"""Tracking and stability metrics for a Rollout, summarised over episodes as JSON.

All per-episode numbers use only the steps where the episode is still *alive* (has not terminated),
and skip step 0 (the reset state, which sits exactly on the reference and would pull every error
towards zero). An episode that terminates on its very first step has no valid steps, and its
tracking errors are NaN. The summaries use nan-aware statistics and report how many episodes
contributed (`n_valid`).

The metrics, and why each one exists:

* **MPJPE (world)**: mean per-joint position error. For every body (31 of them: pelvis, legs,
  spine, arms, hands...), take the distance in metres between where the simulated body is and
  where the reference says it should be, then average over bodies and over time. It is the
  standard pose-tracking number in motion imitation and pose estimation, it is in units anyone
  can picture (0.05 = "5 cm off on average"), and it is exactly the `mpjpe` the env logs during
  training, so eval and training curves are comparable. Its weakness is that it mixes up two
  failures: posing the body wrong, and being in the wrong *place* (drift). A perfect runner that is
  20 cm behind the reference scores 0.20 m.
* **MPJPE (root-relative)**: the same after subtracting each skeleton's own root (pelvis)
  position from all its bodies. This removes drift and keeps only "is the body shaped like the
  reference right now". Orientation is *not* removed, so leaning or turning still counts as error.
  Compare the two: world >> root-relative means the pose is fine but the character is in the
  wrong place.
* **Root drift**: distance between the simulated and reference root (pelvis), over time. We
  report the horizontal (xy) part separately, because height errors are small and bounded while
  horizontal drift can grow without limit. The env terminates at 0.5 m of 3-D drift (training
  default), so drift curves that flatten out well below that are a good sign. Drift that grows
  steadily means the policy runs slightly too slow or too fast.
* **Survival time**: seconds from reset until termination (fall, drift beyond the limit, or
  NaN), capped at the episode length. It is the first thing to look at: tracking errors averaged
  over 0.4 s of a falling baseline and over 30 s of a good policy mean very different things.
  `cap_share` = share of episodes that reached the cap without terminating.
* **Distance travelled**: horizontal displacement of the simulated root from the start to the
  last alive step (straight line, not path length, so jittering on the spot does not count).
  `progress_ratio` = that distance / the reference root's displacement over the same time. 1.0
  means it keeps pace with the clip.
* **Stride cycles**: two ways of counting how many running strides were completed before
  termination. `ref_cycles` = whole reference loops (one loop = one stride, 0.80 s for 16_35)
  that elapsed while alive (time-based: says "how long did it keep going"). `progress_cycles` =
  distance travelled / the reference's per-loop displacement (ground-based: says "how many
  strides' worth of ground did it actually cover"). A character that survives by standing still
  gets ref_cycles > 0 but progress_cycles ~ 0.

Summary statistics: mean, median, and the interquartile range (25th and 75th percentiles), plus
min/max. The median and IQR are there because survival times are often bimodal (most episodes fall
early, a few run to the cap), and the mean of such a distribution describes no actual episode.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eval.rollout import Rollout


def _masked_mean(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean of x[N, T] over T where mask; NaN when an episode has no valid steps."""
    n = mask.sum(1)
    s = np.where(mask, x, 0.0).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def valid_mask(r: Rollout) -> np.ndarray:
    """[N, T+1] steps counted by the metrics: alive and not the reset state."""
    m = r.alive.copy()
    m[:, 0] = False
    return m


def per_step(r: Rollout) -> dict[str, np.ndarray]:
    """Per-step error series [N, T+1] (computed for every step; mask before averaging)."""
    err = np.linalg.norm(r.body_pos - r.body_pos_ref, axis=-1)                 # [N, T+1, B]
    ri = r.root_index
    rel_sim = r.body_pos - r.body_pos[:, :, ri:ri + 1]
    rel_ref = r.body_pos_ref - r.body_pos_ref[:, :, ri:ri + 1]
    err_rel = np.linalg.norm(rel_sim - rel_ref, axis=-1)
    d = r.body_pos[:, :, ri] - r.body_pos_ref[:, :, ri]
    return {
        "mpjpe": err.mean(-1),
        "mpjpe_rootrel": err_rel.mean(-1),
        "root_drift": np.linalg.norm(d, axis=-1),
        "root_drift_xy": np.linalg.norm(d[..., :2], axis=-1),
    }


def last_alive_index(r: Rollout) -> np.ndarray:
    """[N] index of the last step where the episode was alive (0 if it died on step 1)."""
    alive = r.alive
    return np.where(alive.any(1), alive.shape[1] - 1 - np.argmax(alive[:, ::-1], axis=1), 0)


def per_episode(r: Rollout) -> dict[str, np.ndarray]:
    ps = per_step(r)
    mask = valid_mask(r)
    N = r.n_episodes
    last = last_alive_index(r)
    idx = np.arange(N)
    surv = r.survival_s()
    term = r.termination_step()
    terminated = term <= r.n_steps
    root_sim = r.body_pos[:, :, r.root_index, :2]
    root_ref = r.body_pos_ref[:, :, r.root_index, :2]
    dist = np.linalg.norm(root_sim[idx, last] - root_sim[:, 0], axis=-1)
    ref_dist = np.linalg.norm(root_ref[idx, last] - root_ref[:, 0], axis=-1)
    stride_len = float(np.linalg.norm(r.cycle_offset[:2]))
    with np.errstate(invalid="ignore", divide="ignore"):
        progress_ratio = np.where(ref_dist > 1e-6, dist / np.maximum(ref_dist, 1e-6), np.nan)
    # termination cause at the termination step
    cause = np.full(N, "none", dtype=object)
    for i in np.where(terminated)[0]:
        k = term[i]
        cause[i] = ("nan" if r.nan[i, k] else "contact" if r.fell[i, k]
                    else "drift" if r.drifted[i, k] else "other")
    out = {
        "survival_s": surv,
        "terminated": terminated,
        "mpjpe": _masked_mean(ps["mpjpe"], mask),
        "mpjpe_rootrel": _masked_mean(ps["mpjpe_rootrel"], mask),
        "root_drift_mean": _masked_mean(ps["root_drift"], mask),
        "root_drift_xy_mean": _masked_mean(ps["root_drift_xy"], mask),
        "root_drift_xy_final": np.where(mask.any(1), ps["root_drift_xy"][idx, last], np.nan),
        "root_drift_max": np.where(mask.any(1), np.where(mask, ps["root_drift"], -np.inf).max(1), np.nan),
        "distance_m": dist,
        "ref_distance_m": ref_dist,
        "progress_ratio": progress_ratio,
        "ref_cycles": np.floor(surv / r.clip_duration + 1e-6),
        "progress_cycles": dist / stride_len if stride_len > 1e-6 else np.full(N, np.nan),
        "mean_reward": _masked_mean(r.reward, mask),
        "cause": cause,
    }
    return out


def summarise(x: np.ndarray) -> dict:
    """mean / median / IQR / min / max over episodes, ignoring NaN."""
    x = np.asarray(x, np.float64)
    v = x[np.isfinite(x)]
    if v.size == 0:
        return {"n_valid": 0, "mean": None, "median": None, "q25": None, "q75": None,
                "min": None, "max": None}
    q25, med, q75 = np.percentile(v, [25, 50, 75])
    return {"n_valid": int(v.size), "mean": float(v.mean()), "median": float(med),
            "q25": float(q25), "q75": float(q75), "min": float(v.min()), "max": float(v.max())}


def over_time(r: Rollout, every_s: float = 0.1) -> dict:
    """Curves vs episode time: share of episodes alive, and median errors among the alive ones.

    Sampled every 0.1 s by default: baselines fall within a second, so coarser sampling hides
    everything; for a 30 s run it is still only 300 points."""
    ps = per_step(r)
    step = max(1, int(round(every_s / r.dt)))
    ks = np.arange(0, r.n_steps + 1, step)
    alive = r.alive[:, ks]
    out = {"t_s": (ks * r.dt).round(4).tolist(), "alive_share": alive.mean(0).tolist()}
    for k in ("mpjpe", "mpjpe_rootrel", "root_drift_xy"):
        col = []
        for j, kk in enumerate(ks):
            a = alive[:, j]
            col.append(float(np.median(ps[k][a, kk])) if a.any() and kk > 0 else None)
        out[k + "_median"] = col
    return out


def evaluate(r: Rollout, extra: dict | None = None) -> dict:
    """Full metrics dict (JSON-serialisable)."""
    pe = per_episode(r)
    causes, counts = np.unique(pe["cause"].astype(str), return_counts=True)
    res = {
        "policy": r.policy,
        "n_episodes": r.n_episodes,
        "episode_cap_s": r.n_steps * r.dt,
        "dt": r.dt,
        "clip_duration_s": r.clip_duration,
        "cap_share": float(np.mean(~pe["terminated"])),
        "termination_causes": {str(c): int(n) for c, n in zip(causes, counts)},
        "summary": {k: summarise(v) for k, v in pe.items() if k not in ("cause", "terminated")},
        "per_episode": {k: [None if (isinstance(x, float) and not np.isfinite(x)) else
                            (x if isinstance(x, str) else float(x)) for x in v.tolist()]
                        for k, v in pe.items()},
        "over_time": over_time(r),
    }
    res["headline"] = {
        "survival_s_median": res["summary"]["survival_s"]["median"],
        "mpjpe_m_median": res["summary"]["mpjpe"]["median"],
        "mpjpe_rootrel_m_median": res["summary"]["mpjpe_rootrel"]["median"],
    }
    if extra:
        res.update(extra)
    return res


def write_json(obj: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))
    return path


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
