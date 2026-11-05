"""Push-recovery test: shove the running character sideways and see whether it stays up.

Why: tracking error tells you how well the policy copies the clip under ideal conditions. It says
nothing about *balance*. A policy that copies the motion open-loop can look perfect until the
first disturbance. Pushes are the standard robustness probe for physics-based characters
(DeepMimic, section 9.2, pushed its characters with forces on the pelvis for 0.2 s).

Protocol (all parameters in PushConfig, defaults in brackets):
  1. Reset at a random clip phase (reference state initialisation, as in training).
  2. Warm-up: run the policy unperturbed for a random time drawn uniformly from `warmup_s`
     [1.0, 2.0] s. Randomising the moment means pushes land at every phase of the stride (a push
     during flight is very different from one during stance), and different episodes are not all
     pushed at the same phase.
  3. Push: a horizontal force of magnitude F, in a direction drawn uniformly from 0..360 degrees,
     applied to the `body` [root = pelvis] for `duration_s` [0.1 s = 5 policy steps]. It's applied
     through MuJoCo's `xfrc_applied` at the body's centre of mass (see eval/rollout.py).
     Why the pelvis: it is the root of the kinematic tree and carries most of the mass near the
     whole-body centre of mass (6.1 kg of 51.8 kg directly, everything is attached to it), so a
     push there mostly moves the whole body rather than spinning it. DeepMimic pushes the pelvis too.
     `body="thorax"` pushes the upper torso instead, which also creates a large tipping torque
     about the feet (harder).
  4. Recovery window: count the push as **survived** if the episode does not terminate (env
     termination: fall/bad contact, root drift > limit, NaN) from the push onset until
     `recovery_s` [2.0 s] after the push ends. With `fail_on="fall"`, only falls (bad floor contact)
     and NaN count, and drifting off the reference does not. That's useful because after a hard
     shove a good policy may recover balance but end up more than 0.5 m from where the
     reference is.
  5. Only pushes that actually happen count. An episode that terminated *before* the push onset
     (during warm-up) was never pushed. It is excluded from the survival share and counted as
     `n_invalid`. For a policy that can't run at all (the baselines), this means every push is
     invalid and the share is undefined (None), which is the honest answer: you can't measure push
     recovery for a character that falls over on its own. The F = 0 level is the control. Its
     "survival" is the share of episodes that survive the same window *without* a push.

Sweep: F in `forces` [0, 50, ..., 600 N] with `n_per_level` [32] pushes per level. Each level
reports share = survived / valid with a 95% Wilson score interval.

Force vs impulse. What changes the body's motion is the *impulse* J = F * duration (newton
seconds, N s), because momentum changes by exactly J: Δv = J / m. 300 N for 0.1 s = 30 N s, and on
our 51.8 kg humanoid that is a Δv of about 0.58 m/s for the whole body if it were rigid. 150 N for 0.2 s
is the same impulse and roughly the same shove. Reporting the impulse (and Δv) alongside the force
makes results comparable across different push durations and body masses. We still sweep force
at a fixed duration because that is how pushes are usually reported and what you can feel.

Why the Wilson interval (and not "p ± 1.96 sqrt(p(1-p)/n)"). With 32 pushes and a share near 0 or 1
the normal approximation gives nonsense (zero width at 0/32, or limits outside [0, 1]). Wilson's
interval stays inside [0, 1] and is about right for small n. With n = 32 it is roughly ±15
percentage points around 50%: two levels whose intervals overlap heavily are not reliably different.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from env.imitation import RunImitation
from eval.policy import EvalPolicy
from eval.rollout import PushPlan, Rollout, rollout


@dataclass
class PushConfig:
    forces: list[float] = field(default_factory=lambda: [float(f) for f in range(0, 601, 50)])
    n_per_level: int = 32
    duration_s: float = 0.1
    warmup_s: tuple[float, float] = (1.0, 2.0)
    recovery_s: float = 2.0
    body: str = "root"
    fail_on: str = "env"        # "env" (any env termination) or "fall" (bad contact / NaN only)
    seed: int = 0


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float] | tuple[None, None]:
    """95% Wilson score interval for k successes out of n trials."""
    if n == 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def make_plan(cfg: PushConfig, force: float, dt: float, rng: np.random.Generator) -> PushPlan:
    n = cfg.n_per_level
    lo, hi = (int(round(w / dt)) for w in cfg.warmup_s)
    start = rng.integers(lo, hi + 1, size=n)
    theta = rng.uniform(0.0, 2 * np.pi, size=n)
    f = force * np.stack([np.cos(theta), np.sin(theta), np.zeros(n)], axis=1)
    return PushPlan(start_step=start, force=f, n_steps=max(1, int(round(cfg.duration_s / dt))),
                    body=cfg.body)


def score(r: Rollout, cfg: PushConfig) -> dict:
    """Classify each episode of a pushed rollout as invalid / survived / failed."""
    p = r.push
    end = p.start_step + p.n_steps + int(round(cfg.recovery_s / r.dt))
    if cfg.fail_on == "env":
        fail = r.done.astype(bool)
    elif cfg.fail_on == "fall":
        fail = r.fell | r.nan
    else:
        raise ValueError(f"fail_on must be 'env' or 'fall', not {cfg.fail_on!r}")
    T = fail.shape[1] - 1
    first_fail = np.where(fail.any(1), fail.argmax(1), T + 1)
    # The push acts during the steps start..start+n-1, so the state at index `start` is the last
    # unpushed one. Valid = no failure up to and including it.
    valid = first_fail > p.start_step
    survived = valid & (first_fail > np.minimum(end, T)) & (end <= T)
    t_fail_after = np.where(valid & ~survived, (first_fail - p.start_step) * r.dt, np.nan)
    return {"valid": valid, "survived": survived, "fail_time_after_push_s": t_fail_after}


def run_push_test(env: RunImitation, policy: EvalPolicy, cfg: PushConfig | None = None, *,
                  keep_rollouts: bool = False, verbose: bool = False) -> tuple[dict, dict]:
    """Run the force sweep. Returns (results JSON dict, {force: Rollout} if keep_rollouts)."""
    cfg = cfg or PushConfig()
    dt = env.dt
    mass = float(env.mj_model.body_subtreemass[0])
    rng = np.random.default_rng(cfg.seed)
    seconds = cfg.warmup_s[1] + cfg.duration_s + cfg.recovery_s + dt
    levels, rollouts = [], {}
    for j, F in enumerate(cfg.forces):
        plan = make_plan(cfg, float(F), dt, rng)
        r = rollout(env, policy, cfg.n_per_level, seconds, seed=cfg.seed * 1000 + j, push=plan,
                    tail_s=0.0)
        s = score(r, cfg)
        n_valid, k = int(s["valid"].sum()), int(s["survived"].sum())
        lo, hi = wilson_interval(k, n_valid)
        impulse = float(F) * plan.n_steps * dt
        levels.append({
            "force_N": float(F), "impulse_Ns": impulse, "delta_v_mps": impulse / mass,
            "n_pushes": cfg.n_per_level, "n_valid": n_valid, "n_invalid": cfg.n_per_level - n_valid,
            "n_survived": k, "share": (k / n_valid) if n_valid else None, "ci95": [lo, hi],
            "share_of_all": k / cfg.n_per_level,
            "median_fail_time_after_push_s": (float(np.nanmedian(s["fail_time_after_push_s"]))
                                              if np.isfinite(s["fail_time_after_push_s"]).any() else None),
        })
        if verbose:
            sh = "n/a" if not n_valid else f"{k / n_valid:.2f}"
            print(f"[push] {policy.name}: F={F:.0f} N  valid {n_valid}/{cfg.n_per_level}  "
                  f"survived {k}  share {sh}", flush=True)
        if keep_rollouts:
            rollouts[float(F)] = (r, s)
    res = {
        "policy": policy.name,
        "config": {**asdict(cfg), "warmup_s": list(cfg.warmup_s), "duration_steps":
                   max(1, int(round(cfg.duration_s / dt))), "dt": dt},
        "total_mass_kg": mass,
        "levels": levels,
        "headline": headline(levels),
    }
    return res, rollouts


def headline(levels: list[dict]) -> dict:
    """Single numbers: pooled survival share over all F > 0, and F50 (force where the share first
    drops below 50%, linearly interpolated between levels; None if undefined)."""
    pos = [lv for lv in levels if lv["force_N"] > 0]
    nv = sum(lv["n_valid"] for lv in pos)
    ns = sum(lv["n_survived"] for lv in pos)
    pooled = ns / nv if nv else None
    f50 = None
    pts = [(lv["force_N"], lv["share"]) for lv in levels if lv["share"] is not None]
    for (f0, s0), (f1, s1) in zip(pts, pts[1:]):
        if s0 >= 0.5 > s1:
            f50 = f0 + (s0 - 0.5) / (s0 - s1) * (f1 - f0)
            break
    if f50 is None and pts and all(s >= 0.5 for _, s in pts):
        f50 = f">{pts[-1][0]:.0f}"
    if f50 is None and pts and pts[0][1] < 0.5:
        f50 = "undefined: survival already < 50% at the lowest force (check the F=0 control)"
    control = next((lv for lv in levels if lv["force_N"] == 0), None)
    return {"pooled_share_F_gt_0": pooled, "pooled_n_valid": nv, "F50_N": f50,
            "control_share_F0": control["share"] if control else None}
