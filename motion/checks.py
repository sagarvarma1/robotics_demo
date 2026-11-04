"""Quality checks for a retargeted motion: is it a believable, physically sensible run?

Each function returns plain numbers so they can be printed, stored in ``meta`` and asserted
in tests. What each check catches:

* **keypoint error**: does the model's body actually go where the scaled source joint is?
  (catches a bad joint map, a wrong axis convention or IK not converging)
* **joint limits**: a policy can never reproduce a pose outside the joint ranges.
* **stance-foot height and sliding**: a planted foot should sit on the floor and not
  skate. Skating feet in the reference teach the policy to shuffle. Note that the ankle
  and toe origins move a few cm in a real stance anyway, because the foot rolls from heel
  to toe, so a few cm here is normal.
* **loop seam**: the jump from the last frame to the first should look like any other
  frame-to-frame change.
* **speed**: sanity check that this is a run (roughly 2.5–6 m/s), not a walk.
* **direct angle copy**: dm_control simply copies AMC angles onto same-named hinges.
  Because our model was built from a CMU skeleton, IK should land close to that copy on
  the big joints. Large differences are expected only where the subject's bones differ
  from the model's (spine curvature) or where IK has no data (hands, fingers, toes).
"""

from __future__ import annotations

import mujoco
import numpy as np

from motion.retarget import FOOT_BODIES, RetargetConfig, foot_tracks, model_body_pos, stance_mask


def keypoint_error_cm(model, qpos, kp_pos, names, body_map) -> dict:
    bp = model_body_pos(model, qpos, [body_map[n] for n in names])
    err = np.linalg.norm(bp - kp_pos, axis=-1) * 100
    per = {n: {"mean": float(err[:, j].mean()), "max": float(err[:, j].max())}
           for j, n in enumerate(names)}
    return {"mean_cm": float(err.mean()), "max_cm": float(err.max()), "per_keypoint": per}


def joint_limit_violations(model, qpos, tol: float = 1e-6) -> dict:
    lo, hi = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
    q = qpos[:, 7:]
    v = np.maximum(lo - q, 0) + np.maximum(q - hi, 0)
    return {"n_violations": int((v > tol).sum()), "max_violation_rad": float(v.max())}


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segs, start = [], None
    for t, m in enumerate(mask):
        if m and start is None:
            start = t
        if not m and start is not None:
            segs.append((start, t))
            start = None
    if start is not None:
        segs.append((start, len(mask)))
    return segs


def stance_report(model, qpos, fps, cfg: RetargetConfig | None = None, *, cyclic=False,
                  offset_xy=(0.0, 0.0), min_frames: int = 3) -> dict:
    """Per stance phase: mean/min lowest-geom height and horizontal slide of ankle and toe."""
    cfg = cfg or RetargetConfig()
    T = qpos.shape[0]
    q = qpos
    if cyclic:  # unroll two loops so a stance that crosses the seam is seen whole
        q2 = qpos.copy()
        q2[:, :2] += np.asarray(offset_xy)[:2]
        q = np.vstack([qpos, q2])
    tr = foot_tracks(model, q)
    toe_xy = {s: model_body_pos(model, q, [FOOT_BODIES[s][1]])[:, 0, :2] for s in FOOT_BODIES}
    out, seen = [], set()
    for s in FOOT_BODIES:
        m = stance_mask(tr["low"][s], tr["xy"][s], fps, cfg, cyclic=cyclic,
                        offset_xy=2 * np.asarray(offset_xy)) if cyclic else \
            stance_mask(tr["low"][s], tr["xy"][s], fps, cfg)
        for a, b in _segments(m):
            if b - a < min_frames:
                continue
            if cyclic:
                if a == 0 or b == len(m):   # touches the unrolled ends: incomplete
                    continue
                key = (s, a % T)
                if key in seen:
                    continue
                seen.add(key)
            ank = tr["xy"][s][a:b]
            toe = toe_xy[s][a:b]
            out.append({"foot": s, "start": int(a % T), "frames": int(b - a),
                        "height_mean_cm": float(tr["low"][s][a:b].mean() * 100),
                        "height_min_cm": float(tr["low"][s][a:b].min() * 100),
                        "ankle_slide_cm": float(np.linalg.norm(ank[-1] - ank[0]) * 100),
                        "toe_slide_cm": float(np.linalg.norm(toe[-1] - toe[0]) * 100)})
    low_all = np.concatenate([tr["low"][s][:T] for s in FOOT_BODIES])
    return {"stances": out, "min_foot_geom_z_cm": float(low_all.min() * 100)}


def seam_report(model, qpos, qvel, offset_pos) -> dict:
    """Compare the last->first transition with ordinary frame-to-frame changes."""
    nxt = qpos[0].copy()
    nxt[:2] += np.asarray(offset_pos)[:2]
    ext = np.vstack([qpos, nxt[None]])
    dq = np.zeros((ext.shape[0] - 1, model.nv))
    for t in range(ext.shape[0] - 1):
        mujoco.mj_differentiatePos(model, dq[t], 1.0, ext[t], ext[t + 1])
    pose_step = np.linalg.norm(dq[:, 6:], axis=1)          # hinge part, rad
    vel_ext = np.vstack([qvel, qvel[:1]])
    vel_step = np.linalg.norm(np.diff(vel_ext, axis=0)[:, 6:], axis=1)
    return {"pose_jump_rad": float(pose_step[-1]),
            "pose_step_median_rad": float(np.median(pose_step[:-1])),
            "pose_step_max_rad": float(pose_step[:-1].max()),
            "vel_jump": float(vel_step[-1]),
            "vel_step_median": float(np.median(vel_step[:-1])),
            "vel_step_max": float(vel_step[:-1].max()),
            "root_pos_jump_m": float(np.linalg.norm(dq[-1, :3])),
            "root_pos_step_median_m": float(np.median(np.linalg.norm(dq[:-1, :3], axis=1)))}


def speed_report(qpos, fps, offset_pos=None) -> dict:
    T = qpos.shape[0]
    if offset_pos is not None:
        v = float(offset_pos[0] / (T / fps))
    else:
        v = float((qpos[-1, 0] - qpos[0, 0]) / ((T - 1) / fps))
    return {"forward_speed_mps": v}


def direct_copy_rms_deg(model, qpos_ik, qpos_copy) -> dict:
    names = [model.joint(j).name for j in range(1, model.njnt)]
    d = np.degrees(qpos_ik[:, 7:] - qpos_copy[:, 7:])
    rms = np.sqrt((d ** 2).mean(0))
    return {n: float(r) for n, r in zip(names, rms)}
