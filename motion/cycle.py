"""Cut one clean, loopable running stride out of a clip.

Why loop at all: a recorded run lasts ~1.5 s, but we want the policy to run for as long as
we like. Running is periodic: one *stride* (left foot strike -> next left foot strike)
contains everything, and the next stride looks the same, just further along the track.
So we keep a single stride and replay it, moving it forward by ``cycle_offset_pos`` each
time round (see the looping rule in docs/interfaces.md).

Steps:

1. **Heading.** :func:`heading_yaw` reads which way the pelvis faces. We rotate the clip
   about the vertical axis so the runner travels along +x. Then "forward" means the same
   thing in every clip, and the per-loop offset is a pure +x shift.
2. **Find the seam** (:func:`find_cycle`). For every pair of frames (i, j) that are one
   stride apart (0.5–1.0 s), we measure how different the body looks and moves at i and
   at j: joint angles, pelvis height and tilt, and their velocities, each normalised by
   its spread in the clip. The pair with the smallest difference is the best place to
   cut: playing i..j-1 and jumping back to i is then almost invisible. Including
   velocities matters: two frames can have the same pose while the legs move in opposite
   directions.
3. **Blend the seam** (:func:`make_cyclic`). The best pair is never identical, so we fade
   in a correction that makes "frame j" equal to frame i exactly (joint angles, pelvis
   height, pelvis tilt and heading). The correction grows only during the *flight
   phases*, when both feet are in the air. Changing the pose while a foot is planted
   would drag that foot along the floor. Fixing the heading mismatch is why
   ``cycle_offset_yaw`` is 0: the loop runs dead straight instead of slowly curving.
   Fixing the heading mismatch here is why ``cycle_offset_yaw`` is 0: the loop runs
   dead straight instead of slowly curving.
4. **Offset.** ``cycle_offset_pos`` = pelvis xy at j minus at i (z = 0).
"""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

# Forward direction of the model's pelvis in its own body frame. The body frames are
# y-up (CMU convention), and a CMU subject faces +z.
FORWARD_LOCAL = np.array([0.0, 0.0, 1.0])


def heading_yaw(quats_wxyz: np.ndarray) -> np.ndarray:
    """Yaw (rad) of the pelvis's forward direction projected on the ground."""
    f = Rotation.from_quat(quats_wxyz, scalar_first=True).apply(FORWARD_LOCAL)
    return np.arctan2(f[..., 1], f[..., 0])


def rotate_qpos_about_z(qpos: np.ndarray, yaw: float, shift_xy=(0.0, 0.0)) -> np.ndarray:
    """Rigidly rotate the whole body about world z (through the origin), then shift xy."""
    q = qpos.copy()
    Rz = Rotation.from_euler("z", yaw)
    q[:, :3] = Rz.apply(q[:, :3])
    q[:, :2] += np.asarray(shift_xy)
    q[:, 3:7] = (Rz * Rotation.from_quat(q[:, 3:7], scalar_first=True)).as_quat(scalar_first=True)
    return q


def travel_yaw(qpos: np.ndarray) -> float:
    d = qpos[-1, :2] - qpos[0, :2]
    return float(np.arctan2(d[1], d[0]))


def _features(qpos: np.ndarray, fps: float, hinge_weight: np.ndarray):
    """Heading-invariant pose features and their velocities, per frame."""
    yaw = heading_yaw(qpos[:, 3:7])
    R = Rotation.from_quat(qpos[:, 3:7], scalar_first=True)
    tilt = (Rotation.from_euler("z", -yaw[:, None]) * R).as_rotvec()        # pelvis tilt without heading
    pose = np.hstack([qpos[:, 7:] * hinge_weight, qpos[:, 2:3], tilt])
    vel = np.gradient(pose, 1.0 / fps, axis=0)
    # root horizontal velocity in the heading frame
    v_xy = np.gradient(qpos[:, :2], 1.0 / fps, axis=0)
    c, s = np.cos(-yaw), np.sin(-yaw)
    v_local = np.stack([c * v_xy[:, 0] - s * v_xy[:, 1], s * v_xy[:, 0] + c * v_xy[:, 1]], 1)
    vel = np.hstack([vel, v_local])
    return pose, vel


def find_cycle(model: mujoco.MjModel, qpos: np.ndarray, fps: float, *, min_period: float = 0.5,
               max_period: float = 1.0, margin: int = 3, vel_weight: float = 0.5,
               driven_mask: np.ndarray | None = None) -> tuple[int, int, float]:
    """Return (i, j, cost): play frames i..j-1, then frame j is 'frame i of the next loop'."""
    T = qpos.shape[0]
    w = np.ones(qpos.shape[1] - 7) if driven_mask is None else driven_mask.astype(float)
    pose, vel = _features(qpos, fps, w)
    ps = pose.std(0) + 1e-3
    vs = vel.std(0) + 1e-2
    pn, vn = pose / ps, vel / vs
    pmin, pmax = int(round(min_period * fps)), int(round(max_period * fps))
    best = (None, None, np.inf)
    for i in range(margin, T - margin):
        for j in range(i + pmin, min(i + pmax, T - 1 - margin) + 1):
            c = np.mean((pn[j] - pn[i]) ** 2) + vel_weight * np.mean((vn[j] - vn[i]) ** 2)
            if c < best[2]:
                best = (i, j, c)
    if best[0] is None:
        raise ValueError(f"clip too short for a {min_period}-{max_period} s cycle")
    return best


def flight_mask(model: mujoco.MjModel, qpos: np.ndarray, fps: float) -> np.ndarray:
    """True on frames where neither foot is in stance (the flight phase of running)."""
    from motion.retarget import RetargetConfig, foot_tracks, stance_mask
    cfg = RetargetConfig()
    tr = foot_tracks(model, qpos)
    st = np.zeros(qpos.shape[0], bool)
    for s in tr["low"]:
        st |= stance_mask(tr["low"][s], tr["xy"][s], fps, cfg)
    return ~st


def make_cyclic(qpos: np.ndarray, i: int, j: int, *, flight: np.ndarray | None = None):
    """Cut frames i..j-1, blend the seam, align the loop with +x.

    The mismatch between frame j (what should follow the last frame) and frame i is
    removed gradually, but **only during flight frames** (``flight[t]`` True, both feet
    in the air). While a foot is planted the correction stays constant. A changing
    correction would move the planted foot, i.e. make it slide. Heading is corrected as a
    rigid rotation of the whole trajectory (pelvis position *and* orientation) about the
    loop's start point, so a constant heading correction keeps a planted foot planted too.
    Without a flight mask (e.g. walking) the correction is spread evenly over the stride.

    Returns (qpos_cycle [j-i, nq], offset_pos [3], info). ``info`` also holds what
    :func:`transform_points` needs to move keypoints the same way.
    """
    q = qpos[i:j].copy()
    q_next = qpos[j].copy()              # what would follow the last frame
    T = q.shape[0]
    fl = np.ones(T, bool) if flight is None or not np.any(flight[:T]) else flight[:T].astype(bool)
    inc = fl.astype(float)
    w = np.concatenate([[0.0], np.cumsum(inc)[:-1]]) / inc.sum()   # w[0] = 0, w[T] = 1
    c = q[0, :2].copy()                                             # pivot for heading
    raw_yaw = float(heading_yaw(q[0, 3:7]) - heading_yaw(q_next[3:7]))
    raw_yaw = (raw_yaw + np.pi) % (2 * np.pi) - np.pi

    def rigid_yaw(qq, ang):
        out = qq.copy()
        for t in range(len(qq)):
            Rz = Rotation.from_euler("z", ang[t])
            out[t, :2] = c + Rz.apply([*(qq[t, :2] - c), 0.0])[:2]
            out[t, 3:7] = (Rz * Rotation.from_quat(qq[t, 3:7], scalar_first=True)).as_quat(
                scalar_first=True)
        return out

    q = rigid_yaw(q, w * raw_yaw)
    q_next = rigid_yaw(q_next[None], [raw_yaw])[0]
    # remaining mismatch (tilt, joint angles, height): blended with the same weights
    d_hinge = q_next[7:] - q[0, 7:]
    d_z = q_next[2] - q[0, 2]
    R0 = Rotation.from_quat(q[0, 3:7], scalar_first=True)
    Rn = Rotation.from_quat(q_next[3:7], scalar_first=True)
    corr = (R0 * Rn.inv()).as_rotvec()
    q[:, 7:] -= w[:, None] * d_hinge[None]
    q[:, 2] -= w * d_z
    Rw = Rotation.from_rotvec(w[:, None] * corr[None])
    q[:, 3:7] = (Rw * Rotation.from_quat(q[:, 3:7], scalar_first=True)).as_quat(scalar_first=True)
    offset = q_next[:2] - q[0, :2]
    # align: offset along +x, loop starts at x = y = 0
    yaw = -float(np.arctan2(offset[1], offset[0]))
    q = rotate_qpos_about_z(q, yaw)
    shift = q[0, :2].copy()
    q[:, :2] -= shift
    info = {"cut_frames": [int(i), int(j)], "n_flight_frames": int(fl.sum()),
            "seam_hinge_mismatch_rad_max": float(np.abs(d_hinge).max()),
            "seam_root_z_mismatch_m": float(d_z),
            "seam_root_tilt_mismatch_deg": float(np.degrees(np.linalg.norm(corr))),
            "seam_raw_yaw_mismatch_deg": float(np.degrees(raw_yaw)),
            "align_yaw_rad": yaw,
            "_points": {"w": w.tolist(), "raw_yaw": raw_yaw, "pivot": c.tolist(),
                        "align_yaw": yaw, "shift": shift.tolist()}}
    return q, np.array([float(np.linalg.norm(offset)), 0.0, 0.0]), info


def transform_points(pts: np.ndarray, info: dict) -> np.ndarray:
    """Apply make_cyclic's rigid transforms to keypoints ``pts`` [T, J, 3] (frames i..j-1)."""
    p = info["_points"]
    out = pts.copy()
    c = np.array([*p["pivot"], 0.0])
    for t in range(len(out)):
        Rz = Rotation.from_euler("z", p["w"][t] * p["raw_yaw"]).as_matrix()
        out[t] = (out[t] - c) @ Rz.T + c
    Ra = Rotation.from_euler("z", p["align_yaw"]).as_matrix()
    out = out @ Ra.T
    out[..., :2] -= np.asarray(p["shift"])
    return out
