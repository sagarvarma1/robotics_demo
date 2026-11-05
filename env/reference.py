"""Reference motion lookup: "where should the body be at time t?"

The clip is loaded once (`motion.format.load_motion`) and turned into JAX arrays, so during training
every lookup is a cheap array gather, the same on CPU and GPU.

Looping (docs/interfaces.md, section 1). A running clip is a few strides long, but episodes last
10 s, so the clip is played again and again. Replaying it unchanged would teleport the target back
to the start every loop. Instead, at loop k the whole clip is moved forward by k * cycle_offset_pos
(and turned by k * cycle_offset_yaw), so the target keeps running forward smoothly:

    global frame g = t * fps,  frame i = g mod T,  loop k = g // T
    world position  p' = Rz(k*yaw) @ p + k*offset      (xy only; offset z = 0)
    world rotation  q' = qz(k*yaw) * q

Interpolation. If the clip's fps differs from the policy rate (1/ctrl_dt), a policy step can fall
between two frames, so we blend frame g and g+1 linearly (quaternions with nlerp). Each of the
two frames gets its own loop offset, so blending across the end of the clip is also smooth. If the
rates match, we round to the nearest frame instead. (Rounding rather than flooring makes the
lookup immune to float32 round-off in the simulator clock, where 0.06 * 50 may come out as 2.9999.)

Clock. The reference time is read from `data.time`. Reset sets `data.time` to the sampled start
time, and physics advances it. Brax's auto-reset copies a cached `data` back on episode end
(including its time) without resetting `info`, so keeping the clock in `data` keeps the reference
in sync with the body whichever reset mode is used.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from env import quat
from motion.format import EE_BODIES, Motion


class RefFrame(NamedTuple):
    """Reference quantities at one instant (world frame, loop offsets applied)."""
    qpos: jax.Array        # [nq]  root pos/quat with offsets, hinges unchanged
    qvel: jax.Array        # [nv]  root linear velocity rotated by the loop yaw
    body_pos: jax.Array    # [B, 3]
    local_quat: jax.Array  # [B, 4] each body relative to its parent (root: relative to world)
    ee_pos: jax.Array      # [4, 3]
    com: jax.Array         # [3]
    index: jax.Array       # int, frame index i
    loop: jax.Array        # int, loop count k
    phase: jax.Array       # float in [0, 1)


def local_quats_np(model: mujoco.MjModel, body_quat: np.ndarray) -> np.ndarray:
    """Child-relative-to-parent rotations from world body rotations [T, B, 4] (world excluded).

    For the root (parent = world) this is just its world rotation.
    """
    parent = model.body_parentid[1:]                  # parent ids, 0 = world
    q = body_quat.astype(np.float64)
    T, B, _ = q.shape
    world = np.concatenate([np.tile([1.0, 0, 0, 0], (T, 1, 1)), q], axis=1)  # index by body id
    qp = world[:, parent]
    out = np.asarray(quat.mul(quat.conj(jnp.asarray(qp)), jnp.asarray(q)))
    return out.astype(np.float32)


class Reference:
    def __init__(self, motion: Motion, model: mujoco.MjModel, ctrl_dt: float,
                 interpolate: str | bool = "auto"):
        names = [model.body(b).name for b in range(1, model.nbody)]
        if motion.body_names != names:
            raise ValueError("motion body_names do not match the model")
        self.fps = float(motion.fps)
        self.T = motion.num_frames
        self.cyclic = bool(motion.cyclic)
        self.duration = motion.duration
        if interpolate == "auto":
            interpolate = abs(self.fps * ctrl_dt - 1.0) > 1e-6
        self.interpolate = bool(interpolate)
        self.qpos = jnp.asarray(motion.qpos)
        self.qvel = jnp.asarray(motion.qvel)
        self.body_pos = jnp.asarray(motion.body_pos)
        self.local_quat = jnp.asarray(local_quats_np(model, motion.body_quat))
        self.ee_pos = jnp.asarray(motion.ee_pos)
        self.com = jnp.asarray(motion.com)
        self.offset = jnp.asarray(motion.cycle_offset_pos, jnp.float32) * jnp.array([1.0, 1.0, 0.0])
        self.yaw = float(motion.cycle_offset_yaw)
        self.ee_names = EE_BODIES

    # -- frame at an integer global index ----------------------------------------------------
    def _frame(self, g):
        if self.cyclic:
            i, k = jnp.mod(g, self.T), jnp.floor_divide(g, self.T)
        else:
            i, k = jnp.clip(g, 0, self.T - 1), jnp.zeros_like(g)
        kf = k.astype(jnp.float32)
        R = quat.rot_z(kf * self.yaw)
        qz = quat.from_yaw(kf * self.yaw)
        off = kf * self.offset

        def tp(p):  # transform world positions [..., 3]
            return p @ R.T + off

        qpos = self.qpos[i]
        qpos = qpos.at[:3].set(tp(qpos[:3])).at[3:7].set(quat.mul(qz, qpos[3:7]))
        qvel = self.qvel[i]
        qvel = qvel.at[:3].set(R @ qvel[:3])          # angular velocity is body-local: unchanged
        lq = self.local_quat[i]
        lq = lq.at[0].set(quat.mul(qz, lq[0]))        # only the root's rotation is world-relative
        return qpos, qvel, tp(self.body_pos[i]), lq, tp(self.ee_pos[i]), tp(self.com[i]), i, k

    def at(self, time) -> RefFrame:
        f = jnp.asarray(time, jnp.float32) * self.fps
        if self.interpolate:
            g0 = jnp.floor(f)
            alpha = f - g0
            g0 = g0.astype(jnp.int32)
            a, b = self._frame(g0), self._frame(g0 + 1)
            lerp = lambda x, y: (1 - alpha) * x + alpha * y  # noqa: E731
            qpos = lerp(a[0], b[0]).at[3:7].set(quat.nlerp(a[0][3:7], b[0][3:7], alpha))
            fr = (qpos, lerp(a[1], b[1]), lerp(a[2], b[2]), quat.nlerp(a[3], b[3], alpha),
                  lerp(a[4], b[4]), lerp(a[5], b[5]))
            i, k = a[6], a[7]
        else:
            g = jnp.round(f).astype(jnp.int32)
            *fr, i, k = self._frame(g)
        phase = jnp.mod(f / self.T, 1.0) if self.cyclic else jnp.clip(f / max(self.T - 1, 1), 0, 1)
        return RefFrame(*fr, index=i, loop=k, phase=phase)

    def frame_aligned_time(self, u):
        """Map u in [0, 1) to a start time. Snap to a frame if we are not interpolating.

        Reference state initialisation picks a uniformly random point in the clip.
        """
        if self.interpolate:
            return u * self.duration
        return jnp.floor(u * self.T) / self.fps
