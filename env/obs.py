"""Observation: what the policy sees each step, as one flat vector.

Layout (default, 139 numbers):

    slice      size  what
    [0:56]      56   hinge joint angles qpos[7:] (rad)
    [56:112]    56   hinge joint velocities qvel[6:] (rad/s)
    [112]        1   root height above the floor (m)
    [113:119]    6   root orientation in the heading frame, as the first two columns of its
                     rotation matrix ("6D" rotation; the third column is their cross product)
    [119:122]    3   root linear velocity in the heading frame (m/s)
    [122:125]    3   root angular velocity in the heading frame (rad/s)
    [125:137]   12   lhand, rhand, lfoot, rfoot positions minus root position, in the heading frame
    [137:139]    2   sin(2 pi phase), cos(2 pi phase)
    then, for each n in config `obs.future_steps` (default none), 59 numbers:
                56   reference hinge angles n policy steps ahead
                 3   reference root position n steps ahead minus current root, heading frame

Why the heading frame. The heading frame is the root's rotation with only its yaw (turn about
the vertical axis) kept: x = the direction the body faces, flattened onto the ground, z = up.
Running north or running east is the same skill, so the policy shouldn't have to learn it twice.
Expressing velocities and limb positions in this frame makes the observation identical for both.
Tilt (leaning forward, falling sideways) still shows up, because it is what's left after the
yaw is removed. Absolute x/y position is left out for the same reason.

Why the 6D orientation. Quaternions have a sign ambiguity (q and -q are the same rotation) and
Euler angles jump at +-180 deg. Two columns of the rotation matrix are continuous and unique,
which is easier for a neural network (Zhou et al. 2019, "On the continuity of rotation
representations").

Why phase. The policy is told where in the clip it is, as sin/cos so that phase 0.99 and 0.01 are
close together. Without it, the same body state (say, mid-stride) could need different actions
at different times in the clip, and the policy couldn't tell which. DeepMimic's policy gets the
phase as its only reference information.

Future targets (optional, off by default). Showing the policy a few future reference poses tells
it explicitly where to go, which helps for long, non-repeating clips. For one cyclic running
clip, phase already says everything, so DeepMimic's choice (phase only) is the default.

Heading from the root body. In this model the root's local +x axis points to the body's left,
+y up and +z forward. We take forward = left x world_up. It stays well defined when the torso
pitches forward or back (unlike projecting the local +z axis, which vanishes when lying face
down).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from env import quat

BASE_SIZE = 139
FUTURE_SIZE = 59


def heading_yaw(root_xmat):
    """Yaw angle of the body's facing direction, from the root's 3x3 world rotation matrix."""
    left = root_xmat[:, 0]
    fwd = jnp.cross(left, jnp.array([0.0, 0.0, 1.0]))
    return jnp.arctan2(fwd[1], fwd[0])


def heading_frame(root_xmat):
    """3x3 yaw-only rotation H. World vector v in the heading frame is H.T @ v."""
    return quat.rot_z(heading_yaw(root_xmat))


def observation(qpos, qvel, root_xmat, ee_world, phase, future=()):
    """Build the flat observation.

    Args:
      qpos, qvel: simulator state.
      root_xmat: [3, 3] root body rotation (data.xmat[root]).
      ee_world: [4, 3] world positions of lhand, rhand, lfoot, rfoot.
      phase: float in [0, 1).
      future: sequence of (ref_hinge_qpos [56], ref_root_pos [3]) for future steps.
    """
    H = heading_frame(root_xmat)
    root = qpos[:3]
    R_h = H.T @ root_xmat
    ang_world = root_xmat @ qvel[3:6]       # MuJoCo free-joint angular velocity is body-local
    two_pi_phase = 2 * jnp.pi * phase
    parts = [
        qpos[7:], qvel[6:], root[2:3], R_h[:, :2].T.reshape(-1),
        H.T @ qvel[:3], H.T @ ang_world,
        ((ee_world - root) @ H).reshape(-1),
        jnp.stack([jnp.sin(two_pi_phase), jnp.cos(two_pi_phase)]),
    ]
    for hinge, rpos in future:
        parts += [hinge, H.T @ (rpos - root)]
    return jnp.concatenate(parts)


def obs_size(num_future: int) -> int:
    return BASE_SIZE + FUTURE_SIZE * num_future


def _check_layout():  # pragma: no cover - documentation aid
    z = observation(jnp.zeros(63).at[3].set(1), jnp.zeros(62), jnp.eye(3), jnp.zeros((4, 3)), 0.0)
    assert z.shape == (BASE_SIZE,), z.shape


if __name__ == "__main__":
    _check_layout()
    print("ok", jax.default_backend())
