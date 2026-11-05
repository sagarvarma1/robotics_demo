"""Small quaternion and rotation helpers in JAX (MuJoCo convention: q = (w, x, y, z)).

Why our own helpers: they are a few lines each, and having them here (instead of importing MJX's
private `mujoco.mjx._src.math`) keeps the env independent of MJX internals that may move.

A unit quaternion q represents a rotation by angle `a` about unit axis `n` as
    q = (cos(a/2), sin(a/2) * n).
q and -q are the same rotation, which is why `angle` uses |w|.
"""

from __future__ import annotations

import jax.numpy as jnp


def mul(a, b):
    """Hamilton product a*b: rotate by b first, then by a. Broadcasts over leading axes."""
    aw, ax, ay, az = jnp.moveaxis(a, -1, 0)
    bw, bx, by, bz = jnp.moveaxis(b, -1, 0)
    return jnp.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def conj(q):
    """Inverse of a unit quaternion."""
    return q * jnp.array([1.0, -1.0, -1.0, -1.0], q.dtype)


def angle(q):
    """Rotation angle in [0, pi] of a (unit) quaternion. atan2 form is accurate near 0."""
    return 2.0 * jnp.arctan2(jnp.linalg.norm(q[..., 1:], axis=-1), jnp.abs(q[..., 0]))


def from_yaw(yaw):
    """Rotation by `yaw` radians about world z."""
    h = 0.5 * yaw
    z = jnp.zeros_like(h)
    return jnp.stack([jnp.cos(h), z, z, jnp.sin(h)], axis=-1)


def to_mat(q):
    """3x3 rotation matrix (columns = rotated x, y, z axes)."""
    w, x, y, z = jnp.moveaxis(q, -1, 0)
    return jnp.stack([
        jnp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        jnp.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        jnp.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], axis=-2)


def rot_z(yaw):
    """3x3 rotation matrix about world z."""
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    o, z = jnp.ones_like(c), jnp.zeros_like(c)
    return jnp.stack([jnp.stack([c, -s, z], -1), jnp.stack([s, c, z], -1),
                      jnp.stack([z, z, o], -1)], -2)


def nlerp(a, b, alpha):
    """Normalised linear interpolation between quaternions, taking the short way round.

    For the small angles between consecutive mocap frames this is indistinguishable from slerp
    and cheaper.
    """
    b = jnp.where((jnp.sum(a * b, axis=-1, keepdims=True) < 0), -b, b)
    q = (1 - alpha) * a + alpha * b
    return q / jnp.linalg.norm(q, axis=-1, keepdims=True)
