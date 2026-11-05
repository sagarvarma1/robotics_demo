"""DeepMimic tracking reward (Peng et al. 2018, section 5.2 / eq. 5-9).

The reward says "how close is the simulated body to the reference right now", as a weighted sum
of four terms, each of the form exp(-k * error):

    r = w_pose * exp(-k_pose * sum_b angle(q_b^sim, q_b^ref)^2)          body rotations
      + w_vel  * exp(-k_vel  * sum_j (qdot_j^sim - qdot_j^ref)^2)        joint velocities
      + w_ee   * exp(-k_ee   * sum_e |(p_e - p_root)^sim - (p_e - p_root)^ref|^2)  hands and feet
      + w_com  * exp(-k_com  * |c^sim - c^ref|^2)                         centre of mass

Why exp(-k * err). Each term is 1 at a perfect match and decays smoothly toward 0, so the reward is
bounded in [0, 1] (the weights sum to 1). Every small improvement is rewarded, even far from
the target, and one huge error can't make the reward hugely negative. `k` sets how strict a term
is: with k_ee = 40, a summed squared hand/foot error of 0.01 m^2 (e.g. 10 cm on one limb) gives
exp(-0.4) = 0.67, and 0.04 m^2 (20 cm) gives exp(-1.6) = 0.20.

Why these four terms:
  * pose (0.65) is the main signal: does each joint have the right rotation? We compare *local*
    rotations (each body relative to its parent, from MuJoCo's world `xquat`), plus the root's
    world rotation, so a wrong elbow doesn't count as wrong hands and fingers as well. Bodies
    with no joint of their own contribute exactly 0.
  * vel (0.1) asks for the right speed of motion, which a pose match alone doesn't require.
  * ee (0.15): hands and feet matter most for how a motion looks and for contact with the
    ground. Measured relative to the root, so being in the wrong place overall (already covered
    by com) isn't counted twice.
  * com (0.1) keeps the whole body moving along the reference path (world frame).

Deviations from the paper: DeepMimic's end-effector term is in world coordinates. Ours is
relative to the root (see above), with world position left to the CoM term. DeepMimic's velocity
term also covers the root's velocity. Ours covers hinge joints only, as specified in the
project brief.
"""

from __future__ import annotations

import jax.numpy as jnp

from env import quat

TERMS = ("pose", "vel", "ee", "com")


def errors(sim_local_quat, ref_local_quat, sim_qvel, ref_qvel, sim_ee, sim_root, ref_ee, ref_root,
           sim_com, ref_com, vel_mask=None):
    """The four squared errors that go inside the exponentials.

    Velocity error is the MEAN (not sum) of squared hinge-velocity differences over the joints in
    `vel_mask`, i.e. the joints that actually have mocap data. Why (docs/experiment_log.md): summed over
    all 56 hinges, even a near-perfect state gave err ≈ 33, so exp(-0.1·33) ≈ 0.04 and the term never
    produced a learning signal. Averaging makes k independent of how many joints the body has.
    """
    dq = quat.mul(quat.conj(sim_local_quat), ref_local_quat)
    e_pose = jnp.sum(quat.angle(dq) ** 2)
    dv2 = (sim_qvel[6:] - ref_qvel[6:]) ** 2
    if vel_mask is None:
        e_vel = jnp.mean(dv2)
    else:
        e_vel = jnp.sum(dv2 * vel_mask) / jnp.sum(vel_mask)
    e_ee = jnp.sum(((sim_ee - sim_root) - (ref_ee - ref_root)) ** 2)
    e_com = jnp.sum((sim_com - ref_com) ** 2)
    return {"pose": e_pose, "vel": e_vel, "ee": e_ee, "com": e_com}


def tracking_reward(errs: dict, weights, scales):
    """Weighted sum of exp(-k * err). Returns (total, {term: exp value})."""
    terms = {k: jnp.exp(-scales[k] * errs[k]) for k in TERMS}
    total = sum(weights[k] * terms[k] for k in TERMS)
    return total, terms


def local_quats(xquat, parent_ids):
    """Per-body rotation relative to its parent, for bodies 1..nbody-1 (xquat[0] is the world)."""
    return quat.mul(quat.conj(xquat[parent_ids]), xquat[1:])
