"""Reward tests: perfect tracking scores 1, worse tracking scores less."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mujoco import mjx

from env import make_env, quat
from env import reward as reward_lib
from motion.format import save_motion
from tests.fixtures_motion import march_clip


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    p = tmp_path_factory.mktemp("motions") / "march.npz"
    save_motion(p, march_clip())
    return make_env({"motion_path": str(p)})


@pytest.fixture(scope="module")
def evaluate(env):
    @jax.jit
    def f(data):
        data = mjx.forward(env.mjx_model, data)
        _, _, r, done, metrics, flags = env._evaluate(data)
        return r, metrics
    return f


def test_quat_helpers():
    q = quat.from_yaw(jnp.array(0.7))
    assert np.isclose(float(quat.angle(q)), 0.7, atol=1e-6)
    assert np.allclose(quat.mul(q, quat.conj(q)), [1, 0, 0, 0], atol=1e-6)
    R = quat.to_mat(q)
    assert np.allclose(R, quat.rot_z(jnp.array(0.7)), atol=1e-6)


def test_zero_error_gives_sum_of_weights():
    w = {"pose": 0.65, "vel": 0.1, "ee": 0.15, "com": 0.1}
    k = {"pose": 2.0, "vel": 0.1, "ee": 40.0, "com": 10.0}
    total, terms = reward_lib.tracking_reward({t: jnp.zeros(()) for t in reward_lib.TERMS}, w, k)
    assert np.isclose(float(total), 1.0)
    assert all(np.isclose(float(v), 1.0) for v in terms.values())


@pytest.mark.parametrize("t0", [0.0, 0.26, 0.5, 0.98])
def test_reward_is_one_on_reference(env, evaluate, t0):
    s = env.reset_to_time(jnp.float32(t0), jax.random.PRNGKey(0))
    r, m = evaluate(s.data)
    assert float(r) == pytest.approx(1.0, abs=1e-4)
    for term in ("r_pose", "r_vel", "r_ee", "r_com"):
        assert float(m[term]) == pytest.approx(1.0, abs=1e-4), term
    assert float(m["mpjpe"]) < 1e-4
    assert float(m["root_drift"]) < 1e-4


def test_reward_decreases_with_joint_perturbation(env, evaluate):
    s = env.reset_to_time(jnp.float32(0.3), jax.random.PRNGKey(0))
    rng = np.random.default_rng(0)
    direction = jnp.asarray(rng.standard_normal(56), jnp.float32)
    direction = direction / jnp.linalg.norm(direction)
    rewards, poses = [], []
    for eps in (0.0, 0.05, 0.1, 0.2, 0.4, 0.8):
        q = s.data.qpos.at[7:].add(eps * direction)
        r, m = evaluate(s.data.replace(qpos=q))
        rewards.append(float(r))
        poses.append(float(m["r_pose"]))
    print("\n[reward vs joint perturbation 0..0.8 rad]", np.round(rewards, 4))
    assert np.all(np.diff(rewards) < 0), rewards
    assert np.all(np.diff(poses) < 0), poses


def test_reward_decreases_with_velocity_error(env, evaluate):
    s = env.reset_to_time(jnp.float32(0.3), jax.random.PRNGKey(0))
    rs = []
    for eps in (0.0, 0.5, 1.0, 2.0):
        r, _ = evaluate(s.data.replace(qvel=s.data.qvel.at[6:].add(eps)))
        rs.append(float(r))
    assert np.all(np.diff(rs) < 0), rs
