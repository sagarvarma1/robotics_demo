"""Environment tests: reset, reference looping, termination, batching, PD sanity.

Some tests print measurements (zero-action survival, PD tracking error). Run with `-s` to see
them:  uv run pytest tests/test_env.py -s
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mujoco import mjx

from env import make_env, quat
from env import termination as term_lib
from env.reference import Reference
from motion.format import from_qpos_qvel, load_model, save_motion
from tests.fixtures_motion import march_clip, standing_clip, standing_qpos


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    d = tmp_path_factory.mktemp("motions")
    out = {}
    for name, clip in (("march", march_clip()), ("stand", standing_clip())):
        out[name] = d / f"{name}.npz"
        save_motion(out[name], clip)
    return out


@pytest.fixture(scope="module")
def env(paths):
    return make_env({"motion_path": str(paths["march"])})


@pytest.fixture(scope="module")
def evaluate(env):
    @jax.jit
    def f(data):
        data = mjx.forward(env.mjx_model, data)
        return env._evaluate(data)
    return f


# -- reset ---------------------------------------------------------------------------------------
def test_reset_matches_reference(env):
    reset = jax.jit(env.reset)
    m = env.motion
    for seed in range(4):
        s = reset(jax.random.PRNGKey(seed))
        i = int(s.info["ref_index"])
        assert int(s.info["loop"]) == 0 and int(s.info["t"]) == 0
        assert np.allclose(s.data.qpos, m.qpos[i], atol=1e-6)
        assert np.allclose(s.data.qvel, m.qvel[i], atol=1e-6)
        assert float(s.info["phase"]) == pytest.approx(i / m.num_frames, abs=1e-5)
        assert np.allclose(s.data.xpos[1:], m.body_pos[i], atol=1e-5)
        assert float(s.metrics["mpjpe"]) < 1e-4


# -- reference lookup ----------------------------------------------------------------------------
def test_loop_rule_shifts_by_cycle_offset(env):
    ref = env.ref
    for delta in (0.0, 0.1, 0.42):
        a, b = ref.at(delta), ref.at(ref.duration + delta)
        assert int(b.loop) == int(a.loop) + 1 and int(b.index) == int(a.index)
        off = np.asarray(env.motion.cycle_offset_pos)
        assert np.allclose(b.qpos[:3], a.qpos[:3] + off, atol=1e-5)
        assert np.allclose(b.qpos[3:], a.qpos[3:], atol=1e-6)
        assert np.allclose(b.body_pos, a.body_pos + off, atol=1e-5)
        assert np.allclose(b.ee_pos, a.ee_pos + off, atol=1e-5)
        assert np.allclose(b.com, a.com + off, atol=1e-5)
        # two loops later
        c = ref.at(2 * ref.duration + delta)
        assert np.allclose(c.qpos[:3], a.qpos[:3] + 2 * off, atol=1e-5)


def test_loop_rule_with_yaw():
    model = load_model()
    m = dataclasses.replace(march_clip(model=model), cycle_offset_yaw=0.3)
    ref = Reference(m, model, ctrl_dt=0.02)
    delta = 0.2
    a, b = ref.at(delta), ref.at(m.duration + delta)
    R = np.asarray(quat.rot_z(jnp.float32(0.3)))
    assert np.allclose(b.qpos[:3], R @ np.asarray(a.qpos[:3]) + m.cycle_offset_pos, atol=1e-5)
    assert np.allclose(b.qpos[3:7], quat.mul(quat.from_yaw(jnp.float32(0.3)), a.qpos[3:7]), atol=1e-6)
    assert np.allclose(b.body_pos, np.asarray(a.body_pos) @ R.T + m.cycle_offset_pos, atol=1e-5)


def test_interpolation_when_fps_differs():
    model = load_model()
    m = march_clip(num_frames=25, fps=25.0, model=model)
    ref = Reference(m, model, ctrl_dt=0.02)
    assert ref.interpolate
    t = 2.5 / 25.0                                 # halfway between frames 2 and 3
    fr = ref.at(t)
    assert np.allclose(fr.qpos[7:], 0.5 * (m.qpos[2, 7:] + m.qpos[3, 7:]), atol=1e-5)
    # across the loop boundary: halfway between frame T-1 and frame 0 of the next loop
    fr = ref.at((m.num_frames - 0.5) / 25.0)
    expect = 0.5 * (m.qpos[-1, :3] + m.qpos[0, :3] + m.cycle_offset_pos)
    assert np.allclose(fr.qpos[:3], expect, atol=1e-5)


# -- termination ---------------------------------------------------------------------------------
def test_torso_on_floor_terminates(env, evaluate):
    s = env.reset_to_time(jnp.float32(0.0), jax.random.PRNGKey(0))
    q = jnp.asarray(standing_qpos(env.mj_model), jnp.float32)
    # identity root rotation = lying on the back (root +y "up" axis along world +y), torso at floor
    q = q.at[3:7].set(jnp.array([1.0, 0, 0, 0])).at[2].set(0.05)
    _, _, _, done, _, flags = evaluate(s.data.replace(qpos=q, qvel=jnp.zeros_like(s.data.qvel)))
    assert bool(flags["contact"]) and float(done) == 1.0


def test_feet_only_contact_does_not_terminate(env, evaluate):
    s = env.reset_to_time(jnp.float32(0.0), jax.random.PRNGKey(0))
    q = jnp.asarray(standing_qpos(env.mj_model), jnp.float32).at[2].add(-0.005)  # feet 5 mm in
    data = mjx.forward(env.mjx_model, s.data.replace(qpos=q, qvel=jnp.zeros_like(s.data.qvel)))
    c = term_lib.contact_buffer(data)
    geom_body = env.mj_model.geom_bodyid[np.asarray(c.geom)]
    touching = np.asarray(c.dist) < 0
    feet = {env.mj_model.body(n).id for n in term_lib.DEFAULT_ALLOWED}
    assert touching.any(), "fixture should have feet touching the floor"
    assert all(set(b) & feet for b in geom_body[touching])
    _, _, _, _, _, flags = evaluate(data)
    assert not bool(flags["contact"])


def test_root_drift_terminates(env, evaluate):
    s = env.reset_to_time(jnp.float32(0.0), jax.random.PRNGKey(0))
    _, _, _, done, m, flags = evaluate(s.data.replace(qpos=s.data.qpos.at[0].add(0.6)))
    assert bool(flags["drift"]) and float(done) == 1.0
    assert float(m["root_drift"]) == pytest.approx(0.6, abs=1e-4)


# -- batching and rollouts -----------------------------------------------------------------------
def test_jit_vmap_8_envs(env):
    n = 8
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    s = reset(jax.random.split(jax.random.PRNGKey(1), n))
    assert s.obs.shape == (n, env.observation_size)
    a = jax.random.uniform(jax.random.PRNGKey(2), (n, env.action_size), minval=-1, maxval=1)
    for _ in range(5):
        s = step(s, a)
        assert s.obs.shape == (n, env.observation_size)
        assert np.isfinite(np.asarray(s.obs)).all() and np.isfinite(np.asarray(s.reward)).all()
        for k in ("r_pose", "r_vel", "r_ee", "r_com", "mpjpe", "root_drift", "loops"):
            assert s.metrics[k].shape == (n,)
    assert set(s.info) >= {"phase", "t", "loop", "ref_index", "rng"}
    assert np.all(np.asarray(s.info["t"]) == 5)


def _steps_until_done(env, action_fn, t0=0.0, max_steps=500):
    step = jax.jit(env.step)
    s = env.reset_to_time(jnp.float32(t0), jax.random.PRNGKey(0))
    for i in range(max_steps):
        s = step(s, action_fn(s))
        if float(s.done):
            return i + 1, s
    return max_steps, s


def test_zero_action_rollout(paths):
    for name in ("stand", "march"):
        e = make_env({"motion_path": str(paths[name])})
        n, s = _steps_until_done(e, lambda s: jnp.zeros(e.action_size))
        print(f"\n[zero action, {name} clip] terminated after {n} steps ({n * e.dt:.2f} s); "
              f"root_drift={float(s.metrics['root_drift']):.3f}")
        n2, s2 = _steps_until_done(e, lambda s, e=e: e.pose_to_action(e.ref.at(s.data.time).qpos[7:]))
        print(f"[ref-pose PD targets, {name} clip] terminated after {n2} steps ({n2 * e.dt:.2f} s)")
        assert n >= 1


def test_pose_to_action_roundtrip(env):
    q = jnp.asarray(env.motion.qpos[7, 7:])
    assert np.allclose(env.action_to_targets(env.pose_to_action(q)), q, atol=1e-5)


def test_pd_holds_pose_with_root_fixed(env):
    """Root pinned in the air (qpos/qvel reset every physics substep), gravity on, no floor contact.

    Start standing and command a random pose well inside the joint ranges (mid +- 30% of the half
    range, so joint limits don't do the holding), then measure tracking error after 0.1 s and 1 s.
    """
    m = env.mjx_model
    q_start = jnp.asarray(standing_qpos(env.mj_model), jnp.float32).at[2].add(1.0)
    lo, hi = env.mj_model.jnt_range[1:, 0], env.mj_model.jnt_range[1:, 1]
    u = np.random.default_rng(0).uniform(-0.3, 0.3, 56)
    target = jnp.asarray(0.5 * (lo + hi) + u * 0.5 * (hi - lo), jnp.float32)
    ctrl = env.pose_to_action(target)
    s = env.reset_to_time(jnp.float32(0.0), jax.random.PRNGKey(0))
    d0 = s.data.replace(qpos=q_start, qvel=jnp.zeros_like(s.data.qvel), ctrl=ctrl)

    def run(d, n):
        def body(d, _):
            d = mjx.step(m, d.replace(ctrl=ctrl))
            d = d.replace(qpos=d.qpos.at[:7].set(q_start[:7]), qvel=d.qvel.at[:6].set(0.0))
            return d, None
        return jax.lax.scan(body, d, None, length=n)[0]

    run = jax.jit(run, static_argnums=1)
    names = env.motion.joint_names
    for n in (20, 200):                                         # 0.1 s and 1 s
        d = run(d0, n)
        err = np.abs(np.asarray(d.qpos[7:] - target))
        worst = np.argsort(err)[::-1][:4]
        print(f"\n[PD hold, root fixed, {n * 0.005:.1f} s] mean |err| = {np.degrees(err.mean()):.2f} deg, "
              f"max = {np.degrees(err.max()):.2f} deg; worst: "
              + ", ".join(f"{names[j]} {np.degrees(err[j]):.1f}" for j in worst))
    assert np.degrees(err.mean()) < 2.0 and np.degrees(err.max()) < 5.0
