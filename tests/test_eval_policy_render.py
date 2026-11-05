"""Policy loading from the tiny CPU run, and the renderer's frame count.

The tiny run (`runs/mocap_run_v1-tiny-*`, 30k steps) is untrained, but it's a real Brax checkpoint
written by train/brax_ppo.py, so it tests the whole loading path: config.yaml -> env, checkpoint
network config -> network, orbax params -> inference function.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

from eval.policy import jit_obs_act, load_run
from eval.render import count_frames, render_episode
from eval.rollout import rollout

ROOT = Path(__file__).resolve().parents[1]
TINY = sorted(ROOT.glob("runs/mocap_run_v1-tiny-*"))


@pytest.mark.skipif(not TINY, reason="no tiny run directory (train with --tiny first)")
def test_load_tiny_checkpoint_gives_finite_actions(tmp_path):
    env, pol, _ = load_run(TINY[-1])
    assert pol.info["step"] > 0
    act = jit_obs_act(pol)
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    a = np.asarray(act(state.obs, jax.random.PRNGKey(1)))
    assert a.shape == (env.action_size,) == (56,)
    assert np.all(np.isfinite(a)) and np.all(np.abs(a) <= 1.0)
    # batched obs work too, and the deterministic policy is deterministic
    obs = np.stack([state.obs, state.obs])
    b = np.asarray(act(obs, jax.random.PRNGKey(2)))
    assert b.shape == (2, 56) and np.allclose(b[0], a, atol=1e-6)

    # a short rollout + render: frame count = recorded steps + 1 (reset frame)
    r = rollout(env, pol, 2, 5 * env.dt, seed=0, chunk=5)
    assert r.n_steps == 5 and np.all(np.isfinite(r.qpos))
    info = render_episode(env.mj_model, r, 0, tmp_path / "sbs.mp4", mode="side", tail_s=None,
                          width=160, height=128)
    assert info["frames"] == 6
    assert count_frames(tmp_path / "sbs.mp4") == 6
    # left-panel hook (user video): 3 panels wide
    import imageio.v2 as imageio
    fr = [np.full((90, 60, 3), 40 * i, np.uint8) for i in range(4)]
    render_episode(env.mj_model, r, 0, tmp_path / "three.mp4", mode="side", tail_s=None,
                   width=160, height=128, left_frames=fr, left_phases=np.linspace(0, 1, 4, endpoint=False))
    img = imageio.get_reader(tmp_path / "three.mp4").get_data(0)
    assert img.shape[1] == 3 * 160
