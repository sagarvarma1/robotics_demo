"""The push really reaches the physics.

Same start state, same open-loop (reference PD) actions; one rollout gets a 400 N push along +y on
the pelvis for 0.1 s. Momentum bookkeeping: J = 400 N * 0.1 s = 40 N s on a 51.8 kg body is
~0.77 m/s if the whole body moved together. The pelvis is pushed directly and the rest is
dragged along through the joints, so its sideways velocity change should be clearly positive
(we ask for > 0.3 m/s) and point in +y, while the unpushed run hardly moves sideways.
"""

from __future__ import annotations

import numpy as np

from eval.policy import load_baseline
from eval.push_test import PushConfig, score
from eval.rollout import PushPlan, rollout


def test_push_changes_root_velocity_in_push_direction():
    env, pol, _ = load_baseline("refpd")
    start, n = 3, 5
    plan = PushPlan(start_step=np.array([start, start]),
                    force=np.array([[0.0, 400.0, 0.0], [0.0, 0.0, 0.0]]), n_steps=n, body="root")
    r = rollout(env, pol, 2, 0.4, seed=0, start_times=0.0, push=plan, tail_s=0.0)
    # identical before the push
    assert np.allclose(r.qpos[0, :start + 1], r.qpos[1, :start + 1], atol=1e-5)
    # force recorded during the push window only
    assert np.allclose(r.force[0, start + 1:start + n + 1, 1], 400.0)
    assert np.allclose(r.force[0, :start + 1], 0.0) and np.allclose(r.force[0, start + n + 1:], 0.0)
    k = start + n                                    # state right after the push
    v = (r.qpos[:, k + 1, :3] - r.qpos[:, k - 1, :3]) / (2 * r.dt)   # root velocity, central diff
    dv = v[0] - v[1]
    print(f"root velocity change from push: {dv} m/s")
    assert dv[1] > 0.3                               # along the push (+y)
    assert abs(dv[1]) > 3 * abs(dv[0])               # mostly in the push direction
    assert r.alive[:, k].all()


def test_push_scoring_valid_and_survived():
    """Scoring logic on a hand-made rollout-like object: invalid if dead before the push."""
    from types import SimpleNamespace
    T = 20
    done = np.zeros((3, T + 1))
    done[0, 2] = 1      # died before push (start=5) -> invalid
    done[1, 8] = 1      # died during recovery -> failed
    r = SimpleNamespace(push=PushPlan(start_step=np.array([5, 5, 5]), force=np.zeros((3, 3)), n_steps=2),
                        dt=0.1, done=done, fell=done.astype(bool), nan=np.zeros_like(done, bool))
    s = score(r, PushConfig(recovery_s=1.0))
    assert s["valid"].tolist() == [False, True, True]
    assert s["survived"].tolist() == [False, False, True]
