"""Eval metrics on synthetic trajectories whose answers we know (no simulation needed).

* sim == reference          -> MPJPE 0, drift 0, full survival, progress ratio 1.
* sim == reference + 10 cm  -> world MPJPE 0.10, root-relative MPJPE 0 (it's pure drift).
* errors after termination  -> ignored (masking): a fall that sends errors to 5 m must not
                               leak into the averages.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval.metrics import evaluate, per_episode
from eval.push_test import headline, wilson_interval
from eval.rollout import Rollout
from motion.format import load_motion
from env.imitation import resolve_path


def _rollout_from_clip(n_eps=3, steps=100, offset=(0.0, 0.0, 0.0), done_at=None) -> Rollout:
    m = load_motion(resolve_path("data/motions/cmu_16_35.npz"))
    T = m.num_frames
    g = np.arange(steps + 1)
    i, k = g % T, g // T
    shift = k[:, None] * np.asarray(m.cycle_offset_pos)[None]            # loop rule (yaw = 0)
    bp = m.body_pos[i] + shift[:, None, :]
    q = m.qpos[i].copy()
    q[:, :3] += shift
    N = n_eps
    body_ref = np.broadcast_to(bp, (N,) + bp.shape).astype(np.float64).copy()
    body_sim = body_ref + np.asarray(offset)
    done = np.zeros((N, steps + 1))
    if done_at is not None:
        done[0, done_at] = 1.0
        body_sim[0, done_at + 1:] += 5.0          # garbage after the fall
    alive = np.cumprod(1 - done, axis=1) > 0.5
    z = np.zeros((N, steps + 1), bool)
    return Rollout(
        dt=1.0 / m.fps, clip_duration=m.duration, cycle_offset=np.asarray(m.cycle_offset_pos),
        body_names=list(m.body_names), root_index=0, t=np.tile(g / m.fps, (N, 1)),
        qpos=np.tile(q, (N, 1, 1)), qpos_ref=np.tile(q, (N, 1, 1)), body_pos=body_sim,
        body_pos_ref=body_ref, phase=np.tile(i / T, (N, 1)), reward=np.ones((N, steps + 1)),
        done=done, fell=done.astype(bool), drifted=z, nan=z, alive=alive,
        force=np.zeros((N, steps + 1, 3)))


def test_perfect_tracking_gives_zero_error():
    r = _rollout_from_clip()
    res = evaluate(r)
    s = res["summary"]
    assert s["mpjpe"]["max"] == pytest.approx(0.0, abs=1e-9)
    assert s["mpjpe_rootrel"]["max"] == pytest.approx(0.0, abs=1e-9)
    assert s["root_drift_mean"]["max"] == pytest.approx(0.0, abs=1e-9)
    assert s["root_drift_xy_final"]["max"] == pytest.approx(0.0, abs=1e-9)
    assert s["survival_s"]["median"] == pytest.approx(100 / 50.0)
    assert res["cap_share"] == 1.0
    assert s["progress_ratio"]["median"] == pytest.approx(1.0, abs=1e-6)
    # 2.0 s at 0.8 s per stride: 2 full reference cycles, ~2.5 strides of ground
    assert s["ref_cycles"]["median"] == 2
    assert s["progress_cycles"]["median"] == pytest.approx(
        np.linalg.norm(r.body_pos[0, -1, 0, :2] - r.body_pos[0, 0, 0, :2]) / 2.179, rel=1e-3)


def test_constant_offset_is_drift_not_pose_error():
    r = _rollout_from_clip(offset=(0.06, 0.08, 0.0))     # 10 cm horizontal
    pe = per_episode(r)
    assert np.allclose(pe["mpjpe"], 0.10, atol=1e-6)
    assert np.allclose(pe["mpjpe_rootrel"], 0.0, atol=1e-9)
    assert np.allclose(pe["root_drift_xy_mean"], 0.10, atol=1e-6)


def test_errors_after_termination_are_masked():
    r = _rollout_from_clip(done_at=20)
    pe = per_episode(r)
    assert pe["survival_s"][0] == pytest.approx(20 / 50.0)
    assert pe["terminated"][0] and not pe["terminated"][1]
    assert pe["mpjpe"][0] == pytest.approx(0.0, abs=1e-9)       # the +5 m garbage is excluded
    assert pe["cause"][0] == "contact"
    assert pe["ref_cycles"][0] == 0                            # 0.4 s < one 0.8 s stride


def test_wilson_interval_and_headline():
    lo, hi = wilson_interval(0, 32)
    assert lo == 0.0 and 0.09 < hi < 0.12
    lo, hi = wilson_interval(16, 32)
    assert lo == pytest.approx(0.336, abs=0.01) and hi == pytest.approx(0.664, abs=0.01)
    assert wilson_interval(0, 0) == (None, None)
    lv = [{"force_N": f, "share": s, "n_valid": 10, "n_survived": int(s * 10)}
          for f, s in ((0, 1.0), (100, 0.8), (200, 0.2))]
    h = headline(lv)
    assert h["F50_N"] == pytest.approx(150.0)
    assert h["pooled_share_F_gt_0"] == pytest.approx(0.5)
