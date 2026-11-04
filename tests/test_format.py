import numpy as np
import pytest

from motion import format as mf


def _standing_clip(T=10):
    model = mf.load_model()
    qpos = np.tile(model.qpos0, (T, 1)).astype(np.float32)
    # clamp hinge angles into their limits (qpos0 = 0 is outside some ranges, e.g. tibia [1, 170] deg)
    lo, hi = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
    qpos[:, 7:] = np.clip(qpos[:, 7:], lo, hi)
    qpos[:, 0] += np.linspace(0, 1, T)  # move forward
    qvel = np.zeros((T, model.nv), np.float32)
    return mf.from_qpos_qvel(qpos, qvel, 50.0, cyclic=False, source="test", model=model)


def test_roundtrip(tmp_path):
    m = _standing_clip()
    p = tmp_path / "m.npz"
    mf.save_motion(p, m)
    m2 = mf.load_motion(p)
    np.testing.assert_allclose(m2.body_pos, m.body_pos)
    assert m2.body_names == m.body_names and m2.joint_names == m.joint_names
    assert m2.ee_pos.shape == (10, 4, 3)


def test_rejects_limit_violation():
    m = _standing_clip()
    m.qpos[3, 7] = 10.0
    with pytest.raises(ValueError, match="outside its limits"):
        mf.validate(m)


def test_rejects_nonunit_quat():
    m = _standing_clip()
    m.qpos[0, 3:7] = [2, 0, 0, 0]
    with pytest.raises(ValueError, match="unit"):
        mf.validate(m)
