"""ASF/AMC parser + FK cross-checked against MuJoCo FK of a dm_control-style angle copy."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from motion import cmu_amc
from motion.format import hinge_joint_names, load_model

RAW = Path(__file__).resolve().parents[1] / "data/mocap/raw"
pytestmark = pytest.mark.skipif(not (RAW / "16_35.amc").exists(), reason="CMU data not downloaded")


def test_parse():
    skel, vals, fk = cmu_amc.load_cmu(RAW / "16.asf", RAW / "16_35.amc")
    assert vals["root"].shape == (162, 6)
    assert skel.bones["lfemur"].dof == ["rx", "ry", "rz"]
    assert skel.bones["ltibia"].parent == "lfemur"
    assert abs(skel.bones["lfemur"].length - 7.03469 * 0.0254 / 0.45) < 1e-9
    # every ASF dof is a hinge of our model
    assert set(skel.dof_names()) <= set(hinge_joint_names(load_model()))


def test_fk_matches_model_with_direct_copy():
    skel, vals, fk = cmu_amc.load_cmu(RAW / "16.asf", RAW / "16_35.amc")
    model = load_model()
    data = mujoco.MjData(model)
    q = cmu_amc.direct_angle_copy(skel, vals, hinge_joint_names(model))
    for t in (0, 80, 161):
        data.qpos[:] = q[t]
        mujoco.mj_kinematics(model, data)
        for i, b in enumerate(fk.names):
            bid = model.body(b).id
            R = fk.seg_rot[t, i].T @ data.xmat[bid].reshape(3, 3)
            if "thumb" in b:  # dm_control's thumb bodies use their own frame (~20-30 deg off)
                continue
            assert Rotation.from_matrix(R).magnitude() < np.radians(1.0), b
            # positions differ only by subject-vs-model bone lengths
            assert np.linalg.norm(fk.start[t, i] - data.xpos[bid]) < 0.10, b
        np.testing.assert_allclose(fk.start[t, 0], data.xpos[model.body("root").id], atol=1e-9)
