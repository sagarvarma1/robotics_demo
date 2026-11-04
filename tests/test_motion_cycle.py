import numpy as np
from scipy.spatial.transform import Rotation

from motion import cycle
from motion.format import load_model
from motion.retarget import finite_diff_qvel, neutral_qpos


def test_heading_of_default_pose_is_minus_y():
    # root quat = Rx(90): CMU forward (+z, y-up) becomes world -y
    q = Rotation.from_euler("x", 90, degrees=True).as_quat(scalar_first=True)
    assert np.isclose(cycle.heading_yaw(q), -np.pi / 2)


def test_rotate_about_z_rotates_heading():
    model = load_model()
    q = np.tile(model.qpos0, (3, 1))
    q2 = cycle.rotate_qpos_about_z(q, 0.3)
    np.testing.assert_allclose(cycle.heading_yaw(q2[:, 3:7]) - cycle.heading_yaw(q[:, 3:7]), 0.3)


def test_cyclic_qvel_constant_motion():
    model = load_model()
    T, v = 20, 3.0
    q = np.tile(neutral_qpos(model), (T, 1))
    q[:, 0] = np.arange(T) * v / 50
    qv = finite_diff_qvel(model, q, 50.0, cyclic=True, offset_xy=(T * v / 50, 0))
    np.testing.assert_allclose(qv[:, 0], v, atol=1e-9)
    np.testing.assert_allclose(qv[:, 1:], 0, atol=1e-9)
