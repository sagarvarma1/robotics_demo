"""End-to-end checks of the CMU -> model retarget (task 7). Thresholds are loose sanity
bounds; the measured numbers are printed by `uv run python -m motion.build_cmu`."""

from pathlib import Path

import numpy as np
import pytest

from motion import build_cmu, checks
from motion import retarget as rt
from motion.format import load_model, load_motion, validate

RAW = Path(__file__).resolve().parents[1] / "data/mocap/raw"
pytestmark = pytest.mark.skipif(not (RAW / "16_35.amc").exists(), reason="CMU data not downloaded")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("motions")
    report = build_cmu.build("16_35", out_dir=out, verbose=False)
    return report, load_motion(out / "cmu_16_35.npz"), load_motion(out / "cmu_16_35_full.npz"), out


def test_files_validate(built):
    _, mc, mf, _ = built
    model = load_model()
    validate(mc, model)
    validate(mf, model)
    assert mc.cyclic and not mf.cyclic
    assert mc.fps == 50.0
    assert mc.cycle_offset_pos[2] == 0 and abs(mc.cycle_offset_pos[1]) < 1e-6
    assert "retarget" in mc.meta and "frame_range_src" in mc.meta


def test_keypoint_error(built):
    r = built[0]
    assert r["full"]["keypoint_error"]["mean_cm"] < 1.5
    assert r["full"]["keypoint_error"]["max_cm"] < 5.0
    # the cyclic clip deliberately departs from the source to close the loop seam
    # (correction spread over the flight phases), so its bound is looser
    assert r["cyclic"]["keypoint_error"]["mean_cm"] < 2.5
    assert r["cyclic"]["keypoint_error"]["max_cm"] < 8.0


def test_joint_limits(built):
    r = built[0]
    assert r["full"]["limits"]["n_violations"] == 0
    assert r["cyclic"]["limits"]["n_violations"] == 0


def test_feet(built):
    for k in ("full", "cyclic"):
        st = built[0][k]["stance"]
        assert len(st["stances"]) >= 2
        assert st["min_foot_geom_z_cm"] > -1.5
        for s in st["stances"]:
            assert abs(s["height_mean_cm"]) < 2.5
            assert s["toe_slide_cm"] < 5.0
        assert {s["foot"] for s in st["stances"]} == {"left", "right"}


def test_seam_small(built):
    sm = built[0]["cyclic"]["seam"]
    assert sm["pose_jump_rad"] <= sm["pose_step_max_rad"]
    assert sm["vel_jump"] <= sm["vel_step_max"]


def test_speed_is_running(built):
    for k in ("full", "cyclic"):
        assert 2.0 < built[0][k]["forward_speed_mps"] < 6.0


def test_close_to_direct_copy_on_legs(built):
    rms = built[0]["direct_copy_rms_deg"]
    for side in "lr":
        for j in ("femurrx", "femurry", "femurrz", "tibiarx"):
            assert rms[side + j] < 5.0, (side + j, rms[side + j])


def test_upright_and_not_mirrored(built):
    _, mc, _, _ = built
    n = mc.body_names
    z = lambda b: mc.body_pos[:, n.index(b), 2]
    y = lambda b: mc.body_pos[:, n.index(b), 1]
    assert (z("head") > z("root") + 0.3).all() and (z("root") > z("lfoot") + 0.4).all()
    # running along +x means the left hip is on the +y side
    assert (y("lfemur") > y("rfemur")).all()
    assert mc.cycle_offset_pos[0] > 0


def test_keypoint_sidecar_aligned(built):
    _, mc, _, out = built
    kp = np.load(out / "cmu_16_35_keypoints.npz")
    assert kp["pos"].shape[0] == mc.num_frames
    names = list(kp["names"])
    model = load_model()
    err = checks.keypoint_error_cm(model, mc.qpos.astype(float), kp["pos"], names,
                                   dict(zip(names, kp["body"])))
    assert err["mean_cm"] < 2.5


def test_scaling_gives_model_lengths():
    from motion import cmu_amc
    skel, vals, fk = cmu_amc.load_cmu(RAW / "16.asf", RAW / "16_35.amc")
    src = cmu_amc.to_source_motion(skel, fk)
    model = load_model()
    scaled, info = rt.scale_source(src, model)
    Lm = rt.model_segment_lengths(model, src)
    for j, p in enumerate(src.parents):
        if p >= 0:
            L = np.linalg.norm(scaled.pos[:, j] - scaled.pos[:, p], axis=-1)
            np.testing.assert_allclose(L, Lm[j], atol=1e-9)
    assert 0.8 < info["leg_ratio"] < 1.2
