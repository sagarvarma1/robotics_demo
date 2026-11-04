"""Build reference motions from a CMU AMC clip.

    uv run python -m motion.build_cmu                     # 16_35 -> data/motions/cmu_16_35{,_full}.npz
    uv run python -m motion.build_cmu --trial 16_36

Outputs (data/motions/):
  cmu_<trial>.npz                 cyclic single stride (the training reference)
  cmu_<trial>_full.npz            the whole clip, not cyclic
  cmu_<trial>[_full]_keypoints.npz  scaled source keypoints, frame-aligned with the motion
                                  (for playback overlays and checks; not part of the contract)
and a check report at out/motion_check/report_<trial>.json.

Order of operations, and why this order:
  parse + FK (120 fps) -> rotate so the run goes along +x -> scale + IK (120 fps, the
  densest data, so IK warm starts are tiny steps) -> resample to 50 fps
  full clip:   smooth -> ground fix -> velocities
  cyclic clip: find seam -> cut + blend -> smooth with wrap-around -> ground fix -> velocities
Smoothing comes after the cut so the filter can wrap around the seam and smooth it
like any other frame. The ground fix comes after smoothing because smoothing can move
the feet by a millimetre or two.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from motion import checks, cmu_amc, cycle
from motion import retarget as rt
from motion.format import from_qpos_qvel, load_model, save_motion, validate

ROOT = Path(__file__).resolve().parents[1]


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _save_keypoints(path: Path, src: rt.SourceMotion, pos: np.ndarray, fps: float):
    np.savez_compressed(path, names=np.array(src.names), parents=np.array(src.parents),
                        body=np.array([src.body_map[n] for n in src.names]),
                        pos=pos.astype(np.float32), fps=fps)


def build(trial: str = "16_35", subject: str | None = None, raw_dir: Path = ROOT / "data/mocap/raw",
          out_dir: Path = ROOT / "data/motions", cfg: rt.RetargetConfig | None = None,
          verbose: bool = True) -> dict:
    cfg = cfg or rt.RetargetConfig()
    subject = subject or trial.split("_")[0]
    model = load_model()
    asf, amc = raw_dir / f"{subject}.asf", raw_dir / f"{trial}.amc"
    skel, vals, fk = cmu_amc.load_cmu(asf, amc)
    src = cmu_amc.to_source_motion(skel, fk, source=f"cmu:{trial}")
    T_raw = src.num_frames

    # 1. heading: travel along +x, start at the origin
    d = src.pos[-1, 0, :2] - src.pos[0, 0, :2]
    yaw0 = -float(np.arctan2(d[1], d[0]))
    src = src.rotated_about_z(yaw0)
    src = src.rotated_about_z(0.0, shift_xy=-src.pos[0, 0, :2])

    # 2-3. scale + IK at the source rate
    res = rt.retarget(src, cfg, model)
    hinge_names = [model.joint(j).name for j in range(1, model.njnt)]
    direct = cmu_amc.direct_angle_copy(skel, vals, hinge_names)
    rms = checks.direct_copy_rms_deg(model, res.qpos_src_fps, direct)

    # 4. resample to the policy rate
    fps = cfg.fps_out
    q50 = rt.resample(res.qpos_src_fps, src.fps, fps, model)
    kp50 = rt.resample_points(res.scaled.pos, src.fps, fps)[: q50.shape[0]]

    base_meta = {"retarget": cfg.to_dict(), "retarget_info": res.info, "git_sha": _git_sha(),
                 "source_files": [str(asf.relative_to(ROOT)), str(amc.relative_to(ROOT))],
                 "source_fps": src.fps, "source_frames": T_raw, "heading_yaw_applied_rad": yaw0,
                 "keypoints": src.names}
    report = {"trial": trial, "leg_ratio": res.info["leg_ratio"],
              "ik_iters_mean": res.info["ik_iters_mean"], "direct_copy_rms_deg": rms}

    # ---- full clip
    qf = rt.smooth(q50, model, cfg)
    qf, gi = rt.ground_fix(model, qf, fps, cfg)
    vf = rt.finite_diff_qvel(model, qf, fps)
    kpf = kp50.copy()
    kpf[..., 2] += gi["ground_shift_m"]
    meta_f = {**base_meta, "ground": gi, "frame_range_50fps": [0, int(qf.shape[0])],
              "frame_range_src": [0, T_raw]}
    mf = from_qpos_qvel(qf, vf, fps, cyclic=False, source=f"cmu:{trial}", meta=meta_f, model=model)
    out_dir.mkdir(parents=True, exist_ok=True)
    p_full = out_dir / f"cmu_{trial}_full.npz"
    save_motion(p_full, mf)
    _save_keypoints(out_dir / f"cmu_{trial}_full_keypoints.npz", src, kpf, fps)
    report["full"] = {
        "T": int(qf.shape[0]), "fps": fps, **checks.speed_report(qf, fps),
        "keypoint_error": checks.keypoint_error_cm(model, qf, kpf, src.names, src.body_map),
        "limits": checks.joint_limit_violations(model, qf),
        "stance": checks.stance_report(model, qf, fps, cfg), "ground": gi}

    # ---- cyclic stride
    i, j, cost = cycle.find_cycle(model, q50, fps, driven_mask=~rt.undriven_hinge_mask(model))
    flight = cycle.flight_mask(model, q50[i:j], fps)
    qc, offset, ci = cycle.make_cyclic(q50, i, j, flight=flight)
    kpc = cycle.transform_points(kp50[i:j], ci)
    ci.pop("_points")
    qc = rt.smooth(qc, model, cfg, cyclic=True, offset_xy=offset)
    qc, gc = rt.ground_fix(model, qc, fps, cfg, cyclic=True, offset_xy=offset)
    kpc[..., 2] += gc["ground_shift_m"]
    vc = rt.finite_diff_qvel(model, qc, fps, cyclic=True, offset_xy=offset)
    meta_c = {**base_meta, "ground": gc, "cycle": {**ci, "seam_cost": cost},
              "frame_range_50fps": [int(i), int(j)],
              "frame_range_src": [int(round(i * src.fps / fps)), int(round(j * src.fps / fps))]}
    mc = from_qpos_qvel(qc, vc, fps, cyclic=True, cycle_offset_pos=offset, cycle_offset_yaw=0.0,
                        source=f"cmu:{trial}", meta=meta_c, model=model)
    p_cyc = out_dir / f"cmu_{trial}.npz"
    save_motion(p_cyc, mc)
    _save_keypoints(out_dir / f"cmu_{trial}_keypoints.npz", src, kpc, fps)
    report["cyclic"] = {
        "T": int(qc.shape[0]), "fps": fps, "duration_s": qc.shape[0] / fps,
        "cut_frames_50fps": [int(i), int(j)], "cycle_offset_pos": offset.tolist(),
        "cycle_offset_yaw": 0.0, **ci, **checks.speed_report(qc, fps, offset),
        "keypoint_error": checks.keypoint_error_cm(model, qc, kpc, src.names, src.body_map),
        "limits": checks.joint_limit_violations(model, qc),
        "stance": checks.stance_report(model, qc, fps, cfg, cyclic=True, offset_xy=offset),
        "seam": checks.seam_report(model, qc, vc, offset), "ground": gc}
    validate(mf, model)
    validate(mc, model)

    rep_dir = ROOT / "out/motion_check"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / f"report_{trial}.json").write_text(json.dumps(report, indent=2))
    if verbose:
        print_report(report)
        print(f"wrote {p_cyc}, {p_full}, {rep_dir / f'report_{trial}.json'}")
    return report


def print_report(r: dict) -> None:
    print(f"== {r['trial']}: leg ratio {r['leg_ratio']:.3f}, IK iters/frame {r['ik_iters_mean']:.1f}")
    for k in ("full", "cyclic"):
        c = r[k]
        ke = c["keypoint_error"]
        print(f"-- {k}: T={c['T']} @ {c['fps']:.0f} fps, speed {c['forward_speed_mps']:.2f} m/s")
        print(f"   keypoint error mean {ke['mean_cm']:.2f} cm, max {ke['max_cm']:.2f} cm")
        worst = sorted(ke["per_keypoint"].items(), key=lambda x: -x[1]["mean"])[:4]
        print("   worst: " + ", ".join(f"{n} {v['mean']:.1f}/{v['max']:.1f}" for n, v in worst))
        print(f"   joint-limit violations: {c['limits']['n_violations']}")
        st = c["stance"]
        print(f"   ground shift {c['ground']['ground_shift_m'] * 100:.1f} cm; "
              f"min foot geom z {st['min_foot_geom_z_cm']:.2f} cm")
        for s in st["stances"]:
            print(f"   stance {s['foot']:5s} @{s['start']:3d} ({s['frames']:2d} fr): height mean "
                  f"{s['height_mean_cm']:.2f} cm, ankle slide {s['ankle_slide_cm']:.1f} cm, "
                  f"toe slide {s['toe_slide_cm']:.1f} cm")
        if k == "cyclic":
            sm = c["seam"]
            print(f"   cut {c['cut_frames_50fps']}, offset {np.round(c['cycle_offset_pos'], 3)}, "
                  f"raw yaw mismatch {c['seam_raw_yaw_mismatch_deg']:.1f} deg")
            print(f"   seam pose jump {sm['pose_jump_rad']:.3f} rad (median step "
                  f"{sm['pose_step_median_rad']:.3f}, max {sm['pose_step_max_rad']:.3f}); "
                  f"vel jump {sm['vel_jump']:.2f} (median {sm['vel_step_median']:.2f}, "
                  f"max {sm['vel_step_max']:.2f})")
    rms = r["direct_copy_rms_deg"]
    big = {k: v for k, v in rms.items() if not k.startswith(rt.UNDRIVEN_PREFIXES)}
    print("   IK vs direct angle copy, RMS deg (driven joints): " +
          ", ".join(f"{k} {v:.1f}" for k, v in big.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trial", default="16_35")
    ap.add_argument("--subject", default=None)
    args = ap.parse_args()
    build(args.trial, args.subject)


if __name__ == "__main__":
    main()
