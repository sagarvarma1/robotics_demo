"""Kinematic playback of a reference motion: no physics, just "pose the puppet" per frame.

    uv run python -m motion.playback data/motions/cmu_16_35.npz --loops 3 --keypoints
    uv run python -m motion.playback data/motions/cmu_16_35_full.npz --stills 4

For each frame we write qpos into MjData and call ``mj_forward`` (which computes body
poses from joint angles; nothing is simulated), then render offscreen with
``mujoco.Renderer``. A side camera follows the pelvis.

Why look at it at all: numbers can say "error 1 cm" while the character runs with its
knees bent backwards or its legs swapped. A video catches that at once.

``--loops N`` replays a cyclic clip N times using the looping rule from
docs/interfaces.md (position += k * cycle_offset_pos, heading += k * cycle_offset_yaw),
so you can see whether the seam is visible. ``--keypoints`` draws the scaled source
joints as small red spheres, so you can see how closely the model follows the source
(retarget fidelity).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from motion.format import Motion, load_model, load_motion

ROOT = Path(__file__).resolve().parents[1]


def looped_qpos(m: Motion, loops: int) -> np.ndarray:
    """Unroll a cyclic clip ``loops`` times (looping rule from docs/interfaces.md)."""
    if not m.cyclic or loops <= 1:
        return m.qpos.astype(np.float64)
    out = []
    for k in range(loops):
        q = m.qpos.astype(np.float64).copy()
        yaw = k * m.cycle_offset_yaw
        Rz = Rotation.from_euler("z", yaw)
        q[:, :2] = Rz.apply(np.c_[q[:, :2], np.zeros(len(q))])[:, :2] + k * m.cycle_offset_pos[:2]
        q[:, 3:7] = (Rz * Rotation.from_quat(q[:, 3:7], scalar_first=True)).as_quat(scalar_first=True)
        out.append(q)
    return np.vstack(out)


def looped_points(m: Motion, pts: np.ndarray, loops: int) -> np.ndarray:
    if not m.cyclic or loops <= 1:
        return pts
    out = []
    for k in range(loops):
        Rz = Rotation.from_euler("z", k * m.cycle_offset_yaw).as_matrix()
        p = pts @ Rz.T
        p[..., :2] += k * m.cycle_offset_pos[:2]
        out.append(p)
    return np.concatenate(out)


def _add_spheres(scene: mujoco.MjvScene, points: np.ndarray, radius=0.025, rgba=(1, 0.1, 0.1, 0.8)):
    for p in points:
        if scene.ngeom >= scene.maxgeom:
            return
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([radius, 0, 0]), np.asarray(p, dtype=np.float64),
                            np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32))
        scene.ngeom += 1


def render(m: Motion, out_path: Path, *, loops: int = 1, keypoints: np.ndarray | None = None,
           width: int = 640, height: int = 480, distance: float = 4.0, azimuth: float = 90.0,
           elevation: float = -10.0, stills: int = 0, still_dir: Path | None = None,
           model: mujoco.MjModel | None = None) -> list[Path]:
    """Render to mp4; optionally save ``stills`` evenly spaced PNG frames. Returns PNG paths."""
    model = model or load_model()
    data = mujoco.MjData(model)
    qpos = looped_qpos(m, loops)
    kp = None if keypoints is None else looped_points(m, keypoints, loops)
    renderer = mujoco.Renderer(model, height=height, width=width, max_geom=2000)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # stills spread over the first loop (different phases of the stride)
    n_still = m.num_frames if m.cyclic else len(qpos)
    still_idx = set(np.linspace(0, n_still - 1, stills + 1)[:-1].astype(int).tolist()) if stills else set()
    pngs = []
    with imageio.get_writer(out_path, fps=m.fps, codec="libx264", quality=8,
                            macro_block_size=16) as w:
        for t, q in enumerate(qpos):
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            cam.lookat[:] = [q[0], q[1], 0.9]
            renderer.update_scene(data, camera=cam)
            if kp is not None:
                _add_spheres(renderer.scene, kp[t])
            img = renderer.render()
            w.append_data(img)
            if t in still_idx and still_dir is not None:
                still_dir.mkdir(parents=True, exist_ok=True)
                p = still_dir / f"{out_path.stem}_frame{t:04d}.png"
                imageio.imwrite(p, img)
                pngs.append(p)
    renderer.close()
    return pngs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("motion", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="mp4 path (default out/motion_check/<name>.mp4)")
    ap.add_argument("--loops", type=int, default=1)
    ap.add_argument("--keypoints", action="store_true",
                    help="overlay <motion>_keypoints.npz as red spheres")
    ap.add_argument("--stills", type=int, default=0, help="also save N PNG frames")
    ap.add_argument("--azimuth", type=float, default=90.0, help="90 = side view of a +x run")
    ap.add_argument("--distance", type=float, default=4.0)
    args = ap.parse_args()
    m = load_motion(args.motion)
    out = args.out or ROOT / "out/motion_check" / f"{args.motion.stem}.mp4"
    kp = None
    if args.keypoints:
        kp = np.load(args.motion.with_name(args.motion.stem + "_keypoints.npz"))["pos"]
    pngs = render(m, out, loops=args.loops, keypoints=kp, stills=args.stills,
                  still_dir=out.parent, azimuth=args.azimuth, distance=args.distance)
    print(f"wrote {out}")
    for p in pngs:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
