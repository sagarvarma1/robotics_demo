"""Videos: reference vs simulated humanoid, side by side or overlaid, and push clips.

Everything here is *replay*: the rollout already recorded qpos at every policy step, and for each
frame we write qpos into a CPU `mujoco.MjData`, call `mj_forward` (forward kinematics only, no
physics) and render offscreen, exactly as `motion/playback.py` does for the reference clip. That
keeps rendering out of the jitted simulation and makes the two panels trivially frame-aligned:
frame t of both panels comes from the same recorded step t (same clock value), so a lag between
the panels in the video is a real lag in the simulation.

Modes
  side     [reference | simulated]. Same camera settings in both panels (distance, azimuth,
           elevation); each panel's camera follows its *own* pelvis horizontally (`track="self"`),
           so you compare the motions themselves. Drift between the two is printed on the
           simulated panel. Use `track="ref"` to have both cameras follow the reference pelvis,
           so drift shows as the simulated runner sliding out of frame.
  overlay  one scene: the simulated humanoid plus the reference drawn as a translucent blue
           ghost (its geoms added with `mjv_addGeoms`, alpha lowered). The camera follows the
           simulated pelvis. Best for seeing *which* limb is off.
  push     the simulated panel (optionally with ghost) and a red arrow at the pushed body, in
           the push direction, while the force acts. It stays on as a thin outline for 0.5 s
           afterwards so the moment is easy to spot.

Hook for Component C: `left_frames` (a list of HxWx3 uint8 images from the user's own video) adds a
panel on the far left, resized to the panel size. Frames are picked by *phase* when `left_phases`
(one phase in [0, 1) per frame, in the reference clip's phase convention) is given, otherwise by
time at `left_fps` (looping). Phase alignment is what makes a one-stride user clip line up with
a long simulated run: frame "50% through the stride" is shown next to sim frame "50% through".
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from eval.rollout import Rollout
from motion.playback import _add_spheres   # reuse the reference-playback marker helper

# Same defaults as motion.playback.render (side view of a run along +x, pelvis-following).
CAM = {"distance": 4.0, "azimuth": 90.0, "elevation": -10.0, "lookat_z": 0.9}
GHOST_RGBA = (0.25, 0.55, 1.0, 0.35)
PUSH_RGBA = (1.0, 0.1, 0.1, 1.0)


def _camera(**kw) -> mujoco.MjvCamera:
    c = {**CAM, **kw}
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = c["distance"], c["azimuth"], c["elevation"]
    cam.lookat[2] = c["lookat_z"]
    return cam


def _label(img: np.ndarray, lines: list[str], color=(255, 255, 255)) -> np.ndarray:
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default(size=max(12, img.shape[0] // 30))
    except TypeError:            # older Pillow: fixed-size bitmap font
        font = ImageFont.load_default()
    y = 6
    for ln in lines:
        d.text((8, y), ln, fill=color, font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        y += int(img.shape[0] / 30 * 1.3) + 4
    return np.asarray(im)


def _add_ghost(model, ghost_data, scene, rgba=GHOST_RGBA):
    n0 = scene.ngeom
    opt, pert = mujoco.MjvOption(), mujoco.MjvPerturb()
    mujoco.mjv_addGeoms(model, ghost_data, opt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, scene)
    for i in range(n0, scene.ngeom):
        scene.geoms[i].rgba[:] = rgba


def _add_arrow(scene, tail, head, width=0.035, rgba=PUSH_RGBA):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                        np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, np.asarray(tail, np.float64),
                         np.asarray(head, np.float64))
    scene.ngeom += 1


def pick_left_frame(left_frames, t: float, phase: float, left_phases=None, left_fps: float = 30.0):
    """Choose the user-video frame for sim time t / reference phase (see module docstring)."""
    if left_phases is not None:
        lp = np.asarray(left_phases, np.float64)
        dist = np.abs((lp - phase + 0.5) % 1.0 - 0.5)          # circular distance
        return left_frames[int(np.argmin(dist))]
    return left_frames[int(np.floor(t * left_fps)) % len(left_frames)]


def _fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize keeping aspect ratio, letterboxed on black."""
    im = Image.fromarray(np.asarray(img, np.uint8))
    s = min(w / im.width, h / im.height)
    im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))))
    out = Image.new("RGB", (w, h))
    out.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return np.asarray(out)


def frame_range(r: Rollout, ep: int, tail_s: float | None) -> range:
    """Steps to render for episode ep: up to `tail_s` after termination (None: all recorded)."""
    T = r.n_steps
    if tail_s is None:
        return range(T + 1)
    term = int(r.termination_step()[ep])
    return range(min(T, term + int(round(tail_s / r.dt))) + 1)


def render_episode(model: mujoco.MjModel, r: Rollout, ep: int, out_path: str | Path, *,
                   mode: str = "side", track: str = "self", width: int = 480, height: int = 480,
                   tail_s: float | None = 1.0, left_frames: list[np.ndarray] | None = None,
                   left_phases=None, left_fps: float = 30.0, ghost_in_push: bool = True,
                   labels: bool = True, cam: dict | None = None, title: str = "") -> dict:
    """Render one episode of a Rollout to mp4. Returns {"path", "frames", "fps"}.

    mode: "side" | "overlay" | "push". See the module docstring.
    """
    if mode not in ("side", "overlay", "push"):
        raise ValueError(mode)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=height, width=width, max_geom=4000)
    d_sim, d_ref = mujoco.MjData(model), mujoco.MjData(model)
    cam_sim, cam_ref = _camera(**(cam or {})), _camera(**(cam or {}))
    term = int(r.termination_step()[ep])
    push = r.push
    body_id = model.body(push.body).id if push is not None else None
    fps = 1.0 / r.dt
    n = 0
    with imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8, macro_block_size=16) as w:
        for t in frame_range(r, ep, tail_s):
            d_sim.qpos[:] = r.qpos[ep, t]
            d_ref.qpos[:] = r.qpos_ref[ep, t]
            mujoco.mj_forward(model, d_sim)
            mujoco.mj_forward(model, d_ref)
            root_sim, root_ref = r.qpos[ep, t, :2], r.qpos_ref[ep, t, :2]
            drift = float(np.linalg.norm(root_sim - root_ref))
            dead = t >= term
            status = (f"TERMINATED at {term * r.dt:.2f} s" if dead else "")
            tsec = t * r.dt
            panels = []
            if left_frames is not None:
                lf = pick_left_frame(left_frames, tsec, float(r.phase[ep, t]), left_phases, left_fps)
                img = _fit(lf, width, height)
                panels.append(_label(img, ["your video"]) if labels else img)
            if mode == "side":
                cam_ref.lookat[:2] = root_ref
                renderer.update_scene(d_ref, camera=cam_ref)
                img = renderer.render()
                panels.append(_label(img, ["reference", f"t = {tsec:5.2f} s  phase {r.phase[ep, t]:.2f}"])
                              if labels else img)
            cam_sim.lookat[:2] = root_ref if track == "ref" else root_sim
            renderer.update_scene(d_sim, camera=cam_sim)
            if mode == "overlay" or (mode == "push" and ghost_in_push):
                _add_ghost(model, d_ref, renderer.scene)
            ptxt = ""
            if mode == "push" and push is not None:
                s0, nps = int(push.start_step[ep]), int(push.n_steps)
                f = np.asarray(push.force[ep], np.float64)
                fmag = float(np.linalg.norm(f))
                if fmag > 0 and s0 <= t < s0 + nps + int(round(0.5 / r.dt)):
                    u = f / fmag
                    p = d_sim.xipos[body_id].copy()
                    length = 0.25 + 0.6 * min(fmag / 600.0, 1.5)
                    active = t < s0 + nps
                    _add_arrow(renderer.scene, p - u * (length + 0.08), p - u * 0.08,
                               width=0.04 if active else 0.015,
                               rgba=PUSH_RGBA if active else (1.0, 0.5, 0.5, 0.6))
                    if active:
                        _add_spheres(renderer.scene, [p], radius=0.06, rgba=PUSH_RGBA)
                        ptxt = f"PUSH {fmag:.0f} N x {nps * r.dt:.2f} s = {fmag * nps * r.dt:.0f} N s"
            img = renderer.render()
            if labels:
                name = {"side": "simulated", "overlay": "simulated + reference ghost",
                        "push": "simulated (ghost = reference)"}[mode]
                lines = [name + (f"  [{title}]" if title else ""),
                         f"t = {tsec:5.2f} s  root drift {drift:.2f} m"]
                if ptxt:
                    lines.append(ptxt)
                if status:
                    lines.append(status)
                img = _label(img, lines, color=(255, 230, 230) if dead else (255, 255, 255))
            panels.append(img)
            w.append_data(np.concatenate(panels, axis=1))
            n += 1
    renderer.close()
    return {"path": str(out_path), "frames": n, "fps": fps}


def extract_frames(mp4: str | Path, indices: list[int], out_dir: str | Path) -> list[Path]:
    """Save selected frames of an mp4 as PNGs (for checking a video without watching it)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    want, paths = set(indices), []
    reader = imageio.get_reader(mp4)
    for i, fr in enumerate(reader):
        if i in want:
            p = out_dir / f"{Path(mp4).stem}_f{i:04d}.png"
            imageio.imwrite(p, fr)
            paths.append(p)
        if i >= max(want):
            break
    reader.close()
    return paths


def count_frames(mp4: str | Path) -> int:
    reader = imageio.get_reader(mp4)
    n = sum(1 for _ in reader)
    reader.close()
    return n
