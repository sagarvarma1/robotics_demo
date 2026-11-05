"""Synthetic reference clips for testing the environment before real mocap exists.

Why synthetic clips: the env code (reference lookup, looping, reward, termination) can be tested
against motions whose answers we know. A standing clip is trivially cyclic. A "march" moves forward
at a known speed, so the loop rule (each loop adds `cycle_offset_pos`) can be checked exactly.

Both clips go through `motion.format.from_qpos_qvel`, so body positions, end effectors and CoM come
from MuJoCo forward kinematics on our model, just as the real clip's do.

Conventions found by probing the model (env/assets/humanoid_cmu_rlm.xml):
  * the root body's local +z axis is "forward". In the default pose it points along world -y.
  * negative `*femurrx` swings the thigh forward (hip flexion); positive `*tibiarx` bends the knee.
  * at all-zero hinge angles the thighs splay 20 deg sideways, so we add `femurrz = -/+20 deg` to
    stand with the legs vertical.

Run `uv run python -m tests.fixtures_motion <out_dir>` to write both clips as <out_dir>/synthetic_*.npz.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from motion.format import Motion, from_qpos_qvel, load_model, save_motion

ROOT = Path(__file__).resolve().parents[1]
FOOT_BODIES = ("lfoot", "ltoes", "rfoot", "rtoes")


def _hinge(model: mujoco.MjModel, name: str) -> int:
    """qpos address of a hinge joint."""
    return int(model.jnt_qposadr[model.joint(name).id])


def standing_qpos(model: mujoco.MjModel) -> np.ndarray:
    """Upright standing pose, joints inside their ranges, lowest foot point at z = 0."""
    q = model.qpos0.copy()
    q[_hinge(model, "lfemurrz")] = np.radians(-20)
    q[_hinge(model, "rfemurrz")] = np.radians(20)
    lo, hi = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
    q[7:] = np.clip(q[7:], lo + 1e-3, hi - 1e-3)
    q[2] += -lowest_foot_point(model, q)
    return q


def lowest_foot_point(model: mujoco.MjModel, qpos: np.ndarray) -> float:
    """Lowest z of any foot/toe collision geom (capsule ends and spheres, minus their radius)."""
    d = mujoco.MjData(model)
    d.qpos[:] = qpos
    mujoco.mj_kinematics(model, d)
    feet = {model.body(n).id for n in FOOT_BODIES}
    zmin = np.inf
    for g in range(model.ngeom):
        if model.geom_bodyid[g] not in feet:
            continue
        r = model.geom_size[g, 0]
        c = d.geom_xpos[g]
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            ax = d.geom_xmat[g].reshape(3, 3)[:, 2] * model.geom_size[g, 1]
            z = min(c[2] + ax[2], c[2] - ax[2]) - r
        else:
            z = c[2] - r
        zmin = min(zmin, z)
    return float(zmin)


def standing_clip(num_frames: int = 50, fps: float = 50.0, model=None) -> Motion:
    """The standing pose repeated: a cyclic clip with zero velocity and zero loop offset."""
    model = model or load_model()
    q = np.tile(standing_qpos(model), (num_frames, 1))
    v = np.zeros((num_frames, model.nv))
    return from_qpos_qvel(q, v, fps, cyclic=True, source="synthetic:stand", model=model)


def march_clip(num_frames: int = 50, fps: float = 50.0, speed: float = 1.0,
               hip_amp_deg: float = 30.0, knee_amp_deg: float = 40.0, model=None) -> Motion:
    """A sinusoidal hip/knee 'march' sliding forward at `speed` m/s; one gait cycle per clip.

    Not physically consistent (feet slide), but it is cyclic, moves forward, and exercises the loop
    offset. The root moves along the body's forward direction (world -y).
    """
    model = model or load_model()
    base = standing_qpos(model)
    T = num_frames
    t = np.arange(T) / fps
    w = 2 * np.pi / (T / fps)                      # one cycle per clip
    q = np.tile(base, (T, 1))
    hip, knee = np.radians(hip_amp_deg), np.radians(knee_amp_deg)
    for side, sgn in (("l", 1.0), ("r", -1.0)):
        s = np.sin(w * t) * sgn
        q[:, _hinge(model, f"{side}femurrx")] = -hip * np.maximum(s, 0) * 1.0 + hip * 0.3 * np.minimum(s, 0)
        q[:, _hinge(model, f"{side}tibiarx")] = np.radians(5) + knee * np.maximum(s, 0)
    lo, hi = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
    q[:, 7:] = np.clip(q[:, 7:], lo + 1e-3, hi - 1e-3)
    fwd = np.array([0.0, -1.0, 0.0])
    q[:, :3] = base[:3] + np.outer(t * speed, fwd)
    # Velocities: central differences on the cyclic hinge trajectory; root moves at constant speed.
    v = np.zeros((T, model.nv))
    dq = (np.roll(q[:, 7:], -1, axis=0) - np.roll(q[:, 7:], 1, axis=0)) * fps / 2
    v[:, 6:] = dq
    v[:, :3] = fwd * speed
    offset = fwd * speed * (T / fps)
    return from_qpos_qvel(q, v, fps, cyclic=True, cycle_offset_pos=offset, source="synthetic:march",
                          model=model)


def write_fixtures(out_dir: Path | str) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, clip in (("stand", standing_clip()), ("march", march_clip())):
        p = out_dir / f"synthetic_{name}.npz"
        save_motion(p, clip)
        paths[name] = p
    return paths


if __name__ == "__main__":
    import sys
    for k, p in write_fixtures(sys.argv[1] if len(sys.argv) > 1 else "out/fixtures").items():
        print(k, p)
