"""Reference motion file format (see docs/interfaces.md, section 1).

A reference motion is everything the environment needs to know about the clip,
*precomputed* by running MuJoCo forward kinematics on our own model. Why
precompute: during training the GPU runs thousands of environments, and each
one only needs to *look up* "where should the left foot be at phase 0.37?". An
array lookup is far cheaper than redoing kinematics on the reference every step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

import mujoco
import numpy as np

EE_BODIES = ("lhand", "rhand", "lfoot", "rfoot")
MODEL_XML = Path(__file__).resolve().parents[1] / "env/assets/humanoid_cmu_rlm.xml"


@dataclass
class Motion:
    fps: float
    qpos: np.ndarray            # [T, nq]
    qvel: np.ndarray            # [T, nv]
    body_pos: np.ndarray        # [T, B, 3]
    body_quat: np.ndarray       # [T, B, 4]
    ee_pos: np.ndarray          # [T, 4, 3]
    com: np.ndarray             # [T, 3]
    body_names: list[str]
    joint_names: list[str]
    cyclic: bool
    cycle_offset_pos: np.ndarray  # [3]
    cycle_offset_yaw: float
    source: str
    meta: dict = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return self.qpos.shape[0]

    @property
    def duration(self) -> float:
        return self.num_frames / self.fps


def load_model(xml_path: Path | str = MODEL_XML) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(xml_path))


def hinge_joint_names(model: mujoco.MjModel) -> list[str]:
    return [model.joint(j).name for j in range(model.njnt)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]


def from_qpos_qvel(qpos: np.ndarray, qvel: np.ndarray, fps: float, *, cyclic: bool,
                   cycle_offset_pos=(0.0, 0.0, 0.0), cycle_offset_yaw: float = 0.0,
                   source: str = "", meta: dict | None = None,
                   model: mujoco.MjModel | None = None) -> Motion:
    """Fill in all derived quantities (body poses, end effectors, CoM) via forward kinematics."""
    model = model or load_model()
    data = mujoco.MjData(model)
    T = qpos.shape[0]
    B = model.nbody - 1
    body_pos = np.zeros((T, B, 3), np.float32)
    body_quat = np.zeros((T, B, 4), np.float32)
    com = np.zeros((T, 3), np.float32)
    root_body = model.body("root").id
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        body_pos[t] = data.xpos[1:]
        body_quat[t] = data.xquat[1:]
        com[t] = data.subtree_com[root_body]
    names = [model.body(b).name for b in range(1, model.nbody)]
    ee_idx = [names.index(n) for n in EE_BODIES]
    return Motion(
        fps=float(fps), qpos=qpos.astype(np.float32), qvel=qvel.astype(np.float32),
        body_pos=body_pos, body_quat=body_quat, ee_pos=body_pos[:, ee_idx], com=com,
        body_names=names, joint_names=hinge_joint_names(model), cyclic=bool(cyclic),
        cycle_offset_pos=np.asarray(cycle_offset_pos, np.float32),
        cycle_offset_yaw=float(cycle_offset_yaw), source=source, meta=meta or {},
    )


def save_motion(path: Path | str, m: Motion) -> None:
    validate(m)
    d = {f.name: getattr(m, f.name) for f in fields(m) if f.name != "meta"}
    d["body_names"] = np.array(m.body_names)
    d["joint_names"] = np.array(m.joint_names)
    d["meta_json"] = json.dumps(m.meta)
    np.savez_compressed(path, **d)


def load_motion(path: Path | str, *, check: bool = True) -> Motion:
    z = np.load(path, allow_pickle=False)
    m = Motion(
        fps=float(z["fps"]), qpos=z["qpos"], qvel=z["qvel"], body_pos=z["body_pos"],
        body_quat=z["body_quat"], ee_pos=z["ee_pos"], com=z["com"],
        body_names=[str(s) for s in z["body_names"]], joint_names=[str(s) for s in z["joint_names"]],
        cyclic=bool(z["cyclic"]), cycle_offset_pos=z["cycle_offset_pos"],
        cycle_offset_yaw=float(z["cycle_offset_yaw"]), source=str(z["source"]),
        meta=json.loads(str(z["meta_json"])),
    )
    if check:
        validate(m)
    return m


def validate(m: Motion, model: mujoco.MjModel | None = None, *, limit_tol: float = 1e-3,
             floor_tol: float = 0.01) -> None:
    """Raise ValueError if the motion violates the contract in docs/interfaces.md."""
    model = model or load_model()
    T = m.qpos.shape[0]
    B = model.nbody - 1
    expect = {"qpos": (T, model.nq), "qvel": (T, model.nv), "body_pos": (T, B, 3),
              "body_quat": (T, B, 4), "ee_pos": (T, 4, 3), "com": (T, 3)}
    for k, shape in expect.items():
        a = getattr(m, k)
        if a.shape != shape:
            raise ValueError(f"{k}: shape {a.shape} != {shape}")
        if not np.isfinite(a).all():
            raise ValueError(f"{k}: non-finite values")
    if m.fps <= 0:
        raise ValueError("fps must be > 0")
    if m.joint_names != hinge_joint_names(model):
        raise ValueError("joint_names do not match the model's hinge joints")
    qn = np.linalg.norm(m.qpos[:, 3:7], axis=-1)
    if np.abs(qn - 1).max() > 1e-3:
        raise ValueError("root quaternions are not unit length")
    lo, hi = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
    q = m.qpos[:, 7:]
    bad = (q < lo - limit_tol) | (q > hi + limit_tol)
    if bad.any():
        j = int(np.argwhere(bad)[0, 1])
        raise ValueError(f"joint {m.joint_names[j]} outside its limits")
    foot_z = m.ee_pos[:, 2:, 2].min()
    if foot_z < -floor_tol:
        raise ValueError(f"foot below floor: min z = {foot_z:.3f}")
