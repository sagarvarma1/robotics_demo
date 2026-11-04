"""Retarget a source skeleton's motion onto our MuJoCo humanoid with inverse kinematics.

The problem: a mocap actor (or a person in a video) has different bone lengths and a
slightly different joint layout from our simulated body. If we just copied joint angles
across, the *angles* would match but the *positions* would not: with shorter legs the
feet end up in the air or under the floor, and they slide while they should be planted.
For imitation learning what matters most is where the feet, hands and pelvis are, so we
solve for the model's joint angles that put its bodies where the (scaled) source joints
are. That is inverse kinematics (IK).

Pipeline (each step is a function below):

1. **Map** source keypoints onto model bodies (``SourceMotion.body_map``), e.g. the
   source's "left knee" onto the model body ``ltibia`` (whose origin is the knee).
2. **Scale** (:func:`scale_source`). Walking down the tree from the root, each source
   segment keeps its *direction* but gets the *model's* length. The root trajectory
   is scaled by the leg-length ratio, so a shorter-legged model takes proportionally
   shorter strides and keeps the same timing.
   Without this, IK would have to stretch the model's legs to reach targets it can't
   reach, and the errors would show up as bent knees or floating feet.
3. **IK per frame** (:func:`solve_ik`), using the ``mink`` library. mink solves a small
   quadratic program for a joint-space step Δq that best reduces all task errors at
   once (a weighted least squares). Joint limits enter as hard inequality constraints
   (``ConfigurationLimit``), so the solution can never leave the model's range.
   Tasks: a position task per mapped body, orientation tasks where the source rotation
   can be trusted (pelvis, chest, head, feet; they fix the twist about a bone that
   positions alone cannot pin down), and a weak posture task pulling toward a neutral
   pose. The posture task settles joints that no task constrains and keeps the
   problem well conditioned. Each frame starts from the previous solution
   (warm start), so it converges in a few iterations and the angles change smoothly
   from frame to frame.
4. **Ground fix** (:func:`ground_fix`): small offsets add up, so the feet may hover or
   sink a little. We shift the whole motion vertically so that, during stance, the
   lowest point of the foot *geometry* (capsule and sphere surfaces from MuJoCo, not
   body origins, which sit inside the foot) touches z = 0.
5. **Resample** to the policy rate (50 fps): cubic splines for positions and angles,
   slerp for the root quaternion (quaternions can't be interpolated component-wise).
6. **Smooth** with a Savitzky-Golay filter (fits a low-order polynomial in a sliding
   window). It removes marker jitter, which finite differences would otherwise blow up
   into large velocities, while keeping the shape of the motion.
7. **Velocities** by central finite differences. ``mujoco.mj_differentiatePos`` handles
   the free joint properly: angular velocity comes from the quaternion difference,
   expressed in the body frame as MuJoCo expects.

Fingers, thumbs, toes, wrists and hands have no source data. No task acts on them,
so they stay at the neutral pose (0 rad clipped into range).

The source side is generic (:class:`SourceMotion`), so Component C (SMPL joints from
video) can feed this same retargeter.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import mink
import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

from motion.format import load_model

FOOT_BODIES = {"left": ("lfoot", "ltoes"), "right": ("rfoot", "rtoes")}

# Joints with no source data; held at neutral.
UNDRIVEN_PREFIXES = ("lfingers", "rfingers", "lthumb", "rthumb", "ltoes", "rtoes",
                     "lwrist", "rwrist", "lhand", "rhand")

# Suggested mapping for SMPL's 24 joints (Component C). Not tested here; C must verify,
# in particular that SMPL 'pelvis' sits a few cm lower than our root origin.
SMPL_BODY_MAP = {
    "pelvis": "root", "left_hip": "lfemur", "right_hip": "rfemur", "spine1": "upperback",
    "left_knee": "ltibia", "right_knee": "rtibia", "spine2": "thorax",
    "left_ankle": "lfoot", "right_ankle": "rfoot", "left_foot": "ltoes", "right_foot": "rtoes",
    "neck": "lowerneck", "head": "head", "left_shoulder": "lhumerus",
    "right_shoulder": "rhumerus", "left_elbow": "lradius", "right_elbow": "rradius",
    "left_wrist": "lwrist", "right_wrist": "rwrist",
}


@dataclass
class SourceMotion:
    """Per-frame keypoints of any source skeleton, already in metres and z-up.

    names[0] must be the root (pelvis) and map to model body "root". ``parents[j]`` is the
    index of joint j's parent keypoint (-1 for the root); parents come before children.
    ``rot[t, j]`` (optional) is the world rotation of the *model body* that joint j maps
    to, as estimated from the source (i.e. already in the model body's frame convention).
    Only joints listed in ``rot_valid`` are used as orientation targets.
    """
    fps: float
    names: list[str]
    parents: list[int]
    pos: np.ndarray                      # [T, J, 3]
    body_map: dict[str, str]             # source joint name -> model body name
    rot: np.ndarray | None = None        # [T, J, 3, 3]
    rot_valid: tuple[str, ...] = ()
    source: str = ""

    @property
    def num_frames(self) -> int:
        return self.pos.shape[0]

    def replace(self, **kw) -> "SourceMotion":
        return dataclasses.replace(self, **kw)

    def rotated_about_z(self, yaw: float, shift_xy=(0.0, 0.0)) -> "SourceMotion":
        """Rotate the whole scene by ``yaw`` about world z, then shift in xy."""
        Rz = Rotation.from_euler("z", yaw).as_matrix()
        pos = self.pos @ Rz.T
        pos[..., :2] += np.asarray(shift_xy)
        rot = None if self.rot is None else np.einsum("ij,tnjk->tnik", Rz, self.rot)
        return self.replace(pos=pos, rot=rot)


@dataclass
class RetargetConfig:
    fps_out: float = 50.0
    # IK task weights (position cost per metre; orientation cost per radian)
    pos_cost: dict = field(default_factory=lambda: {
        "root": 10.0, "lfemur": 3.0, "rfemur": 3.0, "ltibia": 4.0, "rtibia": 4.0,
        "lfoot": 6.0, "rfoot": 6.0, "ltoes": 4.0, "rtoes": 4.0,
        "upperback": 1.0, "thorax": 1.0, "lowerneck": 1.0, "upperneck": 0.5, "head": 1.0,
        "lhumerus": 2.0, "rhumerus": 2.0, "lradius": 2.0, "rradius": 2.0,
        "lwrist": 2.0, "rwrist": 2.0})
    default_pos_cost: float = 1.0
    ori_cost: dict = field(default_factory=lambda: {
        "root": 2.0, "thorax": 0.5, "head": 0.2, "lfoot": 0.5, "rfoot": 0.5})
    posture_cost: float = 0.02
    damping: float = 1e-3
    solver: str = "daqp"
    max_iters: int = 40
    max_iters_first: int = 400
    tol: float = 1e-5            # stop when |Δq| < tol
    # smoothing (Savitzky-Golay at fps_out)
    smooth_window: int = 5
    smooth_poly: int = 2
    # ground fix
    stance_height: float = 0.03  # m above the foot's lowest point in the clip
    stance_speed: float = 0.8    # m/s horizontal foot speed
    ground_percentile: float = 10.0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- helpers

def neutral_qpos(model: mujoco.MjModel, margin: float = 0.02) -> np.ndarray:
    """0 rad for every hinge, nudged ``margin`` rad inside the range where 0 is on or
    outside a limit (e.g. knee range starts at 1 deg, wrist range ends at 0)."""
    q = model.qpos0.copy()
    lo, hi = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
    q[7:] = np.clip(q[7:], lo + margin, hi - margin)
    return q


def undriven_hinge_mask(model: mujoco.MjModel) -> np.ndarray:
    names = [model.joint(j).name for j in range(1, model.njnt)]
    return np.array([n.startswith(UNDRIVEN_PREFIXES) for n in names])


def model_body_pos(model: mujoco.MjModel, qpos: np.ndarray, bodies: list[str]) -> np.ndarray:
    """[T, len(bodies), 3] world positions of the given bodies."""
    data = mujoco.MjData(model)
    ids = [model.body(b).id for b in bodies]
    out = np.zeros((qpos.shape[0], len(ids), 3))
    for t in range(qpos.shape[0]):
        data.qpos[:] = qpos[t]
        mujoco.mj_kinematics(model, data)
        out[t] = data.xpos[ids]
    return out


def foot_lowest_z(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
    """Lowest z of the foot+toe geom surfaces, per side (uses current data.geom_x*)."""
    out = {}
    for side, bodies in FOOT_BODIES.items():
        zmin = np.inf
        for b in bodies:
            bid = model.body(b).id
            for g in range(model.ngeom):
                if model.geom_bodyid[g] != bid:
                    continue
                c = data.geom_xpos[g]
                r = model.geom_size[g, 0]
                if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    axis_z = data.geom_xmat[g].reshape(3, 3)[2, 2]
                    z = c[2] - abs(axis_z) * model.geom_size[g, 1] - r
                elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
                    z = c[2] - r
                else:  # not used by our feet; fall back to the bounding sphere
                    z = c[2] - model.geom_rbound[g]
                zmin = min(zmin, z)
        out[side] = zmin
    return out


def foot_tracks(model: mujoco.MjModel, qpos: np.ndarray) -> dict[str, np.ndarray]:
    """Per side: lowest geom z [T] and foot body xy [T, 2]."""
    data = mujoco.MjData(model)
    T = qpos.shape[0]
    low = {s: np.zeros(T) for s in FOOT_BODIES}
    xy = {s: np.zeros((T, 2)) for s in FOOT_BODIES}
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_kinematics(model, data)
        lz = foot_lowest_z(model, data)
        for s, (fb, _) in FOOT_BODIES.items():
            low[s][t] = lz[s]
            xy[s][t] = data.xpos[model.body(fb).id, :2]
    return {"low": low, "xy": xy}


def stance_mask(low: np.ndarray, xy: np.ndarray, fps: float, cfg: RetargetConfig,
                cyclic: bool = False, offset_xy=None) -> np.ndarray:
    """A foot is in stance when it is near its lowest height and barely moving."""
    if cyclic:
        nxt = np.vstack([xy[1:], xy[:1] + np.asarray(offset_xy)[:2]])
    else:
        nxt = np.vstack([xy[1:], xy[-1:] + (xy[-1:] - xy[-2:-1])])
    speed = np.linalg.norm(nxt - xy, axis=1) * fps
    return (low < low.min() + cfg.stance_height) & (speed < cfg.stance_speed)


# --------------------------------------------------------------------------- scaling

def model_segment_lengths(model: mujoco.MjModel, src: SourceMotion) -> np.ndarray:
    """Distance from each mapped body to its parent's mapped body, at the neutral pose."""
    q = neutral_qpos(model)
    p = model_body_pos(model, q[None], [src.body_map[n] for n in src.names])[0]
    L = np.zeros(len(src.names))
    for j, pj in enumerate(src.parents):
        if pj >= 0:
            L[j] = np.linalg.norm(p[j] - p[pj])
    return L


def leg_length(lengths: np.ndarray, src: SourceMotion) -> float:
    """Hip->knee->ankle length averaged over both legs (by model body names)."""
    inv = {v: k for k, v in src.body_map.items()}
    idx = {n: i for i, n in enumerate(src.names)}
    tot = []
    for side in "lr":
        tot.append(lengths[idx[inv[f"{side}tibia"]]] + lengths[idx[inv[f"{side}foot"]]])
    return float(np.mean(tot))


def scale_source(src: SourceMotion, model: mujoco.MjModel) -> tuple[SourceMotion, dict]:
    """Give every source segment the model's length, keeping its direction."""
    Lm = model_segment_lengths(model, src)
    seg = np.zeros_like(src.pos)
    for j, pj in enumerate(src.parents):
        if pj >= 0:
            seg[:, j] = src.pos[:, j] - src.pos[:, pj]
    Ls = np.linalg.norm(seg, axis=-1)                      # [T, J]
    leg_src = leg_length(Ls.mean(0), src)
    leg_model = leg_length(Lm, src)
    ratio = leg_model / leg_src
    out = np.zeros_like(src.pos)
    out[:, 0] = src.pos[:, 0] * ratio
    for j, pj in enumerate(src.parents):
        if pj >= 0:
            d = seg[:, j] / np.maximum(Ls[:, j:j + 1], 1e-9)
            out[:, j] = out[:, pj] + d * Lm[j]
    info = {"leg_ratio": ratio, "leg_model_m": leg_model, "leg_source_m": leg_src,
            "segment_ratio": {n: float(Lm[j] / max(Ls[:, j].mean(), 1e-9))
                              for j, n in enumerate(src.names) if src.parents[j] >= 0}}
    return src.replace(pos=out), info


# --------------------------------------------------------------------------- IK

def solve_ik(src: SourceMotion, model: mujoco.MjModel, cfg: RetargetConfig,
             verbose: bool = False) -> tuple[np.ndarray, dict]:
    """Per-frame IK, warm-started. Returns qpos [T, nq] at the source fps."""
    T = src.num_frames
    conf = mink.Configuration(model)
    tasks, frame_tasks = [], []
    for j, name in enumerate(src.names):
        body = src.body_map[name]
        use_rot = src.rot is not None and name in src.rot_valid
        task = mink.FrameTask(body, "body",
                              position_cost=cfg.pos_cost.get(body, cfg.default_pos_cost),
                              orientation_cost=cfg.ori_cost.get(body, 0.0) if use_rot else 0.0)
        frame_tasks.append((j, task, use_rot))
        tasks.append(task)
    posture = mink.PostureTask(model, cost=cfg.posture_cost)
    q_neutral = neutral_qpos(model)
    posture.set_target(q_neutral)
    tasks.append(posture)
    limits = [mink.ConfigurationLimit(model)]
    und = undriven_hinge_mask(model)

    q = q_neutral.copy()
    q[:3] = src.pos[0, 0]
    if src.rot is not None and src.names[0] in src.rot_valid:
        q[3:7] = Rotation.from_matrix(src.rot[0, 0]).as_quat(scalar_first=True)
    conf.update(q)

    out = np.zeros((T, model.nq))
    iters = []
    dt = 1.0  # step size: mink's Δq = v*dt, so dt only rescales v
    for t in range(T):
        for j, task, use_rot in frame_tasks:
            p = src.pos[t, j]
            if use_rot:
                se3 = mink.SE3.from_rotation_and_translation(
                    mink.SO3.from_matrix(src.rot[t, j]), p)
            else:
                se3 = mink.SE3.from_translation(p)
            task.set_target(se3)
        n_max = cfg.max_iters_first if t == 0 else cfg.max_iters
        for it in range(n_max):
            v = mink.solve_ik(conf, tasks, dt, cfg.solver, damping=cfg.damping, limits=limits)
            conf.integrate_inplace(v, dt)
            if np.linalg.norm(v * dt) < cfg.tol:
                break
        iters.append(it + 1)
        out[t] = conf.q
        out[t, 7:][und] = q_neutral[7:][und]   # no data for these: hold neutral exactly
        if verbose and t % 40 == 0:
            print(f"  ik frame {t}/{T}: {it + 1} iters")
    return out, {"ik_iters_mean": float(np.mean(iters)), "ik_iters_max": int(np.max(iters))}


# --------------------------------------------------------------------------- post-processing

def _quat_continuous(qs: np.ndarray) -> np.ndarray:
    qs = qs.copy()
    for t in range(1, len(qs)):
        if np.dot(qs[t], qs[t - 1]) < 0:
            qs[t] = -qs[t]
    return qs


def resample(qpos: np.ndarray, fps_in: float, fps_out: float,
             model: mujoco.MjModel | None = None) -> np.ndarray:
    """Cubic splines can overshoot slightly, so hinges are clipped to range if a model is given."""
    T = qpos.shape[0]
    t_in = np.arange(T) / fps_in
    t_out = np.arange(0.0, t_in[-1] + 1e-9, 1.0 / fps_out)
    out = np.zeros((len(t_out), qpos.shape[1]))
    cols = [0, 1, 2] + list(range(7, qpos.shape[1]))
    out[:, cols] = CubicSpline(t_in, qpos[:, cols], axis=0)(t_out)
    quats = Rotation.from_quat(qpos[:, 3:7], scalar_first=True)
    out[:, 3:7] = Slerp(t_in, quats)(t_out).as_quat(scalar_first=True)
    out[:, 3:7] = _quat_continuous(out[:, 3:7])
    if model is not None:
        out[:, 7:] = np.clip(out[:, 7:], model.jnt_range[1:, 0], model.jnt_range[1:, 1])
    return out


def resample_points(x: np.ndarray, fps_in: float, fps_out: float) -> np.ndarray:
    T = x.shape[0]
    t_in = np.arange(T) / fps_in
    t_out = np.arange(0.0, t_in[-1] + 1e-9, 1.0 / fps_out)
    return CubicSpline(t_in, x, axis=0)(t_out)


def smooth(qpos: np.ndarray, model: mujoco.MjModel, cfg: RetargetConfig, *,
           cyclic: bool = False, offset_xy=(0.0, 0.0)) -> np.ndarray:
    """Savitzky-Golay smoothing. For a cyclic clip, wrap around the seam (root xy is
    detrended by the per-loop offset so it is periodic too). Hinges are clipped back
    into range afterwards because a filter can overshoot a limit slightly."""
    if cfg.smooth_window <= cfg.smooth_poly:
        return qpos.copy()
    q = qpos.copy()
    T = q.shape[0]
    mode = "wrap" if cyclic else "interp"
    ramp = np.zeros((T, 2))
    if cyclic:
        ramp = np.arange(T)[:, None] / T * np.asarray(offset_xy)[None, :2]
    q[:, :2] -= ramp
    q[:, 3:7] = _quat_continuous(q[:, 3:7])
    f = savgol_filter(q, cfg.smooth_window, cfg.smooth_poly, axis=0, mode=mode)
    f[:, :2] += ramp
    f[:, 3:7] /= np.linalg.norm(f[:, 3:7], axis=1, keepdims=True)
    lo, hi = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
    f[:, 7:] = np.clip(f[:, 7:], lo, hi)
    und = undriven_hinge_mask(model)
    f[:, 7:][:, und] = qpos[:, 7:][:, und]
    return f


def ground_fix(model: mujoco.MjModel, qpos: np.ndarray, fps: float, cfg: RetargetConfig, *,
               cyclic: bool = False, offset_xy=(0.0, 0.0)) -> tuple[np.ndarray, dict]:
    """Shift root z so stance feet touch z = 0 (a single constant shift for the clip)."""
    tr = foot_tracks(model, qpos)
    lows = []
    for s in FOOT_BODIES:
        m = stance_mask(tr["low"][s], tr["xy"][s], fps, cfg, cyclic=cyclic, offset_xy=offset_xy)
        lows.append(tr["low"][s][m])
    lows = np.concatenate(lows)
    dz = -float(np.percentile(lows, cfg.ground_percentile))
    q = qpos.copy()
    q[:, 2] += dz
    return q, {"ground_shift_m": dz, "n_stance_frames": int(lows.size)}


def finite_diff_qvel(model: mujoco.MjModel, qpos: np.ndarray, fps: float, *,
                     cyclic: bool = False, offset_xy=(0.0, 0.0)) -> np.ndarray:
    """qvel[t] = average of backward and forward differences (central difference)."""
    T = qpos.shape[0]
    dt = 1.0 / fps
    if cyclic:
        first = qpos[0].copy()
        first[:2] += np.asarray(offset_xy)[:2]
        last = qpos[-1].copy()
        last[:2] -= np.asarray(offset_xy)[:2]
        ext = np.vstack([last[None], qpos, first[None]])
    else:
        # linear extrapolation in tangent space at the ends
        ext = np.zeros((T + 2, model.nq))
        ext[1:-1] = qpos
        v0 = np.zeros(model.nv)
        mujoco.mj_differentiatePos(model, v0, dt, qpos[1], qpos[0])
        ext[0] = qpos[0].copy()
        mujoco.mj_integratePos(model, ext[0], v0, dt)
        v1 = np.zeros(model.nv)
        mujoco.mj_differentiatePos(model, v1, dt, qpos[-2], qpos[-1])
        ext[-1] = qpos[-1].copy()
        mujoco.mj_integratePos(model, ext[-1], v1, dt)
    qvel = np.zeros((T, model.nv))
    vb = np.zeros(model.nv)
    vf = np.zeros(model.nv)
    for t in range(T):
        mujoco.mj_differentiatePos(model, vb, dt, ext[t], ext[t + 1])
        mujoco.mj_differentiatePos(model, vf, dt, ext[t + 1], ext[t + 2])
        qvel[t] = 0.5 * (vb + vf)
    return qvel


# --------------------------------------------------------------------------- one-call API

@dataclass
class RetargetResult:
    qpos_src_fps: np.ndarray     # IK output at the source fps (before resampling)
    scaled: SourceMotion         # scaled source keypoints at the source fps
    info: dict


def retarget(src: SourceMotion, cfg: RetargetConfig | None = None,
             model: mujoco.MjModel | None = None, verbose: bool = False) -> RetargetResult:
    """Steps 1-3 (map, scale, IK). Post-processing is done by the caller (see build_cmu)."""
    cfg = cfg or RetargetConfig()
    model = model or load_model()
    if src.body_map[src.names[0]] != "root" or src.parents[0] != -1:
        raise ValueError("names[0] must be the root and map to model body 'root'")
    scaled, sinfo = scale_source(src, model)
    qpos, iinfo = solve_ik(scaled, model, cfg, verbose=verbose)
    return RetargetResult(qpos, scaled, {**sinfo, **iinfo})
