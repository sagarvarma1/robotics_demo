"""RunImitation: a DeepMimic-style motion-imitation environment in MuJoCo MJX.

The task: a 56-joint humanoid gets a reward for matching a reference clip (a running motion) frame
by frame. The policy never sees the reference pose directly, only the phase (how far through the
clip we are, see env/obs.py). It has to learn *how* to move so that its body ends up where the
reference is, while gravity, contacts and its own momentum push it around.

One environment step (policy rate 50 Hz, ctrl_dt = 20 ms):
  1. action (56 numbers in [-1, 1]) -> data.ctrl. NOTE the action order is the *actuator* order,
     which is alphabetical by joint name (headrx, headry, ...), not the qpos order of the hinges.
     Use `pose_to_action` / `action_to_targets` to convert. Each actuator is a PD servo whose target angle
     is the action mapped linearly onto the joint's range (scripts/build_humanoid.py). The
     policy chooses *where each joint should go*. The servo works out the torque.
  2. physics runs ctrl_dt / sim_dt = 4 substeps of 5 ms.
  3. look up the reference at the new time (env/reference.py), compute the tracking reward
     (env/reward.py) and termination (env/termination.py), and build the observation (env/obs.py).

Reset = reference state initialisation (RSI): start at a uniformly random time in the clip with
the reference's exact pose and velocity. Why it matters (DeepMimic section 6, and its ablations):
if every episode started at frame 0, the policy would have to master the first stride before it
ever saw the later ones. With RSI every part of the clip is practised from the first iteration,
and starting states that already have the right momentum (mid-flight in a running stride) are
reachable at all. DeepMimic reports that without RSI, dynamic skills such as backflips were
not learned.

The reference clock lives in `data.time` (see env/reference.py for why).

PD gain hooks (config `pd.kp_scale`, `pd.kd_scale`). The actuators compute
    force = gain * ctrl + bias0 + bias1 * q,  gain = kp*half_range, bias0 = kp*mid, bias1 = -kp,
so scaling kp means multiplying gainprm[:, 0] and biasprm[:, 0:2] by kp_scale. The target
q* = mid + half*ctrl stays the same. The damping part (-kd * qdot) is joint damping, so kd_scale
multiplies dof_damping of the hinge dofs. Force limits are unchanged. Applied once to the
MjModel before it goes to MJX.

`push` and `randomization` config groups are placeholders (not implemented yet). Setting
`enable: true` raises an error rather than silently doing nothing.
"""

from __future__ import annotations

import re

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from env import obs as obs_lib
from env import reward as reward_lib
from env import termination as term_lib
from env.reference import Reference
from motion.format import EE_BODIES, load_motion

ROOT = Path(__file__).resolve().parents[1]
XML_PATH = ROOT / "env/assets/humanoid_cmu_rlm.xml"


def default_config() -> config_dict.ConfigDict:
    """Every key here appears under `env:` in configs/base.yaml (docs/interfaces.md, section 2)."""
    return config_dict.create(
        motion_path="data/motions/cmu_16_35.npz",
        ctrl_dt=0.02,
        sim_dt=0.005,
        episode_length=500,
        impl="jax",                       # MJX backend; termination reads the jax contact buffer
        reference=config_dict.create(interpolate="auto"),   # auto: only if fps != 1/ctrl_dt
        obs=config_dict.create(future_steps=config_dict.placeholder(tuple)),
        reward=config_dict.create(
            weights=config_dict.create(pose=0.65, vel=0.10, ee=0.15, com=0.10),
            scales=config_dict.create(pose=2.0, vel=1.0, ee=40.0, com=10.0),
            # Hinges whose names match this regex have no mocap data (held neutral by the retargeter),
            # so they are left out of the velocity error. See docs/experiment_log.md.
            vel_exclude="finger|thumb|toes|hand|wrist",
        ),
        pd=config_dict.create(kp_scale=1.0, kd_scale=1.0),
        termination=config_dict.create(
            contact_bodies=list(term_lib.DEFAULT_ALLOWED),   # bodies ALLOWED to touch the floor
            max_root_drift=0.5,
            min_root_height=0.0,
        ),
        push=config_dict.create(enable=False, interval_s=[5.0, 10.0], magnitude=[0.0, 0.0]),
        randomization=config_dict.create(enable=False, friction=[1.0, 1.0], mass_scale=[1.0, 1.0]),
    )


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


class RunImitation(mjx_env.MjxEnv):
    def __init__(self, config: config_dict.ConfigDict | None = None,
                 config_overrides: dict[str, Any] | None = None):
        super().__init__(config or default_config(), config_overrides)
        cfg = self._config
        if cfg.push.enable or cfg.randomization.enable:
            raise NotImplementedError("push / randomization are placeholders, not implemented yet")
        if cfg.impl != "jax":
            raise NotImplementedError("termination reads the MJX jax-impl contact buffer; use impl=jax")
        if abs(self.n_substeps * self.sim_dt - self.dt) > 1e-9:
            raise ValueError("ctrl_dt must be a multiple of sim_dt")

        m = mujoco.MjModel.from_xml_path(str(XML_PATH))
        m.opt.timestep = self.sim_dt
        self._apply_pd_scales(m, cfg.pd.kp_scale, cfg.pd.kd_scale)
        self._mj_model = m
        self._mjx_model = mjx.put_model(m, impl=cfg.impl)
        self._xml_path = XML_PATH.as_posix()

        self.motion = load_motion(resolve_path(cfg.motion_path))
        self.ref = Reference(self.motion, m, self.dt, cfg.reference.interpolate)
        self._future = tuple(int(n) for n in (cfg.obs.future_steps or ()))

        self._root_id = m.body("root").id
        self._ee_ids = np.array([m.body(n).id for n in EE_BODIES])
        self._parent_ids = jnp.asarray(m.body_parentid[1:])
        self._floor_id = m.geom("floor").id
        self._bad_geom = jnp.asarray(term_lib.bad_geom_table(m, tuple(cfg.termination.contact_bodies)))
        hinge_names = [m.joint(j).name for j in range(1, m.njnt)]
        excl = re.compile(cfg.reward.vel_exclude) if cfg.reward.vel_exclude else None
        self._vel_mask = jnp.asarray([0.0 if excl and excl.search(n) else 1.0 for n in hinge_names])
        self._w = {k: float(cfg.reward.weights[k]) for k in reward_lib.TERMS}
        self._k = {k: float(cfg.reward.scales[k]) for k in reward_lib.TERMS}
        # Actuator a drives joint actuator_trnid[a, 0]. scripts/build_humanoid.py emits actuators
        # in qpos order, so this mapping is currently the identity, but we still go through it
        # so pose<->action conversion stays correct if the model ever changes.
        act_jnt = m.actuator_trnid[:, 0]
        self._act_qadr = jnp.asarray(m.jnt_qposadr[act_jnt])
        lo, hi = m.jnt_range[act_jnt, 0], m.jnt_range[act_jnt, 1]
        self._ctrl_mid = jnp.asarray(0.5 * (lo + hi), jnp.float32)
        self._ctrl_half = jnp.asarray(0.5 * (hi - lo), jnp.float32)

    @staticmethod
    def _apply_pd_scales(m: mujoco.MjModel, kp_scale: float, kd_scale: float) -> None:
        if kp_scale != 1.0:
            m.actuator_gainprm[:, 0] *= kp_scale
            m.actuator_biasprm[:, 0:2] *= kp_scale
        if kd_scale != 1.0:
            m.dof_damping[6:] *= kd_scale

    # -- helpers ---------------------------------------------------------------------------------
    def pose_to_action(self, hinge_qpos):
        """Inverse of the actuator map: the action whose PD targets equal `hinge_qpos`.

        `hinge_qpos` is in qpos order (qpos[7:]). The result is in actuator order (= data.ctrl).
        """
        q = jnp.asarray(hinge_qpos)[..., self._act_qadr - 7]
        return jnp.clip((q - self._ctrl_mid) / self._ctrl_half, -1.0, 1.0)

    def action_to_targets(self, action):
        """PD target angles in qpos order (qpos[7:]) for an action in actuator order."""
        tgt = self._ctrl_mid + self._ctrl_half * jnp.clip(action, -1.0, 1.0)
        return jnp.zeros(tgt.shape[:-1] + (56,), tgt.dtype).at[..., self._act_qadr - 7].set(tgt)

    def _evaluate(self, data: mjx.Data):
        """Reference lookup, reward, metrics, done and obs for the current data."""
        fr = self.ref.at(data.time)
        root_sim = data.xpos[self._root_id]
        root_ref = fr.body_pos[self._root_id - 1]
        ee_sim = data.xpos[self._ee_ids]
        com_sim = data.subtree_com[self._root_id]
        errs = reward_lib.errors(
            reward_lib.local_quats(data.xquat, self._parent_ids), fr.local_quat,
            data.qvel, fr.qvel, ee_sim, root_sim, fr.ee_pos, root_ref, com_sim, fr.com,
            vel_mask=self._vel_mask)
        reward, terms = reward_lib.tracking_reward(errs, self._w, self._k)
        root_drift = jnp.linalg.norm(root_sim - root_ref)
        mpjpe = jnp.mean(jnp.linalg.norm(data.xpos[1:] - fr.body_pos, axis=-1))

        c = term_lib.contact_buffer(data)
        flags = term_lib.done_flags(data.qpos, data.qvel, c.geom, c.dist, self._floor_id,
                                    self._bad_geom, root_drift,
                                    self._config.termination.max_root_drift,
                                    self._config.termination.min_root_height)
        done = flags["contact"] | flags["drift"] | flags["low"] | flags["nan"]

        future = []
        for n in self._future:
            ff = self.ref.at(data.time + n * self.dt)
            future.append((ff.qpos[7:], ff.qpos[:3]))
        ob = obs_lib.observation(data.qpos, data.qvel, data.xmat[self._root_id], ee_sim, fr.phase,
                                 future)
        ob = jnp.nan_to_num(ob, nan=0.0, posinf=0.0, neginf=0.0)
        reward = jnp.where(flags["nan"], 0.0, reward)
        metrics = {
            "r_pose": terms["pose"], "r_vel": terms["vel"], "r_ee": terms["ee"], "r_com": terms["com"],
            "mpjpe": mpjpe, "root_drift": root_drift, "loops": fr.loop.astype(jnp.float32),
        }
        return fr, ob, reward, done.astype(jnp.float32), metrics, flags

    # -- MjxEnv API ------------------------------------------------------------------------------
    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, key = jax.random.split(rng)
        t0 = self.ref.frame_aligned_time(jax.random.uniform(key))
        return self.reset_to_time(t0, rng)

    def reset_to_time(self, t0, rng) -> mjx_env.State:
        """Start exactly on the reference at time t0 (loop 0 if t0 < duration)."""
        fr = self.ref.at(t0)
        data = mjx_env.make_data(self._mj_model, qpos=fr.qpos, qvel=fr.qvel, impl=self._config.impl)
        data = data.replace(time=jnp.asarray(t0, data.time.dtype),
                            ctrl=self.pose_to_action(fr.qpos[7:]))
        data = mjx.forward(self._mjx_model, data)
        fr, ob, reward, done, metrics, _ = self._evaluate(data)
        info = {"rng": rng, "t": jnp.zeros((), jnp.int32), "phase": fr.phase,
                "loop": fr.loop.astype(jnp.int32), "ref_index": fr.index.astype(jnp.int32)}
        return mjx_env.State(data, ob, jnp.zeros(()), jnp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        ctrl = jnp.clip(action, -1.0, 1.0)
        data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)
        fr, ob, reward, done, metrics, _ = self._evaluate(data)
        info = dict(state.info)
        info.update(t=state.info["t"] + 1, phase=fr.phase, loop=fr.loop.astype(jnp.int32),
                    ref_index=fr.index.astype(jnp.int32))
        # Update rather than replace: Brax wrappers may add their own keys (e.g. 'reward').
        new_metrics = dict(state.metrics)
        new_metrics.update(metrics)
        return state.replace(data=data, obs=ob, reward=reward, done=done, metrics=new_metrics,
                             info=info)

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def observation_size(self) -> int:
        return obs_lib.obs_size(len(self._future))
