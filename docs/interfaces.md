# Interfaces

These three contracts let the components (A env, B motion, C video, D eval) be built in parallel.
Change them only deliberately: update this file, `motion/format.py`, and the tests together.

## 1. Reference motion file (`.npz`)

Produced by `motion/` (from CMU mocap or from `video/` output), consumed by `env/` and `eval/`.
Load and validate with `motion.format.load_motion(path)`; write with `motion.format.save_motion(path, m)`.

All quantities are in the **simulation model's frame**: world z up, metres, radians, MuJoCo
quaternions `(w, x, y, z)`. They were computed by running MuJoCo forward kinematics on our model
(`env/assets/humanoid_cmu_rlm.xml`), so they match what the simulator would report for the same qpos.

| key | shape / type | meaning |
|---|---|---|
| `fps` | float | frames per second (default 50 = policy rate, 1/ctrl_dt) |
| `qpos` | `[T, 63]` f32 | free root (pos 3 + quat 4) + 56 hinge angles, MuJoCo qpos order |
| `qvel` | `[T, 62]` f32 | root lin vel (world) 3 + root ang vel (local, MuJoCo convention) 3 + 56 joint vels |
| `body_pos` | `[T, B, 3]` f32 | world position of every body (B = `model.nbody - 1`, world excluded) |
| `body_quat` | `[T, B, 4]` f32 | world orientation of every body |
| `ee_pos` | `[T, 4, 3]` f32 | world positions of `lhand, rhand, lfoot, rfoot` bodies |
| `com` | `[T, 3]` f32 | whole-body centre of mass (world) |
| `body_names` | `[B]` str | body names in order |
| `joint_names` | `[56]` str | hinge joint names in qpos order (qpos[7:]) |
| `cyclic` | bool | the clip loops: frame T-1 is followed by frame 0 |
| `cycle_offset_pos` | `[3]` f32 | root xy displacement per loop (z = 0) |
| `cycle_offset_yaw` | float | root heading change per loop (rad, ~0 for straight running) |
| `source` | str | e.g. `cmu:16_35` or `video:<name>` |
| `meta_json` | str | JSON: retarget settings, frame range cut, git sha, etc. |

Looping rule (cyclic clips): at time t, frame index `i = floor(t*fps) mod T`, loop count `k = floor(t*fps / T)`.
The reference root position is `R(k*yaw) @ qpos[i,:3] + k*offset` (applied to all world positions in the
same way; xy only). Quaternions are pre-rotated by `k*yaw` about z.

Validation (`motion.format.validate`) checks: shapes agree, all values finite, unit quaternions,
hinge angles within joint limits (with 1e-3 tolerance), foot body origins (ankles) at z ≥ -1 cm, fps > 0.

## 2. Environment API

`env.imitation.RunImitation(config: ConfigDict | None, config_overrides: dict | None)`, a subclass of
`mujoco_playground.MjxEnv`. Build it with `env.make_env(config)` (we do not rely on
`mujoco_playground.registry`, whose env list is fixed when the module is imported).

- `env.default_config()` returns a `ml_collections.ConfigDict`. Every key appears in the experiment YAML
  under `env:`. Top-level groups: `motion_path, ctrl_dt, sim_dt, episode_length, reward{weights, scales},
  pd{kp_scale, kd_scale}, termination{contact_bodies, max_root_drift, min_root_height}, push{...},
  randomization{...}`.
- `reset(rng) -> State`: reference state initialisation (random phase).
- `step(state, action[56] in [-1,1]) -> State`.
- `state.obs`: flat `jnp` vector (size via `env.observation_size`). Layout documented in `env/obs.py`.
- `state.info` must contain: `phase` (float in [0,1)), `t` (policy steps since reset), `loop` (int),
  `ref_index` (int), `rng`.
- `state.metrics` contains (per step, float): `r_pose, r_vel, r_ee, r_com, mpjpe, root_drift, loops`.

## 3. Run directory and checkpoints

```
runs/<exp_name>-<YYYYmmdd-HHMMSS>/
  config.yaml        # fully resolved experiment config
  meta.json          # git sha + dirty flag, motion sha256, seed, versions (jax, mujoco, brax, playground), device
  metrics.jsonl      # one JSON object per eval/log step: {"step":..., "reward":..., "ep_len":..., "mpjpe":...}
  checkpoints/<step>/  # Brax PPO orbax checkpoint (brax.training.checkpoint format)
```
Resuming: `train/brax_ppo.py --resume runs/<dir>` loads `config.yaml` and the highest `<step>`.
Our own PPO (Phase 6) uses the same layout; its params pytree also holds the observation normaliser.
On Kaggle, `runs/` lives under `/kaggle/working/runs` and is saved as notebook output; to resume in a later
session, attach the previous output as an input dataset and pass that path.
