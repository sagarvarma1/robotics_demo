"""Policies to evaluate: a trained Brax PPO checkpoint, or one of two no-training baselines.

Every policy here has the same shape, `act(state, rng) -> action[56]`: a pure JAX function of the
env state (`mujoco_playground.MjxEnv` State) and a PRNG key, so the rollout code can `vmap` and
`jit` it without caring where it came from. A trained network only looks at `state.obs` (that is
all it saw during training). The reference-PD baseline also reads the clock in `state.data.time`,
which a real policy never gets, which is why it is a *baseline* and not a policy.

Why baselines at all:
  * They make the whole eval harness (rollout, metrics, push test, videos) testable before any
    real training has happened.
  * They give a floor. A trained policy that cannot beat "replay the reference angles open-loop
    through the PD servos" has learned nothing useful about balance.

  zero   action = 0 for every joint, i.e. PD targets at the middle of each joint's range. The
         character goes limp-ish into a fixed posture and falls. Floor for everything.
  refpd  action = env.pose_to_action(ref.at(t + dt).qpos[7:]): the PD servos are told to go to the
         reference joint angles of the *next* step (that is where the env will check them).
         Open loop: no feedback on balance, root position or velocity. It starts on the reference
         (RSI) with the right momentum, so it looks right for a few strides, then small errors
         grow and nothing corrects them. This is the "how far does copying the angles get you"
         baseline that DeepMimic-style control has to beat.

Loading a trained run. `train/brax_ppo.py` saves checkpoints with
`brax.training.checkpoint.save(...)`, whose default network-config file name is `config.json`
(not `ppo_network_config.json`, which `brax.training.agents.ppo.checkpoint.load_policy` expects).
So we do the same three steps `load_policy` does, with the right file name:
    config  = brax.training.checkpoint.load_config(<ckpt>/config.json)
    network = brax.training.checkpoint.get_network(config, ppo_networks.make_ppo_networks)
    params  = brax.training.checkpoint.load(<ckpt>)        # (normaliser, policy, value)
    infer   = ppo_networks.make_inference_fn(network)(params, deterministic=...)
The env is rebuilt from the run's `config.yaml` (`env:` section), so the observation layout and
PD gains are exactly the ones the policy was trained with.

Deterministic vs stochastic. During training, PPO samples actions from a Gaussian (squashed by
tanh) to explore. For evaluation we normally use the *mode* (deterministic=True): the policy's
best guess, without exploration noise. Stochastic evaluation is a robustness check: a good policy
should also survive its own noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import yaml

from env import make_env
from env.imitation import RunImitation

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_CONFIG = ROOT / "configs/mocap_run_v1.yaml"

ActFn = Callable[[Any, jax.Array], jax.Array]   # (env State, rng) -> action[56]


@dataclass
class EvalPolicy:
    """A policy ready for evaluation.

    act:      (state, rng) -> action, pure JAX (vmap/jit friendly).
    obs_act:  (obs, rng) -> action, only for networks (None for baselines that need the clock).
    name:     short label used for output folders and plot legends.
    info:     where it came from (run dir, checkpoint step, ...), copied into the JSON outputs.
    """
    name: str
    act: ActFn
    obs_act: Callable[[jax.Array, jax.Array], jax.Array] | None = None
    info: dict | None = None


# -- baselines ------------------------------------------------------------------------------------
def zero_policy(env: RunImitation) -> EvalPolicy:
    nu = env.action_size

    def act(state, rng):
        del state, rng
        return jnp.zeros((nu,), jnp.float32)

    return EvalPolicy("zero", act, obs_act=lambda obs, rng: jnp.zeros(obs.shape[:-1] + (nu,)),
                      info={"kind": "baseline", "baseline": "zero"})


def refpd_policy(env: RunImitation) -> EvalPolicy:
    def act(state, rng):
        del rng
        fr = env.ref.at(state.data.time + env.dt)
        return env.pose_to_action(fr.qpos[7:])

    return EvalPolicy("refpd", act, obs_act=None, info={"kind": "baseline", "baseline": "refpd"})


BASELINES = {"zero": zero_policy, "refpd": refpd_policy}


def baseline_env(config_path: str | Path = DEFAULT_BASELINE_CONFIG,
                 overrides: dict | None = None) -> tuple[RunImitation, dict]:
    """Env for baselines, from an experiment YAML (default: the first real run's config)."""
    from train.rundir import load_config   # same `base:` inheritance as training
    cfg = load_config(config_path)
    env_cfg = _merge(dict(cfg.get("env", {})), overrides or {})
    return make_env(env_cfg), env_cfg


def load_baseline(name: str, config_path: str | Path = DEFAULT_BASELINE_CONFIG,
                  overrides: dict | None = None) -> tuple[RunImitation, EvalPolicy, dict]:
    if name not in BASELINES:
        raise ValueError(f"unknown baseline {name!r}; choose from {sorted(BASELINES)}")
    env, env_cfg = baseline_env(config_path, overrides)
    return env, BASELINES[name](env), env_cfg


# -- trained Brax PPO checkpoint -------------------------------------------------------------------
def find_checkpoint(run_dir: str | Path, step: int | None = None) -> Path:
    """`<run>/checkpoints/<step>/` (absolute: orbax refuses relative paths). Latest if step is None."""
    ck = Path(run_dir).resolve() / "checkpoints"
    steps = sorted((p for p in ck.iterdir() if p.is_dir() and p.name.isdigit()),
                   key=lambda p: int(p.name)) if ck.exists() else []
    if not steps:
        raise FileNotFoundError(f"no checkpoints under {ck}")
    if step is None:
        return steps[-1]
    for p in steps:
        if int(p.name) == int(step):
            return p
    raise FileNotFoundError(f"no checkpoint for step {step} in {ck} (have {[int(p.name) for p in steps]})")


def load_inference_fn(ckpt: str | Path, deterministic: bool = True):
    """Brax PPO inference function `(obs, rng) -> (action, extras)` from one checkpoint folder."""
    from brax.training import checkpoint as brax_checkpoint
    from brax.training.agents.ppo import networks as ppo_networks

    ckpt = Path(ckpt).resolve()
    cfg_file = ckpt / "config.json"            # name used by brax.training.checkpoint.save
    if not cfg_file.exists():
        cfg_file = ckpt / "ppo_network_config.json"   # name used by ppo.checkpoint.save
    config = _load_network_config(cfg_file)
    network = brax_checkpoint.get_network(config, ppo_networks.make_ppo_networks)
    params = brax_checkpoint.load(ckpt)
    infer = ppo_networks.make_inference_fn(network)(params, deterministic=deterministic)
    return infer, config


def _load_network_config(cfg_file: Path):
    """Same as `brax.training.checkpoint.load_config`, but tolerates `null` kernel-init entries.

    Brax's `save` writes `mean_kernel_init_fn: null` (its default) unchanged, and its
    `load_config` then looks up `KERNEL_INITIALIZER[None]` and raises KeyError (brax at the pinned
    rev). We map names back to functions exactly as brax does, and leave None as None (which is
    `make_ppo_networks`' default for that argument).
    """
    import json

    from brax.training import networks as brax_networks
    from ml_collections import config_dict

    d = json.loads(Path(cfg_file).read_text())
    kw = d["network_factory_kwargs"]
    if isinstance(kw.get("activation"), str):
        kw["activation"] = brax_networks.ACTIVATION[kw["activation"]]
    for k in ("policy_network_kernel_init_fn", "value_network_kernel_init_fn",
              "q_network_kernel_init_fn", "mean_kernel_init_fn"):
        if isinstance(kw.get(k), str):
            kw[k] = brax_networks.KERNEL_INITIALIZER[kw[k]]
    return config_dict.create(**d)


def load_run(run_dir: str | Path, step: int | None = None, *, deterministic: bool = True,
             env_overrides: dict | None = None) -> tuple[RunImitation, EvalPolicy, dict]:
    """Rebuild the env from `<run>/config.yaml` and load a checkpoint as an EvalPolicy.

    `env_overrides` (nested dict) can change env keys for evaluation, e.g.
    `{"termination": {"max_root_drift": 1.0}}`. Leave it empty to evaluate under training conditions.
    Returns (env, policy, env_cfg).
    """
    run = Path(run_dir).resolve()
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    env_cfg = _merge(dict(cfg.get("env", {})), env_overrides or {})
    env = make_env(env_cfg)
    ckpt = find_checkpoint(run, step)
    infer, net_cfg = load_inference_fn(ckpt, deterministic)
    if int(net_cfg.observation_size) != env.observation_size or int(net_cfg.action_size) != env.action_size:
        raise ValueError(f"checkpoint obs/action size {net_cfg.observation_size}/{net_cfg.action_size} "
                         f"does not match env {env.observation_size}/{env.action_size}")

    def obs_act(obs, rng):
        return infer(obs, rng)[0]

    def act(state, rng):
        return obs_act(state.obs, rng)

    name = f"{run.name}@{int(ckpt.name)}"
    info = {"kind": "brax_ppo", "run": str(run), "checkpoint": str(ckpt), "step": int(ckpt.name),
            "deterministic": deterministic}
    return env, EvalPolicy(name, act, obs_act=obs_act, info=info), env_cfg


def jit_obs_act(policy: EvalPolicy):
    """Jitted `act(obs, rng) -> action` for a network policy (the interface asked for in the brief)."""
    if policy.obs_act is None:
        raise ValueError(f"policy {policy.name} needs the env state (clock), not just obs")
    return jax.jit(policy.obs_act)


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out
