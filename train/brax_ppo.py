"""Train the imitation policy with Brax PPO (through MuJoCo Playground's Brax wrapper).

    uv run python -m train.brax_ppo configs/mocap_run_v1.yaml
    uv run python -m train.brax_ppo configs/mocap_run_v1.yaml --tiny --set env.motion_path=<npz>
    uv run python -m train.brax_ppo --resume runs/mocap_run_v1-20261104-160000 [--num-timesteps N]

What happens:
  1. The YAML (with `base:` inheritance, train/rundir.py) gives `env:` and `train:` sections.
     `--set a.b=value` overrides single keys (the value is parsed as YAML). `--tiny` shrinks the
     run so it finishes on a laptop CPU in minutes.
  2. A run directory is created (config.yaml with everything resolved, meta.json; see
     docs/interfaces.md, section 3).
  3. `brax.training.agents.ppo.train.train` runs PPO with thousands of envs stepping in
     parallel inside one jitted function. `wrap_for_brax_training` adds vmap, episode length
     (time limit) and auto-reset. We use `full_reset=True`, so every new episode calls our
     `reset`, i.e. starts at a *fresh* random phase (reference state initialisation). With the
     default (False), each env would always restart from the same cached start state.
  4. After each training epoch we save a checkpoint in Brax's format to
     `<run>/checkpoints/<step>/` and append one line to `<run>/metrics.jsonl`.

Why we save checkpoints ourselves (via `policy_params_fn`) rather than through Brax's
`save_checkpoint_path`: when Brax restores a checkpoint it only restores the network
weights and observation normaliser. Its step counter starts again at 0, so a resumed run would
write over the first run's `<step>` folders. We add the step offset of the restored checkpoint
so step numbers keep counting up across resumes. The Adam optimiser state is not stored in Brax
checkpoints, so it restarts fresh on resume, which shows as a small blip in the learning curve.

metrics.jsonl rows: `step`, `reward` (mean eval episode return), `ep_len` (mean eval episode
length), `mpjpe` and `r_pose/r_vel/r_ee/r_com/root_drift` as *per-step* means (Brax reports
episode sums of our per-step metrics; we divide by the mean episode length, i.e. a
length-weighted average), plus all raw Brax metrics (`eval/...`, `training/...`).
"""

from __future__ import annotations

import os

# Must be set before JAX initialises. MJX on GPU is inaccurate with the default TF32 matmuls
# (MuJoCo Playground README, "GPU Precision Issues").
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import functools  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import yaml  # noqa: E402
from brax.training import checkpoint as brax_checkpoint  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from brax.training.agents.ppo import train as ppo  # noqa: E402
from mujoco_playground._src import wrapper  # noqa: E402

from env import make_env  # noqa: E402
from train.rundir import latest_checkpoint, load_config, log_metrics, make_run_dir  # noqa: E402

# Launcher-only keys in the `train:` section; everything else goes straight to ppo.train.
LAUNCHER_KEYS = ("algo", "network", "full_reset")
DEFAULT_NETWORK = {"policy_hidden_layer_sizes": [512, 256, 128],
                   "value_hidden_layer_sizes": [512, 256, 256, 128]}

TINY = {
    "env": {"episode_length": 100},
    "train": {"num_timesteps": 20_000, "num_envs": 16, "batch_size": 16, "num_minibatches": 4,
              "unroll_length": 10, "num_updates_per_batch": 1, "num_evals": 3,
              "num_eval_envs": 16,
              "network": {"policy_hidden_layer_sizes": [64, 64],
                          "value_hidden_layer_sizes": [64, 64]}},
}
PER_STEP = ("mpjpe", "r_pose", "r_vel", "r_ee", "r_com", "root_drift", "loops")


def set_key(cfg: dict, dotted: str, value) -> None:
    *path, last = dotted.split(".")
    d = cfg
    for k in path:
        d = d.setdefault(k, {})
    d[last] = value


def deep_update(a: dict, b: dict) -> dict:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            deep_update(a[k], v)
        else:
            a[k] = v
    return a


def metrics_row(step: int, metrics: dict) -> dict:
    row = {"step": int(step)}
    if "eval/episode_reward" in metrics:
        ep_len = float(metrics["eval/avg_episode_length"])
        row["reward"] = float(metrics["eval/episode_reward"])
        row["ep_len"] = ep_len
        for k in PER_STEP:
            if f"eval/episode_{k}" in metrics:
                row[k] = float(metrics[f"eval/episode_{k}"]) / max(ep_len, 1.0)
    for k, v in metrics.items():
        try:
            row[k] = float(v)
        except (TypeError, ValueError):
            pass
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", nargs="?", help="experiment YAML (omit with --resume)")
    ap.add_argument("--resume", help="run directory to continue from its latest checkpoint")
    ap.add_argument("--tiny", action="store_true", help="CPU-sized run for a quick end-to-end check")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config key, e.g. env.motion_path=data/motions/x.npz")
    ap.add_argument("--num-timesteps", type=int, help="override train.num_timesteps for this call")
    ap.add_argument("--runs-root", default="runs")
    args = ap.parse_args()

    print("[brax_ppo] jax backend:", jax.default_backend(), jax.devices(),
          "| matmul precision:", os.environ["JAX_DEFAULT_MATMUL_PRECISION"])

    restore, offset = None, 0
    if args.resume:
        run = Path(args.resume).resolve()  # absolute: orbax requires it
        cfg = yaml.safe_load((run / "config.yaml").read_text())
        restore = latest_checkpoint(run)
        if restore is None:
            raise SystemExit(f"no checkpoint found under {run / 'checkpoints'}")
        offset = int(restore.name)
        print(f"[brax_ppo] resuming {run} from checkpoint step {offset}")
    else:
        if not args.config:
            ap.error("give a config YAML or --resume")
        cfg = load_config(args.config)
        if args.tiny:
            deep_update(cfg, json.loads(json.dumps(TINY)))
            cfg["name"] = cfg.get("name", "exp") + "-tiny"
    for kv in args.set:
        k, v = kv.split("=", 1)
        set_key(cfg, k, yaml.safe_load(v))

    tcfg = dict(cfg["train"])
    if args.num_timesteps is not None:
        tcfg["num_timesteps"] = args.num_timesteps
    elif args.resume:
        tcfg["num_timesteps"] = int(tcfg["num_timesteps"]) - offset
        if tcfg["num_timesteps"] <= 0:
            raise SystemExit("run already reached train.num_timesteps; pass --num-timesteps to extend")

    if not args.resume:
        run = make_run_dir(cfg, args.runs_root)
    print(f"[brax_ppo] run dir: {run}")

    env_cfg = cfg["env"]
    env = make_env(env_cfg)
    eval_env = make_env(env_cfg)
    print(f"[brax_ppo] obs size {env.observation_size}, action size {env.action_size}, "
          f"motion {env_cfg['motion_path']} ({env.motion.num_frames} frames @ {env.motion.fps} fps)")

    net_kw = dict(DEFAULT_NETWORK, **tcfg.get("network", {}))
    net_kw = {k: tuple(v) if isinstance(v, list) else v for k, v in net_kw.items()}
    network_factory = functools.partial(ppo_networks.make_ppo_networks, **net_kw)
    wrap = functools.partial(wrapper.wrap_for_brax_training,
                             full_reset=bool(tcfg.get("full_reset", True)))
    ppo_kw = {k: v for k, v in tcfg.items() if k not in LAUNCHER_KEYS}

    ckpt_dir = run / "checkpoints"
    ckpt_config = brax_checkpoint.network_config(
        observation_size=env.observation_size, action_size=env.action_size,
        normalize_observations=bool(ppo_kw.get("normalize_observations", False)),
        network_factory=network_factory)

    def policy_params_fn(step, make_policy, params):
        if step > 0:
            brax_checkpoint.save(ckpt_dir, offset + int(step), params, ckpt_config)

    t0 = time.time()

    def progress_fn(step, metrics):
        row = metrics_row(offset + step, metrics)
        row["wall_s"] = round(time.time() - t0, 1)
        log_metrics(run, row)
        msg = f"[brax_ppo] step {row['step']}: reward {row.get('reward', float('nan')):.3f}"
        if "ep_len" in row:
            msg += f", ep_len {row['ep_len']:.1f}, mpjpe {row.get('mpjpe', float('nan')):.3f} m"
        print(msg + f", {row['wall_s']} s", flush=True)

    _, params, _ = ppo.train(
        environment=env,
        eval_env=eval_env,
        episode_length=int(env_cfg.get("episode_length", env._config.episode_length)),
        wrap_env_fn=wrap,
        network_factory=network_factory,
        seed=int(cfg.get("seed", 0)) + offset,
        restore_checkpoint_path=str(restore) if restore else None,
        policy_params_fn=policy_params_fn,
        progress_fn=progress_fn,
        **ppo_kw,
    )
    last = latest_checkpoint(run)
    print(f"[brax_ppo] done in {time.time() - t0:.1f} s; latest checkpoint: {last}")


if __name__ == "__main__":
    main()
