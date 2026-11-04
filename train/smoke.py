"""Phase 1 smoke test: is the GPU stack healthy, and how fast is our humanoid?

Run on Kaggle (T4):   python -m train.smoke --out /kaggle/working/smoke
Run locally (tiny):   uv run python -m train.smoke --allow-cpu --tiny --out out/smoke

Steps:
 1. JAX sees a GPU (fails loudly otherwise; --allow-cpu skips this check).
 2. Stock Playground task HumanoidWalk trains briefly with Brax PPO via Playground's own
    `learning/train_jax_ppo.py`, saves a checkpoint, then a second run *resumes* from it.
 3. Throughput benchmark: physics steps/sec for N parallel copies of our CMU humanoid and of the
    stock 21-DoF humanoid, for MJX's `jax` and `warp` backends. This number decides how many
    training steps a 12-hour Kaggle session buys us.
Everything is written to <out>/smoke_results.json. Nothing here is a claim until that file exists.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

# Must be set before JAX initialises (see the MuJoCo Playground README, "GPU Precision Issues").
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CMU_XML = ROOT / "env/assets/humanoid_cmu_rlm.xml"


def check_backend(allow_cpu: bool) -> dict:
    info = {"backend": jax.default_backend(), "devices": [str(d) for d in jax.devices()],
            "jax": jax.__version__, "mujoco": mujoco.__version__,
            "matmul_precision": os.environ.get("JAX_DEFAULT_MATMUL_PRECISION")}
    print("[smoke] backend:", info)
    if info["backend"] != "gpu" and not allow_cpu:
        raise SystemExit("JAX is not using the GPU. Fix the CUDA install before anything else.")
    return info


def run_ppo(out: Path, tiny: bool) -> dict:
    """Train HumanoidWalk briefly, then resume from its checkpoint."""
    base = [sys.executable, "-m", "learning.train_jax_ppo", "--env_name=HumanoidWalk",
            "--impl=jax", f"--logdir={out / 'ppo'}"]
    if tiny:
        base += ["--num_timesteps=20000", "--num_envs=16", "--num_eval_envs=16", "--batch_size=16",
                 "--num_minibatches=4", "--num_updates_per_batch=1", "--unroll_length=10",
                 "--num_evals=2", "--episode_length=100", "--num_videos=0"]
    else:
        base += ["--num_timesteps=5000000", "--num_evals=3", "--num_videos=0"]
    res = {}
    t = time.time()
    p1 = subprocess.run(base, capture_output=True, text=True)
    res["first_run"] = {"returncode": p1.returncode, "seconds": round(time.time() - t, 1),
                        "tail": p1.stdout[-3000:] + p1.stderr[-3000:]}
    runs = sorted((out / "ppo").glob("HumanoidWalk-*"))
    ckpt_dir = runs[-1] / "checkpoints" if runs else None
    steps = sorted(int(p.name) for p in ckpt_dir.iterdir() if p.name.isdigit()) if ckpt_dir else []
    res["checkpoints_saved"] = steps
    if p1.returncode == 0 and steps:
        t = time.time()
        p2 = subprocess.run(base + [f"--load_checkpoint_path={ckpt_dir}"], capture_output=True, text=True)
        res["resume_run"] = {"returncode": p2.returncode, "seconds": round(time.time() - t, 1),
                             "restored": "Restoring from" in p2.stdout,
                             "tail": p2.stdout[-3000:] + p2.stderr[-3000:]}
    return res


def bench(xml: Path | None, model: mujoco.MjModel | None, impl: str, nenv: int, nsteps: int) -> dict:
    from mujoco import mjx
    from mujoco_playground._src import mjx_env
    m = model if model is not None else mujoco.MjModel.from_xml_path(str(xml))
    mx = mjx.put_model(m, impl=impl)

    def init(_):
        d = mjx_env.make_data(m, impl=impl, naconmax=nenv * 200, njmax=250)
        return mjx.forward(mx, d)

    data = jax.jit(jax.vmap(init))(jnp.arange(nenv))
    ctrl = jnp.zeros((nenv, m.nu))

    @jax.jit
    def roll(d):
        def body(d, _):
            return jax.vmap(lambda dd: mjx.step(mx, dd.replace(ctrl=jnp.zeros(m.nu))))(d), None
        return jax.lax.scan(body, d, None, length=nsteps)[0]

    d = roll(data)
    jax.block_until_ready(d.qpos)  # compile
    t = time.time()
    d = roll(d)
    jax.block_until_ready(d.qpos)
    dt = time.time() - t
    del ctrl
    return {"impl": impl, "nenv": nenv, "nsteps": nsteps, "seconds": round(dt, 3),
            "physics_steps_per_sec": round(nenv * nsteps / dt), "finite": bool(jnp.isfinite(d.qpos).all())}


def run_bench(tiny: bool) -> list[dict]:
    from mujoco_playground._src.dm_control_suite import humanoid as pg_humanoid
    stock = mujoco.MjModel.from_xml_string(pg_humanoid._XML_PATH.read_text(),
                                           pg_humanoid.common.get_assets())
    nenv, nsteps = (8, 20) if tiny else (2048, 200)
    out = []
    for name, xml, model in (("cmu56", CMU_XML, None), ("stock21", None, stock)):
        for impl in ("jax", "warp"):
            try:
                r = bench(xml, model, impl, nenv, nsteps)
            except Exception as e:  # report and continue: warp may be unavailable on CPU / old GPUs
                r = {"impl": impl, "error": f"{type(e).__name__}: {e}"[:500],
                     "trace": traceback.format_exc()[-1500:]}
            r["model"] = name
            print("[smoke] bench:", {k: v for k, v in r.items() if k != "trace"})
            out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/smoke")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--tiny", action="store_true", help="minutes-scale sizes for a local CPU check")
    ap.add_argument("--skip-ppo", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {"env": check_backend(args.allow_cpu)}
    results["bench"] = run_bench(args.tiny)
    if not args.skip_ppo:
        results["ppo"] = run_ppo(out, args.tiny)
    (out / "smoke_results.json").write_text(json.dumps(results, indent=2))
    print(f"[smoke] wrote {out / 'smoke_results.json'}")


if __name__ == "__main__":
    main()
