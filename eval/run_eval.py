"""Evaluate a trained run or a baseline: metrics, push test, videos, plots.

    uv run python -m eval.run_eval --run runs/mocap_run_v1-20261104-160000 [--step N]
    uv run python -m eval.run_eval --baseline refpd --episodes 16 --seconds 30
    uv run python -m eval.run_eval --baseline zero --pushes-per-level 8 --forces 0,200,400

Writes out/eval/<name>/:
  metrics.json      tracking/stability metrics over `--episodes` episodes of up to `--seconds`
                    (eval/metrics.py). Episodes start at random phases (RSI) with fixed seeds.
  push.json         push-survival sweep (eval/push_test.py)
  side_by_side.mp4  [reference | simulated] for the episode whose survival time is closest to the
                    median (a typical episode, not a hand-picked best one)
  overlay.mp4       the same episode with the reference drawn as a ghost
  push.mp4          one pushed episode (see `push_video` in push.json for which and why)
  plots/            training curves (runs only), push curve, episode curves. `--compare
                    out/eval/<other>` adds other evaluations to the push and episode plots.

Everything is deterministic given --seed (policy in deterministic mode unless --stochastic).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from eval import plots
from eval.metrics import evaluate, write_json
from eval.policy import DEFAULT_BASELINE_CONFIG, load_baseline, load_run
from eval.push_test import PushConfig, run_push_test
from eval.render import render_episode
from eval.rollout import PushPlan, rollout

ROOT = Path(__file__).resolve().parents[1]
PUSH_CAM = {"distance": 4.5, "azimuth": 135.0, "elevation": -25.0}


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def typical_episode(surv: np.ndarray) -> int:
    return int(np.argmin(np.abs(surv - np.median(surv))))


def choose_push_episode(rollouts: dict) -> tuple[float, int, str] | None:
    """Pick (force, episode, why) for the push video from the sweep's rollouts.

    Preference: the largest force that has a survived push (shows a recovery), else the smallest
    force > 0 with a valid but failed push (shows a failure). None if no push was ever valid.
    """
    forces = sorted(f for f in rollouts if f > 0)
    for f in reversed(forces):
        s = rollouts[f][1]
        if s["survived"].any():
            return f, int(np.argmax(s["survived"])), "largest force with a survived push"
    for f in forces:
        s = rollouts[f][1]
        if s["valid"].any():
            return f, int(np.argmax(s["valid"])), "smallest force with a valid (failed) push"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="run directory (runs/<name>-<stamp>)")
    src.add_argument("--baseline", choices=["zero", "refpd"])
    ap.add_argument("--step", type=int, help="checkpoint step (default: latest)")
    ap.add_argument("--config", default=str(DEFAULT_BASELINE_CONFIG), help="env YAML for baselines")
    ap.add_argument("--stochastic", action="store_true", help="sample actions instead of the mode")
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=30.0, help="episode cap (training uses 10 s)")
    ap.add_argument("--forces", default=",".join(str(f) for f in range(0, 601, 50)))
    ap.add_argument("--pushes-per-level", type=int, default=32)
    ap.add_argument("--push-duration", type=float, default=0.1)
    ap.add_argument("--push-warmup", default="1.0,2.0", help="push onset window, seconds after reset")
    ap.add_argument("--recovery", type=float, default=2.0)
    ap.add_argument("--push-body", default="root", choices=["root", "thorax"])
    ap.add_argument("--fail-on", default="env", choices=["env", "fall"])
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--video-seconds", type=float, default=10.0, help="cap on side-by-side length")
    ap.add_argument("--compare", action="append", default=[], help="other out/eval/<name> dirs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/eval")
    ap.add_argument("--name")
    args = ap.parse_args()

    t0 = time.time()
    if args.run:
        env, policy, env_cfg = load_run(args.run, args.step, deterministic=not args.stochastic)
        name = args.name or policy.name.replace("@", "_step")
    else:
        env, policy, env_cfg = load_baseline(args.baseline, args.config)
        name = args.name or f"baseline_{args.baseline}"
    out = (ROOT / args.out / name) if not Path(args.out).is_absolute() else Path(args.out) / name
    (out / "plots").mkdir(parents=True, exist_ok=True)
    print(f"[run_eval] {policy.name} -> {out}", flush=True)

    # 1. long episodes, unperturbed
    r = rollout(env, policy, args.episodes, args.seconds, seed=args.seed, verbose=True)
    res = evaluate(r, extra={"policy_info": policy.info, "env_config": env_cfg, "seed": args.seed})
    write_json(res, out / "metrics.json")
    h = res["headline"]
    print(f"[run_eval] survival median {h['survival_s_median']:.2f} s, MPJPE median "
          f"{h['mpjpe_m_median']}, root-rel {h['mpjpe_rootrel_m_median']}", flush=True)

    if not args.no_video:
        ep = typical_episode(r.survival_s())
        tail = 1.0
        # cap the video length (a 30 s side-by-side is mostly the same stride over and over)
        vmax = int(round(args.video_seconds / r.dt))
        rv = _truncate(r, vmax)
        for mode, fname in (("side", "side_by_side.mp4"), ("overlay", "overlay.mp4")):
            info = render_episode(env.mj_model, rv, ep, out / fname, mode=mode, tail_s=tail,
                                  title=policy.name)
            print(f"[run_eval] wrote {info['path']} ({info['frames']} frames, episode {ep})", flush=True)
        res["video"] = {"episode": ep, "survival_s": float(r.survival_s()[ep]),
                        "why": "episode with survival time closest to the median"}
        write_json(res, out / "metrics.json")

    # 2. push sweep
    if not args.no_push:
        pcfg = PushConfig(forces=_floats(args.forces), n_per_level=args.pushes_per_level,
                          duration_s=args.push_duration, warmup_s=tuple(_floats(args.push_warmup)),
                          recovery_s=args.recovery, body=args.push_body, fail_on=args.fail_on,
                          seed=args.seed)
        pres, prolls = run_push_test(env, policy, pcfg, keep_rollouts=not args.no_video, verbose=True)
        pres["policy_info"] = policy.info
        if not args.no_video:
            pick = choose_push_episode(prolls)
            if pick is not None:
                F, ep, why = pick
                pr = prolls[F][0]
            else:
                # No push was ever valid (the policy falls before the warm-up ends). Still make a
                # clip that shows the push mechanics: push early, at the median force level.
                F = float(np.median([f for f in pcfg.forces if f > 0] or [100.0]))
                n_steps = pres["config"]["duration_steps"]
                plan = PushPlan(start_step=np.array([5]), force=np.array([[0.0, F, 0.0]]),
                                n_steps=n_steps, body=pcfg.body)
                pr = rollout(env, policy, 1, 5 * env.dt + n_steps * env.dt + pcfg.recovery_s,
                             seed=args.seed, push=plan)
                ep, why = 0, ("no valid push in the sweep (policy fell during warm-up); "
                              "demo push at step 5 instead, NOT part of the statistics")
            info = render_episode(env.mj_model, pr, ep, out / "push.mp4", mode="push", tail_s=1.0,
                                  cam=PUSH_CAM, title=policy.name)
            pres["push_video"] = {"force_N": F, "episode": ep, "why": why,
                                  "start_step": int(pr.push.start_step[ep]), "frames": info["frames"]}
            print(f"[run_eval] wrote {info['path']} ({why})", flush=True)
        write_json(pres, out / "push.json")
        print(f"[run_eval] push headline: {pres['headline']}", flush=True)

    # 3. plots
    if args.run:
        plots.training_curves(Path(args.run), out / "plots/training_curves.png")
    pj = {name: out / "push.json"} if (out / "push.json").exists() else {}
    mj = {name: out / "metrics.json"}
    for c in args.compare:
        c = Path(c)
        if (c / "push.json").exists():
            pj[c.name] = c / "push.json"
        if (c / "metrics.json").exists():
            mj[c.name] = c / "metrics.json"
    if pj:
        plots.push_curve(pj, out / "plots/push_survival.png")
    plots.episode_curves(mj, out / "plots/episode_curves.png")
    print(f"[run_eval] done in {time.time() - t0:.0f} s -> {out}", flush=True)
    print(json.dumps({"metrics": res["headline"],
                      "push": json.loads((out / "push.json").read_text())["headline"]
                      if (out / "push.json").exists() else None}, indent=2))


def _truncate(r, n: int):
    """A view of the Rollout cut to its first n steps (for shorter videos)."""
    import dataclasses
    if r.n_steps <= n:
        return r
    cut = {f.name: getattr(r, f.name) for f in dataclasses.fields(r)}
    for k, v in cut.items():
        if isinstance(v, np.ndarray) and v.ndim >= 2 and v.shape[1] == r.n_steps + 1:
            cut[k] = v[:, :n + 1]
    cut["env_metrics"] = {k: v[:, :n + 1] for k, v in r.env_metrics.items()}
    return type(r)(**cut)


if __name__ == "__main__":
    main()
