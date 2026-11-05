# Evaluation notes (Component D: `eval/`)

How we measure a policy: how well it tracks the clip, how long it stays up, and how hard you can
push it before it falls. The code docstrings say the same things next to the code.

```
uv run python -m eval.run_eval --run runs/<run-dir> [--step N]      # trained checkpoint
uv run python -m eval.run_eval --baseline refpd                     # or zero
uv run python -m eval.run_eval --run runs/<dir> --compare out/eval/baseline_refpd --compare out/eval/baseline_zero
uv run pytest tests/test_eval_*.py
```

Output: `out/eval/<name>/{metrics.json, push.json, side_by_side.mp4, overlay.mp4, push.mp4, plots/}`.

## Files

| file | what |
|---|---|
| `eval/policy.py` | load a Brax PPO checkpoint (env rebuilt from the run's `config.yaml`), plus the `zero` and `refpd` baselines |
| `eval/rollout.py` | run N episodes in parallel (vmap), record every step, apply external pushes |
| `eval/metrics.py` | MPJPE (world, root-relative), drift, survival, distance, strides; summaries, JSON |
| `eval/push_test.py` | push-recovery sweep with Wilson confidence intervals |
| `eval/render.py` | side-by-side, overlay (ghost) and push videos; hook for the user's own video |
| `eval/plots.py` | training curves, push-survival curve, survival/tracking-vs-time curves |
| `eval/run_eval.py` | the CLI that runs all of the above |

## The baselines (why they exist)

* **zero**: every action 0, i.e. every PD target is the middle of the joint's range. The servos
  hold one fixed posture and the character falls. It's the absolute floor.
* **refpd** ("reference PD"): action = the reference joint angles of the next step, fed open-loop
  to the PD servos. It starts on the reference with the right momentum (RSI), so for a moment it
  looks right, but nothing corrects balance, and it falls within a stride. Any trained policy
  must beat it clearly, otherwise it has learned nothing about balance. It reads the clock
  (`data.time`), which a real policy doesn't have.

## How episodes are run

* Start at a random phase of the clip, exactly on the reference (the same RSI as training), with
  fixed seeds, so evaluations are repeatable.
* Episode cap 30 s by default, 3x the training episode (10 s). A policy can learn "survive about
  10 s", and the longer cap reveals slow divergence.
* No reset after termination. Physics keeps running (so the videos show the fall), but **every
  metric only uses steps before termination** (`alive` mask, step 0 = the reset state excluded).
* Termination = the env's own rule (non-foot floor contact, root > 0.5 m from the reference root,
  or NaN), so eval and training agree on what "failed" means. The cause is recorded per episode.
* The policy uses its deterministic action (the mean of its Gaussian) unless `--stochastic`.

## Metrics (metrics.json)

All are per episode, then summarised as mean, median, IQR (25th to 75th percentile), min and
max. Use the median and IQR: survival times are often bimodal (most fall early, a few run to the
cap), and the mean then describes no actual episode.

| metric | meaning | why |
|---|---|---|
| `survival_s` | time until termination (capped) | the first number to read; errors from 0.4 s and from 30 s aren't comparable |
| `cap_share` | share of episodes that reached the cap | "solved" means near 1 |
| `mpjpe` | mean distance (m) between each simulated body and its reference body, averaged over 31 bodies and alive steps | the standard tracking error, same as the training log's `mpjpe` |
| `mpjpe_rootrel` | same after subtracting each skeleton's own pelvis position | pose error without drift. If world MPJPE is much larger than this, the shape is right but the place is wrong |
| `root_drift_*` | pelvis distance from the reference pelvis (3-D and horizontal; mean, max, final) | is it falling behind or running ahead? Termination at 0.5 m |
| `distance_m`, `progress_ratio` | straight-line horizontal distance covered while alive, and that / the reference's distance in the same time | 1.0 = keeps pace |
| `ref_cycles` | whole strides of reference time (0.80 s each) survived | "how many strides did it keep going" |
| `progress_cycles` | distance / per-stride displacement (2.18 m) | "how many strides' worth of ground". Standing still survives but scores 0 |
| `over_time` | share alive, and median errors of the alive episodes, every 0.1 s | for the curves |

**Headline tracking number:** median of per-episode `mpjpe` (world), quoted together with median
survival time.

## Push test (push.json)

Protocol (`eval/push_test.py`, defaults):

1. Reset at a random phase. Run unperturbed for a random warm-up of **1.0 to 2.0 s** (randomised per
   episode, so pushes land at every phase of the stride).
2. Push: horizontal force **F**, random direction (0 to 360°), on the **pelvis (`root`)**, for
   **0.1 s** (5 policy steps), via MuJoCo `xfrc_applied` (applied at the body's centre of mass).
   We push the pelvis because it's the root of the body tree near the centre of mass (DeepMimic pushes
   the pelvis too). `--push-body thorax` is harder: it adds a big tipping torque.
3. **Survived** = no termination from push onset until **2.0 s** after the push ends.
   `--fail-on fall` ignores drift and counts only falls and NaN (a good recovery may still leave the
   character > 0.5 m from the reference).
4. **Valid** pushes only: an episode that terminated during warm-up was never pushed. It's counted as
   `n_invalid` and left out of the share. If no push at a level is valid, the share is `null`, not 0.
5. Sweep **F = 0, 50, ..., 600 N**, **32 pushes per level**. F = 0 is the control (survival of the
   same window without a push).
6. Per level: `share` = survived / valid, with a **95% Wilson interval** (stays within [0, 1] and is
   honest at small n. At n = 32 and 50% it is about ±16 points).

**Force vs impulse.** Momentum changes by the impulse J = F × duration (N·s), so Δv = J / m. Our
humanoid is 51.8 kg, so 300 N for 0.1 s = 30 N·s ≈ 0.58 m/s for the whole body. 150 N for 0.2 s
is the same impulse. `push.json` lists `impulse_Ns` and `delta_v_mps` for every level so results
with different durations can be compared.

**Headline push numbers:** `pooled_share_F_gt_0` (share survived across all F > 0) and `F50_N` (force
where the share first drops below 50%, interpolated).

**Does the force reach the physics?** `tests/test_eval_push.py`: same start, same open-loop
actions, one run pushed with 400 N along +y for 0.1 s. The pelvis's sideways velocity changes by
+1.32 m/s in +y (and < 0.05 m/s along x and z). The force is recorded only inside the push window.

## Videos

* `side_by_side.mp4`: [reference | simulated], same camera settings, each following its own pelvis,
  frame t of both panels = recorded step t (same clock). Labels show time, phase, root drift, and
  "TERMINATED" after the end. The episode shown is the one with survival time closest to the
  median (typical, not cherry-picked). Capped at 10 s (`--video-seconds`).
* `overlay.mp4`: the same episode, with the reference as a translucent blue ghost in the same scene.
* `push.mp4`: a red arrow and sphere on the pelvis while the force acts (thin pink arrow for 0.5 s
  after), from an oblique camera so the direction is visible. `push.json → push_video` says
  which push and why. For policies that never survive the warm-up, it's a demo push at step 5,
  labelled as not part of the statistics.
* Hook for the user's video (Component C): `render_episode(..., left_frames=[...],
  left_phases=[...])` adds a left panel, picked by phase (or by time at `left_fps`).

## How to read the plots (plots/)

* `training_curves.png` (trained runs only): eval episode return, eval episode length, MPJPE vs
  environment steps. Return grows both with better tracking and with longer episodes, so read it
  next to episode length (cap 500 = 10 s). MPJPE may briefly rise while episodes get longer.
* `push_survival.png`: share survived vs force, shaded = 95% interval. Where it crosses the dotted
  50% line is F50. Overlapping bands = not a reliable difference. Levels with no valid pushes are
  left out (noted on the plot).
* `episode_curves.png`: left, share of episodes still running vs time (a survival curve). Right,
  median MPJPE of the still-running episodes vs time. A rising right curve means tracking slowly
  degrades. Watch out: late points come from fewer (survivor) episodes.

## Measured baselines

Measured on CPU on 2026-11-04/05 (`out/eval/baseline_*/metrics.json` and `push.json`). 16 episodes,
30 s cap, random start phase, seed 0, env = `configs/mocap_run_v1.yaml`. Median [IQR].

| | zero | refpd |
|---|---|---|
| survival (s) | 0.43 [0.38, 0.44] | 0.74 [0.56, 0.79] |
| episodes reaching the 30 s cap | 0 / 16 | 0 / 16 |
| termination cause | drift 10, contact 6 | drift 15, contact 1 |
| MPJPE world (m) | 0.287 [0.276, 0.348] | 0.272 [0.256, 0.286] |
| MPJPE root-relative (m) | 0.267 [0.259, 0.269] | 0.147 [0.107, 0.159] |
| final horizontal root drift (m) | 0.237 | 0.170 |
| distance travelled (m) | 0.87 | 1.74 |
| progress ratio (vs reference) | 0.78 | 0.93 |
| reference strides survived | 0 | 0 (mean 0.25) |

These per-episode MPJPEs are averages over short episodes whose error grows every step (about
0.05 m after 0.1 s for refpd, 0.75 m just before it terminates; `plots/episode_curves.png`). So
they are not comparable with a policy's MPJPE over 30 s. Always read them next to survival time.

**Push test, default protocol** (warm-up 1–2 s, 2 s recovery, 32 pushes × 13 levels, 0–600 N): both
baselines fall before *every* push (0 valid pushes at every level), so the push share is undefined
(`null`). That is the correct answer for policies that can't run on their own.

**Push test, short-window check** (refpd only, warm-up 0.1–0.2 s, recovery 0.3 s, 32 pushes per level;
`out/eval/baseline_refpd_shortpush/`), to confirm the sweep responds to force:

| F (N) | impulse (N·s) | whole-body Δv (m/s) | survived / valid | 95% CI |
|---|---|---|---|---|
| 0 | 0 | 0 | 20 / 32 | 0.45–0.77 |
| 100 | 10 | 0.19 | 16 / 32 | 0.34–0.66 |
| 200 | 20 | 0.39 | 20 / 32 | 0.45–0.77 |
| 300 | 30 | 0.58 | 12 / 32 | 0.23–0.55 |
| 400 | 40 | 0.77 | 11 / 32 | 0.20–0.52 |
| 500 | 50 | 0.96 | 6 / 32 | 0.09–0.35 |
| 600 | 60 | 1.16 | 5 / 32 | 0.07–0.32 |

Survival falls with force as it should (F50 ≈ 250 N, pooled F > 0 share 0.36). Only 62% survive even
with no push, because refpd falls on its own within ~0.8 s. The 0–200 N levels are
indistinguishable at n = 32 (overlapping intervals).

## Recommended env changes (not made: env/ belongs to Component A)

* A push hook in the env (`push.enable`), so push *training* (robustness) uses the same mechanism
  as this test. Simplest: set `data.xfrc_applied` on the root during a sampled window inside
  `step`, as `eval/rollout.py` does.
* Keep the time limit out of the raw env (it lives in Brax's wrapper today), so eval can run 30 s episodes.
* Expose termination flags (contact / drift / nan) in `state.metrics` or `state.info`, so eval doesn't
  have to recompute them (eval recomputes them with `env/termination.py`'s public functions).
