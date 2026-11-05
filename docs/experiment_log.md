# Experiment log

One entry per run or failure: **symptom → diagnosis → proposed fix → result**. A diagnosis is written down
*before* anything is changed. Numbers here are copied from `runs/*/metrics.jsonl` or `eval/` outputs, never typed
from memory.

| date | run | what | result |
|---|---|---|---|
| 2026-11-04 | `mocap_run_v1-tiny` (CPU, 16 envs, 30k steps) | End-to-end + resume check on the real clip | Works after fixing two bugs: (1) orbax rejects relative checkpoint paths, so run dirs and resume paths are now `.resolve()`d (`tests/test_rundir.py`); (2) actuators were alphabetical, not qpos order, and are now emitted in qpos order by `build_humanoid.py`. No learning expected at this scale. |

### Open issue: velocity reward term is always ≈ 0
**Symptom.** In the tiny run `r_vel` is 0.0 at every eval. Even one step after a reset to the exact reference
state, with PD targets set to the reference, `r_vel` = 0.076 while `r_pose` = 0.985 and `r_ee` = 0.979.
**Diagnosis.** err_vel = Σ over **56** hinges of (Δ joint velocity)², with k = 0.1. Running joint velocities are
several rad/s, so a small lag per joint, summed over 56 joints, already gives err ≈ 25 and exp(-2.5) ≈ 0.08. DeepMimic's
scale was tuned for its own character, which has far fewer degrees of freedom and spherical joints. With r_vel stuck at 0,
the term gives no learning signal and effectively caps the reward at 0.9.
**Proposed fix (not applied yet).** Exclude the 16 hand/finger/thumb/toe/wrist joints, which have no source data, and
use the *mean* squared error over the remaining joints with k chosen so a "good" tracking error gives about 0.5.
Decide after the first GPU baseline run, where we can measure the error distribution of a policy that is actually learning.
**Fix applied (2026-11-05), before the first GPU run.** Measured on the reference-PD baseline (16 resets): summed
over 56 hinges, the error was 33 one step after a reset and 124 after 10 steps; averaged over the 42 joints with data
it was 0.78 and 2.8. Now: mean over data joints (`reward.vel_exclude` regex), with k_vel = 1.0, which gives
r_vel ≈ 0.46 near the reference and ≈ 0.06 while falling. Check: r_vel one step after a reset went 0.076 → 0.476.
