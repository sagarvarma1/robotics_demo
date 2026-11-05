# Notes on the imitation environment (`env/`)

Plain-language notes on how `env.imitation.RunImitation` is built and why. The code docstrings say
the same things closer to the code. This page puts them in one place. Paper: Peng, Abbeel, Levine,
van de Panne, *DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character
Skills*, SIGGRAPH 2018.

## The loop in one paragraph

Every 20 ms the policy looks at the body (observation), outputs 56 target joint angles (action),
and PD servos in the simulator turn those targets into torques for four 5 ms physics steps. We
then look up where the reference clip says the body should be at this moment and reward closeness
(tracking reward). The episode ends early if the character falls or drifts off the reference.
Episodes start at a random point of the clip, already moving the way the clip moves.

## Files

| file | what |
|---|---|
| `env/imitation.py` | the env class: config, reset (RSI), step, PD gain hooks |
| `env/reference.py` | reference lookup by time, looping with offsets, interpolation |
| `env/obs.py` | observation layout (139 numbers by default) and the heading frame |
| `env/reward.py` | the four DeepMimic reward terms |
| `env/termination.py` | early termination: bad floor contact, root drift, NaN |
| `env/quat.py` | quaternion helpers |
| `train/brax_ppo.py` | Brax PPO launcher, run dirs, checkpoints, resume |
| `tests/fixtures_motion.py` | synthetic stand / march clips for testing |

## Observation

What the policy sees (full layout table in `env/obs.py`): joint angles and velocities, root
height, root tilt, root linear and angular velocity, hands and feet relative to the root, and
the clip phase.

**Why the heading frame.** Running north or running east is the same skill, so the policy
shouldn't have to learn it twice. We express every direction-dependent quantity in a frame that
turns with the body's facing direction, but only about the vertical axis (its "heading"). In that
frame x is always "forward along the ground" and z is "up". Leaning and falling are still visible,
because only the yaw has been taken out. Absolute x/y position is left out for the same reason:
where you are on the floor doesn't change how to run.

**Why phase.** The policy isn't shown the reference pose. It's told how far through the clip it
is, as sin/cos of 2π·phase, so that the end and the start of a loop are close together. Without
it, the same body state (mid-stride, say) could call for different actions at different points
of the clip and the policy couldn't tell them apart. This matches DeepMimic, which gives its
policy the phase variable as its only information about the reference.

**Optional future targets.** `env.obs.future_steps: [1, 5]` (for example) adds the reference
pose and root position that many policy steps ahead. That helps with long, non-repeating clips.
For a single cyclic running clip, phase already says everything, so it's off by default.

**Rotation as 6D.** The root's tilt is given as two columns of its rotation matrix in the heading
frame. Quaternions have a sign flip (q = −q) and Euler angles jump, which are both bad for neural
networks. Two matrix columns are smooth and unique.

## Actions: PD targets, not torques

Each of the 56 actions in [−1, 1] is mapped linearly onto its joint's range to give a target
angle. A MuJoCo `general` actuator applies `kp·(target − q)` and joint damping applies `−kd·q̇`
(see `scripts/build_humanoid.py` and decision D2). So the policy says *where* to go and the
servo works out the torque. DeepMimic found PD targets learn faster and look better than
direct torques, because even a random policy then holds a roughly sensible posture.

**Action order = actuator order, which is alphabetical** (`headrx, headry, headrz,
lclaviclery, …`), *not* the qpos order of the hinges (`lfemurrz, lfemurry, …`). For the policy
this makes no difference. For any code that converts poses to actions, use
`env.pose_to_action(hinge_qpos)` and `env.action_to_targets(action)`, which go through
`actuator_trnid`. (We found this the hard way: assuming the same order made a standing pose
collapse within one step.)

**Gain hooks.** `env.pd.kp_scale` multiplies the actuator gain and bias terms (target angles
unchanged), and `env.pd.kd_scale` multiplies the hinge joint damping. Both are applied once to the
model before it's handed to MJX.

**Can the gains hold a pose?** Test `test_pd_holds_pose_with_root_fixed`: root pinned in the air,
gravity on, target = a random pose within ±30% of each joint's half-range. After 1 s the mean
error was 0.27° and the max 1.5° (the spine joints sag slightly under gravity). After 0.1 s the
mean was 5.6° and the max 25° (big arm and leg moves are still under way). So yes, with time to
settle. Watch out: at the clipped all-zero "standing" pose, 10 joints (knees at their 1° lower
limit, clavicles, wrists, fingers, thumbs) sit *on* a joint limit, and there the limit does the
holding, not the servo.

## Reference state initialisation (RSI)

`reset` picks a uniformly random time in the clip (loop 0) and puts the body exactly on the
reference there: pose, joint velocities, root velocity. Why it matters so much: with a fixed start,
the policy has to master the first stride before it ever gets to see the later ones. Some states,
such as the flight phase of a running stride, are practically impossible to reach by exploring
from a standing start. With RSI every part of the clip is practised from the first iteration.
DeepMimic's ablations (section 9 of the paper) show that without RSI, dynamic skills such as
backflips aren't learned at all, and others learn much more slowly.

The Brax wrapper is built with `full_reset=True` in `train/brax_ppo.py`, so every new episode
draws a fresh random start. With Playground's default (`False`), each parallel env always
restarts from its own first cached start state. The reference clock is stored in `data.time`, so
the reference stays in sync with the body in either mode.

## Early termination (ET)

The episode ends when the floor touches any body other than the feet and toes (knee, hand,
pelvis, torso, head), when the root is more than 0.5 m from the reference root, or on NaN.
Why it matters so much: without it, a fallen character keeps collecting data, so most of the
training data shows it lying on the floor, and the value function learns about lying on the
floor. Ending the episode at the fall keeps the data on-task and acts as a penalty, because a
fallen character stops earning the (always positive) reward. In DeepMimic's ablations, policies
trained without ET for running and backflips ended up on the ground and failed to learn the
motion.

Contacts come from MJX's fixed-size contact buffer: 164 slots, one per geom pair allowed by
the collision filter. A slot is touching when its signed distance is negative. See
`env/termination.py`.

## Reward

A weighted sum of four `exp(−k·error)` terms, each 1 at a perfect match. The weights sum to 1, so
the reward is in [0, 1]:

| term | error | weight | k |
|---|---|---|---|
| pose | Σ over bodies of (angle between sim and reference *local* rotation)², root uses its world rotation | 0.65 | 2 |
| vel | Σ over hinges of (joint velocity difference)² | 0.10 | 0.1 |
| ee | Σ over lhand, rhand, lfoot, rfoot of ‖(p − p_root)_sim − (p − p_root)_ref‖² | 0.15 | 40 |
| com | ‖CoM_sim − CoM_ref‖² (world) | 0.10 | 10 |

Local rotations (each body relative to its parent) are used so that one wrong elbow counts once,
not again for every body below it. Exponentials give a smooth, bounded signal: every small
improvement pays, and one huge error can't swamp everything. Per-term values go to
`state.metrics` (`r_pose, r_vel, r_ee, r_com`) along with `mpjpe` (mean world-frame body position
error, metres), `root_drift` and `loops`.

## Looping the clip

A running clip is a few strides long, but an episode lasts 10 s. At loop k the clip is replayed
shifted forward by k·`cycle_offset_pos` (and turned by k·`cycle_offset_yaw`), so the target keeps
running forward instead of teleporting back (docs/interfaces.md, section 1). If the clip's fps
isn't the policy rate (50 Hz), consecutive frames are blended linearly, also across the loop
boundary.

## Deviations from the paper (and why)

- **End-effector term relative to the root.** DeepMimic uses world positions. Ours subtracts the
  root position so that being in the wrong place overall isn't counted twice. The CoM term
  already covers it.
- **Velocity term covers hinge joints only.** DeepMimic includes the root. The project brief
  specified hinges only.
- **Policy rate 50 Hz** (DeepMimic: 30 Hz), matching Playground's humanoid settings (D4).
- **Early termination by contact**, as in DeepMimic. We also added a root-drift limit
  (0.5 m), which DeepMimic didn't have for most skills. It stops episodes where the character is
  upright but no longer following the motion.
- **No pushes / randomisation yet.** The config groups exist as placeholders and raise an error if
  enabled.
- **Gains** come from dm_control's position-controlled CMU walker, not DeepMimic's character.
