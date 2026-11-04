# Motion notes (Component B: `motion/`)

How a CMU mocap run becomes the reference clip the policy imitates, why each step exists, and what
we measured. Rebuild with:

```
uv run python -m motion.build_cmu                       # -> data/motions/cmu_16_35{,_full}.npz + report
uv run python -m motion.playback data/motions/cmu_16_35.npz --loops 3 --keypoints --stills 4
uv run pytest tests/test_retarget.py tests/test_motion_*.py
```

## Pipeline

| step | module | what |
|---|---|---|
| parse | `motion/cmu_amc.py` | our own ASF + AMC parser and forward kinematics (metres, z-up) |
| source | `cmu_amc.to_source_motion` | 20 keypoints + 5 trusted segment rotations, as a generic `SourceMotion` |
| heading | `build_cmu` | rotate the scene so the run goes along +x and starts at x = y = 0 |
| scale | `retarget.scale_source` | each segment gets the model's length; root path scaled by leg ratio |
| IK | `retarget.solve_ik` | mink, per frame at 120 fps, warm-started |
| resample | `retarget.resample` | 120 -> 50 fps (cubic splines; slerp for the root quaternion) |
| cycle | `motion/cycle.py` | find the best one-stride seam, cut, blend, align to +x |
| smooth | `retarget.smooth` | Savitzky-Golay (window 5, order 2 at 50 fps); wraps around for the cyclic clip |
| ground | `retarget.ground_fix` | one constant vertical shift so stance feet touch z = 0 |
| velocity | `retarget.finite_diff_qvel` | central differences with `mj_differentiatePos` |
| save | `motion.format.from_qpos_qvel/save_motion` | body poses, end effectors, CoM by MuJoCo FK |

## Why these steps

**Why IK instead of copying the angles.** Our model is dm_control's CMU humanoid, so its hinges have the
same names and axes as the ASF channels, and a direct copy (what dm_control's `parse_amc.py` does) almost
works. But subject 16's bones are not the model's bones (thigh 39.7 cm vs 40.5 cm, a differently curved
spine, and so on). The same angles on different bones put the feet in different places. They hover, sink or
slide. The imitation reward cares about positions (feet, hands, pelvis), so we solve for the angles that put
the model's bodies where the source joints are. IK is also what lets the *same* code take SMPL joints from
video (Component C), where there are no CMU angles to copy at all. We keep the direct copy as a cross-check:
on the legs, IK and the copy agree to within 1–4 degrees RMS (table below), which tells us the axis conventions
are right.

**Why scale the limbs first.** IK can only reach targets the model's limbs can reach. If the source thigh is
longer, the model has to straighten its knee or lean to reach the knee target, and the error moves into the
pose. So, walking down from the pelvis, each source segment keeps its direction but takes the model's length.
The pelvis path is scaled by the leg-length ratio (model / subject = 0.969). A slightly shorter-legged runner
then takes slightly shorter strides at the same cadence, which keeps the stance foot still instead of sliding.

**What the IK solves.** For each frame mink builds a small quadratic program: find the joint step Δq that best
reduces all weighted task errors. The tasks are: a position task on 20 bodies (pelvis, hips, knees, ankles,
toes, spine, chest, neck, head, shoulders, elbows, wrists), orientation tasks on pelvis, thorax, head and feet,
and a weak posture task toward the neutral pose. Joint limits enter as hard inequality constraints
(`ConfigurationLimit`). The orientation tasks are needed because positions alone can't fix the twist of the
pelvis or of a foot about its own axis. The posture task holds joints that nothing else constrains and keeps
the problem well conditioned. Each frame starts from the previous frame's solution, so consecutive frames
differ by small, smooth steps. Fingers, thumbs, toes, wrists and hands have no data and are held at neutral
(0 rad, nudged 0.02 rad inside the range where 0 is a limit).

**Why the ground fix.** Centimetre-level offsets (different foot geometry, root height scaling) leave the
feet slightly above or below the floor. In physics that is the difference between a foot that pushes off and one
that hits the floor too early or never touches it. We take the *geom* surfaces of the foot and toe capsules and spheres
(body origins sit inside the foot), find the frames where each foot is in stance (near its lowest
point and moving < 0.8 m/s), and shift the whole clip so the 10th percentile of the stance-foot lowest points
is at z = 0. It is one constant shift: per-frame corrections would make the pelvis bob.

**Why smoothing.** Velocities come from finite differences, which amplify jitter: 2 mm of marker noise
at 50 fps becomes 0.1 m/s of velocity noise. The policy is rewarded for matching velocities, so noisy
velocities make for a noisy target. A Savitzky-Golay filter fits a low-order polynomial in a sliding window. It
removes jitter but keeps peaks (such as foot strike) better than a plain moving average. For the cyclic clip
the filter wraps around the seam, so the seam is smoothed like any other frame.

**Why `mj_differentiatePos` for velocity.** The root orientation is a quaternion. Subtracting quaternions
component-wise is not an angular velocity. `mj_differentiatePos` computes the rotation between two
orientations and expresses it in the body frame, which is exactly MuJoCo's free-joint `qvel` convention.

## How the loop works

A recorded run is 1.35 s long. The policy should run indefinitely, so we keep one stride (left strike to
next left strike) and replay it, shifted forward by `cycle_offset_pos` each time (looping rule in
`docs/interfaces.md`).

1. **Find the seam.** For every pair of frames (i, j) 0.5–1.0 s apart, compare joint angles, pelvis height and
   tilt *and their velocities* (each normalised by its spread). Comparing velocities too matters: the same pose
   can occur with the legs moving in opposite directions. The best pair for 16_35 is 50-fps frames 9 → 49
   (source frames 22 → 118), so T = 40 frames = 0.80 s.
2. **Close the gap.** Frame 49 never matches frame 9 exactly: up to 0.22 rad on one joint, 3.5° of heading,
   1.6° of pelvis tilt, 3 mm of height. We fade in a correction that makes "frame j" equal frame i. It grows
   only during *flight* frames (both feet in the air) and stays constant while a foot is planted. A first
   version blended over the last 10 frames, which swung a planted foot by 4.6 cm. Heading is corrected as a
   rigid rotation of the whole trajectory (pelvis position *and* orientation), which keeps a planted foot
   planted. With the heading drift removed, the loop runs dead straight: `cycle_offset_yaw = 0`.
3. **Align.** Rotate so the per-loop displacement is along +x: `cycle_offset_pos = (2.179, 0, 0)` m.

The price is that the cyclic clip departs from the source a little more than the full clip does (keypoint error
mean 1.69 cm vs 0.92 cm). That is deliberate. The policy imitates the .npz, and there continuity and planted feet
matter more than matching the source exactly.

## Measured checks (16_35, from `out/motion_check/report_16_35.json`)

| check | full clip (`cmu_16_35_full.npz`) | cyclic (`cmu_16_35.npz`) |
|---|---|---|
| frames / fps | 68 @ 50 | 40 @ 50 (0.80 s) |
| forward speed | 2.69 m/s | 2.72 m/s (= 2.179 m / 0.80 s) |
| keypoint error, mean / max | 0.92 / 3.49 cm | 1.69 / 6.90 cm |
| worst keypoints (mean/max cm) | thorax 2.5/3.2, lowerneck 2.5/3.5 | rtoes 3.6/6.9, rfoot 3.0/6.0 |
| joint-limit violations | 0 | 0 |
| ground shift applied | +0.8 cm | +0.5 cm |
| lowest foot geom point, whole clip | -0.67 cm | -0.16 cm |
| stance foot height (mean lowest point) | 0.05–1.88 cm | 0.53 (R), 1.51 (L) cm |
| stance slide, ankle / toe origin | 2.2–2.7 / 1.0–1.4 cm | R 2.7 / 1.0, L 3.2 / 1.8 cm |
| seam: hinge pose jump | – | 0.166 rad (median step 0.223, max 0.304) |
| seam: velocity jump | – | 3.44 (median step 4.29, max 6.33) |

Stance slide is the start-to-end horizontal distance of the ankle and toe body origins over one stance
(8–9 frames ≈ 0.17 s). Some of it is the foot rolling from heel to toe, which moves those origins even with no
slip, so a few cm is expected. The seam jumps are *smaller* than a typical frame-to-frame change, so the seam
is invisible in the video (`out/motion_check/seam_strip_37-42.png`).

IK vs dm_control-style direct angle copy, per-joint RMS (degrees, 120 fps, whole clip):

| legs | | spine / head | | arms | |
|---|---|---|---|---|---|
| lfemurrz/ry/rx | 1.1 / 1.9 / 0.9 | lowerback rz/ry/rx | 0.6 / 0.3 / 3.4 | lclaviclery | 8.0 |
| ltibiarx | 2.0 | upperback rz/ry/rx | 4.0 / 1.0 / 9.0 | lhumerus rz/ry/rx | 2.6 / 6.3 / 2.4 |
| lfoot rz/rx | 2.3 / 2.8 | thorax rz/ry/rx | 1.5 / 1.4 / 12.2 | lradiusrx | 0.9 |
| rfemurrz/ry/rx | 0.9 / 2.2 / 1.0 | lowerneck rz/ry/rx | 7.1 / 9.7 / 13.3 | rclaviclery | 4.2 |
| rtibiarx | 2.2 | upperneck rz/ry/rx | 6.1 / 7.5 / 15.9 | rhumerus rz/ry/rx | 3.4 / 3.9 / 3.0 |
| rfoot rz/rx | 2.5 / 3.9 | head rz/ry/rx | 3.1 / 4.2 / 3.3 | rradiusrx | 2.3 |

Legs and elbows agree closely. The spine and neck differ by 9–16° in flexion (rx). Subject 16's spine segments
point in different rest directions from the model's, so the same positions need different angles. IK
spreads the bend over the chain to match positions, and the total trunk orientation is matched by the thorax
orientation task. Clavicles differ because the shoulder positions are matched rather than the clavicle angles.
Undriven joints (wrists, hands, fingers, thumbs, toes) differ by up to ~40° because we hold them at neutral
where the copy follows the (noisy) recorded values.

The spare trial 16_36 was also built as a robustness check (not saved in `data/motions/`): full clip
0.98 / 4.24 cm keypoint error, 2.59 m/s, 0 limit violations. Cyclic: T = 39, offset 2.027 m, seam pose jump
0.165 rad (median step 0.201).

## Visual check

`out/motion_check/cmu_16_35.mp4` (3 loops, side camera following the pelvis, red spheres = scaled source keypoints)
and stills `out/motion_check/cmu_16_35_frame00{00,09,19,29}.png` (four phases of one stride). By eye: upright,
facing and travelling +x, the stance foot flat on the floor, left and right legs alternating (stances: right at
frame 13, left at 33), arms swinging opposite to the legs, no visible pop at the loop seam.

## For the env author

- `cmu_16_35.npz`: cyclic, **T = 40, fps = 50, 0.80 s per loop**, `cycle_offset_pos = (2.179, 0, 0)` m,
  `cycle_offset_yaw = 0`. Forward speed 2.72 m/s along +x (a jog). Root height ≈ 0.96 m mean.
- `cmu_16_35_full.npz`: not cyclic, T = 68 (1.34 s), 2.69 m/s. Useful for evaluation, not for looping.
- Frame 0 of the cyclic clip sits at x = y = 0. Root angular velocity in `qvel[:, 3:6]` is local (MuJoCo).
- `*_keypoints.npz` next to each motion holds the scaled source keypoints aligned frame-by-frame (for overlays
  and checks only; not part of the contract).
- `meta_json` holds the retarget settings, the cut (`frame_range_50fps`, `frame_range_src`) and seam info.

## For Component C (video)

Build a `motion.retarget.SourceMotion`: `names[0]` must be the pelvis mapped to `"root"`, `parents` index into
`names`, positions in metres with world z up, and an optional `rot` for joints in `rot_valid`, expressed as the
world rotation of the *model body* (so it must already include any rest-frame offset). `retarget.SMPL_BODY_MAP`
is a suggested SMPL → model map. It is untested. Note that SMPL's pelvis sits lower than our root origin.
Then reuse the same post-processing as `build_cmu.build` (resample → cycle → smooth → ground fix → velocities).

## Known limitations

- Both CMU clips are short (1.35–1.6 s), so there are few candidate seams. There are only about 1.7 strides to
  choose from.
- The stance detector (height + speed threshold) is simple. It finds 8–9 frames of stance per step, and
  treats the rest of the stride, including heel-strike and toe-off frames, as "flight" for the seam blend.
- dm_control's thumb bodies use their own frame (≈20–30° from the ASF thumb axis). Irrelevant here because
  thumbs are held neutral.
- `body_pos` follows *our* model. Hands carry no source data, so `ee_pos` for hands follows the wrist and
  forearm only.
