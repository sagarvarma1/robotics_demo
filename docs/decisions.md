# Design decisions (and why)

Plain-language notes on every non-obvious choice. Newest at the bottom. When a decision changes,
add a new entry that says what changed and why, rather than editing the old one.

### D1. Body model: dm_control CMU humanoid (56 hinges)
Its joints are named and oriented after the CMU mocap skeleton (`lfemurrx`, …), so CMU clips map onto it
almost one-to-one, and SMPL's joint tree is close to it too. Licensed Apache-2.0. Alternative kept in reserve:
dm_control's 21-DoF humanoid, if the GPU step speed of 56 DoF turns out too slow (measured in the smoke test).

### D2. PD control with MuJoCo position actuators
The policy outputs a *target angle* per joint, not a torque. A PD controller (`tau = kp (q* - q) - kd q̇`)
acts like a spring pulling each joint toward its target. Why this helps learning: a random policy with
torque control flails, while one with PD targets holds a roughly sensible posture, so useful behaviour is
much easier to stumble into (DeepMimic, Peng et al. 2018, §6). The policy's action in [-1, 1] maps
linearly onto each joint's range. MuJoCo computes the spring term in the actuator and the damper term
as joint damping. The `implicitfast` integrator treats damping implicitly, which stays stable with stiff
gains at a 5 ms step. Starting gains come from dm_control's position-controlled CMU walker.
See `scripts/build_humanoid.py`.

### D3. Collision filtering
Body parts collide with the floor, and the lower legs and feet also collide with the *other* leg. No other
self-collision. Why: MJX's JAX backend evaluates every candidate collision pair every step, so the cost
grows with the number of pairs. With filtering the model needs a 164-slot contact buffer. Leg-leg contact
is kept so the policy cannot pass its feet through each other.

### D4. Timing: 5 ms physics step, 20 ms policy step (50 Hz)
Physics needs small steps for stiff contacts. The policy doesn't need to decide 200 times a second, and
fewer decisions per second make credit assignment easier for RL. DeepMimic used 30 Hz. We start at 50 Hz,
which matches Playground's humanoid settings.

### D5. Precomputed reference quantities
See `motion/format.py`: body positions, end effectors and CoM are computed once, offline, with MuJoCo
forward kinematics on *our* model. Training then only does array lookups.

### D6. Sensors removed from the model
Everything the policy observes is computed from qpos/qvel/xpos directly. Touch sensors are not supported
by MJX, and we detect ground contact from the contact list instead.
