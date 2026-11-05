"""Run evaluation episodes and record everything the metrics, push test and videos need.

How it runs. N episodes step in parallel (`jax.vmap` over env states). Time is advanced in
*chunks*: one jitted function does `chunk` policy steps with `jax.lax.scan` and returns the
per-step records, and a plain Python loop calls it until the episode cap is reached. Why chunks
rather than one scan over the whole 30 s: (1) the same compiled function serves any episode length
(no recompiling for 10 s vs 30 s), (2) we can stop early once every episode has ended (a falling
baseline doesn't need 1500 steps), and (3) records come back to numpy in small pieces.

Why the episode cap is longer than training (default 30 s vs 10 s). Training episodes are 10 s,
so a policy may have learned to survive "about 10 s" and then drift or tip over. Evaluating
3x longer shows whether the gait is a stable cycle or slowly falling apart. Note that the raw env
has no time limit (that lives in Brax's training wrapper), so the cap is ours alone.

Termination and masking. The env sets `done` when the character falls (bad floor contact), drifts
more than `termination.max_root_drift` from the reference root, or produces NaNs. We do *not*
reset after `done`. Instead:
  * `alive[:, t]` is True while the episode has not terminated at any step <= t
    (a cumulative product of `1 - done`, so a character that falls and happens to stop touching
    the floor does not come back to life).
  * Physics keeps running after termination, so the videos show the fall itself, but every metric
    only uses steps where `alive` is True. If the state ever becomes non-finite, that episode is
    frozen at its last finite state (so NaN can't spread into averages or the renderer).
  * Index 0 is the reset state (on the reference, error 0). Step t is the state after t policy
    steps, i.e. at time t0 + t*dt.

External pushes (for eval/push_test.py) are applied here, from the eval side, because the env
has no push hook yet. MuJoCo's `data.xfrc_applied[body] = (force_xyz, torque_xyz)` is a Cartesian
wrench in the world frame applied at the body's centre of mass. It is part of `data` and is not
cleared by `mjx.step`, so we set it before every `env.step` (force inside the push window, zero
outside it). The env's step runs 4 physics substeps with the same data, so the force acts for the
whole 20 ms policy step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from env import termination as term_lib
from env.imitation import RunImitation

from eval.policy import EvalPolicy


@dataclass
class PushPlan:
    """One push per episode. start_step [N] (policy step at which the push starts), force [N, 3]
    (newtons, world frame), n_steps (duration in policy steps), body (name of the body pushed)."""
    start_step: np.ndarray
    force: np.ndarray
    n_steps: int
    body: str = "root"


@dataclass
class Rollout:
    """Per-step records, numpy, shape [N, T+1, ...] (index 0 = reset state)."""
    dt: float
    clip_duration: float
    cycle_offset: np.ndarray        # [3] reference root displacement per clip loop
    body_names: list[str]
    root_index: int                 # index of the root in body_pos[..., B, :]
    t: np.ndarray                   # [N, T+1] simulator/reference clock (s)
    qpos: np.ndarray                # [N, T+1, nq] simulated
    qpos_ref: np.ndarray            # [N, T+1, nq] reference at the same clock
    body_pos: np.ndarray            # [N, T+1, B, 3] simulated world body positions
    body_pos_ref: np.ndarray        # [N, T+1, B, 3] reference world body positions
    phase: np.ndarray               # [N, T+1]
    reward: np.ndarray              # [N, T+1]
    done: np.ndarray                # [N, T+1] env done flag at this step (not cumulative)
    fell: np.ndarray                # [N, T+1] bad floor contact
    drifted: np.ndarray             # [N, T+1] root drift above the env limit
    nan: np.ndarray                 # [N, T+1]
    alive: np.ndarray               # [N, T+1] not terminated at or before this step
    force: np.ndarray               # [N, T+1, 3] external force applied *during the step into t*
    env_metrics: dict[str, np.ndarray] = field(default_factory=dict)   # [N, T+1] each
    push: PushPlan | None = None
    policy: str = ""

    @property
    def n_episodes(self) -> int:
        return self.qpos.shape[0]

    @property
    def n_steps(self) -> int:
        return self.qpos.shape[1] - 1

    def termination_step(self) -> np.ndarray:
        """[N] first step with done (T+1 if the episode reached the cap)."""
        d = self.done.astype(bool)
        first = np.where(d.any(1), d.argmax(1), self.n_steps + 1)
        return first

    def survival_s(self) -> np.ndarray:
        """[N] time from reset to termination (capped at T*dt)."""
        return np.minimum(self.termination_step(), self.n_steps) * self.dt


def initial_states(env: RunImitation, n: int, seed: int, start_times=None):
    """Reset N episodes. start_times=None: random phase (reference state initialisation, as in
    training). Otherwise a scalar or [N] array of clip times to start exactly on the reference."""
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    if start_times is None:
        return jax.jit(jax.vmap(env.reset))(keys)
    t0 = jnp.broadcast_to(jnp.asarray(start_times, jnp.float32), (n,))
    return jax.jit(jax.vmap(env.reset_to_time))(t0, keys)


def make_chunk_fn(env: RunImitation, act: Callable, chunk: int, body_id: int,
                  max_root_drift: float):
    """Jitted vmapped `(states, keys, step0, push_start[N], push_n, push_force[N,3]) -> states, recs`."""
    m = env.mj_model
    floor_id = m.geom("floor").id
    bad_geom = jnp.asarray(term_lib.bad_geom_table(m, tuple(env._config.termination.contact_bodies)))
    root_id = m.body("root").id
    nbody = m.nbody

    def one_step(carry, _):
        state, key, t, p_start, p_n, p_force = carry
        key, k_act = jax.random.split(key)
        in_push = (t >= p_start) & (t < p_start + p_n)
        f = jnp.where(in_push, p_force, jnp.zeros(3))
        xfrc = jnp.zeros((nbody, 6), jnp.float32).at[body_id, :3].set(f)
        s_in = state.replace(data=state.data.replace(xfrc_applied=xfrc))
        action = act(s_in, k_act)
        new = env.step(s_in, action)
        d = new.data
        ok = jnp.all(jnp.isfinite(d.qpos)) & jnp.all(jnp.isfinite(d.qvel))
        # freeze a blown-up episode at its last finite state (its 'nan' flag is recorded below)
        kept = jax.tree.map(lambda a, b: jnp.where(ok, a, b), new, state)
        fr = env.ref.at(kept.data.time)
        c = term_lib.contact_buffer(new.data)
        fell = term_lib.fell(c.geom, c.dist, floor_id, bad_geom)
        drift = jnp.linalg.norm(new.data.xpos[root_id] - fr.body_pos[root_id - 1]) > max_root_drift
        rec = _record(kept, fr, f)
        rec.update(done=jnp.maximum(new.done, (~ok).astype(jnp.float32)), fell=fell, drifted=drift,
                   nan=~ok, reward=new.reward)
        # keep the original xfrc out of the carried state: next step sets it again
        return (kept, key, t + 1, p_start, p_n, p_force), rec

    def run(state, key, step0, p_start, p_n, p_force):
        carry = (state, key, step0, p_start, p_n, p_force)
        carry, recs = jax.lax.scan(one_step, carry, None, chunk)
        return carry[0], carry[1], recs

    return jax.jit(jax.vmap(run, in_axes=(0, 0, None, 0, None, 0)))


def _record(state, fr, force):
    d = state.data
    rec = {"t": d.time, "qpos": d.qpos, "qpos_ref": fr.qpos, "body_pos": d.xpos[1:],
           "body_pos_ref": fr.body_pos, "phase": fr.phase, "force": force}
    for k in ("mpjpe", "root_drift", "r_pose", "r_vel", "r_ee", "r_com"):
        if k in state.metrics:
            rec["m_" + k] = state.metrics[k]
    return rec


def rollout(env: RunImitation, policy: EvalPolicy, n_episodes: int = 16, seconds: float = 30.0, *,
            seed: int = 0, start_times=None, push: PushPlan | None = None, chunk: int = 50,
            tail_s: float = 1.0, max_root_drift: float | None = None, verbose: bool = False,
            _cache: dict | None = None) -> Rollout:
    """Run `n_episodes` for up to `seconds` each and return the records.

    Stops early `tail_s` seconds after every episode has terminated (the tail keeps the fall in
    the videos). `max_root_drift` only affects our own `drifted` flag (default: the env's limit);
    the env's `done` always uses the env config.
    """
    n_total = int(round(seconds / env.dt))
    tail = int(round(tail_s / env.dt))
    body = push.body if push is not None else "root"
    body_id = env.mj_model.body(body).id
    mrd = float(env._config.termination.max_root_drift if max_root_drift is None else max_root_drift)
    key = ("chunk", id(env), id(policy.act), chunk, body_id, mrd)
    cache = _cache if _cache is not None else _CHUNK_CACHE
    if key not in cache:
        cache[key] = make_chunk_fn(env, policy.act, chunk, body_id, mrd)
    run_chunk = cache[key]

    state = initial_states(env, n_episodes, seed, start_times)
    keys = jax.random.split(jax.random.PRNGKey(seed + 7919), n_episodes)
    if push is not None:
        p_start = jnp.asarray(push.start_step, jnp.int32)
        p_force = jnp.asarray(push.force, jnp.float32)
        p_n = jnp.asarray(push.n_steps, jnp.int32)
    else:
        p_start = jnp.full((n_episodes,), -1, jnp.int32)
        p_force = jnp.zeros((n_episodes, 3), jnp.float32)
        p_n = jnp.asarray(0, jnp.int32)

    first = jax.jit(jax.vmap(lambda s: _record(s, env.ref.at(s.data.time), jnp.zeros(3))))(state)
    zeros = jnp.zeros((n_episodes,))
    first.update(done=state.done, fell=zeros.astype(bool), drifted=zeros.astype(bool),
                 nan=zeros.astype(bool), reward=state.reward)
    pieces = [jax.tree.map(lambda a: np.asarray(a)[:, None], first)]
    steps, dead_since = 0, None
    while steps < n_total:
        state, keys, recs = run_chunk(state, keys, jnp.asarray(steps, jnp.int32), p_start, p_n, p_force)
        pieces.append(jax.tree.map(np.asarray, recs))
        steps += chunk
        done_any = np.concatenate([p["done"] for p in pieces], axis=1).astype(bool).any(1)
        if verbose:
            print(f"[rollout] {policy.name}: {steps * env.dt:.1f} s, {int((~done_any).sum())}/{n_episodes} alive",
                  flush=True)
        if done_any.all():
            dead_since = steps if dead_since is None else dead_since
            if steps - dead_since >= tail:
                break
    recs = {k: np.concatenate([p[k] for p in pieces], axis=1) for k in pieces[0]}
    T = min(recs["qpos"].shape[1] - 1, n_total)
    recs = {k: v[:, :T + 1] for k, v in recs.items()}
    done = recs["done"].astype(np.float32)
    alive = np.cumprod(1.0 - done, axis=1) > 0.5
    root_index = env.mj_model.body("root").id - 1
    return Rollout(
        dt=float(env.dt), clip_duration=float(env.motion.duration),
        cycle_offset=np.asarray(env.motion.cycle_offset_pos, np.float64),
        body_names=list(env.motion.body_names), root_index=root_index,
        t=recs["t"], qpos=recs["qpos"], qpos_ref=recs["qpos_ref"], body_pos=recs["body_pos"],
        body_pos_ref=recs["body_pos_ref"], phase=recs["phase"], reward=recs["reward"], done=done,
        fell=recs["fell"].astype(bool), drifted=recs["drifted"].astype(bool),
        nan=recs["nan"].astype(bool), alive=alive, force=recs["force"],
        env_metrics={k[2:]: v for k, v in recs.items() if k.startswith("m_")},
        push=push, policy=policy.name)


_CHUNK_CACHE: dict = {}
