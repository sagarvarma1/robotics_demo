"""Early termination: stop the episode as soon as the character has clearly failed.

Why it matters so much. Without it, a policy that falls spends most of each episode lying on the
floor, and nearly all its training data is about lying on the floor. Cutting the episode at the
first fall means the data are about the motion we actually want. It also acts as a penalty: a
fallen character stops collecting the (always positive) tracking reward. DeepMimic's ablation
(Peng et al. 2018, section 9) found that without early termination, policies for running,
backflips and other dynamic skills failed to learn the motion at all and ended up on the ground.

Conditions (any one ends the episode):
  1. Contact between the floor and any body other than the allowed ones (default: feet and toes,
     `termination.contact_bodies`). A knee, hand, hip or torso touching the ground means a fall.
  2. Root drift: the root is more than `max_root_drift` metres (3-D distance) from where the
     reference root is. The character may be upright but has lost the motion (e.g. stopped
     running), and tracking reward from there on would be misleading.
  3. Root lower than `min_root_height` (disabled at 0).
  4. Any NaN in the state: the physics blew up. End the episode before NaNs reach the learner.

How contacts are read in MJX (jax impl). `data._impl.contact` is a fixed-size buffer: one slot
per candidate geom pair the collision filter allows (164 for our model, see docs/decisions.md D3).
Slots exist every step whether touching or not. `contact.geom[:, 0/1]` are the two geom ids, and
`contact.dist` is the signed distance (negative = penetrating). A slot counts as touching when
dist < 0 (our geoms have margin 0). We keep a per-geom table "touching the floor with this geom
is a fall" and look up both geoms of every slot each step (a gather from a 50-entry table, which
is cheap). That way we don't depend on how the slots happen to be ordered.
"""

from __future__ import annotations

import jax.numpy as jnp
import mujoco
import numpy as np

DEFAULT_ALLOWED = ("lfoot", "ltoes", "rfoot", "rtoes")


def contact_buffer(data):
    """The MJX contact struct (jax impl)."""
    impl = getattr(data, "_impl", None)
    return impl.contact if impl is not None and hasattr(impl, "contact") else data.contact


def bad_geom_table(model: mujoco.MjModel, allowed_bodies=DEFAULT_ALLOWED) -> np.ndarray:
    """Boolean per geom: True if this geom touching the floor means a fall."""
    allowed = [model.body(n).id for n in allowed_bodies]
    bad = ~np.isin(model.geom_bodyid, allowed)
    bad &= model.geom_bodyid != 0          # world geoms (the floor itself) never count
    return bad


def fell(contact_geom, contact_dist, floor_id: int, bad_geom):
    """Any touching (dist < 0) contact between the floor and a 'bad' geom."""
    g0, g1 = contact_geom[:, 0], contact_geom[:, 1]
    bad_pair = ((g0 == floor_id) & bad_geom[g1]) | ((g1 == floor_id) & bad_geom[g0])
    return jnp.any((contact_dist < 0.0) & bad_pair)


def done_flags(data_qpos, data_qvel, contact_geom, contact_dist, floor_id, bad_geom, root_drift,
               max_root_drift, min_root_height):
    nan = jnp.any(~jnp.isfinite(data_qpos)) | jnp.any(~jnp.isfinite(data_qvel))
    return {
        "contact": fell(contact_geom, contact_dist, floor_id, bad_geom),
        "drift": root_drift > max_root_drift,
        "low": data_qpos[2] < min_root_height,
        "nan": nan,
    }
