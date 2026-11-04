"""Build our simulation model from dm_control's CMU humanoid.

Input:  env/assets/dm_control/humanoid_CMU.xml  (vendored, Apache-2.0, unmodified)
Output: env/assets/humanoid_cmu_rlm.xml          (generated; do not hand-edit)

Run:    uv run python scripts/build_humanoid.py

What we change and why (also in docs/decisions.md):

1. Torque motors -> PD (position) actuators.
   DeepMimic's policy outputs *target joint angles*, and a PD controller turns
   them into torque: tau = kp * (q_target - q) - kd * qdot.
   We let MuJoCo do this with a `general` actuator:
       force = gain * ctrl + bias0 + bias1 * q
   where ctrl in [-1, 1] maps linearly onto the joint's range, so
       q_target = mid + half * ctrl  ->  gain = kp*half, bias0 = kp*mid, bias1 = -kp.
   The "-kd * qdot" part is the joint's `damping`, which MuJoCo's implicitfast
   integrator handles implicitly (stable even with stiff gains at a 5 ms step).
   kp, force limits and damping start from dm_control's position-controlled
   CMU walker (_POSITION_ACTUATORS_V2020). We don't use its 30 ms actuator
   filter: it adds hidden actuator state the policy can't observe.

2. Joint springs (`stiffness`) set to 0. The original springs pull every joint
   toward 0 rad. With PD control they would only bias the policy's targets.

3. Collision filtering. Contact happens when
   (contype_a & conaffinity_b) or (contype_b & conaffinity_a) is non-zero.
     floor: contype=1, conaffinity=2
     body : contype=2, conaffinity=1   -> touches the floor, not other body parts
     legs : contype=6, conaffinity=5   -> also touch the *other* leg (bit 4)
   MJX's JAX backend checks every candidate geom pair every step at a fixed
   cost, so full self-collision on 50+ geoms would dominate step time. Legs
   still collide with each other so the policy can't "cheat" by passing its
   feet through each other.

4. Physics options: timestep 5 ms, implicitfast integrator, Newton solver with
   few iterations (same settings as Playground's humanoid tasks).

5. Sensors removed. We compute everything from qpos/qvel/xpos directly, and
   some sensor types (touch) are not supported by MJX.
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "env/assets/dm_control/humanoid_CMU.xml"
DST = ROOT / "env/assets/humanoid_cmu_rlm.xml"

# name -> (force limit, kp, joint damping), from dm_control
# locomotion/walkers/cmu_humanoid.py::_POSITION_ACTUATORS_V2020
PD_PARAMS = {
    "headrx": (40, 40, 2), "headry": (40, 40, 2), "headrz": (40, 40, 2),
    "lclaviclery": (80, 80, 20), "lclaviclerz": (80, 80, 20),
    "lfemurrx": (300, 300, 15), "lfemurry": (200, 200, 10), "lfemurrz": (200, 200, 10),
    "lfingersrx": (20, 20, 1), "lfootrx": (120, 120, 6), "lfootrz": (50, 50, 3),
    "lhandrx": (20, 20, 1), "lhandrz": (20, 20, 1),
    "lhumerusrx": (120, 120, 6), "lhumerusry": (120, 120, 6), "lhumerusrz": (120, 120, 6),
    "lowerbackrx": (300, 300, 15), "lowerbackry": (180, 180, 20), "lowerbackrz": (200, 200, 20),
    "lowerneckrx": (120, 120, 20), "lowerneckry": (120, 120, 20), "lowerneckrz": (120, 120, 20),
    "lradiusrx": (90, 90, 5), "lthumbrx": (20, 20, 1), "lthumbrz": (20, 20, 1),
    "ltibiarx": (160, 160, 8), "ltoesrx": (20, 20, 1), "lwristry": (20, 20, 1),
    "rclaviclery": (80, 80, 20), "rclaviclerz": (80, 80, 20),
    "rfemurrx": (300, 300, 15), "rfemurry": (200, 200, 10), "rfemurrz": (200, 200, 10),
    "rfingersrx": (20, 20, 1), "rfootrx": (120, 120, 6), "rfootrz": (50, 50, 3),
    "rhandrx": (20, 20, 1), "rhandrz": (20, 20, 1),
    "rhumerusrx": (120, 120, 6), "rhumerusry": (120, 120, 6), "rhumerusrz": (120, 120, 6),
    "rradiusrx": (90, 90, 5), "rthumbrx": (20, 20, 1), "rthumbrz": (20, 20, 1),
    "rtibiarx": (160, 160, 8), "rtoesrx": (20, 20, 1), "rwristry": (20, 20, 1),
    "thoraxrx": (300, 300, 15), "thoraxry": (80, 80, 8), "thoraxrz": (200, 200, 12),
    "upperbackrx": (300, 300, 15), "upperbackry": (80, 80, 8), "upperbackrz": (200, 200, 12),
    "upperneckrx": (60, 60, 10), "upperneckry": (60, 60, 10), "upperneckrz": (60, 60, 10),
}

LEG_BODIES = {"ltibia", "lfoot", "ltoes", "rtibia", "rfoot", "rtoes"}

# Contact filter bits (see docstring).
FLOOR = dict(contype="1", conaffinity="2")
BODY = dict(contype="2", conaffinity="1")
LEG = dict(contype="6", conaffinity="5")


def _parse_range_deg(s: str) -> tuple[float, float]:
    lo, hi = (float(x) for x in s.split())
    return lo, hi


def build() -> ET.ElementTree:
    tree = ET.parse(SRC)
    root = tree.getroot()

    # --- includes: point them at the vendored copies --------------------------
    for inc in root.findall("include"):
        inc.set("file", "dm_control/" + inc.get("file").lstrip("./"))

    # --- physics options -----------------------------------------------------
    option = ET.Element("option", timestep="0.005", integrator="implicitfast",
                        solver="Newton", iterations="4", ls_iterations="8")
    root.insert(0, option)
    # Note: joint ranges in the XML are in degrees (MJCF default), but qpos and
    # actuator gain/bias parameters are always in radians, so we convert below.

    # --- defaults: drop joint springs, set body collision bits ----------------
    for d in root.iter("default"):
        j = d.find("joint")
        if j is not None and "stiffness" in j.attrib:
            j.set("stiffness", "0")
        if d.get("class") == "humanoid":
            g = d.find("geom")
            g.attrib.update(BODY)

    # --- geoms: floor + leg collision bits -----------------------------------
    world = root.find("worldbody")
    for g in world.findall("geom"):
        if g.get("name") == "floor":
            g.attrib.update(FLOOR)
    for body in world.iter("body"):
        if body.get("name") in LEG_BODIES:
            for g in body.findall("geom"):
                g.attrib.update(LEG)

    # --- joints: damping from PD table ---------------------------------------
    joint_range = {}
    for j in world.iter("joint"):
        name = j.get("name")
        if name in PD_PARAMS:
            j.set("damping", str(PD_PARAMS[name][2]))
            joint_range[name] = j.get("range")

    # --- actuators: motors -> general PD actuators ----------------------------
    act = root.find("actuator")
    motors = list(act)
    for m in motors:
        act.remove(m)
    import math
    for m in motors:
        name = m.get("joint")
        flim, kp, _ = PD_PARAMS[name]
        lo, hi = _parse_range_deg(joint_range[name])
        lo, hi = math.radians(lo), math.radians(hi)
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
        act.append(ET.Element(
            "general", name=name, joint=name,
            ctrllimited="true", ctrlrange="-1 1",
            forcelimited="true", forcerange=f"{-flim} {flim}",
            gaintype="fixed", gainprm=f"{kp * half:.6g}",
            biastype="affine", biasprm=f"{kp * mid:.6g} {-kp:.6g} 0",
        ))

    # --- sensors: remove ------------------------------------------------------
    for s in root.findall("sensor"):
        root.remove(s)

    # Also remove the motor default (no motors left).
    for d in root.iter("default"):
        for m in d.findall("motor"):
            d.remove(m)
    return tree


def main() -> None:
    tree = build()
    ET.indent(tree, space="  ")
    header = ("<!-- GENERATED by scripts/build_humanoid.py from dm_control's humanoid_CMU.xml "
              "(Apache-2.0, see env/assets/dm_control/LICENSE). Do not edit by hand. -->\n")
    DST.write_text(header + ET.tostring(tree.getroot(), encoding="unicode") + "\n")
    print(f"wrote {DST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
