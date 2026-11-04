"""Read CMU motion capture files (ASF skeleton + AMC motion) and run forward kinematics.

The CMU mocap database stores a recording as two plain-text files:

* ``.asf`` ("Acclaim Skeleton File") describes the *person*: one entry per bone with
  its rest-pose direction, its length, the orientation of its local rotation axes
  (``axis``), which rotations it may perform (``dof``, e.g. ``rx ry rz``) and the
  parent/child hierarchy.
* ``.amc`` ("Acclaim Motion Capture") describes the *movement*: for every frame, the
  root translation + rotation and the joint angles of every bone, in degrees.

Forward kinematics (FK) turns "angles" into "where is every joint in space". It works
by walking down the tree from the root: a bone's world orientation is its parent's
orientation times its own local rotation, and the bone's end point is its start point
plus ``length * (world-rotated direction)``. Doing this for every bone gives us the 3-D
keypoints the retargeter (``motion/retarget.py``) tries to match.

The one subtle bit is the ASF ``axis``. Each bone's joint angles are measured in a
local frame ``C`` that is rotated relative to the world frame. So the local rotation of
a bone, *expressed in world axes*, is ``C @ M @ C^-1`` where ``M`` is built from the AMC
angles. Conjugating by ``C`` simply means "rotate about C's axes instead of the world's".

Units and axes:

* ASF lengths are in "units" where ``:units length 0.45`` means 1 unit = 1/0.45 inch.
  So metres = units * 0.0254 / 0.45 = units * 0.056444 (the same constant dm_control uses).
* CMU is **y-up** (subject faces +z, left is +x). Our simulator is **z-up**. Rotating the
  whole scene by +90 degrees about x maps y -> z and z -> -y, which is exactly what
  dm_control's ``parse_amc.py`` does (``euler2quat(90, 0, 0)`` on the root).
  We apply that same rotation (``Y_UP_TO_Z_UP``) to every output position and rotation.

Everything here is pure numpy. Downloaded files are treated as data: they are parsed as
text, never executed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# 1 ASF unit (with ':units length 0.45') in metres; equals dm_control's CONVERSION_LENGTH.
INCH = 0.0254
MOCAP_FPS = 120.0

# +90 deg about x: y-up (CMU) -> z-up (MuJoCo world). Applied on the left.
Y_UP_TO_Z_UP = np.array([[1.0, 0.0, 0.0],
                         [0.0, 0.0, -1.0],
                         [0.0, 1.0, 0.0]])


def rot_axis(axis: str, deg: np.ndarray) -> np.ndarray:
    """Rotation matrices about a principal axis. ``deg`` may be an array -> [..., 3, 3]."""
    a = np.deg2rad(np.asarray(deg, dtype=float))
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    if axis == "x":
        m = [[o, z, z], [z, c, -s], [z, s, c]]
    elif axis == "y":
        m = [[c, z, s], [z, o, z], [-s, z, c]]
    elif axis == "z":
        m = [[c, -s, z], [s, c, z], [z, z, o]]
    else:
        raise ValueError(axis)
    return np.moveaxis(np.array(m), (0, 1), (-2, -1))


def euler_to_mat(angles_deg: np.ndarray, order: str = "XYZ") -> np.ndarray:
    """ASF/AMC Euler convention: order 'XYZ' means rotate about x first, then y, then z.

    With column vectors that is ``R = Rz @ Ry @ Rx`` (the first rotation sits rightmost).
    ``angles_deg[..., i]`` is the angle about axis ``order[i]``.
    """
    angles_deg = np.asarray(angles_deg, dtype=float)
    R = np.broadcast_to(np.eye(3), angles_deg.shape[:-1] + (3, 3)).copy()
    for i, ax in enumerate(order.lower()):
        R = rot_axis(ax, angles_deg[..., i]) @ R
    return R


@dataclass
class Bone:
    name: str
    direction: np.ndarray        # unit vector, ASF (y-up) world frame, rest pose
    length: float                # metres
    axis_deg: np.ndarray         # local frame orientation C, Euler degrees
    axis_order: str              # e.g. "XYZ"
    dof: list[str]               # e.g. ["rx", "ry", "rz"], the order values appear in the AMC
    limits: list[tuple[float, float]]  # degrees, one per dof
    parent: str | None = None
    children: list[str] = field(default_factory=list)

    @property
    def C(self) -> np.ndarray:
        return euler_to_mat(self.axis_deg, self.axis_order)


@dataclass
class Skeleton:
    bones: dict[str, Bone]       # includes a synthetic "root" bone of zero length
    root_order: list[str]        # e.g. ["TX","TY","TZ","RX","RY","RZ"]
    root_axis: str               # e.g. "XYZ"
    length_scale: float          # metres per ASF length unit
    order: list[str]             # bones in parent-before-child order

    def dof_names(self) -> list[str]:
        """Flat list like 'lfemurrx' (matches the hinge names of our MuJoCo model)."""
        return [b + d for b in self.order if b != "root" for d in self.bones[b].dof]


def _tokens(path: Path | str) -> list[str]:
    return Path(path).read_text(errors="replace").replace("\r", "").split("\n")


def parse_asf(path: Path | str) -> Skeleton:
    """Parse an ASF file into a :class:`Skeleton` (lengths converted to metres)."""
    lines = _tokens(path)
    length_unit = 1.0
    root_order, root_axis = ["TX", "TY", "TZ", "RX", "RY", "RZ"], "XYZ"
    bones: dict[str, Bone] = {}
    section = None
    cur: dict | None = None
    hierarchy: list[list[str]] = []
    reading_limits = False
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(":"):
            section = line.split()[0][1:]
            continue
        tok = line.split()
        if section == "units":
            if tok[0] == "length":
                length_unit = float(tok[1])
            elif tok[0] == "angle" and tok[1] != "deg":
                raise ValueError("only degree ASF files are supported")
        elif section == "root":
            if tok[0] == "order":
                root_order = [t.upper() for t in tok[1:]]
            elif tok[0] == "axis":
                root_axis = tok[1].upper()
        elif section == "bonedata":
            if tok[0] == "begin":
                cur = {"limits": [], "dof": [], "axis": np.zeros(3), "order": "XYZ"}
                reading_limits = False
            elif tok[0] == "end":
                assert cur is not None
                bones[cur["name"]] = Bone(
                    name=cur["name"], direction=cur["direction"] / np.linalg.norm(cur["direction"]),
                    length=cur["length"], axis_deg=cur["axis"], axis_order=cur["order"],
                    dof=cur["dof"], limits=cur["limits"])
                cur = None
            elif cur is not None:
                if tok[0] == "name":
                    cur["name"] = tok[1]
                elif tok[0] == "direction":
                    cur["direction"] = np.array([float(t) for t in tok[1:4]])
                elif tok[0] == "length":
                    cur["length"] = float(tok[1])
                elif tok[0] == "axis":
                    cur["axis"] = np.array([float(t) for t in tok[1:4]])
                    cur["order"] = tok[4].upper() if len(tok) > 4 else "XYZ"
                elif tok[0] == "dof":
                    cur["dof"] = [t.lower() for t in tok[1:]]
                elif tok[0] == "limits" or reading_limits:
                    reading_limits = True
                    s = line.replace("limits", "").replace("(", " ").replace(")", " ").split()
                    cur["limits"].append((float(s[0]), float(s[1])))
        elif section == "hierarchy":
            if tok[0] in ("begin", "end"):
                continue
            hierarchy.append(tok)

    scale = INCH / length_unit
    for b in bones.values():
        b.length *= scale
    bones["root"] = Bone("root", np.zeros(3), 0.0, np.zeros(3), root_axis, [], [])
    for row in hierarchy:
        parent, kids = row[0], row[1:]
        for k in kids:
            bones[k].parent = parent
            bones[parent].children.append(k)
    order, stack = [], ["root"]
    while stack:
        b = stack.pop(0)
        order.append(b)
        stack.extend(bones[b].children)
    return Skeleton(bones, root_order, root_axis, scale, order)


def parse_amc(path: Path | str, skel: Skeleton) -> dict[str, np.ndarray]:
    """Parse an AMC file. Returns {bone name: [T, n_values]} (root: 6 values, raw units/deg)."""
    frames: list[dict[str, list[float]]] = []
    cur: dict[str, list[float]] | None = None
    for raw in _tokens(path):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(":"):
            continue
        tok = line.split()
        if len(tok) == 1 and tok[0].isdigit():
            cur = {}
            frames.append(cur)
            continue
        if cur is not None:
            cur[tok[0]] = [float(t) for t in tok[1:]]
    if not frames:
        raise ValueError(f"no frames in {path}")
    out = {}
    for name in frames[0]:
        out[name] = np.array([f[name] for f in frames], dtype=float)
    for name in skel.order:
        if name != "root" and skel.bones[name].dof and name not in out:
            out[name] = np.zeros((len(frames), len(skel.bones[name].dof)))
    return out


@dataclass
class FKResult:
    """Per-frame FK output, already converted to metres and z-up.

    ``start[t, i]`` is where bone ``names[i]`` begins (= its joint with the parent; for
    the root, the root position). ``end`` is where it ends. ``seg_rot[t, i]`` is the world
    orientation of the bone's local axis frame (``matrix @ C``), which is the frame our
    MuJoCo model's body of the same name uses.
    """
    names: list[str]
    start: np.ndarray    # [T, N, 3]
    end: np.ndarray      # [T, N, 3]
    seg_rot: np.ndarray  # [T, N, 3, 3]


def forward_kinematics(skel: Skeleton, amc: dict[str, np.ndarray]) -> FKResult:
    """Global positions/orientations of every bone for every frame (vectorised over time)."""
    T = next(iter(amc.values())).shape[0]
    names = skel.order
    idx = {n: i for i, n in enumerate(names)}
    N = len(names)
    start = np.zeros((T, N, 3))
    end = np.zeros((T, N, 3))
    seg = np.zeros((T, N, 3, 3))
    glob = np.zeros((T, N, 3, 3))  # 'matrix' in ASF terms (world-aligned at rest)

    # Root: translation channels and rotation channels in the order the ASF declares.
    rv = amc["root"]
    trans = np.zeros((T, 3))
    rot_deg = np.zeros((T, 3))
    for k, ch in enumerate(skel.root_order):
        axis = "XYZ".index(ch[1])
        if ch[0] == "T":
            trans[:, axis] = rv[:, k] * skel.length_scale
        else:
            rot_deg[:, axis] = rv[:, k]
    rb = skel.bones["root"]
    C, Ci = rb.C, rb.C.T
    M = euler_to_mat(rot_deg, skel.root_axis)
    glob[:, 0] = C @ M @ Ci
    seg[:, 0] = glob[:, 0] @ C
    start[:, 0] = trans
    end[:, 0] = trans

    for name in names[1:]:
        b = skel.bones[name]
        i, p = idx[name], idx[b.parent]
        ang = np.zeros((T, 3))
        vals = amc.get(name)
        for k, d in enumerate(b.dof):
            ang[:, "xyz".index(d[1])] = vals[:, k]
        C, Ci = b.C, b.C.T
        M = euler_to_mat(ang, "XYZ")   # dof rotations are applied x, then y, then z
        glob[:, i] = glob[:, p] @ C @ M @ Ci
        seg[:, i] = glob[:, i] @ C
        start[:, i] = end[:, p]
        end[:, i] = start[:, i] + b.length * np.einsum("tij,j->ti", glob[:, i], b.direction)

    Rz = Y_UP_TO_Z_UP
    return FKResult(names=list(names), start=start @ Rz.T, end=end @ Rz.T,
                    seg_rot=np.einsum("ij,tnjk->tnik", Rz, seg))


def load_cmu(asf: Path | str, amc: Path | str) -> tuple[Skeleton, dict[str, np.ndarray], FKResult]:
    skel = parse_asf(asf)
    vals = parse_amc(amc, skel)
    return skel, vals, forward_kinematics(skel, vals)


# Keypoints we give the retargeter. Each is the *start* of the ASF bone of the same name,
# which is exactly the origin of the MuJoCo body of that name in our model (the model was
# built from a CMU ASF), so the joint-name map is the identity.
CMU_KEYPOINTS = (
    "root", "lfemur", "ltibia", "lfoot", "ltoes", "rfemur", "rtibia", "rfoot", "rtoes",
    "upperback", "thorax", "lowerneck", "upperneck", "head",
    "lhumerus", "lradius", "lwrist", "rhumerus", "rradius", "rwrist",
)
# Orientations we trust: pelvis, chest, head and feet are rigid, well-marked segments.
CMU_TRUSTED_ROT = ("root", "thorax", "head", "lfoot", "rfoot")


def to_source_motion(skel: Skeleton, fk: FKResult, source: str = "", fps: float = MOCAP_FPS):
    """Convert FK output to a generic :class:`motion.retarget.SourceMotion`."""
    from motion.retarget import SourceMotion  # local import: retarget imports mink
    idx = {n: i for i, n in enumerate(fk.names)}
    names = list(CMU_KEYPOINTS)
    parents = []
    for n in names:
        p = skel.bones[n].parent
        while p is not None and p not in names:
            p = skel.bones[p].parent
        parents.append(-1 if p is None else names.index(p))
    sel = [idx[n] for n in names]
    return SourceMotion(fps=fps, names=names, parents=parents, pos=fk.start[:, sel].copy(),
                        body_map={n: n for n in names}, rot=fk.seg_rot[:, sel].copy(),
                        rot_valid=CMU_TRUSTED_ROT, source=source)


def direct_angle_copy(skel: Skeleton, amc: dict[str, np.ndarray], hinge_names: list[str]) -> np.ndarray:
    """dm_control-style conversion (parse_amc.Amcvals2qpos): copy each AMC angle onto the
    hinge with the same name, root pos * 0.056444 with y-up -> z-up, root quat = Rx(90) * q.
    Returns qpos [T, 7 + len(hinge_names)]. Used only as a cross-check."""
    from scipy.spatial.transform import Rotation
    T = amc["root"].shape[0]
    q = np.zeros((T, 7 + len(hinge_names)))
    rv = amc["root"]
    q[:, :3] = (rv[:, :3] * skel.length_scale) @ Y_UP_TO_Z_UP.T
    Rr = Y_UP_TO_Z_UP @ euler_to_mat(rv[:, 3:6], skel.root_axis)
    q[:, 3:7] = Rotation.from_matrix(Rr).as_quat(scalar_first=True)
    for j, n in enumerate(hinge_names):
        bone, dof = n[:-2], n[-2:]
        if bone in amc and dof in skel.bones[bone].dof:
            q[:, 7 + j] = np.deg2rad(amc[bone][:, skel.bones[bone].dof.index(dof)])
    return q
