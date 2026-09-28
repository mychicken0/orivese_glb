"""Rig + animation export for glb2ori.

glTF skins -> Oriverse `bone(...)` + `skin_weights(...)`, glTF animations -> `#clip` blocks.

Oriverse conventions (measured with inspect_rig): bone rest frames are identity (axis aligned),
child world rotation = parent world * local, key Euler (rx, ry, rz) degrees compose as
Rz * Ry * Rx, `translate:` offsets a joint from rest in its parent's posed frame, and the root
`offset:` is a world translation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

MAX_BONES = 48
MAX_KEYS = 64

CANON = ["Root", "Spine", "Torso", "Neck", "Head"] + [f"{b}{s}" for b in
         ("Scapula", "UpperArm", "Forearm", "Hand", "Thigh", "Shin", "Foot", "Toe") for s in "RL"]

# body-token regex -> canonical base; limb bases get an R/L suffix. Order = priority.
HUMANOID_PATTERNS = [
    (r"^(hips?|pelvis|root)$", "Root", False),
    (r"^(chest|upperchest|torso|ribcage)$", "Torso", False),
    (r"^(spine|belly|abdomen|waist|stomach)$", "Spine", False),
    (r"^neck$", "Neck", False),
    (r"^head$", "Head", False),
    (r"^(shoulder|clavicle|collar|collarbone|scapula)$", "Scapula", True),
    (r"^(upperarm|uparm|arm|humerus)$", "UpperArm", True),
    (r"^(forearm|lowerarm|lowarm|elbow)$", "Forearm", True),
    (r"^(hand|wrist)$", "Hand", True),
    (r"^(thigh|upleg|upperleg|uleg|femur)$", "Thigh", True),
    (r"^(shin|calf|lowleg|lowerleg|leg|tibia)$", "Shin", True),
    (r"^(foot|ankle)$", "Foot", True),
    (r"^(toe|toes|toebase|ball)$", "Toe", True),
    (r"^knee$", "Shin", True),  # only used when no shin-like bone exists on that side
]


@dataclass
class Bone:
    name: str
    node: int
    parent: int      # index into Rig.bones, -1 for the root
    pos: np.ndarray  # rest joint position, Oriverse cm


@dataclass
class Rig:
    bones: list[Bone]
    node_to_bone: dict[int, int]
    bind_world: dict[int, np.ndarray]   # node -> glTF bind-pose world matrix of the joint
    deform_base: dict[int, np.ndarray]  # node -> IBM @ inv(mesh matrix): posed world = J(t) @ deform_base
    frame: np.ndarray
    scale: float
    offset: np.ndarray

    def to_display(self, p: np.ndarray) -> np.ndarray:
        return self.scale * (p @ self.frame.T) - self.offset


@dataclass
class Clip:
    name: str
    duration: float
    times: np.ndarray
    euler: np.ndarray        # (T, B, 3) local rotation degrees (rx, ry, rz)
    trans: np.ndarray        # (T, B, 3) joint offsets from rest in the parent frame, cm
    root_offset: np.ndarray  # (T, 3) cm


# --------------------------------------------------------------------------- glTF node helpers

def _parents(js: dict) -> np.ndarray:
    par = np.full(len(js.get("nodes", [])), -1)
    for i, n in enumerate(js.get("nodes", [])):
        for c in n.get("children", []):
            par[c] = i
    return par


def _topo(par: np.ndarray) -> list[int]:
    kids: dict[int, list[int]] = {}
    for i, p in enumerate(par):
        kids.setdefault(int(p), []).append(i)
    order, stack = [], list(reversed(kids.get(-1, [])))
    while stack:
        n = stack.pop()
        order.append(n)
        stack.extend(reversed(kids.get(n, [])))
    return order


def _quat_mat(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _trs(t, q, s) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = _quat_mat(q) * np.asarray(s)
    m[:3, 3] = t
    return m


def _node_trs(node: dict):
    if "matrix" in node:
        m = np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
        s = np.linalg.norm(m[:3, :3], axis=0)
        r = m[:3, :3] / np.where(s == 0, 1, s)
        return m[:3, 3].copy(), _mat_quat(r), s
    return (np.array(node.get("translation", [0, 0, 0]), float), np.array(node.get("rotation", [0, 0, 0, 1]), float),
            np.array(node.get("scale", [1, 1, 1]), float))


def _mat_quat(r: np.ndarray) -> np.ndarray:
    t = np.trace(r)
    if t > 0:
        s = 2 * np.sqrt(t + 1)
        q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, s / 4]
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(max(1 + r[i, i] - r[j, j] - r[k, k], 1e-12))
        q = [0.0, 0.0, 0.0, (r[k, j] - r[j, k]) / s]
        q[i] = s / 4
        q[j] = (r[j, i] + r[i, j]) / s
        q[k] = (r[k, i] + r[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


def _world(js: dict, order: list[int], par: np.ndarray, local: list[np.ndarray]) -> list[np.ndarray]:
    W = [np.eye(4)] * len(local)
    W = list(W)
    for n in order:
        W[n] = local[n] if par[n] < 0 else W[par[n]] @ local[n]
    return W


def _rotation(m3: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(m3)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def euler_zyx(r: np.ndarray) -> np.ndarray:
    """Degrees (rx, ry, rz) with r = Rz(rz) @ Ry(ry) @ Rx(rx)."""
    ry = np.arcsin(np.clip(-r[2, 0], -1, 1))
    if abs(np.cos(ry)) > 1e-6:
        rx = np.arctan2(r[2, 1], r[2, 2])
        rz = np.arctan2(r[1, 0], r[0, 0])
    else:  # gimbal lock
        rx = np.arctan2(-r[1, 2], r[1, 1])
        rz = 0.0
    return np.degrees([rx, ry, rz])


# --------------------------------------------------------------------------- rig

def _tokens(raw: str) -> list[str]:
    name = re.split(r"[:|]", raw)[-1]
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return re.findall(r"[a-z]+|\d+", name.lower())


def _side(tokens: list[str]) -> str | None:
    if "l" in tokens or "left" in tokens:
        return "L"
    if "r" in tokens or "right" in tokens:
        return "R"
    return None


def _ident(raw: str, used: set[str]) -> str:
    toks = _tokens(raw) or ["bone"]
    base = "".join(t.capitalize() for t in toks)
    if not base[0].isalpha():
        base = "B" + base
    name, k = base, 2
    while name in used:
        name, k = f"{base}{k}", k + 1
    used.add(name)
    return name


def humanoid_names(nodes: list[int], raw: dict[int, str], depth: dict[int, int], pos: dict[int, np.ndarray]) -> dict[int, str]:
    """Map glTF joint nodes to Oriverse canonical humanoid bone names where recognisable."""
    info = {}
    for n in nodes:
        toks = _tokens(raw[n])
        side = _side(toks)
        body = "".join(t for t in toks if t not in ("l", "r", "left", "right") and not t.isdigit())
        body = body.replace("mixamorig", "").replace("bip", "").replace("def", "") or body
        info[n] = (body, side or ("L" if pos[n][1] > 0 else "R"), side is not None)
    out: dict[int, str] = {}
    taken: set[str] = set()
    spine_like = []
    for pat, base, limb in HUMANOID_PATTERNS:
        cands = sorted((n for n in nodes if n not in out and re.match(pat, info[n][0])), key=lambda n: depth[n])
        if base == "Spine":
            spine_like = cands
            cands = cands[:1]
        for n in cands:
            name = base + info[n][1] if limb else base
            if name not in taken:
                out[n] = name
                taken.add(name)
    if "Torso" not in taken and len(spine_like) > 1:  # Spine, Spine1, Spine2 chains
        n = spine_like[1]
        if n not in out:
            out[n] = "Torso"
    return out


def build_rig(src, scale: float, offset: np.ndarray, names: str = "humanoid", log=None) -> Rig | None:
    g, js = src.gltf, src.gltf.json
    skins = js.get("skins", [])
    if not skins:
        return None
    par = _parents(js)
    order = _topo(par)
    joints: list[int] = []
    ibm: dict[int, np.ndarray] = {}
    mesh_m: dict[int, np.ndarray] = {}
    for si, sk in enumerate(skins):
        mats = (g.accessor(sk["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
                if "inverseBindMatrices" in sk else np.tile(np.eye(4), (len(sk["joints"]), 1, 1)))
        mm = (src.skin_mesh_matrix or {}).get(si, np.eye(4))
        for j, m in zip(sk["joints"], mats):
            if j not in ibm:
                joints.append(j)
                ibm[j] = m
                mesh_m[j] = mm
    jset = set(joints)

    def joint_parent(n: int) -> int:
        p = par[n]
        while p >= 0 and p not in jset:
            p = par[p]
        return int(p)

    parent = {j: joint_parent(j) for j in joints}
    # influence per joint, used to prune down to Oriverse's 48 bones
    infl = {j: 0.0 for j in joints}
    for jj, ww in zip(src.joints.ravel(), src.weights.ravel()):
        if jj in infl:
            infl[jj] += ww
    remap = {j: j for j in joints}
    alive = set(joints)
    roots = [j for j in joints if parent[j] < 0]
    desc = {j: 0 for j in joints}
    for j in joints:
        p = parent[j]
        while p >= 0:
            desc[p] += 1
            p = parent[p]
    main_root = max(roots, key=lambda j: desc[j])
    for r in roots:
        if r != main_root:
            parent[r] = main_root
    while len(alive) > MAX_BONES:
        leaves = [j for j in alive if j != main_root and not any(parent[k] == j for k in alive)]
        victim = min(leaves, key=lambda j: infl[j])
        alive.discard(victim)
        for k, v in remap.items():
            if v == victim:
                remap[k] = parent[victim]
        infl[parent[victim]] += infl[victim]
    if len(joints) > MAX_BONES:
        _log(log, f"  rig: pruned {len(joints) - MAX_BONES} low-influence leaf bones (Oriverse max {MAX_BONES})")

    bind_world = {j: mesh_m[j] @ np.linalg.inv(ibm[j]) for j in joints}
    deform_base = {j: ibm[j] @ np.linalg.inv(mesh_m[j]) for j in joints}
    rig = Rig([], {}, bind_world, deform_base, src.frame, scale, offset)
    pos = {j: rig.to_display(bind_world[j][:3, 3][None])[0] for j in joints}
    depth = {}
    for j in joints:
        d, p = 0, parent[j]
        while p >= 0:
            d, p = d + 1, parent[p]
        depth[j] = d
    kept = [j for j in order if j in alive]
    raw = {j: js["nodes"][j].get("name", f"joint{j}") for j in kept}
    canon = humanoid_names(kept, raw, depth, pos) if names == "humanoid" else {}
    used = set(CANON) | set(canon.values())
    final = {j: canon.get(j) or _ident(raw[j], used) for j in kept}
    # the first bone must be the root; declare parents before children (topological order)
    kept.sort(key=lambda j: (j != main_root, depth[j]))
    idx = {j: i for i, j in enumerate(kept)}
    for j in kept:
        rig.bones.append(Bone(final[j], j, idx[parent[j]] if parent[j] >= 0 and j != main_root else -1, pos[j]))
    rig.node_to_bone = {j: idx[remap[j]] for j in joints}
    if canon:
        _log(log, f"  rig: {len(rig.bones)} bones, humanoid names: {', '.join(sorted(set(canon.values())))}")
    return rig


def vertex_skin(rig: Rig, src, points: np.ndarray, step: float = 0.05):
    """Per final point (bone ids (V,4), weights (V,4)) from the nearest source vertex."""
    from scipy.spatial import cKDTree

    src_disp = src.points * rig.scale - rig.offset  # source points are already in Oriverse axes
    _, nn = cKDTree(src_disp).query(points)
    j, w = src.joints[nn], src.weights[nn]
    bone = np.vectorize(lambda n: rig.node_to_bone.get(int(n), 0))(j)
    bone[j < 0] = 0
    out_b = np.zeros((len(points), 4), np.int64)
    out_w = np.zeros((len(points), 4))
    for i in range(len(points)):
        acc: dict[int, float] = {}
        for b, x in zip(bone[i], w[i]):
            if x > 0:
                acc[int(b)] = acc.get(int(b), 0.0) + float(x)
        if not acc:
            acc = {0: 1.0}
        top = sorted(acc.items(), key=lambda kv: -kv[1])[:4]
        tot = sum(x for _, x in top)
        q = [(b, round(x / tot / step) * step) for b, x in top]
        q = [(b, x) for b, x in q if x > 0] or [(top[0][0], 1.0)]
        tot = sum(x for _, x in q)
        for k, (b, x) in enumerate(q):
            out_b[i, k], out_w[i, k] = b, x / tot
    return out_b, out_w


# --------------------------------------------------------------------------- animation

def _sample(times: np.ndarray, values: np.ndarray, interp: str, t: float, is_rot: bool) -> np.ndarray:
    if interp == "CUBICSPLINE":
        values = values[1::3]
    if t <= times[0] or len(times) == 1:
        return values[0]
    if t >= times[-1]:
        return values[-1]
    i = int(np.searchsorted(times, t, side="right") - 1)
    if interp == "STEP":
        return values[i]
    a = (t - times[i]) / max(times[i + 1] - times[i], 1e-12)
    v0, v1 = values[i], values[i + 1]
    if not is_rot:
        return v0 * (1 - a) + v1 * a
    if np.dot(v0, v1) < 0:
        v1 = -v1
    d = np.clip(np.dot(v0, v1), -1, 1)
    if d > 0.9995:
        q = v0 * (1 - a) + v1 * a
    else:
        th = np.arccos(d)
        q = (np.sin((1 - a) * th) * v0 + np.sin(a * th) * v1) / np.sin(th)
    return q / np.linalg.norm(q)


def _clip_name(raw: str, used: set[str]) -> str:
    low = raw.lower()
    for verb in ("idle", "walk", "run", "jump"):
        if re.search(rf"(^|[^a-z]){verb}([^a-z]|$)", low) and verb not in used:
            used.add(verb)
            return verb
    return _ident(raw, used)


def sample_clips(rig: Rig, src, fps: float = 30.0, log=None) -> list[Clip]:
    g, js = src.gltf, src.gltf.json
    par = _parents(js)
    order = _topo(par)
    base = [_node_trs(n) for n in js["nodes"]]
    clips, used = [], set()
    B = len(rig.bones)
    rest = np.array([b.pos for b in rig.bones])
    for ai, anim in enumerate(js.get("animations", [])):
        chans = []
        for c in anim["channels"]:
            tgt = c["target"]
            if "node" not in tgt or tgt["path"] not in ("translation", "rotation", "scale"):
                continue
            smp = anim["samplers"][c["sampler"]]
            chans.append((tgt["node"], tgt["path"], g.accessor(smp["input"]).ravel(),
                          g.accessor(smp["output"]), smp.get("interpolation", "LINEAR")))
        if not chans:
            continue
        duration = float(max(c[2].max() for c in chans))
        n = int(np.clip(round(duration * fps) + 1, 2, MAX_KEYS))
        times = np.linspace(0.0, duration, n)
        eul = np.zeros((n, B, 3)); trans = np.zeros((n, B, 3)); root = np.zeros((n, 3))
        for ti, t in enumerate(times):
            trs = [list(x) for x in base]
            for node, path, tin, vals, interp in chans:
                k = ("translation", "rotation", "scale").index(path)
                trs[node][k] = _sample(tin, vals, interp, t, path == "rotation")
            W = _world(js, order, par, [_trs(*x) for x in trs])
            Q, P = [], []
            for b in rig.bones:
                d = W[b.node] @ rig.deform_base[b.node]  # glTF world deformation from the bind pose
                Q.append(rig.frame @ _rotation(d[:3, :3]) @ rig.frame.T)
                P.append(rig.to_display((d @ np.r_[rig.bind_world[b.node][:3, 3], 1.0])[:3][None])[0])
            for bi, b in enumerate(rig.bones):
                if b.parent < 0:
                    local = Q[bi]
                    root[ti] = P[bi] - rest[bi]
                else:
                    qp = Q[b.parent]
                    local = qp.T @ Q[bi]
                    pred = P[b.parent] + qp @ (rest[bi] - rest[b.parent])
                    trans[ti, bi] = qp.T @ (P[bi] - pred)
                eul[ti, bi] = euler_zyx(local)
        eul = np.degrees(np.unwrap(np.radians(eul), axis=0))
        if np.abs(eul).max() < 0.5 and np.abs(trans).max() < 0.05 and np.abs(root).max() < 0.05:
            _log(log, f"  clip '{anim.get('name', ai)}' skipped: identical to the bind pose")
            continue
        name = _clip_name(anim.get("name", f"clip{ai}"), used)
        clips.append(Clip(name, max(duration, 1e-3), times, eul, trans, root))
        _log(log, f"  clip {name}: {duration:.2f}s, {n} keys")
    return clips


# --------------------------------------------------------------------------- emission

def _f(v: float) -> str:
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def bone_lines(rig: Rig) -> list[str]:
    out = []
    for b in rig.bones:
        p = f"({_f(b.pos[0])}, {_f(b.pos[1])}, {_f(b.pos[2])})"
        out.append(f'bone("{b.name}", pos: {p});' if b.parent < 0 else
                   f'bone("{b.name}", pos: {p}, parent: "{rig.bones[b.parent].name}");')
    return out


def skin_groups(bones: np.ndarray, weights: np.ndarray):
    """Stable point permutation making every identical (bones, weights) mix contiguous, and the mixes."""
    keys = [tuple((int(b), round(float(w), 4)) for b, w in zip(bs, ws) if w > 0) for bs, ws in zip(bones, weights)]
    uniq = {k: i for i, k in enumerate(dict.fromkeys(sorted(set(keys))))}
    gid = np.array([uniq[k] for k in keys])
    perm = np.argsort(gid, kind="stable")
    groups = []
    sorted_gid = gid[perm]
    for k, i in uniq.items():
        idx = np.flatnonzero(sorted_gid == i)
        groups.append((int(idx[0]), int(idx[-1]), k))
    return perm, groups


def skin_lines(rig: Rig, groups) -> list[str]:
    out = []
    for a, b, mix in groups:
        if len(mix) == 1 and mix[0][0] == 0:
            continue  # unbound geometry already rides the root
        names = " ".join(rig.bones[i].name for i, _ in mix)
        ws = ", ".join(_f(w) if w < 1 else "1" for _, w in mix)
        sel = f'verts("{a}-{b}")' if b > a else f"verts({a})"
        out.append(f'g = skin_weights(g, selection: {sel}, bones: "{names}", weights: [{ws}]);')
    return out


def clip_lines(rig: Rig, clip: Clip, mesh: str, loop: bool | None = None) -> list[str]:
    if loop is None:  # glTF has no loop flag: loop only when the last pose returns to the first
        loop = (np.abs(clip.euler[-1] - clip.euler[0]).max() < 5 and np.abs(clip.trans[-1] - clip.trans[0]).max() < 1
                and np.abs(clip.root_offset[-1] - clip.root_offset[0]).max() < 1)
    moving = [bi for bi in range(len(rig.bones)) if np.abs(clip.euler[:, bi]).max() >= 0.05]
    moving_t = [bi for bi in range(len(rig.bones)) if rig.bones[bi].parent >= 0 and np.abs(clip.trans[:, bi]).max() >= 0.01]
    has_root = np.abs(clip.root_offset).max() >= 0.01
    lines = [f"#clip {clip.name} on {mesh}", f"duration: {_f(clip.duration) if clip.duration >= 0.01 else '0.01'}",
             f"loop: {'true' if loop else 'false'}"]
    for ti, t in enumerate(clip.times):
        parts = [_f(t)]
        parts += [f"{rig.bones[bi].name}: ({_f(clip.euler[ti, bi, 0])}, {_f(clip.euler[ti, bi, 1])}, {_f(clip.euler[ti, bi, 2])})"
                  for bi in moving]
        if moving_t:
            tr = ", ".join(f"{rig.bones[bi].name}: ({_f(clip.trans[ti, bi, 0])}, {_f(clip.trans[ti, bi, 1])}, {_f(clip.trans[ti, bi, 2])})"
                           for bi in moving_t)
            parts.append(f"translate: ({tr})")
        if has_root:
            o = clip.root_offset[ti]
            parts.append(f"offset: ({_f(o[0])}, {_f(o[1])}, {_f(o[2])})")
        parts.append('ease: "linear"')
        lines.append(f"key({', '.join(parts)})")
    lines.append("#end")
    return lines


def _log(log, msg: str) -> None:
    if log:
        log(msg)
