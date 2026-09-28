#!/usr/bin/env python3
"""Convert a .glb / .gltf model into a native Oriverse `#mesh` (Weave) block.

Pipeline: parse glTF -> bake node transforms -> weld -> decimate (QEM) ->
per-face colour from texture/vertex colours -> k-means palette ->
manifold clean-up -> emit `mesh_cage` + `paint` Weave source.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import struct
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

import glb_rig

COMPONENT_DTYPES = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
TYPE_SIZES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
NORMALIZE_DIV = {np.int8: 127.0, np.uint8: 255.0, np.int16: 32767.0, np.uint16: 65535.0}


# --------------------------------------------------------------------------- glTF loading

class Gltf:
    def __init__(self, path: Path):
        self.path = path
        data = path.read_bytes()
        self.bin = b""
        if data[:4] == b"glTF":
            _, _, length = struct.unpack_from("<III", data, 0)
            off = 12
            while off < length:
                clen, ctype = struct.unpack_from("<II", data, off)
                chunk = data[off + 8: off + 8 + clen]
                if ctype == 0x4E4F534A:
                    self.json = json.loads(chunk)
                elif ctype == 0x004E4942:
                    self.bin = chunk
                off += 8 + clen
        else:
            self.json = json.loads(data)
        self._buffers: dict[int, bytes] = {}
        self._images: dict[int, np.ndarray] = {}

    def buffer(self, i: int) -> bytes:
        if i not in self._buffers:
            uri = self.json["buffers"][i].get("uri")
            self._buffers[i] = self.bin if uri is None else self._load_uri(uri)
        return self._buffers[i]

    def _load_uri(self, uri: str) -> bytes:
        if uri.startswith("data:"):
            return base64.b64decode(uri.split(",", 1)[1])
        return (self.path.parent / uri).read_bytes()

    def buffer_view(self, i: int) -> tuple[bytes, int, int | None]:
        bv = self.json["bufferViews"][i]
        buf = self.buffer(bv["buffer"])
        off = bv.get("byteOffset", 0)
        return buf[off: off + bv["byteLength"]], bv["byteLength"], bv.get("byteStride")

    def accessor(self, i: int) -> np.ndarray:
        acc = self.json["accessors"][i]
        dtype = COMPONENT_DTYPES[acc["componentType"]]
        ncomp = TYPE_SIZES[acc["type"]]
        count = acc["count"]
        if "sparse" in acc:
            raise NotImplementedError("sparse accessors are not supported")
        if "bufferView" not in acc:
            out = np.zeros((count, ncomp), dtype=dtype)
        else:
            view, _, stride = self.buffer_view(acc["bufferView"])
            off = acc.get("byteOffset", 0)
            itemsize = np.dtype(dtype).itemsize * ncomp
            if stride and stride != itemsize:
                raw = np.frombuffer(view, dtype=np.uint8, count=stride * (count - 1) + itemsize, offset=off)
                rows = np.lib.stride_tricks.as_strided(raw, shape=(count, itemsize), strides=(stride, 1))
                out = np.frombuffer(rows.copy().tobytes(), dtype=dtype).reshape(count, ncomp)
            else:
                out = np.frombuffer(view, dtype=dtype, count=count * ncomp, offset=off).reshape(count, ncomp)
        if acc.get("normalized") and dtype in NORMALIZE_DIV:
            return np.maximum(out.astype(np.float64) / NORMALIZE_DIV[dtype], -1.0)
        return out.astype(np.float64) if dtype == np.float32 else out.copy()

    def image(self, i: int) -> np.ndarray:
        if i not in self._images:
            img = self.json["images"][i]
            raw = self.buffer_view(img["bufferView"])[0] if "bufferView" in img else self._load_uri(img["uri"])
            self._images[i] = np.asarray(Image.open(io.BytesIO(raw)).convert("RGBA"), dtype=np.float32) / 255.0
        return self._images[i]


def node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
    t = np.array(node.get("translation", [0, 0, 0]), dtype=np.float64)
    x, y, z, w = node.get("rotation", [0, 0, 0, 1])
    s = np.array(node.get("scale", [1, 1, 1]), dtype=np.float64)
    r = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    m = np.eye(4)
    m[:3, :3] = r * s
    m[:3, 3] = t
    return m


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def sample_texture(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear, repeat-wrapped sample of an RGBA float image (sRGB) -> linear RGB."""
    h, w = img.shape[:2]
    x = (uv[:, 0] % 1.0) * w - 0.5
    y = (uv[:, 1] % 1.0) * h - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    x0 %= w; y0 %= h
    x1 = (x0 + 1) % w; y1 = (y0 + 1) % h
    c = (img[y0, x0, :3] * (1 - fx) * (1 - fy) + img[y0, x1, :3] * fx * (1 - fy)
         + img[y1, x0, :3] * (1 - fx) * fy + img[y1, x1, :3] * fx * fy)
    return srgb_to_linear(c.astype(np.float64))


@dataclass
class Soup:
    positions: list[np.ndarray] = field(default_factory=list)
    faces: list[np.ndarray] = field(default_factory=list)
    colors: list[np.ndarray] = field(default_factory=list)  # linear RGB per face
    roughness: list[float] = field(default_factory=list)
    metallic: list[float] = field(default_factory=list)
    face_weights: list[int] = field(default_factory=list)
    parts: list[np.ndarray] = field(default_factory=list)  # source primitive id per face
    joints: list[np.ndarray] = field(default_factory=list)  # (V, 4) glTF node index per influence, -1 = none
    weights: list[np.ndarray] = field(default_factory=list)  # (V, 4)
    skin_mesh_matrix: dict = field(default_factory=dict)  # skin index -> world matrix of a mesh node using it
    offset: int = 0


def primitive_triangles(mode: int, idx: np.ndarray) -> np.ndarray:
    if mode == 4:
        return idx.reshape(-1, 3)
    if mode == 5:
        tris = [(idx[i], idx[i + 1], idx[i + 2]) if i % 2 == 0 else (idx[i + 1], idx[i], idx[i + 2]) for i in range(len(idx) - 2)]
        return np.array(tris, dtype=np.int64).reshape(-1, 3)
    if mode == 6:
        return np.array([(idx[0], idx[i], idx[i + 1]) for i in range(1, len(idx) - 1)], dtype=np.int64).reshape(-1, 3)
    return np.zeros((0, 3), dtype=np.int64)


def face_colors_for_primitive(g: Gltf, prim: dict, faces: np.ndarray) -> tuple[np.ndarray, float, float]:
    mats = g.json.get("materials", [])
    mat = mats[prim["material"]] if "material" in prim and prim["material"] < len(mats) else {}
    pbr = mat.get("pbrMetallicRoughness", {})
    factor = np.array(pbr.get("baseColorFactor", [1, 1, 1, 1])[:3], dtype=np.float64)
    attrs = prim["attributes"]
    n = len(faces)
    # sample at the 3 corners and the centroid of each face (barycentric weights)
    bary = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1 / 3, 1 / 3, 1 / 3]], dtype=np.float64)
    col = np.ones((n, 3), dtype=np.float64)
    tex = pbr.get("baseColorTexture")
    if tex is not None and f"TEXCOORD_{tex.get('texCoord', 0)}" in attrs:
        uv = g.accessor(attrs[f"TEXCOORD_{tex.get('texCoord', 0)}"])[:, :2]
        img = g.image(g.json["textures"][tex["index"]]["source"])
        corner_uv = uv[faces]  # (n, 3, 2)
        samples = np.einsum("kc,ncd->nkd", bary, corner_uv).reshape(-1, 2)
        col = sample_texture(img, samples).reshape(n, 4, 3).mean(axis=1)
    if "COLOR_0" in attrs:
        vc = g.accessor(attrs["COLOR_0"])[:, :3]
        col = col * vc[faces].mean(axis=1)
    return col * factor, float(pbr.get("roughnessFactor", 1.0)), float(pbr.get("metallicFactor", 1.0))


def load_scene(g: Gltf, exclude: re.Pattern | None, log=None, soup: Soup | None = None):
    js = g.json
    soup = soup if soup is not None else Soup()
    scene = js.get("scenes", [{"nodes": list(range(len(js.get("nodes", []))))}])[js.get("scene", 0)]
    joint_nodes = {j for sk in js.get("skins", []) for j in sk["joints"]}

    def visit(ni: int, parent: np.ndarray, hidden: bool, joint: int = -1) -> None:
        node = js["nodes"][ni]
        m = parent @ node_matrix(node)
        mesh_i = node.get("mesh")
        name = node.get("name", "")
        skip = hidden or bool(exclude and name and exclude.search(name))
        joint = ni if ni in joint_nodes else joint
        if mesh_i is not None:
            mesh = js["meshes"][mesh_i]
            mname = mesh.get("name", "")
            if skip or (exclude and mname and exclude.search(mname)):
                _log(log, f"  skip node '{name}' mesh '{mname}'")
            else:
                for prim in mesh["primitives"]:
                    pos = g.accessor(prim["attributes"]["POSITION"])[:, :3]
                    idx = g.accessor(prim["indices"]).ravel().astype(np.int64) if "indices" in prim else np.arange(len(pos))
                    faces = primitive_triangles(prim.get("mode", 4), idx)
                    if not len(faces):
                        continue
                    col, rough, metal = face_colors_for_primitive(g, prim, faces)
                    wpos = (np.c_[pos, np.ones(len(pos))] @ m.T)[:, :3]
                    if np.linalg.det(m[:3, :3]) < 0:
                        faces = faces[:, [0, 2, 1]]
                    soup.positions.append(wpos)
                    soup.faces.append(faces + soup.offset)
                    soup.colors.append(col)
                    soup.roughness.append(rough)
                    soup.metallic.append(metal)
                    soup.face_weights.append(len(faces))
                    soup.parts.append(np.full(len(faces), len(soup.parts)))
                    jn = np.full((len(pos), 4), -1, np.int64); wt = np.zeros((len(pos), 4))
                    if "skin" in node and "JOINTS_0" in prim["attributes"] and "WEIGHTS_0" in prim["attributes"]:
                        skin = js["skins"][node["skin"]]
                        soup.skin_mesh_matrix.setdefault(node["skin"], m)
                        jl = g.accessor(prim["attributes"]["JOINTS_0"]).astype(np.int64)
                        wt = g.accessor(prim["attributes"]["WEIGHTS_0"]).astype(np.float64)
                        jn = np.array(skin["joints"], dtype=np.int64)[jl]
                    elif joint >= 0:  # rigid mesh parented under a joint (props, weapons)
                        jn[:, 0] = joint; wt[:, 0] = 1.0
                    soup.joints.append(jn); soup.weights.append(wt)
                    soup.offset += len(pos)
                    _log(log, f"  mesh '{mname or mesh_i}': {len(pos)} verts, {len(faces)} tris")
        for c in node.get("children", []):
            visit(c, m, skip, joint)

    for ni in scene["nodes"]:
        visit(ni, np.eye(4), False)
    if not soup.faces:
        raise SystemExit("no triangle geometry found")
    w = np.array(soup.face_weights, dtype=np.float64)
    return (np.vstack(soup.positions), np.vstack(soup.faces), np.vstack(soup.colors),
            float(np.average(soup.roughness, weights=w)), float(np.average(soup.metallic, weights=w)),
            np.concatenate(soup.parts))


# --------------------------------------------------------------------------- geometry processing

def weld(points: np.ndarray, faces: np.ndarray, tol: float, return_first: bool = False):
    q = np.round(points / tol).astype(np.int64)
    _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    if return_first:
        return points[first], inv.ravel()[faces], first
    return points[first], inv.ravel()[faces]


def tri_areas(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a, b, c = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def face_keys(faces: np.ndarray, n: int) -> np.ndarray:
    s = np.sort(faces, axis=1).astype(np.int64)
    return (s[:, 0] * n + s[:, 1]) * n + s[:, 2]


def decimate(points, faces, colors, target_tris: int, agg: float, parts: np.ndarray | None = None):
    """QEM-decimate and carry per-face colours through the collapse map.

    Multi-part models are decimated part by part with a proportional budget and fixed part
    borders, so thin parts (ribbons, straps, hair cards) are not collapsed away first and the
    seams between parts stay closed."""
    ids = np.unique(parts) if parts is not None else np.zeros(1)
    if len(ids) < 2 or target_tris >= len(faces):
        return _decimate_one(points, faces, colors, target_tris, agg, False)
    out_p, out_f, out_c = [], [], []
    off = 0
    for pid in ids:
        sel = parts == pid
        used, local = np.unique(faces[sel], return_inverse=True)
        f = local.reshape(-1, 3)
        t = max(8, int(round(target_tris * sel.sum() / len(faces))))
        dp, df, dc = _decimate_one(points[used], f, colors[sel], t, agg, True)
        out_p.append(dp); out_f.append(df + off); out_c.append(dc)
        off += len(dp)
    pts = np.vstack(out_p)
    diag = np.linalg.norm(pts.max(0) - pts.min(0))
    pts, fcs = weld(pts, np.vstack(out_f), max(diag * 1e-7, 1e-12))
    return pts, fcs, np.vstack(out_c)


def _decimate_one(points, faces, colors, target_tris: int, agg: float, preserve_border: bool):
    import fast_simplification as fs

    areas = tri_areas(points, faces)
    if target_tris >= len(faces):
        return points, faces, colors
    _, _, collapses = fs.simplify(points.astype(np.float32), faces.astype(np.int32), target_count=target_tris,
                                  agg=agg, preserve_border=preserve_border, return_collapses=True)
    dpts, dfaces, mapping = fs.replay_simplification(points.astype(np.float32), faces.astype(np.int32), collapses)
    dpts = dpts.astype(np.float64)
    dfaces = np.asarray(dfaces, dtype=np.int64)
    n = len(dpts)
    mapped = mapping[faces]
    ok = (mapped[:, 0] != mapped[:, 1]) & (mapped[:, 1] != mapped[:, 2]) & (mapped[:, 0] != mapped[:, 2]) & (mapped >= 0).all(1)
    dkeys = face_keys(dfaces, n)
    order = np.argsort(dkeys)
    okeys = face_keys(mapped[ok], n)
    pos = np.clip(np.searchsorted(dkeys[order], okeys), 0, len(order) - 1)
    hit = dkeys[order][pos] == okeys
    target = order[pos[hit]]
    w = areas[ok][hit]
    acc = np.zeros((len(dfaces), 3)); wsum = np.zeros(len(dfaces))
    np.add.at(acc, target, colors[ok][hit] * w[:, None])
    np.add.at(wsum, target, w)
    # fallback: per decimated vertex average of all original faces touching it
    vacc = np.zeros((n, 3)); vw = np.zeros(n)
    valid = mapping[faces] >= 0
    for k in range(3):
        m = valid[:, k]
        np.add.at(vacc, mapping[faces[m, k]], colors[m] * areas[m, None])
        np.add.at(vw, mapping[faces[m, k]], areas[m])
    vcol = vacc / np.maximum(vw, 1e-30)[:, None]
    dcol = np.where((wsum > 0)[:, None], acc / np.maximum(wsum, 1e-30)[:, None], vcol[dfaces].mean(axis=1))
    return dpts, dfaces, dcol


def edge_table(faces: np.ndarray):
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    fid = np.tile(np.arange(len(faces)), 3)
    key = np.sort(e, axis=1)
    return e, key, fid


def drop_bad_faces(points: np.ndarray, faces: np.ndarray, colors: np.ndarray, min_area: float,
                   nonmanifold: bool = True):
    """Remove degenerate faces, cancel coincident face pairs and (optionally) keep at most two faces per edge."""
    n0 = len(faces)
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    keep &= tri_areas(points, faces) > min_area
    faces, colors = faces[keep], colors[keep]
    # coincident faces are zero-volume fins left by decimation: an even count cancels out
    # completely (the surrounding surface stays closed), an odd count keeps one copy
    _, first, inv, cnt = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_inverse=True, return_counts=True)
    keep_first = first[cnt % 2 == 1]
    keep_first.sort()
    faces, colors = faces[keep_first], colors[keep_first]
    if not nonmanifold:
        return faces, colors, n0 - len(faces)
    _, key, fid = edge_table(faces)
    _, einv, ecount = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    einv = einv.ravel()
    bad_edges = np.flatnonzero(ecount > 2)
    drop = np.zeros(len(faces), bool)
    if len(bad_edges):
        area = tri_areas(points, faces)
        by_edge = defaultdict(list)
        sel = np.isin(einv, bad_edges)
        for e_, f_ in zip(einv[sel], fid[sel]):
            by_edge[e_].append(f_)
        for fl in by_edge.values():
            fl = [f for f in fl if not drop[f]]
            fl.sort(key=lambda f: -area[f])
            drop[fl[2:]] = True
    return faces[~drop], colors[~drop], n0 - int((~drop).sum())


@dataclass
class CleanReport:
    removed_faces: int = 0
    added_faces: int = 0
    holes_filled: int = 0
    holes_left: int = 0
    boundary_edges: int = 0
    seam_vertices: int = 0

    @property
    def watertight(self) -> bool:
        return self.boundary_edges == 0


def clean_mesh(points: np.ndarray, faces: np.ndarray, colors: np.ndarray, min_area: float,
               fill: bool, max_hole: float, precision: int):
    """Make the triangle set acceptable for mesh_cage (no degenerate faces, <=2 faces per edge,
    consistent winding, manifold vertex fans, no unused points) without opening holes:

    1. drop degenerate faces and cancel coincident face pairs (zero-volume fins);
    2. orient every component outward;
    3. with `fill`, close real boundary loops (ear clipping / centroid fan);
    4. cut non-manifold or wrongly wound edges into seams by giving each side its own copy of
       the vertices - every face is kept and the copies coincide, so the surface stays closed;
    5. only if something is still invalid, fall back to dropping the offending faces."""
    rep = CleanReport()
    n0 = len(faces)
    diag = float(np.linalg.norm(points.max(0) - points.min(0)))
    faces, colors = collapse_degenerate(points, faces, colors, min_area)
    faces, colors, _ = drop_bad_faces(points, faces, colors, min_area, nonmanifold=False)
    faces = orient(faces, points)
    if fill:
        for _ in range(6):
            n_before = len(faces)
            points, faces, colors, filled, rep.holes_left = fill_holes(points, faces, colors, max_hole * diag, min_area, precision)
            rep.holes_filled += filled
            rep.added_faces += len(faces) - n_before
            faces, colors, _ = drop_bad_faces(points, faces, colors, min_area, nonmanifold=False)
            faces = orient(faces, points)
            if not filled:
                break
    for _ in range(8):
        faces, points, split = split_along_bad_edges(points, faces)
        rep.seam_vertices += split
        faces, colors, dropped = drop_bad_faces(points, faces, colors, min_area)
        bad = winding_conflicts(points, faces)
        faces, colors = faces[~bad], colors[~bad]
        if not dropped and not bad.any():
            break
        faces = orient(faces, points)
    used = np.unique(faces)
    remap = np.full(len(points), -1, np.int64); remap[used] = np.arange(len(used))
    faces, points = remap[faces], points[used]
    rep.removed_faces = max(0, n0 + rep.added_faces - len(faces))
    rep.boundary_edges = open_boundary_edges(points, faces)
    if not fill or rep.boundary_edges == 0:
        rep.holes_left = 0 if rep.boundary_edges == 0 else len(boundary_loops(faces))
    return points, faces, colors, rep


def collapse_degenerate(points: np.ndarray, faces: np.ndarray, colors: np.ndarray, min_area: float,
                        passes: int = 12):
    """Remove sliver/needle faces by collapsing their shortest edge (merging its two vertices),
    so the neighbouring faces close over the gap instead of leaving a hole."""
    faces = faces.copy()
    for _ in range(passes):
        ok = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
        faces, colors = faces[ok], colors[ok]
        deg = np.flatnonzero(tri_areas(points, faces) <= min_area)
        if not len(deg):
            break
        target = np.arange(len(points))
        touched = np.zeros(len(points), bool)
        for f in deg:
            t = faces[f]
            ls = [np.linalg.norm(points[t[k]] - points[t[(k + 1) % 3]]) for k in range(3)]
            k = int(np.argmin(ls))
            a, b = int(t[k]), int(t[(k + 1) % 3])
            if touched[a] or touched[b]:
                continue
            touched[a] = touched[b] = True
            target[b] = a
        faces = target[faces]
    ok = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    return faces[ok], colors[ok]


def open_boundary_edges(points: np.ndarray, faces: np.ndarray) -> int:
    """Boundary edges that are real openings, i.e. not matched by a coincident seam edge."""
    b = boundary_half_edges(faces)
    if not len(b):
        return 0
    q = np.round(points * 1000).astype(np.int64)
    _, pid = np.unique(q, axis=0, return_inverse=True)
    pid = pid.ravel()
    pairs = np.sort(np.c_[pid[b[:, 0]], pid[b[:, 1]]], axis=1)
    _, inv, cnt = np.unique(pairs, axis=0, return_inverse=True, return_counts=True)
    return int((cnt[inv.ravel()] == 1).sum())


def boundary_half_edges(faces: np.ndarray) -> np.ndarray:
    """Directed face edges (u, v) that have no neighbouring face, plus their face ids in column 2."""
    e, key, fid = edge_table(faces)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    m = cnt[inv.ravel()] == 1
    return np.c_[e[m], fid[m]]


def boundary_loops(faces: np.ndarray) -> list[list[int]]:
    """Closed hole loops, oriented the way the filling faces must traverse them."""
    out: dict[int, list[int]] = defaultdict(list)
    for u, v, _ in boundary_half_edges(faces):
        out[int(v)].append(int(u))
    loops = []
    while out:
        start = next(iter(out))
        loop, cur = [start], start
        while True:
            nxts = out.get(cur)
            if not nxts:
                loop = []  # open chain; cannot happen on a consistently wound mesh
                break
            n = nxts.pop()
            if not nxts:
                del out[cur]
            if n == start:
                break
            if n in loop:  # figure-eight hole: peel off the sub-loop
                i = loop.index(n)
                loops.append(loop[i:])
                loop = loop[:i + 1]
            else:
                loop.append(n)
            cur = n
        if len(loop) >= 3:
            loops.append(loop)
    return [l for l in loops if len(l) >= 3]


def _ear_clip(points: np.ndarray, loop: list[int], edges: set, min_area: float) -> list[tuple[int, int, int]] | None:
    P = points[loop]
    normal = np.cross(P, np.roll(P, -1, axis=0)).sum(0)
    if np.linalg.norm(normal) < 1e-12:
        return None
    normal /= np.linalg.norm(normal)
    ax = np.cross(normal, [1.0, 0, 0] if abs(normal[0]) < 0.9 else [0, 1.0, 0]); ax /= np.linalg.norm(ax)
    ay = np.cross(normal, ax)
    L = list(loop)
    tris = []
    new_edges: set = set()
    while len(L) > 3:
        Q = points[L]
        q2 = np.c_[Q @ ax, Q @ ay]
        best, best_cost = None, np.inf
        n = len(L)
        for i in range(n):
            a, b, c = L[i - 1], L[i], L[(i + 1) % n]
            diag = (min(a, c), max(a, c))
            if diag in edges or diag in new_edges:
                continue
            pa, pb, pc = points[a], points[b], points[c]
            tn = np.cross(pb - pa, pc - pa)
            area = 0.5 * np.linalg.norm(tn)
            if area <= min_area or tn @ normal <= 0:
                continue
            A, B, C = q2[i - 1], q2[i], q2[(i + 1) % n]
            others = np.delete(q2, [(i - 1) % n, i, (i + 1) % n], axis=0)
            if len(others):
                d1 = (B[0] - A[0]) * (others[:, 1] - A[1]) - (B[1] - A[1]) * (others[:, 0] - A[0])
                d2 = (C[0] - B[0]) * (others[:, 1] - B[1]) - (C[1] - B[1]) * (others[:, 0] - B[0])
                d3 = (A[0] - C[0]) * (others[:, 1] - C[1]) - (A[1] - C[1]) * (others[:, 0] - C[0])
                if ((d1 > 0) & (d2 > 0) & (d3 > 0)).any():
                    continue
            cost = np.linalg.norm(pc - pa) / np.sqrt(area)
            if cost < best_cost:
                best, best_cost = i, cost
        if best is None:
            return None
        a, b, c = L[best - 1], L[best], L[(best + 1) % n]
        tris.append((a, b, c))
        new_edges.add((min(a, c), max(a, c)))
        L.pop(best)
    a, b, c = L
    if 0.5 * np.linalg.norm(np.cross(points[b] - points[a], points[c] - points[a])) <= min_area:
        return None
    tris.append((a, b, c))
    return tris


def fill_holes(points: np.ndarray, faces: np.ndarray, colors: np.ndarray, max_perimeter: float,
               min_area: float, precision: int):
    """Close boundary loops: ear clipping in the hole's best-fit plane, centroid fan as fallback.
    New faces inherit the colour of the face across the hole edge."""
    loops = boundary_loops(faces)
    if not loops:
        return points, faces, colors, 0, 0
    _, key, fid = edge_table(faces)
    edges = set(map(tuple, np.unique(key, axis=0).tolist()))
    edge_face = {(int(u), int(v)): int(f) for u, v, f in boundary_half_edges(faces)}
    existing = set(map(tuple, np.sort(faces, axis=1).tolist()))
    fn = np.cross(points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1), 1e-30)[:, None]
    new_pts, new_faces, new_cols = [], [], []
    nxt = len(points)
    filled = left = 0
    for loop in loops:
        P = points[loop]
        if np.linalg.norm(P - np.roll(P, -1, axis=0), axis=1).sum() > max_perimeter:
            left += 1
            continue
        adj = [edge_face[(loop[(i + 1) % len(loop)], loop[i])] for i in range(len(loop))]
        # a real hole continues the surrounding surface; the outline of an open sheet (ribbon,
        # plane, cloth) would be capped by a copy of the sheet facing the other way -> keep it open
        hole_n = np.cross(P, np.roll(P, -1, axis=0)).sum(0)
        if np.linalg.norm(hole_n) > 1e-12 and float(np.mean(fn[adj] @ (hole_n / np.linalg.norm(hole_n)))) < -0.2:
            left += 1
            continue
        col = np.mean(colors[adj], axis=0)
        tris = _ear_clip(points, loop, edges, min_area)
        if tris is not None and any(tuple(sorted(t)) in existing for t in tris):
            left += 1
            continue
        if tris is None:
            c = np.round(P.mean(0), precision)
            tris = [(loop[i], loop[(i + 1) % len(loop)], nxt) for i in range(len(loop))]
            new_pts.append(c)
            nxt += 1
        for t in tris:
            edges.update({(min(t[0], t[1]), max(t[0], t[1])), (min(t[1], t[2]), max(t[1], t[2])), (min(t[0], t[2]), max(t[0], t[2]))})
        new_faces += tris
        new_cols += [col] * len(tris)
        filled += 1
    if new_pts:
        points = np.vstack([points, np.array(new_pts)])
    if new_faces:
        faces = np.vstack([faces, np.array(new_faces, dtype=np.int64)])
        colors = np.vstack([colors, np.array(new_cols)])
    return points, faces, colors, filled, left


def orient(faces: np.ndarray, points: np.ndarray | None = None) -> np.ndarray:
    """Flip faces so every shared edge is traversed in opposite directions (per component);
    with `points`, also turn each component outward (positive signed volume)."""
    e, key, fid = edge_table(faces)
    _, einv = np.unique(key, axis=0, return_inverse=True)
    einv = einv.ravel()
    fwd = e[:, 0] < e[:, 1]
    pairs = defaultdict(list)
    for ei, f, d in zip(einv, fid, fwd):
        pairs[ei].append((f, d))
    adj = defaultdict(list)
    for lst in pairs.values():
        if len(lst) == 2:
            (f1, d1), (f2, d2) = lst
            same = d1 == d2  # same direction -> inconsistent unless one is flipped
            adj[f1].append((f2, same)); adj[f2].append((f1, same))
    flip = np.zeros(len(faces), bool)
    comp = np.full(len(faces), -1)
    ncomp = 0
    for s in range(len(faces)):
        if comp[s] >= 0:
            continue
        comp[s] = ncomp
        stack = [s]
        while stack:
            f = stack.pop()
            for g, same in adj[f]:
                if comp[g] < 0:
                    comp[g] = ncomp
                    flip[g] = flip[f] ^ same
                    stack.append(g)
        ncomp += 1
    out = faces.copy()
    out[flip] = out[flip][:, [0, 2, 1]]
    if points is not None:
        a, b, c = points[out[:, 0]], points[out[:, 1]], points[out[:, 2]]
        vol = np.einsum("ij,ij->i", a, np.cross(b, c))
        inward = np.bincount(comp, weights=vol, minlength=ncomp) < 0
        rev = inward[comp]
        out[rev] = out[rev][:, [0, 2, 1]]
    return out


def winding_conflicts(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Mask of faces to drop so no edge is traversed twice in the same direction."""
    e, _, fid = edge_table(faces)
    _, inv, cnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    drop = np.zeros(len(faces), bool)
    dup = np.flatnonzero(cnt > 1)
    if len(dup):
        area = tri_areas(points, faces)
        groups = defaultdict(list)
        sel = np.isin(inv, dup)
        for k, f in zip(inv[sel], fid[sel]):
            groups[k].append(f)
        for fl in groups.values():
            if sum(not drop[f] for f in fl) > 1:
                drop[min(fl, key=lambda f: area[f])] = True
    return drop


def split_along_bad_edges(points: np.ndarray, faces: np.ndarray):
    """Give each vertex one copy per fan of faces connected through *good* edges (exactly two
    faces, opposite winding). This detaches bow-tie vertices and cuts non-manifold or wrongly
    wound edges into coincident seams, without deleting any face."""
    vf = defaultdict(list)
    for f, tri in enumerate(faces):
        for v in tri:
            vf[int(v)].append(f)
    faces = faces.copy()
    new_pts = [points]
    nxt = len(points)
    split = 0
    for v, fl in vf.items():
        if len(fl) < 2:
            continue
        parent = {f: f for f in fl}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        by_other = defaultdict(list)
        for f in fl:
            t = faces[f]
            i = int(np.flatnonzero(t == v)[0])
            by_other[int(t[(i + 1) % 3])].append((f, True))   # traverses v -> u
            by_other[int(t[(i + 2) % 3])].append((f, False))  # traverses u -> v
        for lst in by_other.values():
            if len(lst) == 2 and lst[0][1] != lst[1][1]:
                parent[find(lst[1][0])] = find(lst[0][0])
        groups = defaultdict(list)
        for f in fl:
            groups[find(f)].append(f)
        if len(groups) < 2:
            continue
        for gi, gl in enumerate(sorted(groups.values(), key=len, reverse=True)):
            if gi == 0:
                continue
            for f in gl:
                faces[f][faces[f] == v] = nxt
            new_pts.append(points[v:v + 1])
            nxt += 1
            split += 1
    return faces, np.vstack(new_pts), split


def kmeans_palette(colors: np.ndarray, weights: np.ndarray, k: int, seed: int = 0, iters: int = 40):
    """Weighted k-means in (approx. perceptual) sRGB space; returns (palette_srgb, labels)."""
    x = linear_to_srgb(colors)
    k = min(k, len(np.unique(np.round(x * 255), axis=0)))
    rng = np.random.default_rng(seed)
    w = weights / weights.sum()
    centers = [x[rng.choice(len(x), p=w)]]
    for _ in range(1, k):
        d = np.min(((x[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        p = d * w
        centers.append(x[rng.choice(len(x), p=p / p.sum())] if p.sum() > 0 else x[rng.integers(len(x))])
    c = np.array(centers)
    for _ in range(iters):
        lab = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
        newc = np.array([np.average(x[lab == i], axis=0, weights=weights[lab == i]) if (lab == i).any() else c[i] for i in range(k)])
        if np.allclose(newc, c, atol=1e-5):
            break
        c = newc
    lab = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
    used = np.unique(lab)
    remap = np.full(k, -1); remap[used] = np.arange(len(used))
    return c[used], remap[lab]


# --------------------------------------------------------------------------- Weave emission

def fmt(v: float, prec: int) -> str:
    s = f"{v:.{prec}f}".rstrip("0").rstrip(".") if prec > 0 else str(int(round(v)))
    return "0" if s in ("-0", "") else s


@dataclass
class MeshLevel:
    points: np.ndarray
    faces: np.ndarray
    labels: np.ndarray
    report: CleanReport
    precision: int = 1
    colors: np.ndarray | None = None  # linear RGB per face, before palette assignment
    skin: tuple | None = None          # (bone ids (V,4), weights (V,4)) per point when rigged


def palette_hex(palette: np.ndarray) -> list[str]:
    return ["#" + "".join(f"{int(round(v * 255)):02x}" for v in np.clip(c, 0, 1)) for c in palette]


def emit_weave(name: str, levels: list[MeshLevel], palette: np.ndarray, roughness: float, metallic: float,
               smooth_angle: float | None, with_object: bool, world: bool, source: str,
               rig=None, clips=()) -> str:
    mats = [f"{name}C{i}" for i in range(len(palette))]
    base = levels[0]
    lines = [f"// Generated by glb2ori.py from {source}",
             f"// {len(base.points)} points, {len(base.faces)} triangles, {len(palette)} colours"
             + "".join(f"; lod {i}: {len(l.faces)} triangles" for i, l in enumerate(levels[1:], 1))]
    dress = [f"#texture {m} color={hx} roughness={fmt(roughness, 2)} metallic={fmt(metallic, 2)}"
             for m, hx in zip(mats, palette_hex(palette))]
    for i, lvl in enumerate(levels):
        lines += emit_mesh_block(name if i == 0 else f"{name} lod: {i}", lvl, mats, dress if i == 0 else [], smooth_angle,
                                 rig, declare_bones=i == 0)
        if i == 0:
            for c in clips:
                lines += glb_rig.clip_lines(rig, c, name)
    if world:
        lines += ["", 'object Ground { model="cube" bounds_min=(-3000,-3000,-20) bounds_max=(3000,3000,0) }']
    if with_object or world:
        lines += ["", f'object {name}Object {{ model = "mesh.{name}" position = (0, {"300" if world else "0"}, 0) }}']
    return "\n".join(lines) + "\n"


def emit_mesh_block(header: str, lvl: MeshLevel, mats: list[str], dress: list[str],
                    smooth_angle: float | None, rig=None, declare_bones: bool = True) -> list[str]:
    prec = lvl.precision
    order = np.argsort(lvl.labels, kind="stable")
    faces, labels, points = lvl.faces[order], lvl.labels[order], lvl.points
    groups = []
    if rig is not None and lvl.skin is not None:
        # order points so every identical skin mix is one contiguous verts("a-b") range
        perm, groups = glb_rig.skin_groups(*lvl.skin)
        inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
        points, faces = points[perm], inv[faces]
    tri = len(faces)
    lines = [f"#mesh {header}", *dress]
    # mesh_cage accounts ~9 indices per triangle (faces + cage edges); baked vertices grow with
    # material seams and hard edges, so reserve ~3x the cage points
    need_i, need_v = tri * 9 + 64, len(points) * 3
    if need_i > 90000 or need_v > 50000:
        lines.append(f"budget(verts: {min(200000, max(50000, need_v))}, indices: {min(600000, max(90000, need_i))});")
    if rig is not None and declare_bones:
        lines += glb_rig.bone_lines(rig)
    per_line = 8
    pts = [f"({fmt(p[0], prec)},{fmt(p[1], prec)},{fmt(p[2], prec)})" for p in points]
    fcs = [f"[{a},{b},{c}]" for a, b, c in faces]
    lines.append("let cage = mesh_cage(points: [")
    lines += ["    " + ",".join(pts[i:i + per_line]) + ("," if i + per_line < len(pts) else "") for i in range(0, len(pts), per_line)]
    lines.append("], faces: [")
    lines += ["    " + ",".join(fcs[i:i + per_line]) + ("," if i + per_line < len(fcs) else "") for i in range(0, len(fcs), per_line)]
    lines.append(f'], material: "{mats[0]}");')
    lines.append("var g = cage;")
    starts = np.searchsorted(labels, np.arange(len(mats)))
    ends = np.searchsorted(labels, np.arange(len(mats)), side="right")
    for i in range(1, len(mats)):
        if ends[i] > starts[i]:
            lines.append(f'g = paint(g, material: "{mats[i]}", selection: faces("{starts[i]}-{ends[i] - 1}"));')
    if groups:
        lines += glb_rig.skin_lines(rig, groups)
    lines.append(f"out.geo = smooth_shade(g, angle: {fmt(smooth_angle, 1)});" if smooth_angle is not None else "out.geo = g;")
    lines.append("#end")
    return lines


AXES = {"+x": (0, 1), "-x": (0, -1), "+y": (1, 1), "-y": (1, -1), "+z": (2, 1), "-z": (2, -1)}


def to_oriverse_frame(p: np.ndarray, faces: np.ndarray, up: str, forward: str, face_to: str, mirror: bool):
    """Map source axes (glTF: +Y up, +Z front) onto Oriverse (Z up, model front faces `face_to`)."""
    r = frame_matrix(up, forward, face_to, mirror)
    out = p @ r.T
    if np.linalg.det(r) < 0:
        faces = faces[:, [0, 2, 1]]
    return out, faces


def frame_matrix(up: str, forward: str, face_to: str, mirror: bool) -> np.ndarray:
    ui, us = AXES[up]; fi, fs_ = AXES[forward]
    src_up = np.zeros(3); src_up[ui] = us
    src_fw = np.zeros(3); src_fw[fi] = fs_
    src_side = np.cross(src_up, src_fw)
    basis_src = np.stack([src_fw, src_side, src_up])  # rows: forward, side, up in source coords
    ti, ts = AXES[face_to]
    dst_fw = np.zeros(3); dst_fw[ti] = ts
    dst_up = np.array([0.0, 0.0, 1.0])
    if abs(dst_fw @ dst_up) > 0.5:
        raise ValueError("face must be a horizontal axis (+x,-x,+y,-y)")
    dst_side = np.cross(dst_up, dst_fw)
    basis_dst = np.stack([dst_fw, dst_side, dst_up])
    r = basis_dst.T @ basis_src
    if mirror:
        r = np.diag([1.0, -1.0, 1.0]) @ r
    return r


# --------------------------------------------------------------------------- public API

MAX_TRIS = 64_000  # mesh_cage uses ~9 indices per triangle; Oriverse caps a mesh at 600k indices


@dataclass
class Source:
    points: np.ndarray
    faces: np.ndarray
    colors: np.ndarray
    roughness: float
    metallic: float
    meshes: list[str]
    parts: np.ndarray | None = None
    joints: np.ndarray | None = None   # (V, 4) glTF node per influence (-1 none), aligned with points
    weights: np.ndarray | None = None  # (V, 4)
    gltf: Gltf | None = None
    frame: np.ndarray | None = None    # 3x3 glTF world -> Oriverse axes
    skin_mesh_matrix: dict | None = None

    @property
    def rigged(self) -> bool:
        return bool(self.gltf and self.gltf.json.get("skins")) and self.joints is not None and (self.joints >= 0).any()


_SOURCE_CACHE: dict[tuple, Source] = {}


def load_source(path: Path, exclude: str | None = None, up: str = "+y", forward: str = "+z", face: str = "+x",
                mirror: bool = False, weld_tol: float = 1e-5, log=None) -> Source:
    """Parse, transform to the Oriverse frame and weld a model (cached per file state and options)."""
    path = Path(path)
    key = (str(path.resolve()), path.stat().st_mtime_ns, exclude, up, forward, face, mirror, weld_tol)
    if key in _SOURCE_CACHE:
        return _SOURCE_CACHE[key]
    g = Gltf(path)
    msgs: list[str] = []

    def collect(m: str) -> None:
        msgs.append(m.strip())
        _log(log, m)
    soup = Soup()
    pts, faces, cols, rough, metal, parts = load_scene(g, re.compile(exclude, re.I) if exclude else None, collect, soup)
    vj, vw = np.vstack(soup.joints), np.vstack(soup.weights)
    pts, faces = to_oriverse_frame(pts, faces, up, forward, face, mirror)
    diag = np.linalg.norm(pts.max(0) - pts.min(0))
    pts, faces, first = weld(pts, faces, max(diag * weld_tol, 1e-9), return_first=True)
    vj, vw = vj[first], vw[first]
    faces, cp, _ = drop_bad_faces(pts, faces, np.c_[cols, parts], 0.0)
    cols, parts = cp[:, :3], cp[:, 3].astype(np.int64)
    used = np.unique(faces)
    remap = np.full(len(pts), -1, np.int64); remap[used] = np.arange(len(used))
    src = Source(pts[used], remap[faces], cols, rough, metal, msgs, parts, vj[used], vw[used], g,
                 frame_matrix(up, forward, face, mirror), soup.skin_mesh_matrix)
    if len(_SOURCE_CACHE) > 4:
        _SOURCE_CACHE.pop(next(iter(_SOURCE_CACHE)))
    _SOURCE_CACHE[key] = src
    return src


def _log(log, msg: str) -> None:
    if log:
        log(msg)


def resolve_target(src_tris: int, tris: int | None, ratio: float | None, max_verts: int | None) -> int:
    """Pick the decimation target from exactly one of tris / ratio / max_verts (closed mesh: V ~ T/2)."""
    if ratio is not None:
        if not 0 < ratio <= 1:
            raise ValueError("ratio must be in (0, 1]")
        t = int(round(src_tris * ratio))
    elif max_verts is not None:
        t = int(max_verts) * 2
    else:
        t = int(tris or 8000)
    return max(8, min(t, src_tris, MAX_TRIS))


def auto_precision(points: np.ndarray, faces: np.ndarray) -> int:
    """Decimal places (cm) so rounding moves points by < ~1/40 of a typical edge (0..3)."""
    e = points[faces[:, [1, 2, 0]]] - points[faces]
    med = float(np.median(np.linalg.norm(e, axis=2)))
    return int(np.clip(np.ceil(-np.log10(max(med, 1e-9) / 40)), 0, 3))


def build_level(src: Source, target: int, scale: float, offset: np.ndarray, precision: int | None, fill: bool,
                max_hole: float, agg: float) -> MeshLevel:
    pts, faces, cols = decimate(src.points, src.faces, src.colors, target, agg, src.parts)
    pts = pts * scale - offset
    prec = auto_precision(pts, faces) if precision is None else precision
    # snap to the emitted precision first so validation sees exactly what Oriverse will parse
    pts, faces = weld(np.round(pts, prec), faces, 10.0 ** -(prec + 3))
    step = 10.0 ** -prec
    pts, faces, cols, rep = clean_mesh(pts, faces, cols, 0.05 * step * step, fill, max_hole, prec)
    return MeshLevel(pts, faces, np.zeros(len(faces), np.int64), rep, prec, cols)


def default_name(path: Path) -> str:
    name = re.sub(r"\W", "", Path(path).stem.title().replace("_", "").replace("-", "")) or "Model"
    return name if name[0].isalpha() else "M" + name


def convert(input: str | Path, output: str | Path | None = None, *, name: str | None = None,
            tris: int | None = 8000, ratio: float | None = None, max_verts: int | None = None,
            colors: int = 12, height: float | None = None, scale: float = 100.0,
            up: str = "+y", forward: str = "+z", face: str = "+x", mirror: bool = False,
            exclude: str | None = None, precision: int | None = None, smooth: float | None = 50.0, agg: float = 7.0,
            weld_tol: float = 1e-5, fill_holes: bool = True, max_hole: float = 1.0,
            lods: list[float] | tuple[float, ...] = (), with_object: bool = False, world: bool = False,
            rig: bool | None = None, rig_names: str = "humanoid", clips: bool = True, weight_step: float = 0.05,
            fps: float = 30.0, log=None) -> dict:
    """Convert a glTF/GLB model to an Oriverse `#mesh` .weave file and return a summary dict.

    Size: give one of `tris` (target triangles), `ratio` (fraction of the source triangles) or
    `max_verts` (target cage points). `lods` are extra far-display levels as fractions of the base
    triangle count, e.g. [0.3, 0.1]. `fill_holes` closes boundary loops whose perimeter is at most
    `max_hole` x the bounding-box diagonal (1.0 = every hole -> watertight).

    Rig: `rig` (default: auto when the file has a skin) exports bones + skin weights; `rig_names`
    "humanoid" renames recognisable bones to Oriverse's canonical humanoid names (free
    idle/walk/run retargeting) or "keep"; `clips` exports glTF animations as `#clip` blocks
    sampled at `fps` (max 64 keys); `weight_step` quantises skin weights."""
    input = Path(input)
    if not 1 <= colors <= 20:
        raise ValueError("colors must be 1..20 (Oriverse material slot cap)")
    if len(lods) > 3 or any(not 0 < r < 1 for r in lods):
        raise ValueError("lods: up to 3 fractions in (0, 1)")
    name = name or default_name(input)
    if not re.fullmatch(r"[A-Za-z]\w*", name):
        raise ValueError("name must be an identifier (letters, digits, _)")
    out_path = Path(output) if output else input.with_suffix(".weave")
    _log(log, f"loading {input}")
    src = load_source(input, exclude, up, forward, face, mirror, weld_tol, log)
    _log(log, f"  source: {len(src.points)} verts, {len(src.faces)} tris")
    lo, hi = src.points.min(0), src.points.max(0)
    s = height / (hi[2] - lo[2]) if height else scale
    offset = np.array([(hi[0] + lo[0]) / 2 * s, (hi[1] + lo[1]) / 2 * s, lo[2] * s])
    target = resolve_target(len(src.faces), tris, ratio, max_verts)
    warnings = []
    if (tris or 0) > MAX_TRIS or (max_verts or 0) * 2 > MAX_TRIS:
        warnings.append(f"target clamped to {MAX_TRIS} triangles (Oriverse mesh budget)")

    rig_obj = glb_rig.build_rig(src, s, offset, rig_names, log) if (rig if rig is not None else src.rigged) else None
    if rig and rig_obj is None:
        warnings.append("rig requested but the file has no skin")
    base = build_level(src, target, s, offset, precision, fill_holes, max_hole, agg)
    pts, faces, cols, rep = base.points, base.faces, base.colors, base.report
    if len(faces) > 1.3 * target:
        warnings.append(f"got {len(faces)} triangles for target {target}: part borders are kept fixed so "
                        "multi-part models stay seamless; exclude or simplify tiny parts to go lower")
    _log(log, f"  base: {len(faces)} tris, {len(pts)} points, precision {base.precision}, watertight={rep.watertight} "
              f"(filled {rep.holes_filled} holes, {rep.seam_vertices} seam vertices)")
    palette, labels = kmeans_palette(cols, tri_areas(pts, faces) + 1e-12, colors)
    base.labels = labels
    levels = [base]
    for r in lods:
        lvl = build_level(src, max(8, int(len(faces) * r)), s, offset, precision, fill_holes, max_hole, agg)
        lvl.labels = np.argmin(((linear_to_srgb(lvl.colors)[:, None, :] - palette[None]) ** 2).sum(-1), axis=1)
        levels.append(lvl)
        _log(log, f"  lod {len(levels) - 1}: {len(lvl.faces)} tris, watertight={lvl.report.watertight}")
    clip_list = []
    if rig_obj is not None:
        for lvl in levels:
            lvl.skin = glb_rig.vertex_skin(rig_obj, src, lvl.points, weight_step)
        if clips:
            clip_list = glb_rig.sample_clips(rig_obj, src, fps, log)
    text = emit_weave(name, levels, palette, src.roughness, src.metallic,
                      smooth if smooth is not None and smooth >= 0 else None, with_object, world, input.name,
                      rig_obj, clip_list)
    out_path.write_text(text)
    for i, l in enumerate(levels):
        if not l.report.watertight:
            warnings.append(f"level {i}: {l.report.boundary_edges} open boundary edges remain ({l.report.holes_left} holes)")
    size = pts.max(0) - pts.min(0)
    _log(log, f"wrote {out_path} ({len(text) / 1024:.0f} KB): mesh.{name}, size {size[0]:.0f} x {size[1]:.0f} x {size[2]:.0f} cm")
    return {
        "output": str(out_path.resolve()), "name": name, "model": f"mesh.{name}", "bytes": len(text),
        "source": {"vertices": int(len(src.points)), "triangles": int(len(src.faces)), "meshes": src.meshes},
        "points": int(len(pts)), "triangles": int(len(faces)), "precision_decimals": base.precision, "size_cm": [round(float(v), 1) for v in size],
        "palette": palette_hex(palette), "watertight": rep.watertight, "boundary_edges": rep.boundary_edges,
        "holes_filled": rep.holes_filled, "holes_left": rep.holes_left, "faces_removed": rep.removed_faces,
        "faces_added": rep.added_faces, "seam_vertices": rep.seam_vertices,
        "lods": [{"lod": i, "triangles": int(len(l.faces)), "points": int(len(l.points)), "watertight": l.report.watertight}
                 for i, l in enumerate(levels) if i],
        "rig": None if rig_obj is None else {
            "bones": [b.name for b in rig_obj.bones],
            "humanoid": sorted(set(b.name for b in rig_obj.bones) & set(glb_rig.CANON)),
            "skin_groups": len(glb_rig.skin_groups(*base.skin)[1]) if base.skin else 0,
            "clips": [{"name": c.name, "model": f"anim.mesh.{name}.{c.name}", "duration_s": round(c.duration, 3),
                       "keys": len(c.times)} for c in clip_list],
        },
        "warnings": warnings,
    }


def inspect_model(input: str | Path) -> dict:
    """Describe a glTF/GLB file (meshes, nodes, materials, textures, bounds) without converting it."""
    g = Gltf(Path(input))
    js = g.json
    meshes = []
    for i, m in enumerate(js.get("meshes", [])):
        verts = tris = 0
        mats, tex, vcol = set(), False, False
        for p in m["primitives"]:
            acc = js["accessors"]
            verts += acc[p["attributes"]["POSITION"]]["count"]
            n = acc[p["indices"]]["count"] if "indices" in p else acc[p["attributes"]["POSITION"]]["count"]
            tris += n // 3 if p.get("mode", 4) == 4 else max(0, n - 2)
            if "material" in p:
                mat = js["materials"][p["material"]]
                mats.add(mat.get("name", f"material{p['material']}"))
                tex |= "baseColorTexture" in mat.get("pbrMetallicRoughness", {})
            vcol |= "COLOR_0" in p["attributes"]
        meshes.append({"index": i, "name": m.get("name", ""), "vertices": verts, "triangles": tris,
                       "materials": sorted(mats), "base_color_texture": tex, "vertex_colors": vcol})
    images = []
    for i, im in enumerate(js.get("images", [])):
        h, w = g.image(i).shape[:2]
        images.append({"index": i, "name": im.get("name", ""), "size": [w, h]})
    nodes = [{"index": i, "name": n.get("name", ""), "mesh": n["mesh"]} for i, n in enumerate(js.get("nodes", [])) if "mesh" in n]
    src = load_source(Path(input))
    lo, hi = src.points.min(0), src.points.max(0)
    junk = re.compile(r"badge|watermark|logo|signature|shadow", re.I)
    suspicious = []
    for n in nodes:
        m = meshes[n["mesh"]]
        if any(junk.search(s) for s in [n["name"], m["name"], *m["materials"]]):
            suspicious.append(f"^{re.escape(n['name'] or m['name'])}$")
    return {
        "file": str(Path(input).resolve()), "generator": js.get("asset", {}).get("generator", ""),
        "meshes": meshes, "mesh_nodes": nodes, "images": images,
        "animations": len(js.get("animations", [])), "skins": len(js.get("skins", [])),
        "total_triangles": sum(m["triangles"] for m in meshes),
        "size_source_units_oriverse_axes": [round(float(v), 4) for v in hi - lo],
        "hints": {
            "exclude_suggestion": "|".join(dict.fromkeys(suspicious)) or None,
            "notes": ([f"Rig found ({sum(len(s['joints']) for s in js.get('skins', []))} joints): exported as bone() + "
                       "skin_weights; recognisable bones get Oriverse humanoid names (rig_names='humanoid')."]
                      if js.get("skins") else ["No rig: static mesh (an agent can add bone() + skin_auto, see SKILL.md)."])
                     + ([f"{len(js['animations'])} animation(s): exported as #clip blocks (clips=false to skip)."]
                        if js.get("animations") else []),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help=".glb or .gltf file")
    ap.add_argument("--inspect", action="store_true", help="print model info as JSON and exit")
    ap.add_argument("-o", "--output", type=Path, help="output .weave (default: <input>.weave)")
    ap.add_argument("-n", "--name", help="mesh name (default: from file name)")
    lvl = ap.add_mutually_exclusive_group()
    lvl.add_argument("--tris", type=int, help="target triangle count after decimation (default 8000)")
    lvl.add_argument("--ratio", type=float, help="target as a fraction of the source triangles, e.g. 0.02")
    lvl.add_argument("--max-verts", type=int, help="target cage point count")
    ap.add_argument("--lods", default="", help="far LOD levels as fractions of the base, e.g. '0.3,0.1' (max 3)")
    ap.add_argument("--colors", type=int, default=12, help="palette size / material slots, max 20 (default 12)")
    sz = ap.add_mutually_exclusive_group()
    sz.add_argument("--height", type=float, help="scale uniformly so the model is this tall (cm)")
    sz.add_argument("--scale", type=float, default=100.0, help="source units -> cm (default 100: metres)")
    ap.add_argument("--up", default="+y", choices=AXES, help="source up axis (glTF: +y)")
    ap.add_argument("--forward", default="+z", choices=AXES, help="source front axis (glTF: +z)")
    ap.add_argument("--face", default="+x", choices=["+x", "-x", "+y", "-y"], help="Oriverse axis the model front should face (default +x, the engine's character facing)")
    ap.add_argument("--mirror", action="store_true", help="mirror left/right (fix handedness if the result looks flipped)")
    ap.add_argument("--exclude", help="regex of node/mesh names to skip (e.g. 'badge|watermark')")
    ap.add_argument("--no-fill", action="store_true", help="do not fill holes (keep open boundaries)")
    ap.add_argument("--max-hole", type=float, default=1.0, help="fill holes up to this perimeter / bbox diagonal (default 1 = all)")
    ap.add_argument("--precision", type=int, help="decimal places for cm coordinates (default: auto from edge length)")
    ap.add_argument("--smooth", type=float, default=50.0, help="smooth_shade angle in degrees; negative = flat facets (default 50)")
    ap.add_argument("--agg", type=float, default=7.0, help="decimation aggressiveness (default 7)")
    ap.add_argument("--weld", type=float, default=1e-5, help="weld tolerance relative to bbox diagonal (default 1e-5)")
    ap.add_argument("--with-object", action="store_true", help="append an `object` that places the mesh")
    ap.add_argument("--world", action="store_true", help="emit a standalone test world (ground + object)")
    rg = ap.add_mutually_exclusive_group()
    rg.add_argument("--rig", dest="rig", action="store_true", default=None, help="export bones + skin weights (default: auto when the file has a skin)")
    rg.add_argument("--no-rig", dest="rig", action="store_false", help="static mesh only")
    ap.add_argument("--rig-names", choices=["humanoid", "keep"], default="humanoid", help="rename bones to Oriverse humanoid names (default) or keep")
    ap.add_argument("--no-clips", action="store_true", help="do not export glTF animations as #clip blocks")
    ap.add_argument("--weight-step", type=float, default=0.05, help="skin weight quantisation (default 0.05)")
    ap.add_argument("--fps", type=float, default=30.0, help="animation sampling rate, max 64 keys per clip (default 30)")
    ap.add_argument("--json", action="store_true", help="print the result summary as JSON on stdout")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.inspect:
        print(json.dumps(inspect_model(a.input), indent=1))
        return 0
    log = None if a.quiet else (lambda m: print(m, file=sys.stderr))
    try:
        res = convert(a.input, a.output, name=a.name, tris=a.tris, ratio=a.ratio, max_verts=a.max_verts,
                      colors=a.colors, height=a.height, scale=a.scale, up=a.up, forward=a.forward, face=a.face,
                      mirror=a.mirror, exclude=a.exclude, precision=a.precision, smooth=a.smooth, agg=a.agg,
                      weld_tol=a.weld, fill_holes=not a.no_fill, max_hole=a.max_hole,
                      lods=[float(x) for x in a.lods.split(",") if x.strip()],
                      with_object=a.with_object, world=a.world, rig=a.rig, rig_names=a.rig_names,
                      clips=not a.no_clips, weight_step=a.weight_step, fps=a.fps, log=log)
    except ValueError as e:
        ap.error(str(e))
    if a.json:
        print(json.dumps(res, indent=1))
    for w in res["warnings"]:
        print(f"warning: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
