"""Write a small synthetic rigged + animated GLB (no third-party assets) for tests and demos.

A capped cylinder (1 m tall, red) skinned to two joints; the upper joint bends 60 degrees
about X over one second. A tiny "badge" quad is included to exercise --exclude.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np


def cylinder(radius=0.1, height=1.0, seg=16, rings=10):
    pts, faces = [], []
    for r in range(rings + 1):
        y = height * r / rings
        for s in range(seg):
            a = 2 * np.pi * s / seg
            pts.append((radius * np.cos(a), y, radius * np.sin(a)))
    for r in range(rings):
        for s in range(seg):
            a, b = r * seg + s, r * seg + (s + 1) % seg
            c, d = a + seg, b + seg
            faces += [(a, c, b), (b, c, d)]
    bot, top = len(pts), len(pts) + 1
    pts += [(0, 0, 0), (0, height, 0)]
    for s in range(seg):
        faces.append((bot, s, (s + 1) % seg))
        faces.append((top, rings * seg + (s + 1) % seg, rings * seg + s))
    return np.array(pts, np.float32), np.array(faces, np.uint32)


def write(path: Path) -> Path:
    pts, faces = cylinder()
    y = pts[:, 1]
    w1 = np.clip((y - 0.3) / 0.4, 0, 1)  # blend zone 0.3..0.7 m around the knee at 0.5 m
    joints = np.tile(np.array([0, 1, 0, 0], np.uint16), (len(pts), 1))
    weights = np.c_[1 - w1, w1, np.zeros(len(pts)), np.zeros(len(pts))].astype(np.float32)
    ibm = np.stack([np.eye(4), np.eye(4)]).astype(np.float32)
    ibm[1][1, 3] = -0.5  # inverse of translate(0, 0.5, 0)
    times = np.linspace(0, 1, 11, dtype=np.float32)
    ang = np.radians(60) * times / 2
    rots = np.c_[np.sin(ang), np.zeros(11), np.zeros(11), np.cos(ang)].astype(np.float32)
    badge = np.array([[0.2, 1.1, 0], [0.4, 1.1, 0], [0.4, 1.2, 0], [0.2, 1.2, 0]], np.float32)
    badge_idx = np.array([0, 1, 2, 0, 2, 3], np.uint32)

    blobs, views, accessors = [], [], []

    def add(arr: np.ndarray, comp: int, typ: str, minmax: bool = False) -> int:
        off = sum(len(b) for b in blobs)
        data = arr.tobytes()
        data += b"\0" * (-len(data) % 4)
        blobs.append(data)
        views.append({"buffer": 0, "byteOffset": off, "byteLength": arr.nbytes})
        acc = {"bufferView": len(views) - 1, "componentType": comp, "count": int(len(arr)), "type": typ}
        if minmax:
            acc["min"], acc["max"] = arr.min(0).tolist(), arr.max(0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    a_pos = add(pts, 5126, "VEC3", True)
    a_idx = add(faces.ravel(), 5125, "SCALAR")
    a_j = add(joints, 5123, "VEC4")
    a_w = add(weights, 5126, "VEC4")
    a_ibm = add(ibm.transpose(0, 2, 1).reshape(-1, 16), 5126, "MAT4")
    a_t = add(times, 5126, "SCALAR", True)
    a_r = add(rots, 5126, "VEC4")
    a_bp = add(badge, 5126, "VEC3", True)
    a_bi = add(badge_idx, 5125, "SCALAR")
    gltf = {
        "asset": {"version": "2.0", "generator": "glb2ori tests/make_sample.py"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1, 3]}],
        "nodes": [
            {"name": "Body", "mesh": 0, "skin": 0},
            {"name": "hips", "children": [2]},
            {"name": "spine", "translation": [0, 0.5, 0]},
            {"name": "badge", "mesh": 1},
        ],
        "meshes": [
            {"name": "Body", "primitives": [{"attributes": {"POSITION": a_pos, "JOINTS_0": a_j, "WEIGHTS_0": a_w},
                                              "indices": a_idx, "material": 0}]},
            {"name": "badge", "primitives": [{"attributes": {"POSITION": a_bp}, "indices": a_bi, "material": 0}]},
        ],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.1, 0.1, 1], "metallicFactor": 0, "roughnessFactor": 0.7}}],
        "skins": [{"joints": [1, 2], "inverseBindMatrices": a_ibm}],
        "animations": [{"name": "Bend", "samplers": [{"input": a_t, "output": a_r, "interpolation": "LINEAR"}],
                        "channels": [{"sampler": 0, "target": {"node": 2, "path": "rotation"}}]}],
        "buffers": [{"byteLength": sum(len(b) for b in blobs)}],
        "bufferViews": views,
        "accessors": accessors,
    }
    js = json.dumps(gltf).encode()
    js += b" " * (-len(js) % 4)
    bin_ = b"".join(blobs)
    out = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(bin_))
    out += struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(bin_), 0x004E4942) + bin_
    path.write_bytes(out)
    return path


if __name__ == "__main__":
    print(write(Path(sys.argv[1] if len(sys.argv) > 1 else "sample_rigged.glb")))
