#!/usr/bin/env python3
"""MCP server (stdio) exposing the GLB -> Oriverse native #mesh converter to any AI agent.

Register it in an MCP client, e.g.:
    {"mcpServers": {"glb2ori": {"command": "/path/.venv/bin/python", "args": ["/path/glb2ori_mcp.py"]}}}
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from mcp.server.fastmcp import FastMCP

import glb2ori
import ori_push

Axis = Literal["+x", "-x", "+y", "-y", "+z", "-z"]
Facing = Literal["+x", "-x", "+y", "-y"]

mcp = FastMCP(
    "glb2ori",
    instructions=(
        "Convert .glb/.gltf 3D models into native, editable Oriverse `#mesh` Weave source "
        "(mesh_cage + per-face colour palette), automatically decimated and repaired to be watertight. "
        "Workflow: inspect_model -> convert_model (pick size with tris / ratio / max_verts, set height_cm, "
        "exclude watermark meshes) -> push_to_oriverse to validate in the Oriverse editor. "
        "Use absolute file paths. Output is a static mesh (rigs/animations are ignored)."
    ),
)


def _path(p: str) -> Path:
    path = Path(os.path.expanduser(p))
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    return path


@mcp.tool()
def inspect_model(path: str) -> dict:
    """Describe a .glb/.gltf file before converting: meshes (names, vertex/triangle counts,
    materials, textures), images, rig/animation presence, overall size in source units
    (already in Oriverse axes: x forward, y side, z up) and a suggested `exclude` regex for
    watermark/badge meshes."""
    return glb2ori.inspect_model(_path(path))


@mcp.tool()
def convert_model(
    path: str,
    output: str | None = None,
    name: str | None = None,
    tris: int | None = None,
    ratio: float | None = None,
    max_verts: int | None = None,
    colors: int = 12,
    height_cm: float | None = None,
    scale: float = 100.0,
    exclude: str | None = None,
    lods: list[float] | None = None,
    fill_holes: bool = True,
    max_hole: float = 1.0,
    smooth_angle: float = 50.0,
    precision: int | None = None,
    up: Axis = "+y",
    forward: Axis = "+z",
    face: Facing = "+x",
    mirror: bool = False,
    with_object: bool = False,
    world: bool = False,
    rig: bool | None = None,
    rig_names: Literal["humanoid", "keep"] = "humanoid",
    clips: bool = True,
    weight_step: float = 0.05,
    fps: float = 30.0,
) -> dict:
    """Convert a .glb/.gltf model into an Oriverse `#mesh` .weave file.

    Size (give at most one; default 8000 triangles): `tris` = target triangles, `ratio` = fraction
    of the source triangles (e.g. 0.01), `max_verts` = target cage points. Max ~64000 triangles.
    `colors` = palette size / material slots (1..20). `height_cm` scales the model to that height;
    otherwise `scale` converts source units to cm (100 = metres). `exclude` is a regex of node/mesh
    names to drop (e.g. "badge|watermark"). `lods` adds up to 3 far LOD levels as fractions of the
    base triangle count (e.g. [0.3, 0.1]). `fill_holes` closes holes whose perimeter is at most
    `max_hole` x bbox diagonal (1.0 = all -> watertight). `smooth_angle` < 0 gives flat facets.
    `precision` = decimals for cm coordinates (default auto). Axes: source `up`/`forward`
    (glTF +y/+z); the model front ends up facing Oriverse `face` (+x = engine character facing).
    `with_object` appends an object placing `mesh.<name>`; `world` emits a standalone test world.
    Rig: `rig` (default auto when the file has a skin) exports `bone(...)` + `skin_weights(...)`;
    `rig_names="humanoid"` renames recognisable bones (Root/Spine/Torso/Neck/Head/ThighR/...) so
    Oriverse can auto-play canonical idle/walk/run on it; `clips` exports glTF animations as
    `#clip` blocks (play with `obj.PlayAnimationOnRepeat(Ori.anim.mesh.<Name>.<clip>)`).
    Returns stats (triangles, points, size_cm, palette, watertight, holes filled, output path)."""
    src = _path(path)
    log: list[str] = []
    res = glb2ori.convert(
        src, Path(output).expanduser() if output else None, name=name, tris=tris, ratio=ratio,
        max_verts=max_verts, colors=colors, height=height_cm, scale=scale, up=up, forward=forward,
        face=face, mirror=mirror, exclude=exclude, precision=precision,
        smooth=smooth_angle if smooth_angle >= 0 else None, fill_holes=fill_holes, max_hole=max_hole,
        lods=lods or [], with_object=with_object, world=world, rig=rig, rig_names=rig_names, clips=clips,
        weight_step=weight_step, fps=fps, log=log.append)
    res["log"] = log
    return res


@mcp.tool()
def read_weave(path: str, offset: int = 0, max_chars: int = 20000) -> dict:
    """Read part of a generated .weave file (e.g. to paste into another tool). Large meshes are
    hundreds of KB, so page with `offset`; prefer push_to_oriverse for loading into the editor."""
    text = _path(path).read_text()
    chunk = text[offset: offset + max_chars]
    return {"total_chars": len(text), "offset": offset, "next_offset": offset + len(chunk) if offset + len(chunk) < len(text) else None,
            "text": chunk}


@mcp.tool()
def push_to_oriverse(weave_path: str, agent: str | None = None, submit: bool = False,
                     oriverse_url: str = ori_push.DEFAULT_URL) -> dict:
    """Stage a .weave file as the Oriverse draft world through the Oriverse MCP server
    (default http://127.0.0.1:39837/mcp): compiles and simulates it and returns the validation
    report. `agent` is the Oriverse editing-session id (reuse the one from Oriverse's
    list_editors so the draft lands in the same session; omitted = new session).
    `submit=True` also proposes it to the connected editor as a [Load] button. Note: staging
    replaces that session's current draft."""
    return ori_push.push(_path(weave_path), agent, submit, oriverse_url)


if __name__ == "__main__":
    mcp.run()
