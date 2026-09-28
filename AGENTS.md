# Notes for coding agents working on this repo

Files: `glb2ori.py` (library + CLI), `glb_rig.py` (rig/clips), `glb2ori_mcp.py` (stdio MCP server),
`ori_push.py` (stage into Oriverse via its MCP), `tests/` (synthetic sample + unittest),
agent skill in `.devin/skills/glb-to-oriverse/`.

## Setup / verify
- `./install.sh` (venv + pinned deps + tests) or `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
- Tests: `.venv/bin/python -m unittest discover -s tests -v`
- Validate in Oriverse: `.venv/bin/python ori_push.py file.weave --agent <uuid>` (Oriverse MCP at http://127.0.0.1:39837/mcp).
- Dependency policy: pin versions published > 7 days ago (see requirements.txt comments).

## Oriverse `mesh_cage` rules learned (validation errors seen)
- No unused points (`E_INVALID_POLYGON ... unused point`), no zero-area faces.
- Every edge max 2 faces (`edge ... is nonmanifold`) and opposite winding on shared edges (`inconsistent winding`).
- Coincident duplicate points in separate fans are accepted; `//` full-line comments are accepted.
- CCW winding (right-handed, Z up) = front face. `paint(g, material:, selection: faces("a-b"))` works on the cage.
- `mesh_cage` counts ~9 indices per triangle; defaults 50k verts / 90k indices, `budget(verts:, indices:)` up to 200k / 600k. Max 20 material slots.
- `#mesh Name lod: N` blocks without `#texture`/`bone` lines validate (they reuse the base slots and rig).
- No per-corner UV input on `mesh_cage`, so textures are baked to a per-face colour palette.
- `preview_mesh` fails above ~16 MB snapshot (30k tris + LODs); validate those with `ori_push.py`/playtest.
- `submit_pending_world` requires a `summary` argument.

## Rig conventions (measured with inspect_rig on a test rig)
- Bone rest frames are identity; child world rot = parent world * local; key Euler (rx,ry,rz) = Rz*Ry*Rx.
- `translate: (Bone: v)` offsets a joint in its parent's posed frame; root `offset:` is world translation.
- Characters face +X, left = +Y (R bones at -Y). Max 48 bones, 64 keys per clip; keys use `ease: "linear"`.
- `skin_weights(g, selection: verts("a-b"), bones: "A B", weights: [..])` works on mesh_cage; points are
  sorted so each weight mix is one contiguous range. `skin_auto(g, radius:)` needs leaf bones at extremities.
- Hornet sample check: Oriverse joint positions matched glTF skinning within 0.01 cm, rotations within 0.02 deg.

## Repair pipeline (why it is watertight)
- Rounding to the emitted precision happens BEFORE cleaning; precision auto = edge median / 40.
- Multi-part models: each glTF primitive decimated separately (proportional budget, fixed borders), re-welded.
- Slivers: edge-collapse (never delete -> no hole). Coincident face pairs from decimation cancel.
- Holes: ear clipping in best-fit plane, centroid fan fallback; open-sheet outlines are skipped.
- Non-manifold / mis-wound edges: split vertices per good-edge fan (coincident seams), faces kept.
