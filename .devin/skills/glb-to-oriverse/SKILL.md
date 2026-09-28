---
name: glb-to-oriverse
description: "Convert .glb/.gltf 3D models (static or rigged/animated) into native, editable Oriverse #mesh Weave source - decimated, watertight, colour palette, bones, skin weights and #clip animations - and load them into the Oriverse editor. Also covers rigging models that have no skeleton. Use when a user wants a GLB/glTF model in Oriverse without hand-modelling it."
---

# GLB -> Oriverse native #mesh

Repo root = the folder containing `glb2ori.py` (install: `./install.sh`, venv at `.venv/`).
Prefer the `glb2ori` MCP tools when connected; otherwise the CLI (`.venv/bin/python glb2ori.py ...`).

## 1. Inspect
`inspect_model(path)` / `glb2ori.py model.glb --inspect`
- `hints.exclude_suggestion`: watermark / badge / shadow-plane meshes to drop.
- `skins` > 0: the file has a rig -> it is exported automatically. `animations` > 0: clips too.

## 2. Convert
`convert_model(path, output, name, tris|ratio|max_verts, height_cm, colors, exclude, lods)`
CLI: `glb2ori.py model.glb -o out.weave -n Name --tris 8000 --height 150 --colors 16 --exclude '^SHADOW$' --json`
- Detail: props 2k-8k tris, characters 8k-20k, hard cap ~64k. `lods: [0.3, 0.1]` for far LODs.
- `height_cm` = real-world height (player is 180 cm). Model front ends up facing +X (engine facing).
- Check the result: `watertight` should be true; read `warnings`.
- Rig result (`rig`): `bones`, `humanoid` (bones renamed to Oriverse canonical names), `clips`
  (`anim.mesh.<Name>.<clip>`). Humanoid-named rigs without own clips auto-play the engine's
  canonical idle/walk/run/jumpHover; use `clips=false` (`--no-clips`) if the file's clips are just poses.

## 3. Load into Oriverse
`push_to_oriverse(weave_path, agent=<agent id from Oriverse list_editors>, submit=true)`
- Staging validates (compile + 60-frame sim); `submit` shows the user a [Load] button.
- Staging replaces that session's draft. To add to an existing world, paste only the `#mesh`
  (+ `#clip`) blocks and place `object X { model = "mesh.Name" position = (x, y, 0) }`.
- Play a clip from Weave: `obj.PlayAnimationOnRepeat(Ori.anim.mesh.Name.clip)`.

## 4. Verify
Oriverse `preview_mesh {name, clip}` shows the clip as a strip; `inspect_rig {target, query: {clip, times_s}}`
returns joint positions; `playtest` with `views` for the final render.

## Rigging a model that has NO skeleton (AI step)
The converter cannot invent a rig; do it in the generated #mesh with the Oriverse DSL
(read Oriverse `mesh_dsl` section `bones_and_clips` first):
1. Convert statically, then read joint locations from `preview_mesh` front/right views (cm, Z up, +X front, left = +Y).
2. Insert top-level `bone("Root", pos: (x,y,z));` then children with `parent:`. Use the canonical
   humanoid names (Root, Spine, Torso, Neck, Head, ScapulaR/L, UpperArmR/L, ForearmR/L, HandR/L,
   ThighR/L, ShinR/L, FootR/L, ToeR/L; R = -Y side) to get free idle/walk/run.
3. After the paint lines add `g = skin_auto(g, radius: <cm>);` (envelope skinning) before `out.geo`.
   Envelopes run along bone segments, so give every extremity a leaf bone (head top, finger/toe
   tips, tail tip); `E_SKIN_UNCOVERED` lists missed vertices -> add a leaf bone or raise `radius`.
4. Validate, then check a canonical clip on it (`preview_mesh {name, clip: "walk"}`) and fix joint
   placement / radius until limbs bend cleanly. Author extra verbs as `#clip Name on Mesh` blocks.

## Fixes
- Faces the wrong way: `forward`/`face`; lying down: `up`; mirrored: `mirror`.
- Intentional openings got closed: `fill_holes=false` or `max_hole=0.1`.
- Too faceted: raise `tris`; blocky colours: raise `colors` (max 20); file too big: lower `tris` or `weight_step=0.1`.
