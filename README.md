# glb2ori: GLB/glTF → native Oriverse `#mesh`

Convert any `.glb` / `.gltf` model into **editable Oriverse Weave source**: decimated, repaired to be
watertight, coloured, and (when the file has one) **rigged and animated**, so you don't have to model
it by hand in the mesh DSL. Built to be driven by AI agents over **MCP**, and usable as a CLI too.

```
model.glb ──► inspect ──► convert (decimate · repair · palette · rig · clips) ──► model.weave ──► Oriverse editor
```

## Install (one command)

```bash
git clone git@github.com:mychicken0/orivese_glb.git glb2ori && cd glb2ori && ./install.sh
```

`install.sh` creates `.venv`, installs pinned deps, runs the self-test and prints the MCP config
with absolute paths. Requires Python 3.10+.

## Use from an AI agent (MCP)

Add the server to your client (paths printed by `install.sh`):

```json
{
  "mcpServers": {
    "glb2ori": {
      "command": "/ABS/PATH/glb2ori/.venv/bin/python",
      "args": ["/ABS/PATH/glb2ori/glb2ori_mcp.py"]
    }
  }
}
```

Claude Code: `claude mcp add glb2ori -- /ABS/PATH/.venv/bin/python /ABS/PATH/glb2ori_mcp.py`

| Tool | What it does |
|---|---|
| `inspect_model(path)` | Meshes, triangle counts, textures, rig/animation info, size, and an `exclude` suggestion for watermark/badge/shadow meshes |
| `convert_model(path, ...)` | Writes the `.weave` file and returns stats: `triangles`, `size_cm`, `watertight`, `palette`, `rig.bones`, `rig.clips`, `warnings` |
| `read_weave(path, offset, max_chars)` | Pages through a generated file (these are 50 KB to 1 MB) |
| `push_to_oriverse(weave_path, agent, submit)` | Stages the file in the Oriverse editor through its local MCP (`http://127.0.0.1:39837/mcp`), validates it (compile + 60-frame sim), and with `submit=true` shows the user a **[Load]** button |

**Recommended agent workflow**

1. `inspect_model`: note `hints.exclude_suggestion`, `skins`, `animations`.
2. `convert_model(path, name="Hero", tris=8000, height_cm=170, exclude=<hint>, lods=[0.3])`.
   - `tris`: props 2k–8k, characters 8k–20k, cap ~64k (or `ratio` / `max_verts`).
   - Check `watertight == true` and read `warnings`.
3. `push_to_oriverse(output, agent=<id from Oriverse list_editors>, submit=true)`.
4. Verify with Oriverse `preview_mesh` / `playtest`.

The agent skill with more detail (including how to rig models that have no skeleton) lives in
[`.devin/skills/glb-to-oriverse/SKILL.md`](.devin/skills/glb-to-oriverse/SKILL.md). Copy it into your
agent's skills folder if your client supports skills.

## Use from the CLI

```bash
.venv/bin/python glb2ori.py model.glb --inspect
.venv/bin/python glb2ori.py model.glb -o model.weave -n Hero --tris 8000 --height 170 \
    --colors 16 --exclude '^SHADOW$' --lods 0.3,0.1 --with-object --json
.venv/bin/python ori_push.py model.weave --agent <oriverse-session-uuid> --submit
```

| Flag | Meaning |
|---|---|
| `--tris N` / `--ratio R` / `--max-verts N` | Target detail (QEM decimation per part; part seams stay fixed) |
| `--lods 0.3,0.1` | Extra far-display LODs (`#mesh Name lod: N`) |
| `--height CM` / `--scale K` | Uniform size (default: metres → cm) |
| `--colors N` | Palette size = material slots (1–20) |
| `--exclude REGEX` | Skip nodes/meshes by name |
| `--no-fill`, `--max-hole F` | Keep intentional openings (default fills every hole) |
| `--rig` / `--no-rig` | Bones + skin weights (default: auto when the file has a skin) |
| `--rig-names humanoid\|keep` | Rename bones to Oriverse's canonical humanoid names (default) |
| `--no-clips`, `--fps 30`, `--weight-step 0.05` | Animation export / sampling / skin-weight quantisation |
| `--up/--forward/--face/--mirror` | Axis fixes (glTF +Y up, +Z front → Oriverse Z up, front faces +X) |
| `--smooth DEG` (`-1` = flat), `--precision N` | Shading angle, coordinate decimals (default auto) |
| `--world` / `--with-object` | Emit a standalone test world / a placing `object` |

## What you get

```weave
#mesh Hero
#texture HeroC0 color=#fa760c roughness=0.8 metallic=0
bone("Root", pos: (10.7, 2.7, 71.4));
bone("Spine", pos: (15.1, 2.7, 78.8), parent: "Root");
let cage = mesh_cage(points: [...], faces: [...], material: "HeroC0");
var g = cage;
g = paint(g, material: "HeroC1", selection: faces("140-389"));
g = skin_weights(g, selection: verts("673-696"), bones: "ThighL KneeL", weights: [0.6, 0.4]);
out.geo = smooth_shade(g, angle: 50);
#end
#clip walk on Hero
duration: 1.2
loop: true
key(0, ThighR: (0, -25, 0), ..., ease: "linear")
#end
```

- **Watertight repair**: sliver faces are edge-collapsed, zero-volume fins from decimation are
  cancelled, holes are filled (ear clipping, centroid-fan fallback), non-manifold or wrongly wound
  edges become invisible coincident seams instead of deleted faces, and open sheets (ribbons, cloth)
  are left open.
- **Colours**: base-colour texture × factor × vertex colour, baked per face and k-means clustered.
  `mesh_cage` has no UV input, so fine texture detail becomes per-face colour (raise `--tris` for more).
- **Rig**: glTF skin → `bone()` (≤48, low-influence leaves pruned) + 4-influence `skin_weights`;
  animations → `#clip` (≤64 linear keys, Euler Z·Y·X, measured against Oriverse's `inspect_rig`,
  joint error < 0.01 cm). Humanoid names (`Root, Spine, Torso, Neck, Head, ScapulaR/L, UpperArmR/L,
  ForearmR/L, HandR/L, ThighR/L, ShinR/L, FootR/L, ToeR/L`) let Oriverse retarget its canonical
  idle/walk/run/jumpHover. Play clips with `obj.PlayAnimationOnRepeat(Ori.anim.mesh.Hero.walk)`.

## Limits

- ~9 indices per triangle count against Oriverse's 600k index cap → ≤ ~64k triangles per mesh.
- Scale animation and morph targets are ignored. Colliders don't follow bones (Oriverse rule).
- Default object collider is the AABB; `collision = mesh` wants ≤ ~1k vertices.
- `preview_mesh` in Oriverse can't handle meshes much above ~30k triangles plus LODs; use `playtest` for those.

## Development

```bash
.venv/bin/python -m unittest discover -s tests -v    # synthetic rigged sample, no assets needed
.venv/bin/python tests/make_sample.py examples/sample_rigged.glb
```

Files: `glb2ori.py` (loader, decimation, repair, palette, Weave emit, CLI), `glb_rig.py` (bones,
skin weights, clips), `glb2ori_mcp.py` (MCP server), `ori_push.py` (Oriverse MCP client),
`AGENTS.md` (notes for coding agents working on this repo).

---

**ภาษาไทยสั้น ๆ:** ติดตั้งด้วย `./install.sh` แล้วใส่ config MCP ที่ script พิมพ์ออกมา AI จะเรียก
`inspect_model` → `convert_model` → `push_to_oriverse` ได้เอง ได้ `#mesh` แบบ native ที่ลด poly
ได้ ซ่อมรูอัตโนมัติ และมี rig/animation ถ้าไฟล์ GLB มีมาให้
