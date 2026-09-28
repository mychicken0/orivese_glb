"""Run: .venv/bin/python -m unittest discover -s tests -v"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import glb2ori  # noqa: E402
import glb_rig  # noqa: E402
import make_sample  # noqa: E402


def compose(euler_deg: np.ndarray) -> np.ndarray:
    """Oriverse key convention: R = Rz @ Ry @ Rx (measured with inspect_rig)."""
    rx, ry, rz = np.radians(euler_deg)
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class ConvertTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.glb = make_sample.write(cls.tmp / "sample_rigged.glb")
        cls.res = glb2ori.convert(cls.glb, cls.tmp / "sample.weave", name="Sample", tris=400, height=100,
                                  exclude="^badge$", lods=[0.5], world=True)
        cls.text = (cls.tmp / "sample.weave").read_text()

    def test_mesh_is_watertight_and_scaled(self):
        self.assertTrue(self.res["watertight"])
        self.assertAlmostEqual(self.res["size_cm"][2], 100, delta=0.5)
        self.assertIn("#mesh Sample\n", self.text)
        self.assertIn("#mesh Sample lod: 1", self.text)
        self.assertNotIn("badge", " ".join(self.res["source"]["meshes"][:1]))

    def test_rig_exported_with_humanoid_names(self):
        rig = self.res["rig"]
        self.assertEqual(rig["bones"], ["Root", "Spine"])
        self.assertIn('bone("Root", pos:', self.text)
        self.assertIn('bone("Spine", pos:', self.text)
        self.assertIn("skin_weights(g, selection: verts(", self.text)
        self.assertEqual([c["name"] for c in rig["clips"]], ["Bend"])
        self.assertIn("#clip Bend on Sample", self.text)

    def test_clip_reproduces_gltf_joint_motion(self):
        src = glb2ori.load_source(self.glb, "^badge$")
        lo, hi = src.points.min(0), src.points.max(0)
        s = 100 / (hi[2] - lo[2])
        off = np.array([(hi[0] + lo[0]) / 2 * s, (hi[1] + lo[1]) / 2 * s, lo[2] * s])
        rig = glb_rig.build_rig(src, s, off)
        clip = glb_rig.sample_clips(rig, src)[0]
        rest = np.array([b.pos for b in rig.bones])
        # the spine joint rotates 60 deg about glTF X -> Oriverse Y after the frame change; an
        # end-effector 50 cm above it must follow the glTF arc
        for ti, t in enumerate(clip.times):
            q_root = compose(clip.euler[ti, 0])
            q_sp = q_root @ compose(clip.euler[ti, 1])
            p_sp = rest[0] + clip.root_offset[ti] + q_root @ (rest[1] - rest[0] + clip.trans[ti, 1])
            tip = p_sp + q_sp @ np.array([0, 0, 50.0])
            a = np.radians(60) * t
            expect = rest[1] + 50 * np.array([np.sin(a), 0, np.cos(a)])  # glTF +X bend = +Z(fwd)->+X
            np.testing.assert_allclose(tip, expect, atol=0.2)

    def test_euler_roundtrip(self):
        for e in ([30, 40, 50], [-120, 10, 170], [5, -80, -45]):
            np.testing.assert_allclose(compose(glb_rig.euler_zyx(compose(np.array(e, float)))), compose(np.array(e, float)), atol=1e-9)


class RepairTest(unittest.TestCase):
    def test_fill_holes_closes_open_cylinder(self):
        pts, faces = make_sample.cylinder()
        keep = np.ones(len(faces), bool)
        keep[-32:] = False  # remove both caps
        p = pts.astype(float) * 100
        cols = np.ones((keep.sum(), 3))
        _, f, _, rep = glb2ori.clean_mesh(p, faces[keep].astype(np.int64), cols, 1e-6, True, 1.0, 2)
        self.assertTrue(rep.watertight)
        self.assertEqual(rep.holes_filled, 2)


if __name__ == "__main__":
    unittest.main()
