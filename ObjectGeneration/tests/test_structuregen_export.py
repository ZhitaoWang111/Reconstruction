"""Stage 4: the structure handoff written from a real agent run reads back as it says.

Skipped when Blender or a real (non-fake) agent run is missing.
"""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from structuregen import config  # noqa: E402

RUN = (config.OUTPUT_ROOT / config.SCENE / "structuregen/latest")


def _fake_run():
    """A scripted (no model, no cost) agent run on the latest white model, made once per test run."""
    from structuregen.agent import run as agent
    out = Path(tempfile.mkdtemp(prefix="sg_fake_"))
    agent.main([str(RUN.resolve()), "--fake", "--out", str(out / "agent_fake")])
    return out / "agent_fake"


READY = config.BLENDER.is_file() and (RUN / "evidence/task_card.json").is_file()


@unittest.skipUnless(READY, "needs Blender and a committed agent run")
class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from structuregen.export import handoff as export
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "structure_handoff"
        cls.agent = _fake_run()
        export.main([str(cls.agent), "--out", str(cls.out)])
        cls.manifest = json.loads((cls.out / "manifest.json").read_text())
        cls.structure = json.loads((cls.out / "structure.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
        shutil.rmtree(cls.agent.parent, ignore_errors=True)

    def test_files_and_hashes(self):
        from structuregen.export.handoff import sha256
        names = sorted(p.name for p in self.out.iterdir())
        self.assertTrue({"README.md", "manifest.json", "structure.json", "structure.usdc"} <= set(names), names)
        self.assertEqual(self.manifest["files"]["usd_sha256"], sha256(self.out / "structure.usdc"))
        for tex in self.manifest["files"]["textures"]:
            self.assertTrue((self.out / tex).is_file(), tex)
        walls = json.loads((Path(json.loads((RUN / "plan.json").read_text())["handoff_dir"]) / "walls.json").read_text())
        self.assertEqual(self.manifest["coordinate_frame"], walls["coordinate_frame"])      # default: the handoff's own

    def test_usd_static_and_coloured(self):
        from pxr import Usd, UsdGeom, UsdPhysics
        stage = Usd.Stage.Open(str(self.out / "structure.usdc"))
        self.assertEqual(str(stage.GetDefaultPrim().GetPath()), "/Structure")
        self.assertEqual(sorted(p.GetName() for p in stage.GetPrimAtPath("/Structure").GetChildren()),
                         ["Floors", "Walls", "_materials"])
        meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
        self.assertTrue(meshes)
        for p in meshes:
            self.assertTrue(p.HasAPI(UsdPhysics.CollisionAPI), p.GetPath())
            self.assertFalse(p.HasAPI(UsdPhysics.RigidBodyAPI), p.GetPath())
        for p in stage.Traverse():
            self.assertFalse(any(a.GetName().startswith("userProperties:blender:") for a in p.GetAttributes()))
            self.assertFalse(p.IsA(UsdGeom.Camera))

    def test_floors_sit_at_the_wall_base(self):
        for f in self.structure["floors"]:
            self.assertAlmostEqual(f["top_z"], self.structure["segments"][0]["base_z"], places=4)
            self.assertIn("top", f["faces"])
            self.assertEqual(len(f["faces"]["top"]["rgb"]), 3)

    def test_every_face_has_a_colour_and_basis(self):
        card = json.loads((RUN / "evidence/task_card.json").read_text())
        framed = {c["id"]: bool(c["frames"]) for c in card["segments"]}
        for s in self.structure["segments"]:
            # a wall no key frame shows is painted from its neighbours: inferred, not photo
            self.assertEqual(s["faces"]["front"]["basis"], "photo" if framed[s["id"]] else "inferred", s["id"])
            for zone, f in s["faces"].items():
                self.assertEqual(len(f["rgb"]), 3, (s["id"], zone))
                self.assertIn(f["basis"], ("photo", "inferred"), (s["id"], zone))

    def test_frame(self):
        """Native scannet_scan walls only get z back to the annotated floor; amb3r_floor walls are moved
        upright onto gt/'s floor, within 2 cm of the GT handoffs' full matrix."""
        import numpy as np
        x = self.manifest["frame"]
        if self.manifest["coordinate_frame"] == "scannet_scan" and "from_walls_handoff" in x:
            fz = x["from_walls_handoff"]["floor_z"]
            for s in self.structure["segments"]:
                self.assertAlmostEqual(s["base_z"], fz, places=4)
            return
        from structuregen.export import handoff as export
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "scan"
            export.main([str(self.agent), "--out", str(out), "--frame", "scannet_scan"])
            m = json.loads((out / "manifest.json").read_text())
            s = json.loads((out / "structure.json").read_text())
        mv = m["frame"]["from_native"]
        self.assertLess(mv["residual_vs_full_matrix_m"]["horizontal_max"], 0.02)
        T = np.array(mv["matrix_column_vectors"])
        for a_, b_ in zip(self.structure["segments"], s["segments"]):
            self.assertAlmostEqual(b_["base_z"], mv["floor_z"], places=4)
            for p, q in zip(a_["front_floor"], b_["front_floor"]):
                np.testing.assert_allclose(T[:2, :2] @ p + T[:2, 3], q, atol=1e-4)

    def test_refuses_uncommitted_run(self):
        from structuregen.export import handoff as export
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "agent_x"
            fake.mkdir()
            (fake / "report.json").write_text(json.dumps({"status": "stopped: budget", "walls": []}))
            with self.assertRaises(SystemExit):
                export.main([str(fake), "--out", str(Path(tmp) / "out")])


if __name__ == "__main__":
    unittest.main()
