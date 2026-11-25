"""Verify run isolation and preservation of stage-4 visual assets."""
import os
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import trimesh
from pxr import Usd, UsdPhysics

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from objgen.layout import STAGES, ASSETS_ROOT, object_output_parent, create_run, resolve_physics_input


class ArtifactLayoutTests(unittest.TestCase):
    def test_runs_are_distinct_and_named_after_rgb(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, started = create_run(tmp, "/source/xuebi.jpg")
            second, _ = create_run(tmp, "/source/xuebi.jpg")
            self.assertNotEqual(first, second)
            self.assertEqual(first.parent.name, "xuebi")
            self.assertRegex(first.name, r"^\d{8}_\d{6}_\d{6}$")
            self.assertTrue(started.endswith("+08:00"))
            self.assertEqual({p.name for p in first.iterdir()}, set(STAGES))

    def test_asset_categories_and_multiview_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            for image, expected in [("Beverages_And_snacks/1.jpg", "Beverages_And_snacks/1"),
                                    ("Beverages_And_snacks/1/0.jpg", "Beverages_And_snacks/1"),
                                    ("SKU/0/0.jpg", "SKU/0"),
                                    ("SKU/0/90.jpg", "SKU/0")]:
                self.assertEqual(object_output_parent(tmp, ASSETS_ROOT/image), Path(tmp)/expected)
            run, _ = create_run(tmp, ASSETS_ROOT/"SKU/0/0.jpg")
            self.assertEqual(run.parent, Path(tmp)/"SKU/0")
            self.assertEqual({p.name for p in run.iterdir()}, set(STAGES))

    def test_physics_cannot_overwrite_source(self):
        with self.assertRaises(ValueError):
            resolve_physics_input("/tmp/visual.usda", "/tmp/visual.usda")

    def test_conversion_and_physics_keep_separate_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, _ = create_run(tmp, "/source/box.png")
            mesh = trimesh.creation.box()
            mesh.visual.vertex_colors = np.tile([80, 140, 200, 255], (len(mesh.vertices), 1))
            mesh.export(run / "sam3d/mesh.glb")
            for script, args in [
                ("mesh_to_usd", ["--height", "0.3", "--prim-name", "Box"]),
                ("configure_usd_physics", ["--mass", "0.3", "--solver-position-iterations", "64", "--solver-velocity-iterations", "8"]),
            ]:
                result = subprocess.run([sys.executable, "-m", f"objgen.to_usd.{script}", str(run), *args], capture_output=True, text=True,
                                        env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                if script == "mesh_to_usd":
                    visual_digest = hashlib.sha256((run / "usd/object.usdc").read_bytes()).hexdigest()
                    conversion_bytes = (run / "usd/object.conversion.json").read_bytes()
            self.assertEqual(visual_digest, hashlib.sha256((run / "usd/object.usdc").read_bytes()).hexdigest())
            self.assertEqual(conversion_bytes, (run / "usd/object.conversion.json").read_bytes())
            visual = Usd.Stage.Open(str(run / "usd/object.usdc"))
            physical = Usd.Stage.Open(str(run / "physics/object.usdc"))
            self.assertFalse(visual.GetDefaultPrim().HasAPI(UsdPhysics.RigidBodyAPI))
            self.assertTrue(physical.GetDefaultPrim().HasAPI(UsdPhysics.RigidBodyAPI))
            self.assertAlmostEqual(UsdPhysics.MassAPI(physical.GetDefaultPrim()).GetMassAttr().Get(), 0.3)
            self.assertEqual(physical.GetDefaultPrim().GetAttribute("physxRigidBody:solverVelocityIterationCount").Get(), 8)
            self.assertEqual(physical.GetDefaultPrim().GetAttribute("physxRigidBody:solverPositionIterationCount").Get(), 64)
            self.assertFalse(visual.GetDefaultPrim().GetAttribute("physxRigidBody:solverVelocityIterationCount"))
            self.assertFalse(json.loads(conversion_bytes)["physics_configured"])
            self.assertTrue(json.loads((run / "physics/object.conversion.json").read_text())["physics_configured"])


if __name__ == "__main__":
    unittest.main()
