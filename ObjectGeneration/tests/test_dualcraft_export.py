"""DualCraft import and articulated export: structure kept, physics rebuilt.

A synthetic DualCraft product stands in for a real one: a box carcass with a door
on a +z hinge, a handle on the door (fixed joint) and a shelf in the carcass
(fixed joint), each link an OBJ of convex boards.  Everything runs without the
server, a GPU or Isaac Sim.
"""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "tools"))

from objgen.dualcraft import configure_articulated_physics as cap  # noqa: E402
from objgen.dualcraft import dualcraft_product as dp  # noqa: E402
from objgen import handoff_io as hio  # noqa: E402
from objgen.dualcraft import import_dualcraft  # noqa: E402

# Carcass 0.4 deep (x), 0.8 wide (y), 1.0 high; door hinged on its +y edge.
URDF = """<robot name="test_cabinet">
  <link name="housing"><visual><geometry><mesh filename="assets/meshes/housing/mesh.obj"/></geometry></visual></link>
  <link name="shelf"><visual><geometry><mesh filename="assets/meshes/shelf/mesh.obj"/></geometry></visual></link>
  <link name="door_1"><visual><geometry><mesh filename="assets/meshes/door/mesh.obj"/></geometry></visual></link>
  <link name="handle"><visual><geometry><mesh filename="assets/meshes/handle/mesh.obj"/></geometry></visual></link>
  <joint name="shelf_joint" type="fixed"><parent link="housing"/><child link="shelf"/></joint>
  <joint name="door_1_joint" type="revolute"><origin xyz="0.21 {hinge_y} 0.5" rpy="0 0 0"/>
    <parent link="housing"/><child link="door_1"/><axis xyz="0 0 {axis_z}"/>
    <limit effort="20" velocity="1" lower="0" upper="1.5708"/></joint>
  <joint name="handle_joint" type="fixed"><origin xyz="0.02 {handle_y} 0" rpy="0 0 0"/>
    <parent link="door_1"/><child link="handle"/></joint>
</robot>"""


def boards(*specs) -> trimesh.Trimesh:
    """Disjoint axis-aligned boxes, one connected piece each: (centre, extents)."""
    return trimesh.util.concatenate([trimesh.creation.box(extents=e).apply_translation(c) for c, e in specs])


def write_product(directory: Path, hinge_side: float = 1.0) -> Path:
    """``hinge_side`` +1 hinges the door on the +y edge with a +z axis (opens outward);
    -1 mirrors the hinge to the -y edge but keeps the +z axis, so it swings inward."""
    product = directory / "product"
    meshes = {
        # back, two sides, top, bottom: five convex boards, open at the front (+x)
        "housing": boards(([-0.19, 0, 0.5], [0.02, 0.8, 1.0]), ([0, 0.39, 0.5], [0.4, 0.02, 1.0]),
                          ([0, -0.39, 0.5], [0.4, 0.02, 1.0]), ([0, 0, 0.99], [0.4, 0.76, 0.02]),
                          ([0, 0, 0.01], [0.4, 0.76, 0.02])),
        # full-depth shelf: a door swinging inward cuts through it, as on a real cabinet
        "shelf": boards(([0, 0, 0.5], [0.36, 0.76, 0.02])),
        # door in its own frame: hinge at the origin, panel extends toward -y (or +y)
        "door": boards(([0, -hinge_side * 0.39, 0], [0.02, 0.78, 0.98])),
        "handle": boards(([0, 0, 0], [0.02, 0.02, 0.2])),
    }
    for name, mesh in meshes.items():
        path = product / "assets" / "meshes" / name / "mesh.obj"
        path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(path)
    (product / "model.urdf").write_text(URDF.format(
        hinge_y=hinge_side * 0.40, axis_z=1, handle_y=-hinge_side * 0.70))
    (product / "compile_report.json").write_text(json.dumps({"status": "success", "metrics": {"turn_count": 3}}))
    inputs = directory / "inputs"
    (inputs / "blueprint").mkdir(parents=True)
    parts = [{"id": "housing", "parent": None, "appearance": {"finish": "wood"}},
             {"id": "shelf", "parent": "housing", "appearance": {"finish": "wood"}},
             {"id": "door_1", "parent": "housing", "appearance": {"finish": "wood"}},
             {"id": "handle", "parent": "door_1", "appearance": {"finish": "brushed metal"}}]
    (inputs / "prompt.json").write_text(json.dumps({"text": "test cabinet", "parts": parts}))
    (inputs / "blueprint" / "source.json").write_text(json.dumps(
        {"estimates": {"mass_kg": 30, "materials": ["wood", "metal"]}}))
    (inputs / "scannet_handoff.json").write_text(json.dumps({
        "object": "obj_0001", "front_frame": 1,
        "size": {"width": 0.8, "depth": 0.42, "height": 1.0, "measured": True},
        "views": [{"file": "photo_0.png", "keyframe": 1, "azimuth_deg": -10.0, "elevation_deg": 5.0}],
        "object_frame": {"origin_xy": [2.0, 3.0], "bottom_z": 0.05, "front": [0.0, -1.0, 0.0],
                         "left": [1.0, 0.0, 0.0], "up": [0, 0, 1]}}))
    return directory


def write_generation(base: Path) -> Path:
    """A handoff with one object and an empty SAM3D generation for it."""
    handoff = base / "handoff" / "scene_t" / "gt"
    (handoff / "keyframes" / "obj_0001").mkdir(parents=True)
    Image.fromarray(np.full((6, 8, 3), 100, np.uint8)).save(handoff / "keyframes/obj_0001/0001_rgb.png")
    Image.fromarray(np.full((6, 8), 255, np.uint8)).save(handoff / "keyframes/obj_0001/0001_mask.png")
    (handoff / "manifest.json").write_text(json.dumps({"schema_version": "handoff/v1", "scene_id": "scene_t"}))
    (handoff / "objects.json").write_text(json.dumps([{
        "object_id": "obj_0001", "label": "cabinet", "structural": False,
        "key_frames": [{"role": "primary", "seq": 1, "rgb": "keyframes/obj_0001/0001_rgb.png",
                        "mask": "keyframes/obj_0001/0001_mask.png"}]}]))
    generation = base / "gen"
    tasks = hio.plan(handoff, json.loads((handoff / "objects.json").read_text()), generation)
    hio.write_index(generation, handoff, json.loads((handoff / "manifest.json").read_text()), tasks,
                    {"seed": 42})
    return generation


class ProductTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.run_dir = write_product(self.tmp / "run")
        self.product = dp.read_product(self.run_dir / "product")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_fixed_links_merge_into_their_body(self):
        self.assertEqual(dp.rigid_groups(self.product),
                         {"housing": ["housing", "shelf"], "door_1": ["door_1", "handle"]})

    def test_mass_split_by_hull_volume_and_material(self):
        materials = dp.link_materials(self.product, dp.read_blueprint(self.run_dir))
        self.assertEqual(materials["handle"]["material"], "metal")
        bodies = dp.build_bodies(self.product, materials)
        summary = dp.distribute_mass(bodies, 30.0)
        self.assertAlmostEqual(sum(b.mass for b in bodies.values()), 30.0, places=6)
        # every board is a box, so hulls are exact and convexity is 1
        for body in bodies.values():
            for stats in body.pieces.values():
                self.assertAlmostEqual(stats["convexity_min"], 1.0, places=6)
        wood = summary["hull_volume_m3"]["wood"]
        metal = summary["hull_volume_m3"]["metal"]
        factor = 30.0 / (wood * 600 + metal * 7800)
        self.assertAlmostEqual(summary["density_factor"], round(factor, 4), places=4)
        door = bodies["door_1"]
        expected = factor * (0.02 * 0.78 * 0.98 * 600 + 0.02 * 0.02 * 0.2 * 7800)
        self.assertAlmostEqual(door.mass, expected, places=6)

    def test_inertia_matches_trimesh_for_one_material(self):
        materials = dp.link_materials(self.product, dp.read_blueprint(self.run_dir))
        bodies = dp.build_bodies(self.product, materials)
        dp.distribute_mass(bodies, 30.0)
        housing = bodies["housing"]          # wood only
        merged = trimesh.util.concatenate(housing.hulls)
        merged.density = housing.mass / merged.volume
        np.testing.assert_allclose(housing.centre_of_mass, merged.center_mass, atol=1e-9)
        np.testing.assert_allclose(housing.inertia, merged.moment_inertia, atol=1e-9)

    def test_rest_bounds_and_offset(self):
        bounds = dp.rest_bounds(self.product)
        np.testing.assert_allclose(bounds[0], [-0.2, -0.4, 0.0], atol=1e-6)
        self.assertAlmostEqual(bounds[1][2], 1.0, places=6)
        offset = dp.asset_offset(bounds)
        self.assertAlmostEqual(offset[2], 0.0, places=6)

    def test_outward_door_is_clean(self):
        bodies = dp.build_bodies(self.product, dp.link_materials(self.product, dp.read_blueprint(self.run_dir)))
        dp.distribute_mass(bodies, 30.0)
        checks = dp.interference(self.product, bodies)
        self.assertEqual(checks["swept_into_parent"], [])
        self.assertEqual(checks["moves_inward"], [])

    def test_door_hinged_the_wrong_way_is_caught(self):
        run = write_product(self.tmp / "wrong", hinge_side=-1.0)
        product = dp.read_product(run / "product")
        bodies = dp.build_bodies(product, dp.link_materials(product, dp.read_blueprint(run)))
        dp.distribute_mass(bodies, 30.0)
        checks = dp.interference(product, bodies)
        # flush against the side wall, so the overlap test alone misses it
        self.assertEqual([s["joint"] for s in checks["moves_inward"]], ["door_1_joint"])

    def test_joint_frame_keeps_axis_sign(self):
        for axis in ([0, 0, 1], [0, 0, -1], [1, 0, 0], [0, -1, 0]):
            token, local = cap.joint_frame(np.array(axis, dtype=float))
            index = "XYZ".index(token)
            np.testing.assert_allclose(local[:, index], axis, atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(local), 1.0, places=12)


class ExportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.run_dir = write_product(self.tmp / "run")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_usd_layout(self):
        from pxr import Usd, UsdGeom, UsdPhysics
        out = self.tmp / "usd" / "physics.usdc"
        report = cap.build(self.run_dir / "product", out, 30.0, "static", self.run_dir, "Obj1",
                           visual_output=self.tmp / "usd" / "visual.usdc")
        stage = Usd.Stage.Open(str(out))
        root = stage.GetDefaultPrim()
        self.assertTrue(root.HasAPI(UsdPhysics.ArticulationRootAPI))
        bodies = [p for p in root.GetChildren() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        self.assertEqual(sorted(p.GetName() for p in bodies), ["door_1", "housing"])
        masses = [UsdPhysics.MassAPI(p).GetMassAttr().Get() for p in bodies]
        self.assertAlmostEqual(sum(masses), 30.0, places=4)
        # only collision/hull_* collide; visual meshes carry no CollisionAPI at all
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Mesh):
                self.assertEqual(prim.HasAPI(UsdPhysics.CollisionAPI), prim.GetName().startswith("hull_"),
                                 prim.GetPath())
        joints = {p.GetName(): p for p in stage.GetPrimAtPath("/Obj1/joints").GetChildren()}
        self.assertEqual(sorted(joints), ["door_1_joint", "fix_base"])
        door = UsdPhysics.RevoluteJoint(joints["door_1_joint"])
        self.assertEqual(door.GetAxisAttr().Get(), "Z")
        self.assertAlmostEqual(door.GetUpperLimitAttr().Get(), 90.0, places=2)
        # the asset frame puts the closed object on z = 0 with its XY box centred
        box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(root).ComputeAlignedRange()
        self.assertAlmostEqual(box.GetMin()[2], 0.0, places=5)
        self.assertAlmostEqual(box.GetMin()[0] + box.GetMax()[0], 0.0, places=5)
        self.assertEqual(report["collision_hulls"], 5 + 1 + 1 + 1)
        visual = Usd.Stage.Open(str(self.tmp / "usd" / "visual.usdc"))
        self.assertFalse(any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in Usd.PrimRange(visual.GetDefaultPrim())))

    def test_dynamic_has_no_fixed_base(self):
        from pxr import Usd
        out = self.tmp / "usd" / "physics.usdc"
        cap.build(self.run_dir / "product", out, 30.0, "dynamic", self.run_dir, "Obj1")
        stage = Usd.Stage.Open(str(out))           # keep the stage alive while reading
        joints = stage.GetPrimAtPath("/Obj1/joints").GetChildren()
        self.assertEqual([p.GetName() for p in joints], ["door_1_joint"])


class ImportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.run_dir = write_product(self.tmp / "run")
        self.generation = write_generation(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_import_points_the_index_at_dualcraft(self):
        import_dualcraft.import_run(self.generation, "obj_0001", self.run_dir)
        record = json.loads((self.generation / "objects.json").read_text())[0]
        self.assertEqual(record["generator"], "dualcraft")
        self.assertEqual(record["status"], "success")
        self.assertEqual(record["urdf"], "objects/obj_0001/primary/dualcraft/product/model.urdf")
        self.assertTrue((self.generation / record["urdf"]).is_file())
        self.assertEqual(record["stats"]["movable_joints"], ["door_1_joint"])
        manifest = json.loads((self.generation / "manifest.json").read_text())
        self.assertEqual(manifest["generator_counts"], {"dualcraft": 1})
        self.assertEqual(manifest["generator"], {"seed": 42})   # SAM3D settings kept

    def test_reindex_keeps_the_import(self):
        import_dualcraft.import_run(self.generation, "obj_0001", self.run_dir)
        hio.reindex(self.generation)
        self.assertEqual(json.loads((self.generation / "objects.json").read_text())[0]["generator"], "dualcraft")

    def test_missing_object_frame_is_refused(self):
        conversion = self.run_dir / "inputs" / "scannet_handoff.json"
        data = json.loads(conversion.read_text())
        del data["object_frame"]
        conversion.write_text(json.dumps(data))
        with self.assertRaises(SystemExit):
            import_dualcraft.import_run(self.generation, "obj_0001", self.run_dir)


class ScenePoseTest(unittest.TestCase):
    def test_pose_maps_asset_into_the_scan(self):
        from objgen.handoff import build_assembly_handoff as bah
        frame = {"origin_xy": [2.0, 3.0], "bottom_z": 0.05, "front": [0.0, -1.0, 0.0],
                 "left": [1.0, 0.0, 0.0], "up": [0, 0, 1]}
        pose = bah.scene_pose(frame, [0.01, 0.0, 0.0])
        matrix = np.array(pose["matrix"])
        # asset origin = DualCraft point (0.01, 0, 0): 0.01 m in front of the frame origin
        np.testing.assert_allclose(matrix @ [0, 0, 0, 1], [2.0, 3.0 - 0.01, 0.05, 1], atol=1e-6)
        # asset +X (front) points along the recorded front, +Y along left
        np.testing.assert_allclose(matrix[:3, 0], [0, -1, 0], atol=1e-6)
        np.testing.assert_allclose(matrix[:3, 1], [1, 0, 0], atol=1e-6)

    def test_rounded_frame_is_orthonormalised(self):
        from objgen.handoff import build_assembly_handoff as bah
        frame = {"origin_xy": [0, 0], "bottom_z": 0, "front": [-0.966, -0.259, 0.0],
                 "left": [0.259, -0.966, 0.0], "up": [0, 0, 1]}
        rotation = np.array(bah.scene_pose(frame, [0, 0, 0])["rotation"])
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
