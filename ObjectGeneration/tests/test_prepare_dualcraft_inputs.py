"""prepare_dualcraft_inputs: which key frames go to DualCraft, and how they are cut.

A synthetic scene stands in for the handoff: a box-shaped object seen by cameras at
known positions, its mask rendered by projecting the object's points, so every rule is
checked against an answer worked out by hand.
"""
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from objgen.dualcraft import prepare_dualcraft_inputs as pdi  # noqa: E402

CAMERA = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480}
CENTRE = np.array([2.0, 3.0, 1.0])


def box_cloud(seed: int = 0) -> np.ndarray:
    """1.0 x 0.5 x 2.0 m, long side turned 30 deg from +X, standing on z = 0."""
    rng = np.random.default_rng(seed)
    local = rng.uniform([-0.5, -0.25, 0.0], [0.5, 0.25, 2.0], size=(20000, 3))
    a = math.radians(30.0)
    rotation = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    return local @ rotation.T + [CENTRE[0], CENTRE[1], 0.0]


def look_at(eye, target) -> list:
    """camera_to_world, OpenCV axes (+X right, +Y down, +Z forward), world +Z up."""
    forward = np.asarray(target, float) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([right, down, forward])
    pose[:3, 3] = eye
    return pose.tolist()


def camera_at(azimuth_deg: float, distance: float, aim=CENTRE) -> list:
    a = math.radians(azimuth_deg)
    eye = CENTRE + distance * np.array([math.cos(a), math.sin(a), 0.0])
    return look_at(eye, aim)


def render_mask(points: np.ndarray, pose: list) -> np.ndarray:
    pose = np.asarray(pose)
    local = (points - pose[:3, 3]) @ pose[:3, :3]
    ahead = local[:, 2] > 1e-3
    u = (CAMERA["fx"] * local[ahead, 0] / local[ahead, 2] + CAMERA["cx"]).astype(int)
    v = (CAMERA["fy"] * local[ahead, 1] / local[ahead, 2] + CAMERA["cy"]).astype(int)
    keep = (u >= 0) & (u < CAMERA["width"]) & (v >= 0) & (v < CAMERA["height"])
    mask = np.zeros((CAMERA["height"], CAMERA["width"]), bool)
    mask[v[keep], u[keep]] = True
    return mask


class GeometryTest(unittest.TestCase):
    def test_scan_box_finds_the_turned_box(self):
        box = pdi.scan_box(box_cloud())
        # 1-99 percentile of a uniform spread covers 98 % of each side
        self.assertAlmostEqual(box["long"], 0.98, delta=0.02)
        self.assertAlmostEqual(box["short"], 0.49, delta=0.02)
        self.assertAlmostEqual(box["height"], 1.96, delta=0.03)
        np.testing.assert_allclose(box["centre_xy"], CENTRE[:2], atol=0.01)
        long_axis = np.array(box["long_axis_xy"])
        self.assertGreater(abs(long_axis @ [math.cos(math.radians(30)), math.sin(math.radians(30))]), 0.999)

    def test_out_of_picture(self):
        points = box_cloud()
        self.assertEqual(pdi.out_of_picture(points, camera_at(0, 6.0), CAMERA), 0.0)
        # looking away: nothing in the picture
        self.assertEqual(pdi.out_of_picture(points, look_at(CENTRE + [6, 0, 0], CENTRE + [12, 0, 0]), CAMERA), 1.0)
        # aimed past the object's edge: about half of it falls outside
        aim = CENTRE + np.array([0.0, 0.0, 1.0])            # aim at the top: the bottom half drops out
        share = pdi.out_of_picture(points, camera_at(0, 1.5, aim), CAMERA)
        self.assertGreater(share, 0.3)
        self.assertLess(share, 1.0)

    def test_crop_box_margin_and_clip(self):
        mask = np.zeros((480, 640), bool)
        mask[50:150, 100:200] = True
        self.assertEqual(pdi.crop_box(mask), [85, 35, 215, 165])
        mask = np.zeros((480, 640), bool)
        mask[0:100, 600:640] = True                         # touching the top-right corner
        left, top, right, bottom = pdi.crop_box(mask)
        self.assertEqual((top, right), (0, 640))
        self.assertEqual(left, 594)
        self.assertEqual(bottom, 115)


class SelectionTest(unittest.TestCase):
    def frames(self, specs):
        return [{"seq": seq, "role": role, "camera_to_world": pose}
                for seq, role, pose in specs]

    def test_drops_cut_and_duplicate_views(self):
        points = box_cloud()
        aim_top = CENTRE + np.array([0.0, 0.0, 1.6])
        frames = self.frames([
            (1, "primary", camera_at(0, 6.0)),
            (2, "secondary", camera_at(5, 6.0)),            # 5 deg from frame 1: same view
            (3, "third", camera_at(60, 1.2, aim_top)),      # most of the object below the picture
        ])
        cut = {f["seq"]: pdi.out_of_picture(points, f["camera_to_world"], CAMERA) for f in frames}
        self.assertGreater(cut[3], pdi.OUT_OF_PICTURE_MAX)
        kept, dropped = pdi.select_frames(frames, cut, CENTRE)
        self.assertEqual([f["seq"] for f in kept], [1])
        reasons = {d["seq"]: d["reason"] for d in dropped}
        self.assertIn("same view", reasons[2])
        self.assertIn("out of the picture", reasons[3])

    def test_keeps_separate_views_in_order(self):
        points = box_cloud()
        frames = self.frames([(1, "primary", camera_at(0, 6.0)), (2, "secondary", camera_at(40, 6.0)),
                              (3, "third", camera_at(-35, 6.0))])
        cut = {f["seq"]: pdi.out_of_picture(points, f["camera_to_world"], CAMERA) for f in frames}
        kept, dropped = pdi.select_frames(frames, cut, CENTRE)
        self.assertEqual([f["seq"] for f in kept], [1, 2, 3])
        self.assertEqual(dropped, [])

    def test_never_drops_the_least_cut_frame(self):
        frames = self.frames([(1, "primary", camera_at(0, 6.0)), (2, "secondary", camera_at(60, 6.0))])
        kept, _ = pdi.select_frames(frames, {1: 0.7, 2: 0.9}, CENTRE)
        self.assertEqual([f["seq"] for f in kept], [1])

    def test_max_photos(self):
        frames = self.frames([(i, "primary", camera_at(i * 30.0, 6.0)) for i in range(5)])
        kept, dropped = pdi.select_frames(frames, {i: 0.0 for i in range(5)}, CENTRE, max_photos=3)
        self.assertEqual(len(kept), 3)
        self.assertTrue(all("more than 3" in d["reason"] for d in dropped))


class PrepareObjectTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_writes_photos_and_record(self):
        import trimesh
        handoff = self.tmp / "handoff"
        (handoff / "keyframes" / "obj_0001").mkdir(parents=True)
        (handoff / "clouds").mkdir()
        points = box_cloud()
        trimesh.PointCloud(points).export(handoff / "clouds" / "obj_0001.ply")
        key_frames = []
        for seq, role, azimuth in ((10, "primary", 0.0), (20, "secondary", 4.0), (30, "third", 50.0)):
            pose = camera_at(azimuth, 6.0)
            mask = render_mask(points, pose)
            Image.fromarray(np.full((480, 640, 3), 90, np.uint8)).save(handoff / f"keyframes/obj_0001/{seq:04d}_rgb.jpg")
            Image.fromarray((mask * 255).astype(np.uint8)).save(handoff / f"keyframes/obj_0001/{seq:04d}_mask.png")
            key_frames.append({"seq": seq, "role": role, "source_frame_id": seq * 10,
                               "rgb": f"keyframes/obj_0001/{seq:04d}_rgb.jpg",
                               "mask": f"keyframes/obj_0001/{seq:04d}_mask.png",
                               "camera_to_world": pose, "visibility": 1.0, "clarity": 100.0})
        record = {"object_id": "obj_0001", "label": "cabinet", "cloud": "clouds/obj_0001.ply",
                  "key_frames": key_frames}
        out = self.tmp / "out" / "obj_0001"
        conversion = pdi.prepare_object(self.tmp, handoff, record, CAMERA, out)
        self.assertEqual([p["seq"] for p in conversion["photos"]], [10, 30])
        self.assertEqual([d["seq"] for d in conversion["dropped"]], [20])
        saved = json.loads((out / "conversion.json").read_text())
        self.assertEqual(saved["schema_version"], pdi.SCHEMA)
        for photo in saved["photos"]:
            with Image.open(out / photo["file"]) as image:
                size = image.size
            left, top, right, bottom = photo["crop"]
            self.assertEqual(size, (right - left, bottom - top))   # under MAX_SIDE: not resized
            self.assertEqual(photo["sha256"], pdi.sha256(out / photo["file"]))


if __name__ == "__main__":
    unittest.main()
