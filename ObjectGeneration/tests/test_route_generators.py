"""route_generators: the message it builds and the checks on the answer.  No model call."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from objgen.planning import route_generators as rg  # noqa: E402
from test_prepare_dualcraft_inputs import CAMERA, box_cloud, camera_at, render_mask  # noqa: E402


def answer(route="dualcraft", parts=None, confidence="high", best=1):
    return {"identity": "cabinet", "label_matches": True,
            "moving_parts": parts if parts is not None else
            [{"part": "door", "motion": "hinged_door", "count": 2, "visible": True}],
            "route": route, "confidence": confidence, "reasoning": "x",
            "front": {"best_image": best, "facing": "toward_camera", "cue": "doors"}}


class CheckTest(unittest.TestCase):
    def test_consistent_answers_pass(self):
        self.assertEqual(rg.check(answer(), images=2), [])
        self.assertEqual(rg.check(answer("sam3d", parts=[]), images=1), [])
        # a knob alone does not make an object articulated
        knob = [{"part": "dial", "motion": "knob_or_dial", "count": 1, "visible": True}]
        self.assertEqual(rg.check(answer("sam3d", parts=knob), images=1), [])

    def test_inconsistent_answers_are_flagged(self):
        self.assertEqual(rg.check(answer(best=3), images=2), [])             # image 3: the context frame
        self.assertTrue(rg.check(answer(best=4), images=2))                  # no image 4
        self.assertTrue(rg.check(answer(parts=[]), images=1))                # dualcraft, nothing moves
        self.assertTrue(rg.check(answer("sam3d"), images=1))                 # sam3d, a door listed, high
        self.assertEqual(rg.check(answer("sam3d", confidence="low"), images=1), [])

    def test_front_seq_maps_context_to_first_frame(self):
        shown = [{"image": 1, "seq": 145}, {"image": 2, "seq": 124}, {"image": 3, "seq": 145, "role": "context"}]
        self.assertEqual(rg.front_seq(answer(best=2), shown), 124)
        self.assertEqual(rg.front_seq(answer(best=3), shown), 145)
        self.assertIsNone(rg.front_seq(answer(best=0), shown))

    def test_schema_is_strict(self):
        def walk(node):
            if node.get("type") == "object":
                self.assertFalse(node.get("additionalProperties", True))
                self.assertEqual(set(node["required"]), set(node["properties"]))
                for child in node["properties"].values():
                    walk(child)
            if node.get("type") == "array":
                walk(node["items"])
        walk(rg.ROUTE_SCHEMA)


class OverrideTest(unittest.TestCase):
    def test_override_wins_and_survives_a_rerun(self):
        objects = [{"object_id": "obj_0001", "route": "dualcraft"}, {"object_id": "obj_0002", "route": "sam3d"}]
        overrides = {"obj_0001": {"route": "sam3d", "note": "by hand"}, "obj_0009": {"route": "sam3d"}}
        unknown = rg.apply_overrides(objects, overrides)
        self.assertEqual(unknown, ["obj_0009"])
        self.assertEqual((objects[0]["route"], objects[0]["model_route"]), ("sam3d", "dualcraft"))
        self.assertEqual(objects[0]["override"]["note"], "by hand")
        self.assertEqual((objects[1]["route"], objects[1]["override"]), ("sam3d", None))
        # applying again (an --overrides-only run) changes nothing
        rg.apply_overrides(objects, overrides)
        self.assertEqual((objects[0]["route"], objects[0]["model_route"]), ("sam3d", "dualcraft"))
        # a fresh model answer for the object: the override still holds, model_route updates
        fresh = {"object_id": "obj_0001", "route": "dualcraft"}
        rg.apply_overrides([fresh], overrides)
        self.assertEqual((fresh["route"], fresh["model_route"]), ("sam3d", "dualcraft"))
        # the override removed: the model's answer comes back
        rg.apply_overrides(objects, {})
        self.assertEqual((objects[0]["route"], objects[0]["override"]), ("dualcraft", None))

    def test_check_judges_the_model_answer(self):
        overridden = {**answer(), "route": "sam3d", "model_route": "dualcraft"}
        self.assertEqual(rg.check(overridden, images=1), [])


class ContentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_sends_the_dualcraft_photos_and_context(self):
        import trimesh
        handoff = self.tmp / "handoff"
        (handoff / "keyframes" / "obj_0001").mkdir(parents=True)
        (handoff / "clouds").mkdir()
        points = box_cloud()
        trimesh.PointCloud(points).export(handoff / "clouds" / "obj_0001.ply")
        frames = []
        for seq, role, azimuth in ((10, "primary", 0.0), (20, "secondary", 4.0), (30, "third", 50.0)):
            pose = camera_at(azimuth, 6.0)
            mask = render_mask(points, pose)
            stem = f"keyframes/obj_0001/{seq:04d}"
            Image.fromarray(np.full((480, 640, 3), 90, np.uint8)).save(handoff / f"{stem}_rgb.jpg")
            Image.fromarray(np.full((480, 640, 3), 90, np.uint8)).save(handoff / f"{stem}_boxed.jpg")
            Image.fromarray((mask * 255).astype(np.uint8)).save(handoff / f"{stem}_mask.png")
            frames.append({"seq": seq, "role": role, "rgb": f"{stem}_rgb.jpg", "mask": f"{stem}_mask.png",
                           "boxed": f"{stem}_boxed.jpg", "camera_to_world": pose})
        record = {"object_id": "obj_0001", "label": "object", "label_quality": "generic",
                  "cloud": "clouds/obj_0001.ply", "key_frames": frames}
        content, shown, extra = rg.build_content(handoff, record, CAMERA, floor_z=0.0)
        # frame 20 is 4 deg from frame 10: same view, not sent (the converter's rule)
        self.assertEqual([s["seq"] for s in shown], [10, 30, 10])
        self.assertEqual(shown[-1]["role"], "context")
        self.assertEqual([d["seq"] for d in extra["dropped"]], [20])
        self.assertEqual(sum(1 for b in content if b["type"] == "image"), 3)
        text = content[0]["text"]
        self.assertIn("generic label", text)
        self.assertIn("Images 1-2", text)
        self.assertAlmostEqual(extra["size"]["height"], 1.96, delta=0.03)


if __name__ == "__main__":
    unittest.main()
