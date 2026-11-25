"""Handoff-driven generation: selection, layout, failure isolation and resume.

The SAM3D model is replaced by a fake that returns a box mesh, so every test
runs without a GPU while still exercising the real file layout and reports.
"""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from objgen.generate import generate  # noqa: E402
from objgen import handoff_io as hio  # noqa: E402

FAIL_MARK = 7  # rgb value that makes the fake pipeline raise


class FakeGaussian:
    def save_ply(self, path):
        Path(path).write_bytes(b"ply\n")


class FakePipeline:
    def __init__(self):
        self.calls = 0

    def __call__(self, image, mask, seed):
        import trimesh
        self.calls += 1
        if image[0, 0, 0] == FAIL_MARK:
            raise RuntimeError("synthetic failure")
        return {"glb": trimesh.creation.box(), "gaussian": [FakeGaussian()],
                "rotation": np.eye(3), "scale": np.ones(3)}


def write_frame(folder, seq, value=100, mask_on=True, mask_size=(8, 6)):
    folder.mkdir(parents=True, exist_ok=True)
    rgb = folder / f"{seq:04d}_rgb.png"
    Image.fromarray(np.full((6, 8, 3), value, np.uint8)).save(rgb)
    mask = np.zeros(mask_size[::-1], np.uint8)
    if mask_on:
        mask[1:4, 2:6] = 255
    Image.fromarray(mask).save(folder / f"{seq:04d}_mask.png")


def make_handoff(base):
    """Five objects: two good, one that fails in the model, one structural,
    one whose mask is empty."""
    handoff = base / "handoff" / "scene_x" / "gt"
    specs = [("obj_0001", "chair", False, {}), ("obj_0002", "desk", False, {}),
             ("obj_0003", "lamp", False, {"value": FAIL_MARK}), ("obj_0004", "wall", True, {}),
             ("obj_0005", "cup", False, {"mask_on": False})]
    records = []
    for index, (object_id, label, structural, frame) in enumerate(specs):
        seq = 10 + index
        write_frame(handoff / "keyframes" / object_id, seq, **frame)
        records.append({"object_id": object_id, "label": label, "structural": structural,
                        "key_frames": [{"seq": seq, "role": "primary",
                                        "rgb": f"keyframes/{object_id}/{seq:04d}_rgb.png",
                                        "mask": f"keyframes/{object_id}/{seq:04d}_mask.png"}]})
    (handoff / "objects.json").write_text(json.dumps(records))
    (handoff / "manifest.json").write_text(json.dumps({"schema_version": "handoff/v1", "scene_id": "scene_x"}))
    return handoff


class GenerateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.handoff = make_handoff(self.base)
        self.output = self.base / "out"
        self.config = self.base / "pipeline.yaml"
        self.config.write_text("fake: true\n")
        self.pipeline = FakePipeline()
        self.loads = 0
        for target, kwargs in (("torch.cuda.reset_peak_memory_stats", {}),
                               ("torch.cuda.max_memory_allocated", {"return_value": 0}),
                               ("torch.cuda.empty_cache", {})):
            patcher = patch(target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self):
        self.temp.cleanup()

    def load(self, args):
        self.loads += 1
        return self.pipeline

    def run_main(self, *extra, source=None):
        argv = [*(source or ["--handoff", str(self.handoff)]), "--output", str(self.output),
                "--config", str(self.config), *extra]
        with patch.object(generate, "load_model", self.load), contextlib.redirect_stdout(io.StringIO()):
            return generate.main(argv)

    def index(self):
        return {r["object_id"]: r for r in json.loads((self.output / "objects.json").read_text())}

    def stage(self, object_id):
        return self.output / "objects" / object_id / "primary" / "sam3d"

    def test_every_handoff_object_is_accounted_for(self):
        self.assertEqual(self.run_main(), 1)  # one object failed
        records = self.index()
        self.assertEqual(set(records), {"obj_0001", "obj_0002", "obj_0003", "obj_0004", "obj_0005"})
        status = {key: value["status"] for key, value in records.items()}
        self.assertEqual(status, {"obj_0001": "success", "obj_0002": "success", "obj_0003": "failed",
                                  "obj_0004": "skipped", "obj_0005": "skipped"})
        self.assertIn("synthetic failure", records["obj_0003"]["error"])
        self.assertEqual(records["obj_0004"]["reason"], "structural object")
        self.assertEqual(records["obj_0005"]["reason"], "mask is empty")
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["schema_version"], hio.SCHEMA)
        self.assertEqual(sum(manifest["status_counts"].values()), 5)

    def test_success_writes_outputs_and_binds_them_to_the_report(self):
        self.run_main("--objects", "obj_0001")
        stage = self.stage("obj_0001")
        for name in hio.OUTPUTS:
            self.assertTrue((stage / name).is_file(), name)
        report = json.loads((stage / "report.json").read_text())
        self.assertEqual(report["outputs"]["mesh.glb"], hio.sha256(stage / "mesh.glb"))
        self.assertEqual(report["inputs"]["mask_pixels"], 12)
        pose = json.loads((stage / "pose.json").read_text())
        self.assertEqual(pose["rotation"], np.eye(3).tolist())
        self.assertEqual(self.index()["obj_0001"]["mesh"], "objects/obj_0001/primary/sam3d/mesh.glb")

    def test_selection_marks_others_without_dropping_them(self):
        self.run_main("--objects", "obj_0002")
        records = self.index()
        self.assertEqual(records["obj_0002"]["status"], "success")
        self.assertEqual(records["obj_0001"]["status"], "pending")
        self.assertEqual(self.pipeline.calls, 1)

    def test_unknown_object_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "obj_9999"):
            self.run_main("--objects", "obj_9999")

    def test_rerun_reuses_finished_objects_without_loading_the_model(self):
        self.run_main("--objects", "obj_0001,obj_0002")
        self.assertEqual(self.run_main("--objects", "obj_0001,obj_0002"), 0)
        self.assertEqual((self.loads, self.pipeline.calls), (1, 2))

    def test_changed_input_or_tampered_output_regenerates(self):
        self.run_main("--objects", "obj_0001,obj_0002")
        write_frame(self.handoff / "keyframes" / "obj_0001", 10, value=150)
        (self.stage("obj_0002") / "mesh.glb").write_bytes(b"tampered")
        self.run_main("--objects", "obj_0001,obj_0002")
        self.assertEqual(self.pipeline.calls, 4)

    def test_failed_object_is_retried_and_force_regenerates_all(self):
        self.run_main("--objects", "obj_0001,obj_0003")
        self.run_main("--objects", "obj_0001,obj_0003")
        self.assertEqual(self.pipeline.calls, 3)  # only the failed object ran again
        self.run_main("--objects", "obj_0001,obj_0003", "--force")
        self.assertEqual(self.pipeline.calls, 5)

    def test_failure_leaves_no_mesh_from_an_earlier_run(self):
        self.run_main("--objects", "obj_0001")
        write_frame(self.handoff / "keyframes" / "obj_0001", 10, value=FAIL_MARK)
        self.run_main("--objects", "obj_0001")
        self.assertFalse((self.stage("obj_0001") / "mesh.glb").exists())
        self.assertEqual(self.index()["obj_0001"]["status"], "failed")

    def test_dry_run_writes_nothing(self):
        self.assertEqual(self.run_main("--dry-run"), 0)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.loads, 0)


class InputInspectionTests(unittest.TestCase):
    def test_bad_frames_are_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            write_frame(folder, 1, mask_size=(5, 5))
            reason, _ = hio.inspect_inputs(folder / "0001_rgb.png", folder / "0001_mask.png")
            self.assertIn("sizes differ", reason)
            reason, _ = hio.inspect_inputs(folder / "missing.png", folder / "0001_mask.png")
            self.assertIn("missing", reason)


class SingleImageTests(unittest.TestCase):
    def test_single_image_becomes_a_one_object_handoff_with_a_stable_home(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            write_frame(base / "photos", 3)
            image, mask = base / "photos/0003_rgb.png", base / "photos/0003_mask.png"
            first = hio.write_single_image(image, mask, "bottle", handoffs_root=base / "h")
            again = hio.write_single_image(image, mask, "bottle", handoffs_root=base / "h")
            self.assertEqual(first, again)
            _, manifest, objects = hio.load_handoff(first)
            self.assertEqual(manifest["producer"]["name"], "single_image")
            self.assertEqual([o["object_id"] for o in objects], ["obj_0000"])
            write_frame(base / "other", 3, value=200)
            other = hio.write_single_image(base / "other/0003_rgb.png", base / "other/0003_mask.png",
                                           handoffs_root=base / "h")
            self.assertNotEqual(first, other)  # same file name, different photo


if __name__ == "__main__":
    unittest.main()
