"""CPU checks for instance selection and failure handling before GPU model loading."""
import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from objgen.generate.sam31_mask import select_candidates


class PromptInputTests(unittest.TestCase):
    def test_rank_individual_masks_and_filter_empty(self):
        masks = np.zeros((4, 1, 3, 5), dtype=bool)
        masks[0, 0, 0, 0] = True
        masks[1, 0, 1, 1] = True
        masks[3, 0, 2, 2] = True
        normalized, candidates = select_candidates({
            "out_binary_masks": masks, "out_probs": [.6, .8, .99, .8],
            "out_boxes_xywh": [[.2, 0, .4, 1]] * 4,
        }, 5, 3, .5)
        self.assertEqual([c["model_output_index"] for c in candidates], [1, 3, 0])
        np.testing.assert_allclose(candidates[0]["box_xyxy"], [1., 0., 3., 3.])
        self.assertEqual(int(normalized[candidates[0]["model_output_index"]].sum()), 1)

    def test_reject_invalid_or_missing_masks(self):
        for masks, scores, boxes in [
            ([], [], []),
            (np.ones((1, 2, 2)), [.9], [[0, 0, 1, 1]]),
            (np.ones((1, 3, 5)), [float("nan")], [[0, 0, 1, 1]]),
            (np.ones((1, 3, 5)), [.2], [[0, 0, 1, 1]]),
        ]:
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                select_candidates({"out_binary_masks": masks, "out_probs": scores, "out_boxes_xywh": boxes}, 5, 3, .5)

    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-m", "objgen.generate.sam3d_smoke", *map(str, args)], capture_output=True, text=True,
                              env={**os.environ, "PYTHONPATH": str(ROOT / "src")})

    def test_cli_rejects_conflicting_or_incomplete_inputs(self):
        for args in [
            ["--prompt", "bottle", "--mask", "mask.png"],
            ["--prompt", "   "],
            ["--sam31-threshold", "nan"],
            ["--sam31-instance-index", "-1"],
        ]:
            with self.subTest(args=args):
                self.assertEqual(self.run_cli(*args).returncode, 2)

    def test_failed_segmentation_does_not_start_sam3d(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            image = folder / "input.png"
            Image.new("RGB", (5, 3)).save(image)
            result = self.run_cli("--image", image, "--prompt", "bottle", "--output", folder / "out",
                                  "--sam31-python", sys.executable, "--sam31-checkpoint", folder / "missing.pt")
            self.assertEqual(result.returncode, 1)
            run = next((folder / "out/input").iterdir())
            report = json.loads((run / "sam3d/report.json").read_text())
            self.assertEqual(report["stage"], "segmentation")
            self.assertNotIn("load_seconds", report)
            self.assertFalse((run / "sam3d/mesh.glb").exists())
            failure = json.loads((run / "sam3d/sam31/segmentation.json").read_text())
            self.assertEqual(failure["error_type"], "FileNotFoundError")

    def test_explicit_mask_checked_before_model_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            Image.new("RGB", (5, 3)).save(folder / "input.png")
            Image.new("L", (5, 3), 0).save(folder / "mask.png")
            result = self.run_cli("--image", folder / "input.png", "--mask", folder / "mask.png", "--output", folder / "out")
            self.assertEqual(result.returncode, 1)
            run = next((folder / "out/input").iterdir())
            report = json.loads((run / "sam3d/report.json").read_text())
            self.assertEqual(report["stage"], "input_validation")
            self.assertEqual(report["error"], "Object mask is empty")
            self.assertFalse((run / "sam3d/sam31").exists())
            self.assertEqual((run / "input/rgb.png").read_bytes(), (folder / "input.png").read_bytes())
            self.assertEqual((run / "input/mask.png").read_bytes(), (folder / "mask.png").read_bytes())

    def test_interactive_worker_must_explicitly_confirm(self):
        for status, exitcode in [("success", 0), ("cancelled", 1)]:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                image = folder / "input.png"
                Image.new("RGB", (5, 3)).save(image)
                worker = folder / "fake_python"
                payload = {"status": status, "user_confirmed": False}
                worker.write_text(
                    f"#!{sys.executable}\n"
                    "import sys,json\nfrom pathlib import Path\n"
                    "out=Path(sys.argv[sys.argv.index('--output')+1])\n"
                    f"(out/'segmentation.json').write_text(json.dumps({payload!r}))\n"
                    f"sys.exit({exitcode})\n")
                worker.chmod(0o755)
                result = self.run_cli("--image", image, "--output", folder / "out", "--sam31-python", worker)
                self.assertEqual(result.returncode, 1)
                run = next((folder / "out/input").iterdir())
                report = json.loads((run / "sam3d/report.json").read_text())
                self.assertEqual(report["selection_mode"], "points")
                self.assertEqual(report["stage"], "segmentation")
                self.assertNotIn("load_seconds", report)
                self.assertFalse((run / "sam3d/mesh.glb").exists())
                if status == "success":
                    self.assertIn("was not confirmed", report["error"])


if __name__ == "__main__":
    unittest.main()
