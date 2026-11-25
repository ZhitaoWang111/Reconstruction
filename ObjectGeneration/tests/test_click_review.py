"""Test point review gates and HTTP interactions without loading a GPU model."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from objgen.generate.sam31_click import PointModel, Review, ReviewError, make_server, validate_points
from objgen.run_pipeline import parse_args, sam3d_command


class FakeModel:
    def segment(self, points):
        mask = Image.new("L", (100, 80))
        for point in points:
            x, y = point["x"] * 99, point["y"] * 79
            ImageDraw.Draw(mask).ellipse((x-12, y-12, x+12, y+12), fill=255 if point["label"] else 0)
        return mask

    def close(self):
        pass


class ClickReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.review = Review(Image.new("RGB", (100, 80), "white"), Path(self.tmp.name))
        self.worker = threading.Thread(target=self.review.work, args=(FakeModel,))
        self.worker.start()
        self.wait_for("ready")

    def tearDown(self):
        self.review.cancel()
        self.worker.join(timeout=3)
        self.assertFalse(self.worker.is_alive())
        self.tmp.cleanup()

    def wait_for(self, phase):
        deadline = time.monotonic() + 3
        while self.review.public()["phase"] != phase and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertEqual(self.review.public()["phase"], phase)

    def test_no_prompt_selects_interactive_route(self):
        args = parse_args(["--image", "image.png"])
        self.assertIsNone(args.prompt)
        self.assertIsNone(args.mask)
        self.assertNotIn("--prompt", sam3d_command(args))

    def test_point_adapter_initializes_cache_and_resets_in_inference_mode(self):
        import numpy as np
        import torch
        from types import SimpleNamespace
        with torch.inference_mode():
            tensor = torch.ones(1)
        state = {"cached_frame_outputs": {0: {1: "old mask"}}}
        def reset(current):
            tensor.zero_()  # Like upstream: in-place mutation of an inference tensor.
            current["cached_frame_outputs"].clear()
        def request(data):
            self.assertEqual(state["cached_frame_outputs"], {0: {}})
            self.assertEqual(data["points"], [[.3, .4], [.7, .8]])
            self.assertEqual(data["point_labels"], [1, 0])
            self.assertTrue(data["rel_coordinates"])
            return {"outputs": {"out_obj_ids": np.array([1]),
                                "out_binary_masks": np.ones((1, 1, 80, 100), dtype=bool)}}
        model = PointModel.__new__(PointModel)
        model.state, model.session, model.image = state, "test", self.review.image
        model.predictor = SimpleNamespace(model=SimpleNamespace(reset_state=reset), handle_request=request)
        mask = model.segment([{"x":.3,"y":.4,"label":1}, {"x":.7,"y":.8,"label":0}])
        self.assertEqual(mask.getbbox(), (0, 0, 100, 80))

    def test_invalid_points_are_rejected(self):
        for points in [None, [{"x":float("nan"),"y":.5,"label":1}],
                       [{"x":1.1,"y":.5,"label":1}], [{"x":.5,"y":.5,"label":2}],
                       [{"x":.5,"y":.5,"label":0}], [{"x":True,"y":.5,"label":1}]]:
            with self.subTest(points=points), self.assertRaises(ReviewError):
                validate_points(points)

    def test_confirm_requires_latest_preview_then_freezes_selection(self):
        with self.assertRaises(ReviewError):
            self.review.confirm(0)
        self.review.submit([{"x":.5,"y":.5,"label":1}], 0)
        self.wait_for("preview")
        with self.assertRaises(ReviewError):
            self.review.confirm(1)  # Ready mask, but not yet displayed.
        self.assertTrue(self.review.preview(1).startswith(b'\x89PNG'))
        self.review.submit([{"x":.5,"y":.5,"label":1},{"x":.58,"y":.5,"label":0}], 1)
        with self.assertRaises(ReviewError):
            self.review.confirm(1)
        self.wait_for("preview")
        with self.assertRaises(ReviewError):
            self.review.confirm(2)
        self.review.preview(2)
        self.review.confirm(2)
        self.assertTrue(self.review.done.is_set())
        self.assertEqual(self.review.phase, "confirmed")
        with self.assertRaises(ReviewError):
            self.review.submit([], 2)
        saved = json.loads((Path(self.tmp.name)/"interaction.json").read_text())
        self.assertEqual(saved['history'][-1]['action'], 'confirmed')
        self.assertEqual(len(saved['points']), 2)

    def test_clear_invalidates_mask_and_cancel_does_not_confirm(self):
        self.review.submit([{"x":.5,"y":.5,"label":1}], 0)
        self.wait_for("preview")
        self.review.preview(1)
        self.review.submit([], 1)
        self.assertIsNone(self.review.mask)
        with self.assertRaises(ReviewError):
            self.review.confirm(2)
        self.review.cancel()
        self.assertIsNone(self.review.confirmed_at)
        self.assertEqual(self.review.phase, "cancelled")

    def test_old_inference_cannot_restore_mask_after_clear(self):
        self.review.cancel()
        self.worker.join()
        entered, release = threading.Event(), threading.Event()
        class SlowModel(FakeModel):
            def segment(self, points):
                entered.set()
                release.wait(timeout=2)
                return super().segment(points)
        self.review = Review(Image.new('RGB',(100,80)), Path(self.tmp.name))
        self.worker = threading.Thread(target=self.review.work, args=(SlowModel,))
        self.worker.start()
        self.wait_for('ready')
        self.review.submit([{"x":.5,"y":.5,"label":1}],0)
        self.assertTrue(entered.wait(timeout=2))
        self.review.submit([],1)
        release.set()
        self.review.cancel()
        self.worker.join(timeout=2)
        self.assertIsNone(self.review.mask)
        self.assertFalse(self.review.public()['can_confirm'])

    def test_http_page_preview_gate_and_cancel(self):
        server, url = make_server(self.review)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        def post(route, data, origin=None):
            headers = {'Content-Type':'application/json'}
            if origin:
                headers['Origin'] = origin
            request = urllib.request.Request(url+route, data=json.dumps(data).encode(), headers=headers)
            with urllib.request.urlopen(request) as response:
                return json.load(response)
        try:
            with urllib.request.urlopen(url) as response:
                self.assertIn('确认分割，继续生成', response.read().decode())
            with self.assertRaises(urllib.error.HTTPError) as error:
                post('points',{'revision':0,'points':[]},'https://unrelated.example')
            self.assertEqual(error.exception.code,403)
            post('points',{'revision':0,'points':[{'x':.5,'y':.5,'label':1}]})
            self.wait_for('preview')
            with self.assertRaises(urllib.error.HTTPError):
                post('confirm',{'revision':1})
            with urllib.request.urlopen(url+'preview?revision=1') as response:
                self.assertEqual(response.headers['Content-Type'],'image/png')
            self.assertTrue(self.review.public()['can_confirm'])
            self.assertEqual(post('cancel',{})['phase'],'cancelled')
            with self.assertRaises(urllib.error.HTTPError):
                post('confirm',{'revision':1})
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
