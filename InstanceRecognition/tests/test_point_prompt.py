"""Point-prompt object perception (src.perception.point_prompt): tools, rules and source packing, without SAM or a model."""
import json

import numpy as np
import pytest
from PIL import Image

from src.perception.point_prompt import objects as ob
from src.perception.point_prompt import source as src_mod

W, H = 160, 120


class FakeSam:
    def __init__(self):
        import threading
        self.lock = threading.Lock()

    def set_image(self, img, key):
        pass

    def segment(self, pos, neg):
        m = np.zeros((H, W), bool)
        xs, ys = [p[0] for p in pos], [p[1] for p in pos]
        m[max(0, min(ys) - 10):max(ys) + 11, max(0, min(xs) - 10):max(xs) + 11] = True
        return m, 0.9, 0.8


@pytest.fixture
def frame(tmp_path):
    frames = tmp_path / "frames"; frames.mkdir()
    Image.new("RGB", (W, H), (120, 120, 120)).save(frames / "frame_000056.jpg")
    return ob.ObjectFrame(56, FakeSam(), tmp_path / "run/frame_000056", frames_dir=frames)


def test_free_labels_unless_a_vocabulary_is_given(frame, tmp_path):
    res, _ = frame.call("plan", {"objects": [{"label": "tv cabinet", "where": "left"}, {"label": "bookcase", "where": "right"}]})
    assert [p["plan_id"] for p in res["planned"]] == ["P1", "P2"]
    frames = tmp_path / "frames"
    strict = ob.ObjectFrame(56, FakeSam(), tmp_path / "run2", frames_dir=frames, vocab=["chair"])
    with pytest.raises(ValueError, match="not in this run's vocabulary"):
        strict.call("plan", {"objects": [{"label": "sofa", "where": "x"}]})


def test_combine_needs_a_reason_and_pieces_of_one_plan(frame):
    frame.call("plan", {"objects": [{"label": "counter", "where": "a"}, {"label": "stool", "where": "b"}]})
    frame.call("segment", {"items": [{"plan_id": "P1", "positive": [[20, 20]], "negative": []},
                                     {"plan_id": "P1", "positive": [[80, 20]], "negative": []},
                                     {"plan_id": "P2", "positive": [[120, 90]], "negative": []}]})
    with pytest.raises(ValueError, match="why_one_unit is empty"):
        frame.call("combine", {"plan_id": "P1", "union": ["C1", "C2"], "subtract": [], "why_one_unit": " "})
    with pytest.raises(ValueError, match="other plan ids"):
        frame.call("combine", {"plan_id": "P1", "union": ["C1", "C3"], "subtract": [], "why_one_unit": "one worktop"})
    res, _ = frame.call("combine", {"plan_id": "P1", "union": ["C1", "C2"], "subtract": [], "why_one_unit": "one worktop"})
    assert res["candidate"] == "C4" and res["combined_from"] == ["C1", "C2"]


def test_finish_and_written_result(frame, tmp_path):
    frame.call("plan", {"objects": [{"label": "sofa", "where": "a"}, {"label": "lamp", "where": "b"}]})
    frame.call("segment", {"items": [{"plan_id": "P1", "positive": [[30, 30], [90, 70]], "negative": []}]})
    frame.call("keep", {"candidates": ["C1"]})
    with pytest.raises(ValueError, match="neither kept nor skipped"):
        frame.call("finish", {"skipped": []})
    frame.call("finish", {"skipped": [{"plan_id": "P2", "reason": "behind the sofa"}]})
    frame.fr.finished = frame.finished
    r = frame.fr.write(3, {"prompt_tokens": 1, "completion_tokens": 1}, 1.0)
    on_disk = json.loads((tmp_path / "run/frame_000056/result.json").read_text())
    assert on_disk["finished"] and on_disk["kept"][0]["label"] == "sofa" and on_disk["skipped"][0]["label"] == "lamp"
    assert r["kept"][0]["mask_file"] == "masks/001_P1.png"


def test_pack_writes_the_lifting_manifest(frame, tmp_path):
    """One finished frame -> the manifest shape amb3r_mask_source reads, with the uniform-500 slot."""
    frame.call("plan", {"objects": [{"label": "sofa", "where": "a"}]})
    frame.call("segment", {"items": [{"plan_id": "P1", "positive": [[30, 30], [90, 70]], "negative": [[5, 5]]}]})
    frame.call("keep", {"candidates": ["C1"]})
    frame.call("finish", {"skipped": []})
    frame.fr.finished = True
    frame.fr.write(3, {"prompt_tokens": 1, "completion_tokens": 1}, 1.0)
    run_dir = tmp_path / "run"
    (run_dir / "config.json").write_text(json.dumps({"model": "m", "prompt_sha256": "abc"}))
    upstream = tmp_path / "upstream"; upstream.mkdir()
    (upstream / "manifest.json").write_text(json.dumps({"frames": [{"source_frame_id": 0, "sequence_index": 0},
                                                                   {"source_frame_id": 56, "sequence_index": 5}]}))
    with pytest.raises(SystemExit, match="does not match the upstream"):       # frame 56 alone is not slots 0, 5, ...
        src_mod.pack(run_dir, tmp_path / "out", upstream)


def test_defaults_are_the_run_that_fed_harness_v15():
    import hashlib
    assert ob.MODEL == "openai/gpt-6-sol" and ob.MAX_ROUNDS == 24
    assert hashlib.sha256(ob.PROMPT.read_bytes()).hexdigest().startswith("4cde57c3")
    assert [t["function"]["name"] for t in ob.TOOLS] == ["plan", "segment", "zoom", "combine", "keep", "finish"]
