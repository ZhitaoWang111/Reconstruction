"""2D structure segmentation (src.structure.segment): the window's tool routing and rules, without SAM or a model."""
import json

import numpy as np
import pytest
from PIL import Image

from src.structure.segment import window as wn
from src.perception.point_prompt.clicks import MIN_AREA_FRACTION

W, H = 160, 120


class FakeSam:
    """Returns the rectangle spanned by the positive points, grown by 10 px; records each image it is given."""

    def __init__(self):
        import threading
        self.lock = threading.Lock()
        self.images = []

    def set_image(self, img, key):
        self.images.append(key)

    def segment(self, pos, neg):
        m = np.zeros((H, W), bool)
        xs, ys = [p[0] for p in pos], [p[1] for p in pos]
        m[max(0, min(ys) - 10):max(ys) + 11, max(0, min(xs) - 10):max(xs) + 11] = True
        return m, 0.9, 0.8


@pytest.fixture
def win(tmp_path):
    frames = tmp_path / "frames"; frames.mkdir()
    for fid in (10, 20, 30):
        Image.new("RGB", (W, H), (120, 120, 120)).save(frames / f"frame_{fid:06d}.jpg")
    return wn.Window(0, [10, 20, 30], FakeSam(), tmp_path / "run", frames_dir=frames)


def plan(win, *items):
    res, _ = win.call("plan", {"objects": [{"frame": f, "label": l, "where": "x"} for f, l in items]})
    return [p["plan_id"] for p in res["planned"]]


def test_plan_ids_count_across_the_window_and_name_one_frame(win):
    p = plan(win, (10, "wall"), (20, "floor"), (30, "wall"))
    assert p == ["P1", "P2", "P3"]
    assert win.plan_frame == {"P1": 10, "P2": 20, "P3": 30}


def test_labels_outside_the_vocabulary_are_refused_before_any_id_is_given(win):
    with pytest.raises(ValueError, match="not allowed"):
        win.call("plan", {"objects": [{"frame": 10, "label": "wall", "where": "x"},
                                      {"frame": 10, "label": "doorframe", "where": "x"}]})
    assert win.plan_frame == {}                     # the valid item was not planned either


def test_unknown_frame_is_refused(win):
    with pytest.raises(ValueError, match="not a frame of this window"):
        plan(win, (99, "wall"))


def test_segment_groups_by_frame_and_sets_each_image(win):
    p1, p2 = plan(win, (10, "wall"), (20, "wall"))
    res, imgs = win.call("segment", {"items": [
        {"plan_id": p1, "positive": [[20, 20], [60, 60]], "negative": [[40, 40]]},
        {"plan_id": p2, "positive": [[30, 30], [90, 80]], "negative": []}]})
    rows = res["candidates"]
    assert [r["frame"] for r in rows] == [10, 20] and win.cand_frame == {"C1": 10, "C2": 20}
    assert win.sam.images == [10, 20] and len(imgs) == 1
    assert any("negative points inside the mask" in w for w in rows[0]["warnings"])


def test_combine_refuses_candidates_from_another_frame(win):
    p1, p2 = plan(win, (10, "wall"), (20, "wall"))
    win.call("segment", {"items": [{"plan_id": p1, "positive": [[20, 20], [60, 60]], "negative": []},
                                   {"plan_id": p2, "positive": [[20, 20], [60, 60]], "negative": []}]})
    with pytest.raises(ValueError, match="belongs to frame 20"):
        win.call("combine", {"plan_id": p1, "union": ["C1", "C2"], "subtract": [], "reason": "x"})


def test_tiny_masks_cannot_be_kept(win):
    (p,) = plan(win, (10, "window"))
    win.call("segment", {"items": [{"plan_id": p, "positive": [[5, 5]], "negative": []}]})
    win.frames[10].cands["C1"]["area"] = int(MIN_AREA_FRACTION * W * H) - 1     # just under the floor
    with pytest.raises(ValueError, match="below the downstream minimum"):
        win.call("keep", {"candidates": ["C1"]})


def test_finish_needs_every_plan_kept_or_skipped(win):
    p1, p2 = plan(win, (10, "wall"), (30, "floor"))
    win.call("segment", {"items": [{"plan_id": p1, "positive": [[20, 20], [60, 60]], "negative": []}]})
    win.call("keep", {"candidates": ["C1"]})
    with pytest.raises(ValueError, match="neither kept nor skipped"):
        win.call("finish", {"skipped": []})
    res, _ = win.call("finish", {"skipped": [{"plan_id": p2, "reason": "covered by a rug"},
                                             {"plan_id": "-", "reason": "a wardrobe side, not a wall"}]})
    assert res == {"finished": True, "kept": 1, "skipped": 1} and win.not_planned == ["a wardrobe side, not a wall"]


def test_written_result_is_what_lift_reads(win, tmp_path):
    p1, = plan(win, (20, "wall"))
    win.call("segment", {"items": [{"plan_id": p1, "positive": [[20, 20], [100, 80]], "negative": []}]})
    win.call("keep", {"candidates": ["C1"]})
    win.call("finish", {"skipped": []})
    fr = win.frames[20]
    r = fr.write(3, {"prompt_tokens": 1, "completion_tokens": 1}, 1.0)
    on_disk = json.loads((tmp_path / "run/frame_000020/result.json").read_text())
    assert on_disk["frame_id"] == 20 and on_disk["kept"][0]["label"] == "wall"
    mask = np.asarray(Image.open(tmp_path / "run/frame_000020" / on_disk["kept"][0]["mask_file"])) > 0
    assert mask.shape == (H, W) and int(mask.sum()) == r["kept"][0]["mask_pixels"]


def test_defaults_are_the_accepted_run(tmp_path):
    """struct_v4_win5: astra, windows of 5, structure_v4 prompt (sha 659e0824), vocabulary of four."""
    import hashlib
    assert wn.MODEL == "coding-openai/gpt-6-astra" and wn.WINDOW == 5 and wn.MAX_ROUNDS == 30
    assert wn.VOCAB == ("wall", "floor", "door", "window")
    assert hashlib.sha256(wn.PROMPT.read_bytes()).hexdigest().startswith("659e0824")
