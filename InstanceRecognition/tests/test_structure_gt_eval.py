"""tools.structure.gt_eval: the pieces on synthetic inputs, and the real baseline when it exists."""
import json

import numpy as np
import pytest

from tools.structure import gt_eval as ge

FACE = {"key": "F1", "name": "south", "c": [0.0, 0.0], "n": [0.0, 1.0], "u": [1.0, 0.0], "a": [-3.0, 3.0],
        "area_m2": 15.0, "z": [0.0, 2.5]}


def _plane(pid, offset, a=(-2.0, 2.0), heading_deg=90.0):
    t = np.radians(heading_deg)
    n = [np.cos(t), np.sin(t)]
    return {"id": pid, "point": [0.0, offset], "normal": n, "along": [-n[1], n[0]], "a": list(a)}


def test_match_gt_takes_the_face_the_plane_lies_on():
    m = ge.match_gt(_plane("W1", 0.08), [FACE])
    assert m["face"] == "F1" and m["offset_cm"] == pytest.approx(8.0) and m["overlap_m"] == pytest.approx(4.0)
    assert ge.match_gt(_plane("W2", 0.6), [FACE]) is None                 # too far in front
    assert ge.match_gt(_plane("W3", 0.0, heading_deg=-90), [FACE]) is None  # facing the other way


def test_planes_report_faces_without_a_plane():
    other = {**FACE, "key": "F2", "c": [5.0, 5.0]}
    W = {"walls": [_plane("W1", 0.05)], "rejected": [_plane("R1", 5.0)]}
    out = ge.eval_planes(W, [FACE, other])
    assert out["faces_with_plane"] == 1 and out["kept_on_no_face"] == []
    f1, f2 = out["faces"]
    assert f1["planes"] == ["W1"] and f1["offset_cm"] == pytest.approx(5.0) and f1["length_covered"] == pytest.approx(4 / 6, abs=1e-4)
    assert f2["planes"] == [] and f2["offset_cm"] is None


def test_room_size_is_the_gap_between_the_matched_faces():
    faces = [{**FACE, "key": k, "c": c, "n": n} for k, c, n in
             (("FW", [-3.0, 0.0], [1.0, 0.0]), ("FE", [3.5, 0.0], [-1.0, 0.0]),
              ("FS", [0.0, -3.0], [0.0, 1.0]), ("FN", [0.0, 3.2], [0.0, -1.0]))]
    planes = {"matches": {"A": {"face": "FW", "name": "w"}, "B": {"face": "FS", "name": "s"},
                          "C": {"face": "FE", "name": "e"}, "D": {"face": "FN", "name": "n"}}}
    sides = [{"side": s, "k": k, "from": "observed", "walls": [w], "normal": nn, "offset": off}
             for s, k, w, nn, off in (("W", 0, "A", [1, 0], -2.9), ("S", 1, "B", [0, 1], -2.9),
                                      ("E", 2, "C", [-1, 0], -3.4), ("N", 3, "D", [0, -1], -3.1))]
    C = {"sides": sides, "rectangle": {"width_m": 6.3, "depth_m": 6.0, "area_m2": 37.8, "seen_outside_share": 0.0}}
    out = ge.eval_room(C, planes, faces)
    assert {x["axis"]: x["diff_cm"] for x in out["size"]} == {"W-E": pytest.approx(-20.0), "S-N": pytest.approx(-20.0)}
    assert {x["side"]: x["inward_cm"] for x in out["sides"]} == {"W": pytest.approx(10.0), "S": pytest.approx(10.0),
                                                                 "E": pytest.approx(10.0), "N": pytest.approx(10.0)}


def test_compare_lists_what_moved(capsys):
    base = {"run": "a", "commit": "x", "headline": {"seg.wall.purity": 0.9, "openings.gt_found": 1}}
    cur = {"run": "a", "commit": "y", "headline": {"seg.wall.purity": 0.9, "openings.gt_found": 3}}
    ge.compare(cur, base)
    out = capsys.readouterr().out
    assert "openings.gt_found" in out and "seg.wall.purity" not in out and "1 of 2" in out


real = pytest.mark.skipif(not ge.BASELINE.is_file(), reason="no structure GT baseline saved")


@real
def test_real_baseline_has_every_headline():
    doc = json.loads(ge.BASELINE.read_text())
    h = doc["headline"]
    for k in ("seg.wall.purity", "coverage.masks", "planes.faces_with_plane", "planes.mean_abs_offset_cm",
              "height.mean_abs_storey_error_cm", "openings.gt_found", "openings.gt_total"):
        assert k in h and h[k] is not None
    assert h["openings.gt_total"] == 4 and h["planes.faces_total"] == 9
