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


def _fake_gt(monkeypatch, pts):
    pts = np.asarray(pts, float)
    monkeypatch.setattr(ge, "gt_surface", lambda: {"points": pts, "label": np.array(["wall"] * len(pts)),
                                                   "instance": np.zeros(len(pts), int), "wall": np.ones(len(pts), bool),
                                                   "per_m2": 2000})


def _sheet(y, x=(0.0, 4.0), z=(0.0, 2.5), step=0.02):
    """GT wall points on the plane y = const."""
    X, Z = np.meshgrid(np.arange(*x, step), np.arange(*z, step))
    return np.c_[X.ravel(), np.full(X.size, y), Z.ravel()]


def test_plane_offset_is_measured_against_the_gt_points(monkeypatch):
    _fake_gt(monkeypatch, _sheet(-0.08))               # GT wall 8 cm behind our plane at y = 0 (room on +y)
    W = {"walls": [{"id": "W1", "point": [2.0, 0.0], "normal": [0.0, 1.0], "along": [-1.0, 0.0], "a": [-1.5, 1.5], "z": [0.1, 2.3]}],
         "rejected": [{"id": "R1", "point": [2.0, 3.0], "normal": [0.0, 1.0], "along": [-1.0, 0.0], "a": [-1.0, 1.0], "z": [0.1, 2.0]}]}
    out = ge.eval_planes(W)
    w1, r1 = out["planes"]
    assert w1["offset_cm"] == pytest.approx(8.0, abs=0.5) and r1["offset_cm"] is None
    assert out["kept_without_gt"] == [] and out["mean_abs_offset_cm"] == pytest.approx(8.0, abs=0.5)


def test_room_sides_are_measured_against_the_gt_points(monkeypatch):
    # GT room 0..4 x 0..3, our rectangle 0.1..3.9 x 0.1..2.9: every side 10 cm inside
    pts = np.r_[_sheet(0.0), _sheet(3.0), _sheet(0.0, x=(0, 3)) @ np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
                _sheet(4.0, x=(0, 3)) @ np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])]
    _fake_gt(monkeypatch, pts)
    sides = [{"side": "W", "from": "observed", "p0": [0.1, 2.9], "p1": [0.1, 0.1], "normal": [1.0, 0.0]},
             {"side": "S", "from": "observed", "p0": [0.1, 0.1], "p1": [3.9, 0.1], "normal": [0.0, 1.0]},
             {"side": "E", "from": "observed", "p0": [3.9, 0.1], "p1": [3.9, 2.9], "normal": [-1.0, 0.0]},
             {"side": "N", "from": "observed", "p0": [3.9, 2.9], "p1": [0.1, 2.9], "normal": [0.0, -1.0]}]
    C = {"sides": sides, "rectangle": {"width_m": 3.8, "depth_m": 2.8, "area_m2": 10.64, "seen_outside_share": 0.0}}
    out = ge.eval_room(C)
    assert all(x["inward_cm"] == pytest.approx(10.0, abs=1.0) for x in out["sides"])
    assert {x["axis"]: round(x["gt_m"], 2) for x in out["size"]} == {"W-E": 4.0, "S-N": 3.0}


def test_compare_lists_what_moved(capsys):
    base = {"run": "a", "commit": "x", "headline": {"wall.on_face": 0.9, "openings.gt_found": 1}}
    cur = {"run": "a", "commit": "y", "headline": {"wall.on_face": 0.9, "openings.gt_found": 3}}
    ge.compare(cur, base)
    out = capsys.readouterr().out
    assert "openings.gt_found" in out and "wall.on_face" not in out and "1 of 2" in out


real = pytest.mark.skipif(not ge.BASELINE.is_file(), reason="no structure GT baseline saved")


@real
def test_real_baseline_has_every_headline():
    doc = json.loads(ge.BASELINE.read_text())
    h = doc["headline"]
    for k in ("wall.in_solid", "wall.on_face", "wall.support", "wall.median_offset_cm",
              "wall.mean_abs_height_error_cm", "chain.rays", "chain.masks", "openings.gt_found", "openings.gt_total"):
        assert k in h and h[k] is not None
    assert h["openings.gt_total"] == 4 and h["planes.kept_without_gt"] == 0
    assert h["chain.rays"] >= h["chain.masks"] >= h["chain.planes"]       # evidence only gets lost down the chain
    assert "F1" not in json.dumps(doc)                                     # no GT face partition anywhere


def _room(tmp_wall_h=2.5):
    """One straight wall along x, room on +y: start (0,0) -> end (4,0), thickness 0.2 behind (-y)."""
    return {"walls": [{"id": "ext_S", "start_floor": [0.0, 0.0], "end_floor": [4.0, 0.0], "normal_to_back": [0.0, -1.0],
                       "thickness_m": 0.2, "height_m": tmp_wall_h,
                       "footprint_floor": [[0, 0], [4, 0], [4, -0.2], [0, -0.2]],
                       "openings": [{"from_m": 1.0, "to_m": 2.0, "z_floor": [0.0, 2.0]}]}]}


def test_solids_measure_inside_face_and_hole():
    S = ge.Solids(_room())
    X = np.array([[0.5, -0.1, 1.0],      # inside the wall
                  [0.5, 0.05, 1.0],      # 5 cm into the room, in front of the face
                  [1.5, -0.1, 1.0],      # inside the opening
                  [0.5, -0.1, 2.7]])     # above the wall top
    d, which, hole = S.distance(X)
    assert d[0] == 0 and d[1] == pytest.approx(0.05) and hole.tolist() == [False, False, True, False]
    assert d[3] == pytest.approx(0.2)
    f = S.face_distance(X)
    assert f[1] == pytest.approx(0.05) and f[0] == pytest.approx(0.1) and np.isinf(f[2])
    pts, idx = S.face_samples()
    assert not ((pts[:, 0] > 1.0) & (pts[:, 0] < 2.0) & (pts[:, 2] < 2.0)).any()     # nothing sampled in the hole
    assert pts[:, 2].max() < 2.5 and np.allclose(pts[:, 1], 0)
