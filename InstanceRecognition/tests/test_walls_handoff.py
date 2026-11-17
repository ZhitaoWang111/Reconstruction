"""Wall handoff (walls/v1): the checker catches each broken promise, and the real handoff holds them."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.handoff import schema, walls_check, walls_schema as ws

REAL = Path(__file__).resolve().parents[1] / "artifacts/handoff/scene0000_00/walls_harness"


def _pose(yaw=0.0, centre=(0.0, 0.0, 1.3)):
    """OpenCV camera looking horizontally along world +X turned by yaw about +Z."""
    c, s = np.cos(yaw), np.sin(yaw)
    forward = np.array([c, s, 0.0]); down = np.array([0.0, 0.0, -1.0])
    T = np.eye(4); T[:3, :3] = np.column_stack([np.cross(down, forward), down, forward]); T[:3, 3] = centre
    return T.tolist()


def _wall(wid, a, b, t, role, joins):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = (b - a) / np.linalg.norm(b - a)
    back = np.array([d[1], -d[0]])                     # right of start->end: the room is on the left
    return {"id": wid, "role": role, "start_floor": a.tolist(), "end_floor": b.tolist(),
            "length_m": float(np.linalg.norm(b - a)), "normal_to_back": back.tolist(),
            "thickness_m": t, "thickness_kind": "assumed", "height_m": 2.8,
            "footprint_floor": [a.tolist(), b.tolist(), (b + t * back).tolist(), (a + t * back).tolist()],
            "segments": [{"from_m": 0.0, "to_m": float(np.linalg.norm(b - a)), "kind": "observed",
                          "source_planes": ["W1"]}],
            "joins": joins, "openings": [], "key_frames": [], "evidence": {}}


def _square(tmp_path, size=4.0, t=0.3):
    """A closed 4 m square room, exterior corners mitred, one key frame on the south wall."""
    s = size
    inner = [(0, 0), (s, 0), (s, s), (0, s)]
    outer = [(-t, -t), (s + t, -t), (s + t, s + t), (-t, s + t)]
    ids = ["ext_S", "ext_E", "ext_N", "ext_W"]
    walls = []
    for i, wid in enumerate(ids):
        w = _wall(wid, inner[i], inner[(i + 1) % 4], t, "exterior",
                  {"start": {"wall": ids[i - 1], "type": "mitre"}, "end": {"wall": ids[(i + 1) % 4], "type": "mitre"}})
        w["footprint_floor"] = [list(inner[i]), list(inner[(i + 1) % 4]), list(outer[(i + 1) % 4]), list(outer[i])]
        if wid != "ext_S":                                 # only the south wall was seen; the rest is closure
            w["segments"][0].update(kind="inferred", source_planes=[])
        walls.append(w)
    width, height = 64, 48
    (tmp_path / "keyframes/ext_S").mkdir(parents=True)
    mask = np.zeros((height, width), np.uint8); mask[10:30, 20:40] = 255
    Image.fromarray(mask).save(tmp_path / "keyframes/ext_S/0001_mask.png")
    for kind in ("rgb", "boxed"):
        Image.new("RGB", (width, height)).save(tmp_path / f"keyframes/ext_S/0001_{kind}.jpg")
    walls[0]["key_frames"] = [{
        "role": "primary", "seq": 1, "source_frame_id": 11, "rgb": "keyframes/ext_S/0001_rgb.jpg",
        "mask": "keyframes/ext_S/0001_mask.png", "boxed": "keyframes/ext_S/0001_boxed.jpg",
        "bbox_xyxy": [20, 10, 40, 30], "bbox_convention": schema.BBOX_CONVENTION, "visible_pixels": 400,
        "covers_m": [1.0, 3.0], "view_angle_deg": 5.0, "distance_m": 2.0,
        "camera_to_world": _pose(-np.pi / 2, (2.0, 2.0, 1.3))}]
    doc = {"schema_version": ws.SCHEMA_VERSION, "scene_id": "test", "producer": {"name": "test"},
           "coordinate_frame": "scannet_scan", "units": "meters", "up_axis": "Z",
           "floor_frame_to_scan": np.eye(4).tolist(),
           "camera": schema.camera(width, height, 50.0, 50.0, 31.5, 23.5, source="test"),
           "storey": {"height_m": 2.8, "kind": "assumed"},
           "thickness_defaults": {"exterior_m": t, "interior_m": 0.15, "kind": "assumed"},
           "outline": {"shape": "rectangle", "inner_corners_floor": [list(p) for p in inner],
                       "outer_corners_floor": [list(p) for p in outer]},
           "walls": walls, "not_included": list(ws.NOT_INCLUDED), "known_issues": []}
    (tmp_path / "walls.json").write_text(json.dumps(doc))
    return doc


def _rewrite(tmp_path, doc):
    (tmp_path / "walls.json").write_text(json.dumps(doc))
    return walls_check.check(tmp_path)


def test_square_room_passes(tmp_path):
    _square(tmp_path)
    assert walls_check.check(tmp_path) == []


def test_room_on_the_wrong_side_is_caught(tmp_path):
    doc = _square(tmp_path)
    w = doc["walls"][1]
    w["start_floor"], w["end_floor"] = w["end_floor"], w["start_floor"]
    assert any("room is not on the left" in p for p in _rewrite(tmp_path, doc))


def test_open_exterior_corner_is_caught(tmp_path):
    doc = _square(tmp_path)
    doc["walls"][1]["start_floor"] = [4.0, 0.2]                 # pulled 20 cm off the south wall's end
    doc["walls"][1]["length_m"] = 3.8
    doc["walls"][1]["segments"][0]["to_m"] = 3.8
    problems = _rewrite(tmp_path, doc)
    assert any("room-side corner is open" in p for p in problems)
    assert any("off ext_E" in p or "off ext_S" in p or "corner" in p for p in problems)


def test_segments_must_cover_the_wall(tmp_path):
    doc = _square(tmp_path)
    doc["walls"][0]["segments"] = [{"from_m": 0.0, "to_m": 1.5, "kind": "inferred", "source_planes": []},
                                   {"from_m": 2.0, "to_m": 4.0, "kind": "observed", "source_planes": ["W1"]}]
    assert any("gap or overlap" in p for p in _rewrite(tmp_path, doc))


def test_observed_segment_names_its_planes(tmp_path):
    doc = _square(tmp_path)
    doc["walls"][0]["segments"][0]["source_planes"] = []
    assert any("without source_planes" in p for p in _rewrite(tmp_path, doc))


def test_footprint_must_match_thickness(tmp_path):
    doc = _square(tmp_path)
    doc["walls"][2]["thickness_m"] = 0.2                        # footprint still drawn 0.3 deep
    assert any("thickness_m behind" in p for p in _rewrite(tmp_path, doc))


def test_join_to_unknown_wall_is_caught(tmp_path):
    doc = _square(tmp_path)
    doc["walls"][0]["joins"]["end"]["wall"] = "ext_X"
    assert any("unknown wall" in p for p in _rewrite(tmp_path, doc))


def _opening(w, a0, a1, z0, z1):
    """A measured opening on a wall, in the shape the producer writes."""
    a = np.asarray(w["start_floor"], float); b = np.asarray(w["end_floor"], float)
    d = (b - a) / np.linalg.norm(b - a)
    quad = [[*(a + s * d), z] for s, z in ((a0, z0), (a1, z0), (a1, z1), (a0, z1))]
    return {"from_m": a0, "to_m": a1, "z_floor": [z0, z1], "width_m": round(a1 - a0, 3),
            "height_m": round(z1 - z0, 3), "kind": "opening", "source": "ray_test", "rays_through": 500,
            "closed_above": True, "corners_floor": [[round(v, 5) for v in q] for q in quad],
            "corners_scan": [[round(v, 5) for v in q] for q in quad], "note": "rays see through"}


def test_a_measured_opening_passes(tmp_path):
    """The fixture's floor frame is the scan frame, so the two corner sets coincide."""
    doc = _square(tmp_path)
    doc["walls"][0]["openings"] = [_opening(doc["walls"][0], 1.0, 1.9, 0.0, 2.0)]
    assert _rewrite(tmp_path, doc) == []


@pytest.mark.parametrize("change,message", [
    ({"from_m": 3.5, "to_m": 4.5}, "outside the wall"),
    ({"z_floor": [0.5, 3.5], "height_m": 3.0}, "not inside"),
    ({"width_m": 0.5}, "width_m disagrees"),
    ({"kind": "door"}, "kind must be opening"),
    ({"source": "guess"}, "kind must be opening"),
    ({"corners_floor": [[0, 1, 0]] * 4}, "not on the wall's face"),        # 1 m off the south wall's line
    ({"corners_scan": [[0, 0, 0]] * 4}, "corners_scan is not corners_floor"),
])
def test_a_bad_opening_is_caught(tmp_path, change, message):
    doc = _square(tmp_path)
    op = _opening(doc["walls"][0], 1.0, 1.9, 0.0, 2.0)
    op.update(change)
    if "from_m" in change:
        op["width_m"] = round(op["to_m"] - op["from_m"], 3)   # keep only the extent wrong
    doc["walls"][0]["openings"] = [op]
    assert any(message in p for p in _rewrite(tmp_path, doc))


def test_overlapping_openings_are_caught(tmp_path):
    doc = _square(tmp_path)
    doc["walls"][0]["openings"] = [_opening(doc["walls"][0], 1.0, 2.0, 0.0, 2.0),
                                   _opening(doc["walls"][0], 1.5, 2.5, 0.0, 2.0)]
    assert any("overlap" in p for p in _rewrite(tmp_path, doc))


@pytest.mark.parametrize("field,value,message", [
    ("visible_pixels", 399, "visible_pixels"),
    ("bbox_xyxy", [20, 10, 41, 30], "bbox_xyxy"),
    ("covers_m", [1.0, 4.5], "covers_m"),
    ("camera_to_world", (np.eye(4) * 2).tolist(), "rigid"),
])
def test_key_frame_records_agree_with_their_files(tmp_path, field, value, message):
    doc = _square(tmp_path)
    doc["walls"][0]["key_frames"][0][field] = value
    assert any(message in p for p in _rewrite(tmp_path, doc))


def test_key_frame_roles_in_order(tmp_path):
    doc = _square(tmp_path)
    doc["walls"][0]["key_frames"][0]["role"] = "secondary"
    assert any("roles" in p for p in _rewrite(tmp_path, doc))


def test_missing_key_frame_image_is_caught(tmp_path):
    _square(tmp_path)
    (tmp_path / "keyframes/ext_S/0001_rgb.jpg").unlink()
    assert any("rgb image missing" in p for p in walls_check.check(tmp_path))


def test_convention_fields_match_the_object_handoff(tmp_path):
    doc = _square(tmp_path)
    doc["camera"]["extrinsics"] = {**schema.EXTRINSICS_CONVENTION, "axes": "opengl"}
    assert any("extrinsics convention" in p for p in _rewrite(tmp_path, doc))


# ---------------------------------------------------------------- the real handoff
real = pytest.mark.skipif(not (REAL / "walls.json").is_file(), reason="wall handoff not built")


@real
def test_real_handoff_passes_the_checker():
    assert walls_check.check(REAL) == []


@real
def test_real_key_frame_masks_land_on_their_wall():
    """Independent of the producer: cast every mask pixel's ray through the stored pose and
    intrinsics onto the plane this frame saw, and it must land on the wall - within its length,
    between floor and storey height - and where `covers_m` says. A wrong pose, the wrong
    intrinsics, a mask from another wall or a mask spilling onto furniture all fail this.

    The reverse test (project the wall face, expect it inside the mask) does not work:
    furniture in front of the wall is correctly left out of the mask."""
    doc = json.loads((REAL / "walls.json").read_text())
    Minv = np.linalg.inv(np.asarray(doc["floor_frame_to_scan"])); K = np.asarray(doc["camera"]["K"])
    rng = np.random.default_rng(0)
    for w in doc["walls"]:
        a = np.asarray(w["start_floor"]); d = (np.asarray(w["end_floor"]) - a) / w["length_m"]
        back = np.asarray(w["normal_to_back"])
        for k in w["key_frames"]:
            vs, us = np.nonzero(np.asarray(Image.open(REAL / k["mask"])) > 0)
            pick = rng.choice(len(us), min(4000, len(us)), replace=False); us, vs = us[pick], vs[pick]
            T = np.asarray(k["camera_to_world"])
            rays = np.c_[(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones(len(us))] @ T[:3, :3].T
            o = Minv[:3, :3] @ T[:3, 3] + Minv[:3, 3]; r = rays @ Minv[:3, :3].T
            face = a - list(k["plane_offset_m"].values())[0] * back
            t = ((face - o[:2]) @ back) / (r[:, :2] @ back)
            X = o + t[:, None] * r
            s, z = (X[:, :2] - a) @ d, X[:, 2]
            on = (t > 0) & (s > -0.2) & (s < w["length_m"] + 0.2) & (z > -0.2) & (z < w["height_m"] + 0.2)
            tag = (w["id"], k["source_frame_id"])
            assert on.mean() > 0.9, (*tag, round(on.mean(), 3))
            lo, hi = np.percentile(s[on], [2, 98])
            assert abs(lo - k["covers_m"][0]) < 0.25 and abs(hi - k["covers_m"][1]) < 0.25, (*tag, lo, hi, k["covers_m"])


@real
def test_real_handoff_marks_what_is_assumed():
    doc = json.loads((REAL / "walls.json").read_text())
    assert doc["storey"]["kind"] == "assumed" and doc["storey"]["height_m"] == pytest.approx(2.8)
    t = {w["role"]: w["thickness_m"] for w in doc["walls"]}
    assert t == {"exterior": 0.3, "interior": 0.15}
    assert all(w["thickness_kind"] == "assumed" for w in doc["walls"])
    inferred = {w["id"]: w["evidence"]["inferred_m"] / w["length_m"] for w in doc["walls"]}
    assert inferred["ext_W"] > 0.7 and inferred["ext_S"] < 0.1


@real
@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "pagegen/templates/walls_handoff/page.js").is_file(),
                    reason="page templates are not under version control")
def test_page_builder_publishes_every_key_frame(tmp_path):
    from pagegen import build_walls_handoff as page
    page.build(REAL, tmp_path / "page", rays=tmp_path / "no_rays.npz")          # no rays: no background
    text = (tmp_path / "page/walls.js").read_text()
    payload = json.loads(text[len("window.WALLS="):-2])
    assert payload["background"] is None
    frames = [k for w in payload["doc"]["walls"] for k in w["key_frames"]]
    assert frames and all((tmp_path / "page" / k[f]).is_file() for k in frames for f in ("rgb", "mask", "boxed"))
    assert {"index.html", "page.js", "page.css"} <= {p.name for p in (tmp_path / "page").iterdir()}
    assert json.loads((REAL / "walls.json").read_text())["walls"][0]["key_frames"][0]["rgb"].startswith("keyframes/")


@real
def test_real_handoff_carries_the_measured_opening():
    doc = json.loads((REAL / "walls.json").read_text())
    holes = [(w["id"], o) for w in doc["walls"] for o in w["openings"]]
    assert len(holes) == 1
    wid, o = holes[0]
    assert (wid, o["kind"], o["source"]) == ("int_3", "opening", "ray_test")
    assert o["width_m"] == pytest.approx(0.6, abs=0.05) and o["height_m"] == pytest.approx(1.7, abs=0.05)
    assert o["closed_above"] is True
    assert "openings" not in doc["not_included"]
