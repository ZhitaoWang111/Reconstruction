"""Handoff contract: camera intrinsics, extrinsics, floor_z and the amb3r_floor frame block."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.handoff import check, schema
from src.handoff.from_gt import floor_height, read_scene_meta


def _pose(yaw=0.0, centre=(1.0, 2.0, 1.5)):
    """OpenCV camera looking horizontally along world +X (rotated by yaw about Z)."""
    c, s = np.cos(yaw), np.sin(yaw)
    forward = np.array([c, s, 0.0])
    down = np.array([0.0, 0.0, -1.0])
    right = np.cross(down, forward)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([right, down, forward])
    pose[:3, 3] = centre
    return pose


def _write_ply(path, points):
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\nend_header\n")
    path.write_bytes(header.encode() + np.asarray(points, "<f4").tobytes())


def _handoff(tmp_path, width=64, height=48):
    camera = schema.camera(width, height, 50.0, 50.0, 31.5, 23.5, source="test")
    (tmp_path / "clouds").mkdir()
    (tmp_path / "keyframes/obj_0001").mkdir(parents=True)
    _write_ply(tmp_path / "clouds/obj_0001.ply", [[3.0, 2.0, 1.5]] * 4)
    mask = np.zeros((height, width), np.uint8)
    mask[20:28, 28:36] = 255
    Image.fromarray(mask).save(tmp_path / "keyframes/obj_0001/0000_mask.png")
    for name in ("rgb", "boxed"):
        Image.fromarray(np.zeros((height, width, 3), np.uint8)).save(
            tmp_path / f"keyframes/obj_0001/0000_{name}.jpg")
    frame = schema.empty_frame()
    frame.update({"rgb": "keyframes/obj_0001/0000_rgb.jpg", "mask": "keyframes/obj_0001/0000_mask.png",
                  "boxed": "keyframes/obj_0001/0000_boxed.jpg", "bbox_xyxy": [28, 20, 36, 28],
                  "visible_pixels": 64, "camera_to_world": _pose().tolist()})
    record = schema.empty_object("obj_0001")
    record.update({"label": "chair", "cloud": "clouds/obj_0001.ply", "point_count": 4,
                   "key_frames": [frame]})
    frame = {"definition": "docs/amb3r-floor-frame.md", "file": "test", "native_to_frame": np.eye(4).tolist(),
             "scale": {"value": 1.0, "source": "native"}, "first_frame": {"amb3r_index": 0}}
    manifest = schema.manifest("scene_test", {"name": "test"}, objects=1, camera=camera,
                               floor_z=0.0, frame=frame)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "objects.json").write_text(json.dumps([record]))
    return tmp_path


def _edit(directory, which, change):
    path = directory / f"{which}.json"
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def test_complete_handoff_passes(tmp_path):
    assert check.check(_handoff(tmp_path)) == []


def test_camera_to_world_is_required(tmp_path):
    directory = _handoff(tmp_path)
    _edit(directory, "objects", lambda d: d[0]["key_frames"][0].pop("camera_to_world"))
    assert any("missing ['camera_to_world']" in p for p in check.check(directory))


@pytest.mark.parametrize("pose", [None, np.eye(3).tolist(), (np.eye(4) * 2).tolist(),
                                  np.diag([1.0, 1.0, -1.0, 1.0]).tolist()])
def test_camera_to_world_must_be_rigid(tmp_path, pose):
    directory = _handoff(tmp_path)
    _edit(directory, "objects", lambda d: d[0]["key_frames"][0].update(camera_to_world=pose))
    assert any("camera_to_world" in p for p in check.check(directory))


@pytest.mark.parametrize("key", ["camera", "floor_z", "frame"])
def test_manifest_scene_fields_are_required(tmp_path, key):
    directory = _handoff(tmp_path)
    _edit(directory, "manifest", lambda d: d.pop(key))
    assert any(key in p for p in check.check(directory))


def test_k_must_agree_with_fx_fy_cx_cy(tmp_path):
    directory = _handoff(tmp_path)
    _edit(directory, "manifest", lambda d: d["camera"].update(cy=99.0))
    assert any("K disagrees" in p for p in check.check(directory))


def test_extrinsics_convention_is_fixed(tmp_path):
    directory = _handoff(tmp_path)
    _edit(directory, "manifest", lambda d: d["camera"]["extrinsics"].update(axes="opengl"))
    assert any("extrinsics convention" in p for p in check.check(directory))


def test_manifest_camera_and_pose_project_the_object_into_its_mask(tmp_path):
    """The documented formula, applied to the stored fields, lands on the mask."""
    directory = _handoff(tmp_path)
    manifest = json.loads((directory / "manifest.json").read_text())
    frame = json.loads((directory / "objects.json").read_text())[0]["key_frames"][0]
    cam = manifest["camera"]
    p = np.linalg.inv(np.asarray(frame["camera_to_world"])) @ np.array([3.0, 2.0, 1.5, 1.0])
    u = cam["fx"] * p[0] / p[2] + cam["cx"]
    v = cam["fy"] * p[1] / p[2] + cam["cy"]
    mask = np.asarray(Image.open(directory / frame["mask"])) > 0
    assert p[2] > 0 and mask[int(v), int(u)]


def test_scene_meta_and_floor(tmp_path):
    scene = tmp_path / "scene_x"
    scene.mkdir()
    (scene / "scene_x.txt").write_text(
        "axisAlignment = 1 0 0 -1 0 1 0 -2 0 0 1 0 0 0 0 1 \ncolorWidth = 1296\n"
        "fx_color = 1170.187988\nsceneType = Apartment\n")
    meta = read_scene_meta(scene)
    assert meta["colorWidth"] == 1296 and meta["fx_color"] == pytest.approx(1170.187988)
    assert np.asarray(meta["axisAlignment"]).reshape(4, 4)[1, 3] == -2
    assert meta["sceneType"] == "Apartment"
    rng = np.random.default_rng(0)
    floor = np.column_stack([rng.uniform(0, 4, 400), rng.uniform(0, 4, 400), 0.05 + rng.normal(0, 0.005, 400)])
    rug = np.column_stack([rng.uniform(0, 1, 40), rng.uniform(0, 1, 40), np.full(40, 0.3)])
    chair = np.column_stack([rng.uniform(0, 1, 50), rng.uniform(0, 1, 50), rng.uniform(0.0, 1.0, 50)])
    points = np.vstack([floor, rug, chair]).astype(np.float32)
    owner = np.array([3] * 440 + [7] * 50)
    result = floor_height(points, owner, {3: "floor", 7: "chair"})
    assert result["value"] == pytest.approx(0.05, abs=0.01)
    # The rug lifts the median by under a centimetre; the minimum or the mean would not
    # be robust to it. Tilt is fitted on everything, so only a loose bound holds here.
    assert result["floor_object_ids"] == ["obj_0003"] and result["plane_tilt_deg"] < 5.0
    flat = floor_height(floor.astype(np.float32), np.full(400, 3), {3: "floor"})
    assert flat["plane_tilt_deg"] < 1.0
    with pytest.raises(ValueError, match="floor"):
        floor_height(points, owner, {3: "carpet", 7: "chair"})


def test_real_handoff_if_present():
    """The committed-to artifact satisfies the contract and projects onto its masks."""
    directory = Path("artifacts/handoff/scene0000_00/gt_amb3r_floor")
    if not (directory / "manifest.json").is_file():
        pytest.skip("handoff artifact not built on this machine")
    assert check.check(directory) == []


@pytest.mark.parametrize("change,message", [
    (lambda d: d.update(coordinate_frame="scannet_scan"), "coordinate_frame"),
    (lambda d: d.update(floor_z=0.5), "floor is at z = 0"),
    (lambda d: d.update(axis_alignment={"matrix": np.eye(4).tolist()}), "axis_alignment is not part"),
    (lambda d: d["frame"].update(native_to_frame=np.diag([1.0, 1.0, -1.0, 1.0]).tolist()), "times a rotation"),
])
def test_manifest_is_in_amb3r_floor(tmp_path, change, message):
    directory = _handoff(tmp_path)
    _edit(directory, "manifest", change)
    assert any(message in p for p in check.check(directory))
