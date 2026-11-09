"""Tests for src.geometry.extract_scannet_sens."""
import io
import json
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.geometry.extract_scannet_sens import (
    extract,
    load_frame_indices,
    uniform_sample_indices,
)


# ---------------------------------------------------------------------------
# uniform_sample_indices
# ---------------------------------------------------------------------------

def test_uniform_sample_both_endpoints_included():
    indices = uniform_sample_indices(100, 5)
    assert indices[0] == 0
    assert indices[-1] == 99


def test_uniform_sample_single_frame_returns_zero():
    assert uniform_sample_indices(10, 1) == [0]


def test_uniform_sample_count_equals_frame_count():
    assert uniform_sample_indices(5, 5) == [0, 1, 2, 3, 4]


def test_uniform_sample_count_exceeds_frame_count():
    assert uniform_sample_indices(3, 10) == [0, 1, 2]


def test_uniform_sample_returns_correct_count():
    indices = uniform_sample_indices(5578, 500)
    assert len(indices) == 500


def test_uniform_sample_indices_sorted_and_unique():
    indices = uniform_sample_indices(5578, 500)
    assert indices == sorted(set(indices))


def test_uniform_sample_known_values():
    # round(i * 9 / 2) for i in 0..2 → 0, 4 (round(4.5)=4), 9
    indices = uniform_sample_indices(10, 3)
    assert indices == [0, round(1 * 9 / 2), 9]


def test_uniform_sample_invalid_inputs():
    with pytest.raises(ValueError):
        uniform_sample_indices(0, 5)
    with pytest.raises(ValueError):
        uniform_sample_indices(10, 0)


# ---------------------------------------------------------------------------
# load_frame_indices
# ---------------------------------------------------------------------------

def test_load_frame_indices_valid(tmp_path):
    p = tmp_path / "batch.json"
    p.write_text(json.dumps({"frame_indices": [0, 11, 22]}))
    assert load_frame_indices(p) == [0, 11, 22]


def test_load_frame_indices_missing_key(tmp_path):
    p = tmp_path / "batch.json"
    p.write_text(json.dumps({"something_else": [0, 1]}))
    with pytest.raises(ValueError, match="non-empty integer"):
        load_frame_indices(p)


def test_load_frame_indices_not_sorted(tmp_path):
    p = tmp_path / "batch.json"
    p.write_text(json.dumps({"frame_indices": [11, 0, 22]}))
    with pytest.raises(ValueError, match="sorted and unique"):
        load_frame_indices(p)


def test_load_frame_indices_duplicates(tmp_path):
    p = tmp_path / "batch.json"
    p.write_text(json.dumps({"frame_indices": [0, 0, 11]}))
    with pytest.raises(ValueError, match="sorted and unique"):
        load_frame_indices(p)


def test_load_frame_indices_empty(tmp_path):
    p = tmp_path / "batch.json"
    p.write_text(json.dumps({"frame_indices": []}))
    with pytest.raises(ValueError, match="non-empty integer"):
        load_frame_indices(p)


# ---------------------------------------------------------------------------
# extract — smoke test with a synthetic .sens file
# ---------------------------------------------------------------------------

def _make_sens(tmp_path: Path, num_frames: int = 4) -> Path:
    """Write a minimal valid .sens file with tiny synthetic frames."""
    sens = tmp_path / "scene.sens"
    sensor_name = b"TestSensor"

    identity = np.eye(4, dtype=np.float32)

    # depth image: 4x4, all 500 mm
    depth_width, depth_height = 4, 4
    depth_mm = np.full((depth_height, depth_width), 500, dtype=np.uint16)
    depth_bytes = zlib.compress(depth_mm.tobytes())

    # color image: tiny JPEG (4x4 RGB)
    color_img = Image.new("RGB", (4, 4), color=(128, 64, 32))
    color_buf = io.BytesIO()
    color_img.save(color_buf, format="JPEG", quality=95)
    color_bytes = color_buf.getvalue()

    with sens.open("wb") as f:
        f.write(struct.pack("<I", 4))                    # version
        f.write(struct.pack("<Q", len(sensor_name)))
        f.write(sensor_name)
        for _ in range(4):                               # 4 matrices
            f.write(identity.tobytes())
        f.write(struct.pack("<ii", 2, 1))                # JPEG color, zlib depth
        f.write(struct.pack("<IIII", 4, 4, depth_width, depth_height))
        f.write(struct.pack("<f", 1000.0))               # depth_shift
        f.write(struct.pack("<Q", num_frames))
        for frame_idx in range(num_frames):
            f.write(identity.tobytes())                  # camera_to_world
            f.write(struct.pack("<QQ", frame_idx * 1000, frame_idx * 1000))
            f.write(struct.pack("<QQ", len(color_bytes), len(depth_bytes)))
            f.write(color_bytes)
            f.write(depth_bytes)
    return sens


def test_extract_writes_manifest_and_files(tmp_path):
    sens = _make_sens(tmp_path)
    out = tmp_path / "out"
    source_indices = [0, 2, 3]
    selection = {"method": "explicit", "count": 3}

    manifest_path = extract(sens, out, source_indices, selection)

    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["num_source_frames"] == 4
    assert len(manifest["frames"]) == 3
    assert manifest["frames"][0]["source_frame_id"] == 0
    assert manifest["frames"][1]["source_frame_id"] == 2
    assert manifest["frames"][2]["source_frame_id"] == 3
    # sequence indices are 0-based
    assert [f["sequence_index"] for f in manifest["frames"]] == [0, 1, 2]

    # JPEG and NPZ should exist for each selected frame
    for seq in range(3):
        assert (out / "input" / f"{seq:04d}.jpg").exists()
        assert (out / "rgb" / f"{seq:04d}.jpg").exists()
        assert (out / "input" / f"{seq:04d}.npz").exists()


def test_extract_skips_unselected_frames(tmp_path):
    sens = _make_sens(tmp_path, num_frames=4)
    out = tmp_path / "out"
    source_indices = [1]
    selection = {"method": "explicit", "count": 1}

    extract(sens, out, source_indices, selection)
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["frames"]) == 1
    assert manifest["frames"][0]["source_frame_id"] == 1
    assert not (out / "input" / "0001.jpg").exists()  # frame 0 not extracted


def test_extract_out_of_range_raises(tmp_path):
    sens = _make_sens(tmp_path, num_frames=4)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="recording has"):
        extract(sens, out, [0, 99], {"method": "test", "count": 2})


def test_extract_npz_contains_source_frame_id(tmp_path):
    sens = _make_sens(tmp_path, num_frames=3)
    out = tmp_path / "out"
    extract(sens, out, [2], {"method": "test", "count": 1})
    npz = np.load(str(out / "input" / "0000.npz"))
    assert int(npz["source_frame_id"]) == 2
