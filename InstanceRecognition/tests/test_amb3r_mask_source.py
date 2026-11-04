"""Contracts for the AMB3R lift, the one geometry this tree builds.

Two defects lived here and neither was visible until the lift was actually run
end to end: the frame paths were carried over from the upstream source verbatim,
so every mask resolved against the wrong directory, and the point colours were
written under a key (`colors`) the Harness does not read. Both produced a source
that looked well-formed and could not be consumed. So what is pinned here is the
shape of what lands on disk and that every path in the manifest resolves from it.

The point maps are a fixture, not AMB3R's 1.6 GB npz: the lift is arithmetic on a
grid, so a 2-frame one is enough to exercise it. No GPU and no network.
"""
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.geometry import amb3r_mask_source as lift
from src.geometry.backproject_masks import crop_box, write_ply

FRAME = (1296, 968)  # width, height of the RGB frames the masks were drawn on


def write_ply_file(path, count=4):
    points = np.arange(count * 3, dtype=np.float32).reshape(count, 3)
    write_ply(path, points, np.full((count, 3), 180, np.uint8),
              np.zeros(count, np.uint32), np.zeros(count, np.uint32),
              np.arange(count, dtype=np.uint32))
    return path


def instance(instance_id=1, class_name="office chair"):
    return {
        "raw_candidate_id": instance_id, "instance_id": instance_id,
        "class_id": 0, "prompt_index": 0, "class_name": class_name,
        "sam_prompt": class_name, "sam_prompt_type": "text",
        "gpt_certainty": "clear", "visible_scope": "partial",
        "evidence": "visible", "ambiguity": "", "sam_score": 0.9,
        "sam_box_xyxy": [1.0, 1.0, 4.0, 4.0], "mask_pixels": 4,
        "mask_file": f"masks/raw_masks/{instance_id:03d}_{class_name.replace(' ', '_')}.png",
    }


class Amb3r:
    """An upstream mask source, its RGB frames, and a 2-frame point map."""

    def __init__(self, root, frames=2, conf=0.9):
        self.root = Path(root)
        self.source = self.root / "upstream"
        self.frames_dir = self.root / "frames"
        self.masks = self.source / "masks/raw_masks"
        self.masks.mkdir(parents=True)
        (self.frames_dir / "scene0000_00").mkdir(parents=True)

        self.sim3 = self.root / "evaluation.json"
        backdrop = write_ply_file(self.root / "canonical_cloud.ply")
        self.sim3.write_text(json.dumps({
            "sim3_oracle_scale": {
                "alignment_scale": 0.8875,
                "alignment_rotation": np.eye(3).tolist(),
                "alignment_translation_m": [0.1, 0.2, 0.3],
            },
            "sim3_ply": str(backdrop),
        }))

        records = []
        for seq in range(frames):
            fid = seq * 11
            # A gradient, so a point's colour identifies which pixel it came from.
            gradient = np.zeros((FRAME[1], FRAME[0], 3), dtype=np.uint8)
            gradient[:, :, 0] = np.arange(FRAME[0], dtype=np.uint8)[None, :]
            gradient[:, :, 1] = np.arange(FRAME[1], dtype=np.uint8)[:, None]
            with Image.fromarray(gradient) as rgb_frame:
                rgb_frame.save(self.frames_dir / "scene0000_00" / f"frame_{fid:06d}.jpg")

            items = [instance(1), instance(2, "table")]
            for item in items:
                mask = np.zeros((FRAME[1], FRAME[0]), dtype=np.uint8)
                mask[100:300, 100:300] = 255
                Image.fromarray(mask).save(self.source / item["mask_file"])
            predictions = self.source / f"predictions/frame_{fid:06d}/sam31"
            predictions.mkdir(parents=True)
            (predictions / "predictions.json").write_text(json.dumps({"instances": items}))
            Image.new("RGB", FRAME, "white").save(predictions / "annotated.jpg")
            # Upstream also ships its own per-frame RGB copy and a display asset.
            (self.source / "frames").mkdir(exist_ok=True)
            Image.new("RGB", FRAME, "grey").save(self.source / f"frames/frame_{fid:06d}_rgb.jpg")
            (self.source / "assets").mkdir(exist_ok=True)
            (self.source / f"assets/frame_{fid:06d}.js").write_text("// upstream asset\n")

            records.append({
                "sequence_index": seq, "source_frame_id": fid,
                "npz": f"frames/frame_{fid:06d}.npz",
                "asset": f"assets/frame_{fid:06d}.js",
                "rgb_image": f"frames/frame_{fid:06d}_rgb.jpg",
                # The pre-refactor RGB path: it is dead, and the lift must not
                # depend on it when writing the frame's relocation.
                "original_rgb_image": f"../../../runtime/frames/scene0000_00/frame_{fid:06d}.jpg",
                "predictions_json": f"predictions/frame_{fid:06d}/sam31/predictions.json",
                "instances": items,
            })
        (self.source / "manifest.json").write_text(json.dumps({
            "scene_id": "scene0000_00", "method": "gpt", "model": "openai/gpt-5.6-terra",
            "policy": "Frame-local IDs", "frame_count": frames,
            "mask_count": sum(len(r["instances"]) for r in records),
            "background_asset": "assets/background.js",
            "bounds": {"minimum": [0, 0, 0], "maximum": [1, 1, 1]},
            "frames": records,
        }))

        # AMB3R's point maps are (N, 392, 518, 3) positions and (N, 392, 518) conf.
        shape = (frames, lift.AMBR3R_SIZE[1], lift.AMBR3R_SIZE[0])
        rng = np.random.default_rng(0)
        np.savez(
            self.root / "pointmaps.npz",
            pts=rng.normal(scale=2.0, size=shape + (3,)).astype(np.float32),
            conf=np.full(shape, conf, dtype=np.float32),
        )

    def build(self, output, **kwargs):
        return lift.build(self.root / "pointmaps.npz", self.sim3,
                          self.source, output, 0.3, 0, None,
                          self.frames_dir, **kwargs)


def test_the_frame_paths_resolve_from_the_output(tmp_path):
    """Upstream's paths are correct for upstream, which is not where this lands."""
    output = Amb3r(tmp_path).build(tmp_path / "source")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["lifting_policy"] == "amb3r_per_frame_v1"
    missing = []
    for frame in manifest["frames"]:
        for key, value in frame.items():
            if key in lift.FRAME_PATH_KEYS or key in ("npz", "asset"):
                if not (output / value).resolve().is_file():
                    missing.append((frame["sequence_index"], key, value))
        for item in frame["instances"]:
            if not (output / item["mask_file"]).resolve().is_file():
                missing.append((frame["sequence_index"], "mask_file", item["mask_file"]))
    assert missing == [], f"paths that do not resolve from the output: {missing}"


def test_the_rgb_is_located_where_the_caller_says_not_upstream(tmp_path):
    """Upstream's recorded RGB path is the pre-refactor one and does not exist."""
    perception = Amb3r(tmp_path)
    recorded = json.loads((perception.source / "manifest.json").read_text())
    recorded = recorded["frames"][0]["original_rgb_image"]
    assert not (perception.source / recorded).is_file()

    output = perception.build(tmp_path / "source")
    frame = json.loads((output / "manifest.json").read_text())["frames"][0]
    relocated = (output / frame["original_rgb_image"]).resolve()
    assert relocated.is_file() and relocated.name == "frame_000000.jpg"


def test_the_npz_carries_the_point_colours_under_the_key_the_harness_reads(tmp_path):
    """`observations.py` reads z['rgb']; a different key is a KeyError, not a fallback."""
    output = Amb3r(tmp_path).build(tmp_path / "source")
    frame = json.loads((output / "manifest.json").read_text())["frames"][0]
    with np.load((output / frame["npz"]).resolve()) as z:
        assert set(z.files) >= {"points_world", "rgb", "mask_bits", "pixel_indices",
                                "grid_shape_hw", "sequence_index"}
        assert z["rgb"].shape == z["points_world"].shape
        assert z["rgb"].dtype == np.uint8


def test_a_point_takes_the_colour_of_the_pixel_its_mask_claimed(tmp_path):
    """The sampled colour must come through the same crop the masks go through.

    A crop offset here would give every point the colour of a pixel a few columns
    away from the mask that claimed it — the mask's own RGB evidence would be
    subtly wrong rather than absent.
    """
    perception = Amb3r(tmp_path)
    output = perception.build(tmp_path / "source")
    manifest = json.loads((output / "manifest.json").read_text())
    frame = manifest["frames"][0]

    with np.load((output / frame["npz"]).resolve()) as z:
        indices, rgb = z["pixel_indices"], z["rgb"]

    with Image.open(perception.frames_dir / "scene0000_00/frame_000000.jpg") as image:
        expected = np.asarray(
            image.convert("RGB")
            .resize(lift.AMBR3R_INTERMEDIATE, Image.Resampling.BILINEAR)
            .crop(crop_box(lift.AMBR3R_INTERMEDIATE, lift.AMBR3R_SIZE))
            .resize(lift.AMBR3R_SIZE, Image.Resampling.BILINEAR)
        ).reshape(-1, 3)[indices]

    assert np.array_equal(rgb, expected), "point colours are not the claimed pixels"


def test_the_backdrop_is_encoded_from_the_cloud_the_sim3_came_from(tmp_path):
    """The viewer draws this behind the points, so the two must share one frame."""
    perception = Amb3r(tmp_path)
    output = perception.build(tmp_path / "source")
    manifest = json.loads((output / "manifest.json").read_text())

    asset = output / manifest["background_asset"]
    assert asset.is_file(), "the source must carry its own backdrop, not a path to one"
    assert asset.read_text().startswith('window.registerMaskCloud("background",')
    assert manifest["bounds"]["maximum"] == [9.0, 10.0, 11.0]  # 4 backdrop points

    report = json.loads((output / "projection_report.json").read_text())
    assert report["policy"] == "amb3r_per_frame_v1"
    assert report["backdrop_points"] == 4
    assert report["backdrop_sha256"] == hashlib.sha256(
        (perception.root / "canonical_cloud.ply").read_bytes()).hexdigest()


def test_the_frame_asset_carries_every_field_the_viewer_decodes(tmp_path):
    """The result page decodes these four; a missing one white-screens the frame.

    `incremental.js` calls `atob()` on each, and `atob(undefined)` throws, which
    kills the whole render — overlay, cloud and object list — rather than
    degrading. So the asset's payload keys are a contract, not a convenience.
    """
    output = Amb3r(tmp_path).build(tmp_path / "source")
    frame = json.loads((output / "manifest.json").read_text())["frames"][0]
    text = (output / frame["asset"]).read_text()

    # The wrapper names the cloud by source frame id, matching how it is registered.
    assert text.startswith(f'window.registerMaskCloud({frame["source_frame_id"]},')
    payload = json.loads(text.split(",", 1)[1].rstrip(");\n"))

    for key in ("positions_f32_base64", "colors_u8_base64",
                "mask_bits_u32_base64", "pixel_indices_u32_base64"):
        assert key in payload, f"{key} missing: the viewer decodes it with atob()"
        assert payload[key], f"{key} is empty"

    # Each decodes, and the payload describes the same points as the npz.
    decoded = {name: np.frombuffer(base64.b64decode(payload[key]), dtype=dtype)
               for name, key, dtype in (
                   ("positions", "positions_f32_base64", "<f4"),
                   ("colors", "colors_u8_base64", "u1"),
                   ("mask_bits", "mask_bits_u32_base64", "<u4"),
                   ("pixel_indices", "pixel_indices_u32_base64", "<u4"))}
    assert payload["count"] == len(decoded["positions"]) // 3
    assert len(decoded["positions"]) == 3 * payload["count"]
    assert len(decoded["colors"]) == 3 * payload["count"]
    assert len(decoded["mask_bits"]) == payload["count"]
    assert len(decoded["pixel_indices"]) == payload["count"]

    with np.load((output / frame["npz"]).resolve()) as z:
        assert np.array_equal(decoded["pixel_indices"], z["pixel_indices"])
        assert np.array_equal(decoded["mask_bits"], z["mask_bits"])
        assert np.allclose(decoded["positions"].reshape(-1, 3), z["points_world"])


def test_the_default_source_is_the_one_this_tree_owns(tmp_path):
    """The CLI default must name a source inside this tree.

    It used to point at the pre-refactor tree's `mask_pointclouds/`. That tree was
    deleted on 2026-10-19, so a default left pointing into it would make a bare
    `python -m src.geometry.amb3r_mask_source` fail with a missing-directory error
    instead of naming the wrong source.
    """
    assert lift.DEFAULT_SOURCE.is_dir(), f'missing default source: {lift.DEFAULT_SOURCE}'
    assert lift.DEFAULT_SOURCE.is_relative_to(lift.config.ARTIFACTS_ROOT)
    manifest = json.loads((lift.DEFAULT_SOURCE / "manifest.json").read_text())
    frame = manifest["frames"][0]
    for key in ("sequence_index", "source_frame_id", "predictions_json",
                "annotated_image"):
        assert key in frame, f"default source frame lacks {key}"
    assert frame["instances"][0]["mask_file"]


def test_a_low_confidence_pixel_contributes_no_point(tmp_path):
    """AMB3R reports where it does not trust its own depth, and those are dropped."""
    perception = Amb3r(tmp_path, conf=0.1)
    output = perception.build(tmp_path / "source")
    frame = json.loads((output / "manifest.json").read_text())["frames"][0]
    assert frame["point_count"] == 0


def test_the_output_directory_must_be_new(tmp_path):
    perception = Amb3r(tmp_path)
    perception.build(tmp_path / "source")
    with pytest.raises(ValueError, match="must be empty"):
        perception.build(tmp_path / "source")
