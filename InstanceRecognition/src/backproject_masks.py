#!/usr/bin/env python3
"""Lift existing SAM masks through AMB3R point maps, without instance fusion.

Point maps already contain one backprojected world point per model-input pixel.
We transfer each independent binary mask through the SAME image preprocessing,
then apply the recorded global GT Sim(3). No GT depth, new model inference,
confidence filtering, denoising, voxelization, or cross-frame association is used.
"""

from __future__ import annotations

import argparse
import base64
import colorsys
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
GEOMETRY_ROOT = PROJECT_ROOT / "Video2PointCloud"
DEFAULT_GEOMETRY = GEOMETRY_ROOT / "output/amb3r/vo_uniform500"


def crop_box(size: tuple[int, int], target: tuple[int, int]) -> tuple[int, int, int, int]:
    """Match the official AMB3R-VO DemoDataset center crop, including rounding."""
    width, height = size
    target_width, target_height = target
    if min(width, height, target_width, target_height) <= 0:
        raise ValueError("Image dimensions must be positive")
    aspect = target_width / target_height
    if width / height > aspect:
        cropped_width = int(round(height * aspect))
        left = (width - cropped_width) // 2
        return left, 0, left + cropped_width, height
    if width / height < aspect:
        cropped_height = int(round(width / aspect))
        top = (height - cropped_height) // 2
        return 0, top, width, top + cropped_height
    return 0, 0, width, height


def align_mask(mask: np.ndarray, intermediate_size: tuple[int, int],
               target_size: tuple[int, int]) -> np.ndarray:
    """Original mask -> extracted RGB grid -> VO crop -> point-map grid.

    Nearest-neighbor sampling preserves binary membership. This is resolution
    alignment, not morphological cleanup. Overlapping masks remain independent.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D binary mask, got {mask.shape}")
    if not np.isin(mask, [0, 1, 255]).all():
        raise ValueError("Mask contains non-binary values")
    image = Image.fromarray((mask > 0).astype(np.uint8))
    image = image.resize(intermediate_size, Image.Resampling.NEAREST)
    image = image.crop(crop_box(intermediate_size, target_size))
    return np.asarray(image.resize(target_size, Image.Resampling.NEAREST)) > 0


def membership_bits(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Bit i means membership in masks[i], NOT a globally shared instance ID."""
    if len(masks) > 32:
        raise ValueError("This lossless uint32 format supports at most 32 masks per frame")
    bits = np.zeros(shape, dtype=np.uint32)
    for index, mask in enumerate(masks):
        if mask.shape != shape:
            raise ValueError(f"Mask shape mismatch: {mask.shape} != {shape}")
        bits[mask] |= np.uint32(1 << index)
    return bits


def apply_sim3(points: np.ndarray, transform: dict) -> np.ndarray:
    rotation = np.asarray(transform["alignment_rotation"], dtype=np.float64)
    translation = np.asarray(transform["alignment_translation_m"], dtype=np.float64)
    scale = float(transform["alignment_scale"])
    if rotation.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid Sim(3)")
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("Nonfinite Sim(3)")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < 0:
        raise ValueError("Sim(3) rotation is not a proper rotation")
    return (scale * (points.astype(np.float64) @ rotation.T) + translation).astype(np.float32)


def select_source_ply_pixels(pointmap: np.ndarray, confidence: np.ndarray,
                             threshold: float, limit: int,
                             rng: np.random.Generator) -> np.ndarray:
    """Replay ONLY the existing PLY sampling to recover its pixel provenance.

    This selection is never applied to the raw per-frame mask exports.
    """
    valid = (np.isfinite(pointmap).all(axis=-1) & np.isfinite(confidence)
             & (confidence >= threshold))
    selected = np.flatnonzero(valid)
    if len(selected) > limit:
        selected = rng.choice(selected, limit, replace=False)
    return selected


def instance_color(frame_id: int, instance_id: int) -> list[int]:
    hue = ((frame_id * 37 + instance_id * 137) % 360) / 360
    return [round(channel * 255) for channel in colorsys.hsv_to_rgb(hue, 0.68, 0.96)]


def display_colors(bits: np.ndarray, instances: list[dict]) -> np.ndarray:
    """No winner for overlaps: multi-membership is displayed in magenta."""
    colors = np.full((bits.size, 3), 75, dtype=np.uint8)
    for index, instance in enumerate(instances):
        colors[bits == np.uint32(1 << index)] = instance["color"]
    overlap = (bits & (bits - np.uint32(1))) != 0
    colors[overlap] = [235, 65, 210]
    return colors


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray,
              bits: np.ndarray, source_frame_ids: np.ndarray,
              pixel_indices: np.ndarray) -> None:
    """Binary PLY with recoverable multi-label and source-pixel provenance."""
    count = len(points)
    if not (colors.shape == (count, 3) and len(bits) == len(source_frame_ids) == len(pixel_indices) == count):
        raise ValueError("PLY attribute lengths differ")
    vertex = np.empty(count, dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("mask_bits", "<u4"), ("source_frame_id", "<u4"), ("pixel_index", "<u4"),
    ])
    for axis, name in enumerate(("x", "y", "z")):
        vertex[name] = points[:, axis]
    for axis, name in enumerate(("red", "green", "blue")):
        vertex[name] = colors[:, axis]
    vertex["mask_bits"], vertex["source_frame_id"], vertex["pixel_index"] = bits, source_frame_ids, pixel_indices
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment raw per-frame mask observations; mask_bits refers to frame-local instance list\n"
        f"element vertex {count}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uint mask_bits\nproperty uint source_frame_id\nproperty uint pixel_index\nend_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        vertex.tofile(handle)


def read_point_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read our existing binary vertex-only PLY without starting native thread pools."""
    types = {"float": "<f4", "double": "<f8", "uchar": "u1", "uint": "<u4"}
    fields = []
    count = None
    with path.open("rb") as handle:
        if handle.readline() != b"ply\n" or handle.readline() != b"format binary_little_endian 1.0\n":
            raise ValueError("Expected a binary little-endian PLY")
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("Truncated PLY header")
            tokens = line.decode("ascii").split()
            if tokens[:2] == ["element", "vertex"]:
                count = int(tokens[2])
            elif tokens[:1] == ["element"]:
                raise ValueError("Expected a vertex-only point cloud")
            elif tokens[:1] == ["property"]:
                if len(tokens) != 3 or tokens[1] not in types:
                    raise ValueError(f"Unsupported PLY property: {tokens}")
                fields.append((tokens[2], types[tokens[1]]))
            elif tokens == ["end_header"]:
                break
        if count is None:
            raise ValueError("Missing PLY vertex count")
        vertices = np.fromfile(handle, dtype=fields, count=count)
        if len(vertices) != count:
            raise ValueError("Truncated PLY vertices")
    points = np.column_stack([vertices[name] for name in ("x", "y", "z")]).astype(np.float32)
    colors = np.column_stack([vertices[name] for name in ("red", "green", "blue")]).astype(np.uint8)
    return points, colors


def encode(array: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode("ascii")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_cloud_asset(path: Path, key: str | int, payload: dict) -> None:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    path.write_text(f"window.registerMaskCloud({json.dumps(key)},{data});\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path, directory: Path) -> str:
    return Path(os.path.relpath(path.resolve(), directory.resolve())).as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", default="scene0000_00")
    parser.add_argument("--prediction", type=Path, default=DEFAULT_GEOMETRY / "scene_scene0000_00_uniform500_metric_head_results.npz")
    parser.add_argument("--alignment", type=Path, default=DEFAULT_GEOMETRY / "pointcloud_gt_corrected_evaluation.json")
    parser.add_argument("--pointcloud", type=Path, default=DEFAULT_GEOMETRY / "pointcloud_gt_corrected_rgb.ply")
    parser.add_argument("--data", type=Path, default=GEOMETRY_ROOT / "data/scannet/scene0000_00_uniform500")
    parser.add_argument("--results", type=Path, default=APP_ROOT / "runtime/results")
    parser.add_argument("--batch", type=Path, default=APP_ROOT / "runtime/batches/scene0000_00/target_500.json")
    parser.add_argument("--output", type=Path, default=APP_ROOT / "runtime/mask_pointclouds/scene0000_00/amb3r_gt_raw")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output}; use a new --output directory")
    evaluation = json.loads(args.alignment.read_text())
    data_manifest = json.loads((args.data / "manifest.json").read_text())
    batch = json.loads(args.batch.read_text())
    frame_ids = evaluation["source_frame_ids"]
    metadata_ids = [frame["source_frame_id"] for frame in data_manifest["frames"]]
    if frame_ids != metadata_ids or frame_ids != batch["frame_indices"]:
        raise ValueError("Geometry, metadata, and mask batch frame lists differ")
    if batch["scene_id"] != args.scene_id or len(set(frame_ids)) != len(frame_ids):
        raise ValueError("Scene or source frame identities are inconsistent")
    if Path(evaluation["prediction"]).resolve() != args.prediction.resolve():
        raise ValueError("Alignment was fitted to a different prediction archive")
    if Path(evaluation["sim3_ply"]).resolve() != args.pointcloud.resolve():
        raise ValueError("Alignment describes a different canonical PLY")
    transform = evaluation["sim3_oracle_scale"]
    print(f"Loading AMB3R point maps for {len(frame_ids)} frames (no GPU/model loading)...", flush=True)
    with np.load(args.prediction, allow_pickle=False) as archive:
        points = archive["pts"]
        confidence = archive["conf"]
        images = archive["images"]
    if points.ndim != 4 or points.shape[-1] != 3 or points.shape[0] != len(frame_ids):
        raise ValueError(f"Unexpected point-map shape: {points.shape}")
    count, height, width, _ = points.shape
    if confidence.shape != points.shape[:3] or images.shape != (count, 3, height, width):
        raise ValueError("Confidence/image arrays do not share the point-map grid")
    target_size = (width, height)
    original_size = tuple(data_manifest["source_color_size"])
    records = []
    for sequence, frame_id in enumerate(frame_ids):
        frame_dir = args.results / args.scene_id / f"frame_{frame_id:06d}"
        summary = json.loads((frame_dir / "summary.json").read_text())
        predictions_path = frame_dir / "sam31/predictions.json"
        predictions = json.loads(predictions_path.read_text())
        if summary["scene_id"] != args.scene_id or summary["frame_index"] != frame_id:
            raise ValueError(f"Wrong mask frame identity in {frame_dir}")
        if predictions["image_shape_hwc"][:2] != [original_size[1], original_size[0]]:
            raise ValueError(f"Wrong mask image dimensions in {frame_dir}")
        if predictions["instance_count"] != len(predictions["instances"]) or len(predictions["instances"]) > 32:
            raise ValueError(f"Unsupported or inconsistent mask count in {frame_dir}")
        with np.load(args.data / "input" / f"{sequence:04d}.npz", allow_pickle=False) as metadata:
            if int(metadata["source_frame_id"]) != frame_id:
                raise ValueError(f"Metadata frame mismatch at sequence {sequence}")
        for instance in predictions["instances"]:
            if not (predictions_path.parent / instance["mask_file"]).is_file():
                raise FileNotFoundError(predictions_path.parent / instance["mask_file"])
        records.append((frame_dir, summary, predictions_path, predictions))
    canonical_points, canonical_rgb = read_point_ply(args.pointcloud)
    if len(canonical_points) != evaluation["predicted_point_count"] or canonical_rgb.shape != canonical_points.shape:
        raise ValueError("Canonical PLY count or colors disagree with provenance")
    for subdir in (output, output / "frames", output / "assets"):
        subdir.mkdir(parents=True, exist_ok=True)
    write_cloud_asset(output / "assets/background.js", "background", {
        "count": len(canonical_points),
        "positions_f32_base64": encode(canonical_points, "<f4"),
        "colors_u8_base64": encode(canonical_rgb, "u1"),
    })
    sampling = evaluation["sampling"]
    rng = np.random.default_rng(sampling["seed"])
    per_frame_limit = max(1, int(sampling["max_pred_points"]) // count)
    canonical_bits = np.zeros(len(canonical_points), dtype=np.uint32)
    canonical_frame_ids = np.zeros(len(canonical_points), dtype=np.uint32)
    canonical_pixel_ids = np.zeros(len(canonical_points), dtype=np.uint32)
    canonical_colors = np.empty_like(canonical_rgb)
    frames = []
    cursor = 0
    max_coordinate_error = 0.0
    max_rgb_error = 0
    extraction_rgb_mae = []
    total_memberships = total_overlap = total_invalid = total_points = total_empty = 0
    for sequence, (frame_id, record) in enumerate(zip(frame_ids, records)):
        frame_dir, summary, predictions_path, predictions = record
        source_jpeg = args.data / "rgb" / f"{sequence:04d}.jpg"
        with Image.open(source_jpeg) as image:
            intermediate_size = image.size
            processed = np.asarray(image.convert("RGB").crop(crop_box(image.size, target_size))
                                   .resize(target_size, Image.Resampling.LANCZOS))
            intermediate_rgb = np.asarray(image.convert("RGB"))
        rgb = np.clip(np.rint((np.moveaxis(images[sequence], 0, -1) + 1.0) * 127.5), 0, 255).astype(np.uint8)
        rgb_error = int(np.abs(rgb.astype(np.int16) - processed.astype(np.int16)).max())
        max_rgb_error = max(max_rgb_error, rgb_error)
        if rgb_error > 1:
            raise ValueError(f"Frame {frame_id}: NPZ RGB != preprocessed source JPEG (max error {rgb_error})")
        # Confirm the detector and reconstruction saw the same source frame.
        with Image.open(summary["source"]) as image:
            resized_original = np.asarray(image.convert("RGB").resize(intermediate_size, Image.Resampling.BILINEAR))
        rgb_mae = float(np.abs(resized_original.astype(np.float32) - intermediate_rgb).mean())
        extraction_rgb_mae.append(rgb_mae)
        if rgb_mae > 8:
            raise ValueError(f"Frame {frame_id}: original/extracted RGB mismatch (MAE {rgb_mae})")
        masks, instances = [], []
        for instance in predictions["instances"]:
            source_mask = predictions_path.parent / instance["mask_file"]
            with Image.open(source_mask) as image:
                if image.size != original_size:
                    raise ValueError(f"Wrong mask dimensions: {source_mask}")
                mask_original = np.asarray(image)
            mask = align_mask(mask_original, intermediate_size, target_size)
            masks.append(mask)
            item = dict(instance)
            item.update({
                "confidence": instance["yoloe_confidence"],
                "color": instance_color(frame_id, instance["instance_id"]),
                "mask_file": relative(source_mask, output),
                "original_mask_pixels": int(np.count_nonzero(mask_original)),
                "aligned_mask_pixels": int(mask.sum()),
            })
            instances.append(item)
        bits_grid = membership_bits(masks, (height, width))
        flat_points = points[sequence].reshape(-1, 3)
        finite = np.isfinite(flat_points).all(axis=1)
        flat_bits = bits_grid.reshape(-1)
        selected = np.flatnonzero((flat_bits != 0) & finite).astype(np.uint32)
        invalid = int(np.count_nonzero((flat_bits != 0) & ~finite))
        world = apply_sim3(flat_points[selected], transform)
        bits = flat_bits[selected]
        colors = rgb.reshape(-1, 3)[selected]
        overlap = int(np.count_nonzero(bits & (bits - np.uint32(1))))
        memberships = 0
        for index, instance in enumerate(instances):
            instance["point_count"] = int(np.count_nonzero(bits & np.uint32(1 << index)))
            instance["nonfinite_points_skipped"] = int(np.count_nonzero(masks[index].reshape(-1) & ~finite))
            memberships += instance["point_count"]
            total_empty += int(instance["point_count"] == 0)
        stem = f"frame_{frame_id:06d}"
        ids = np.full(len(world), frame_id, dtype=np.uint32)
        np.savez_compressed(
            output / "frames" / f"{stem}.npz",
            points_world=world, rgb=colors, mask_bits=bits, pixel_indices=selected,
            geometry_confidence=confidence[sequence].reshape(-1)[selected],
            instance_ids=np.asarray([item["instance_id"] for item in instances], dtype=np.uint32),
            source_frame_id=np.int64(frame_id), sequence_index=np.int64(sequence),
            grid_shape_hw=np.asarray([height, width], dtype=np.int32),
        )
        write_ply(output / "frames" / f"{stem}.ply", world, display_colors(bits, instances), bits, ids, selected)
        Image.fromarray(rgb).save(output / "frames" / f"{stem}_rgb.jpg", quality=95)
        write_cloud_asset(output / "assets" / f"{stem}.js", frame_id, {
            "source_frame_id": frame_id, "count": len(world), "width": width, "height": height,
            "original_size": list(original_size), "intermediate_size": list(intermediate_size),
            "crop_box": list(crop_box(intermediate_size, target_size)),
            "positions_f32_base64": encode(world, "<f4"), "colors_u8_base64": encode(colors, "u1"),
            "mask_bits_u32_base64": encode(bits, "<u4"), "pixel_indices_u32_base64": encode(selected, "<u4"),
        })
        # Prove that these same point maps generate the EXACT supplied PLY.
        sampled = select_source_ply_pixels(points[sequence], confidence[sequence],
                                           sampling["confidence_threshold"], per_frame_limit, rng)
        end = cursor + len(sampled)
        regenerated = apply_sim3(flat_points[sampled], transform)
        if end > len(canonical_points):
            raise ValueError("Replayed PLY sampling exceeds supplied point count")
        error = float(np.max(np.abs(regenerated - canonical_points[cursor:end]))) if len(sampled) else 0.0
        max_coordinate_error = max(max_coordinate_error, error)
        if error > 1e-5:
            raise ValueError(f"Frame {frame_id}: source PLY replay mismatch ({error} m)")
        if not np.array_equal(rgb.reshape(-1, 3)[sampled], canonical_rgb[cursor:end]):
            raise ValueError(f"Frame {frame_id}: source PLY colors do not match")
        canonical_bits[cursor:end] = flat_bits[sampled]
        canonical_frame_ids[cursor:end] = frame_id
        canonical_pixel_ids[cursor:end] = sampled
        canonical_colors[cursor:end] = display_colors(flat_bits[sampled], instances)
        frame = {
            "sequence_index": sequence, "source_frame_id": frame_id,
            "asset": f"assets/{stem}.js", "npz": f"frames/{stem}.npz", "ply": f"frames/{stem}.ply",
            "rgb_image": f"frames/{stem}_rgb.jpg", "original_rgb_image": relative(Path(summary["source"]), output),
            "annotated_image": relative(frame_dir / "sam31/annotated.jpg", output),
            "predictions_json": relative(predictions_path, output), "instances": instances,
            "point_count": len(world), "membership_count": memberships,
            "overlap_point_count": overlap, "nonfinite_points_skipped": invalid,
            "canonical_ply_point_range": [cursor, end],
        }
        frames.append(frame)
        write_json(output / "frames" / f"{stem}.json", frame)
        cursor = end
        total_points += len(world)
        total_memberships += memberships
        total_overlap += overlap
        total_invalid += invalid
        if sequence % 10 == 0 or sequence + 1 == count:
            print(f"{sequence + 1}/{count}: source frame {frame_id}, {len(instances)} masks, "
                  f"{len(world):,} points, {overlap:,} overlapping; {time.monotonic()-started:.1f}s", flush=True)
    if cursor != len(canonical_points):
        raise ValueError("Replayed PLY sampling has incomplete point coverage")
    write_ply(output / "pointcloud_gt_corrected_mask_observations.ply", canonical_points,
              canonical_colors, canonical_bits, canonical_frame_ids, canonical_pixel_ids)
    np.savez_compressed(output / "canonical_mask_memberships.npz",
                        source_frame_ids=canonical_frame_ids, pixel_indices=canonical_pixel_ids,
                        mask_bits=canonical_bits)
    manifest = {
        "scene_id": args.scene_id, "frame_count": count,
        "mask_count": sum(len(frame["instances"]) for frame in frames),
        "background_asset": "assets/background.js", "frames": frames,
        "bounds": {"minimum": canonical_points.min(axis=0).tolist(), "maximum": canonical_points.max(axis=0).tolist()},
        "policy": "Raw SAM3.1 mask observations; IDs are local to each frame; overlaps preserved; no new filtering or association",
    }
    write_json(output / "manifest.json", manifest)
    (output / "manifest.js").write_text("window.MASK_PROJECTION_MANIFEST=" +
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + ";\n", encoding="utf-8")
    report = {
        "scene_id": args.scene_id, "frame_count": count, "mask_count": manifest["mask_count"],
        "raw_union_point_count_sum_over_frames": total_points,
        "raw_mask_membership_count_sum_over_frames": total_memberships,
        "overlap_point_count_sum_over_frames": total_overlap, "empty_projected_masks": total_empty,
        "nonfinite_mask_points_skipped": total_invalid,
        "source_prediction": str(args.prediction.resolve()), "source_prediction_sha256": sha256_file(args.prediction),
        "source_ply": str(args.pointcloud.resolve()), "source_ply_sha256": sha256_file(args.pointcloud),
        "source_alignment": str(args.alignment.resolve()), "source_alignment_sha256": sha256_file(args.alignment),
        "source_batch_sha256": sha256_file(args.batch), "source_frame_ids": frame_ids,
        "sim3": {key: transform[key] for key in ("alignment_scale", "alignment_rotation", "alignment_translation_m")},
        "projection": "P_world = s * R * P_amb3r(pixel) + t; P_amb3r is the already-backprojected per-frame point map",
        "preprocessing": {"original_size": list(original_size), "intermediate_size": list(intermediate_size),
                          "crop_box_on_intermediate": list(crop_box(intermediate_size, target_size)),
                          "point_map_size": list(target_size), "mask_interpolation": "nearest-neighbor at BOTH resize steps"},
        "validation": {"frame_lists_equal": True, "all_source_rgb_preprocessing_verified": True,
                       "npz_vs_processed_rgb_max_channel_error": max_rgb_error,
                       "original_vs_extracted_rgb_mae_max": max(extraction_rgb_mae),
                       "canonical_point_count": len(canonical_points),
                       "canonical_replay_max_coordinate_error_m": max_coordinate_error,
                       "canonical_rgb_exact_match": True,
                       "canonical_points_with_any_mask": int(np.count_nonzero(canonical_bits)),
                       "canonical_points_with_overlapping_masks": int(np.count_nonzero(canonical_bits & (canonical_bits - np.uint32(1))))},
        "postprocessing": {"new_confidence_threshold": None, "mask_morphology": False,
                           "denoising": False, "voxel_downsampling": False, "cross_frame_fusion": False,
                           "semantic_normalization": False, "overlap_winner_selection": False,
                           "uses_gt_depth": False, "uses_recorded_gt_sim3": True},
        "existing_input_policy": {"yoloe_confidence_threshold": 0.25, "yoloe_max_detections": 16,
                                  "canonical_ply_sampling_inherited_only": sampling,
                                  "raw_per_frame_outputs_use_confidence_filter": False},
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(output / "projection_report.json", report)
    template = APP_ROOT / "web/mask_pointcloud"
    for name in ("index.html", "styles.css", "viewer.js"):
        if (template / name).is_file():
            shutil.copy2(template / name, output / name)
    print(json.dumps({key: report[key] for key in ("frame_count", "mask_count", "raw_union_point_count_sum_over_frames", "elapsed_seconds")}), flush=True)
    print(f"Outputs: {output}", flush=True)
    return report


if __name__ == "__main__":
    run(parse_args())
