#!/usr/bin/env python3
"""Shared, model-independent ScanNet metric evaluation and PLY export helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_encoded_depth(metadata_path: Path) -> np.ndarray:
    with np.load(metadata_path) as metadata:
        maximum_depth = float(metadata["maximum_depth"])
    encoded_path = metadata_path.with_name(metadata_path.stem + "_depth.png")
    encoded = np.asarray(Image.open(encoded_path), dtype=np.float32)
    return encoded / 65535.0 * maximum_depth


def preprocess_depth(depth: np.ndarray, target_hw: tuple[int, int], mode: str) -> np.ndarray:
    target_h, target_w = target_hw
    image = Image.fromarray(depth.astype(np.float32), mode="F")
    if mode == "da3_resize":
        return np.asarray(image.resize((target_w, target_h), Image.Resampling.NEAREST), dtype=np.float32)
    if mode == "mapanything_cover_crop":
        source_w, source_h = image.size
        scale = max(target_w / source_w, target_h / source_h) + 1e-8
        resized_w, resized_h = np.floor(np.asarray([source_w, source_h]) * scale).astype(int)
        image = image.resize((int(resized_w), int(resized_h)), Image.Resampling.NEAREST)
        left = (resized_w - target_w) // 2
        top = (resized_h - target_h) // 2
        return np.asarray(image.crop((left, top, left + target_w, top + target_h)), dtype=np.float32)
    raise ValueError(f"Unknown preprocessing mode: {mode}")


def load_ground_truth(data: Path, frame_count: int, target_hw: tuple[int, int], mode: str):
    metadata_paths = sorted(data.glob("[0-9][0-9][0-9][0-9].npz"))[:frame_count]
    if len(metadata_paths) != frame_count:
        raise ValueError(f"Need {frame_count} metadata files, found {len(metadata_paths)}")
    depths, poses, source_ids = [], [], []
    for path in metadata_paths:
        depths.append(preprocess_depth(load_encoded_depth(path), target_hw, mode))
        with np.load(path) as metadata:
            poses.append(metadata["camera_pose"].astype(np.float64))
            source_ids.append(int(metadata["source_frame_id"]))
    return np.stack(depths), np.stack(poses), np.asarray(source_ids, dtype=np.int64)


def unproject_world(depth: np.ndarray, intrinsics: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    frame_count, height, width = depth.shape
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    result = []
    for index in range(frame_count):
        z = depth[index]
        k = intrinsics[index]
        camera = np.stack(((u - k[0, 2]) / k[0, 0] * z, (v - k[1, 2]) / k[1, 1] * z, z), axis=-1)
        pose = camera_to_world[index]
        world = camera.reshape(-1, 3) @ pose[:3, :3].T + pose[:3, 3]
        result.append(world.reshape(height, width, 3))
    return np.stack(result).astype(np.float32)


def write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    vertex = np.empty(
        len(points),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")],
    )
    vertex["x"], vertex["y"], vertex["z"] = points.T
    colors = np.asarray(colors)
    if np.issubdtype(colors.dtype, np.floating):
        colors = np.clip(np.rint(colors * 255.0), 0, 255)
    colors = colors.astype(np.uint8)
    vertex["r"], vertex["g"], vertex["b"] = colors.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertex)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header)
        vertex.tofile(handle)


def export_point_cloud(prediction: dict, output: Path, max_points: int = 2_000_000,
                       confidence_percentile: float = 40.0) -> dict:
    depth = prediction["depth"].astype(np.float32)
    intrinsics = prediction["intrinsics"].astype(np.float32)
    camera_to_world = prediction["camera_to_world"].astype(np.float32)
    images = prediction["images"]
    confidence = prediction.get("confidence", np.ones_like(depth)).astype(np.float32)
    mask = prediction.get("mask", np.ones_like(depth, dtype=bool)).astype(bool)
    world = unproject_world(depth, intrinsics, camera_to_world)
    threshold = float(np.nanpercentile(confidence[np.isfinite(confidence)], confidence_percentile))
    valid = (
        np.isfinite(world).all(axis=-1) & np.isfinite(depth) & (depth > 0)
        & np.isfinite(confidence) & (confidence >= threshold) & mask
    )
    points, colors = world[valid], images[valid]
    if len(points) > max_points:
        keep = np.random.default_rng(0).choice(len(points), max_points, replace=False)
        points, colors = points[keep], colors[keep]
    write_binary_ply(output, points.astype(np.float32), colors)
    return {
        "ply": str(output.resolve()),
        "point_count": int(len(points)),
        "confidence_percentile": confidence_percentile,
        "confidence_threshold": threshold,
        "finite_fraction": float(np.isfinite(points).all(axis=-1).mean()),
    }


def depth_metrics(predicted: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    valid = mask & np.isfinite(predicted) & np.isfinite(target) & (predicted > 0) & (target > 0) & (target < 10.0)
    pred = predicted[valid].astype(np.float64)
    gt = target[valid].astype(np.float64)
    if not len(pred):
        raise ValueError("No valid overlapping depth pixels")
    scale = float(np.median(gt / pred))

    def compute(values: np.ndarray) -> dict:
        ratio = np.maximum(values / gt, gt / values)
        return {
            "abs_rel": float(np.mean(np.abs(values - gt) / gt)),
            "rmse_m": float(np.sqrt(np.mean((values - gt) ** 2))),
            "mae_m": float(np.mean(np.abs(values - gt))),
            "delta1": float(np.mean(ratio < 1.25)),
            "delta2": float(np.mean(ratio < 1.25 ** 2)),
        }

    return {
        "valid_pixel_count": int(len(pred)),
        "valid_pixel_fraction": float(valid.mean()),
        "median_metric_scale_gt_over_pred": scale,
        "raw_metric": compute(pred),
        "median_scale_aligned": compute(pred * scale),
        "pred_depth_median_m": float(np.median(pred)),
        "gt_depth_median_m": float(np.median(gt)),
    }


def umeyama(source: np.ndarray, target: np.ndarray, with_scale: bool):
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    source_centered, target_centered = source - source_mean, target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1] = -1
    rotation = u @ np.diag(correction) @ vt
    scale = float(np.sum(singular_values * correction) / np.mean(np.sum(source_centered ** 2, axis=1))) if with_scale else 1.0
    translation = target_mean - scale * (rotation @ source_mean)
    aligned = (scale * (rotation @ source.T)).T + translation
    return aligned, scale


def trajectory_metrics(predicted_poses: np.ndarray, target_poses: np.ndarray) -> dict:
    predicted = predicted_poses[:, :3, 3].astype(np.float64)
    target = target_poses[:, :3, 3].astype(np.float64)
    pred_step = np.linalg.norm(np.diff(predicted, axis=0), axis=1)
    target_step = np.linalg.norm(np.diff(target, axis=0), axis=1)
    aligned_se3, _ = umeyama(predicted, target, with_scale=False)
    aligned_sim3, scale = umeyama(predicted, target, with_scale=True)

    def rmse(values, reference):
        return float(np.sqrt(np.mean(np.sum((values - reference) ** 2, axis=1))))

    return {
        "frame_count": int(len(predicted)),
        "path_length_pred_m": float(pred_step.sum()),
        "path_length_gt_m": float(target_step.sum()),
        "path_length_ratio_pred_over_gt": float(pred_step.sum() / target_step.sum()),
        "sim3_scale_gt_over_pred": scale,
        "ate_rmse_se3_m": rmse(aligned_se3, target),
        "ate_rmse_sim3_m": rmse(aligned_sim3, target),
        "pred_step_median_m": float(np.median(pred_step)),
        "pred_step_max_m": float(np.max(pred_step)),
        "pred_step_max_over_median": float(np.max(pred_step) / max(np.median(pred_step), 1e-12)),
        "gt_step_median_m": float(np.median(target_step)),
        "gt_step_max_m": float(np.max(target_step)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--preprocess", choices=("da3_resize", "mapanything_cover_crop"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ply", type=Path)
    parser.add_argument("--max-points", type=int, default=2_000_000)
    args = parser.parse_args()

    with np.load(args.prediction) as archive:
        prediction = {key: archive[key] for key in archive.files}
    depth = prediction["depth"].astype(np.float32)
    confidence = prediction.get("confidence", np.ones_like(depth)).astype(np.float32)
    mask = prediction.get("mask", np.ones_like(depth, dtype=bool)).astype(bool)
    gt_depth, gt_poses, source_ids = load_ground_truth(args.data, len(depth), depth.shape[1:], args.preprocess)
    metrics = {
        "prediction": str(args.prediction.resolve()),
        "preprocess": args.preprocess,
        "source_frame_ids": source_ids.tolist(),
        "finite_depth_fraction": float(np.isfinite(depth).mean()),
        "depth": depth_metrics(depth, gt_depth, mask & np.isfinite(confidence)),
        "trajectory": trajectory_metrics(prediction["camera_to_world"], gt_poses),
    }
    if args.ply:
        metrics["point_cloud"] = export_point_cloud(prediction, args.ply, args.max_points)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
