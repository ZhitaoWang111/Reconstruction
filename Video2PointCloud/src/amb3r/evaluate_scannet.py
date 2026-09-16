#!/usr/bin/env python3
"""Evaluate AMB3R/AMB3R-VO metric depth and trajectory on extracted ScanNet frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--mode", choices=("base", "vo"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--max-depth", type=float, default=10.0)
    return parser.parse_args()


def load_metric_depth(metadata_path: Path) -> np.ndarray:
    with np.load(metadata_path) as metadata:
        maximum_depth = float(metadata["maximum_depth"])
    encoded = np.asarray(Image.open(metadata_path.with_name(metadata_path.stem + "_depth.png")), dtype=np.float32)
    return encoded / 65535.0 * maximum_depth


def center_crop_resize_depth(depth: np.ndarray, target=(518, 392)) -> np.ndarray:
    target_width, target_height = target
    height, width = depth.shape
    target_aspect = target_width / target_height
    image_aspect = width / height
    if image_aspect > target_aspect:
        new_width = int(round(height * target_aspect))
        left = (width - new_width) // 2
        depth = depth[:, left : left + new_width]
    elif image_aspect < target_aspect:
        new_height = int(round(width / target_aspect))
        top = (height - new_height) // 2
        depth = depth[top : top + new_height]
    return np.asarray(
        Image.fromarray(depth.astype(np.float32), mode="F").resize(
            (target_width, target_height), Image.Resampling.NEAREST
        ),
        dtype=np.float32,
    )


def load_ground_truth(args: argparse.Namespace, frame_count: int):
    metadata_paths = sorted(args.data.glob("[0-9][0-9][0-9][0-9].npz"))[:frame_count]
    if len(metadata_paths) != frame_count:
        raise ValueError(f"Need {frame_count} metadata files, found {len(metadata_paths)}")
    poses = []
    source_ids = []
    for path in metadata_paths:
        with np.load(path) as metadata:
            poses.append(metadata["camera_pose"].astype(np.float64))
            source_ids.append(int(metadata["source_frame_id"]))

    if args.mode == "base":
        sys.path.insert(0, str(args.repo.resolve()))
        sys.path.insert(0, str((args.repo / "thirdparty").resolve()))
        from amb3r.datasets import Demo

        dataset = Demo(
            ROOT=str(args.data), resolution=(518, 392), num_seq=1, full_video=True,
            kf_every=1, disable_crop=False, max_images=frame_count
        )
        views, _ = dataset[0]
        depths = [np.asarray(view["depthmap"], dtype=np.float32) for view in views]
    else:
        depths = [center_crop_resize_depth(load_metric_depth(path)) for path in metadata_paths]
    return np.stack(depths), np.stack(poses), np.asarray(source_ids, dtype=np.int64)


def world_points_to_depth(points: np.ndarray, poses: np.ndarray) -> np.ndarray:
    depths = []
    for pointmap, camera_to_world in zip(points, poses):
        rotation = camera_to_world[:3, :3]
        translation = camera_to_world[:3, 3]
        camera_points = (pointmap.reshape(-1, 3) - translation) @ rotation
        depths.append(camera_points[:, 2].reshape(pointmap.shape[:2]))
    return np.stack(depths)


def depth_metrics(predicted: np.ndarray, target: np.ndarray, confidence: np.ndarray,
                  threshold: float, max_depth: float) -> dict:
    valid = (
        np.isfinite(predicted) & np.isfinite(target) & np.isfinite(confidence)
        & (predicted > 0) & (target > 0) & (target < max_depth) & (confidence >= threshold)
    )
    pred = predicted[valid].astype(np.float64)
    gt = target[valid].astype(np.float64)
    if len(pred) == 0:
        raise ValueError("No valid overlapping depth pixels")
    scale = float(np.median(gt / pred))

    def compute(p: np.ndarray) -> dict:
        ratio = np.maximum(p / gt, gt / p)
        return {
            "abs_rel": float(np.mean(np.abs(p - gt) / gt)),
            "rmse_m": float(np.sqrt(np.mean((p - gt) ** 2))),
            "mae_m": float(np.mean(np.abs(p - gt))),
            "delta1": float(np.mean(ratio < 1.25)),
            "delta2": float(np.mean(ratio < 1.25 ** 2)),
        }

    return {
        "valid_pixel_count": int(len(pred)),
        "valid_pixel_fraction": float(valid.mean()),
        "median_metric_scale_gt_over_pred": scale,
        "raw_metric": compute(pred),
        "median_scale_aligned": compute(pred * scale),
        "pred_depth_m": {
            "p05": float(np.percentile(pred, 5)),
            "median": float(np.median(pred)),
            "p95": float(np.percentile(pred, 95)),
        },
        "gt_depth_m": {
            "p05": float(np.percentile(gt, 5)),
            "median": float(np.median(gt)),
            "p95": float(np.percentile(gt, 95)),
        },
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
    if with_scale:
        variance = np.mean(np.sum(source_centered ** 2, axis=1))
        scale = float(np.sum(singular_values * correction) / variance)
    else:
        scale = 1.0
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

    def rmse(a, b):
        return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))

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
    args = parse_args()
    with np.load(args.prediction) as prediction:
        points = prediction["pts"].astype(np.float32)
        confidence = prediction["conf"].astype(np.float32)
        poses = prediction["pose"].astype(np.float64)
    if points.ndim == 5 and points.shape[0] == 1:
        points = points[0]
    if confidence.ndim == 4 and confidence.shape[0] == 1:
        confidence = confidence[0]
    if poses.ndim == 4 and poses.shape[0] == 1:
        poses = poses[0]

    target_depth, target_poses, source_ids = load_ground_truth(args, len(points))
    predicted_depth = world_points_to_depth(points, poses)
    if predicted_depth.shape != target_depth.shape:
        raise ValueError(f"Depth shape mismatch: predicted {predicted_depth.shape}, target {target_depth.shape}")
    if confidence.shape != predicted_depth.shape:
        raise ValueError(f"Confidence shape mismatch: {confidence.shape} vs {predicted_depth.shape}")

    metrics = {
        "mode": args.mode,
        "prediction": str(args.prediction.resolve()),
        "source_frame_ids": source_ids.tolist(),
        "finite_point_fraction": float(np.isfinite(points).all(axis=-1).mean()),
        "point_bbox_extent_m": (np.nanmax(points, axis=(0, 1, 2)) - np.nanmin(points, axis=(0, 1, 2))).tolist(),
        "depth": depth_metrics(
            predicted_depth, target_depth, confidence, args.confidence_threshold, args.max_depth
        ),
        "trajectory": trajectory_metrics(poses, target_poses),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
