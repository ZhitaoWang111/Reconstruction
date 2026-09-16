#!/usr/bin/env python3
"""Evaluate a predicted point cloud against a ScanNet mesh.

The predicted cloud is aligned from model coordinates to ScanNet coordinates
using corresponding predicted/GT camera centres.  SE(3) results preserve the
model's native scale; Sim(3) results are an oracle-scale diagnostic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--format", choices=("common", "amb3r"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--gt-mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--se3-ply", type=Path)
    parser.add_argument("--sim3-ply", type=Path)
    parser.add_argument("--max-pred-points", type=int, default=500_000)
    parser.add_argument("--gt-sample-points", type=int, default=500_000)
    parser.add_argument("--confidence-percentile", type=float, default=40.0)
    return parser.parse_args()


def load_gt_poses(data: Path, frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    paths = sorted(data.glob("[0-9][0-9][0-9][0-9].npz"))[:frame_count]
    if len(paths) != frame_count:
        raise ValueError(f"Need {frame_count} metadata files, found {len(paths)}")
    poses, source_ids = [], []
    for path in paths:
        with np.load(path) as metadata:
            poses.append(metadata["camera_pose"].astype(np.float64))
            source_ids.append(int(metadata["source_frame_id"]))
    return np.stack(poses), np.asarray(source_ids, dtype=np.int64)


def umeyama_transform(source: np.ndarray, target: np.ndarray, with_scale: bool):
    valid = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source, target = source[valid], target[valid]
    if len(source) < 3:
        raise ValueError("At least three finite camera correspondences are required")
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    source_centered, target_centered = source - source_mean, target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1] = -1
    rotation = u @ np.diag(correction) @ vt
    variance = np.mean(np.sum(source_centered ** 2, axis=1))
    scale = float(np.sum(singular_values * correction) / variance) if with_scale else 1.0
    translation = target_mean - scale * (rotation @ source_mean)
    aligned = (scale * (rotation @ source.T)).T + translation
    ate = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    return rotation, scale, translation, ate, int(valid.sum())


def sample_indices(valid: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    indices = np.flatnonzero(valid)
    if len(indices) > limit:
        indices = rng.choice(indices, limit, replace=False)
    return indices


def sample_common(prediction: dict, max_points: int, percentile: float):
    depth = prediction["depth"].astype(np.float32)
    confidence = prediction.get("confidence", np.ones_like(depth)).astype(np.float32)
    mask = prediction.get("mask", np.ones_like(depth, dtype=bool)).astype(bool)
    intrinsics = prediction["intrinsics"].astype(np.float64)
    poses = prediction["camera_to_world"].astype(np.float64)
    images = prediction.get("images")
    finite_confidence = confidence[np.isfinite(confidence)]
    threshold = float(np.percentile(finite_confidence, percentile))
    per_frame = max(1, max_points // len(depth))
    rng = np.random.default_rng(0)
    point_parts, color_parts = [], []
    height, width = depth.shape[1:]
    for frame in range(len(depth)):
        z = depth[frame].reshape(-1)
        valid = (
            np.isfinite(z) & (z > 0) & np.isfinite(confidence[frame].reshape(-1))
            & (confidence[frame].reshape(-1) >= threshold) & mask[frame].reshape(-1)
        )
        selected = sample_indices(valid, per_frame, rng)
        v, u = np.divmod(selected, width)
        k = intrinsics[frame]
        camera = np.stack(
            ((u - k[0, 2]) / k[0, 0] * z[selected],
             (v - k[1, 2]) / k[1, 1] * z[selected], z[selected]), axis=1
        )
        pose = poses[frame]
        world = camera @ pose[:3, :3].T + pose[:3, 3]
        finite = np.isfinite(world).all(axis=1)
        point_parts.append(world[finite].astype(np.float32))
        if images is not None:
            color_parts.append(images[frame].reshape(-1, 3)[selected][finite])
    points = np.concatenate(point_parts)
    colors = np.concatenate(color_parts) if color_parts else np.full((len(points), 3), 180, np.uint8)
    return points, colors, poses, threshold


def sample_amb3r(prediction: dict, max_points: int, percentile: float):
    points = prediction["pts"].astype(np.float32)
    confidence = prediction["conf"].astype(np.float32)
    poses = prediction["pose"].astype(np.float64)
    images = prediction.get("images")
    for name, array in (("pts", points), ("conf", confidence), ("pose", poses)):
        if array.shape[0] == 1 and array.ndim in (4, 5):
            if name == "pts": points = points[0]
            elif name == "conf": confidence = confidence[0]
            else: poses = poses[0]
    if images is not None:
        images = np.asarray(images)
        if images.shape[0] == 1 and images.ndim == 5:
            images = images[0]
        if len(images) != len(points):
            raise ValueError(
                f"AMB3R images/point maps disagree: {len(images)} != {len(points)}"
            )
    finite_confidence = confidence[np.isfinite(confidence)]
    threshold = float(np.percentile(finite_confidence, percentile))
    per_frame = max(1, max_points // len(points))
    rng = np.random.default_rng(0)
    parts, color_parts = [], []
    for frame_index, (pointmap, conf) in enumerate(zip(points, confidence)):
        flat = pointmap.reshape(-1, 3)
        valid = np.isfinite(flat).all(axis=1) & np.isfinite(conf.reshape(-1)) & (conf.reshape(-1) >= threshold)
        selected = sample_indices(valid, per_frame, rng)
        parts.append(flat[selected])
        if images is not None:
            image = images[frame_index]
            if image.ndim != 3:
                raise ValueError(f"Expected a 3D AMB3R image, got {image.shape}")
            if image.shape[0] == 3:
                image = np.moveaxis(image, 0, -1)
            if image.shape != pointmap.shape:
                raise ValueError(
                    f"AMB3R image/point-map shapes disagree: {image.shape} != {pointmap.shape}"
                )
            image = image.astype(np.float32)
            if image.min() >= -1.01 and image.max() <= 1.01:
                image = (image + 1.0) * 127.5
            elif image.max() <= 1.01:
                image = image * 255.0
            color_parts.append(np.clip(np.rint(image), 0, 255).astype(np.uint8).reshape(-1, 3)[selected])
    sampled = np.concatenate(parts).astype(np.float32)
    colors = np.concatenate(color_parts) if color_parts else np.full((len(sampled), 3), 180, np.uint8)
    return sampled, colors, poses, threshold


def transform_points(points: np.ndarray, rotation: np.ndarray, scale: float, translation: np.ndarray):
    return (scale * (points.astype(np.float64) @ rotation.T) + translation).astype(np.float32)


def sample_gt_mesh(path: Path, count: int) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty() or not mesh.has_triangles():
        raise ValueError(f"Not a triangle mesh: {path}")
    o3d.utility.random.seed(0)
    cloud = mesh.sample_points_uniformly(number_of_points=count)
    return np.asarray(cloud.points, dtype=np.float32)


def distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    tree = cKDTree(target)
    # A single worker is intentional: large all-core queries proved unstable in
    # the AMB3R environment and can create avoidable host-memory pressure.
    result, _ = tree.query(source, k=1, workers=1)
    return result


def geometry_metrics(predicted: np.ndarray, target: np.ndarray) -> dict:
    accuracy = distances(predicted, target)
    completeness = distances(target, predicted)
    result = {
        "accuracy_mean_m": float(accuracy.mean()),
        "accuracy_median_m": float(np.median(accuracy)),
        "completeness_mean_m": float(completeness.mean()),
        "completeness_median_m": float(np.median(completeness)),
        "chamfer_l1_m": float((accuracy.mean() + completeness.mean()) / 2.0),
    }
    for threshold in (0.05, 0.10, 0.20):
        precision = float(np.mean(accuracy < threshold))
        recall = float(np.mean(completeness < threshold))
        f_score = 2.0 * precision * recall / max(precision + recall, 1e-12)
        key = f"{int(threshold * 100):02d}cm"
        result[f"precision_{key}"] = precision
        result[f"recall_{key}"] = recall
        result[f"fscore_{key}"] = f_score
    return result


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False, compressed=False):
        raise RuntimeError(f"Failed to write {path}")


def main() -> None:
    args = parse_args()
    with np.load(args.prediction) as archive:
        prediction = {key: archive[key] for key in archive.files}
    if args.format == "common":
        points, colors, predicted_poses, confidence_threshold = sample_common(
            prediction, args.max_pred_points, args.confidence_percentile
        )
    else:
        points, colors, predicted_poses, confidence_threshold = sample_amb3r(
            prediction, args.max_pred_points, args.confidence_percentile
        )
    gt_poses, source_ids = load_gt_poses(args.data, len(predicted_poses))
    predicted_centres = predicted_poses[:, :3, 3]
    gt_centres = gt_poses[:, :3, 3]
    se3 = umeyama_transform(predicted_centres, gt_centres, with_scale=False)
    sim3 = umeyama_transform(predicted_centres, gt_centres, with_scale=True)
    se3_points = transform_points(points, *se3[:3])
    sim3_points = transform_points(points, *sim3[:3])
    gt_points = sample_gt_mesh(args.gt_mesh, args.gt_sample_points)
    result = {
        "prediction": str(args.prediction.resolve()),
        "prediction_format": args.format,
        "gt_mesh": str(args.gt_mesh.resolve()),
        "source_frame_ids": source_ids.tolist(),
        "frame_count": int(len(predicted_poses)),
        "predicted_point_count": int(len(points)),
        "gt_sample_point_count": int(len(gt_points)),
        "sampling": {
            "seed": 0,
            "confidence_percentile": args.confidence_percentile,
            "confidence_threshold": confidence_threshold,
            "max_pred_points": args.max_pred_points,
        },
        "se3_native_scale": {
            "trajectory_ate_rmse_m": se3[3],
            "camera_correspondence_count": se3[4],
            "alignment_scale": se3[1],
            "alignment_rotation": se3[0].tolist(),
            "alignment_translation_m": se3[2].tolist(),
            **geometry_metrics(se3_points, gt_points),
        },
        "sim3_oracle_scale": {
            "trajectory_ate_rmse_m": sim3[3],
            "camera_correspondence_count": sim3[4],
            "alignment_scale": sim3[1],
            "alignment_rotation": sim3[0].tolist(),
            "alignment_translation_m": sim3[2].tolist(),
            **geometry_metrics(sim3_points, gt_points),
        },
    }
    if args.se3_ply:
        write_ply(args.se3_ply, se3_points, colors)
        result["se3_ply"] = str(args.se3_ply.resolve())
    if args.sim3_ply:
        write_ply(args.sim3_ply, sim3_points, colors)
        result["sim3_ply"] = str(args.sim3_ply.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
