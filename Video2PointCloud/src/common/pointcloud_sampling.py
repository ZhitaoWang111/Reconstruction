"""Sampling and I/O shared by evaluation and the aligned-cloud export.

Both the evaluator and `fit_sim3_from_gt.py` need the same point set: the
evaluation metrics are computed on it, and the canonical PLY handed to the mask
lifter is the very same cloud after the Sim(3) is applied. Keeping one
implementation is what makes those two descriptions true of the same file.
"""
from __future__ import annotations

import numpy as np
import open3d as o3d


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


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False, compressed=False):
        raise RuntimeError(f"Failed to write {path}")
