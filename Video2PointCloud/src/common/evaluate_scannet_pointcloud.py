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

from pointcloud_sampling import sample_amb3r, sample_common, sample_indices, transform_points, write_ply  # noqa: F401

# The Sim(3) that puts the prediction in the scan frame is also the geometry the
# mask lifter consumes, so it lives in its own module. Fitting it here as well
# would let evaluation and the published cloud drift apart.
from fit_sim3_from_gt import fit_alignment, load_gt_poses, umeyama_transform  # noqa: F401


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
    alignment = fit_alignment(predicted_poses, gt_poses)
    se3, sim3 = alignment["se3_native_scale"], alignment["sim3_oracle_scale"]
    se3_points = transform_points(
        points, np.asarray(se3["alignment_rotation"]), se3["alignment_scale"],
        np.asarray(se3["alignment_translation_m"]))
    sim3_points = transform_points(
        points, np.asarray(sim3["alignment_rotation"]), sim3["alignment_scale"],
        np.asarray(sim3["alignment_translation_m"]))
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
        "se3_native_scale": {**se3, **geometry_metrics(se3_points, gt_points)},
        "sim3_oracle_scale": {**sim3, **geometry_metrics(sim3_points, gt_points)},
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
