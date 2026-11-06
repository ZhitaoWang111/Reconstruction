#!/usr/bin/env python3
"""Fit the Sim(3) that maps a predicted trajectory into the ScanNet scan frame.

This is the step that puts AMB3R's geometry where the rest of the pipeline
expects it: ScanNet world coordinates, metres. It is separate from evaluation
because the mask lifter consumes its output — the geometry used downstream is
calibrated against GT camera poses, and this file is where that happens.

What is fitted, and against what
--------------------------------
Input is two ordered lists of camera centres, both pairs describing the same
photographs:

  source  predicted centres: ``pose[i][:3, 3]`` from the AMB3R archive. These
          are camera-to-world translations in AMB3R's own world, which has an
          arbitrary origin, orientation and scale.
  target  GT centres: ``camera_pose[i][:3, 3]`` from ScanNet's per-frame
          metadata (``input/NNNN.npz``), already in the scan frame.

The fit solves ``target ≈ scale * R @ source + t`` in the least-squares sense
(Umeyama). Three quantities come out:

  R      rotation  — AMB3R's world axes → scan axes
  scale  scale     — AMB3R's units → metres
  t      translation — AMB3R's origin → scan origin

SE(3) and Sim(3) differ only in whether ``scale`` is free. SE(3) keeps the
prediction's native scale (``scale = 1``), so it measures the trajectory error
the model itself would be judged on; Sim(3) absorbs the scale error, so its
residual is what remains after a single global scale is corrected. Both are
reported because the gap between them is the scale error, which is the part of
the error a method without GT would have to estimate some other way.

Why camera centres and not points
---------------------------------
Centres are available for every frame and are exact to the pose prediction;
matching point clouds instead would need nearest-neighbour correspondences
that the depth error would bias. The fitted transform is then applied to all
points, so the whole cloud inherits this one rigid transform.

The ATE reported here is RMS distance between aligned predicted centres and GT
centres, in metres, over the frames that had finite values on both sides.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_gt_poses(data: Path, frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Read GT camera-to-world matrices for the first ``frame_count`` frames.

    ``data`` is the extracted ScanNet directory (``.../input``) whose files are
    named ``NNNN.npz`` in trajectory order; sorting them by name therefore
    reproduces the order the prediction was made in.
    """
    paths = sorted(Path(data).glob("[0-9][0-9][0-9][0-9].npz"))[:frame_count]
    if len(paths) != frame_count:
        raise ValueError(f"Need {frame_count} metadata files, found {len(paths)}")
    poses, source_ids = [], []
    for path in paths:
        with np.load(path) as metadata:
            poses.append(metadata["camera_pose"].astype(np.float64))
            source_ids.append(int(metadata["source_frame_id"]))
    return np.stack(poses), np.asarray(source_ids, dtype=np.int64)


def predicted_poses_from_archive(prediction: Path) -> np.ndarray:
    """Read the archive's ``pose`` array as (frames, 4, 4), camera-to-world."""
    with np.load(prediction) as archive:
        if "pose" not in archive.files:
            raise ValueError(f"{prediction} has no 'pose' array: {archive.files}")
        poses = archive["pose"].astype(np.float64)
    # Some archives keep a leading batch axis of size one.
    if poses.ndim == 4 and poses.shape[0] == 1:
        poses = poses[0]
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected pose shape (frames, 4, 4), got {poses.shape}")
    return poses


def umeyama_transform(source: np.ndarray, target: np.ndarray, with_scale: bool):
    """Least-squares similarity ``target ≈ scale * R @ source + t``.

    ``with_scale=False`` fixes the scale at 1, which is plain Umeyama SE(3).
    The reflection correction (``correction[-1] = -1`` when ``det(u @ vt) < 0``)
    keeps the result a rotation rather than a mirror, which a degenerate cloud
    layout could otherwise produce.

    Returns ``(rotation, scale, translation, ate_m, correspondence_count)``.
    """
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


def fit_alignment(predicted_poses: np.ndarray, gt_poses: np.ndarray) -> dict:
    """Fit both SE(3) and Sim(3) and return the records written to JSON."""
    predicted_centres = predicted_poses[:, :3, 3]
    gt_centres = gt_poses[:, :3, 3]
    se3 = umeyama_transform(predicted_centres, gt_centres, with_scale=False)
    sim3 = umeyama_transform(predicted_centres, gt_centres, with_scale=True)

    def record(fit):
        rotation, scale, translation, ate, count = fit
        return {
            "trajectory_ate_rmse_m": ate,
            "camera_correspondence_count": count,
            "alignment_scale": scale,
            "alignment_rotation": rotation.tolist(),
            "alignment_translation_m": translation.tolist(),
        }

    return {"se3_native_scale": record(se3), "sim3_oracle_scale": record(sim3)}


def apply_sim3(points: np.ndarray, record: dict) -> np.ndarray:
    """Apply a fitted record to world points; the lifter uses the same maths."""
    rotation = np.asarray(record["alignment_rotation"], dtype=np.float64)
    scale = float(record["alignment_scale"])
    translation = np.asarray(record["alignment_translation_m"], dtype=np.float64)
    return (scale * (np.asarray(points, dtype=np.float64) @ rotation.T) + translation).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prediction", type=Path, required=True,
                        help="AMB3R archive containing pose (already at metric-head scale)")
    parser.add_argument("--data", type=Path, required=True,
                        help="Extracted ScanNet input directory holding NNNN.npz per frame")
    parser.add_argument("--output", type=Path, required=True,
                        help="JSON to write the fitted transforms to")
    parser.add_argument("--sim3-ply", type=Path,
                        help="Optional: write the aligned canonical cloud here")
    parser.add_argument("--se3-ply", type=Path,
                        help="Optional: write the SE(3)-aligned cloud (native scale) here")
    parser.add_argument("--max-points", type=int, default=500_000,
                        help="Point budget for the exported clouds (evaluator default: 500000)")
    parser.add_argument("--confidence-percentile", type=float, default=40.0,
                        help="Confidence percentile kept when sampling (evaluator default: 40)")
    return parser.parse_args()


def write_aligned_cloud(prediction: Path, record: dict, output: Path,
                        max_points: int = 500_000, percentile: float = 40.0) -> int:
    """Sample the archive, apply the record, and write the canonical PLY.

    This is the cloud the mask lifter reads as its scene backdrop, so it has to
    be produced by the same sampling the evaluator measures — that is why the
    sampling lives in `pointcloud_sampling` and not in the evaluator. The PLY
    and the transform therefore come from one fit and one point set.

    Imports are local so a caller that only wants the JSON does not need open3d.
    """
    from pointcloud_sampling import sample_amb3r, transform_points, write_ply

    with np.load(prediction) as archive:
        prediction_data = {key: archive[key] for key in archive.files}
    points, colors, _poses, _threshold = sample_amb3r(prediction_data, max_points, percentile)
    aligned = transform_points(
        points, np.asarray(record["alignment_rotation"]), record["alignment_scale"],
        np.asarray(record["alignment_translation_m"]))
    write_ply(output, aligned, colors)
    return int(len(aligned))


def main() -> None:
    args = parse_args()
    predicted = predicted_poses_from_archive(args.prediction)
    gt_poses, source_ids = load_gt_poses(args.data, len(predicted))
    result = {
        "prediction": str(args.prediction.resolve()),
        "prediction_format": "amb3r",
        "gt_metadata": str(Path(args.data).resolve()),
        "source_frame_ids": source_ids.tolist(),
        "frame_count": int(len(predicted)),
        **fit_alignment(predicted, gt_poses),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for record_key, ply_key, path in (("se3_native_scale", "se3_ply", args.se3_ply),
                                      ("sim3_oracle_scale", "sim3_ply", args.sim3_ply)):
        if path:
            points = write_aligned_cloud(args.prediction, result[record_key], path,
                                         args.max_points, args.confidence_percentile)
            result[ply_key] = str(path.resolve())
            result.setdefault("exported_point_count", points)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("frame_count", "se3_native_scale", "sim3_oracle_scale")},
                     indent=2))


if __name__ == "__main__":
    main()
