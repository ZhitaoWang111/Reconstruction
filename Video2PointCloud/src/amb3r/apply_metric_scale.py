#!/usr/bin/env python3
"""Apply AMB3R's own metric-head scale to an official AMB3R-VO result.

This is deliberately a separate post-processing step: the official VO archive is
kept unchanged, and the provenance JSON records exactly where the scale came from.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--scale-reference", type=Path, required=True)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--output-prediction", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.scale_reference) as reference:
        scale = float(reference["metric_scale"])

    with np.load(args.prediction) as source:
        result = {key: source[key] for key in source.files}
    result["pts"] = result["pts"].astype(np.float32, copy=False) * np.float32(scale)
    poses = result["pose"].astype(np.float32, copy=True)
    poses[..., :3, 3] *= np.float32(scale)
    result["pose"] = poses
    result["metric_scale"] = np.float32(scale)

    args.output_prediction.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_prediction, **result)

    cloud = o3d.io.read_point_cloud(str(args.input_ply))
    points = np.asarray(cloud.points)
    if len(points) == 0 or not np.isfinite(points).all():
        raise ValueError("Input PLY is empty or contains non-finite points")
    cloud.points = o3d.utility.Vector3dVector(points * scale)
    args.output_ply.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(args.output_ply), cloud, write_ascii=False, compressed=False):
        raise RuntimeError(f"Failed to write {args.output_ply}")

    provenance = {
        "operation": "global_metric_scale_postprocess",
        "official_vo_prediction_unchanged": str(args.prediction.resolve()),
        "metric_scale": scale,
        "metric_scale_source": str(args.scale_reference.resolve()),
        "metric_scale_source_description": (
            "AMB3R frontend depth_metric/depth median on the first 20 evaluated RGB frames"
        ),
        "output_prediction": str(args.output_prediction.resolve()),
        "output_ply": str(args.output_ply.resolve()),
        "point_count": int(len(points)),
        "warning": (
            "This is a local global-scale postprocess, not the unmodified official AMB3R-VO output."
        ),
    }
    args.provenance.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
