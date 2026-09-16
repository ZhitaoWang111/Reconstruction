#!/usr/bin/env python3
"""Convert official DA3-Streaming outputs to the shared evaluation schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--streaming-output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=float)
    parser.add_argument("--max-rss-kib", type=int)
    args = parser.parse_args()

    frame_paths = sorted(
        (args.streaming_output / "results_output").glob("frame_*.npz"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    if not frame_paths:
        raise ValueError("No DA3-Streaming frame archives found")
    expected = [f"frame_{index}.npz" for index in range(len(frame_paths))]
    if [path.name for path in frame_paths] != expected:
        raise ValueError("DA3-Streaming frame archives are not consecutive")

    poses = np.loadtxt(args.streaming_output / "camera_poses.txt", dtype=np.float64).reshape(-1, 4, 4)
    if len(poses) != len(frame_paths):
        raise ValueError(f"Pose/frame mismatch: {len(poses)} vs {len(frame_paths)}")

    depths, confidence, intrinsics, images = [], [], [], []
    for path in frame_paths:
        with np.load(path) as frame:
            depths.append(frame["depth"].astype(np.float32))
            confidence.append(frame["conf"].astype(np.float32))
            intrinsics.append(frame["intrinsics"].astype(np.float32))
            images.append(frame["image"].astype(np.uint8))

    metadata_paths = sorted(args.metadata.glob("[0-9][0-9][0-9][0-9].npz"))[: len(frame_paths)]
    if len(metadata_paths) != len(frame_paths):
        raise ValueError(f"Metadata/frame mismatch: {len(metadata_paths)} vs {len(frame_paths)}")
    source_ids = []
    for path in metadata_paths:
        with np.load(path) as metadata:
            source_ids.append(int(metadata["source_frame_id"]))

    depth = np.stack(depths)
    confidence_array = np.stack(confidence)
    mask = np.isfinite(depth) & (depth > 0) & np.isfinite(confidence_array)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        depth=depth,
        confidence=confidence_array,
        intrinsics=np.stack(intrinsics),
        camera_to_world=poses.astype(np.float32),
        images=np.stack(images),
        mask=mask,
        source_frame_ids=np.asarray(source_ids, dtype=np.int64),
        streaming_chunk_size=np.int64(30),
        streaming_overlap=np.int64(15),
        streaming_loop_closure=np.int64(0),
    )
    summary = {
        "input_mode": "rgb_only",
        "official_mode": "DA3-Streaming",
        "num_frames": len(frame_paths),
        "processed_shape": list(depth.shape),
        "chunk_size": 30,
        "overlap": 15,
        "alignment": "official dense Sim(3), adjacent windows only",
        "loop_closure": False,
        "wall_seconds": args.wall_seconds,
        "max_rss_kib": args.max_rss_kib,
        "finite_depth_fraction": float(np.isfinite(depth).mean()),
        "prediction": str(args.output.resolve()),
    }
    summary_path = args.output.with_name("run_metrics.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
