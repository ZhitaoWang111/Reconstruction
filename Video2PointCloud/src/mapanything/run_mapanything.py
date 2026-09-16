#!/usr/bin/env python3
"""Run the official MapAnything image-only model and save standardized results."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="facebook/map-anything")
    parser.add_argument("--cache-dir", type=Path, default=Path("/home/wzt/HDD/huggingface"))
    parser.add_argument("--num-frames", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this MapAnything test")
    paths = sorted(args.images.glob("*.jpg"))[: args.num_frames]
    metadata_paths = sorted(args.metadata.glob("[0-9][0-9][0-9][0-9].npz"))[: len(paths)]
    if not paths or len(paths) != len(metadata_paths):
        raise ValueError(f"Found {len(paths)} images and {len(metadata_paths)} metadata files")
    source_ids = []
    for path in metadata_paths:
        with np.load(path) as metadata:
            source_ids.append(int(metadata["source_frame_id"]))

    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    args.output.mkdir(parents=True, exist_ok=True)
    load_start = time.perf_counter()
    model = MapAnything.from_pretrained(args.model, cache_dir=str(args.cache_dir)).to("cuda").eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    views = load_images([str(path) for path in paths], verbose=True)

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    predictions = model.infer(
        views,
        memory_efficient_inference=True,
        minibatch_size=1,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
        apply_confidence_mask=False,
    )
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - start

    depth = np.stack([pred["depth_z"][0, ..., 0].float().cpu().numpy() for pred in predictions])
    confidence = np.stack([pred["conf"][0].float().cpu().numpy() for pred in predictions])
    intrinsics = np.stack([pred["intrinsics"][0].float().cpu().numpy() for pred in predictions])
    camera_to_world = np.stack([pred["camera_poses"][0].float().cpu().numpy() for pred in predictions])
    images = np.stack([
        np.clip(np.rint(pred["img_no_norm"][0].float().cpu().numpy() * 255), 0, 255).astype(np.uint8)
        for pred in predictions
    ])
    mask = np.stack([pred["mask"][0, ..., 0].bool().cpu().numpy() for pred in predictions])
    scaling = np.stack([pred["metric_scaling_factor"][0].float().cpu().numpy() for pred in predictions])
    result_path = args.output / "prediction.npz"
    np.savez_compressed(
        result_path,
        depth=depth.astype(np.float32),
        confidence=confidence.astype(np.float32),
        intrinsics=intrinsics.astype(np.float32),
        camera_to_world=camera_to_world.astype(np.float32),
        images=images,
        mask=mask,
        metric_scaling_factor=scaling.astype(np.float32),
        source_frame_ids=np.asarray(source_ids, dtype=np.int64),
    )
    summary = {
        "model": args.model,
        "input_mode": "rgb_only",
        "memory_efficient_inference": True,
        "minibatch_size": 1,
        "num_frames": len(paths),
        "source_frame_ids": source_ids,
        "processed_shape": list(depth.shape),
        "model_load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "fps": len(paths) / inference_seconds,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "finite_depth_fraction": float(np.isfinite(depth).mean()),
        "metric_scaling_factor_median": float(np.median(scaling)),
        "prediction": str(result_path.resolve()),
    }
    (args.output / "run_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
