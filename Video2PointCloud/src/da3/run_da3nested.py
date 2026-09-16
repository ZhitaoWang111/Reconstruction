#!/usr/bin/env python3
"""Run the official DA3Nested model and save a standardized RGB-only result."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    parser.add_argument("--cache-dir", type=Path, default=Path("/home/wzt/HDD/huggingface"))
    parser.add_argument("--num-frames", type=int, default=40)
    parser.add_argument("--process-res", type=int, default=504)
    return parser.parse_args()


def homogeneous_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics
    if extrinsics.shape[-2:] != (3, 4):
        raise ValueError(f"Unexpected extrinsics shape {extrinsics.shape}")
    bottom = np.broadcast_to(np.asarray([0, 0, 0, 1], dtype=extrinsics.dtype), (*extrinsics.shape[:-2], 1, 4))
    return np.concatenate((extrinsics, bottom), axis=-2)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DA3Nested-Giant-Large")
    paths = sorted(args.images.glob("*.jpg"))[: args.num_frames]
    metadata_paths = sorted(args.metadata.glob("[0-9][0-9][0-9][0-9].npz"))[: len(paths)]
    if not paths or len(paths) != len(metadata_paths):
        raise ValueError(f"Found {len(paths)} images and {len(metadata_paths)} metadata files")
    source_ids = []
    for path in metadata_paths:
        with np.load(path) as metadata:
            source_ids.append(int(metadata["source_frame_id"]))

    from depth_anything_3.api import DepthAnything3

    args.output.mkdir(parents=True, exist_ok=True)
    load_start = time.perf_counter()
    model = DepthAnything3.from_pretrained(args.model, cache_dir=str(args.cache_dir)).to("cuda").eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    prediction = model.inference(
        [str(path) for path in paths],
        process_res=args.process_res,
        process_res_method="upper_bound_resize",
        ref_view_strategy="middle",
    )
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - start

    depth = np.asarray(prediction.depth, dtype=np.float32)
    confidence = np.ones_like(depth) if prediction.conf is None else np.asarray(prediction.conf, dtype=np.float32)
    extrinsics = homogeneous_extrinsics(np.asarray(prediction.extrinsics, dtype=np.float64))
    camera_to_world = np.linalg.inv(extrinsics).astype(np.float32)
    intrinsics = np.asarray(prediction.intrinsics, dtype=np.float32)
    images = np.asarray(prediction.processed_images, dtype=np.uint8)
    sky = np.zeros_like(depth, dtype=bool) if prediction.sky is None else np.asarray(prediction.sky, dtype=bool)
    mask = np.isfinite(depth) & (depth > 0) & ~sky
    result_path = args.output / "prediction.npz"
    np.savez_compressed(
        result_path,
        depth=depth,
        confidence=confidence,
        intrinsics=intrinsics,
        camera_to_world=camera_to_world,
        images=images,
        mask=mask,
        sky=sky,
        source_frame_ids=np.asarray(source_ids, dtype=np.int64),
        is_metric=np.int64(prediction.is_metric),
        scale_factor=np.float32(prediction.scale_factor),
    )
    summary = {
        "model": args.model,
        "input_mode": "rgb_only",
        "num_frames": len(paths),
        "source_frame_ids": source_ids,
        "processed_shape": list(depth.shape),
        "reported_is_metric": int(prediction.is_metric),
        "predicted_scale_factor": float(prediction.scale_factor),
        "model_load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "fps": len(paths) / inference_seconds,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "finite_depth_fraction": float(np.isfinite(depth).mean()),
        "prediction": str(result_path.resolve()),
    }
    (args.output / "run_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
