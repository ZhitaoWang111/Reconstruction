#!/usr/bin/env python3
"""Run MapAnything on a long sequence with overlapping serial windows.

MapAnything has no upstream long-sequence runner.  Each window uses the
official RGB-only infer path.  Adjacent windows are joined using only their
overlapping predicted camera centres; ScanNet GT is never used for stitching.
"""

from __future__ import annotations

import argparse
import gc
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
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/home/wzt/HDD/huggingface"))
    parser.add_argument("--num-frames", type=int, default=500)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--overlap", type=int, default=15)
    return parser.parse_args()


def umeyama(source: np.ndarray, target: np.ndarray):
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    source_centered, target_centered = source - source_mean, target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1] = -1
    rotation = u @ np.diag(correction) @ vt
    variance = np.mean(np.sum(source_centered ** 2, axis=1))
    scale = float(np.sum(singular_values * correction) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    aligned = scale * (source @ rotation.T) + translation
    rmse = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    return scale, rotation, translation, rmse


def transform_poses(poses: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray):
    result = poses.copy()
    result[:, :3, :3] = rotation @ poses[:, :3, :3]
    result[:, :3, 3] = scale * (poses[:, :3, 3] @ rotation.T) + translation
    return result


def to_numpy(predictions):
    def stack(name, selector):
        return np.stack([selector(pred[name]) for pred in predictions])

    return {
        "depth": stack("depth_z", lambda x: x[0, ..., 0].float().cpu().numpy()),
        "confidence": stack("conf", lambda x: x[0].float().cpu().numpy()),
        "intrinsics": stack("intrinsics", lambda x: x[0].float().cpu().numpy()),
        "camera_to_world": stack("camera_poses", lambda x: x[0].float().cpu().numpy()),
        "images": stack(
            "img_no_norm",
            lambda x: np.clip(np.rint(x[0].float().cpu().numpy() * 255), 0, 255).astype(np.uint8),
        ),
        "mask": stack("mask", lambda x: x[0, ..., 0].bool().cpu().numpy()),
        "metric_scaling_factor": stack(
            "metric_scaling_factor", lambda x: np.asarray(x[0].float().cpu().numpy()).reshape(-1)
        ),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 0 < args.overlap < args.chunk_size:
        raise ValueError("Require 0 < overlap < chunk-size")
    image_paths = sorted(args.images.glob("*.jpg"))[: args.num_frames]
    metadata_paths = sorted(args.metadata.glob("[0-9][0-9][0-9][0-9].npz"))[: len(image_paths)]
    if len(image_paths) != args.num_frames or len(metadata_paths) != len(image_paths):
        raise ValueError(f"Found {len(image_paths)} images and {len(metadata_paths)} metadata files")
    source_ids = []
    for path in metadata_paths:
        with np.load(path) as metadata:
            source_ids.append(int(metadata["source_frame_id"]))

    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    args.output.mkdir(parents=True, exist_ok=True)
    load_start = time.perf_counter()
    model = MapAnything.from_pretrained(str(args.model), cache_dir=str(args.cache_dir)).to("cuda").eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start

    step = args.chunk_size - args.overlap
    starts = list(range(0, len(image_paths) - args.overlap, step))
    chunks = [(start, min(start + args.chunk_size, len(image_paths))) for start in starts]
    output = {name: [] for name in (
        "depth", "confidence", "intrinsics", "camera_to_world", "images", "mask", "metric_scaling_factor"
    )}
    transforms, chunk_seconds = [], []
    previous_global_overlap_poses = None
    torch.cuda.reset_peak_memory_stats()
    total_start = time.perf_counter()

    for chunk_index, (start, end) in enumerate(chunks):
        chunk_start = time.perf_counter()
        views = load_images([str(path) for path in image_paths[start:end]], verbose=False)
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
        current = to_numpy(predictions)
        del predictions, views
        torch.cuda.empty_cache()
        gc.collect()

        if chunk_index == 0:
            scale, rotation, translation, overlap_rmse = 1.0, np.eye(3), np.zeros(3), 0.0
        else:
            current_overlap = current["camera_to_world"][: args.overlap, :3, 3].astype(np.float64)
            scale, rotation, translation, overlap_rmse = umeyama(
                current_overlap, previous_global_overlap_poses[:, :3, 3]
            )
        current["camera_to_world"] = transform_poses(
            current["camera_to_world"].astype(np.float64), scale, rotation, translation
        ).astype(np.float32)
        current["depth"] = (current["depth"] * scale).astype(np.float32)
        previous_global_overlap_poses = current["camera_to_world"][-args.overlap:].astype(np.float64)

        save_count = len(current["depth"]) if chunk_index == len(chunks) - 1 else step
        for name in output:
            output[name].append(current[name][:save_count])
        elapsed = time.perf_counter() - chunk_start
        chunk_seconds.append(elapsed)
        transforms.append({
            "chunk_index": chunk_index,
            "frame_range_end_exclusive": [start, end],
            "saved_frame_range_end_exclusive": [start, start + save_count],
            "scale_to_global": scale,
            "rotation_to_global": rotation.tolist(),
            "translation_to_global": translation.tolist(),
            "overlap_camera_rmse": overlap_rmse,
            "seconds": elapsed,
        })
        print(json.dumps({"chunk": chunk_index + 1, "total": len(chunks), "frames": [start, end],
                          "scale": scale, "overlap_camera_rmse": overlap_rmse, "seconds": elapsed}))

    inference_seconds = time.perf_counter() - total_start
    arrays = {name: np.concatenate(parts, axis=0) for name, parts in output.items()}
    if any(len(value) != len(image_paths) for value in arrays.values()):
        raise RuntimeError({name: len(value) for name, value in arrays.items()})
    result_path = args.output / "prediction.npz"
    np.savez_compressed(
        result_path,
        **arrays,
        source_frame_ids=np.asarray(source_ids, dtype=np.int64),
        streaming_chunk_size=np.int64(args.chunk_size),
        streaming_overlap=np.int64(args.overlap),
    )
    summary = {
        "model": str(args.model.resolve()),
        "input_mode": "rgb_only",
        "upstream_long_sequence_runner": False,
        "stitching_gt_used": False,
        "stitching": "overlap predicted camera centres, sequential Sim(3)",
        "memory_efficient_inference": True,
        "minibatch_size": 1,
        "num_frames": len(image_paths),
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "num_chunks": len(chunks),
        "model_load_seconds": load_seconds,
        "inference_and_stitch_seconds": inference_seconds,
        "fps": len(image_paths) / inference_seconds,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "finite_depth_fraction": float(np.isfinite(arrays["depth"]).mean()),
        "metric_scaling_factor_median": float(np.median(arrays["metric_scaling_factor"])),
        "prediction": str(result_path.resolve()),
        "chunks": transforms,
    }
    (args.output / "run_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
