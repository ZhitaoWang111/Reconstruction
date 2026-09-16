#!/usr/bin/env python3
"""Run DA3 Nested on ten independent ScanNet images (never as a sequence)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


DEFAULT_INDICES = [0, 55, 111, 166, 222, 277, 333, 388, 444, 499]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=DEFAULT_INDICES)
    parser.add_argument("--process-res", type=int, default=504)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by DA3 Nested Giant-Large")
    from depth_anything_3.api import DepthAnything3

    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    load_start = time.perf_counter()
    model = DepthAnything3.from_pretrained(str(args.model)).to("cuda").eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start

    depths, confidences, skies, source_ids, scale_factors, metric_flags = [], [], [], [], [], []
    elapsed_per_frame = []
    torch.cuda.reset_peak_memory_stats()
    for position, index in enumerate(args.indices, start=1):
        image_path = args.images / f"{index:04d}.jpg"
        with np.load(args.metadata / f"{index:04d}.npz") as metadata:
            source_id = int(metadata["source_frame_id"])
        start = time.perf_counter()
        prediction = model.inference(
            [str(image_path)],
            process_res=args.process_res,
            process_res_method="upper_bound_resize",
            ref_view_strategy="middle",
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        depth = np.asarray(prediction.depth[0], dtype=np.float32)
        confidence = (
            np.ones_like(depth) if prediction.conf is None
            else np.asarray(prediction.conf[0], dtype=np.float32)
        )
        sky = (
            np.zeros_like(depth, dtype=bool) if prediction.sky is None
            else np.asarray(prediction.sky[0], dtype=bool)
        )
        if not np.isfinite(depth).all():
            raise RuntimeError(f"Non-finite DA3 output for frame {index}")
        depths.append(depth)
        confidences.append(confidence)
        skies.append(sky)
        source_ids.append(source_id)
        scale_factors.append(float(prediction.scale_factor))
        metric_flags.append(int(prediction.is_metric))
        elapsed_per_frame.append(elapsed)
        print(f"DA3 single frame {position}/{len(args.indices)}: index={index}, source={source_id}, {elapsed:.3f}s")
        del prediction

    prediction_path = args.output / "prediction.npz"
    depth_stack = np.stack(depths)
    np.savez_compressed(
        prediction_path,
        depth=depth_stack,
        confidence=np.stack(confidences),
        sky=np.stack(skies),
        sample_indices=np.asarray(args.indices, dtype=np.int64),
        source_frame_ids=np.asarray(source_ids, dtype=np.int64),
        scale_factors=np.asarray(scale_factors, dtype=np.float32),
        is_metric=np.asarray(metric_flags, dtype=np.int64),
    )
    summary = {
        "method": "Depth Anything 3 Nested Giant-Large 1.1",
        "input_mode": "ten independent one-image inference calls",
        "model": str(args.model.resolve()),
        "sample_indices": args.indices,
        "source_frame_ids": source_ids,
        "processed_shape": list(depth_stack.shape),
        "reported_is_metric": metric_flags,
        "predicted_scale_factors": scale_factors,
        "model_load_seconds": load_seconds,
        "inference_seconds_total": float(sum(elapsed_per_frame)),
        "inference_seconds_per_frame": elapsed_per_frame,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "finite_depth_fraction": float(np.isfinite(depth_stack).mean()),
        "prediction": str(prediction_path.resolve()),
    }
    (args.output / "run_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
