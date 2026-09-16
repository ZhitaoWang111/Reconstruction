#!/usr/bin/env python3
"""Run AMB3R's metric-depth frontend on ten independent ScanNet images."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


DEFAULT_INDICES = [0, 55, 111, 166, 222, 277, 333, 388, 444, 499]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=DEFAULT_INDICES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by AMB3R")
    sys.path.insert(0, str(args.repo.resolve()))
    from amb3r.model import AMB3R

    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    load_start = time.perf_counter()
    model = AMB3R(metric_scale=True)
    model.load_weights(str(args.checkpoint))
    model.cuda().eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start

    depths, confidences, source_ids, elapsed_per_frame = [], [], [], []
    torch.cuda.reset_peak_memory_stats()
    for position, index in enumerate(args.indices, start=1):
        image_path = args.data / f"{index:04d}.jpg"
        metadata_path = args.data / f"{index:04d}.npz"
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        target_aspect = 518 / 392
        if width / height > target_aspect:
            crop_width = int(round(height * target_aspect))
            left = (width - crop_width) // 2
            image = image.crop((left, 0, left + crop_width, height))
        elif width / height < target_aspect:
            crop_height = int(round(width / target_aspect))
            top = (height - crop_height) // 2
            image = image.crop((0, top, width, top + crop_height))
        image = image.resize((518, 392), Image.Resampling.LANCZOS)
        tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()
        tensor = (tensor / 255.0 * 2.0 - 1.0)[None, None].cuda()
        with np.load(metadata_path) as metadata:
            source_id = int(metadata["source_frame_id"])

        start = time.perf_counter()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = model({"images": tensor}, iters=0)[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        depth = result["depth_metric"][0, 0, ..., 0].float().cpu().numpy()
        confidence = result["depth_conf"][0, 0, ..., 0].float().cpu().numpy()
        if depth.shape != (392, 518) or not np.isfinite(depth).all():
            raise RuntimeError(f"Invalid AMB3R output for frame {index}: {depth.shape}")
        depths.append(depth.astype(np.float32))
        confidences.append(confidence.astype(np.float32))
        source_ids.append(source_id)
        elapsed_per_frame.append(elapsed)
        print(f"AMB3R single frame {position}/{len(args.indices)}: index={index}, source={source_id}, {elapsed:.3f}s")
        del tensor, result

    prediction_path = args.output / "prediction.npz"
    np.savez_compressed(
        prediction_path,
        depth=np.stack(depths),
        confidence=np.stack(confidences),
        sample_indices=np.asarray(args.indices, dtype=np.int64),
        source_frame_ids=np.asarray(source_ids, dtype=np.int64),
    )
    summary = {
        "method": "AMB3R metric-depth frontend",
        "input_mode": "ten independent one-image calls; frontend only; iters=0",
        "sample_indices": args.indices,
        "source_frame_ids": source_ids,
        "processed_shape": list(np.stack(depths).shape),
        "model_load_seconds": load_seconds,
        "inference_seconds_total": float(sum(elapsed_per_frame)),
        "inference_seconds_per_frame": elapsed_per_frame,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "finite_depth_fraction": float(np.isfinite(depths).mean()),
        "prediction": str(prediction_path.resolve()),
    }
    (args.output / "run_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
