#!/usr/bin/env python3
"""Run the official AMB3R base model without opening an Open3D window."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument(
        "--frontend-only",
        action="store_true",
        help="Run only the AMB3R frontend, matching VO windows that pass the early-return confidence gate.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--max-points", type=int, default=1_000_000)
    return parser.parse_args()


def write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    vertex = np.empty(
        len(points),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")],
    )
    vertex["x"], vertex["y"], vertex["z"] = points.T
    color_u8 = np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8)
    vertex["r"], vertex["g"], vertex["b"] = color_u8.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertex)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        vertex.tofile(handle)


def point_cloud_from_result(result: dict, threshold: float, max_points: int, seed: int = 0):
    points = result["world_points"][0].detach().float().cpu().numpy().reshape(-1, 3)
    confidence_raw = result["world_points_conf"][0].detach().float().cpu().numpy().reshape(-1)
    confidence = (confidence_raw - 1.0) / confidence_raw
    colors = (
        result["images"][0]
        .detach()
        .float()
        .cpu()
        .permute(0, 2, 3, 1)
        .numpy()
        .reshape(-1, 3)
    )
    colors = (colors + 1.0) / 2.0
    mask = np.isfinite(points).all(axis=1) & np.isfinite(confidence) & (confidence >= threshold)
    points, colors = points[mask], colors[mask]
    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(points), size=max_points, replace=False)
        points, colors = points[keep], colors[keep]
    return points.astype(np.float32), colors.astype(np.float32), confidence, mask


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the official AMB3R implementation")
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.repo.resolve()))

    from amb3r.model import AMB3R
    from amb3r.datasets import Demo

    torch.manual_seed(0)
    np.random.seed(0)
    model = AMB3R()
    model.load_weights(str(args.checkpoint))
    model.cuda().eval()

    dataset = Demo(
        ROOT=str(args.data),
        resolution=(518, 392),
        num_seq=1,
        full_video=True,
        kf_every=1,
        disable_crop=False,
        max_images=args.num_frames,
    )
    _, views_all = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
    for key in views_all:
        views_all[key] = views_all[key].cuda()

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.no_grad():
        if args.frontend_only:
            images, patch_tokens = model.front_end.encode_patch_tokens(views_all)
            results = [model.front_end.decode_patch_tokens_and_heads(images, patch_tokens)]
        else:
            results = model(views_all)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    source_frame_ids = []
    for metadata_path in sorted(args.data.glob("[0-9][0-9][0-9][0-9].npz"))[: args.num_frames]:
        with np.load(metadata_path) as metadata:
            source_frame_ids.append(int(metadata["source_frame_id"]))

    stage_results = [("frontend", results[0])]
    if not args.frontend_only:
        stage_results.append(("backend", results[-1]))
    summary = {
        "mode": "amb3r_base",
        "num_frames": int(views_all["images"].shape[1]),
        "source_frame_ids": source_frame_ids,
        "runtime_seconds": elapsed,
        "fps": float(views_all["images"].shape[1] / elapsed),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "confidence_threshold": args.confidence_threshold,
        "stages": {},
    }
    for index, (stage, result) in enumerate(stage_results):
        points, colors, confidence, mask = point_cloud_from_result(
            result, args.confidence_threshold, args.max_points, seed=index
        )
        write_binary_ply(args.output / f"amb3r_{stage}.ply", points, colors)
        depth = result["depth"].detach().float()
        depth_metric = result["depth_metric"].detach().float()
        metric_scale = torch.median(depth_metric / depth).item()
        metric_result = dict(result)
        metric_result["world_points"] = result["world_points"] * metric_scale
        metric_pose = result["pose"].clone()
        metric_pose[..., :3, 3] *= metric_scale
        metric_result["pose"] = metric_pose
        metric_points, metric_colors, _, _ = point_cloud_from_result(
            metric_result, args.confidence_threshold, args.max_points, seed=index
        )
        write_binary_ply(args.output / f"amb3r_{stage}_metric.ply", metric_points, metric_colors)
        np.savez_compressed(
            args.output / f"amb3r_{stage}_results.npz",
            pts=result["world_points"][0].detach().float().cpu().numpy(),
            conf=((result["world_points_conf"][0].detach().float().cpu().numpy() - 1.0)
                  / result["world_points_conf"][0].detach().float().cpu().numpy()),
            pose=result["pose"][0].detach().float().cpu().numpy(),
            images=result["images"][0].detach().float().cpu().numpy(),
            source_frame_ids=np.asarray(source_frame_ids, dtype=np.int64),
        )
        np.savez_compressed(
            args.output / f"amb3r_{stage}_metric_results.npz",
            pts=metric_result["world_points"][0].detach().float().cpu().numpy(),
            conf=((result["world_points_conf"][0].detach().float().cpu().numpy() - 1.0)
                  / result["world_points_conf"][0].detach().float().cpu().numpy()),
            pose=metric_pose[0].detach().float().cpu().numpy(),
            images=result["images"][0].detach().float().cpu().numpy(),
            depth=depth[0].cpu().numpy(),
            depth_metric=depth_metric[0].cpu().numpy(),
            metric_scale=np.float32(metric_scale),
            source_frame_ids=np.asarray(source_frame_ids, dtype=np.int64),
        )
        summary["stages"][stage] = {
            "finite_point_fraction": float(np.isfinite(result["world_points"].detach().float().cpu().numpy()).all(axis=-1).mean()),
            "retained_point_count": int(len(points)),
            "confidence_min": float(np.nanmin(confidence)),
            "confidence_median": float(np.nanmedian(confidence)),
            "confidence_max": float(np.nanmax(confidence)),
            "retained_fraction": float(mask.mean()),
            "predicted_metric_scale": metric_scale,
        }

    (args.output / "run_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
