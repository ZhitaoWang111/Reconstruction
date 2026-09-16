#!/usr/bin/env python3
"""Fair common-FOV evaluation and contact sheet for single-frame metric depth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--amb3r", type=Path, required=True)
    parser.add_argument("--da3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_gt(metadata_path: Path) -> np.ndarray:
    with np.load(metadata_path) as metadata:
        maximum_depth = float(metadata["maximum_depth"])
    encoded = np.asarray(Image.open(metadata_path.with_name(metadata_path.stem + "_depth.png")), dtype=np.float32)
    return encoded / 65535.0 * maximum_depth


def amb3r_crop_bounds(width: int, height: int) -> tuple[int, int, int, int]:
    target_aspect = 518 / 392
    if width / height > target_aspect:
        crop_width = int(round(height * target_aspect))
        left = (width - crop_width) // 2
        return left, 0, left + crop_width, height
    crop_height = int(round(width / target_aspect))
    top = (height - crop_height) // 2
    return 0, top, width, top + crop_height


def resize(array: np.ndarray, size: tuple[int, int], nearest: bool = False) -> np.ndarray:
    method = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return np.asarray(Image.fromarray(array.astype(np.float32), mode="F").resize(size, method), dtype=np.float32)


def common_fov(array: np.ndarray, source_size: tuple[int, int], nearest: bool = False) -> np.ndarray:
    original_w, original_h = source_size
    restored = resize(array, (original_w, original_h), nearest=nearest)
    left, top, right, bottom = amb3r_crop_bounds(original_w, original_h)
    return resize(restored[top:bottom, left:right], (518, 392), nearest=nearest)


def metric_values(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict:
    p, g = pred[valid].astype(np.float64), gt[valid].astype(np.float64)
    ratio = np.maximum(p / g, g / p)
    return {
        "abs_rel": float(np.mean(np.abs(p - g) / g)),
        "rmse_m": float(np.sqrt(np.mean((p - g) ** 2))),
        "mae_m": float(np.mean(np.abs(p - g))),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25 ** 2)),
    }


def evaluate(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict:
    scale = float(np.median(gt[valid] / pred[valid]))
    return {
        "valid_pixel_count": int(valid.sum()),
        "median_scale_gt_over_pred": scale,
        "raw_metric": metric_values(pred, gt, valid),
        "median_scale_aligned": metric_values(pred * scale, gt, valid),
        "pred_depth_median_m": float(np.median(pred[valid])),
        "gt_depth_median_m": float(np.median(gt[valid])),
    }


def macro_average(per_frame: list[dict], branch: str) -> dict:
    keys = per_frame[0][branch].keys()
    return {key: float(np.mean([frame[branch][key] for frame in per_frame])) for key in keys}


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.amb3r) as archive:
        amb_depth = archive["depth"].astype(np.float32)
        indices = archive["sample_indices"].astype(int)
        source_ids = archive["source_frame_ids"].astype(int)
    with np.load(args.da3) as archive:
        da_depth_native = archive["depth"].astype(np.float32)
        da_indices = archive["sample_indices"].astype(int)
        da_source_ids = archive["source_frame_ids"].astype(int)
        da_metric_flags = archive["is_metric"].astype(int)
    if not (np.array_equal(indices, da_indices) and np.array_equal(source_ids, da_source_ids)):
        raise ValueError("AMB3R and DA3 frame selections differ")
    if not np.all(da_metric_flags == 1):
        raise ValueError(f"DA3 did not report metric depth on every frame: {da_metric_flags.tolist()}")

    gt_depths, rgbs, da_depths = [], [], []
    for index, da_depth in zip(indices, da_depth_native):
        rgb_image = Image.open(args.data / f"{index:04d}.jpg").convert("RGB")
        source_size = rgb_image.size
        bounds = amb3r_crop_bounds(*source_size)
        rgbs.append(np.asarray(rgb_image.crop(bounds).resize((518, 392), Image.Resampling.LANCZOS)))
        gt_depths.append(common_fov(load_gt(args.data / f"{index:04d}.npz"), source_size, nearest=True))
        da_depths.append(common_fov(da_depth, source_size, nearest=False))
    gt_depth = np.stack(gt_depths)
    da_depth = np.stack(da_depths)
    rgb = np.stack(rgbs)

    shared_valid = (
        np.isfinite(gt_depth) & (gt_depth > 0) & (gt_depth < 10.0)
        & np.isfinite(amb_depth) & (amb_depth > 0)
        & np.isfinite(da_depth) & (da_depth > 0)
    )
    amb_per_frame, da_per_frame, rows = [], [], []
    for i, (sample_index, source_id) in enumerate(zip(indices, source_ids)):
        amb = evaluate(amb_depth[i], gt_depth[i], shared_valid[i])
        da = evaluate(da_depth[i], gt_depth[i], shared_valid[i])
        amb_per_frame.append(amb)
        da_per_frame.append(da)
        rows.append({
            "sample_index": int(sample_index),
            "source_frame_id": int(source_id),
            "valid_pixel_count": int(shared_valid[i].sum()),
            "amb3r": amb,
            "da3": da,
            "raw_abs_rel_winner": "AMB3R" if amb["raw_metric"]["abs_rel"] < da["raw_metric"]["abs_rel"] else "DA3",
        })

    amb_global = evaluate(amb_depth, gt_depth, shared_valid)
    da_global = evaluate(da_depth, gt_depth, shared_valid)
    result = {
        "protocol": {
            "scene": "scene0000_00",
            "selection": "10 endpoint-inclusive uniformly spaced samples from uniform-500",
            "single_frame_definition": "each model received one RGB image per inference call; no temporal or multi-view context",
            "common_fov": "AMB3R center-crop field of view, 518x392; DA3 predictions reprojected from full-frame native output",
            "ground_truth": "ScanNet sensor depth; valid 0<depth<10m",
            "shared_mask": "same valid pixels for both methods",
            "primary": "raw metric scale; no GT scale alignment",
            "diagnostic": "per-frame median GT/pred scale alignment",
        },
        "sample_indices": indices.tolist(),
        "source_frame_ids": source_ids.tolist(),
        "amb3r": {
            "global_pixel_weighted": amb_global,
            "macro_raw_metric": macro_average(amb_per_frame, "raw_metric"),
            "macro_median_scale_aligned": macro_average(amb_per_frame, "median_scale_aligned"),
        },
        "da3": {
            "global_pixel_weighted": da_global,
            "macro_raw_metric": macro_average(da_per_frame, "raw_metric"),
            "macro_median_scale_aligned": macro_average(da_per_frame, "median_scale_aligned"),
        },
        "raw_abs_rel_win_count": {
            "AMB3R": sum(row["raw_abs_rel_winner"] == "AMB3R" for row in rows),
            "DA3": sum(row["raw_abs_rel_winner"] == "DA3" for row in rows),
        },
        "per_frame": rows,
    }
    output_json = args.output / "comparison.json"
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    valid_values = gt_depth[shared_valid]
    vmin, vmax = float(np.percentile(valid_values, 2)), float(np.percentile(valid_values, 98))
    fig, axes = plt.subplots(len(indices), 6, figsize=(18, 3.15 * len(indices)), constrained_layout=True)
    headers = ["RGB", "ScanNet GT", "AMB3R depth", "DA3 depth", "AMB3R |error|", "DA3 |error|"]
    depth_handle, error_handle = None, None
    for row_index in range(len(indices)):
        axes[row_index, 0].imshow(rgb[row_index])
        valid = shared_valid[row_index]
        gt_vis = np.where(valid, gt_depth[row_index], np.nan)
        depth_handle = axes[row_index, 1].imshow(gt_vis, cmap="turbo", vmin=vmin, vmax=vmax)
        axes[row_index, 2].imshow(amb_depth[row_index], cmap="turbo", vmin=vmin, vmax=vmax)
        axes[row_index, 3].imshow(da_depth[row_index], cmap="turbo", vmin=vmin, vmax=vmax)
        amb_error = np.where(valid, np.abs(amb_depth[row_index] - gt_depth[row_index]), np.nan)
        da_error = np.where(valid, np.abs(da_depth[row_index] - gt_depth[row_index]), np.nan)
        error_handle = axes[row_index, 4].imshow(amb_error, cmap="magma", vmin=0, vmax=1.0)
        axes[row_index, 5].imshow(da_error, cmap="magma", vmin=0, vmax=1.0)
        for column in range(6):
            axes[row_index, column].axis("off")
            if row_index == 0:
                axes[row_index, column].set_title(headers[column], fontsize=12)
        axes[row_index, 0].set_ylabel(f"u500 {indices[row_index]:04d}\nsrc {source_ids[row_index]}", fontsize=9)
        axes[row_index, 0].yaxis.set_visible(True)
        axes[row_index, 0].set_yticks([])
    fig.colorbar(depth_handle, ax=axes[:, 1:4], shrink=0.45, label="depth (m)")
    fig.colorbar(error_handle, ax=axes[:, 4:6], shrink=0.45, label="absolute error (m), clipped at 1m")
    figure_path = args.output / "depth_comparison_10_frames.png"
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    print(json.dumps({"comparison": str(output_json.resolve()), "visualization": str(figure_path.resolve()), "summary": result}, indent=2))


if __name__ == "__main__":
    main()
