#!/usr/bin/env python3
"""Render deterministic PCA-aligned point-cloud previews without a GUI."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=120_000)
    parser.add_argument("--title", default="AMB3R point cloud")
    return parser.parse_args()


def equal_limits(values: np.ndarray, first: int, second: int):
    extent = max(np.ptp(values[:, first]), np.ptp(values[:, second])) * 0.52
    center = np.median(values[:, [first, second]], axis=0)
    return (center[0] - extent, center[0] + extent), (center[1] - extent, center[1] + extent)


def main() -> None:
    args = parse_args()
    cloud = o3d.io.read_point_cloud(str(args.ply))
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    valid = np.isfinite(points).all(axis=1)
    points, colors = points[valid], colors[valid]
    if not len(points):
        raise ValueError("Point cloud has no finite points")
    if len(points) > args.max_points:
        keep = np.random.default_rng(0).choice(len(points), args.max_points, replace=False)
        points, colors = points[keep], colors[keep]
    if colors.shape != points.shape:
        colors = np.full_like(points, 0.35)

    centered = points - np.median(points, axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ axes.T

    fig, plots = plt.subplots(1, 2, figsize=(14, 6), dpi=160, facecolor="white")
    for plot, (first, second), label in zip(
        plots, ((0, 1), (0, 2)), ("principal view", "side view")
    ):
        plot.scatter(projected[:, first], projected[:, second], c=np.clip(colors, 0, 1), s=0.15, linewidths=0)
        xlim, ylim = equal_limits(projected, first, second)
        plot.set_xlim(*xlim)
        plot.set_ylim(*ylim)
        plot.set_aspect("equal")
        plot.axis("off")
        plot.set_title(label, fontsize=11)
    fig.suptitle(args.title, fontsize=14)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"wrote {args.output} from {len(points)} sampled points")


if __name__ == "__main__":
    main()
