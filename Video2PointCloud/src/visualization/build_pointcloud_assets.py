#!/usr/bin/env python3
"""Pack the evaluated and GT PLY clouds into file://-loadable JavaScript assets."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import open3d as o3d


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLOUDS = {
    "amb3r_gt": PROJECT_ROOT / "output/amb3r/vo_uniform500/pointcloud_gt_corrected_rgb.ply",
    "amb3r": PROJECT_ROOT / "output/amb3r/vo_uniform500/pointcloud_se3_scan_world_rgb.ply",
    "da3": PROJECT_ROOT / "output/da3/scene0000_00_uniform500_streaming_r2/pointcloud_se3_scan_world.ply",
}
STATIC_FILES = ("index.html", "styles.css", "viewer.js")


def encode(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output/visualization/assets",
        help="Generated browser asset directory (default: output/visualization/assets)",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = {"format": "quantized-u16-position-plus-u8-rgb", "clouds": {}}
    for name, path in CLOUDS.items():
        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points, dtype=np.float32)
        colors = np.asarray(cloud.colors)
        valid = np.isfinite(points).all(axis=1)
        points = points[valid]
        if colors.shape == np.asarray(cloud.points).shape:
            colors = colors[valid]
        else:
            colors = np.full(points.shape, 0.7, dtype=np.float32)
        colors = np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8)

        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        extent = maximum - minimum
        safe_extent = np.where(extent > 0, extent, 1.0)
        quantized = np.clip(
            np.rint((points - minimum) / safe_extent * 65535.0), 0, 65535
        ).astype("<u2")
        source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        payload = {
            "count": int(len(points)),
            "minimum": minimum.tolist(),
            "maximum": maximum.tolist(),
            "source": str(path),
            "source_sha256": source_sha,
            "positions_u16_base64": encode(quantized),
            "colors_u8_base64": encode(colors),
        }
        javascript = (
            "window.POINT_CLOUD_PAYLOADS = window.POINT_CLOUD_PAYLOADS || {};\n"
            f"window.POINT_CLOUD_PAYLOADS[{json.dumps(name)}] = "
            f"{json.dumps(payload, separators=(',', ':'))};\n"
        )
        output = args.output / f"{name}.cloud.js"
        output.write_text(javascript, encoding="utf-8")
        manifest["clouds"][name] = {
            "count": int(len(points)),
            "minimum": minimum.tolist(),
            "maximum": maximum.tolist(),
            "source": str(path),
            "source_sha256": source_sha,
            "asset": output.name,
            "asset_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }
        print(f"{name}: {len(points):,} points -> {output}")

    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    site_dir = args.output.parent
    for filename in STATIC_FILES:
        shutil.copy2(Path(__file__).with_name(filename), site_dir / filename)
    print(f"viewer: {site_dir / 'index.html'}")


if __name__ == "__main__":
    main()
