#!/usr/bin/env python3
"""Generate real scene0000_00 visual evidence for the local ScanNet guide."""

from __future__ import annotations

import io
import json
import struct
import zlib
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from PIL import Image
from plyfile import PlyData


SCENE_ID = "scene0000_00"
FRAME_ID = 162
SOURCE = Path("/home/wzt/HDD/datasets/scannet_v2/scans") / SCENE_ID
OUTPUT = Path(__file__).resolve().parent / "assets"


def read_matrix(handle) -> tuple[float, ...]:
    return struct.unpack("<16f", handle.read(64))


def read_axis_alignment() -> np.ndarray:
    metadata = SOURCE / f"{SCENE_ID}.txt"
    for line in metadata.read_text().splitlines():
        if line.startswith("axisAlignment ="):
            values = np.fromstring(line.split("=", 1)[1], sep=" ")
            return values.reshape(4, 4)
    raise ValueError("axisAlignment is missing from scene metadata")


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (homogeneous @ matrix.T)[:, :3]


def extract_rgbd() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sens_path = SOURCE / f"{SCENE_ID}.sens"
    with sens_path.open("rb") as handle:
        version = struct.unpack("<I", handle.read(4))[0]
        name_length = struct.unpack("<Q", handle.read(8))[0]
        handle.seek(name_length, 1)
        color_intrinsic = np.asarray(read_matrix(handle)).reshape(4, 4)
        color_extrinsic = np.asarray(read_matrix(handle)).reshape(4, 4)
        depth_intrinsic = np.asarray(read_matrix(handle)).reshape(4, 4)
        depth_extrinsic = np.asarray(read_matrix(handle)).reshape(4, 4)
        color_compression = struct.unpack("<i", handle.read(4))[0]
        depth_compression = struct.unpack("<i", handle.read(4))[0]
        color_width, color_height, depth_width, depth_height = struct.unpack(
            "<IIII", handle.read(16)
        )
        depth_shift = struct.unpack("<f", handle.read(4))[0]
        frame_count = struct.unpack("<Q", handle.read(8))[0]

        if version != 4 or color_compression != 2 or depth_compression != 1:
            raise ValueError("Unexpected ScanNet .sens encoding")
        if FRAME_ID >= frame_count:
            raise IndexError(f"Frame {FRAME_ID} is outside {frame_count} frames")

        color_data = depth_data = None
        trajectory = []
        for index in range(frame_count):
            pose = np.asarray(read_matrix(handle)).reshape(4, 4)
            handle.seek(8 + 8, 1)
            color_bytes, depth_bytes = struct.unpack("<QQ", handle.read(16))
            if index % 10 == 0 and np.isfinite(pose).all():
                trajectory.append(pose[:3, 3])
            if index == FRAME_ID:
                color_data = handle.read(color_bytes)
                depth_data = handle.read(depth_bytes)
            else:
                handle.seek(color_bytes + depth_bytes, 1)

        imu_count = struct.unpack("<Q", handle.read(8))[0]
        imu_values = np.empty((imu_count, 15), dtype=np.float64)
        imu_timestamps = np.empty(imu_count, dtype=np.uint64)
        for index in range(imu_count):
            record = struct.unpack("<15dQ", handle.read(128))
            imu_values[index] = record[:15]
            imu_timestamps[index] = record[15]

    if color_data is None or depth_data is None:
        raise RuntimeError("Requested RGB-D frame was not found")

    color = Image.open(io.BytesIO(color_data)).convert("RGB")
    color.thumbnail((1100, 820), Image.Resampling.LANCZOS)
    color.save(OUTPUT / "frame-0162-rgb.webp", "WEBP", quality=88)

    depth = np.frombuffer(zlib.decompress(depth_data), dtype=np.uint16).reshape(
        depth_height, depth_width
    )
    depth_m = depth.astype(np.float32) / depth_shift
    valid = depth_m > 0
    ceiling = float(np.percentile(depth_m[valid], 98))
    normalized = np.clip(depth_m / ceiling, 0, 1)
    rgba = plt.get_cmap("turbo_r")(normalized)
    rgba[~valid, :3] = (0.025, 0.045, 0.075)
    depth_rgb = Image.fromarray((rgba[:, :, :3] * 255).astype(np.uint8))
    depth_rgb = depth_rgb.resize((color_width, color_height), Image.Resampling.NEAREST)
    depth_rgb.thumbnail((1100, 820), Image.Resampling.LANCZOS)
    depth_rgb.save(OUTPUT / "frame-0162-depth.webp", "WEBP", quality=90)
    return np.asarray(trajectory), imu_values, imu_timestamps


def palette_image(values: np.ndarray, semantic: bool) -> Image.Image:
    colors = np.array(
        [
            [12, 24, 42],
            [229, 95, 64],
            [38, 166, 154],
            [244, 184, 72],
            [98, 126, 234],
            [184, 86, 147],
            [71, 142, 91],
            [236, 121, 55],
            [68, 180, 200],
            [161, 116, 214],
            [199, 72, 72],
            [113, 156, 64],
        ],
        dtype=np.uint8,
    )
    result = np.zeros((*values.shape, 3), dtype=np.uint8)
    result[:] = colors[0]
    for rank, value in enumerate(np.unique(values)):
        if value == 0:
            continue
        color_index = int(value) % (len(colors) - 1) + 1 if semantic else rank % (len(colors) - 1) + 1
        result[values == value] = colors[color_index]
    return Image.fromarray(result)


def extract_masks() -> None:
    for kind, output_name, semantic in (
        ("label", "frame-0162-semantic-raw.webp", True),
        ("label-filt", "frame-0162-semantic-filt.webp", True),
        ("instance", "frame-0162-instance-raw.webp", False),
        ("instance-filt", "frame-0162-instance-filt.webp", False),
    ):
        archive = SOURCE / f"{SCENE_ID}_2d-{kind}.zip"
        member = f"{kind}/{FRAME_ID}.png"
        with ZipFile(archive) as zipped:
            values = np.asarray(Image.open(io.BytesIO(zipped.read(member))))
        image = palette_image(values, semantic)
        image.thumbnail((1100, 820), Image.Resampling.NEAREST)
        image.save(OUTPUT / output_name, "WEBP", quality=92, method=6)


def category_colors(values: np.ndarray) -> np.ndarray:
    """Return stable high-contrast RGB colors for integer IDs."""
    values = values.astype(np.int64)
    hue = np.mod(values * 0.61803398875, 1.0)
    rgba = plt.get_cmap("hsv")(hue)
    rgb = rgba[:, :3] * 0.72 + 0.28
    rgb[values < 0] = (0.15, 0.18, 0.23)
    return rgb


def render_points(
    xyz: np.ndarray,
    rgb: np.ndarray,
    output_name: str,
    *,
    trajectory: np.ndarray | None = None,
    point_size: float = 0.55,
) -> None:
    figure = plt.figure(figsize=(12, 8), dpi=150, facecolor="#07101d")
    axis = figure.add_subplot(111, projection="3d", facecolor="#07101d")
    axis.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=point_size, linewidths=0)
    bounds = xyz
    if trajectory is not None and len(trajectory):
        axis.plot(
            trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
            color="#67dacb", linewidth=1.6, alpha=0.92,
        )
        axis.scatter(
            trajectory[::14, 0], trajectory[::14, 1], trajectory[::14, 2],
            color="#b8fff5", s=5, linewidths=0, alpha=0.8,
        )
        axis.scatter(*trajectory[0], color="#63e6be", s=48, edgecolors="white", linewidths=0.6)
        axis.scatter(*trajectory[-1], color="#ff785f", s=48, edgecolors="white", linewidths=0.6)
        bounds = np.vstack((xyz, trajectory))
    axis.view_init(elev=31, azim=-62)
    spans = np.ptp(bounds, axis=0)
    centers = (bounds.max(axis=0) + bounds.min(axis=0)) / 2
    radius = spans.max() / 2
    axis.set_xlim(centers[0] - radius, centers[0] + radius)
    axis.set_ylim(centers[1] - radius, centers[1] + radius)
    axis.set_zlim(centers[2] - radius * 0.72, centers[2] + radius * 0.72)
    axis.set_axis_off()
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    figure.savefig(
        OUTPUT / output_name,
        format="webp",
        dpi=150,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(figure)


def read_mesh(mesh_name: str) -> tuple[np.ndarray, np.ndarray]:
    mesh = o3d.io.read_triangle_mesh(str(SOURCE / mesh_name))
    vertices = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)
    if not len(vertices):
        raise ValueError(f"{mesh_name} has no vertices")
    if len(colors) != len(vertices):
        colors = np.full_like(vertices, 0.72)
    return vertices, colors


def sample_indices(length: int, maximum: int, seed: int) -> np.ndarray:
    if length <= maximum:
        return np.arange(length)
    return np.random.default_rng(seed).choice(length, maximum, replace=False)


def read_segments(filename: str) -> np.ndarray:
    payload = json.loads((SOURCE / filename).read_text())
    return np.asarray(payload["segIndices"], dtype=np.int64)


def instance_ids(segments: np.ndarray, aggregation_name: str) -> np.ndarray:
    groups = json.loads((SOURCE / aggregation_name).read_text())["segGroups"]
    mapping = {segment: group["objectId"] for group in groups for segment in group["segments"]}
    return np.asarray([mapping.get(int(segment), -1) for segment in segments])


def render_scene_views(trajectory: np.ndarray) -> None:
    alignment = read_axis_alignment()
    dense_xyz, dense_rgb = read_mesh(f"{SCENE_ID}_vh_clean.ply")
    simple_xyz, simple_rgb = read_mesh(f"{SCENE_ID}_vh_clean_2.ply")
    dense_segments = read_segments(f"{SCENE_ID}_vh_clean.segs.json")
    simple_segments = read_segments(f"{SCENE_ID}_vh_clean_2.0.010000.segs.json")

    if len(dense_segments) != len(dense_xyz) or len(simple_segments) != len(simple_xyz):
        raise ValueError("Segment arrays do not match their mesh vertex counts")

    dense_keep = sample_indices(len(dense_xyz), 65_000, 11)
    simple_keep = sample_indices(len(simple_xyz), 55_000, 7)
    dense_sample = transform_points(dense_xyz[dense_keep], alignment)
    simple_sample = transform_points(simple_xyz[simple_keep], alignment)

    render_points(dense_sample, dense_rgb[dense_keep], "scene0000_00-dense-rgb.webp")
    render_points(
        dense_sample,
        category_colors(dense_segments[dense_keep]),
        "scene0000_00-dense-segments.webp",
    )
    render_points(simple_sample, simple_rgb[simple_keep], "scene0000_00-simple-rgb.webp")
    render_points(
        simple_sample,
        category_colors(simple_segments[simple_keep]),
        "scene0000_00-simple-segments.webp",
    )

    labeled = PlyData.read(SOURCE / f"{SCENE_ID}_vh_clean_2.labels.ply")
    semantic_labels = np.asarray(labeled["vertex"]["label"], dtype=np.int64)
    render_points(
        simple_sample,
        category_colors(semantic_labels[simple_keep]),
        "scene0000_00-semantic-3d.webp",
    )

    objects = instance_ids(simple_segments, f"{SCENE_ID}.aggregation.json")
    render_points(
        simple_sample,
        category_colors(objects[simple_keep]),
        "scene0000_00-instance-3d.webp",
    )

    aligned_trajectory = transform_points(trajectory, alignment)
    trajectory_keep = sample_indices(len(simple_xyz), 25_000, 17)
    trajectory_mesh = transform_points(simple_xyz[trajectory_keep], alignment)
    render_points(
        trajectory_mesh,
        np.full((len(trajectory_mesh), 3), (0.31, 0.36, 0.44)),
        "scene0000_00-trajectory.webp",
        trajectory=aligned_trajectory,
        point_size=0.42,
    )

    # Keep the original hero asset name while making it use the aligned view.
    render_points(simple_sample, simple_rgb[simple_keep], "scene0000_00-mesh.webp")


def render_imu(values: np.ndarray, timestamps: np.ndarray) -> None:
    seconds = (timestamps.astype(np.float64) - float(timestamps[0])) / 1_000_000
    rotation_rate = np.linalg.norm(values[:, 0:3], axis=1)
    acceleration = np.linalg.norm(values[:, 3:6], axis=1)

    figure, axes = plt.subplots(2, 1, figsize=(10, 3.4), dpi=150, sharex=True)
    figure.patch.set_facecolor("#0b1523")
    for axis in axes:
        axis.set_facecolor("#0b1523")
        axis.grid(axis="y", color="white", alpha=0.07, linewidth=0.6)
        axis.tick_params(colors="#77869a", labelsize=7, length=0)
        axis.spines[:].set_visible(False)
    axes[0].plot(seconds, rotation_rate, color="#67dacb", linewidth=0.65)
    axes[1].plot(seconds, acceleration, color="#ff785f", linewidth=0.65)
    axes[0].set_ylabel("rotation", color="#93a1b4", fontsize=7)
    axes[1].set_ylabel("accel.", color="#93a1b4", fontsize=7)
    axes[1].set_xlabel("time / s", color="#93a1b4", fontsize=7)
    figure.subplots_adjust(left=0.08, right=0.985, bottom=0.18, top=0.96, hspace=0.18)
    figure.savefig(
        OUTPUT / "scene0000_00-imu.webp",
        format="webp",
        dpi=150,
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    trajectory, imu_values, imu_timestamps = extract_rgbd()
    extract_masks()
    render_scene_views(trajectory)
    render_imu(imu_values, imu_timestamps)
    print(f"Prepared example assets in {OUTPUT}")


if __name__ == "__main__":
    main()
