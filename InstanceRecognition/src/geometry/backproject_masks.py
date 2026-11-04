#!/usr/bin/env python3
"""Shared helpers for mask lifting and its published artifacts.

The point-cloud lifting pipeline that used to live here has been replaced by the
AMB3R per-frame lift (``amb3r_mask_source``). What remains are the helpers that
the lift, the assembler and the page builders all share: PLY read/write, mask
loading and resolution alignment, JSON/asset writers, hashing, relative paths and
instance colours.
"""

from __future__ import annotations

import base64
import colorsys
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def crop_box(size: tuple[int, int], target: tuple[int, int]) -> tuple[int, int, int, int]:
    """Match the official AMB3R-VO DemoDataset center crop, including rounding."""
    width, height = size
    target_width, target_height = target
    if min(width, height, target_width, target_height) <= 0:
        raise ValueError("Image dimensions must be positive")
    aspect = target_width / target_height
    if width / height > aspect:
        cropped_width = int(round(height * aspect))
        left = (width - cropped_width) // 2
        return left, 0, left + cropped_width, height
    if width / height < aspect:
        cropped_height = int(round(width / aspect))
        top = (height - cropped_height) // 2
        return 0, top, width, top + cropped_height
    return 0, 0, width, height


def align_mask(mask: np.ndarray, intermediate_size: tuple[int, int],
               target_size: tuple[int, int]) -> np.ndarray:
    """Original mask -> extracted RGB grid -> VO crop -> point-map grid.

    Nearest-neighbor sampling preserves binary membership. This is resolution
    alignment, not morphological cleanup. Overlapping masks remain independent.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D binary mask, got {mask.shape}")
    if not np.isin(mask, [0, 1, 255]).all():
        raise ValueError("Mask contains non-binary values")
    image = Image.fromarray((mask > 0).astype(np.uint8))
    image = image.resize(intermediate_size, Image.Resampling.NEAREST)
    image = image.crop(crop_box(intermediate_size, target_size))
    return np.asarray(image.resize(target_size, Image.Resampling.NEAREST)) > 0


def instance_masks(frame: dict, root: Path) -> tuple[list[np.ndarray], dict[str, str]]:
    """Boolean masks in the frame's own image grid, one per instance.

    Reads the exported mask files rather than any display artifact (the
    `instance_id_map.png` the perception stage also writes), so what geometry
    lifts is exactly what a mask asserted.
    """
    masks, hashes = [], {}
    for item in frame["instances"]:
        path = (root / item["mask_file"]).resolve()
        with Image.open(path) as image:
            masks.append(np.asarray(image.convert("L")) > 0)
        hashes[str(path)] = sha256_file(path)
    return masks, hashes


def membership_bits(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Bit i means membership in masks[i], NOT a globally shared instance ID."""
    if len(masks) > 32:
        raise ValueError("This lossless uint32 format supports at most 32 masks per frame")
    bits = np.zeros(shape, dtype=np.uint32)
    for index, mask in enumerate(masks):
        if mask.shape != shape:
            raise ValueError(f"Mask shape mismatch: {mask.shape} != {shape}")
        bits[mask] |= np.uint32(1 << index)
    return bits


def apply_sim3(points: np.ndarray, transform: dict) -> np.ndarray:
    rotation = np.asarray(transform["alignment_rotation"], dtype=np.float64)
    translation = np.asarray(transform["alignment_translation_m"], dtype=np.float64)
    scale = float(transform["alignment_scale"])
    if rotation.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid Sim(3)")
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("Nonfinite Sim(3)")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < 0:
        raise ValueError("Sim(3) rotation is not a proper rotation")
    return (scale * (points.astype(np.float64) @ rotation.T) + translation).astype(np.float32)


def instance_color(frame_id: int, instance_id: int) -> list[int]:
    hue = ((frame_id * 37 + instance_id * 137) % 360) / 360
    return [round(channel * 255) for channel in colorsys.hsv_to_rgb(hue, 0.68, 0.96)]


def display_colors(bits: np.ndarray, instances: list[dict]) -> np.ndarray:
    """No winner for overlaps: multi-membership is displayed in magenta."""
    colors = np.full((bits.size, 3), 75, dtype=np.uint8)
    for index, instance in enumerate(instances):
        colors[bits == np.uint32(1 << index)] = instance["color"]
    overlap = (bits & (bits - np.uint32(1))) != 0
    colors[overlap] = [235, 65, 210]
    return colors


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray,
              bits: np.ndarray, source_frame_ids: np.ndarray,
              pixel_indices: np.ndarray) -> None:
    """Binary PLY with recoverable multi-label and source-pixel provenance."""
    count = len(points)
    if not (colors.shape == (count, 3) and len(bits) == len(source_frame_ids) == len(pixel_indices) == count):
        raise ValueError("PLY attribute lengths differ")
    vertex = np.empty(count, dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("mask_bits", "<u4"), ("source_frame_id", "<u4"), ("pixel_index", "<u4"),
    ])
    for axis, name in enumerate(("x", "y", "z")):
        vertex[name] = points[:, axis]
    for axis, name in enumerate(("red", "green", "blue")):
        vertex[name] = colors[:, axis]
    vertex["mask_bits"], vertex["source_frame_id"], vertex["pixel_index"] = bits, source_frame_ids, pixel_indices
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment raw per-frame mask observations; mask_bits refers to frame-local instance list\n"
        f"element vertex {count}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uint mask_bits\nproperty uint source_frame_id\nproperty uint pixel_index\nend_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        vertex.tofile(handle)


def read_point_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read our existing binary vertex-only PLY without starting native thread pools."""
    types = {"float": "<f4", "double": "<f8", "uchar": "u1", "uint": "<u4"}
    fields = []
    count = None
    with path.open("rb") as handle:
        if handle.readline() != b"ply\n" or handle.readline() != b"format binary_little_endian 1.0\n":
            raise ValueError("Expected a binary little-endian PLY")
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("Truncated PLY header")
            tokens = line.decode("ascii").split()
            if tokens[:2] == ["element", "vertex"]:
                count = int(tokens[2])
            elif tokens[:1] == ["element"]:
                raise ValueError("Expected a vertex-only point cloud")
            elif tokens[:1] == ["property"]:
                if len(tokens) != 3 or tokens[1] not in types:
                    raise ValueError(f"Unsupported PLY property: {tokens}")
                fields.append((tokens[2], types[tokens[1]]))
            elif tokens == ["end_header"]:
                break
        if count is None:
            raise ValueError("Missing PLY vertex count")
        vertices = np.fromfile(handle, dtype=fields, count=count)
        if len(vertices) != count:
            raise ValueError("Truncated PLY vertices")
    points = np.column_stack([vertices[name] for name in ("x", "y", "z")]).astype(np.float32)
    colors = np.column_stack([vertices[name] for name in ("red", "green", "blue")]).astype(np.uint8)
    return points, colors


def encode(array: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode("ascii")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_cloud_asset(path: Path, key: str | int, payload: dict) -> None:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    path.write_text(f"window.registerMaskCloud({json.dumps(key)},{data});\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path, directory: Path) -> str:
    return Path(os.path.relpath(path.resolve(), directory.resolve())).as_posix()
