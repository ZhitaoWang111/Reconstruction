#!/usr/bin/env python3
"""Extract a bounded RGB-D sequence from a ScanNet v2 .sens recording.

The output directory layout is consumed by ``amb3r_mask_source.py`` and the
AMB3R runners.  The paired depth, pose, and intrinsics metadata are retained
next to each JPEG so the exact same preprocessing can be used for metric-scale
evaluation.

Run from InstanceRecognition::

    python -m src.geometry.extract_scannet_sens \\
      --sens /path/to/scene0000_00.sens \\
      --output artifacts/sources/scene0000_00/frames_uniform500 \\
      --target-frames 500

Three frame-selection modes:

* ``--target-frames N``  — uniform endpoints-inclusive sample (recommended).
* ``--start/--count/--stride``  — fixed stride (legacy).
* ``--frame-indices-json``  — explicit list from a JSON file.
"""

from __future__ import annotations

import argparse
import io
import json
import struct
import zlib
from pathlib import Path

import numpy as np
from PIL import Image


COLOR_COMPRESSION = {0: "raw", 1: "png", 2: "jpeg"}
DEPTH_COMPRESSION = {0: "raw_ushort", 1: "zlib_ushort", 2: "occi_ushort"}


def read(fmt: str, handle):
    size = struct.calcsize(fmt)
    payload = handle.read(size)
    if len(payload) != size:
        raise EOFError(f"Expected {size} bytes, got {len(payload)}")
    return struct.unpack(fmt, payload)


def read_matrix(handle) -> np.ndarray:
    return np.asarray(read("<16f", handle), dtype=np.float32).reshape(4, 4)


def uniform_sample_indices(frame_count: int, target_frames: int) -> list[int]:
    """Deterministic endpoints-inclusive uniform sampler.

    Selects ``target_frames`` indices from ``[0, frame_count)`` such that both
    endpoints are always included and the spacing is as even as possible.

    >>> uniform_sample_indices(10, 3)
    [0, 4, 9]
    >>> uniform_sample_indices(5, 5)
    [0, 1, 2, 3, 4]
    >>> uniform_sample_indices(5, 1)
    [0]
    """
    if frame_count <= 0 or target_frames <= 0:
        raise ValueError("frame_count and target_frames must be positive")
    if frame_count <= target_frames:
        return list(range(frame_count))
    if target_frames == 1:
        return [0]
    last = frame_count - 1
    return [round(index * last / (target_frames - 1)) for index in range(target_frames)]


def load_frame_indices(path: Path) -> list[int]:
    """Load an explicit frame-index list from a JSON file.

    The file must have a ``frame_indices`` key containing a sorted, unique,
    non-empty list of non-negative integers.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    indices = payload.get("frame_indices")
    if not isinstance(indices, list) or not indices or not all(isinstance(i, int) for i in indices):
        raise ValueError(f"Expected a non-empty integer frame_indices list in {path}")
    if indices != sorted(set(indices)):
        raise ValueError(f"frame_indices must be sorted and unique in {path}")
    return indices


def extract(sens_path: Path, output: Path, source_indices: list[int], selection: dict) -> Path:
    """Extract frames from a .sens file and write the output directory.

    Returns the path to the written ``manifest.json``.
    """
    input_dir = output / "input"
    rgb_dir = output / "rgb"
    input_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    records = []

    selected = {src: seq for seq, src in enumerate(source_indices)}

    with sens_path.open("rb") as handle:
        version = read("<I", handle)[0]
        if version != 4:
            raise ValueError(f"Unsupported .sens version {version}; expected 4")
        sensor_name_length = read("<Q", handle)[0]
        sensor_name = handle.read(sensor_name_length).decode("utf-8", errors="replace")
        intrinsic_color = read_matrix(handle)
        extrinsic_color = read_matrix(handle)
        intrinsic_depth = read_matrix(handle)
        extrinsic_depth = read_matrix(handle)
        color_compression_id, depth_compression_id = read("<ii", handle)
        color_width, color_height, depth_width, depth_height = read("<IIII", handle)
        depth_shift = read("<f", handle)[0]
        num_frames = read("<Q", handle)[0]

        if color_compression_id != 2 or depth_compression_id != 1:
            raise ValueError(
                "This extractor expects JPEG color and zlib uint16 depth, got "
                f"{COLOR_COMPRESSION.get(color_compression_id, color_compression_id)} / "
                f"{DEPTH_COMPRESSION.get(depth_compression_id, depth_compression_id)}"
            )
        if selected and max(selected) >= num_frames:
            raise ValueError(
                f"Requested source frame {max(selected)}, but recording has {num_frames} frames"
            )

        for source_index in range(num_frames):
            camera_to_world = read_matrix(handle)
            timestamp_color, timestamp_depth = read("<QQ", handle)
            color_size, depth_size = read("<QQ", handle)

            if source_index not in selected:
                handle.seek(color_size + depth_size, io.SEEK_CUR)
                continue

            color_bytes = handle.read(color_size)
            depth_bytes = handle.read(depth_size)
            sequence_index = selected[source_index]
            stem = f"{sequence_index:04d}"

            color = Image.open(io.BytesIO(color_bytes)).convert("RGB")
            # AMB3R's ScanNet loader resizes RGB to the registered depth grid.
            color = color.resize((depth_width, depth_height), Image.Resampling.BILINEAR)
            color.save(input_dir / f"{stem}.jpg", quality=95)
            # AMB3R-VO's demo loader accepts every image extension and would
            # otherwise mistake paired *_depth.png files for RGB frames.
            color.save(rgb_dir / f"{stem}.jpg", quality=95)

            depth_mm = np.frombuffer(zlib.decompress(depth_bytes), dtype="<u2").reshape(
                depth_height, depth_width
            )
            maximum_depth = float(depth_mm.max()) / depth_shift
            if maximum_depth <= 0:
                encoded_depth = np.zeros_like(depth_mm)
            else:
                encoded_depth = np.rint(
                    (depth_mm.astype(np.float32) / depth_shift) / maximum_depth * 65535.0
                ).astype(np.uint16)
            # Write depth as I-mode (32-bit signed) PNG. uint16 values fit
            # without loss; the reader must cast back to uint16 on load.
            import warnings as _warnings
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", DeprecationWarning)
                Image.fromarray(encoded_depth.astype(np.int32), mode="I").save(
                    input_dir / f"{stem}_depth.png"
                )
            np.savez_compressed(
                input_dir / f"{stem}.npz",
                camera_pose=camera_to_world,
                camera_intrinsics=intrinsic_depth[:3, :3],
                maximum_depth=np.float32(maximum_depth),
                source_frame_id=np.int64(source_index),
                timestamp_color=np.uint64(timestamp_color),
                timestamp_depth=np.uint64(timestamp_depth),
            )
            records.append(
                {
                    "sequence_index": sequence_index,
                    "source_frame_id": source_index,
                    "timestamp_color": timestamp_color,
                    "timestamp_depth": timestamp_depth,
                    "pose_finite": bool(np.isfinite(camera_to_world).all()),
                    "maximum_depth_m": maximum_depth,
                }
            )

    records.sort(key=lambda item: item["sequence_index"])
    manifest = {
        "source_sens": str(sens_path.resolve()),
        "source_sens_size_bytes": sens_path.stat().st_size,
        "sens_version": version,
        "sensor_name": sensor_name,
        "num_source_frames": num_frames,
        "selection": selection,
        "color_compression": COLOR_COMPRESSION[color_compression_id],
        "depth_compression": DEPTH_COMPRESSION[depth_compression_id],
        "source_color_size": [color_width, color_height],
        "source_depth_size": [depth_width, depth_height],
        "depth_shift": depth_shift,
        "intrinsic_color": intrinsic_color.tolist(),
        "extrinsic_color": extrinsic_color.tolist(),
        "intrinsic_depth": intrinsic_depth.tolist(),
        "extrinsic_depth": extrinsic_depth.tolist(),
        "frames": records,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument(
        "--target-frames",
        type=int,
        help="Uniformly sample this many frames over the complete recording, including both endpoints",
    )
    parser.add_argument(
        "--frame-indices-json",
        type=Path,
        help="Read exact source indices from a JSON object's frame_indices field",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start < 0 or args.count <= 0 or args.stride <= 0:
        raise ValueError("start must be >= 0; count and stride must be > 0")
    if args.target_frames is not None and args.frame_indices_json is not None:
        raise ValueError("Use either --target-frames or --frame-indices-json, not both")

    # Resolve the .sens header to learn num_frames before choosing indices.
    # We open the file twice: once to read the header, once in extract().
    with args.sens.open("rb") as handle:
        _version = read("<I", handle)[0]
        _snl = read("<Q", handle)[0]
        handle.seek(_snl, io.SEEK_CUR)
        for _ in range(4):
            read("<16f", handle)  # skip 4 matrices
        read("<ii", handle)  # color/depth compression
        read("<IIII", handle)  # widths/heights
        read("<f", handle)  # depth_shift
        num_frames = read("<Q", handle)[0]

    if args.frame_indices_json is not None:
        source_indices = load_frame_indices(args.frame_indices_json)
        selection = {
            "method": "explicit_json",
            "frame_indices_json": str(args.frame_indices_json.resolve()),
            "count": len(source_indices),
        }
    elif args.target_frames is not None:
        source_indices = uniform_sample_indices(num_frames, args.target_frames)
        selection = {
            "method": "uniform_endpoints_inclusive",
            "target_frames": args.target_frames,
            "count": len(source_indices),
        }
    else:
        source_indices = [args.start + i * args.stride for i in range(args.count)]
        selection = {
            "method": "fixed_stride",
            "start": args.start,
            "count": args.count,
            "stride": args.stride,
        }

    manifest_path = extract(args.sens, args.output, source_indices, selection)
    frames = json.loads(manifest_path.read_text())["frames"]
    print(f"Extracted {len(frames)} frames to {args.output / 'input'}")
    print(f"RGB-only VO input: {args.output / 'rgb'}")
    print(f"Source frame ids: {frames[0]['source_frame_id']}..{frames[-1]['source_frame_id']}")


if __name__ == "__main__":
    main()
