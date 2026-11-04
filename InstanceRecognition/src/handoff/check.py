"""Check a handoff directory against the contract, whoever produced it.

This runs on the consumer's side of the boundary too: the point of a contract is
that the producer cannot silently drift, and a directory that parses is not the
same as a directory whose numbers agree with its files. Every check here is a
disagreement that would otherwise surface inside the object-generation module as
a confusing downstream symptom.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import schema


def _ply_vertex_count(path: Path) -> int:
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError("not a PLY")
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("truncated header")
            tokens = line.decode("ascii").split()
            if tokens[:1] == ["element"] and tokens[1] == "vertex":
                return int(tokens[2])
            if tokens == ["end_header"]:
                raise ValueError("no vertex element")


def check(directory: Path, strict_missing: bool = True) -> list[str]:
    """Return every problem found; an empty list means the directory is usable."""
    problems: list[str] = []
    manifest_path = directory / "manifest.json"
    objects_path = directory / "objects.json"
    for path in (manifest_path, objects_path):
        if not path.is_file():
            problems.append(f"missing {path.name}")
    if problems:
        return problems

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != schema.SCHEMA_VERSION:
        problems.append(f"schema_version {manifest.get('schema_version')!r} "
                        f"!= {schema.SCHEMA_VERSION!r}")
    if manifest.get("coordinate_frame") != "scannet_scan":
        problems.append(f"unexpected coordinate_frame {manifest.get('coordinate_frame')!r}")
    if manifest.get("units") != "meters":
        problems.append(f"unexpected units {manifest.get('units')!r}")

    records = json.loads(objects_path.read_text())
    if not isinstance(records, list) or not records:
        problems.append("objects.json is not a non-empty list")
        return problems
    if manifest.get("objects") != len(records):
        problems.append(f"manifest says {manifest.get('objects')} objects, "
                        f"objects.json holds {len(records)}")

    seen = set()
    for record in records:
        name = record.get("object_id", "<no id>")
        missing = [key for key in schema.OBJECT_REQUIRED if key not in record]
        if missing and strict_missing:
            problems.append(f"{name}: missing required fields {missing}")
            continue
        if name in seen:
            problems.append(f"{name}: duplicate object_id")
        seen.add(name)

        extra = set(record) - set(schema.OBJECT_REQUIRED) - set(schema.OBJECT_OPTIONAL)
        if extra:
            problems.append(f"{name}: unknown fields {sorted(extra)}")

        cloud = directory / record["cloud"]
        if not cloud.is_file():
            problems.append(f"{name}: cloud missing at {record['cloud']}")
        else:
            count = _ply_vertex_count(cloud)
            if count != record["point_count"]:
                problems.append(f"{name}: point_count {record['point_count']} "
                                f"!= {count} vertices in the PLY")

        quality = schema.label_quality(record["label"])
        if quality != record["label_quality"]:
            problems.append(f"{name}: label_quality {record['label_quality']!r} "
                            f"disagrees with label {record['label']!r} ({quality})")
        if record["structural"] != (quality == "structural"):
            problems.append(f"{name}: structural flag disagrees with label_quality")

        frames = record.get("key_frames") or []
        if not frames:
            problems.append(f"{name}: no key frames")
        roles = [frame.get("role") for frame in frames]
        if len(set(roles)) != len(roles):
            problems.append(f"{name}: duplicate key frame roles {roles}")
        if len(frames) > len(schema.KEYFRAME_ROLES):
            problems.append(f"{name}: {len(frames)} key frames, max "
                            f"{len(schema.KEYFRAME_ROLES)}")
        for frame in frames:
            frame_missing = [key for key in schema.FRAME_REQUIRED if key not in frame]
            if frame_missing:
                problems.append(f"{name} seq {frame.get('seq')}: missing {frame_missing}")
                continue
            if frame["bbox_convention"] != schema.BBOX_CONVENTION:
                problems.append(f"{name} seq {frame['seq']}: bbox_convention differs")
            for key in ("rgb", "mask", "boxed"):
                if not (directory / frame[key]).is_file():
                    problems.append(f"{name} seq {frame['seq']}: {key} image missing")
            mask_path = directory / frame["mask"]
            if mask_path.is_file():
                width, height, _ = _mask_shape(mask_path)
                box = frame["bbox_xyxy"]
                if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
                    problems.append(f"{name} seq {frame['seq']}: bbox {box} outside "
                                    f"{width}x{height} mask")
                inside = int((_mask_array(mask_path) > 0).sum())
                if inside != frame["visible_pixels"]:
                    problems.append(f"{name} seq {frame['seq']}: visible_pixels "
                                    f"{frame['visible_pixels']} != {inside} set pixels")

    return problems


def _mask_array(path: Path) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as image:
        return np.asarray(image)


def _mask_shape(path: Path) -> tuple[int, int, int]:
    array = _mask_array(path)
    return array.shape[1], array.shape[0], 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    problems = check(args.directory)
    if problems:
        for problem in problems:
            print(f"FAIL {problem}")
        raise SystemExit(1)
    print(f"OK {args.directory}")


if __name__ == "__main__":
    main()
