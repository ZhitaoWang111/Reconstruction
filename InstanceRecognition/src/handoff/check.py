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


def frame_problems(block) -> list[str]:
    """The `frame` block both handoffs carry: a scaled rotation plus translation, and a named scale source."""
    if not isinstance(block, dict):
        return ["frame is not an object"]
    out = []
    missing = [k for k in schema.COORD_FRAME_REQUIRED if k not in block]
    if missing:
        return [f"frame missing {missing}"]
    M = np.asarray(block["native_to_frame"], dtype=np.float64)
    scale = block["scale"] or {}
    s = scale.get("value")
    if scale.get("source") not in schema.SCALE_SOURCES:
        out.append(f"frame.scale.source {scale.get('source')!r} not one of {schema.SCALE_SOURCES}")
    if not isinstance(s, (int, float)) or not s > 0:
        out.append(f"frame.scale.value {s!r} is not a positive number")
    elif M.shape != (4, 4) or not np.allclose(M[3], [0, 0, 0, 1]):
        out.append("frame.native_to_frame is not a 4x4 affine matrix")
    else:
        R = M[:3, :3] / s
        if not (np.allclose(R @ R.T, np.eye(3), atol=1e-6) and np.linalg.det(R) > 0):
            out.append("frame.native_to_frame is not scale.value times a rotation")
    return out


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
    if manifest.get("coordinate_frame") != schema.FRAME:
        problems.append(f"unexpected coordinate_frame {manifest.get('coordinate_frame')!r}")
    if manifest.get("units") != "meters":
        problems.append(f"unexpected units {manifest.get('units')!r}")
    missing = [key for key in schema.MANIFEST_REQUIRED if key not in manifest]
    if missing:
        problems.append(f"manifest missing required fields {missing}")
    camera = manifest.get("camera") or {}
    camera_missing = [key for key in schema.CAMERA_REQUIRED if key not in camera]
    if camera_missing:
        problems.append(f"manifest camera missing {camera_missing}")
    else:
        K = np.asarray(camera["K"], dtype=np.float64)
        expected = np.array([[camera["fx"], 0, camera["cx"]], [0, camera["fy"], camera["cy"]], [0, 0, 1]])
        if K.shape != (3, 3) or not np.allclose(K, expected):
            problems.append("manifest camera K disagrees with fx/fy/cx/cy")
        if camera["extrinsics"] != schema.EXTRINSICS_CONVENTION:
            problems.append(f"manifest camera extrinsics convention {camera['extrinsics']} "
                            f"!= {schema.EXTRINSICS_CONVENTION}")
    if "floor_z" in manifest and not (isinstance(manifest["floor_z"], (int, float)) and abs(manifest["floor_z"]) < 0.1):
        problems.append(f"floor_z {manifest['floor_z']!r}: the floor is at z = 0 in {schema.FRAME}, "
                        "give or take how far this producer's clouds sit from the reconstructed floor")
    if "axis_alignment" in manifest:
        problems.append("axis_alignment is not part of the contract any more (the frame is amb3r_floor)")
    if "frame" in manifest:
        problems += frame_problems(manifest["frame"])

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
            pose = np.asarray(frame["camera_to_world"] if frame["camera_to_world"] is not None
                              else np.full((4, 4), np.nan), dtype=np.float64)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                problems.append(f"{name} seq {frame['seq']}: camera_to_world is not a finite 4x4")
            elif (not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6)
                  or not np.allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-3)
                  or np.linalg.det(pose[:3, :3]) < 0):
                problems.append(f"{name} seq {frame['seq']}: camera_to_world is not a rigid transform")
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
