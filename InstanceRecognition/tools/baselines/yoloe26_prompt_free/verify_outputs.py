#!/usr/bin/env python3
"""Validate the artifact contract produced by infer_prompt_free.py."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    output_dir = parse_args().output_dir.expanduser().resolve()
    manifest_path = output_dir / "predictions.json"
    manifest = json.loads(manifest_path.read_text())

    source = cv2.imread(manifest["source"], cv2.IMREAD_COLOR)
    annotated = cv2.imread(str(output_dir / manifest["artifacts"]["annotated_image"]))
    instance_map = cv2.imread(
        str(output_dir / manifest["artifacts"]["instance_id_map"]), cv2.IMREAD_UNCHANGED
    )
    if source is None or annotated is None or instance_map is None:
        raise RuntimeError("A required image artifact could not be loaded")
    if annotated.shape != source.shape:
        raise RuntimeError(f"Annotated/source shape mismatch: {annotated.shape} vs {source.shape}")
    if instance_map.shape != source.shape[:2]:
        raise RuntimeError(
            f"Instance map/source shape mismatch: {instance_map.shape} vs {source.shape[:2]}"
        )

    instances = manifest["instances"]
    if len(instances) != manifest["instance_count"] or not instances:
        raise RuntimeError("Manifest instance count is empty or inconsistent")
    if len(manifest["unique_classes"]) != len(set(item["class_name"] for item in instances)):
        raise RuntimeError("Unique class list is inconsistent with instance records")

    for instance in instances:
        mask_path = output_dir / instance["mask_file"]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Missing mask: {mask_path}")
        if mask.shape != source.shape[:2]:
            raise RuntimeError(f"Mask shape mismatch for {mask_path}: {mask.shape}")
        values = set(np.unique(mask).tolist())
        if not values.issubset({0, 255}):
            raise RuntimeError(f"Mask is not binary: {mask_path}, values={sorted(values)}")
        pixels = int(np.count_nonzero(mask))
        if pixels == 0 or pixels != instance["mask_pixels"]:
            raise RuntimeError(
                f"Mask pixel count mismatch for {mask_path}: file={pixels}, manifest={instance['mask_pixels']}"
            )

    max_id = int(instance_map.max())
    if max_id > manifest["instance_count"]:
        raise RuntimeError(
            f"Instance map contains id {max_id}, above count {manifest['instance_count']}"
        )

    print("VERIFY_OK")
    print(f"output_dir={output_dir}")
    print(f"image_shape={source.shape[1]}x{source.shape[0]}")
    print(f"instances={manifest['instance_count']}")
    print(f"unique_classes={len(manifest['unique_classes'])}")
    print(f"binary_masks={len(instances)}")


if __name__ == "__main__":
    main()
