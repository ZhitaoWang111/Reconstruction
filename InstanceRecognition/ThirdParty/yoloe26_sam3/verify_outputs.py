#!/usr/bin/env python3
"""Validate YOLOE-26 + SAM3/SAM3.1 mask artifacts."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    output_dir = parser.parse_args().output_dir.expanduser().resolve()
    manifest = json.loads((output_dir / "predictions.json").read_text())
    source = cv2.imread(manifest["source"], cv2.IMREAD_COLOR)
    annotated = cv2.imread(str(output_dir / manifest["artifacts"]["annotated_image"]))
    instance_map = cv2.imread(
        str(output_dir / manifest["artifacts"]["instance_id_map"]), cv2.IMREAD_UNCHANGED
    )
    if source is None or annotated is None or instance_map is None:
        raise RuntimeError("A required image artifact could not be read")
    if source.shape != annotated.shape or source.shape[:2] != instance_map.shape:
        raise RuntimeError("Artifact dimensions do not match the source image")
    if manifest["yoloe_masks_consumed"] is not False:
        raise RuntimeError("Pipeline must use YOLOE boxes, not upstream masks")

    instances = manifest["instances"]
    if not instances or len(instances) != manifest["instance_count"]:
        raise RuntimeError("Instance count is empty or inconsistent")
    upstream = json.loads(Path(manifest["yoloe_predictions"]).read_text())
    if len(instances) != upstream["instance_count"]:
        raise RuntimeError("Not every YOLOE detection received a SAM mask")

    for item in instances:
        mask = cv2.imread(str(output_dir / item["mask_file"]), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != source.shape[:2]:
            raise RuntimeError(f"Missing or invalid mask: {item['mask_file']}")
        values = set(np.unique(mask).tolist())
        if not values.issubset({0, 255}):
            raise RuntimeError(f"Mask is not binary: {item['mask_file']}")
        pixels = int(np.count_nonzero(mask))
        if pixels == 0 or pixels != item["mask_pixels"]:
            raise RuntimeError(f"Mask pixel count mismatch: {item['mask_file']}")
        if not (0 <= item["prompt_box_iou"] <= 1):
            raise RuntimeError("Invalid prompt/mask-box IoU")

    if int(instance_map.max()) > len(instances):
        raise RuntimeError("Instance map contains an out-of-range ID")
    print("VERIFY_OK")
    print(f"sam_version={manifest['sam_version']}")
    print(f"instances={len(instances)}")
    print(f"unique_classes={len(manifest['unique_classes'])}")
    print(f"image_shape={source.shape[1]}x{source.shape[0]}")


if __name__ == "__main__":
    main()
