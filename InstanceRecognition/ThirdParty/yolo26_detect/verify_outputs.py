#!/usr/bin/env python3
"""Validate artifacts produced by infer_yolo26_detect.py."""

import argparse
import json
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    output_dir = parse_args().output_dir.expanduser().resolve()
    manifest = json.loads((output_dir / "predictions.json").read_text())

    source = cv2.imread(manifest["source"], cv2.IMREAD_COLOR)
    annotated = cv2.imread(str(output_dir / manifest["artifacts"]["annotated_image"]))
    if source is None or annotated is None:
        raise RuntimeError("A required image artifact could not be loaded")
    if annotated.shape != source.shape:
        raise RuntimeError(f"Annotated/source shape mismatch: {annotated.shape} vs {source.shape}")

    detections = manifest["detections"]
    if len(detections) != manifest["detection_count"] or not detections:
        raise RuntimeError("Manifest detection count is empty or inconsistent")
    classes = {item["class_name"] for item in detections}
    if sorted(classes) != manifest["unique_classes"]:
        raise RuntimeError("Unique class list is inconsistent with detection records")

    height, width = source.shape[:2]
    threshold = float(manifest["confidence_threshold"])
    for detection in detections:
        x1, y1, x2, y2 = detection["box_xyxy"]
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise RuntimeError(f"Invalid bounding box: {detection['box_xyxy']}")
        confidence = float(detection["confidence"])
        if not (threshold <= confidence <= 1.0):
            raise RuntimeError(f"Invalid confidence: {confidence}")

    print("VERIFY_OK")
    print(f"output_dir={output_dir}")
    print(f"image_shape={width}x{height}")
    print(f"detections={manifest['detection_count']}")
    print(f"unique_classes={len(manifest['unique_classes'])}")
    print("masks=0")


if __name__ == "__main__":
    main()
