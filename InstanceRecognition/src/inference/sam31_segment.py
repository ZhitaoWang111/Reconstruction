#!/usr/bin/env python3
"""Second-stage SAM3.1 masks from first-stage YOLOE boxes."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
SAM3_REPO = ROOT / "ThirdParty/SAM3"
KERNEL_CACHE = ROOT / "runtime/cache/torch"
KERNEL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("PYTORCH_KERNEL_CACHE_PATH", str(KERNEL_CACHE))
sys.path.insert(0, str(SAM3_REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from sam3.model_builder import build_sam3_multiplex_video_predictor  # noqa: E402


PALETTE = [
    (255, 99, 71),
    (30, 144, 255),
    (50, 205, 50),
    (255, 193, 7),
    (186, 85, 211),
    (0, 206, 209),
    (255, 105, 180),
    (255, 140, 0),
    (70, 130, 180),
    (154, 205, 50),
    (220, 20, 60),
    (64, 224, 208),
    (138, 43, 226),
    (210, 180, 140),
    (0, 191, 255),
    (240, 128, 128),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.0)
    return parser.parse_args()


def safe_name(label: str) -> str:
    value = re.sub(r"[^0-9A-Za-z._-]+", "_", label.strip())
    return value.strip("_") or "object"


def normalize_xywh(box_xyxy: Sequence[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box_xyxy
    return [x1 / width, y1 / height, (x2 - x1) / width, (y2 - y1) / height]


def normalized_xywh_to_xyxy(boxes_xywh: np.ndarray, width: int, height: int) -> np.ndarray:
    boxes = np.asarray(boxes_xywh, dtype=np.float32).reshape(-1, 4).copy()
    if not len(boxes):
        return boxes
    boxes[:, [0, 2]] *= width
    boxes[:, [1, 3]] *= height
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    return boxes


def box_iou(boxes: np.ndarray, target: Sequence[float]) -> np.ndarray:
    if not len(boxes):
        return np.empty(0, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    left_top = np.maximum(boxes[:, :2], target[:2])
    right_bottom = np.minimum(boxes[:, 2:], target[2:])
    intersection_size = np.maximum(right_bottom - left_top, 0)
    intersection = intersection_size[:, 0] * intersection_size[:, 1]
    box_area = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(
        boxes[:, 3] - boxes[:, 1], 0
    )
    target_area = max(target[2] - target[0], 0) * max(target[3] - target[1], 0)
    return intersection / np.maximum(box_area + target_area - intersection, 1e-6)


def select_box_candidate(outputs: dict, prompt_box: Sequence[float], width: int, height: int) -> dict:
    masks = np.asarray(outputs["out_binary_masks"]).astype(bool)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    boxes = normalized_xywh_to_xyxy(outputs["out_boxes_xywh"], width, height)
    scores = np.asarray(outputs["out_probs"], dtype=np.float32).reshape(-1)
    if masks.ndim != 3 or not len(masks):
        raise RuntimeError("SAM3.1 returned no box-prompt mask candidates")
    if not (len(masks) == len(boxes) == len(scores)):
        raise RuntimeError("SAM3.1 box-prompt output cardinality mismatch")
    ious = box_iou(boxes, prompt_box)
    index = int(np.argmax(ious))
    return {
        "mask": masks[index],
        "sam_box_xyxy": boxes[index].round(2).tolist(),
        "sam_score": float(scores[index]),
        "prompt_box_iou": float(ious[index]),
        "candidate_count": len(masks),
        "prompt_type": "positive_box",
    }


def select_point_object(
    outputs: dict,
    obj_id: int,
    prompt_box: Sequence[float],
    width: int,
    height: int,
) -> dict:
    obj_ids = np.asarray(outputs["out_obj_ids"]).reshape(-1)
    matches = np.flatnonzero(obj_ids == obj_id)
    if len(matches) != 1:
        raise RuntimeError(f"SAM3.1 point fallback did not return obj_id={obj_id}")
    index = int(matches[0])
    boxes = normalized_xywh_to_xyxy(outputs["out_boxes_xywh"], width, height)
    mask = np.asarray(outputs["out_binary_masks"])[index].astype(bool)
    return {
        "mask": mask,
        "sam_box_xyxy": boxes[index].round(2).tolist(),
        "sam_score": float(np.asarray(outputs["out_probs"])[index]),
        "prompt_box_iou": float(box_iou(boxes[index : index + 1], prompt_box)[0]),
        "candidate_count": len(obj_ids),
        "prompt_type": "center_point_fallback",
    }


def render_masks(image: np.ndarray, instances: list[dict], masks: list[np.ndarray]) -> np.ndarray:
    output = image.astype(np.float32)
    for index, mask in enumerate(masks):
        color = np.asarray(PALETTE[index % len(PALETTE)], dtype=np.float32)
        output[mask] = output[mask] * 0.46 + color * 0.54
    output = np.clip(output, 0, 255).astype(np.uint8)
    scale = max(0.55, min(output.shape[:2]) / 1000)
    thickness = max(2, round(scale * 3))
    for index, (instance, mask) in enumerate(zip(instances, masks)):
        color = PALETTE[index % len(PALETTE)]
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(output, contours, -1, color, thickness, cv2.LINE_AA)
        x1, y1, x2, y2 = [round(value) for value in instance["yoloe_box_xyxy"]]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        text = f"{instance['class_name']} {instance['yoloe_confidence']:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )
        text_y = max(text_height + baseline + 4, y1)
        cv2.rectangle(
            output,
            (x1, text_y - text_height - baseline - 6),
            (min(output.shape[1] - 1, x1 + text_width + 7), text_y + 2),
            color,
            -1,
        )
        cv2.putText(
            output,
            text,
            (x1 + 3, text_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return output


def run_sam31(
    predictor, image: Image.Image, detections: list[dict], threshold: float
) -> tuple[list[dict], float]:
    width, height = image.size
    results = []
    with tempfile.TemporaryDirectory(prefix="instance_recognition_sam31_") as temp_dir:
        image.convert("RGB").save(
            Path(temp_dir) / "000000.jpg", quality=100, subsampling=0
        )
        inference_state = predictor.model.init_state(
            resource_path=temp_dir,
            offload_video_to_cpu=False,
            async_loading_frames=False,
        )
        session_id = str(uuid.uuid4())
        now = time.time()
        predictor._all_inference_states[session_id] = {
            "state": inference_state,
            "session_id": session_id,
            "start_time": now,
            "last_use_time": now,
        }
        prompt_started = time.perf_counter()
        try:
            for index, detection in enumerate(detections):
                box = detection["box_xyxy"]
                response = predictor.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": 0,
                        "bounding_boxes": [normalize_xywh(box, width, height)],
                        "bounding_box_labels": [1],
                        "output_prob_thresh": threshold,
                    }
                )
                try:
                    result = select_box_candidate(
                        response["outputs"], box, width, height
                    )
                except RuntimeError:
                    x1, y1, x2, y2 = box
                    obj_id = 10000 + index
                    response = predictor.handle_request(
                        {
                            "type": "add_prompt",
                            "session_id": session_id,
                            "frame_index": 0,
                            "points": [[
                                ((x1 + x2) / 2) / width,
                                ((y1 + y2) / 2) / height,
                            ]],
                            "point_labels": [1],
                            "obj_id": obj_id,
                        }
                    )
                    result = select_point_object(
                        response["outputs"], obj_id, box, width, height
                    )
                results.append(result)
        finally:
            predictor.handle_request(
                {"type": "close_session", "session_id": session_id}
            )
        prompt_seconds = time.perf_counter() - prompt_started
    return results, prompt_seconds


class Sam31Segmenter:
    """Reusable SAM3.1 predictor with per-image sessions and resident weights."""

    def __init__(self, checkpoint: Path, threshold: float = 0.0) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        self.threshold = threshold
        started = time.perf_counter()
        self.predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=str(self.checkpoint),
            max_num_objects=16,
            multiplex_count=16,
            use_fa3=False,
            use_rope_real=False,
            compile=False,
            warm_up=False,
            async_loading_frames=False,
            default_output_prob_thresh=threshold,
        )
        self.load_seconds = time.perf_counter() - started

    def infer(
        self, source: Path, detections_path: Path, output_dir: Path
    ) -> dict:
        source = Path(source).expanduser().resolve()
        detections_path = Path(detections_path).expanduser().resolve()
        output_dir = Path(output_dir).expanduser().resolve()
        for path in (source, detections_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        upstream = json.loads(detections_path.read_text())
        if Path(upstream["source"]).resolve() != source:
            raise RuntimeError("YOLOE source does not match the SAM3.1 source")
        if upstream.get("native_masks_forwarded") is not False:
            raise RuntimeError("This pipeline accepts YOLOE boxes, never YOLOE masks")
        detections = upstream["instances"]

        image_pil = Image.open(source).convert("RGB")
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not decode {source}")
        if detections:
            results, prompt_seconds = run_sam31(
                self.predictor, image_pil, detections, self.threshold
            )
            timing = {
                "model_load_seconds": round(self.load_seconds, 3),
                "session_and_prompts_seconds": round(prompt_seconds, 3),
                "model_reused": True,
            }
        else:
            results = []
            timing = {
                "model_load_seconds": round(self.load_seconds, 3),
                "model_reused": True,
                "skipped": "no YOLOE detections",
            }
        if len(results) != len(detections):
            raise RuntimeError("Not every YOLOE detection received a SAM3.1 mask")

        output_dir.mkdir(parents=True, exist_ok=True)
        mask_dir = output_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        for path in mask_dir.glob("*.png"):
            path.unlink()
        instance_map = np.zeros(image.shape[:2], dtype=np.uint16)
        masks = []
        instances = []
        for instance_id, (detection, result) in enumerate(
            zip(detections, results), start=1
        ):
            mask = np.asarray(result.pop("mask"), dtype=bool)
            if mask.shape != image.shape[:2] or not mask.any():
                raise RuntimeError(f"Invalid SAM3.1 mask for detection {instance_id}")
            masks.append(mask)
            mask_path = (
                mask_dir
                / f"{instance_id:03d}_{safe_name(detection['class_name'])}.png"
            )
            if not cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255):
                raise RuntimeError(f"Could not write {mask_path}")
            instance_map[mask] = instance_id
            instances.append(
                {
                    "instance_id": instance_id,
                    "class_id": detection["class_id"],
                    "class_name": detection["class_name"],
                    "yoloe_confidence": detection["confidence"],
                    "yoloe_box_xyxy": detection["box_xyxy"],
                    "sam_score": round(result["sam_score"], 6),
                    "sam_box_xyxy": result["sam_box_xyxy"],
                    "prompt_box_iou": round(result["prompt_box_iou"], 6),
                    "sam_candidate_count": result["candidate_count"],
                    "sam_prompt_type": result["prompt_type"],
                    "mask_pixels": int(mask.sum()),
                    "mask_file": str(mask_path.relative_to(output_dir)),
                }
            )

        annotated_path = output_dir / "annotated.jpg"
        if not cv2.imwrite(str(annotated_path), render_masks(image, instances, masks)):
            raise RuntimeError(f"Could not write {annotated_path}")
        instance_map_path = output_dir / "instance_id_map.png"
        if not cv2.imwrite(str(instance_map_path), instance_map):
            raise RuntimeError(f"Could not write {instance_map_path}")
        manifest = {
            "stage": "SAM3.1 masks prompted by YOLOE-26 boxes",
            "source": str(source),
            "yoloe_detections": str(detections_path),
            "yoloe_masks_consumed": False,
            "checkpoint": str(self.checkpoint),
            "instance_count": len(instances),
            "unique_classes": upstream["unique_classes"],
            "image_shape_hwc": list(image.shape),
            "timing": timing,
            "artifacts": {
                "annotated_image": annotated_path.name,
                "instance_id_map": instance_map_path.name,
                "mask_directory": mask_dir.name,
            },
            "instances": instances,
        }
        (output_dir / "predictions.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )
        return manifest


def main() -> None:
    args = parse_args()
    segmenter = Sam31Segmenter(args.checkpoint, args.threshold)
    manifest = segmenter.infer(args.source, args.detections, args.output_dir)
    print(f"SAM31_OK instances={manifest['instance_count']} output={args.output_dir}")


if __name__ == "__main__":
    main()
