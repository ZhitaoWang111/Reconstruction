#!/usr/bin/env python3
"""Refine existing YOLOE-26 detections into masks with SAM3 or SAM3.1."""

import argparse
import gc
import json
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parent
SAM3_REPO = ROOT.parent / "SAM3"
TORCH_KERNEL_CACHE = ROOT / ".cache" / "torch"
TORCH_KERNEL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("PYTORCH_KERNEL_CACHE_PATH", str(TORCH_KERNEL_CACHE))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402
from sam3.model_builder import (  # noqa: E402
    build_sam3_image_model,
    build_sam3_multiplex_video_predictor,
)


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
    parser = argparse.ArgumentParser(
        description=(
            "Use only YOLOE-26 labels and boxes as upstream detections, then obtain "
            "one mask per detection from SAM3 or SAM3.1."
        )
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--yoloe-predictions", required=True, type=Path)
    parser.add_argument("--sam-version", required=True, choices=["sam3", "sam3.1"])
    parser.add_argument("--sam-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sam-threshold", type=float, default=0.0)
    return parser.parse_args()


def safe_name(label: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", label.strip())
    return cleaned.strip("_") or "object"


def box_iou(boxes: np.ndarray, target: Sequence[float]) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.float32)
    target_box = np.asarray(target, dtype=np.float32)
    lt = np.maximum(boxes[:, :2], target_box[:2])
    rb = np.minimum(boxes[:, 2:], target_box[2:])
    wh = np.maximum(rb - lt, 0)
    intersection = wh[:, 0] * wh[:, 1]
    box_area = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(
        boxes[:, 3] - boxes[:, 1], 0
    )
    target_area = max(target_box[2] - target_box[0], 0) * max(
        target_box[3] - target_box[1], 0
    )
    return intersection / np.maximum(box_area + target_area - intersection, 1e-6)


def normalize_cxcywh(box_xyxy: Sequence[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = box_xyxy
    return [
        ((x1 + x2) / 2) / width,
        ((y1 + y2) / 2) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    ]


def normalize_xywh(box_xyxy: Sequence[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = box_xyxy
    return [x1 / width, y1 / height, (x2 - x1) / width, (y2 - y1) / height]


def xywh_normalized_to_xyxy(
    boxes_xywh: np.ndarray, width: int, height: int
) -> np.ndarray:
    if len(boxes_xywh) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = np.asarray(boxes_xywh, dtype=np.float32).copy()
    boxes[:, 0] *= width
    boxes[:, 1] *= height
    boxes[:, 2] *= width
    boxes[:, 3] *= height
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    return boxes


def select_candidate(
    masks: np.ndarray,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    prompt_box: Sequence[float],
) -> Tuple[np.ndarray, List[float], float, float, int]:
    masks = np.asarray(masks).astype(bool)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3 or len(masks) == 0:
        raise RuntimeError("SAM returned no usable mask candidates for a YOLOE box")
    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if not (len(masks) == len(boxes_xyxy) == len(scores)):
        raise RuntimeError(
            "SAM candidate count mismatch: "
            f"masks={len(masks)}, boxes={len(boxes_xyxy)}, scores={len(scores)}"
        )
    ious = box_iou(boxes_xyxy, prompt_box)
    selected = int(np.argmax(ious))
    return (
        masks[selected],
        boxes_xyxy[selected].round(2).tolist(),
        float(scores[selected]),
        float(ious[selected]),
        len(masks),
    )


def select_object_id(
    outputs: Dict,
    obj_id: int,
    width: int,
    height: int,
    prompt_box: Sequence[float],
) -> Tuple[np.ndarray, List[float], float, float, int]:
    obj_ids = np.asarray(outputs["out_obj_ids"]).reshape(-1)
    matches = np.flatnonzero(obj_ids == obj_id)
    if len(matches) != 1:
        raise RuntimeError(
            f"SAM3.1 point fallback expected obj_id={obj_id}, got {obj_ids.tolist()}"
        )
    index = int(matches[0])
    boxes_xyxy = xywh_normalized_to_xyxy(
        np.asarray(outputs["out_boxes_xywh"]), width, height
    )
    iou = float(box_iou(boxes_xyxy[index : index + 1], prompt_box)[0])
    return (
        np.asarray(outputs["out_binary_masks"])[index].astype(bool),
        boxes_xyxy[index].round(2).tolist(),
        float(np.asarray(outputs["out_probs"])[index]),
        iou,
        len(obj_ids),
    )


def infer_sam3_image(
    image: Image.Image,
    detections: List[Dict],
    checkpoint: Path,
    threshold: float,
) -> Tuple[List[Dict], Dict]:
    started = time.perf_counter()
    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        device="cuda",
        eval_mode=True,
        compile=False,
    )
    load_seconds = time.perf_counter() - started
    processor = Sam3Processor(model, device="cuda", confidence_threshold=threshold)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    with torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)
        encode_seconds = time.perf_counter() - started - load_seconds
        width, height = image.size
        results = []
        prompt_started = time.perf_counter()
        for detection in detections:
            processor.reset_all_prompts(state)
            state = processor.add_geometric_prompt(
                box=normalize_cxcywh(detection["box_xyxy"], width, height),
                label=True,
                state=state,
            )
            selected = select_candidate(
                state["masks"].detach().cpu().numpy(),
                state["boxes"].detach().float().cpu().numpy(),
                state["scores"].detach().float().cpu().numpy(),
                detection["box_xyxy"],
            )
            results.append(dict(zip(
                ["mask", "sam_box_xyxy", "sam_score", "prompt_box_iou", "candidate_count"],
                selected,
            )))
    timing = {
        "model_load_seconds": round(load_seconds, 3),
        "image_encode_seconds": round(encode_seconds, 3),
        "all_box_prompts_seconds": round(time.perf_counter() - prompt_started, 3),
    }
    del processor, state, model
    gc.collect()
    torch.cuda.empty_cache()
    return results, timing


def infer_sam31_video(
    image: Image.Image,
    detections: List[Dict],
    checkpoint: Path,
    threshold: float,
) -> Tuple[List[Dict], Dict]:
    started = time.perf_counter()
    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path=str(checkpoint),
        max_num_objects=16,
        multiplex_count=16,
        use_fa3=False,
        # The released multiplex checkpoint stores complex `freqs_cis` tensors.
        # The builder default is aimed at compiled real-valued RoPE and otherwise
        # leaves those checkpoint tensors unmatched.
        use_rope_real=False,
        compile=False,
        warm_up=False,
        async_loading_frames=False,
        default_output_prob_thresh=threshold,
    )
    load_seconds = time.perf_counter() - started
    width, height = image.size
    results = []
    with tempfile.TemporaryDirectory(prefix="yoloe_sam31_") as temp_dir:
        frame_path = Path(temp_dir) / "000000.jpg"
        image.convert("RGB").save(frame_path, quality=100, subsampling=0)
        # The shared predictor currently forwards `offload_state_to_cpu`, while
        # the official SAM3.1 model's init_state does not accept that SAM3-only
        # keyword. Register the otherwise standard session directly until the
        # upstream wrapper signatures converge.
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
            for detection_index, detection in enumerate(detections):
                response = predictor.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": 0,
                        "bounding_boxes": [
                            normalize_xywh(detection["box_xyxy"], width, height)
                        ],
                        "bounding_box_labels": [1],
                        "output_prob_thresh": threshold,
                    }
                )
                outputs = response["outputs"]
                boxes_xyxy = xywh_normalized_to_xyxy(
                    outputs["out_boxes_xywh"], width, height
                )
                try:
                    selected = select_candidate(
                        outputs["out_binary_masks"],
                        boxes_xyxy,
                        outputs["out_probs"],
                        detection["box_xyxy"],
                    )
                    prompt_type = "positive_box"
                except RuntimeError:
                    x1, y1, x2, y2 = detection["box_xyxy"]
                    obj_id = 10000 + detection_index
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
                    selected = select_object_id(
                        response["outputs"],
                        obj_id,
                        width,
                        height,
                        detection["box_xyxy"],
                    )
                    prompt_type = "center_point_fallback"
                result = dict(zip(
                    ["mask", "sam_box_xyxy", "sam_score", "prompt_box_iou", "candidate_count"],
                    selected,
                ))
                result["prompt_type"] = prompt_type
                results.append(result)
        finally:
            predictor.handle_request(
                {"type": "close_session", "session_id": session_id}
            )
        prompt_seconds = time.perf_counter() - prompt_started
    timing = {
        "model_load_seconds": round(load_seconds, 3),
        "image_session_and_prompts_seconds": round(prompt_seconds, 3),
    }
    del predictor
    gc.collect()
    torch.cuda.empty_cache()
    return results, timing


def render_result(image_bgr: np.ndarray, instances: List[Dict], masks: List[np.ndarray]) -> np.ndarray:
    output = image_bgr.astype(np.float32)
    for index, mask in enumerate(masks):
        color = np.asarray(PALETTE[index % len(PALETTE)], dtype=np.float32)
        output[mask] = output[mask] * 0.48 + color * 0.52
    output = np.clip(output, 0, 255).astype(np.uint8)
    for index, (instance, mask) in enumerate(zip(instances, masks)):
        color = PALETTE[index % len(PALETTE)]
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(output, contours, -1, color, 2, cv2.LINE_AA)
        x1, y1, x2, y2 = [int(round(value)) for value in instance["yoloe_box_xyxy"]]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f"{instance['class_name']} {instance['yoloe_confidence']:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.45, min(output.shape[:2]) / 1100)
        thickness = max(1, int(round(font_scale * 2)))
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )
        text_y = max(text_height + baseline + 2, y1)
        cv2.rectangle(
            output,
            (x1, text_y - text_height - baseline - 4),
            (min(output.shape[1] - 1, x1 + text_width + 5), text_y + 2),
            color,
            -1,
        )
        cv2.putText(
            output,
            label,
            (x1 + 2, text_y - baseline),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return output


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    predictions_path = args.yoloe_predictions.expanduser().resolve()
    checkpoint = args.sam_checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for required in [source, predictions_path, checkpoint]:
        if not required.is_file():
            raise FileNotFoundError(required)

    upstream = json.loads(predictions_path.read_text())
    if Path(upstream["source"]).resolve() != source:
        raise RuntimeError("YOLOE manifest source does not match --source")
    detections = upstream["instances"]
    if not detections:
        raise RuntimeError("YOLOE manifest contains no detections")

    image = Image.open(source).convert("RGB")
    image_bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Could not read source image: {source}")
    if args.sam_version == "sam3":
        sam_results, timing = infer_sam3_image(
            image, detections, checkpoint, args.sam_threshold
        )
        backend = "SAM3 image model with independent positive box prompts"
    else:
        sam_results, timing = infer_sam31_video(
            image, detections, checkpoint, args.sam_threshold
        )
        backend = "SAM3.1 multiplex video model on a one-frame image session"
    if len(sam_results) != len(detections):
        raise RuntimeError("Did not obtain one SAM result per YOLOE detection")

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    for old_mask in mask_dir.glob("*.png"):
        old_mask.unlink()

    height, width = image_bgr.shape[:2]
    instance_map = np.zeros((height, width), dtype=np.uint16)
    manifest_instances = []
    masks = []
    for index, (detection, sam_result) in enumerate(
        zip(detections, sam_results), start=1
    ):
        mask = np.asarray(sam_result.pop("mask"), dtype=bool)
        if mask.shape != (height, width) or not mask.any():
            raise RuntimeError(f"Invalid SAM mask for detection {index}: {mask.shape}")
        masks.append(mask)
        label = detection["class_name"]
        mask_path = mask_dir / f"{index:03d}_{safe_name(label)}.png"
        if not cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255):
            raise RuntimeError(f"Could not save {mask_path}")
        instance_map[mask] = index
        manifest_instances.append(
            {
                "instance_id": index,
                "class_id": detection["class_id"],
                "class_name": label,
                "yoloe_confidence": detection["confidence"],
                "yoloe_box_xyxy": detection["box_xyxy"],
                "sam_score": round(sam_result["sam_score"], 6),
                "sam_box_xyxy": sam_result["sam_box_xyxy"],
                "prompt_box_iou": round(sam_result["prompt_box_iou"], 6),
                "sam_candidate_count": sam_result["candidate_count"],
                "sam_prompt_type": sam_result.get("prompt_type", "positive_box"),
                "mask_pixels": int(mask.sum()),
                "mask_file": str(mask_path.relative_to(output_dir)),
            }
        )

    annotated_path = output_dir / "annotated.jpg"
    annotated = render_result(image_bgr, manifest_instances, masks)
    if not cv2.imwrite(str(annotated_path), annotated):
        raise RuntimeError(f"Could not save {annotated_path}")
    instance_map_path = output_dir / "instance_id_map.png"
    if not cv2.imwrite(str(instance_map_path), instance_map):
        raise RuntimeError(f"Could not save {instance_map_path}")

    manifest = {
        "task": "image-only YOLOE-26 detection followed by SAM masks",
        "system_text_prompt": None,
        "sam_version": args.sam_version,
        "sam_backend": backend,
        "source": str(source),
        "yoloe_predictions": str(predictions_path),
        "yoloe_checkpoint": upstream["checkpoint"],
        "yoloe_masks_consumed": False,
        "sam_checkpoint": str(checkpoint),
        "sam_checkpoint_size_bytes": checkpoint.stat().st_size,
        "sam_threshold": args.sam_threshold,
        "image_shape_hwc": list(image_bgr.shape),
        "instance_count": len(manifest_instances),
        "unique_classes": sorted({item["class_name"] for item in manifest_instances}),
        "timing": timing,
        "artifacts": {
            "annotated_image": annotated_path.name,
            "instance_id_map": instance_map_path.name,
            "mask_directory": mask_dir.name,
        },
        "instances": manifest_instances,
    }
    (output_dir / "predictions.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print("CHECK_OK")
    print(f"sam_version={args.sam_version}")
    print(f"instances={manifest['instance_count']}")
    print(f"unique_classes={len(manifest['unique_classes'])}")
    print(f"yoloe_masks_consumed={manifest['yoloe_masks_consumed']}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
