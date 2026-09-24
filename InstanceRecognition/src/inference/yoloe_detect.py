#!/usr/bin/env python3
"""First-stage YOLOE-26 Prompt-Free detection for the local application.

The checkpoint has a segmentation head, but this stage deliberately records and
renders only labels, confidence scores, and boxes. Its native masks are never
saved or forwarded to SAM3.1.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "runtime/config"
KERNEL_CACHE = ROOT / "runtime/cache/torch"
(CONFIG_ROOT / "Ultralytics").mkdir(parents=True, exist_ok=True)
KERNEL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(CONFIG_ROOT))
os.environ.setdefault("PYTORCH_KERNEL_CACHE_PATH", str(KERNEL_CACHE))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLOE, __version__ as ultralytics_version  # noqa: E402


PALETTE = [
    (255, 120, 72),
    (71, 160, 255),
    (89, 196, 116),
    (198, 113, 255),
    (70, 207, 219),
    (255, 158, 94),
    (232, 108, 151),
    (120, 184, 91),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-det", type=int, default=16)
    return parser.parse_args()


def render_boxes(image: np.ndarray, instances: list[dict]) -> np.ndarray:
    output = image.copy()
    scale = max(0.55, min(output.shape[:2]) / 1000)
    thickness = max(2, round(scale * 3))
    for index, instance in enumerate(instances):
        color = PALETTE[index % len(PALETTE)]
        x1, y1, x2, y2 = [round(value) for value in instance["box_xyxy"]]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        text = f"{instance['class_name']} {instance['confidence']:.2f}"
        (width, height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )
        label_y = max(height + baseline + 4, y1)
        cv2.rectangle(
            output,
            (x1, label_y - height - baseline - 6),
            (min(output.shape[1] - 1, x1 + width + 7), label_y + 2),
            color,
            -1,
        )
        cv2.putText(
            output,
            text,
            (x1 + 3, label_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return output


class YoloeDetector:
    """Reusable YOLOE detector whose weights stay resident between requests."""

    def __init__(
        self,
        checkpoint: Path,
        device: str = "0",
        imgsz: int = 640,
        conf: float = 0.25,
        max_det: int = 16,
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.max_det = max_det
        started = time.perf_counter()
        self.model = YOLOE(str(self.checkpoint))
        self.load_seconds = time.perf_counter() - started

    def infer(self, source: Path, output_dir: Path) -> dict:
        source = Path(source).expanduser().resolve()
        output_dir = Path(output_dir).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        output_dir.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        results = self.model.predict(
            source=str(source),
            device=self.device,
            imgsz=self.imgsz,
            conf=self.conf,
            max_det=self.max_det,
            retina_masks=False,
            verbose=False,
        )
        inference_seconds = time.perf_counter() - started
        if len(results) != 1:
            raise RuntimeError(f"Expected one image result, got {len(results)}")
        result = results[0]
        boxes = (
            np.empty((0, 4), dtype=np.float32)
            if result.boxes is None
            else result.boxes.xyxy.detach().cpu().numpy()
        )
        scores = (
            np.empty(0, dtype=np.float32)
            if result.boxes is None
            else result.boxes.conf.detach().cpu().numpy()
        )
        classes = (
            np.empty(0, dtype=np.int64)
            if result.boxes is None
            else result.boxes.cls.detach().cpu().numpy().astype(int)
        )
        if not (len(boxes) == len(scores) == len(classes)):
            raise RuntimeError("YOLOE output cardinality mismatch")

        instances = []
        for index, (box, score, class_id) in enumerate(
            zip(boxes, scores, classes), start=1
        ):
            label = str(result.names.get(int(class_id), int(class_id)))
            instances.append(
                {
                    "instance_id": index,
                    "class_id": int(class_id),
                    "class_name": label,
                    "confidence": round(float(score), 6),
                    "box_xyxy": [round(float(value), 2) for value in box],
                }
            )

        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not decode {source}")
        annotated_path = output_dir / "annotated.jpg"
        if not cv2.imwrite(str(annotated_path), render_boxes(image, instances)):
            raise RuntimeError(f"Could not write {annotated_path}")

        manifest = {
            "stage": "YOLOE-26 Prompt-Free detection",
            "source": str(source),
            "checkpoint": str(self.checkpoint),
            "text_prompt": None,
            "native_masks_forwarded": False,
            "confidence_threshold": self.conf,
            "image_size": self.imgsz,
            "max_detections": self.max_det,
            "instance_count": len(instances),
            "unique_classes": sorted({item["class_name"] for item in instances}),
            "image_shape_hwc": list(image.shape),
            "timing": {
                "model_load_seconds": round(self.load_seconds, 3),
                "inference_seconds": round(inference_seconds, 3),
                "model_reused": True,
            },
            "runtime": {
                "ultralytics": ultralytics_version,
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
            },
            "artifacts": {"annotated_image": annotated_path.name},
            "instances": instances,
        }
        (output_dir / "detections.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )
        return manifest


def main() -> None:
    args = parse_args()
    detector = YoloeDetector(
        args.checkpoint,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        max_det=args.max_det,
    )
    manifest = detector.infer(args.source, args.output_dir)
    print(f"YOLOE_OK instances={manifest['instance_count']} output={args.output_dir}")


if __name__ == "__main__":
    main()
