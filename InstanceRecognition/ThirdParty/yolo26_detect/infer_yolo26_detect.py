#!/usr/bin/env python3
"""Run closed-set YOLO26 object detection on one local image."""

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator


ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = ROOT / ".runtime_config"
TORCH_KERNEL_CACHE = ROOT / ".cache" / "torch" / "kernels"
(CONFIG_ROOT / "Ultralytics").mkdir(parents=True, exist_ok=True)
TORCH_KERNEL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(CONFIG_ROOT))
os.environ.setdefault("PYTORCH_KERNEL_CACHE_PATH", str(TORCH_KERNEL_CACHE))

import cv2  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO, __version__ as ultralytics_version  # noqa: E402


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Image-only YOLO26 closed-set object detection. "
            "Writes COCO labels, bounding boxes and an annotated image."
        )
    )
    parser.add_argument("--source", required=True, type=Path, help="Input image path")
    parser.add_argument(
        "--model",
        default="yolo26s.pt",
        help="YOLO26 detection checkpoint name or local path",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "latest")
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def load_model(model_arg: str) -> tuple:
    """Load a local checkpoint, or let Ultralytics download it into weights/."""
    candidate = Path(model_arg).expanduser()
    if candidate.is_file():
        return YOLO(str(candidate.resolve())), candidate.resolve()

    cached = ROOT / "weights" / candidate.name
    if cached.is_file():
        return YOLO(str(cached)), cached

    (ROOT / "weights").mkdir(parents=True, exist_ok=True)
    with working_directory(ROOT / "weights"):
        model = YOLO(candidate.name)
    if not cached.is_file():
        raise FileNotFoundError(f"Checkpoint download did not create {cached}")
    return model, cached


def class_name(names: Dict[int, str], class_id: int) -> str:
    return str(names.get(class_id, names.get(str(class_id), class_id)))


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input image not found: {source}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model(args.model)
    results = model.predict(
        source=str(source),
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        verbose=True,
    )
    if len(results) != 1:
        raise RuntimeError(f"Expected one image result, got {len(results)}")

    result = results[0]
    if result.boxes is None:
        raise RuntimeError("YOLO26 Detect did not return bounding boxes")
    if result.masks is not None:
        raise RuntimeError("Detection checkpoint unexpectedly returned masks")

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    count = len(boxes)
    if count == 0:
        raise RuntimeError("Inference completed but returned zero detections")
    if not (count == len(scores) == len(classes)):
        raise RuntimeError(
            "Output cardinality mismatch: "
            f"boxes={count}, scores={len(scores)}, classes={len(classes)}"
        )

    annotated_path = output_dir / "annotated.jpg"
    if not cv2.imwrite(str(annotated_path), result.plot()):
        raise RuntimeError(f"Failed to save {annotated_path}")

    detections = []
    labels = []
    for index, (box, score, class_id) in enumerate(
        zip(boxes, scores, classes), start=1
    ):
        label = class_name(result.names, int(class_id))
        labels.append(label)
        detections.append(
            {
                "detection_id": index,
                "class_id": int(class_id),
                "class_name": label,
                "confidence": round(float(score), 6),
                "box_xyxy": [round(float(value), 2) for value in box],
            }
        )

    manifest = {
        "task": "YOLO26 closed-set image-only object detection",
        "user_text_prompt": None,
        "class_vocabulary": "COCO 80 classes from the pretrained checkpoint",
        "output_type": "class labels and bounding boxes; no masks",
        "source": str(source),
        "checkpoint": str(checkpoint),
        "ultralytics_version": ultralytics_version,
        "torch_version": torch.__version__,
        "device_requested": args.device,
        "device_used": (
            f"cuda:{args.device}" if args.device != "cpu" and torch.cuda.is_available() else "cpu"
        ),
        "imgsz": args.imgsz,
        "confidence_threshold": args.conf,
        "result_image_shape_hwc": list(result.orig_img.shape),
        "detection_count": count,
        "unique_classes": sorted(set(labels)),
        "timing_ms": {key: round(float(value), 3) for key, value in result.speed.items()},
        "artifacts": {"annotated_image": annotated_path.name},
        "detections": detections,
    }
    manifest_path = output_dir / "predictions.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    print("\nCHECK_OK")
    print(f"detections={count}")
    print(f"unique_classes={', '.join(manifest['unique_classes'])}")
    print(f"boxes={len(boxes)} masks=0 labels={len(labels)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
