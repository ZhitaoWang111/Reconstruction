#!/usr/bin/env python3
"""Verify the local WeDetect-Anything model and output artifacts."""

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
from PIL import Image


ROOT = Path(__file__).resolve().parent
EXPECTED_CHECKPOINT_SHA256 = (
    "e595578aeeb5f2a36cca383c23279cf7ea72e0bfb7c546dc39420359bf2d2c82"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_output(source: Path, output: Path, log: Path) -> None:
    if not output.is_file() or not log.is_file():
        raise FileNotFoundError(f"Missing output or log: {output}, {log}")
    with Image.open(source) as source_image, Image.open(output) as output_image:
        if source_image.size != output_image.size:
            raise RuntimeError(
                f"Output/source size mismatch: {output_image.size} vs {source_image.size}"
            )
    log_text = log.read_text()
    if "[result]" not in log_text or "labeled ->" not in log_text:
        raise RuntimeError(f"Inference completion marker missing from {log}")


def main() -> None:
    checkpoint = ROOT / "checkpoints" / "wedetect_base_uni.pth"
    text_features = ROOT / "wedetect_anything" / "prompt_free.npy"
    onnx_path = ROOT / "wedetect_anything" / "onnx_models" / "wedetect_anything_base.onnx"
    meta_path = ROOT / "wedetect_anything" / "onnx_models" / "wedetect_anything_base_meta.json"
    output_dir = ROOT / "wedetect_anything" / "outputs"

    actual_hash = sha256(checkpoint)
    if actual_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"Checkpoint SHA-256 mismatch: {actual_hash}")

    features = np.load(text_features, mmap_mode="r")
    if features.shape != (4585, 768) or features.dtype != np.float32:
        raise RuntimeError(f"Unexpected vocabulary features: {features.shape} {features.dtype}")

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    session = onnxruntime.InferenceSession(
        str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or inputs[0].shape != [1, 3, 640, 640]:
        raise RuntimeError(f"Unexpected ONNX input signature: {[(x.name, x.shape) for x in inputs]}")
    if [item.name for item in outputs] != ["scores", "bboxes", "embeds"]:
        raise RuntimeError(f"Unexpected ONNX outputs: {[item.name for item in outputs]}")

    meta = json.loads(meta_path.read_text())
    if sum(meta["grid_size"]) != 8400 or meta["prompt_dim"] != 768:
        raise RuntimeError(f"Unexpected export metadata: {meta}")

    examples = [
        (
            ROOT / "wedetect_anything" / "examples" / "1.jpg",
            output_dir / "official_example_1.jpg",
            output_dir / "official_example_1.log",
        ),
        (
            ROOT.parent / "yoloe26_prompt_free" / "inputs" / "official_bus.jpg",
            output_dir / "official_bus.jpg",
            output_dir / "official_bus.log",
        ),
        (
            ROOT.parents[2] / "ScanNet" / "assets" / "frame-0162-rgb.webp",
            output_dir / "scannet_scene0000_00_frame0162.jpg",
            output_dir / "scannet_scene0000_00_frame0162.log",
        ),
    ]
    for source, output, log in examples:
        verify_output(source, output, log)

    print("VERIFY_OK")
    print(f"checkpoint_sha256={actual_hash}")
    print(f"vocabulary_features={features.shape}")
    print(f"onnx_input={inputs[0].name}:{inputs[0].shape}")
    print(f"onnx_outputs={','.join(item.name for item in outputs)}")
    print(f"onnx_providers={','.join(session.get_providers())}")
    print(f"verified_outputs={len(examples)}")


if __name__ == "__main__":
    main()
