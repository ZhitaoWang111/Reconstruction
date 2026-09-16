"""Filesystem and runtime configuration for the local instance-recognition app."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT.parent
THIRD_PARTY = ROOT / "ThirdParty"

SCANNET_ROOT = Path(
    os.environ.get(
        "SCANNET_ROOT", "/home/wzt/HDD/datasets/scannet_v2/scans"
    )
).expanduser()
RUNTIME_ROOT = Path(
    os.environ.get("INSTANCE_RECOGNITION_RUNTIME", ROOT / "runtime")
).expanduser()

YOLOE_PYTHON = Path(
    os.environ.get(
        "YOLOE_PYTHON", THIRD_PARTY / "yoloe26_prompt_free/.venv/bin/python"
    )
).expanduser()
SAM31_PYTHON = Path(
    os.environ.get("SAM31_PYTHON", "/home/wzt/.conda/envs/scene/bin/python")
).expanduser()

YOLOE_CHECKPOINT = Path(
    os.environ.get(
        "YOLOE_CHECKPOINT",
        THIRD_PARTY / "yoloe26_prompt_free/weights/yoloe-26s-seg-pf.pt",
    )
).expanduser()
SAM31_CHECKPOINT = Path(
    os.environ.get("SAM31_CHECKPOINT", WORKSPACE / "weights/sam3.1_multiplex.pt")
).expanduser()
SAM3_REPO = THIRD_PARTY / "SAM3"

DEFAULT_CONFIDENCE = 0.25
DEFAULT_IMAGE_SIZE = 640
DEFAULT_MAX_DETECTIONS = 16
