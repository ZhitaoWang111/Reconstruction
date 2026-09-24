"""Filesystem and runtime configuration for the local instance-recognition app."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT.parent
THIRD_PARTY = ROOT / "ThirdParty"
WEIGHTS_ROOT = WORKSPACE / "weights"
LOCAL_PYTHON = ROOT / ".venv/bin/python"
DEFAULT_PYTHON = LOCAL_PYTHON if LOCAL_PYTHON.is_file() else Path(sys.executable)
LEGACY_YOLOE_PYTHON = THIRD_PARTY / "yoloe26_prompt_free/.venv/bin/python"

SCANNET_ROOT = Path(
    os.environ.get(
        "SCANNET_ROOT", WORKSPACE.parent / "data/scannet_v2"
    )
).expanduser()
RUNTIME_ROOT = Path(
    os.environ.get("INSTANCE_RECOGNITION_RUNTIME", ROOT / "runtime")
).expanduser()

YOLOE_PYTHON = Path(
    os.environ.get(
        "YOLOE_PYTHON",
        LEGACY_YOLOE_PYTHON if LEGACY_YOLOE_PYTHON.is_file() else DEFAULT_PYTHON,
    )
).expanduser()
SAM31_PYTHON = Path(
    os.environ.get("SAM31_PYTHON", DEFAULT_PYTHON)
).expanduser()

YOLOE_CHECKPOINT = Path(
    os.environ.get(
        "YOLOE_CHECKPOINT",
        WEIGHTS_ROOT / "yoloe26_prompt_free/weights/yoloe-26s-seg-pf.pt",
    )
).expanduser()
SAM31_CHECKPOINT = Path(
    os.environ.get("SAM31_CHECKPOINT", WEIGHTS_ROOT / "sam3.1_multiplex.pt")
).expanduser()
SAM3_REPO = THIRD_PARTY / "SAM3"

DEFAULT_CONFIDENCE = 0.25
DEFAULT_IMAGE_SIZE = 640
DEFAULT_MAX_DETECTIONS = 16

# GPT text prompts are the primary method; legacy YOLOE remains opt-in.
DEFAULT_METHOD = os.environ.get("INSTANCE_RECOGNITION_METHOD", "gpt_sam31")
GPT_MODEL = os.environ.get("INSTANCE_GPT_MODEL", "openai/gpt-5.6-terra")
GPT_API_DIR = Path(os.environ.get("INSTANCE_GPT_API_DIR", WORKSPACE.parent / "API/scannet_gpt_sam"))
GPT_SAM_THRESHOLD = 0.5
