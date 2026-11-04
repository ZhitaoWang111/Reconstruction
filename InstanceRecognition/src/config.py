"""Filesystem and runtime configuration for the instance-recognition pipeline.

This tree keeps four things apart, and the path constants below are the single
place that decides where each one lives:

  1. functional code   -> ``src/``
  2. page build code   -> ``pagegen/`` (with its templates under ``pagegen/templates/``)
  3. pipeline output   -> ``ARTIFACTS_ROOT`` (``artifacts/``)
  4. generated pages   -> ``SITE_ROOT`` (``site/``)

Both output roots are independently redirectable by environment variable, so a
run can publish its pages somewhere other than next to its data. Nothing else in
the tree should build an output path by hand.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT.parent
WEIGHTS_ROOT = WORKSPACE / "weights"
LOCAL_PYTHON = ROOT / ".venv/bin/python"
DEFAULT_PYTHON = LOCAL_PYTHON if LOCAL_PYTHON.is_file() else Path(sys.executable)

SCANNET_ROOT = Path(
    os.environ.get(
        "SCANNET_ROOT", WORKSPACE.parent / "data/scannet_v2"
    )
).expanduser()

# Pipeline output: point-cloud sources, frames, geometry caches and per-run
# records (calls, versions, evidence). This is the tree that gets archived.
ARTIFACTS_ROOT = Path(
    os.environ.get("IR_ARTIFACTS_ROOT", ROOT / "artifacts")
).expanduser()

# Generated pages and the display data they load (states, objects, manifest.js).
# Kept apart from ARTIFACTS_ROOT so an experiment's records can be archived or
# deleted without touching its published pages, and vice versa.
SITE_ROOT = Path(
    os.environ.get("IR_SITE_ROOT", ROOT / "site")
).expanduser()

# Torch compiles kernels into a cache on first use. It belongs to the artifacts
# tree rather than to either venue's runtime directory, so removing a run's
# output cannot invalidate the other tree's cache.
TORCH_KERNEL_CACHE = Path(
    os.environ.get("IR_TORCH_KERNEL_CACHE", ARTIFACTS_ROOT / "cache/torch")
).expanduser()

# Page templates live with the code that reads them, not with the output.
WEB_ROOT = ROOT / "pagegen/templates"

# Cross-experiment navigation is shared, hand-written source that lives outside
# this tree. The pages link to it relatively; see pagegen/build_records.py.
NAV_ROOT = Path(
    os.environ.get("IR_NAV_ROOT", WORKSPACE / "docs")
).expanduser()

# Activate scene-understanding for SAM3.1, GPT and Harness. The conda environment
# sets SAM31_PYTHON; otherwise use the active interpreter, never the old tree.
# The JPEG/text inference path requires no OpenCV.
SAM31_PYTHON = Path(os.environ.get("SAM31_PYTHON", sys.executable)).expanduser()

SAM31_CHECKPOINT = Path(
    os.environ.get("SAM31_CHECKPOINT", WEIGHTS_ROOT / "sam3.1_multiplex.pt")
).expanduser()

# The gateway retired gpt-5.6-terra; gpt-6-sol matches the Harness default so both
# stages of an experiment ask the same model family to read a frame.
GPT_MODEL = os.environ.get("INSTANCE_GPT_MODEL", "openai/gpt-6-sol")
# Shared by perception and Harness; neither imports the other to read credentials.
GATEWAY_ENV_FILE = Path(os.environ.get("IR_GATEWAY_ENV_FILE", WORKSPACE.parent / "API/.env")).expanduser()
# The prompt is versioned with the code that sends it, so a prompt edit and the
# code that depends on its output schema move together.
PROMPT_ROOT = ROOT / "src/inference/prompts"
