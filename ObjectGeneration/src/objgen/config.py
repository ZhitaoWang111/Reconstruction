"""Paths: the single source of ROOT and of everything rooted at it.

Every module asks here instead of counting ``parents`` from its own file, so files can
move between subpackages without breaking path resolution.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]            # ObjectGeneration/
RECONSTRUCTION = ROOT.parent                          # repo root; third-party repos live here
SUBMODULE = RECONSTRUCTION / "Submodule"
SRC = ROOT / "src"
OUTPUT_ROOT = ROOT / "output"
ASSETS_ROOT = ROOT / "assets"
CONFIGS = ROOT / "configs"
PAGEGEN = ROOT / "pagegen"
SITE = ROOT / "site"
DOCS = ROOT / "docs"
SAM3D_REPO = SUBMODULE / "sam-3d-objects"
DUALCRAFT_REPO = SUBMODULE / "dual-craft"
BLENDER_MCP = SUBMODULE / "mcp-for-blender"

VENS = {"sam3d": ROOT / ".venv-sam3d", "appearance": ROOT / ".venv-appearance"}
PYTHON = VENS["sam3d"] / "bin/python"                 # numpy, trimesh, pxr, torch
VENV_PYTHON = PYTHON
APPEARANCE_PYTHON = VENS["appearance"] / "bin/python"  # mcp, yaml

# Sibling modules of this repository, resolved from the parent of ObjectGeneration/.
SAM31 = RECONSTRUCTION / "InstanceRecognition"
ROBOT_SIM = RECONSTRUCTION.parent / "robot_sim"
ISAAC_ENV = ROBOT_SIM / "scripts/activate_env.sh"
SAM31_PYTHON = SAM31 / ".venv/bin/python"
SAM31_REPO = RECONSTRUCTION / "Submodule/SAM3"   # moved out of InstanceRecognition/ThirdParty
SAM31_CHECKPOINT = RECONSTRUCTION / "weights/sam3.1_multiplex.pt"

SAM3D_CONFIG = ROOT / ".runtime_config/sam3d/pipeline.yaml"
APPEARANCE_CONFIG = CONFIGS / "appearance/default.yaml"
PIPELINE_ENV = ["PYTHONPATH=" + str(SRC)]
