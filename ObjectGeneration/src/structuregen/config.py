"""Paths for structure generation."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]                    # ObjectGeneration/
SRC = ROOT / "src"
OUTPUT_ROOT = ROOT / "output"
BLENDER = Path("/home/seer/wzt/Blender/.local/opt/blender/blender")
RECONSTRUCTION = ROOT.parent
SUBMODULE = RECONSTRUCTION / "Submodule"                  # third-party repos, one copy for the whole repo
BLENDER_MCP = SUBMODULE / "mcp-for-blender"
HANDOFF_ROOT = RECONSTRUCTION / "InstanceRecognition/artifacts/handoff"
SCENE = "scene0000_00"
DEFAULT_HANDOFF = HANDOFF_ROOT / SCENE / "walls_harness"

# One colour per segment, in handoff order (sRGB). The plan view, the page's 3D viewer and the
# segment table all use these, so a segment has the same colour everywhere.
SEGMENT_COLOURS = [(214, 92, 68), (222, 158, 61), (186, 186, 74), (104, 168, 96), (74, 158, 158),
                   (78, 122, 190), (124, 104, 186), (176, 96, 152), (128, 128, 132)]
