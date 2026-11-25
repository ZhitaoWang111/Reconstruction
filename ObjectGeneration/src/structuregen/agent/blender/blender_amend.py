"""Runs inside Blender: apply hand edits to a finished appearance model.

    blender -b <appearance.blend> --python src/structuregen/agent/blender/blender_amend.py -- <edits.json> <out.blend>

edits.json is a list of {"seg", "face", "rgb", "roughness"}; each goes through the same sg.paint
operator the agent uses, with item "manual", so the material records that a person set it.
"""
import json
from pathlib import Path
import sys

import bpy

edits_path, out = sys.argv[sys.argv.index("--") + 1:][:2]
here = Path(__file__).resolve().parent
ns = {"__name__": "sg_amend"}
exec(compile((here / "sg_ops.py").read_text(), str(here / "sg_ops.py"), "exec"), ns)
ns["register"]()
for e in json.loads(Path(edits_path).read_text()):
    bpy.ops.sg.paint(seg=e["seg"], face=e["face"], rgb=tuple(e["rgb"]), roughness=e.get("roughness", 0.9), item="manual")
bpy.ops.wm.save_as_mainfile(filepath=out, copy=True)
print("SG_AMENDED", out)
