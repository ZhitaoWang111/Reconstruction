"""Step 4, the structure handoff for scene assembly.

    python -m structuregen.export.handoff [agent run] [--frame amb3r_floor|scannet_scan] [--out <dir>]

handoff.py exports appearance.blend through blender_export.py, sorts it into /Structure/Walls and
/Structure/Floors, adds static colliders, reads it back and checks it, then writes structure.json,
manifest.json and README.md. blender_dump.py reads a model's triangles and materials (page, amend).
"""
