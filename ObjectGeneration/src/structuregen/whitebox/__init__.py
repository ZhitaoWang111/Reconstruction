"""Step 1, the white model.

    python -m structuregen.whitebox.build --handoff <walls handoff> [--opening-inset 0.15] [--floor-thickness 0.05]
    python -m structuregen.whitebox.validate <run dir>

segments.py plans the wall segments (pure data, no Blender); build.py runs blender_build.py to
make walls, floors and cameras, checks closedness and volumes, renders the key-frame views;
validate.py reads the USD back with pxr.
"""
