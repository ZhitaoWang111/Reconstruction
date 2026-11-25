"""Runs inside Blender: export the finished appearance model to USD.

    blender -b <agent run>/appearance.blend --python src/structuregen/export/blender_export.py -- <out.usda>

Same exporter settings as the white model (blender_build.py): /Structure root, one material per
face zone bound through GeomSubsets, UsdPreviewSurface, custom properties as userProperties, plus
the UV map ("st") and the tile images of sg.tiles written next to the file under textures/.
Bodies come out flat under /Structure; export.py sorts them into Walls/ and Floors/ and adds
physics and clean-up with pxr.
"""
import sys

import bpy

out = sys.argv[sys.argv.index("--") + 1]
for ob in list(bpy.data.objects):            # key-frame cameras are not part of the deliverable
    if ob.type != "MESH":
        bpy.data.objects.remove(ob)
bpy.ops.wm.usd_export(filepath=out, root_prim_path="/Structure", selected_objects_only=False,
                      export_materials=True, generate_preview_surface=True, export_cameras=False,
                      export_lights=False, export_custom_properties=True,
                      export_textures=True, export_textures_mode="NEW", overwrite_textures=True,
                      convert_world_material=False, export_uvmaps=True, xform_op_mode="MAT")
print("SG_EXPORTED", out)
