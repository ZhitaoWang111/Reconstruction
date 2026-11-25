"""Runs inside Blender: render the white model from the handoff's key-frame cameras.

    blender -b <run>/walls.blend --python src/structuregen/whitebox/blender_views.py -- <out dir> <width> <height>

Key-frame views are rendered at the photo's own size (the cameras' lens and shift were
set for it), on a transparent background, so a page can lay them over the photos.
An overview of the whole room is rendered too.
"""
import sys
from pathlib import Path

import bpy

args = sys.argv[sys.argv.index("--") + 1:]
OUT = Path(args[0])
WIDTH, HEIGHT = int(args[1]), int(args[2])
OUT.mkdir(parents=True, exist_ok=True)
scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
shading = scene.display.shading
shading.light, shading.color_type = "STUDIO", "MATERIAL"
shading.show_shadows = False
shading.show_cavity = True
shading.cavity_type = "BOTH"
scene.display.render_aa = "8"
scene.view_settings.view_transform, scene.view_settings.look = "Standard", "None"
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.film_transparent = True

count = 0
scene.render.resolution_x, scene.render.resolution_y = WIDTH, HEIGHT
for ob in sorted((o for o in bpy.data.objects if o.type == "CAMERA"), key=lambda o: o.name):
    scene.camera = ob
    scene.render.filepath = str(OUT / f"{ob.name}.png")
    bpy.ops.render.render(write_still=True)
    count += 1

scene.render.resolution_x, scene.render.resolution_y = 1200, 800
scene.render.film_transparent = False
cam = bpy.data.cameras.new("overview")
cam.lens = 26
ob = bpy.data.objects.new("overview", cam)
scene.collection.objects.link(ob)
ob.location = (5.5, -9.5, 8.5)
ob.rotation_euler = (0.82, 0.0, 0.42)
scene.camera = ob
scene.render.filepath = str(OUT / "overview.png")
bpy.ops.render.render(write_still=True)
print("RENDERED", count + 1, "views to", OUT)
