"""Deterministic six-view canonical-geometry baseline; no API or MCP required."""
import json
from pathlib import Path
import sys
import bpy
import numpy as np
from mathutils import Vector

folder = Path(sys.argv[sys.argv.index('--') + 1])
arrays = np.load(folder / 'canonical_mesh.npz')
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
mesh = bpy.data.meshes.new('CanonicalMesh')
mesh.from_pydata(arrays['points'].tolist(), [], arrays['indices'].reshape(-1, 3).tolist())
mesh.update()
obj = bpy.data.objects.new('CanonicalObject', mesh)
bpy.context.collection.objects.link(obj)
color = mesh.color_attributes.new(name='SourceColor', type='FLOAT_COLOR', domain='POINT')
rgba = np.column_stack((arrays['colors'], np.ones(len(arrays['points'])))).astype(np.float32)
color.data.foreach_set('color', rgba.ravel())
assert np.array_equal(np.array([v.co[:] for v in mesh.vertices], dtype='<f4'), arrays['points'])
assert np.array_equal(np.array([p.vertices[:] for p in mesh.polygons], dtype='<i4').ravel(), arrays['indices'])
material = bpy.data.materials.new('BaselineVertexColor')
material.use_nodes = True
bsdf = material.node_tree.nodes.get('Principled BSDF')
bsdf.inputs['Roughness'].default_value = .45
vertex_color = material.node_tree.nodes.new('ShaderNodeVertexColor')
vertex_color.layer_name = color.name
material.node_tree.links.new(vertex_color.outputs['Color'], bsdf.inputs['Base Color'])
obj.data.materials.append(material)
center = Vector(tuple((arrays['points'].min(axis=0) + arrays['points'].max(axis=0)) / 2))
size = float(np.ptp(arrays['points'], axis=0).max())
bpy.ops.object.camera_add()
camera = bpy.context.object
camera.data.type = 'ORTHO'
camera.data.ortho_scale = size * 1.25
camera.data.clip_start = .001
bpy.context.scene.camera = camera
bpy.ops.object.light_add(type='AREA')
light = bpy.context.object
light.data.energy = 400 * size * size
light.data.size = size * 2
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE_NEXT'
scene.render.resolution_x = scene.render.resolution_y = 512
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.world.color = (.15, .15, .15)
views = {'front': (0, -1, 0), 'back': (0, 1, 0), 'left': (-1, 0, 0),
         'right': (1, 0, 0), 'top': (0, 0, 1), 'three_quarter': (1, -1, .6)}
cameras = {}
for name, direction in views.items():
    camera.location = center + Vector(direction).normalized() * size * 3
    camera.rotation_euler = (center - camera.location).to_track_quat('-Z', 'Y').to_euler()
    light.location = camera.location + Vector((0, 0, size * 1.5))
    light.rotation_euler = (center - light.location).to_track_quat('-Z', 'Y').to_euler()
    scene.render.filepath = str(folder / 'renders' / (name + '.png'))
    bpy.ops.render.render(write_still=True)
    cameras[name] = {'position': list(camera.location), 'rotation_euler': list(camera.rotation_euler),
                     'ortho_scale': camera.data.ortho_scale}
(folder / 'baseline_cameras.json').write_text(json.dumps(cameras, indent=2) + '\n')
bpy.ops.wm.save_as_mainfile(filepath=str(folder / 'work.blend'))
print('APPEARANCE_BASELINE_RENDERED', bpy.app.version_string, flush=True)
