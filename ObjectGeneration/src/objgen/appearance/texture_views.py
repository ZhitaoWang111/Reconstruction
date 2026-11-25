"""Executed inside the texture template with its canonical object and material."""
from mathutils import Matrix
scene.render.engine='BLENDER_EEVEE_NEXT'
scene.render.resolution_x=512; scene.render.resolution_y=768
scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG'
scene.view_settings.view_transform='Standard'
scene.world.color=(.18,.18,.18)
bpy.ops.object.camera_add()
view=bpy.context.object; scene.camera=view; view.data.type='ORTHO'; view.data.clip_start=.001
# Match the fitted photo projection exactly, including off-centre framing.
view.location=Vector(camera['center'])+toward*height*3
view.rotation_euler=Matrix(np.array(camera['basis']).T.tolist()).to_euler()
view.data.ortho_scale=camera['height_m']
scene.render.resolution_x=512; scene.render.resolution_y=round(512/camera['aspect'])
view.data.shift_x=(.5-camera['offset'][0])*camera['aspect']
view.data.shift_y=.5-camera['offset'][1]
# Emission comparisons isolate the sampled base colour from illumination.
for kind in ('baseline','refined'):
    nodes.clear(); out=nodes.new('ShaderNodeOutputMaterial'); emit=nodes.new('ShaderNodeEmission')
    if kind=='baseline':
        color=nodes.new('ShaderNodeVertexColor'); color.layer_name='SourceColor'
    else:
        color=nodes.new('ShaderNodeTexImage'); color.image=target
        coord=nodes.new('ShaderNodeUVMap'); coord.uv_map='st'; links.new(coord.outputs[0],color.inputs['Vector'])
    links.new(color.outputs['Color'],emit.inputs[0]); links.new(emit.outputs[0],out.inputs['Surface'])
    scene.render.filepath=str(folder/'renders'/f'aligned_{kind}.png'); bpy.ops.render.render(write_still=True)
nodes.clear(); out=nodes.new('ShaderNodeOutputMaterial'); bsdf=nodes.new('ShaderNodeBsdfPrincipled'); bsdf.inputs['Roughness'].default_value=.45
tex=nodes.new('ShaderNodeTexImage'); tex.image=target
coord=nodes.new('ShaderNodeUVMap'); coord.uv_map='st'; links.new(coord.outputs[0],tex.inputs['Vector']); links.new(tex.outputs['Color'],bsdf.inputs['Base Color']); links.new(bsdf.outputs[0],out.inputs['Surface'])
bpy.ops.object.light_add(type='AREA'); light=bpy.context.object; light.data.energy=400*height*height; light.data.size=height*2
scene.render.resolution_x=scene.render.resolution_y=512
view.data.shift_x=view.data.shift_y=0; view.data.ortho_scale=height*1.25
for name,direction in {'front':(0,-1,0),'back':(0,1,0),'left':(-1,0,0),'right':(1,0,0),'top':(0,0,1),'three_quarter':(1,-1,.6)}.items():
    view.location=Vector(center)+Vector(direction).normalized()*height*3
    view.rotation_euler=(Vector(center)-view.location).to_track_quat('-Z','Y').to_euler()
    light.location=view.location+Vector((0,0,height*1.5)); light.rotation_euler=(Vector(center)-light.location).to_track_quat('-Z','Y').to_euler()
    scene.render.filepath=str(folder/'renders'/f'refined_{name}.png'); bpy.ops.render.render(write_still=True)
