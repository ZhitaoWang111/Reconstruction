"""Bake photo projection with ray visibility onto a geometry-preserving atlas."""
import json
from pathlib import Path
import sys
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

folder = Path(sys.argv[sys.argv.index('--')+1])
camera = json.loads((folder/'camera.json').read_text())
config = json.loads((folder/'config.json').read_text())
arrays = np.load(folder/'canonical_mesh.npz')
points, faces = arrays['points'], arrays['indices'].reshape(-1,3)
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
mesh = bpy.data.meshes.new('CanonicalMesh')
mesh.from_pydata(points.tolist(), [], faces.tolist()); mesh.update()
obj = bpy.data.objects.new('CanonicalObject',mesh); bpy.context.collection.objects.link(obj)
bpy.context.view_layer.objects.active=obj; obj.select_set(True)
assert np.array_equal(np.array([v.co[:] for v in mesh.vertices], dtype='<f4'), points)
assert np.array_equal(np.array([p.vertices[:] for p in mesh.polygons], dtype='<i4'), faces)
height=float(np.ptp(points[:,2])); center=(points.min(axis=0)+points.max(axis=0))/2
normals=np.cross(points[faces[:,1]]-points[faces[:,0]],points[faces[:,2]]-points[faces[:,0]])
normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-20)
# Separate the visible photo chart from the fallback cylindrical chart.
xyz=points[faces]
angle=np.arctan2(xyz[:,:,1]-center[1],xyz[:,:,0]-center[0])/(2*np.pi)+.5
seam=np.ptp(angle,axis=1)>.5
angle[seam]=np.where(angle[seam]<.5,angle[seam]+1,angle[seam])
uv=np.stack([.75+.14*angle,.02+.96*(xyz[:,:,2]-points[:,2].min())/height],axis=-1)
atlas=mesh.uv_layers.new(name='st')
atlas.data.foreach_set('uv',uv.astype(np.float32).ravel())
B=np.array(camera['basis']); toward=Vector(B[2]); q=(points-np.array(camera['center']))@B.T
proj=q[:,:2]/camera['height_m']; proj[:,0]/=camera['aspect']; proj+=camera['offset']
projection=mesh.uv_layers.new(name='PhotoProjection')
projection.data.foreach_set('uv',proj[faces].astype(np.float32).ravel())
mesh.uv_layers.active=atlas; atlas.active_render=True
bvh=BVHTree.FromPolygons([Vector(p) for p in points],faces.tolist(),all_triangles=True)
mask_image=bpy.data.images.load(str(folder/'reference_mask.png'))
mask=np.array(mask_image.pixels[:],dtype=np.float32).reshape(mask_image.size[1],mask_image.size[0],4)[::-1,:,0]>.5
# Mask erosion keeps background out of boundary texels.
for _ in range(3):
    mask[1:-1,1:-1] &= mask[:-2,1:-1]&mask[2:,1:-1]&mask[1:-1,:-2]&mask[1:-1,2:]
weights=np.zeros(len(points),np.float32)
for i,p in enumerate(points):
    u,v=proj[i]; x=int(u*(mask.shape[1]-1)); y=int((1-v)*(mask.shape[0]-1))
    if not (0<=u<=1 and 0<=v<=1 and mask[y,x]): continue
    n=Vector(mesh.vertices[i].normal); facing=n.dot(toward)
    if facing<=.15: continue
    hit=bvh.ray_cast(Vector(p)+toward*height*1e-4,toward,height*3)[0]
    if hit is None: weights[i]=min(1.,max(0.,(facing-.15)/.20))
# All observed faces share the same photo coordinates, so overlapping projections
# sample identical colours instead of competing with the unseen vertex-colour bake.
observed=(weights[faces]>0).any(axis=1)
uv[observed,:,0]=.02+.65*proj[faces[observed],0]
uv[observed,:,1]=.02+.96*proj[faces[observed],1]
assert np.isfinite(uv).all() and uv.min()>=0 and uv.max()<=1
atlas.data.foreach_set('uv',uv.astype(np.float32).ravel())
def attr(name,values):
    a=mesh.color_attributes.new(name=name,type='FLOAT_COLOR',domain='POINT')
    a.data.foreach_set('color',np.column_stack([values,np.ones(len(points))]).astype(np.float32).ravel())
attr('SourceColor',arrays['colors']); attr('PhotoWeight',np.repeat(weights[:,None],3,axis=1))
material=bpy.data.materials.new('PhotoGrounded'); material.use_nodes=True
nodes=material.node_tree.nodes; nodes.clear(); links=material.node_tree.links
output=nodes.new('ShaderNodeOutputMaterial'); emission=nodes.new('ShaderNodeEmission')
links.new(emission.outputs[0],output.inputs['Surface'])
source=nodes.new('ShaderNodeVertexColor'); source.layer_name='SourceColor'
weight=nodes.new('ShaderNodeVertexColor'); weight.layer_name='PhotoWeight'
photo=nodes.new('ShaderNodeTexImage'); photo.image=bpy.data.images.load(str(folder/'reference_crop.png')); photo.extension='CLIP'
coords=nodes.new('ShaderNodeUVMap'); coords.uv_map='PhotoProjection'; links.new(coords.outputs[0],photo.inputs['Vector'])
mix=nodes.new('ShaderNodeMixRGB'); links.new(weight.outputs['Color'],mix.inputs[0]); links.new(source.outputs['Color'],mix.inputs[1]); links.new(photo.outputs['Color'],mix.inputs[2]); links.new(mix.outputs[0],emission.inputs[0])
size=config['texture_size']; target=bpy.data.images.new('BaseColor',width=size,height=size,alpha=False)
target.colorspace_settings.name='sRGB'; bake=nodes.new('ShaderNodeTexImage'); bake.image=target; nodes.active=bake
obj.data.materials.append(material)
scene=bpy.context.scene; scene.render.engine='CYCLES'; scene.cycles.device='CPU'; scene.cycles.samples=1
scene.render.bake.margin=8; scene.render.bake.use_clear=True
bpy.ops.object.bake(type='EMIT')
target.filepath_raw=str(folder/'textures/basecolor.png'); target.file_format='PNG'; target.save()
links.new(weight.outputs['Color'],emission.inputs[0])
coverage=bpy.data.images.new('PhotoCoverage',width=size,height=size,alpha=False); coverage.colorspace_settings.name='Non-Color'; bake.image=coverage
bpy.ops.object.bake(type='EMIT'); coverage.filepath_raw=str(folder/'textures/photo_coverage.png'); coverage.file_format='PNG'; coverage.save()
nodes.clear(); output=nodes.new('ShaderNodeOutputMaterial'); bsdf=nodes.new('ShaderNodeBsdfPrincipled'); bsdf.inputs['Roughness'].default_value=.45
tex=nodes.new('ShaderNodeTexImage'); tex.image=target
coords=nodes.new('ShaderNodeUVMap'); coords.uv_map='st'; links.new(coords.outputs[0],tex.inputs['Vector']); links.new(tex.outputs['Color'],bsdf.inputs['Base Color']); links.new(bsdf.outputs[0],output.inputs['Surface'])
np.save(folder/'uv.npy',uv.astype(np.float32))
np.save(folder/'observed_faces.npy',observed)
fallback=bpy.data.materials.new('UnobservedOriginal'); fallback.use_nodes=True
fc=fallback.node_tree.nodes.new('ShaderNodeVertexColor'); fc.layer_name='SourceColor'
fb=fallback.node_tree.nodes.get('Principled BSDF'); fb.inputs['Roughness'].default_value=.45
fallback.node_tree.links.new(fc.outputs['Color'],fb.inputs['Base Color'])
mesh.materials.append(fallback)
for face,seen in zip(mesh.polygons,observed): face.material_index=0 if seen else 1
exec(compile((Path(__file__).parent/'texture_views.py').read_text(), 'texture_views.py', 'exec'))
bpy.ops.wm.save_as_mainfile(filepath=str(folder/'work.blend'))
(folder/'projection.json').write_text(json.dumps({'photo_weighted_vertices':int((weights>0).sum()),'vertices':len(points),'unobserved_policy':'original vertex colors','seam_crossing_faces':int(seam.sum()), 'geometry_preserved':True, 'limitations':['Photo lighting is retained', 'Unobserved cylindrical chart may stretch near caps; inspect side views']},indent=2))
print('APPEARANCE_TEXTURE_BAKED',flush=True)
