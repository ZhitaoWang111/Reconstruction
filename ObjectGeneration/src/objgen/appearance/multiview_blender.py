"""Deterministic cumulative photo charts, selected by visibility and incidence."""
import json
from pathlib import Path
import sys
import bpy
import numpy as np
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
folder=Path(sys.argv[sys.argv.index('--')+1]); config=json.loads((folder/'config.json').read_text())
views=json.loads((folder/'views.json').read_text()); arrays=np.load(folder/'canonical_mesh.npz')
points=arrays['points']; faces=arrays['indices'].reshape(-1,3)
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
mesh=bpy.data.meshes.new('CanonicalMesh'); mesh.from_pydata(points.tolist(),[],faces.tolist());mesh.update()
obj=bpy.data.objects.new('CanonicalObject',mesh);bpy.context.collection.objects.link(obj)
bpy.context.view_layer.objects.active=obj;obj.select_set(True)
height=float(np.ptp(points[:,2]));center=(points.min(axis=0)+points.max(axis=0))/2
attr=mesh.color_attributes.new(name='SourceColor',type='FLOAT_COLOR',domain='POINT')
attr.data.foreach_set('color',np.column_stack([arrays['colors'],np.ones(len(points))]).astype(np.float32).ravel())
bvh=BVHTree.FromPolygons([Vector(p) for p in points],faces.tolist(),all_triangles=True)
centroids=points[faces].mean(axis=1)
normals=np.cross(points[faces[:,1]]-points[faces[:,0]],points[faces[:,2]]-points[faces[:,0]])
normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-20)
best=np.zeros(len(faces));assignment=np.full(len(faces),-1,dtype=np.int16);uv=np.zeros((len(faces),3,2),np.float32)
# Each source photo has a disjoint, padded chart; unseen faces retain source colour.
for index,view in enumerate(views):
    ref=folder/view['folder'];cam=json.loads((ref/'camera.json').read_text());B=np.array(cam['basis']);toward=Vector(B[2])
    proj=(points-np.array(cam['center']))@B.T
    proj=proj[:,:2]/cam['height_m'];proj[:,0]/=cam['aspect'];proj+=cam['offset'];tri=proj[faces]
    mask_image=bpy.data.images.load(str(ref/'reference_mask.png'))
    mask=np.array(mask_image.pixels[:]).reshape(mask_image.size[1],mask_image.size[0],4)[:,:,0]>.5
    for _ in range(3): mask[1:-1,1:-1]&=mask[:-2,1:-1]&mask[2:,1:-1]&mask[1:-1,:-2]&mask[1:-1,2:]
    valid=((tri>=0)&(tri<=1)).all(axis=(1,2));xy=np.rint(tri*np.array([mask.shape[1]-1,mask.shape[0]-1])).astype(int)
    xy[:,:,0]=xy[:,:,0].clip(0,mask.shape[1]-1);xy[:,:,1]=xy[:,:,1].clip(0,mask.shape[0]-1)
    valid &= mask[xy[:,:,1],xy[:,:,0]].all(axis=1)
    confidence=normals@B[2];valid &= (confidence>.25)&(confidence>best+.025)
    accepted=[]
    for i in np.flatnonzero(valid):
        hit=bvh.ray_cast(Vector(centroids[i])+toward*height*1e-4,toward,height*3)[0]
        if hit is None: accepted.append(i)
    accepted=np.asarray(accepted,dtype=int);best[accepted]=confidence[accepted];assignment[accepted]=index
    # 3x2 atlas supports all four side views plus top/bottom, in requested order.
    uv[accepted,:,0]=(index%3+.02+.96*tri[accepted,:,0])/3
    uv[accepted,:,1]=(1-index//3+.02+.96*tri[accepted,:,1])/2
    print('MULTIVIEW_ACCUMULATED',view['angle'],len(accepted),flush=True)
observed=assignment>=0
atlas=mesh.uv_layers.new(name='st');atlas.data.foreach_set('uv',uv.ravel())
material=bpy.data.materials.new('PhotoGrounded');material.use_nodes=True
nodes=material.node_tree.nodes;nodes.clear();links=material.node_tree.links
output=nodes.new('ShaderNodeOutputMaterial');bsdf=nodes.new('ShaderNodeBsdfPrincipled');bsdf.inputs['Roughness'].default_value=.45
tex=nodes.new('ShaderNodeTexImage');target=bpy.data.images.load(str(folder/'textures/basecolor.png'));tex.image=target;tex.extension='EXTEND'
coords=nodes.new('ShaderNodeUVMap');coords.uv_map='st';links.new(coords.outputs[0],tex.inputs['Vector']);links.new(tex.outputs['Color'],bsdf.inputs['Base Color']);links.new(bsdf.outputs[0],output.inputs['Surface'])
mesh.materials.append(material)
fallback=bpy.data.materials.new('UnobservedOriginal');fallback.use_nodes=True
fc=fallback.node_tree.nodes.new('ShaderNodeVertexColor');fc.layer_name='SourceColor';fb=fallback.node_tree.nodes.get('Principled BSDF');fb.inputs['Roughness'].default_value=.45;fallback.node_tree.links.new(fc.outputs['Color'],fb.inputs['Base Color']);mesh.materials.append(fallback)
for face,seen in zip(mesh.polygons,observed):face.material_index=0 if seen else 1
np.save(folder/'uv.npy',uv);np.save(folder/'observed_faces.npy',observed);np.save(folder/'face_sources.npy',assignment)
scene=bpy.context.scene
camera=json.loads((folder/views[0]['folder']/'camera.json').read_text());toward=Vector(camera['basis'][2])
exec(compile((Path(__file__).parent/'texture_views.py').read_text(),'texture_views.py','exec'))
# Render all reference directions after every cumulative update.
scene.render.resolution_x=512
for v in views:
    camera=json.loads((folder/v['folder']/'camera.json').read_text());toward=Vector(camera['basis'][2])
    view.location=Vector(camera['center'])+toward*height*3;view.rotation_euler=Matrix(np.array(camera['basis']).T.tolist()).to_euler();view.data.ortho_scale=camera['height_m']
    view.data.shift_x=(.5-camera['offset'][0])*camera['aspect'];view.data.shift_y=.5-camera['offset'][1];scene.render.resolution_y=round(512/camera['aspect'])
    nodes.clear();out=nodes.new('ShaderNodeOutputMaterial');emit=nodes.new('ShaderNodeEmission');tex=nodes.new('ShaderNodeTexImage');tex.image=target;coords=nodes.new('ShaderNodeUVMap');coords.uv_map='st';links.new(coords.outputs[0],tex.inputs['Vector']);links.new(tex.outputs['Color'],emit.inputs[0]);links.new(emit.outputs[0],out.inputs['Surface'])
    scene.render.filepath=str(folder/'renders'/('view_'+v['angle']+'.png'));bpy.ops.render.render(write_still=True)
bpy.ops.wm.save_as_mainfile(filepath=str(folder/'work.blend'))
assert np.array_equal(np.array([v.co[:] for v in mesh.vertices],dtype='<f4'),points)
assert np.array_equal(np.array([p.vertices[:] for p in mesh.polygons],dtype='<i4'),faces)
(folder/'projection.json').write_text(json.dumps({'geometry_preserved':True,'order':[v['angle'] for v in views],'observed_faces':int(observed.sum()),'faces':len(faces),'source_faces':{v['angle']:int((assignment==i).sum()) for i,v in enumerate(views)},'overlap_policy':'replace only if visible incidence improves by >0.025; ties keep earlier view','limitations':['Photographic highlights retained','Orthographic pose approximation','Hard chart boundaries may remain','Unseen faces retain original vertex colours']},indent=2))
