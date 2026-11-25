#!/usr/bin/env python3
"""Export exact ScanNet annotation-mesh vertices and ownership for offline viewing."""
import argparse,base64,colorsys,hashlib,json,os
from pathlib import Path
import numpy as np
from plyfile import PlyData,PlyElement

HERE=Path(__file__).resolve().parent

def encode(a,dtype):return base64.b64encode(np.asarray(a,dtype=dtype).tobytes()).decode('ascii')
def write_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def color(oid):return [round(v*255) for v in colorsys.hsv_to_rgb((oid*.61803398875)%1,.65,.95)]

def export_scene(source,out):
    sid=source.name
    paths={k:source/(sid+s) for k,s in {'mesh':'_vh_clean_2.ply','segments':'_vh_clean_2.0.010000.segs.json','aggregation':'.aggregation.json','labels':'_vh_clean_2.labels.ply','metadata':'.txt'}.items()}
    if not all(p.is_file() for p in paths.values()):return None
    mesh=PlyData.read(paths['mesh']);v=mesh['vertex'];xyz=np.column_stack([v[k] for k in ('x','y','z')]).astype(np.float32)
    rgb=np.column_stack([v[k] for k in ('red','green','blue')]).astype(np.uint8)
    faces=np.asarray(mesh['face']['vertex_indices'].tolist(),dtype=np.uint32)
    segments=np.asarray(json.loads(paths['segments'].read_text())['segIndices'],dtype=np.int32)
    annotation=json.loads(paths['aggregation'].read_text());groups=annotation['segGroups']
    labeled=PlyData.read(paths['labels'])['vertex'];nyu=np.asarray(labeled['label'],dtype=np.uint16)
    assert len(segments)==len(xyz)==len(nyu)
    assert np.array_equal(xyz,np.column_stack([labeled[k] for k in ('x','y','z')]))
    assert faces.shape[1]==3 and faces.max()<len(xyz)
    metadata={k.strip():val.strip() for line in paths['metadata'].read_text().splitlines() if '=' in line for k,val in [line.split('=',1)]}
    alignment=np.fromstring(metadata['axisAlignment'],sep=' ').reshape(4,4)
    aligned=(np.column_stack([xyz,np.ones(len(xyz))])@alignment.T)[:,:3].astype(np.float32)
    assert np.isfinite(aligned).all()
    # Rendering-only, area-weighted vertex normals; ownership is unchanged.
    normals=np.zeros_like(aligned)
    face_normals=np.cross(aligned[faces[:,1]]-aligned[faces[:,0]],aligned[faces[:,2]]-aligned[faces[:,0]])
    for corner in range(3):np.add.at(normals,faces[:,corner],face_normals)
    lengths=np.linalg.norm(normals,axis=1);normals/=np.maximum(lengths[:,None],1e-12)
    owner=np.full(len(xyz),-1,np.int32);objects=[];known_segments=set(segments.tolist());ids=set()
    dest=out/sid;dest.mkdir(parents=True,exist_ok=True);(dest/'objects').mkdir(exist_ok=True)
    for group in groups:
        oid=int(group['objectId']);assert oid>=0 and oid not in ids;ids.add(oid)
        missing=set(group['segments'])-known_segments
        assert not missing,(sid,oid,'segments missing',missing)
        take=np.isin(segments,group['segments']);assert np.all(owner[take]==-1),(sid,oid,'conflicting annotations')
        owner[take]=oid;indices=np.flatnonzero(take).astype(np.uint32);pts=aligned[take]
        bounds={'minimum':pts.min(0).tolist(),'maximum':pts.max(0).tolist()} if len(pts) else None
        values,counts=np.unique(nyu[take],return_counts=True)
        obj={'object_id':oid,'group_id':group['id'],'label':group['label'],'segments':group['segments'],
             'vertex_count':int(take.sum()),'bounds':bounds,'color':color(oid),
             'nyu40_histogram':{str(int(k)):int(c) for k,c in zip(values,counts)},
             'ply':f'objects/object_{oid:04d}.ply','npz':f'objects/object_{oid:04d}.npz'}
        objects.append(obj)
        np.savez_compressed(dest/obj['npz'],points_world=pts,points_scan=xyz[take],rgb=rgb[take],
            vertex_indices=indices,segment_ids=segments[take],object_id=np.int32(oid),nyu40_ids=nyu[take],axis_alignment=alignment)
        data=np.empty(len(pts),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),('vertex_index','<u4'),('segment_id','<i4'),('object_id','<i4')])
        for j,key in enumerate(('x','y','z')):data[key]=pts[:,j]
        for j,key in enumerate(('red','green','blue')):data[key]=rgb[take,j]
        data['vertex_index']=indices;data['segment_id']=segments[take];data['object_id']=oid
        PlyData([PlyElement.describe(data,'vertex')],text=False,byte_order='<').write(dest/obj['ply'])
    assert sum(o['vertex_count'] for o in objects)+int((owner<0).sum())==len(xyz)
    meta={'scene_id':sid,'scene_type':metadata.get('sceneType',''),'objects':objects,'vertex_count':len(xyz),
          'face_count':len(faces),'annotated_vertices':int((owner>=0).sum()),'unannotated_vertices':int((owner<0).sum()),
          'bounds':{'minimum':aligned.min(0).tolist(),'maximum':aligned.max(0).tolist()},'axis_alignment':alignment.tolist(),
          'coordinates':'meters; annotation mesh scan coordinates transformed once by axisAlignment; +Z upright',
          'representation':'all vertices of the simplified annotation mesh, no sampling or object regrouping',
          'semantic_id_space':'raw aggregation label; NYU40 IDs from labels.ply (not 2D ScanNet IDs)',
          'sources':{k:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in paths.items()}}
    write_json(dest/'manifest.json',meta)
    np.savez_compressed(dest/'scene.npz',points_world=aligned,points_scan=xyz,rgb=rgb,object_ids=owner,segment_ids=segments,nyu40_ids=nyu,faces=faces,axis_alignment=alignment)
    payload={'meta':meta,'positions':encode(aligned,'<f4'),'normals':encode(normals,'<f4'),'rgb':encode(rgb,'u1'),'owners':encode(owner,'<i4'),'segments':encode(segments,'<i4'),'faces':encode(faces,'<u4')}
    (dest/'data.js').write_text('window.registerGTScene('+json.dumps(sid)+','+json.dumps(payload,ensure_ascii=False,separators=(',',':'))+');\n')
    print(f'{sid}: {len(objects)} objects, {len(xyz)} vertices, {meta["unannotated_vertices"]} unannotated',flush=True)
    return {'id':sid,'type':meta['scene_type'],'objects':len(objects),'vertices':len(xyz),'asset':f'{sid}/data.js'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(os.environ.get('SCANNET_ROOT',HERE.parents[1]/'data/scannet_v2')));args=p.parse_args()
    out=HERE/'assets/object_gt';out.mkdir(parents=True,exist_ok=True)
    scenes=[s for source in sorted(args.root.glob('scene*')) if source.is_dir() for s in [export_scene(source,out)] if s]
    if not scenes:raise RuntimeError('No scene has all required annotation files')
    (out/'catalog.js').write_text('window.GT_CATALOG='+json.dumps(scenes,ensure_ascii=False)+';\n')
    base=(HERE.parent/'InstanceRecognition/src/web/frontend/mask_pointcloud/viewer.js').read_text().rstrip()
    assert base.endswith('main();');(out/'renderer.js').write_text(base[:-len('main();')])
    write_json(out/'build_report.json',{'scenes':scenes,'source_root':str(args.root.resolve()),'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})

if __name__=='__main__':main()
