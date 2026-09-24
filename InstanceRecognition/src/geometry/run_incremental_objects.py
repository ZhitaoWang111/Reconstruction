"""Replay the first sampled frames using fixed AMB3R geometry and existing masks."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from src.geometry.incremental_objects import Associator,voxelize
from src.geometry.backproject_masks import apply_sim3,encode,instance_color
from src.visualization.publish_gpt_viewer import rebase_manifest

ROOT=Path(__file__).resolve().parents[2]

def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def ply(path,arrays,selection):
    n=int(selection.sum());dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),('object_id','<u4'),('support_frame_bits','<u4')]
    data=np.empty(n,dtype=dtype)
    for i,k in enumerate(('x','y','z')):data[k]=arrays['points_world'][selection,i]
    for i,k in enumerate(('red','green','blue')):data[k]=arrays['rgb'][selection,i]
    data['object_id']=arrays['object_ids'][selection];data['support_frame_bits']=arrays['support_frame_bits'][selection]
    with path.open('wb') as f:
        f.write(('ply\nformat binary_little_endian 1.0\nelement vertex '+str(n)+'\n'+''.join('property '+('float' if t=='<f4' else 'uint' if t=='<u4' else 'uchar')+' '+k+'\n' for k,t in dtype)+'end_header\n').encode());data.tofile(f)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frames',type=int,default=30)
    ap.add_argument('--voxel',type=float,default=.02)
    ap.add_argument('--distance',type=float,default=.05)
    ap.add_argument('--output',type=Path,default=ROOT/'runtime/experiments/incremental_objects_first30_v2')
    args=ap.parse_args();assert 1<=args.frames<=30 and args.voxel>0 and args.distance>0
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError('Completed run exists; use a new output directory')
    start=time.monotonic();source=ROOT/'runtime/mask_pointclouds/scene0000_00/amb3r_gt_raw'
    before=digest(source/'manifest.json');manifest=json.loads((source/'manifest.json').read_text())
    report=json.loads((source/'projection_report.json').read_text());geometry=Path(report['source_prediction'])
    assert digest(geometry)==report['source_prediction_sha256']
    alignment=Path(report['source_alignment']);assert digest(alignment)==report['source_alignment_sha256']
    transform=json.loads(alignment.read_text())['sim3_oracle_scale']
    frames=manifest['frames'][:args.frames];assert [f['sequence_index'] for f in frames]==list(range(args.frames))
    config={'frames':args.frames,'sampled_sequences':[0,args.frames-1],'source_frames':[f['source_frame_id'] for f in frames],
        'source_manifest':str(source/'manifest.json'),'source_manifest_sha256':before,
        'geometry_sha256':report['source_prediction_sha256'],'alignment_sha256':report['source_alignment_sha256'],
        'voxel_m':args.voxel,'match_distance_m':args.distance,'ambiguity_margin':.12,
        'policy':'previous-state association, provisional new objects, deferred ambiguity, no automatic object union; group granularity separate',
        'surface_support':'nearest correspondence to full current point map; not calibrated camera reprojection or negative visibility evidence',
        'multi_frame_support':'at least 3 distinct sampled frames; correlated views are not independent confirmations',
        'input_budget':'existing exported masks only; SAM candidates beyond the original 16-mask display budget not included',
        'geometry_policy':'fixed full-video GT-corrected AMB3R input; incremental association only, not end-to-end online reconstruction',
        'models_called':False}
    save(out/'config.json',config);save(out/'source_manifest.json',manifest)
    print('Loading fixed point maps',flush=True)
    with np.load(geometry) as z:maps=z['pts'][:args.frames].copy()
    associator=Associator(args.voxel,args.distance);surfaces={};snapshots=[];input_audit=[]
    for seq,f in enumerate(frames):
        tick=time.monotonic();path=(source/f['npz']).resolve()
        with np.load(path) as z:
            points=z['points_world'];rgb=z['rgb'];bits=z['mask_bits'];pixels=z['pixel_indices'];shape=z['grid_shape_hw']
        world=apply_sim3(maps[seq].reshape(-1,3),transform);valid=np.isfinite(world).all(1)
        grid_bits=np.zeros(len(world),dtype=np.uint32);grid_bits[pixels]=bits
        assert np.max(np.abs(world[pixels]-points))<1e-5
        surfaces[seq]=(cKDTree(world[valid]),grid_bits[valid])
        observations=[]
        for index,item in enumerate(f['instances']):
            take=(bits & np.uint32(1<<index))!=0
            keys,down,_=voxelize(points[take],rgb[take],args.voxel)
            if len(down)==0:raise ValueError('Empty input observation')
            observations.append({'id':f'{seq:04d}:{item["instance_id"]:03d}','seq':seq,'source_frame_id':f['source_frame_id'],
                'mask_index':index,'instance_id':item['instance_id'],'category_label':item['class_name'],
                'points':down,'raw_points':points[take],'raw_colors':rgb[take],'pixels':pixels[take],
                'source_prediction':str((source/f['predictions_json']).resolve()),'source_mask':str((source/item['mask_file']).resolve()),
                'source_points':str(path),'source_mask_sha256':digest(source/item['mask_file'])})
        associator.step(seq,observations,*surfaces[seq]);associator.retry_pending(surfaces)
        objects,arrays=associator.snapshot()
        for obj in objects:obj['color']=instance_color(0,obj['id'])
        assignments=[{k:o[k] for k in ('id','seq','source_frame_id','instance_id','category_label','status','object_id','candidates')} for o in observations]
        snapshot={'sequence_index':seq,'source_frame_id':f['source_frame_id'],'objects':objects,
            'assignments':assignments,'pending':list(associator.pending),
            'events':[e for e in associator.events if e['at']==seq],
            'voxel_count':len(arrays['points_world']),'conflict_entries':int(arrays['ownership_conflict'].sum()),
            'new_voxels':int((arrays['added_at']==seq).sum()),'elapsed_seconds':round(time.monotonic()-tick,3)}
        (out/'states').mkdir(exist_ok=True)
        np.savez_compressed(out/f'states/{seq:04d}.npz',**arrays)
        payload={'snapshot':snapshot,'positions':encode(arrays['points_world'],'<f4'),'rgb':encode(arrays['rgb'],'u1'),
            'ids':encode(arrays['object_ids'],'<u4'),'support':encode(arrays['support_frame_bits'],'<u4'),
            'first':encode(arrays['first_seen'],'u1'),'added':encode(arrays['added_at'],'u1'),'conflicts':encode(arrays['ownership_conflict'],'u1')}
        (out/f'states/{seq:04d}.js').write_text('window.registerIncrementalState('+str(seq)+','+json.dumps(payload,ensure_ascii=False,separators=(',',':'))+');\n')
        save(out/f'states/{seq:04d}.json',snapshot);snapshots.append(snapshot)
        input_audit.append({'sequence':seq,'npz':str(path),'sha256':digest(path)})
        print(f'STEP {seq:02d}: masks={len(observations)} objects={len(objects)} pending={len(associator.pending)} voxels={len(arrays["object_ids"])} seconds={snapshot["elapsed_seconds"]}',flush=True)
    (out/'objects').mkdir(exist_ok=True)
    for obj in objects:
        sel=arrays['object_ids']==obj['id'];ply(out/f'objects/object_{obj["id"]:04d}.ply',arrays,sel)
        np.savez_compressed(out/f'objects/object_{obj["id"]:04d}.npz',**{k:v[sel] for k,v in arrays.items()})
    provenance=[{k:v for k,v in o.items() if not isinstance(v,np.ndarray)} for o in associator.observations.values()]
    save(out/'observations.json',provenance);save(out/'events.json',associator.events);save(out/'input_audit.json',input_audit)
    summary={'frame_count':args.frames,'observation_count':len(provenance),'object_candidates':len(objects),
        'multi_frame_objects':sum(o['status']=='multi_frame' for o in objects),
        'semantic_groups':sum(o['kind']=='semantic_group' for o in objects),
        'pending_observations':len(associator.pending),'deferred_resolutions':sum(e['action']=='resolve_pending' for e in associator.events),
        'final_voxel_entries':len(arrays['object_ids']),'ownership_conflict_entries':int(arrays['ownership_conflict'].sum()),
        'elapsed_seconds':round(time.monotonic()-start,3),'objects':objects,
        'accuracy_claim':False,'automatic_object_merge_or_split':False}
    assert digest(source/'manifest.json')==before
    save(out/'summary.json',summary)
    m=dict(manifest,frames=frames,frame_count=args.frames,mask_count=sum(len(f['instances']) for f in frames))
    m=rebase_manifest(m,source,out)
    (out/'manifest.js').write_text('window.MASK_PROJECTION_MANIFEST='+json.dumps(m,ensure_ascii=False,separators=(',',':'))+';\n')
    from src.visualization.build_incremental_viewer import build
    build(out,args.frames)
    print(json.dumps({k:v for k,v in summary.items() if k!='objects'},ensure_ascii=False))

if __name__=='__main__':main()
