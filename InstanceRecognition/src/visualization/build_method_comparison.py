#!/usr/bin/env python3
"""Build an offline method comparison using the existing mask viewer and AMB3R maps."""
from __future__ import annotations
import argparse, copy, json, shutil, statistics, time
from pathlib import Path
import numpy as np
from PIL import Image
from src.geometry.backproject_masks import (APP_ROOT, align_mask, apply_sim3, crop_box, display_colors, encode,
    instance_color, membership_bits, relative, sha256_file, write_cloud_asset, write_json, write_ply)

TEMPLATE=APP_ROOT/'web/method_comparison'
VIEWER=APP_ROOT/'web/mask_pointcloud'

def viewer_bundle(folder, manifest, method):
    folder.mkdir(parents=True,exist_ok=True)
    write_json(folder/'manifest.json',manifest)
    (folder/'manifest.js').write_text('window.MASK_PROJECTION_MANIFEST='+json.dumps(manifest,ensure_ascii=False,separators=(',',':'),allow_nan=False)+';\n')
    html=(VIEWER/'index.html').read_text()
    if method=='gpt':
        html=html.replace('YOLOE-26 + SAM3.1 / AMB3R GT-corrected', 'GPT-5.6 Terra + SAM3.1 / AMB3R GT-corrected')
        html=html.replace('没有新增置信度阈值','SAM 文本提示阈值 0.5；最多展示 16 个 mask')
    html=html.replace('原始 2D Mask 反投影检查','2D Mask 与 3D 落点')
    html=html.replace('<script src="viewer.js"></script>','<script src="viewer.js"></script>\n  <script src="bridge.js"></script>')
    (folder/'index.html').write_text(html)
    script=(VIEWER/'viewer.js').read_text()
    script=script.replace('${index.toString().padStart(3, "0")}', '${frame.sequence_index.toString().padStart(3, "0")}')
    script=script.replace('${Number(instance.confidence).toFixed(3)} /', '${escapeHtml(instance.score_type || "YOLOE")} ${Number(instance.confidence).toFixed(3)} /')
    (folder/'viewer.js').write_text(script)
    shutil.copy2(TEMPLATE/'bridge.js',folder/'bridge.js')
    style=(VIEWER/'styles.css').read_text()
    # Main page owns frame navigation; standalone viewers retain their controls.
    style+='\n'+(TEMPLATE/'embedded.css').read_text()
    (folder/'styles.css').write_text(style)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--labels',type=Path,required=True)
    ap.add_argument('--results',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--primary-only',action='store_true',help='Export GPT mask viewer only, without the comparison shell')
    ap.add_argument('--baseline',type=Path,default=APP_ROOT/'runtime/mask_pointclouds/scene0000_00/amb3r_gt_raw')
    args=ap.parse_args(); start=time.monotonic()
    output=args.output.resolve(); baseline=args.baseline.resolve()
    if json.loads((baseline/'manifest.json').read_text()).get('method') == 'gpt':
        baseline = baseline.parent/'yoloe26_gt_raw'
    run=json.loads((args.labels/'config.json').read_text())
    base_manifest=json.loads((baseline/'manifest.json').read_text())
    base_report=json.loads((baseline/'projection_report.json').read_text())
    entries=run['frames']; baseline_lookup={f['sequence_index']:f for f in base_manifest['frames']}
    predictions=[]; labeling=[]
    for rec in entries:
        seq=rec['sequence_index']; fid=rec['source_frame_id']
        if baseline_lookup[seq]['source_frame_id']!=fid: raise ValueError('Sample/source mapping mismatch')
        label_path=args.labels/f'labels/{seq:04d}.json'
        pred_path=args.results/f'frame_{fid:06d}/sam31/predictions.json'
        label=json.loads(label_path.read_text()); pred=json.loads(pred_path.read_text())
        if label['source_sha256']!=sha256_file(Path(rec['source'])): raise ValueError('Source RGB changed')
        if pred['provenance']['labels_sha256']!=sha256_file(label_path): raise ValueError('SAM labels changed')
        if pred['source_frame_id']!=fid or pred['sequence_index']!=seq: raise ValueError('SAM frame mismatch')
        predictions.append((pred_path,pred)); labeling.append((label_path,label))
    prediction_path=Path(base_report['source_prediction'])
    if sha256_file(prediction_path)!=base_report['source_prediction_sha256']: raise ValueError('Geometry archive changed')
    alignment=json.loads(Path(base_report['source_alignment']).read_text())
    if sha256_file(Path(base_report['source_alignment']))!=base_report['source_alignment_sha256']: raise ValueError('Alignment changed')
    transform=alignment['sim3_oracle_scale']
    prep=base_report['preprocessing']; intermediate=tuple(prep['intermediate_size']); target=tuple(prep['point_map_size'])
    width,height=target; original_size=tuple(prep['original_size'])
    print('Loading existing AMB3R point/RGB maps; no reconstruction inference',flush=True)
    with np.load(prediction_path,allow_pickle=False) as archive:
        points=archive['pts']; images=archive['images']
    if points.shape[1:3]!=(height,width): raise ValueError('Geometry shape mismatch')
    output.mkdir(parents=True,exist_ok=True)
    gpt=output if args.primary_only else output/'gpt'; base=output/'baseline'
    for path in (gpt/'frames',gpt/'assets') + (() if args.primary_only else (base,)): path.mkdir(parents=True,exist_ok=True)
    base_frames=[]; gpt_frames=[]; comparison=[]; validation=[]
    for rec,(pred_path,pred),(label_path,label) in zip(entries,predictions,labeling):
        seq=rec['sequence_index']; fid=rec['source_frame_id']; stem=f'frame_{fid:06d}'
        old=baseline_lookup[seq]; bf=copy.deepcopy(old)
        for key in ('asset','npz','ply','rgb_image','original_rgb_image','annotated_image','predictions_json'):
            bf[key]=relative(baseline/old[key],base)
        for item in bf['instances']:
            item['mask_file']=relative(baseline/item['mask_file'],base); item['score_type']='YOLOE'
        base_frames.append(bf)
        flat=points[seq].reshape(-1,3); finite=np.isfinite(flat).all(axis=1)
        rgb=np.clip(np.rint((np.moveaxis(images[seq],0,-1)+1.0)*127.5),0,255).astype(np.uint8)
        # Replay all existing baseline union points as an independent coordinate/color check.
        with np.load(baseline/old['npz'],allow_pickle=False) as cached:
            base_pixels=cached['pixel_indices']; expected=cached['points_world']
            replay=apply_sim3(flat[base_pixels],transform)
            error=float(np.abs(replay-expected).max()) if len(replay) else 0.
            rgb_equal=np.array_equal(rgb.reshape(-1,3)[base_pixels],cached['rgb'])
            if error>1e-5 or not rgb_equal: raise ValueError(f'Baseline geometry replay mismatch at {seq}')
        masks=[]; items=[]
        for item in pred['instances']:
            path=pred_path.parent/item['mask_file']
            with Image.open(path) as image:
                if image.size!=original_size: raise ValueError('Mask size mismatch')
                raw=np.asarray(image)
            mask=align_mask(raw,intermediate,target); masks.append(mask)
            items.append(dict(item,mask_file=relative(path,gpt),color=instance_color(fid,item['instance_id']),
                confidence=item['sam_score'],score_type='SAM',original_mask_pixels=int(np.count_nonzero(raw)),
                aligned_mask_pixels=int(mask.sum())))
        bits_grid=membership_bits(masks,(height,width)); flat_bits=bits_grid.reshape(-1)
        selected=np.flatnonzero((flat_bits!=0)&finite).astype(np.uint32)
        world=apply_sim3(flat[selected],transform); bits=flat_bits[selected]; colors=rgb.reshape(-1,3)[selected]
        invalid=int(np.count_nonzero((flat_bits!=0)&~finite))
        overlap=int(np.count_nonzero(bits & (bits-np.uint32(1))))
        for index,item in enumerate(items):
            item['point_count']=int(np.count_nonzero(bits & np.uint32(1<<index)))
            item['nonfinite_points_skipped']=int(np.count_nonzero(masks[index].reshape(-1)&~finite))
        np.savez_compressed(gpt/'frames'/f'{stem}.npz',points_world=world,rgb=colors,mask_bits=bits,pixel_indices=selected,
            instance_ids=np.array([x['instance_id'] for x in items],dtype=np.uint32),source_frame_id=np.int64(fid),
            sequence_index=np.int64(seq),grid_shape_hw=np.array([height,width],dtype=np.int32))
        write_ply(gpt/'frames'/f'{stem}.ply',world,display_colors(bits,items),bits,np.full(len(world),fid,dtype=np.uint32),selected)
        write_cloud_asset(gpt/'assets'/f'{stem}.js',fid,{'source_frame_id':fid,'count':len(world),'width':width,'height':height,
            'original_size':list(original_size),'intermediate_size':list(intermediate),'crop_box':list(crop_box(intermediate,target)),
            'positions_f32_base64':encode(world,'<f4'),'colors_u8_base64':encode(colors,'u1'),
            'mask_bits_u32_base64':encode(bits,'<u4'),'pixel_indices_u32_base64':encode(selected,'<u4')})
        gf={'sequence_index':seq,'source_frame_id':fid,'asset':f'assets/{stem}.js','npz':f'frames/{stem}.npz','ply':f'frames/{stem}.ply',
            'rgb_image':relative(baseline/old['rgb_image'],gpt),'original_rgb_image':relative(Path(rec['source']),gpt),
            'predictions_json':relative(pred_path,gpt),'instances':items,'point_count':len(world),
            'membership_count':sum(x['point_count'] for x in items),'overlap_point_count':overlap,'nonfinite_points_skipped':invalid}
        gpt_frames.append(gf); write_json(gpt/'frames'/f'{stem}.json',gf)
        prompts=copy.deepcopy(pred['prompts'])
        for prompt in prompts: prompt['selected_count']=sum(x['prompt_index']==prompt['prompt_index'] for x in items)
        comparison.append({'sequence_index':seq,'source_frame_id':fid,'baseline_masks':len(old['instances']),
            'baseline_classes':sorted({x['class_name'] for x in old['instances']}),'gpt_masks':len(items),
            'gpt_raw_masks':pred['raw_candidate_count'],'gpt_categories':len(prompts),'prompts':prompts,
            'gpt_seconds':label['elapsed_seconds'],'sam_seconds':pred['elapsed_seconds'],'usage':label['usage'],
            'gpt_labels':relative(label_path,output),'gpt_predictions':relative(pred_path,output),
            'baseline_predictions':relative(baseline/old['predictions_json'],output),'unresolved_regions':pred['unresolved_regions']})
        validation.append({'sequence_index':seq,'source_frame_id':fid,'baseline_replay_max_error_m':error,
            'baseline_rgb_exact':rgb_equal,'nonfinite_points_skipped':invalid,'empty_projected_masks':sum(x['point_count']==0 for x in items)})
        print(f'PROJECT sequence={seq} masks={len(items)} points={len(world)}',flush=True)
    bundles=((gpt,gpt_frames,'gpt'),) if args.primary_only else ((base,base_frames,'baseline'),(gpt,gpt_frames,'gpt'))
    for folder,frames,method in bundles:
        manifest={'scene_id':run['scene_id'],'frame_count':len(frames),'mask_count':sum(len(x['instances']) for x in frames),
            'background_asset':relative(baseline/base_manifest['background_asset'],folder),'bounds':base_manifest['bounds'],
            'frames':frames,'policy':'Frame-local IDs; overlaps preserved; no temporal fusion', 'method':method,
            'model':run['model'] if method=='gpt' else 'YOLOE-26'}
        viewer_bundle(folder,manifest,method)
    report={'scene_id':run['scene_id'],'model':run['model'],'frame_count':len(comparison),'frames':comparison,
        'baseline_mask_count':sum(f['baseline_masks'] for f in comparison),'gpt_mask_count':sum(f['gpt_masks'] for f in comparison),
        'raw_mask_count':sum(f['gpt_raw_masks'] for f in comparison),
        'gpt_median_seconds':statistics.median(f['gpt_seconds'] for f in comparison),
        'gpt_total_reported_tokens':sum((f['usage'] or {}).get('total_tokens',0) for f in comparison),
        'labels_config':str((args.labels/'config.json').resolve()),'sam_policy':predictions[0][1]['policy'],
        'baseline_report':str((baseline/'projection_report.json').resolve()),'geometry_prediction_sha256':base_report['source_prediction_sha256'],
        'alignment_sha256':base_report['source_alignment_sha256'],'elapsed_seconds':round(time.monotonic()-start,3),
        'scope':'End-to-end comparison; no accuracy ground truth; raw GPT labels; no temporal smoothing'}
    write_json(output/'comparison.json',report)
    (output/'comparison.js').write_text('window.METHOD_COMPARISON='+json.dumps(report,ensure_ascii=False,separators=(',',':'),allow_nan=False)+';\n')
    write_json(output/'projection_validation.json',{'frames':validation,'frame_count':len(validation),
        'max_baseline_replay_error_m':max(x['baseline_replay_max_error_m'] for x in validation),
        'all_baseline_rgb_exact':all(x['baseline_rgb_exact'] for x in validation),'preprocessing':prep,
        'uses_gt_depth':False,'uses_recorded_gt_sim3':True,'new_reconstruction_inference':False})
    if args.primary_only:
        projection = dict(base_report)
        projection.update({'method':'gpt_sam31','model':run['model'],'frame_count':len(gpt_frames),
            'labels_config':str((args.labels/'config.json').resolve()),
            'mask_results':str(args.results.resolve()),
            'baseline_report':str((baseline/'projection_report.json').resolve()),
            'mask_count':sum(len(f['instances']) for f in gpt_frames),
            'source_frame_ids':[f['source_frame_id'] for f in gpt_frames],
            'raw_union_point_count_sum_over_frames':sum(f['point_count'] for f in gpt_frames),
            'raw_mask_membership_count_sum_over_frames':sum(f['membership_count'] for f in gpt_frames),
            'overlap_point_count_sum_over_frames':sum(f['overlap_point_count'] for f in gpt_frames),
            'empty_projected_masks':sum(x['empty_projected_masks'] for x in validation),
            'nonfinite_mask_points_skipped':sum(x['nonfinite_points_skipped'] for x in validation),
            'existing_input_policy':predictions[0][1]['policy'],
            'elapsed_seconds':round(time.monotonic()-start,3)})
        projection['validation']={k:v for k,v in base_report['validation'].items()
            if k not in {'canonical_points_with_any_mask','canonical_points_with_overlapping_masks'}}
        projection['validation']['baseline_union_replay_max_error_m']=max(x['baseline_replay_max_error_m'] for x in validation)
        write_json(output/'projection_report.json',projection)
    else:
        for name in ('index.html','styles.css','app.js'): shutil.copy2(TEMPLATE/name,output/name)
    print('Viewer ready: '+str(output/'index.html'),flush=True)

if __name__=='__main__': main()
