#!/usr/bin/env python3
"""SAM3.1 text-prompt branch consuming independently generated GPT labels."""
from __future__ import annotations
import argparse, hashlib, json, tempfile, time, uuid
from pathlib import Path
import numpy as np
from PIL import Image
from src import config
from src.inference.sam31_segment import Sam31Segmenter, normalized_xywh_to_xyxy, safe_name


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def segment(predictor, source, categories, output, threshold=.5, max_masks=16):
    """Keep raw candidates on disk, including those outside the display budget."""
    image=Image.open(source).convert('RGB'); width,height=image.size
    instances=[]; prompt_records=[]; started=time.monotonic()
    output.mkdir(parents=True,exist_ok=True)
    (output/'raw_masks').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='gpt_sam31_') as tmp:
        image.save(Path(tmp)/'000000.jpg',quality=100,subsampling=0)
        state=predictor.model.init_state(resource_path=tmp,offload_video_to_cpu=False,async_loading_frames=False)
        session=str(uuid.uuid4()); now=time.time()
        predictor._all_inference_states[session]={'state':state,'session_id':session,'start_time':now,'last_use_time':now}
        try:
            for prompt_index,category in enumerate(categories):
                tick=time.monotonic()
                response=predictor.handle_request({'type':'add_prompt','session_id':session,'frame_index':0,
                    'text':category['sam_prompt'],'output_prob_thresh':threshold})
                outputs=response['outputs']
                masks=np.asarray(outputs['out_binary_masks'],dtype=bool)
                if masks.ndim==4 and masks.shape[1]==1: masks=masks[:,0]
                scores=np.asarray(outputs['out_probs']).reshape(-1)
                boxes=normalized_xywh_to_xyxy(outputs['out_boxes_xywh'],width,height)
                if not len(scores): masks=np.zeros((0,height,width),dtype=bool)
                if masks.shape!=(len(scores),height,width) or len(boxes)!=len(scores):
                    raise ValueError('SAM text output shape mismatch')
                ids=[]; empty=[]
                for candidate_index,(mask,score,box) in enumerate(zip(masks,scores,boxes)):
                    if not np.isfinite(score) or not np.isfinite(box).all(): raise ValueError('Nonfinite SAM output')
                    if not mask.any(): empty.append(candidate_index); continue
                    raw_id=len(instances)+1; ids.append(raw_id)
                    relative=f'raw_masks/{raw_id:03d}_{safe_name(category["category"])}.png'
                    Image.fromarray(mask.astype(np.uint8)*255).save(output/relative)
                    instances.append({'raw_candidate_id':raw_id,'class_name':category['category'],
                        'class_id':prompt_index,'prompt_index':prompt_index,'sam_prompt':category['sam_prompt'],
                        'gpt_certainty':category['certainty'],'visible_scope':category['visible_scope'],
                        'evidence':category['evidence'],'ambiguity':category['ambiguity'],
                        'sam_score':float(score),'sam_box_xyxy':box.round(2).tolist(),
                        'sam_prompt_type':'text','mask_file':relative,'mask_pixels':int(mask.sum())})
                prompt_records.append({'prompt_index':prompt_index,**category,'raw_candidate_ids':ids,
                    'empty_candidate_indices':empty,'elapsed_seconds':round(time.monotonic()-tick,3)})
        finally:
            predictor.handle_request({'type':'close_session','session_id':session})
    # Same displayed instance budget as YOLOE baseline. Never delete discarded raw masks.
    selected=sorted(instances,key=lambda x:(-x['sam_score'],x['raw_candidate_id']))[:max_masks]
    selected=sorted(selected,key=lambda x:x['raw_candidate_id'])
    keep={x['raw_candidate_id'] for x in selected}
    selected=[dict(x,instance_id=i+1) for i,x in enumerate(selected)]
    return {'stage':'GPT categories -> SAM3.1 text-prompt masks','source':str(Path(source).resolve()),
        'image_shape_hwc':[height,width,3], 'instance_count':len(selected),'instances':selected,
        'raw_candidates':instances,'raw_candidate_count':len(instances),
        'excluded_by_budget':[x['raw_candidate_id'] for x in instances if x['raw_candidate_id'] not in keep],
        'prompts':prompt_records,'unique_classes':sorted({x['class_name'] for x in selected}),
        'policy':{'sam_output_prob_threshold':threshold,'max_selected_masks':max_masks,
            'selection':'top SAM scores, ties by prompt/candidate order; all raw masks retained',
            'uncertain_categories_segmented':True,'cross_category_nms':False,'temporal_propagation':False,
            'empty_prompts_preserved':True,'yoloe_input_consumed':False},
        'elapsed_seconds':round(time.monotonic()-started,3)}


def write_display_artifacts(prediction, output):
    """Create web previews; independent binary masks remain authoritative for overlaps."""
    import cv2
    from src.inference.sam31_segment import PALETTE
    output=Path(output)
    image=cv2.imread(prediction['source'])
    if image is None: raise ValueError('Cannot read source RGB')
    preview=image.astype(np.float32); masks=[]
    id_map=np.zeros(image.shape[:2],dtype=np.uint16)
    for index,item in enumerate(prediction['instances']):
        with Image.open(output/item['mask_file']) as handle: mask=np.asarray(handle)>0
        if mask.shape!=image.shape[:2]: raise ValueError('Mask dimensions differ from RGB')
        masks.append(mask); color=np.asarray(PALETTE[index%len(PALETTE)],dtype=np.float32)
        preview[mask]=preview[mask]*.46+color*.54; id_map[mask]=item['instance_id']
    preview=np.clip(preview,0,255).astype(np.uint8)
    scale=max(.55,min(image.shape[:2])/1000)
    for index,(item,mask) in enumerate(zip(prediction['instances'],masks)):
        color=PALETTE[index%len(PALETTE)]
        contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(preview,contours,-1,color,2)
        x,y,_,_=map(round,item['sam_box_xyxy']); y=max(26,y)
        text=f"{item['instance_id']} {item['class_name']} SAM {item['sam_score']:.2f}"
        cv2.putText(preview,text,(max(0,x),y),cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),4,cv2.LINE_AA)
        cv2.putText(preview,text,(max(0,x),y),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),1,cv2.LINE_AA)
    if not cv2.imwrite(str(output/'annotated.jpg'),preview): raise RuntimeError('Cannot save annotated RGB')
    if not cv2.imwrite(str(output/'instance_id_map.png'),id_map): raise RuntimeError('Cannot save instance ID map')
    prediction['artifacts']={'annotated_image':'annotated.jpg','instance_id_map':'instance_id_map.png'}
    prediction['instance_id_map_policy']='display-only; overlaps overwritten by later instance; independent binary masks are authoritative'
    prediction['timing']={'session_and_prompts_seconds':prediction['elapsed_seconds']}
    return prediction


class GptSam31Segmenter:
    """Persistent worker adapter sharing the same segmentation logic as the batch."""
    def __init__(self,checkpoint,threshold=.5):
        self.segmenter=Sam31Segmenter(checkpoint,threshold)
        self.threshold=threshold

    def infer(self,source,labels_path,output):
        labels=json.loads(Path(labels_path).read_text())
        if Path(labels['source']).resolve()!=Path(source).resolve(): raise ValueError('GPT/SAM source path mismatch')
        if labels['source_sha256']!=hashlib.sha256(Path(source).read_bytes()).hexdigest(): raise ValueError('GPT/SAM source hash mismatch')
        result=segment(self.segmenter.predictor,source,labels['labels']['categories'],Path(output),self.threshold,16)
        result.update({'gpt_model':labels['requested_model'],'gpt_elapsed_seconds':labels['elapsed_seconds'],
            'gpt_usage':labels['usage'],'unresolved_regions':labels['labels']['unresolved_regions']})
        write_display_artifacts(result,output)
        save(Path(output)/'predictions.json',result)
        return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--labels',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,default=config.SAM31_CHECKPOINT)
    ap.add_argument('--threshold',type=float,default=.5)
    ap.add_argument('--max-masks',type=int,default=16)
    ap.add_argument('--wait-for-labels',type=float,default=0,help='Wait up to N seconds per frame for an active labeling job')
    args=ap.parse_args()
    if not 0<=args.threshold<=1 or not 1<=args.max_masks<=32: ap.error('Invalid mask settings')
    run_config=json.loads((args.labels/'config.json').read_text())
    records=run_config['frames']
    if not args.wait_for_labels:
        for record in records:
            label_path=args.labels/f'labels/{record["sequence_index"]:04d}.json'
            if not label_path.is_file(): raise SystemExit(f'Missing GPT labels: {label_path}')
    segmenter=None; results=[]
    for record in records:
        seq=record['sequence_index']; fid=record['source_frame_id']
        label_path=args.labels/f'labels/{seq:04d}.json'
        deadline=time.monotonic()+args.wait_for_labels
        while not label_path.is_file() and time.monotonic()<deadline:
            time.sleep(2)
        if not label_path.is_file(): raise SystemExit(f'Missing GPT labels: {label_path}')
        labels=json.loads(label_path.read_text())
        source=Path(record['source']); source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
        if labels['source_frame_id']!=fid or labels['source_sha256']!=source_sha: raise ValueError('GPT/source mismatch')
        provenance={'source_sha256':source_sha,'labels_sha256':hashlib.sha256(label_path.read_bytes()).hexdigest(),
            'checkpoint':str(args.checkpoint.resolve()),'checkpoint_size':args.checkpoint.stat().st_size,
            'checkpoint_mtime_ns':args.checkpoint.stat().st_mtime_ns,'threshold':args.threshold,'max_masks':args.max_masks}
        out=args.output/f'frame_{fid:06d}'/'sam31'; path=out/'predictions.json'
        if path.exists():
            prediction=json.loads(path.read_text())
            if prediction['provenance']!=provenance: raise ValueError('SAM cached config mismatch; use a new output directory')
            for item in prediction['raw_candidates']:
                with Image.open(out/item['mask_file']) as mask:
                    if mask.size!=tuple(prediction['image_shape_hwc'][1::-1]): raise ValueError('Cached mask size mismatch')
        else:
            if segmenter is None: segmenter=Sam31Segmenter(args.checkpoint,threshold=args.threshold)
            prediction=segment(segmenter.predictor,source,labels['labels']['categories'],out,args.threshold,args.max_masks)
            prediction.update({'sequence_index':seq,'source_frame_id':fid,'provenance':provenance,
                'gpt_model':labels['requested_model'],'gpt_elapsed_seconds':labels['elapsed_seconds'],
                'gpt_usage':labels['usage'],'unresolved_regions':labels['labels']['unresolved_regions']})
            write_display_artifacts(prediction,out)
            save(path,prediction)
        if 'artifacts' not in prediction or any(not (out/name).is_file() for name in prediction['artifacts'].values()):
            write_display_artifacts(prediction,out)
            save(path,prediction)
        results.append({'sequence_index':seq,'source_frame_id':fid,'prediction':str(path.resolve()),
            'instance_count':prediction['instance_count'],'raw_candidate_count':prediction['raw_candidate_count']})
        print(f'SAM sequence={seq} instances={prediction["instance_count"]} raw={prediction["raw_candidate_count"]}',flush=True)
    save(args.output/'summary.json',{'frame_count':len(results),'frames':results,'labels_config':str((args.labels/'config.json').resolve())})

if __name__=='__main__': main()
