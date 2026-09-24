"""Run bounded model-led incremental association on existing masks and geometry."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from .observations import Observations,DEFAULT_SOURCE,ROOT,digest,model_observation
from .state import initial,save
from .tools import Tools,TOOLS
from .client import Client,image
from .publish import materialize,publish
from .inputs import build_window_content


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--output',type=Path,default=ROOT/'runtime/experiments/harness_first30_v2')
    parser.add_argument('--frames',type=int,default=30);parser.add_argument('--window',type=int,default=5)
    parser.add_argument('--max-calls',type=int,default=14);parser.add_argument('--voxel',type=float,default=.02)
    parser.add_argument('--model',default='openai/gpt-5.6-terra');parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if not 1<=args.frames<=30 or not 1<=args.window<=args.frames or args.max_calls<4 or args.voxel<=0:parser.error('Invalid limits')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    # One process per run. Advisory lock automatically releases after a crash.
    import fcntl
    lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(out/'code_versions'/f'{time.time_ns()}.json', {str(p.relative_to(ROOT)):digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
    prompt=Path(__file__).with_name('prompt.txt').read_text()
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='prepare_only'}
    config.update(input_policy='history_on_demand_rgb_bbox_v2_spatial_top3_global_obb',prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),manifest_sha256=digest(args.source/'manifest.json'),
                  geometry_policy='fixed full-video GT-corrected AMB3R; incremental identity association only; no GT instance labels supplied')
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=config:raise ValueError('Resume config changed; use new output')
    save(out/'config.json',config)
    prior_audit=json.loads((out/'input_audit.json').read_text()) if (out/'input_audit.json').exists() else None
    data=Observations(args.source,args.frames,out)
    if prior_audit and prior_audit!=data.audit:raise ValueError('Input changed; cannot resume')
    for seq in range(args.frames):
        data.view(seq)
        for oid,o in data.items.items():
            if o['seq']==seq:data.view(seq,oid)
    print(f'PREPARED {len(data.items)} observations, {args.frames} sampled frames',flush=True)
    if args.prepare_only:return
    client=Client(args.model,out)
    checkpoint=json.loads((out/'checkpoint.json').read_text()) if (out/'checkpoint.json').exists() else {'version':0}
    state=json.loads((out/'versions'/f'{checkpoint["version"]:04d}'/'state.json').read_text()) if checkpoint['version'] else initial()
    for first in range(state['cutoff']+1,args.frames,args.window):
        cutoff=min(first+args.window,args.frames)-1;tools=Tools(data,state,cutoff)
        content=build_window_content(data,tools,first,cutoff,args.max_calls)
        messages=[{'role':'system','content':prompt},{'role':'user','content':content}]
        committed=False;tick=time.monotonic();run_tag=str(time.time_ns())
        for step in range(args.max_calls):
            if step==args.max_calls-2:
                messages.append({'role':'user','content':'Budget nearly exhausted: submit a complete valid proposal now, deferring unresolved observations. Then commit.'})
            tag=f'w{first:04d}_{run_tag}_call{step:02d}'
            response=client.call(messages,TOOLS,tag,final=step==args.max_calls-2 and tools.pending is None)
            messages.append(response)
            calls=response['tool_calls']
            evidence_messages=[]
            for tool_index,call in enumerate(calls):
                name=call['function']['name'];images=[]
                try:
                    arguments=json.loads(call['function']['arguments']);result,images=tools.call(name,arguments)
                    if name=='commit_update':
                        checked=tools.pending;version=checked['state']['version'];temp=out/'versions'/f'.{version:04d}.tmp'
                        if temp.exists():shutil.rmtree(temp)
                        previous=out/'versions'/f'{state["version"]:04d}' if state['version'] else None
                        materialize(data,checked,temp,previous,args.voxel)
                        target=out/'versions'/f'{version:04d}'
                        if target.exists():shutil.rmtree(target) # unreferenced orphan only; checkpoint still points to previous
                        temp.rename(target);save(out/'checkpoint.json',{'version':version,'cutoff':cutoff})
                        state=checked['state'];committed=True;result={'committed':version,'cutoff':cutoff}
                except (ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:result={'error':str(exc)}
                save(out/'calls'/f'{tag}_tool{tool_index:02d}.json',{'name':name,'arguments':call['function']['arguments'],'result':result,'images':[str(p) for p in images]})
                messages.append({'role':'tool','tool_call_id':call['id'],'content':json.dumps(result,ensure_ascii=False)})
                if images:
                    visual=[{'type':'text','text':f'Visual evidence from {name}; tool_call_id={call["id"]}'}]
                    for descriptor,path in zip(result['visual_images'],images,strict=True):
                        visual.extend([{'type':'text','text':json.dumps(descriptor,ensure_ascii=False)},image(path)])
                    evidence_messages.append({'role':'user','content':visual})
                print(f'WINDOW {first}-{cutoff} call {step+1}: {name} '+('ERROR '+result['error'] if 'error' in result else 'OK'),flush=True)
                if committed:break
            messages.extend(evidence_messages)
            if committed:break
        if not committed:raise RuntimeError('Window budget exhausted without valid commit; prior checkpoint preserved')
        save(out/'versions'/f'{state["version"]:04d}'/'timing.json',{'elapsed_seconds':time.monotonic()-tick,'calls':step+1})
        publish(data,out,args.voxel)
    publish(data,out,args.voxel)
    metas=[json.loads(p.read_text()) for p in (out/'calls').glob('*_meta.json')]
    usage={k:sum((m.get('usage') or {}).get(k,0) or 0 for m in metas) for k in ('prompt_tokens','completion_tokens','total_tokens')}
    summary={'completed':state['cutoff']==args.frames-1,'frames':args.frames,'windows':state['version'],'objects':len(state['objects']),
             'observations':len(state['assignments']),'pending':sum(a['object_id'] is None for a in state['assignments'].values()),
             'api_successful_calls':len(metas),'usage':usage,'cost_currency':None,'cost_note':'Gateway price not configured; tokens recorded, no fabricated cost',
             'accuracy_evaluated':False,'geometry_policy':config['geometry_policy']}
    save(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
