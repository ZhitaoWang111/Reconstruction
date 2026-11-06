"""Run bounded model-led incremental association on existing masks and geometry."""
import argparse
import hashlib
import json
import math
import re
import shutil
import time
from pathlib import Path
from .observations import Observations,DEFAULT_SOURCE,digest
from .state import initial,save
from .tools import Tools,TOOLS
from .client import Client,GatewayNetworkError,image
from .publish import materialize
from .inputs import build_window_content
from .paths import resolve_recorded_path
from src import config as app_config
from src.geometry.amb3r_mask_source import POLICY as AMB3R_FRAME_POLICY
from pagegen.build_site import publish

# The one source policy this tree builds. Exact matching is deliberate: it is
# what stops an accidentally mis-pointed --source from silently running the
# wrong geometry.
ACCEPTED_LIFTING_POLICIES = (AMB3R_FRAME_POLICY,)


def resolve_output(name, run_dir=None, site_dir=None):
    """Map an experiment name onto the two output trees.

    Pipeline records and generated pages are written to different roots by
    design, so a run is normally identified by a NAME and the two directories are
    derived from it. An absolute path is also accepted as a direct locator for
    the records tree, which keeps `--publish-only --output <existing dir>` working
    on runs created before this split.
    """
    name = Path(name)
    if run_dir is None and (name.is_absolute() or len(name.parts) > 1):
        run_dir = name
        if site_dir is None:
            site_dir = name
    run_dir = Path(run_dir).expanduser().resolve() if run_dir else app_config.ARTIFACTS_ROOT / 'experiments' / name
    site_dir = Path(site_dir).expanduser().resolve() if site_dir else app_config.SITE_ROOT / 'experiments' / name
    return run_dir, site_dir



def process_window(data, state, first, cutoff, args, prompt, out, client):
    """Run one window transaction, leaving the supplied checkpoint state untouched on failure."""
    tools=Tools(data,state,cutoff)
    content=build_window_content(data,tools,first,cutoff,args.max_calls)
    messages=[{'role':'system','content':prompt},{'role':'user','content':content}]
    committed=False;tick=time.monotonic();run_tag=str(time.time_ns())
    for step in range(args.max_calls):
        if step==args.max_calls-2:
            # The count is in rounds, and one round may carry a single tool call.
            # A model that spent its last round on propose_update returned a valid
            # proposal id and was then killed with no round left for commit_update,
            # losing the whole window. Say how many rounds remain and that both calls
            # are needed.
            messages.append({'role':'user','content':
                f'Budget nearly exhausted: you have {args.max_calls-step} rounds left. '
                'Submit a complete valid proposal now - hold what you can, discard the rest in '
                'entries of at most four observations - and then call commit_update with the '
                'proposal id. Two calls, both required: a proposal that is never committed '
                'loses the whole window.'})
            # From here on there is no room to inspect a held object and resubmit,
            # so the re-identification gate stands down for the rest of the window.
            tools.gate_relaxed=True
        tag=f'w{first:04d}_{run_tag}_call{step:02d}'
        for recovery in range(args.window_network_retries + 1):
            try:
                response=client.call(messages,TOOLS,tag,final=step==args.max_calls-2 and tools.pending is None)
                break
            except GatewayNetworkError as exc:
                exhausted=recovery == args.window_network_retries
                save(out/'network_recovery'/f'{tag}_recovery{recovery}.json', {
                    'first': first, 'cutoff': cutoff, 'call': step+1,
                    'checkpoint_cutoff': state['cutoff'], 'exhausted': exhausted,
                    'error': str(exc), 'delay_seconds': None if exhausted else 30})
                if exhausted:
                    raise RuntimeError(
                        f'WINDOW {first}-{cutoff} call {step+1}: network recovery exhausted; '
                        f'checkpoint preserved at cutoff {state["cutoff"]}: {exc}') from None
                print(f'WINDOW NETWORK RECOVERY {recovery+1}/{args.window_network_retries} '
                      f'for {first}-{cutoff} call {step+1}; waiting 30s; '
                      'keeping conversation and checked proposal, retrying only this request', flush=True)
                time.sleep(30)

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
            outcome=('ERROR '+result['error'] if 'error' in result else
                     'HELD '+','.join(sorted({c['object_id'] for c in result.get('checks',())}))
                     if result.get('status')=='verification_required' else 'OK')
            print(f'WINDOW {first}-{cutoff} call {step+1}: {name} {outcome}',flush=True)
            if committed:break
        messages.extend(evidence_messages)
        if committed:break
    if not committed:raise RuntimeError('Window budget exhausted without valid commit; prior checkpoint preserved')
    return state, {'elapsed_seconds':time.monotonic()-tick,'calls':step+1}


def publish_completed(run_dir, site_dir=None):
    """Rebuild completed results without model calls or inference-config changes."""
    import fcntl
    run_dir = Path(run_dir).resolve()
    if not (run_dir/'config.json').is_file():
        raise ValueError('Cannot publish: config.json is missing')
    with (run_dir/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=json.loads((run_dir/'config.json').read_text())
        if site_dir is None:
            # The run recorded where its pages belong; reuse that unless overridden.
            site_dir = Path(config.get('site_dir', run_dir))
        site_dir = Path(site_dir).resolve()
        if not (run_dir/'checkpoint.json').is_file():
            raise ValueError('Cannot publish: checkpoint.json is missing')
        checkpoint=json.loads((run_dir/'checkpoint.json').read_text())
        if checkpoint.get('cutoff') != config['frames']-1:
            raise ValueError(f'Cannot publish: task is incomplete at cutoff {checkpoint.get("cutoff")}; '
                             f'finish all {config["frames"]} frames first')
        state=json.loads((run_dir/'versions'/f'{checkpoint["version"]:04d}'/'state.json').read_text())
        if state['cutoff'] != checkpoint['cutoff'] or state['version'] != checkpoint['version']:
            raise ValueError('Checkpoint and committed state disagree')
        print(f'PUBLISHING completed results from {run_dir}; no model calls',flush=True)
        data=Observations(resolve_recorded_path(config['source'], run_dir),config['frames'],run_dir)
        publish(data,run_dir,site_dir,config['voxel'])
        print(f'PUBLISHED {site_dir / "index.html"}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--output',type=Path,required=True,
                        help='Experiment NAME (not a path), e.g. harness-v3-sol-20frames-YYYYMMDD-HHMM; '
                             'records go to <artifacts>/experiments/<name>, pages to <site>/experiments/<name>')
    parser.add_argument('--run-dir',type=Path,default=None,
                        help='Override the pipeline-records directory (default: <artifacts>/experiments/<name>)')
    parser.add_argument('--site-dir',type=Path,default=None,
                        help='Override the generated-pages directory (default: <site>/experiments/<name>)')
    parser.add_argument('--frames',type=int,default=30);parser.add_argument('--window',type=int,default=5)
    parser.add_argument('--max-calls',type=int,default=14);parser.add_argument('--voxel',type=float,default=.02)
    parser.add_argument('--timeout',type=float,default=120, help='Per-request gateway read timeout in seconds')
    parser.add_argument('--network-retries',type=int,default=5,
                        help='Retries after transient gateway errors for each request (default: 5)')
    parser.add_argument('--window-network-retries',type=int,default=2,
                        help='Recovery batches for the same failed request; preserve window context (default: 2)')
    parser.add_argument('--retry-backoff',type=float,default=2.0,
                        help='Initial seconds for exponential network retry backoff (default: 2)')
    parser.add_argument('--model',default='openai/gpt-6-sol')
    parser.add_argument('--reasoning-effort',choices=['none','low','medium','high'],default=None,
                        help='Default: none for GPT-6 Sol; Astra requires low or higher')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only',action='store_true')
    mode.add_argument('--skip-publish',action='store_true', help='Save results only; build pages later with --publish-only')
    mode.add_argument('--publish-only',action='store_true',
                        help='Build the visualization once from a completed output; does not call the model')
    args=parser.parse_args()
    if args.publish_only:
        # Prefer the site directory recorded in the run's own config.json; only an
        # explicit --site-dir overrides it. That keeps re-publishing a completed
        # run idempotent without re-stating where its pages belong.
        if args.run_dir is not None or args.site_dir is not None:
            run_dir, site_dir = resolve_output(args.output, args.run_dir, args.site_dir)
        else:
            run_dir = Path(args.output).expanduser().resolve()
            site_dir = None
        publish_completed(run_dir, site_dir)
        return
        return
    is_astra=args.model.rsplit("/",1)[-1].startswith("gpt-6-astra")
    args.reasoning_effort=args.reasoning_effort or ("low" if is_astra else "none")
    if is_astra and args.reasoning_effort=="none":
        parser.error("GPT-6 Astra requires reasoning effort low or higher")
    if not is_astra and args.reasoning_effort!="none":
        parser.error("Non-Astra harness tool calls currently require reasoning effort none")
    manifest_path = args.source / 'manifest.json'
    try:
        source_manifest = json.loads(manifest_path.read_text())
        manifest_frame_count = len(source_manifest['frames'])
        lifting_policy = source_manifest.get('lifting_policy')
        if lifting_policy not in ACCEPTED_LIFTING_POLICIES:
            parser.error(
                f'Harness requires lifting_policy in {ACCEPTED_LIFTING_POLICIES}; '
                f'got {lifting_policy!r} from {args.source}. Build the source with '
                'python -m src.geometry.amb3r_mask_source, which lifts masks onto '
                'AMB3R per-frame point maps.'
            )
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(f'Invalid source manifest: {exc}')
    if (not 1<=args.frames<=manifest_frame_count or not 1<=args.window<=args.frames or args.max_calls<4 or
            args.voxel<=0 or args.timeout<=0 or args.network_retries<0 or args.window_network_retries<0 or args.retry_backoff<0 or
            not all(math.isfinite(v) for v in (args.voxel,args.timeout,args.retry_backoff))):
        parser.error(f'Invalid limits: frames must be in 1..{manifest_frame_count}')
    run_dir,site_dir=resolve_output(args.output,args.run_dir,args.site_dir)
    run_dir.mkdir(parents=True,exist_ok=True)
    out=run_dir
    # One process per run. Advisory lock automatically releases after a crash.
    import fcntl
    lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(out/'code_versions'/f'{time.time_ns()}.json', {str(p.relative_to(app_config.ROOT)):digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
    prompt=Path(__file__).with_name('prompt.txt').read_text()
    # Say what geometry the model actually saw; a source must never be described as
    # the projection it does not use. This field participates in the resume
    # comparison below, so it is derived from the source, not from anything
    # run-specific.
    geometry_policy = {
        AMB3R_FRAME_POLICY:
            'AMB3R per-frame point maps, Sim(3)-corrected to the GT-camera frame; '
            'mask pixels unprojected onto the network-predicted surface their own '
            'pixel sees (one point per retained mask pixel); incremental identity '
            'association only; no GT instance labels supplied',
    }.get(lifting_policy)
    if geometry_policy is None:
        parser.error(f'No geometry description for lifting_policy {lifting_policy!r}; '
                     'add one so the audit trail cannot misdescribe the source')
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
            if k not in {'prepare_only','publish_only','skip_publish','run_dir','site_dir'}}
    # Record where the two output trees actually are. These participate in the
    # resume comparison below, so a run cannot silently resume into a different
    # pair of directories.
    config.update(run_dir=str(run_dir),site_dir=str(site_dir))
    config.update(input_policy='history_on_demand_rgb_bbox_v2_spatial_top3_global_obb',prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),manifest_sha256=digest(args.source/'manifest.json'),
                  geometry_policy=geometry_policy)
    if (out/'config.json').exists():
        existing=json.loads((out/'config.json').read_text())
        # New retry/publishing controls are operational settings. Permit a
        # legacy output to resume while still rejecting changed data/model policy.
        operational={'network_retries','window_network_retries','retry_backoff'}
        if ({k:v for k,v in existing.items() if k not in operational} !=
                {k:v for k,v in config.items() if k not in operational}):
            raise ValueError('Resume config changed; use new output')
    save(out/'config.json',config)
    save(out/'run_options'/f'{time.time_ns()}.json', {
        'network_retries':args.network_retries, 'window_network_retries':args.window_network_retries,
        'retry_backoff':args.retry_backoff, 'publish_policy':'manual' if args.skip_publish else 'on_completion'})
    print(f'PREPARING {args.frames} frames ({math.ceil(args.frames/args.window)} windows); '
          f'model={args.model}; output={out}',flush=True)
    prior_audit=json.loads((out/'input_audit.json').read_text()) if (out/'input_audit.json').exists() else None
    data=Observations(args.source,args.frames,out)
    if prior_audit and prior_audit!=data.audit:raise ValueError('Input changed; cannot resume')
    for seq in range(args.frames):
        if seq % 10 == 0:
            print(f'PREPARE EVIDENCE frame {seq+1}/{args.frames}',flush=True)
        data.view(seq)
        for oid,o in data.items.items():
            if o['seq']==seq:data.view(seq,oid)
    print(f'PREPARED {len(data.items)} observations, {args.frames} sampled frames',flush=True)
    if args.prepare_only:return
    checkpoint_path=out/'checkpoint.json'
    client=Client(args.model,out,timeout=args.timeout,reasoning_effort=args.reasoning_effort,
                  network_retries=args.network_retries,retry_backoff=args.retry_backoff)
    checkpoint=json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {'version':0}
    state=json.loads((out/'versions'/f'{checkpoint["version"]:04d}'/'state.json').read_text()) if checkpoint['version'] else initial()
    for first in range(state['cutoff']+1,args.frames,args.window):
        cutoff=min(first+args.window,args.frames)-1
        print(f'WINDOW {state["version"]+1}/{math.ceil(args.frames/args.window)} START: frames {first}-{cutoff}',flush=True)
        state,timing=process_window(data,state,first,cutoff,args,prompt,out,client)
        save(out/'versions'/f'{state["version"]:04d}'/'timing.json',timing)
        print(f'PROGRESS {state["version"]}/{math.ceil(args.frames/args.window)} windows; '
              f'{cutoff+1}/{args.frames} frames committed; {len(state["objects"])} objects; '
              f'{timing["elapsed_seconds"]:.1f}s',flush=True)
    metas=[json.loads(p.read_text()) for p in (out/'calls').glob('*_meta.json')]
    usage={k:sum((m.get('usage') or {}).get(k,0) or 0 for m in metas) for k in ('prompt_tokens','completion_tokens','total_tokens')}
    # A window that failed after spending calls is retried under a fresh run_tag, so
    # one window can hold several sessions in `calls/`. Counting every meta would
    # report the retries as if they had contributed to the committed state; report
    # the split instead of one ambiguous total.
    sessions={}
    for p in (out/'calls').glob('*_meta.json'):
        match=re.match(r'w(\d+)_(\d+)_call',p.name)
        if match:sessions.setdefault(int(match.group(1)),set()).add(match.group(2))
    retried={w:sorted(s) for w,s in sessions.items() if len(s)>1}
    committed_calls=abandoned_calls=0
    for first,sids in sessions.items():
        counts=sorted(len(list((out/'calls').glob(f'w{first:04d}_{sid}_call*_meta.json'))) for sid in sids)
        committed_calls+=counts[-1];abandoned_calls+=sum(counts[:-1])
    summary={'completed':state['cutoff']==args.frames-1,'frames':args.frames,'windows':state['version'],'objects':len(state['objects']),
             'observations':len(state['assignments']),'discarded':sum(a['status']=='discard' for a in state['assignments'].values()),
             'excluded':sum(a['status']=='exclude' for a in state['assignments'].values()),
             'api_successful_calls':len(metas),'api_calls_in_committed_windows':committed_calls,
             'api_calls_in_abandoned_attempts':abandoned_calls,'retried_windows':retried,
             'usage':usage,'cost_currency':None,'cost_note':'Gateway price not configured; tokens recorded, no fabricated cost',
             'accuracy_evaluated':False,'geometry_policy':config['geometry_policy']}
    save(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
    if args.skip_publish:
        print('PROCESSING COMPLETE; pages not built (--skip-publish). Use --publish-only --output to build later.',flush=True)
    else:
        print('PROCESSING COMPLETE; PUBLISHING final visualization once',flush=True)
        publish(data,out,site_dir,args.voxel)
        print(f'PUBLISHED {site_dir / "index.html"}',flush=True)


if __name__=='__main__':main()
