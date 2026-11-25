"""M5 existing-run appearance workflow, fingerprinted recovery and verified delivery selection."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import yaml
from .baseline import ROOT
from .contracts import atomic_json, digest, json_digest, resolve_inputs, run_lock, validate_config
from .delivery import deliver
from .review import candidate_snapshot
from .texturing import build


def verified_delivery(run, delivery, revision=None):
    run,delivery=Path(run).resolve(),Path(delivery).resolve()
    parent=run/'physics/appearance'
    if not delivery.is_relative_to(parent) or delivery.parent.parent!=parent:
        raise ValueError('Delivery must belong to this run')
    report=json.loads((delivery/'report.json').read_text())
    if not all(report.get(k) is True for k in ('passed','physics_preserved','isaac_display_verified','drop_verified','texture_sampled')) or report.get('status')!='success':
        raise ValueError('Delivery verification failed')
    name=report['source_revision']
    if delivery.parent.name!=name or (revision is not None and revision!=name): raise ValueError('Delivery revision mismatch')
    snapshot=candidate_snapshot(run,run/'usd/appearance'/name)
    if not snapshot['complete'] or digest(run/'usd/appearance'/name/'object.usda')!=report['candidate_sha256']:
        raise ValueError('Candidate differs from delivered version')
    if report.get('candidate_snapshot') and snapshot!=report['candidate_snapshot']:
        raise ValueError('Delivered candidate fingerprint changed')
    if digest(run/'physics/object.usda')!=report['baseline_physics_sha256']:
        raise ValueError('Baseline physics changed')
    hashes=report.get('output_sha256')
    if not hashes or 'asset/object.usda' not in hashes: raise ValueError('Missing delivery hashes')
    for relative,expected in hashes.items():
        path=(delivery/relative).resolve()
        if not path.is_relative_to(delivery) or not path.is_file() or digest(path)!=expected:
            raise ValueError('Delivery dependency changed: '+relative)
    return {'revision':name,'delivery':str(delivery),'asset':str(delivery/'asset/object.usda'),
            'report':str(delivery/'report.json'),'report_sha256':digest(delivery/'report.json'),
            'candidate':snapshot}


def selected_asset(run):
    """Technical selection is distinct from historical human review selection.json."""
    run=Path(run).resolve()
    record=json.loads((run/'usd/appearance/delivery_selection.json').read_text())
    if record.get('mode')!='technical_verified' or record.get('human_approved') is not False:
        raise ValueError('Unsupported selection policy')
    current=verified_delivery(run,record['delivery'],record['revision'])
    if current!=record['verification']: raise ValueError('Selection fingerprint changed')
    return Path(current['asset'])


def execute(run, config, isaac_env, revision=None, resume=None, feedback=True):
    run=Path(run).resolve();isaac_env=Path(isaac_env).resolve()
    inputs=resolve_inputs(run)
    if 'physics_object.usda' not in inputs: raise ValueError('Run stage 5 first; baseline physics is required')
    tools={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
    contract={'inputs':inputs,'config':config,'tool_sha256':tools,'blender_sha256':digest(config['blender']),
              'isaac_env':str(isaac_env),'isaac_env_sha256':digest(isaac_env),'feedback':feedback,'requested_revision':revision}
    parent=run/'usd/appearance'
    if resume:
        folder=Path(resume).resolve()
        if folder.parent!=parent or not folder.name.startswith('m5_'): raise ValueError('Invalid recovery directory')
        report=json.loads((folder/'report.json').read_text())
        if report['contract']!=contract: raise ValueError('Recovery inputs/config/tools changed; create a new task')
    else:
        folder=parent/('m5_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));folder.mkdir(parents=True)
        report={'schema_version':1,'milestone':'M5','status':'running','contract':contract,'stages':{},
                'run_dir':str(run),'human_approved':False,'selection_policy':'technical_verified','view_mode':'display_only'}
    def save(): atomic_json(folder/'report.json',report)
    print('APPEARANCE_TASK',folder,flush=True)
    # Task lock prevents concurrent recovery of the same ledger; component operations own the run lock.
    import fcntl
    with (folder/'.task.lock').open('a+') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('This appearance task is already running') from None
        try:
            report.update(status='running');save()
            if 'candidate' in report:
                candidate=parent/report['candidate'];candidate_snapshot(run,candidate)
            elif revision:
                candidate=parent/revision
                if not candidate_snapshot(run,candidate)['complete']: raise ValueError('Revision has not passed technical checks')
                report['candidate']=candidate.name;save()
            else:
                before=set(parent.glob('v_*'));report['stages']['texturing']={'status':'running'};save()
                try: candidate=build(run,config)
                finally:
                    created=sorted(set(parent.glob('v_*'))-before)
                    if created: report['attempted_revision']=created[-1].name;save()
                report['candidate']=candidate.name;report['stages']['texturing']={'status':'success'};save()
            if feedback:
                stage=report['stages'].get('feedback',{})
                if stage.get('status') in ('running','failed'):
                    raise ValueError('Interrupted API feedback is not replayed; start a new task with --revision and --skip-feedback')
                if stage.get('status')!='success':
                    report['stages']['feedback']={'status':'running'};save()
                    from .feedback import run_feedback
                    output=run_feedback(run,candidate,config)
                    result=json.loads((output/'report.json').read_text())
                    report['stages']['feedback']={'status':'failed' if result.get('error') else 'success','report':str(output/'report.json'),'report_sha256':digest(output/'report.json')};save()
                    if result.get('error'): raise RuntimeError('Feedback failed; see '+str(output/'report.json'))
                    candidate=parent/result['best_revision'];candidate_snapshot(run,candidate)
                    report['candidate']=candidate.name;save()
                else:
                    path=Path(stage['report'])
                    if digest(path)!=stage['report_sha256']: raise ValueError('Feedback report changed')
                    result=json.loads(path.read_text());candidate=parent/result['best_revision']
                    candidate_snapshot(run,candidate);report['candidate']=candidate.name;save()
            saved=report.get('delivery')
            if saved:
                verification=verified_delivery(run,saved,candidate.name)
            else:
                report['stages']['delivery']={'status':'running'};save()
                delivery=deliver(run,candidate.name,isaac_env)
                verification=verified_delivery(run,delivery,candidate.name)
                report['delivery']=str(delivery);report['stages']['delivery']={'status':'success'};save()
            with run_lock(run):
                # Recheck immediately before recording the asset for consumers.
                verification=verified_delivery(run,verification['delivery'],candidate.name)
                atomic_json(parent/'delivery_selection.json',{'schema_version':1,'mode':'technical_verified',
                    'human_approved':False,'revision':candidate.name,'delivery':verification['delivery'],
                    'verification':verification,'created_at':datetime.now().astimezone().isoformat()})
            report.update(status='success',passed=True,asset=verification['asset'],candidate_snapshot=verification['candidate'],
                          selection=str(parent/'delivery_selection.json'),delivery_report=verification['report'])
        except BaseException as exc:
            report.update(status='cancelled' if isinstance(exc,KeyboardInterrupt) else 'failed',passed=False,error=f'{type(exc).__name__}: {exc}')
            raise
        finally: save()
    print('APPEARANCE_SUCCESS',report['asset'],flush=True)
    print('APPEARANCE_TASK_REPORT',folder/'report.json',flush=True)
    return folder, report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/appearance/default.yaml')
    p.add_argument('--isaac-env',type=Path,default=ROOT.parent.parent/'robot_sim/scripts/activate_env.sh')
    p.add_argument('--revision',help='Reuse a technically valid candidate instead of rebaking')
    p.add_argument('--resume',type=Path,help='Recover an m5 task with unchanged sources/config/tools')
    p.add_argument('--skip-feedback',action='store_true',help='Use deterministic photo projection without GPT API calls')
    a=p.parse_args()
    try:
        _,report=execute(a.run_dir,validate_config(yaml.safe_load(a.config.read_text())),a.isaac_env,a.revision,a.resume,not a.skip_feedback)
        atomic_json(a.run_dir.resolve()/'usd/appearance/pipeline.json',report)
    except (Exception,KeyboardInterrupt) as exc:
        print('APPEARANCE_FAILED',f'{type(exc).__name__}: {exc}',file=sys.stderr);return 130 if isinstance(exc,KeyboardInterrupt) else 1
    return 0


if __name__=='__main__': raise SystemExit(main())
