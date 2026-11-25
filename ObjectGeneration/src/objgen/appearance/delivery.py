"""M4 independent physical package, relocated Isaac render/ablation/drop verification."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import sys
from PIL import Image, ImageChops
from .baseline import ROOT
from .contracts import atomic_json, digest, resolve_inputs, run_lock, valid_stage
from .review import candidate_snapshot


def isaac_command(env,script,*args):
    return ['bash','-c','set -e; source "$1"; shift; exec taskset -c "${SIM_CPU_AFFINITY:-16-23}" python -u -X faulthandler "$@"',
            'appearance-isaac',str(env),str(script),*map(str,args)]


def deliver(run,revision_id,isaac_env):
    with run_lock(run):
        source=(run/'usd/appearance'/revision_id).resolve()
        if source.parent!=(run/'usd/appearance').resolve(): raise ValueError('Invalid revision')
        report=json.loads((source/'report.json').read_text())
        if not report.get('technical_checks_passed') or not report.get('stages') or not all(valid_stage(s,source) for s in report['stages'].values()):
            raise ValueError('Invalid candidate artifacts')
        inputs=resolve_inputs(run); manifest=json.loads((source/'manifest.json').read_text())
        if inputs!=manifest['inputs']: raise ValueError('Candidate sources changed')
        if 'physics_object.usda' not in inputs: raise ValueError('Verified baseline physical USD required')
        delivery=run/'physics/appearance'/revision_id/('m4_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        delivery.mkdir(parents=True)
        result={'milestone':'M4','passed':False,'source_revision':revision_id,'status':'packaging',
                'review_scope':'single_photo_visible_side; technical verification only; no human approval inferred',
                'human_approved':False,
                'candidate_snapshot':candidate_snapshot(run,source),
                'candidate_sha256':digest(source/'object.usda'),'baseline_physics_sha256':digest(run/'physics/object.usda'),
                'tool_sha256':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}}
        atomic_json(delivery/'report.json',result)
        print('APPEARANCE_DELIVERY',delivery,flush=True)
        env=dict(os.environ,OPENBLAS_CORETYPE='Haswell',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        def command(name,args,timeout=240):
            print('APPEARANCE_M4_STAGE',name,flush=True)
            with (delivery/(name+'.log')).open('w') as log: subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout,env=env)
        hidden=source.with_name('.m4_hidden_'+source.name)
        temp=Path(tempfile.mkdtemp(prefix='appearance_m4_relocated_'))
        try:
            command('package',['taskset','-c','0',str(ROOT/'.venv-sam3d/bin/python'),str(Path(__file__).parent/'package_asset.py'),str(source/'object.usda'),str(run/'physics/object.usda'),str(delivery/'asset')])
            shutil.copytree(delivery/'asset',temp/'asset')
            # Keep the actual output package unavailable too, proving no fallback resolution.
            hidden_package=delivery/'asset_hidden'
            atomic_json(delivery/'relocation_journal.json',{'source':str(source),'hidden_source':str(hidden),'package':str(delivery/'asset'),'hidden_package':str(hidden_package),'relocated':str(temp),'restored':False})
            source.rename(hidden)
            try:
                (delivery/'asset').rename(hidden_package)
                try:
                    assert not source.exists() and not (delivery/'asset').exists()
                    command('render',isaac_command(isaac_env,ROOT/'src/objgen/verify/check_usd_isaacsim.py',temp/'asset/object.usda','--output',temp/'render','--physics-preview'))
                    render=json.loads((temp/'render/isaacsim_verification.json').read_text()); assert render['passed']
                    shutil.copytree(temp/'asset',temp/'ablation')
                    for texture in (temp/'ablation/textures').iterdir():
                        with Image.open(texture) as im: size=im.size
                        Image.new('RGB',size,(255,0,255)).save(texture)
                    command('ablation',isaac_command(isaac_env,ROOT/'src/objgen/verify/check_usd_isaacsim.py',temp/'ablation/object.usda','--output',temp/'ablation_render','--physics-preview'))
                    with Image.open(temp/'render/object_isaacsim.png') as a,Image.open(temp/'ablation_render/object_isaacsim.png') as b:
                        diff=ImageChops.difference(a.convert('RGB'),b.convert('RGB'))
                        channels=diff.split()
                        maximum=ImageChops.lighter(ImageChops.lighter(channels[0],channels[1]),channels[2])
                        changed=sum(maximum.histogram()[21:])
                    assert changed>1000,f'Texture ablation had insufficient effect: {changed} pixels'
                    command('drop',isaac_command(isaac_env,ROOT/'src/objgen/verify/check_usd_physics_isaacsim.py',temp/'asset/object.usda','--output',temp/'drop'))
                    drop=json.loads((temp/'drop/isaacsim_physics_verification.json').read_text()); assert drop['passed']
                    result.update(relocated_source_hidden=True,texture_sampled=True,ablation_changed_pixels=changed,
                                  authored_mass_kg=drop['authored_mass_kg'],runtime_mass_kg=drop['runtime_mass_kg'])
                finally:
                    if hidden_package.exists(): hidden_package.rename(delivery/'asset')
            finally:
                if hidden.exists(): hidden.rename(source)
                journal=json.loads((delivery/'relocation_journal.json').read_text()); journal['restored']=True; atomic_json(delivery/'relocation_journal.json',journal)
            for name in ('render','ablation_render','drop'): shutil.copytree(temp/name,delivery/name)
            # Stage.Export preserves relative texture paths; viewing scenes need their own dependency folder.
            shutil.copytree(temp/'asset/textures',delivery/'render/textures')
            shutil.copytree(temp/'ablation/textures',delivery/'ablation_render/textures')
            command('localize_views',['taskset','-c','0',str(ROOT/'.venv-sam3d/bin/python'),str(Path(__file__).parent/'localize_view.py'),str(delivery/'render/object_view.usda'),str(delivery/'ablation_render/object_view.usda')])
            assert resolve_inputs(run)==inputs
            result.update(status='success',passed=True,physics_preserved=True,isaac_display_verified=True,drop_verified=True,
                          package='asset/object.usda',relocated_test_directory=str(temp),
                          output_sha256={str(p.relative_to(delivery)):digest(p) for p in delivery.rglob('*') if p.is_file() and p.name not in ('report.json','relocation_journal.json')})
        except BaseException as exc:
            result.update(status='cancelled' if isinstance(exc,KeyboardInterrupt) else 'failed',error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            atomic_json(delivery/'report.json',result)
        print('APPEARANCE_M4_REPORT',delivery/'report.json',flush=True)
        return delivery


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True,type=Path);p.add_argument('--revision',required=True)
    p.add_argument('--isaac-env',type=Path,default=ROOT.parent.parent/'robot_sim/scripts/activate_env.sh')
    a=p.parse_args()
    try: deliver(a.run_dir.resolve(),a.revision,a.isaac_env.resolve())
    except Exception as exc: print('APPEARANCE_M4_FAILED',str(exc),file=sys.stderr);return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
