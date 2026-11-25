"""M2 bottle photo projection: immutable candidate, geometry-preserving textured USD."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml
from PIL import Image, ImageDraw
from .baseline import ROOT, execute_stage, prepare_images
from .contracts import allocate_revision, atomic_json, digest, resolve_inputs, run_lock, validate_config


def comparison(revision):
    canvas=Image.new('RGB',(1100,750),'#dddddd'); draw=ImageDraw.Draw(canvas)
    for i,(label,name) in enumerate([('Reference photo','reference_isolated.png'),('Original vertex colours','renders/aligned_baseline.png'),('Photo projection','renders/aligned_refined.png'),('Silhouette overlay','renders/alignment.png')]):
        with Image.open(revision/name) as raw:
            image=raw.convert('RGB'); image.thumbnail((260,690))
        canvas.paste(image,(i*275+(275-image.width)//2,40)); draw.text((i*275+10,10),label,fill='black')
    canvas.save(revision/'renders/comparison.png')


def build(run, config, camera=None, deadline=None):
    with run_lock(run):
        inputs=resolve_inputs(run); revision=allocate_revision(run)
        for name in ('textures','renders'): (revision/name).mkdir()
        source=Path(__file__).parent
        atomic_json(revision/'config.json',config)
        atomic_json(revision/'manifest.json',{'schema_version':1,'milestone':'M2','run_dir':str(run),'revision':revision.name,'created_at':datetime.now().astimezone().isoformat(),'inputs':inputs,'tool_sha256':{p.name:digest(p) for p in source.glob('*.py')},'blender_sha256':digest(config['blender'])})
        if camera:
            # Capture the override, so a mutable external file cannot change this candidate.
            atomic_json(revision/'camera_override.json',json.loads(camera.read_text()))
        report={'milestone':'M2','revision':revision.name,'status':'pending','passed':False,'stages':{}}
        print('APPEARANCE_REVISION',revision,flush=True)
        env=dict(os.environ,OPENBLAS_CORETYPE='Haswell',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        python=['taskset','-c','0',str(ROOT/'.venv-sam3d/bin/python')]
        def command(name,args,timeout=300):
            if deadline is not None:
                remaining=deadline-time.monotonic()
                if remaining<=0: raise TimeoutError('Appearance wall budget exhausted')
                timeout=min(timeout,remaining)
            with (revision/(name+'.log')).open('w') as log:
                subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
        try:
            execute_stage(report,revision,'geometry',lambda:command('geometry',python+[str(source/'extract_geometry.py'),inputs['usd']['path'],str(revision)]),['canonical_mesh.npz','geometry.json'])
            execute_stage(report,revision,'reference',lambda:prepare_images(inputs,revision),['reference_crop.png','reference_mask.png','reference_isolated.png','reference.json'])
            execute_stage(report,revision,'alignment',lambda:command('alignment',python+[str(source/'camera_fit.py'),str(revision)]+([str(revision/'camera_override.json')] if camera else [])),['camera.json','renders/alignment.png'])
            views=['front','back','left','right','top','three_quarter']
            execute_stage(report,revision,'texturing',lambda:command('texturing',[config['blender'],'--background','--factory-startup','--python-exit-code','1','--python',str(source/'texture_blender.py'),'--',str(revision)],1200),['work.blend','uv.npy','observed_faces.npy','textures/basecolor.png','textures/photo_coverage.png','projection.json','renders/aligned_baseline.png','renders/aligned_refined.png',*[f'renders/refined_{v}.png' for v in views]])
            execute_stage(report,revision,'usd',lambda:command('usd',python+[str(source/'usd_appearance.py'),str(revision)]),['object.usda','usd_validation.json'])
            execute_stage(report,revision,'comparison',lambda:comparison(revision),['renders/comparison.png'])
            if resolve_inputs(run)!=inputs: raise RuntimeError('Source inputs changed')
            report.update(status='awaiting_review',passed=True,technical_checks_passed=True,human_approved=False,isaac_verified=False)
        except BaseException as exc:
            report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            if (revision/'camera.json').exists() and not json.loads((revision/'camera.json').read_text()).get('alignment_accepted',False): report['status']='needs_alignment'
            raise
        finally: atomic_json(revision/'report.json',report)
        return revision


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/appearance/default.yaml')
    parser.add_argument('--camera',type=Path,help='JSON camera field overrides captured in candidate')
    args=parser.parse_args()
    try: revision=build(args.run_dir.resolve(),validate_config(yaml.safe_load(args.config.read_text())),args.camera)
    except Exception as exc:
        print('APPEARANCE_TEXTURING_FAILED',str(exc),file=sys.stderr); return 1
    print('APPEARANCE_TEXTURING_REPORT',revision/'report.json'); return 0


if __name__=='__main__': raise SystemExit(main())
