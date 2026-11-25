"""Ordered multiview photo refinement with cumulative candidates and verified delivery."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import yaml
from PIL import Image, ImageDraw
from .api_client import APIClient, image_content
from .baseline import ROOT, execute_stage, prepare_images
from .contracts import allocate_revision, atomic_json, digest, resolve_inputs, run_lock, validate_config
from .texturing import comparison
from .orchestrator import execute

ORDER=('0','90','180','270','up','down')

def discover(folder):
    found={}
    for p in Path(folder).iterdir():
        if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png','.webp') and p.stem.lower() in ORDER:
            key=p.stem.lower()
            if key in found: raise ValueError('Duplicate view: '+key)
            found[key]=p.resolve()
    if '0' not in found: raise ValueError('A front photograph named 0 is required')
    return [(angle,found[angle]) for angle in ORDER if angle in found]

def sheet(revision,views):
    canvas=Image.new('RGB',(len(views)*480,680),'#dddddd');d=ImageDraw.Draw(canvas)
    for i,v in enumerate(views):
        d.text((i*480+12,10),'Angle '+v['angle']+' | reference / cumulative model',fill='black')
        for j,name in enumerate([v['folder']+'/reference_isolated.png','renders/view_'+v['angle']+'.png']):
            im=Image.open(revision/name).convert('RGB');im.thumbnail((235,640));canvas.paste(im,(i*480+j*240,35))
    canvas.save(revision/'renders/multiview_comparison.png')

def build(run,config,views):
    with run_lock(run):
        inputs=resolve_inputs(run);revision=allocate_revision(run);source=Path(__file__).parent
        for n in ('textures','renders','views'): (revision/n).mkdir()
        atomic_json(revision/'config.json',config)
        atomic_json(revision/'manifest.json',{'schema_version':1,'milestone':'multiview','revision':revision.name,'run_dir':str(run),'inputs':inputs,'tool_sha256':{p.name:digest(p) for p in source.glob('*.py')},'blender_sha256':digest(config['blender'])})
        report={'milestone':'multiview','revision':revision.name,'status':'running','passed':False,'stages':{}}
        env=dict(os.environ,OPENBLAS_CORETYPE='Haswell',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        python=['taskset','-c','0',str(ROOT/'.venv-sam3d/bin/python')]
        def command(name,args,timeout=1200):
            with (revision/(name+'.log')).open('w') as log: subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
        try:
            execute_stage(report,revision,'geometry',lambda:command('geometry',python+[str(source/'extract_geometry.py'),inputs['usd']['path'],str(revision)]),['canonical_mesh.npz','geometry.json'])
            records=[];files=[]
            def reference():
                for angle,ref in views:
                    dest=revision/'views'/angle;shutil.copytree(ref,dest)
                    records.append({'angle':angle,'folder':'views/'+angle})
                atomic_json(revision/'views.json',records)
                for p in (revision/'views').rglob('*'):
                    if p.is_file():files.append(str(p.relative_to(revision)))
            execute_stage(report,revision,'reference',reference,files)
            execute_stage(report,revision,'alignment',lambda:command('alignment',python+[str(source/'multiview_geometry.py'),str(revision)]),['views.json',*[f"views/{v['angle']}/{n}" for v in records for n in ('camera.json','alignment.png')]])
            for n in ('camera.json','reference_crop.png','reference_mask.png','reference_isolated.png','reference.json'):
                shutil.copy2(revision/'views/0'/n,revision/n)
            shutil.copy2(revision/'views/0/alignment.png',revision/'renders/alignment.png')
            def atlas():
                size=config['texture_size'];a=Image.new('RGB',(size,size),(128,128,128))
                for i,v in enumerate(records):
                    x0=round((i%3+.02)/3*size);x1=round((i%3+.98)/3*size)
                    y0=round((i//3+.02)/2*size);y1=round((i//3+.98)/2*size)
                    im=Image.open(revision/v['folder']/'reference_crop.png').convert('RGB').resize((x1-x0,y1-y0),Image.Resampling.LANCZOS);a.paste(im,(x0,y0))
                a.save(revision/'textures/basecolor.png')
            execute_stage(report,revision,'atlas',atlas,['textures/basecolor.png'])
            execute_stage(report,revision,'texturing',lambda:command('texturing',['taskset','-c','0',config['blender'],'--background','--factory-startup','--python-exit-code','1','--python',str(source/'multiview_blender.py'),'--',str(revision)]),['work.blend','uv.npy','observed_faces.npy','face_sources.npy','projection.json','renders/aligned_baseline.png','renders/aligned_refined.png',*[f'renders/refined_{v}.png' for v in ('front','back','left','right','top','three_quarter')],*[f"renders/view_{v['angle']}.png" for v in records]])
            execute_stage(report,revision,'usd',lambda:command('usd',python+[str(source/'usd_appearance.py'),str(revision)]),['object.usda','usd_validation.json'])
            def comparisons(): comparison(revision);sheet(revision,records)
            execute_stage(report,revision,'comparison',comparisons,['renders/comparison.png','renders/multiview_comparison.png','camera.json','reference_isolated.png','renders/alignment.png'])
            if resolve_inputs(run)!=inputs:raise RuntimeError('Baseline inputs changed')
            report.update(status='awaiting_review',passed=True,technical_checks_passed=True,human_approved=False,isaac_verified=False)
        except BaseException as e:
            report.update(status='failed',error=f'{type(e).__name__}: {e}');raise
        finally:atomic_json(revision/'report.json',report)
        return revision

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True,type=Path);p.add_argument('--views-dir',required=True,type=Path);p.add_argument('--prompt',default='bottle');p.add_argument('--config',type=Path,default=ROOT/'configs/appearance/default.yaml');p.add_argument('--isaac-env',type=Path,default=ROOT.parent.parent/'robot_sim/scripts/activate_env.sh')
    a=p.parse_args();run=a.run_dir.resolve();config=validate_config(yaml.safe_load(a.config.read_text()));photos=discover(a.views_dir)
    if len(photos)>config['max_api_calls']:raise ValueError('One API check per view required; increase max_api_calls in config')
    task=run/'input'/('multiview_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));task.mkdir()
    ledger={'status':'running','order':[k for k,_ in photos],'steps':[]};atomic_json(task/'report.json',ledger)
    client=APIClient(config)
    try:
        views=[]
        for angle,photo in photos:
            print('MULTIVIEW_STAGE',angle,flush=True)
            ref=task/angle;ref.mkdir();shutil.copy2(photo,ref/('rgb'+photo.suffix))
            snap=ref/('rgb'+photo.suffix)
            if angle=='0':
                shutil.copy2(resolve_inputs(run)['mask']['path'],ref/'mask.png')
                if digest(snap)!=resolve_inputs(run)['image']['sha256']:raise ValueError('Front photograph differs from baseline input')
            else:
                command=['taskset','-c','0',str(ROOT.parent/'InstanceRecognition/.venv/bin/python'),'-u',str(ROOT/'src/objgen/generate/sam31_mask.py'),'--image',str(snap),'--prompt',a.prompt,'--output',str(ref/'sam31'),'--repo',str(ROOT.parent/'Submodule/SAM3'),'--checkpoint',str(ROOT.parent/'weights/sam3.1_multiplex.pt')]
                with (ref/'segmentation.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180,env=dict(os.environ,OPENBLAS_CORETYPE='Haswell',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
                shutil.copy2(ref/'sam31/mask.png',ref/'mask.png')
            prepare_images({'image':{'path':str(snap)},'mask':{'path':str(ref/'mask.png')}},ref)
            views.append((angle,ref));candidate=build(run,config,views)
            compact=task/('comparison_'+angle+'.jpg');im=Image.open(candidate/'renders/multiview_comparison.png');im.thumbnail((1800,720));im.convert('RGB').save(compact,quality=90)
            used=sum(r.get('usage',{}).get('total_tokens',0) for r in client.records)
            if used+12000+config['max_completion_tokens']>config['max_total_tokens']:raise RuntimeError('API token budget exhausted')
            message=client.chat([{'role':'system','content':'You review cumulative multi-view photo projection onto a fixed mesh. Compare each reference/model pair; previous views must remain recognizable. Do not claim physical validation. Return JSON with status (pass or discuss), problems (list), and summary. Use discuss only for severe incorrect view orientation, missing label, gross deformation, or a prior view being destroyed. Modest seams, photographic highlights, transparent bottle rendered opaque, and minor alignment errors are known limitations; document these without blocking.'},{'role':'user','content':[{'type':'text','text':'Added angle '+angle+' in order 0,90,180,270,up,down. Each pair is reference on left and cumulative model on right. Inspect all pairs.'},image_content(compact)]}],response_format={'type':'json_object'})
            evaluation=json.loads(message['content']);atomic_json(task/('agent_'+angle+'.json'),{'evaluation':evaluation,'api_record':client.records[-1]})
            ledger['steps'].append({'angle':angle,'candidate':candidate.name,'evaluation':evaluation,'reference_sha256':digest(snap)});atomic_json(task/'report.json',ledger)
            if evaluation.get('status')!='pass':raise RuntimeError('Agent requested discussion: '+json.dumps(evaluation,ensure_ascii=False))
        _,result=execute(run,config,a.isaac_env,revision=candidate.name,feedback=False)
        ledger.update(status='success',asset=result['asset'],candidate=candidate.name,api_records=client.records);atomic_json(task/'report.json',ledger)
        atomic_json(run/'usd/appearance/pipeline.json',result)
        print('MULTIVIEW_SUCCESS',result['asset'],flush=True)
    except BaseException as e:
        ledger.update(status='failed',error=f'{type(e).__name__}: {e}');atomic_json(task/'report.json',ledger);raise

if __name__=='__main__':main()
