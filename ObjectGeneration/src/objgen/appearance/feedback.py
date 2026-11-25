"""M3 bounded GPT camera feedback; model cannot approve or execute arbitrary code."""
import argparse
import asyncio
from datetime import datetime
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time
import yaml
from PIL import Image
from ..config import BLENDER_MCP, ROOT
from .api_client import APIClient, image_content
from .contracts import atomic_json, digest, validate_config
from .mcp_client import blender_session, execute_template
from .review import candidate_snapshot
from .texturing import build

TOOLS=[{'type':'function','function':{'name':'adjust_camera','description':'Create a geometry-preserving candidate with absolute camera angles and bounded framing. Do not change camera merely to improve backside appearance: unseen areas lack evidence.',
    'strict':True,'parameters':{'type':'object','properties':{
        'elevation_degrees':{'type':'number'},'roll_degrees':{'type':'number'},'yaw_degrees':{'type':'number'},
        'scale_factor':{'type':'number'},'offset':{'type':'array','items':{'type':'number'},'minItems':2,'maxItems':2},
        'reason':{'type':'string'}},'required':['elevation_degrees','roll_degrees','yaw_degrees','scale_factor','offset','reason'],'additionalProperties':False}}},
    {'type':'function','function':{'name':'stop_evaluation','description':'Stop iterating and hand all candidates to human review. This does not approve any candidate.',
    'strict':True,'parameters':{'type':'object','properties':{'reason':{'type':'string'}},'required':['reason'],'additionalProperties':False}}}]


def validate_action(message):
    calls=message.get('tool_calls') or []
    if len(calls)!=1 or not isinstance(calls[0].get('id'),str): raise ValueError('Exactly one identified tool call required')
    call=calls[0];fn=call['function'];args=json.loads(fn['arguments']);name=fn['name']
    if not isinstance(args,dict) or not isinstance(args.get('reason'),str) or not 1<=len(args['reason'])<=2000: raise ValueError('Invalid reason')
    if name=='stop_evaluation':
        if set(args)!={'reason'}: raise ValueError('Unknown stop field')
    elif name=='adjust_camera':
        fields={'elevation_degrees':(-60,60),'roll_degrees':(-30,30),'yaw_degrees':(-180,180),'scale_factor':(.7,1.5)}
        if set(args)!=set(fields)|{'offset','reason'}: raise ValueError('Invalid camera arguments')
        for k,(lo,hi) in fields.items():
            v=args[k]
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not lo<=v<=hi: raise ValueError('Camera bound exceeded: '+k)
        if not isinstance(args['offset'],list) or len(args['offset'])!=2 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not .2<=v<=.8 for v in args['offset']): raise ValueError('Invalid offset')
    else: raise ValueError('Tool not allowed')
    return call,name,args


def reserve(report, config, started):
    """Reserve worst-case image/prompt allowance before sending, including failed calls."""
    allowance=12000+config['max_completion_tokens']
    if len(report['requests'])>=config.get('max_api_calls',4): return 'api_call_budget'
    if (len(report['requests'])+1)*allowance>config.get('max_total_tokens',60000): return 'token_budget'
    if time.monotonic()-started>=config.get('max_wall_seconds',1200): return 'time_budget'
    report['requests'].append({'status':'reserved','token_allowance':allowance})
    return None


async def mcp_preview(config, revision, output):
    """Only trusted fixed template code reaches Blender MCP."""
    port=config['mcp_port'];repo=BLENDER_MCP
    with socket.socket() as s: s.bind(('127.0.0.1',port))
    with (output/'blender.log').open('w') as log:
        process=subprocess.Popen([config['blender'],'--factory-startup','--python',str(Path(__file__).with_name('blender_bootstrap.py')),'--',str(repo/'addon.py'),str(port)],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,BLENDER_MCP_DISABLE_TELEMETRY='1'))
        try:
            deadline=time.monotonic()+60
            while True:
                if process.poll() is not None: raise RuntimeError('Dedicated Blender exited')
                with socket.socket() as sock:
                    if sock.connect_ex(('127.0.0.1',port))==0: break
                if time.monotonic()>deadline: raise TimeoutError('Dedicated Blender startup timeout')
                await asyncio.sleep(.5)
            camera=json.loads((revision/'camera.json').read_text())
            code=f'''import bpy
from mathutils import Matrix, Vector
bpy.ops.wm.open_mainfile(filepath={str(revision/'work.blend')!r})
camera={camera!r}
scene=bpy.context.scene
view=scene.camera
view.location=Vector(camera['center'])+Vector(camera['basis'][2])*camera['height_m']*3
view.rotation_euler=Matrix(camera['basis']).transposed().to_euler()
view.data.type='ORTHO'; view.data.ortho_scale=camera['height_m']
view.data.shift_x=(.5-camera['offset'][0])*camera['aspect']; view.data.shift_y=.5-camera['offset'][1]
scene.render.resolution_x=256;scene.render.resolution_y=round(256/camera['aspect']);scene.render.resolution_percentage=100
scene.render.filepath={str(output/'mcp_preview.png')!r}
bpy.ops.render.render(write_still=True)
print('M3_PREVIEW_COMPLETE')
'''
            async with blender_session(repo,port) as session:
                result=await asyncio.wait_for(execute_template(session,code,'受限相机反馈渲染；只读候选工程，不保存修改。'),170)
                if 'M3_PREVIEW_COMPLETE' not in result: raise RuntimeError('Missing MCP completion marker')
            with Image.open(output/'mcp_preview.png') as im: im.verify()
        finally:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill();process.wait()


def reduced(source,target):
    with Image.open(source) as im:
        im=im.convert('RGB');im.thumbnail((768,768));im.save(target)


def run_feedback(run,revision,config):
    candidate_snapshot(run,revision)
    if not json.loads((revision/'report.json').read_text()).get('technical_checks_passed'): raise ValueError('Correct needs_alignment in review UI first')
    output=run/'usd/appearance'/('m3_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));output.mkdir()
    atomic_json(output/'config.json',config)
    report={'schema_version':1,'milestone':'M3','status':'running','human_approved':False,'candidates':[revision.name],
        'requests':[],'iterations':[],'best_revision':revision.name,'scope':'single_photo_visible_side',
        'tool_sha256':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}}
    def save(): atomic_json(output/'report.json',report)
    save();print('APPEARANCE_FEEDBACK_REPORT',output/'report.json',flush=True)
    started=time.monotonic();api=APIClient(config);current=revision;stale=0
    best=json.loads((current/'camera.json').read_text())['silhouette_iou']
    history=[]
    try:
        for index in range(config['max_iterations']):
            if time.monotonic()-started>=config.get('max_wall_seconds',1200): report['stop_reason']='time_budget';break
            step=output/f'iteration_{index:02d}';step.mkdir()
            remaining=config.get('max_wall_seconds',1200)-(time.monotonic()-started)
            asyncio.run(asyncio.wait_for(mcp_preview(config,current,step),timeout=remaining))
            reduced(current/'renders/comparison.png',step/'comparison.jpg');reduced(step/'mcp_preview.png',step/'preview.jpg')
            camera=json.loads((current/'camera.json').read_text())
            # Start a compact context each round; include exact previous call/tool result protocol.
            messages=[{'role':'system','content':'You assess photo projection camera alignment only. Never approve a candidate. Preserve geometry. Symmetric bottle yaw is ambiguous; unseen backside cannot be repaired from one photo. Prefer stopping if silhouette IoU >= .90 and label projection is adequate. One bounded tool call only. Angles are absolute degrees; scale_factor multiplies current fitted height. Image comparison shows reference, original, refined, contour. Numeric IoU is alignment evidence, not texture quality.'}]+history+[
                {'role':'user','content':[{'type':'text','text':'Current camera: '+json.dumps(camera)+'; candidate '+current.name},image_content(step/'comparison.jpg'),image_content(step/'preview.jpg')]}]
            reason=reserve(report,config,started)
            if reason: report['stop_reason']=reason;break
            save() # Reserved calls are never replayed after interruption. Resume by review only.
            remaining=config.get('max_wall_seconds',1200)-(time.monotonic()-started)
            api.config={**config,'request_timeout_seconds':max(1,min(config['request_timeout_seconds'],remaining))}
            message=api.chat(messages,tools=TOOLS,tool_choice='required',parallel_tool_calls=False)
            call,name,args=validate_action(message)
            record=report['requests'][-1];record.update(status='completed',response=message,api=api.records[-1]);save()
            actual=api.records[-1].get('usage') or {}
            if not isinstance(actual.get('total_tokens'),int) or actual['total_tokens']>record['token_allowance']:
                report['stop_reason']='usage_unknown_or_exceeds_reservation';break
            if name=='stop_evaluation':
                report['stop_reason']=args['reason'];break
            if time.monotonic()-started>=config.get('max_wall_seconds',1200): report['stop_reason']='time_budget';break
            override={**camera,**{k:v for k,v in args.items() if k!='reason'}}
            atomic_json(step/'camera_override.json',override)
            before=set((run/'usd/appearance').glob('v_*'))
            try:
                proposal=build(run,config,step/'camera_override.json',deadline=started+config.get('max_wall_seconds',1200))
                score=json.loads((proposal/'camera.json').read_text())['silhouette_iou']
                report['candidates'].append(proposal.name)
                report['iterations'].append({'revision':proposal.name,'reason':args['reason'],'silhouette_iou':score})
                if score>best+.001: best=score;report['best_revision']=proposal.name;stale=0;current=proposal
                else: stale+=1
                tool_result=json.dumps({'revision':proposal.name,'silhouette_iou':score,'best_revision':report['best_revision'],'human_review_required':True})
            except Exception as exc:
                failed=sorted(set((run/'usd/appearance').glob('v_*'))-before)
                report['failed_candidates']=[p.name for p in failed];report['stop_reason']='candidate_failed';report['error']=str(exc);break
            history=[message,{'role':'tool','tool_call_id':call['id'],'content':tool_result}]
            save()
            if stale>=2: report['stop_reason']='two_iterations_without_alignment_improvement';break
        else: report['stop_reason']='iteration_budget'
    except (Exception,KeyboardInterrupt) as exc:
        report['stop_reason']='interrupted' if isinstance(exc,KeyboardInterrupt) else 'tool_or_api_error'
        from .preflight import error_detail
        report['error']=error_detail(exc)
    finally:
        report.update(status='awaiting_review',elapsed_seconds=time.monotonic()-started,api_usage=api.records)
        save()
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--revision',required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/appearance/default.yaml');a=p.parse_args()
    config=validate_config(yaml.safe_load(a.config.read_text()));run=a.run_dir.resolve()
    output=run_feedback(run,run/'usd/appearance'/a.revision,config)
    report=json.loads((output/'report.json').read_text())
    print('APPEARANCE_AWAITING_REVIEW',report['best_revision']);print('STOP_REASON',report['stop_reason'])
    return 0 if 'error' not in report else 1


if __name__=='__main__': raise SystemExit(main())
