"""Read-only loopback appearance viewer and candidate fingerprint utilities."""
import argparse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
from urllib.parse import urlsplit, parse_qs
import yaml
from .baseline import ROOT
from .contracts import atomic_json, digest, json_digest, resolve_inputs, run_lock, valid_stage, validate_config


def candidate_snapshot(run, revision):
    run, revision = Path(run).resolve(), Path(revision).resolve()
    if revision.parent != run/'usd/appearance' or not revision.name.startswith('v_'):
        raise ValueError('Candidate must be a revision of this run')
    manifest=json.loads((revision/'manifest.json').read_text())
    if resolve_inputs(run)!=manifest['inputs']: raise ValueError('Source fingerprints changed')
    report=json.loads((revision/'report.json').read_text())
    complete=report.get('technical_checks_passed') is True
    if complete and not all(valid_stage(stage,revision) for stage in report['stages'].values()):
        raise ValueError('Candidate fingerprints changed')
    files={'manifest.json','config.json','report.json'}
    for stage in report['stages'].values(): files.update(stage.get('outputs',{}))
    for name in ('camera.json','renders/alignment.png','reference_isolated.png'):
        if (revision/name).exists(): files.add(name)
    if not complete and report.get('status')!='needs_alignment': raise ValueError('Candidate is not reviewable')
    hashes={}
    for relative in sorted(files):
        p=(revision/relative).resolve()
        if not p.is_relative_to(revision) or not p.is_file(): raise ValueError('Invalid candidate dependency')
        hashes[relative]=digest(p)
    return {'revision':revision.name,'complete':complete,'files':hashes,'fingerprint':json_digest(hashes)}


def decide(run, revision, expected, action, note=''):
    if action not in ('accept','reject','original'): raise ValueError('Invalid decision')
    with run_lock(run):
        snapshot=candidate_snapshot(run,revision)
        if snapshot['fingerprint']!=expected: raise ValueError('Candidate changed; reopen review')
        if action=='accept' and not snapshot['complete']: raise ValueError('Alignment must pass before acceptance')
        record={'schema_version':1,'decision':action,'scope':'single_photo_visible_side',
                'created_at':datetime.now().astimezone().isoformat(),'candidate':snapshot,'note':note}
        atomic_json(Path(revision)/'review.json',record)
        selection_path=Path(run)/'usd/appearance/selection.json'
        if action in ('accept','original'):
            atomic_json(selection_path,{'schema_version':1,'selected':snapshot['revision'] if action=='accept' else 'original',
                'human_approved':True,'review':record,'baseline':resolve_inputs(run)})
        elif selection_path.exists():
            previous=json.loads(selection_path.read_text())
            if previous.get('selected')==snapshot['revision']:
                atomic_json(selection_path,{'schema_version':1,'selected':None,'human_approved':False,'review':record})
        return record


def verified_selection(run):
    data=json.loads((Path(run)/'usd/appearance/selection.json').read_text())
    if data.get('human_approved') is not True: raise ValueError('No human-approved selection')
    if resolve_inputs(run)!=data['baseline']: raise ValueError('Selected source changed')
    if data['selected']!='original':
        revision=Path(run)/'usd/appearance'/data['selected']
        snapshot=candidate_snapshot(run,revision)
        if snapshot!=data['review']['candidate']: raise ValueError('Selected candidate changed; confirmation invalid')
        if json.loads((revision/'review.json').read_text())!=data['review']: raise ValueError('Review was superseded')
    return data


PAGE='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>外观生成效果展示</title>
<style>body{font:16px sans-serif;max-width:1200px;margin:25px auto;background:#eee}img{max-width:100%;max-height:720px}button,input,select{padding:9px;margin:5px}#views{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}#views figure{margin:0;background:white;padding:12px;border-radius:8px}#views img{width:100%}figcaption{line-height:1.6}h2{margin-top:30px}.guide{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}.guide p{background:white;padding:12px;margin:0;line-height:1.7}.hint{line-height:1.7;color:#444}pre{white-space:pre-wrap}</style>
<h1>外观生成效果展示</h1><p>选择版本查看原照片、精修效果和不同方向的渲染图。未被输入照片覆盖的区域保留原始顶点颜色。</p>
<label>选择要查看的候选版本：<select id="candidates"></select></label><pre id="state"></pre>
<h2>1. 正面外观对比</h2>
<p class="hint">下方合成图从左到右依次为以下四项。先比较原照片与照片投射效果，再检查青色轮廓是否对齐。</p>
<div class="guide">
<p><strong>① 原照片（Reference photo）</strong><br>从输入照片中裁出目标物体，灰色区域是移除的背景。它是外观参考，重点看瓶盖、标签、文字和颜色。</p>
<p><strong>② 原始模型外观（Original vertex colours）</strong><br>SAM3D 生成的原始顶点颜色，尚未叠加照片贴图。用它判断精修前后的变化；文字和图案可能较模糊。</p>
<p><strong>③ 照片投射效果（Photo projection）</strong><br>将照片图案贴到三维模型后，从拟合的照片视角渲染。重点检查瓶盖、标签和文字是否贴在正确位置，有没有偏移、拉伸或混入背景。这张图不额外施加光照，便于比较图案。</p>
<p><strong>④ 轮廓对齐检查（Silhouette overlay）</strong><br>原照片上叠加模型的青色轮廓线。检查它是否沿着物体边缘，明显偏离说明照片与模型对齐不足。轮廓吻合并不保证文字位置或背面纹理准确。</p>
</div>
<img id="comparison" alt="从左到右：原照片、原始模型外观、照片投射效果、轮廓对齐检查">
<div id="multi" hidden><h2>多视角累计对比</h2><p>按 0→90→180→270→up→down 排列；每对左侧是参考照片，右侧是同一个累计模型在该照片方向的效果。检查新增视角后，已有视角的文字、图案是否仍然正确。缺少的照片不显示。</p><img id="multicomparison" alt="多视角参考照片与累计模型对比"></div><h2>2. 放大的轮廓对齐图</h2>
<p class="hint">这张图与上方第四列相同，单独放大便于检查。青色线表示模型在照片视角下的轮廓，照片中的背景仍保留作为位置参考。</p>
<img id="alignment" alt="原照片上的模型青色轮廓叠加图">
<h2>3. 精修模型的六个方向</h2>
<p class="hint">下面是同一个候选模型在统一光照下的六个视图，可检查侧面拉伸、接缝和未观测区域。正面、背面等名称按模型坐标定义，不一定与拍照方向完全一致。照片中的高光和阴影仍可能保留在贴图中。</p>
<div id="views"></div>
<script>
const token=new URLSearchParams(location.search).get('token');let state,current;
async function refresh(){let r=await fetch('/state?token='+token);state=await r.json();let sel=document.getElementById('candidates');let old=sel.value;
sel.replaceChildren(...state.candidates.map(c=>new Option(c.revision,c.revision)));sel.value=state.candidates.some(c=>c.revision===old)?old:state.candidates.at(-1).revision;show();}
function asset(name){return '/asset?token='+token+'&revision='+current.revision+'&name='+encodeURIComponent(name)}
function show(){current=state.candidates.find(c=>c.revision===document.getElementById('candidates').value);document.getElementById('state').textContent=current.revision+' / '+(current.complete?'生成完成（技术检查已通过）':'对齐不足（仅展示诊断图）');
let ismulti=!!current.files['renders/multiview_comparison.png'];document.getElementById('multi').hidden=!ismulti;if(ismulti)document.getElementById('multicomparison').src=asset('renders/multiview_comparison.png');
document.getElementById('comparison').src=asset(current.complete?'renders/comparison.png':'reference_isolated.png');document.getElementById('alignment').src=asset('renders/alignment.png');
document.getElementById('views').replaceChildren(...(current.complete?['front','back','left','right','top','three_quarter']:[]).map(v=>{const descriptions={front:['正面','检查标签、文字和瓶盖外观。'],back:['背面','对照背面参考照片检查图案；未覆盖区域仍保留原始顶点颜色。'],left:['左侧','检查图案是否被拉长，以及贴图与原始颜色的交界。'],right:['右侧','检查侧面拉伸、接缝和颜色过渡。'],top:['俯视','检查顶部、瓶盖以及边缘颜色是否异常。'],three_quarter:['斜前方','综合检查图案在曲面上的连续性和立体观感。']};let f=document.createElement('figure');let i=new Image();i.src=asset('renders/refined_'+v+'.png');i.alt=descriptions[v][0];let caption=document.createElement('figcaption');let title=document.createElement('strong');title.textContent=descriptions[v][0];caption.append(title,document.createElement('br'),descriptions[v][1]);f.append(i,caption);return f}));
}
document.getElementById('candidates').onchange=show;refresh();
</script></html>'''


def serve(run, revisions, config, port=0):
    run=Path(run).resolve(); parent=run/'usd/appearance'
    token=secrets.token_urlsafe(32); snapshots={p.name:candidate_snapshot(run,p) for p in revisions}
    state={'candidates':list(snapshots.values()),'status':'display_only'}
    session=parent/('review_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def reply(self,status,body,kind='application/json'):
            data=json.dumps(body,ensure_ascii=False).encode() if kind=='application/json' else body
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store'); self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(data)
        def do_GET(self):
            url=urlsplit(self.path); q=parse_qs(url.query)
            if q.get('token')!=[token]: return self.reply(403,{'error':'Invalid token'})
            try:
                if url.path=='/': return self.reply(200,PAGE.encode(),'text/html; charset=utf-8')
                if url.path=='/state':
                    data=json.loads(json.dumps(state))
                    return self.reply(200,data)
                if url.path=='/asset':
                    revision=q['revision'][0]; name=q['name'][0]
                    if revision not in snapshots or name not in snapshots[revision]['files'] or not name.endswith('.png'): raise ValueError('Image not allowed')
                    p=parent/revision/name
                    if digest(p)!=snapshots[revision]['files'][name]: raise ValueError('Image changed')
                    return self.reply(200,p.read_bytes(),'image/png')
                return self.reply(404,{'error':'Not found'})
            except (KeyError,ValueError,FileNotFoundError) as exc: return self.reply(409,{'error':str(exc)})
        def do_POST(self):
            return self.reply(405,{'error':'Display-only page; actions are disabled'})
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    url=f'http://127.0.0.1:{server.server_port}/?token={token}'
    atomic_json(session,state); print('APPEARANCE_REVIEW_URL',url,flush=True)
    # Store URL separately with private permissions; token never goes to the API.
    private=session.with_suffix('.url');private.touch(mode=0o600);private.write_text(url+'\n')
    try: server.serve_forever()
    finally: server.server_close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--revision',action='append',required=True,help='Repeat to compare immutable candidates')
    p.add_argument('--port',type=int,default=0);p.add_argument('--config',type=Path,default=ROOT/'configs/appearance/default.yaml')
    a=p.parse_args();run=a.run_dir.resolve()
    try: serve(run,[run/'usd/appearance'/v for v in a.revision],validate_config(yaml.safe_load(a.config.read_text())),a.port)
    except KeyboardInterrupt: return 0
    return 0


if __name__=='__main__': raise SystemExit(main())
