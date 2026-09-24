"""Publish committed windows with the existing offline WebGL point renderer."""
import json
import os
import re
import shutil
import numpy as np
from .state import save
from .observations import ROOT
from src.geometry.backproject_masks import encode, instance_color
from src.visualization.publish_gpt_viewer import rebase_manifest
from src.visualization.build_incremental_viewer import build
from src.geometry.run_incremental_objects import ply


def materialize(data,checked,version_dir,previous,voxel):
    state=checked['state'];(version_dir/'clouds').mkdir(parents=True,exist_ok=True)
    for oid in state['objects']:
        target=version_dir/'clouds'/f'{oid}.npz'
        if oid not in checked['affected'] and previous and (previous/'clouds'/f'{oid}.npz').exists():
            os.link(previous/'clouds'/f'{oid}.npz',target)
        else:
            p,c=data.fused(state,oid,voxel);np.savez_compressed(target,points=p,rgb=c)
    save(version_dir/'transaction.json',checked)
    save(version_dir/'state.json',state)


def publish(data,out,voxel):
    current=json.loads((out/'checkpoint.json').read_text())['version'];frames=[];states=[]
    for v in range(1,current+1):
        folder=out/'versions'/f'{v:04d}';state=json.loads((folder/'state.json').read_text());states.append(state)
        frame=data.frames[state['cutoff']];frames.append(frame)
    manifest=dict(data.manifest,frames=frames,frame_count=len(frames),mask_count=sum(len(f['instances']) for f in frames))
    manifest=rebase_manifest(manifest,data.source,out)
    (out/'manifest.js').write_text('window.MASK_PROJECTION_MANIFEST='+json.dumps(manifest,ensure_ascii=False)+';\n')
    build(out,len(frames));(out/'states').mkdir(exist_ok=True);(out/'objects').mkdir(exist_ok=True)
    for index,state in enumerate(states):
        folder=out/'versions'/f'{index+1:04d}';transaction=json.loads((folder/'transaction.json').read_text())
        points=[];colors=[];ids=[];objects=[]
        for oid,obj in state['objects'].items():
            with np.load(folder/'clouds'/f'{oid}.npz') as z:p=z['points'];c=z['rgb']
            n=int(oid[6:]);members=[k for k,a in state['assignments'].items() if a['object_id']==oid]
            seqs=sorted({data.items[k]['seq'] for k in members})
            objects.append({'id':n,'object_label':obj['object_label'],'frames':seqs,'voxel_count':len(p),
                'kind':'semantic_group' if 'pair' in obj['object_label'].lower() else 'object',
                'status':'multi_frame' if len(seqs)>=3 else 'provisional','color':instance_color(0,n),
                'observations':members})
            points.append(p);colors.append(c);ids.append(np.full(len(p),n,np.uint32))
        p=np.concatenate(points) if points else np.empty((0,3),np.float32)
        c=np.concatenate(colors) if colors else np.empty((0,3),np.uint8)
        obj_ids=np.concatenate(ids) if ids else np.empty(0,np.uint32)
        zeros=np.zeros(len(p),np.uint32)
        arrays={'points_world':p,'rgb':c,'object_ids':obj_ids,'support_frame_bits':zeros}
        assignments=[]
        for k,o in data.items.items():
            if o['seq']==state['cutoff']:
                a=state['assignments'][k]
                assignments.append({'id':k,'instance_id':o['instance_id'],'object_id':int(a['object_id'][6:]) if a['object_id'] else None,
                                    'status':a['status'],'reason':a['reason'],'evidence':a['evidence']})
        snapshot={'objects':objects,'assignments':assignments,'pending':[k for k,a in state['assignments'].items() if a['object_id'] is None],
                  'events':transaction['proposal']['actions'],'conflict_entries':0,'cutoff':state['cutoff'],'version':state['version']}
        save(out/'states'/f'{index:04d}.json',snapshot);np.savez_compressed(out/'states'/f'{index:04d}.npz',**arrays)
        payload={'snapshot':snapshot,'positions':encode(p,'<f4'),'rgb':encode(c,'u1'),'ids':encode(obj_ids,'<u4'),
                 'support':encode(zeros,'<u4'),'first':encode(zeros,'u1'),'added':encode(zeros,'u1'),'conflicts':encode(zeros,'u1')}
        (out/'states'/f'{index:04d}.js').write_text('window.registerIncrementalState('+str(index)+','+json.dumps(payload,ensure_ascii=False)+');\n')
        if index==len(states)-1:
            for obj in objects:ply(out/'objects'/f'object_{obj["id"]:04d}.ply',arrays,obj_ids==obj['id'])
    html=(out/'index.html').read_text()
    html=re.sub(r'<title>.*?</title>','<title>模型 Harness · 窗口式物体点云</title>',html)
    html=html.replace('固定 AMB3R 几何 · 逐帧关联实验','固定 AMB3R 几何 · 模型逐窗审阅').replace('前 30 帧 · 物体点云如何积累','Harness · 窗口式物体点云')
    html=re.sub(r'<p class="experimentNote">.*?</p>', '<p class="experimentNote">每窗审阅多个抽样帧；RGB 与 mask 显示窗口末帧，点云显示本窗提交后的累计状态。固定全视频校正几何用于身份关联实验；不代表在线重建或准确率。<a href="audit.html">窗口全部观测与修改证据</a> · <a href="summary.json">运行汇总</a> · <a href="config.json">配置</a></p>', html, flags=re.S)
    html=html.replace('本帧关联与未决记录','本窗口提交记录').replace('当前帧观测','窗口末帧观测').replace('截至本帧的累计点云','截至本窗口的累计点云')
    html=re.sub(r'<select id="cloudFilter".*?</select>','<select id="cloudFilter"><option value="all">全部累计点（未统计逐体素支持数）</option></select>',html,flags=re.S)
    (out/'index.html').write_text(html)
    js=(out/'incremental.js').read_text();start=js.index('function drawEvents()');end=js.index('updateCloudBuffers=function',start)
    js=js[:start]+'''function drawEvents(){ $('events').replaceChildren(); for(const e of stateCloud.snapshot.events){const d=document.createElement('div');d.className='event';d.textContent=`${e.action} ${e.source_object||''} → ${e.object_id||''} ${JSON.stringify(e.observations||[])}：${e.reason}；证据 ${e.evidence.join(', ')}`;$('events').appendChild(d);}}
'''+js[end:]
    js=js.replace('空间归属重叠 ${stateCloud.snapshot.conflict_entries.toLocaleString()} 条','空间重叠未评估')
    js=js.replace('抽样 ${currentFrameIndex} / 源帧','窗口 ${currentFrameIndex+1} / 末抽样帧 ${f.sequence_index} / 源帧')
    js=js.replace('已加载 ${index} / 原始帧', '已加载窗口 ${index+1} / 原始帧')
    js=js.replace("${String(i).padStart(3,'0')} / ${f.source_frame_id}", '窗口 ${i+1} · 抽样 ${f.sequence_index} / 源 ${f.source_frame_id}')
    js=js.replace('selectedInstanceIndices.add(a.instance_id-1)','selectedInstanceIndices.add(manifest.frames[currentFrameIndex].instances.findIndex(x=>x.instance_id===a.instance_id))')
    (out/'incremental.js').write_text(js)
    audit={'states':states,'observations':data.items,'transactions':[json.loads((out/'versions'/f'{v:04d}'/'transaction.json').read_text()) for v in range(1,current+1)]}
    (out/'audit.js').write_text('window.AUDIT='+json.dumps(audit,ensure_ascii=False)+';')
    (out/'audit.html').write_text('''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Harness 证据回放</title><style>body{font:16px/1.7 sans-serif;max-width:1100px;margin:25px auto;padding:15px;background:#f4f8f7}article{background:white;padding:18px;margin:12px 0;border:1px solid #ccd}img{max-width:100%;max-height:500px}pre{white-space:pre-wrap;overflow-wrap:anywhere}select{font-size:18px}</style><h1>窗口归属与纠错记录</h1><a href="index.html">回到三维点云</a><p>从窗口 1 开始查看。原始观测保留；下方显示本次修改的归属前后与证据，待定观测同样计入。</p><select id="window"></select><main id="content"></main><script src="audit.js"></script><script>
const sel=document.getElementById('window'),content=document.getElementById('content');AUDIT.states.forEach((s,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`窗口 ${i+1} · 抽样 0–${s.cutoff} · 版本 ${s.version}`;sel.append(o)});function add(tag,text,parent=content){const e=document.createElement(tag);e.textContent=text;parent.append(e);return e}function show(){content.replaceChildren();const i=Number(sel.value),s=AUDIT.states[i],old=i?AUDIT.states[i-1].assignments:{};const a=add('article','');add('h2','修改单',a);add('pre',JSON.stringify(AUDIT.transactions[i].proposal,null,2),a);for(const [id,now] of Object.entries(s.assignments)){if(JSON.stringify(now)===JSON.stringify(old[id]))continue;const obs=AUDIT.observations[id],card=add('article','');add('h3',`${id} · ${obs.category_label}`,card);add('p',`${old[id]?.object_id||old[id]?.status||'未处理'} → ${now.object_id||now.status} · ${now.reason}`,card);add('p','证据：'+now.evidence.join(', '),card);const img=document.createElement('img');img.loading='lazy';img.src='evidence/'+id.replace(':','_')+'.jpg';img.alt=id+' mask crop';card.append(img)}}sel.onchange=show;show();</script>''')

    from src.visualization.build_harness_result import build as build_result
    build_result(out)
