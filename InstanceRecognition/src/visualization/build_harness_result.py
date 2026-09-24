"""Apply RGB/overlay/category + cloud layout to published Harness results."""
import json
import os
from pathlib import Path
from .publish_gpt_viewer import rebase_manifest
from .object_bboxes import export_harness_bboxes


def build(out):
    out=Path(out).resolve();here=Path(__file__).parent
    config=json.loads((out/'config.json').read_text());source=Path(config['source'])
    original=json.loads((source/'manifest.json').read_text())
    checkpoint=json.loads((out/'checkpoint.json').read_text())
    states=[json.loads((out/'versions'/f'{v:04d}'/'state.json').read_text()) for v in range(1,checkpoint['version']+1)]
    export_harness_bboxes(out, states=states)
    frames=original['frames'][:states[-1]['cutoff']+1]
    manifest=rebase_manifest(dict(original,frames=frames,frame_count=len(frames)),source,out)
    for f in manifest['frames']: f['rgb_image']=f['original_rgb_image']
    (out/'manifest.js').write_text('window.MASK_PROJECTION_MANIFEST='+json.dumps(manifest,ensure_ascii=False)+';')
    observations=json.loads((out/'observations.json').read_text())
    mapping=[]
    for f in frames:
        seq=f['sequence_index'];index=next(i for i,s in enumerate(states) if s['cutoff']>=seq);s=states[index]
        assignments=[]
        for item in f['instances']:
            oid=f'{seq:04d}:{item["instance_id"]:03d}';a=s['assignments'][oid]
            assignments.append(dict(a,id=oid,instance_id=item['instance_id'],object_id=int(a['object_id'][6:]) if a['object_id'] else None))
        overlay=f'evidence/frame_{seq:04d}.jpg'
        if not (out/overlay).is_file():
            raise FileNotFoundError(f'Missing saved agent input: {out/overlay}')
        colors={str(item['instance_id']):observations[f'{seq:04d}:{item["instance_id"]:03d}']['overlay_color_rgb'] for item in f['instances']}
        row={'window_index':index,'assignments':assignments,'agent_overlay':overlay,'input_colors':colors,'input_mask_ids':{str(item['instance_id']):observations[f'{seq:04d}:{item["instance_id"]:03d}']['mask_id'] for item in f['instances']}}
        preview='numbered_overlay_preview/'+overlay
        if (out/preview).is_file():
            row['agent_overlay']=preview
            row['overlay_note']='编号 overlay 重绘预览；历史 Agent 决策未重跑'
        mapping.append(row)
    changes=[]
    previous={'objects':{},'assignments':{}}
    for index,state in enumerate(states):
        actions=json.loads((out/'versions'/f'{index+1:04d}'/'transaction.json').read_text())['proposal']['actions']
        entries={}
        def entry(oid):
            return entries.setdefault(oid,{'object_id':int(oid[6:]),'object_label':state['objects'].get(oid,previous['objects'].get(oid,{})).get('object_label',oid),'changes':[]})
        def add(oid,text):
            if oid and text not in entry(oid)['changes']:entry(oid)['changes'].append(text)
        for a in actions:
            kind=a['action'];oid=a.get('object_id')
            if kind=='create':add(oid,'新增')
            elif kind=='relabel':add(oid,'修正类别：'+previous['objects'].get(oid,{}).get('object_label','')+' → '+a['object_label'])
            elif kind=='merge':
                add(oid,'合并：'+a['source_object']+' → '+oid)
                add(a['source_object'],'合并至 '+oid)
            elif kind in ('assign','defer','repair'):
                for obs in a['observations']:
                    old=previous['assignments'].get(obs)
                    if old and (old.get('object_id')!=oid or old['status']!=('assigned' if kind=='assign' else kind)):
                        add(old.get('object_id'),'修正观测归属')
                        add(oid,'修正观测归属')
                if kind=='assign' and oid in previous['objects']:add(oid,'补充观测')
        changes.append({'window_index':index,'first':previous.get('cutoff',-1)+1,'cutoff':state['cutoff'],'objects':list(entries.values())})
        previous=state
    (out/'result_data.js').write_text('window.RESULT_FRAMES='+json.dumps(mapping,ensure_ascii=False)+';\nwindow.RESULT_CHANGES='+json.dumps(changes,ensure_ascii=False)+';')
    (out/'index.html').write_text((here/'harness_result.html').read_text())
    (out/'result.css').write_text((here/'harness_result.css').read_text())
    (out/'harness_input_display.js').write_text((here/'harness_input_display.js').read_text())
    (out/'harness_object_groups.js').write_text((here/'harness_object_groups.js').read_text())
    (out/'harness_bbox_display.js').write_text((here/'harness_bbox_display.js').read_text())
    (out/'harness_cloud_pick.js').write_text((here/'harness_cloud_pick.js').read_text())
    p=out/'incremental.js';js=p.read_text()
    if 'RESULT_FRAMES' not in js:
        js=js.replace('stateFor(index)', 'stateFor(RESULT_FRAMES[index].window_index)')
        js=js.replace('stateCloud=st;', 'stateCloud={...st,snapshot:{...st.snapshot,assignments:RESULT_FRAMES[index].assignments}};$("rawRgb").src=f.original_rgb_image;')
        js=js.replace('st.snapshot.assignments','stateCloud.snapshot.assignments')
        js=js.replace('已加载窗口 ${index+1}', '已加载抽样帧 ${f.sequence_index} · 窗口 ${RESULT_FRAMES[index].window_index+1}')
        js=js.replace('${item.instance_id} · ${escapeHtml(item.class_name)}','${escapeHtml(item.class_name)}')
        js=js.replace('窗口 ${currentFrameIndex+1} / 末抽样帧 ${f.sequence_index}', '抽样帧 ${f.sequence_index}')
        js=js.replace('${String(currentFrameIndex).padStart(4,\'0\')}', '${String(RESULT_FRAMES[currentFrameIndex].window_index).padStart(4,\'0\')}')
        js=js.replace('窗口 ${i+1} · 抽样 ${f.sequence_index} / 源 ${f.source_frame_id}', '抽样帧 ${f.sequence_index} · 原始帧 ${f.source_frame_id}')
        js=js.replace("setupControls();initWebgl();", "setupControls();$('clearSelection').onclick=()=>$('allObjects').click();initWebgl();")

    # Keep category swatches identical to the saved per-frame input overlay.
    js=js.replace('item.color=obj?obj.color:[255,183,72];', 'item.color=RESULT_FRAMES[index].input_colors[String(item.instance_id)];')
    if 'glState.pickIds=' not in js:
        js=js.replace('const positions=[],colors=[],filter=', 'const positions=[],colors=[],pickIds=[],filter=')
        js=js.replace('colors.push(255,183,72);', 'colors.push(255,183,72);pickIds.push(0);')
        js=js.replace('colors.push(...c);', 'colors.push(...c);pickIds.push(id);')
        js=js.replace('glState.drawCount=positions.length/3;', 'glState.drawCount=positions.length/3;glState.pickIds=pickIds;')
    # Object naming is a display-only change in the object list.
    js=js.replace('<span>O${String(o.id)', '<span>Object ${String(o.id)')
    js=js.replace('`O${o.id}', '`Object ${o.id}').replace('`O${String(a.object_id)', '`Object ${String(a.object_id)')
    js=js.replace('if(selectedObjectId!==null&&id!==selectedObjectId)continue;', 'if(selectedObjectIds!==null?!selectedObjectIds.has(id):(selectedObjectId!==null&&id!==selectedObjectId))continue;')
    p.write_text(js)
    p=out/'viewer.js';js=p.read_text().replace('`${instance.instance_id} · ${instance.class_name}`','`${instance.class_name}`').replace('if (imageMode === "selected") {', 'if (imageMode !== "rgb") {').replace('gl.clearColor(0.02, 0.03, 0.05, 1);','gl.clearColor(0, 0, 0, 1);').replace('gl.clearColor(0.949, 0.937, 0.973, 1);','gl.clearColor(0, 0, 0, 1);');p.write_text(js)
    print(out/'index.html')

if __name__=='__main__':
    import sys
    build(sys.argv[1])
