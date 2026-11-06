"""Apply RGB/overlay/category + cloud layout to published Harness results.

Reads the pipeline records from `run_dir` and writes only to `site_dir` (the
pages tree). Paths that point back into `run_dir` — the saved agent-input
overlays under `evidence/` — are computed relative to the page, because the two
trees are siblings rather than nested.
"""
import json
import os
from pathlib import Path
from src.visualization.publish_gpt_viewer import rebase_manifest
from src.visualization.object_bboxes import export_harness_bboxes
from src import config as app_config
from src.harness.paths import resolve_recorded_path
from src.harness.state import dumps_sanitized
from src.geometry.amb3r_mask_source import POLICY as AMB3R_FRAME_POLICY

# The result page replaces index.html wholesale, so publish.py's own experiment
# note never reaches the reader. The geometry disclosure therefore has to be
# applied here too, or the lift the source actually used goes undescribed.
RESULT_NOTE = '窗口内各帧共享同一次提交结果，并非逐帧独立推理。独立场景点云几何；通过 RGB 相机可见性投影得到 mask 点云；仅累积已观测表面，未评估识别准确率。'
RESULT_NOTE_AMB3R = ('窗口内各帧共享同一次提交结果，并非逐帧独立推理。AMB3R 逐帧深度点图几何（经 GT 相机位姿 Sim(3) 校正）；'
                     '每帧将 mask 像素反投影到该像素自身所见表面（每个 mask 像素恰好一个点）；'
                     '仅累积已观测表面，未评估识别准确率。')


def build(run_dir, site_dir):
    run_dir = Path(run_dir).resolve()
    out = Path(site_dir).resolve()
    config=json.loads((run_dir/'config.json').read_text());source=resolve_recorded_path(config['source'],run_dir)
    original=json.loads((source/'manifest.json').read_text())
    checkpoint=json.loads((run_dir/'checkpoint.json').read_text())
    states=[json.loads((run_dir/'versions'/f'{v:04d}'/'state.json').read_text()) for v in range(1,checkpoint['version']+1)]
    export_harness_bboxes(out, run_dir=run_dir, states=states)
    frames=original['frames'][:states[-1]['cutoff']+1]
    # This rewrites manifest.js after build_site wrote it, so every count that
    # build_site recomputed has to be recomputed here too. `frame_count` was the
    # only one carried over, which left `mask_count` reporting the whole source's
    # masks (4632 for a 500-frame source) on a run that covered 150 of them.
    manifest=rebase_manifest(dict(original,frames=frames,frame_count=len(frames),
                                  mask_count=sum(len(f['instances']) for f in frames)),source,out)
    for f in manifest['frames']: f['rgb_image']=f['original_rgb_image']
    (out/'manifest.js').write_text('window.MASK_PROJECTION_MANIFEST='+dumps_sanitized(manifest,ensure_ascii=False)+';')
    observations=json.loads((run_dir/'observations.json').read_text())
    # The overlays live with the pipeline records; the page links to them across
    # the two output trees.
    evidence_root=os.path.relpath(run_dir/'evidence',out)
    preview_root=os.path.relpath(run_dir/'numbered_overlay_preview',out)
    mapping=[]
    for f in frames:
        seq=f['sequence_index'];index=next(i for i,s in enumerate(states) if s['cutoff']>=seq);s=states[index]
        assignments=[]
        for item in f['instances']:
            oid=f'{seq:04d}:{item["instance_id"]:03d}';a=s['assignments'][oid]
            assignments.append(dict(a,id=oid,instance_id=item['instance_id'],object_id=int(a['object_id'][6:]) if a['object_id'] else None))
        overlay=f'evidence/frame_{seq:04d}.jpg'
        if not (run_dir/overlay).is_file():
            raise FileNotFoundError(f'Missing saved agent input: {run_dir/overlay}')
        colors={str(item['instance_id']):observations[f'{seq:04d}:{item["instance_id"]:03d}']['overlay_color_rgb'] for item in f['instances']}
        row={'window_index':index,'assignments':assignments,'agent_overlay':f'{evidence_root}/frame_{seq:04d}.jpg','input_colors':colors,'input_instance_ids':[int(item['instance_id']) for item in f['instances']]}
        if (run_dir/'numbered_overlay_preview'/overlay).is_file():
            row['agent_overlay']=f'{preview_root}/frame_{seq:04d}.jpg'
            row['overlay_note']='编号 overlay 重绘预览；历史 Agent 决策未重跑'
        mapping.append(row)
    changes=[]
    previous={'objects':{},'assignments':{}}
    for index,state in enumerate(states):
        actions=json.loads((run_dir/'versions'/f'{index+1:04d}'/'transaction.json').read_text())['proposal']['actions']
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
            # `repair` -> `exclude` and `assign`/`defer` -> `hold`/`discard` are
            # renames; a run may predate either, and its pages still have to render.
            elif kind in ('hold', 'discard', 'exclude', 'assign', 'defer', 'repair'):
                for obs in a['observations']:
                    old=previous['assignments'].get(obs)
                    new_status='assigned' if kind in ('hold','assign') else ('discard' if kind in ('discard','defer') else kind)
                    if old and (old.get('object_id')!=oid or old['status']!=new_status):
                        add(old.get('object_id'),'修正观测归属')
                        add(oid,'修正观测归属')
                if kind in ('hold','assign') and oid in previous['objects']:add(oid,'补充观测')
        changes.append({'window_index':index,'first':previous.get('cutoff',-1)+1,'cutoff':state['cutoff'],'objects':list(entries.values())})
        previous=state
    (out/'result_data.js').write_text('window.RESULT_FRAMES='+dumps_sanitized(mapping,ensure_ascii=False)+';\nwindow.RESULT_CHANGES='+dumps_sanitized(changes,ensure_ascii=False)+';\n'
                                      'window.RESULT_EVIDENCE_ROOT='+dumps_sanitized(evidence_root,ensure_ascii=False)+';')
    templates = app_config.WEB_ROOT / 'result'
    html=(templates/'harness_result.html').read_text()
    # The geometry disclosure is chosen by the source's own lifting policy, so a
    # page can never describe geometry it was not built from.
    policy=original.get('lifting_policy')
    if policy != AMB3R_FRAME_POLICY:
        raise ValueError(
            f'No display wording for lifting_policy {policy!r}; this tree builds '
            f'only {AMB3R_FRAME_POLICY!r}')
    if RESULT_NOTE not in html:
        raise ValueError(f'Result template geometry note changed; {policy} disclosure not applied')
    html=html.replace(RESULT_NOTE,RESULT_NOTE_AMB3R)
    # Shared navigation lives outside this tree; link to it relative to the page
    # so the two output roots can move independently.
    html=html.replace('@@NAV@@',os.path.relpath(app_config.NAV_ROOT/'assets',out))
    # The run's own records (summary/config/input audit) live in the artifacts
    # tree, not beside the page, so these links have to be rewritten rather than
    # left as siblings.
    for name in ('summary.json','config.json','input_audit.json'):
        if not (run_dir/name).exists():
            raise FileNotFoundError(f'Missing run record referenced by the result page: {run_dir/name}')
        html=html.replace(f'href="{name}"', f'href="{os.path.relpath(run_dir/name, out)}"')
    if '@@' in html:
        raise ValueError(f'Unsubstituted placeholder left in result page: {html[html.index("@@"):][:40]}')
    (out/'index.html').write_text(html)
    (out/'result.css').write_text((templates/'harness_result.css').read_text())
    (out/'harness_input_display.js').write_text((templates/'harness_input_display.js').read_text())
    (out/'harness_object_groups.js').write_text((templates/'harness_object_groups.js').read_text())
    (out/'harness_bbox_display.js').write_text((templates/'harness_bbox_display.js').read_text())
    (out/'harness_cloud_pick.js').write_text((templates/'harness_cloud_pick.js').read_text())
    (out/'harness_observation_panel.js').write_text((templates/'harness_observation_panel.js').read_text())
    # Keep the all500 viewer layout as the canonical result-page template.
    for name in ('all500-layout.css', 'all500-layout.js'):
        (out/name).write_bytes((templates/name).read_bytes())
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
