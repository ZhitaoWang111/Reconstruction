"""Publish committed windows: pipeline output to one tree, pages to another.

Two roots, deliberately:

* `run_dir`  — pipeline records (`config.json`, `checkpoint.json`, `versions/`,
  `calls/`, `evidence/`, `observations.json`). Written by `run.py`.
* `site_dir` — the generated pages and the display data they load (`index.html`,
  `audit.html`, `manifest.js`, `states/`, `objects/`).

`states/` and `objects/` are site data, not pipeline data: they are derived from
`versions/` and can be regenerated from it at any time, so an experiment's records
can be archived without its pages and vice versa.
"""
import json
import hashlib
from pathlib import Path
import re
import numpy as np

from src.harness.state import dumps_sanitized, save
from src.harness.observations import ROOT
from src.geometry.backproject_masks import encode, instance_color
from src.geometry.amb3r_mask_source import POLICY as AMB3R_FRAME_POLICY
from src.visualization.publish_gpt_viewer import rebase_manifest

from . import build_viewer
from . import build_result as result_builder


def ply(path, arrays, selection):
    """Write the selected rows of an accumulated cloud as a binary PLY.

    The reader is the page's own picker, so only geometry, colour and the object
    id that picker resolves are carried.
    """
    count = int(selection.sum())
    dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'), ('green', 'u1'),
             ('blue', 'u1'), ('object_id', '<u4'), ('support_frame_bits', '<u4')]
    data = np.empty(count, dtype=dtype)
    for axis, name in enumerate(('x', 'y', 'z')):
        data[name] = arrays['points_world'][selection, axis]
    for axis, name in enumerate(('red', 'green', 'blue')):
        data[name] = arrays['rgb'][selection, axis]
    data['object_id'] = arrays['object_ids'][selection]
    data['support_frame_bits'] = arrays['support_frame_bits'][selection]
    header = ('ply\nformat binary_little_endian 1.0\nelement vertex ' + str(count) + '\n' +
              ''.join('property ' + ('float' if t == '<f4' else 'uint' if t == '<u4' else 'uchar') +
                      ' ' + k + '\n' for k, t in dtype) + 'end_header\n')
    with path.open('wb') as stream:
        stream.write(header.encode('ascii'))
        data.tofile(stream)


def _file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _snapshot_cache_key(data, folder, state):
    # Content hashes deliberately cover source bytes, not just modification times.
    dependencies = [__file__, encode.__code__.co_filename,
                    instance_color.__code__.co_filename, ply.__code__.co_filename]
    return {'schema': 1, 'numpy': np.__version__,
            'code': {str(p): _file_hash(p) for p in set(dependencies)},
            'state': _file_hash(folder / 'state.json'),
            'transaction': _file_hash(folder / 'transaction.json'),
            'clouds': {oid: _file_hash(folder / 'clouds' / f'{oid}.npz') for oid in state['objects']},
            'observations': hashlib.sha256(json.dumps(data.items, sort_keys=True).encode()).hexdigest()}


def _snapshot_cache_hit(site_dir, cache_path, key, last):
    try:
        cached = json.loads(cache_path.read_text())
        if cached['key'] != key or (last and not cached.get('latest_ply_exported', False)):
            return False
        groups = ['artifacts', 'ply'] if last else ['artifacts']
        return all(_file_hash(site_dir / name) == digest for group in groups
                   for name, digest in cached[group].items())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def publish(data, run_dir, site_dir, voxel):
    """Build the site for a completed run.

    `run_dir` holds the pipeline records and is read-only here; everything this
    function writes goes to `site_dir`.
    """
    run_dir = Path(run_dir).resolve()
    site_dir = Path(site_dir).resolve()
    site_dir.mkdir(parents=True, exist_ok=True)

    # ---- pipeline inputs (read from run_dir) ----------------------------------
    current = json.loads((run_dir / 'checkpoint.json').read_text())['version']
    frames = []
    states = []
    for v in range(1, current + 1):
        folder = run_dir / 'versions' / f'{v:04d}'
        state = json.loads((folder / 'state.json').read_text())
        states.append(state)
        frames.append(data.frames[state['cutoff']])

    # ---- site output ----------------------------------------------------------
    manifest = dict(data.manifest, frames=frames, frame_count=len(frames),
                    mask_count=sum(len(f['instances']) for f in frames))
    # rebase_manifest rewrites every external path relative to the DESTINATION,
    # so pointing it at the site dir is what makes the pages find the sources and
    # frames that live under a different root.
    manifest = rebase_manifest(manifest, data.source, site_dir)
    (site_dir / 'manifest.js').write_text(
        'window.MASK_PROJECTION_MANIFEST=' + dumps_sanitized(manifest, ensure_ascii=False) + ';\n')

    build_viewer.build(site_dir, len(frames))
    (site_dir / 'states').mkdir(exist_ok=True)
    (site_dir / 'objects').mkdir(exist_ok=True)

    for index, state in enumerate(states):
        folder = run_dir / 'versions' / f'{index + 1:04d}'
        transaction = json.loads((folder / 'transaction.json').read_text())
        cache_path = site_dir / 'visualization_cache' / 'snapshots' / f'{index:04d}.json'
        cache_key = _snapshot_cache_key(data, folder, state)
        if _snapshot_cache_hit(site_dir, cache_path, cache_key, index == len(states) - 1):
            continue
        points = []
        colors = []
        ids = []
        objects = []
        for oid, obj in state['objects'].items():
            with np.load(folder / 'clouds' / f'{oid}.npz') as z:
                p = z['points']
                c = z['rgb']
            n = int(oid[6:])
            members = [k for k, a in state['assignments'].items() if a['object_id'] == oid]
            seqs = sorted({data.items[k]['seq'] for k in members})
            objects.append({'id': n, 'object_label': obj['object_label'], 'frames': seqs, 'voxel_count': len(p),
                            'kind': 'semantic_group' if 'pair' in obj['object_label'].lower() else 'object',
                            'status': 'multi_frame' if len(seqs) >= 3 else 'provisional', 'color': instance_color(0, n),
                            'observations': members})
            points.append(p)
            colors.append(c)
            ids.append(np.full(len(p), n, np.uint32))
        p = np.concatenate(points) if points else np.empty((0, 3), np.float32)
        c = np.concatenate(colors) if colors else np.empty((0, 3), np.uint8)
        obj_ids = np.concatenate(ids) if ids else np.empty(0, np.uint32)
        zeros = np.zeros(len(p), np.uint32)
        arrays = {'points_world': p, 'rgb': c, 'object_ids': obj_ids, 'support_frame_bits': zeros}
        assignments = []
        for k, o in data.items.items():
            if o['seq'] == state['cutoff']:
                a = state['assignments'][k]
                assignments.append({'id': k, 'instance_id': o['instance_id'],
                                    'object_id': int(a['object_id'][6:]) if a['object_id'] else None,
                                    'status': a['status'], 'reason': a['reason'], 'evidence': a['evidence']})
        snapshot = {'objects': objects, 'assignments': assignments,
                    # `deferrals` is a count of closed decisions, not a queue. The
                    # harness stopped re-sending them (see Tools.listing) because a
                    # deferral's mask never returns; the page keeps the number so a
                    # run's coverage gap stays visible instead of silently vanishing.
                    'deferrals': [k for k, a in state['assignments'].items() if a['status'] == 'defer'],
                    'excluded': [k for k, a in state['assignments'].items() if a['status'] == 'exclude'],
                    'events': transaction['proposal']['actions'], 'conflict_entries': 0,
                    'cutoff': state['cutoff'], 'version': state['version']}
        save(site_dir / 'states' / f'{index:04d}.json', snapshot)
        np.savez_compressed(site_dir / 'states' / f'{index:04d}.npz', **arrays)
        payload = {'snapshot': snapshot, 'positions': encode(p, '<f4'), 'rgb': encode(c, 'u1'),
                   'ids': encode(obj_ids, '<u4'), 'support': encode(zeros, '<u4'),
                   'first': encode(zeros, 'u1'), 'added': encode(zeros, 'u1'), 'conflicts': encode(zeros, 'u1')}
        (site_dir / 'states' / f'{index:04d}.js').write_text(
            'window.registerIncrementalState(' + str(index) + ',' + dumps_sanitized(payload, ensure_ascii=False) + ');\n')
        if index == len(states) - 1:
            for obj in objects:
                ply(site_dir / 'objects' / f'object_{obj["id"]:04d}.ply', arrays, obj_ids == obj['id'])
        artifact_paths = [site_dir / 'states' / f'{index:04d}.{ext}' for ext in ('json', 'npz', 'js')]
        ply_paths = ([site_dir / 'objects' / f'object_{obj["id"]:04d}.ply' for obj in objects]
                     if index == len(states) - 1 else [])
        save(cache_path, {'key': cache_key, 'latest_ply_exported': index == len(states) - 1,
                          'artifacts': {str(p.relative_to(site_dir)): _file_hash(p) for p in artifact_paths},
                          'ply': {str(p.relative_to(site_dir)): _file_hash(p) for p in ply_paths}})

    # ---- result page ----------------------------------------------------------
    html = (site_dir / 'index.html').read_text()
    html = re.sub(r'<title>.*?</title>', '<title>模型 Harness · 窗口式物体点云</title>', html)
    # The geometry label is chosen by the source's own lifting policy, so a page
    # can never describe geometry it was not built from.
    policy = data.manifest.get('lifting_policy')
    if policy != AMB3R_FRAME_POLICY:
        raise ValueError(
            f'No display wording for lifting_policy {policy!r}; this tree builds '
            f'only {AMB3R_FRAME_POLICY!r}')
    html = html.replace('真值网格首表面 · 模型逐窗审阅', 'AMB3R 逐帧深度 · 模型逐窗审阅').replace(
        '独立场景点云投影 · 模型逐窗审阅', 'AMB3R 逐帧深度 · 模型逐窗审阅').replace(
        '固定 AMB3R 几何 · 逐帧关联实验', 'AMB3R 逐帧深度 · 模型逐窗审阅').replace(
        '固定 AMB3R 几何 · 模型逐窗审阅', 'AMB3R 逐帧深度 · 模型逐窗审阅')
    experiment_note = ('使用 AMB3R 逐帧预测的深度点图（经 GT 相机位姿 Sim(3) 校正），'
                       '将 mask 像素反投影为该像素自身所见表面上的一个点；'
                       '遮挡表面不进入候选；不代表在线重建或准确率。')
    html = html.replace('前 30 帧 · 物体点云如何积累', 'Harness · 窗口式物体点云')
    html = re.sub(
        r'<p class="experimentNote">.*?</p>',
        f'<p class="experimentNote">每窗审阅多个抽样帧；RGB 与 mask 显示窗口末帧，点云显示本窗提交后的累计状态。'
        f'{experiment_note}<a href="audit.html">窗口全部观测与修改证据</a> · '
        f'<a href="summary.json">运行汇总</a> · <a href="config.json">配置</a></p>', html, flags=re.S)
    html = html.replace('本帧关联与未决记录', '本窗口提交记录').replace(
        '当前帧观测', '窗口末帧观测').replace('截至本帧的累计点云', '截至本窗口的累计点云')
    html = re.sub(r'<select id="cloudFilter".*?</select>',
                  '<select id="cloudFilter"><option value="all">全部累计点（未统计逐体素支持数）</option></select>',
                  html, flags=re.S)
    (site_dir / 'index.html').write_text(html)

    js = (site_dir / 'incremental.js').read_text()
    start = js.index('function drawEvents()')
    end = js.index('updateCloudBuffers=function', start)
    js = js[:start] + '''function drawEvents(){ $('events').replaceChildren(); for(const e of stateCloud.snapshot.events){const d=document.createElement('div');d.className='event';d.textContent=`${e.action} ${e.source_object||''} → ${e.object_id||''} ${JSON.stringify(e.observations||[])}：${e.reason}；证据 ${e.evidence.join(', ')}`;$('events').appendChild(d);}}
''' + js[end:]
    js = js.replace('空间归属重叠 ${stateCloud.snapshot.conflict_entries.toLocaleString()} 条', '空间重叠未评估')
    js = js.replace('抽样 ${currentFrameIndex} / 源帧', '窗口 ${currentFrameIndex+1} / 末抽样帧 ${f.sequence_index} / 源帧')
    js = js.replace('已加载 ${index} / 原始帧', '已加载窗口 ${index+1} / 原始帧')
    js = js.replace("${String(i).padStart(3,'0')} / ${f.source_frame_id}", '窗口 ${i+1} · 抽样 ${f.sequence_index} / 源 ${f.source_frame_id}')
    js = js.replace('selectedInstanceIndices.add(a.instance_id-1)',
                    'selectedInstanceIndices.add(manifest.frames[currentFrameIndex].instances.findIndex(x=>x.instance_id===a.instance_id))')
    (site_dir / 'incremental.js').write_text(js)

    # ---- audit page -----------------------------------------------------------
    audit = {'states': states, 'observations': data.items,
             'transactions': [json.loads((run_dir / 'versions' / f'{v:04d}' / 'transaction.json').read_text())
                              for v in range(1, current + 1)]}
    (site_dir / 'audit.js').write_text('window.AUDIT=' + dumps_sanitized(audit, ensure_ascii=False) + ';')
    # The audit page shows evidence crops. Those live with the pipeline records,
    # so the img src is computed relative to the site page, not hardcoded.
    evidence_rel = _relative_evidence_path(run_dir, site_dir)
    (site_dir / 'audit.html').write_text(_AUDIT_HTML.replace('@@EVIDENCE@@', evidence_rel))

    # ---- result page assets ---------------------------------------------------
    result_builder.build(run_dir, site_dir)


def _relative_evidence_path(run_dir, site_dir):
    import os
    return os.path.relpath(run_dir / 'evidence', site_dir)


_AUDIT_HTML = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Harness 证据回放</title><style>body{font:16px/1.7 sans-serif;max-width:1100px;margin:25px auto;padding:15px;background:#f4f8f7}article{background:white;padding:18px;margin:12px 0;border:1px solid #ccd}img{max-width:100%;max-height:500px}pre{white-space:pre-wrap;overflow-wrap:anywhere}select{font-size:18px}</style><h1>窗口归属与纠错记录</h1><a href="index.html">回到三维点云</a><p>从窗口 1 开始查看。原始观测保留；下方显示本次修改的归属前后与证据，待定观测同样计入。</p><select id="window"></select><main id="content"></main><script src="audit.js"></script><script>
const sel=document.getElementById('window'),content=document.getElementById('content');AUDIT.states.forEach((s,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`窗口 ${i+1} · 抽样 0–${s.cutoff} · 版本 ${s.version}`;sel.append(o)});function add(tag,text,parent=content){const e=document.createElement(tag);e.textContent=text;parent.append(e);return e}function show(){content.replaceChildren();const i=Number(sel.value),s=AUDIT.states[i],old=i?AUDIT.states[i-1].assignments:{};const a=add('article','');add('h2','修改单',a);add('pre',JSON.stringify(AUDIT.transactions[i].proposal,null,2),a);for(const [id,now] of Object.entries(s.assignments)){if(JSON.stringify(now)===JSON.stringify(old[id]))continue;const obs=AUDIT.observations[id],card=add('article','');add('h3',`${id} · ${obs.category_label}`,card);add('p',`${old[id]?.object_id||old[id]?.status||'未处理'} → ${now.object_id||now.status} · ${now.reason}`,card);add('p','证据：'+now.evidence.join(', '),card);const img=document.createElement('img');img.loading='lazy';img.src='@@EVIDENCE@@/'+id.replace(':','_')+'.jpg';img.alt=id+' mask crop';card.append(img)}}sel.onchange=show;show();</script>'''
