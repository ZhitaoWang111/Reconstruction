"""Score a finished run: eval.json + report.html, and a re-score of the pilot run for comparison."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from initial import InitialPoses
from state import SceneState

from .baselines import B0, B1, M3, baseline_matrices
from .frames import STEP, Holdout
from .score import Scorer

HERE = Path(__file__).resolve().parent
SCENE_ASSEMBLY = HERE.parents[1]
PILOT = SCENE_ASSEMBLY / 'runs' / 'harness-v1-opus55-batch1-20261111-r2'


def _pilot_matrices(hs, assets) -> dict:
    """The pilot run's final poses as 4x4 matrices (its pose had one footprint scale and one height scale)."""
    from state.pose import Pose
    snap = json.loads((PILOT / 'states' / 'v0008.json').read_text())
    out = {}
    for oid, st in snap['objects'].items():
        q = st['pose']
        pose = Pose(x=q['x'], y=q['y'], yaw_deg=q['yaw_deg'], up=q['up'], scale=(q['scale_xy'], q['scale_xy'], q['scale_z']))
        out[oid] = (pose.matrix(assets.verts(oid), float(q['z'])), pose)
    return out


def _overlay(scene, holdout, scorer, oid, matrix, path: Path):
    """The object's first key frame with the final silhouette outlined, for eyeballing."""
    from inputs.usdmesh import load_visual
    seq = min(f.seq for f in scene.objects[oid].key_frames)
    f = next(k for k in scene.objects[oid].key_frames if k.seq == seq)
    photo = Image.open(f.rgb).convert('RGB').resize((scene.camera.width // STEP, scene.camera.height // STEP))
    mesh = scorer._posed(oid, matrix)
    from observe.render import camera_rays
    c = scene.camera
    W, H = c.width // STEP, c.height // STEP
    o, d = camera_rays(f.c2w, c.fx / STEP, c.fy / STEP, c.cx / STEP, c.cy / STEP, W, H)
    loc, ray, _ = mesh.ray.intersects_location(o, d, multiple_hits=False)
    m = np.zeros((H, W), bool)
    m.reshape(-1)[ray] = True
    e = np.zeros_like(m)
    e[1:] |= m[1:] != m[:-1]
    e[:-1] |= m[:-1] != m[1:]
    e[:, 1:] |= m[:, 1:] != m[:, :-1]
    e[:, :-1] |= m[:, :-1] != m[:, 1:]
    a = np.asarray(photo).copy()
    a[e] = [40, 225, 90]
    im = Image.fromarray(a)
    d2 = ImageDraw.Draw(im)
    d2.rectangle([0, 0, 150, 18], fill=(255, 255, 255))
    d2.text((3, 2), f'{oid}  key frame {seq}', font=ImageFont.truetype(
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', 13), fill=(20, 30, 40))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=85)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 4) if xs else None


def evaluate(run_dir: Path, compare_pilot=True, overlays=True) -> dict:
    from inputs import handoff, structure as structure_mod
    run_dir = Path(run_dir)
    scene = handoff.load()
    state = SceneState.load(scene, structure_mod.load(scene=scene), run_dir / 'state')
    initial = InitialPoses(scene, state.structure, run_dir / 'init')
    initial.compute_all()
    hold = Holdout(scene)
    scorer = Scorer(scene, hold, state.assets)
    out_dir = run_dir / 'eval'
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = {}
    for oid, st in sorted(state.objects.items()):
        row = {'status': st.status, 'label': scene.objects[oid].label}
        row['final'] = scorer.score(oid, st.matrix)
        for name, m in baseline_matrices(scene, initial, oid, state.assets.verts(oid)).items():
            row[name] = scorer.score(oid, m)['iou']
        rows[oid] = row
        if overlays:
            _overlay(scene, hold, scorer, oid, st.matrix, out_dir / 'overlays' / f'{oid}.jpg')

    pilot = {}
    if compare_pilot and (PILOT / 'states' / 'v0008.json').exists():
        for oid, (m, _p) in _pilot_matrices(scene, state.assets).items():
            pilot[oid] = scorer.score(oid, m)['iou']

    done = [r['final']['iou'] for o, r in rows.items() if r['status'] == 'done']
    failed = [r['final']['iou'] for o, r in rows.items() if r['status'] == 'failed']
    # geometry and physics of the final state
    from verify import Config
    from verify.measure import Measure
    measure = Measure(state, state.structure, None, wall_exempt_labels=Config().wall_exempt_labels)
    m = measure.all()
    phys = {}
    p = run_dir / 'final_physics.json'
    if p.exists():
        ph = json.loads(p.read_text())
        phys = {'fell_or_dropped': ph.get('stop', []),
                'moved_more_than_5cm': [i['object'] for i in ph['moved'] if i['level'] == 'report'],
                'per_object': {i['object']: {k: i[k] for k in ('dz', 'dxy', 'tilt_deg', 'settled', 'level', 'note')}
                               for i in ph['moved']}}
    prog = json.loads((run_dir / 'progress.json').read_text()) if (run_dir / 'progress.json').exists() else {}
    batches = {b: {'turns': r.get('turns'), 'usd': r.get('usd'), 'stop': r.get('stop'),
                   'objects': r.get('objects')} for b, r in prog.get('batches', {}).items()}
    undo = {}
    vlog = run_dir / 'verify' / 'verify.jsonl'
    if vlog.exists():
        for line in vlog.open():
            e = json.loads(line)
            undo[e['object']] = undo.get(e['object'], 0) + (1 if e['verdict'] == 'undo' else 0)

    report = {
        'run': run_dir.name, 'scene': scene.scene_id, 'time': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'state_version': state.version,
        'frames': hold.summary(),
        'iou': {'final_done': _mean(done), 'final_failed': _mean(failed), 'final_all': _mean(done + failed),
                'B0': _mean([r[B0] for r in rows.values()]), 'B1': _mean([r[B1] for r in rows.values()]),
                'M3_start': _mean([r[M3] for r in rows.values()]),
                'pilot_r2_rescored': _mean([pilot[o] for o in rows if o in pilot]) if pilot else None,
                'objects': {o: {'final': r['final'], 'B0': r[B0], 'B1': r[B1], 'M3_start': r[M3],
                                'pilot_r2': pilot.get(o), 'status': r['status'], 'label': r['label']}
                            for o, r in rows.items()}},
        'geometry': {'contacts_not_declared_as_support': sorted(tuple(sorted(x)) for x in m.contacts),
                     'in_wall_over_2cm': {o: v for o, v in m.wall_pen.items() if v > 0.02},
                     'outside_room_over_5cm': {o: v for o, v in m.outside.items() if v > 0.05},
                     'wall_exempt': sorted(m.exempt)},
        'physics': phys,
        'process': {'batches': batches, 'usd_total': prog.get('usd'), 'auto_undos': undo},
        'pilot_batch1': _pilot_compare(rows, pilot, state.assets.verts),
        'objects': {o: r['status'] for o, r in rows.items()},
    }
    (out_dir / 'eval.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    (out_dir / 'report.html').write_text(_html(report))
    return report


def _pilot_compare(rows, pilot, verts_of):
    """The pilot scored six objects; on those the new frame set should land near its numbers
    (B0 0.472, B1 0.477, r2 0.532)."""
    oids = [o for o in rows if o in pilot]
    if not oids:
        return None
    return {'objects': oids, 'B0': _mean([rows[o][B0] for o in oids]), 'B1': _mean([rows[o][B1] for o in oids]),
            'M3_start': _mean([rows[o][M3] for o in oids]), 'pilot_r2_rescored': _mean([pilot[o] for o in oids]),
            'pilot_r2_original': 0.532, 'pilot_B0_original': 0.472, 'pilot_B1_original': 0.477}


# ---------------------------------------------------------------- the page

def _html(rep: dict) -> str:
    iou = rep['iou']
    def pct(x):
        return '—' if x is None else f'{x:.3f}'
    head = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>场景组装 · 评估 · {rep['run']}</title>
<style>
body{{margin:0;background:#fbfcfd;color:#16191d;font:15px/1.6 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif}}
.wrap{{max-width:1100px;margin:0 auto;padding:36px 28px 80px}}
h1{{font-size:26px;margin:0 0 6px}} h2{{font-size:19px;margin:34px 0 10px;border-bottom:1px solid #dfe3e8;padding-bottom:6px}}
.meta{{color:#6b7480;font-family:ui-monospace,monospace;font-size:12.5px}}
table{{border-collapse:collapse;width:100%;font-size:13.5px;background:#fff;margin:10px 0}}
th,td{{border:1px solid #dfe3e8;padding:6px 9px;text-align:left}} th{{background:#eef1f4;font-size:11.5px;color:#6b7480;
letter-spacing:.05em;text-transform:uppercase}} td.n{{text-align:right;font-family:ui-monospace,monospace}}
.big{{display:flex;gap:14px;flex-wrap:wrap;margin:14px 0}}
.card{{border:1px solid #dfe3e8;border-radius:10px;background:#fff;padding:12px 16px;min-width:150px}}
.card b{{display:block;font-size:24px;font-family:ui-monospace,monospace;color:#0f766e}}
.card span{{font-size:12px;color:#6b7480}}
img{{width:196px;border:1px solid #dfe3e8;border-radius:6px;margin:3px}}
.grid{{display:flex;flex-wrap:wrap}}
</style></head><body><div class="wrap">
<h1>场景组装评估 · {rep['run']}</h1>
<div class="meta">scene {rep['scene']} · state v{rep['state_version']} · {rep['time']} · 留出帧 {rep['frames']['with_frames']} 个物体有 / 每物体最多 16 张</div>
<h2>轮廓 IoU（留出帧）</h2>
<div class="big">
  <div class="card"><b>{pct(iou['final_all'])}</b><span>全部物体</span></div>
  <div class="card"><b>{pct(iou['final_done'])}</b><span>只算完成</span></div>
  <div class="card"><b>{pct(iou['final_failed'])}</b><span>只算失败</span></div>
  <div class="card"><b>{pct(iou['M3_start'])}</b><span>起点位姿</span></div>
  <div class="card"><b>{pct(iou['B1'])}</b><span>B1 SAM3D</span></div>
  <div class="card"><b>{pct(iou['B0'])}</b><span>B0 扫描盒</span></div>
  <div class="card"><b>{pct(iou['pilot_r2_rescored'])}</b><span>r2 重打分</span></div>
</div>
<h2>逐物体</h2><table><thead><tr><th>物体</th><th>标签</th><th>状态</th><th>最终</th><th>B1</th><th>B0</th><th>起点</th><th>r2</th><th>帧数</th></tr></thead><tbody>"""
    body = []
    for o, r in sorted(iou['objects'].items()):
        body.append(f"<tr><td>{o}</td><td>{r['label']}</td><td>{r['status']}</td>"
                    f"<td class=n>{pct(r['final']['iou'])}</td><td class=n>{pct(r['B1'])}</td>"
                    f"<td class=n>{pct(r['B0'])}</td><td class=n>{pct(r['M3_start'])}</td>"
                    f"<td class=n>{pct(r['pilot_r2'])}</td><td class=n>{r['final']['frames']}</td></tr>")
    g = rep['geometry']
    ph = rep['physics']
    mid = f"""</tbody></table>
<h2>几何与物理（最终状态）</h2>
<table><tbody>
<tr><th>没声明支撑却接触</th><td>{len(g['contacts_not_declared_as_support'])} 对 {g['contacts_not_declared_as_support']}</td></tr>
<tr><th>穿墙 &gt; 2 cm</th><td>{g['in_wall_over_2cm']}</td></tr>
<tr><th>出房间 &gt; 5 cm</th><td>{g['outside_room_over_5cm']}</td></tr>
<tr><th>门窗豁免</th><td>{g['wall_exempt']}</td></tr>
<tr><th>全局物理：倒下或掉落</th><td>{ph.get('fell_or_dropped', '（没有运行）')}</td></tr>
<tr><th>全局物理：位移 &gt; 5 cm</th><td>{ph.get('moved_more_than_5cm', [])}</td></tr>
</tbody></table>
<h2>每批</h2><table><thead><tr><th>批</th><th>停因</th><th>轮数</th><th>花费</th><th>物体状态</th></tr></thead><tbody>"""
    prows = []
    for b, r in sorted(rep['process']['batches'].items(), key=lambda kv: int(kv[0])):
        prows.append(f"<tr><td>{b}</td><td>{r['stop']}</td><td class=n>{r['turns']}</td>"
                     f"<td class=n>{r['usd']}</td><td>{r['objects']}</td></tr>")
    tail = f"""</tbody></table>
<p class=meta>总花费 ${rep['process'].get('usd_total')} · 自动撤回次数 {rep['process']['auto_undos']}</p>
<h2>最终位姿（首个关键帧，绿线是渲染轮廓）</h2><div class="grid">"""
    imgs = [f'<a href="overlays/{o}.jpg"><img src="overlays/{o}.jpg" title="{o}"></a>' for o in sorted(iou['objects'])]
    return head + ''.join(body) + mid + ''.join(prows) + tail + ''.join(imgs) + '</div></div></body></html>'
