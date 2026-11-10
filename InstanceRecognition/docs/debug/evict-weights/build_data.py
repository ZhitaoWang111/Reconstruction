"""Build data.js for the auto-evict weight test page (temporary).

One entry per test case: an object, one window of a v14 run, and every
candidate that window's auto-evict saw (observations held before the window
plus the ones the window held). The page reruns the selection in the browser,
so this script only precomputes what needs point clouds: pairwise view angles
and per-point coverage bitmasks.

Run from InstanceRecognition: PYTHONPATH=. .venv/bin/python docs/debug/evict-weights/build_data.py
"""
import json, pathlib, tempfile, collections, numpy as np
from scipy.spatial import cKDTree
from src.harness.observations import Observations

SRC = pathlib.Path('artifacts/sources/scene0000_00/scene_amb3r_full100_v1')
RUN = 'harness-v14-amb3r-full100frames-20261109-1744'
EXP = pathlib.Path('artifacts/experiments') / RUN
OUT = pathlib.Path('docs/debug/evict-weights/data.js')
RADII = [5, 10, 15, 20]  # cm


def cabinet_part(p):
    z, y = np.median(p[:, 2]), np.median(p[:, 1])
    return ('左' if y < 5.82 else '右') + ('上门' if z > 1.5 else '下门')


def tv_part(p, seq):
    # Grouped by where the camera stood, not by which part of the screen: every
    # mask here covers the whole screen.
    return '正前方近拍' if 40 <= seq <= 50 else '斜侧远拍'


def bookshelf_part(p):
    """Group by wall section: x coordinate separates the left vs right cabinet unit."""
    x = np.median(p[:, 0])
    return '左段（帧 35–36）' if x < 1.0 else '中右段（帧 46–49）'


def barchair_part(p, seq):
    """Bar stool: group by distance from the object centroid (single chair viewed from different angles/distances)."""
    pts_count = len(p)
    return '近拍（>8000 点）' if pts_count >= 8000 else '远拍（<8000 点）'


def tv2_part(p, seq):
    """Object59 w19: near frontal vs far oblique."""
    return '正面近拍（帧 50–57）' if seq < 70 else '斜侧远拍（帧 93–94）'


CASES = [
    {'key': 'Object64-w10', 'object': 'Object64', 'label': 'cabinet', 'window': 10, 'part_fn': cabinet_part,
     'parts': ['左下门', '右下门', '左上门', '右上门'],
     'summary': '电视柜的四块门板，帧 44–49 同一段连续拍摄；视角差最大约 21°，点数/像素全在 0.163–0.164。'},
    {'key': 'Object63-w19', 'object': 'Object63', 'label': 'television', 'window': 19, 'part_fn': tv_part,
     'parts': ['正前方近拍', '斜侧远拍'],
     'summary': '同一块电视屏幕从房间两侧拍：帧 45–49 在柜子正前方约 3 m，帧 36 与 93–94 从沙发/吧台一侧约 5 m 斜拍，'
                '两组视角差约 40–60°。屏幕反光，点数/像素 0.039–0.154，占比和点数不再一致。'
                '远拍组投影回近拍画面 79–85% 落在同一块屏幕上，但深度差约 0.27 m，新增表面会把它算成新表面。'},
    {'key': 'Object53-w19', 'object': 'Object53', 'label': 'bar stool', 'window': 19, 'part_fn': barchair_part,
     'parts': ['近拍（>8000 点）', '远拍（<8000 点）'],
     'summary': '同一把吧台椅的 6 个候选：早期 2 条 held（帧 34 和 88，点数只有 2800–4000），'
                '新来 4 条（帧 90–93，视角差约 47°，点数/像素有一条低到 0.121）。'
                '这个用例测试 A 和 P 适度分歧、以及点数项对低质量 mask 的压制效果。'},
    {'key': 'Object59-w19', 'object': 'Object59', 'label': 'bookcase/cabinet', 'window': 19, 'part_fn': tv2_part,
     'parts': ['正面近拍（帧 50–57）', '斜侧远拍（帧 93–94）'],
     'summary': '大型书柜/储物柜墙，视角差约 119°（本批最大）。持有的 5 条从正面 2–3 m 拍，每条 14000–19000 点；'
                '新来 8 条从吧台侧面 119° 斜拍，每条 1600–10100 点。'
                '新增表面率全部 100%——视角太大导致 AMB3R 点云对不上，不是因为真的拍到了新部位。'
                '这一窗暴露了新增表面项在极端视角差下的虚高问题。'},
]


def window_candidates(obj, v):
    prev = json.loads((EXP / f'versions/{v-1:04d}/state.json').read_text())['assignments']
    held = sorted(k for k, r in prev.items() if r.get('object_id') == obj and r.get('status') in ('held', 'assigned'))
    new = [o for a in json.loads((EXP / f'versions/{v:04d}/transaction.json').read_text())['proposal']['actions']
           if a['action'] == 'hold' and a['object_id'] == obj for o in a['observations']]
    kept = sorted(k for k, r in json.loads((EXP / f'versions/{v:04d}/state.json').read_text())['assignments'].items()
                  if r.get('object_id') == obj and r.get('status') in ('held', 'assigned'))
    return held, new, kept


manifest = json.loads((SRC / 'manifest.json').read_text())
mask_px = {f'{f["sequence_index"]:04d}:{i["instance_id"]:03d}': i['mask_pixels']
           for f in manifest['frames'] for i in f['instances']}
cam = lambda s: np.load(SRC / f'frames/{s:04d}.npz')['camera_to_world'][:3, 3].astype(np.float64)


def build_case(d, case):
    held, new, kept = window_candidates(case['object'], case['window'])
    ids = held + new
    W, H = d.items[ids[0]]['image_size_wh']
    P = [d.points[o] for o in ids]
    Cc = [np.asarray(d.items[o]['centroid']) for o in ids]
    T = [cam(d.items[o]['seq']) for o in ids]
    n = len(ids)
    theta = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                m = (Cc[i] + Cc[j]) / 2
                a, b = T[i] - m, T[j] - m
                theta[i][j] = round(float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))), 3)
    trees = [cKDTree(p) for p in P]
    dist = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i][j] = trees[j].query(P[i], distance_upper_bound=max(RADII) / 100 + 1e-6)[0]
    hist = {}
    for r in RADII:
        hist[str(r)] = []
        for i in range(n):
            bits = np.zeros(len(P[i]), dtype=np.int64)
            for j in range(n):
                if i != j:
                    bits |= (dist[i][j] <= r / 100).astype(np.int64) << j
            c = collections.Counter(bits.tolist())
            hist[str(r)].append(sorted([[int(k), int(v)] for k, v in c.items()], key=lambda kv: -kv[1]))
    cands = []
    for k, o in enumerate(ids):
        it = d.items[o]
        seq = it['seq']
        part = case['part_fn'](P[k]) if case['part_fn'] is cabinet_part else case['part_fn'](P[k], seq)
        cands.append({
            'id': o, 'seq': seq, 'frame': it['source_frame_id'], 'part': part,
            'held_before': o in held, 'mask_px': mask_px[o], 'area': mask_px[o] / (W * H),
            'points': len(P[k]), 'pts_per_px': round(len(P[k]) / mask_px[o], 4),
            'cam_dist': round(float(np.linalg.norm(T[k] - Cc[k])), 2),
            'img': f'../../../artifacts/experiments/{RUN}/evidence/{o.replace(":", "_")}.jpg',
        })
    return {'key': case['key'], 'object': case['object'], 'label': case['label'], 'window': case['window'],
            'summary': case['summary'], 'parts': case['parts'], 'limit': 5, 'image_wh': [W, H],
            'candidates': cands, 'theta': theta, 'hist': hist, 'v14_kept': kept}


with tempfile.TemporaryDirectory() as tmp:
    d = Observations(SRC, 100, tmp)
    cases = [build_case(d, c) for c in CASES]

data = {'run': RUN, 'source': SRC.name, 'radii_cm': RADII, 'cases': cases}
OUT.write_text('// Generated by build_data.py; auto-evict weight test cases from ' + RUN + '.\n'
               'var EVICT_DATA = ' + json.dumps(data, ensure_ascii=False, separators=(',', ':')) + ';\n'
               "if (typeof module !== 'undefined') module.exports = EVICT_DATA;\n")
print('wrote', OUT, OUT.stat().st_size, 'bytes')
for c in cases:
    print(c['key'], len(c['candidates']), 'candidates; v14 kept', c['v14_kept'])
    for x in c['candidates']:
        print('  ', x['id'], x['part'], f"{x['area']:.4f}", x['points'], f"{x['pts_per_px']:.3f}", 'held' if x['held_before'] else '')
