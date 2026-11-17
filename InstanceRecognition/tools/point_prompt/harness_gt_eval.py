"""Score a finished harness run against the ScanNet 2D instance annotation.

    .venv/bin/python -m tools.point_prompt.harness_gt_eval harness-v14-... harness-v15-...

GT is used for scoring only; the harness never saw it. Same mapping as the v11
study (debug-log 1105-v1): an observation's GT object is the ScanNet 2D instance
that owns most of its mask pixels.

Per run, over the objects that are live at the end (status not merged/retired):
  purity          share of an object's held observations on its majority GT object
  impure objects  objects whose held observations come from >= 2 GT objects
                  (counting counter#3 + cabinet#42 as one, user decision 2026-11-11)
  duplicates      GT objects that are the majority object of >= 2 live objects
  found           non-structural GT objects that are the majority of some live object
"""
import io
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from src import config

SCENE = config.SCANNET_ROOT / 'scene0000_00' / 'scene0000_00'   # a file stem: SCENE.aggregation.json, SCENE_2d-instance.zip
STRUCTURAL = {'wall', 'floor', 'ceiling', 'window', 'doorframe'}
SAME_UNIT = [{3, 42}]


def unit(i):
    for g in SAME_UNIT:
        if i in g:
            return min(g)
    return i


def gt_labels():
    agg = json.loads(Path(f'{SCENE}.aggregation.json').read_text())
    return {g['objectId']: g['label'] for g in agg['segGroups']}


def observation_gt(run_dir, archive):
    """observation id -> (majority GT id or None, share of mask pixels on it)."""
    obs = json.loads((run_dir / 'observations.json').read_text())
    cache, out = {}, {}
    for oid, o in obs.items():
        fid = int(o['source_frame_id'])
        if fid not in cache:
            cache[fid] = np.asarray(Image.open(io.BytesIO(archive.read(f'instance/{fid}.png')))).astype(np.int32) - 1
        gt = cache[fid]
        m = np.asarray(Image.open(o['mask']).convert('L')) > 0
        if m.shape != gt.shape:
            m = np.asarray(Image.fromarray(m.astype(np.uint8)).resize((gt.shape[1], gt.shape[0]), Image.NEAREST)) > 0
        v = gt[m]
        v = v[v >= 0]
        if not len(v):
            out[oid] = (None, 0.0)
            continue
        ids, counts = np.unique(v, return_counts=True)
        out[oid] = (int(ids[counts.argmax()]), float(counts.max() / m.sum()))
    return out


def score(run):
    run_dir = config.ARTIFACTS_ROOT / 'experiments' / run
    ck = json.loads((run_dir / 'checkpoint.json').read_text())
    state = json.loads((run_dir / 'versions' / f'{ck["version"]:04d}' / 'state.json').read_text())
    labels = gt_labels()
    archive = zipfile.ZipFile(f'{SCENE}_2d-instance.zip')
    gt_of = observation_gt(run_dir, archive)
    held = defaultdict(list)
    for oid, a in state['assignments'].items():
        if a['status'] == 'held' and a.get('object_id'):
            held[a['object_id']].append(oid)
    live = {k: v for k, v in state['objects'].items() if v.get('status') not in ('merged', 'retired') and held.get(k)}
    rows = []
    for obj, info in live.items():
        units = Counter(unit(gt_of[o][0]) for o in held[obj] if gt_of[o][0] is not None)
        if not units:
            continue
        major, n = units.most_common(1)[0]
        rows.append({'object': obj, 'label': info.get('object_label'), 'held': len(held[obj]),
                     'major': major, 'major_label': labels.get(major), 'purity': n / sum(units.values()),
                     'units': dict(units)})
    by_gt = defaultdict(list)
    for r in rows:
        by_gt[r['major']].append(r)
    structural = {i for i, l in labels.items() if l in STRUCTURAL}
    nonstruct = [r for r in rows if r['major'] not in structural]
    impure = [r for r in nonstruct if len(r['units']) >= 2]
    dups = {g: rs for g, rs in by_gt.items() if len(rs) >= 2 and g not in structural}
    seen = {unit(g) for g, _ in gt_of.values() if g is not None and labels.get(g) not in STRUCTURAL}
    found = {r['major'] for r in nonstruct}
    return {'run': run, 'objects': len(live), 'held': sum(r['held'] for r in rows),
            'mean_purity': float(np.mean([r['purity'] for r in nonstruct])),
            'impure': impure, 'dups': dups, 'found': found, 'seen': seen, 'rows': rows, 'labels': labels}


def report(s):
    L = s['labels']
    print(f"== {s['run']}")
    print(f"   live objects {s['objects']} | held observations {s['held']} | "
          f"mean purity {s['mean_purity']:.3f}")
    print(f"   GT objects found {len(s['found'])} of {len(s['seen'])} visible in some observation")
    print(f"   impure objects (held obs from >=2 GT objects): {len(s['impure'])}")
    for r in sorted(s['impure'], key=lambda r: r['purity'])[:12]:
        mix = ', '.join(f"{L.get(g)}#{g}x{n}" for g, n in sorted(r['units'].items(), key=lambda kv: -kv[1]))
        print(f"      {r['object']:10s} {r['label']!r:24} purity {r['purity']:.2f}  [{mix}]")
    print(f"   GT objects split across >=2 live objects: {len(s['dups'])}")
    for g, rs in sorted(s['dups'].items(), key=lambda kv: -len(kv[1])):
        print(f"      {L.get(g)}#{g}: " + ', '.join(f"{r['object']}({r['label']}, {r['held']})" for r in rs))


def main():
    out = {}
    for run in sys.argv[1:]:
        s = score(run)
        report(s)
        out[run] = {k: v for k, v in s.items() if k not in ('labels',)}
        out[run]['found'] = sorted(s['found']); out[run]['seen'] = sorted(s['seen'])
        out[run]['dups'] = {str(g): [r['object'] for r in rs] for g, rs in s['dups'].items()}
    path = config.ARTIFACTS_ROOT / 'experiments' / 'analysis' / 'harness_gt_eval.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
