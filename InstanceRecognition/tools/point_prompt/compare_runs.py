"""Compare two point-prompt runs on the frames they share, against ScanNet GT.

    .venv/bin/python -m tools.point_prompt.compare_runs point-full100_v1 point-sep_v2
Reports, on the common frames only: merged masks (>=2 GT units), split GT objects,
the cabinet#8 / shelf#56 outcome per frame, the counter + base-cabinet outcome,
fragmentation of multi-part furniture, and cost.
"""
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from src.perception.point_prompt.objects import run_dir_of

from . import gt_eval as G

A, B = 8, 56
KITCHEN = (3, 42)


def run_frames(run):
    out = {}
    for fid, masks in G.point_masks(run):
        out[fid] = masks
    return out


def classify(fid, masks, archive):
    """cabinet/shelf outcome for one frame: merged / separated / partial / n/a."""
    gt = np.asarray(Image.open(io.BytesIO(archive.read(f'instance/{fid}.png')))).astype(np.int32) - 1
    if (gt == A).mean() < 0.03 or (gt == B).mean() < 0.03:
        return 'n/a'
    kinds = []
    for _, path in masks:
        m = np.asarray(Image.open(path)) > 0
        sa, sb, n = (m & (gt == A)).sum(), (m & (gt == B)).sum(), m.sum()
        ca, cb = sa / (gt == A).sum(), sb / (gt == B).sum()
        if ca >= .25 and cb >= .25 and sa / n >= .15 and sb / n >= .15:
            kinds.append('merged')
        elif ca >= .25 and sa / n >= .5:
            kinds.append('A')
        elif cb >= .25 and sb / n >= .5:
            kinds.append('B')
    if 'merged' in kinds:
        return 'merged'
    return 'separated' if ('A' in kinds and 'B' in kinds) else 'partial'


def kitchen_outcome(fid, masks, archive):
    gt = np.asarray(Image.open(io.BytesIO(archive.read(f'instance/{fid}.png')))).astype(np.int32) - 1
    if (gt == KITCHEN[0]).mean() < 0.01 or (gt == KITCHEN[1]).mean() < 0.03:
        return 'n/a'
    best = 0.0
    for _, path in masks:
        m = np.asarray(Image.open(path)) > 0
        ca = (m & (gt == KITCHEN[0])).sum() / (gt == KITCHEN[0]).sum()
        cb = (m & (gt == KITCHEN[1])).sum() / (gt == KITCHEN[1]).sum()
        best = max(best, min(ca, cb))
    return 'one mask' if best >= .25 else 'not together'


def fragments(frames, archive, labels):
    """GT units owned (majority of pixels) by >= 2 masks, multi-part furniture only."""
    furniture = {i for i, l in labels.items() if any(w in l for w in ('cabinet', 'shelf', 'counter', 'door', 'couch', 'bookshelf'))}
    count = 0
    for fid, masks in frames.items():
        gt = np.asarray(Image.open(io.BytesIO(archive.read(f'instance/{fid}.png')))).astype(np.int32) - 1
        own = Counter()
        for _, path in masks:
            m = np.asarray(Image.open(path)) > 0
            v = gt[m]; v = v[v >= 0]
            if not len(v):
                continue
            i, c = np.unique(v, return_counts=True)
            j = G.unit(int(i[c.argmax()]))
            if c.max() / m.sum() >= .5 and j in furniture:
                own[j] += 1
        count += sum(1 for n in own.values() if n >= 2)
    return count


def cost(run, fids):
    rs = [json.loads((run_dir_of(run) / f'frame_{f:06d}' / 'result.json').read_text()) for f in fids]
    return {'rounds': sum(r['rounds'] for r in rs), 'prompt_tokens': sum(r['usage']['prompt_tokens'] for r in rs),
            'kept': sum(len(r['kept']) for r in rs), 'skipped': sum(len(r['skipped']) for r in rs)}


def main():
    old, new = sys.argv[1], sys.argv[2]
    fo, fn = run_frames(old), run_frames(new)
    common = sorted(set(fo) & set(fn))
    fo = {f: fo[f] for f in common}; fn = {f: fn[f] for f in common}
    archive = zipfile.ZipFile(f'{G.SCENE}_2d-instance.zip')
    labels = G.gt_labels()
    text = {fid: masks for fid, masks in G.text_masks() if fid in common}
    print(f'{len(common)} common frames')
    rows = {}
    for name, frames in (('text', text), (old, fo), (new, fn)):
        merges, splits, n, _ = G.evaluate(frames.items())
        cab = Counter(classify(f, m, archive) for f, m in frames.items())
        kit = Counter(kitchen_outcome(f, m, archive) for f, m in frames.items())
        rows[name] = (n, len(merges), len(splits), fragments(frames, archive, labels), cab, kit, merges)
        print(f'\n== {name}: {n} masks | merged {len(merges)} | split {len(splits)} | furniture fragments '
              f'{rows[name][3]}')
        print('   cabinet#8/shelf#56:', dict(cab))
        print('   counter#3 + cabinet#42:', dict(kit))
        for k, v in Counter(' + '.join(sorted(f'{p[1]}#{p[0]}' for p in r['gt'])) for r in merges).most_common(6):
            print(f'     {v:3d}  {k}')
    print('\ncost', old, cost(old, common))
    print('cost', new, cost(new, common))
    per = []
    for f in common:
        a, b = classify(f, fo[f], archive), classify(f, fn[f], archive)
        if a != 'n/a':
            per.append((f, a, b))
    print('\ncabinet/shelf per frame (old -> new):')
    for f, a, b in per:
        print(f'   {f}: {a:10s} -> {b}')
    why = []
    for f in common:
        r = json.loads((run_dir_of(new) / f'frame_{f:06d}' / 'result.json').read_text())
        why += [(f, k['label'], k['why_one_unit']) for k in r['kept'] if k.get('why_one_unit')]
    print(f'\ncombine reasons in {new} ({len(why)}):')
    for f, l, w in why:
        print(f'   {f} {l!r}: {w[:150]}')


if __name__ == '__main__':
    main()
