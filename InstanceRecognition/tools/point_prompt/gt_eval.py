"""Score perception masks against the ScanNet 2D instance annotation (scene0000_00).

A mask MERGES when two or more non-structural GT objects each take a real share
of it (>= SHARE of the mask's pixels) and each is substantially inside it
(>= COVER of that object's visible pixels). A GT object is SPLIT when two or more
masks each cover >= COVER of it.

    .venv/bin/python -m tools.point_prompt.gt_eval --run point-full100_v2   # point-prompt run
    .venv/bin/python -m tools.point_prompt.gt_eval --text                    # text-prompt pipeline, same frames

Evaluation only: the perception never sees GT. Results go to
artifacts/experiments/<run>/perception/point_prompt/gt_eval.json (or analysis/ for --text).
"""
import argparse
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from src import config
from src.perception.point_prompt.objects import run_dir_of

SCENE = config.SCANNET_ROOT / 'scene0000_00' / 'scene0000_00'   # a file stem: SCENE.aggregation.json, SCENE_2d-instance.zip
SOURCE = config.ARTIFACTS_ROOT / 'sources/scene0000_00/scene_amb3r_full100_v1'
TEXT_OUT = config.ARTIFACTS_ROOT / 'experiments/gpt_sam31_uniform500/analysis/gt_eval_text_full100.json'


STRUCTURAL = {'wall', 'floor', 'ceiling', 'window', 'doorframe'}
SHARE, COVER = 0.15, 0.25
# GT objects that count as one unit for asset generation (user decision 2026-11-11):
# the worktop is part of the base-cabinet run it sits on.
SAME_UNIT = [{3, 42}]  # kitchen counter#3 + cabinet#42


def unit(i):
    for group in SAME_UNIT:
        if i in group:
            return min(group)
    return i


def gt_labels():
    agg = json.loads(Path(f'{SCENE}.aggregation.json').read_text())
    return {g['objectId']: g['label'] for g in agg['segGroups']}


def point_masks(run):
    for d in sorted(run_dir_of(run).glob('frame_*')):
        if not (d / 'result.json').exists():
            continue          # frame still running
        r = json.loads((d / 'result.json').read_text())
        yield r['frame_id'], [(f'{k["plan_id"]} {k["label"]}', d / k['mask_file']) for k in r['kept']]


def text_masks():
    m = json.loads((SOURCE / 'manifest.json').read_text())
    for f in m['frames']:
        yield f['source_frame_id'], [(f'#{i["instance_id"]} {i["class_name"]}', (SOURCE / i['mask_file']).resolve())
                                     for i in f['instances']]


def evaluate(frames):
    labels = gt_labels()
    archive = zipfile.ZipFile(f'{SCENE}_2d-instance.zip')
    merges, splits, n_masks, per_frame = [], [], 0, {}
    for fid, masks in frames:
        gt = np.asarray(Image.open(io.BytesIO(archive.read(f'instance/{fid}.png')))).astype(np.int32) - 1
        ids, counts = np.unique(gt[gt >= 0], return_counts=True)
        size = {int(i): int(c) for i, c in zip(ids, counts) if labels.get(int(i)) not in STRUCTURAL}
        cover_by_obj = {}
        frame_merges = []
        for name, path in masks:
            m = np.asarray(Image.open(path)) > 0
            n_masks += 1
            inside = gt[m]
            ids, counts = np.unique(inside[inside >= 0], return_counts=True)
            parts = []
            for i, c in zip(ids.tolist(), counts.tolist()):
                if i not in size:
                    continue
                share, cover = c / m.sum(), c / size[i]
                if cover >= COVER:
                    cover_by_obj.setdefault(i, []).append(name)
                if share >= SHARE and cover >= COVER:
                    parts.append((i, labels[i], round(share, 2), round(cover, 2)))
            if len({unit(p[0]) for p in parts}) >= 2:
                rec = {'frame': fid, 'mask': name, 'area': round(float(m.mean()), 3), 'gt': parts}
                merges.append(rec); frame_merges.append(rec)
        for i, names in cover_by_obj.items():
            if len(names) >= 2:
                splits.append({'frame': fid, 'gt': (i, labels[i]), 'masks': names})
        per_frame[fid] = frame_merges
    return merges, splits, n_masks, per_frame


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--run')
    g.add_argument('--text', action='store_true')
    ap.add_argument('--show', type=int, default=0, help='print this many merge records')
    args = ap.parse_args()
    frames = list(text_masks() if args.text else point_masks(args.run))
    merges, splits, n_masks, per_frame = evaluate(frames)
    name = 'text pipeline' if args.text else args.run
    print(f'{name}: {len(frames)} frames, {n_masks} masks')
    print(f'  merged masks (>=2 GT objects): {len(merges)} in {sum(1 for v in per_frame.values() if v)} frames')
    pairs = Counter(' + '.join(sorted(f'{p[1]}#{p[0]}' for p in r['gt'])) for r in merges)
    for k, v in pairs.most_common(12):
        print(f'    {v:3d}  {k}')
    print(f'  split GT objects (>=2 masks each covering >={COVER:.0%}): {len(splits)}')
    for k, v in Counter(f'{s["gt"][1]}#{s["gt"][0]}' for s in splits).most_common(8):
        print(f'    {v:3d}  {k}')
    for r in merges[:args.show]:
        print('   ', r)
    out = TEXT_OUT if args.text else run_dir_of(args.run) / 'gt_eval.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'merges': merges, 'splits': splits, 'masks': n_masks}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
