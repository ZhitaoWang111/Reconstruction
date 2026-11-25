"""Run one batch end to end: baselines, the agent session, and the holdout score.

    .venv/bin/python src/legacy/run_experiment.py --name harness-v1-opus55-batch1-20261111

Writes runs/<name>/: config.json, baselines.json, states/, images/, calls.jsonl,
summary.json, eval.json. A rerun with the same name refuses to overwrite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

import agent
import evaluate
import handoff
from scene import Scene

BATCH_1 = ['obj_0037', 'obj_0049', 'obj_0048', 'obj_0062', 'obj_0016', 'obj_0061']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--objects', nargs='+', default=BATCH_1)
    ap.add_argument('--effort', default='high', choices=['low', 'medium', 'high', 'xhigh', 'max'])
    ap.add_argument('--max-turns', type=int, default=60)
    ap.add_argument('--budget-tokens', type=int, default=1_500_000)
    args = ap.parse_args()

    out = HERE.parent / 'runs' / args.name
    if out.exists():
        raise SystemExit(f'{out} exists; choose a new name')
    out.mkdir(parents=True)
    hs = handoff.load()
    code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob('*.py'))}
    config = {'name': args.name, 'model': agent.MODEL, 'effort': args.effort, 'objects': args.objects,
              'max_turns': args.max_turns, 'budget_tokens': args.budget_tokens, 'scene_id': hs.scene_id,
              'asset_handoff': str(handoff.ASSET_HANDOFF), 'system_sha256': hashlib.sha256(agent.SYSTEM.encode()).hexdigest(),
              'code_sha256': code, 'started': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    (out / 'config.json').write_text(json.dumps(config, indent=1))

    scene = Scene(hs, args.objects, out)
    ev = evaluate.Evaluator(scene)
    base = evaluate.baselines(scene, ev)
    evaluate.write(out / 'baselines.json', base)
    print('baselines (holdout IoU):', {k: round(float(np.mean([r[k] for r in base.values()])), 3)
                                      for k in ('B0_bbox_yaw0', 'B1_sam3d', 'B2_best_residual', 'oracle_best_candidate')}, flush=True)

    summary = agent.run_batch(scene, out, effort=args.effort, max_turns=args.max_turns, budget_tokens=args.budget_tokens)
    print('agent:', {k: summary[k] for k in ('turns', 'finished', 'tokens', 'seconds')}, flush=True)

    result = {}
    for oid in args.objects:
        p = scene.placed.get(oid)
        result[oid] = {'agent': evaluate_or_none(ev, oid, p), 'status': p.status if p else 'unplaced',
                       'flag': scene.flags.get(oid), **{k: base[oid][k] for k in ('B0_bbox_yaw0', 'B1_sam3d', 'B2_best_residual', 'oracle_best_candidate')}}
    placed = [r for r in result.values() if r['agent'] is not None]
    means = {k: round(float(np.mean([r[k] for r in placed])), 4) for k in ('agent', 'B0_bbox_yaw0', 'B1_sam3d', 'B2_best_residual', 'oracle_best_candidate')} if placed else {}
    evaluate.write(out / 'eval.json', {'per_object': result, 'mean_over_placed': means, 'placed': len(placed), 'of': len(args.objects)})
    print('holdout IoU, mean over placed objects:', means, flush=True)


def evaluate_or_none(ev, oid, placement):
    return ev.score(oid, placement.pose)['iou'] if placement is not None else None


if __name__ == '__main__':
    main()
