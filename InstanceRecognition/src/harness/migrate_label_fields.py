"""One-off schema migration for saved observations/objects, including embedded JSON.

No inference is performed. Proposal IDs remain historical opaque identifiers.
Unknown label-bearing structures abort before any writes. --apply records hashes.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT/'InstanceRecognition/runtime/experiments'


def migrate(value, counts):
    if isinstance(value, dict):
        target = None
        if 'label' in value:
            if 'seq' in value or ('mask_id' in value and 'bbox_xyxy' in value):
                target = 'category_label'
            elif (value.get('action') == 'create' or 'created_at' in value
                  or 'voxel_count' in value or 'representatives' in value
                  or ('observations' in value and str(value.get('id', '')).startswith('O'))):
                target = 'object_label'
            else:
                raise ValueError(f'Unrecognized label structure: {sorted(value)}')
            if target in value:
                raise ValueError(f'Duplicate old/new keys: {target}')
            counts[target] += 1
        return {(target if k == 'label' and target else k): migrate(v, counts) for k,v in value.items()}
    if isinstance(value, list):
        return [migrate(v, counts) for v in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return value
        if isinstance(parsed, (dict,list)):
            new = migrate(parsed, counts)
            if new != parsed:
                return json.dumps(new, ensure_ascii=False)
    return value


def decode(text, suffix):
    if suffix == '.json':
        return '', json.loads(text), '\n'
    match = re.match(r'(\s*(?:window\.)?[\w.]+\s*=\s*)', text)
    if not match:
        return None
    try:
        value, end = json.JSONDecoder().raw_decode(text[match.end():])
    except ValueError:
        return None
    tail = text[match.end()+end:]
    if tail.strip() not in ('', ';'):
        return None
    return match[0], value, tail


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    roots=[p for p in RUNS.iterdir() if p.name.startswith(('harness_', 'incremental_objects_'))]
    roots += [ROOT/'docs/assets/harness-tools', ROOT/'docs/assets/harness-records']
    changes=[];total=Counter()
    for root in roots:
        for p in sorted(root.rglob('*')):
            if p.suffix not in ('.json','.js') or 'code_versions' in p.parts: continue
            text=p.read_text();decoded=decode(text,p.suffix)
            if decoded is None: continue
            prefix,value,tail=decoded;counts=Counter();new=migrate(value,counts)
            if new==value: continue
            output=prefix+json.dumps(new,ensure_ascii=False,indent=2 if p.suffix=='.json' else None).replace('</','<\\/')+tail
            changes.append((p,output,{'path':str(p.relative_to(ROOT)), 'fields':dict(counts),
                'sha256_before':hashlib.sha256(p.read_bytes()).hexdigest(),
                'sha256_after':hashlib.sha256(output.encode()).hexdigest()}));total.update(counts)
    if args.apply:
        for p,output,_ in changes:
            temp=p.with_suffix(p.suffix+'.migration-tmp');temp.write_text(output);temp.replace(p)
        if changes:
            report=ROOT/'InstanceRecognition/runtime/docs/label_migration_manifest.json';report.parent.mkdir(parents=True,exist_ok=True)
            report.write_text(json.dumps({'date':'2026-09-22','fields':dict(total),
                'note':'Schema migration only; no inference. Historical code snapshots, prompt prose and opaque proposal IDs retained. Embedded JSON payload keys migrated.',
                'files':[x[2] for x in changes]},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'apply':args.apply,'files':len(changes),'fields':dict(total)}))

if __name__=='__main__':main()
