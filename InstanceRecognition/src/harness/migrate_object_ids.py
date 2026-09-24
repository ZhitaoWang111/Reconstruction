"""Migrate saved Harness object identifiers from O1 to Object1 without inference.

Touches standalone identifier tokens in parsed data, including JSON embedded in
message strings and human-readable trace text. Numeric viewer IDs, observation
IDs, opaque proposal IDs, image payloads and code-version snapshots are retained.
The manifest records before/after hashes; migrated traces are not original bytes.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from .migrate_label_fields import ROOT, RUNS, decode

TOKEN = re.compile(r'(?<![A-Za-z0-9_])O([1-9][0-9]*)(?![A-Za-z0-9_])')


def rename(value, counts):
    def replace(match):
        counts['identifier_tokens'] += 1
        return 'Object' + match.group(1)
    counts['format_placeholders'] += value.count('O<number>')
    value = value.replace('O<number>', 'Object<number>')
    return TOKEN.sub(replace, value)


def migrate(value, counts):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            new_key = rename(key, counts)
            if new_key in result:
                raise ValueError(f'Identifier key collision: {new_key}')
            result[new_key] = migrate(item, counts)
        return result
    if isinstance(value, list):
        return [migrate(item, counts) for item in value]
    if isinstance(value, str):
        # Images are opaque data, never text or identifier references.
        if value.startswith('data:'):
            return value
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return rename(value, counts)
        if isinstance(parsed, (dict, list)):
            new = migrate(parsed, counts)
            return json.dumps(new, ensure_ascii=False) if new != parsed else value
        return rename(value, counts)
    return value


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    roots = sorted(p for p in RUNS.iterdir() if p.name.startswith(('harness_', 'incremental_objects_')))
    roots += [ROOT / 'docs/assets/harness-tools', ROOT / 'docs/assets/harness-records']
    changes, renames, total = [], [], Counter()
    for root in roots:
        for path in sorted(root.rglob('*')):
            if not path.is_file() or 'code_versions' in path.parts:
                continue
            if path.suffix == '.npz' and re.fullmatch(r'O[1-9][0-9]*', path.stem):
                target = path.with_name('Object' + path.name[1:])
                if target.exists():
                    raise ValueError(f'Cloud destination exists: {target}')
                renames.append((path, target, sha(path.read_bytes())))
            if path.suffix not in ('.json', '.js'):
                continue
            text = path.read_text()
            decoded = decode(text, path.suffix)
            if decoded is None:
                continue
            prefix, value, tail = decoded
            counts = Counter()
            new = migrate(value, counts)
            if new == value:
                continue
            output = prefix + json.dumps(new, ensure_ascii=False, indent=2 if path.suffix == '.json' else None).replace('</', '<\\/') + tail
            changes.append((path, output, {
                'path': str(path.relative_to(ROOT)), 'counts': dict(counts),
                'sha256_before': sha(path.read_bytes()), 'sha256_after': sha(output.encode()),
            }))
            total.update(counts)
    if args.apply:
        for path, output, _ in changes:
            temp = path.with_suffix(path.suffix + '.migration-tmp')
            temp.write_text(output)
            temp.replace(path)
        for path, target, _ in renames:
            path.rename(target)
        if changes or renames:
            report = ROOT / 'InstanceRecognition/runtime/docs/object_id_migration_manifest.json'
            report.parent.mkdir(parents=True, exist_ok=True)
            previous = json.loads(report.read_text()) if report.exists() else {}
            file_entries = {item['path']: item for item in previous.get('files', [])}
            for _, _, item in changes:
                if item['path'] in file_entries:
                    item['sha256_before'] = file_entries[item['path']]['sha256_before']
                file_entries[item['path']] = item
            report.write_text(json.dumps({
                'date': '2026-09-22', 'counts': dict(Counter(previous.get('counts', {})) + total),
                'note': 'Identifier schema migration only; no inference. Historical trace text and embedded payloads now use ObjectN and are not byte-original responses. Numeric viewer IDs, observation IDs, image data, opaque proposal IDs, historical hashes and code-version snapshots retained. Cloud bytes unchanged.',
                'files': list(file_entries.values()),
                'renames': previous.get('renames', []) + [{'from': str(p.relative_to(ROOT)), 'to': str(t.relative_to(ROOT)), 'sha256': digest} for p,t,digest in renames],
            }, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'apply': args.apply, 'files': len(changes), 'clouds_renamed': len(renames), 'counts': dict(total)}))


if __name__ == '__main__':
    main()
