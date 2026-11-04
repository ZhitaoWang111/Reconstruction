"""Batch RGB -> GPT category prompts. No masks or temporal context enter the request."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from src import config
from src.gateway import settings
from .io import perception_dir, read_json, sha256, write_json

CATEGORY_FIELDS = {
    'category': {'type': 'string'}, 'sam_prompt': {'type': 'string'},
    'certainty': {'type': 'string', 'enum': ['clear', 'uncertain']},
    'visible_scope': {'type': 'string', 'enum': ['whole', 'partial', 'unclear']},
    'evidence': {'type': 'string'}, 'ambiguity': {'type': 'string'},
}
REGION_FIELDS = {key: {'type': 'string'} for key in ('location', 'reason')}
SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['categories', 'unresolved_regions'],
    'properties': {
        name: {'type': 'array', 'items': {
            'type': 'object', 'additionalProperties': False,
            'required': list(fields), 'properties': fields,
        }} for name, fields in [('categories', CATEGORY_FIELDS), ('unresolved_regions', REGION_FIELDS)]
    },
}


def validate(data):
    if not isinstance(data, dict) or set(data) != set(SCHEMA['required']):
        raise ValueError('Invalid labeling schema')
    for name, fields in [('categories', CATEGORY_FIELDS), ('unresolved_regions', REGION_FIELDS)]:
        if not isinstance(data[name], list):
            raise ValueError('Expected arrays')
        seen = set()
        for item in data[name]:
            if not isinstance(item, dict) or set(item) != set(fields):
                raise ValueError('Invalid labeling fields')
            for key, spec in fields.items():
                if not isinstance(item[key], str):
                    raise ValueError('Expected string field')
                if 'enum' in spec and item[key] not in spec['enum']:
                    raise ValueError('Invalid enum')
            if name == 'categories':
                if not item['category'].strip() or not item['sam_prompt'].strip():
                    raise ValueError('Empty category/prompt')
                category = item['category'].strip().casefold()
                if category in seen:
                    raise ValueError('Repeated category')
                seen.add(category)
    return data


def _response_labels(body):
    try:
        choice = body['choices'][0]
        if choice['finish_reason'] != 'stop':
            raise ValueError('Incomplete model response')
        return validate(json.loads(choice['message']['content']))
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ValueError('Malformed gateway response') from exc


def label_one(record, output, model, base, key, prompt, attempts=3, force=False):
    if attempts < 1:
        raise ValueError('attempts must be positive')
    output = Path(output)
    source_bytes = Path(record['source']).read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != record['source_sha256']:
        raise ValueError('RGB changed since the batch was prepared')
    sequence = record['sequence_index']
    target = output / f'labels/{sequence:04d}.json'
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    provenance = {'source_sha256': record['source_sha256'],
                  'requested_model': model, 'prompt_sha256': prompt_hash}
    if target.exists() and not force:
        try:
            cached = read_json(target)
        except ValueError:
            cached = None
        if isinstance(cached, dict) and isinstance(cached.get('labels'), dict):
            if any(cached.get(k) != v for k, v in provenance.items()):
                raise ValueError('Cached labels came from a different image, model or prompt')
            validate(cached['labels'])
            return cached
    if target.exists():
        target.replace(target.with_suffix('.superseded.json'))
    if not key:
        raise ValueError('Missing VIBEG_API_KEY')
    payload = {
        'model': model, 'reasoning_effort': 'none', 'max_completion_tokens': 4096,
        'messages': [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {
                'url': 'data:image/jpeg;base64,' + base64.b64encode(source_bytes).decode(),
                'detail': 'high',
            }}]},
        ],
        'response_format': {'type': 'json_schema', 'json_schema': {
            'name': 'scene_categories', 'strict': True, 'schema': SCHEMA}},
    }
    started, failures = time.monotonic(), []
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(base.rstrip('/') + '/chat/completions',
                                        data=json.dumps(payload).encode(), headers={
                                            'Authorization': 'Bearer ' + key,
                                            'Content-Type': 'application/json'})
        raw, body = None, None
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                raw = response.read().decode().replace(key, '[REDACTED]')
            body = json.loads(raw)
            labels = _response_labels(body)
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as exc:
            retry = True
            message = f'{type(exc).__name__}: {exc}'
            if isinstance(exc, urllib.error.HTTPError):
                message = f'HTTP {exc.code}: ' + exc.read(1500).decode(errors='replace')
                retry = exc.code in (408, 429, 500, 502, 503, 504)
            message = message.replace(key, '[REDACTED]')
            failures.append(message)
            write_json(output / f'responses/{sequence:04d}_attempt{attempt}.json',
                       {'error': message, 'body': body, 'raw': raw})
            if retry and attempt < attempts:
                time.sleep(min(2 ** attempt, 8))
                continue
            break
        # Disk errors must stop the run, not repeat a paid request.
        write_json(output / f'responses/{sequence:04d}_attempt{attempt}.json', body)
        result = {**record, **provenance, 'labels': labels, 'returned_model': body.get('model'),
                  'usage': body.get('usage'), 'response_id': body.get('id'), 'attempts': attempt,
                  'prior_errors': failures, 'elapsed_seconds': round(time.monotonic() - started, 3)}
        write_json(target, result)
        (output / f'errors/{sequence:04d}.json').unlink(missing_ok=True)
        print(f'LABELED {sequence}: {len(labels["categories"])} categories', flush=True)
        return result
    result = {**record, 'error': failures, 'elapsed_seconds': round(time.monotonic() - started, 3)}
    write_json(output / f'errors/{sequence:04d}.json', result)
    return result


def build_records(batch_path, frames_dir):
    batch = read_json(batch_path)
    scene = batch['scene_id']
    ids = batch['frame_indices']
    if not ids or any(type(fid) is not int or fid < 0 for fid in ids) or len(set(ids)) != len(ids):
        raise ValueError('Batch needs unique nonnegative frame indices')
    records = []
    for seq, fid in enumerate(ids):
        source = (Path(frames_dir) / scene / f'frame_{fid:06d}.jpg').resolve()
        if not source.is_file():
            raise SystemExit(f'Missing frame image: {source}')
        records.append({'scene_id': scene, 'sequence_index': seq, 'source_frame_id': fid,
                        'source': str(source), 'source_sha256': sha256(source)})
    return scene, batch, records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='Experiment name under artifacts/experiments')
    parser.add_argument('--batch', required=True, type=Path, help='JSON: scene_id and frame_indices')
    parser.add_argument('--frames-dir', required=True, type=Path, help='Parent directory of scene RGB frames')
    parser.add_argument('--model', default=config.GPT_MODEL)
    parser.add_argument('--prompt-file', type=Path, default=config.PROMPT_ROOT / 'prompt_en.txt')
    parser.add_argument('--first', type=int, default=0)
    parser.add_argument('--count', type=int)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(argv)
    if args.first < 0 or (args.count is not None and args.count < 1) or not 1 <= args.workers <= 8:
        parser.error('Require first >= 0, count >= 1 and workers in 1..8')
    scene, _, records = build_records(args.batch, args.frames_dir)
    records = records[args.first:args.first + args.count if args.count is not None else None]
    if not records:
        parser.error('No frames selected')
    output = perception_dir(args.run) / 'labels'
    prompt = args.prompt_file.read_text()
    credentials = settings()
    base = credentials.get('VIBEG_OPENAI_BASE_URL', 'https://vibeg.seer-robotics.cn/openai/v1').rstrip('/')
    run_config = {'format': 'gpt_categories_v2', 'scene_id': scene, 'model': args.model,
                  'base_url': base, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                  'schema': SCHEMA, 'frame_count': len(records), 'frames': records,
                  'batch_sha256': sha256(args.batch), 'reasoning_effort': 'none',
                  'image_detail': 'high', 'max_completion_tokens': 4096}
    config_path = output / 'config.json'
    if config_path.exists() and read_json(config_path) != run_config:
        raise ValueError('Run inputs changed; choose a new --run')
    write_json(config_path, run_config)
    if args.prepare_only:
        print(f'Prepared {len(records)} frames: {output}')
        return output
    key = credentials.get('VIBEG_API_KEY', '')
    if not key:
        raise ValueError('Missing VIBEG_API_KEY')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda r: label_one(r, output, args.model, base, key, prompt), records))
    failures = [r['sequence_index'] for r in results if 'error' in r]
    write_json(output / 'summary.json', {'requested_frames': len(records),
               'completed_frames': len(records) - len(failures), 'failed_frames': failures})
    if failures:
        raise SystemExit(f'Failed frames {failures}; rerun the same command to resume')
    return output


if __name__ == '__main__':
    main()
