"""Publish immutable harness logs, loading one window at a time.

Reasons are shown in their original English; this tree does not translate them.
Source logs are read-only.

Reads the pipeline records from the run directory and writes the page and its
resources to the site directory. The two are separate trees, so every path the
page emits is computed relative to the site directory rather than assumed to be
a sibling of the logs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.harness.state import dumps_sanitized
from src.visualization.page_naming import canonical_run_id, records_page_name, records_asset_name
from src import config as app_config
TEMPLATES = app_config.WEB_ROOT / 'records'
LOG_NAME = re.compile(r'w(\d+)_(\d+)_call(\d+)_(request|tool\d+|\d+_(?:response|meta|error))\.json')


def dump_js(path, prefix, value):
    path.write_text(prefix + dumps_sanitized(value, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/') + ';\n')


def _display_path(path):
    """Repo-relative when the file is inside this tree, absolute otherwise.

    The records root is redirectable, so a log is not guaranteed to live under
    ROOT; falling back to the absolute path keeps the auditor's reference usable
    instead of raising.
    """
    path = Path(path).resolve()
    for base in (app_config.ROOT, app_config.ARTIFACTS_ROOT):
        try:
            return str(path.relative_to(base))
        except ValueError:
            continue
    return str(path)


def rank(event):
    kind = event['kind']
    if kind == 'request':
        return 0, 0, 0
    if kind.startswith('tool'):
        return 2, int(kind[4:]), 0
    attempt, typ = kind.split('_')
    return 1, int(attempt), {'response': 0, 'meta': 1, 'error': 2}[typ]


def compact(event):
    """Statistics retain only real tool execution records, never request repeats."""
    output = {k: event[k] for k in ('kind', 'file', 'url', 'sha256')}
    data = event['data']
    if event['kind'].startswith('tool'):
        output['data'] = {'name': data.get('name'), 'arguments': data.get('arguments'),
                          'result': {k: v for k, v in data.get('result', {}).items()
                                     if k in ('error', 'proposal_id', 'committed', 'actions',
                                              'status', 'checks')}}
    else:
        output['data'] = {}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='Experiment name; determines the default run/site directories')
    parser.add_argument('--run-dir', type=Path, help='Pipeline-record directory (default: <artifacts>/experiments/<run>)')
    parser.add_argument('--site-dir', type=Path, help='Pages directory (default: <site>/experiments/<run>)')
    parser.add_argument('--page', help='Optional HTML filename override for legacy reports')
    args = parser.parse_args()
    try:
        args.run = canonical_run_id(args.run)
        args.page = args.page or records_page_name(args.run)
        asset_name = records_asset_name(args.page)
    except ValueError as exc:
        parser.error(str(exc))
    folder = (args.run_dir.expanduser().resolve() if args.run_dir
              else app_config.ARTIFACTS_ROOT / 'experiments' / args.run)
    site = (args.site_dir.expanduser().resolve() if args.site_dir
            else app_config.SITE_ROOT / 'experiments' / args.run)
    if not (folder / 'config.json').is_file() or not (folder / 'calls').is_dir():
        parser.error(f'Missing config.json or calls/ in {folder}')
    # The page sits in the site tree; its resources sit beside it, and links back
    # to the logs are computed relative to the page.
    page = site / args.page
    assets = site / asset_name
    site.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    windows, count = {}, 0
    for file in sorted((folder / 'calls').glob('*.json')):
        match = LOG_NAME.fullmatch(file.name)
        if not match:
            raise ValueError(f'Unrecognized log filename: {file}')
        start, session, call, kind = match.groups()
        raw = file.read_bytes()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            print('Skipped incomplete live log:', file.name)
            continue
        event = {'kind': kind, 'file': _display_path(file),
                 'url': os.path.relpath(file, page.parent),
                 'sha256': hashlib.sha256(raw).hexdigest(), 'data': data}
        windows.setdefault(int(start), {}).setdefault(session, {}).setdefault(int(call), []).append(event)
        count += 1
    manifest = []
    for start, sessions in sorted(windows.items()):
        win = {'start': start, 'sessions': [{'id': sid, 'calls': [{'index': index, 'events': sorted(events, key=rank)}
                 for index, events in sorted(calls.items())]} for sid, calls in sorted(sessions.items())]}
        cache_key = f'{args.run}:{start}'
        file = assets / f'window-{start:04d}.js'
        dump_js(file, 'window.AGENT_WINDOW_CACHE=window.AGENT_WINDOW_CACHE||{};window.AGENT_WINDOW_CACHE[' + json.dumps(cache_key) + ']=', win)
        stub = {'start': start, 'cache_key': cache_key, 'data_url': os.path.relpath(file, page.parent),
                'sessions': [{'id': session['id'], 'calls': [{'index': call['index'], 'events': [compact(e) for e in call['events']]}
                   for call in session['calls']]} for session in win['sessions']]}
        manifest.append(stub)
    config = json.loads((folder / 'config.json').read_text())
    dump_js(assets / 'records.js', 'window.AGENT_RECORDS=', [{'name': args.run, 'config': config, 'windows': manifest}])
    for name in ('page.js', 'page.css'):
        (assets / name).write_bytes((TEMPLATES / name).read_bytes())
    coverage = f'本次快照：{len(manifest)} 个有日志的窗口。'
    html = (TEMPLATES / 'page.html').read_text()
    # The shared navigation script and its stylesheet are hand-written source that
    # lives outside this tree; link to them relative to the page rather than
    # assuming a fixed depth.
    docs_root = app_config.NAV_ROOT
    html = (html.replace('@@ASSETS@@', os.path.relpath(assets, page.parent))
                .replace('@@NAV@@', os.path.relpath(docs_root / 'assets', page.parent))
                .replace('@@DOCS@@', os.path.relpath(docs_root, page.parent))
                .replace('@@RUN@@', args.run)
                .replace('@@COVERAGE@@', coverage)
                .replace('@@MODEL@@', str(config.get('model', 'unknown')))
                .replace('@@FRAMES@@', str(config.get('frames', 'unknown')))
                .replace('@@WINDOW@@', str(config.get('window', 'unknown'))))
    page.write_text(html)
    print(coverage)
    print(f'Exported {count} original logs to {page}')


if __name__ == '__main__':
    main()
