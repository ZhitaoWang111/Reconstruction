"""Ask each candidate model one Harness turn, and report time and behaviour.

Each model gets the same messages, the same tool list, and one request. The turn
is then *executed* through the real `Tools` object, because a model that picks an
unusable tool has not done the job however fast it answered -- a proposal naming
observations it never inspected is a different outcome from one that inspected
them first, and only running it shows the difference.

Nothing is committed and no state advances. A model's speed here is the speed of
one turn, not of a window.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.harness.client import Client, GatewayNetworkError
from src.harness.inputs import build_window_content
from src.harness.observations import Observations
from src.harness.state import initial, save
from src.harness.tools import TOOLS, Tools

from .window_input import DEFAULT_OUT, DEFAULT_SOURCE, build, load

# Enough of the candidate field to compare, and bounded so one long model does
# not bury the others in the report.
CANDIDATES = (
    'openai/gpt-6-sol',
    'openai/gpt-6.1-sol',
    'openai/gpt-6-astra',
    'anthropic/claude-opus-5-5',
    'anthropic/claude-sonnet-5-5',
    'gemini/gemini-4-argon',
    'xai/grok-4.7',
    'dashscope/qwen3.8-max',
    'deepseek/deepseek-v4.1-flash',
    'zhipu/glm-5.3',
)


def _arguments(call):
    try:
        return json.loads(call['function']['arguments'])
    except (KeyError, TypeError, json.JSONDecodeError):
        return None


def probe(model, messages, out, timeout, reasoning_effort=None):
    """One turn against `model`, executed, with the timing it took.

    A model that the gateway rejects is a result, not a crash: the pilot exists to
    find out which of these are usable on this exact input.
    """
    model_root = Path(out).resolve() / 'turns' / model.replace('/', '__')
    model_root.mkdir(parents=True, exist_ok=True)

    record = {'model': model, 'reasoning_effort': reasoning_effort}
    try:
        client = Client(model, model_root, timeout=timeout,
                        reasoning_effort=reasoning_effort,
                        network_retries=0, retry_backoff=0.0)
    except (RuntimeError, ValueError) as exc:
        record.update(ok=False, error=f'{type(exc).__name__}: {exc}')
        return record

    tick = time.monotonic()
    try:
        message = client.call(messages, TOOLS, 'turn', final=False)
    except (GatewayNetworkError, RuntimeError, ValueError) as exc:
        record.update(ok=False, elapsed_seconds=round(time.monotonic() - tick, 3),
                      error=f'{type(exc).__name__}: {str(exc)[:600]}')
        return record
    record['elapsed_seconds'] = round(time.monotonic() - tick, 3)

    # `Client` writes this on every completed request. Treat its absence as a
    # missing measurement rather than a crash: a model that answered without
    # producing a tool call has still told us something.
    meta_path = model_root / 'calls' / 'turn_0_meta.json'
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        record['usage'] = meta.get('usage')
        record['returned_model'] = meta.get('returned_model')

    calls = message.get('tool_calls') or []
    record['tool_calls'] = [c['function']['name'] for c in calls]
    record['finish_reason'] = 'tool_calls' if calls else 'none'
    if calls:
        record['arguments'] = _arguments(calls[0])
    record['ok'] = bool(calls)

    # Execute the turn against a throwaway state, so a bad tool choice is visible
    # as an error rather than as an impressive-looking latency.
    if calls:
        record['execution'] = _execute(calls, out)
    return record


def _execute(calls, out):
    """Run the model's turn through the real Tools against a throwaway state."""
    data = Observations(DEFAULT_SOURCE, 5, Path(out).resolve() / 'window')
    tools = Tools(data, initial(), 4)
    build_window_content(data, tools, 0, 4, max_calls=4)
    results = []
    for call in calls:
        name = call['function']['name']
        args = _arguments(call)
        try:
            if args is None:
                raise ValueError('arguments are not valid JSON')
            result, _ = tools.call(name, args)
            results.append({'tool': name, 'ok': True, 'summary': _summarize(result)})
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            results.append({'tool': name, 'ok': False,
                            'error': f'{type(exc).__name__}: {str(exc)[:200]}'})
    return results


def _summarize(result, limit=160):
    """What the result says, small enough to read in a table."""
    if isinstance(result, dict):
        if 'error' in result:
            return result['error'][:limit]
        if 'proposal_id' in result:
            actions = result.get('actions') or []
            kinds = [a.get('action') for a in actions if isinstance(a, dict)]
            return f"proposal {result['proposal_id']}: {len(actions)} actions {kinds}"
        if 'observations' in result and isinstance(result['observations'], list):
            return f"{len(result['observations'])} observations"
        if 'objects' in result:
            return f"{len(result['objects'])} objects, version {result.get('version')}"
    return str(result)[:limit]


def run(models, out=DEFAULT_OUT, timeout=300, reasoning_effort=None):
    """One turn per model, written to `turns/<model>/` and to a single report."""
    messages, _ = None, None
    try:
        messages = load(out)
    except SystemExit:
        messages, _ = build(out=out)
    out = Path(out).resolve()
    records = []
    for model in models:
        print(f'TURN {model} ...', flush=True)
        record = probe(model, messages, out, timeout, reasoning_effort)
        state = 'ok' if record.get('ok') else 'FAILED'
        print(f'  {state} {record.get("elapsed_seconds", "?")}s '
              f'{record.get("error", "") or record.get("tool_calls", "")}', flush=True)
        records.append(record)
        save(out / 'turns' / f'{model.replace("/", "__")}.json', record)
    save(out / 'report.json', {'models': records, 'timeout': timeout})
    return records


def summarize(records):
    ok = [r for r in records if r.get('ok')]
    ok.sort(key=lambda r: r['elapsed_seconds'])
    lines = [f'{"model":34} {"sec":>7} {"tools":>22} {"tokens":>9} {"forced":>7}  result']
    for r in sorted(records, key=lambda r: r.get('elapsed_seconds', 1e9)):
        if not r.get('ok'):
            lines.append(f'{r["model"]:34} {"-":>7} {"-":>22} {"-":>9} '
                         f'{str(r.get("tool_choice", "")):>7}  '
                         f'{r.get("error", "failed")[:60]}')
            continue
        usage = (r.get('usage') or {}).get('total_tokens')
        tools = ','.join(r.get('tool_calls') or []) or '(none)'
        execution = r.get('execution') or []
        verdict = '; '.join(
            f'{e["tool"]}={"ok" if e["ok"] else "ERROR"}' for e in execution) or '(not executed)'
        # A turn taken under `auto` may answer with prose instead of acting, so the
        # choice is shown rather than left to be inferred from the tool list.
        forced = 'yes' if r.get('tool_choice') == 'required' else 'no'
        lines.append(f'{r["model"]:34} {r["elapsed_seconds"]:7.1f} {tools:>22} '
                     f'{str(usage):>9} {forced:>7}  {verdict[:60]}')
    lines.append('')
    lines.append(f'{len(ok)}/{len(records)} returned a usable tool call')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', default=list(CANDIDATES))
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--reasoning-effort', default=None,
                        help='omit to use each model\'s own default')
    args = parser.parse_args()
    records = run(args.models, args.out, args.timeout, args.reasoning_effort)
    print()
    print(summarize(records))


if __name__ == '__main__':
    main()
