"""Ask a Claude model one Harness turn through the gateway's Claude route.

Two routes exist and only one works for this job:

* `/anthropic/v1/messages` — the native Messages API. It refuses a forced tool
  choice (`tool_choice: any`) for these models, and the Harness forces a tool call
  on every non-final turn. Asking with `auto` instead is not a substitute: Haiku
  answers with a page of scene description and stops (`end_turn`, no `tool_use`),
  which measures its willingness to narrate rather than its ability to act.
* `/openai/v1/chat/completions` — accepts `tool_choice: required` for
  `anthropic/claude-*` and returns tool calls normally.

So this module posts the Harness's own messages, unmodified, to the OpenAI-shaped
route. That also means the input is byte-identical to what the other models get,
which is what makes the timing comparison mean anything.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from src.harness.state import save
from src.harness.tools import TOOLS

KEY_FILE = Path('/home/seer/wzt/Gateway/key.json')

# The gateway routes Claude here, and this route honours a forced tool choice --
# but only for some models. The newer revisions reject `tool_choice: required`
# with HTTP 400 ("type \"tool\" and \"any\" are not supported"), so the ones that
# cannot be forced are asked with `auto` and reported as such: under `auto` a
# model is free to answer with prose, and that is a real difference in behaviour
# rather than a harness limitation.
FORCED_TOOL_CHOICE = {
    'anthropic/claude-haiku-5-5': 'required',
    'anthropic/claude-sonnet-5': 'required',
}
DEFAULT_TOOL_CHOICE = 'auto'

DEFAULT_MODELS = ('anthropic/claude-opus-5-5', 'anthropic/claude-sonnet-5-5',
                  'anthropic/claude-fable-5-1', 'anthropic/claude-haiku-5-5')


def credentials(path=KEY_FILE):
    """The gateway key and OpenAI-compatible base URL from the shared key file."""
    config = json.loads(Path(path).read_text())
    base = (config.get('OpenAI_URL') or '').rstrip('/')
    key = config.get('token') or config.get('Authorization', '').removeprefix('Bearer ')
    if not base or not key:
        raise SystemExit(f'{path} must carry OpenAI_URL and token')
    return key, base


def probe(model, messages, out, timeout=600, max_completion_tokens=10000,
          tool_choice=None):
    """One turn, recorded in the same shape as the other pilots."""
    model_root = Path(out).resolve() / 'turns' / model.replace('/', '__')
    model_root.mkdir(parents=True, exist_ok=True)
    choice = tool_choice or FORCED_TOOL_CHOICE.get(model, DEFAULT_TOOL_CHOICE)
    record = {'model': model, 'api': 'openai_chat_completions', 'tool_choice': choice}

    try:
        key, base = credentials()
    except (SystemExit, OSError, json.JSONDecodeError) as exc:
        record.update(ok=False, error=f'{type(exc).__name__}: {exc}')
        return record

    payload = {'model': model, 'max_completion_tokens': max_completion_tokens,
               'messages': messages, 'tools': TOOLS, 'parallel_tool_calls': False,
               'tool_choice': choice, 'stream': False}
    # The request carries the whole window; the audit copy keeps only its shape.
    save(model_root / 'calls' / 'turn_0_request.json',
         {**{k: v for k, v in payload.items() if k != 'messages'},
          'messages': f'[{len(messages)} messages, '
                      f'{len(json.dumps(messages)) // 1024} KB omitted]'})

    request = urllib.request.Request(
        base + '/v1/chat/completions', json.dumps(payload).encode(),
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    tick = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        record.update(ok=False, elapsed_seconds=round(time.monotonic() - tick, 3),
                      error=f'HTTP {exc.code}: {exc.read(500).decode(errors="replace")[:400]}')
        return record
    except Exception as exc:
        record.update(ok=False, elapsed_seconds=round(time.monotonic() - tick, 3),
                      error=f'{type(exc).__name__}: {str(exc)[:400]}')
        return record
    record['elapsed_seconds'] = round(time.monotonic() - tick, 3)

    save(model_root / 'calls' / 'turn_0_response.json', body)
    choice = (body.get('choices') or [{}])[0]
    message = choice.get('message') or {}
    calls = message.get('tool_calls') or []
    record['returned_model'] = body.get('model')
    record['usage'] = body.get('usage')
    record['finish_reason'] = choice.get('finish_reason')
    record['tool_calls'] = [c['function']['name'] for c in calls]
    record['ok'] = bool(calls)
    if calls:
        try:
            record['arguments'] = json.loads(calls[0]['function']['arguments'])
        except json.JSONDecodeError:
            record['arguments'] = calls[0]['function']['arguments']
    if message.get('content'):
        record['text'] = str(message['content'])[:600]
    return record


def main():
    import argparse
    from .run_pilot import summarize
    from .window_input import load

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', default=list(DEFAULT_MODELS))
    parser.add_argument('--out', type=Path, default=Path('artifacts/pilot_experiment'))
    parser.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args()

    messages = load(args.out)
    records = []
    for model in args.models:
        print(f'TURN {model} ...', flush=True)
        record = probe(model, messages, args.out, args.timeout)
        print(f'  {"ok" if record.get("ok") else "FAILED"} '
              f'{record.get("elapsed_seconds", "?")}s '
              f'{record.get("error", "") or record.get("tool_calls", "")}', flush=True)
        records.append(record)
        save(Path(args.out).resolve() / 'turns' / f'{model}.json', record)
    print()
    print(summarize(records))


if __name__ == '__main__':
    main()
