"""Assemble the one window input every pilot model is asked to answer.

The input is the Harness's own: `build_window_content` over the real lifted
source, the real prompt, and the real tool list. Building it once and handing the
same bytes to each model is what makes a timing comparison mean anything -- if
each model got a freshly built window, a difference in the input would be
indistinguishable from a difference in the model.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src import config
from src.harness.observations import Observations
from src.harness.inputs import build_window_content
from src.harness.state import initial
from src.harness.state import save
from src.harness.tools import Tools

DEFAULT_SOURCE = config.ARTIFACTS_ROOT / 'sources/scene0000_00/scene_amb3r_v1'
DEFAULT_OUT = config.ARTIFACTS_ROOT / 'pilot_experiment'
PROMPT = config.ROOT / 'src/harness/prompt.txt'


def build(source=DEFAULT_SOURCE, frames=5, window=5, first=0, out=DEFAULT_OUT,
          rebuild=False, max_calls=4):
    """Write the window input and return (messages, out).

    `messages` is the exact list the Harness would post: the system prompt plus
    the window content. `frames` may exceed the window; only the window is sent.

    `max_calls` is only used to fill the `max_calls` field the content carries, so
    the model plans against the same budget it would really have. The pilot still
    stops after one request.
    """
    source, out = Path(source).resolve(), Path(out).resolve()
    window_root = out / 'window'
    if window_root.exists():
        if not rebuild:
            raise SystemExit(f'{window_root} exists; pass --rebuild to replace it')
        shutil.rmtree(window_root)
    window_root.mkdir(parents=True)

    data = Observations(source, frames, window_root)
    cutoff = min(first + window, frames) - 1
    tools = Tools(data, initial(), cutoff)
    content = build_window_content(data, tools, first, cutoff, max_calls=max_calls)

    messages = [{'role': 'system', 'content': PROMPT.read_text()},
                {'role': 'user', 'content': content}]
    save(window_root / 'messages.json', messages)
    save(window_root / 'window.json', {
        'source': str(source), 'frames': frames, 'window': [first, cutoff],
        'observations': len(data.items), 'prompt': str(PROMPT),
        'content_parts': len(content), 'max_calls': max_calls,
        'bytes': len(json.dumps(messages, ensure_ascii=False).encode()),
    })
    return messages, window_root


def load(out=DEFAULT_OUT):
    """The built window input, or a refusal naming how to build it."""
    path = Path(out).resolve() / 'window/messages.json'
    if not path.is_file():
        raise SystemExit(f'No window input at {path}; run `build` first')
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--frames', type=int, default=5)
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--first', type=int, default=0)
    parser.add_argument('--rebuild', action='store_true')
    parser.add_argument('--max-calls', type=int, default=4,
                        help='max_calls the content advertises to the model')
    args = parser.parse_args()
    if not 1 <= args.window <= args.frames:
        parser.error('Require 1 <= window <= frames')
    messages, window_root = build(args.source, args.frames, args.window, args.first,
                                  args.out, args.rebuild, args.max_calls)
    meta = json.loads((window_root / 'window.json').read_text())
    print(json.dumps({'window_root': str(window_root), **meta}, indent=2))


if __name__ == '__main__':
    main()
