"""One OpenAI Responses call through the vibeg gateway, with usage and cost per call.

Adapted from ObjectGeneration's structuregen/agent/llm.py (copied, not imported: different
project). The key comes from the gateway key file (or VIBEG_API_KEY) and never reaches a log:
requests are saved with images replaced by their file path and the key redacted. Prices are
GPT-6 Astra's list price: $10 / $1 cached / $50 per 1M tokens, prompts above 272K at $20 / $2 / $75.
"""
from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import httpx

KEY_FILE = Path('/home/seer/wzt/Gateway/key.json')
MODEL = '七牛云/coding-openai/gpt-6-astra'
PRICE = {'in': 10.0, 'cached': 1.0, 'out': 50.0, 'threshold': 272_000, 'in_hi': 20.0, 'cached_hi': 2.0, 'out_hi': 75.0}
RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504, 524}


def credentials() -> tuple:
    data = json.loads(KEY_FILE.read_text())
    return os.environ.get('VIBEG_API_KEY') or data['token'], data['OpenAI_URL'].rstrip('/') + '/v1'


def image_part(path, detail='high') -> dict:
    data = Path(path).read_bytes()
    mime = 'image/png' if str(path).endswith('.png') else 'image/jpeg'
    return {'type': 'input_image', 'image_url': f'data:{mime};base64,{base64.b64encode(data).decode()}',
            'detail': detail, '_path': str(path)}


def text_part(t) -> dict:
    return {'type': 'input_text', 'text': t}


def cost(usage: dict) -> float:
    inp = usage.get('input_tokens', 0) or 0
    cached = (usage.get('input_tokens_details') or {}).get('cached_tokens', 0) or 0
    out = usage.get('output_tokens', 0) or 0
    hi = inp > PRICE['threshold']
    p_in, p_c, p_o = (PRICE['in_hi'], PRICE['cached_hi'], PRICE['out_hi']) if hi else (PRICE['in'], PRICE['cached'], PRICE['out'])
    return ((inp - cached) * p_in + cached * p_c + out * p_o) / 1e6


def _clean(value):
    """For logs: no image bytes."""
    if isinstance(value, dict):
        if value.get('type') == 'input_image':
            return {'type': 'input_image', 'image': value.get('_path'), 'detail': value.get('detail')}
        if value.get('type') == 'reasoning':
            return {'type': 'reasoning', 'summary': value.get('summary'), 'encrypted_content': '<omitted>'}
        return {k: _clean(v) for k, v in value.items() if k != '_path'}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _strip_private(value):
    if isinstance(value, dict):
        return {k: _strip_private(v) for k, v in value.items() if k != '_path'}
    if isinstance(value, list):
        return [_strip_private(v) for v in value]
    return value


class Client:
    def __init__(self, log_dir: Path, model: str = MODEL, effort: str = 'high', timeout: float = 600.0,
                 max_output_tokens: int = 16000):
        self.key, self.base = credentials()
        self.model, self.effort, self.max_out = model, effort, max_output_tokens
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.http = httpx.Client(timeout=timeout)
        self.n = 0

    def respond(self, instructions: str, items: list, tools: list) -> dict:
        """One model reply: {output, calls, text, reasoning, usage, cost, seconds, model}."""
        self.n += 1
        payload = {'model': self.model, 'instructions': instructions, 'input': _strip_private(items),
                   'tools': tools, 'tool_choice': 'auto', 'parallel_tool_calls': False,
                   'reasoning': {'effort': self.effort, 'summary': 'auto'}, 'include': ['reasoning.encrypted_content'],
                   'store': False, 'max_output_tokens': self.max_out}
        (self.log_dir / f'{self.n:03d}_request.json').write_text(
            json.dumps(_clean({**payload, 'input': items[-6:], 'input_items': len(items)}), ensure_ascii=False, indent=1))
        delay, error = 4.0, ''
        for attempt in range(6):
            started = time.time()
            try:
                r = self.http.post(self.base + '/responses', json=payload, headers={'Authorization': f'Bearer {self.key}'})
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                error = f'{type(exc).__name__}: {exc}'
            else:
                if r.status_code == 200:
                    body = r.json()
                    if body.get('status') in ('completed', 'incomplete') and not body.get('error'):
                        break
                    error = f"status={body.get('status')} error={body.get('error')}"
                elif r.status_code in RETRY_STATUS:
                    error = f'HTTP {r.status_code}: {r.text[:300]}'
                else:
                    raise RuntimeError(f'HTTP {r.status_code}: {r.text[:800]}'.replace(self.key, '<hidden>'))
            (self.log_dir / f'{self.n:03d}_error_{attempt}.txt').write_text(error.replace(self.key, '<hidden>'))
            time.sleep(delay)
            delay = min(delay * 2, 60)
        else:
            raise RuntimeError(f'gateway failed 6 times; last: {error}'.replace(self.key, '<hidden>'))
        output = body.get('output', [])
        calls = [{'call_id': o['call_id'], 'name': o['name'], 'arguments': o['arguments']}
                 for o in output if o.get('type') == 'function_call']
        text = ''.join(p.get('text', '') for o in output if o.get('type') == 'message'
                       for p in o.get('content', []) if p.get('type') == 'output_text')
        reasoning = [s.get('text', '') for o in output if o.get('type') == 'reasoning' for s in (o.get('summary') or [])]
        usage = body.get('usage') or {}
        result = {'output': output, 'calls': calls, 'text': text, 'reasoning': reasoning, 'usage': usage,
                  'cost': cost(usage), 'seconds': round(time.time() - started, 1), 'model': body.get('model'),
                  'status': body.get('status')}
        (self.log_dir / f'{self.n:03d}_response.json').write_text(json.dumps(
            {k: v for k, v in result.items() if k != 'output'} | {'output_types': [o.get('type') for o in output]},
            ensure_ascii=False, indent=1))
        return result
