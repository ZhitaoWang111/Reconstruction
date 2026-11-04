"""Bounded gateway requests with redacted logs and complete-response validation."""
import base64
import copy
import json
import errno
import http.client
import math
import ssl
from email.utils import parsedate_to_datetime
import socket
import time
import urllib.request
import urllib.error
from src.gateway import settings
from .state import save


class GatewayNetworkError(RuntimeError):
    """A request failed after bounded retries for a transient gateway error."""

    def __init__(self, message, *, attempts, last_error):
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


class IncompleteSSEError(ValueError):
    """The gateway closed an SSE response before a complete tool call arrived."""


def image(path):
    return {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(path.read_bytes()).decode(),'detail':'high'}}


def redacted(value, key):
    if isinstance(value,str):
        if value.startswith('data:image/'):return '[image bytes stored in evidence/]'
        return value.replace(key,'[REDACTED]') if key else value
    if isinstance(value,list):return [redacted(v,key) for v in value]
    if isinstance(value,dict):return {k:redacted(v,key) for k,v in value.items()}
    return value


def read_stream(response):
    """Assemble SSE tool-call fragments; truncated streams never execute."""
    calls={};body={'choices':[]};content='';finish=None;done=False
    for raw in response:
        line=raw.decode('utf-8').strip()
        if not line.startswith('data:'):continue
        data=line[5:].strip()
        if data=='[DONE]':done=True;break
        chunk=json.loads(data)
        for k in ('id','model','usage'):
            if chunk.get(k) is not None:body[k]=chunk[k]
        for choice in chunk.get('choices',[]):
            if choice.get('finish_reason'):finish=choice['finish_reason']
            delta=choice.get('delta',{});content+=delta.get('content') or ''
            for c in delta.get('tool_calls',[]):
                item=calls.setdefault(c['index'],{'id':'','type':'function','function':{'name':'','arguments':''}})
                if c.get('id'):item['id']+=c['id']
                for k in ('name','arguments'):item['function'][k]+=c.get('function',{}).get(k) or ''
    if not done or finish is None:raise IncompleteSSEError('Incomplete SSE stream')
    body['choices']=[{'finish_reason':finish,'message':{'role':'assistant','content':content or None,'tool_calls':[calls[i] for i in sorted(calls)]}}]
    return body


RESPONSES_OUTPUT_KEY = '_responses_output'

# An empty tool round is normally a one-off. Three attempts were enough while the
# gateway always returned a tool call; on 2026-10-20 a model narrating its plan
# instead of acting pushed that to ~20% of rounds, so retries got more room and
# now carry the narration back into the conversation (see reply_narration).
MAX_RESPONSE_FAILURES = 6
NUDGE = ('Your last reply contained no tool call. Every reply must contain exactly '
         'one tool call: the harness cannot continue on text alone. Call the least '
         'expensive tool that advances this window now.')


def responses_input(messages):
    """Translate harness history and replay original Responses items verbatim."""
    items = []
    for message in messages:
        role = message['role']
        if role == 'assistant' and RESPONSES_OUTPUT_KEY in message:
            items.extend(copy.deepcopy(message[RESPONSES_OUTPUT_KEY]))
            continue
        if role == 'tool':
            items.append({'type': 'function_call_output', 'call_id': message['tool_call_id'],
                          'output': message['content']})
            continue
        content = message.get('content')
        if content is not None:
            if isinstance(content, list):
                parts = []
                for part in content:
                    if part['type'] == 'text':
                        parts.append({'type': 'output_text' if role == 'assistant' else 'input_text',
                                      'text': part['text']})
                    elif part['type'] == 'image_url':
                        source = part['image_url']
                        parts.append({'type': 'input_image', 'image_url': source['url'],
                                      'detail': source.get('detail', 'auto')})
                    else:
                        raise ValueError('Unsupported chat content type: ' + part['type'])
                content = parts
            items.append({'role': role, 'content': content})
        for call in message.get('tool_calls', []):
            items.append({'type': 'function_call', 'call_id': call['id'],
                          'name': call['function']['name'], 'arguments': call['function']['arguments']})
    return items


def reply_narration(body):
    """Text the model produced when it should have called a tool, or ''.

    A reply that announces the next action and then stops is not a valid tool
    round, but its text is the model's own state. Replaying the identical request
    reproduces it; handing the text back as dialogue is what lets the model
    continue instead of narrating the same plan again.
    """
    if not isinstance(body, dict):
        return ''
    choices = body.get('choices') or []
    text = (choices[0].get('message') or {}).get('content') if choices else None
    if not text:
        for item in body.get('output') or []:
            if item.get('type') == 'message':
                text = ''.join(part.get('text', '') for part in item.get('content') or []
                               if part.get('type') == 'output_text')
                if text:
                    break
    return (text or '').strip()


def responses_message(body):
    """Only complete responses may expose executable tool calls to the harness."""
    if body.get('status') != 'completed' or body.get('error'):
        raise ValueError('Incomplete Responses response: ' + str(body.get('status')))
    output = body.get('output', [])
    calls, text = [], []
    for item in output:
        if item.get('status') not in (None, 'completed'):
            raise ValueError('Incomplete Responses output item')
        if item.get('type') == 'function_call':
            if not item.get('call_id') or not item.get('name') or not isinstance(item.get('arguments'), str):
                raise ValueError('Malformed Responses function call')
            calls.append({'id': item['call_id'], 'type': 'function',
                          'function': {'name': item['name'], 'arguments': item['arguments']}})
        elif item.get('type') == 'message':
            text.extend(part['text'] for part in item.get('content', []) if part.get('type') == 'output_text')
    if not calls:
        raise ValueError('Missing complete tool call response')
    return {'role': 'assistant', 'content': ''.join(text) or None, 'tool_calls': calls,
            RESPONSES_OUTPUT_KEY: copy.deepcopy(output)}


def responses_usage(raw):
    """Retain provider fields and add Chat-compatible aggregate token names."""
    if raw is None:
        return None
    return {**raw, 'prompt_tokens': raw.get('input_tokens', 0),
            'completion_tokens': raw.get('output_tokens', 0),
            'total_tokens': raw.get('total_tokens', (raw.get('input_tokens') or 0) + (raw.get('output_tokens') or 0))}


class Client:
    def __init__(self, model, out, timeout=120, reasoning_effort=None,
                 network_retries=5, retry_backoff=2.0, retry_backoff_max=30.0):
        cfg=settings();self.key=cfg.get('VIBEG_API_KEY','')
        if not self.key:raise RuntimeError('Missing VIBEG_API_KEY')
        self.base=cfg.get('VIBEG_OPENAI_BASE_URL','https://vibeg.seer-robotics.cn/openai/v1').rstrip('/')
        if not isinstance(network_retries, int) or network_retries < 0:
            raise ValueError('network_retries must be a non-negative integer')
        if not all(math.isfinite(v) and v >= 0 for v in (retry_backoff, retry_backoff_max)):
            raise ValueError('retry backoff values must be non-negative')
        self.model,self.out,self.timeout=model,out,timeout
        self._next_attempt = {}
        self.network_retries = network_retries
        self.retry_backoff = retry_backoff
        self.retry_backoff_max = retry_backoff_max
        self.use_responses = model.split('/')[-1].startswith('gpt-6-astra')
        self.reasoning_effort = reasoning_effort if reasoning_effort is not None else ('low' if self.use_responses else 'none')
        if self.use_responses and self.reasoning_effort == 'none':
            raise ValueError('GPT-6 Astra does not support reasoning effort none; use low or higher')
        if not self.use_responses and self.reasoning_effort != 'none':
            raise ValueError('Non-Astra harness tool calls currently require reasoning effort none')

    def _payload(self, messages, tools, final):
        """Request body for one round; identical shape on retries."""
        endpoint = '/chat/completions'
        payload={'model':self.model,'reasoning_effort':self.reasoning_effort,'max_completion_tokens':10000,
                 'messages':messages,'tools':tools,'parallel_tool_calls':False,'stream':False,
                 'tool_choice':{'type':'function','function':{'name':'propose_update'}} if final else 'required'}
        if self.use_responses:
            endpoint = '/responses'
            payload = {'model': self.model, 'reasoning': {'effort': self.reasoning_effort},
                       'max_output_tokens': 10000, 'input': responses_input(messages),
                       'tools': [{'type': 'function', **tool['function']} for tool in tools],
                       'parallel_tool_calls': False, 'stream': False, 'store': False,
                       'include': ['reasoning.encrypted_content'],
                       'tool_choice': {'type': 'function', 'name': 'propose_update'} if final else 'required'}
        return endpoint, payload

    def call(self,messages,tools,tag,final=False):
        # Keep the Sol path on Chat Completions and use a complete JSON response.
        # This avoids replaying Responses reasoning items on every tool round and
        # preserves complete tool-call responses for audit/replay.
        endpoint, payload = self._payload(messages, tools, final)
        # Retries extend this copy with whatever the model said, never `messages`.
        sent = list(messages)
        save(self.out/'calls'/f'{tag}_request.json',redacted(payload,self.key))
        network_failures = response_failures = 0
        attempt = self._next_attempt.get(tag, 0)
        while True:
            self._next_attempt[tag] = attempt + 1
            start=time.monotonic()
            request=urllib.request.Request(self.base+endpoint,json.dumps(payload).encode(),
                headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
            print(f'API REQUEST {tag} attempt {attempt + 1}; timeout={self.timeout:g}s', flush=True)
            try:
                # Only transport/response parsing lives inside this block. Disk
                # failures while saving an audit must not trigger another API call.
                with urllib.request.urlopen(request,timeout=self.timeout) as r:
                    body=read_stream(r) if 'text/event-stream' in r.headers.get('Content-Type','') else json.loads(r.read())
            except Exception as exc:
                if self._is_retryable(exc):
                    network_failures += 1
                    exhausted = network_failures > self.network_retries
                    delay = self._retry_delay(exc, network_failures) if not exhausted else None
                    error = self._log_error(tag, attempt, start, exc, 'network', delay)
                    if exhausted:
                        raise GatewayNetworkError(
                            f'Gateway network retries exhausted ({network_failures} failures): {error}',
                            attempts=attempt + 1, last_error=exc) from None
                    print(f'NETWORK RETRY {network_failures}/{self.network_retries} {tag}: '
                          f'{error}; waiting {delay:g}s', flush=True)
                elif isinstance(exc, (json.JSONDecodeError, UnicodeDecodeError)):
                    response_failures += 1
                    delay = min(self.retry_backoff_max, self.retry_backoff * 2 ** (response_failures - 1))
                    error = self._log_error(tag, attempt, start, exc, 'response',
                                            delay if response_failures < 3 else None)
                    if response_failures >= 3:
                        raise RuntimeError(error) from None
                    print(f'RESPONSE RETRY {response_failures}/2 {tag}: {error}; waiting {delay:g}s', flush=True)
                else:
                    error = self._log_error(tag, attempt, start, exc, 'fatal', None)
                    raise RuntimeError(error) from None
                time.sleep(delay)
                attempt += 1
                continue
            save(self.out/'calls'/f'{tag}_{attempt}_response.json',redacted(body,self.key))
            try:
                if not isinstance(body, dict):
                    raise ValueError('Expected a JSON object response')
                if self.use_responses:
                    message = responses_message(body)
                else:
                    choices = body.get('choices') or [{}]
                    choice=choices[0];message=choice.get('message') or {}
                    if choice.get('finish_reason') not in ('stop','tool_calls') or not message.get('tool_calls'):
                        raise ValueError('Missing complete tool call response')
            except (ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
                response_failures += 1
                delay = min(self.retry_backoff_max, self.retry_backoff * 2 ** (response_failures - 1))
                # A reply that only narrates the plan is the model's own text, not
                # a transport problem. Re-sending the identical request just gets
                # the same narration, so the earlier attempts are folded into the
                # next request as dialogue. `messages` itself is never mutated:
                # the harness replays its conversation on every round.
                narration = reply_narration(body)
                if narration:
                    sent = sent + [{'role': 'assistant', 'content': narration},
                                   {'role': 'user', 'content': NUDGE}]
                    _, payload = self._payload(sent, tools, final)
                error = self._log_error(tag, attempt, start, exc, 'response',
                                        delay if response_failures < MAX_RESPONSE_FAILURES else None)
                if response_failures >= MAX_RESPONSE_FAILURES:
                    raise RuntimeError(error) from None
                print(f'RESPONSE RETRY {response_failures}/{MAX_RESPONSE_FAILURES-1} {tag}: {error}; '
                      f'waiting {delay:g}s', flush=True)
                time.sleep(delay)
                attempt += 1
                continue
            save(self.out/'calls'/f'{tag}_{attempt}_meta.json',{'elapsed_seconds':time.monotonic()-start,'usage':responses_usage(body.get('usage')) if self.use_responses else body.get('usage'),
                'raw_usage':body.get('usage'),
                'requested_model':self.model,'returned_model':body.get('model'),'response_id':body.get('id')})
            return message

    def _log_error(self, tag, attempt, start, exc, kind, delay):
        error=f'{type(exc).__name__}: {exc}'
        if isinstance(exc, urllib.error.HTTPError):
            try:
                error += ': ' + exc.read(1800).decode(errors='replace')
            except (OSError, http.client.HTTPException):
                pass
            finally:
                exc.close()
        error=redacted(error,self.key)
        save(self.out/'calls'/f'{tag}_{attempt}_error.json', {
            'error': error, 'seconds': time.monotonic()-start, 'kind': kind,
            'retryable': kind in {'network', 'response'}, 'retry_delay_seconds': delay,
            'http_status': exc.code if isinstance(exc, urllib.error.HTTPError) else None})
        return error

    def _retry_delay(self, exc, failures):
        delay=min(self.retry_backoff_max, self.retry_backoff * 2 ** min(failures - 1, 20))
        if isinstance(exc, urllib.error.HTTPError) and exc.headers:
            value=exc.headers.get('Retry-After')
            if value:
                try:
                    requested=float(value)
                except ValueError:
                    try:
                        requested=parsedate_to_datetime(value).timestamp()-time.time()
                    except (ValueError, TypeError, OverflowError):
                        requested=0
                if math.isfinite(requested):
                    # Bound even server-specified waits so the runner remains responsive.
                    delay=max(delay, min(60.0, max(0.0, requested)))
        return delay

    @staticmethod
    def _is_retryable(exc):
        if isinstance(exc, urllib.error.HTTPError):
            # 524 is the gateway giving up on a slow upstream, not a rejection of
            # the request: the same call succeeded after a retry on 2026-10-20
            # after a 130 s wait. Treating it as fatal ended a whole window.
            return exc.code in {408, 425, 429, 500, 502, 503, 504, 524}
        if isinstance(exc, ssl.SSLCertVerificationError):
            return False
        if isinstance(exc, urllib.error.URLError):
            return Client._is_retryable(exc.reason) if isinstance(exc.reason, BaseException) else False
        if isinstance(exc, (TimeoutError, ConnectionError, http.client.IncompleteRead,
                            http.client.RemoteDisconnected, ssl.SSLEOFError, IncompleteSSEError)):
            return True
        if isinstance(exc, socket.gaierror):
            return exc.errno == socket.EAI_AGAIN
        return isinstance(exc, OSError) and exc.errno in {
            errno.ETIMEDOUT, errno.ECONNRESET, errno.ECONNREFUSED,
            errno.ECONNABORTED, errno.EHOSTUNREACH, errno.ENETUNREACH, errno.EPIPE}
