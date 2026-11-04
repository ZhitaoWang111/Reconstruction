"""Offline gateway contract tests; no API keys or network required."""
import copy
import io
import json

import pytest

from src.harness import client as module

TOOLS = [{'type': 'function', 'function': {
    'name': 'propose_update', 'description': 'Propose',
    'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}}]
OUTPUT = [
    {'type': 'reasoning', 'id': 'rs_1', 'summary': [], 'encrypted_content': 'opaque-reasoning'},
    {'type': 'message', 'id': 'msg_1', 'role': 'assistant', 'status': 'completed',
     'content': [{'type': 'output_text', 'text': 'Ready', 'annotations': []}]},
    {'type': 'function_call', 'id': 'fc_1', 'call_id': 'call_1', 'status': 'completed',
     'name': 'propose_update', 'arguments': '{}'},
]


def body():
    return {'id': 'resp_1', 'model': 'gpt-6-astra', 'status': 'completed',
            'output': copy.deepcopy(OUTPUT), 'usage': {
                'input_tokens': 12, 'output_tokens': 8, 'total_tokens': 20,
                'output_tokens_details': {'reasoning_tokens': 5}}}


@pytest.fixture
def gateway(monkeypatch):
    requests, replies = [], []
    monkeypatch.setattr(module, 'settings', lambda: {
        'VIBEG_API_KEY': 'fake-key', 'VIBEG_OPENAI_BASE_URL': 'https://gateway.invalid/v1'})
    monkeypatch.setattr(module.time, 'sleep', lambda _: None)

    def urlopen(request, timeout):
        requests.append((request.full_url, json.loads(request.data)))
        reply = replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        response = io.BytesIO(json.dumps(reply).encode())
        response.headers = {'Content-Type': 'application/json'}
        return response

    monkeypatch.setattr(module.urllib.request, 'urlopen', urlopen)
    return requests, replies


def test_astra_request_conversion_replay_and_audit(gateway, tmp_path):
    requests, replies = gateway
    replies.extend([body(), body()])
    client = module.Client('openai/gpt-6-astra', tmp_path)
    messages = [{'role': 'system', 'content': 'Check observations'},
                {'role': 'user', 'content': [
                    {'type': 'text', 'text': 'Inspect'},
                    {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,YQ==', 'detail': 'high'}}]}]
    original = copy.deepcopy(messages)
    result = client.call(messages, TOOLS, 'first', final=True)
    endpoint, payload = requests[0]
    assert endpoint == 'https://gateway.invalid/v1/responses'
    assert payload['reasoning'] == {'effort': 'low'}
    assert payload['store'] is False and payload['stream'] is False
    assert payload['include'] == ['reasoning.encrypted_content']
    assert payload['tool_choice'] == {'type': 'function', 'name': 'propose_update'}
    assert payload['tools'] == [{'type': 'function', **TOOLS[0]['function']}]
    assert payload['input'][1]['content'] == [
        {'type': 'input_text', 'text': 'Inspect'},
        {'type': 'input_image', 'image_url': 'data:image/jpeg;base64,YQ==', 'detail': 'high'}]
    assert 'messages' not in payload and 'max_completion_tokens' not in payload
    assert result['tool_calls'] == [{'id': 'call_1', 'type': 'function',
                                    'function': {'name': 'propose_update', 'arguments': '{}'}}]
    assert result['content'] == 'Ready'
    assert result[module.RESPONSES_OUTPUT_KEY] == OUTPUT
    assert messages == original
    messages.extend([result, {'role': 'tool', 'tool_call_id': 'call_1', 'content': '{"ok":true}'}])
    client.call(messages, TOOLS, 'second')
    replay = requests[1][1]
    assert replay['input'][2:-1] == OUTPUT
    assert replay['input'][-1] == {'type': 'function_call_output', 'call_id': 'call_1', 'output': '{"ok":true}'}
    assert replay['tool_choice'] == 'required'
    meta = json.loads((tmp_path / 'calls/first_0_meta.json').read_text())
    assert meta['usage']['prompt_tokens'] == 12
    assert meta['usage']['completion_tokens'] == 8
    assert meta['raw_usage'] == body()['usage']
    assert json.loads((tmp_path / 'calls/first_0_response.json').read_text()) == body()
    assert 'data:image/' not in (tmp_path / 'calls/first_request.json').read_text()


@pytest.mark.parametrize('status', ['incomplete', 'failed', 'in_progress', None])
def test_incomplete_responses_never_return_calls(gateway, tmp_path, status):
    requests, replies = gateway
    bad = body()
    bad['status'] = status
    attempts = module.MAX_RESPONSE_FAILURES
    replies.extend([bad] * attempts)
    with pytest.raises(RuntimeError, match='Incomplete Responses response'):
        module.Client('openai/gpt-6-astra', tmp_path).call([], TOOLS, 'bad')
    assert len(requests) == attempts
    assert not list((tmp_path / 'calls').glob('*_meta.json'))
    assert len(list((tmp_path / 'calls').glob('*_response.json'))) == attempts


def test_incomplete_items_and_missing_calls_rejected():
    bad = body()
    bad['output'][-1]['status'] = 'in_progress'
    with pytest.raises(ValueError, match='Incomplete Responses output item'):
        module.responses_message(bad)
    bad['output'] = OUTPUT[:2]
    with pytest.raises(ValueError, match='Missing complete tool call'):
        module.responses_message(bad)


def test_effort_selection_and_legacy_chat(gateway, tmp_path):
    requests, replies = gateway
    with pytest.raises(ValueError, match='does not support'):
        module.Client('openai/gpt-6-astra', tmp_path, reasoning_effort='none')
    assert module.Client('gpt-6-astra', tmp_path, reasoning_effort='high').reasoning_effort == 'high'
    assert module.Client('openai/gpt-6-astra-2026-09-28', tmp_path).use_responses
    with pytest.raises(ValueError, match='Non-Astra'):
        module.Client('openai/gpt-6-sol', tmp_path, reasoning_effort='low')
    message = {'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': 'old_call', 'type': 'function', 'function': {'name': 'propose_update', 'arguments': '{}'}}]}
    replies.append({'choices': [{'finish_reason': 'tool_calls', 'message': message}],
                    'usage': {'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 5}})
    client = module.Client('openai/gpt-6-sol', tmp_path)
    assert client.call([{'role': 'user', 'content': 'Go'}], TOOLS, 'chat') == message
    endpoint, payload = requests[0]
    assert endpoint.endswith('/chat/completions')
    assert payload['reasoning_effort'] == 'none'
    assert payload['tools'] == TOOLS and payload['stream'] is False
    assert 'stream_options' not in payload
    assert 'input' not in payload and 'reasoning' not in payload


def http_error(code, headers=None):
    return module.urllib.error.HTTPError('https://gateway.invalid', code, 'gateway failure',
                                         headers or {}, io.BytesIO(b'fake-key error body'))


def test_network_failure_recovers_without_mutating_request(gateway, tmp_path, monkeypatch, capsys):
    requests, replies = gateway
    delays = []
    monkeypatch.setattr(module.time, 'sleep', delays.append)
    replies.extend([TimeoutError('read timed out')] * 4 + [body()])
    messages = [{'role': 'user', 'content': 'unchanged input'}]
    module.Client('openai/gpt-6-astra', tmp_path).call(messages, TOOLS, 'retry')
    assert len(requests) == 5 and all(r == requests[0] for r in requests)
    assert messages == [{'role': 'user', 'content': 'unchanged input'}]
    assert delays == [2, 4, 8, 16]
    assert 'NETWORK RETRY 4/5' in capsys.readouterr().out
    assert len(list((tmp_path/'calls').glob('*_error.json'))) == 4


@pytest.mark.parametrize('code', [408, 425, 429, 500, 502, 503, 504, 524])
def test_transient_http_error_recovers_and_redacts(gateway, tmp_path, code):
    requests, replies = gateway
    replies.extend([http_error(code), body()])
    module.Client('openai/gpt-6-astra', tmp_path).call([], TOOLS, 'http')
    assert len(requests) == 2
    text = (tmp_path/'calls/http_0_error.json').read_text()
    assert 'fake-key' not in text and '[REDACTED]' in text
    assert json.loads(text)['kind'] == 'network'


@pytest.mark.parametrize('code', [400, 401, 403, 404, 413, 422])
def test_permanent_http_error_fails_immediately(gateway, tmp_path, code):
    requests, replies = gateway
    replies.append(http_error(code))
    with pytest.raises(RuntimeError, match='HTTPError') as caught:
        module.Client('openai/gpt-6-astra', tmp_path).call([], TOOLS, 'fatal')
    assert not isinstance(caught.value, module.GatewayNetworkError)
    assert len(requests) == 1
    assert 'fake-key' not in str(caught.value)
    assert json.loads((tmp_path/'calls/fatal_0_error.json').read_text())['retryable'] is False


def test_network_exhaustion_can_resume_same_tag_without_overwriting_audit(gateway, tmp_path):
    requests, replies = gateway
    client = module.Client('openai/gpt-6-astra', tmp_path, network_retries=1)
    replies.extend([TimeoutError('read timed out')] * 2)
    with pytest.raises(module.GatewayNetworkError):
        client.call([], TOOLS, 'resume')
    before = {p.name: p.read_bytes() for p in (tmp_path/'calls').glob('*_error.json')}
    replies.append(body())
    client.call([], TOOLS, 'resume')
    assert (tmp_path/'calls/resume_2_meta.json').is_file()
    assert before == {p.name: p.read_bytes() for p in (tmp_path/'calls').glob('*_error.json')}
    assert len(requests) == 3


def test_retry_after_delta_and_date(gateway, tmp_path, monkeypatch):
    from email.utils import formatdate
    delays = []
    monkeypatch.setattr(module.time, 'sleep', delays.append)
    monkeypatch.setattr(module.time, 'time', lambda: 1000.0)
    gateway[1].extend([http_error(429, {'Retry-After': '10'}),
                       http_error(503, {'Retry-After': formatdate(1020, usegmt=True)}), body()])
    module.Client('openai/gpt-6-astra', tmp_path).call([], TOOLS, 'retry_after')
    assert delays == [10, 20]


def test_disk_error_is_not_retried(gateway, tmp_path, monkeypatch):
    requests, replies = gateway
    replies.append(body())
    original = module.save
    def failing_save(path, payload):
        if path.name.endswith('_response.json'):
            raise OSError(module.errno.ENOSPC, 'disk full')
        return original(path, payload)
    monkeypatch.setattr(module, 'save', failing_save)
    with pytest.raises(OSError, match='disk full'):
        module.Client('openai/gpt-6-astra', tmp_path).call([], TOOLS, 'disk')
    assert len(requests) == 1


@pytest.mark.parametrize('exc,retryable', [
    (module.urllib.error.URLError(TimeoutError('timeout')), True),
    (ConnectionResetError('reset'), True),
    (module.http.client.IncompleteRead(b'partial'), True),
    (module.IncompleteSSEError('Incomplete SSE stream'), True),
    (module.socket.gaierror(module.socket.EAI_AGAIN, 'try again'), True),
    (module.ssl.SSLCertVerificationError('invalid certificate'), False),
    (module.urllib.error.URLError(module.ssl.SSLCertVerificationError('bad cert')), False),
    (PermissionError('permission'), False),
    (OSError(module.errno.ENOSPC, 'full'), False),
])
def test_retryable_transport_classification(exc, retryable):
    assert module.Client._is_retryable(exc) == retryable


def narration(text):
    """A reply that announces the next action but calls no tool."""
    return {'id': 'resp_n', 'model': 'gpt-6-sol',
            'choices': [{'index': 0, 'finish_reason': 'tool_calls',
                         'message': {'role': 'assistant', 'content': text}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}


def complete():
    return {'id': 'resp_c', 'model': 'gpt-6-sol',
            'choices': [{'index': 0, 'finish_reason': 'tool_calls',
                         'message': {'role': 'assistant', 'content': None, 'tool_calls': [
                             {'index': 0, 'id': 'call_1', 'type': 'function',
                              'function': {'name': 'propose_update', 'arguments': '{}'}}]}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}


def test_narration_is_fed_back_and_caller_messages_untouched(gateway, tmp_path):
    requests, replies = gateway
    replies.extend([narration('I will inspect the archived views first.'), complete()])
    client = module.Client('openai/gpt-6-sol', tmp_path)
    conversation = [{'role': 'system', 'content': 'sys'}, {'role': 'user', 'content': 'go'}]
    original = copy.deepcopy(conversation)
    result = client.call(conversation, TOOLS, 'n')
    assert result['tool_calls'][0]['function']['name'] == 'propose_update'
    # The retry must carry the model's own words plus the nudge, so it continues
    # instead of narrating the same plan again...
    retry = requests[1][1]['messages']
    assert retry[-1]['content'] == module.NUDGE
    assert retry[-2] == {'role': 'assistant', 'content': 'I will inspect the archived views first.'}
    # ...while the harness's conversation stays exactly as it was built.
    assert conversation == original


def test_nudges_accumulate_across_retries(gateway, tmp_path):
    requests, replies = gateway
    replies.extend([narration('first plan'), narration('second plan'), complete()])
    module.Client('openai/gpt-6-sol', tmp_path).call(
        [{'role': 'user', 'content': 'go'}], TOOLS, 'n2')
    third = requests[2][1]['messages']
    assert [m['content'] for m in third if m['role'] == 'assistant'] == ['first plan', 'second plan']
