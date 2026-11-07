"""Offline two-window runs with real proposals, materialization and checkpoints."""
import copy
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.harness import run, client


@pytest.fixture
def runner(tmp_path, monkeypatch):
    source, out, site = tmp_path/'source', tmp_path/'output', tmp_path/'site'
    source.mkdir()
    (source/'manifest.json').write_text(json.dumps({
        'lifting_policy': 'amb3r_per_frame_v1', 'frames': [{}, {}]}))
    publications, requests = [], []
    faults = {'commit_timeouts': 0, 'fatal_second_window': False}

    class Data:
        def __init__(self, source, count, out):
            self.items = {f'{i:04d}:001': {'seq': i} for i in range(count)}
            self.audit = {}
        def view(self, *args):
            return None

    def content(data, tools, first, cutoff, budget):
        tools.evidence.update(data.items)
        return f'Window {first}'

    def gateway(request, timeout):
        payload = json.loads(request.data)
        requests.append(copy.deepcopy(payload))
        messages = payload['messages']
        first = int(messages[1]['content'].split()[-1])
        if first == 1 and faults['fatal_second_window']:
            raise client.urllib.error.HTTPError(request.full_url, 401, 'invalid auth', {}, io.BytesIO())
        if messages[-1]['role'] == 'tool':
            if faults['commit_timeouts']:
                faults['commit_timeouts'] -= 1
                raise TimeoutError('temporary read timeout')
            preview = json.loads(messages[-1]['content'])
            name, args = 'commit_update', {'proposal_id': preview['proposal_id']}
        else:
            name = 'propose_update'
            obs = f'{first:04d}:001'
            proposal = {'base_version': first, 'actions': [{
                'action': 'discard', 'observations': [obs], 'reason': 'Ambiguous identity', 'evidence': [obs]}]}
            args = {'proposal_json': json.dumps(proposal)}
        response = {'choices': [{'finish_reason': 'tool_calls', 'message': {
            'role': 'assistant', 'tool_calls': [{'id': f'c{first}_{name}', 'type': 'function',
                                               'function': {'name': name, 'arguments': json.dumps(args)}}]}}]}
        stream = io.BytesIO(json.dumps(response).encode())
        stream.headers = {'Content-Type': 'application/json'}
        return stream

    monkeypatch.setattr(run, 'Observations', Data)
    monkeypatch.setattr(run, 'build_window_content', content)
    monkeypatch.setattr(client, 'settings', lambda: {'VIBEG_API_KEY': 'fake-key'})
    monkeypatch.setattr(client.urllib.request, 'urlopen', gateway)
    monkeypatch.setattr(client.time, 'sleep', lambda _: None)

    def publish(data, folder, site_dir, voxel):
        # Publisher now receives the two roots separately: pipeline records and
        # the pages that are generated from them.
        checkpoint = json.loads((folder/'checkpoint.json').read_text())
        assert checkpoint == {'version': 2, 'cutoff': 1}
        assert (folder/'versions/0001/state.json').is_file()
        assert (folder/'versions/0002/state.json').is_file()
        assert Path(site_dir) == site.resolve()
        publications.append(checkpoint)
    monkeypatch.setattr(run, 'publish', publish)

    def invoke(*options, run_dir=None, site_dir=None):
        monkeypatch.setattr(sys, 'argv', ['run', '--source', str(source), '--output', 'test-run',
                                        '--run-dir', str(run_dir or out), '--site-dir', str(site_dir or site),
                                        '--frames', '2', '--window', '1', '--max-calls', '4',
                                        '--network-retries', '1', '--window-network-retries', '1', *options])
        run.main()
    return SimpleNamespace(out=out, site=site, source=source, faults=faults, requests=requests,
                           publications=publications, invoke=invoke)


def test_two_windows_publish_once_after_recovery_without_duplicate_tools(runner, capsys):
    runner.faults['commit_timeouts'] = 2
    runner.invoke()
    assert len(runner.publications) == 1
    assert runner.requests[1] == runner.requests[2] == runner.requests[3]
    events = [json.loads(p.read_text()) for p in (runner.out/'calls').glob('*_tool*.json')]
    assert len(events) == 4
    assert sum(e['name'] == 'commit_update' for e in events) == 2
    assert all('error' not in e['result'] for e in events)
    assert len(list((runner.out/'calls').glob('*_error.json'))) == 2
    from pagegen.build_records import LOG_NAME
    assert all(LOG_NAME.fullmatch(p.name) for p in (runner.out/'calls').glob('*.json'))
    text = capsys.readouterr().out
    assert 'WINDOW NETWORK RECOVERY' in text
    assert 'PROGRESS 2/2 windows; 2/2 frames committed' in text
    assert json.loads((runner.out/'summary.json').read_text())['completed']


def test_fatal_failure_keeps_checkpoint_and_existing_page(runner):
    runner.out.mkdir(); runner.site.mkdir()
    (runner.site/'index.html').write_text('old complete page')
    runner.faults['fatal_second_window'] = True
    with pytest.raises(RuntimeError, match='401'):
        runner.invoke()
    assert json.loads((runner.out/'checkpoint.json').read_text()) == {'version': 1, 'cutoff': 0}
    assert (runner.site/'index.html').read_text() == 'old complete page'
    assert runner.publications == []
    assert not (runner.out/'versions/0002').exists()


def test_skip_then_publish_only_uses_saved_config_without_api(runner, monkeypatch):
    runner.invoke('--skip-publish')
    assert runner.publications == []
    config = (runner.out/'config.json').read_bytes()
    count = len(runner.requests)
    monkeypatch.setattr(sys, 'argv', ['run', '--publish-only', '--output', str(runner.out)])
    run.main()
    assert len(runner.publications) == 1 and len(runner.requests) == count
    assert (runner.out/'config.json').read_bytes() == config


def test_publish_only_rejects_partial_run_without_api(runner, monkeypatch):
    runner.faults['fatal_second_window'] = True
    with pytest.raises(RuntimeError):
        runner.invoke()
    count = len(runner.requests)
    monkeypatch.setattr(sys, 'argv', ['run', '--publish-only', '--output', str(runner.out)])
    with pytest.raises(ValueError, match='incomplete'):
        run.main()
    assert runner.publications == [] and len(runner.requests) == count


def test_recovery_exhaustion_preserves_uncommitted_window(runner):
    runner.faults['commit_timeouts'] = 50
    with pytest.raises(RuntimeError, match='checkpoint preserved at cutoff -1'):
        runner.invoke()
    assert not (runner.out/'checkpoint.json').exists()
    assert runner.publications == []
    events = [json.loads(p.read_text()) for p in (runner.out/'calls').glob('*_tool*.json')]
    assert [e['name'] for e in events] == ['propose_update']
    assert len(list((runner.out/'calls').glob('*_error.json'))) == 4


def test_resume_legacy_config_with_new_retry_options(runner):
    runner.faults['fatal_second_window'] = True
    with pytest.raises(RuntimeError):
        runner.invoke()
    config_path = runner.out/'config.json'
    config = json.loads(config_path.read_text())
    for key in ('network_retries', 'window_network_retries', 'retry_backoff'):
        del config[key]
    config_path.write_text(json.dumps(config))
    prior = (runner.out/'versions/0001/state.json').read_bytes()
    count = len(runner.requests)
    runner.faults['fatal_second_window'] = False
    runner.invoke('--network-retries', '3')
    assert (runner.out/'versions/0001/state.json').read_bytes() == prior
    assert len(runner.requests) == count + 2
    assert len(runner.publications) == 1


def test_run_completes_under_the_accepted_policy(runner):
    """The one buildable policy drives the whole two-window path."""
    manifest = json.loads((runner.source/'manifest.json').read_text())
    assert manifest['lifting_policy'] == 'amb3r_per_frame_v1'

    runner.invoke()
    assert len(runner.publications) == 1
    assert runner.publications[0] == {'version': 2, 'cutoff': 1}
    assert json.loads((runner.out/'summary.json').read_text())['completed']


def test_a_retired_policy_is_rejected_not_silently_run(runner):
    """A source from a deleted lift must fail the gate, not run its geometry.

    The modules that built these were removed, so accepting one would run a source
    nobody can rebuild from the code in this tree.
    """
    manifest = json.loads((runner.source/'manifest.json').read_text())
    manifest['lifting_policy'] = 'mesh_first_surface_v1'
    (runner.source/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(SystemExit):
        runner.invoke()


def test_unknown_policy_is_still_rejected(runner):
    """Widening the accepted set must not turn the gate into a no-op."""
    manifest = json.loads((runner.source/'manifest.json').read_text())
    manifest['lifting_policy'] = 'something_else_v9'
    (runner.source/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(SystemExit):
        runner.invoke()


def test_verify_rounds_attach_images_once_and_force_without_duplicate_results(runner, monkeypatch, tmp_path):
    """A proposal verify keeps flagging is shown twice, then let through.

    Each round's images arrive in one user turn, not also as ordinary tool
    evidence, and the forced round replaces its tool result rather than adding
    a second one for the same call.
    """
    from PIL import Image
    from src.harness import verify
    shots = []

    def checks(pending, data, state):
        path = tmp_path / f'shot{len(shots)}.jpg'
        Image.new('RGB', (8, 8), (40 * len(shots), 0, 0)).save(path)
        shots.append(path)
        return ([{'action_index': 0, 'action': 'discard', 'object_id': 'Object1',
                  'observation_id': None, 'note': 'look', 'verify_fail': True,
                  'images': [str(path)]}], [path])
    monkeypatch.setattr(verify, 'build_checks', checks)

    # The gateway answers every tool result with another proposal until it is
    # given a proposal_id outside a verify, then commits.
    def gateway(request, timeout):
        payload = json.loads(request.data)
        runner.requests.append(copy.deepcopy(payload))
        messages = payload['messages']
        first = int(messages[1]['content'].split()[-1])
        last_tool = next((m for m in reversed(messages) if m['role'] == 'tool'), None)
        result = json.loads(last_tool['content']) if last_tool else {}
        if result.get('forced_after_verify'):
            name, args = 'commit_update', {'proposal_id': result['proposal_id']}
        elif 'committed' in result or 'verify_required' not in json.dumps(result) and result.get('proposal_id'):
            name, args = 'commit_update', {'proposal_id': result['proposal_id']}
        else:
            obs = f'{first:04d}:001'
            name, args = 'propose_update', {'proposal_json': json.dumps({'base_version': first, 'actions': [{
                'action': 'discard', 'observations': [obs], 'reason': 'r', 'evidence': [obs]}]})}
        n = len(runner.requests)
        response = {'choices': [{'finish_reason': 'tool_calls', 'message': {
            'role': 'assistant', 'tool_calls': [{'id': f'c{n}', 'type': 'function',
                                               'function': {'name': name, 'arguments': json.dumps(args)}}]}}]}
        stream = io.BytesIO(json.dumps(response).encode())
        stream.headers = {'Content-Type': 'application/json'}
        return stream
    monkeypatch.setattr(client.urllib.request, 'urlopen', gateway)

    monkeypatch.setattr(sys, 'argv', ['run', '--source', str(runner.source), '--output', 'test-run',
                                      '--run-dir', str(runner.out), '--site-dir', str(runner.site),
                                      '--frames', '2', '--window', '1', '--max-calls', '6',
                                      '--network-retries', '1', '--window-network-retries', '1'])
    run.main()

    assert json.loads((runner.out/'checkpoint.json').read_text()) == {'version': 2, 'cutoff': 1}
    assert len(list((runner.out/'verify').glob('*_force.json'))) == 2
    for payload in runner.requests:
        ids = [m['tool_call_id'] for m in payload['messages'] if m['role'] == 'tool']
        assert len(ids) == len(set(ids)), 'a tool call received two results'
        # Every verify image is sent exactly once.
        sent = [p['image_url']['url'] for m in payload['messages'] if m['role'] == 'user'
                and isinstance(m['content'], list) for p in m['content'] if p.get('type') == 'image_url']
        assert len(sent) == len(set(sent))
    # Two shown rounds per window: the last request of window 0 carries both.
    window0 = [p for p in runner.requests if p['messages'][1]['content'] == 'Window 0']
    images = [p for m in window0[-1]['messages'] if m['role'] == 'user' and isinstance(m['content'], list)
              for p in m['content'] if p.get('type') == 'image_url']
    assert len(images) == 2
