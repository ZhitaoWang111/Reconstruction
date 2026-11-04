"""Contracts for the pilot: the input is the Harness's own, and failures show.

The pilot's value is that a timing comparison is only meaningful if every model
answered the same question, and that a model which answered badly is reported as
badly rather than omitted. Those two are what is pinned here; the gateway call
itself is stubbed, so these run offline.
"""
import json
from pathlib import Path

import pytest

from pilot_experiment import run_pilot, window_input


@pytest.fixture
def source(tmp_path):
    """A minimal lifted source: two frames, one instance each."""
    import numpy as np
    from PIL import Image

    root = tmp_path / 'source'
    (root / 'frames').mkdir(parents=True)
    (tmp_path / 'rgb').mkdir()
    records = []
    for seq, fid in enumerate((0, 11)):
        mask = Image.new('L', (64, 48), 0)
        mask.paste(255, (8, 8, 40, 32))
        mask_dir = root / 'masks'
        mask_dir.mkdir(exist_ok=True)
        mask.save(mask_dir / f'{seq:03d}.png')
        rgb = tmp_path / 'rgb' / f'frame_{fid:06d}.jpg'
        Image.new('RGB', (64, 48), 'white').save(rgb)
        np.savez_compressed(
            root / 'frames' / f'{seq:04d}.npz',
            points_world=np.random.default_rng(seq).normal(size=(200, 3)).astype(np.float32),
            rgb=np.full((200, 3), 120, np.uint8),
            mask_bits=np.full(200, 1, np.uint32),
            pixel_indices=np.arange(200, dtype=np.int64),
            grid_shape_hw=np.array([48, 64], dtype=np.int32),
            sequence_index=np.int64(seq))
        records.append({
            'sequence_index': seq, 'source_frame_id': fid,
            'npz': f'frames/{seq:04d}.npz',
            'original_rgb_image': str(rgb),
            'predictions_json': str(rgb),
            'instances': [{'instance_id': 1, 'class_name': 'table', 'color': [10, 20, 30],
                           'mask_file': f'masks/{seq:03d}.png', 'sam_prompt': 'table',
                           'sam_score': 0.9, 'gpt_certainty': 'clear',
                           'evidence': 'visible', 'ambiguity': '', 'visible_scope': 'whole',
                           'class_id': 0, 'prompt_index': 0, 'raw_candidate_id': 1,
                           'sam_box_xyxy': [8, 8, 40, 32], 'mask_pixels': 768,
                           'sam_prompt_type': 'text'}],
        })
    (root / 'manifest.json').write_text(json.dumps({
        'scene_id': 'scene0000_00', 'method': 'gpt', 'model': 'test',
        'policy': 'test', 'frame_count': 2, 'mask_count': 2,
        'lifting_policy': 'amb3r_per_frame_v1',
        'background_asset': 'assets/background.js',
        'bounds': {'minimum': [-1, -1, -1], 'maximum': [1, 1, 1]},
        'frames': records,
    }))
    return root


def test_the_window_input_is_the_harness_own_messages(source, tmp_path, monkeypatch):
    """It must be the system prompt plus build_window_content, not a rewrite."""
    monkeypatch.setattr(window_input, 'PROMPT', Path(window_input.__file__).parents[1]
                        / 'src/harness/prompt.txt')
    messages, root = window_input.build(source, frames=2, window=2, out=tmp_path / 'pilot')
    assert [m['role'] for m in messages] == ['system', 'user']
    assert messages[0]['content'] == window_input.PROMPT.read_text()
    content = messages[1]['content']
    # Each sampled frame contributes observation text plus its two images.
    assert sum(1 for part in content if part['type'] == 'image_url') == 4
    payload = json.loads(content[0]['text'])
    assert payload['window'] == [0, 1]
    assert payload['max_calls'] == 4


def test_a_rebuild_is_required_to_replace_an_existing_input(source, tmp_path, monkeypatch):
    monkeypatch.setattr(window_input, 'PROMPT', Path(window_input.__file__).parents[1]
                        / 'src/harness/prompt.txt')
    out = tmp_path / 'pilot'
    window_input.build(source, frames=2, window=2, out=out)
    with pytest.raises(SystemExit, match='exists'):
        window_input.build(source, frames=2, window=2, out=out)
    window_input.build(source, frames=2, window=2, out=out, rebuild=True)


def test_a_model_that_fails_is_recorded_not_dropped(tmp_path, monkeypatch):
    """A rejected model is a result; the report must not silently skip it."""
    def refuse(*args, **kwargs):
        raise ValueError('GPT-6 Astra does not support reasoning effort none')

    monkeypatch.setattr(run_pilot, 'Client', refuse)
    record = run_pilot.probe('openai/gpt-6-astra', [], tmp_path, 10, 'none')
    assert record['ok'] is False
    assert 'Astra' in record['error']
    report = run_pilot.summarize([record])
    assert 'openai/gpt-6-astra' in report and 'failed' in report.lower() or 'Astra' in report


def test_a_turn_without_a_tool_call_is_not_a_success(tmp_path, monkeypatch):
    """Latency alone is not an outcome: no tool call means nothing was done."""
    class Silent:
        def __init__(self, *args, **kwargs):
            pass

        def call(self, *args, **kwargs):
            return {'role': 'assistant', 'content': 'I have reviewed the frames.'}

    monkeypatch.setattr(run_pilot, 'Client', Silent)
    record = run_pilot.probe('some/model', [], tmp_path, 10)
    assert record['ok'] is False
    assert record['tool_calls'] == []


def test_a_turn_is_executed_against_the_real_tools(tmp_path, source, monkeypatch):
    """A tool call is run, so an unusable choice is an error rather than a number."""
    calls = [{'function': {'name': 'inspect_observations',
                           'arguments': json.dumps({'ids': ['9999:999']})}}]
    results = run_pilot._execute(calls, tmp_path)
    assert results[0]['ok'] is False
    assert 'Unknown or future observation' in results[0]['error']


def test_the_summary_ranks_by_time_and_marks_execution(tmp_path):
    records = [
        {'model': 'slow/model', 'ok': True, 'elapsed_seconds': 90.0,
         'tool_calls': ['list_objects'], 'usage': {'total_tokens': 100},
         'execution': [{'tool': 'list_objects', 'ok': True}]},
        {'model': 'fast/model', 'ok': True, 'elapsed_seconds': 12.5,
         'tool_calls': ['propose_update'], 'usage': {'total_tokens': 200},
         'execution': [{'tool': 'propose_update', 'ok': False, 'error': 'Stale base_version'}]},
    ]
    report = run_pilot.summarize(records)
    lines = [line for line in report.splitlines() if '/model' in line]
    assert lines[0].startswith('fast/model'), 'fastest first'
    assert 'propose_update=ERROR' in report
    assert '2/2 returned a usable tool call' in report
