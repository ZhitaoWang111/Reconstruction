"""Exercise the real label CLI, SAM adapter and frame publication with fake external calls."""
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src import config
from src.inference import gpt_sam31, label_frames
from src.inference.io import perception_dir, read_json, write_json


class Predictor:
    def __init__(self, fail_frame=None, invalid_shape=False):
        self.opens = 0
        self.closes = 0
        self.shutdowns = 0
        self.fail_frame = fail_frame
        self.invalid_shape = invalid_shape
        self.prompts = []
        self.frame_bytes = []

    def handle_request(self, request):
        if request['type'] == 'start_session':
            self.opens += 1
            self.frame_bytes.append((Path(request['resource_path']) / '000000.jpg').read_bytes())
            return {'session_id': str(self.opens)}
        if request['type'] == 'close_session':
            self.closes += 1
            return {}
        assert set(request) == {'type', 'session_id', 'frame_index', 'text', 'output_prob_thresh'}
        self.prompts.append(request['text'])
        if self.opens == self.fail_frame:
            raise RuntimeError('simulated SAM interruption')
        masks = np.zeros((1, 1, 6, 8), dtype=bool)
        masks[0, 0, 1:5, 1:7] = True
        return {'outputs': {'out_binary_masks': masks,
                            'out_probs': [0.9 if request['text'] == 'table' else 0.8],
                            'out_boxes_xywh': [] if self.invalid_shape else [[1/8, 1/6, 6/8, 4/6]]}}

    def shutdown(self):
        self.shutdowns += 1


@pytest.fixture
def batch(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'ARTIFACTS_ROOT', tmp_path / 'artifacts')
    frames = tmp_path / 'frames/scene_test'
    frames.mkdir(parents=True)
    for fid in (11, 22):
        Image.new('RGB', (8, 6), (fid, 80, 120)).save(frames / f'frame_{fid:06d}.jpg')
    batch_path = tmp_path / 'batch.json'
    write_json(batch_path, {'scene_id': 'scene_test', 'frame_indices': [11, 22]})
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'fake model')
    monkeypatch.setattr(label_frames, 'settings', lambda: {
        'VIBEG_API_KEY': 'test-secret', 'VIBEG_OPENAI_BASE_URL': 'https://gateway.invalid/v1'})
    requests = []

    def respond(request, timeout):
        payload = json.loads(request.data)
        requests.append(payload)
        assert [m['role'] for m in payload['messages']] == ['system', 'user']
        assert len(payload['messages'][1]['content']) == 1
        assert payload['messages'][1]['content'][0]['type'] == 'image_url'
        labels = {'categories': [
            {'category': name, 'sam_prompt': name, 'certainty': 'clear',
             'visible_scope': 'whole', 'evidence': 'visible shape', 'ambiguity': ''}
            for name in ('table', 'chair')], 'unresolved_regions': []}
        return io.BytesIO(json.dumps({'model': payload['model'], 'usage': {'total_tokens': 1},
            'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(labels)}}]}).encode())

    monkeypatch.setattr(label_frames.urllib.request, 'urlopen', respond)
    args = ['--run', 'test-run', '--batch', str(batch_path), '--frames-dir', str(frames.parent), '--workers', '1']
    return args, checkpoint, frames, requests


def test_cli_chain_writes_real_masks_and_preserves_overlaps(batch, monkeypatch):
    args, checkpoint, frames, requests = batch
    label_frames.main(args)
    predictor = Predictor()
    loaded = []
    monkeypatch.setattr(gpt_sam31, 'load_predictor', lambda *a: loaded.append(a) or predictor)
    output = gpt_sam31.main(['--run', 'test-run', '--checkpoint', str(checkpoint)])
    assert output == config.ARTIFACTS_ROOT / 'experiments/test-run/perception/masks'
    assert len(requests) == 2 and len(loaded) == 1
    assert predictor.opens == predictor.closes == 2 and predictor.shutdowns == 1
    assert predictor.prompts == ['table', 'chair', 'table', 'chair']
    assert predictor.frame_bytes[0] == (frames / 'frame_000011.jpg').read_bytes()
    folder = output / 'frame_000011/sam31'
    result = read_json(folder / 'predictions.json')
    masks = [np.asarray(Image.open(folder / item['mask_file'])) > 0 for item in result['instances']]
    assert len(masks) == 2 and (masks[0] & masks[1]).sum() == 24
    assert result['instances'][0]['sam_box_xyxy'] == [1, 1, 7, 5]
    assert result['instances'][0]['mask_pixels'] == 24
    assert np.asarray(Image.open(folder / 'instance_id_map.png'))[2, 2] == 2
    assert (folder / 'annotated.jpg').is_file()
    # Both stages reuse verified outputs without calling either external system.
    label_frames.main(args)
    gpt_sam31.main(['--run', 'test-run', '--checkpoint', str(checkpoint)])
    assert len(requests) == 2 and len(loaded) == 1
    # Geometry reads these same independent masks, not the display ID map.
    from src.geometry.backproject_masks import instance_masks
    decoded, _ = instance_masks({'instances': result['instances']}, folder)
    assert np.array_equal(decoded[0], masks[0])
    assert np.array_equal(decoded[1], masks[1])


def test_failed_frame_is_unpublished_and_resume_keeps_completed_frame(batch, monkeypatch):
    args, checkpoint, _, _ = batch
    label_frames.main(args)
    predictor = Predictor(fail_frame=2)
    monkeypatch.setattr(gpt_sam31, 'load_predictor', lambda *a: predictor)
    cli = ['--run', 'test-run', '--checkpoint', str(checkpoint)]
    with pytest.raises(RuntimeError, match='interruption'):
        gpt_sam31.main(cli)
    output = perception_dir('test-run') / 'masks'
    first = output / 'frame_000011/sam31/predictions.json'
    before = first.read_bytes()
    assert not (output / 'frame_000022/sam31').exists()
    assert predictor.closes == 2 and predictor.shutdowns == 1
    resumed = Predictor()
    monkeypatch.setattr(gpt_sam31, 'load_predictor', lambda *a: resumed)
    gpt_sam31.main(cli)
    assert first.read_bytes() == before
    assert resumed.opens == 1
    assert read_json(output / 'summary.json')['frame_count'] == 2


def test_shape_error_closes_session_and_does_not_publish(batch, monkeypatch):
    args, checkpoint, _, _ = batch
    label_frames.main(args)
    predictor = Predictor(invalid_shape=True)
    monkeypatch.setattr(gpt_sam31, 'load_predictor', lambda *a: predictor)
    with pytest.raises(ValueError, match='shape mismatch'):
        gpt_sam31.main(['--run', 'test-run', '--checkpoint', str(checkpoint)])
    assert predictor.closes == predictor.shutdowns == 1
    assert not (perception_dir('test-run') / 'masks/frame_000011/sam31').exists()


def test_budget_keeps_raw_candidates_and_damaged_cache_is_refused(batch, monkeypatch):
    args, checkpoint, _, _ = batch
    label_frames.main(args)
    monkeypatch.setattr(gpt_sam31, 'load_predictor', lambda *a: Predictor())
    cli = ['--run', 'test-run', '--checkpoint', str(checkpoint), '--max-masks', '1']
    output = gpt_sam31.main(cli)
    folder = output / 'frame_000011/sam31'
    result = read_json(folder / 'predictions.json')
    assert result['instance_count'] == 1 and result['raw_candidate_count'] == 2
    assert result['excluded_by_budget'] == [2]
    (folder / result['raw_candidates'][1]['mask_file']).unlink()
    with pytest.raises(ValueError, match='missing or changed'):
        gpt_sam31.main(cli)


def test_prepare_uses_no_gateway_and_changed_run_is_refused(batch):
    args, _, _, requests = batch
    output = label_frames.main(args + ['--prepare-only'])
    assert requests == [] and (output / 'config.json').exists()
    with pytest.raises(ValueError, match='Run inputs changed'):
        label_frames.main(args + ['--prepare-only', '--count', '1'])


def test_outputs_reject_external_symlink_and_path_names(tmp_path, monkeypatch):
    root = tmp_path / 'artifacts'
    root.mkdir()
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.setattr(config, 'ARTIFACTS_ROOT', root)
    (root / 'experiments').symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        perception_dir('test')
    with pytest.raises(ValueError, match='not a path'):
        perception_dir('../test')


def test_atomic_json_keeps_original_on_failed_publish(tmp_path, monkeypatch):
    from src.inference import io as module
    path = tmp_path / 'data.json'
    write_json(path, {'value': 'old'})

    def fail(*args):
        raise OSError('disk failure')

    monkeypatch.setattr(module.os, 'replace', fail)
    with pytest.raises(OSError, match='disk failure'):
        write_json(path, {'value': 'new'})
    assert read_json(path) == {'value': 'old'}
    assert list(tmp_path.iterdir()) == [path]
