"""The result page rewrites manifest.js, so it must recompute what it rewrites.

`build_site.publish` writes `manifest.js` and `build_result.build` writes it again
a moment later. The second write recomputed `frame_count` but carried the source's
`mask_count` through untouched, so a run covering 150 of a 500-frame source
advertised all 4632 source masks as its own. The page does not read the field,
which is exactly why it survived: nothing failed, the number was just wrong for
anyone who did read it.

No real point clouds are involved — the bounding-box export is stubbed, because
what is under test is the manifest arithmetic, not the geometry.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from pagegen import build_result, build_viewer
from src import config as app_config


def source_manifest(frames: int, instances: int):
    """A source manifest whose mask count belongs to the whole source."""
    return {
        'scene_id': 'scene0000_00', 'method': 'gpt', 'model': 'openai/gpt-6-sol',
        'policy': 'test', 'lifting_policy': 'amb3r_per_frame_v1',
        'background_asset': 'assets/background.js',
        'bounds': {'minimum': [-1, -1, -1], 'maximum': [1, 1, 1]},
        'frames': [{
            'sequence_index': seq, 'source_frame_id': seq * 11,
            'npz': f'frames/{seq:04d}.npz', 'asset': f'assets/frame_{seq:06d}.js',
            'rgb_image': f'frames/frame_{seq:06d}.jpg',
            'original_rgb_image': f'frames/frame_{seq:06d}.jpg',
            'predictions_json': f'predictions/{seq:04d}.json',
            'annotated_image': f'annotated/{seq:04d}.jpg',
            'instances': [{'instance_id': i + 1, 'class_name': 'table',
                           'mask_file': f'masks/{seq:04d}_{i:03d}.png'}
                          for i in range(instances)],
        } for seq in range(frames)],
    }


@pytest.fixture
def source(tmp_path):
    """A 6-frame source receiving 2 masks a frame, with its files present."""
    root = tmp_path / 'source'
    manifest = source_manifest(frames=6, instances=2)
    manifest['frame_count'] = 6
    manifest['mask_count'] = 12
    (root / 'assets').mkdir(parents=True, exist_ok=True)
    (root / 'assets/background.js').write_text('window.registerMaskCloud("background",{});')
    for frame in manifest['frames']:
        for key in ('npz', 'asset', 'rgb_image', 'original_rgb_image',
                    'predictions_json', 'annotated_image'):
            (root / frame[key]).parent.mkdir(parents=True, exist_ok=True)
            (root / frame[key]).write_text('x')
        for item in frame['instances']:
            (root / item['mask_file']).parent.mkdir(parents=True, exist_ok=True)
            (root / item['mask_file']).write_text('x')
    (root / 'manifest.json').write_text(json.dumps(manifest))
    return root


@pytest.fixture
def run_dir(tmp_path, source, monkeypatch):
    """A run that committed one window over the first 2 of the 6 frames."""
    run = tmp_path / 'run'
    (run / 'evidence').mkdir(parents=True)
    (run / 'versions' / '0001').mkdir(parents=True)
    # Every published observation needs an assignment, or build_result refuses.
    assignments = {f'{seq:04d}:{i + 1:03d}': {'object_id': 'Object1', 'status': 'assigned'}
                   for seq in range(2) for i in range(2)}
    state = {'version': 1, 'cutoff': 1,
             'objects': {'Object1': {'object_label': 'table', 'created_at': 0,
                                     'status': 'multi_frame'}},
             'assignments': assignments, 'aliases': {}}
    (run / 'versions' / '0001' / 'state.json').write_text(json.dumps(state))
    (run / 'versions' / '0001' / 'transaction.json').write_text(
        json.dumps({'proposal': {'actions': []}}))
    (run / 'checkpoint.json').write_text(json.dumps({'version': 1, 'cutoff': 1}))
    (run / 'config.json').write_text(json.dumps({'source': str(source)}))
    # build_result reads per-observation overlay colours back from the audit file.
    (run / 'observations.json').write_text(json.dumps({
        f'{seq:04d}:{i + 1:03d}': {'overlay_color_rgb': [10, 20, 30]}
        for seq in range(2) for i in range(2)}))
    for seq in range(2):
        (run / 'evidence' / f'frame_{seq:04d}.jpg').write_text('x')
    # The result page links to these; it refuses to publish without them.
    for name in ('summary.json', 'input_audit.json'):
        (run / name).write_text(json.dumps({}))
    monkeypatch.setattr(build_result, 'export_harness_bboxes', lambda *a, **k: [])
    return run


def test_mask_count_covers_the_run_not_the_whole_source(tmp_path, run_dir):
    """The reported mask count must describe this run's frames."""
    out = tmp_path / 'site'
    out.mkdir()
    build_viewer.build(out)          # the result page edits the shared bundle
    build_result.build(run_dir, out)

    manifest = json.loads((out / 'manifest.js').read_text().split('=', 1)[1].rstrip(';\n'))
    assert manifest['frame_count'] == 2, 'only the committed frames are published'
    assert manifest['mask_count'] == 4, (
        '2 frames x 2 masks; the source holds 12 across 6 frames and that number '
        'must not be published as this run\'s')
    assert manifest['mask_count'] == sum(
        len(frame['instances']) for frame in manifest['frames'])


def test_frame_count_and_mask_count_agree_with_each_other(tmp_path, run_dir):
    """Two counts over the same frame list cannot disagree."""
    out = tmp_path / 'site'
    out.mkdir()
    build_viewer.build(out)
    build_result.build(run_dir, out)
    manifest = json.loads((out / 'manifest.js').read_text().split('=', 1)[1].rstrip(';\n'))
    assert manifest['frame_count'] == len(manifest['frames'])
    assert manifest['mask_count'] <= manifest['frame_count'] * 2
