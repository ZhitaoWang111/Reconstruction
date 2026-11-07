"""Tests for verify.py: post-proposal evidence checks."""
import json
import numpy as np
import pytest
from types import SimpleNamespace
from pathlib import Path
from src.harness import verify


# ── test helpers ──────────────────────────────────────────────────────────────

def _state(assignments=None, objects=None):
    return {
        'version': 1, 'cutoff': 10,
        'objects': objects or {},
        'assignments': assignments or {},
        'aliases': {},
    }


def _pending(actions, state):
    """Minimal pending-proposal dict as produced by state.propose."""
    return {'proposal': {'actions': actions}, 'state': state}


def _data(points=None, items=None, source=None, out=None):
    return SimpleNamespace(
        points=points or {},
        items=items or {},
        source=source,
        out=out or Path('/tmp'),
    )


def _pts(centre, n=50):
    """Small cloud around ``centre``."""
    rng = np.random.default_rng(0)
    return (np.array(centre, dtype=np.float32)
            + rng.normal(scale=0.05, size=(n, 3)).astype(np.float32))


# ── hold checks ───────────────────────────────────────────────────────────────

def test_hold_check_note_without_camera_pose():
    """Without camera_to_world, back-projection is unavailable but no error."""
    state = _state(
        objects={'Object1': {'object_label': 'chair', 'status': 'provisional', 'created_at': 0}},
        assignments={'0001:001': {'object_id': 'Object1', 'status': 'held',
                                   'reason': 'x', 'evidence': [], 'decided_at': 1}},
    )
    pending = _pending(
        [{'action': 'hold', 'object_id': 'Object1', 'observations': ['0005:001'],
          'evidence': ['0005:001'], 'reason': 'test'}],
        state)
    data = _data(
        points={'0001:001': _pts([1, 0, 0]), '0005:001': _pts([1.1, 0, 0])},
        items={'0005:001': {'seq': 5, 'image': '/nonexistent/rgb.jpg',
                            'bbox_xyxy': [10, 10, 100, 100]}})
    checks, images = verify.build_checks(pending, data, state)
    assert len(checks) == 1
    assert checks[0]['action'] == 'hold'
    assert checks[0]['object_id'] == 'Object1'
    # No camera_to_world: no comparison is possible, and that is not a failure.
    assert 'no comparison' in checks[0]['note'].lower()
    assert checks[0]['hit_rate'] is None
    assert not checks[0]['verify_fail']


def test_create_no_overlap_not_a_failure():
    """A create with no overlap should not raise verify_fail."""
    state = _state(
        objects={'Object1': {'object_label': 'table', 'status': 'provisional', 'created_at': 0}},
        assignments={'0001:001': {'object_id': 'Object1', 'status': 'held',
                                   'reason': 'x', 'evidence': [], 'decided_at': 1}},
    )
    # New Object2 is far from Object1.
    proposed = dict(state)
    proposed = {**state,
                'objects': {**state['objects'],
                             'Object2': {'object_label': 'lamp', 'status': 'provisional', 'created_at': 5}},
                'assignments': {**state['assignments'],
                                 '0005:001': {'object_id': 'Object2', 'status': 'held',
                                               'reason': 'x', 'evidence': [], 'decided_at': 5}}}
    pending = _pending(
        [{'action': 'create', 'object_id': 'Object2', 'object_label': 'lamp',
          'evidence': ['0005:001'], 'reason': 'new object'},
         {'action': 'hold', 'object_id': 'Object2', 'observations': ['0005:001'],
          'evidence': ['0005:001'], 'reason': 'hold'}],
        proposed)
    data = _data(
        points={'0001:001': _pts([0, 0, 0]), '0005:001': _pts([5, 0, 0])},
        items={'0005:001': {'seq': 5, 'image': '/nonexistent/rgb.jpg', 'bbox_xyxy': [10, 10, 50, 50]}})
    checks, _ = verify.build_checks(pending, data, state)
    create_checks = [c for c in checks if c['action'] == 'create']
    assert len(create_checks) == 1
    assert not create_checks[0]['verify_fail']


def test_create_high_overlap_is_verify_fail():
    """A create where the new cloud overlaps an existing object should trigger verify_fail."""
    state = _state(
        objects={'Object1': {'object_label': 'chair', 'status': 'provisional', 'created_at': 0}},
        assignments={'0001:001': {'object_id': 'Object1', 'status': 'held',
                                   'reason': 'x', 'evidence': [], 'decided_at': 1}},
    )
    centre = [1.0, 2.0, 0.5]
    proposed = {**state,
                'objects': {**state['objects'],
                             'Object2': {'object_label': 'chair', 'status': 'provisional', 'created_at': 5}},
                'assignments': {**state['assignments'],
                                 '0005:001': {'object_id': 'Object2', 'status': 'held',
                                               'reason': 'x', 'evidence': [], 'decided_at': 5}}}
    pending = _pending(
        [{'action': 'create', 'object_id': 'Object2', 'object_label': 'chair',
          'evidence': ['0005:001'], 'reason': 'new'},
         {'action': 'hold', 'object_id': 'Object2', 'observations': ['0005:001'],
          'evidence': ['0005:001'], 'reason': 'hold'}],
        proposed)
    # Object1 and Object2 clouds at the same location → high overlap.
    data = _data(
        points={'0001:001': _pts(centre, 200), '0005:001': _pts(centre, 200)},
        items={'0005:001': {'seq': 5, 'image': '/nonexistent/rgb.jpg', 'bbox_xyxy': [10, 10, 50, 50]}})
    checks, _ = verify.build_checks(pending, data, state)
    create_checks = [c for c in checks if c['action'] == 'create']
    assert create_checks[0]['verify_fail']
    assert 'Object1' in create_checks[0]['note'] or 'overlap' in create_checks[0]['note'].lower()


def test_merge_over_limit_is_verify_fail():
    """A merge that leaves the target above HOLD_LIMIT triggers verify_fail."""
    from src.harness.state import HOLD_LIMIT
    # Object1 has 3 held, Object2 has 3 held; after merge Object1 gets 6.
    assignments = {}
    for i in range(3):
        assignments[f'000{i}:001'] = {'object_id': 'Object1', 'status': 'held',
                                       'reason': 'x', 'evidence': [], 'decided_at': i}
    for i in range(3):
        assignments[f'001{i}:001'] = {'object_id': 'Object2', 'status': 'held',
                                       'reason': 'x', 'evidence': [], 'decided_at': i}
    state = _state(
        objects={'Object1': {'object_label': 'chair', 'status': 'provisional', 'created_at': 0},
                 'Object2': {'object_label': 'chair', 'status': 'provisional', 'created_at': 1}},
        assignments=assignments)
    # Proposed state: Object2 merged into Object1 (all 6 held).
    proposed_assignments = {k: {'object_id': 'Object1', 'status': 'held',
                                  'reason': 'merged', 'evidence': [], 'decided_at': 5}
                             for k in assignments}
    proposed = {**state,
                'objects': {'Object1': state['objects']['Object1']},
                'assignments': proposed_assignments}
    pending = _pending(
        [{'action': 'merge', 'object_id': 'Object1', 'source_object': 'Object2',
          'evidence': ['0000:001'], 'reason': 'same object'}],
        proposed)
    data = _data()
    checks, _ = verify.build_checks(pending, data, state)
    merge_checks = [c for c in checks if c['action'] == 'merge']
    assert len(merge_checks) == 1
    assert merge_checks[0]['verify_fail']
    assert str(HOLD_LIMIT) in merge_checks[0]['note']


def test_merge_within_limit_is_not_a_failure():
    """A merge leaving the target at or below HOLD_LIMIT should not fail."""
    from src.harness.state import HOLD_LIMIT
    assignments = {}
    for i in range(2):
        assignments[f'000{i}:001'] = {'object_id': 'Object1', 'status': 'held',
                                       'reason': 'x', 'evidence': [], 'decided_at': i}
    for i in range(2):
        assignments[f'001{i}:001'] = {'object_id': 'Object2', 'status': 'held',
                                       'reason': 'x', 'evidence': [], 'decided_at': i}
    state = _state(
        objects={'Object1': {'object_label': 'chair', 'status': 'provisional', 'created_at': 0},
                 'Object2': {'object_label': 'chair', 'status': 'provisional', 'created_at': 1}},
        assignments=assignments)
    proposed_assignments = {k: {'object_id': 'Object1', 'status': 'held',
                                  'reason': 'merged', 'evidence': [], 'decided_at': 5}
                             for k in assignments}
    proposed = {**state,
                'objects': {'Object1': state['objects']['Object1']},
                'assignments': proposed_assignments}
    pending = _pending(
        [{'action': 'merge', 'object_id': 'Object1', 'source_object': 'Object2',
          'evidence': ['0000:001'], 'reason': 'same'}],
        proposed)
    data = _data()
    checks, _ = verify.build_checks(pending, data, state)
    merge_checks = [c for c in checks if c['action'] == 'merge']
    assert not merge_checks[0]['verify_fail']


def test_format_verify_message_contains_warnings():
    checks = [
        {'action_index': 0, 'action': 'hold', 'object_id': 'Object1',
         'note': 'Cyan dots off-centre.', 'verify_fail': True},
        {'action_index': 1, 'action': 'create', 'object_id': 'Object2',
         'note': 'No overlap.', 'verify_fail': False},
    ]
    msg = verify.format_verify_message(checks)
    assert '⚠' in msg
    assert '✓' in msg
    assert 'commit_update' in msg


# ── hold checks against a real projection ─────────────────────────────────────

W, H = 200, 100
K = [[100.0, 0, 100.0, 0], [0, 100.0, 50.0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def _camera_source(tmp_path, seqs):
    """A source whose frames all look down +z from the origin."""
    from PIL import Image
    src = tmp_path / 'src'
    (src / 'frames').mkdir(parents=True)
    for seq in seqs:
        np.savez(src / 'frames' / f'{seq:04d}.npz', camera_to_world=np.eye(4))
    (src / 'manifest.json').write_text(json.dumps({'intrinsic_color': K, 'frames': []}))
    rgb = tmp_path / 'rgb.jpg'
    Image.new('RGB', (W, H), 'grey').save(rgb)
    return src, rgb


def _mask(tmp_path, name, box):
    from PIL import Image
    m = Image.new('L', (W, H), 0)
    m.paste(255, box)
    path = tmp_path / f'{name}.png'
    m.save(path)
    return str(path)


def _hold_case(tmp_path, new_mask_box, monkeypatch):
    src, rgb = _camera_source(tmp_path, [0, 5])
    monkeypatch.setattr(verify, '_load_frame_assets',
                        lambda data, seq: (np.eye(4), np.asarray(K)[:3, :3], (W, H)))
    # Object1 already holds a patch 2 m ahead, which projects near the image centre.
    rng = np.random.default_rng(1)
    patch = np.c_[rng.uniform(-0.2, 0.2, 300), rng.uniform(-0.2, 0.2, 300), np.full(300, 2.0)]
    state = _state(
        objects={'Object1': {'object_label': 'box', 'status': 'provisional', 'created_at': 0}},
        assignments={'0000:001': {'object_id': 'Object1', 'status': 'held',
                                  'reason': 'x', 'evidence': [], 'decided_at': 0}})
    proposed = {**state, 'assignments': {**state['assignments'],
                '0005:001': {'object_id': 'Object1', 'status': 'held',
                             'reason': 'x', 'evidence': [], 'decided_at': 5}}}
    pending = _pending([{'action': 'hold', 'object_id': 'Object1',
                         'observations': ['0005:001'], 'evidence': ['0005:001'],
                         'reason': 't'}], proposed)
    data = _data(points={'0000:001': patch.astype(np.float32),
                         '0005:001': patch.astype(np.float32) + [0, 0, 0.01]},
                 items={'0005:001': {'seq': 5, 'image': str(rgb),
                                     'mask': _mask(tmp_path, 'm5', new_mask_box),
                                     'bbox_xyxy': list(new_mask_box)}},
                 source=src, out=tmp_path / 'out')
    return verify.build_checks(pending, data, state)


def test_a_hold_whose_object_lands_in_the_mask_passes_without_an_image(tmp_path, monkeypatch):
    checks, images = _hold_case(tmp_path, (80, 40, 120, 60), monkeypatch)
    assert checks[0]['hit_rate'] > 0.9
    assert not checks[0]['verify_fail']
    assert images == []


def test_a_hold_whose_object_lands_elsewhere_is_flagged_with_its_image(tmp_path, monkeypatch):
    checks, images = _hold_case(tmp_path, (0, 0, 30, 20), monkeypatch)
    assert checks[0]['hit_rate'] < verify.HOLD_HIT_THRESHOLD
    assert checks[0]['verify_fail']
    assert len(images) == 1 and images[0].is_file()


def test_the_new_observation_is_not_compared_against_itself(tmp_path, monkeypatch):
    """With only the new observation held, there is nothing to compare."""
    src, rgb = _camera_source(tmp_path, [5])
    monkeypatch.setattr(verify, '_load_frame_assets',
                        lambda data, seq: (np.eye(4), np.asarray(K)[:3, :3], (W, H)))
    proposed = _state(
        objects={'Object1': {'object_label': 'box', 'status': 'provisional', 'created_at': 5}},
        assignments={'0005:001': {'object_id': 'Object1', 'status': 'held',
                                  'reason': 'x', 'evidence': [], 'decided_at': 5}})
    pending = _pending([{'action': 'hold', 'object_id': 'Object1',
                         'observations': ['0005:001'], 'evidence': ['0005:001'],
                         'reason': 't'}], proposed)
    data = _data(points={'0005:001': np.array([[0, 0, 2.0]] * 50, np.float32)},
                 items={'0005:001': {'seq': 5, 'image': str(rgb),
                                     'mask': _mask(tmp_path, 'm', (0, 0, 10, 10))}},
                 source=src, out=tmp_path / 'out')
    checks, images = verify.build_checks(pending, data, _state())
    assert checks[0]['hit_rate'] is None and not checks[0]['verify_fail']
    assert images == []


def test_verify_attaches_at_most_the_image_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, 'MAX_VERIFY_IMAGES', 2)
    src, rgb = _camera_source(tmp_path, [0, 5])
    monkeypatch.setattr(verify, '_load_frame_assets',
                        lambda data, seq: (np.eye(4), np.asarray(K)[:3, :3], (W, H)))
    rng = np.random.default_rng(2)
    patch = np.c_[rng.uniform(-0.2, 0.2, 300), rng.uniform(-0.2, 0.2, 300),
                  np.full(300, 2.0)].astype(np.float32)
    held = {'0000:001': {'object_id': 'Object1', 'status': 'held', 'reason': 'x',
                         'evidence': [], 'decided_at': 0}}
    new = [f'0005:{i:03d}' for i in range(1, 6)]
    proposed = _state(
        objects={'Object1': {'object_label': 'box', 'status': 'provisional', 'created_at': 0}},
        assignments={**held, **{k: {**held['0000:001'], 'decided_at': 5} for k in new}})
    pending = _pending([{'action': 'hold', 'object_id': 'Object1', 'observations': new,
                         'evidence': new, 'reason': 't'}], proposed)
    corner = _mask(tmp_path, 'corner', (0, 0, 10, 10))
    data = _data(points={'0000:001': patch, **{k: patch + [5, 0, 0] for k in new}},
                 items={k: {'seq': 5, 'image': str(rgb), 'mask': corner} for k in new},
                 source=src, out=tmp_path / 'out')
    checks, images = verify.build_checks(pending, data, _state(
        objects=proposed['objects'], assignments=held))
    assert sum(c['verify_fail'] for c in checks) == 5
    assert len(images) == 2
    assert sum('No image' in c['note'] for c in checks) == 3
