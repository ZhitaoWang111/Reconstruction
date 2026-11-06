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
    # No camera_to_world → back-projection unavailable, not a hard failure.
    assert 'unavailable' in checks[0]['note'].lower()
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
