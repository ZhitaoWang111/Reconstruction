from types import SimpleNamespace
import numpy as np
from src.harness.spatial_context import build_spatial_context


def _data_with(points, items=None):
    """Build a SimpleNamespace data stub with points and optional item metadata."""
    return SimpleNamespace(points=points, items=items or {})


def _held_state(objects, held):
    """Build a state dict: objects map + assignments mapping obs->object (held)."""
    assignments = {}
    for obs_id, obj_id in held:
        assignments[obs_id] = {'object_id': obj_id, 'status': 'held'}
    return {'version': 1, 'cutoff': 99, 'objects': objects, 'assignments': assignments, 'aliases': {}}


def test_candidates_sorted_by_gap_with_overlap_fields():
    # Three objects at x=1, 2, 3; observation at x=1 (closest to Object1).
    objects = {f'Object{i}': {'object_label': 'cat'} for i in range(1, 4)}
    held = {f'Object{i}': [f'h{i}'] for i in range(1, 4)}
    flat = [(obs, oid) for oid, obs_list in held.items() for obs in obs_list]
    state = _held_state(objects, flat)
    pts = {f'h{i}': np.array([[float(i), 0, 0]]) for i in range(1, 4)}
    pts['a'] = np.array([[1.0, 0, 0], [1.0, 0, 0], [4.0, 0, 0]])
    pts['empty'] = np.empty((0, 3))
    data = _data_with(pts)
    _, candidates = build_spatial_context(data, state, [{'id': 'a'}, {'id': 'empty'}])
    assert [c['object_id'] for c in candidates['a']] == ['Object1', 'Object2', 'Object3']
    assert candidates['a'][0]['gap_m'] == 0.0
    assert 'overlap' in candidates['a'][0]
    assert candidates['empty'] == []


def test_first_window_no_history():
    _, candidates = build_spatial_context(
        SimpleNamespace(points={}, items={}),
        {'version': 0, 'cutoff': -1, 'objects': {}, 'assignments': {}, 'aliases': {}},
        [{'id': 'a'}])
    assert candidates == {'a': []}


def test_far_objects_are_dropped():
    # Object at x=1, object at x=5 (>2m away).
    objects = {'Object1': {'object_label': 'a'}, 'Object2': {'object_label': 'b'}}
    state = _held_state(objects, [('h1', 'Object1'), ('h2', 'Object2')])
    pts = {'h1': np.array([[1, 0, 0]]), 'h2': np.array([[5, 0, 0]]),
           'a': np.array([[1, 0, 0]])}
    data = _data_with(pts)
    _, candidates = build_spatial_context(data, state, [{'id': 'a'}])
    assert [c['object_id'] for c in candidates['a']] == ['Object1']


def test_dense_extension_past_eight():
    # 10 objects at x=0.04..0.40 (all within 0.5m), observation at x=0.
    objects = {f'Object{i}': {'object_label': 'row'} for i in range(1, 11)}
    flat = [(f'h{i}', f'Object{i}') for i in range(1, 11)]
    state = _held_state(objects, flat)
    pts = {f'h{i}': np.array([[i * 0.04, 0, 0]]) for i in range(1, 11)}
    pts['a'] = np.array([[0, 0, 0]])
    data = _data_with(pts)
    _, candidates = build_spatial_context(data, state, [{'id': 'a'}])
    # First 8 by gap, then extras with gap < 0.5 (all of them), capped at 15.
    assert len(candidates['a']) == 10
