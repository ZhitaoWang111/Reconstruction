"""The gate as the harness sees it: hold, inspect, release.

The pure geometry is covered in test_reid_gate.py.  This file drives the same
ruling through Tools.call, so it also pins the parts the model interacts with:
the status word, the fact that a hold keeps the proposal pending, and the two
ways out of it.
"""
import json
import pytest
from src.harness.tools import Tools

FAR = [6.77, 5.96, 0.52]        # the bin two rooms away
NEAR = [3.705, 1.225, 0.21]     # the archived bin, three centimetres later
CURRENT = '0100:001'
HISTORY = [f'00{index:02d}:001' for index in range(8)]


def tools_for(centroid=FAR):
    """One archived Object25 with eight observations, plus one open observation."""
    items, assignments = {}, {}
    for index in range(8):
        obs = HISTORY[index]
        items[obs] = {'seq': index, 'centroid': [3.70 + .01 * index, 1.23, 0.21],
                      'category_label': 'wastebasket'}
        assignments[obs] = {'object_id': 'Object25', 'status': 'assigned'}
    items[CURRENT] = {'seq': 100, 'centroid': centroid, 'category_label': 'trash can'}
    state = {'version': 1, 'cutoff': 100, 'objects': {
        'Object25': {'object_label': 'wastebasket', 'created_at': 0, 'status': 'provisional'}},
        'assignments': assignments, 'aliases': {}}
    tools = Tools(type('Data', (), {'items': items})(), state, 100)
    # build_window_content seeds only the current window's observation ids into
    # `evidence`, before any tool runs. Archived ids appear there only when
    # something is actually inspected, which is exactly what releases a hold.
    tools.evidence.add(CURRENT)
    return tools


def propose_assign(tools, object_id='Object25', observations=(CURRENT,)):
    body = {'base_version': 1, 'actions': [{
        'action': 'hold', 'object_id': object_id, 'observations': list(observations),
        'evidence': list(observations), 'reason': 'same bin seen before'}]}
    return tools.call('propose_update', {'proposal_json': json.dumps(body)})[0]


def test_distant_assign_is_held_and_stays_pending():
    tools = tools_for(FAR)
    payload = propose_assign(tools)
    assert payload['status'] == 'verification_required'
    assert payload['proposal_held'] is True
    assert [c['object_id'] for c in payload['checks']] == ['Object25']
    check = payload['checks'][0]
    assert check['worst_distance_m'] > 5.0
    assert check['threshold_m'] == 1.0
    # The proposal is still committable in principle -- just not yet.
    assert tools.pending is not None


def test_nearby_assign_passes_without_a_hold():
    payload = propose_assign(tools_for(NEAR))
    assert 'proposal_id' in payload


def test_commit_is_refused_until_the_object_is_inspected():
    tools = tools_for(FAR)
    propose_assign(tools)
    with pytest.raises(ValueError, match='Object25'):
        tools.call('commit_update', {'proposal_id': tools.pending['id']})


def test_inspecting_the_object_releases_the_hold():
    tools = tools_for(FAR)
    propose_assign(tools)
    # Stands in for inspect_object, which puts one of the object's archived
    # observations into `evidence`; that intersection is the release signal.
    tools.evidence.add(HISTORY[3])
    payload = propose_assign(tools)
    assert 'proposal_id' in payload
    ready, _ = tools.call('commit_update', {'proposal_id': payload['proposal_id']})
    assert ready == {'ready_to_commit': True, 'proposal_id': payload['proposal_id']}


def test_inspecting_a_current_observation_does_not_release():
    tools = tools_for(FAR)
    propose_assign(tools)
    # A current-window id is in `evidence` from the start; it must not count.
    tools.evidence.add('0100:002')
    payload = propose_assign(tools)
    assert payload['status'] == 'verification_required'


def test_budget_pressure_lets_the_hold_through():
    tools = tools_for(FAR)
    tools.gate_relaxed = True
    payload = propose_assign(tools)
    assert 'proposal_id' in payload
    ready, _ = tools.call('commit_update', {'proposal_id': payload['proposal_id']})
    assert ready['ready_to_commit'] is True


def test_creating_an_object_is_never_held():
    tools = tools_for(FAR)
    body = {'base_version': 1, 'actions': [
        {'action': 'create', 'object_id': 'Object26', 'object_label': 'bin',
         'evidence': [CURRENT], 'reason': 'new bin'},
        {'action': 'hold', 'object_id': 'Object26', 'observations': [CURRENT],
         'evidence': [CURRENT], 'reason': 'assign the new bin'}]}
    payload = tools.call('propose_update', {'proposal_json': json.dumps(body)})[0]
    assert 'proposal_id' in payload
