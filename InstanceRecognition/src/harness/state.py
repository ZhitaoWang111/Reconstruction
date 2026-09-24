"""Pure, testable transaction validation. Original observations are never mutated."""
import copy
import hashlib
import json
import os
from pathlib import Path


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def initial():
    return {'version': 0, 'cutoff': -1, 'objects': {}, 'assignments': {}, 'aliases': {}}


def propose(state, proposal, observations, cutoff, evidence):
    """Build a detached next state; caller publishes only after geometry succeeds."""
    if proposal.get('base_version') != state['version']:
        raise ValueError('Stale base_version')
    if cutoff < state['cutoff']:
        raise ValueError('Cannot move cutoff backwards')
    actions = proposal.get('actions')
    if not isinstance(actions, list) or not actions:
        raise ValueError('actions must be a nonempty list')
    new = copy.deepcopy(state)
    affected = set()
    visible = {k for k, o in observations.items() if o['seq'] <= cutoff}
    touched=set()
    for a in actions:
        refs = a.get('evidence', [])
        if not a.get('reason') or not isinstance(refs, list) or not refs or any(r not in evidence for r in refs):
            raise ValueError('Every action needs a reason and accessible evidence IDs')
        kind = a.get('action')
        oid = a.get('object_id')
        if kind == 'create':
            if not isinstance(oid, str) or not oid.startswith('Object') or not oid[6:].isdigit() or int(oid[6:]) <= 0:
                raise ValueError('Use object IDs Object1, Object2, ...')
            if oid != f'Object{int(oid[6:])}' or oid in new['objects'] or oid in new['aliases']:
                raise ValueError('Object ID already used or not canonical')
            object_label = a.get('object_label')
            if not isinstance(object_label, str) or not object_label.strip(): raise ValueError('New object needs object_label')
            new['objects'][oid] = {'object_label': object_label, 'created_at': cutoff, 'status': 'provisional'}
            affected.add(oid)
        elif kind == 'relabel':
            if not isinstance(oid, str) or oid not in new['objects']:
                raise ValueError('relabel needs an active object_id')
            if set(a) - {'action', 'object_id', 'object_label', 'evidence', 'reason'}:
                raise ValueError('relabel only changes object_label; category_label and other fields are not allowed')
            object_label = a.get('object_label')
            if not isinstance(object_label, str) or not object_label.strip():
                raise ValueError('relabel needs a nonempty object_label')
            object_label = object_label.strip()
            obj = new['objects'][oid]
            if object_label == obj['object_label']:
                raise ValueError('relabel must change object_label')
            obj.setdefault('label_history', []).append({
                'previous_label': obj['object_label'], 'object_label': object_label,
                'evidence': copy.deepcopy(refs), 'reason': a['reason'], 'decided_at': cutoff})
            obj['object_label'] = object_label
            # affected tracks geometry changes; a category correction reuses the cloud.
        elif kind in ('assign', 'defer', 'repair'):
            ids = a.get('observations')
            if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids) or any(x not in visible for x in ids):
                raise ValueError('Invalid, repeated, or future observation IDs')
            if touched.intersection(ids): raise ValueError('Observation changed twice in one proposal')
            touched.update(ids)
            if kind == 'assign' and oid not in new['objects']: raise ValueError('Unknown target object')
            for obs in ids:
                old = new['assignments'].get(obs, {}).get('object_id')
                if old: affected.add(old)
                new['assignments'][obs] = {'object_id': oid if kind == 'assign' else None,
                    'status': 'assigned' if kind == 'assign' else kind, 'reason': a['reason'], 'evidence': refs,
                    'decided_at': cutoff}
            if kind == 'assign': affected.add(oid)
        elif kind == 'merge':
            source = a.get('source_object')
            if source == oid or source not in new['objects'] or oid not in new['objects']:
                raise ValueError('Invalid merge endpoints')
            for record in new['assignments'].values():
                if record['object_id'] == source:
                    record.update(object_id=oid, reason=a['reason'], evidence=refs, decided_at=cutoff)
            del new['objects'][source]
            new['aliases'][source] = oid
            for alias, target in list(new['aliases'].items()):
                if target == source: new['aliases'][alias] = oid
            affected.update((source, oid))
        else:
            raise ValueError(f'Unknown action {kind}')
    if set(new['assignments']) != visible:
        raise ValueError('Every visible observation needs assigned/defer/repair status; missing: '+str(sorted(visible-set(new['assignments']))))
    for oid in new['objects']:
        if not any(a['object_id'] == oid for a in new['assignments'].values()):
            raise ValueError(f'Empty object {oid}; assign observations or merge it')
    new.update(version=state['version']+1, cutoff=cutoff)
    proposal_id = hashlib.sha256(json.dumps({'before':state, 'proposal':proposal,'cutoff':cutoff},sort_keys=True).encode()).hexdigest()[:20]
    return {'id': proposal_id, 'state': new, 'affected': sorted(affected), 'proposal': copy.deepcopy(proposal)}
