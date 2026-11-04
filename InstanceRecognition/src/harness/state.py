"""Pure, testable transaction validation. Original observations are never mutated."""
import copy
import hashlib
import json
import os
from pathlib import Path


_LEGACY_OBSERVATION_KEYS = frozenset({'mask_id'})
_LEGACY_OBSERVATION_TOKEN = 'mask_id'
_CURRENT_OBSERVATION_TOKEN = 'instance_id'

# A deferral is a per-observation judgement, not a way to clear a window. Four is
# generous enough for a genuinely ambiguous cluster and small enough that a
# whole-window disposal has to be written out one observation at a time.
DEFER_BATCH_LIMIT = 4


def _strip_legacy_observation_keys(value):
    """Remove retired observation fields before writing a new run artifact.

    The source manifest and historical runs may still contain the retired field,
    but every artifact produced by the current harness must use ``instance_id``
    as its only frame-local instance identifier.  The copy is recursive so the
    guard also covers nested request, response, transaction, and audit payloads.
    """
    if isinstance(value, dict):
        return {key: _strip_legacy_observation_keys(item)
                for key, item in value.items()
                if key not in _LEGACY_OBSERVATION_KEYS}
    if isinstance(value, list):
        return [_strip_legacy_observation_keys(item) for item in value]
    if isinstance(value, str):
        # Tool arguments and Responses function-call arguments are JSON encoded
        # strings nested inside the audit payload. Normalize those first so a
        # retired key cannot survive inside a serialized proposal or response.
        stripped = value.lstrip()
        if stripped.startswith(('{', '[')):
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError, json.JSONDecodeError):
                parsed = None
            if isinstance(parsed, (dict, list)):
                cleaned = _strip_legacy_observation_keys(parsed)
                if cleaned != parsed:
                    return json.dumps(cleaned, ensure_ascii=False, separators=(',', ':'))
        # A model can still mention the retired spelling in free-form text,
        # such as a reason or provider error. New artifacts must not retain the
        # spelling anywhere, so normalize it to the current field name too.
        return value.replace(_LEGACY_OBSERVATION_TOKEN, _CURRENT_OBSERVATION_TOKEN)
    return value


def dumps_sanitized(value, **kwargs):
    """Serialize a new artifact after removing the retired observation token."""
    return json.dumps(_strip_legacy_observation_keys(value), **kwargs)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = _strip_legacy_observation_keys(value)
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
    for index, a in enumerate(actions):
        refs = a.get('evidence', [])
        # One combined check used to answer "Every action needs a reason and
        # accessible evidence IDs" for three different faults. A model that cited
        # an archived id it had read in the registry got that sentence, could not
        # tell which of its twenty actions was wrong, and resubmitted the same
        # proposal until the window's budget ran out. Each fault now names itself.
        if not a.get('reason'):
            raise ValueError(f'Action {index} ({a.get("action")}) has no reason')
        if not isinstance(refs, list) or not refs:
            raise ValueError(f'Action {index} ({a.get("action")}) needs a nonempty evidence list')
        unreachable = [r for r in refs if r not in evidence]
        if unreachable:
            # The registry lists every observation an archived Object owns, and
            # reading that list is not the same as having looked: only an
            # inspection makes an id citable. Say which id and whose it is.
            owners = {}
            for r in unreachable:
                owner = (state['assignments'].get(r) or {}).get('object_id')
                owners.setdefault(owner, []).append(r)
            detail = '; '.join(f'{", ".join(ids)}' + (f' belongs to {o}; call inspect_object({o}) to make them citable'
                                                      if o else ' is not an observation id you can cite')
                               for o, ids in owners.items())
            raise ValueError(f'Action {index} ({a.get("action")}) cites observations you have not inspected: {detail}')
        kind = a.get('action')
        # `repair` promised a fix that never happened: the action only withholds
        # an observation from object fusion and runs no pixel repair, a caveat the
        # old prompt had to spell out. `exclude` names the consequence. Proposals
        # and committed states written before the rename still say `repair`, so it
        # is accepted on input and normalised here; nothing is written under the
        # old name.
        if kind == 'repair':
            kind = 'exclude'
        oid = a.get('object_id')
        if kind == 'create':
            # `observations` on a create was silently dropped: the branch below reads
            # object_id and object_label only, so a model that expected create to
            # file its masks got "these observations have no state" at the coverage
            # check instead of an error naming the field. It then resubmitted the
            # same shape. relabel already rejects unknown keys; create now does too.
            extra = set(a) - {'action', 'object_id', 'object_label', 'evidence', 'reason'}
            if extra:
                # `observations` is the one that actually happens: a create that carries
                # it looks complete to the model and files nothing, and the coverage
                # check then reports the missing ids as if they were never mentioned.
                if 'observations' in extra:
                    raise ValueError('create takes no observations; add a separate assign action for them')
                raise ValueError(f'create accepts only action, object_id, object_label, evidence and '
                                 f'reason; got {sorted(extra)}')
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
        elif kind in ('assign', 'defer', 'exclude'):
            ids = a.get('observations')
            if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids) or any(x not in visible for x in ids):
                raise ValueError('Invalid, repeated, or future observation IDs')
            # One deferral is one decision. Without a ceiling, "defer everything
            # in this window under one reason" is a legal, zero-cost action: the
            # window's own ids are already in evidence, so it needs no inspection
            # and always passes. That is what turned 34 deferred observations in
            # the P1 run into 535 in the P3 run, with single actions disposing of
            # 41 at once. Deferring may still happen; it has to be per observation.
            if kind == 'defer' and len(ids) > DEFER_BATCH_LIMIT:
                raise ValueError(
                    f'Defer at most {DEFER_BATCH_LIMIT} observations per action; got {len(ids)}. '
                    'Each deferral needs its own reason, because a group deferral under one '
                    'reason is not a decision. Submit one defer action per observation you '
                    'actually judged ambiguous, after inspecting it.')
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
        raise ValueError('Every visible observation needs assigned/defer/exclude status; missing: '+str(sorted(visible-set(new['assignments']))))
    for oid in new['objects']:
        if not any(a['object_id'] == oid for a in new['assignments'].values()):
            raise ValueError(f'Empty object {oid}; assign observations or merge it')
    new.update(version=state['version']+1, cutoff=cutoff)
    proposal_id = hashlib.sha256(json.dumps({'before':state, 'proposal':proposal,'cutoff':cutoff},sort_keys=True).encode()).hexdigest()[:20]
    return {'id': proposal_id, 'state': new, 'affected': sorted(affected), 'proposal': copy.deepcopy(proposal)}
