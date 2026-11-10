"""Pure, testable transaction validation. Original observations are never mutated."""
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np


_LEGACY_OBSERVATION_KEYS = frozenset({'mask_id'})
_LEGACY_OBSERVATION_TOKEN = 'mask_id'
_CURRENT_OBSERVATION_TOKEN = 'instance_id'

# A discard is a per-observation judgement, not a way to clear a window. Ten is
# generous enough for a genuinely unusable cluster and small enough that a
# whole-window disposal still has to be written out observation by observation.
# `defer` was the name until 2026-11; the input no longer accepts it, because a
# deferral was never reclaimed (0 of 532 in the v11 run) and the name promised a
# later decision the host never offered.
DISCARD_BATCH_LIMIT = 10
# Each object keeps at most this many held observations. Staying small keeps
# the point cloud honest: a single wrong hold pollutes at most one slot and
# can be evicted when a better view arrives.
HOLD_LIMIT = 5


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


# Auto-evict weights for mask area, point count, view angle and new surface.
# Set 1:1:1:1 on the evict-weights test page (docs/debug/evict-weights), where
# the four v14 cases kept every part of the object under these values.
EVICT_WEIGHTS = (0.25, 0.25, 0.25, 0.25)
# A point farther than this from every selected cloud counts as new surface.
NEW_SURFACE_RADIUS = 0.10

_VIEW_PARAMS = {}


def _camera(data, seq):
    """Camera centre for ``seq`` from the source frames, or None if unavailable."""
    source = getattr(data, 'source', None)
    if source is None or seq is None:
        return None
    frames = Path(source) / 'frames'
    if frames not in _VIEW_PARAMS:
        from .view_params import ViewParams
        _VIEW_PARAMS[frames] = ViewParams(frames)
    return _VIEW_PARAMS[frames].position(seq)


def _view_angle(a, b):
    """Degrees between the two cameras as seen from the midpoint of the two clouds.

    Measured at the object, not at the world origin: two cameras on either side
    of an object can point the same way from the origin. Each observation uses
    its own centroid because the Object's fused centroid is unreliable while it
    holds one or two observations. None when a camera or centroid is missing.
    """
    if a['camera'] is None or b['camera'] is None or a['centroid'] is None or b['centroid'] is None:
        return None
    m = (a['centroid'] + b['centroid']) / 2
    u, v = a['camera'] - m, b['camera'] - m
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return None
    return float(np.degrees(np.arccos(np.clip(u @ v / nu / nv, -1.0, 1.0))))


def _select_holds(obs_ids, data, limit=HOLD_LIMIT):
    """Greedily pick up to ``limit`` observations to keep, best first.

    score(i | S) = w_a·A + w_p·P + w_v·V + w_c·N, where S is the set chosen so far:
      A  mask pixels / frame pixels, divided by the largest among the candidates
      P  point count, divided by the largest among the candidates
      V  (1 − cos θ) / 2 for θ the smallest view angle to any member of S
      N  share of this observation's points farther than NEW_SURFACE_RADIUS
         from every cloud in S
    S starts empty, so held and new observations compete on equal terms. In the
    first round N is 1 and V is 0 for every candidate, so A and P pick the first.
    Ties go to the larger mask, then to the earlier candidate in ``obs_ids``.
    Missing data scores 0 on its term (no camera: V; no mask size: A).
    """
    from scipy.spatial import cKDTree
    items = getattr(data, 'items', None) or {}
    clouds = getattr(data, 'points', None) or {}
    feats = []
    for oid in obs_ids:
        item = items.get(oid, {})
        w, h = item.get('image_size_wh') or (0, 0)
        px = item.get('mask_pixels')
        cloud = np.asarray(clouds.get(oid, np.empty((0, 3))), dtype=np.float64).reshape(-1, 3)
        centroid = item.get('centroid')
        if centroid is None and len(cloud):
            centroid = cloud.mean(0)
        feats.append({'area': px / (w * h) if px and w and h else 0.0,
                      'points': item.get('point_count', len(cloud)),
                      'cloud': cloud,
                      'centroid': None if centroid is None else np.asarray(centroid, dtype=np.float64),
                      'camera': _camera(data, item.get('seq'))})
    max_area = max((f['area'] for f in feats), default=0) or 1
    max_points = max((f['points'] for f in feats), default=0) or 1
    wa, wp, wv, wc = EVICT_WEIGHTS
    # covered[i]: which of i's points lie within the radius of a selected cloud.
    covered = [np.zeros(len(f['cloud']), dtype=bool) for f in feats]
    selected, remaining = [], list(range(len(feats)))
    while remaining and len(selected) < limit:
        best = None
        for i in remaining:
            f = feats[i]
            a, p = f['area'] / max_area, f['points'] / max_points
            v, n = 0.0, 1.0
            if selected:
                angles = [t for t in (_view_angle(f, feats[j]) for j in selected) if t is not None]
                v = (1 - np.cos(np.radians(min(angles)))) / 2 if angles else 0.0
                n = float((~covered[i]).mean()) if len(covered[i]) else 0.0
            key = (wa * a + wp * p + wv * v + wc * n, a)
            if best is None or key > best[0]:
                best = (key, i)
        pick = best[1]
        selected.append(pick)
        remaining.remove(pick)
        if len(feats[pick]['cloud']):
            tree = cKDTree(feats[pick]['cloud'])
            for i in remaining:
                if len(feats[i]['cloud']):
                    d = tree.query(feats[i]['cloud'], distance_upper_bound=NEW_SURFACE_RADIUS)[0]
                    covered[i] |= d <= NEW_SURFACE_RADIUS
    return [obs_ids[i] for i in selected]


def _auto_evict(obj_id, held_ids, new_ids, data, cutoff):
    """Select which observations to keep and which to auto-evict.

    Called only when held + new would exceed HOLD_LIMIT.  Returns
    (keep_new, auto_discard_new, auto_discard_held):
      - keep_new         new observation ids to keep
      - auto_discard_new new ids that are dropped (capacity only)
      - auto_discard_held held ids that are dropped to make room
    """
    keep = set(_select_holds(list(held_ids) + list(new_ids), data))
    keep_new     = [i for i in new_ids   if i in keep]
    drop_new     = [i for i in new_ids   if i not in keep]
    drop_held    = [i for i in held_ids  if i not in keep]
    return keep_new, drop_new, drop_held


def _describe(action, kind):
    """Short label for an action in an error message: its kind and target."""
    return f'{kind} {action["object_id"]}' if kind == 'hold' and action.get('object_id') else kind


def propose(state, proposal, observations, cutoff, evidence, data=None):
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
    # Where each observation was first placed, so a second placement can name
    # both actions instead of only saying that one exists somewhere.
    touched = {}
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
                    raise ValueError('create takes no observations; add a separate hold action for them')
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
        elif kind in ('hold', 'discard', 'exclude'):
            ids = a.get('observations')
            # Three faults used to share one sentence with no id in it.
            if not isinstance(ids, list) or not ids:
                raise ValueError(f'Action {index} ({kind}) needs a nonempty observations list')
            repeated = sorted({x for x in ids if ids.count(x) > 1})
            if repeated:
                raise ValueError(f'Action {index} ({kind}) lists {", ".join(repeated)} more than once')
            unknown = [x for x in ids if x not in observations]
            if unknown:
                raise ValueError(f'Action {index} ({kind}) lists ids that are not observations: {", ".join(unknown)}')
            future = [x for x in ids if x not in visible]
            if future:
                raise ValueError(f'Action {index} ({kind}) lists observations after this window: {", ".join(future)}')
            # One discard is one decision. Without a ceiling, "discard everything
            # in this window under one reason" is a legal, zero-cost action: the
            # window's own ids are already in evidence, so it needs no inspection
            # and always passes. That is what turned 34 discarded observations in
            # the P1 run into 535 in the P3 run, with single actions disposing of
            # 41 at once. Discarding may still happen; it has to be per observation.
            if kind == 'discard' and len(ids) > DISCARD_BATCH_LIMIT:
                raise ValueError(
                    f'Discard at most {DISCARD_BATCH_LIMIT} observations per action; got {len(ids)}. '
                    'Each discard needs its own reason, because a group discard under one '
                    'reason is not a decision. Submit one discard action per observation you '
                    'actually judged unusable, after inspecting it.')
            # The message names every clash and both places it sits. Said without
            # them, a model with forty actions cannot find the one id it pasted
            # twice; in the v12 trial it resubmitted for eleven rounds.
            clashes = [x for x in ids if x in touched]
            if clashes:
                raise ValueError('Each observation may appear in only one action; ' + '; '.join(
                    f'{x} is in action {touched[x]} and action {index} ({_describe(a, kind)})'
                    for x in clashes) + '. Keep it in the one you mean and remove it from the other.')
            for x in ids:
                touched[x] = f'{index} ({_describe(a, kind)})'
            if kind == 'hold' and oid not in new['objects']:
                raise ValueError(f'Action {index} (hold) targets {oid!r}, which is not an active object'
                                 + (f'; it was merged into {new["aliases"][oid]}' if oid in new['aliases'] else ''))
            if kind == 'hold':
                # evict accepts a single id (legacy) or a list.
                evict_raw = a.get('evict')
                if evict_raw is None:
                    evict_list = []
                elif isinstance(evict_raw, str):
                    evict_list = [evict_raw]
                elif isinstance(evict_raw, list):
                    evict_list = evict_raw
                else:
                    raise ValueError(f'Action {index} (hold {oid}) evict must be a string or list of strings')
                for evict in evict_list:
                    if not isinstance(evict, str):
                        raise ValueError(f'Action {index} (hold {oid}) evict list contains a non-string: {evict!r}')
                    evict_rec = new['assignments'].get(evict, {})
                    if evict_rec.get('object_id') != oid or evict_rec.get('status') not in ('held', 'assigned'):
                        holding = sorted(k for k, r in new['assignments'].items()
                                         if r.get('object_id') == oid and r.get('status') in ('held', 'assigned'))
                        owner = evict_rec.get('object_id')
                        raise ValueError(
                            f'Action {index} (hold {oid}) evicts {evict}, which {oid} does not hold'
                            + (f' ({evict} belongs to {owner})' if owner else '')
                            + f'. {oid} holds: {", ".join(holding) or "nothing"}.')
                    if evict in touched:
                        raise ValueError(
                            f'Each observation may appear in only one action; {evict} is in action '
                            f'{touched[evict]} and is evicted by action {index} ({_describe(a, kind)}). '
                            'An evicted observation is discarded already; remove it from the other action.')
                    touched[evict] = f'{index} (evict by hold {oid})'
                    new['assignments'][evict] = {**evict_rec,
                        'object_id': None, 'status': 'discard',
                        'reason': f'evicted to make room: {a["reason"]}',
                        'decided_at': cutoff}
                    affected.add(oid)
            for obs in ids:
                old_owner = new['assignments'].get(obs, {}).get('object_id')
                if old_owner: affected.add(old_owner)
                new['assignments'][obs] = {'object_id': oid if kind == 'hold' else None,
                    'status': 'held' if kind == 'hold' else kind, 'reason': a['reason'], 'evidence': refs,
                    'decided_at': cutoff}
            if kind == 'hold': affected.add(oid)
        elif kind == 'merge':
            source = a.get('source_object')
            if source == oid:
                raise ValueError(f'Action {index} (merge) merges {oid} into itself')
            for role, end in (('source_object', source), ('object_id', oid)):
                if end not in new['objects']:
                    raise ValueError(f'Action {index} (merge) {role} {end!r} is not an active object'
                                     + (f'; it was merged into {new["aliases"][end]}' if end in new['aliases'] else ''))
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
        raise ValueError('Every visible observation needs a hold, discard or exclude; missing: '
                         + ', '.join(sorted(visible - set(new['assignments']))))
    # Enforce the per-object hold cap. Count both 'held' (new name) and
    # 'assigned' (pre-rename runs read back as held) so the check works
    # consistently regardless of which run wrote the state. Every object over
    # the cap is reported at once: reporting the first only made the v12 trial
    # fix them one per round, each fix revealing the next.
    held = {oid: sorted(k for k, a in new['assignments'].items()
                        if a.get('object_id') == oid and a.get('status') in ('held', 'assigned'))
            for oid in new['objects']}
    over = {oid: ids for oid, ids in held.items() if len(ids) > HOLD_LIMIT}
    if over:
        # Auto-evict: pick the best HOLD_LIMIT observations by mask area, point
        # count, view angle and new surface (_select_holds), discard the rest. The model can steer this by naming
        # specific observations in the evict list; this runs on whatever is left.
        for oid, ids in over.items():
            # Split into already-held-before-this-proposal and new ones.
            prior_held = sorted(k for k, r in state['assignments'].items()
                                if r.get('object_id') == oid and r.get('status') in ('held', 'assigned'))
            new_held = [k for k in ids if k not in set(prior_held)]
            keep_new, drop_new, drop_old = _auto_evict(oid, prior_held, new_held, data, cutoff)
            for drop in drop_new + drop_old:
                rec = new['assignments'][drop]
                new['assignments'][drop] = {**rec,
                    'object_id': None, 'status': 'discard',
                    'reason': 'auto-evicted: lower score (mask area + point count + view angle + new surface)',
                    'decided_at': cutoff}
            affected.add(oid)
    # Re-compute held after auto-eviction for the empty-object check.
    held = {oid: sorted(k for k, a in new['assignments'].items()
                        if a.get('object_id') == oid and a.get('status') in ('held', 'assigned'))
            for oid in new['objects']}
    empty = [oid for oid, ids in held.items() if not ids]
    if empty:
        raise ValueError(f'These objects would hold nothing: {", ".join(empty)}. '
                         'Hold an observation on each, or merge it into another object.')
    new.update(version=state['version']+1, cutoff=cutoff)
    proposal_id = hashlib.sha256(json.dumps({'before':state, 'proposal':proposal,'cutoff':cutoff},sort_keys=True).encode()).hexdigest()[:20]
    return {'id': proposal_id, 'state': new, 'affected': sorted(affected), 'proposal': copy.deepcopy(proposal)}
