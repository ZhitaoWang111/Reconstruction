"""Read-only spatial hints from the window's prior committed state.

Candidate selection rules (1105-v1 item 2)
-------------------------------------------
For each new observation, the host computes two metrics against every archived
object that has held observations:

  gap      5th-percentile of the nearest-neighbour distances from the
           observation's points to the object's point cloud.  The 5th
           percentile rather than the minimum suppresses single stray
           points at the mask boundary that would otherwise make distant
           objects look artificially close.

  overlap  Fraction of the observation's points within 10 cm of the
           object's point cloud.  10 cm absorbs the typical AMB3R
           cross-frame alignment error (median 6.6 cm, p90 13.1 cm)
           without swallowing the 0.4-0.74 m gaps between the stools.

Candidates are then selected:
  1. Discard objects whose gap exceeds 2 m (too far to be relevant).
  2. Sort the rest by gap ascending; ties broken by Object id.
  3. Keep the first 8.
  4. Extend with objects ranked 9th and beyond whose gap < 0.5 m,
     up to a total of 15 (handles dense clusters like a shoe rack).

The object's cloud is built only from its held observations (status
'held' or the pre-rename 'assigned').  Using the full history would let
a single wrong hold inflate the object's cloud and make it look closer
than it is, which is what caused the stool-76 error in v11.
"""
import numpy as np
from scipy.spatial import cKDTree

_GAP_TOL = 0.10    # overlap tolerance, metres
_GAP_CUTOFF = 2.0  # objects beyond this are not candidates
_DENSE_BAND = 0.5  # gap threshold for extending past 8
_CAP_BASE = 8      # return at most this many without the dense extension
_CAP_TOTAL = 15    # hard ceiling including the dense extension
_CENTROID_PRE = 2.5  # centroid pre-filter radius, metres


def _held_cloud(data, state, object_id):
    """Voxelised (2 cm) point cloud from held observations only."""
    ids = [k for k, a in state['assignments'].items()
           if a.get('object_id') == object_id
           and a.get('status') in ('held', 'assigned')]
    if not ids:
        return np.empty((0, 3), np.float32)
    parts = [data.points[k] for k in ids if k in data.points and len(data.points[k])]
    if not parts:
        return np.empty((0, 3), np.float32)
    pts = np.concatenate(parts)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if not len(pts):
        return np.empty((0, 3), np.float32)
    keys = np.floor(pts / 0.02).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[idx]


def build_spatial_context(data, state, observations):
    """Return ``({}, candidates)``; global OBB removed in 1105-v1."""
    candidates = {o['id']: [] for o in observations}
    if not state['objects']:
        return {}, candidates

    # Pre-compute per-object centroid from held observations' metadata.
    obj_centroid = {}
    for oid in state['objects']:
        pts = [data.items[k].get('centroid') for k, a in state['assignments'].items()
               if a.get('object_id') == oid and a.get('status') in ('held', 'assigned')
               and k in data.items]
        pts = [c for c in pts if c and len(c) == 3 and np.isfinite(c).all()]
        if pts:
            obj_centroid[oid] = np.mean(pts, axis=0)

    for observation in observations:
        oid = observation['id']
        pts = data.points.get(oid)
        if pts is None or not len(pts):
            continue
        pts = pts[np.isfinite(pts).all(axis=1)]
        if not len(pts):
            continue
        # Subsample to keep queries fast.
        step = max(1, len(pts) // 2000)
        qpts = pts[::step]
        obs_c = np.array(data.items.get(oid, {}).get('centroid') or [0, 0, 0], dtype=float)

        scored = []
        for obj_id, label in [(o, state['objects'][o]['object_label'])
                               for o in state['objects']]:
            # Pre-filter by centroid distance to avoid building trees for
            # every object in a large scene.
            if obj_id in obj_centroid:
                if np.linalg.norm(obj_centroid[obj_id] - obs_c) > _CENTROID_PRE:
                    continue
            cloud = _held_cloud(data, state, obj_id)
            if not len(cloud):
                continue
            tree = cKDTree(cloud)
            dist, _ = tree.query(qpts, k=1)
            gap = float(np.percentile(dist, 5))
            if gap > _GAP_CUTOFF:
                continue
            overlap = float((dist <= _GAP_TOL).mean())
            scored.append((gap, int(obj_id.removeprefix('Object')), obj_id, label, overlap))

        scored.sort()  # primary: gap, secondary: object number
        base = scored[:_CAP_BASE]
        extra = [r for r in scored[_CAP_BASE:] if r[0] < _DENSE_BAND]
        selected = (base + extra)[:_CAP_TOTAL]
        candidates[oid] = [
            {'object_id': r[2], 'object_label': r[3],
             'gap_m': round(r[0], 3), 'overlap': round(r[4], 3)}
            for r in selected]

    return {}, candidates
