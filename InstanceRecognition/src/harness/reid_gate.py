"""Re-identification gate for assigns onto an archived object.

An `assign` makes two different claims at once: that the listed masks are one
entity, and that the entity is the archived object it is filed under.  The first
claim is answered by the window's own images; the second needs the archive.  On
the 150-frame run an assign put eight observations of a bin 5.56 m away -- a
different room, 52 frames after the object was last seen -- under an object whose
observations never moved more than 4 cm from each other.  The reason text argued
the first claim only.

This module answers the second claim with geometry alone: how far is the new
surface from anywhere the object has already been observed.  It only reads the
window-start state, never the proposal's own result, so a proposal cannot make
itself look consistent.

Nothing here decides ownership.  A check is a request to look at the archived
object before committing, not a rejection: after inspecting, the original
assign is still allowed.
"""
import math

# A fixed floor keeps the rule from firing on objects with very tight spread,
# where 20x a 4 cm self-scale would land below the sensor's own noise.
DEFAULT_FLOOR_M = 1.0

# The relative term only matters for large objects whose observations legitimately
# sit far apart.  On the 150-frame run it never bound (every object that actually
# moved rooms had a self-scale under 10 cm), so the floor is the working
# parameter and this multiplier is the safety valve for a scene not seen yet.
SCALE_MULTIPLIER = 20.0

# Below this many archived observations the p95 estimate is just the maximum of a
# handful of numbers, which is too noisy to scale a threshold by.
MIN_HISTORY_FOR_SCALE = 6


def distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def self_scale(centroids):
    """p95 of each archived observation's distance to its nearest sibling.

    This is the object's own idea of "the same place".  A wastebasket is 4 cm; a
    rug is 25 cm because its observations land on different parts of its surface.
    Returns None when the history is too small to estimate from.
    """
    if len(centroids) < MIN_HISTORY_FOR_SCALE:
        return None
    nearest = [math.inf] * len(centroids)
    for left in range(len(centroids)):
        for right in range(left + 1, len(centroids)):
            span = distance(centroids[left], centroids[right])
            nearest[left] = min(nearest[left], span)
            nearest[right] = min(nearest[right], span)
    nearest.sort()
    # int() floors, so a short history yields its maximum rather than an
    # interpolated quantile. Conservative in the permissive direction: a larger
    # scale means a larger threshold and fewer holds.
    return nearest[min(len(nearest) - 1, int(0.95 * len(nearest)))]


def threshold_for(centroids, floor_m=DEFAULT_FLOOR_M, scale_multiplier=SCALE_MULTIPLIER):
    scale = self_scale(centroids)
    return (floor_m if scale is None else max(floor_m, scale_multiplier * scale)), scale


def check_assign(action_index, action, prior_assignments, centroids,
                 floor_m=DEFAULT_FLOOR_M, scale_multiplier=SCALE_MULTIPLIER):
    """Return one check when this assign lands beyond the target's own spread.

    History comes from the window-start assignments, never from the proposal, so
    an object's existing observations cannot be used to justify a distance the
    proposal itself would have created.
    """
    object_id = action.get('object_id')
    history = [obs for obs, record in prior_assignments.items()
               if record.get('object_id') == object_id and obs in centroids]
    if not history:
        # A newly created object has nothing to re-identify against.
        return None
    points = [centroids[obs] for obs in history]
    limit, scale = threshold_for(points, floor_m, scale_multiplier)

    measured = []
    for obs in action.get('observations') or []:
        if obs not in centroids:
            continue
        nearest_obs, nearest = min(((h, distance(centroids[obs], centroids[h]))
                                    for h in history), key=lambda pair: pair[1])
        measured.append((obs, nearest, nearest_obs))
    if not measured:
        return None

    offending = [row for row in measured if row[1] > limit]
    if not offending:
        return None
    worst = max(offending, key=lambda row: row[1])
    return {'action_index': action_index, 'object_id': object_id,
            'assigned_count': len(action.get('observations') or []),
            'offending_count': len(offending),
            # A count alone does not say which mask to detach: an assign of eight
            # observations with one outlier has to be split, and the model cannot
            # guess which one from the count and a single nearest neighbour.
            'offending_observations': [row[0] for row in offending],
            'worst_distance_m': round(worst[1], 3),
            'threshold_m': round(limit, 3),
            'object_self_scale_m': None if scale is None else round(scale, 3),
            'history_count': len(history),
            'nearest_archived_observation': worst[2]}


def verify_reidentification(actions, prior_assignments, centroids,
                            floor_m=DEFAULT_FLOOR_M, scale_multiplier=SCALE_MULTIPLIER):
    """Screen every hold in a proposal; report the ones that need a look.

    Only `hold` is screened.  `create` has no archive to re-identify against,
    and `discard`/`exclude`/`relabel`/`merge` do not put new surfaces into an object.
    """
    checks = []
    for index, action in enumerate(actions or []):
        if action.get('action') != 'hold':
            continue
        check = check_assign(index, action, prior_assignments, centroids,
                             floor_m, scale_multiplier)
        if check:
            checks.append(check)
    return {'status': 'verification_required' if checks else 'ok', 'checks': checks}
