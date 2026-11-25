"""M4 verification: measure, compare with the previous version, keep or undo.

    from verify import Verifier
    v = Verifier(state, structure, isaac)
    before = v.measure()
    report = state.set_pose(...)
    result = v.review('obj_0049', before, report)     # {'verdict': 'keep'|'undo', 'issues': [...]}

Penetration is a yes/no question here (M4 decision 2026-11-20): the scene is
asked which objects are in contact (Isaac's contact report), and a contact pair
that was not there before the change is a violation. A pair with the object's
own declared support, in either direction, is expected and allowed. Distances
for penetration are not measured yet.

The other hard checks are geometric and analytic: sinking into a wall, leaving
the room, a declared support that does not hold (no surface, or too small a
contact area), and a declared wall contact that is not against the wall.
Everything else (scan overlap, scale) is reported, never undoes a change.
"""
from .rules import Config
from .measure import Metrics, Measure
from .check import Verifier

__all__ = ['Config', 'Metrics', 'Measure', 'Verifier']
