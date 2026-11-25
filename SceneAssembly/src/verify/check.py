"""Compare a change with the version before it, then keep it or take it back.

The rule (D06, confirmed 2026-11-20): a change is judged by whether it makes
anything worse, not by whether the scene is perfect. Each hard check has a
violation amount; the change is undone when an amount grows by more than the
tolerance. Penetration is yes/no, so its violation amount is 0 or 1 and a pair
that appears where there was none is worse by 1.

Soft checks (scan overlap, scale) are reported and never undo anything.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .measure import Measure, Metrics
from .rules import Config


def _allowed(state, pair) -> bool:
    """A pair is expected when one of the two is the other's declared support."""
    a, b = tuple(pair)
    for o in (a, b):
        st = state.objects.get(o)
        if st and st.support.kind == 'object' and st.support.on in (a, b) and st.support.on != o:
            return True
    return False


class Verifier:
    def __init__(self, state, structure, isaac=None, config: Config | None = None, log_dir: Path | None = None):
        self.state = state
        self.structure = structure
        self.isaac = isaac
        self.cfg = config or Config()
        self.measure = Measure(state, structure, isaac, wall_exempt_labels=self.cfg.wall_exempt_labels)
        self.streaks: dict = {}            # oid -> consecutive automatic undos
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.cfg.write(self.log_dir / 'verify_config.json')

    # ------------------------------------------------------------ measuring

    def measure_now(self) -> Metrics:
        return self.measure.all()

    def baseline(self) -> Metrics:
        """The state as it is: what the next change is compared against."""
        return self.measure_now()

    # ------------------------------------------------------------ judging

    def _compare(self, oid: str, before: Metrics, after: Metrics) -> list:
        c = self.cfg
        issues = []

        def add(check, value, was, level, note=''):
            issues.append({'object': oid, 'check': check, 'value': value, 'before': was,
                           'threshold': getattr(c, f'{check}_stop_m', None), 'level': level, 'note': note})

        # penetration: new pairs involving this object that were not there before
        allowed_before = {p for p in before.contacts if _allowed(self.state, p)}
        allowed_after = {p for p in after.contacts if _allowed(self.state, p)}
        new = (after.contacts - before.contacts) - allowed_after
        gone = (before.contacts - after.contacts) - allowed_before
        for p in sorted(new, key=lambda s: tuple(sorted(s))):
            if oid in p:
                other = [x for x in p if x != oid]
                issues.append({'object': oid, 'check': 'penetration', 'with': other[0] if other else None,
                               'value': 1, 'before': 0, 'level': 'stop',
                               'note': 'in contact now, was not before'})
        for p in sorted(gone, key=lambda s: tuple(sorted(s))):
            if oid in p:
                other = [x for x in p if x != oid]
                issues.append({'object': oid, 'check': 'penetration', 'with': other[0] if other else None,
                               'value': 0, 'before': 1, 'level': 'ok', 'note': 'no longer in contact'})

        def worse(check, val, was, report, stop):
            grow = (val or 0) - (was or 0)
            if val is None:
                return
            if stop is not None and grow > c.tolerance_m and val > stop:
                add(check, val, was, 'stop')
            elif val > report and (was is None or grow > c.tolerance_m):
                add(check, val, was, 'report')

        worse('wall_pen', after.wall_pen.get(oid), before.wall_pen.get(oid), c.wall_report_m, c.wall_stop_m)
        worse('outside', after.outside.get(oid), before.outside.get(oid), c.room_report_m, c.room_stop_m)

        # support: floating above it or sunk into it
        g, g0 = after.support_gap.get(oid), before.support_gap.get(oid)
        if g is not None:
            if g < 0:                                        # sunk
                if g0 is None or abs(g) - abs(min(g0, 0)) > c.tolerance_m:
                    lvl = 'stop' if abs(g) > c.support_sink_stop_m else 'report'
                    issues.append({'object': oid, 'check': 'support_sink', 'value': round(abs(g), 4),
                                   'before': None if g0 is None else round(abs(min(g0, 0)), 4), 'level': lvl})
            else:                                            # floating
                if g0 is None or g - max(g0, 0) > c.tolerance_m:
                    lvl = 'stop' if g > c.support_float_stop_m else 'report'
                    issues.append({'object': oid, 'check': 'support_float', 'value': round(g, 4),
                                   'before': None if g0 is None else round(max(g0, 0), 4), 'level': lvl})
        r, r0 = after.contact_ratio.get(oid), before.contact_ratio.get(oid)
        if r is not None and r < c.support_contact_ratio_min and (r0 is None or r0 >= c.support_contact_ratio_min):
            issues.append({'object': oid, 'check': 'support_declaration', 'value': r, 'before': r0,
                           'level': 'stop', 'note': 'the declared support is barely under the object'})

        wg = after.wall_gap.get(oid)
        if wg is not None and wg > c.wall_gap_report_m:
            lvl = 'stop' if wg > c.wall_gap_stop_m else 'report'
            issues.append({'object': oid, 'check': 'wall_gap', 'value': wg, 'before': before.wall_gap.get(oid),
                           'level': lvl, 'note': 'declared against a wall but not against it'})

        # soft: reported only
        s = after.scale_dev.get(oid)
        lo, hi = c.scale_ok
        if s is not None and after.scale_dev.get(oid) != before.scale_dev.get(oid):
            st = self.state.objects[oid].pose.scale
            if any(x < lo or x > hi for x in st):
                issues.append({'object': oid, 'check': 'scale', 'value': list(st), 'before': None,
                               'level': 'report', 'note': 'scale far from the asset size'})
        cb, ca = before.cloud.get(oid), after.cloud.get(oid)
        if cb and ca and (ca['cover'] < cb['cover'] - 0.05 or ca['spill'] > cb['spill'] + 0.05):
            issues.append({'object': oid, 'check': 'cloud', 'value': ca, 'before': cb, 'level': 'report',
                           'note': 'fits the scan worse than before'})
        return issues

    def review(self, oid: str, before: Metrics, change_report: dict | None = None, reason: str = '') -> dict:
        """Judge the change just made to `oid`; undo it if it made a hard check worse."""
        after = self.measure_now()
        issues = self._compare(oid, before, after)
        stop = [i for i in issues if i['level'] == 'stop']
        result = {'object': oid, 'version_before': before and getattr(before, 'version', None),
                  'issues': issues, 'moved': (change_report or {}).get('moved', [])}
        if stop:
            self.streaks[oid] = self.streaks.get(oid, 0) + 1
            undone = False
            try:
                self.state.undo(oid, reason=f'verify: {stop[0]["check"]}', auto=True)
                undone = True
            except Exception as e:                  # nothing to undo: the starting pose itself is wrong
                result['undo_error'] = str(e)
            result.update(verdict='undo', undone=undone, streak=self.streaks[oid])
            if self.streaks[oid] >= self.cfg.max_undo_streak:
                self.state.set_status(oid, 'failed', note=f'{self.streaks[oid]} consecutive undos')
                result['failed'] = True
        else:
            self.streaks[oid] = 0
            result.update(verdict='keep', streak=0)
        result['after'] = after.as_dict()
        if self.log_dir:
            with open(self.log_dir / 'verify.jsonl', 'a') as f:
                f.write(json.dumps(result, ensure_ascii=False, default=str) + '\n')
        return result

    # ------------------------------------------------------------ physics

    def review_physics(self, seconds: float = 5.0, movable=None, undo: bool = True) -> dict:
        """One gravity run: let `movable` (default: the whole scene's dynamic objects) move,
        then report and undo whatever fell over or dropped."""
        if self.isaac is None:
            raise RuntimeError('physics needs the Isaac service')
        from state.store import IN_SCENE
        oids = movable if movable is not None else [o for o, s in self.state.objects.items() if s.status in IN_SCENE]
        movable = [o for o in oids if o in self.state.objects]
        r = self.isaac.settle(self.state.isaac_objects(), movable, seconds=seconds)
        c = self.cfg
        findings, stop = [], []
        for oid, m in r['moved'].items():
            base = self.state.objects[oid]
            drop = m['dz'] <= -c.phys_drop_m and base.bottom_z + m['dz'] <= self.structure.floor_z + c.phys_near_floor_m
            fall = m['tilt_deg'] >= c.phys_fall_tilt_deg
            item = {'object': oid, 'dz': m['dz'], 'dxy': m['dxy'], 'tilt_deg': m['tilt_deg'],
                    'settled': m['settled'], 'early_jump': m['early_jump'],
                    'level': 'stop' if (drop or fall) else ('report' if (m['dxy'] > c.phys_report_move_m or not m['settled']) else 'ok'),
                    'note': 'dropped' if drop else ('fell over' if fall else '')}
            findings.append(item)
            if item['level'] == 'stop':
                stop.append(oid)
        undone = []
        for oid in (stop if undo else []):     # undo=False: report only (the final whole-scene run, M8)
            try:
                self.state.undo(oid, reason='verify: physics', auto=True)
                undone.append(oid)
            except Exception:
                pass
        out = {'moved': findings, 'fixed': r.get('fixed', []), 'undo': undone, 'stop': stop, 'ms': r.get('ms')}
        if self.log_dir:
            with open(self.log_dir / 'verify_physics.jsonl', 'a') as f:
                f.write(json.dumps(out, ensure_ascii=False, default=str) + '\n')
        return out
