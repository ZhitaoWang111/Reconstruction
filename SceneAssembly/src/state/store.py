"""The scene state, its versions, and undo.

The harness is the only owner of poses. Every change makes a new version and
writes a full snapshot (states/vNNNN.json) plus one line in actions.jsonl, so a
run can be resumed from any version and every step can be replayed.

Changes (D06 M2):
  set_pose   the agent sets an object's pose and support; it takes effect at once.
             Everything resting on the object (recursively) keeps its pose
             relative to it and is lowered onto it again. The reply lists every
             object whose placement changed.
  undo       takes back an object's last change. What rests on it goes back with
             it, but keeps its own later changes.
  set_status untouched | working | done | failed.

The agent declares supports; nothing here infers them (D17). A support must be
in the scene: done, or in the current batch (D06 M2-9).
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .assets import Assets
from .pose import Pose, Support
from .resolve import Placement, Rejected, Resolver

STATUSES = ('untouched', 'working', 'done', 'failed')
IN_SCENE = ('working', 'done', 'failed')       # rendered and collided with (D06 M2-8: failed objects stay)


@dataclass
class ObjectState:
    pose: Pose
    support: Support
    matrix: np.ndarray
    bottom_z: float
    surface_z: float | None
    walls: list = field(default_factory=list)
    status: str = 'untouched'
    batch: int | None = None
    history: list = field(default_factory=list)    # snapshots before each change of *this* object

    def placement(self) -> dict:
        """What undo restores."""
        return {'pose': self.pose, 'support': self.support, 'matrix': self.matrix.copy(),
                'bottom_z': self.bottom_z, 'surface_z': self.surface_z, 'walls': copy.deepcopy(self.walls)}

    def restore(self, p: dict):
        self.pose, self.support, self.matrix = p['pose'], p['support'], p['matrix'].copy()
        self.bottom_z, self.surface_z, self.walls = p['bottom_z'], p['surface_z'], copy.deepcopy(p['walls'])

    def to_dict(self) -> dict:
        return {'pose': self.pose.to_dict(), 'support': self.support.to_dict(),
                'matrix': np.round(self.matrix, 6).tolist(), 'bottom_z': round(self.bottom_z, 4),
                'surface_z': None if self.surface_z is None else round(self.surface_z, 4),
                'walls': self.walls, 'status': self.status, 'batch': self.batch,
                'history': [{'pose': h['pose'].to_dict(), 'support': h['support'].to_dict(),
                             'matrix': np.round(h['matrix'], 6).tolist(), 'bottom_z': round(h['bottom_z'], 4),
                             'surface_z': None if h['surface_z'] is None else round(h['surface_z'], 4),
                             'walls': h['walls'], 'version': h['version'],
                             'parent_matrix': None if h.get('parent_matrix') is None else np.round(h['parent_matrix'], 6).tolist()}
                            for h in self.history]}

    @classmethod
    def from_dict(cls, d) -> 'ObjectState':
        hist = [{'pose': Pose.from_dict(h['pose']), 'support': Support.from_dict(h['support']),
                 'matrix': np.array(h['matrix']), 'bottom_z': h['bottom_z'], 'surface_z': h['surface_z'],
                 'walls': h['walls'], 'version': h['version'],
                 'parent_matrix': None if h.get('parent_matrix') is None else np.array(h['parent_matrix'])} for h in d['history']]
        return cls(pose=Pose.from_dict(d['pose']), support=Support.from_dict(d['support']),
                   matrix=np.array(d['matrix']), bottom_z=d['bottom_z'], surface_z=d['surface_z'],
                   walls=d['walls'], status=d['status'], batch=d['batch'], history=hist)


def _rot(m):
    u, _, vt = np.linalg.svd(np.asarray(m, float)[:3, :3])
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def relative_pose(old_parent: np.ndarray, new_parent: np.ndarray, child: Pose, child_matrix: np.ndarray):
    """The child's pose after its support moved from old_parent to new_parent, keeping
    where it sits on the support. Rotation follows the parent's yaw; the child's own
    up/tilt/scale stay; height is redone by dropping onto the support."""
    T = new_parent @ np.linalg.inv(old_parent)              # parent's motion (may include a rescale)
    p_new = T @ np.array([child.x, child.y, old_parent[2, 3], 1.0])
    R = _rot(new_parent) @ _rot(old_parent).T
    dyaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    return child.moved(x=float(p_new[0]), y=float(p_new[1]), yaw_deg=child.yaw_deg + dyaw,
                       tilt_toward_deg=child.tilt_toward_deg + dyaw)


class SceneState:
    def __init__(self, scene, structure, out: Path, assets: Assets | None = None):
        self.scene = scene                  # inputs.handoff.Scene
        self.structure = structure          # inputs.structure.Structure
        self.assets = assets or Assets(scene)
        self.resolver = Resolver(self.assets, structure)
        self.objects: dict[str, ObjectState] = {}
        self.version = 0
        self.current_batch: int | None = None
        self.out = Path(out)
        (self.out / 'states').mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ queries

    def in_scene(self, include=()) -> dict:
        """oid -> 4x4 for what exists now (working, done, failed), plus `include`."""
        return {o: s.matrix for o, s in self.objects.items() if s.status in IN_SCENE or o in include}

    def isaac_objects(self, include=()) -> dict:
        """For isaac.client: oid -> (asset usd, 4x4)."""
        return {o: (self.assets.usd(o), m) for o, m in self.in_scene(include).items()}

    def resting_on(self, oid) -> list:
        """Objects whose declared support is `oid`, recursively, nearest first."""
        out, frontier = [], [oid]
        while frontier:
            nxt = [o for o, s in self.objects.items() if s.support.kind == 'object' and s.support.on in frontier and o not in out]
            out += nxt
            frontier = nxt
        return out

    # ------------------------------------------------------------ changes

    def init_object(self, oid, pose: Pose, support: Support, batch=None, note='initial pose'):
        """Harness-made starting placement (M3). Not undoable: it is where history begins."""
        if oid in self.objects:
            raise ValueError(f'{oid} already has a state')
        matrices = self.in_scene()
        pl, pose = self.resolver.resolve(oid, pose, support, matrices)
        self.objects[oid] = ObjectState(pose, support, pl.matrix, pl.bottom_z, pl.surface_z, pl.walls,
                                        status='untouched', batch=batch)
        return self._commit('init', oid, {'note': note}, [oid])

    def set_status(self, oid, status, note=''):
        if status not in STATUSES:
            raise ValueError(f'status must be one of {STATUSES}')
        self.objects[oid].status = status
        return self._commit('status', oid, {'status': status, 'note': note}, [])

    def _check_support_present(self, oid, support: Support):
        if support.kind != 'object':
            return
        on = self.objects.get(support.on)
        if on is None or on.status == 'untouched' and on.batch != self.current_batch:
            raise Rejected(f'{support.on} is not in the scene yet (not processed and not in this batch); '
                           f'bring it into the batch first, or stand {oid} on the floor for now', on=support.on)
        if support.on in [oid, *self.resting_on(oid)]:
            raise Rejected(f'{oid} cannot rest on {support.on}: {support.on} already rests on {oid}', on=support.on)

    def set_pose(self, oid, pose: Pose, support: Support, reason=''):
        """Apply now. Returns the change report; raises Rejected (state untouched) if it cannot be done."""
        st = self.objects[oid]
        self._check_support_present(oid, support)
        others = {o: m for o, m in self.in_scene(include=self._batch_members()).items() if o != oid}
        pl, pose = self.resolver.resolve(oid, pose, support, others, current_bottom=st.bottom_z)

        # work on copies: either the whole change (object + everything on it) applies, or none of it
        staged = {oid: (pose, support, pl)}
        old_mat = {oid: st.matrix}
        new_mat = dict(others)
        new_mat[oid] = pl.matrix
        for child in self.resting_on(oid):
            cs = self.objects[child]
            parent = cs.support.on
            parent_old = old_mat.get(parent, self.objects[parent].matrix)
            parent_new = new_mat[parent]
            cpose = relative_pose(parent_old, parent_new, cs.pose, cs.matrix)
            try:
                cpl, cpose = self.resolver.resolve(child, cpose, cs.support,
                                                   {o: m for o, m in new_mat.items() if o != child},
                                                   hint_z=self._moved_surface(cs, parent_old, parent_new))
            except Rejected as e:
                raise Rejected(f'{child} rests on {parent} and could not follow it: {e.reason}', follower=child) from e
            staged[child] = (cpose, cs.support, cpl)
            old_mat[child] = cs.matrix
            new_mat[child] = cpl.matrix

        for o, (p, s, pl_) in staged.items():
            os_ = self.objects[o]
            if o == oid:
                os_.history.append({**os_.placement(), 'version': self.version,
                                    'parent_matrix': self._parent_matrix(os_.support)})
            os_.pose, os_.support, os_.matrix = p, s, pl_.matrix
            os_.bottom_z, os_.surface_z, os_.walls = pl_.bottom_z, pl_.surface_z, pl_.walls
        if st.status in ('untouched', 'done'):
            st.status = 'working'
        moved = [o for o in staged if not np.allclose(old_mat[o], new_mat[o], atol=1e-6)]
        return self._commit('set_pose', oid, {'reason': reason, 'pose': pose.to_dict(), 'support': support.to_dict()},
                            moved, staged)

    def undo(self, oid, reason='', auto=False):
        """Take back `oid`'s last change; what rests on it follows it back (keeping its own changes)."""
        st = self.objects[oid]
        if not st.history:
            raise Rejected(f'{oid} has no change to undo')
        prev = dict(st.history[-1])
        others = {o: m for o, m in self.in_scene(include=self._batch_members()).items() if o != oid}
        then = prev.get('parent_matrix')
        if then is not None and prev['support'].on in others and not np.allclose(then, others[prev['support'].on], atol=1e-6):
            # its support has moved since: keep where it sat on the support, not where it was in the room
            ppose = relative_pose(then, others[prev['support'].on], prev['pose'], prev['matrix'])
            try:
                ppl, ppose = self.resolver.resolve(oid, ppose, prev['support'], others,
                                                   hint_z=self._moved_surface_z(prev['surface_z'], then, others[prev['support'].on]))
            except Rejected as e:
                raise Rejected(f'undo of {oid}: its earlier place on {prev["support"].on} no longer works: {e.reason}') from e
            prev.update(pose=ppose, matrix=ppl.matrix, bottom_z=ppl.bottom_z, surface_z=ppl.surface_z, walls=ppl.walls)
        old_mat = {oid: st.matrix}
        new_mat = dict(others)
        new_mat[oid] = prev['matrix']
        staged = {}
        for child in self.resting_on(oid):
            cs = self.objects[child]
            parent = cs.support.on
            parent_old = old_mat.get(parent, self.objects[parent].matrix)
            parent_new = new_mat[parent]
            cpose = relative_pose(parent_old, parent_new, cs.pose, cs.matrix)
            try:
                cpl, cpose = self.resolver.resolve(child, cpose, cs.support,
                                                   {o: m for o, m in new_mat.items() if o != child},
                                                   hint_z=self._moved_surface(cs, parent_old, parent_new))
            except Rejected as e:
                raise Rejected(f'undo of {oid}: {child} rests on it and could not follow it back: {e.reason}',
                               follower=child) from e
            staged[child] = (cpose, cs.support, cpl)
            old_mat[child] = cs.matrix
            new_mat[child] = cpl.matrix
        st.history.pop()
        st.restore(prev)
        for o, (p, s, pl_) in staged.items():
            os_ = self.objects[o]
            os_.pose, os_.matrix, os_.bottom_z, os_.surface_z, os_.walls = p, pl_.matrix, pl_.bottom_z, pl_.surface_z, pl_.walls
        moved = [oid] + [o for o in staged if not np.allclose(old_mat[o], new_mat[o], atol=1e-6)]
        return self._commit('auto_undo' if auto else 'undo', oid, {'reason': reason, 'back_to_version': prev['version']},
                            moved, staged)

    # ------------------------------------------------------------ helpers

    def _parent_matrix(self, support: Support):
        if support.kind == 'object' and support.on in self.objects:
            return self.objects[support.on].matrix.copy()
        return None

    def _batch_members(self):
        return [o for o, s in self.objects.items() if s.batch == self.current_batch and self.current_batch is not None]

    @staticmethod
    def _moved_surface_z(surface_z, parent_old, parent_new):
        """Where a surface of the parent is after the parent moved (for the drop's search window).
        Follows the parent's vertical shift and rescale of height."""
        if surface_z is None:
            return None
        T = np.asarray(parent_new) @ np.linalg.inv(parent_old)
        return float((T @ np.array([parent_old[0, 3], parent_old[1, 3], surface_z, 1.0]))[2])

    def _moved_surface(self, cs: ObjectState, parent_old, parent_new):
        return self._moved_surface_z(cs.surface_z, parent_old, parent_new)

    def _commit(self, action, oid, detail, moved, staged=None):
        self.version += 1
        report = {'version': self.version, 'action': action, 'object': oid, **detail, 'moved': []}
        for o in moved:
            s = self.objects[o]
            item = {'object': o, 'pose': s.pose.to_dict(), 'support': s.support.to_dict(),
                    'bottom_z': round(s.bottom_z, 4)}
            if o != oid:
                item['because'] = f'rests on {s.support.on}'
            if s.walls:
                item['walls'] = s.walls
            report['moved'].append(item)
        snap = {'version': self.version, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'current_batch': self.current_batch,
                'action': report, 'objects': {o: s.to_dict() for o, s in self.objects.items()}}
        (self.out / 'states' / f'v{self.version:04d}.json').write_text(json.dumps(snap, indent=1, ensure_ascii=False))
        with open(self.out / 'actions.jsonl', 'a') as f:
            f.write(json.dumps(report, ensure_ascii=False) + '\n')
        return report

    @classmethod
    def load(cls, scene, structure, out: Path, version: int | None = None, assets=None) -> 'SceneState':
        """Resume from states/vNNNN.json (latest if version is None). Later snapshots are left on disk."""
        out = Path(out)
        files = sorted((out / 'states').glob('v*.json'))
        if version is not None:
            files = [f for f in files if int(f.stem[1:]) <= version]
        snap = json.loads(files[-1].read_text())
        s = cls(scene, structure, out, assets)
        s.version = snap['version']
        s.current_batch = snap['current_batch']
        s.objects = {o: ObjectState.from_dict(d) for o, d in snap['objects'].items()}
        return s
