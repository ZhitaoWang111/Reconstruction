"""What the scene looks like right now, measured. No judgement (that is check.py).

Geometry is measured on the visual meshes, the same shape the agent looks at in the
pictures and the same one heights are computed on. Penetration is the one exception:
it is a yes/no answer from Isaac's contact report, on the collision shapes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from state import Pose


@dataclass
class Metrics:
    contacts: set = field(default_factory=set)          # frozenset({a, b}) of objects in contact
    contact_ms: int = 0
    wall_pen: dict = field(default_factory=dict)        # oid -> deepest point inside a wall, m
    outside: dict = field(default_factory=dict)         # oid -> furthest footprint point outside the room, m
    support_gap: dict = field(default_factory=dict)     # oid -> gap under the support (negative: sunk)
    contact_ratio: dict = field(default_factory=dict)   # oid -> share of the bottom that has support under it
    wall_gap: dict = field(default_factory=dict)        # oid -> gap to the wall it is declared against
    scale_dev: dict = field(default_factory=dict)       # oid -> largest |scale - 1|
    cloud: dict = field(default_factory=dict)           # oid -> {'cover':.., 'spill':..}
    exempt: set = field(default_factory=set)            # objects the wall and room checks skip (doors, windows)

    def as_dict(self) -> dict:
        return {'contacts': sorted(tuple(sorted(p)) for p in self.contacts),
                'wall_pen': self.wall_pen, 'outside': self.outside, 'support_gap': self.support_gap,
                'contact_ratio': self.contact_ratio, 'wall_gap': self.wall_gap,
                'scale_dev': self.scale_dev, 'cloud': self.cloud, 'contact_ms': self.contact_ms,
                'wall_exempt': sorted(self.exempt)}


class Measure:
    def __init__(self, state, structure, isaac=None, wall_exempt_labels=('door', 'window')):
        self.wall_exempt_labels = tuple(wall_exempt_labels)
        self.state = state
        self.structure = structure
        self.isaac = isaac
        self.assets = state.assets
        self._scan_tree = {}

    # ------------------------------------------------------------ helpers

    def world_vertices(self, oid) -> np.ndarray:
        v = self.assets.verts(oid)
        M = self.state.objects[oid].matrix
        return v @ M[:3, :3].T + M[:3, 3]

    def wall_exempt(self, oid) -> bool:
        return self.state.scene.objects[oid].label.strip().lower() in self.wall_exempt_labels

    def _scene(self) -> list:
        return [o for o, s in self.state.objects.items() if s.status != 'untouched']

    def surface_points(self, oid, n=1500) -> np.ndarray:
        """Points spread over the posed surface: vertices alone miss a wall slab that
        a box straddles (its vertices sit on either side, none inside)."""
        v, f = self.assets.verts_faces(oid)
        M = self.state.objects[oid].matrix
        m = trimesh.Trimesh(v @ M[:3, :3].T + M[:3, 3], f, process=False)
        if len(m.faces) <= n // 3:
            return m.vertices
        pts, _ = trimesh.sample.sample_surface(m, n, seed=0)
        return pts

    def _scan(self, oid) -> cKDTree:
        if oid not in self._scan_tree:
            self._scan_tree[oid] = cKDTree(self.state.scene.objects[oid].points())
        return self._scan_tree[oid]

    # ------------------------------------------------------------ checks

    def contacts(self) -> tuple:
        """Object-object contact pairs from Isaac. Wall and floor contacts are dropped here;
        walls are checked analytically below."""
        if self.isaac is None:
            return set(), 0
        pairs = self.isaac.contacts(self.state.isaac_objects())
        out = set()
        for p in pairs:
            a, b = p['a'], p['b']
            if a in self.state.objects and b in self.state.objects:
                out.add(frozenset((a, b)))
        return out, 0

    def wall_penetration(self, oid) -> float:
        pts = self.surface_points(oid)
        worst = 0.0
        for w in self.structure.walls.values():
            q = w.to_wall(pts)
            inside = ((q[:, 0] > 0) & (q[:, 0] < w.length) & (q[:, 1] < 0) & (q[:, 1] > -w.thickness)
                      & (q[:, 2] > 0) & (q[:, 2] < w.height))
            if not inside.any():
                continue
            for o in w.openings:            # a hole is not wall
                in_hole = inside & (q[:, 0] > o.from_m) & (q[:, 0] < o.to_m) & (q[:, 2] > o.z0) & (q[:, 2] < o.z1)
                inside &= ~in_hole
            if inside.any():
                worst = max(worst, float(-q[inside, 1].min()))     # deepest point inside the wall
        return worst

    def outside_room(self, oid) -> float:
        pts = self.world_vertices(oid)[:, :2]
        inside = self.structure.in_room(pts)
        if inside.all():
            return 0.0
        out = pts[~inside]
        # how far out: nearest floor boundary segment
        best = np.full(len(out), np.inf)
        for f in self.structure.floors.values():
            for ring in f.rings:
                a, b = ring, np.roll(ring, -1, 0)
                for (x0, y0), (x1, y1) in zip(a, b):
                    d = np.array([x1 - x0, y1 - y0])
                    L2 = float(d @ d)
                    if L2 <= 1e-12:
                        continue
                    t = np.clip(((out - [x0, y0]) @ d) / L2, 0, 1)
                    proj = np.array([x0, y0]) + t[:, None] * d
                    best = np.minimum(best, np.linalg.norm(out - proj, axis=1))
        return float(best.max()) if np.isfinite(best).any() else 0.0

    def support(self, oid) -> tuple:
        """(gap under the declared support, share of the sampled bottom within 2 cm of it).

        The support is measured against the support object's visual mesh, or the floor plane."""
        st = self.state.objects[oid]
        pts = self.surface_points(oid, 1200)
        low = pts[pts[:, 2] <= pts[:, 2].min() + 0.05]
        if st.support.kind == 'object' and st.support.on in self.state.objects:
            sup = self.state.objects[st.support.on]
            m = trimesh.Trimesh(self.assets.verts(st.support.on) @ sup.matrix[:3, :3].T + sup.matrix[:3, 3],
                                self.assets.verts_faces(st.support.on)[1], process=False)
            loc, ray, _ = m.ray.intersects_location(low, np.tile([0, 0, -1.0], (len(low), 1)), multiple_hits=True)
            if not len(ray):
                return np.inf, 0.0
            gap = np.full(len(low), np.inf)
            np.minimum.at(gap, ray, low[ray, 2] - loc[:, 2])         # nearest surface under each point
            hit = np.isfinite(gap)
            return float(gap[hit].min()) if hit.any() else np.inf, float(np.mean(gap[hit] < 0.05)) if hit.any() else 0.0
        if st.support.kind == 'floor':
            gap = low[:, 2] - self.structure.floor_z
            return float(gap.min()), float((gap < 0.02).mean())
        return 0.0, 1.0                                            # wall-hung / height: nothing under it

    def wall_gap_of(self, oid) -> float:
        """Gap to the wall the object is declared against (inf when not declared)."""
        st = self.state.objects[oid]
        if not st.support.against:
            return np.inf
        pts = self.world_vertices(oid)
        names = st.support.against
        best = np.inf
        for w in self.structure.walls.values():
            if names != ('auto',) and w.id not in names:
                continue
            q = w.to_wall(pts)
            if names == ('auto',) and not (q[:, 0].max() > 0 and q[:, 0].min() < w.length):
                continue
            best = min(best, float(q[:, 1].min()))
        return abs(best)

    def cloud(self, oid) -> dict:
        pts = self.world_vertices(oid)
        tree = self._scan(oid)
        scan = self.state.scene.objects[oid].points()
        verts = tree.query(pts)[0]
        scan_tree = cKDTree(pts)
        return {'cover': float((scan_tree.query(scan)[0] < 0.05).mean()),
                'spill': float((verts > 0.05).mean())}

    # ------------------------------------------------------------ all of it

    def all(self) -> Metrics:
        m = Metrics()
        oids = self._scene()
        if self.isaac is not None:
            m.contacts, m.contact_ms = self.contacts()
        for oid in oids:
            if self.wall_exempt(oid):          # a door or window sits in the wall by nature (D07-3)
                m.wall_pen[oid], m.outside[oid] = 0.0, 0.0
                m.exempt.add(oid)
            else:
                m.wall_pen[oid] = round(self.wall_penetration(oid), 4)
                m.outside[oid] = round(self.outside_room(oid), 4)
            gap, ratio = self.support(oid)
            m.support_gap[oid] = None if not np.isfinite(gap) else round(gap, 4)
            m.contact_ratio[oid] = round(ratio, 3)
            g = self.wall_gap_of(oid)
            if np.isfinite(g):
                m.wall_gap[oid] = round(g, 4)
            s = self.state.objects[oid].pose.scale
            m.scale_dev[oid] = round(max(abs(x - 1.0) for x in s), 3)
            m.cloud[oid] = {k: round(v, 3) for k, v in self.cloud(oid).items()}
        return m
