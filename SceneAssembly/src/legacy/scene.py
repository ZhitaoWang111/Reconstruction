"""Assembly state, the actions that change it, and the checks that warn about it.

State is an immutable chain: every commit writes states/vNNNN.json and never
touches an older one. A move is always proposed first. propose() changes
nothing; it returns the warnings and a before/after render so the agent can
look before it commits. commit() refuses a proposal it has not seen.

Geometry here gives evidence only. Warnings name a problem and a number; they
never block a commit, but the commit reason has to answer them.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

import handoff
from icp import refine
from pose import Pose, apply, matrix, nearest_upright
from usdmesh import load_visual, srgb_faces

PENETRATION_WARN = 0.02    # m of one asset inside another
HOVER_WARN = 0.03          # m of air under an object with no declared support
SINK_WARN = 0.02           # m below its declared support surface
SCALE_WARN = 0.20          # |scale - 1| on footprint or height
SCAN_HEIGHT_WARN = 0.06    # m between the placed bottom and the scanned bottom
OUTSIDE_WARN = 0.15        # m the footprint centre lies outside the scanned extent


@dataclass
class Asset:
    oid: str
    ob: handoff.Object
    verts: np.ndarray
    faces: np.ndarray
    colours: np.ndarray       # sRGB per face
    scan: np.ndarray          # scan points, scan frame

    def mesh(self, pose: Pose) -> trimesh.Trimesh:
        return trimesh.Trimesh(apply(matrix(self.verts, pose), self.verts), self.faces, process=False)


@dataclass
class Placement:
    pose: Pose
    support: str | None = None          # 'floor', another oid, or None
    status: str = 'placed'              # placed | accepted | stale | flagged
    notes: str = ''

    def to_dict(self):
        return {'pose': self.pose.to_dict(), 'support': self.support, 'status': self.status, 'notes': self.notes}


@dataclass
class Proposal:
    pid: str
    oid: str
    pose: Pose
    support: str | None
    reason: str
    warnings: list = field(default_factory=list)


class Scene:
    def __init__(self, scene: handoff.Scene, oids: list[str], out: Path):
        self.handoff = scene
        self.oids = list(oids)
        self.out = Path(out)
        (self.out / 'states').mkdir(parents=True, exist_ok=True)
        self.assets = {}
        for oid in oids:
            ob = scene.objects[oid]
            v, f, col, interp = load_visual(ob.usd)
            self.assets[oid] = Asset(oid, ob, v, f, srgb_faces(col, interp, len(f), f), ob.points())
        self.placed: dict[str, Placement] = {}
        self.flags: dict[str, str] = {}
        self.version = 0
        self.proposals: dict[str, Proposal] = {}
        self._n = 0
        self._candidates = {}

    # ------------------------------------------------------------- evidence
    def candidates(self, oid: str) -> list[dict]:
        """SAM3D plus four yaw-locked scan registrations, each as a full Pose. Cached.

        Every candidate is refined against the scan with the mesh turned the way
        it will be rendered: SAM3D keeps its own up choice, the four
        registrations assume the asset's +Z is up.
        """
        if oid in self._candidates:
            return self._candidates[oid]
        a = self.assets[oid]
        z0 = float(a.scan.min(0)[2])
        up, sam_yaw, tilt = nearest_upright(a.ob.rotation_world)
        starts = [('sam3d', up, sam_yaw)] + [(f'icp{d}', '+Z', float(d)) for d in (0, 90, 180, 270)]
        out = []
        for name, cup, y0 in starts:
            turned = self.turned(oid, cup)
            r = refine(turned, a.scan, np.radians(y0))
            c = dict(name=name, pose=_pose_from_fit(turned, r, cup, z0), residual_cm=round(r['residual'] * 100, 1),
                     inliers=round(r['inlier'], 2), diverged=bool(r['diverged']))
            if name == 'sam3d':
                c.update(sam3d_yaw=round(sam_yaw, 1), sam3d_tilt_deg=round(tilt, 1))
            out.append(c)
        self._candidates[oid] = out
        return out

    def turned(self, oid: str, up: str) -> np.ndarray:
        """Asset vertices with `up` turned to +Z, footprint centred, lowest point at z=0, yaw 0."""
        a = self.assets[oid]
        return apply(matrix(a.verts, Pose(0, 0, 0, 0, up)), a.verts)

    def fit(self, oid: str, pose: Pose, yaw_window=15.0, xy_window=0.25) -> tuple[Pose, dict]:
        """Bounded scan fit starting from `pose`: yaw within +/-yaw_window, XY within xy_window. z, up and scale kept."""
        a = self.assets[oid]
        turned = self.turned(oid, pose.up) * [pose.scale_xy, pose.scale_xy, pose.scale_z]
        lo, hi = turned.min(0), turned.max(0)
        off = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
        Q = (turned - off) @ _Rz(pose.yaw_deg).T
        qlo, qhi = Q.min(0), Q.max(0)
        t0 = np.array([pose.x - (qlo[0] + qhi[0]) / 2, pose.y - (qlo[1] + qhi[1]) / 2, pose.z - qlo[2]])
        r = refine(turned, a.scan, np.radians(pose.yaw_deg), t0=t0,
                   bound=np.radians(yaw_window), maxt=xy_window)
        fitted = replace(_pose_from_fit(turned, r, pose.up, pose.z), scale_xy=pose.scale_xy, scale_z=pose.scale_z)
        return fitted, dict(residual_cm=round(r['residual'] * 100, 1), start_residual_cm=round(r['residual_start'] * 100, 1),
                            kept_start=bool(r['diverged']))

    def contacts(self, oid: str) -> list[dict]:
        """Scan points of other batch objects just under this object's scan bottom. Direction is ambiguous."""
        v = self.assets[oid].scan
        bot = float(np.percentile(v[:, 2], 2))
        lo, hi = v[:, :2].min(0) - 0.04, v[:, :2].max(0) + 0.04
        out = []
        for other in self.oids:
            if other == oid:
                continue
            s = self.assets[other].scan
            sel = ((s[:, 2] > bot - 0.05) & (s[:, 2] < bot + 0.01) & (s[:, 0] > lo[0]) & (s[:, 0] < hi[0])
                   & (s[:, 1] > lo[1]) & (s[:, 1] < hi[1]))
            if sel.sum() >= 5:
                out.append(dict(oid=other, points=int(sel.sum()),
                                z=[round(float(s[sel, 2].min()), 3), round(float(s[sel, 2].max()), 3)]))
        return sorted(out, key=lambda d: -d['points'])

    def rest_z(self, oid: str, pose: Pose, support: str) -> float | None:
        """z at which `oid` comes to rest when lowered straight down onto `support` (None if it misses)."""
        if support == 'floor':
            return self.handoff.floor_z
        if support not in self.placed:
            return None
        s = self.assets[support].mesh(self.placed[support].pose)
        high = pose.with_z(float(s.bounds[1][2]) + 1.0)
        m = self.assets[oid].mesh(high)
        pts = m.vertices[::max(1, len(m.vertices) // 2500)]
        hit, ray, _ = s.ray.intersects_location(pts, np.tile([0, 0, -1.0], (len(pts), 1)), multiple_hits=False)
        if not len(ray):
            return None
        drop = float((pts[ray, 2] - hit[:, 2]).min())
        return high.z - drop

    def scan_bounds(self, oid: str):
        s = self.assets[oid].scan
        return s.min(0), s.max(0)

    # ------------------------------------------------------------- geometry
    def penetration(self, oid: str, pose: Pose, other: str) -> float:
        """How far `oid` at `pose` reaches inside the placed `other`, in metres (0 if apart).

        Inside-ness is exact (ray parity on a watertight mesh); depth is the distance
        to the nearest of ~20k surface samples of `other`, so it is good to ~1 cm.
        """
        m = self.assets[oid].mesh(pose)
        o = self.assets[other].mesh(self.placed[other].pose)
        if not _boxes_overlap(m.bounds, o.bounds):
            return 0.0
        pts = m.vertices[::max(1, len(m.vertices) // 3000)]
        pts = pts[np.all((pts >= o.bounds[0]) & (pts <= o.bounds[1]), axis=1)]
        if not len(pts):
            return 0.0
        inside = o.contains(pts)
        if not inside.any():
            return 0.0
        surface, _ = trimesh.sample.sample_surface(o, 20000, seed=0)
        d, _ = cKDTree(surface).query(pts[inside])
        return float(d.max())

    def warnings(self, oid: str, pose: Pose, support: str | None) -> list[str]:
        out = []
        for other in self.placed:
            if other == oid or other == support:
                continue
            d = self.penetration(oid, pose, other)
            if d > PENETRATION_WARN:
                out.append(f'penetration: {d*100:.0f} cm inside {other} ({self.assets[other].ob.label})')
        if support is not None:
            if support != 'floor' and support not in self.placed:
                out.append(f'support: {support} is not placed yet')
            else:
                rest = self.rest_z(oid, pose, support)
                if rest is None:
                    out.append(f'support: lowered straight down, it misses {support} entirely')
                else:
                    gap = pose.z - rest
                    if gap > HOVER_WARN:
                        out.append(f'support: {gap*100:.0f} cm of air between it and {support} (it would rest at z {rest:.3f})')
                    elif gap < -SINK_WARN:
                        out.append(f'support: sunk {-gap*100:.0f} cm into {support} (it would rest at z {rest:.3f})')
        elif pose.z - self.handoff.floor_z > HOVER_WARN:
            out.append(f'hover: bottom is {(pose.z - self.handoff.floor_z)*100:.0f} cm above the floor and no support is declared')
        # where the scan says the object's bottom was; robust to stray low points
        seen = float(np.percentile(self.assets[oid].scan[:, 2], 2))
        if pose.z < seen - SCAN_HEIGHT_WARN:
            out.append(f'height: bottom at z {pose.z:.3f} is {(seen - pose.z)*100:.0f} cm below where the scan saw it '
                       f'(z {seen:.3f}); if it rests on something, that support may be too low')
        elif pose.z > seen + SCAN_HEIGHT_WARN:
            out.append(f'height: bottom at z {pose.z:.3f} is {(pose.z - seen)*100:.0f} cm above where the scan saw it (z {seen:.3f})')
        for name, s in (('footprint', pose.scale_xy), ('height', pose.scale_z)):
            if abs(s - 1) > SCALE_WARN:
                out.append(f'scale: {name} {s:.2f}x the generated size')
        slo, shi = self.scan_bounds(oid)
        beyond = max(0.0, slo[0] - pose.x, pose.x - shi[0], slo[1] - pose.y, pose.y - shi[1])
        if beyond > OUTSIDE_WARN:
            out.append(f'outside scan: footprint centre is {beyond*100:.0f} cm outside the scanned extent')
        return out

    # ------------------------------------------------------------- actions
    def propose(self, oid: str, pose: Pose, support: str | None, reason: str) -> Proposal:
        if oid not in self.assets:
            raise ValueError(f'{oid} is not in this batch')
        self._n += 1
        p = Proposal(f'p{self._n:03d}', oid, pose, support, reason, self.warnings(oid, pose, support))
        self.proposals[p.pid] = p
        return p

    def commit(self, pid: str, reason: str) -> list[str]:
        p = self.proposals.pop(pid, None)
        if p is None:
            raise ValueError(f'unknown proposal {pid}; propose first')
        moved_from = self.placed.get(p.oid)
        self.placed[p.oid] = Placement(p.pose, p.support, 'placed', reason)
        stale = []
        if moved_from is not None and moved_from.pose != p.pose:
            for oid, pl in self.placed.items():
                if pl.support == p.oid and pl.status != 'stale':
                    pl.status = 'stale'
                    stale.append(oid)
        # every other pending proposal was checked against the old state
        self.proposals.clear()
        self._write(f'commit {p.pid} {p.oid}', {'proposal': p.pid, 'oid': p.oid, 'reason': reason,
                                                 'warnings': p.warnings, 'stale': stale})
        return stale

    def accept(self, oid: str, note: str):
        if oid not in self.placed:
            raise ValueError(f'{oid} is not placed')
        self.placed[oid].status = 'accepted'
        self.placed[oid].notes = note
        self._write(f'accept {oid}', {'oid': oid, 'note': note})

    def flag(self, oid: str, problem: str):
        self.flags[oid] = problem
        if oid in self.placed:
            self.placed[oid].status = 'flagged'
        self._write(f'flag {oid}', {'oid': oid, 'problem': problem})

    def _write(self, what: str, detail: dict):
        self.version += 1
        doc = {'version': self.version, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'action': what,
               'detail': detail, 'objects': {o: p.to_dict() for o, p in self.placed.items()}, 'flags': self.flags}
        path = self.out / 'states' / f'v{self.version:04d}.json'
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(doc, indent=1, ensure_ascii=False))

    def summary(self) -> str:
        if not self.placed and not self.flags:
            return 'nothing placed yet'
        lines = []
        for oid in self.oids:
            p = self.placed.get(oid)
            if p is None:
                lines.append(f'{oid}: unplaced' + (f' (flagged: {self.flags[oid]})' if oid in self.flags else ''))
                continue
            q = p.pose
            lines.append(f'{oid}: {p.status} at ({q.x:.3f}, {q.y:.3f}, {q.z:.3f}) yaw {q.yaw_deg:.0f} up {q.up} '
                         f'scale xy {q.scale_xy:.2f} z {q.scale_z:.2f} support {p.support}')
        return '\n'.join(lines)


def _boxes_overlap(a, b, pad=0.0):
    return bool(np.all(a[0] - pad <= b[1]) and np.all(b[0] - pad <= a[1]))


def _Rz(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _pose_from_fit(turned, r, up, z) -> Pose:
    """Pose whose placed footprint equals the one `refine` produced for `turned`."""
    P = (turned - r['off']) @ _Rz(np.degrees(r['yaw'])).T + r['t']
    lo, hi = P.min(0), P.max(0)
    return Pose(float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2), float(z),
                float(np.degrees(r['yaw'])) % 360, up)
