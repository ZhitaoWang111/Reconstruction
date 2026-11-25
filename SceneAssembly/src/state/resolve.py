"""Pose + support -> a definite 4x4, or a rejection that says why.

Height comes from the support; walls only move the object along their normal.
Nothing here guesses: when the support surface or the wall is not where the
request says, the request is rejected with the reason (D06 M2-4, M2-5).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .pose import Pose, Support

SEARCH_ABOVE_M = 0.15          # with surface_z: search from surface_z + this ...
SEARCH_DEPTH_M = 0.30          # ... down this far
NO_HINT_ABOVE_M = 0.05         # without surface_z: from the current bottom + this, same depth
UP_FACING = 0.3                # a face counts as a support surface if its normal's z exceeds this
WALL_MAX_GAP_M = 0.30          # auto wall: the nearest wall must be within this
WALL_AMBIGUOUS_M = 0.15        # auto wall: rejected when the second wall is this close to the first
CORNER_MAX_COS = 0.26          # two walls: their normals must be 75..105 degrees apart
COPLANAR_M = 0.01


class Rejected(Exception):
    """The request cannot be carried out; `reason` goes back to the agent as is."""

    def __init__(self, reason: str, **detail):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


@dataclass
class Placement:
    matrix: np.ndarray
    bottom_z: float
    surface_z: float | None            # height of the surface it rests on (floor top, the object top it sits on)
    walls: list = field(default_factory=list)   # [{"wall", "face", "push_m"}]


@dataclass(frozen=True)
class WallFace:
    wall: str
    face: str                          # front (room side of the handoff) | back (other side of an interior wall)
    origin: np.ndarray                 # (x, y) of the face line start
    along: np.ndarray                  # unit (x, y)
    normal: np.ndarray                 # unit (x, y), pointing away from the wall into the space in front of this face
    length: float
    base_z: float
    height: float
    openings: tuple

    def coords(self, p):
        """(..., 2|3) -> along, offset in front of this face (0 on the face, < 0 inside the wall)."""
        d = np.asarray(p, float)[..., :2] - self.origin
        return d @ self.along, d @ self.normal

    def plane_key(self):
        return self.normal, float(self.origin @ self.normal)


def wall_faces(structure) -> list:
    faces = []
    for w in structure.walls.values():
        common = dict(wall=w.id, along=w.direction, length=w.length, base_z=w.base_z, height=w.height, openings=w.openings)
        faces.append(WallFace(face='front', origin=w.start, normal=w.normal, **common))
        if w.role == 'interior':       # an interior wall's back is another room's wall
            faces.append(WallFace(face='back', origin=w.start - w.thickness * w.normal, normal=-w.normal, **common))
    return faces


class Resolver:
    def __init__(self, assets, structure):
        self.assets = assets
        self.structure = structure
        self.floor_z = structure.floor_z
        self.ceiling_z = structure.floor_z + structure.storey_height
        self.faces = wall_faces(structure)

    # ------------------------------------------------------------ height

    def drop_onto(self, oid, pose: Pose, on: str, on_matrix: np.ndarray, start_z: float):
        """Lower `oid` (bottom starting at start_z) straight down onto `on`. Returns (bottom_z, surface_z)."""
        verts = self.assets.sample(oid, 3000)
        M = pose.matrix(verts, start_z)
        pts = verts @ M[:3, :3].T + M[:3, 3]
        support = self.assets.posed(on, on_matrix)
        down = np.tile([0, 0, -1.0], (len(pts), 1))
        loc, ray, tri = support.ray.intersects_location(pts, down, multiple_hits=True)
        if len(ray):
            ok = (support.face_normals[tri][:, 2] > UP_FACING) & (loc[:, 2] <= start_z + 1e-4) & (loc[:, 2] <= pts[ray, 2] + 1e-6)
            loc, ray = loc[ok], ray[ok]
        if not len(ray):
            raise Rejected(f'nothing of {on} lies under {oid} at ({pose.x:.2f}, {pose.y:.2f}) below z {start_z:.2f}',
                           on=on, start_z=start_z)
        gap = pts[ray, 2] - loc[:, 2]
        k = int(np.argmin(gap))
        surface = float(loc[k, 2])
        if surface < start_z - SEARCH_DEPTH_M:
            raise Rejected(f'no surface of {on} within {SEARCH_DEPTH_M:.2f} m below z {start_z:.2f} under {oid}; '
                           f'the nearest is at z {surface:.2f}. Give surface_z for the surface it should rest on',
                           on=on, start_z=start_z, nearest_surface_z=surface)
        return start_z - float(gap[k]), surface

    def height(self, oid, pose: Pose, support: Support, matrices: dict, hint_z=None, current_bottom=None):
        """(bottom_z, surface_z) for the declared support."""
        if support.kind == 'floor':
            return self.floor_z, self.floor_z
        if support.kind in ('wall', 'height'):
            return self.floor_z + float(support.height), None
        on = support.on
        if on not in matrices:
            raise Rejected(f'{on} is not in the scene, so {oid} cannot rest on it', on=on)
        if support.surface_z is not None:
            start = float(support.surface_z) + SEARCH_ABOVE_M
        elif hint_z is not None:
            start = float(hint_z) + SEARCH_ABOVE_M
        elif current_bottom is not None:
            start = float(current_bottom) + NO_HINT_ABOVE_M
        else:
            raise Rejected(f'give surface_z: where on {on} should {oid} rest?', on=on)
        return self.drop_onto(oid, pose, on, matrices[on], start)

    # ------------------------------------------------------------ walls

    def _distance(self, face: WallFace, pts):
        a, off = face.coords(pts)
        return a, off, float(off.min())

    def _auto_face(self, oid, pts):
        centre = (pts.min(0) + pts.max(0)) / 2
        cands = []
        for f in self.faces:
            a, off, gap = self._distance(f, pts)
            ca, coff = f.coords(centre)
            if coff <= 0 or a.max() < 0 or a.min() > f.length:
                continue                 # behind the face, or entirely beyond its ends
            cands.append((abs(gap), f))
        if not cands:
            raise Rejected(f'no wall faces {oid}; give the wall id(s) in `against`')
        cands.sort(key=lambda c: c[0])
        d0, f0 = cands[0]
        if d0 > WALL_MAX_GAP_M:
            raise Rejected(f'{oid} is {d0:.2f} m from the nearest wall ({f0.wall}); against a wall means within '
                           f'{WALL_MAX_GAP_M:.2f} m. Check x, y or name the wall', nearest=f0.wall, gap_m=d0)
        n0, c0 = f0.plane_key()
        for d, f in cands[1:]:
            n, c = f.plane_key()
            if np.dot(n, n0) > 0.999 and abs(c - c0) < COPLANAR_M:
                continue                 # another piece of the same flat wall
            if d - d0 < WALL_AMBIGUOUS_M:
                raise Rejected(f'{oid} is near two walls ({f0.wall} {d0:.2f} m, {f.wall} {d:.2f} m); '
                               f'name the wall(s) it is against in `against`', walls=[f0.wall, f.wall])
            break
        return [f0]

    def _named_faces(self, oid, names, pts):
        centre = (pts.min(0) + pts.max(0)) / 2
        out = []
        for name in names:
            mine = [f for f in self.faces if f.wall == name]
            if not mine:
                raise Rejected(f'unknown wall {name!r}; walls are {", ".join(self.structure.walls)}')
            ahead = [f for f in mine if f.coords(centre)[1] > -1e-6]
            if not ahead:
                raise Rejected(f'{oid} is behind wall {name}, not in front of it', wall=name)
            out.append(max(ahead, key=lambda f: f.coords(centre)[1]) if len(ahead) == 1 else
                       min(ahead, key=lambda f: abs(f.coords(centre)[1])))
        if len(out) == 2 and abs(float(np.dot(out[0].normal, out[1].normal))) > CORNER_MAX_COS:
            raise Rejected(f'walls {names[0]} and {names[1]} are not at a corner (about 90 degrees apart); '
                           'an object can be pushed against two walls only in a corner', walls=list(names))
        return out

    def push_to_walls(self, oid, pts, against):
        """Shift (dx, dy) that puts the object flush with the chosen wall face(s), and what was done."""
        faces = self._auto_face(oid, pts) if against == ('auto',) else self._named_faces(oid, against, pts)
        shift = np.zeros(2)
        for _ in range(3):              # two faces at ~90 degrees settle in one or two rounds
            for f in faces:
                _, _, gap = self._distance(f, pts[:, :2] + shift)
                shift = shift - gap * f.normal
        report = []
        for f in faces:
            a, off, gap = self._distance(f, pts[:, :2] + shift)
            push = float(-np.dot(shift, f.normal))
            z = pts[:, 2] - f.base_z
            for o in f.openings:
                if a.max() > o.from_m and a.min() < o.to_m and z.max() > o.z0 and z.min() < o.z1:
                    raise Rejected(f'{oid} against {f.wall} would cover its opening '
                                   f'({o.from_m:.2f}-{o.to_m:.2f} m along the wall)', wall=f.wall)
            if a.max() < 0 or a.min() > f.length:
                raise Rejected(f'{oid} is beyond the ends of wall {f.wall}', wall=f.wall)
            report.append({'wall': f.wall, 'face': f.face, 'push_m': round(push, 4)})
        return shift, report

    # ------------------------------------------------------------ all together

    def resolve(self, oid, pose: Pose, support: Support, matrices: dict, hint_z=None, current_bottom=None) -> Placement:
        """matrices: oid -> 4x4 of every other object in the scene (supports are looked up here)."""
        if support.kind == 'object':
            if support.on == oid:
                raise Rejected(f'{oid} cannot rest on itself')
        verts = self.assets.verts(oid)
        bottom, surface = self.height(oid, pose, support, matrices, hint_z, current_bottom)
        walls = []
        if support.against:
            M = pose.matrix(verts, bottom)
            pts = verts @ M[:3, :3].T + M[:3, 3]
            shift, walls = self.push_to_walls(oid, pts, support.against)
            if np.linalg.norm(shift) > 1e-9:
                pose = pose.moved(x=pose.x + float(shift[0]), y=pose.y + float(shift[1]))
                if support.kind == 'object':     # moved sideways: rest on the support again where it now is
                    bottom, surface = self.height(oid, pose, support, matrices, hint_z=surface)
        M = pose.matrix(verts, bottom)
        top = float((verts @ M[:3, :3].T + M[:3, 3])[:, 2].max())
        if top > self.ceiling_z + 1e-3:
            raise Rejected(f'{oid} would reach z {top:.2f}, above the ceiling at {self.ceiling_z:.2f}', top_z=top)
        return Placement(matrix=M, bottom_z=float(bottom), surface_z=surface, walls=walls), pose
