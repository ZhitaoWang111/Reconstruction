"""Load the structure handoff (walls and floors) scene assembly builds the room from.

The handoff is ObjectGeneration's `structure_handoff/v2`: `structure.usdc` holds
the wall bodies, trims and floor slabs as static triangle-mesh colliders, and
`structure.json` describes the same geometry in numbers. Assembly references
the USD as it is and uses the JSON for the checks and placements that need the
geometry (wall faces, room outline).

It must be in the same frame as the object handoff. That is checked here, along
with the file hashes the manifest records, so a regenerated structure is never
silently paired with objects from another frame.

Conventions (from the handoff):
  - a segment's `front` face is the room side; `front_floor` is that face's
    footprint line [start, end] on the floor, `normal_room` points into the room;
  - `along_m` and opening ranges are measured from the front line's start,
    heights (`z_floor`) from the wall base `base_z`;
  - a floor's `top_polygons` are rings in (x, y); its top is at `top_z`.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STRUCTURE_HANDOFF = Path(os.environ.get(
    'STRUCTURE_HANDOFF',
    Path(__file__).resolve().parents[3]
    / 'ObjectGeneration/output/scene0000_00/structuregen/structure_handoff_scannet_scan'))
SCHEMA = 'structure_handoff/v2'
FLOOR_TOLERANCE_M = 0.01   # structure floor vs object-handoff floor_z


@dataclass(frozen=True)
class Opening:
    from_m: float              # along the front line, from its start
    to_m: float
    z0: float                  # above the wall base
    z1: float


@dataclass(frozen=True, eq=False)
class Wall:
    id: str
    prim: str                  # Xform; the body mesh is `body`
    body: str
    role: str                  # exterior | interior
    start: np.ndarray          # front (room-side) face footprint, (x, y)
    end: np.ndarray
    normal: np.ndarray         # unit (x, y) pointing into the room
    thickness: float
    height: float
    base_z: float
    openings: tuple

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    @property
    def direction(self) -> np.ndarray:
        return (self.end - self.start) / self.length

    def to_wall(self, p) -> np.ndarray:
        """Scan-frame points (..., 2 or 3) -> (along, offset into the room, height above the base).

        offset is 0 on the front face, positive in the room, negative inside the wall.
        """
        p = np.asarray(p, float)
        d = p[..., :2] - self.start
        out = np.stack([d @ self.direction, d @ self.normal], -1)
        if p.shape[-1] == 3:
            out = np.concatenate([out, p[..., 2:3] - self.base_z], -1)
        return out

    def from_wall(self, along, offset, height=None) -> np.ndarray:
        """Inverse of to_wall for one point."""
        xy = self.start + along * self.direction + offset * self.normal
        return xy if height is None else np.array([*xy, self.base_z + height])


@dataclass(frozen=True, eq=False)
class Floor:
    id: str
    kind: str                  # room | threshold
    prim: str
    top_z: float
    rings: tuple               # each an (N, 2) array; even-odd fill, so holes work too
    area: float

    def contains(self, xy) -> np.ndarray:
        """Even-odd point-in-polygon over all rings; xy is (..., 2)."""
        xy = np.asarray(xy, float)
        flat = xy.reshape(-1, 2)
        inside = np.zeros(len(flat), bool)
        for r in self.rings:
            a, b = r, np.roll(r, -1, 0)
            for (x0, y0), (x1, y1) in zip(a, b):
                crosses = (y0 > flat[:, 1]) != (y1 > flat[:, 1])
                with np.errstate(divide='ignore', invalid='ignore'):
                    xc = x0 + (flat[:, 1] - y0) * (x1 - x0) / (y1 - y0)
                inside ^= crosses & (flat[:, 0] < xc)
        return inside.reshape(xy.shape[:-1])


@dataclass(frozen=True, eq=False)
class Structure:
    root: Path
    usd: Path
    scene_id: str
    frame: str
    floor_z: float             # wall bases / floor tops
    storey_height: float
    walls: dict                # id -> Wall
    floors: dict               # id -> Floor
    objects_in_walls: dict     # oid -> the handoff's record (segment, share of points inside, depth p95)

    def in_room(self, xy) -> np.ndarray:
        """True where (x, y) is over any floor (rooms and thresholds together)."""
        xy = np.asarray(xy, float)
        out = np.zeros(xy.shape[:-1], bool)
        for f in self.floors.values():
            out |= f.contains(xy)
        return out


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(root: Path = STRUCTURE_HANDOFF, scene=None) -> Structure:
    """Read and check the structure handoff. Pass the object handoff's `handoff.Scene`
    to also check that the two share a scene, a frame and a floor."""
    root = Path(root)
    man = json.loads((root / 'manifest.json').read_text())
    if man['schema_version'] != SCHEMA:
        raise ValueError(f"{root}: schema {man['schema_version']}, expected {SCHEMA}")
    files = man['files']
    usd, js = root / files['usd'], root / 'structure.json'
    for path, want in ((usd, files['usd_sha256']), (js, files['structure_sha256'])):
        if _sha(path) != want:
            raise ValueError(f'{path} does not match the sha256 in its manifest')
    if not man.get('checks', {}).get('passed'):
        raise ValueError(f'{root}: the export checks did not pass')
    s = json.loads(js.read_text())
    if s['coordinate_frame'] != man['coordinate_frame'] or s['units'] != 'meters' or s['up_axis'] != 'Z':
        raise ValueError(f'{js}: frame/units/up differ from the manifest')

    walls = {}
    for g in s['segments']:
        (x0, y0), (x1, y1) = g['front_floor']
        n = np.array(g['normal_room'][:2], float)
        walls[g['id']] = Wall(
            id=g['id'], prim=g['prim'], body=g['body'], role=g['role'],
            start=np.array([x0, y0], float), end=np.array([x1, y1], float), normal=n / np.linalg.norm(n),
            thickness=float(g['thickness_m']), height=float(g['height_m']), base_z=float(g['base_z']),
            openings=tuple(Opening(o['from_m'], o['to_m'], *o['z_floor']) for o in g['openings']))
    floors = {f['id']: Floor(f['id'], f['kind'], f['prim'], float(f['top_z']),
                             tuple(np.array(r, float) for r in f['top_polygons']), float(f['area_m2']))
              for f in s['floors']}
    st = Structure(root=root, usd=usd, scene_id=s['scene_id'], frame=s['coordinate_frame'],
                   floor_z=float(man['frame']['floor_z']), storey_height=float(man['storey']['height_m']),
                   walls=walls, floors=floors,
                   objects_in_walls={o['object_id']: o for o in s['objects_in_walls']['objects']})

    if scene is not None:
        if st.scene_id != scene.scene_id:
            raise ValueError(f'structure is {st.scene_id}, objects are {scene.scene_id}')
        if st.frame != scene.frame:
            raise ValueError(f'structure frame {st.frame} != object frame {scene.frame}')
        if abs(st.floor_z - scene.floor_z) > FLOOR_TOLERANCE_M:
            raise ValueError(f'structure floor {st.floor_z} vs object floor_z {scene.floor_z}')
    return st
