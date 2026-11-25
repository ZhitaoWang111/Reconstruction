"""The pose the agent sets, and the 4x4 it becomes.

An asset is a USD mesh in its own frame (metres, Z-up). A pose puts it in the
scan frame in this order:

  1. `up`     which of the asset's own axes (+X, -X, +Y, -Y, +Z, -Z) points to the sky;
              the asset is now "upright";
  2. `scale`  three factors on the upright asset's own length, width and height;
  3. `yaw_deg` turn about the vertical, counter-clockwise from +X;
  4. `tilt_deg` / `tilt_toward_deg` tip the object over by tilt_deg, its top moving
              toward the horizontal direction tilt_toward_deg (same angle convention
              as yaw). 0 for anything standing straight, which is almost everything.

All turning happens about the object's own centre: the centre of the upright,
scaled asset's bounding box. (x, y) is where that centre ends up. The agent never
sets z: it follows from the support (see Support and resolve.py).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import numpy as np


def Rz(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _Rx(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def _Ry(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


# asset axis that ends up pointing to world +Z, and the rotation that does it
UP = {'+Z': np.eye(3), '-Z': _Rx(180), '+X': _Ry(-90), '-X': _Ry(90), '+Y': _Rx(90), '-Y': _Rx(-90)}
AXIS = {'+X': (0, 1), '-X': (0, -1), '+Y': (1, 1), '-Y': (1, -1), '+Z': (2, 1), '-Z': (2, -1)}


def tilt_matrix(tilt_deg, toward_deg):
    """Rotation that tips +Z by tilt_deg toward the horizontal direction toward_deg."""
    if not tilt_deg:
        return np.eye(3)
    t = np.radians(toward_deg)
    axis = np.array([-np.sin(t), np.cos(t), 0.0])          # z x d: rotating +Z about it moves it toward d
    a = np.radians(tilt_deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    yaw_deg: float = 0.0
    up: str = '+Z'
    scale: tuple = (1.0, 1.0, 1.0)      # upright length, width, height
    tilt_deg: float = 0.0
    tilt_toward_deg: float = 0.0

    def __post_init__(self):
        if self.up not in UP:
            raise ValueError(f'up must be one of {sorted(UP)}, got {self.up!r}')
        s = tuple(float(v) for v in self.scale)
        if len(s) != 3 or min(s) <= 0:
            raise ValueError(f'scale must be three positive numbers, got {self.scale!r}')
        object.__setattr__(self, 'scale', s)
        if not 0 <= self.tilt_deg <= 180:
            raise ValueError(f'tilt_deg must be within 0..180, got {self.tilt_deg}')

    # rotation and linear part --------------------------------------------------
    def rotation(self) -> np.ndarray:
        """Upright frame -> scan frame (yaw, then tilt). Scale and `up` excluded."""
        return tilt_matrix(self.tilt_deg, self.tilt_toward_deg) @ Rz(self.yaw_deg)

    def linear(self) -> np.ndarray:
        """Asset frame -> scan frame, without translation."""
        return self.rotation() @ np.diag(self.scale) @ UP[self.up]

    def upright(self, verts: np.ndarray) -> np.ndarray:
        """Asset vertices -> the upright, scaled asset (before yaw and tilt)."""
        return verts @ (np.diag(self.scale) @ UP[self.up]).T

    def matrix(self, verts: np.ndarray, bottom_z: float) -> np.ndarray:
        """4x4 asset -> scan frame, with the object's centre at (x, y) and its lowest point at bottom_z."""
        u = self.upright(verts)
        centre = (u.min(0) + u.max(0)) / 2
        R = self.rotation()
        w = (u - centre) @ R.T
        M = np.eye(4)
        M[:3, :3] = self.linear()
        M[:3, 3] = np.array([self.x, self.y, bottom_z - w[:, 2].min()]) - R @ centre
        return M

    def moved(self, **kw) -> 'Pose':
        return replace(self, **kw)

    def to_dict(self) -> dict:
        d = asdict(self)
        d['scale'] = [round(v, 4) for v in self.scale]
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}

    @classmethod
    def from_dict(cls, d) -> 'Pose':
        d = dict(d)
        d['scale'] = tuple(d.get('scale', (1.0, 1.0, 1.0)))
        return cls(**d)

    @classmethod
    def from_rotation(cls, R: np.ndarray, x: float, y: float, scale=(1.0, 1.0, 1.0), up: str | None = None) -> 'Pose':
        """Express an arbitrary asset -> scan rotation as up + yaw + tilt.
        `up` defaults to the asset axis that points closest to the sky (smallest tilt)."""
        R = np.asarray(R, float)
        if up is None:
            up = max(UP, key=lambda k: (R @ (UP[k].T @ [0, 0, 1.0]))[2])
        Q = R @ UP[up].T                                  # upright frame -> scan
        u = Q[:, 2]
        tilt = float(np.degrees(np.arccos(np.clip(u[2], -1, 1))))
        toward = float(np.degrees(np.arctan2(u[1], u[0]))) if tilt > 1e-6 else 0.0
        Y = tilt_matrix(tilt, toward).T @ Q               # pure rotation about +Z
        yaw = float(np.degrees(np.arctan2(Y[1, 0], Y[0, 0])))
        return cls(x=float(x), y=float(y), yaw_deg=yaw, up=up, scale=tuple(scale),
                   tilt_deg=round(tilt, 6), tilt_toward_deg=toward)


SUPPORT_KINDS = ('floor', 'object', 'wall', 'height')


@dataclass(frozen=True)
class Support:
    """What carries the object, which fixes its height.

    floor   stands on the floor.
    object  rests on another object `on`. `surface_z` is the rough height of the surface
            it rests on (a shelf, a bed top); the drop searches just above it and below.
            Without it the search starts at the object's current bottom.
    wall    hangs on a wall; `height` is its bottom above the floor.
    height  bottom at `height` above the floor and nothing carries it. Only for the
            harness's own placements (an initial pose before the agent has declared a
            support); the agent's tools do not offer it.

    `against` is about walls, not weight: () means not against a wall; ('auto',) means
    against the nearest wall; one or two wall ids name them. A wall-hung object is
    always against a wall, so for kind 'wall' an empty `against` means ('auto',).
    """
    kind: str = 'floor'
    on: str | None = None
    surface_z: float | None = None
    height: float | None = None
    against: tuple = field(default_factory=tuple)

    def __post_init__(self):
        if self.kind not in SUPPORT_KINDS:
            raise ValueError(f'support kind must be one of {SUPPORT_KINDS}, got {self.kind!r}')
        if self.kind == 'object' and not self.on:
            raise ValueError("support kind 'object' needs `on` (the supporting object's id)")
        if self.kind != 'object' and self.on:
            raise ValueError(f"`on` only goes with kind 'object', not {self.kind!r}")
        if self.kind in ('wall', 'height') and self.height is None:
            raise ValueError(f"support kind {self.kind!r} needs `height` (bottom above the floor, m)")
        a = tuple(self.against)
        if self.kind == 'wall' and not a:
            a = ('auto',)
        if len(a) > 2 or ('auto' in a and len(a) != 1):
            raise ValueError(f"`against` is (), ('auto',), or one or two wall ids; got {self.against!r}")
        object.__setattr__(self, 'against', a)

    def to_dict(self) -> dict:
        d = {'kind': self.kind}
        if self.on:
            d['on'] = self.on
        if self.surface_z is not None:
            d['surface_z'] = round(float(self.surface_z), 4)
        if self.height is not None:
            d['height'] = round(float(self.height), 4)
        if self.against:
            d['against'] = list(self.against)
        return d

    @classmethod
    def from_dict(cls, d) -> 'Support':
        d = dict(d)
        d['against'] = tuple(d.get('against', ()))
        return cls(**d)
