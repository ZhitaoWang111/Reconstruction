"""Pose of an asset in the scan frame, in the terms the agent works with.

A USD asset is metres, Z-up, its XY bounding box centred on the origin and its
lowest point at z=0. A pose maps it into the scan frame: turn the chosen asset
axis (`up`) to world +Z, scale the footprint by `scale_xy` and the height by
`scale_z`, rotate by `yaw_deg` about +Z, then move it so its axis-aligned
footprint is centred on (x, y) and its lowest point sits at z. So z is the
height the object stands on - the floor is floor_z, "on the bed" is the bed's
top - whichever way the asset is turned.

Footprint and height scale separately because generated assets get their
height right far more often than their proportions: on scene0000_00 batch 1
every asset height is within 6% of the annotation, while footprints are off by
up to 1.9x. Shrinking a bed uniformly to fix its footprint also lowers its top,
and everything resting on it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

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


# Which asset axis ends up pointing to world +Z, and the rotation that does it.
UP = {'+Z': np.eye(3), '-Z': _Rx(180), '+X': _Ry(-90), '-X': _Ry(90), '+Y': _Rx(90), '-Y': _Rx(-90)}


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    z: float
    yaw_deg: float
    up: str = '+Z'
    scale_xy: float = 1.0
    scale_z: float = 1.0

    def with_z(self, z: float) -> 'Pose':
        return replace(self, z=float(z))

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def matrix(verts: np.ndarray, pose: Pose) -> np.ndarray:
    """4x4 that takes the asset's own USD points to the scan frame.

    The footprint is centred after the yaw, so (x, y) is exactly the centre of the
    placed object's axis-aligned footprint whatever the yaw.
    """
    S = np.diag([pose.scale_xy, pose.scale_xy, pose.scale_z])
    L = Rz(pose.yaw_deg) @ S @ UP[pose.up]
    c = verts @ L.T
    lo, hi = c.min(0), c.max(0)
    M = np.eye(4)
    M[:3, :3] = L
    M[:3, 3] = [pose.x - (lo[0] + hi[0]) / 2, pose.y - (lo[1] + hi[1]) / 2, pose.z - lo[2]]
    return M


def apply(M: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ M[:3, :3].T + M[:3, 3]


def nearest_upright(R_world: np.ndarray) -> tuple[str, float, float]:
    """Express a full rotation (asset -> scan) as the closest (up, yaw) and the tilt left over."""
    best = None
    for name, U in UP.items():
        axis = U.T @ [0, 0, 1.0]                      # asset axis this choice turns upward
        tilt = float(np.degrees(np.arccos(np.clip((R_world @ axis)[2], -1, 1))))
        if best is None or tilt < best[2]:
            Rp = R_world @ U.T
            best = (name, float(np.degrees(np.arctan2(Rp[1, 0], Rp[0, 0]))) % 360, tilt)
    return best
