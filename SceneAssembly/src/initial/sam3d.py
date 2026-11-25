"""SAM3D's rotation -> the starting orientation, in the terms the agent works with.

SAM3D gives one thing about an object's orientation: a 3x3 rotation taking the
asset's own axes to the scan frame (no position, no scale). Two rules turn it into
an `up` axis and a `yaw` (D03 M3-2, M3-3):

  * the asset axis pointing most nearly to +Z becomes `up`;
  * when that is the asset's -Z, `up` is set to +Z instead and the object is marked
    `flipped`. Generated assets are upright, so -Z means SAM3D's whole rotation is
    turned over (fridge outlet underneath, microwave feet up), which is a known
    SAM3D failure rather than a real upside-down object. A side axis is kept as it
    is: those assets were generated lying down (pillow, guitar case).
  * `yaw` is the horizontal heading; it is only trustworthy when the object is not
    flipped (on the six flipped ones it is off, sometimes by 180 degrees).
  * tilt is always 0: the leftover tilt after snapping to an axis is a median 6.3
    degrees, mostly SAM3D error, and a tilted start with no declared support only
    makes objects poke into their neighbours. An agent that wants a tilt sets it.
"""
from __future__ import annotations

import numpy as np

from state.pose import AXIS, Pose, UP


def orthonormal(R) -> np.ndarray:
    """Nearest rotation (SAM3D's output can carry a little scale or noise)."""
    u, _, vt = np.linalg.svd(np.asarray(R, float)[:3, :3])
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def _heading(R: np.ndarray) -> float:
    """Horizontal direction of the asset's +X after R, deg, unless it is nearly vertical."""
    for col in (0, 1, 2):
        h = R[:2, col]
        if np.linalg.norm(h) > 0.2:
            return float(np.degrees(np.arctan2(h[1], h[0])))
    return 0.0


def orientation(R) -> dict:
    """dict(up, yaw_deg, flipped, sam3d_up, deviation_deg)."""
    R = orthonormal(R)
    scores = {k: float((R @ (UP[k].T @ np.array([0, 0, 1.0])))[2]) for k in UP}
    sam3d_up = max(scores, key=scores.get)
    up, flipped = ('+Z', True) if sam3d_up == '-Z' else (sam3d_up, False)
    if flipped:
        yaw = _heading(R)
    else:
        Q = R @ UP[up].T                    # upright frame -> scan frame
        yaw = float(np.degrees(np.arctan2(Q[1, 0], Q[0, 0])))
    return {'up': up, 'yaw_deg': yaw, 'flipped': flipped, 'sam3d_up': sam3d_up,
            'deviation_deg': float(np.degrees(np.arccos(np.clip(scores[sam3d_up], -1, 1))))}


def start_pose(obj, x: float, y: float) -> tuple:
    """The unrefined starting pose for a handoff object at (x, y), and its record."""
    o = orientation(obj.rotation_world)
    pose = Pose(x=float(x), y=float(y), yaw_deg=o['yaw_deg'], up=o['up'], scale=(1.0, 1.0, 1.0),
                tilt_deg=0.0, tilt_toward_deg=0.0)
    return pose, o
