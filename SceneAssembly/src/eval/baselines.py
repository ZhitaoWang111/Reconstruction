"""Fixed answers to compare the agent against, on the same holdout frames and renderer.

B0  scan bounding-box centre, yaw 0, upright, scale 1, bottom at the scan's lowest point
B1  raw SAM3D orientation (as estimated, including the six that come out upside down),
    at the scan box centre, scale 1
M3  the agent's starting pose (SAM3D corrected, one bounded registration, scan height)

The pilot's B2 (best-residual registration) and its oracle (best of five candidates) are gone:
they depended on the candidate mechanism that was dropped in D03.
"""
from __future__ import annotations

import numpy as np

from state.pose import Pose, UP

B0 = 'B0_scan_box_yaw0'
B1 = 'B1_sam3d_raw'
M3 = 'M3_start'


def _nearest_upright(R):
    """(up, yaw) whose upright frame is closest to this rotation, and the tilt left over - the pilot's rule."""
    best = None
    for name, U in UP.items():
        axis = U.T @ np.array([0, 0, 1.0])
        tilt = float(np.degrees(np.arccos(np.clip((R @ axis)[2], -1, 1))))
        if best is None or tilt < best[2]:
            Rp = R @ U.T
            best = (name, float(np.degrees(np.arctan2(Rp[1, 0], Rp[0, 0]))), tilt)
    return best


def baseline_matrices(scene, initial, oid, verts) -> dict:
    """oid -> {name: 4x4} for the three baselines."""
    o = scene.objects[oid]
    scan = o.points()
    lo, hi = scan.min(0), scan.max(0)
    x, y = float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)
    b0 = Pose(x=x, y=y, yaw_deg=0.0, up='+Z', scale=(1.0, 1.0, 1.0))
    up, yaw, _tilt = _nearest_upright(o.rotation_world)
    b1 = Pose(x=x, y=y, yaw_deg=yaw, up=up, scale=(1.0, 1.0, 1.0))
    out = {B0: b0.matrix(verts, float(lo[2])), B1: b1.matrix(verts, float(lo[2]))}
    d = initial[oid]
    out[M3] = Pose.from_dict(d['pose']).matrix(verts, float(d['bottom_z']))
    return out
