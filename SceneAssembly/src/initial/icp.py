"""Bounded registration of an upright asset against its scan points.

Copied from the pilot harness (`src/legacy/icp.py`) and adapted: there it carried a
position and its own yaw convention, here it works on vertices already upright and
scaled and returns a yaw about the object's centre. Kept as a copy on purpose - the
pilot code will be removed, and this version is ours to change.

The fit is deliberately small: yaw may move at most +-BOUND from the start, the
centre at most MAXT horizontally, and the height never moves. A bad start therefore
cannot wander somewhere else and report a small residual.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

BOUND_DEG = 20.0
MAX_SHIFT_M = 0.25
MAX_STEP_M = 0.06
MAX_STEP_DEG = 5.0
TRIM = 0.85
ITERS = 30
INLIER_M = 0.05


def Rz(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def fit(points: np.ndarray, scan: np.ndarray, yaw_deg: float, centre=(0.0, 0.0)):
    """Refine `yaw_deg` and the centre of `points` (already centred on their own bbox) to `scan`.

    Returns dict(yaw_deg, centre, residual_m, residual_start_m, inlier, diverged).
    `diverged` means the fit did not improve and the start was kept.
    """
    points = np.asarray(points, float)
    sub = points[::max(1, len(points) // 4000)]
    tree = cKDTree(scan)
    yaw0 = float(yaw_deg)
    c0 = np.array([float(centre[0]), float(centre[1])], float)

    def cost(yaw, c):
        p = sub @ Rz(yaw).T + [c[0], c[1], 0.0]
        d, _ = tree.query(p, workers=-1)
        return float(np.median(d)), d

    res0, _ = cost(yaw0, c0)
    yaw, c = yaw0, c0.copy()
    for _ in range(ITERS):
        p = sub @ Rz(yaw).T + [c[0], c[1], 0.0]
        d, j = tree.query(p, workers=-1)
        keep = d <= np.quantile(d, TRIM)
        src, dst = p[keep], scan[j[keep]]
        cs, cd = src.mean(0), dst.mean(0)
        a, b = src[:, :2] - cs[:2], dst[:, :2] - cd[:2]
        H = a.T @ b
        dyaw = float(np.degrees(np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])))
        dyaw = float(np.clip(dyaw, -MAX_STEP_DEG, MAX_STEP_DEG))
        dyaw = float(np.clip(yaw0 + dyaw, yaw0 - BOUND_DEG, yaw0 + BOUND_DEG)) - yaw   # never past the window
        step = (cd - (Rz(dyaw) @ cs))[:2]
        step = np.clip(step, -MAX_STEP_M, MAX_STEP_M)
        c_new = c + step
        off = c_new - c0
        if np.linalg.norm(off) > MAX_SHIFT_M:                                         # never past the window
            c_new = c0 + off / np.linalg.norm(off) * MAX_SHIFT_M
        if abs(dyaw) < 1e-5 and np.linalg.norm(c_new - c) < 1e-6:
            yaw, c = yaw + dyaw, c_new
            break
        yaw, c = yaw + dyaw, c_new
    res, d = cost(yaw, c)
    if not np.isfinite(res) or res > res0:                       # refinement never makes things worse
        res, d = res0, cost(yaw0, c0)[1]
        return dict(yaw_deg=yaw0, centre=tuple(c0), residual_m=res0, residual_start_m=res0,
                    inlier=float((d < INLIER_M).mean()), diverged=True)
    return dict(yaw_deg=yaw, centre=(float(c[0]), float(c[1])), residual_m=res, residual_start_m=res0,
                inlier=float((d < INLIER_M).mean()), diverged=False)
