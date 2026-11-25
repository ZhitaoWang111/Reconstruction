"""Yaw-locked refinement of an upright asset against the object's own scan points.

The hypothesis start is given (yaw from a discrete set + the scan-bbox footprint translation).
Refinement is bounded: yaw may move at most +/-BOUND from the start, z is locked to the scan
bottom, XY translation may move at most MAXT. This keeps a bad hypothesis from wandering into
a different place and reporting a falsely small residual.
"""
import numpy as np
from scipy.spatial import cKDTree

BOUND = np.radians(20.0)
MAXT = 0.25


def yaw_of(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def footprint_placement(verts, scan):
    lo, hi = verts.min(0), verts.max(0)
    off = np.array([(lo[0]+hi[0])/2, (lo[1]+hi[1])/2, lo[2]])
    slo, shi = scan.min(0), scan.max(0)
    t = np.array([(slo[0]+shi[0])/2, (slo[1]+shi[1])/2, slo[2]])
    return off, t


def cost(P, tree):
    d, j = tree.query(P, workers=-1)
    return float(np.median(d)), d, j


def refine(verts, scan, yaw0, iters=30, trim=0.85, t0=None, bound=BOUND, maxt=MAXT):
    """Bounded yaw + XY fit of `verts` to `scan`. Starts from the scan footprint unless `t0` is given.

    `t0` is where the unrotated bbox centre-bottom (`off`) of `verts` lands; z stays at t0[2].
    """
    off, t_fp = footprint_placement(verts, scan)
    t0 = t_fp if t0 is None else np.asarray(t0, float)
    tree = cKDTree(scan)
    sub = verts[::max(1, len(verts)//4000)]
    yaw, t = float(yaw0), t0.copy()
    res0 = cost((sub-off) @ Rz(yaw).T + t, tree)[0]
    for _ in range(iters):
        P = (sub-off) @ Rz(yaw).T + t
        _, d, j = cost(P, tree)
        k = d <= np.quantile(d, trim)
        src, dst = P[k], scan[j[k]]
        cs, cd = src.mean(0), dst.mean(0)
        a, b = src[:, :2]-cs[:2], dst[:, :2]-cd[:2]
        H = a.T @ b
        dyaw = float(np.arctan2(H[0, 1]-H[1, 0], H[0, 0]+H[1, 1]))
        dyaw = float(np.clip(dyaw, -np.radians(5), np.radians(5)))
        dyaw = float(np.clip(yaw0+dyaw, yaw0-bound, yaw0+bound)) - yaw
        step = cd - (Rz(dyaw) @ cs)
        step[2] = 0
        step[:2] = np.clip(step[:2], -0.06, 0.06)
        t_new = t + step
        dt = t_new[:2]-t0[:2]
        if np.linalg.norm(dt) > maxt:
            t_new[:2] = t0[:2] + dt/np.linalg.norm(dt)*maxt
        t_new[2] = t0[2]
        if abs(dyaw) < 1e-5 and np.linalg.norm(t_new-t) < 1e-6:
            t = t_new
            break
        yaw, t = yaw+dyaw, t_new
    P = (sub-off) @ Rz(yaw).T + t
    _, d, _ = cost(P, tree)
    res = float(np.median(d))
    # refinement is never allowed to make a hypothesis worse than its start
    if res > res0:
        return dict(yaw=float(yaw0), t=t0, residual=float(res0), residual_start=float(res0),
                    inlier=float((cost((sub-off) @ Rz(yaw0).T + t0, tree)[1] < 0.05).mean()),
                    start=float(yaw0), off=off, diverged=True)
    return dict(yaw=yaw, t=t, residual=res, residual_start=float(res0),
                inlier=float((d < 0.05).mean()), start=float(yaw0), off=off, diverged=False)
