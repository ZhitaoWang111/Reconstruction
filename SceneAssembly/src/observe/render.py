"""Offline renderer for the four-side views (an asset on its own, no room needed).

Copied from the pilot harness (`src/legacy/render.py`) rather than imported: the
pilot code is going away. Embree ray casting on the visual mesh, sRGB face colours.
"""
from __future__ import annotations

import numpy as np


def camera_rays(C2W, fx, fy, cx, cy, W, H):
    us, vs = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(us)], -1).reshape(-1, 3)
    d = d @ C2W[:3, :3].T
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    o = np.broadcast_to(C2W[:3, 3], d.shape).copy()
    return o, d


def render_mesh(tr, facecol, C2W, fx, fy, cx, cy, W, H, ambient=0.45, gain=0.95, near=0.02):
    """rgb (H, W, 3) float in 0..1 and a mask of covered pixels."""
    o, d = camera_rays(C2W, fx, fy, cx, cy, W, H)
    rgb = np.zeros((H, W, 3), np.float32)
    mask = np.zeros((H, W), bool)
    loc, ray_i, tri_i = tr.ray.intersects_location(o, d, multiple_hits=False)
    if len(ray_i):
        dist = np.linalg.norm(loc - o[ray_i], axis=1)
        keep = dist > near
        ray_i, tri_i = ray_i[keep], tri_i[keep]
        n = tr.face_normals[tri_i]
        lam = np.clip(np.abs((n * d[ray_i]).sum(1)), 0, 1) * 0.72 + 0.28
        shade = (ambient + (1 - ambient) * lam) * gain
        c = np.clip(facecol[tri_i] * shade[:, None], 0, 1)
        yy, xx = ray_i // W, ray_i % W
        rgb[yy, xx] = c
        mask[yy, xx] = True
    return rgb, mask


def look_at(center, target, up=(0, 0, 1)):
    """OpenCV-convention camera-to-world: +Z forward, +Y down."""
    center = np.asarray(center, float)
    f = np.asarray(target, float) - center
    f /= np.linalg.norm(f)
    r = np.cross(f, np.asarray(up, float))
    if np.linalg.norm(r) < 1e-6:
        r = np.cross(f, [1.0, 0, 0])
    r /= np.linalg.norm(r)
    dn = np.cross(f, r)
    C = np.eye(4)
    C[:3, 0], C[:3, 1], C[:3, 2], C[:3, 3] = r, dn, f, center
    return C
