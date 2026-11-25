"""Exact-camera rendering: embree ray casting on the USD visual mesh, sRGB shading."""
import numpy as np
import trimesh


def camera_rays(C2W, fx, fy, cx, cy, W, H, window=None):
    """Ray origins/dirs for pixels in [u0,u1)x[v0,v1). Returns (u0,v0,u1,v1, origin(N,3), dir(N,3))."""
    if window is None:
        u0, v0, u1, v1 = 0, 0, W, H
    else:
        u0, v0, u1, v1 = window
        u0, v0 = max(0, int(u0)), max(0, int(v0))
        u1, v1 = min(W, int(u1)), min(H, int(v1))
    us, vs = np.meshgrid(np.arange(u0, u1) + 0.5, np.arange(v0, v1) + 0.5)
    d = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(us)], -1).reshape(-1, 3)
    d = d @ C2W[:3, :3].T
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    o = np.broadcast_to(C2W[:3, 3], d.shape).copy()
    return (u0, v0, u1, v1), o, d


def render_asset(tr, facecol, C2W, fx, fy, cx, cy, W, H, window=None,
                 ambient=0.38, gain=0.75, near=0.02):
    """Render one mesh under the given camera. Returns rgb(H,W,3) float, mask(H,W) bool, depth."""
    (u0, v0, u1, v1), o, d = camera_rays(C2W, fx, fy, cx, cy, W, H, window)
    rgb = np.zeros((H, W, 3), np.float32)
    mask = np.zeros((H, W), bool)
    depth = np.full((H, W), np.inf, np.float32)
    loc, ray_i, tri_i = tr.ray.intersects_location(o, d, multiple_hits=False)
    if len(ray_i):
        dist = np.linalg.norm(loc - o[ray_i], axis=1)
        keep = dist > near
        loc, ray_i, tri_i, dist = loc[keep], ray_i[keep], tri_i[keep], dist[keep]
        # back-face culling: face normal against view direction
        n = tr.face_normals[tri_i]
        vis = (n * d[ray_i]).sum(1) < 0
        loc, ray_i, tri_i, dist, n = loc[vis], ray_i[vis], tri_i[vis], dist[vis], n[vis]
        lam = np.clip((-(n * d[ray_i]).sum(1)), 0, 1) * 0.72 + 0.28
        shade = (ambient + (1 - ambient) * lam) * gain
        c = np.clip(facecol[tri_i] * shade[:, None], 0, 1)
        yy, xx = (ray_i // (u1 - u0)) + v0, (ray_i % (u1 - u0)) + u0
        rgb[yy, xx] = c
        mask[yy, xx] = True
        depth[yy, xx] = dist
    return rgb, mask, depth


def persp_camera(center, target, up=(0, 0, 1)):
    """OpenCV convention C2W: +Z forward, +Y down."""
    center = np.asarray(center, float)
    f = np.asarray(target, float) - center
    f /= np.linalg.norm(f)
    up = np.asarray(up, float)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:
        r = np.cross(f, [1.0, 0, 0])
    r /= np.linalg.norm(r)
    dn = np.cross(f, r)
    C = np.eye(4)
    C[:3, 0], C[:3, 1], C[:3, 2], C[:3, 3] = r, dn, f, center
    return C
