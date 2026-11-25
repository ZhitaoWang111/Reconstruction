"""Where each object of a batch is in a key frame: project its scan points, keep the
visible ones (sensor depth), take the box. Falls back to the object's own mask."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

FRAMES = Path('/home/seer/wzt/Reconstruction/Video2PointCloud/data/scannet/scene0000_00_uniform500/input')
OCCLUSION_SLACK_M = 0.10       # a scan point counts as visible when the sensor sees nothing this much in front of it
MIN_VISIBLE_POINTS = 30
MIN_VISIBLE_SHARE = 0.03       # of the object's points that fall inside the image
BOX_PERCENTILE = 2.0           # trim a few stray points before taking the box


class Sensor:
    """Sensor depth of a frame, resampled onto the colour camera (same as the legacy evaluator)."""

    def __init__(self, camera):
        self.cam = camera
        self._cache = {}

    def depth(self, seq):
        if seq in self._cache:
            return self._cache[seq]
        npz = FRAMES / f'{seq:04d}.npz'
        png = FRAMES / f'{seq:04d}_depth.png'
        if not (npz.exists() and png.exists()):
            self._cache[seq] = None
            return None
        z = np.load(npz)
        raw = np.asarray(Image.open(png), np.float32) / 65535.0 * float(z['maximum_depth'])
        Kd = z['camera_intrinsics'].astype(float)
        c = self.cam
        us, vs = np.meshgrid(np.arange(c.width) + 0.5, np.arange(c.height) + 0.5)
        x = (us - c.cx) / c.fx
        y = (vs - c.cy) / c.fy
        ud = np.clip(np.round(Kd[0, 0] * x + Kd[0, 2]).astype(int), 0, raw.shape[1] - 1)
        vd = np.clip(np.round(Kd[1, 1] * y + Kd[1, 2]).astype(int), 0, raw.shape[0] - 1)
        zd = raw[vd, ud]                                  # depth along the optical axis
        out = np.where(zd > 0, zd, np.inf).astype(np.float32)
        self._cache[seq] = out
        return out


def project(points, c2w, cam):
    """Scan points -> (u, v, depth along the optical axis), plus the mask of points in front of the camera."""
    w2c = np.linalg.inv(c2w)
    p = points @ w2c[:3, :3].T + w2c[:3, 3]
    front = p[:, 2] > 0.05
    u = cam.fx * p[:, 0] / np.where(front, p[:, 2], 1) + cam.cx
    v = cam.fy * p[:, 1] / np.where(front, p[:, 2], 1) + cam.cy
    return u, v, p[:, 2], front


def box_from_points(points, c2w, cam, sensor_depth):
    """(x0, y0, x1, y1, visible_points) or None when too little of the object is in view."""
    u, v, z, front = project(points, c2w, cam)
    inside = front & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
    if inside.sum() < MIN_VISIBLE_POINTS:
        return None
    ui, vi = u[inside].astype(int), v[inside].astype(int)
    if sensor_depth is not None:
        seen = sensor_depth[vi, ui]
        vis = z[inside] <= seen + OCCLUSION_SLACK_M
    else:
        vis = np.ones(inside.sum(), bool)
    if vis.sum() < MIN_VISIBLE_POINTS or vis.sum() < MIN_VISIBLE_SHARE * inside.sum():
        return None
    uu, vv = u[inside][vis], v[inside][vis]
    lo = np.percentile(np.c_[uu, vv], BOX_PERCENTILE, axis=0)
    hi = np.percentile(np.c_[uu, vv], 100 - BOX_PERCENTILE, axis=0)
    return (int(max(0, lo[0])), int(max(0, lo[1])), int(min(cam.width - 1, hi[0])), int(min(cam.height - 1, hi[1])),
            int(vis.sum()))


def box_from_mask(mask_path):
    m = np.asarray(Image.open(mask_path).convert('L')) > 127
    ys, xs = np.nonzero(m)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), int(m.sum())
