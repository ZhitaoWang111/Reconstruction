"""Silhouette IoU on the holdout frames (the pilot's metric, copied from src/legacy/evaluate.py).

A rendered pixel counts only where the sensor saw nothing closer: a nightstand hidden by the bed
is not penalised for being hidden. Both sides are also limited to pixels the sensor saw at all.
"""
from __future__ import annotations

import numpy as np
import trimesh

from observe.render import camera_rays

from .frames import STEP

OCCLUSION_SLACK_M = 0.10


class Scorer:
    def __init__(self, scene, holdout, assets):
        self.scene = scene
        self.h = holdout
        self.assets = assets
        self.cam = scene.camera
        self._mesh = {}

    def _posed(self, oid, matrix):
        key = oid
        if key not in self._mesh:
            v, f = self.assets.verts_faces(oid)
            m = trimesh.Trimesh(v, f, process=False)
            self._mesh[key] = (m, f)
        m = self._mesh[key][0].copy()
        m.vertices = np.asarray(m.vertices, float) @ np.asarray(matrix)[:3, :3].T + np.asarray(matrix)[:3, 3]
        return m

    def silhouette(self, mesh, seq):
        c = self.cam
        W, H = c.width // STEP, c.height // STEP
        o, d = camera_rays(self.h.frame_c2w(seq), c.fx / STEP, c.fy / STEP, c.cx / STEP, c.cy / STEP, W, H)
        sensor = self.h.sensor(seq).reshape(-1)
        loc, ray, _ = mesh.ray.intersects_location(o, d, multiple_hits=False)
        m = np.zeros(W * H, bool)
        if len(ray):
            dist = np.linalg.norm(loc - o[ray], axis=1)
            m[ray[dist <= sensor[ray] + OCCLUSION_SLACK_M]] = True
        return m.reshape(H, W)

    def score(self, oid, matrix) -> dict:
        frames = self.h.frames[oid]
        if not frames:
            return {'iou': None, 'frames': 0, 'per_frame': []}
        mesh = self._posed(oid, matrix)
        ious = []
        for seq in frames:
            sensor = self.h.sensor(seq)
            m = self.silhouette(mesh, seq)
            ok = np.isfinite(sensor)
            a, b = m & ok, self.h.gt(oid, seq) & ok
            u = (a | b).sum()
            ious.append(float((a & b).sum() / u) if u else 0.0)
        return {'iou': round(float(np.mean(ious)), 4), 'frames': len(ious), 'per_frame': [round(x, 3) for x in ious]}
