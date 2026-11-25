"""Score placed objects on frames the agent never sees.

For each object, the holdout frames are every sampled frame (of the 500) where
its ScanNet 2-D instance mask covers at least 1% of the image, minus its key
frames. On each frame the asset is rendered under the frame's real camera and
compared with the photo mask. A rendered pixel counts only where the asset is
not behind a real surface: sensor depth says what was actually in front, so a
nightstand hidden by the bed is not penalised for being hidden.

The score is the mean IoU over a fixed, seeded subset of holdout frames. The
same subset is used for every method, so the numbers compare.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from pose import Pose
from render import camera_rays

FRAMES = Path('/home/seer/wzt/Reconstruction/Video2PointCloud/data/scannet/scene0000_00_uniform500/input')
STEP = 2              # render at half resolution
OCCLUSION_SLACK = 0.10  # m: sensor surface this much in front of the asset hides it
MIN_COVER = 0.01
MAX_FRAMES = 16


class Evaluator:
    def __init__(self, scene, seed=0):
        self.s = scene
        hs = scene.handoff
        self.cam = hs.camera
        self.frames = {}
        with zipfile.ZipFile(hs.instance_2d) as z:
            masks = {s: np.array(Image.open(io.BytesIO(z.read(f'instance/{fid}.png'))))[::STEP, ::STEP]
                     for s, fid in enumerate(hs.frame_ids)}
        rng = np.random.default_rng(seed)
        for oid in scene.oids:
            value = int(oid[4:]) + 1
            keys = {f.seq for f in scene.assets[oid].ob.key_frames}
            hold = [s for s, m in masks.items() if s not in keys and (m == value).mean() >= MIN_COVER]
            pick = sorted(rng.choice(hold, size=min(MAX_FRAMES, len(hold)), replace=False).tolist()) if hold else []
            self.frames[oid] = [(s, masks[s] == value) for s in pick]
        self._depth = {}

    def _sensor(self, seq):
        """Sensor depth resampled to the colour camera at render resolution, and the C2W pose."""
        if seq not in self._depth:
            z = np.load(FRAMES / f'{seq:04d}.npz')
            raw = np.asarray(Image.open(FRAMES / f'{seq:04d}_depth.png'), np.float32) / 65535.0 * float(z['maximum_depth'])
            Kd = z['camera_intrinsics'].astype(float)
            c = self.cam
            W, H = c.width // STEP, c.height // STEP
            us, vs = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
            # colour pixel ray -> depth pixel (same optical centre, different intrinsics)
            x = (us - c.cx / STEP) / (c.fx / STEP)
            y = (vs - c.cy / STEP) / (c.fy / STEP)
            ud = np.clip(np.round(Kd[0, 0] * x + Kd[0, 2]).astype(int), 0, raw.shape[1] - 1)
            vd = np.clip(np.round(Kd[1, 1] * y + Kd[1, 2]).astype(int), 0, raw.shape[0] - 1)
            zd = raw[vd, ud]
            ray_len = zd * np.sqrt(1 + x ** 2 + y ** 2)        # depth along the optical axis -> along the ray
            self._depth[seq] = (np.where(zd > 0, ray_len, np.inf), z['camera_pose'].astype(float))
        return self._depth[seq]

    def silhouette(self, mesh, seq):
        c = self.cam
        W, H = c.width // STEP, c.height // STEP
        sensor, C2W = self._sensor(seq)
        _, o, d = camera_rays(C2W, c.fx / STEP, c.fy / STEP, c.cx / STEP, c.cy / STEP, W, H)
        loc, ray, _ = mesh.ray.intersects_location(o, d, multiple_hits=False)
        m = np.zeros(W * H, bool)
        if len(ray):
            dist = np.linalg.norm(loc - o[ray], axis=1)
            visible = dist <= sensor.reshape(-1)[ray] + OCCLUSION_SLACK
            m[ray[visible]] = True
        return m.reshape(H, W)

    def score(self, oid, pose: Pose) -> dict:
        frames = self.frames[oid]
        if not frames:
            return {'iou': None, 'frames': 0}
        mesh = self.s.assets[oid].mesh(pose)
        ious = []
        for seq, gt in frames:
            sensor, _ = self._sensor(seq)
            m = self.silhouette(mesh, seq)
            # where the sensor saw nothing, neither side can be checked
            ok = np.isfinite(sensor)
            a, b = m & ok, gt & ok
            u = (a | b).sum()
            ious.append(float((a & b).sum() / u) if u else 0.0)
        return {'iou': round(float(np.mean(ious)), 4), 'frames': len(frames), 'per_frame': [round(x, 3) for x in ious]}


def baselines(scene, ev: Evaluator) -> dict:
    """Each geometric baseline scored per object: B0 scan bbox + yaw 0, B1 SAM3D, B2 best-residual registration."""
    out = {}
    for oid in scene.oids:
        a = scene.assets[oid]
        lo, hi = a.scan.min(0), a.scan.max(0)
        cands = scene.candidates(oid)
        icp = [c for c in cands if c['name'].startswith('icp') and not c['diverged']] or [c for c in cands if c['name'].startswith('icp')]
        b2 = min(icp, key=lambda c: c['residual_cm'])
        poses = {'B0_bbox_yaw0': Pose((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, float(lo[2]), 0.0, '+Z'),
                 'B1_sam3d': next(c['pose'] for c in cands if c['name'] == 'sam3d'),
                 'B2_best_residual': b2['pose']}
        # oracle: the best of the five candidates on the holdout set itself (an upper bound for picking among them)
        scores = {k: ev.score(oid, p)['iou'] for k, p in poses.items()}
        cand_scores = {c['name']: ev.score(oid, c['pose'])['iou'] for c in cands}
        scores['oracle_best_candidate'] = max(cand_scores.values())
        scores['candidates'] = cand_scores
        out[oid] = scores
    return out


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=1, ensure_ascii=False))
