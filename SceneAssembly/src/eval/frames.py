"""Frames the agent never saw, per object.

Source: ScanNet's 2D instance annotation (one PNG per sampled frame; pixel value = object id + 1)
and the sensor depth next to the colour frames. A frame counts for an object when its annotation
covers at least MIN_COVER of the picture. Every object's first key frame is removed - those are the
frames a batch shares, so any object can be seen there by the agent - and 16 frames per object are
picked with a seed from the object's id, so the set is fixed no matter how the scene is batched.
"""
from __future__ import annotations

import io
import json
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

STEP = 2                 # work at half resolution: 648 x 484
MIN_COVER = 0.01
MAX_FRAMES = 16


def _scene_sources(scene):
    """The scene handoff's manifest, which the asset handoff names and checksums."""
    from inputs import handoff
    man = json.loads((Path(handoff.ASSET_HANDOFF) / 'manifest.json').read_text())
    src = man['source_handoff']
    sm = json.loads((Path(src['path']) / 'manifest.json').read_text())
    return sm


class _LRU:
    def __init__(self, n):
        self.n, self.d = n, OrderedDict()

    def get(self, k):
        if k in self.d:
            self.d.move_to_end(k)
            return self.d[k]
        return None

    def put(self, k, v):
        self.d[k] = v
        if len(self.d) > self.n:
            self.d.popitem(last=False)


class Holdout:
    def __init__(self, scene, seed_from_id=True):
        self.scene = scene
        self.cam = scene.camera
        sm = _scene_sources(scene)
        sources = sm['producer']['sources']
        self.instance_zip = Path(sources['instance_2d'])
        self.frames_dir = Path(sources['camera_to_world'])
        self.frame_ids = {s: fid for s, fid in enumerate(sm['source_frame_ids'])}
        self._masks = _LRU(32)
        self._depth = _LRU(64)
        self._zip = zipfile.ZipFile(self.instance_zip)
        first = {min(f.seq for f in o.key_frames) for o in scene.objects.values()}
        self.excluded = first
        self.frames = self._select(seed_from_id)

    # ------------------------------------------------------------ data

    def instance(self, seq):
        m = self._masks.get(seq)
        if m is not None:
            return m
        name = f"instance/{self.frame_ids[seq]}.png"
        a = np.asarray(Image.open(io.BytesIO(self._zip.read(name))))
        a = a[::STEP, ::STEP]
        self._masks.put(seq, a)
        return a

    def sensor(self, seq):
        """Depth along the optical axis, resampled onto the colour camera at the working resolution."""
        d = self._depth.get(seq)
        if d is not None:
            return d
        z = np.load(self.frames_dir / f'{seq:04d}.npz')
        raw = np.asarray(Image.open(self.frames_dir / f'{seq:04d}_depth.png'), np.float32) / 65535.0 * float(z['maximum_depth'])
        Kd = z['camera_intrinsics'].astype(float)
        c = self.cam
        W, H = c.width // STEP, c.height // STEP
        us, vs = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        x = (us - c.cx / STEP) / (c.fx / STEP)
        y = (vs - c.cy / STEP) / (c.fy / STEP)
        ud = np.clip(np.round(Kd[0, 0] * x + Kd[0, 2]).astype(int), 0, raw.shape[1] - 1)
        vd = np.clip(np.round(Kd[1, 1] * y + Kd[1, 2]).astype(int), 0, raw.shape[0] - 1)
        zd = raw[vd, ud]
        # depth along the optical axis -> distance along the ray: the silhouette test compares
        # Euclidean distances from the camera, the way the pilot's evaluator did
        ray_len = zd * np.sqrt(1 + x ** 2 + y ** 2)
        d = np.where(zd > 0, ray_len, np.inf).astype(np.float32)
        self._depth.put(seq, d)
        return d

    # ------------------------------------------------------------ selection

    def _select(self, seed_from_id):
        # stream the frames once: one mask decode per sequence, coverage for every object at once
        cover = {oid: [] for oid in self.scene.objects}
        for s in sorted(self.frame_ids):
            a = self.instance(s)
            vals, counts = np.unique(a, return_counts=True)
            for v, c in zip(vals, counts):
                oid = f'obj_{int(v) - 1:04d}' if v > 0 else None
                if oid in cover and c / a.size >= MIN_COVER:
                    cover[oid].append(s)
        out = {}
        rng_global = np.random.default_rng(0)
        for oid, o in self.scene.objects.items():
            keys = {f.seq for f in o.key_frames}
            hold = [s for s in cover[oid] if s not in self.excluded and s not in keys]
            rng = np.random.default_rng(int(oid[4:])) if seed_from_id else rng_global
            pick = sorted(rng.choice(hold, size=min(MAX_FRAMES, len(hold)), replace=False).tolist()) if hold else []
            out[oid] = pick
        return out

    def frame_c2w(self, seq):
        """The camera pose of a sampled frame: from its npz, which the scene handoff checks against its own."""
        z = np.load(self.frames_dir / f'{seq:04d}.npz')
        return z['camera_pose'].astype(float)

    def gt(self, oid, seq):
        return self.instance(seq) == int(oid[4:]) + 1

    def summary(self) -> dict:
        n = [len(v) for v in self.frames.values()]
        return {'objects': len(n), 'with_frames': sum(1 for x in n if x), 'frames_per_object': {
            'min': int(min(n)), 'median': float(np.median(n)), 'max': int(max(n))},
            'excluded_sequences': sorted(self.excluded)}
