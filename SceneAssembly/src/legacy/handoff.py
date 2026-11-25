"""Load the two handoffs scene assembly depends on, joined by object_id.

The entry point is the ObjectGeneration assembly handoff. Its manifest names the
InstanceRecognition scene handoff it was built against and records that
handoff's sha256, so a regenerated scene is never silently paired with assets
made for an older one. Only the fields assembly uses are kept.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

ASSET_HANDOFF = Path(os.environ.get(
    'ASSET_HANDOFF',
    Path(__file__).resolve().parents[3] / 'ObjectGeneration/output/scene0000_00/gt/assembly_handoff'))
ROLES = ('primary', 'secondary', 'third')


@dataclass(frozen=True)
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True, eq=False)
class KeyFrame:
    seq: int
    role: str
    rgb: Path
    mask: Path
    visible_pixels: int
    c2w: np.ndarray            # 4x4 camera -> scan frame, OpenCV axes


@dataclass(frozen=True, eq=False)
class Object:
    oid: str
    label: str
    generic: bool              # label names no specific thing (ScanNet "object")
    cloud: Path
    extent: np.ndarray         # scan bounds max - min: what was scanned, not the object's size
    key_frames: tuple          # KeyFrame, primary first
    usd: Path
    dims: np.ndarray           # asset bounding box in its own frame, metres
    height_m: float
    scale_factor: float
    size_source: str
    body_type: str
    collider: str
    pose_frame: KeyFrame       # the key frame SAM3D estimated the pose in
    pose_rotation: np.ndarray  # USD asset-local -> pose_frame camera; scale/translation not metric

    @property
    def primary(self) -> KeyFrame:
        return self.key_frames[0]

    @property
    def rotation_world(self) -> np.ndarray:
        """USD asset-local -> scan frame. Rotation only: SAM3D gives no metric position."""
        return self.pose_frame.c2w[:3, :3] @ self.pose_rotation

    def points(self) -> np.ndarray:
        return np.asarray(trimesh.load(self.cloud).vertices, float)


@dataclass(frozen=True, eq=False)
class Scene:
    scene_id: str
    camera: Camera
    floor_z: float
    frame: str                 # the scene handoff's coordinate_frame (scannet_scan for handoff/v2, amb3r_floor from v3)
    frame_ids: tuple           # seq -> source video frame id
    instance_2d: Path | None   # ScanNet 2-D instance zip; only the GT producer has one
    objects: dict              # oid -> Object, generated assets only


def _read(path: Path, sha256: str | None = None):
    raw = path.read_bytes()
    if sha256 is not None and hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError(f'{path} changed after the assets were handed off; '
                         'rebuild the assembly handoff before assembling')
    return json.loads(raw)


def load(asset_dir: Path = ASSET_HANDOFF) -> Scene:
    asset_dir = Path(asset_dir)
    am = _read(asset_dir / 'manifest.json')
    src = am['source_handoff']
    scene_dir = Path(src['path'])
    sm = _read(scene_dir / 'manifest.json', src['manifest_sha256'])
    if sm['scene_id'] != am['scene_id']:
        raise ValueError(f"scene_id differs: {sm['scene_id']} vs {am['scene_id']}")
    scene_objects = {r['object_id']: r for r in _read(scene_dir / 'objects.json', src['objects_sha256'])}

    objects = {}
    for a in _read(asset_dir / 'objects.json'):
        if a['status'] != 'success':
            continue
        oid = a['object_id']
        s = scene_objects.get(oid)
        if s is None:
            raise ValueError(f'{oid} has an asset but is not in {scene_dir}')
        frames = sorted((KeyFrame(f['seq'], f['role'], scene_dir / f['rgb'], scene_dir / f['mask'],
                                  f['visible_pixels'], np.array(f['camera_to_world'], float))
                         for f in s['key_frames']), key=lambda f: ROLES.index(f.role))
        size, physics = a['size'], a['physics']
        objects[oid] = Object(
            oid=oid, label=s['label'], generic=s['label_quality'] == 'generic',
            cloud=scene_dir / s['cloud'],
            extent=np.subtract(s['bounds']['maximum'], s['bounds']['minimum']),
            key_frames=tuple(frames), usd=asset_dir / a['usd'],
            dims=np.array(size['dimensions_m'], float), height_m=size['height_m'],
            scale_factor=size['scale_factor'], size_source=size['source'],
            body_type=physics['body_type'], collider=physics['collision_approximation'],
            pose_frame=next(f for f in frames if f.seq == a['generation']['seq']),
            pose_rotation=np.array(a['pose']['rotation'], float),
        )

    c = sm['camera']
    instance_2d = sm['producer'].get('sources', {}).get('instance_2d')
    return Scene(scene_id=sm['scene_id'],
                 camera=Camera(c['fx'], c['fy'], c['cx'], c['cy'], c['width'], c['height']),
                 floor_z=sm['floor_z'], frame=sm.get('coordinate_frame', 'scannet_scan'),
                 frame_ids=tuple(sm['source_frame_ids']),
                 instance_2d=Path(instance_2d) if instance_2d else None,
                 objects=dict(sorted(objects.items())))
