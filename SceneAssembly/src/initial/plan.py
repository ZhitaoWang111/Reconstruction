"""Compute and cache the starting pose of every object, and write it into the state."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from inputs.handoff import ASSET_HANDOFF
from state import Pose, Support

from .icp import fit
from .sam3d import start_pose

SCAN_BOTTOM_PERCENTILE = 2.0


def _handoff_dir() -> Path:
    return ASSET_HANDOFF


def _fingerprint(handoff_dir: Path) -> str:
    return hashlib.sha256((Path(handoff_dir) / 'manifest.json').read_bytes()).hexdigest()[:16]


def initial_pose(obj, assets, structure) -> dict:
    """The starting pose of one object and the record of where it came from."""
    verts = assets.verts(obj.oid)
    scan = obj.points()
    lo, hi = scan.min(0), scan.max(0)
    x, y = float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)
    bottom = float(np.percentile(scan[:, 2], SCAN_BOTTOM_PERCENTILE))
    bottom_scan = bottom
    bottom = max(bottom, structure.floor_z)

    pose, o = start_pose(obj, x, y)
    yaw0 = pose.yaw_deg

    u = pose.upright(verts)
    centre = (u.min(0) + u.max(0)) / 2
    r = fit(u - centre, scan, pose.yaw_deg, centre=(x, y))
    pose = pose.moved(x=r['centre'][0], y=r['centre'][1], yaw_deg=r['yaw_deg'])

    ceiling = structure.floor_z + structure.storey_height
    top = float((verts @ pose.matrix(verts, bottom)[:3, :3].T + pose.matrix(verts, bottom)[:3, 3])[:, 2].max())
    clamped = top > ceiling + 1e-6
    if clamped:
        bottom -= top - ceiling

    return {
        'object': obj.oid, 'label': obj.label,
        'pose': pose.to_dict(), 'bottom_z': round(bottom, 4),
        'height_m': round(float(u.ptp(0)[2]), 4),
        # where it came from
        'sam3d': {'up': o['sam3d_up'], 'deviation_deg': round(o['deviation_deg'], 1),
                  'flipped': o['flipped'], 'yaw_confirmed': not o['flipped'],
                  'tilt_of_sam3d_deg': round(o['deviation_deg'], 1), 'starting_tilt_deg': 0.0},
        'scan': {'x': round(x, 4), 'y': round(y, 4), 'bottom_z': round(bottom_scan, 4),
                 'bottom_is_below_floor': bool(bottom_scan < structure.floor_z)},
        'registration': {'yaw0_deg': round(yaw0, 2),
                         'yaw_deg': round(r['yaw_deg'], 2),
                         'shift_m': round(float(np.hypot(r['centre'][0] - x, r['centre'][1] - y)), 4),
                         'residual_m': round(r['residual_m'], 4),
                         'residual_start_m': round(r['residual_start_m'], 4),
                         'inlier': round(r['inlier'], 3), 'improved': not r['diverged']},
        'support': None, 'clamped_to_ceiling': bool(clamped),
        'note': '未声明支撑，高度取自扫描底面' + ('（超出层高，已下移）' if clamped else ''),
    }


class InitialPoses:
    """All objects' starting poses, cached in <run>/initial_poses.json."""

    def __init__(self, scene, structure, run_dir: Path, assets=None):
        self.scene = scene
        self.structure = structure
        self.out = Path(run_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.path = self.out / 'initial_poses.json'
        self.fingerprint = _fingerprint(_handoff_dir())
        if assets is None:
            from state.assets import Assets
            assets = Assets(scene)
        self.assets = assets
        self.data = {}

    def compute_all(self, force=False) -> dict:
        if not force and self.path.exists():
            cached = json.loads(self.path.read_text())
            if cached.get('fingerprint') == self.fingerprint and cached.get('scene_id') == self.scene.scene_id:
                self.data = cached['objects']
                return self.data
        self.data = {oid: initial_pose(o, self.assets, self.structure) for oid, o in self.scene.objects.items()}
        self.path.write_text(json.dumps({'scene_id': self.scene.scene_id, 'fingerprint': self.fingerprint,
                                         'objects': self.data}, indent=1, ensure_ascii=False))
        return self.data

    def __getitem__(self, oid) -> dict:
        if not self.data:
            self.compute_all()
        return self.data[oid]

    def card(self, oid) -> dict:
        """The part the agent sees in the object card (M5)."""
        d = self[oid]
        return {'pose': d['pose'], 'support': None, 'note': d['note'],
                'orientation_confirmed': d['sam3d']['yaw_confirmed'],
                'flipped_from_sam3d': d['sam3d']['flipped']}

    def apply(self, state, oid, batch=None) -> dict:
        """Write it into the state as this object's first version."""
        d = self[oid]
        pose = Pose.from_dict(d['pose'])
        support = Support('height', height=float(d['bottom_z'] - self.structure.floor_z))
        return state.init_object(oid, pose, support, batch=batch, note='M3 initial pose')

    def summary(self) -> dict:
        d = self.data or self.compute_all()
        flipped = [o for o, v in d.items() if v['sam3d']['flipped']]
        side = [o for o, v in d.items() if v['sam3d']['up'] not in ('+Z', '-Z')]
        clamped = [o for o, v in d.items() if v['clamped_to_ceiling']]
        resid = np.array([v['registration']['residual_m'] for v in d.values()])
        return {'objects': len(d), 'flipped_from_sam3d': flipped, 'side_up': side,
                'clamped_to_ceiling': clamped,
                'registration_residual_cm': {'median': round(float(np.median(resid)) * 100, 1),
                                             'p90': round(float(np.percentile(resid, 90)) * 100, 1)},
                'deviated_from_start': [o for o, v in d.items() if v['registration']['shift_m'] > 0.05]}
