"""Shared plumbing: where things live, the AMB3R point maps, the floor frame, and the rays.

Three coordinate frames meet in this package and every array says which it is in:

* scan    - ScanNet scan frame, metres, +Z up (what the AMB3R Sim(3) maps into, and the
            frame of every handoff file);
* lift    - the scan frame with ScanNet's axisAlignment applied, only so the room is
            roughly square to the axes; `lift.py` writes its points here;
* floor   - `walls.py` refits the floor plane and turns its normal onto +z: z = 0 is the
            floor and walls are vertical. Everything from `walls.py` on is in this frame.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src import config
from src.geometry.amb3r_mask_source import (AMBR3R_SIZE, DEFAULT_NPZ, DEFAULT_SIM3, _apply_sim3,
                                           _apply_sim3_to_pose, _load_sim3)

# Inputs
SLOTS = config.ARTIFACTS_ROOT / "sources/scene0000_00/gpt_sam31_uniform500_full_v1/manifest.json"
FRAMES = config.ARTIFACTS_ROOT / "frames/scene0000_00"
SCENE = config.SCANNET_ROOT / "scene0000_00"

# Outputs: everything per run under artifacts/structure/<run>/ (segmentation/ first, then
# points, planes, closure); the scene rays do not depend on the run.
OUT_ROOT = config.ARTIFACTS_ROOT / "structure"
RAYS = OUT_ROOT / "scene_rays.npz"

CONF = 0.3                 # AMB3R confidence floor (the lifting default)


def masks_dir(run: str) -> Path:
    """The 2D segmentation of a run (src.structure.segment): one frame_<id>/result.json per frame."""
    return OUT_ROOT / run / "segmentation"


def run_dir(run: str) -> Path:
    path = OUT_ROOT / run
    path.mkdir(parents=True, exist_ok=True)
    return path


def frame_slots() -> dict[int, int]:
    """ScanNet frame id -> index into the AMB3R arrays."""
    return {int(f["source_frame_id"]): int(f["sequence_index"]) for f in json.loads(SLOTS.read_text())["frames"]}


def axis_alignment(scene: Path = SCENE) -> np.ndarray:
    for line in (scene / f"{scene.name}.txt").read_text().splitlines():
        if line.startswith("axisAlignment"):
            return np.array([float(x) for x in line.split("=")[1].split()]).reshape(4, 4)
    raise ValueError(f"no axisAlignment in {scene}")


class PointMaps:
    """The AMB3R point maps, confidences and poses, read once (each npz access reloads the array)."""

    def __init__(self, npz: Path = DEFAULT_NPZ, sim3: Path = DEFAULT_SIM3, scene: Path = SCENE):
        z = np.load(str(npz))
        self.pts, self.conf, self.pose = z["pts"], z["conf"], z["pose"]
        self.transform, _ = _load_sim3(sim3)
        self.A = axis_alignment(scene)
        self.H, self.W = AMBR3R_SIZE[1], AMBR3R_SIZE[0]

    def scan(self, seq: int) -> np.ndarray:
        """(H*W, 3) points of one frame in the scan frame."""
        return _apply_sim3(self.pts[seq].reshape(-1, 3), self.transform).astype(np.float64)

    def lift(self, seq: int) -> np.ndarray:
        """(H*W, 3) points of one frame in the lift frame."""
        p = self.scan(seq)
        return p @ self.A[:3, :3].T + self.A[:3, 3]

    def camera_to_world_scan(self, seq: int) -> np.ndarray:
        return _apply_sim3_to_pose(self.pose[seq], self.transform).astype(np.float64)

    def camera_lift(self, seq: int) -> np.ndarray:
        return (self.A @ self.camera_to_world_scan(seq))[:3, 3]

    def valid(self, seq: int, points: np.ndarray) -> np.ndarray:
        return (self.conf[seq].reshape(-1) >= CONF) & np.isfinite(points).all(1)


def rotation_to_z(n: np.ndarray) -> np.ndarray:
    """Rotation turning n onto +z about a horizontal axis only, so headings are kept."""
    v = np.cross(n, [0, 0, 1.0]); s = np.linalg.norm(v); c = n[2]
    if s < 1e-12:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)


class FloorFrame:
    """lift <-> floor, from the floor plane `walls.py` fitted (stored in walls.json)."""

    def __init__(self, floor: dict):
        self.c = np.asarray(floor["point_in_lift_frame"], dtype=np.float64)
        self.R = rotation_to_z(np.asarray(floor["up_in_lift_frame"], dtype=np.float64))

    def __call__(self, p_lift):
        return (np.asarray(p_lift, dtype=np.float64) - self.c) @ self.R.T


class Rays:
    """All rays of the sequence (see rays.py), in the floor frame, and the ray test on a wall plane.

    A ray runs from a camera to the surface its pixel saw; the space along it is empty.
    On the vertical plane through c with normal n, `grid` counts per cell (along x height)
    the rays that end on the plane (`solid`: something is there) and the rays that pass
    through it (`through`: nothing is there). `drift` is how far one surface's depth may
    disagree between frames: a ray ending within it ends on the plane, one ending further
    behind passed through.
    """

    def __init__(self, to_floor, path: Path = RAYS):
        z = np.load(path)
        self.P = to_floor(z["P"].astype(np.float64))
        self.C = to_floor(z["cams"].astype(np.float64))[z["cam_of"]]
        self.frames = len(z["cams"])

    def grid(self, c, n, lo, hi, cell, drift, z_lo, z_hi):
        u = np.array([-n[1], n[0]])
        dp, dc = (self.P[:, :2] - c) @ n, (self.C[:, :2] - c) @ n
        na, nz = int(np.ceil((hi - lo) / cell)), int(np.ceil((z_hi - z_lo) / cell))

        def count(X):
            ia = np.floor(((X[:, :2] - c) @ u - lo) / cell).astype(int)
            iz = np.floor((X[:, 2] - z_lo) / cell).astype(int)
            ok = (ia >= 0) & (ia < na) & (iz >= 0) & (iz < nz)
            g = np.zeros((na, nz), int)
            np.add.at(g, (ia[ok], iz[ok]), 1)
            return g

        solid = count(self.P[np.abs(dp) < drift])
        thr = (dp * dc < 0) & (np.abs(dp) > drift) & (np.abs(dc) > 0.05)
        t = dc[thr] / (dc[thr] - dp[thr])
        through = count(self.C[thr] + t[:, None] * (self.P[thr] - self.C[thr]))
        return solid, through
