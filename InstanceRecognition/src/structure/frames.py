"""Shared plumbing: where things live, the AMB3R point maps in the delivery frame, and the rays.

One coordinate frame runs through this package: `amb3r_floor` (docs/amb3r-floor-frame.md,
src.geometry.floor_frame) - AMB3R's reconstruction scaled and stood upright, z = 0 on the
floor, metres. Every point, camera, plane and wall from `lift.py` on is in it, and the
handoff is written in it too, so there is no conversion at the end.

Neither ScanNet's axisAlignment nor its GT poses are read. The frame file says where its
scale came from.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src import config
from src.geometry.amb3r_mask_source import AMBR3R_SIZE, DEFAULT_NPZ
from src.geometry.floor_frame import Frame, frame_path

# Inputs
SLOTS = config.ARTIFACTS_ROOT / "sources/scene0000_00/gpt_sam31_uniform500_full_v1/manifest.json"
FRAMES = config.ARTIFACTS_ROOT / "frames/scene0000_00"
SCENE = config.SCANNET_ROOT / "scene0000_00"     # camera intrinsics (the handoff's `camera` block); GT pages

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


class PointMaps:
    """The AMB3R point maps, confidences and poses, read once, delivered in the amb3r_floor frame.

    `frame=None` gives the native arrays only (what the frame itself is fitted from).
    """

    def __init__(self, npz: Path = DEFAULT_NPZ, frame: Frame | None | str = "default"):
        z = np.load(str(npz))
        self.pts, self.conf, self.pose = z["pts"], z["conf"], z["pose"]
        self.frame = Frame.load() if frame == "default" else frame
        self.H, self.W = AMBR3R_SIZE[1], AMBR3R_SIZE[0]

    def native(self, seq: int) -> np.ndarray:
        """(H*W, 3) points of one frame in AMB3R's own frame and units."""
        return self.pts[seq].reshape(-1, 3).astype(np.float64)

    def points(self, seq: int) -> np.ndarray:
        """(H*W, 3) points of one frame in the amb3r_floor frame."""
        return self.frame.points(self.native(seq))

    def camera_to_world(self, seq: int) -> np.ndarray:
        """4x4 camera-to-world (OpenCV axes) in the amb3r_floor frame."""
        return self.frame.pose(self.pose[seq])

    def camera(self, seq: int) -> np.ndarray:
        return self.camera_to_world(seq)[:3, 3]

    def valid(self, seq: int, points: np.ndarray) -> np.ndarray:
        return (self.conf[seq].reshape(-1) >= CONF) & np.isfinite(points).all(1)


def axis_alignment_for_eval(scene: Path = SCENE) -> np.ndarray:
    """ScanNet's axisAlignment (scan -> NYU axis-aligned). For comparing with GT pages only;
    nothing in the wall pipeline reads it."""
    for line in (scene / f"{scene.name}.txt").read_text().splitlines():
        if line.startswith("axisAlignment"):
            return np.array([float(x) for x in line.split("=")[1].split()]).reshape(4, 4)
    raise ValueError(f"no axisAlignment in {scene}")


def rotation_to_z(n: np.ndarray) -> np.ndarray:
    """Rotation turning n onto +z about a horizontal axis only, so headings are kept."""
    v = np.cross(n, [0, 0, 1.0]); s = np.linalg.norm(v); c = n[2]
    if s < 1e-12:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)


class Rays:
    """All rays of the sequence (see rays.py), in the amb3r_floor frame, and the ray test on a wall plane.

    A ray runs from a camera to the surface its pixel saw; the space along it is empty.
    On the vertical plane through c with normal n, `grid` counts per cell (along x height)
    the rays that end on the plane (`solid`: something is there) and the rays that pass
    through it (`through`: nothing is there). `drift` is how far one surface's depth may
    disagree between frames: a ray ending within it ends on the plane, one ending further
    behind passed through.
    """

    def __init__(self, path: Path = RAYS, frame: Frame | None = None):
        z = np.load(path)
        frame = frame or Frame.load()
        if "native_to_frame" not in z.files or not np.allclose(z["native_to_frame"], frame.M, atol=1e-9):
            raise SystemExit(f"{path} was not written in the current amb3r_floor frame; rerun python -m src.structure.rays")
        self.P = z["P"].astype(np.float64)
        self.C = z["cams"].astype(np.float64)[z["cam_of"]]
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
