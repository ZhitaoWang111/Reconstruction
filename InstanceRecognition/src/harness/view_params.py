"""Camera viewpoint parameters for observation comparison.

Reads the per-frame ``camera_to_world`` (4×4, scan frame) that the lifted
source writes into each frame npz, and derives the direction and distance each
observation was made from. The Sim(3) fit already ran when the source was
built (see ``amb3r_mask_source.py``), so the harness never re-derives it here.

A source whose frame npz files do not carry ``camera_to_world`` (a run predating
the field, such as ``scene_amb3r_v1``) yields ``None`` from :func:`position`;
callers treat that as "viewpoint unavailable" and omit the fields.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class ViewParams:
    """Lazy, cached access to camera positions for a harness run.

    The positions array is loaded once on first use and indexed by sequence
    index. Observations whose ``seq`` is outside the loaded range (a source
    without the field, or a seq past the last frame) get ``None`` back, and the
    caller omits the viewpoint fields for that one.
    """

    def __init__(self, frames_dir: Path):
        self._frames_dir = Path(frames_dir)
        self._positions: np.ndarray | None = None
        self._loaded = False
        self._max_seq = -1

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        # Scan the frames directory once. A missing file just means that seq is
        # unavailable; a directory with no npz at all means the source predates
        # camera_to_world, so the whole feature stays off.
        positions = []
        seq = 0
        while True:
            path = self._frames_dir / f"{seq:04d}.npz"
            if not path.is_file():
                break
            try:
                with np.load(path) as z:
                    if "camera_to_world" not in z.files:
                        # This source does not carry the field at all; stop
                        # loading and leave positions empty so every lookup
                        # returns None rather than a half-filled array.
                        positions = []
                        return
                    positions.append(z["camera_to_world"][:3, 3].astype(np.float64))
            except Exception:
                positions = []
                return
            seq += 1
        self._max_seq = len(positions) - 1
        self._positions = np.asarray(positions, dtype=np.float64) if positions else None

    def position(self, seq: int) -> np.ndarray | None:
        """Camera centre for sequence index ``seq``, or None if unavailable."""
        self._ensure_loaded()
        if self._positions is None or seq < 0 or seq > self._max_seq:
            return None
        return self._positions[seq]

    def view_direction(self, seq: int, centroid) -> list[float] | None:
        """Unit vector from object centroid toward the camera, or None."""
        cam = self.position(seq)
        if cam is None or centroid is None:
            return None
        c = np.asarray(centroid, dtype=np.float64)
        direction = cam - c
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            return None
        return [round(float(v), 3) for v in (direction / norm)]

    def camera_distance(self, seq: int, centroid) -> float | None:
        """Distance from camera to centroid in metres, or None."""
        cam = self.position(seq)
        if cam is None or centroid is None:
            return None
        return round(float(np.linalg.norm(cam - np.asarray(centroid, dtype=np.float64))), 2)
