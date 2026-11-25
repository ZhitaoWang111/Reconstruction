"""Asset meshes in their own frame, loaded once.

The visual mesh is what the agent sees, so heights and wall contact are computed
on it (D06 M2-3), not on the collision hull.
"""
from __future__ import annotations

import numpy as np
import trimesh

from inputs.usdmesh import load_visual


class Assets:
    def __init__(self, scene):
        self.scene = scene                 # inputs.handoff.Scene
        self._mesh = {}

    def verts_faces(self, oid):
        if oid not in self._mesh:
            v, f, _, _ = load_visual(self.scene.objects[oid].usd)
            self._mesh[oid] = (np.asarray(v, float), np.asarray(f, np.int64))
        return self._mesh[oid]

    def verts(self, oid):
        return self.verts_faces(oid)[0]

    def usd(self, oid) -> str:
        return str(self.scene.objects[oid].usd)

    def posed(self, oid, M) -> trimesh.Trimesh:
        v, f = self.verts_faces(oid)
        return trimesh.Trimesh(v @ M[:3, :3].T + M[:3, 3], f, process=False)

    def sample(self, oid, n=4000) -> np.ndarray:
        """Up to n vertices, evenly strided, for the cheap geometric queries."""
        v = self.verts(oid)
        return v[::max(1, len(v) // n)]
