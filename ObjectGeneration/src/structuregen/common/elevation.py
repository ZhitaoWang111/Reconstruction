"""Wall-plane geometry for the key frames: points on a segment's room-side face and their pixels.

A point on a wall's room-side face is (along, z): along from the wall's start, z above the floor.
With a key frame's camera (OpenCV, camera_to_world, as the handoff gives it) it maps to a pixel.
"""
from __future__ import annotations

import numpy as np


def wall_point(wall: dict, along, z, back: float = 0.0):
    s = np.array(wall["start_floor"][:2], float)
    e = np.array(wall["end_floor"][:2], float)
    u = (e - s) / np.linalg.norm(e - s)
    n = np.array(wall["normal_to_back"][:2], float)
    along = np.asarray(along, float)
    z = np.asarray(z, float)
    return np.stack([s[0] + along * u[0] + back * n[0], s[1] + along * u[1] + back * n[1], z, np.ones_like(z)], -1)


def project(camera: dict, c2w, points):
    pc = points @ np.linalg.inv(np.asarray(c2w, float)).T
    with np.errstate(divide="ignore", invalid="ignore"):
        u = camera["fx"] * pc[..., 0] / pc[..., 2] + camera["cx"]
        v = camera["fy"] * pc[..., 1] / pc[..., 2] + camera["cy"]
    return u, v, pc[..., 2]
