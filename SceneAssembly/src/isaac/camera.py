"""Pinhole intrinsics and OpenCV poses as USD cameras. numpy + pxr only, so both the
harness (.venv) and the Isaac service (Isaac's own Python) can import it.

A USD camera looks down -Z with +Y up; the handoffs' camera_to_world is OpenCV
(+Z forward, +Y down). Focal length, apertures and aperture offsets are chosen so
a render through the camera lines up pixel for pixel with the photo.
"""
from __future__ import annotations

import numpy as np
from pxr import Gf, UsdGeom

OPENCV_TO_USD = np.diag([1.0, -1.0, -1.0])
APERTURE_MM = 36.0   # horizontal film width; focal length is derived from it so fx is exact


def gf_matrix(m) -> Gf.Matrix4d:
    """numpy column-vector 4x4 -> Gf.Matrix4d, which uses row vectors (hence the transpose)."""
    return Gf.Matrix4d(*[float(x) for x in np.asarray(m, float).T.reshape(-1)])


def np_matrix(g: Gf.Matrix4d) -> np.ndarray:
    """Inverse of gf_matrix."""
    return np.array(g, float).T


def usd_camera_to_world(c2w_opencv) -> np.ndarray:
    c2w = np.array(c2w_opencv, float).copy()
    c2w[:3, :3] = c2w[:3, :3] @ OPENCV_TO_USD
    return c2w


def set_intrinsics(camera: UsdGeom.Camera, fx, fy, cx, cy, width, height, clip=(0.05, 50.0)):
    """Square pixels assumed (fx == fy to well under a pixel for ScanNet)."""
    pixel_mm = APERTURE_MM / width
    camera.CreateHorizontalApertureAttr(APERTURE_MM)
    camera.CreateVerticalApertureAttr(pixel_mm * height)
    camera.CreateFocalLengthAttr(fx * pixel_mm)
    # principal point offset from the image centre, in film millimetres; image +v is down, film +y is up
    camera.CreateHorizontalApertureOffsetAttr((cx - width / 2) * pixel_mm)
    camera.CreateVerticalApertureOffsetAttr(-(cy - height / 2) * pixel_mm)
    camera.CreateClippingRangeAttr(Gf.Vec2f(*clip))
    camera.CreateProjectionAttr(UsdGeom.Tokens.perspective)
