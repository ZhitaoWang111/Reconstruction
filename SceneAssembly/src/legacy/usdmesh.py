"""Read the handoff visual.usdc into numpy arrays, preserving the face-varying displayColor."""
import json
import numpy as np
from pxr import Usd, UsdGeom


def load_visual(usdc):
    st = Usd.Stage.Open(str(usdc))
    prim = st.GetDefaultPrim()
    mprim = None
    for p in Usd.PrimRange(prim):
        if p.IsA(UsdGeom.Mesh):
            mprim = p
            break
    m = UsdGeom.Mesh(mprim)
    pts = np.array(m.GetPointsAttr().Get(), dtype=np.float64)
    idx = np.array(m.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    cnt = np.array(m.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    assert (cnt == 3).all(), np.unique(cnt)
    faces = idx.reshape(-1, 3)
    varname, interp, col = None, None, None
    for pv in UsdGeom.PrimvarsAPI(mprim).GetPrimvars():
        if pv.GetName() in ('primvars:displayColor', 'displayColor'):
            varname, interp, col = 'displayColor', pv.GetInterpolation(), np.array(pv.Get(), dtype=np.float64)
    return pts, faces, col, interp


def srgb_faces(col, interp, nfaces, faces=None):
    """Linear displayColor -> sRGB, one colour per face."""
    if col is None:
        return np.tile([0.72, 0.70, 0.66], (nfaces, 1))
    c = np.clip(col[:, :3], 0, None)
    c = np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)
    if interp == 'faceVarying':
        return c.reshape(nfaces, 3, 3).mean(1)
    if interp == 'uniform':
        return c[:nfaces]
    if interp == 'vertex' and faces is not None:
        return c[faces].mean(1)
    return np.tile(c.mean(0), (nfaces, 1))
