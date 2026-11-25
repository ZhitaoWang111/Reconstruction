"""Write a committed state out as one USD scene, and the key-frame cameras with it.

Each placed object becomes /World/Objects/<oid>, an Xform holding the pose's 4x4
(the exact matrix the offline renderer used), with the asset's physics.usdc
referenced underneath, so the USD is the same scene the agent saw, ready for
Isaac Sim or any USD viewer. Cameras are written as real USD cameras whose
focal length, aperture and aperture offsets reproduce the ScanNet intrinsics,
so a render through them lines up with the photos.

    .venv/bin/python src/legacy/export_usd.py runs/<name>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import handoff
from pose import Pose, matrix
from usdmesh import load_visual

from usdcam import gf_matrix as _gf, set_intrinsics, usd_camera_to_world


def add_camera(stage, path: str, c2w_opencv: np.ndarray, cam: handoff.Camera):
    """A USD camera equivalent to (fx, fy, cx, cy, width, height) at this pose. Square pixels assumed."""
    camera = UsdGeom.Camera.Define(stage, path)
    set_intrinsics(camera, cam.fx, cam.fy, cam.cx, cam.cy, cam.width, cam.height)
    UsdGeom.Xformable(camera).AddTransformOp().Set(_gf(usd_camera_to_world(c2w_opencv)))
    prim = camera.GetPrim()
    prim.CreateAttribute('scannet:fx', Sdf.ValueTypeNames.Double).Set(cam.fx)
    prim.CreateAttribute('scannet:fy', Sdf.ValueTypeNames.Double).Set(cam.fy)
    prim.CreateAttribute('scannet:resolution', Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(cam.width, cam.height))
    return camera


def export(run: Path, out: Path | None = None) -> Path:
    run = Path(run)
    summary = json.loads((run / 'summary.json').read_text())
    state = summary['state']
    hs = handoff.load()
    out = Path(out or run / 'export' / 'scene.usda')
    out.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Xform.Define(stage, '/World/Objects')
    UsdGeom.Xform.Define(stage, '/World/Cameras')
    stage.GetRootLayer().customLayerData = {'scene_id': hs.scene_id, 'run': run.name, 'frame': f'{hs.frame}, metres, +Z up',
                                            'floor_z': hs.floor_z}

    cameras = {}
    for oid, st in state.items():
        ob = hs.objects[oid]
        verts = load_visual(ob.usd)[0]
        M = matrix(verts, Pose(**st['pose']))
        x = UsdGeom.Xform.Define(stage, f'/World/Objects/{oid}')
        x.AddTransformOp().Set(_gf(M))
        p = x.GetPrim()
        p.GetReferences().AddReference(str(ob.usd))
        p.SetCustomDataByKey('label', ob.label)
        p.SetCustomDataByKey('support', st['support'] or '')
        p.SetCustomDataByKey('status', st['status'])
        for f in ob.key_frames:
            cameras.setdefault(f.seq, f)

    for seq, f in sorted(cameras.items()):
        add_camera(stage, f'/World/Cameras/seq_{seq:04d}', f.c2w, hs.camera)

    # a flat floor at the scanned floor height: something to stand on when Play runs physics
    floor = UsdGeom.Mesh.Define(stage, '/World/Floor')
    lo = np.array([1e9, 1e9]); hi = -lo
    for oid, st in state.items():
        lo = np.minimum(lo, [st['pose']['x'], st['pose']['y']]); hi = np.maximum(hi, [st['pose']['x'], st['pose']['y']])
    lo, hi, z = lo - 2.0, hi + 2.0, hs.floor_z
    floor.CreatePointsAttr([Gf.Vec3f(lo[0], lo[1], z), Gf.Vec3f(hi[0], lo[1], z), Gf.Vec3f(hi[0], hi[1], z), Gf.Vec3f(lo[0], hi[1], z)])
    floor.CreateFaceVertexCountsAttr([4])
    floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    floor.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.55, 0.55)])
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    floor.GetPrim().SetCustomDataByKey('note', 'flat stand-in for the scanned floor at floor_z; the offline renderer has no floor')
    physics = UsdPhysics.Scene.Define(stage, '/World/PhysicsScene')
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    physics.CreateGravityMagnitudeAttr(9.81)

    light = UsdLux.DomeLight.Define(stage, '/World/DomeLight')
    light.CreateIntensityAttr(1000.0)
    stage.GetRootLayer().Save()
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('run', type=Path)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    print(export(a.run, a.out))
