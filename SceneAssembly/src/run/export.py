"""Write the assembled scene as one USD: the room (structure.usdc), every object with its pose,
the key-frame cameras, a physics scene and a light.

Adapted from the pilot's export (`src/legacy/export_usd.py`, copied, not imported): the pose matrix
now comes straight from the state; the flat stand-in floor is gone because the structure brings the
real walls and floors.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

from isaac.camera import gf_matrix, set_intrinsics, usd_camera_to_world


def add_camera(stage, path: str, c2w_opencv, cam):
    camera = UsdGeom.Camera.Define(stage, path)
    set_intrinsics(camera, cam.fx, cam.fy, cam.cx, cam.cy, cam.width, cam.height)
    UsdGeom.Xformable(camera).AddTransformOp().Set(gf_matrix(usd_camera_to_world(c2w_opencv)))
    prim = camera.GetPrim()
    prim.CreateAttribute('scannet:fx', Sdf.ValueTypeNames.Double).Set(cam.fx)
    prim.CreateAttribute('scannet:fy', Sdf.ValueTypeNames.Double).Set(cam.fy)
    prim.CreateAttribute('scannet:resolution', Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(cam.width, cam.height))
    return camera


def export_scene(state, scene, structure, out: Path, run_name: str = '', include_untouched=False) -> Path:
    """One USD of the state as it is. Untouched objects (never processed) are left out unless asked."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(world.GetPrim())
    stage.GetRootLayer().customLayerData = {
        'scene_id': scene.scene_id, 'run': run_name, 'frame': f'{structure.frame}, metres, +Z up',
        'floor_z': float(structure.floor_z), 'state_version': int(state.version)}

    room = stage.DefinePrim('/World/Structure', 'Xform')
    room.GetReferences().AddReference(str(Path(structure.usd).resolve()))

    UsdGeom.Xform.Define(stage, '/World/Objects')
    counts = {}
    cameras = {}
    for oid, st in sorted(state.objects.items()):
        if st.status == 'untouched' and not include_untouched:
            continue
        ob = scene.objects[oid]
        x = UsdGeom.Xform.Define(stage, f'/World/Objects/{oid}')
        x.AddTransformOp().Set(gf_matrix(st.matrix))
        p = x.GetPrim()
        p.GetReferences().AddReference(str(Path(ob.usd).resolve()))
        p.SetCustomDataByKey('label', ob.label)
        p.SetCustomDataByKey('status', st.status)
        p.SetCustomDataByKey('support', str(st.support.to_dict()))
        p.SetCustomDataByKey('pose', str(st.pose.to_dict()))
        if st.status == 'failed':
            p.SetCustomDataByKey('note', 'marked failed during assembly; kept at its last accepted pose')
        counts[st.status] = counts.get(st.status, 0) + 1
        for f in ob.key_frames:
            cameras.setdefault(f.seq, f)

    UsdGeom.Xform.Define(stage, '/World/Cameras')
    for seq, f in sorted(cameras.items()):
        add_camera(stage, f'/World/Cameras/seq_{seq:04d}', f.c2w, scene.camera)

    physics = UsdPhysics.Scene.Define(stage, '/World/PhysicsScene')
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    physics.CreateGravityMagnitudeAttr(9.81)
    UsdLux.DomeLight.Define(stage, '/World/DomeLight').CreateIntensityAttr(1000.0)
    layer = stage.GetRootLayer()
    data = dict(layer.customLayerData)
    data['objects'] = {k: int(v) for k, v in counts.items()}
    layer.customLayerData = data
    layer.Save()
    return out
