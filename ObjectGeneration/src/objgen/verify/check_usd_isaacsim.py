"""Load and render a USD asset in a separate, headless Isaac Sim process."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("usd", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--physics-preview", action="store_true", help="Include a ground collider and gravity in the saved viewing scene")
args = parser.parse_args()
args.usd = args.usd.resolve()
args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=True)

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 800, "height": 800})
try:
    import numpy as np
    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
    from PIL import Image
    from isaacsim.core.utils.viewports import set_camera_view
    from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file

    assert omni.usd.get_context().open_stage(str(args.usd))
    for _ in range(10):
        app.update()
    stage = omni.usd.get_context().get_stage()
    meshes = [UsdGeom.Mesh(p) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    assert meshes
    assert all(len(m.GetPointsAttr().Get()) > 0 for m in meshes)
    assert all(UsdShade.MaterialBindingAPI(m.GetPrim()).ComputeBoundMaterial()[0] for m in meshes)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    bounds = cache.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
    target = bounds.GetMidpoint()
    height = bounds.GetSize()[2]
    if args.physics_preview:
        scene = UsdPhysics.Scene.Define(stage, "/PhysicsScene")
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        scene.CreateGravityMagnitudeAttr(9.81)
        ground = UsdGeom.Cube.Define(stage, "/PreviewGround")
        ground.CreateSizeAttr(1.0)
        ground.CreateDisplayColorAttr([Gf.Vec3f(0.65)])
        ground.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.005))
        ground.AddScaleOp().Set(Gf.Vec3d(max(1.0, height * 4), max(1.0, height * 4), 0.01))
        UsdPhysics.CollisionAPI.Apply(ground.GetPrim()).CreateCollisionEnabledAttr(True)
    light = UsdLux.DomeLight.Define(stage, "/PreviewLight")
    light.CreateIntensityAttr(1000)
    viewport = get_active_viewport()
    assert viewport is not None
    viewport.resolution = (800, 800)
    framing_scale = max(height, float(bounds.GetSize()[0]) * 0.65, float(bounds.GetSize()[1]) * 0.65)
    eye = np.array(target) + np.array([0.35, -1.9, 0.25]) * framing_scale
    set_camera_view(eye, np.array(target), viewport_api=viewport)
    for _ in range(45):
        app.update()
    image_path = args.output / f"{args.usd.stem}_isaacsim.png"
    capture = capture_viewport_to_file(viewport, str(image_path))
    future = app.run_coroutine(capture.wait_for_result(completion_frames=0), run_until_complete=False)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        app.update()
        if future.done():
            future.result()
            try:
                with Image.open(image_path) as image:
                    assert image.size == (800, 800)
                    image.verify()
                break
            except (OSError, SyntaxError):
                pass
        time.sleep(0.01)
    else:
        raise RuntimeError("Isaac Sim screenshot did not finish writing")
    scene_path = args.output / f"{args.usd.stem}_view.usda"
    stage.Export(str(scene_path))
    saved = Usd.Stage.Open(str(scene_path))
    assert saved.GetPrimAtPath(str(viewport.camera_path)).IsA(UsdGeom.Camera)
    report = {"passed": True, "isaacsim": version("isaacsim"), "asset": str(args.usd), "mesh_count": len(meshes), "vertex_count": sum(len(m.GetPointsAttr().Get()) for m in meshes), "screenshot": str(image_path), "view_scene": str(scene_path), "dimensions_m": list(bounds.GetSize()), "physics_preview": args.physics_preview}
    (args.output / "isaacsim_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print("USD_ISAACSIM_VERIFIED", json.dumps(report), flush=True)
finally:
    app.close()
