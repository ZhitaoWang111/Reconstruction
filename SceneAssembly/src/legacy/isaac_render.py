"""Render an exported scene in Isaac Sim through its key-frame cameras (RTX, headless).

Run with the Isaac Sim environment, not SceneAssembly/.venv:

    cd /home/seer/wzt/robot_sim && source scripts/activate_env.sh
    python /home/seer/wzt/Reconstruction/SceneAssembly/src/legacy/isaac_render.py \
        runs/<name>/export/scene.usda --seqs 63 27 --out runs/<name>/isaac

Writes one PNG per camera at the photo's resolution. Physics is not run: the
renders show the committed poses exactly as the offline renderer placed them.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('scene', type=Path)
ap.add_argument('--seqs', nargs='*', type=int, help='camera seqs to render (default: all)')
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--width', type=int, default=1296)
ap.add_argument('--height', type=int, default=968)
ap.add_argument('--warmup', type=int, default=60)
args = ap.parse_args()

from isaacsim import SimulationApp

# fast_shutdown: with it off, this kit build was seen to hang in Python after every
# extension had shut down. Everything is written before close(), so nothing is lost.
app = SimulationApp({'headless': True, 'width': args.width, 'height': args.height, 'fast_shutdown': True})
try:
    import omni.usd
    from PIL import Image
    from pxr import UsdGeom
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    scene = args.scene.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    assert omni.usd.get_context().open_stage(str(scene)), scene
    for _ in range(args.warmup):
        app.update()
    stage = omni.usd.get_context().get_stage()
    cams = sorted((p for p in stage.Traverse() if p.IsA(UsdGeom.Camera) and str(p.GetPath()).startswith('/World/Cameras/')),
                  key=lambda p: p.GetName())
    if args.seqs:
        want = {f'seq_{s:04d}' for s in args.seqs}
        cams = [p for p in cams if p.GetName() in want]
    assert cams, 'no matching cameras'

    report = {'scene': str(scene), 'resolution': [args.width, args.height], 'renders': {}}
    viewport = get_active_viewport()
    viewport.resolution = (args.width, args.height)

    def capture(path):
        request = capture_viewport_to_file(viewport, str(path))
        future = app.run_coroutine(request.wait_for_result(completion_frames=0), run_until_complete=False)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            app.update()
            if future.done():
                future.result()
                try:
                    with Image.open(path) as im:
                        im.verify()
                    return
                except (OSError, SyntaxError):
                    pass
            time.sleep(0.002)
        raise RuntimeError(f'capture timed out: {path}')

    for cam in cams:
        viewport.set_active_camera(str(cam.GetPath()))
        for _ in range(30):            # let RTX accumulate after the camera jump
            app.update()
        path = args.out / f'{cam.GetName()}.png'
        capture(path)
        report['renders'][cam.GetName()] = str(path)
        print(f'RENDERED {path}', flush=True)
    (args.out / 'isaac_report.json').write_text(json.dumps(report, indent=1))
    print('ISAAC_RENDER_DONE', flush=True)
finally:
    app.close()
