"""Stage 1: build the white model of a walls handoff, one closed body per segment.

    PYTHONPATH=src .venv-sam3d/bin/python -m structuregen.whitebox.build \\
        [--handoff <walls_harness dir>] [--height 2.8] [--opening-inset 0.15] [--out <dir>]

Writes ``plan.json`` (pure geometry, checked here), then runs Blender headless on
``blender_build.py`` and checks what it built against the plan: every segment closed,
its measured volume equal to footprint x height minus its openings. A timestamped run
under ``output/<scene>/structuregen/`` holds ``walls.blend``, ``walls_base.usda``,
``build_report.json``, ``plan.json`` and ``views/`` (the model seen from every key-frame
camera, at the photo's size, transparent background)), next to a ``latest`` symlink.

Floors are built in Blender after the walls (blender_build.build_floors) and checked here: every
floor closed, its volume its top area x thickness, and the rooms' area plus every interior wall's
footprint equal to the handoff's inner outline. ``structure_base.usda`` holds /Structure/Walls and
/Structure/Floors; the floors are added to plan.json under ``floors``.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

from .. import config
from ..common import handoff as handoff_io
from ..common.usd_layout import regroup
from . import segments

SCRIPT = Path(__file__).with_name("blender_build.py")
VIEWS = Path(__file__).with_name("blender_views.py")
TOLERANCE_M3 = 1e-4


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--handoff", type=Path, default=config.DEFAULT_HANDOFF)
    parser.add_argument("--height", type=float, default=None,
                        help="Override the handoff's assumed storey height (metres)")
    parser.add_argument("--opening-inset", type=float, default=0.0,
                        help="Narrow every handoff opening by this much on each side (metres), e.g. when "
                             "the handoff's opening is an annotated door frame's outer edge")
    parser.add_argument("--floor-thickness", type=float, default=0.05,
                        help="Floor slab thickness below z = 0 (metres)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Run directory; default output/<scene>/structuregen/<timestamp>")
    parser.add_argument("--blender", type=Path, default=config.BLENDER)
    return parser.parse_args(argv)


def _area(pts) -> float:
    return abs(sum(pts[k][0] * pts[(k + 1) % len(pts)][1] - pts[(k + 1) % len(pts)][0] * pts[k][1]
                   for k in range(len(pts)))) / 2


def check_floors(doc, planned, report) -> list[str]:
    """Floors closed, volume = top area x thickness, rooms + interior walls = the outline."""
    bad = []
    t = planned["floor_thickness_m"]
    outline = doc.get("outline", {}).get("inner_corners_floor")
    if not outline:
        return ["handoff has no outline.inner_corners_floor: no floors built"]
    for f in report["floors"]:
        if not f["closed"]:
            bad.append(f"{f['id']}: not closed")
        if abs(f["volume_m3"] - f["area_m2"] * t) > 1e-4:
            bad.append(f"{f['id']}: volume {f['volume_m3']:.5f} != area x thickness {f['area_m2'] * t:.5f}")
    rooms = sum(f["area_m2"] for f in report["floors"] if f["kind"] == "room")
    inner = sum(s["footprint_area_m2"] for s in planned["segments"] if s["role"] == "interior")
    if abs(rooms + inner - _area(outline)) > 1e-3:
        bad.append(f"rooms {rooms:.4f} m2 + interior walls {inner:.4f} m2 != outline {_area(outline):.4f} m2")
    if not any(f["kind"] == "room" for f in report["floors"]):
        bad.append("no room floor")
    return bad


def main(argv=None) -> int:
    args = parse_args(argv)
    doc = handoff_io.load(args.handoff)
    planned = segments.plan(doc, args.height, args.opening_inset)
    problems = segments.check(planned)
    if problems:
        for p in problems:
            print("PLAN_PROBLEM", p, file=sys.stderr)
        return 2
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out or (config.OUTPUT_ROOT / doc["scene_id"] / "structuregen" / stamp)
    out.mkdir(parents=True, exist_ok=True)
    planned["handoff_dir"] = str(Path(args.handoff).resolve())
    planned["floor_thickness_m"] = args.floor_thickness
    (out / "plan.json").write_text(json.dumps(planned, indent=1))

    result = subprocess.run([str(args.blender), "-b", "--factory-startup", "--python", str(SCRIPT),
                             "--", str(out / "plan.json"), str(out)],
                            capture_output=True, text=True)
    (out / "blender.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        print("BLENDER_FAILED", result.returncode, file=sys.stderr)
        print((result.stdout + result.stderr)[-2000:], file=sys.stderr)
        return 1
    report = json.loads((out / "build_report.json").read_text())

    failures = []
    by_id = {s["id"]: s for s in planned["segments"]}
    for built in report["segments"]:
        seg = by_id[built["id"]]
        if not built["closed"]:
            failures.append(f"{built['id']}: not closed")
        if abs(built["volume_error_m3"]) > TOLERANCE_M3:
            failures.append(f"{built['id']}: volume {built['volume_m3']:.6f}, expected "
                            f"{seg['expected_volume_m3']:.6f}")
    missing = sorted(set(by_id) - {s["id"] for s in report["segments"]})
    if missing:
        failures.append("not built: " + ", ".join(missing))

    failures += check_floors(doc, planned, report)
    planned["floors"] = report["floors"]
    (out / "plan.json").write_text(json.dumps(planned, indent=1))
    from pxr import Usd
    stage = Usd.Stage.Open(str(out / "structure_base.usda"))
    regroup(stage)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/Structure"))
    stage.GetRootLayer().Save()

    cam = doc["camera"]
    views = subprocess.run([str(args.blender), "-b", str(out / "walls.blend"), "--python", str(VIEWS),
                            "--", str(out / "views"), str(cam["width"]), str(cam["height"])],
                           capture_output=True, text=True)
    (out / "blender_views.log").write_text(views.stdout + views.stderr)
    if views.returncode:
        failures.append(f"view rendering failed ({views.returncode}); see blender_views.log")

    latest = out.parent / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(out.name)

    for f in report["floors"]:
        print(f"  {f['id']:20s} closed={f['closed']} area={f['area_m2']:.3f} m2 volume={f['volume_m3']:.4f}")
    volume = sum(s["volume_m3"] for s in report["segments"])
    print(f"STRUCTUREGEN_PLAN {len(planned['segments'])} segments, {len(report['frames'])} cameras, "
          f"wall volume {volume:.4f} m3")
    for s in report["segments"]:
        print(f"  {s['id']:9s} closed={s['closed']} volume={s['volume_m3']:.4f} "
              f"expected={s['expected_volume_m3']:.4f} faces={s['faces']} zones={','.join(s['zones'])}")
    print("STRUCTUREGEN_OUT", out)
    if failures:
        for f in failures:
            print("BUILD_FAILURE", f, file=sys.stderr)
        return 3
    print("STRUCTUREGEN_OK", len(report["segments"]), "segments")
    return 0


if __name__ == "__main__":
    sys.exit(main())
