"""Publish a wall handoff (walls/v2) as a page for the people who will build the walls.

    python -m pagegen.build_walls_handoff --handoff artifacts/handoff/scene0000_00/walls_harness

The page reads the handoff directory and nothing the handoff does not point to. It shows
what the wall generator will receive: the walls in plan and in 3D, each wall's observed
and inferred stretches, its joins and thickness, and its key frames with the wall face
projected into them through the camera the file carries - so a wrong pose, wrong
intrinsics or a wall in the wrong place is visible at a glance, not only in a checker.

Background: the scene rays the walls were cut with (every pixel of every frame), already in
the handoff's amb3r_floor frame. When the file is gone, or was written in another frame, the
page publishes without a background, as the object page does.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from src import config
from src.handoff.walls_from_harness import known_issues_zh
from src.structure.frames import RAYS

TEMPLATES = config.WEB_ROOT / "walls_handoff"
DEFAULT_OUTPUT = config.SITE_ROOT / "handoff"
BACKGROUND_VOXEL = 0.04       # m; about 165k points for this scene
PLAN_CELL = 0.04              # m; top-view density image
PLAN_BAND = (0.10, 2.60)      # m; heights that go into it (the floor itself would paint everything)


def b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def plan_image(points: np.ndarray, path: Path) -> dict:
    """Top view of the scene: how many points stack in each cell. A vertical surface (a wall, a
    cabinet side) stacks dozens, a horizontal one (floor, bed, desk) one or two, so the walls
    come out dark and the furniture tops faint. Row 0 is the largest y (north up)."""
    keep = (points[:, 2] > PLAN_BAND[0]) & (points[:, 2] < PLAN_BAND[1])
    xy = points[keep, :2]
    x0, y0 = np.floor(points[:, :2].min(0) / PLAN_CELL) * PLAN_CELL
    x1, y1 = np.ceil(points[:, :2].max(0) / PLAN_CELL) * PLAN_CELL
    nx, ny = int(round((x1 - x0) / PLAN_CELL)), int(round((y1 - y0) / PLAN_CELL))
    ix = np.clip(((xy[:, 0] - x0) / PLAN_CELL).astype(int), 0, nx - 1)
    iy = np.clip(((xy[:, 1] - y0) / PLAN_CELL).astype(int), 0, ny - 1)
    count = np.zeros((ny, nx), np.int32)
    np.add.at(count, (ny - 1 - iy, ix), 1)
    t = np.where(count <= 2, 0.10, np.minimum(0.85, 0.10 + 0.75 * np.clip((count - 2) / 30, 0, 1) ** 0.8))
    ink = np.array([69, 76, 85], np.float64)
    rgba = np.zeros((ny, nx, 4), np.uint8)
    rgba[..., :3] = ink
    rgba[..., 3] = np.where(count > 0, np.round(255 * t), 0).astype(np.uint8)
    Image.fromarray(rgba, "RGBA").save(path)
    return {"x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1), "cell_m": PLAN_CELL,
            "band_m": list(PLAN_BAND)}


def background(doc: dict, rays: Path, media: Path) -> dict | None:
    if not rays.is_file():
        return None
    z = np.load(rays)
    M = np.asarray(doc["frame"]["native_to_frame"], dtype=np.float64)
    if "native_to_frame" not in z.files or not np.allclose(z["native_to_frame"], M, atol=1e-6):
        print(f"{rays} is not in this handoff's frame; no background")
        return None
    points = z["P"].astype(np.float64)
    plan = plan_image(points, media / "plan_density.png")
    plan["image"] = "media/plan_density.png"
    keys = np.floor(points / BACKGROUND_VOXEL).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    points, rgb = points[first], z["rgb"][first]
    low, high = points.min(0), points.max(0)
    quantised = np.round((points - low) / np.maximum(high - low, 1e-6) * 65535).astype(np.uint16)
    return {"n": int(len(points)), "lo": low.round(5).tolist(), "hi": high.round(5).tolist(),
            "pos": b64(quantised), "rgb": b64(rgb.astype(np.uint8)), "voxel_m": BACKGROUND_VOXEL,
            "rays": int(len(z["P"])), "frames": int(len(z["cams"])), "plan": plan,
            "source": "every 4th pixel of every frame (src.structure.rays), amb3r_floor"}


def build(handoff: Path, output: Path, rays: Path = RAYS) -> None:
    handoff = Path(os.path.abspath(handoff))      # abspath, not resolve: keep the path the user knows
    doc = json.loads((handoff / "walls.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    media = output / "media"
    if media.exists():
        shutil.rmtree(media)
    media.mkdir()
    for wall in doc["walls"]:
        for frame in wall["key_frames"]:
            for kind in ("rgb", "mask", "boxed"):
                target = media / frame[kind].split("keyframes/", 1)[1]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(handoff / frame[kind], target)
                frame[kind] = str(target.relative_to(output))
    payload = {"doc": doc, "background": background(doc, rays, media), "known_issues_zh": known_issues_zh(doc),
               "handoff_dir": os.path.relpath(handoff, output), "handoff_abs": str(handoff)}
    (output / "walls.js").write_text("window.WALLS=" + json.dumps(payload, ensure_ascii=False,
                                                                  separators=(",", ":")) + ";\n")
    for name in ("index.html", "page.css", "page.js"):
        (output / name).write_text((TEMPLATES / name).read_text())
    size = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    frames = sum(len(w["key_frames"]) for w in doc["walls"])
    print(f"wrote {output}: {len(doc['walls'])} walls, {frames} key frames, "
          f"background {'yes' if payload['background'] else 'no'}, {size / 1e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--handoff", type=Path, default=config.ARTIFACTS_ROOT / "handoff/scene0000_00/walls_harness")
    parser.add_argument("--output", type=Path, default=None,
                        help="pages directory (default: <site>/handoff/<scene>/<handoff dir name>)")
    parser.add_argument("--rays", type=Path, default=RAYS, help="scene rays for the background (optional)")
    args = parser.parse_args()
    output = args.output or DEFAULT_OUTPUT / args.handoff.resolve().parent.name / args.handoff.resolve().name
    build(args.handoff, output, args.rays)


if __name__ == "__main__":
    main()
