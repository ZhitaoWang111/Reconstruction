"""compare(frame): the photo on the left, the current model from the same camera on the right.

    PYTHONPATH=src .venv-sam3d/bin/python -m structuregen.agent.compare <scene.blend> <frame> [<frame> ...] --out <dir>

The model half is flat colour (every pixel shows the colour that was set) darkened only in
corners by ambient occlusion, with thin outlines where faces meet at an angle, at a depth step,
and around trims. No numbers are given: the agent judges colour and placement from the image.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2
import numpy as np

from .. import config
from ..common import handoff as handoff_io

SCRIPT = Path(__file__).parent / "blender" / "blender_compare.py"
SCALE = 0.5                 # model render at half the photo size; the image is 1400 px wide anyway
AO_FLOOR = 0.55             # AO only darkens, and never below this share of the set colour
EDGE_COLOUR = (40, 40, 40)  # BGR
CREASE_DEG = 30.0
DEPTH_STEP = 0.04           # relative distance jump that counts as a silhouette
WIDTH = 1400
BACKGROUND = (60, 60, 60)


def passes(blend: Path, cameras: list[str], out: Path, blender: Path = config.BLENDER) -> None:
    result = subprocess.run([str(blender), "-b", str(blend), "--python", str(SCRIPT), "--",
                             str(out), str(SCALE), *cameras], capture_output=True, text=True)
    if result.returncode or "SG_COMPARE_PASSES" not in result.stdout:
        raise RuntimeError("compare passes failed:\n" + (result.stdout + result.stderr)[-2000:])


def model_image(stem: Path) -> np.ndarray:
    flat = cv2.imread(f"{stem}_flat.png", cv2.IMREAD_UNCHANGED)
    ao = cv2.imread(f"{stem}_ao.png", cv2.IMREAD_UNCHANGED)
    geo = cv2.imread(f"{stem}_geo.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
    dist = cv2.imread(f"{stem}_dist.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
    alpha = flat[..., 3] > 0
    shade = AO_FLOOR + (1 - AO_FLOOR) * (ao[..., :3].mean(-1).astype(np.float32) / 255.0)
    shade[~alpha] = 1.0
    img = flat[..., :3].astype(np.float32) * shade[..., None]
    img[~alpha] = BACKGROUND

    normal = geo[..., 2::-1] * 2 - 1                        # BGR file -> xyz
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-6)
    d = dist[..., 2]                                        # R channel (BGR order in OpenCV)
    trim = dist[..., 1] > 0.5
    edge = np.zeros(alpha.shape, bool)
    cos_t = np.cos(np.radians(CREASE_DEG))
    for dy, dx in ((0, 1), (1, 0)):
        a, b = (slice(None, -dy or None), slice(None, -dx or None)), (slice(dy, None), slice(dx, None))
        both = alpha[a] & alpha[b]
        crease = both & ((normal[a] * normal[b]).sum(-1) < cos_t)
        step = both & (np.abs(d[a] - d[b]) > DEPTH_STEP * np.minimum(d[a], d[b]))
        border = (alpha[a] != alpha[b]) | (both & (trim[a] != trim[b]))
        e = crease | step | border
        edge[a] |= e
        edge[b] |= e
    img[edge] = EDGE_COLOUR
    return img.clip(0, 255).astype(np.uint8)


def compose(photo: np.ndarray, model: np.ndarray, label: str) -> np.ndarray:
    model = cv2.resize(model, (photo.shape[1], photo.shape[0]), interpolation=cv2.INTER_NEAREST)
    pair = np.hstack([photo, np.full((photo.shape[0], 8, 3), 255, np.uint8), model])
    pair = cv2.resize(pair, (WIDTH, int(round(pair.shape[0] * WIDTH / pair.shape[1]))), interpolation=cv2.INTER_AREA)
    bar = np.full((30, WIDTH, 3), 255, np.uint8)
    cv2.putText(bar, label, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
    return np.vstack([bar, pair])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("blend", type=Path)
    parser.add_argument("frames", type=int, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, default=config.DEFAULT_HANDOFF)
    args = parser.parse_args(argv)
    doc = handoff_io.load(args.handoff)
    rgb = {k["source_frame_id"]: args.handoff / k["rgb"] for w in doc["walls"] for k in w["key_frames"]}
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        cams = [f"frame_{f:04d}" for f in args.frames]
        passes(args.blend, cams, Path(tmp))
        for f, cam in zip(args.frames, cams):
            img = compose(cv2.imread(str(rgb[f])), model_image(Path(tmp) / cam),
                          f"frame {f}   left: photo   right: current model from the same camera")
            cv2.imwrite(str(args.out / f"{f:04d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("COMPARE", len(args.frames), "->", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
