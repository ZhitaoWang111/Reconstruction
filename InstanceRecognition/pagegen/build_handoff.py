"""Publish a handoff directory as a page for the downstream module authors.

The handoff contract is judged by people who will write the object generator and
the scene assembler, not by the producer. They need to see, per object: what it is
called, what it looks like from each key frame, whether the box actually contains
it, and how good the geometry is. That is what this page shows - it is a reading
instrument for the contract, not a demo of the pipeline.

Clouds are re-encoded here rather than copied: the PLY files are the contract's
format, but a browser cannot read a binary PLY cheaply, and quantising each
object to 7 bytes per point keeps the whole scene under a megabyte.

The 3D panel needs a background, and the honest one is the whole annotated mesh
the object clouds were cut from - not an unrelated reconstruction. It is read from
`producer.sources.mesh` when that path still exists, and omitted otherwise, so a
handoff from a producer that cannot supply a scene cloud still publishes.
"""
from __future__ import annotations

import argparse
import base64
import colorsys
import json
import shutil
from pathlib import Path

import numpy as np

from src import config

TEMPLATES = config.WEB_ROOT / "handoff"
DEFAULT_OUTPUT = config.SITE_ROOT / "handoff"

# Background thinning, metres. 0.05 keeps furniture legible at about 72k points;
# coarser values blur a chair into a blob, finer ones stall the first paint.
BACKGROUND_VOXEL = 0.05


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Positions and colours from a PLY, ignoring any element after `vertex`.

    The contract's own clouds carry no faces, but the GT background source is the
    full annotated mesh and does. Its elements are read only up to the end of the
    header, so a `property list ...` line in the face element is skipped rather
    than crashing the page build.
    """
    fields, count, element = [], None, None
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY: {path}")
        handle.readline()
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Truncated PLY header: {path}")
            tokens = line.decode("ascii").split()
            if tokens[:1] == ["element"]:
                element = tokens[1]
                if element == "vertex":
                    count = int(tokens[2])
            elif tokens[:1] == ["property"] and element == "vertex":
                fields.append((tokens[2], {"float": "<f4", "uchar": "u1"}[tokens[1]]))
            elif tokens == ["end_header"]:
                break
        if count is None:
            raise ValueError(f"PLY has no vertex element: {path}")
        vertices = np.fromfile(handle, dtype=np.dtype(fields), count=count)
    points = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float32)
    rgb = np.column_stack([vertices["red"], vertices["green"], vertices["blue"]]).astype(np.uint8)
    return points, rgb


def encode_cloud(points: np.ndarray, rgb: np.ndarray) -> dict:
    """Quantise one object's points to 3 x u8 offset + 3 x u8 colour.

    Per-object extents, not scene-wide: a chair 40 cm across would otherwise
    quantise to a handful of distinct positions inside an 8.4 m scene box, which
    turns a point cloud into confetti at the zoom level the reader will use.
    """
    low = points.min(axis=0)
    span = np.maximum(points.max(axis=0) - low, 1e-5)
    quantised = np.round((points - low) / span * 255).astype(np.uint8)
    packed = np.empty((len(points), 6), dtype=np.uint8)
    packed[:, :3] = quantised
    packed[:, 3:] = rgb
    return {
        "minimum": [round(float(v), 5) for v in low],
        "span": [round(float(v), 5) for v in span],
        "centroid": [round(float(v), 5) for v in points.mean(axis=0)],
        "count": int(len(points)),
        "payload": base64.b64encode(packed.tobytes()).decode("ascii"),
    }


def encode_background(points: np.ndarray, rgb: np.ndarray, voxel: float) -> dict:
    """Scene-wide cloud, voxel-thinned and colour-averaged.

    Scene extents here rather than per-object, because the background is read as
    one room. 0.05 m retains the shape of furniture while cutting 492k points to
    about 72k, which is the difference between a page that opens and one that
    stalls. Colour is averaged per voxel so the wall keeps its own tone.
    """
    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable")
    sums = np.zeros((len(counts), 3), dtype=np.float64)
    rs = np.zeros((len(counts), 3), dtype=np.float64)
    np.add.at(sums, inverse, points.astype(np.float64))
    np.add.at(rs, inverse, rgb.astype(np.float64))
    centres = sums / counts[:, None]
    colours = (rs / counts[:, None]).round().clip(0, 255).astype(np.uint8)
    return encode_cloud(centres.astype(np.float32), colours)


def object_colour(object_id: str) -> str:
    """Stable hex colour per object, matching the swatch the list will show."""
    number = int("".join(ch for ch in object_id if ch.isdigit()) or 0)
    hue = (number * 0.61803398875) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 0.95)
    return "#%02x%02x%02x" % (round(red * 255), round(green * 255), round(blue * 255))


def build(handoff: Path, output: Path) -> None:
    handoff = handoff.resolve()
    manifest = json.loads((handoff / "manifest.json").read_text())
    records = json.loads((handoff / "objects.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    media = output / "media"
    if media.exists():
        shutil.rmtree(media)

    published = []
    for record in records:
        name = record["object_id"]
        entry = {key: record[key] for key in
                 ("object_id", "label", "label_quality", "structural", "description",
                  "point_count", "centroid", "bounds", "frames_visible", "semantic_id")}
        entry["key_frames"] = []
        for frame in record["key_frames"]:
            published_frame = {key: frame[key] for key in
                               ("seq", "source_frame_id", "bbox_xyxy", "visible_pixels",
                                "pixel_ratio", "bbox_fill", "clarity", "touches_border",
                                "view_angle_deg", "role")}
            for kind in ("rgb", "mask", "boxed"):
                source = handoff / frame[kind]
                relative = Path("media") / name / source.name
                (output / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, output / relative)
                published_frame[kind] = str(relative)
            entry["key_frames"].append(published_frame)
        points, rgb = read_ply(handoff / record["cloud"])
        entry["cloud"] = encode_cloud(points, rgb)
        entry["colour"] = object_colour(name)
        published.append(entry)

    background = None
    mesh_source = (manifest.get("producer", {}).get("sources", {}) or {}).get("mesh")
    if mesh_source and Path(mesh_source).is_file():
        points, rgb = read_ply(Path(mesh_source))
        background = encode_background(points, rgb, BACKGROUND_VOXEL)
        background["source"] = "annotated mesh, voxel-thinned"
        background["voxel"] = BACKGROUND_VOXEL

    payload = {"manifest": manifest, "objects": published, "background": background}
    (output / "handoff.js").write_text(
        "window.HANDOFF=" + json.dumps(payload, ensure_ascii=False,
                                       separators=(",", ":")) + ";\n")
    for name in ("index.html", "page.css", "page.js"):
        (output / name).write_text((TEMPLATES / name).read_text())
    size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    print(f"wrote {output}: {len(published)} objects, {size / 1e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None,
                        help="Pages directory (default: <site>/handoff/<scene>/<producer>)")
    args = parser.parse_args()
    output = args.output
    if output is None:
        scene = args.handoff.parent.name
        output = DEFAULT_OUTPUT / scene / args.handoff.name
    build(args.handoff, output)


if __name__ == "__main__":
    main()
