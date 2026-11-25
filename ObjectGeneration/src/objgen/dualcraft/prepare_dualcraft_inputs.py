"""Turn handoff objects into DualCraft photo-route inputs: cropped key frames, nothing else.

DualCraft's photo route takes photos of unknown cameras: its own model anchors each one
and estimates the size while drafting the blueprint.  So the only input is the photos.
Size, front and placement stay on our side: the export scales the asset to the scan and
turns it into the scene afterwards, from the record this script writes beside the photos.

For every selected object (``--objects``, or the ``dualcraft`` objects of ``routing.json``)
the script takes the handoff's key frames, the frames upstream already chose for that
object (best visibility and clarity, then widest separated views, at most three), and

1. drops a frame that leaves too much of the object out of the picture: the object's scan
   points are projected through the frame's camera and more than ``OUT_OF_PICTURE_MAX``
   land outside it (obj_0038's third frame: 53 %, the cabinet's right half).  The mask
   cannot tell: it only covers what is in the picture, and that half-cabinet has 5 % of
   its mask at the edge.  A frame is never dropped for this if it is the least cut one;
2. drops a frame within ``MIN_SEPARATION_DEG`` of a frame already kept, judged by the
   direction from the object to each camera (upstream widens its separation down to 6 deg
   to fill three slots; a near-duplicate adds no information and one more anchoring
   guess);
3. crops each kept frame to the mask's box plus ``MARGIN`` of its size on every side
   (DualCraft's own video route uses the same 15 %): the full frame shows several objects
   and the photo route only knows its target as the one in the middle;
4. scales the crop's long side to ``MAX_SIDE`` (DualCraft does the same on intake).

Output, one directory per object, and one ``dualcraft_inputs.json`` index::

    <generation>/dualcraft_inputs/<object_id>/
        photo_0.png ...     the photos, in the order handed over (best key frame first)
        conversion.json     where each photo came from, what was dropped and why, and the
                            object's scan box and frame for the export to scale and place it

``conversion.json`` carries the whole provenance; DualCraft never reads it and it never goes
to the server.  There a job is made of the photos alone (``create_job(photos)``, no
``video``); the import reads this file from here when the job comes back.

Usage::

    PYTHONPATH=src .venv-sam3d/bin/python -m objgen.dualcraft.prepare_dualcraft_inputs \\
        output/scene0000_00/gt --objects obj_0038 [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

from ..config import ROOT

from .. import handoff_io as hio  # noqa: E402

SCHEMA = "dualcraft_inputs/v1"
MARGIN = 0.15
MAX_SIDE = 1024
# ScanNet's camera stands close: big furniture rarely fits whole (obj_0038's best key frame
# leaves 22 % of the scan out, its second 30 %; the couch's best one 41 % and it still reads
# as the couch), so the bar is for frames that miss the larger part: the shower at 42 % shows
# its whole front, the fridge at 53 % is a strip of door behind the counter, the toaster oven
# at 58 % sits in a corner of a frame about the cabinets below it.
OUT_OF_PICTURE_MAX = 0.50
MIN_SEPARATION_DEG = 10.0
MAX_PHOTOS = 3          # upstream keeps at most three key frames; DualCraft takes up to 12
FIT_STEP_DEG = 0.5


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def out_of_picture(points: np.ndarray, camera_to_world: list, camera: dict) -> float:
    """Share of the object's scan points that the frame's camera does not see in its
    picture: behind the camera or projected outside the image (OpenCV axes)."""
    pose = np.asarray(camera_to_world, dtype=np.float64)
    local = (points - pose[:3, 3]) @ pose[:3, :3]
    ahead = local[:, 2] > 1e-3
    u = camera["fx"] * local[ahead, 0] / local[ahead, 2] + camera["cx"]
    v = camera["fy"] * local[ahead, 1] / local[ahead, 2] + camera["cy"]
    inside = (u >= 0) & (u < camera["width"]) & (v >= 0) & (v < camera["height"])
    return 1.0 - float(inside.sum()) / len(points)


def crop_box(mask: np.ndarray, margin: float = MARGIN) -> list[int]:
    """The mask's box widened by ``margin`` of its size on every side, clipped to the
    picture; [left, top, right, bottom), pixels."""
    ys, xs = np.nonzero(mask)
    height, width = mask.shape
    left, right, top, bottom = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    pad_x, pad_y = margin * (right - left), margin * (bottom - top)
    return [int(max(0, math.floor(left - pad_x))), int(max(0, math.floor(top - pad_y))),
            int(min(width, math.ceil(right + pad_x))), int(min(height, math.ceil(bottom + pad_y)))]


def bearing(camera_to_world: list, centre: np.ndarray) -> np.ndarray | None:
    """Unit direction from the object's centre to the camera, in the scan frame."""
    delta = np.asarray(camera_to_world, dtype=np.float64)[:3, 3] - centre
    norm = float(np.linalg.norm(delta))
    return delta / norm if norm > 1e-6 else None


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b))))))


def scan_box(points: np.ndarray) -> dict:
    """Oriented footprint box over the 1-99 percentile of the scan cloud (min-area
    rectangle, ``FIT_STEP_DEG`` steps) and the height range.  What the scan saw, in
    metres: the export scales the asset to it and turns the asset into the scene."""
    low, high = np.percentile(points[:, 2], [1, 99])
    xy = points[:, :2]
    best = None
    for degrees in np.arange(0.0, 90.0, FIT_STEP_DEG):
        a = math.radians(degrees)
        axes = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
        projected = xy @ axes.T
        lo, hi = np.percentile(projected, [1, 99], axis=0)
        area = float(np.prod(hi - lo))
        if best is None or area < best[0]:
            best = (area, axes, lo, hi)
    _, axes, lo, hi = best
    extent = hi - lo
    long_index = int(np.argmax(extent))
    centre_xy = ((lo + hi) / 2) @ axes
    return {
        "centre_xy": [round(float(v), 4) for v in centre_xy],
        "bottom_z": round(float(low), 4), "top_z": round(float(high), 4),
        "height": round(float(high - low), 4),
        "long": round(float(extent[long_index]), 4),
        "short": round(float(extent[1 - long_index]), 4),
        "long_axis_xy": [round(float(v), 6) for v in axes[long_index]],
        "short_axis_xy": [round(float(v), 6) for v in axes[1 - long_index]],
        "points": int(len(points)),
        "method": f"min-area rectangle over the 1-99 percentile XY of the scan cloud, "
                  f"{FIT_STEP_DEG:g} deg steps; height = 1-99 percentile Z",
        "caution": "a scan covers only what the cameras saw: an object against a wall is "
                   "usually shallower here than it is, and a base the scan missed makes it lower",
    }


def select_frames(key_frames: list[dict], cut: dict[int, float], centre: np.ndarray,
                  max_photos: int = MAX_PHOTOS) -> tuple[list[dict], list[dict]]:
    """Key frames to hand over, in upstream order, and the ones dropped with the reason.
    ``cut``: per key frame, the share of the object out of the picture."""
    kept, dropped = [], []
    least_cut = min(cut.values())
    for frame in key_frames:
        share = cut[frame["seq"]]
        if share > OUT_OF_PICTURE_MAX and share > least_cut:
            dropped.append({"seq": frame["seq"], "role": frame["role"],
                            "reason": f"{share:.0%} of the object out of the picture "
                                      f"(over {OUT_OF_PICTURE_MAX:.0%})"})
            continue
        direction = bearing(frame["camera_to_world"], centre)
        near = [(k, angle_deg(direction, k["_bearing"])) for k in kept
                if direction is not None and k["_bearing"] is not None]
        close = [(k, a) for k, a in near if a < MIN_SEPARATION_DEG]
        if close:
            other, angle = close[0]
            dropped.append({"seq": frame["seq"], "role": frame["role"],
                            "reason": f"{angle:.1f} deg from key frame {other['seq']}, which is kept "
                                      f"(under {MIN_SEPARATION_DEG:g} deg: the same view)"})
            continue
        if len(kept) >= max_photos:
            dropped.append({"seq": frame["seq"], "role": frame["role"],
                            "reason": f"more than {max_photos} photos"})
            continue
        kept.append({**frame, "_bearing": direction, "_out_of_picture": share})
    return kept, dropped


def prepare_object(generation: Path, handoff: Path, record: dict, camera: dict, out: Path, *,
                   dry_run: bool = False) -> dict:
    import trimesh
    oid = record["object_id"]
    cloud = np.asarray(trimesh.load(handoff / record["cloud"]).vertices, dtype=np.float64)
    box = scan_box(cloud)
    centre = np.array([*box["centre_xy"], (box["bottom_z"] + box["top_z"]) / 2])
    masks = {k["seq"]: np.asarray(Image.open(handoff / k["mask"])) > 0 for k in record["key_frames"]}
    cut = {k["seq"]: out_of_picture(cloud, k["camera_to_world"], camera) for k in record["key_frames"]}
    kept, dropped = select_frames(record["key_frames"], cut, centre)
    if not kept:
        raise ValueError(f"{oid}: no usable key frame")
    photos = []
    if not dry_run:
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
    for index, frame in enumerate(kept):
        box_px = crop_box(masks[frame["seq"]])
        name = f"photo_{index}.png"
        entry = {
            "file": name, "seq": frame["seq"], "role": frame["role"],
            "source_frame_id": frame["source_frame_id"],
            "rgb": frame["rgb"], "mask": frame["mask"], "crop": box_px,
            "visibility": frame.get("visibility"), "clarity": frame.get("clarity"),
            "out_of_picture": round(frame["_out_of_picture"], 4),
            "camera_to_world": frame["camera_to_world"],
            "azimuth_from_object_deg": round(math.degrees(math.atan2(
                frame["_bearing"][1], frame["_bearing"][0])), 2) if frame["_bearing"] is not None else None,
        }
        if not dry_run:
            image = Image.open(handoff / frame["rgb"]).convert("RGB").crop(tuple(box_px))
            image.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
            image.save(out / name)
            entry.update(size_px=list(image.size), sha256=sha256(out / name))
        photos.append(entry)
    conversion = {
        "schema_version": SCHEMA,
        "object_id": oid, "label": record.get("label"),
        "route": "photos",
        "note": "DualCraft gets the photos only: it anchors their cameras and estimates the "
                "size itself.  Everything else here is for our export (scale to scan_box, "
                "place in the scene) and for review.",
        "source": {"handoff": str(handoff), "cloud": record["cloud"],
                   "cloud_sha256": sha256(handoff / record["cloud"])},
        "photos": photos,
        "dropped": dropped,
        "scan_box": box,
        "rules": {"margin": MARGIN, "max_side": MAX_SIDE, "out_of_picture_max": OUT_OF_PICTURE_MAX,
                  "min_separation_deg": MIN_SEPARATION_DEG, "max_photos": MAX_PHOTOS},
        "created_at": hio.now(),
    }
    if not dry_run:
        hio.atomic_json(out / "conversion.json", conversion)
    return conversion


def selected_objects(generation: Path, objects: list[str] | None) -> list[str]:
    if objects:
        return objects
    routing = generation / "routing.json"
    if not routing.is_file():
        raise SystemExit("give --objects, or run route_generators.py first (no routing.json)")
    data = json.loads(routing.read_text())
    return sorted(o["object_id"] for o in data.get("objects", []) if o.get("route") == "dualcraft")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("generation", type=Path, help="generation directory (has manifest.json)")
    parser.add_argument("--objects", help="comma-separated object ids; default: routing.json's dualcraft objects")
    parser.add_argument("--output", type=Path, help="default: <generation>/dualcraft_inputs")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    args = parser.parse_args(argv)
    generation = args.generation.resolve()
    manifest = json.loads((generation / "manifest.json").read_text())
    handoff = Path(manifest["source_handoff"]["path"])
    scene = {o["object_id"]: o for o in json.loads((handoff / "objects.json").read_text())}
    camera = json.loads((handoff / "manifest.json").read_text())["camera"]
    ids = selected_objects(generation, args.objects.split(",") if args.objects else None)
    unknown = [oid for oid in ids if oid not in scene]
    if unknown:
        raise SystemExit("not in the handoff: " + ", ".join(unknown))
    root = (args.output or generation / "dualcraft_inputs").resolve()
    index = []
    for oid in ids:
        conversion = prepare_object(generation, handoff, scene[oid], camera, root / oid, dry_run=args.dry_run)
        kept = ", ".join(f"{p['seq']}({p['role'][0]})" for p in conversion["photos"])
        dropped = "; ".join(f"{d['seq']}: {d['reason']}" for d in conversion["dropped"]) or "-"
        print(f"{oid} {scene[oid]['label']:<16} photos {kept:<22} dropped {dropped}")
        index.append({"object_id": oid, "label": scene[oid].get("label"),
                      "photos": len(conversion["photos"]), "dropped": len(conversion["dropped"]),
                      "directory": f"{oid}"})
    if not args.dry_run:
        previous = {}
        if (root / "dualcraft_inputs.json").is_file():
            previous = {o["object_id"]: o for o in json.loads((root / "dualcraft_inputs.json").read_text())["objects"]}
        previous.update({o["object_id"]: o for o in index})
        hio.atomic_json(root / "dualcraft_inputs.json", {
            "schema_version": SCHEMA, "generation": str(generation), "handoff": str(handoff),
            "objects": sorted(previous.values(), key=lambda o: o["object_id"]), "updated_at": hio.now()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
