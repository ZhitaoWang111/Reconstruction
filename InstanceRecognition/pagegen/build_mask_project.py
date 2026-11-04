"""Build the mask-projection inspection page.

This page exists to answer one question per mask: what did the lift actually
produce? For the selected frame and mask it shows the 2D ground truth (the mask
drawn on its RGB frame) beside the 3D points that mask lifted onto, so a bad
projection is visible rather than inferred.

There is one source, the AMB3R per-frame lift, because that is the only geometry
this tree builds. The source keeps its own cropped grid (392x518), so the page
warps masks into that grid and shows the same centre crop the RGB went through.

Two things are drawn on the 2D side, because their difference is the diagnostic:
pixels the mask claims that landed on a surface, and pixels it claims that have no
surface at all (the mask covers sky, or AMB3R's depth never resolved there). The
second group silently contributes no points, so it would otherwise be invisible.

Frames are loaded on demand: a 20-frame scene is ~10M points, and a page that
loads them all upfront would be useless to scroll.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from src import config
from src.geometry.backproject_masks import write_json, align_mask, crop_box, read_point_ply
from src.geometry.amb3r_mask_source import AMBR3R_INTERMEDIATE

TEMPLATES = config.WEB_ROOT / "mask_project"
DEFAULT_SOURCE = config.ARTIFACTS_ROOT / "sources/scene0000_00/scene_amb3r_v1"
DEFAULT_OUTPUT = config.SITE_ROOT / "mask_project"
# Positions are stored as unsigned 16-bit offsets from the scene's minimum corner.
# Over a room-sized span that is well under a tenth of a millimetre, far finer
# than the mesh it came from, while halving the payload against float32.
QUANT_MAX = 65534
MAX_INSTANCES = 16


def encode(array, dtype):
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode("ascii")


def quantize(points, origin, scale):
    return np.clip(np.rint((points - origin) / scale), 0, QUANT_MAX).astype("<u2")


def pack_bits(flags):
    """Pack a boolean grid into bytes, LSB first, for compact transfer."""
    return base64.b64encode(np.packbits(np.asarray(flags, dtype=bool)).tobytes()).decode("ascii")


def write_background(path, points, key):
    """Quantize a scene cloud against its own bounds into a loadable backdrop.

    The page paints the backdrop flat grey, so only positions ship: one u16 triple
    per point. For AMB3R's canonical cloud (492k points) that is 3 MB, against the
    10 MB the legacy viewer spent on float32 positions plus RGB nobody reads.
    """
    points = np.ascontiguousarray(points, dtype=np.float64)
    low = points.min(axis=0)
    high = points.max(axis=0)
    scale = np.maximum(high - low, 1e-6) / QUANT_MAX
    encoded = np.ascontiguousarray(quantize(points, low, scale).reshape(-1))
    path.write_text(
        "window.registerMaskProjectBackground(" + json.dumps(key) + "," + json.dumps({
            "count": int(len(points)),
            "positions_u16": encode(encoded, "<u2"),
            "origin": low.tolist(),
            "scale": scale.tolist(),
        }, separators=(",", ":")) + ");\n")


def amb3r_background_cloud(amb3r_source, override=None):
    """The canonical AMB3R scene cloud behind the mask source, or None.

    A mask source carries only the pixels its masks claimed (~124k of the frame),
    so it cannot serve as its own backdrop. The full cloud is the PLY the Sim(3)
    was fitted against, which the source names in its sim3_source provenance.

    A source built before the Sim(3) was recorded has no such key at all, so the
    lookup is tolerant: an absent key degrades to "no backdrop" exactly like a
    missing file, instead of raising before the caller's warning can run.
    """
    if override is not None:
        return Path(override) if Path(override).is_file() else None
    manifest = json.loads((Path(amb3r_source) / "manifest.json").read_text())
    evaluation = manifest.get("sim3_source")
    if not evaluation or not Path(evaluation).is_file():
        return None
    cloud = Path(json.loads(Path(evaluation).read_text()).get("sim3_ply", ""))
    return cloud if cloud.is_file() else None


def instance_metadata(source, frame, page_root):
    """Per-instance facts the page shows without needing the point payload.

    Mask and RGB paths are resolved through the source and then re-expressed
    relative to page_root (the directory containing index.html), because the masks
    live in the upstream experiment tree while the page lives under the site root.
    """
    rows = []
    with Image.open(source / frame["original_rgb_image"]) as image:
        width, height = image.size
    for item in frame["instances"]:
        path = (source / item["mask_file"]).resolve()
        with Image.open(path) as mask_image:
            mask = np.asarray(mask_image.convert("L")) > 0
        rows.append({
            "instance_id": int(item["instance_id"]),
            "class_name": item["class_name"],
            "sam_prompt": item.get("sam_prompt"),
            "gpt_certainty": item.get("gpt_certainty"),
            "evidence": item.get("evidence"),
            "sam_score": item.get("sam_score"),
            "color": item.get("color") or [255, 255, 255],
            "mask_url": os.path.relpath(path, page_root),
            "claimed_pixels": int(mask.sum()),
        })
    return rows, width, height


def build_single_source(source, output_subdir, first=0, count=None):
    """Build one source's frame data into a subdirectory."""
    source = Path(source).resolve()
    output_subdir = Path(output_subdir).resolve()
    page_root = output_subdir.parent  # URLs are relative to index.html, not the source/ subdir
    manifest = json.loads((source / "manifest.json").read_text())
    frames = manifest["frames"][first:first + count if count else None]
    if not frames:
        raise ValueError("No frames selected")

    output_subdir.mkdir(parents=True, exist_ok=True)
    (output_subdir / "frames").mkdir(exist_ok=True)

    # Compute global bounds from all frames' points. The AMB3R lift keeps its own
    # natural bounds, which are not the RGB frame's.
    all_points = []
    for frame in frames:
        with np.load((source / frame["npz"]).resolve(), allow_pickle=False) as z:
            all_points.append(z["points_world"].astype(np.float64))
    all_points = np.vstack(all_points)
    low = all_points.min(axis=0)
    high = all_points.max(axis=0)
    span = np.maximum(high - low, 1e-6)
    scale = span / QUANT_MAX

    meta_frames = []
    for frame in frames:
        seq = int(frame["sequence_index"])
        instances, width, height = instance_metadata(source, frame, page_root)
        if len(instances) > MAX_INSTANCES:
            raise ValueError(f"Frame {seq} has {len(instances)} instances; the page packs "
                             f"membership into 16 bits")
        with np.load((source / frame["npz"]).resolve(), allow_pickle=False) as z:
            points = z["points_world"].astype(np.float64)
            bits = z["mask_bits"].astype(np.uint32)
            pixels = np.asarray(z["pixel_indices"], dtype=np.int64)
            grid = tuple(int(v) for v in z["grid_shape_hw"])
        # Work in the point grid, not the RGB frame: masks must be warped into
        # AMB3R's cropped resolution before they can be compared pixel-for-pixel
        # with the lifted indices.
        grid_h, grid_w = grid

        masks_grid = []
        for item in frame["instances"]:
            path = (source / item["mask_file"]).resolve()
            with Image.open(path) as mask_image:
                raw = np.asarray(mask_image.convert("L")) > 0
            if raw.shape != (grid_h, grid_w):
                raw = align_mask(raw, AMBR3R_INTERMEDIATE, (grid_w, grid_h))
            masks_grid.append(raw)

        # The same center crop the masks went through, expressed in source-RGB
        # pixels, so the 2D backdrop lines up with the overlay.
        if (grid_h, grid_w) != (height, width):
            left, top, right, bottom = crop_box(AMBR3R_INTERMEDIATE, (grid_w, grid_h))
            mid_w, mid_h = AMBR3R_INTERMEDIATE
            rgb_crop = [
                left * width / mid_w, top * height / mid_h,
                (right - left) * width / mid_w, (bottom - top) * height / mid_h,
            ]
        else:
            rgb_crop = None

        lifted = np.zeros(grid_h * grid_w, dtype=bool)
        lifted[pixels] = True
        for index, row in enumerate(instances):
            mask = np.uint32(1 << index)
            row["lifted_points"] = int(((bits & mask) != 0).sum())

        claimed_bits = np.zeros(grid_h * grid_w, dtype=np.uint16)
        for i, m in enumerate(masks_grid):
            claimed_bits[m.reshape(-1)] |= np.uint16(1 << i)

        payload = {
            "sequence_index": seq,
            "source_frame_id": int(frame["source_frame_id"]),
            "count": int(len(points)),
            "grid_shape_hw": [grid_h, grid_w],
            "origin": low.tolist(),
            "scale": scale.tolist(),
            "positions_u16": encode(quantize(points, low, scale).reshape(-1), "<u2"),
            "mask_bits_u16": encode(bits.astype("<u2"), "<u2"),
            "lifted_bitmap": pack_bits(lifted),
            "claimed_bits_u16": encode(claimed_bits, "<u2"),
        }
        (output_subdir / "frames" / f"{seq:04d}.js").write_text(
            "window.registerMaskProjectFrame(" + json.dumps(seq) + "," +
            json.dumps(payload, separators=(",", ":")) + ");\n")
        meta_frames.append({
            "sequence_index": seq,
            "source_frame_id": int(frame["source_frame_id"]),
            "rgb_url": os.path.relpath((source / frame["original_rgb_image"]).resolve(), page_root),
            # The page indexes claimed_bits/lifted by this grid, so it must match
            # the payload, not the source RGB size (AMB3R crops to 518×392).
            "grid_shape_hw": [grid_h, grid_w],
            "rgb_shape_hw": [height, width],
            "rgb_crop": rgb_crop,
            "points": int(len(points)),
            "claimed_pixels": int(sum(r["claimed_pixels"] for r in instances)),
            "instances": instances,
        })
        print(f"PACKED frame {seq} ({frame['source_frame_id']}): {len(points)} points, "
              f"{len(instances)} masks", flush=True)

    return {
        "scene_id": manifest.get("scene_id"),
        "lifting_policy": manifest.get("lifting_policy"),
        "source": str(source),
        "bounds": {"minimum": low.tolist(), "maximum": high.tolist()},
        "frames": meta_frames,
    }


def build(amb3r_source, output, first=0, count=None, amb3r_background=None):
    """Build the AMB3R source and write the inspection page."""
    amb3r_source, output = (Path(p).resolve() for p in (amb3r_source, output))
    # Backdrop scripts live beside the generated tree and are rebuilt on demand, so
    # the guard must not mistake a rerun for clobbering user data.
    TEMPLATE_FILES = {"index.html", "page.css", "page.js", "background_amb3r.js"}
    if output.exists():
        existing = {p.name for p in output.iterdir()}
        nonempty = existing - TEMPLATE_FILES - {"README.md"}
        if nonempty:
            raise ValueError(f"Refusing to overwrite nonempty output: {output} (has: {nonempty})")
    output.mkdir(parents=True, exist_ok=True)

    # The backdrop is the canonical AMB3R cloud -- the same geometry the lift's
    # points sit on, so the page never frames points against a surface they are not
    # on. It is quantized against its own bounds; the page supplies that cloud's
    # origin/scale when decoding.
    backgrounds = {"amb3r": None}
    amb3r_cloud_path = amb3r_background_cloud(amb3r_source, amb3r_background)
    if amb3r_cloud_path is None:
        print("WARNING: no AMB3R background cloud; the view will show points "
              "over an empty scene", flush=True)
    else:
        amb3r_points, _ = read_point_ply(amb3r_cloud_path)
        write_background(output / "background_amb3r.js", amb3r_points, "amb3r")
        backgrounds["amb3r"] = {"asset": "background_amb3r.js", "points": int(len(amb3r_points))}

    amb3r_meta = build_single_source(amb3r_source, output / "amb3r", first, count)

    # Write the page metadata. The source names its own backdrop: the page loads it
    # on boot, so a source without one (no AMB3R cloud available) degrades to points
    # over an empty scene rather than rendering the wrong geometry.
    write_json(output / "meta.json", {
        "scene_id": amb3r_meta["scene_id"],
        "sources": {"amb3r": amb3r_meta},
        "backgrounds": backgrounds,
    })
    (output / "meta.js").write_text(
        "window.MASK_PROJECT_META=" + json.dumps({
            "scene_id": amb3r_meta["scene_id"],
            "sources": {
                "amb3r": {
                    "policy": amb3r_meta["lifting_policy"],
                    "bounds": amb3r_meta["bounds"],
                    "background": backgrounds["amb3r"],
                    "frames": amb3r_meta["frames"],
                },
            },
        }, separators=(",", ":")) + ";\n")
    for name in ("index.html", "page.css", "page.js"):
        (output / name).write_bytes((TEMPLATES / name).read_bytes())
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amb3r-source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--amb3r-background", type=Path, default=None,
                        help="Scene cloud for the backdrop; defaults to the "
                             "sim3_ply its manifest points at")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--count", type=int, default=None)
    args = parser.parse_args()
    result = build(args.amb3r_source, args.output, args.first, args.count,
                   args.amb3r_background)
    print(json.dumps({"output": str(result)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
