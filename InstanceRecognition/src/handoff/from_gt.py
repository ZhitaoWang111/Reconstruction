"""Write a handoff directory from the ScanNet instance annotation.

This exists so the object-generation and scene-assembly modules can start before
scene understanding is good enough to feed them. It produces the same format a
harness producer will produce, so the consumer never learns which one wrote a
given directory; only `manifest.json` says.

Three source quirks are handled here and must not be rediscovered downstream:

1. The 2D instance PNGs store `objectId + 1`. Value 0 is unannotated background.
   Verified three ways: the value range is 0..67 against objectIds 0..66; the
   log-log correlation between 2D pixel counts and 3D vertex counts is 0.85 at
   offset +1 and 0.38 at offset 0; and reading value 10 as objectId 10 makes a
   362-vertex sink cover 5.5 million pixels while offset +1 makes it the floor.

2. The GT mesh coordinates are the raw scan frame, *not* the axis-aligned frame.
   The AMB3R source was Sim(3)-corrected into the raw scan frame - median surface
   distance 0.12 m against the AMB3R cloud, versus 2.97 m if `axisAlignment` is
   applied first. Writing the aligned mesh here would silently put GT and harness
   clouds in different coordinate frames, which is exactly the mistake the whole
   `coordinate_frame` field exists to prevent.

3. Key frames are computed, not looked up. ScanNet has no key-frame annotation;
   5578 2D instance masks are the whole of it, and visibility has to be derived
   from them.

Usage::

    python -m src.handoff.from_gt --scene /path/to/scene0000_00 \\
        --output artifacts/handoff/scene0000_00/gt
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import laplace

from src import config
from . import schema

# The harness sample defines `seq`; the ScanNet archive defines `source_frame_id`.
# A consumer that knows only one of the two cannot join this directory to a
# harness run later, so both are written in every frame record.
SOURCE = config.ARTIFACTS_ROOT / "sources/scene0000_00/scene_amb3r_v1"
FRAMES = config.ARTIFACTS_ROOT / "frames/scene0000_00"
DEFAULT_SCENE = config.SCANNET_ROOT / "scene0000_00"
DEFAULT_OUTPUT = config.ARTIFACTS_ROOT / "handoff/scene0000_00/gt"
DEFAULT_GT_INPUT = config.WORKSPACE / "Video2PointCloud/data/scannet/scene0000_00_uniform500/input"

# Separation thresholds, tried in order until three views are found. Objects flat
# against a wall cannot reach the first one - a clock on a wall spans 4 degrees in
# this scene - so the fallback is deliberately generous rather than pretending
# every object has three distinct viewpoints available.
# Tried in order until three views are found. No zero entry: an object flat
# against a wall spans a few degrees, and "no separation requirement" filled its
# three slots with adjacent frames.
SEPARATIONS_DEG = (20.0, 14.0, 10.0, 6.0)
MAX_VIEWS = 3

# A frame may show as little as this fraction of the object's best frame and still
# be chosen. Below it the object is mostly occluded, and a key frame the consumer
# cannot see the object in is worse than no key frame at all.
#
# Structural pieces get a lower floor because they are the opposite case: a wall is
# never fully visible in any frame, so holding them to the object standard would
# throw away most of the frames that do show which wall it is. They are not fed to
# the object generator anyway - the floor here is about documenting the room, not
# about providing buildable input.
VISIBILITY_FLOOR = 0.20
VISIBILITY_FLOOR_STRUCTURAL = 0.05

# What edge contact costs a candidate's score. Small on purpose: it breaks ties
# towards complete views without overriding how much of the object is visible.
EDGE_PENALTY = 0.85

# Two chosen frames must sit at least this many frames apart. A toilet in the
# corner has hundreds of usable frames and they are all the same view; without a
# gap the selector happily returns frames 347, 348, 349, which is one view printed
# three times.
MIN_FRAME_GAP = 5

BOX_COLOUR = (255, 210, 0)
BOX_WIDTH = 3


def read_vertex_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Positions and colours from a binary PLY, ignoring any element after vertex.

    The tree's other reader rejects files that carry faces because the clouds it
    reads never do; the ScanNet mesh has 153,587 of them and only the vertices
    matter here.
    """
    types = {"float": "<f4", "double": "<f8", "uchar": "u1", "uint": "<u4",
             "int": "<i4", "short": "<i2"}
    fields, count, element = [], None, None
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY: {path}")
        handle.readline()  # format line; verified by the dtype table below
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Truncated PLY header: {path}")
            tokens = line.decode("ascii").split()
            if tokens[:1] == ["element"]:
                element, size = tokens[1], int(tokens[2])
                if element == "vertex":
                    count = size
            elif tokens[:1] == ["property"] and element == "vertex":
                if tokens[1] == "list":
                    raise ValueError("Vertex list properties are unsupported")
                fields.append((tokens[2], types[tokens[1]]))
            elif tokens == ["end_header"]:
                break
        if count is None:
            raise ValueError(f"PLY has no vertex element: {path}")
        vertices = np.fromfile(handle, dtype=np.dtype(fields), count=count)
    if len(vertices) != count:
        raise ValueError(f"Truncated PLY vertices: {path}")
    points = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float32)
    rgb = np.column_stack([vertices["red"], vertices["green"], vertices["blue"]]).astype(np.uint8)
    return points, rgb


def write_ply(path: Path, points: np.ndarray, rgb: np.ndarray) -> None:
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    data = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                        ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for axis, name in enumerate(("x", "y", "z")):
        data[name] = points[:, axis]
    for axis, name in enumerate(("red", "green", "blue")):
        data[name] = rgb[:, axis]
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        data.tofile(stream)


def load_objects(scene: Path, points: np.ndarray):
    """One owner per vertex, plus the label table, straight from aggregation.json."""
    stem = scene.name
    segments = np.asarray(json.loads(
        (scene / f"{stem}_vh_clean_2.0.010000.segs.json").read_text())["segIndices"])
    if len(segments) != len(points):
        raise ValueError(f"{len(segments)} segments for {len(points)} vertices")
    aggregation = json.loads((scene / f"{stem}.aggregation.json").read_text())
    segment_owner = {}
    for group in aggregation["segGroups"]:
        for segment in group["segments"]:
            segment_owner[segment] = int(group["objectId"])
    # Vertices whose segment is in no group are annotated as belonging to nothing;
    # they stay -1 and are simply not exported.
    owner = np.fromiter((segment_owner.get(int(s), -1) for s in segments),
                        dtype=np.int32, count=len(segments))
    labels = {int(g["objectId"]): g["label"] for g in aggregation["segGroups"]}
    return owner, labels


def camera_centres(sequence_count: int) -> np.ndarray | None:
    """Camera centre per sampled frame, in the scan frame the clouds live in.

    AMB3R's pose is camera-to-world and its world is not the scan frame; the
    evaluation record carries the Sim(3) that maps one to the other, and that is
    applied here rather than leaving the caller to guess a convention.
    """
    manifest_path = SOURCE / "manifest.json"
    if not manifest_path.is_file():
        return None
    source = json.loads(manifest_path.read_text())
    record_path = source.get("sim3_source")
    if not record_path or not Path(record_path).is_file():
        return None
    record = json.loads(Path(record_path).read_text())["sim3_oracle_scale"]
    scale = float(record["alignment_scale"])
    rotation = np.asarray(record["alignment_rotation"], dtype=np.float64)
    translation = np.asarray(record["alignment_translation_m"], dtype=np.float64)
    # The evaluation record names its own prediction file only in some versions,
    # so fall back to the sibling default rather than failing the whole export:
    # camera centres are optional metadata, not a reason to write no handoff.
    npz = Path(record["npz"]) if record.get("npz") else None
    if npz is None or not npz.is_file():
        npz = Path(record_path).parent / "scene_scene0000_00_uniform500_metric_head_results.npz"
    if not npz.is_file():
        return None
    with np.load(npz) as archive:
        poses = archive["pose"][:, :3, 3]
    if len(poses) < sequence_count:
        return None
    centres = (scale * (rotation @ poses[:sequence_count].T).T) + translation
    return centres


def frame_clarity(path: Path) -> float:
    """Variance of the discrete Laplacian on the greyscale frame.

    The same quantity `harness/history.py` uses to pick a representative, so a GT
    key frame and a harness representative are chosen by the same standard.
    """
    with Image.open(path) as source:
        grey = np.asarray(source.convert("L"), dtype=np.float64)
    return float(laplace(grey, mode="reflect").var())


def draw_boxed(rgb: Image.Image, box, label: str) -> Image.Image:
    """Full frame with the object boxed, in the harness's own style.

    Yellow outline, label on a black chip at the box corner. Deliberately the same
    drawing as `historical_bbox_view` so a consumer sees one house style no matter
    which producer made the directory.
    """
    image = rgb.copy()
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", max(17, image.width // 45))
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), outline=BOX_COLOUR, width=BOX_WIDTH)
    text_bounds = draw.textbbox((0, 0), label, font=font)
    x = min(box[0], max(0, image.width - (text_bounds[2] - text_bounds[0])))
    y = min(box[1], max(0, image.height - (text_bounds[3] - text_bounds[1])))
    position = (x - text_bounds[0], y - text_bounds[1])
    draw.rectangle(draw.textbbox(position, label, font=font), fill=(0, 0, 0))
    draw.text(position, label, font=font, fill=(255, 255, 255))
    return image


def select_keyframes(candidates: list[dict], structural: bool = False) -> list[dict]:
    """Up to three frames: best scoring first, then the widest distinct views.

    Visible pixel count is the only occlusion signal available without a depth
    buffer, and it has to rank first. Treating "box touches the image edge" as the
    primary key instead let a couch frame with 24 visible pixels outrank one with
    799,016, because a bar counter occluded the couch and left it entirely inside
    the frame: the box was clean and empty. A frame that barely sees the object is
    useless however well framed it is, so those are dropped outright, and edge
    contact only discounts an otherwise strong candidate.

    `structural` relaxes only the visibility floor. A wall is never fully in view
    from any single camera, so the object-grade floor would discard the frames that
    do show it; the separation and gap rules still apply, because three adjacent
    frames of a wall are still one view.
    """
    usable = [c for c in candidates if c["bearing"] is not None]
    if not usable:
        usable = candidates
    if not usable:
        return []
    # Relative to this object's own best frame, so the floor means "the object is
    # mostly hidden here", not "the object is small in the scene".
    floor = VISIBILITY_FLOOR_STRUCTURAL if structural else VISIBILITY_FLOOR
    best = max(c["pixels"] for c in usable)
    visible = [c for c in usable if c["pixels"] >= floor * best]
    if not visible:
        visible = usable
    for candidate in visible:
        candidate["visibility"] = candidate["pixels"] / max(best, 1)
        # Edge contact costs 15% of the score: enough to prefer a complete view
        # when one exists, not enough to promote a near-invisible frame.
        penalty = EDGE_PENALTY if candidate["touches_border"] else 1.0
        candidate["rank"] = candidate["score"] * penalty
    ordered = sorted(visible, key=lambda c: -c["rank"])
    chosen: list[dict] = []
    for separation in SEPARATIONS_DEG:
        limit = np.cos(np.radians(separation))
        chosen = []
        for candidate in ordered:
            bearing = candidate["bearing"]
            if bearing is not None and any(
                    float(np.dot(bearing, other["bearing"])) >= limit
                    for other in chosen if other["bearing"] is not None):
                continue
            # Near-identical frames are one view, not two. Enforced here rather
            # than by a separation threshold because a corner object can satisfy
            # the angle and still return three consecutive frames.
            if any(abs(candidate["seq"] - other["seq"]) < MIN_FRAME_GAP for other in chosen):
                continue
            chosen.append(candidate)
            if len(chosen) == MAX_VIEWS:
                break
        if len(chosen) == MAX_VIEWS:
            return chosen
    # Return fewer views rather than padding with duplicates: an object visible
    # from one direction has one view, and saying so is more useful to the
    # consumer than three copies of it.
    return chosen


def build(scene: Path, output: Path, source: Path = SOURCE, frames: Path = FRAMES,
         gt_input: Path = DEFAULT_GT_INPUT) -> None:
    scene, output = scene.resolve(), output.resolve()
    source_root = source.resolve()
    frame_root = frames.resolve()
    manifest_path = source_root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing harness source manifest: {manifest_path}")
    source_manifest = json.loads(manifest_path.read_text())
    sequence = [int(frame["source_frame_id"]) for frame in source_manifest["frames"]]
    seq_to_source = {index: value for index, value in enumerate(sequence)}

    points, rgb = read_vertex_ply(scene / f"{scene.name}_vh_clean_2.ply")
    owner, labels = load_objects(scene, points)
    centres = camera_centres(len(sequence))
    # Per-frame GT camera-to-world poses (4x4, scan frame). Used directly because
    # this handoff is GT-derived; no AMB3R or Sim(3) is involved.
    gt_poses = {}
    gt_input_dir = Path(gt_input).resolve()
    if gt_input_dir.is_dir():
        for seq_idx in range(len(sequence)):
            npz_path = gt_input_dir / f"{seq_idx:04d}.npz"
            if npz_path.is_file():
                with np.load(npz_path) as z:
                    if "camera_pose" in z.files:
                        gt_poses[seq_idx] = z["camera_pose"].astype(np.float64)

    instance_zip = scene / f"{scene.name}_2d-instance.zip"
    with zipfile.ZipFile(instance_zip) as archive:
        masks = {int(name.split("/")[-1][:-4]): name
                 for name in archive.namelist() if name.endswith(".png")}
        print(f"reading {len(sequence)} instance masks", flush=True)
        clarity = {}
        candidates: dict[int, list[dict]] = {}
        for seq in range(len(sequence)):
            source_frame = seq_to_source[seq]
            if source_frame not in masks:
                continue
            label = np.asarray(Image.open(
                io.BytesIO(archive.read(masks[source_frame]))))
            height, width = label.shape
            total = float(label.size)
            clarity[seq] = frame_clarity(frame_root / f"frame_{source_frame:06d}.jpg")
            for value in np.unique(label):
                value = int(value)
                if value == 0:
                    continue
                object_id = value - 1
                ys, xs = np.nonzero(label == value)
                pixels = int(len(xs))
                box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                candidate = {
                    "seq": seq, "source_frame_id": source_frame, "pixels": pixels,
                    "ratio": pixels / total, "bbox": box,
                    "fill": pixels / max(1.0, (box[2] - box[0]) * (box[3] - box[1])),
                    "touches_border": bool(box[0] == 0 or box[1] == 0
                                           or box[2] == width or box[3] == height),
                    "bearing": None,
                }
                candidates.setdefault(object_id, []).append(candidate)

    # Clarity is frame-wide, so the score can only be computed once every frame
    # has been measured; doing it inside the loop would need two passes anyway.
    all_clarity = np.asarray([value for value in clarity.values() if value > 0])
    reference = float(np.median(all_clarity)) if len(all_clarity) else 1.0
    for entries in candidates.values():
        for entry in entries:
            sharpness = min(clarity.get(entry["seq"], 0.0) / max(reference, 1e-6), 2.0)
            entry["score"] = entry["ratio"] * (0.5 + 0.5 * sharpness)

    # Bearings need the object centroid, which is known only after the geometry
    # pass below; computed here to keep the ordering obvious.
    object_rows = []
    for object_id in sorted(set(int(v) for v in np.unique(owner)) - {-1}):
        take = owner == object_id
        cloud = points[take]
        if not len(cloud):
            continue
        object_rows.append({
            "object_id": object_id,
            "label": labels.get(object_id, "unknown"),
            "points": cloud,
            "rgb": rgb[take],
            "centroid": cloud.mean(axis=0),
            "minimum": cloud.min(axis=0),
            "maximum": cloud.max(axis=0),
        })

    clouds_dir = output / "clouds"
    frames_dir = output / "keyframes"
    clouds_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for row in object_rows:
        object_id = row["object_id"]
        name = f"obj_{object_id:04d}"
        cloud_path = clouds_dir / f"{name}.ply"
        write_ply(cloud_path, row["points"], row["rgb"])

        entries = candidates.get(object_id, [])
        if centres is not None:
            for entry in entries:
                delta = centres[entry["seq"]] - row["centroid"]
                norm = float(np.linalg.norm(delta))
                entry["bearing"] = delta / norm if norm > 1e-6 else None
                entry["camera_center"] = centres[entry["seq"]].tolist()
        quality = schema.label_quality(row["label"])
        chosen = select_keyframes(entries, structural=quality == "structural") if entries else []

        frame_records = []
        primary = chosen[0]["bearing"] if chosen and chosen[0]["bearing"] is not None else None
        for index, entry in enumerate(chosen):
            frame_dir = frames_dir / name
            frame_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{entry['seq']:04d}"
            source_frame = entry["source_frame_id"]
            shutil.copyfile(frame_root / f"frame_{source_frame:06d}.jpg",
                            frame_dir / f"{stem}_rgb.jpg")
            with zipfile.ZipFile(instance_zip) as archive:
                label = np.asarray(Image.open(io.BytesIO(
                    archive.read(masks[source_frame]))))
            binary = np.where(label == object_id + 1, 255, 0).astype(np.uint8)
            Image.fromarray(binary, mode="L").save(frame_dir / f"{stem}_mask.png")
            draw_boxed(Image.open(frame_dir / f"{stem}_rgb.jpg").convert("RGB"),
                       entry["bbox"], f"{name} {row['label']}").save(
                frame_dir / f"{stem}_boxed.jpg", quality=95, subsampling=0)
            angle = None
            if primary is not None and entry["bearing"] is not None:
                angle = round(float(np.degrees(np.arccos(
                    np.clip(float(np.dot(entry["bearing"], primary)), -1.0, 1.0)))), 2)
            frame = schema.empty_frame()
            frame.update({
                "seq": entry["seq"], "source_frame_id": source_frame,
                "rgb": f"keyframes/{name}/{stem}_rgb.jpg",
                "mask": f"keyframes/{name}/{stem}_mask.png",
                "boxed": f"keyframes/{name}/{stem}_boxed.jpg",
                "bbox_xyxy": entry["bbox"],
                "visible_pixels": entry["pixels"],
                "pixel_ratio": round(entry["ratio"], 6),
                "bbox_fill": round(entry["fill"], 4),
                "clarity": round(clarity.get(entry["seq"], 0.0), 3),
                "visibility": round(entry.get("visibility", 1.0), 4),
                "touches_border": entry["touches_border"],
                "camera_center": entry.get("camera_center"),
                "camera_to_world_gt": gt_poses.get(entry["seq"]).tolist()
                                     if entry["seq"] in gt_poses else None,
                "view_angle_deg": angle,
                "role": schema.KEYFRAME_ROLES[min(index, 2)],
            })
            frame_records.append(frame)

        record = schema.empty_object(name)
        record.update({
            "label": row["label"],
            "label_quality": quality,
            "structural": schema.label_quality(row["label"]) == "structural",
            "semantic_id": object_id,
            "cloud": f"clouds/{name}.ply",
            "point_count": int(len(row["points"])),
            "centroid": [round(float(v), 5) for v in row["centroid"]],
            "bounds": {"minimum": [round(float(v), 5) for v in row["minimum"]],
                       "maximum": [round(float(v), 5) for v in row["maximum"]]},
            "frames_visible": len(entries),
            "key_frames": frame_records,
        })
        records.append(record)

    records.sort(key=lambda r: (-r["point_count"], r["object_id"]))
    counts = {"structural": 0, "generic": 0, "specific": 0}
    for record in records:
        counts[record["label_quality"]] += 1
    views = {}
    for record in records:
        views[len(record["key_frames"])] = views.get(len(record["key_frames"]), 0) + 1

    manifest = schema.manifest(
        scene.name,
        producer={
            "name": "scannet_ground_truth",
            "note": "ScanNet instance annotation, not a scene-understanding run; "
                    "geometry is the reconstructed annotation mesh, which is denser "
                    "and cleaner than a harness cloud.",
            "sources": {
                "mesh": str(scene / f"{scene.name}_vh_clean_2.ply"),
                "segments": str(scene / f"{scene.name}_vh_clean_2.0.010000.segs.json"),
                "aggregation": str(scene / f"{scene.name}.aggregation.json"),
                "instance_2d": str(instance_zip),
                "rgb": str(frame_root),
                "camera_source": str(source_root),
            },
            "instance_id_encoding": "2D PNG value = objectId + 1; 0 is unannotated",
        },
        frames=len(sequence),
        objects=len(records),
        label_quality_counts=counts,
        keyframe_view_counts=views,
        source_frame_ids=sequence,
        axis_alignment_note="clouds use the raw scan frame; apply axisAlignment "
                            "from the .txt only if the NYU frame is required",
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (output / "objects.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    digest = hashlib.sha256((output / "objects.json").read_bytes()).hexdigest()
    print(f"wrote {output}: {len(records)} objects, {counts}, views {views}")
    print(f"objects.json sha256 {digest[:16]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gt-input", type=Path, default=DEFAULT_GT_INPUT,
                        help="Directory of per-frame GT npz files (NNNN.npz) with camera_pose")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--frames", type=Path, default=FRAMES)
    args = parser.parse_args()
    build(args.scene, args.output, args.source, args.frames, args.gt_input)


if __name__ == "__main__":
    main()
