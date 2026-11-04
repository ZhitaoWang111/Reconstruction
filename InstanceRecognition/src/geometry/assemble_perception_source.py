"""Assemble the perception stage's labels and masks into a liftable source.

The perception stage writes two independent trees -- ``labels/`` (GPT) and
``masks/`` (SAM3.1) -- and neither knows about the other. Every lifting module
instead consumes one source directory whose ``manifest.json`` names, per frame,
the RGB image and each mask with its category. This module is that bridge.

It lifts nothing and writes no point. It resolves the RGB image, which the
labels record only as a path the refactor invalidated, and it proves the two
halves belong together before writing anything:

  * each mask records the SHA-256 of the label file it was produced from, so a
    label/mask mismatch stops the run instead of quietly lifting the wrong
    categories;
  * the RGB hash recorded by the labels, by the masks, and recomputed from the
    file must all agree.

Mask and RGB paths are written relative to the output directory and point into
the perception tree in place; the masks are not copied.

``lifting_policy`` is deliberately absent. Until a lifting module has run there
is no geometry to describe, and the Harness refuses a source whose policy it
does not know, so pointing the Harness here fails loudly instead of silently
lifting nothing.

Run from InstanceRecognition::

    python -m src.geometry.assemble_perception_source \\
      --run gpt_sam31_uniform500 \\
      --output artifacts/sources/scene0000_00/gpt_sam31_uniform500_v1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src import config
from src.geometry.backproject_masks import (
    encode,
    instance_color,
    read_point_ply,
    relative,
    sha256_file,
    write_cloud_asset,
    write_json,
)
from src.inference.io import perception_dir
from src.visualization.publish_gpt_viewer import write_manifest

POLICY = "gpt_sam31_perception_v1"

# One bit per instance is carried in a uint32, so a frame cannot hold more.
MAX_INSTANCES = 32

DEFAULT_FRAMES = config.ARTIFACTS_ROOT / "frames"

# The cloud the viewer draws behind the lifted points. It has to be the one the
# lifting will align to or the two sit in different coordinates, and AMB3R's
# canonical cloud is what the recorded Sim(3) was fitted to -- so it is the
# backdrop for anything lifted from AMB3R point maps.
DEFAULT_BACKDROP = (
    config.WORKSPACE
    / "Video2PointCloud/output/amb3r/vo_uniform500/pointcloud_gt_corrected_rgb.ply"
)


def _read(path: Path, what: str) -> dict:
    if not path.is_file():
        raise SystemExit(f"Missing {what}: {path}")
    return json.loads(path.read_text())


def _frame(record, labels_run, masks_run, frames_dir, scene_id, output):
    """One frame's manifest entry, or an error naming what disagrees."""
    seq = int(record["sequence_index"])
    fid = int(record["source_frame_id"])
    label_path = labels_run / f"labels/{seq:04d}.json"
    predictions_path = masks_run / f"frame_{fid:06d}/sam31/predictions.json"
    labels = _read(label_path, "GPT labels")
    predictions = _read(predictions_path, "SAM3.1 predictions")

    # The halves are keyed differently -- labels by sequence, masks by source
    # frame -- so a mis-pairing is the easy mistake, and catching it here is what
    # keeps it from surfacing much later as wrong geometry.
    for name, document in (("labels", labels), ("predictions", predictions)):
        if int(document["sequence_index"]) != seq or int(document["source_frame_id"]) != fid:
            raise ValueError(f"{name} disagree on frame {seq} (source frame {fid})")

    # A mask is only evidence about the labels it was actually produced from.
    provenance = predictions["provenance"]
    if provenance["labels_sha256"] != sha256_file(label_path):
        raise ValueError(
            f"Frame {seq}: the masks were produced from different labels than "
            f"{label_path}; the two perception halves do not belong together"
        )

    rgb = (frames_dir / scene_id / f"frame_{fid:06d}.jpg").resolve()
    if not rgb.is_file():
        raise SystemExit(f"Missing RGB for source frame {fid}: {rgb}")
    digest = sha256_file(rgb)
    for name, value in (("labels batch", record["source_sha256"]),
                        ("label file", labels["source_sha256"]),
                        ("predictions", provenance["source_sha256"])):
        if value != digest:
            raise ValueError(f"Frame {seq}: {name} describe a different RGB than {rgb}")

    instances = predictions["instances"]
    if predictions["instance_count"] != len(instances):
        raise ValueError(f"Frame {seq}: instance_count disagrees with instances")
    if len(instances) > MAX_INSTANCES:
        raise ValueError(f"Frame {seq}: {len(instances)} instances exceeds the mask_bits capacity")

    folder = predictions_path.parent
    for item in instances:
        item["mask_file"] = relative(folder / item["mask_file"], output)
        # Perception records no display colour, but every viewer keys a mask by it:
        # without one the page falls back to white for the swatch and for the
        # projected points alike, so all masks look identical. The hue is derived
        # from the frame/instance pair, so it is stable across runs and rebuilds.
        item["color"] = instance_color(seq, int(item["instance_id"]))
    return {
        "sequence_index": seq,
        "source_frame_id": fid,
        "original_rgb_image": relative(rgb, output),
        "annotated_image": relative(folder / "annotated.jpg", output),
        "predictions_json": relative(predictions_path, output),
        "instances": instances,
    }, predictions["gpt_model"]


def build(run, output, frames_dir=DEFAULT_FRAMES, backdrop=DEFAULT_BACKDROP,
          first=0, count=None):
    """Assemble `run`'s perception output into the new source directory `output`."""
    output = Path(output).resolve()
    frames_dir, backdrop = Path(frames_dir).resolve(), Path(backdrop).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Use a new empty output directory: {output}")
    if not backdrop.is_file():
        raise SystemExit(f"Missing backdrop cloud: {backdrop}")
    started = time.monotonic()

    perception = perception_dir(run).resolve()
    labels_run, masks_run = perception / "labels", perception / "masks"
    run_config = _read(labels_run / "config.json", "labeling run config")
    scene_id = run_config["scene_id"]
    records = run_config["frames"][first:first + count if count is not None else None]
    if not records:
        raise ValueError("No frames selected")
    sequences = [int(r["sequence_index"]) for r in records]
    if sequences != list(range(len(sequences))):
        raise ValueError("Selected frames must be a contiguous run starting at sequence 0")

    output.mkdir(parents=True, exist_ok=True)
    (output / "assets").mkdir()

    frames, models, mask_count = [], set(), 0
    for record in records:
        entry, model = _frame(record, labels_run, masks_run, frames_dir, scene_id, output)
        frames.append(entry)
        models.add(model)
        mask_count += len(entry["instances"])
        print(f"ASSEMBLED sequence {entry['sequence_index']} "
              f"(frame {entry['source_frame_id']}): {len(entry['instances'])} masks", flush=True)
    if len(models) != 1:
        raise ValueError(f"Frames were labeled by more than one model: {sorted(models)}")

    points, colors = read_point_ply(backdrop)
    write_cloud_asset(output / "assets/background.js", "background", {
        "count": int(len(points)),
        "positions_f32_base64": encode(points, "<f4"),
        "colors_u8_base64": encode(colors, "u1"),
    })

    # The viewer frames its camera on these. No lifted point exists yet, so they
    # describe the backdrop -- the extent a lift will place points inside.
    manifest = {
        "scene_id": scene_id,
        "method": "gpt",
        "model": models.pop(),
        "policy": "Frame-local IDs; overlaps preserved; no temporal fusion",
        "frame_count": len(frames),
        "mask_count": mask_count,
        "background_asset": "assets/background.js",
        "bounds": {"minimum": points.min(axis=0).tolist(),
                   "maximum": points.max(axis=0).tolist()},
        "frames": frames,
    }
    write_manifest(output, manifest)
    write_json(output / "assembly_report.json", {
        "policy": POLICY,
        "run": run,
        "perception": str(perception),
        "labels_config": str(labels_run / "config.json"),
        "labels_config_sha256": sha256_file(labels_run / "config.json"),
        "scene_id": scene_id,
        "frame_count": len(frames),
        "mask_count": mask_count,
        "backdrop": str(backdrop),
        "backdrop_sha256": sha256_file(backdrop),
        "backdrop_points": int(len(points)),
        "note": "Paths are relative to this directory; masks and RGB are referenced in "
                "place. lifting_policy is set by whichever lifting module consumes this.",
        "elapsed_seconds": round(time.monotonic() - started, 3),
    })
    print(json.dumps({"output": str(output), "frames": len(frames),
                      "masks": mask_count, "policy": POLICY}, indent=2), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="Perception experiment name")
    parser.add_argument("--output", type=Path, required=True, help="New source directory")
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES,
                        help="Parent directory of the scene's RGB frames")
    parser.add_argument("--backdrop", type=Path, default=DEFAULT_BACKDROP,
                        help="Scene cloud the viewer draws behind the lifted points")
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--count", type=int, default=None)
    args = parser.parse_args()
    if args.first < 0 or (args.count is not None and args.count < 1):
        parser.error("Require first >= 0 and count >= 1")
    build(args.run, args.output, args.frames_dir, args.backdrop, args.first, args.count)


if __name__ == "__main__":
    main()
