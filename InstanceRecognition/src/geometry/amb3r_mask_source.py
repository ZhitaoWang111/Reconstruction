"""Build a Harness mask-projection source from AMB3R per-frame point maps.

Each pixel in AMB3R's output already has a predicted 3D position; the depth is
from the neural network rather than from a GT mesh. That is what makes this the
one lift in this tree: a GT mesh has surface-reconstruction gaps (black chair
backs, thin structures, regions scanned at a bad angle), and a mask pixel there
projects onto whatever geometry happens to be behind it instead of the surface
the camera actually saw.

The lifting direction is:

    mask pixel → AMB3R depth at that pixel → one 3D point

One pixel yields at most one point, so there is no multi-ray ambiguity and no
occlusion leakage. The cost is that the depth estimate is noisier than a GT
mesh, especially in low-texture or symmetric regions.

Usage::

    python -m src.geometry.amb3r_mask_source \\
      --npz  Video2PointCloud/output/amb3r/vo_uniform500/scene_*.npz \\
      --sim3 Video2PointCloud/output/amb3r/vo_uniform500/pointcloud_gt_corrected_evaluation.json \\
      --source <upstream source with masks> \\
      --output artifacts/sources/scene0000_00/scene_amb3r_v1 \\
      --conf  0.3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src import config
from src.inference.sam31_segment import DEFAULT_MIN_MASK_AREA_FRACTION
from src.geometry.backproject_masks import (
    align_mask,
    crop_box,
    encode,
    membership_bits,
    read_point_ply,
    relative,
    sha256_file,
    write_cloud_asset,
    write_json,
)

POLICY = "amb3r_per_frame_v1"

# AMB3R's internal resolution after its own center-crop pipeline.
AMBR3R_SIZE = (518, 392)          # width, height
AMBR3R_INTERMEDIATE = (640, 480)  # intermediate resize before crop

# Frame-level paths carried in from the upstream source and re-expressed against
# this output. `original_rgb_image` is deliberately absent: upstream records the
# pre-refactor RGB location, so it is taken from `frames_dir` instead.
FRAME_PATH_KEYS = ("predictions_json", "rgb_image", "annotated_image")

DEFAULT_NPZ = (
    config.WORKSPACE
    / "Video2PointCloud/output/amb3r/vo_uniform500"
    / "scene_scene0000_00_uniform500_metric_head_results.npz"
)
DEFAULT_SIM3 = (
    config.WORKSPACE
    / "Video2PointCloud/output/amb3r/vo_uniform500"
    / "pointcloud_gt_corrected_evaluation.json"
)
DEFAULT_SOURCE = (
    config.ARTIFACTS_ROOT
    / "sources/scene0000_00/gpt_sam31_uniform500_v1"
)
DEFAULT_OUTPUT = (
    config.ARTIFACTS_ROOT
    / "sources/scene0000_00/scene_amb3r_v1"
)

# Where the scene's RGB frames live. The upstream mask source records an RGB path
# from before the refactor, which no longer exists, so the frames are located here
# instead -- the same directory the assembler reads them from.
DEFAULT_FRAMES = config.ARTIFACTS_ROOT / "frames"


def _load_sim3(sim3_path: Path) -> tuple[dict, Path]:
    """The Sim(3) to apply, plus the cloud the viewer draws behind the points.

    The backdrop has to be the cloud this Sim(3) was fitted against, or the two
    sit in different coordinate frames in the viewer. It is read from the same
    evaluation file that supplies the transform, so the two cannot be picked
    independently.
    """
    evaluation = json.loads(sim3_path.read_text())
    backdrop = Path(evaluation.get("sim3_ply", ""))
    if not backdrop.is_file():
        raise SystemExit(f"Missing canonical AMB3R cloud: {backdrop}")
    return evaluation["sim3_oracle_scale"], backdrop


def _sample_rgb(rgb_path: Path, pixel_indices: np.ndarray) -> np.ndarray:
    """The RGB frame's pixels at the given grid positions, in AMB3R's own grid.

    The transform is deliberately the same resize -> crop -> resize that
    ``align_mask`` applies to the masks: if the two disagreed by even the crop
    offset, a point would carry the colour of a pixel a few columns away from the
    mask that claimed it.
    """
    with Image.open(rgb_path) as image:
        frame = image.convert("RGB")
        resized = frame.resize(AMBR3R_INTERMEDIATE, Image.Resampling.BILINEAR)
        cropped = resized.crop(crop_box(AMBR3R_INTERMEDIATE, AMBR3R_SIZE))
        grid = cropped.resize(AMBR3R_SIZE, Image.Resampling.BILINEAR)
    return np.asarray(grid).reshape(-1, 3)[pixel_indices].copy()


def _apply_sim3(points: np.ndarray, transform: dict) -> np.ndarray:
    """Apply Sim(3): P_world = s * R @ P_amb3r + t."""
    R = np.asarray(transform["alignment_rotation"], dtype=np.float64)
    t = np.asarray(transform["alignment_translation_m"], dtype=np.float64)
    s = float(transform["alignment_scale"])
    if not np.isfinite(s) or s <= 0:
        raise ValueError("Sim(3) scale must be positive and finite")
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-5):
        raise ValueError("Sim(3) rotation is not orthogonal")
    return (s * (points.astype(np.float64) @ R.T) + t).astype(np.float32)


def _apply_sim3_to_pose(pose: np.ndarray, transform: dict) -> np.ndarray:
    """Transform a camera-to-world 4x4 from AMB3R coords into the scan frame.

    The Sim(3) scale corrects distances, not orientations, so the rotation
    block is re-rotated but not rescaled; only the translation takes the scale.

        R_scan = R_sim3 @ R_amb3r
        t_scan = s * (R_sim3 @ t_amb3r) + t_sim3
    """
    R = np.asarray(transform["alignment_rotation"], dtype=np.float64)
    t = np.asarray(transform["alignment_translation_m"], dtype=np.float64)
    s = float(transform["alignment_scale"])
    pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R @ pose[:3, :3]
    out[:3, 3] = s * (R @ pose[:3, 3]) + t
    return out.astype(np.float32)


def _load_intrinsic_color(sim3_path: Path) -> list | None:
    """Read the RGB intrinsics from the ScanNet extraction manifest.

    The extraction manifest sits at ``<Video2PointCloud>/data/scannet/<scene>/manifest.json``.
    The VO output directory is ``<Video2PointCloud>/output/amb3r/...``, so the
    project root is four levels up from the sim3 evaluation file. None is
    returned when the manifest is not found so the build continues without
    intrinsics rather than failing.
    """
    # The sim3 file is at <root>/output/amb3r/<run>/pointcloud_gt_corrected_evaluation.json
    project_root = sim3_path.resolve().parents[3]
    candidates = sorted(project_root.glob("data/scannet/*/manifest.json"))
    for candidate in candidates:
        try:
            data = json.loads(candidate.read_text())
            val = data.get("intrinsic_color")
            if val:
                return val
        except Exception:
            continue
    return None


def build(
    npz_path: Path,
    sim3_path: Path,
    source: Path,
    output: Path,
    conf_threshold: float = 0.3,
    first: int = 0,
    count: int | None = None,
    frames_dir: Path = DEFAULT_FRAMES,
    min_area_fraction: float = DEFAULT_MIN_MASK_AREA_FRACTION,
) -> Path:
    npz_path, sim3_path, source, output = (
        Path(p).resolve() for p in (npz_path, sim3_path, source, output)
    )
    frames_dir = Path(frames_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "frames").mkdir()
    (output / "assets").mkdir()

    transform, backdrop = _load_sim3(sim3_path)
    source_manifest = json.loads((source / "manifest.json").read_text())
    frames = source_manifest["frames"]
    if count is not None:
        frames = frames[first: first + count]
    else:
        frames = frames[first:]

    # Load per-frame point maps once; the file is 1.6 GB so we stream it with
    # mmap_mode so only the slices we touch are paged in.
    amb3r = np.load(str(npz_path), mmap_mode="r")
    pts_all = amb3r["pts"]    # (N, 392, 518, 3), AMB3R coordinates
    conf_all = amb3r["conf"]  # (N, 392, 518)
    pose_all = amb3r["pose"]  # (N, 4, 4), camera-to-world in AMB3R coords

    out_frames = []
    total_points = 0
    total_masks = 0
    total_dropped = 0

    for frame in frames:
        seq = int(frame["sequence_index"])
        src_id = int(frame["source_frame_id"])

        # pts_all[seq] is the point map for this sequence slot.
        pts_frame = np.array(pts_all[seq])   # (392, 518, 3)
        conf_frame = np.array(conf_all[seq]) # (392, 518)

        # Masks segmented before SAM gained its area filter still carry the
        # small ones; drop them here with the same rule so a source lifted from
        # old perception output matches what a fresh SAM run would have kept.
        # The rule is applied to the original-resolution pixel count, as in SAM.
        all_instances = frame.get("instances") or []
        frame_area = int(frame.get("image_width") or 1296) * int(frame.get("image_height") or 968)
        instances = [item for item in all_instances
                     if item["mask_pixels"] >= min_area_fraction * frame_area]
        total_dropped += len(all_instances) - len(instances)
        masks_aligned = []
        for item in instances:
            mask_path = (source / item["mask_file"]).resolve()
            raw = np.asarray(Image.open(mask_path).convert("L")) > 0
            aligned = align_mask(raw, AMBR3R_INTERMEDIATE, AMBR3R_SIZE)
            masks_aligned.append(aligned)

        bits = membership_bits(masks_aligned, (AMBR3R_SIZE[1], AMBR3R_SIZE[0]))

        # Collect pixels that (a) belong to at least one mask and
        # (b) have sufficient confidence.
        claimed_flat = bits.reshape(-1) > 0
        conf_flat = conf_frame.reshape(-1)
        valid = claimed_flat & (conf_flat >= conf_threshold)
        pixel_indices = np.flatnonzero(valid).astype(np.int64)

        # Pull the 3D positions and apply GT Sim(3).
        pts_flat = pts_frame.reshape(-1, 3)
        local_pts = pts_flat[pixel_indices]          # (K, 3), AMB3R coords
        world_pts = _apply_sim3(local_pts, transform) # (K, 3), GT world coords

        bits_flat = bits.reshape(-1)[pixel_indices].astype(np.uint32)

        # Colors come from the RGB frame the masks were drawn on, sampled through
        # the same resize-then-crop the masks went through, so a point's colour is
        # the pixel the mask actually claimed rather than an invented hue. The
        # Harness shows these points to the model, so a synthetic colour there is
        # evidence the model reads but the image does not support. The frame is
        # located through `frames_dir`, not the frame's recorded path, which is the
        # pre-refactor location.
        rgb_frame = frames_dir / source_manifest["scene_id"] / f"frame_{src_id:06d}.jpg"
        rgb = _sample_rgb(rgb_frame, pixel_indices)

        # Build per-instance statistics for the manifest.
        inst_meta = []
        for i, item in enumerate(instances):
            bit = np.uint32(1 << i)
            pt_count = int(((bits_flat & bit) != 0).sum())
            inst_meta.append({
                **{k: v for k, v in item.items()},
                "point_count": pt_count,
                "nonfinite_points_skipped": 0,
            })
            total_masks += 1

        # The AMB3R pose for this frame, transformed into the scan coordinate
        # frame the points live in. Downstream uses it for back-projection and
        # viewpoint comparison without re-running the Sim(3) fit.
        pose_frame = np.array(pose_all[seq]).reshape(4, 4)
        camera_to_world = _apply_sim3_to_pose(pose_frame, transform)

        # Write NPZ (the schema the Harness and the page builders read).
        np.savez_compressed(
            output / "frames" / f"{seq:04d}.npz",
            points_world=world_pts,
            rgb=rgb,
            mask_bits=bits_flat,
            pixel_indices=pixel_indices,
            grid_shape_hw=np.array([AMBR3R_SIZE[1], AMBR3R_SIZE[0]], dtype=np.int32),
            sequence_index=np.int64(seq),
            camera_to_world=camera_to_world,
        )

        # Write JS asset (used by viewer).
        asset_key = src_id
        write_cloud_asset(
            output / "assets" / f"frame_{src_id:06d}.js",
            asset_key,
            {
                "source_frame_id": src_id,
                "count": len(world_pts),
                "width": AMBR3R_SIZE[0],
                "height": AMBR3R_SIZE[1],
                "original_size": [
                    int(frame.get("image_width") or 1296),
                    int(frame.get("image_height") or 968),
                ],
                "intermediate_size": list(AMBR3R_INTERMEDIATE),
                "crop_box": list(crop_box(AMBR3R_INTERMEDIATE, AMBR3R_SIZE)),
                "positions_f32_base64": encode(world_pts, "<f4"),
                "colors_u8_base64": encode(rgb, "u1"),
                "mask_bits_u32_base64": encode(bits_flat.astype("<u4"), "<u4"),
                # The result page decodes this alongside mask_bits to map each
                # point back to the pixel it came from. Omitting it is not a
                # degraded view: `atob(undefined)` throws and the whole frame —
                # overlay, cloud and object list — fails to render.
                "pixel_indices_u32_base64": encode(
                    pixel_indices.astype("<u4"), "<u4"),
            },
        )

        out_frame = {k: v for k, v in frame.items() if k != "instances"}
        out_frame["instances"] = inst_meta
        out_frame["point_count"] = len(world_pts)
        out_frame["membership_count"] = len(world_pts)
        out_frame["overlap_point_count"] = int(
            sum(1 for b in bits_flat if bin(b).count("1") > 1)
        )
        out_frame["nonfinite_points_skipped"] = 0
        # The frame's paths are carried over from upstream, but upstream is a
        # different directory: a path that was correct there does not resolve here,
        # and a mask the Harness cannot open is a source it cannot use. Re-express
        # each against this output. `original_rgb_image` is not among them in the
        # legacy input -- its recorded RGB path is the pre-refactor one -- so it is
        # taken from wherever the caller said the frames live.
        out_frame["original_rgb_image"] = relative(
            frames_dir / source_manifest["scene_id"] / f"frame_{src_id:06d}.jpg", output
        )
        for item in out_frame["instances"]:
            item["mask_file"] = relative((source / item["mask_file"]).resolve(), output)
        for key in FRAME_PATH_KEYS:
            if key in out_frame:
                out_frame[key] = relative((source / frame[key]).resolve(), output)
        out_frame["npz"] = relative(
            output / "frames" / f"{seq:04d}.npz", output
        )
        out_frame["asset"] = relative(
            output / "assets" / f"frame_{src_id:06d}.js", output
        )
        out_frames.append(out_frame)
        total_points += len(world_pts)
        print(
            f"PROJECTED frame {seq} ({src_id}): {len(world_pts)} points "
            f"from {int(valid.sum())} claimed pixels",
            flush=True,
        )

    # The viewer draws this behind the lifted points. The upstream manifest copied
    # a `background_asset` path here, but copying a path does not copy the file it
    # names, and the path pointed at the producing run rather than this output. So
    # the backdrop is encoded from the same cloud the Sim(3) came from: the two are
    # then guaranteed to be in one frame, and the asset travels with the source.
    backdrop_points, backdrop_colors = read_point_ply(backdrop)
    write_cloud_asset(output / "assets/background.js", "background", {
        "count": int(len(backdrop_points)),
        "positions_f32_base64": encode(backdrop_points, "<f4"),
        "colors_u8_base64": encode(backdrop_colors, "u1"),
    })

    manifest = {
        **{k: v for k, v in source_manifest.items() if k != "frames"},
        "lifting_policy": POLICY,
        "frame_count": len(out_frames),
        "frames": out_frames,
        "background_asset": "assets/background.js",
        # The lifted points sit on the backdrop's own surface, so the viewer frames
        # its camera on these rather than on the points of the sampled subset.
        "bounds": {
            "minimum": backdrop_points.min(axis=0).tolist(),
            "maximum": backdrop_points.max(axis=0).tolist(),
        },
        "conf_threshold": conf_threshold,
        "min_mask_area_fraction": min_area_fraction,
        "masks_dropped_by_area": total_dropped,
        "sim3_source": str(sim3_path),
        # RGB camera intrinsics (1296×968).  Stored here so downstream never has
        # to locate a separate manifest.  Read from the ScanNet extraction manifest
        # that lives beside the sim3 evaluation file; None if not found so the
        # build does not fail when that manifest is absent.
        "intrinsic_color": _load_intrinsic_color(sim3_path),
    }
    write_json(output / "manifest.json", manifest)

    report = {
        "policy": POLICY,
        "conf_threshold": conf_threshold,
        "backdrop": str(backdrop),
        "backdrop_sha256": sha256_file(backdrop),
        "backdrop_points": int(len(backdrop_points)),
        "frames": [
            {
                "sequence_index": f["sequence_index"],
                "source_frame_id": f["source_frame_id"],
                "points": f["point_count"],
                "claimed_pixels": int(
                    sum(i["point_count"] for i in f["instances"])
                ),
                "pixels_without_surface": 0,
                "overlap_points": f["overlap_point_count"],
            }
            for f in out_frames
        ],
    }
    write_json(output / "projection_report.json", report)
    print(
        json.dumps({"output": str(output), "policy": POLICY}, indent=2),
        flush=True,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--sim3", type=Path, default=DEFAULT_SIM3)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES,
                        help="Parent directory of the scene's RGB frames")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="AMB3R confidence threshold (default 0.3)")
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--min-area-fraction", type=float,
                        default=DEFAULT_MIN_MASK_AREA_FRACTION,
                        help="Drop masks below this fraction of the frame (0 keeps all)")
    args = parser.parse_args()
    build(args.npz, args.sim3, args.source, args.output,
          args.conf, args.first, args.count, args.frames_dir,
          args.min_area_fraction)


if __name__ == "__main__":
    main()
