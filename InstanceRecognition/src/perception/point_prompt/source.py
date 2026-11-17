"""Turn a point-prompt run into a source the lifting and the harness read.

    python -m src.perception.point_prompt.source pack --run full100_v2 \\
        --output artifacts/sources/scene0000_00/gpt_point_full100_v2
    python -m src.geometry.amb3r_mask_source --source artifacts/sources/scene0000_00/gpt_point_full100_v2 \\
        --output artifacts/sources/scene0000_00/scene_amb3r_point_full100_v2 --conf 0.3
    python -m src.perception.point_prompt.source renumber artifacts/sources/scene0000_00/scene_amb3r_point_full100_v2

`pack` writes the manifest shape of the text-prompt source (gpt_sam31_uniform500_v1), with
the point-prompt masks and labels in place of the text-prompt ones, so amb3r_mask_source
needs no changes. `sequence_index` keeps the uniform-500 slot the AMB3R npz is indexed by
(read from the 100-frame upstream source: its frames are every fifth slot, so frame 56 is
slot 5), because the lifting needs it.

`renumber` then rewrites the lifted source to sequence_index 0..99, which the harness
requires (harness/observations.py refuses non-contiguous indices). npz files are renamed to
match; per-frame assets are named by source frame id and stay where they are.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from src import config

from .objects import run_dir_of

SCENE = "scene0000_00"
UPSTREAM = config.ARTIFACTS_ROOT / "sources/scene0000_00/gpt_sam31_uniform500_full_v1"


def pack(run_dir: Path, out: Path, upstream: Path = UPSTREAM) -> dict:
    if out.exists():
        raise SystemExit(f"{out} exists; move it away first")
    masks_dir = out / "masks"
    masks_dir.mkdir(parents=True)
    slot = {int(f["source_frame_id"]): int(f["sequence_index"])
            for f in json.loads((upstream / "manifest.json").read_text())["frames"]}
    frames, total, by_label = [], 0, {}
    for folder in sorted(run_dir.glob("frame_*")):
        if not (folder / "result.json").exists():
            continue                                            # frame not finished
        result = json.loads((folder / "result.json").read_text())
        fid = int(result["frame_id"])
        if fid not in slot:
            raise SystemExit(f"frame {fid} is not in the 100-frame source; its AMB3R slot is unknown")
        seq = slot[fid]
        instances = []
        for i, kept in enumerate(sorted(result["kept"], key=lambda k: k["plan_id"]), start=1):
            src = folder / kept["mask_file"]
            dst = masks_dir / f"{seq:04d}_{i:03d}_{kept['plan_id']}.png"
            shutil.copyfile(src, dst)
            mask = np.asarray(Image.open(src).convert("L")) > 0
            instances.append({
                "raw_candidate_id": i, "class_name": kept["label"], "class_id": 0, "prompt_index": i - 1,
                "sam_prompt": kept["label"], "sam_prompt_type": "points", "sam_score": round(float(kept["sam_score"]), 4),
                "mask_file": str(dst.relative_to(out)), "mask_pixels": int(mask.sum()), "instance_id": i,
                "plan_id": kept["plan_id"], "points_positive": len(kept["positive"]),
                "points_negative": len(kept["negative"]), "combined_from": kept.get("combined_from"),
                "why_one_unit": kept.get("why_one_unit"),
            })
            by_label[kept["label"]] = by_label.get(kept["label"], 0) + 1
        total += len(instances)
        frames.append({"sequence_index": seq, "source_frame_id": fid, "image_width": 1296, "image_height": 968,
                       "original_rgb_image": f"../../../frames/{SCENE}/frame_{fid:06d}.jpg", "instances": instances})
    frames.sort(key=lambda f: f["sequence_index"])
    if [f["sequence_index"] for f in frames] != [5 * i for i in range(len(frames))]:
        raise SystemExit("sequence_index does not match the upstream 100-frame slots")
    cfg = json.loads((run_dir / "config.json").read_text())
    manifest = {"scene_id": SCENE, "method": "gpt_point_prompt", "model": cfg["model"],
                "policy": "Frame-local IDs from GPT point prompts + SAM3 image model; overlaps preserved",
                "frame_count": len(frames), "mask_count": total, "prompt_sha256": cfg["prompt_sha256"],
                "labels": dict(sorted(by_label.items(), key=lambda kv: -kv[1])), "frames": frames}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(frames)} frames, {total} masks -> {out}")
    return manifest


def renumber(src: Path) -> None:
    src = src.resolve()
    manifest = json.loads((src / "manifest.json").read_text())
    frames = manifest["frames"]
    slots = [int(f["sequence_index"]) for f in frames]
    if slots == list(range(len(frames))):
        print("already contiguous; nothing to do")
        return
    if sorted(slots) != slots:
        raise SystemExit(f"frames are not in slot order: {slots[:5]}")
    # the new npz set is built beside the old one and swapped in whole
    staging = src / "_frames_renumbered"
    staging.mkdir()
    for i, frame in enumerate(frames):
        with np.load((src / frame["npz"]).resolve()) as z:
            data = {k: z[k] for k in z.files}
        data["sequence_index"] = np.int64(i)
        np.savez_compressed(staging / f"{i:04d}.npz", **data)
        frame["npz"] = f"frames/{i:04d}.npz"
        frame["sequence_index"] = i
        if not (src / frame["asset"]).exists():
            raise SystemExit(f"missing asset {frame['asset']}")
    shutil.rmtree(src / "frames")
    staging.rename(src / "frames")
    manifest["renumbered_from_slots"] = slots
    (src / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(frames)} frames renumbered 0..{len(frames) - 1}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack", help="run -> upstream mask source")
    p.add_argument("--run", required=True, help="experiment name (artifacts/experiments/<run>/perception/point_prompt)")
    p.add_argument("--run-dir", type=Path, default=None, help="read the run from here instead")
    p.add_argument("--output", type=Path, required=True)
    r = sub.add_parser("renumber", help="lifted source -> sequence_index 0..n-1, in place")
    r.add_argument("source", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "pack":
        out = a.output if a.output.is_absolute() else config.ROOT / a.output
        pack(a.run_dir or run_dir_of(a.run), out)
    else:
        renumber(a.source)


if __name__ == "__main__":
    main()
