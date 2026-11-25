"""Generate SAM 3D meshes for the objects of a handoff, or for one image and mask.

    ./scripts/run_generate.sh --handoff <handoff dir> [--objects obj_0018,obj_0032]
    ./scripts/run_generate.sh --image photo.jpg --mask mask.png [--label bottle]

A single image is wrapped as a one-object handoff first, so both inputs share
one output layout (see handoff_io). The model is loaded once and every queued
object runs in the same process; one object failing is recorded and the batch
moves on.
"""
import argparse
import os
from pathlib import Path
import sys
import time
import traceback

from .. import handoff_io as hio
from ..config import SAM3D_REPO

ROOT = hio.ROOT
REPO = SAM3D_REPO
POSE_KEYS = ("rotation", "translation", "scale", "downsample_factor")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--handoff", type=Path, help="Handoff directory containing manifest.json and objects.json")
    source.add_argument("--image", type=Path, help="Single RGB image; requires --mask")
    parser.add_argument("--mask", type=Path, help="Binary mask for --image (non-zero = object)")
    parser.add_argument("--label", help="Object label recorded for --image")
    parser.add_argument("--scene-id", help="Scene id for --image; default is the image name plus a content hash")
    parser.add_argument("--objects", help="Comma-separated object_id list; default is every non-structural object")
    parser.add_argument("--view", choices=hio.ROLES, default="primary", help="Which key frame of each object to use")
    parser.add_argument("--include-structural", action="store_true", help="Also generate walls, floors and similar")
    parser.add_argument("--output", type=Path, help="Output root; default output/<scene_id>/<handoff name>")
    parser.add_argument("--force", action="store_true", help="Regenerate even when a matching result exists")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without loading the model or writing")
    parser.add_argument("--config", type=Path, default=hio.ROOT / ".runtime_config/sam3d/pipeline.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low-vram", action="store_true", help="Offload inactive model stages to CPU")
    parser.add_argument("--sequential-condition", action="store_true",
                        help="With --low-vram, load the two condition embedders one at a time")
    args = parser.parse_args(argv)
    if args.image is not None and args.mask is None:
        parser.error("--image requires --mask")
    if args.handoff is not None and (args.mask or args.label or args.scene_id):
        parser.error("--mask, --label and --scene-id only apply to --image")
    if args.sequential_condition and not args.low_vram:
        parser.error("--sequential-condition requires --low-vram")
    args.only = {item.strip() for item in args.objects.split(",") if item.strip()} if args.objects else None
    return args


def plain(value):
    """Turn tensors and arrays from the SAM3D result into JSON values."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def load_model(args):
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "notebook"))
    os.chdir(REPO)
    from inference import Inference
    config = str(args.config.resolve())
    if not args.low_vram:
        return Inference(config, compile=False)
    from .sam3d_offload import enable_stage_offload, load_on_cpu
    with load_on_cpu():
        pipeline = Inference(config, compile=False)
    enable_stage_offload(pipeline._pipeline, sequential_condition=args.sequential_condition)
    return pipeline


def clear_stage(directory: Path):
    """Remove every output before regenerating, so a failed run can never leave
    an old mesh sitting next to a new report."""
    for name in (*hio.OUTPUTS, "report.json"):
        (directory / name).unlink(missing_ok=True)
    directory.mkdir(parents=True, exist_ok=True)


def generate_one(pipeline, task, facts, key, args):
    import numpy as np
    import torch
    from PIL import Image
    directory = task["stage_dir"]
    clear_stage(directory)
    report = {"object_id": task["object_id"], "label": task["label"], "view": task["role"],
              "status": "running", "source": task["source"], "inputs": facts,
              "generator_key": key, "started_at": hio.now()}
    hio.atomic_json(directory / "report.json", report)
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        image = np.asarray(Image.open(task["rgb"]).convert("RGB"))
        mask = np.asarray(Image.open(task["mask"]).convert("L")) > 0
        result = pipeline(image, mask, seed=args.seed)
        mesh = result["glb"]
        mesh.export(directory / "mesh.glb")
        result["gaussian"][0].save_ply(str(directory / "splat.ply"))
        pose = {name: plain(result[name]) for name in POSE_KEYS if name in result}
        hio.atomic_json(directory / "pose.json", {"frame": hio.FRAMES["pose"], "available": bool(pose), **pose})
        report.update(status="success", vertices=len(mesh.vertices), faces=len(mesh.faces),
                      watertight=bool(mesh.is_watertight), pose_available=bool(pose),
                      result_keys=sorted(result),
                      outputs={name: hio.sha256(directory / name) for name in hio.OUTPUTS})
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        report.update(seconds=round(time.monotonic() - start, 2), finished_at=hio.now(),
                      peak_vram_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3))
        hio.atomic_json(directory / "report.json", report)
        torch.cuda.empty_cache()
    return report


def main(argv=None):
    args = parse_args(argv)
    args.config = args.config.expanduser().resolve()
    if args.image is not None:
        if args.dry_run:
            raise SystemExit("--dry-run is only supported with --handoff")
        args.handoff = hio.write_single_image(args.image, args.mask, args.label, args.scene_id)
    handoff, manifest, objects = hio.load_handoff(args.handoff)
    root = (args.output.expanduser().resolve() if args.output else hio.default_output_root(handoff, manifest))
    tasks = hio.plan(handoff, objects, root, args.view, args.only, args.include_structural)
    key = {"config_sha256": hio.sha256(args.config), "seed": args.seed}
    generator = {**key, "config": str(args.config), "low_vram": args.low_vram,
                 "sequential_condition": args.sequential_condition}

    queue, reused, rejected = [], [], []
    for task in tasks:
        if not task["selected"] or "skip" in task:
            continue
        reason, facts = hio.inspect_inputs(task["rgb"], task["mask"])
        if reason:
            rejected.append((task, reason, facts))
        elif not args.force and hio.up_to_date(task["stage_dir"], facts, key):
            reused.append(task)
        else:
            queue.append((task, facts))
    skipped = sum(task["selected"] and "skip" in task for task in tasks)
    print(f"PLAN handoff={handoff} output={root}", flush=True)
    print(f"PLAN objects={len(tasks)} generate={len(queue)} reuse={len(reused)} "
          f"bad_input={len(rejected)} skipped={skipped} not_selected={sum(not t['selected'] for t in tasks)}",
          flush=True)
    for task, reason, _ in rejected:
        print(f"BAD_INPUT {task['object_id']} {reason}", flush=True)
    if args.dry_run:
        for task, _ in queue:
            print(f"WOULD_GENERATE {task['object_id']} {task['label']} {task['source']['rgb']}", flush=True)
        return 0

    for task, reason, facts in rejected:
        clear_stage(task["stage_dir"])
        hio.atomic_json(task["stage_dir"] / "report.json",
                        {"object_id": task["object_id"], "label": task["label"], "view": task["role"],
                         "status": "skipped", "reason": reason, "source": task["source"], "inputs": facts,
                         "finished_at": hio.now()})
    hio.write_index(root, handoff, manifest, tasks, generator)
    if not queue:
        print("GENERATE_DONE nothing to generate", flush=True)
        return 0

    load_start = time.monotonic()
    pipeline = load_model(args)
    print(f"MODEL_LOADED {time.monotonic() - load_start:.1f}s", flush=True)
    failures = 0
    for number, (task, facts) in enumerate(queue, 1):
        print(f"GENERATE {number}/{len(queue)} {task['object_id']} {task['label']}", flush=True)
        report = generate_one(pipeline, task, facts, key, args)
        failures += report["status"] != "success"
        print(f"RESULT {task['object_id']} {report['status']} {report['seconds']}s "
              f"{report.get('error', '')}".rstrip(), flush=True)
        hio.write_index(root, handoff, manifest, tasks, generator)
    summary = hio.write_index(root, handoff, manifest, tasks, generator)
    print("GENERATE_DONE", summary["status_counts"], root / "objects.json", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
