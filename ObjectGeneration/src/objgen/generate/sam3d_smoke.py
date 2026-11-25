"""Run SAM 3D Objects with a mask, a text prompt, or confirmed interactive points."""
import argparse
import json
import os
import math
from pathlib import Path
import subprocess
import shutil
import hashlib
import sys
import time
import traceback

from ..layout import create_run

from ..config import ROOT, SAM3D_REPO, SAM31_CHECKPOINT, SAM31_PYTHON, SAM31_REPO, SAM3D_CONFIG
REPO = SAM3D_REPO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sample = REPO / "notebook/images/shutterstock_stylish_kidsroom_1640806567"
    parser.add_argument("--image", type=Path)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--mask", type=Path, help="Use an existing binary mask")
    inputs.add_argument("--prompt", help="Use SAM3.1 to segment this object (English recommended)")
    parser.add_argument("--sam31-python", type=Path, default=SAM31_PYTHON)
    parser.add_argument("--sam31-repo", type=Path, default=SAM31_REPO)
    parser.add_argument("--sam31-checkpoint", type=Path, default=SAM31_CHECKPOINT)
    parser.add_argument("--sam31-threshold", type=float, default=0.5)
    parser.add_argument("--sam31-instance-index", type=int, default=0, help="Zero-based index after sorting nonempty masks by descending score")
    parser.add_argument("--config", type=Path, default=SAM3D_CONFIG)
    parser.add_argument("--output", type=Path, default=ROOT / "output", help="Output root; mirrors assets category/object folders, then adds start timestamp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--low-vram", action="store_true", help="Offload inactive model stages to CPU")
    parser.add_argument("--sequential-condition", action="store_true", help="With --low-vram, offload condition encoders before loading generators")
    args = parser.parse_args()
    if args.sequential_condition and not args.low_vram:
        parser.error("--sequential-condition requires --low-vram")
    if args.prompt is not None and not args.prompt.strip():
        parser.error("--prompt must not be empty")
    if not math.isfinite(args.sam31_threshold) or not 0 <= args.sam31_threshold <= 1:
        parser.error("--sam31-threshold must be between 0 and 1")
    if args.sam31_instance_index < 0:
        parser.error("--sam31-instance-index must be nonnegative")
    interactive = args.image is not None and args.mask is None and args.prompt is None and not args.load_only
    sample_mask = args.image is None and args.mask is None and args.prompt is None
    args.image = (args.image or sample / "image.png").resolve()
    args.mask = args.mask.resolve() if args.mask is not None else (sample / "0.png" if sample_mask else None)
    args.config = args.config.resolve()
    original_image, original_mask = args.image, args.mask
    run_dir, started_at = create_run(args.output, args.image)
    args.output = run_dir / "sam3d"
    print("RUN_DIR", run_dir, flush=True)
    selection_mode = "points" if interactive else "text" if args.prompt is not None else "mask"
    report = {"status": "running", "image": str(args.image), "mask": str(args.mask) if args.mask else None, "prompt": args.prompt, "mask_source": "sam3.1_points" if interactive else "sam3.1" if args.prompt is not None else "file", "selection_mode": selection_mode, "load_only": args.load_only, "config": str(args.config), "seed": args.seed, "low_vram": args.low_vram, "sequential_condition": args.sequential_condition}
    report.update(run_dir=str(run_dir), started_at=started_at, stage="input_snapshot")
    inputs = {
        "started_at": started_at, "timezone": "Asia/Shanghai", "image_name": original_image.stem,
        "source_image": str(original_image), "source_mask": str(original_mask) if original_mask else None,
        "prompt": args.prompt, "seed": args.seed, "load_only": args.load_only,
        "selection_mode": selection_mode,
        "low_vram": args.low_vram, "sequential_condition": args.sequential_condition,
        "sam31_threshold": args.sam31_threshold, "sam31_instance_index": args.sam31_instance_index,
        "config_source": str(args.config), "argv": sys.argv[1:],
    }
    start = time.monotonic()
    try:
        if not args.load_only:
            args.image = run_dir / "input" / ("rgb" + original_image.suffix)
            shutil.copy2(original_image, args.image)
            inputs.update(image=str(args.image), image_sha256=hashlib.sha256(args.image.read_bytes()).hexdigest())
            if original_mask is not None:
                args.mask = run_dir / "input" / ("mask" + original_mask.suffix)
                shutil.copy2(original_mask, args.mask)
                inputs.update(mask=str(args.mask), mask_sha256=hashlib.sha256(args.mask.read_bytes()).hexdigest())
            report.update(image=str(args.image), mask=str(args.mask) if args.mask else None)
        (run_dir / "input/input.json").write_text(json.dumps(inputs, indent=2) + "\n")
        # Finish the separate SAM3.1 process before importing/loading SAM3D on CUDA.
        if (args.prompt is not None or interactive) and not args.load_only:
            if not args.image.is_file():
                raise FileNotFoundError(args.image)
            segmentation = args.output / "sam31"
            segmentation.mkdir(parents=True, exist_ok=True)
            segmentation_report = segmentation / "segmentation.json"
            report.update(stage="segmentation", segmentation_report=str(segmentation_report))
            command = [
                # Do not resolve this symlink: Python needs the venv executable path.
                str(args.sam31_python.expanduser().absolute()), "-u", "-X", "faulthandler",
                str(Path(__file__).with_name("sam31_click.py" if interactive else "sam31_mask.py")),
                "--image", str(args.image),
                "--output", str(segmentation), "--repo", str(args.sam31_repo.expanduser().resolve()),
                "--checkpoint", str(args.sam31_checkpoint.expanduser().resolve()),
            ]
            if not interactive:
                command += ["--prompt", args.prompt, "--threshold", str(args.sam31_threshold),
                            "--instance-index", str(args.sam31_instance_index)]
            print("SAM31_INTERACTIVE_START" if interactive else "SAM31_SEGMENTING", args.prompt or "点击选物", flush=True)
            segment_start = time.monotonic()
            with (segmentation / "run.log").open("w") as log:
                if interactive:
                    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                          text=True, errors="replace") as process:
                        for line in process.stdout:
                            log.write(line)
                            log.flush()
                            if line.startswith("SAM31_INTERACTIVE_"):
                                print(line, end="", flush=True)
                        returncode = process.wait()
                else:
                    returncode = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
            report["segmentation_seconds"] = time.monotonic() - segment_start
            if returncode:
                raise RuntimeError(f"SAM3.1 failed or selection cancelled (exit {returncode}); see {segmentation / 'run.log'}")
            metadata = json.loads(segmentation_report.read_text())
            if metadata.get("status") != "success":
                raise RuntimeError(f"SAM3.1 did not produce a valid mask; see {segmentation_report}")
            if interactive and metadata.get("user_confirmed") is not True:
                raise RuntimeError("Interactive mask was not confirmed; SAM3D will not start")
            args.mask = Path(metadata["mask"])
            report.update(mask=str(args.mask), sam31_selected=metadata["selected"])
            if interactive:
                report["user_confirmed"] = True
                inputs.update(points=metadata["points"], confirmed_at=metadata["confirmed_at"],
                              mask=str(args.mask), mask_sha256=metadata["mask_sha256"])
            print("SAM31_MASK_READY", args.mask, flush=True)
        report["stage"] = "input_validation"
        import numpy as np
        from PIL import Image
        if not args.load_only:
            image = np.asarray(Image.open(args.image).convert("RGB"))
            mask = np.asarray(Image.open(args.mask).convert("L")) > 0
            if image.shape[:2] != mask.shape:
                raise ValueError("Image and mask dimensions must match")
            if not mask.any():
                raise ValueError("Object mask is empty")
        report["stage"] = "model_loading"
        load_start = time.monotonic()
        import torch
        report.update(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(), cpu_affinity=sorted(os.sched_getaffinity(0)))
        print(json.dumps(report), flush=True)
        sys.path.insert(0, str(REPO))
        sys.path.insert(0, str(REPO / "notebook"))
        os.chdir(REPO)
        from inference import Inference
        if args.low_vram:
            from .sam3d_offload import load_on_cpu, enable_stage_offload
            with load_on_cpu():
                pipeline = Inference(str(args.config.resolve()), compile=False)
            enable_stage_offload(pipeline._pipeline, sequential_condition=args.sequential_condition)
        else:
            pipeline = Inference(str(args.config.resolve()), compile=False)
        report["load_seconds"] = time.monotonic() - load_start
        print("MODEL_LOADED", flush=True)
        if not args.load_only:
            report["stage"] = "generation"
            result = pipeline(image, mask, seed=args.seed)
            mesh = result["glb"]
            mesh.export(args.output / "mesh.glb")
            result["gaussian"][0].save_ply(str(args.output / "splat.ply"))
            report.update(vertices=len(mesh.vertices), faces=len(mesh.faces), watertight=bool(mesh.is_watertight))
        report["status"] = "success"
        report["stage"] = "complete"
        report["peak_vram_gib"] = torch.cuda.max_memory_allocated() / 2**30
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        (run_dir / "input/input.json").write_text(json.dumps(inputs, indent=2) + "\n")
        report["elapsed_seconds"] = time.monotonic() - start
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
