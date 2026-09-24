#!/usr/bin/env python3
"""Run resumable GPT + SAM3.1 inference (or opt-in YOLOE baseline) over sampled ScanNet frames."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

from src import config
from src.inference.model_service import ModelServiceManager
from src.inference.pipeline import InstanceRecognitionPipeline
from src.data.sampling import uniform_sample_indices
from src.data.scannet_sens import ScanNetSensReader


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def wait_until_loaded(manager: ModelServiceManager, client_id: str) -> dict:
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        status = manager.heartbeat(client_id)
        errors = {
            name: value["last_error"]
            for name, value in status["workers"].items()
            if value["last_error"]
        }
        if errors:
            raise RuntimeError(f"Model preload failed: {errors}")
        if all(value["loaded"] for value in status["workers"].values()):
            return status
        time.sleep(1)
    raise TimeoutError("Models did not finish loading within 600 seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", default="scene0000_00")
    parser.add_argument("--target-frames", type=int, default=500)
    parser.add_argument("--force", action="store_true", help="ignore per-frame caches")
    parser.add_argument("--method", choices=("gpt_sam31", "yoloe_sam31"), default=config.DEFAULT_METHOD)
    parser.add_argument(
        "--runtime-root", type=Path, default=config.RUNTIME_ROOT
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime_root = args.runtime_root.expanduser().resolve()
    reader = ScanNetSensReader(config.SCANNET_ROOT, runtime_root)
    header = reader.header(args.scene_id)
    indices = uniform_sample_indices(header.frame_count, args.target_frames)
    batch_path = (
        runtime_root
        / ("batches_gpt" if args.method == "gpt_sam31" else "batches")
        / args.scene_id
        / f"target_{args.target_frames}.json"
    )
    manifest = {
        "scene_id": args.scene_id,
        "method": args.method,
        "source_frame_count": header.frame_count,
        "target_frames": args.target_frames,
        "sampled_frame_count": len(indices),
        "sampling": "uniform endpoints-inclusive",
        "frame_indices": indices,
        "completed": {},
        "failures": {},
        "status": "extracting",
    }
    if batch_path.is_file():
        previous = json.loads(batch_path.read_text())
        if previous.get("frame_indices") == indices:
            manifest["completed"] = previous.get("completed", {})
            manifest["failures"] = previous.get("failures", {})
    write_json_atomic(batch_path, manifest)

    print(
        f"SAMPLE scene={args.scene_id} source={header.frame_count} "
        f"selected={len(indices)} first={indices[0]} last={indices[-1]}",
        flush=True,
    )
    extract_started = time.perf_counter()
    reader.frame_paths(args.scene_id, indices)
    print(
        f"EXTRACT_OK seconds={time.perf_counter() - extract_started:.3f}", flush=True
    )

    manager = ModelServiceManager(runtime_root, method=args.method)
    pipeline = InstanceRecognitionPipeline(reader, runtime_root, manager, method=args.method)
    client_id = f"batch-{os.getpid()}"
    manager.open_client(client_id)
    failures = 0
    batch_started = time.perf_counter()
    try:
        status = wait_until_loaded(manager, client_id)
        load_times = {
            name: value["load_seconds"] for name, value in status["workers"].items()
        }
        print(f"MODELS_READY load_seconds={load_times}", flush=True)
        manifest["status"] = "running"
        manifest["model_load_seconds"] = load_times
        write_json_atomic(batch_path, manifest)

        for position, frame_index in enumerate(indices, start=1):
            manager.heartbeat(client_id)
            started = time.perf_counter()
            try:
                summary = pipeline.run(args.scene_id, frame_index, force=args.force)
                manifest["completed"][str(frame_index)] = {
                    "cached": summary.get("cached", False),
                    "wall_seconds": summary["wall_seconds"],
                    "instance_count": summary["sam31"]["instance_count"],
                    "unique_classes": summary["sam31"]["unique_classes"],
                }
                manifest["failures"].pop(str(frame_index), None)
                print(
                    f"[{position:03d}/{len(indices)}] frame={frame_index:04d} "
                    f"instances={summary['sam31']['instance_count']} "
                    f"cached={summary.get('cached', False)} "
                    f"seconds={time.perf_counter() - started:.3f}",
                    flush=True,
                )
            except Exception as error:
                failures += 1
                manifest["completed"].pop(str(frame_index), None)
                manifest["failures"][str(frame_index)] = str(error)
                print(
                    f"[{position:03d}/{len(indices)}] frame={frame_index:04d} "
                    f"FAILED {error}",
                    flush=True,
                )
            write_json_atomic(batch_path, manifest)
    finally:
        manager.close_client(client_id)
        manager.shutdown()

    class_counts = Counter()
    total_instances = 0
    for item in manifest["completed"].values():
        total_instances += item["instance_count"]
        class_counts.update(item["unique_classes"])
    manifest["status"] = "complete" if not manifest["failures"] else "complete_with_failures"
    manifest["batch_wall_seconds"] = round(time.perf_counter() - batch_started, 3)
    manifest["summary"] = {
        "successful_frames": len(manifest["completed"]),
        "failed_frames": len(manifest["failures"]),
        "total_instances": total_instances,
        "frames_containing_class": dict(class_counts.most_common()),
    }
    write_json_atomic(batch_path, manifest)
    print(
        f"BATCH_{'OK' if not manifest['failures'] else 'INCOMPLETE'} "
        f"success={len(manifest['completed'])} failures={len(manifest['failures'])} "
        f"manifest={batch_path}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
