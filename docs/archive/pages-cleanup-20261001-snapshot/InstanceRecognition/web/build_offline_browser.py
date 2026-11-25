#!/usr/bin/env python3
"""Build a file:// compatible browser for completed inference results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def relative_url(path: Path, output_dir: Path) -> str:
    return Path(os.path.relpath(path.resolve(), output_dir.resolve())).as_posix()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Offline browser artifact is missing: {path}")
    return path


def build_index(results_root: Path, output_dir: Path) -> dict:
    results_root = results_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    runtime_root = results_root.parent
    scenes = []
    total_instances = 0

    for scene_root in sorted(path for path in results_root.iterdir() if path.is_dir()):
        frames = []
        summaries = sorted(
            scene_root.glob("frame_*/summary.json"),
            key=lambda path: int(path.parent.name.removeprefix("frame_")),
        )
        for summary_path in summaries:
            summary = json.loads(summary_path.read_text())
            scene_id = str(summary["scene_id"])
            frame_index = int(summary["frame_index"])
            if scene_id != scene_root.name:
                raise ValueError(f"Scene mismatch in {summary_path}")

            run_root = summary_path.parent
            source = require_file(
                runtime_root / "frames" / scene_id / f"frame_{frame_index:06d}.jpg"
            )
            yoloe_image = require_file(run_root / "yoloe/annotated.jpg")
            yoloe_json = require_file(run_root / "yoloe/detections.json")
            sam_image = require_file(run_root / "sam31/annotated.jpg")
            sam_json = require_file(run_root / "sam31/predictions.json")
            instance_map = require_file(run_root / "sam31/instance_id_map.png")

            instances = []
            for item in summary["sam31"]["instances"]:
                mask = require_file(run_root / "sam31" / item["mask_file"])
                instances.append(
                    {
                        "id": int(item["instance_id"]),
                        "class_name": str(item["class_name"]),
                        "confidence": float(item["yoloe_confidence"]),
                        "prompt": str(item["sam_prompt_type"]),
                        "iou": float(item["prompt_box_iou"]),
                        "mask_pixels": int(item["mask_pixels"]),
                        "mask": relative_url(mask, output_dir),
                    }
                )

            yoloe = summary["yoloe"]
            sam31 = summary["sam31"]
            total_instances += len(instances)
            frames.append(
                {
                    "frame_index": frame_index,
                    "wall_seconds": float(summary["wall_seconds"]),
                    "source": relative_url(source, output_dir),
                    "yoloe_image": relative_url(yoloe_image, output_dir),
                    "sam_image": relative_url(sam_image, output_dir),
                    "instance_map": relative_url(instance_map, output_dir),
                    "yoloe_json": relative_url(yoloe_json, output_dir),
                    "sam_json": relative_url(sam_json, output_dir),
                    "instance_count": int(sam31["instance_count"]),
                    "classes": list(yoloe["unique_classes"]),
                    "yoloe_seconds": float(yoloe["timing"]["inference_seconds"]),
                    "sam_seconds": float(
                        sam31["timing"].get("session_and_prompts_seconds", 0.0)
                    ),
                    "instances": instances,
                }
            )
        if frames:
            scenes.append(
                {
                    "scene_id": scene_root.name,
                    "frame_count": len(frames),
                    "first_frame": frames[0]["frame_index"],
                    "last_frame": frames[-1]["frame_index"],
                    "frames": frames,
                }
            )

    if not scenes:
        raise ValueError(f"No completed inference results found under {results_root}")
    return {
        "scene_count": len(scenes),
        "frame_count": sum(scene["frame_count"] for scene in scenes),
        "instance_count": total_instances,
        "scenes": scenes,
    }


def write_data(index: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "window.OFFLINE_RESULTS_INDEX = " + json.dumps(
        index, ensure_ascii=False, separators=(",", ":")
    ) + ";\n"
    temporary = output_path.with_suffix(".js.tmp")
    temporary.write_text(payload)
    os.replace(temporary, output_path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=root / "runtime/results")
    parser.add_argument("--output-dir", type=Path, default=root / "offline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = build_index(args.results_root, args.output_dir)
    output = args.output_dir / "data.js"
    write_data(index, output)
    print(
        f"OFFLINE_INDEX_OK scenes={index['scene_count']} "
        f"frames={index['frame_count']} instances={index['instance_count']} "
        f"output={output}"
    )


if __name__ == "__main__":
    main()
