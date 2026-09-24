#!/usr/bin/env python3
"""Local ScanNet frame browser with GPT + SAM3.1 inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

from src import config
from src.model_service import ModelServiceManager
from src.pipeline import InferenceFailure, InstanceRecognitionPipeline
from src.scannet_sens import ScanNetSensReader


def create_app(
    scans_root: Path = config.SCANNET_ROOT,
    runtime_root: Path = config.RUNTIME_ROOT,
    model_service: ModelServiceManager | None = None,
    method: str = config.DEFAULT_METHOD,
) -> Flask:
    app = Flask(__name__, static_folder=None)
    runtime_root = Path(runtime_root).resolve()
    reader = ScanNetSensReader(scans_root, runtime_root)
    model_service = model_service or ModelServiceManager(runtime_root, method=method)
    pipeline = InstanceRecognitionPipeline(reader, runtime_root, model_service, method=method)
    app.extensions["model_service"] = model_service

    @app.get("/")
    def index():
        return send_from_directory(Path(__file__).resolve().parent, "index.html")

    @app.get("/assets/<path:filename>")
    def static_asset(filename: str):
        return send_from_directory(Path(__file__).resolve().parent, filename)

    @app.get("/api/health")
    def health():
        dependencies = {
            "scannet": Path(scans_root).is_dir(),
            "yoloe_python": config.YOLOE_PYTHON.is_file(),
            "yoloe_checkpoint": config.YOLOE_CHECKPOINT.is_file(),
            "sam31_python": config.SAM31_PYTHON.is_file(),
            "sam31_checkpoint": config.SAM31_CHECKPOINT.is_file(),
            "sam3_source": (config.SAM3_REPO / "sam3/model_builder.py").is_file(),
        }
        if method == "gpt_sam31":
            dependencies.pop("yoloe_python")
            dependencies.pop("yoloe_checkpoint")
            dependencies["gpt_client"] = (config.GPT_API_DIR / "label_frames.py").is_file()
        return jsonify(
            {
                "method": method,
                "ok": all(dependencies.values()),
                "dependencies": dependencies,
                "models": model_service.status(),
            }
        )

    @app.post("/api/client/open")
    def open_client():
        body = request.get_json(silent=True) or {}
        return jsonify(model_service.open_client(str(body.get("client_id", ""))))

    @app.post("/api/client/heartbeat")
    def client_heartbeat():
        body = request.get_json(silent=True) or {}
        return jsonify(model_service.heartbeat(str(body.get("client_id", ""))))

    @app.post("/api/client/close")
    def close_client():
        body = request.get_json(silent=True) or {}
        return jsonify(model_service.close_client(str(body.get("client_id", ""))))

    @app.get("/api/models/status")
    def model_status():
        return jsonify(model_service.status())

    @app.post("/api/models/unload")
    def unload_models():
        model_service.unload_all()
        return jsonify(model_service.status())

    @app.get("/api/scenes")
    def scenes():
        values = reader.list_scenes()
        return jsonify({"count": len(values), "scenes": values})

    @app.get("/api/frame/<scene_id>/<int:frame_index>.jpg")
    def frame(scene_id: str, frame_index: int):
        return send_file(
            reader.frame_path(scene_id, frame_index),
            mimetype="image/jpeg",
            max_age=86400,
            conditional=True,
        )

    @app.get("/api/run/<scene_id>/<int:frame_index>")
    def existing_run(scene_id: str, frame_index: int):
        reader.header(scene_id)
        run_root = pipeline.result_root(scene_id, frame_index)
        summary_path = run_root / "summary.json"
        if not summary_path.is_file():
            return jsonify({"available": False})
        summary = json.loads(summary_path.read_text())
        pipeline._validate(summary, run_root)
        summary["cached"] = True
        return jsonify({"available": True, **public_summary(summary)})

    @app.post("/api/infer")
    def infer():
        body = request.get_json(silent=True) or {}
        scene_id = str(body.get("scene_id", ""))
        try:
            frame_index = int(body.get("frame_index"))
        except (TypeError, ValueError):
            raise ValueError("frame_index must be an integer")
        summary = pipeline.run(scene_id, frame_index, force=bool(body.get("force", False)))
        return jsonify(public_summary(summary))

    @app.get("/api/artifact/<scene_id>/<int:frame_index>/<stage>/<filename>")
    def artifact(scene_id: str, frame_index: int, stage: str, filename: str):
        reader.header(scene_id)
        if stage not in {"yoloe", "sam31", "gpt"} or filename not in {
            "annotated.jpg",
            "detections.json",
            "labels.json",
            "predictions.json",
            "instance_id_map.png",
        }:
            raise ValueError("Artifact is not exposed")
        root = pipeline.result_root(scene_id, frame_index) / stage
        return send_from_directory(root, filename, conditional=True, max_age=0)

    @app.errorhandler(FileNotFoundError)
    @app.errorhandler(IndexError)
    @app.errorhandler(ValueError)
    def bad_request(error):
        return jsonify({"error": str(error)}), 400

    @app.errorhandler(InferenceFailure)
    def inference_failure(error):
        return jsonify({"error": str(error)}), 500

    return app


def public_summary(summary: dict) -> dict:
    scene_id = summary["scene_id"]
    frame_index = summary["frame_index"]
    prefix = f"/api/artifact/{scene_id}/{frame_index}"
    if summary.get("method") == "gpt_sam31":
        sam = summary["sam31"]
        return {"scene_id": scene_id, "frame_index": frame_index, "method": "gpt_sam31",
            "cached": summary.get("cached", False), "wall_seconds": summary["wall_seconds"],
            "contract": summary["contract"], "input_image": f"/api/frame/{scene_id}/{frame_index}.jpg",
            "gpt": {"model": summary["gpt"]["requested_model"],
                "categories": summary["gpt"]["labels"]["categories"],
                "unresolved_regions": summary["gpt"]["labels"]["unresolved_regions"],
                "manifest": f"{prefix}/gpt/labels.json", "elapsed_seconds": summary["gpt"]["elapsed_seconds"]},
            "sam31": {"image": f"{prefix}/sam31/annotated.jpg", "manifest": f"{prefix}/sam31/predictions.json",
                "instance_map": f"{prefix}/sam31/instance_id_map.png", "instance_count": sam["instance_count"],
                "unique_classes": sam["unique_classes"], "timing": sam["timing"], "instances": sam["instances"],
                "empty_prompt_count": sum(not x["raw_candidate_ids"] for x in sam["prompts"])}}
    yoloe = summary["yoloe"]
    sam31 = summary["sam31"]
    return {
        "scene_id": scene_id,
        "frame_index": frame_index,
        "cached": summary.get("cached", False),
        "wall_seconds": summary["wall_seconds"],
        "contract": summary["contract"],
        "input_image": f"/api/frame/{scene_id}/{frame_index}.jpg",
        "yoloe": {
            "image": f"{prefix}/yoloe/annotated.jpg",
            "manifest": f"{prefix}/yoloe/detections.json",
            "instance_count": yoloe["instance_count"],
            "unique_classes": yoloe["unique_classes"],
            "timing": yoloe["timing"],
            "instances": yoloe["instances"],
        },
        "sam31": {
            "image": f"{prefix}/sam31/annotated.jpg",
            "manifest": f"{prefix}/sam31/predictions.json",
            "instance_map": f"{prefix}/sam31/instance_id_map.png",
            "instance_count": sam31["instance_count"],
            "unique_classes": sam31["unique_classes"],
            "timing": sam31["timing"],
            "instances": sam31["instances"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    application = create_app()
    try:
        application.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True,
            use_reloader=False,
        )
    finally:
        application.extensions["model_service"].shutdown()
