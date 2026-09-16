"""Orchestrate YOLOE-26 and SAM3.1 in their compatible Python environments."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from . import config
from .model_service import ModelServiceError, ModelServiceManager
from .scannet_sens import ScanNetSensReader


class InferenceFailure(RuntimeError):
    pass


class InstanceRecognitionPipeline:
    def __init__(
        self,
        reader: ScanNetSensReader,
        runtime_root: Path = config.RUNTIME_ROOT,
        model_service: ModelServiceManager | None = None,
    ):
        self.reader = reader
        self.runtime_root = Path(runtime_root).resolve()
        self.model_service = model_service or ModelServiceManager(self.runtime_root)
        self._inference_lock = threading.Lock()

    def _assert_dependencies(self) -> None:
        required = [
            config.YOLOE_PYTHON,
            config.SAM31_PYTHON,
            config.YOLOE_CHECKPOINT,
            config.SAM31_CHECKPOINT,
            config.SAM3_REPO / "sam3/model_builder.py",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise InferenceFailure("Missing runtime dependencies: " + ", ".join(missing))

    def run(self, scene_id: str, frame_index: int, force: bool = False) -> dict:
        self._assert_dependencies()
        frame_path = self.reader.frame_path(scene_id, frame_index)
        run_root = (
            self.runtime_root / "results" / scene_id / f"frame_{frame_index:06d}"
        )
        summary_path = run_root / "summary.json"
        if summary_path.is_file() and not force:
            summary = json.loads(summary_path.read_text())
            self._validate(summary, run_root)
            summary["cached"] = True
            return summary

        with self._inference_lock:
            if summary_path.is_file() and not force:
                summary = json.loads(summary_path.read_text())
                self._validate(summary, run_root)
                summary["cached"] = True
                return summary
            run_root.mkdir(parents=True, exist_ok=True)
            yoloe_root = run_root / "yoloe"
            sam31_root = run_root / "sam31"
            started = time.perf_counter()

            try:
                yoloe = self.model_service.infer_yoloe(frame_path, yoloe_root)
                sam31 = self.model_service.infer_sam31(
                    frame_path, yoloe_root / "detections.json", sam31_root
                )
            except ModelServiceError as error:
                raise InferenceFailure(str(error)) from error
            summary = {
                "scene_id": scene_id,
                "frame_index": frame_index,
                "source": str(frame_path),
                "cached": False,
                "wall_seconds": round(time.perf_counter() - started, 3),
                "contract": {
                    "input": "ScanNet RGB frame only",
                    "text_prompt": None,
                    "yoloe_masks_consumed": False,
                    "maximum_detections": config.DEFAULT_MAX_DETECTIONS,
                },
                "yoloe": yoloe,
                "sam31": sam31,
            }
            self._validate(summary, run_root)
            temporary = summary_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
            )
            os.replace(temporary, summary_path)
            return summary

    @staticmethod
    def _validate(summary: dict, run_root: Path) -> None:
        yoloe = summary["yoloe"]
        sam31 = summary["sam31"]
        if yoloe.get("native_masks_forwarded") is not False:
            raise InferenceFailure("YOLOE mask isolation contract was violated")
        if sam31.get("yoloe_masks_consumed") is not False:
            raise InferenceFailure("SAM3.1 consumed a YOLOE mask")
        if yoloe["instance_count"] != sam31["instance_count"]:
            raise InferenceFailure("YOLOE detection count and SAM3.1 mask count differ")
        artifacts = [
            run_root / "yoloe" / yoloe["artifacts"]["annotated_image"],
            run_root / "sam31" / sam31["artifacts"]["annotated_image"],
            run_root / "sam31" / sam31["artifacts"]["instance_id_map"],
        ]
        if any(not path.is_file() for path in artifacts):
            raise InferenceFailure("A required result artifact is missing")
