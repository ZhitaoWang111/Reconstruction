"""Orchestrate GPT labels and SAM3.1, with an opt-in YOLOE baseline."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from src import config
from src.inference.model_service import ModelServiceError, ModelServiceManager
from src.data.scannet_sens import ScanNetSensReader


class InferenceFailure(RuntimeError):
    pass


class InstanceRecognitionPipeline:
    def __init__(
        self,
        reader: ScanNetSensReader,
        runtime_root: Path = config.RUNTIME_ROOT,
        model_service: ModelServiceManager | None = None,
        method: str = config.DEFAULT_METHOD,
    ):
        self.reader = reader
        self.runtime_root = Path(runtime_root).resolve()
        if method not in {"gpt_sam31", "yoloe_sam31"}:
            raise ValueError("Unknown recognition method: " + method)
        self.method = method
        self.model_service = model_service or ModelServiceManager(self.runtime_root, method=method)
        self._inference_lock = threading.Lock()

    def result_root(self, scene_id: str, frame_index: int) -> Path:
        directory = "results_gpt" if self.method == "gpt_sam31" else "results"
        return self.runtime_root / directory / scene_id / f"frame_{frame_index:06d}"

    def _assert_dependencies(self) -> None:
        required = [config.SAM31_PYTHON, config.SAM31_CHECKPOINT,
                    config.SAM3_REPO / "sam3/model_builder.py"]
        if self.method == "gpt_sam31":
            required.extend([config.GPT_API_DIR / "label_frames.py", config.GPT_API_DIR / "prompt_en.txt"])
        else:
            required.extend([config.YOLOE_PYTHON, config.YOLOE_CHECKPOINT])
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise InferenceFailure("Missing runtime dependencies: " + ", ".join(missing))

    def run(self, scene_id: str, frame_index: int, force: bool = False) -> dict:
        self._assert_dependencies()
        frame_path = self.reader.frame_path(scene_id, frame_index)
        run_root = self.result_root(scene_id, frame_index)
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
            if self.method == "gpt_sam31":
                return self._run_gpt(scene_id, frame_index, frame_path, run_root, force)
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

    def _run_gpt(self, scene_id, frame_index, frame_path, run_root, force):
        from src.inference.gpt_labels import label_frame
        started = time.perf_counter()
        try:
            labels = label_frame(frame_path, scene_id, frame_index, run_root / "gpt", force=force)
            sam31 = self.model_service.infer_sam31_text(frame_path, run_root / "gpt/labels.json", run_root / "sam31")
        except Exception as error:
            raise InferenceFailure(str(error)) from error
        summary = {"scene_id": scene_id, "frame_index": frame_index, "source": str(frame_path),
            "method": "gpt_sam31", "cached": False, "wall_seconds": round(time.perf_counter()-started, 3),
            "contract": {"method": "gpt_sam31", "input": "ScanNet RGB frame only", "model": config.GPT_MODEL,
                "reasoning_effort": "none", "maximum_detections": 16, "sam_text_threshold": config.GPT_SAM_THRESHOLD,
                "yoloe_masks_consumed": False, "text_prompt": "GPT categories"},
            "gpt": labels, "sam31": sam31}
        self._validate(summary, run_root)
        temporary = run_root / "summary.json.tmp"
        temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
        os.replace(temporary, run_root / "summary.json")
        return summary

    @staticmethod
    def _validate(summary: dict, run_root: Path) -> None:
        if summary.get("method") == "gpt_sam31":
            sam = summary["sam31"]
            if sam["instance_count"] != len(sam["instances"]):
                raise InferenceFailure("GPT/SAM instance count mismatch")
            if sam["policy"].get("yoloe_input_consumed") is not False:
                raise InferenceFailure("GPT branch consumed YOLOE input")
            artifacts = [run_root / "gpt/labels.json", run_root / "sam31/predictions.json",
                         run_root / "sam31" / sam["artifacts"]["annotated_image"],
                         run_root / "sam31" / sam["artifacts"]["instance_id_map"]]
            artifacts.extend(run_root / "sam31" / x["mask_file"] for x in sam["raw_candidates"])
            if any(not path.is_file() for path in artifacts):
                raise InferenceFailure("A required GPT/SAM result artifact is missing")
            return
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
