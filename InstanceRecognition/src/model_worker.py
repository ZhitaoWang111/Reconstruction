#!/usr/bin/env python3
"""JSON-lines model worker used by the long-lived web service."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("yoloe", "sam31"), required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-det", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.0)
    return parser.parse_args()


def send(protocol, payload: dict) -> None:
    protocol.write(json.dumps(payload, ensure_ascii=False) + "\n")
    protocol.flush()


def main() -> None:
    args = parse_args()
    protocol = sys.stdout
    # Third-party libraries print progress to stdout. Keep stdout protocol-only.
    sys.stdout = sys.stderr
    model = None

    def load_model():
        nonlocal model
        if model is not None:
            return 0.0
        started = time.perf_counter()
        if args.stage == "yoloe":
            from .yoloe_detect import YoloeDetector

            model = YoloeDetector(
                args.checkpoint,
                device=args.device,
                imgsz=args.imgsz,
                conf=args.conf,
                max_det=args.max_det,
            )
        else:
            from .sam31_segment import Sam31Segmenter

            model = Sam31Segmenter(args.checkpoint, args.threshold)
        return time.perf_counter() - started

    for line in sys.stdin:
        request_id = None
        try:
            message = json.loads(line)
            request_id = message.get("id")
            command = message.get("command")
            if command == "load":
                load_seconds = load_model()
                send(
                    protocol,
                    {
                        "id": request_id,
                        "ok": True,
                        "status": "loaded",
                        "load_seconds": round(load_seconds, 3),
                    },
                )
            elif command == "infer":
                load_seconds = load_model()
                if args.stage == "yoloe":
                    result = model.infer(
                        Path(message["source"]), Path(message["output_dir"])
                    )
                else:
                    result = model.infer(
                        Path(message["source"]),
                        Path(message["detections"]),
                        Path(message["output_dir"]),
                    )
                send(
                    protocol,
                    {
                        "id": request_id,
                        "ok": True,
                        "load_seconds": round(load_seconds, 3),
                        "result": result,
                    },
                )
            elif command == "shutdown":
                send(protocol, {"id": request_id, "ok": True, "status": "stopped"})
                return
            else:
                raise ValueError(f"Unknown command: {command}")
        except Exception as error:  # keep the worker alive for recoverable requests
            send(
                protocol,
                {
                    "id": request_id,
                    "ok": False,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )


if __name__ == "__main__":
    main()
