#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON="$SCRIPT_DIR/../yoloe26_prompt_free/.venv/bin/python"

"$PYTHON" "$SCRIPT_DIR/infer_yolo26_seg.py" \
  --source "$SCRIPT_DIR/../yoloe26_prompt_free/inputs/official_bus.jpg" \
  --model yolo26s-seg.pt \
  --output-dir "$SCRIPT_DIR/outputs/official_bus_s" \
  --device 0 \
  --imgsz 640 \
  --conf 0.25

"$PYTHON" "$SCRIPT_DIR/verify_outputs.py" "$SCRIPT_DIR/outputs/official_bus_s"

"$PYTHON" "$SCRIPT_DIR/infer_yolo26_seg.py" \
  --source "$WORKSPACE/ScanNet/assets/frame-0162-rgb.webp" \
  --model yolo26s-seg.pt \
  --output-dir "$SCRIPT_DIR/outputs/scannet_scene0000_00_frame0162_s" \
  --device 0 \
  --imgsz 640 \
  --conf 0.25

"$PYTHON" "$SCRIPT_DIR/verify_outputs.py" \
  "$SCRIPT_DIR/outputs/scannet_scene0000_00_frame0162_s"
