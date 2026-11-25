#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/infer_prompt_free.py" \
  --source "$SCRIPT_DIR/inputs/official_bus.jpg" \
  --model yoloe-26s-seg-pf.pt \
  --output-dir "$SCRIPT_DIR/outputs/official_bus_s" \
  --device 0 \
  --imgsz 640 \
  --conf 0.25

"$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/verify_outputs.py" \
  "$SCRIPT_DIR/outputs/official_bus_s"
