#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$ROOT/../../.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
CHECKPOINT="$ROOT/checkpoints/wedetect_base_uni.pth"
ONNX_DIR="$ROOT/wedetect_anything/onnx_models"
ONNX_MODEL="$ONNX_DIR/wedetect_anything_base.onnx"
META="$ONNX_DIR/wedetect_anything_base_meta.json"
OUTPUT_DIR="$ROOT/wedetect_anything/outputs"
DEVICE="${WEDETECT_DEVICE:-cuda}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing environment: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing checkpoint: $CHECKPOINT" >&2
  exit 1
fi

mkdir -p "$ONNX_DIR" "$OUTPUT_DIR"

if [[ ! -f "$ONNX_MODEL" || ! -f "$META" ]]; then
  "$PYTHON" "$ROOT/wedetect_anything/export_onnx.py" \
    --variant base \
    --checkpoint "$CHECKPOINT" \
    --img-size 640 \
    --output-dir "$ONNX_DIR"
fi

run_one() {
  local image="$1"
  local output="$2"
  local log="$3"
  "$PYTHON" "$ROOT/wedetect_anything/infer_anything.py" \
    --onnx "$ONNX_MODEL" \
    --meta "$META" \
    --text-feat "$ROOT/wedetect_anything/prompt_free.npy" \
    --image "$image" \
    --output "$output" \
    --font "$ROOT/simsun.ttc" \
    --score-thr 0.1 \
    --iou-thr 0.7 \
    --keep-top-k 100 \
    --match-thr 0.4 \
    --device "$DEVICE" | tee "$log"
}

run_one \
  "$ROOT/wedetect_anything/examples/1.jpg" \
  "$OUTPUT_DIR/official_example_1.jpg" \
  "$OUTPUT_DIR/official_example_1.log"

run_one \
  "$ROOT/../yoloe26_prompt_free/inputs/official_bus.jpg" \
  "$OUTPUT_DIR/official_bus.jpg" \
  "$OUTPUT_DIR/official_bus.log"

run_one \
  "$WORKSPACE/ScanNet/assets/frame-0162-rgb.webp" \
  "$OUTPUT_DIR/scannet_scene0000_00_frame0162.jpg" \
  "$OUTPUT_DIR/scannet_scene0000_00_frame0162.log"

"$PYTHON" "$ROOT/verify_local_reproduction.py"
