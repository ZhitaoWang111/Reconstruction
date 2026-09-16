#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON="/home/wzt/.conda/envs/scene/bin/python"
export PYTHONPATH="$SCRIPT_DIR/../SAM3${PYTHONPATH:+:$PYTHONPATH}"

run_one() {
  local version="$1"
  local checkpoint="$2"
  local source="$3"
  local detections="$4"
  local output_dir="$5"
  "$PYTHON" "$SCRIPT_DIR/infer_yoloe_boxes_sam3.py" \
    --source "$source" \
    --yoloe-predictions "$detections" \
    --sam-version "$version" \
    --sam-checkpoint "$checkpoint" \
    --sam-threshold 0.0 \
    --output-dir "$output_dir"
  "$PYTHON" "$SCRIPT_DIR/verify_outputs.py" "$output_dir"
}

YOLOE_ROOT="$SCRIPT_DIR/../yoloe26_prompt_free"
BUS_IMAGE="$YOLOE_ROOT/inputs/official_bus.jpg"
SCAN_IMAGE="$WORKSPACE/ScanNet/assets/frame-0162-rgb.webp"

run_one sam3 "$WORKSPACE/weights/sam3.pt" \
  "$BUS_IMAGE" "$YOLOE_ROOT/outputs/official_bus_s/predictions.json" \
  "$SCRIPT_DIR/outputs/sam3/official_bus_s"
run_one sam3 "$WORKSPACE/weights/sam3.pt" \
  "$SCAN_IMAGE" "$YOLOE_ROOT/outputs/scannet_scene0000_00_frame0162_s/predictions.json" \
  "$SCRIPT_DIR/outputs/sam3/scannet_scene0000_00_frame0162_s"

run_one sam3.1 "$WORKSPACE/weights/sam3.1_multiplex.pt" \
  "$BUS_IMAGE" "$YOLOE_ROOT/outputs/official_bus_s/predictions.json" \
  "$SCRIPT_DIR/outputs/sam3.1/official_bus_s"
run_one sam3.1 "$WORKSPACE/weights/sam3.1_multiplex.pt" \
  "$SCAN_IMAGE" "$YOLOE_ROOT/outputs/scannet_scene0000_00_frame0162_s/predictions.json" \
  "$SCRIPT_DIR/outputs/sam3.1/scannet_scene0000_00_frame0162_s"
