#!/usr/bin/env bash
# Rebuild the existing scene0000_00 uniform-500 mask-to-point-cloud page.
set -euo pipefail

module_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
project_root="$(dirname "$module_root")"
python_bin="${AMB3R_PYTHON:-$module_root/.venv/bin/python}"
checkpoint="${AMB3R_CHECKPOINT:-$(dirname "$project_root")/weights/amb3r/amb3r.pt}"
scene_root="${SCANNET_ROOT:-$(dirname "$project_root")/data/scannet_v2}/scene0000_00"
batch="$project_root/InstanceRecognition/runtime/batches/scene0000_00/target_500.json"
data="$module_root/data/scannet/scene0000_00_uniform500"
geometry="$module_root/output/amb3r/vo_uniform500"
reference="$module_root/output/amb3r/base_uniform500_first20"
viewer="$project_root/InstanceRecognition/runtime/mask_pointclouds/scene0000_00/amb3r_gt_raw"
logs="$module_root/output/amb3r/restore_logs"
prediction="$geometry/scene_scene0000_00_uniform500_results.npz"
metric_prediction="$geometry/scene_scene0000_00_uniform500_metric_head_results.npz"
alignment="$geometry/pointcloud_gt_corrected_evaluation.json"
pointcloud="$geometry/pointcloud_gt_corrected_rgb.ply"
runner=(taskset -c "${AMB3R_CPU:-0}" env HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$python_bin" -u)

test -x "$python_bin"
test -f "$checkpoint"
test -f "$batch"
mkdir -p "$logs" "$geometry" "$reference"

if [[ ! -f "$data/manifest.json" ]]; then
  "${runner[@]}" "$module_root/src/amb3r/extract_scannet_sens.py" \
    --sens "$scene_root/scene0000_00.sens" --output "$data" \
    --frame-indices-json "$batch" > "$logs/extraction.log" 2>&1
fi
if [[ ! -f "$reference/amb3r_frontend_metric_results.npz" ]]; then
  echo 'Running first-20 RGB metric-scale reference'
  "${runner[@]}" "$module_root/src/amb3r/run_headless_amb3r.py" \
    --repo "$module_root/ThirdParty/amb3r" --data "$data/input" \
    --checkpoint "$checkpoint" --output "$reference" \
    --num-frames 20 --frontend-only --sdpa-attention > "$logs/frontend.log" 2>&1
fi
if [[ ! -f "$prediction" ]]; then
  echo 'Running official AMB3R-VO on the 500 RGB frames'
  "${runner[@]}" "$module_root/src/amb3r/run_vo_headless.py" \
    --repo "$module_root/ThirdParty/amb3r" --data "$data/rgb" \
    --checkpoint "$checkpoint" --output "$geometry" --sdpa-attention \
    > "$logs/vo.log" 2>&1
fi
if [[ ! -f "$metric_prediction" ]]; then
  echo 'Applying the RGB-predicted metric scalar'
  "${runner[@]}" "$module_root/src/amb3r/apply_metric_scale.py" \
    --prediction "$prediction" --scale-reference "$reference/amb3r_frontend_metric_results.npz" \
    --input-ply "$geometry/raw_preview_rgb.ply" --output-prediction "$metric_prediction" \
    --output-ply "$geometry/metric_head_preview_rgb.ply" \
    --provenance "$geometry/metric_scale_provenance.json" > "$logs/metric-scale.log" 2>&1
fi
if [[ ! -f "$alignment" ]]; then
  # The fit and the cloud it produces are one step: the canonical PLY handed to
  # the mask lifter has to be the same point set under the same transform, so
  # both come out of this call rather than from evaluation.
  echo 'Fitting the original GT Sim(3) and exporting the canonical colored cloud'
  "${runner[@]}" "$module_root/src/common/fit_sim3_from_gt.py" \
    --prediction "$metric_prediction" --data "$data/input" --output "$alignment" \
    --se3-ply "$geometry/pointcloud_se3_scan_world_rgb.ply" --sim3-ply "$pointcloud" \
    > "$logs/sim3-fit.log" 2>&1
fi
if [[ ! -f "$viewer/index.html" ]]; then
  gpt_viewer="$(dirname "$viewer")/gpt_terra_gt_raw"
  if [[ -f "$(dirname "$viewer")/gpt_terra_pairs_v1_gt_raw/manifest.json" ]]; then
    gpt_viewer="$(dirname "$viewer")/gpt_terra_pairs_v1_gt_raw"
  fi
  if [[ -f "$gpt_viewer/manifest.json" ]]; then
    echo 'Restoring the completed GPT + SAM primary viewer'
    PYTHONPATH="$project_root/InstanceRecognition" "${runner[@]}" -m src.publish_gpt_viewer \
      --source "$gpt_viewer" --destination "$viewer" \
      > "$logs/projection.log" 2>&1
  else
    echo 'Projecting existing YOLOE baseline masks'
    "${runner[@]}" "$project_root/InstanceRecognition/src/backproject_masks.py" \
      --prediction "$metric_prediction" --alignment "$alignment" --pointcloud "$pointcloud" \
      --data "$data" --batch "$batch" --output "$viewer" > "$logs/projection.log" 2>&1
  fi
fi
echo "Offline viewer: $viewer/index.html"
