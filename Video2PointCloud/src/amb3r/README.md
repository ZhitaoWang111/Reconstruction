# AMB3R / AMB3R-VO on ScanNet v2 `scene0000_00`

> The exact InstanceRecognition uniform-500 run is complete. See the [combined 500-frame report](../../output/reports/SCANNET_SCENE0000_00_UNIFORM500_RESULTS.md) and [`output/amb3r/vo_uniform500`](../../output/amb3r/vo_uniform500). The sections below preserve the earlier 40-frame reproduction.

This directory is a reproducible, bounded test of the official
[AMB3R repository](https://github.com/HengyiWang/amb3r) on the first ScanNet v2
scene. Inference uses RGB only. ScanNet depth and poses are decoded separately
and used only for evaluation.

## Result in one paragraph

The official code and checkpoint run successfully on an RTX 4090. The predicted
geometry is finite and visually coherent. The base model's metric head produces
roughly metric geometry, but on this sequence it is about 11% too large
(20-frame depth AbsRel 11.12%). The unmodified AMB3R-VO archive is **not metric
scale** in this test: its median GT/predicted depth scale is 2.024 and raw depth
AbsRel is 50.65%. Applying the base model's own 20-frame scale estimate (2.25598)
as an explicitly separate postprocess reduces VO raw depth AbsRel to 11.44%.
After median GT scale alignment, VO depth AbsRel is 1.59% and trajectory Sim(3)
ATE is 2.92 cm, showing that most of the discrepancy is a single scale factor.

## Provenance and environment

- Upstream commit: `92c4081f910f98e683503092b85301861519175e`
- Checkpoint SHA-256: `2d9342282cce0a20e1074ad1513d9a18c3d100dc06c734a4a11f2d1baeea5239`
- Conda prefix: `/home/wzt/.conda/envs/amb3r`
- Python 3.9.25; PyTorch 2.5.0+cu118; torchvision 0.20.0+cu118
- xformers 0.0.28.post2; Open3D 0.18.0; flash-attn 2.7.3
- spconv 2.3.8; torch-scatter 2.1.2+pt25cu118; PyTorch3D 0.7.8
- GPU: NVIDIA GeForce RTX 4090, 24,564 MiB; driver 570.133.07
- `pip check`: no broken requirements

The upstream repository currently has no `LICENSE` file. Check licensing with
the authors before redistributing code, checkpoint, or derived packages.

## Test data

- Source: `/home/wzt/HDD/datasets/scannet_v2/scans/scene0000_00/scene0000_00.sens`
- Source recording: 5,578 frames, 3,699,853,377 bytes
- Evaluated subset: source frames `0, 10, ..., 390` (40 RGB frames)
- RGB/depth evaluation grid: 640 x 480 before AMB3R preprocessing
- All 40 ScanNet poses are finite
- Ground-truth sampled path: 5.3275 m
- Extraction record: [`data/scannet/scene0000_00_stride10_40f/manifest.json`](../../data/scannet/scene0000_00_stride10_40f/manifest.json)

The `input/` folder contains paired RGB, depth, and metadata for evaluation.
The `rgb/` folder intentionally contains JPEG files only because the official VO
demo loader treats every image extension as an RGB frame.

## Quantitative results

| Run | Frames | Native scale | Depth AbsRel | Depth RMSE | delta1 | Trajectory path ratio | ATE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AMB3R frontend + metric head | 20 | GT/pred 0.8997 | 11.12% | 0.2395 m | 99.81% | 1.082 | SE(3) 3.06 cm |
| Official AMB3R-VO output | 40 | GT/pred 2.0241 | 50.65% | 1.0311 m | 0.05% | 0.497 | SE(3) 27.73 cm |
| VO + AMB3R metric-head scale | 40 | GT/pred 0.8972 | 11.44% | 0.2392 m | 99.62% | 1.121 | SE(3) 7.62 cm |
| VO + oracle alignment (diagnostic) | 40 | aligned | 1.59% | 0.0633 m | 99.65% | — | Sim(3) 2.92 cm |

The oracle row is diagnostic only and is not a deployable result. The
metric-head row uses no ScanNet depth at inference time; its scale comes from
AMB3R's predicted `depth_metric/depth` ratio on the first 20 RGB frames.

Runtime measurements:

- AMB3R 4-frame frontend + backend: 1.197 s model time, 7.02 GB peak allocated GPU memory.
- AMB3R 20-frame frontend: 1.273 s model time, 7.85 GB peak allocated GPU memory.
- Official 40-frame VO: 7.58 reported FPS; 28.24 s end-to-end wall time; 10.95 GB maximum host RSS.

## Important code-path finding

The upstream frontend computes `depth_metric`, but leaves `world_points` and
pose translations at their original scale (`amb3r/frontend.py:138-146`). The VO
configuration selects raw `world_points` (`slam/slam_config.yaml:29`), and the
official saver writes those points and poses directly (`slam/run.py:74-115`).
Therefore the unmodified VO PLY/NPZ should not be treated as metric output in
this checkout. `apply_metric_scale.py` keeps the official files unchanged and
records its separate transformation in `metric_scale_provenance.json`.

## Main artifacts

- Official VO metrics: [`output/amb3r/vo_40f/evaluation.json`](../../output/amb3r/vo_40f/evaluation.json)
- Metric-head VO metrics: [`output/amb3r/vo_40f/metric_head_evaluation.json`](../../output/amb3r/vo_40f/metric_head_evaluation.json)
- Metric-head scale provenance: [`output/amb3r/vo_40f/metric_scale_provenance.json`](../../output/amb3r/vo_40f/metric_scale_provenance.json)
- Metric-head VO point cloud: [`output/amb3r/vo_40f/scene_scene0000_00_s0_n40_stride10_metric_head_kf_pts.ply`](../../output/amb3r/vo_40f/scene_scene0000_00_s0_n40_stride10_metric_head_kf_pts.ply)
- Point-cloud preview: [`output/amb3r/vo_40f/metric_head_preview.png`](../../output/amb3r/vo_40f/metric_head_preview.png)
- 20-frame base metrics: [`output/amb3r/base_20f_frontend/evaluation.json`](../../output/amb3r/base_20f_frontend/evaluation.json)
- 4-frame frontend/backend comparison: [`output/amb3r/base_4f/run_metrics.json`](../../output/amb3r/base_4f/run_metrics.json)

## Exact rerun commands

All commands use absolute paths and leave the official Git checkout unchanged.

Extract the bounded ScanNet sequence:

```bash
/home/wzt/.conda/envs/amb3r/bin/python /home/wzt/wzt/CVPR_2027/Video2PointCloud/src/amb3r/extract_scannet_sens.py \
  --sens /home/wzt/HDD/datasets/scannet_v2/scans/scene0000_00/scene0000_00.sens \
  --output /home/wzt/wzt/CVPR_2027/Video2PointCloud/data/scannet/scene0000_00_stride10_40f \
  --start 0 --count 40 --stride 10
```

Run the official AMB3R-VO entrypoint:

```bash
cd /home/wzt/wzt/CVPR_2027/Video2PointCloud/ThirdParty/amb3r
CUDA_VISIBLE_DEVICES=0 /home/wzt/.conda/envs/amb3r/bin/python slam/run.py \
  --data_path /home/wzt/wzt/CVPR_2027/Video2PointCloud/data/scannet/scene0000_00_stride10_40f/rgb \
  --demo_name scene0000_00_s0_n40_stride10 \
  --results_path /home/wzt/wzt/CVPR_2027/Video2PointCloud/output/amb3r/vo_40f \
  --model_name amb3r \
  --ckpt_path /home/wzt/wzt/CVPR_2027/Video2PointCloud/checkpoints/amb3r/amb3r.pt
```

Run the 20-frame metric head and evaluate it:

```bash
cd /home/wzt/wzt/CVPR_2027/Video2PointCloud/ThirdParty/amb3r
CUDA_VISIBLE_DEVICES=0 /home/wzt/.conda/envs/amb3r/bin/python \
  /home/wzt/wzt/CVPR_2027/Video2PointCloud/src/amb3r/run_headless_amb3r.py \
  --repo /home/wzt/wzt/CVPR_2027/Video2PointCloud/ThirdParty/amb3r \
  --data /home/wzt/wzt/CVPR_2027/Video2PointCloud/data/scannet/scene0000_00_stride10_40f/input \
  --checkpoint /home/wzt/wzt/CVPR_2027/Video2PointCloud/checkpoints/amb3r/amb3r.pt \
  --output /home/wzt/wzt/CVPR_2027/Video2PointCloud/output/amb3r/base_20f_frontend \
  --num-frames 20 --frontend-only
```

Apply the predicted scale without overwriting the official VO result:

```bash
/home/wzt/.conda/envs/amb3r/bin/python \
  /home/wzt/wzt/CVPR_2027/Video2PointCloud/src/amb3r/apply_metric_scale.py \
  --prediction /home/wzt/wzt/CVPR_2027/Video2PointCloud/output/amb3r/vo_40f/scene_scene0000_00_s0_n40_stride10_results.npz \
  --scale-reference /home/wzt/wzt/CVPR_2027/Video2PointCloud/output/amb3r/base_20f_frontend/amb3r_frontend_metric_results.npz \
  --input-ply /home/wzt/wzt/CVPR_2027/Video2PointCloud/output/amb3r/vo_40f/scene_scene0000_00_s0_n40_stride10_downsampled_kf_pts.ply \
  --output-prediction /home/wzt/wzt/CVPR_2027/Video2PointCloud/output/amb3r/vo_40f/scene_scene0000_00_s0_n40_stride10_metric_head_results.npz \
  --output-ply /home/wzt/wzt/CVPR_2027/Video2PointCloud/output/amb3r/vo_40f/scene_scene0000_00_s0_n40_stride10_metric_head_kf_pts.ply \
  --provenance /home/wzt/wzt/CVPR_2027/Video2PointCloud/output/amb3r/vo_40f/metric_scale_provenance.json
```

Evaluate either NPZ by selecting `--mode vo` or `--mode base`:

```bash
/home/wzt/.conda/envs/amb3r/bin/python \
  /home/wzt/wzt/CVPR_2027/Video2PointCloud/src/amb3r/evaluate_scannet.py \
  --repo /home/wzt/wzt/CVPR_2027/Video2PointCloud/ThirdParty/amb3r \
  --data /home/wzt/wzt/CVPR_2027/Video2PointCloud/data/scannet/scene0000_00_stride10_40f/input \
  --prediction /home/wzt/wzt/CVPR_2027/Video2PointCloud/output/amb3r/vo_40f/scene_scene0000_00_s0_n40_stride10_metric_head_results.npz \
  --mode vo \
  --output /home/wzt/wzt/CVPR_2027/Video2PointCloud/output/amb3r/vo_40f/metric_head_evaluation.json
```

## Evidence boundaries

- This is one bounded scene segment, not a ScanNet benchmark result.
- The geometry is visually coherent, but contains normal multi-view thickness
  and edge noise; finite arrays alone are not taken as reconstruction quality.
- The metric-head correction has an approximately 10-12% scale bias here.
- Model FPS excludes model loading, file I/O, masks, and serialization. The VO
  end-to-end wall time includes those costs.
- All downloaded weights are kept outside the upstream checkout, whose tracked
  and untracked Git status is clean after this workspace reorganization.
