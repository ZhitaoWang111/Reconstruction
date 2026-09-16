# ScanNet `scene0000_00` uniform-500 point-cloud benchmark

Run date: 2026-09-13. All three methods were run **serially**, never concurrently, on one RTX 4090 (24 GB). This is a one-scene reproduction, not a ScanNet leaderboard.

## Conclusion

On the native-scale 500-frame reconstruction, **Depth Anything 3 Nested is best**: 0.135 m point-cloud Chamfer-L1, 0.519 F-score@10 cm, 7.59% raw depth AbsRel, and 0.193 m SE(3) trajectory ATE. AMB3R-VO plus its own RGB-predicted metric-head scalar ranks second. MapAnything's short-window predictions are locally plausible, but the required 33-window stitching accumulates substantial scale and rotation drift.

For follow-up applications, the active outputs are grouped into three categories:

| Category | Scale source | Intended use |
| --- | --- | --- |
| **GT-corrected AMB3R-VO** | Full Sim(3) fitted from 500 ScanNet GT/predicted camera correspondences | Dataset processing and oracle diagnosis |
| **Original AMB3R-VO** | AMB3R's RGB-only metric-head scalar; no GT scale correction | Globally consistent point cloud on unseen video |
| **DA3 Nested** | DA3 native metric prediction | Single-frame metric depth |

The GT-corrected colored cloud is [`pointcloud_gt_corrected_rgb.ply`](../amb3r/vo_uniform500/pointcloud_gt_corrected_rgb.ply). It contains 491,666 points in ScanNet coordinates. Its fitted scale relative to the original page cloud is `0.891636`; Chamfer-L1 is 0.067 m and F@10 cm is 0.820. This remains a GT-assisted/oracle result rather than a deployable RGB-only output.

## Exact input selection

The source is `/home/wzt/HDD/datasets/scannet_v2/scans/scene0000_00/scene0000_00.sens` (5,578 frames). The frame selection is copied exactly from InstanceRecognition:

```python
last = frame_count - 1
indices = [round(i * last / (target_frames - 1)) for i in range(target_frames)]
```

The authoritative list is `/home/wzt/wzt/CVPR_2027/InstanceRecognition/runtime/batches/scene0000_00/target_500.json`. It contains exactly 500 unique frames, includes both endpoints, starts with `0, 11, 22, 34, 45, 56, 67, 78`, and ends with `5499, 5510, 5521, 5532, 5543, 5555, 5566, 5577`. The extracted manifest is [`data/scannet/scene0000_00_uniform500/manifest.json`](../../data/scannet/scene0000_00_uniform500/manifest.json). All 500 GT poses are finite; the GT camera path is 63.003 m.

Inference receives only the 500 RGB images. ScanNet depth, intrinsics, poses, and mesh are used only by the evaluator.

## Native-scale results

Lower is better for AbsRel, RMSE, ATE, accuracy, completeness, and Chamfer-L1; the depth-scale ratio is best near 1.0; higher is better for F-score.

| Method | Raw depth AbsRel | Depth RMSE | GT/pred depth scale | SE(3) ATE | Cloud accuracy | Cloud completeness | Chamfer-L1 | F@5 cm | F@10 cm | F@20 cm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **DA3 Nested streaming** | **7.59%** | **0.226 m** | **1.053** | **0.193 m** | **0.125 m** | **0.146 m** | **0.135 m** | **0.258** | **0.519** | **0.819** |
| AMB3R-VO + predicted metric scalar | 11.98% | 0.293 m | 0.895 | 0.248 m | 0.191 m | 0.172 m | 0.182 m | 0.103 | 0.296 | 0.617 |
| MapAnything streaming | 57.78% | 1.524 m | 0.692 | 1.465 m | 1.352 m | 0.317 m | 0.834 m | 0.040 | 0.084 | 0.165 |

`accuracy` is prediction-to-GT nearest-neighbour distance; `completeness` is GT-to-prediction distance; Chamfer-L1 is their arithmetic mean. The point-cloud comparison uses the native predicted scale and only an SE(3) camera-centre alignment into the ScanNet coordinate frame.

AMB3R's official VO archive is arbitrary-scale in this checkout. Before applying its RGB-only predicted scalar, raw depth AbsRel is 51.43%, SE(3) ATE is 0.977 m, and the predicted/GT path ratio is 0.484. The reported AMB3R row multiplies the official VO depth, points, and pose translations by the scalar 2.3023 predicted from the first 20 selected RGB frames; no ScanNet GT is used for that correction.

## Oracle-scale diagnostic

This table permits one global GT-derived scale and is therefore **not deployable**. It helps separate scale error from shape/drift error.

| Method | Aligned depth AbsRel | Sim(3) ATE | Sim(3) cloud scale | Cloud Chamfer-L1 | F@10 cm |
| --- | ---: | ---: | ---: | ---: | ---: |
| AMB3R-VO + metric scalar | **2.55%** | **0.091 m** | 0.892 | **0.067 m** | **0.820** |
| DA3 Nested streaming | 6.95% | 0.180 m | **1.039** | 0.121 m | 0.570 |
| MapAnything streaming | 15.88% | 0.696 m | 0.578 | 0.360 m | 0.178 |

AMB3R has the best oracle-aligned geometry, but its native metric scale is less accurate than DA3. MapAnything remains poor after one global Sim(3), showing that its failure is not a single scalar bias.

## Long-sequence execution modes

| Method | 500-frame path | Window | Wall time | Peak GPU allocation | Max host RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| AMB3R-VO | official VO, then separate 20-frame metric-head scalar | VO-native | 264.23 s | metric head: 7.31 GiB | 10.66 GiB |
| DA3 Nested | official `DA3-Streaming`, adjacent dense Sim(3) | 30 / overlap 15 | 139.99 s | not emitted upstream | 13.44 GiB |
| MapAnything | local serial wrapper around official `model.infer` | 30 / overlap 15 | 184.16 s total; 136.60 s infer+stitch | 9.34 GiB | 9.62 GiB |

Direct single-pass 500-frame inference was attempted separately for DA3 and MapAnything and exceeded 24 GB GPU memory. DA3 therefore uses its official streaming implementation. MapAnything has no upstream long-sequence stitcher in this checkout, so [`src/mapanything/run_mapanything_streaming.py`](../../src/mapanything/run_mapanything_streaming.py) aligns adjacent windows using only the 15 overlapping **predicted** camera centres. It never accesses ScanNet GT. Its largest recorded overlap-camera RMSE is 0.126 m, and the resulting drift is part of the measured result.

DA3 loop closure is disabled because no SALAD checkpoint was supplied; its 33 windows use the official adjacent dense Sim(3) path. All methods produced 500 finite depth maps.

## Point-cloud protocol

- GT surface: 500,000 points uniformly sampled with seed 0 from `/home/wzt/HDD/datasets/scannet_v2/scans/scene0000_00/scene0000_00_vh_clean_2.ply`.
- Prediction: at most 500,000 points, sampled approximately equally per frame after retaining the top 60% by each method's confidence and its native validity mask.
- Coordinate alignment: Umeyama on all 500 predicted/GT camera-centre correspondences. The main table fixes scale to 1.0; the diagnostic table estimates scale.
- Threshold metrics use strict distance `< 0.05/0.10/0.20 m`.
- The evaluator is [`src/common/evaluate_scannet_pointcloud.py`](../../src/common/evaluate_scannet_pointcloud.py). All three evaluation JSON files contain exactly the reference 500 source IDs and only finite numeric metrics.

## Reproduction commands

Extract the exact InstanceRecognition selection:

```bash
/home/wzt/.conda/envs/amb3r/bin/python /home/wzt/wzt/CVPR_2027/Video2PointCloud/src/amb3r/extract_scannet_sens.py \
  --sens /home/wzt/HDD/datasets/scannet_v2/scans/scene0000_00/scene0000_00.sens \
  --output /home/wzt/wzt/CVPR_2027/Video2PointCloud/data/scannet/scene0000_00_uniform500 \
  --frame-indices-json /home/wzt/wzt/CVPR_2027/InstanceRecognition/runtime/batches/scene0000_00/target_500.json
```

Run each command separately. Do not launch the three methods concurrently.

AMB3R-VO:

```bash
cd /home/wzt/wzt/CVPR_2027/Video2PointCloud/ThirdParty/amb3r
CUDA_VISIBLE_DEVICES=0 /home/wzt/.conda/envs/amb3r/bin/python slam/run.py \
  --data_path /home/wzt/wzt/CVPR_2027/Video2PointCloud/data/scannet/scene0000_00_uniform500/rgb \
  --demo_name scene0000_00_uniform500 \
  --results_path /home/wzt/wzt/CVPR_2027/Video2PointCloud/output/amb3r/vo_uniform500 \
  --model_name amb3r \
  --ckpt_path /home/wzt/wzt/CVPR_2027/Video2PointCloud/checkpoints/amb3r/amb3r.pt
```

DA3 Nested official streaming:

```bash
cd /home/wzt/wzt/CVPR_2027/Video2PointCloud/ThirdParty/depth-anything-3/da3_streaming
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 /home/wzt/.conda/envs/da3nested/bin/python da3_streaming.py \
  --image_dir /home/wzt/wzt/CVPR_2027/Video2PointCloud/data/scannet/scene0000_00_uniform500/rgb \
  --config /home/wzt/wzt/CVPR_2027/Video2PointCloud/src/da3/da3_streaming_uniform500.yaml \
  --output_dir /home/wzt/wzt/CVPR_2027/Video2PointCloud/output/da3/scene0000_00_uniform500_streaming_r2
```

MapAnything serial streaming wrapper:

```bash
cd /home/wzt/wzt/CVPR_2027/Video2PointCloud/ThirdParty/map-anything
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 /home/wzt/.conda/envs/mapanything/bin/python \
  /home/wzt/wzt/CVPR_2027/Video2PointCloud/src/mapanything/run_mapanything_streaming.py \
  --images /home/wzt/wzt/CVPR_2027/Video2PointCloud/data/scannet/scene0000_00_uniform500/rgb \
  --metadata /home/wzt/wzt/CVPR_2027/Video2PointCloud/data/scannet/scene0000_00_uniform500/input \
  --output /home/wzt/wzt/CVPR_2027/Video2PointCloud/output/mapanything/scene0000_00_uniform500_streaming \
  --model /home/wzt/HDD/models/map-anything-a1d87e9 \
  --num-frames 500 --chunk-size 30 --overlap 15
```

## Artifacts

Open the three active output categories together in the [interactive 3D comparison page](../visualization/index.html). It works directly from `file://` and provides independent or synchronized rotate, pan, zoom, color, preset-view, and fullscreen controls. Rebuild instructions and asset provenance are in [`src/visualization/README.md`](../../src/visualization/README.md).

| Method | Full metrics | Point-cloud metrics | Native-scale cloud | Preview |
| --- | --- | --- | --- | --- |
| GT-corrected AMB3R-VO | — | [`pointcloud_gt_corrected_evaluation.json`](../amb3r/vo_uniform500/pointcloud_gt_corrected_evaluation.json) | [`pointcloud_gt_corrected_rgb.ply`](../amb3r/vo_uniform500/pointcloud_gt_corrected_rgb.ply) | [interactive viewer](../visualization/index.html) |
| AMB3R | [`evaluation_metric_head.json`](../amb3r/vo_uniform500/evaluation_metric_head.json) | [`pointcloud_evaluation.json`](../amb3r/vo_uniform500/pointcloud_evaluation.json) | [`pointcloud_se3_scan_world.ply`](../amb3r/vo_uniform500/pointcloud_se3_scan_world.ply) | [`pointcloud_se3_preview.png`](../amb3r/vo_uniform500/pointcloud_se3_preview.png) |
| DA3 Nested | [`evaluation.json`](../da3/scene0000_00_uniform500_streaming_r2/evaluation.json) | [`pointcloud_evaluation.json`](../da3/scene0000_00_uniform500_streaming_r2/pointcloud_evaluation.json) | [`pointcloud_se3_scan_world.ply`](../da3/scene0000_00_uniform500_streaming_r2/pointcloud_se3_scan_world.ply) | [`pointcloud_se3_preview.png`](../da3/scene0000_00_uniform500_streaming_r2/pointcloud_se3_preview.png) |
| MapAnything | [`evaluation.json`](../mapanything/scene0000_00_uniform500_streaming/evaluation.json) | [`pointcloud_evaluation.json`](../mapanything/scene0000_00_uniform500_streaming/pointcloud_evaluation.json) | [`pointcloud_se3_scan_world.ply`](../mapanything/scene0000_00_uniform500_streaming/pointcloud_se3_scan_world.ply) | [`pointcloud_se3_preview.png`](../mapanything/scene0000_00_uniform500_streaming/pointcloud_se3_preview.png) |

## Evidence limits

- One scene cannot establish dataset-level ranking.
- DA3 and MapAnything streaming do not have the same full-sequence context as a hypothetical single-pass 500-frame run.
- AMB3R's scalar comes from only the first 20 selected RGB frames because its metric frontend was used as a bounded scale probe; it is predicted, not GT-fitted.
- Main point-cloud alignment uses GT camera correspondences to establish the ScanNet coordinate frame, but does not alter scale. The Sim(3) table explicitly does use GT scale and is diagnostic only.
- Different official preprocessing grids and confidence definitions remain method-specific.
