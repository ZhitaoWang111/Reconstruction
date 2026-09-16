# Depth Anything 3 Nested on ScanNet `scene0000_00`

> The exact InstanceRecognition uniform-500 run is complete. It uses official DA3-Streaming with 30-frame windows and 15-frame overlap. See the [combined 500-frame report](../../output/reports/SCANNET_SCENE0000_00_UNIFORM500_RESULTS.md) and [`output/da3/scene0000_00_uniform500_streaming_r2`](../../output/da3/scene0000_00_uniform500_streaming_r2). The sections below preserve the earlier 40-frame reproduction.

This is an RGB-only reproduction of the official Depth Anything 3 Nested inference path. ScanNet depth and poses are used only after inference for evaluation.

## Frozen inputs and versions

- Upstream: `/home/wzt/wzt/CVPR_2027/Video2PointCloud/ThirdParty/depth-anything-3`
- Upstream commit: `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`
- Model: `depth-anything/DA3NESTED-GIANT-LARGE-1.1`, revision `b2359bdf726fb44ef62acca04d629dcf158053e7`
- The `-1.1` checkpoint is the upstream-recommended replacement for the deprecated original checkpoint after a training-bug fix.
- Model file: `/home/wzt/HDD/models/da3nested-gl-1.1-b2359bd/model.safetensors`
- Model SHA-256: `8ebe871a022ed58d2fc8fdfb2ebdb31d57b60fe39611c849095851a7b7c6020c`
- Model license: CC BY-NC 4.0; upstream code license: Apache-2.0
- Environment: `/home/wzt/.conda/envs/da3nested` (Python 3.11, PyTorch 2.5.0+cu118, xFormers 0.0.28.post2)
- Input: the first ScanNet v2 scene, frames `0,10,...,390` (40 RGB frames at 640x480)
- Model preprocessing: official upper-bound resize to 504x378; reference-view strategy `middle`

The optional `gsplat` package is not installed because this reproduction exports a depth-unprojected point cloud, not the optional 3D Gaussian rendering branch.

## Re-run

```bash
env -u ALL_PROXY -u all_proxy HF_HOME=/home/wzt/HDD/huggingface PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 /home/wzt/.conda/envs/da3nested/bin/python /home/wzt/wzt/CVPR_2027/Video2PointCloud/src/da3/run_da3nested.py --images /home/wzt/wzt/CVPR_2027/Video2PointCloud/data/scannet/scene0000_00_stride10_40f/rgb --metadata /home/wzt/wzt/CVPR_2027/Video2PointCloud/data/scannet/scene0000_00_stride10_40f/input --output /home/wzt/wzt/CVPR_2027/Video2PointCloud/output/da3/scene0000_00_40f --num-frames 40 --model /home/wzt/HDD/models/da3nested-gl-1.1-b2359bd

PYTHONNOUSERSITE=1 /home/wzt/.conda/envs/da3nested/bin/python /home/wzt/wzt/CVPR_2027/Video2PointCloud/src/common/scannet_repro_common.py --prediction /home/wzt/wzt/CVPR_2027/Video2PointCloud/output/da3/scene0000_00_40f/prediction.npz --data /home/wzt/wzt/CVPR_2027/Video2PointCloud/data/scannet/scene0000_00_stride10_40f/input --preprocess da3_resize --output /home/wzt/wzt/CVPR_2027/Video2PointCloud/output/da3/scene0000_00_40f/evaluation.json --ply /home/wzt/wzt/CVPR_2027/Video2PointCloud/output/da3/scene0000_00_40f/pointcloud.ply
```

## Measured result

- 40-frame inference: 3.84 s (10.43 FPS), peak allocated GPU memory 14.82 GB
- Finite depth: 100%; exported cloud: 2,000,000 finite points
- Raw metric depth: AbsRel 0.0237, RMSE 0.0726 m, delta1 0.9955
- Median-scale-aligned depth: AbsRel 0.0184, RMSE 0.0661 m, delta1 0.9957
- Metric scale is close: GT/pred median scale 0.9846 (about 1.6% high)
- Trajectory: SE(3) ATE 0.0387 m; Sim(3) ATE 0.0339 m; estimated path 5.475 m vs GT 5.327 m
- Continuity check: maximum step 0.333 m, median step 0.113 m, ratio 2.94

On this 40-frame clip, both the metric depth and camera trajectory are credible, and the visually inspected point cloud preserves the room layout without the large global scale bias seen in the MapAnything run.

Artifacts are in [`output/da3/scene0000_00_40f`](../../output/da3/scene0000_00_40f). The 4-frame smoke output is retained separately under `output/da3/smoke_4f`.
