# MapAnything on ScanNet `scene0000_00`

> The exact InstanceRecognition uniform-500 run is complete. A direct 500-frame call exceeds 24 GB VRAM, so the recorded result uses a local serial 30/15-overlap wrapper around the official inference API. See the [combined 500-frame report](../../output/reports/SCANNET_SCENE0000_00_UNIFORM500_RESULTS.md) and [`output/mapanything/scene0000_00_uniform500_streaming`](../../output/mapanything/scene0000_00_uniform500_streaming). The sections below preserve the earlier 40-frame reproduction.

This is an RGB-only reproduction of the official MapAnything image inference path. ScanNet depth and poses are used only after inference for evaluation.

## Frozen inputs and versions

- Upstream: `/home/wzt/wzt/CVPR_2027/Video2PointCloud/ThirdParty/map-anything`
- Upstream commit: `3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9`
- Model: `facebook/map-anything`, revision `a1d87e9086706fb9974f3be5a3e3a0ca5401c5aa`
- Model file: `/home/wzt/HDD/models/map-anything-a1d87e9/model.safetensors`
- Model SHA-256: `981f060c64664dff3272b5f5a823d350abe71a2f144444db4cfc325f3ed5a3a0`
- Model license: CC-BY-NC 4.0; upstream code license: Apache-2.0
- Environment: `/home/wzt/.conda/envs/mapanything` (Python 3.11, PyTorch 2.5.0+cu118)
- Input: the first ScanNet v2 scene, frames `0,10,...,390` (40 RGB frames at 640x480)
- Model preprocessing: official `load_images`, producing 518x392 images

## Re-run

```bash
env -u ALL_PROXY -u all_proxy HF_HOME=/home/wzt/HDD/huggingface PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 /home/wzt/.conda/envs/mapanything/bin/python /home/wzt/wzt/CVPR_2027/Video2PointCloud/src/mapanything/run_mapanything.py --images /home/wzt/wzt/CVPR_2027/Video2PointCloud/data/scannet/scene0000_00_stride10_40f/rgb --metadata /home/wzt/wzt/CVPR_2027/Video2PointCloud/data/scannet/scene0000_00_stride10_40f/input --output /home/wzt/wzt/CVPR_2027/Video2PointCloud/output/mapanything/scene0000_00_40f --num-frames 40 --model /home/wzt/HDD/models/map-anything-a1d87e9

PYTHONNOUSERSITE=1 /home/wzt/.conda/envs/mapanything/bin/python /home/wzt/wzt/CVPR_2027/Video2PointCloud/src/common/scannet_repro_common.py --prediction /home/wzt/wzt/CVPR_2027/Video2PointCloud/output/mapanything/scene0000_00_40f/prediction.npz --data /home/wzt/wzt/CVPR_2027/Video2PointCloud/data/scannet/scene0000_00_stride10_40f/input --preprocess mapanything_cover_crop --output /home/wzt/wzt/CVPR_2027/Video2PointCloud/output/mapanything/scene0000_00_40f/evaluation.json --ply /home/wzt/wzt/CVPR_2027/Video2PointCloud/output/mapanything/scene0000_00_40f/pointcloud.ply
```

## Measured result

- 40-frame inference: 5.44 s (7.35 FPS), peak allocated GPU memory 10.91 GB
- Finite depth: 100%; exported cloud: 2,000,000 finite points
- Raw metric depth: AbsRel 0.4507, RMSE 0.9189 m, delta1 0.0013
- Median-scale-aligned depth: AbsRel 0.0240, RMSE 0.0670 m, delta1 0.9986
- Learned output is about 1.452x too large on this clip (GT/pred median scale 0.6887)
- Trajectory: SE(3) ATE 0.2423 m; Sim(3) ATE 0.0465 m; estimated path 7.834 m vs GT 5.327 m
- Continuity check: maximum step 0.459 m, median step 0.180 m, ratio 2.55

The preview is geometrically coherent, but the raw metric scale is not credible on this clip. The low aligned depth and pose errors show that most of the discrepancy is a common scale bias, not a failure to recover scene shape.

Artifacts are in [`output/mapanything/scene0000_00_40f`](../../output/mapanything/scene0000_00_40f). The 4-frame smoke output is retained separately under `output/mapanything/smoke_4f`.
