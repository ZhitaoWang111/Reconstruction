# ScanNet single-frame depth comparison

This experiment evaluates AMB3R and Depth Anything 3 Nested on ten independent
RGB images from the existing `scene0000_00` uniform-500 extraction. Each call
contains exactly one image, so neither method receives temporal or multi-view
context.

The primary comparison uses each model's native metric-depth prediction without
fitting scale to ScanNet. A per-frame median-scale-aligned score is included only
as a diagnostic. Both methods are evaluated on the same AMB3R center-crop field
of view and exactly the same valid ScanNet depth pixels.

The models must be run serially to fit the 24 GB GPU:

```bash
/home/wzt/.conda/envs/amb3r/bin/python /home/wzt/wzt/CVPR_2027/Video2PointCloud/src/single_frame_depth/run_amb3r_single_frame.py \
  --repo /home/wzt/wzt/CVPR_2027/Video2PointCloud/ThirdParty/amb3r \
  --data /home/wzt/wzt/CVPR_2027/Video2PointCloud/data/scannet/scene0000_00_uniform500/input \
  --checkpoint /home/wzt/wzt/CVPR_2027/Video2PointCloud/checkpoints/amb3r/amb3r.pt \
  --output /home/wzt/wzt/CVPR_2027/Video2PointCloud/output/single_frame_depth/amb3r

PYTHONNOUSERSITE=1 /home/wzt/.conda/envs/da3nested/bin/python /home/wzt/wzt/CVPR_2027/Video2PointCloud/src/single_frame_depth/run_da3_single_frame.py \
  --images /home/wzt/wzt/CVPR_2027/Video2PointCloud/data/scannet/scene0000_00_uniform500/rgb \
  --metadata /home/wzt/wzt/CVPR_2027/Video2PointCloud/data/scannet/scene0000_00_uniform500/input \
  --model /home/wzt/HDD/models/da3nested-gl-1.1-b2359bd \
  --output /home/wzt/wzt/CVPR_2027/Video2PointCloud/output/single_frame_depth/da3
```

Then run the evaluator after both model processes have exited:

```bash
/home/wzt/.conda/envs/amb3r/bin/python \
  /home/wzt/wzt/CVPR_2027/Video2PointCloud/src/single_frame_depth/evaluate_and_visualize.py \
  --data /home/wzt/wzt/CVPR_2027/Video2PointCloud/data/scannet/scene0000_00_uniform500/input \
  --amb3r /home/wzt/wzt/CVPR_2027/Video2PointCloud/output/single_frame_depth/amb3r/prediction.npz \
  --da3 /home/wzt/wzt/CVPR_2027/Video2PointCloud/output/single_frame_depth/da3/prediction.npz \
  --output /home/wzt/wzt/CVPR_2027/Video2PointCloud/output/single_frame_depth/comparison
```
