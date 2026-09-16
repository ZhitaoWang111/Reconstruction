# Video2PointCloud

This workspace keeps upstream repositories, local integration code, inputs, checkpoints, and generated artifacts separate:

```text
ThirdParty/   unmodified upstream Git checkouts only
src/          local runners, evaluators, and visualization source
data/         extracted ScanNet inputs
checkpoints/  local model weights or links to external model stores
output/       experiment results, reports, checksums, and the built viewer
```

The canonical downstream point cloud is
[`output/amb3r/vo_uniform500/pointcloud_gt_corrected_rgb.ply`](output/amb3r/vo_uniform500/pointcloud_gt_corrected_rgb.ply).
Its provenance and fitted GT Sim(3) are stored beside it in
[`pointcloud_gt_corrected_evaluation.json`](output/amb3r/vo_uniform500/pointcloud_gt_corrected_evaluation.json).

See [`src/README.md`](src/README.md) for runner locations and
[`output/README.md`](output/README.md) for the artifact layout.
