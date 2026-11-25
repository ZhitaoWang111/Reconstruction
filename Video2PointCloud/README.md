# Video2PointCloud

This workspace keeps upstream repositories, local integration code, inputs, checkpoints, and generated artifacts separate:

```text
src/          local runners, evaluators, and visualization source
data/         extracted ScanNet inputs
output/       experiment results, reports, checksums, and the built viewer
.venv/        runtime environment (pytorch3d and utils3d are built into it)
```

Upstream checkouts live in the repository-root `Submodule/` (`amb3r`,
`depth-anything-3`, `map-anything`, `pytorch3d`, `utils3d`); scripts take the
checkout path from `../Submodule/<name>`.

The canonical downstream point cloud is
[`output/amb3r/vo_uniform500/pointcloud_gt_corrected_rgb.ply`](output/amb3r/vo_uniform500/pointcloud_gt_corrected_rgb.ply).
Its provenance and fitted GT Sim(3) are stored beside it in
[`pointcloud_gt_corrected_evaluation.json`](output/amb3r/vo_uniform500/pointcloud_gt_corrected_evaluation.json).

On 2026-09-17 the AMB3R uniform-500 reconstruction and downstream mask viewer
were regenerated on the RTX 5070 Ti, using `/home/seer/wzt/weights/amb3r/amb3r.pt`.
The existing viewer is at
[`InstanceRecognition/runtime/mask_pointclouds/scene0000_00/amb3r_gt_raw/index.html`](../InstanceRecognition/runtime/mask_pointclouds/scene0000_00/amb3r_gt_raw/index.html).
With the prepared local `.venv`, rerun missing stages from the repository root:

```bash
bash Video2PointCloud/src/amb3r/restore_mask_viewer.sh
```

`AMB3R_CHECKPOINT`, `AMB3R_PYTHON`, and `SCANNET_ROOT` override local paths.
See the [current restoration report](../InstanceRecognition/runtime/reports/MASK_POINTCLOUD_RESTORE_2026-09-17.md)
for validation, runtime compatibility changes, and environment provenance.
Historical 2026-09-13 reports retain the original machine's measurements.

See [`src/README.md`](src/README.md) for runner locations and
[`output/README.md`](output/README.md) for the artifact layout.
