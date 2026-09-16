# RGB video to scaled point cloud: `scene0000_00`

> The current full-scene-style experiment uses the exact 500-frame sampling from InstanceRecognition. See [`SCANNET_SCENE0000_00_UNIFORM500_RESULTS.md`](SCANNET_SCENE0000_00_UNIFORM500_RESULTS.md). The table below is the earlier bounded 40-frame experiment.

All rows use the same ScanNet v2 RGB frames `0,10,...,390`. No ScanNet depth, intrinsics, or poses are supplied to inference; depth and poses are used only by the evaluator. Results are a bounded 40-frame reproduction, not a ScanNet benchmark.

| Method | Metric source | Raw depth AbsRel | Raw depth RMSE | GT/pred depth scale | SE(3) ATE | Sim(3) ATE | Pred/GT path |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DA3Nested-Giant-Large-1.1 | nested metric head | **2.37%** | **0.0726 m** | **0.9846** | **3.87 cm** | 3.39 cm | **1.028** |
| MapAnything | learned metric scale | 45.07% | 0.9189 m | 0.6887 | 24.23 cm | 4.65 cm | 1.470 |
| AMB3R-VO + predicted metric-head scalar | separate 20-frame AMB3R metric estimate | 11.44% | 0.2392 m | 0.8972 | 7.62 cm | **2.92 cm** | 1.121 |
| Official AMB3R-VO output | arbitrary-scale VO | 50.65% | 1.0311 m | 2.0241 | 27.73 cm | **2.92 cm** | 0.497 |

Interpretation: DA3 Nested is the only tested path whose raw output is close to ScanNet metric scale on this clip. MapAnything reconstructs shape and motion well after a single global scale alignment (aligned depth AbsRel 2.40%, Sim(3) ATE 4.65 cm), but its raw scale is about 45% too large. AMB3R-VO is arbitrary-scale in the tested official path; applying AMB3R's separately predicted metric scalar narrows but does not eliminate its scale error.

Native inference grids differ by official preprocessing: DA3 uses 504x378, while MapAnything and this AMB3R run use 518x392. Therefore the table is useful for this concrete clip, but is not a controlled leaderboard.

Detailed provenance, commands, visual previews, full metrics, and artifacts:

- [Depth Anything 3 Nested](../../src/da3/README.md)
- [MapAnything](../../src/mapanything/README.md)
- [AMB3R / AMB3R-VO](../../src/amb3r/README.md)
