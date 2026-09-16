# `scene0000_00`: AMB3R vs DA3 single-frame metric depth

## Conclusion

On these ten endpoint-inclusive, uniformly spaced frames, **DA3 Nested is more
accurate in native metric scale**, while **AMB3R preserves per-frame depth shape
slightly better once an oracle scalar is removed**. The first result is the
deployable comparison; the second only diagnoses where each model's error comes
from.

| Macro average over 10 frames | AMB3R | DA3 Nested | Better |
|---|---:|---:|---|
| AbsRel (native metric) | 9.58% | **5.48%** | DA3 |
| RMSE (native metric) | 0.251 m | **0.178 m** | DA3 |
| MAE (native metric) | 0.212 m | **0.120 m** | DA3 |
| delta1 (native metric) | **98.84%** | 97.52% | AMB3R |
| AbsRel (per-frame median-scale aligned; diagnostic) | **2.06%** | 2.81% | AMB3R |

DA3 wins native-scale AbsRel on 6/10 frames. Its mean AbsRel is about 43% lower
than AMB3R's. AMB3R's predicted depth is usually too large: its pooled median
GT/prediction scale is `0.906`, versus `0.984` for DA3. This systematic scale
bias explains why AMB3R can look geometrically clean while scoring worse in
meters.

The delta1 exception is caused mainly by source frame 1241: DA3 has a larger
structural/outlier region there (`delta1=80.44%`), even though its AbsRel remains
below AMB3R's. Therefore no single metric alone captures all visible failure
modes.

## Per-frame native-scale results

| uniform-500 index | ScanNet source frame | AMB3R AbsRel | DA3 AbsRel | AMB3R RMSE | DA3 RMSE | AbsRel winner |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 13.56% | **3.79%** | 0.262 m | **0.085 m** | DA3 |
| 55 | 615 | 13.59% | **4.81%** | 0.352 m | **0.136 m** | DA3 |
| 111 | 1241 | 15.75% | **6.80%** | **0.406 m** | 0.462 m | DA3 |
| 166 | 1855 | 8.16% | **2.86%** | 0.265 m | **0.145 m** | DA3 |
| 222 | 2481 | **1.81%** | 1.81% | **0.079 m** | 0.086 m | AMB3R |
| 277 | 3096 | **2.50%** | 7.96% | **0.128 m** | 0.167 m | AMB3R |
| 333 | 3722 | **2.89%** | 8.13% | **0.142 m** | 0.218 m | AMB3R |
| 388 | 4336 | 16.19% | **2.68%** | 0.404 m | **0.138 m** | DA3 |
| 444 | 4962 | **10.22%** | 11.31% | **0.209 m** | 0.223 m | AMB3R |
| 499 | 5577 | 11.16% | **4.66%** | 0.261 m | **0.120 m** | DA3 |

## Protocol and boundaries

- Every inference call contains exactly one RGB frame. There is no temporal or
  multi-view context, and AMB3R-VO is not involved.
- AMB3R is its metric-depth frontend (`AMB3R(metric_scale=True)`, `iters=0`);
  DA3 is `DA3NESTED-GIANT-LARGE-1.1` at official `process_res=504`.
- Both are evaluated on AMB3R's 518x392 center-crop field of view and the exact
  same 1,848,116 valid ScanNet depth pixels (`0 < depth < 10 m`).
- Native metric scores do not use ScanNet to fit scale. Per-frame median-scale
  alignment is explicitly an oracle diagnostic and cannot be used at inference.
- Ten frames are enough to expose method behavior on this scene, but not enough
  to establish a dataset-wide ranking.

Machine-readable metrics are in [`output/single_frame_depth/comparison/comparison.json`](../single_frame_depth/comparison/comparison.json); the visual
contact sheet is [`depth_comparison_10_frames.png`](../single_frame_depth/comparison/depth_comparison_10_frames.png).
