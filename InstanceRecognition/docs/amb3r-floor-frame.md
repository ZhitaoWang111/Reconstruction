# 交付坐标系 `amb3r_floor`

2026-11-18 定并实施。所有交接文件（物体交接 `handoff/v2` 的后续版本、墙体交接 `walls/v1` 的后续版本）统一使用这个坐标系，替换现在的 `scannet_scan`。

## 一句话定义

`amb3r_floor` 由 AMB3R 自己的重建坐标系扶正得到。原点在第一帧相机正下方的地面上，+Z 垂直于地面向上，+X 是第一帧相机水平方向上的朝向，单位为米。

整个定义只用 AMB3R 的输出和我们自己分割出的地面。不读 ScanNet 的任何东西：既不读 GT 位姿，也不读 `axisAlignment`，也不读标注。唯一的例外是尺度，见下文“尺度”一节。

## 轴和原点

| | 定义 |
|---|---|
| 原点 | 序列第一帧（AMB3R `pose[0]`）的相机中心，沿地面法向投到地面平面上的点 |
| +Z | 地面平面的法向，取相机所在的那一侧（所有相机都在地面上方）。z = 0 就是地面 |
| +X | 第一帧相机光轴（OpenCV +Z）在水平面上的投影，归一化 |
| +Y | +Z × +X（右手系；从上往下看，+Y 在 +X 左边 90°） |
| 单位 | 米 |
| 手性 | 右手系，`det(R) = +1` |

和 ScanNet 的 `axisAlignment` 不同，这里不做“摆正”：墙不一定和 X、Y 轴平行。房间主轴由墙体模块自己估计（`close.py` 的 `axis_deg`），写在墙体交接里，下游需要时自己转。

## 怎么从 AMB3R 得到它

记 AMB3R 原生坐标系里的点为 `p_n`、相机位姿为 `T_n`（4×4 camera-to-world，OpenCV 约定：+X 右、+Y 下、+Z 前；scene0000_00 已验证：第一帧所有有效点在相机前方，图像上边缘的点 y < 0）。

1. **尺度**：`p_s = s · p_n`。`s` 的来源见下一节。
2. **地面**：取结构分割（`src/structure/segment`）中所有 `floor` mask 的像素，抬到 `p_s`。用 RANSAC 拟合一个平面（内点带宽 3 cm，随机种子固定为 0），再用内点做 PCA 重新拟合，得到点 `c` 和单位法向 `n`。
3. **朝向**：如果相机中心到平面的有符号距离的中位数小于 0，就令 `n = -n`。不假设原生坐标系的哪根轴朝上。
4. **轴**：
   - `z = n`
   - `f = R_n0 · [0, 0, 1]`，即第一帧的光轴方向
   - `x = normalize(f − (f·z) z)`
   - `y = z × x`
   - 如果第一帧几乎垂直向下或向上拍（`|f·z| > 0.95`），水平分量太小，就依次换用后面的帧，并把实际用的帧号记入定义文件。
5. **原点**：`o = c0 − ((c0 − c)·n) n`，其中 `c0 = s · t_n0` 是第一帧相机中心。
6. **变换**：`R = [x; y; z]`（按行排），则 `p_floor = R (s · p_n − o)`。

合成一个 4×4 矩阵 `M_native_to_floor`（含尺度）：

```
M = | s·R   −R·o |
    | 0       1  |
p_floor = M @ [p_n, 1]
```

相机位姿（camera-to-world，OpenCV）的变换。旋转不乘尺度，平移乘尺度：

```
R_floor = R · R_n
t_floor = R · (s · t_n − o)
```

相机内参不受影响。

## 尺度

AMB3R 的 metric head 有尺度误差：scene0000_00 上偏大约 12.7%。因此 `s` 的取值方式要写明，并在交接文件里标注：

| `scale.source` | 含义 | 何时用 |
|---|---|---|
| `oracle_gt_poses` | `s` 取 AMB3R 相机中心对 ScanNet GT 相机中心做 Umeyama Sim(3) 拟合得到的尺度（`pointcloud_gt_corrected_evaluation.json` → `sim3_oracle_scale.alignment_scale`）。只有尺度来自 GT，旋转和平移都不用。 | 仅限 ScanNet 场景。scene0000_00 目前用它，`s = 0.887500745504967` |
| `native` | `s = 1`，直接用 metric head 的尺度 | 没有任何尺度参照时 |
| （待定） | 不依赖 GT 的尺度估计，例如用门高、门宽、台面高等已知尺寸校正 | 换场景前要做的工作 |

`oracle_gt_poses` 是暂时的折中：坐标系先定下来，下游拿到的尺寸不会突然变大 12.7%。用到它的交接文件必须在 manifest 里写明。

## 交接文件里怎么写

manifest（墙体交接的顶层字段同理）：

```json
"coordinate_frame": "amb3r_floor",
"units": "meters",
"up_axis": "Z",
"frame": {
  "definition": "docs/amb3r-floor-frame.md",
  "file": "<场景的坐标系定义文件，见下>",
  "native_to_frame": [[...4x4, 含尺度...]],
  "scale": {"value": 0.8875, "source": "oracle_gt_poses",
            "note": "AMB3R 尺度由 ScanNet GT 相机位姿校正；只有尺度来自 GT。换场景时没有这个值"},
  "first_frame": {"source_frame_id": 0, "amb3r_index": 0}
}
```

- 地面就是 z = 0，所以现有的 `floor_z` 字段固定为 0，保留只是为了兼容。
- `axis_alignment` 字段删除。
- 关键帧的 `camera_to_world` 用上面的公式转到这个坐标系，约定仍是 OpenCV。

每个场景的坐标系只定义一次，存成一个文件，物体交接和墙体交接都引用它，不各自再拟合一遍。建议路径是 `artifacts/scenes/<scene>/amb3r_floor.json`，内容为：

- `native_to_frame`
- 尺度及其来源
- 地面拟合的统计：点数、残差中位数和 p90、相机离地高度
- 使用的帧
- 生成它的结构分割 run

## 和 ScanNet 的关系（只用于评估）

在 ScanNet 场景上，为了和 GT 比较，可以额外算一个刚体变换 `scan_to_frame`：

```
p_floor = R · (R_sim3ᵀ (p_scan − t_sim3) − o)
```

- 这个变换只写进评估代码和评估结果，不写进交接文件，也不能被任何生产代码读取。
- GT 交接（`src/handoff/from_gt.py`）用它把 GT 网格和 GT 位姿转到 `amb3r_floor`。这样 GT 交接和 harness 交接仍然可以直接叠放，下游不需要知道是谁生成的。
- `axisAlignment` 只在评估时、需要 NYU 摆正坐标系时才用，同样不进交接文件。

## scene0000_00 的数值（2026-11-18 实测）

数值来自 `struct_v4_win5` 的 38 个地面 mask，共 160 万点，尺度为 `oracle_gt_poses`。

| 项 | 值 |
|---|---|
| 地面拟合残差 | 中位数 1.26 cm，p90 3.55 cm |
| 地面法向（缩放后的原生坐标系） | (−0.008, −0.911, −0.413)，与原生 +Z 相差约 114° |
| 相机离地高度（500 帧） | 中位数 1.31 m，最低 1.16 m |
| 第一帧 | 离地 1.33 m，俯视 24°，水平分量 0.914 |
| `scan → amb3r_floor` | 绕竖直轴约 105°；scan 的 +Z 和本坐标系 +Z 相差 0.45° |

`M_native_to_floor`（含尺度 0.8875）：

```
[[ 0.0223  -0.3663   0.8081   0.0136]
 [-0.8872  -0.0027   0.0233   0.0098]
 [-0.0071  -0.8084  -0.3663   1.3366]
 [ 0.       0.       0.       1.    ]]
```

`scan_to_frame`（评估用，刚体变换）：

```
[[-0.2579  -0.9662   0.0015   3.5869]
 [ 0.9662  -0.2578   0.0077  -1.8036]
 [-0.0071   0.0035   1.      -0.0718]
 [ 0.       0.       0.       1.    ]]
```

数值即 `artifacts/scenes/scene0000_00/amb3r_floor.json`（`python -m src.geometry.floor_frame --run struct_v4_win5` 生成）。

## 和现在坐标系的对照

| | 现在 | 改为 |
|---|---|---|
| 交付坐标系 | `scannet_scan`：AMB3R 经 GT 位姿拟合的 Sim(3) 转入 | `amb3r_floor`：AMB3R 原生坐标系乘尺度后扶正 |
| 竖直方向 | ScanNet 扫描坐标系的 +Z | 自己拟合的地面法向 |
| 水平朝向、原点 | ScanNet 扫描时的随意设定 | 第一帧相机 |
| 尺度 | GT（oracle） | 暂时 GT（oracle），并标明来源；之后换成不依赖 GT 的估计 |
| 墙体流水线内部 | scan → lift（`axisAlignment`）→ floor | 原生 → `amb3r_floor`，一步到位，lift 坐标系取消 |
| 用到 GT 的地方 | Sim(3) 的旋转、平移、尺度，以及 `axisAlignment` | 只有尺度，且在 manifest 里标明 |

## 实施情况（2026-11-18）

| 项 | 位置 |
|---|---|
| 拟合与变换 | `src/geometry/floor_frame.py`：`build` 写出 `artifacts/scenes/<scene>/amb3r_floor.json`；`Frame.points` / `Frame.pose` 做原生 → 本坐标系；`scannet_scan_to_frame` 只给评估和 GT 交接用 |
| 墙体流水线 | `src/structure/*`：lift、rays、walls、close 都直接在本坐标系里算，lift 坐标系和 `axisAlignment` 已删除；`scene_rays.npz` 记下写入时的 `native_to_frame`，坐标系变了会拒绝读取 |
| 墙体交接 | `walls/v2`：`coordinate_frame: amb3r_floor`，新增 `frame`，删除 `floor_frame_to_scan`、开口的 `corners_scan`、`storey.floor_plane_scan` |
| 物体交接 | `handoff/v3`：`coordinate_frame: amb3r_floor`，新增 `frame`，删除 `axis_alignment`；GT 交接输出到 `artifacts/handoff/scene0000_00/gt_amb3r_floor/`，旧的 `gt/`（v2）暂留给下游 |
| 下游 | `SceneAssembly/src/export_usd.py` 的坐标系名改为从交接 manifest 读（`handoff.py` 加了 `frame` 字段，旧交接默认 `scannet_scan`） |

和规定有出入的一处：`floor_z` 不是固定为 0。harness 交接的地面就是 z = 0；GT 交接的物体站在 GT 网格的地面上，它在本坐标系里位于 z = −0.026 m（oracle Sim(3) 没有把两个地面完全对齐），所以 GT 交接的 `floor_z` 写 −0.026。检查器要求 `|floor_z| < 0.1 m`。

迁移核对（把旧交接用 `scannet_scan_to_frame` 换到本坐标系再比）：

- 墙体：7 面墙一一对上，墙端点最多差 1.6 cm，关键帧位姿差 < 1e-6，16 张关键帧里 15 张不变（int_3 少了第 1341 帧，它新增的墙面不到 0.15 m²）。外墙的名字（W/S/E/N）按房间主方向起，主方向换了，所以同一面墙换了名字（旧 ext_W → 新 ext_S，依次轮换）。
- 开口：同一处，宽高变了一格（0.60 × 1.90 → 0.70 × 1.80 m；GT 0.63 × 1.95 m）。射线检验在 10 cm 格子上算，格子相对门洞的位置一变，边界就挪一格。
- 稀疏点的面积：墙点按 3 cm 体素去重来算面积，体素原来和坐标轴对齐，墙斜 35° 时面积会被算大约 13%。现在体素转到房间主方向（`walls.params.voxel_axis_deg`），结果不再依赖 +X 朝哪。
- GT 交接：67 个物体，点与位姿换算后差 < 1e-5 m，关键帧选择和图片逐字节相同。
