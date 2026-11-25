# ScanNet 原始数据：实测经验与坑

本文记录直接读 ScanNet v2 原始文件时核对出的事实，ScanNet 官方文档里没写，或者写得容易误读。

**范围**：除另有说明，所有数字都在 `scene0000_00` 上实测（2026-11 前后），其他场景没有逐一复核。每条都标了依据：

- **实测**：在本机数据上跑过，给了数字。
- **早期实测**：之前的会话测过，结论记在代码注释或项目记录里，2026-11-11 写本文时没有重跑。
- **推断**：和实测结果一致，但没有找到官方代码或文档证实。

数据根目录：`/home/seer/wzt/data/scannet_v2/<scene>/`，没有 `scans/` 这一层，可以用 `SCANNET_ROOT` 覆盖。本机数据布局见 [`docs/scannet-data.md`](../../docs/scannet-data.md)，网页介绍见 [`../README.md`](../README.md)。

---

## 1. 一个场景目录里有什么

| 文件 | 内容 | 本项目用途 |
|---|---|---|
| `<scene>.sens` | 原始 RGB-D 流：彩色 JPEG、深度 zlib 压缩、逐帧位姿、文件头带内参 | 抽帧（`InstanceRecognition/src/geometry/extract_scannet_sens.py`） |
| `<scene>.txt` | 场景元数据：`axisAlignment`、彩色/深度内参、宽高、`colorToDepthExtrinsics`、帧数 | 交接文件的内参和轴对齐矩阵 |
| `<scene>_vh_clean_2.ply` | 简化标注网格（scene0000_00：81,369 个顶点、153,587 个面），带顶点颜色 | GT 物体点云 |
| `<scene>_vh_clean_2.0.010000.segs.json` | 每个顶点属于哪个 segment（`segIndices`，长度等于顶点数） | 顶点 → segment |
| `<scene>.aggregation.json` | segment → 物体实例（`segGroups`，带 `objectId` 和 `label`） | segment → 物体 |
| `<scene>_2d-instance.zip` / `-filt.zip` | 每帧的 2D 实例图 | 关键帧 mask |
| `<scene>_2d-label.zip` / `-filt.zip` | 每帧的 2D 语义图 | 未使用 |
| `<scene>_vh_clean_2.labels.ply` | 带 NYU40 语义标签的网格 | 未使用 |
| `<scene>_vh_clean.ply` 等 | 稠密网格及其 segs / aggregation | 未使用 |

---

## 2. `.sens`

**实测，scene0000_00：**

- 格式版本 4，共 5,578 帧；彩色 1296×968 JPEG，深度 640×480 zlib uint16。`depth_shift = 1000`，即深度值单位为毫米。
- **所有帧的时间戳都是 0**：彩色和深度都一样，全场景只有一个时间戳值。所以没法按时间抽帧，只能按帧号均匀抽。传感器帧率恒定，按帧号抽等价于按时间抽。
- **5,578 帧位姿全部有限**（逐帧扫描确认）。官方说明里提到，其他场景可能有跟踪失败的帧，位姿是 `-inf`；抽帧代码会在每帧记录 `pose_finite`，换场景时先看这一项。
- 文件头里的彩色外参和深度外参都是单位阵。
- 文件头的传感器名是 `StructureSensor (calibrated)`。

---

## 3. 两套内参（最容易出错）

`.txt` 和 `.sens` 文件头里各有一套彩色内参和一套深度内参，数值不同：

| | 彩色 fx / fy / cx / cy | 深度 fx / fy / cx / cy |
|---|---|---|
| `.txt` | 1170.188 / 1170.188 / 647.75 / 483.75 | 571.624 / 571.624 / 319.5 / 239.5 |
| `.sens` 文件头 | 1169.62 / 1167.11 / 646.30 / 489.93 | 577.59 / 578.73 / 318.91 / 242.68 |

两套的特征不同（**实测**）：`.txt` 的深度主点正好在 640×480 的几何中心 (319.5, 239.5)，fx 和 fy 完全相等，是标称值；`.sens` 的主点偏离中心，fx 和 fy 也不相等，像逐台设备标定的结果。

**各自和哪份数据对得上（实测）：**

- **彩色内参对 2D 实例标注**：把每个物体的 GT 顶点用 GT 位姿投影到关键帧（61 个物体的 187 个关键帧），和实例 mask 比对。
  - 用 `.txt` 内参，最佳对齐偏移是 **(0, 0) px**，89.9% 的投影顶点落在 mask 内。
  - 用 `.sens` 内参，要偏移 **(+2, −6) px** 才对得上，落在 mask 内的是 87.4%。
- **深度内参对 GT 网格**：抽 20 帧深度图反投影，算到 GT 网格的中位距离。
  - `.sens` 是 2.48 cm，`.txt` 是 2.68 cm，差别很小。

**推断**：2D 标注是用 `.txt` 的内参从 3D 网格渲染出来的，`.sens` 里是采集时记录的设备标定值。

**怎么用：**

- 要和 2D 标注（instance / label PNG）或从网格渲染出的图对齐时，用 **`.txt`** 的彩色内参。交接文件 `handoff/v2` 用的就是这套。
- 用深度图反投影时，两套都可以，`.sens` 略好一点。
- 注意：`Video2PointCloud/data/scannet/<scene>_uniform500/manifest.json` 里的 `intrinsic_color` 和 npz 里的 `camera_intrinsics` 都来自 **`.sens`**，其中 npz 存的是深度内参。拿它们去和标注 mask 对齐，会有约 6 px 的竖直偏差。

---

## 4. 位姿

- **约定（实测）**：`.sens` 里的位姿是 **camera→world**，坐标轴是 **OpenCV 约定**（+X 右、+Y 下、+Z 朝前）。用第 3 节同样的投影比对，按这个约定理解时 87% 的投影顶点落在 mask 内；当成 world→camera 只有 2.5%，按 OpenGL 约定（−Z 朝前）是 0%。
- **世界坐标系就是网格的原始扫描坐标系**：位姿和 `vh_clean_2.ply` 可以直接配合使用，**不需要**乘 `axisAlignment`。
- **彩色和深度用同一个位姿（实测）**：`.txt` 里的 `colorToDepthExtrinsics` 不是单位阵（平移约 4 cm）。但把它补乘到位姿上之后，深度反投影到网格的距离反而变差，从 2.48 cm 变成 3.54 cm（用 `.txt` 深度内参是 2.68 → 3.80 cm）。彩色这边，不加任何补偿，标注就是 (0, 0) 偏移对齐。所以用这份数据时，直接拿 `.sens` 的位姿就行，**不要**再叠加 `colorToDepthExtrinsics`。
- 用 AMB3R 等方法估出的位姿经 Sim(3) 对齐到扫描坐标系后，相机中心和 GT 位姿平移部分的差距，在 187 个关键帧上中位 8.0 cm、最大 20.9 cm。需要精确相机位置时用 GT 位姿。

---

## 5. 坐标系与 `axisAlignment`

- 网格、位姿和 2D 标注的投影都在**原始扫描坐标系**里：单位米，Z 轴朝上。
- **原始扫描坐标系本身已经和重力对齐（实测）**：floor 实例拟合出的平面法线和 +Z 的夹角只有 0.135°。
- `axisAlignment`（在 `.txt` 里，按行展开的 4×4 矩阵）在 scene0000_00 上**只做绕 Z 轴的旋转加平移**（旋转部分约 19°，Z 方向平移 −0.064 m），作用是把墙面摆正到坐标轴方向，方便做 NYU 风格的轴对齐处理。
- **不要在不需要的时候应用它（早期实测，见 `InstanceRecognition/src/handoff/schema.py` 文件头）**：AMB3R 点云经 Sim(3) 对齐到原始扫描坐标系后，和 GT 网格的中位表面距离是 0.12 m；如果先对网格应用了 `axisAlignment`，这个距离会变成 2.97 m。
- `ScanNet/objects/` 浏览器页和 `prepare_object_gt.py` 都应用了一次 `axisAlignment`；`handoff/v2` 没有应用，并把矩阵记在 `manifest.axis_alignment` 里。混用这两份数据之前，先确认双方在同一个坐标系。

---

## 6. 3D 实例标注

- **归属链**：顶点 → `segIndices[顶点]` → segment → `aggregation.segGroups[*].segments` → `objectId`。
- scene0000_00 有 67 个 `segGroups`，`objectId` 从 0 到 66，每组的 `id` 等于 `objectId`。有 4,357 个顶点的 segment 不属于任何组，即没有标注。
- `objectId` 只在当前扫描内有效，不能用来跨场景对应物体。
- **标签质量（实测，67 个实例）**：
  - 15 个是建筑结构：wall ×9、floor、ceiling、window ×2、doorframe ×2；
  - 5 个标签只写了 `object`，没有具体类别；
  - 其余 47 个有具体类别。
- **包围盒 ≠ 物体尺寸（实测）**：AABB 反映的是扫描覆盖的范围。例如 table 的 Z 跨度是 1.21 m；L 形 couch 的 X/Y 跨度是 2.83 / 2.90 m；floor 实例里混入了地毯等东西，最高点到 0.36 m。
- **地面高度**：floor 实例顶点 z 的中位数是 **0.0544 m**，5%–95% 分位在 0.029–0.123 m 之间。用最小值或均值都会被异常点拉偏，用中位数。
- 有些物体底部扫描缺失：nightstand（obj_0050）扫描到的最低点在 0.21 m，stool（obj_0031）在 0.13 m，但它们实际都立在地上。不能拿扫描最低点判断物体是否落地。

---

## 7. 2D 实例 / 语义图

- **实例 PNG 的像素值 = `objectId + 1`，0 表示没有标注（早期实测，见 `from_gt.py` 文件头；三条证据）**：
  1. 像素值范围是 0..67，`objectId` 范围是 0..66；
  2. 2D 像素数和 3D 顶点数的对数相关系数，按 +1 偏移对应是 0.85，不偏移是 0.38；
  3. 如果不偏移，把值 10 读成 objectId 10，一个只有 362 个顶点的水槽会覆盖 550 万像素；按 +1 偏移读，这些像素属于地面。
- 实例 PNG 是 8 位灰度（`L` 模式），1296×968，和彩色图同分辨率。
- 语义 PNG（`2d-label`）是 16 位（`I;16`），像素值是 **ScanNet 类别 ID**，不是 NYU40。网格 `labels.ply` 里用的是 NYU40，两者的数字不能直接比较。
- 每种都有 `-filt` 版本，经过平滑或过滤。第 162 帧的 filtered 实例图和原始图有 30,803 个像素不同。本项目用的是**不带 filt** 的版本。
- **2D 标注是从 3D 标注渲染出来的（实测）**：交接文件里的关键帧 mask 和对应帧的实例 PNG 逐像素完全相同（52 个非结构物体，没有多出也没有缺失的像素）。因此：
  - 以下三条是早期实测（2026-11-04，ObjectGeneration 接入交接文件时测的），本次没有重跑：
  - 实例边界会把贴着的配件算进去，比如有一帧的抱枕被算进了 couch；
  - 镂空物体的 mask 有内部空洞（四把 stool 有 10%–21% 的空洞，cabinet 8.4%）；
  - bbox 碰到画面边缘不等于 mask 被截断：67 个碰边的关键帧里，大多数 mask 落在 8 px 边带内的面积都不到 2.5%。

---

## 8. 本项目派生数据的约定

这些不是 ScanNet 原始数据，而是我们抽帧或导出时定下的约定，读取时容易踩坑：

- **抽帧深度 PNG 是归一化的**：`extract_scannet_sens.py` 写的 `NNNN_depth.png` 不是毫米值。还原公式是 `depth_m = png / 65535 × maximum_depth`，`maximum_depth` 在同名 npz 里。
- **抽帧的 RGB 被缩小了**：`Video2PointCloud/.../input/NNNN.jpg` 被缩到深度分辨率 640×480，是给 AMB3R 用的。原分辨率帧在 `InstanceRecognition/artifacts/frames/<scene>/frame_NNNNNN.jpg`（1296×968）。
- **两套帧编号**：
  - `source_frame_id` 是 `.sens` 里的原始帧号；
  - `seq` 是抽样后的序号，从 0 开始连续。
  - 500 帧采样和 100 帧 source 的 `seq` 编号不一样，跨数据集对齐时用 `source_frame_id`。

---

## 相关代码

- `InstanceRecognition/src/handoff/from_gt.py`：从 GT 生成交接文件，第 1–4 条源数据坑也写在它的文件头注释里。
- `InstanceRecognition/src/geometry/extract_scannet_sens.py`：`.sens` 抽帧。
- `ScanNet/prepare_object_gt.py`：物体 GT 浏览器的数据导出，这里应用了 `axisAlignment`。
