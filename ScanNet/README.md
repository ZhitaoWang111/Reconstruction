# ScanNet 数据介绍页

用浏览器直接打开 [index.html](index.html)，无需启动服务。

读原始文件前先看 [docs/raw-data-notes.md](docs/raw-data-notes.md)：两套内参、位姿约定、`axisAlignment`、实例 PNG 的 +1 编码等实测经验。

## 物体级 3D 真值浏览器

打开 [objects/index.html](objects/index.html)。当前导出 3 个具备所需标注文件的本地场景，不要求 `.sens` 已下载完整：

| 场景 | 原始聚合实例 | 简化网格顶点 | 未归属顶点 |
| --- | ---: | ---: | ---: |
| scene0000_00 | 67 | 81,369 | 4,357 |
| scene0000_02 | 71 | 86,001 | 3,723 |
| scene0001_01 | 23 | 124,429 | 7,276 |

按类别或原始 `objectId` 搜索，点击实例列表或三维表面选择，Shift + 点击多选；选择后可聚焦，或回到全场景视角。可切换网格／全部顶点、实例色／RGB／segment 色，以及灰色场景参照。默认隐藏墙、地面和天花板的彩色标注，取消勾选可显示。原始 ID 从 0 开始，在当前扫描内有效，不作为不同扫描间的共享身份。

数据来自简化标注网格 `*_vh_clean_2.ply`，使用对应 `segs.json` 与 `aggregation.json` 精确取得顶点归属，保留全部顶点及原始类别／实例粒度，未抽样、融合或按预测类别重分组。坐标为米，应用一次 `axisAlignment`。三角面只在三个顶点属于同一标注实例时显示，跨实例边界面不渲染；顶点模式包含全部标注点。光照法线仅用于显示，按原网格三角面计算，不改变几何或归属。默认关闭背面剔除，便于查看扫描表面。

尺寸显示的是对齐世界坐标系的 AABB，不是物体的真实长宽高或位姿标注。扫描本身可能不完整，小物体在简化网格上可能只有很少顶点。例如 scene0000_00 的原始 `shoes` 实例 #58 只有 43 个顶点；马桶为 #36，有 454 个顶点。GT 的划分粒度也未必等于我们的目标粒度。

单选实例后可下载：

- PLY：对齐后的顶点 XYZ、原始 RGB、原网格顶点索引、segment ID、objectId，**是顶点点云，不含三角面**。
- NPZ：上述信息，以及原始扫描坐标、NYU40 顶点标签和对齐矩阵。
- 页面底部“整场景 NPZ”：包含所有顶点、原始三角面、实例归属和坐标变换；`object_ids=-1` 为未归属。
- 每场景 `manifest.json`：原始类别、segment 清单、NYU40 分布、包围盒，以及原文件路径和 SHA-256。

生成与验证（仓库根目录）：

```bash
taskset -c 0 Video2PointCloud/.venv/bin/python ScanNet/prepare_object_gt.py
taskset -c 0 Video2PointCloud/.venv/bin/python ScanNet/check_object_gt.py
```

生成脚本支持 `--root /path/to/scannet_v2` 或 `SCANNET_ROOT`，仅读取原数据，输出到 `ScanNet/assets/object_gt/`。该生成目录沿用仓库资产忽略规则；换机时需迁移或重新生成。依赖 NumPy、plyfile；校验还使用 Chrome 与 websocket-client。渲染器复用本仓库已有 WebGL 相机控制，不需要网络或推理服务。

已核对三个场景全部 161 个实例的原始顶点索引、原坐标、对齐坐标、segment 归属及 PLY 导出；浏览器逐一检查首场景 67 个实例的选择结果，并验证三种颜色、网格／顶点切换、搜索／类别筛选、多选、表面拾取、跨场景切换和移动布局，无运行时异常。[检查结果与截图目录](assets/object_gt/qa/checks.json)。

## 当前机器恢复记录（2026-09-18）

换机时 `assets/` 未迁移，页面引用的 15 张 WebP 全部缺失。已用本地 `/home/seer/wzt/data/scannet_v2/scene0000_00` 原始数据和现有脚本重新生成：

- 原始第 162 帧 RGB、深度、原始／过滤后的 2D 语义与实例图，共 6 张。
- 稠密／简化网格、对应 segment、3D 语义与实例、相机轨迹和首页预览，共 8 张。
- IMU 曲线 1 张。

图片中的颜色用于展示，不改变原始标签。3D 网格图片是抽样顶点预览，不是完整网格文件；2D 标签来自对三维标注的投影。深度图在网页中缩放展示，不能把该预览图当成已完成 RGB-D 标定对齐的数据。

已通过真实 Chrome 的 `file://` 加载检查：15 张图片解码成功、12 个图集选项均可切换、清单筛选正常、移动布局无横向溢出，无 JavaScript 运行时异常。[检查记录](assets/restoration_checks.json)。同时复核了页面的网格顶点／面数、668／669 个 segment、67 个聚合实例，以及第 162 帧 filtered 标注的 30,803 个变化像素。

页面已更新数据根路径，并将原来的“本地 20 条”明确标为旧电脑记录。当前 scene0000_00、scene0000_02 各有 13 个预期命名文件；scene0001_01 没有正式 `.sens`，只有 `.sens.larkcache`；scene0000_01、scene0002_00 为空目录。文件清单检查不代表完整性认证。

## 再次换机时恢复

`assets/` 被仓库现有 Git 规则忽略；仅拉取代码不会带回图片。可以迁移整个 `ScanNet/assets/`，或准备 scene0000_00 的 13 个文件后，在仓库根目录运行：

```bash
SCANNET_ROOT=/home/seer/wzt/data/scannet_v2 MPLBACKEND=Agg taskset -c 0 Video2PointCloud/.venv/bin/python ScanNet/prepare_example_assets.py
```

脚本依赖 NumPy、Matplotlib、Pillow、Open3D、plyfile，当前 `Video2PointCloud/.venv` 已具备。`SCANNET_ROOT` 指向包含场景目录的数据根目录，脚本输出固定在本目录的 `assets/` 中。

## 数据对本项目的意义

除了 RGB，ScanNet 还提供度量深度、相机标定与逐帧位姿、重建表面网格，以及二维／三维语义和实例标注。`segs.json` 将网格顶点映射到几何分段，`aggregation.json` 将分段归入带类别的物体实例，因此可以从标注构建每个物体的 GT 表面点集合，检查我们跨帧关联得到的点云。

这些 GT 数据可用于诊断和评测；若用于算法输入，应单独标明该实验使用了 GT。扫描网格表示观测到的表面，不等于完整物体几何或隐藏部分的体积。2D 语义 PNG 使用 ScanNet 类别 ID，3D `labels.ply` 使用 NYU40 类别 ID，不能直接比较数字。

参考：[ScanNet 官方数据组织与格式](https://github.com/ScanNet/ScanNet#data-organization)。
