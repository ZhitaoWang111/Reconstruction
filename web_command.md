# 本仓库网页打开方法

更新日期：2026-09-18。仓库位置：`/home/seer/wzt/Reconstruction`。

本文收录本项目制作或整理的网页入口，包括离线结果、需要后端服务的推理页面和源码模板。不包含第三方依赖自带的示例或文档。

## 先选对页面

| 想查看什么 | 页面文件 | 打开方式 |
| --- | --- | --- |
| 每帧 RGB、YOLOE 检测、SAM3.1 mask，以及 episode 类别统计 | [InstanceRecognition/runtime/offline/index.html](InstanceRecognition/runtime/offline/index.html) | 离线直接打开，见 1.1。 |
| 每帧 mask 反投影后的 3D 点云 | 页面已于 2026-10-02 清理 | 点云与投影数据仍在 `InstanceRecognition/runtime/mask_pointclouds/scene0000_00/`；需要页面时按 1.2 重新生成。 |
| 模型 Harness：逐窗身份关联、累计点云与审计 | [InstanceRecognition/runtime/experiments/harness-v1-gpt56-terra-30frames-20260921/index.html](InstanceRecognition/runtime/experiments/harness-v1-gpt56-terra-30frames-20260921/index.html) | 离线直接打开；[全部窗口观测与修改证据](InstanceRecognition/runtime/experiments/harness-v1-gpt56-terra-30frames-20260921/audit.html)。运行范围和完成状态见 [报告](InstanceRecognition/runtime/reports/HARNESS_FIRST30_2026-09-21.md)。 |
| 模型 Harness：逐窗口 Agent 交互记录（请求、响应、工具返回、理由中译） | 结果页同目录的 `records.html`，最新一轮为 [harness-v2-gpt6-sol-50frames-10windows-20261013/records.html](InstanceRecognition/runtime/experiments/harness-v2-gpt6-sol-50frames-10windows-20261013/records.html) | 离线直接打开。**每个实验的结果页与交互记录页同在一个实验目录内**；两页可互相跳转，也可用页面上的实验切换栏换轮次。记录页由 `build_harness_records.py --run <实验标识>` 生成，见 1.6。 |
| 前 30 帧增量物体点云：跨帧 ID、累计点云与待定归属 | 页面已于 2026-09-30 清理 | 实验记录仍保留：[InstanceRecognition/runtime/reports/INCREMENTAL_OBJECTS_FIRST30_2026-09-18.md](InstanceRecognition/runtime/reports/INCREMENTAL_OBJECTS_FIRST30_2026-09-18.md)。 |
| 抽样序号 80–140：YOLOE26 与 GPT-5.6 Terra + SAM3.1 对比 | [InstanceRecognition/runtime/comparisons/scene0000_00_0080_0140/index.html](InstanceRecognition/runtime/comparisons/scene0000_00_0080_0140/index.html) | 离线直接打开，见 1.4。 |
| GT 修正 AMB3R、原始 AMB3R、DA3 的点云对比 | [Video2PointCloud/output/visualization/index.html](Video2PointCloud/output/visualization/index.html) | **需先导出**（`output/visualization/` 当前不存在），见 1.3。 |
| 选择 ScanNet 场景／帧，并点击按钮运行模型 | 启动后端后访问 <http://127.0.0.1:8090/>（源码 `InstanceRecognition/src/web/frontend/index.html`，不能直接 `file://` 打开） | 必须启动后端，再访问 HTTP 地址，见第 2 节。 |
| ScanNet 数据包含的信息及示例 | [ScanNet/index.html](ScanNet/index.html) | 离线直接打开，见 3.1。 |
| ScanNet 物体级 3D 标注真值 | [ScanNet/objects/index.html](ScanNet/objects/index.html) | 离线直接打开；选择场景、类别和 objectId，查看网格／顶点并导出。见 3.1。 |
| 三个视觉任务的备选方法 | [memorandum/three-vision-task-methods.html](memorandum/three-vision-task-methods.html) | 离线直接打开，见 3.2。 |
| 视频到 SimReady 的方案讨论与思路演进 | [docs/scene-reconstruction-discussions.html](docs/scene-reconstruction-discussions.html) | 离线直接打开，见 3.3。 |
| 重建论文时间线 | `news/` 下的三个时间线页面 | 离线直接打开，见第 4 节。 |
| 论文阅读记录 | [news/papers/paper_reading_record/index.html](news/papers/paper_reading_record/index.html) 及五篇详情 | 离线直接打开，见第 5 节。 |
| 手写签名采集 | [scannetpp-signature-pad.html](docs/scannetpp-signature-pad.html) | 离线直接打开，见第 6 节。 |

对于下面的离线页面，任选一种打开方式：

- 在终端执行对应的 `google-chrome` 命令；所有命令均使用完整路径，不需要先切换目录。
- 将命令引号内的 `file:///...` 地址复制到浏览器地址栏并按回车。
- 在文件管理器中找到对应 HTML，用浏览器打开。

离线页面不需要启动网站服务或推理模型。请保留 HTML 周围的脚本、样式、数据和图片目录，不要只把单个 HTML 搬走。点击论文、GitHub 等外部链接时才需要访问对应网站。

## 1. 已有实验结果：不需要重新推理

### 1.1 YOLOE-26 与 SAM3.1 离线结果浏览器

```bash
/usr/bin/google-chrome "file:///home/seer/wzt/Reconstruction/InstanceRecognition/runtime/offline/index.html"
```

可以切换已处理的场景和帧，查看原始 RGB、YOLOE 检测、SAM3.1 mask、实例 ID 图，以及按置信度和检出帧数筛选的 episode 类别统计。

2026-09-17 已在当前机器重跑 `scene0000_00`：原始 5,578 帧按原有均匀采样逻辑选出 500 帧（含首帧 0 和末帧 5,577），500 帧全部成功，共 6,790 个 mask，批次耗时约 18 分 48 秒。页面保留之前的测试帧 162、163，因此显示 **502 帧、6,822 个 mask**。全部输出文件与页面交互已检查，查看时无需启动服务。详见 [本次批量结果记录](InstanceRecognition/runtime/reports/BATCH_RESULT_2026-09-17.md)。

它显示离线索引中已经保存的结果，不会自动扫描尚未处理的帧，也不会启动新推理。若新增了推理结果但页面未收录，可以单独更新索引：

```bash
/home/seer/wzt/Reconstruction/InstanceRecognition/.venv/bin/python -m src.web.build_offline_browser
```

### 1.2 每帧 mask 反投影到 3D 点云的检查页

```bash
# 页面已于 2026-10-02 清理，原命令不再可用；点云与投影数据仍在原位。
```

该页面的 `index.html` 曾由反投影脚本从模板 `InstanceRecognition/src/web/frontend/mask_pointcloud/` 复制生成，2026-10-02 随 `mask_pointclouds/*/index.html` 一并清理。当前目录下只剩数据（`manifest.js`、`manifest.json`、`frames/`、`assets/`、`projection_report.json`、`projection_validation.json` 等），用 `file://` 打开不会再显示页面。需要时重新生成：

```bash
/home/seer/wzt/Reconstruction/InstanceRecognition/.venv/bin/python -m src.geometry.backproject_masks \
  --output /tmp/amb3r_gt_raw_viewer
```

`backproject_masks` 出于安全考虑拒绝写入非空目录，因此不能直接重建到原路径；请指定新的 `--output`。重建后的内容与 2026-10-02 前一致（模板未变）。

该入口已于 2026-09-18 更新：继续使用 **GPT-5.6 Terra + SAM3.1**，正式 prompt 允许有视觉依据的 `pair of shoes` 成对类别。原均匀抽样清单的 500 帧不变，其中原标签出现 shoe/shoes 的 83 帧已重新运行 GPT → SAM，其余 417 帧保留原结果。当前展示 **4,481 张 mask**，原始候选 4,895 张，414 张预算外候选仍保留；背景仍为原来的 492,555 个点。左侧查看 2D 图像和当前帧 mask，右侧查看对应的 3D 落点，直接打开无需服务。

没有增加 mask 配对、合并或额外 GPT 后处理。第 6 帧由 GPT 自动输出 `pair of shoes`，SAM 返回两张对应两双鞋的 mask；其他帧可能保留单只类别或得到空分割。完整结果、局限与验证见 [成对类别更新报告](InstanceRecognition/runtime/reports/SHOE_PAIR_REVISION_2026-09-18.md)。

旧 GPT 单数版目录 `gpt_terra_singular_v1_archive/` 与原 YOLOE 目录 `yoloe26_gt_raw/` 的数据均保留，但两者的浏览页面已于 2026-10-02 一并清理。80–140 对比页仍展示原实验数据；[2026-09-17 全场景报告](InstanceRecognition/runtime/reports/GPT_SAM_FULL_SCENE_2026-09-17.md) 为历史版本记录。

几何复用此前使用 `/home/seer/wzt/weights/amb3r/amb3r.pt` 生成的 AMB3R GT 修正结果，本次只替换实例 mask。原重建与恢复记录见 [MASK_POINTCLOUD_RESTORE_2026-09-17.md](InstanceRecognition/runtime/reports/MASK_POINTCLOUD_RESTORE_2026-09-17.md)。当前环境下可复用以下命令恢复缺失阶段，已存在的阶段产物会跳过：

```bash
bash /home/seer/wzt/Reconstruction/Video2PointCloud/src/amb3r/restore_mask_viewer.sh
```

- 左键拖动：旋转。
- 右键拖动：平移。
- 滚轮：缩放。
- “重置视角”：恢复旋转、缩放和平移。

此页展示逐帧原始观测，不代表已经完成跨帧实例合并；实例 ID 只在帧内有效。几何使用 GT 修正结果，不应误解为无 GT 的纯 RGB 最终结果。

### 1.3 三类点云输出对比页

```bash
# 先导出，再打开：
cd /home/seer/wzt/Reconstruction/Video2PointCloud
/usr/bin/python3 src/visualization/build_pointcloud_assets.py     # 默认输出到 output/visualization/
/usr/bin/google-chrome "file:///home/seer/wzt/Reconstruction/Video2PointCloud/output/visualization/index.html"
```

`output/visualization/index.html` 是构建产物，当前**尚未导出**，仓库里现存的是模板 `Video2PointCloud/src/visualization/index.html`。导出后即可离线打开。

对比 GT 修正的 AMB3R-VO、原始 AMB3R-VO 和 DA3 点云，支持同步视角、预设视角、RGB／高度配色及单窗口全屏。左键旋转，右键或 Shift 拖动平移，滚轮缩放。

GT 修正版本属于借助真值的诊断结果，不能与无 GT 方法混同。打开已有页面不需要重新导出点云。

### 1.4 YOLOE26 / GPT + SAM3.1 两种方案对比

```bash
/usr/bin/google-chrome "file:///home/seer/wzt/Reconstruction/InstanceRecognition/runtime/comparisons/scene0000_00_0080_0140/index.html"
```

已完成原 500 帧清单中抽样序号 80–140（含两端）的 61 帧。左侧复用 YOLOE26 框提示 SAM3.1 结果，右侧为 GPT-5.6 Terra 类别文本提示 SAM3.1 结果。页面统一选帧，可查看 RGB、逐个/多个 mask、对应 3D 落点，以及 GPT 的依据、不确定类别和未分出 mask 的提示。无需启动服务或模型。

基线为 792 张 mask，新方案为 479 张；没有人工 GT 评测，数量不同不代表准确率高低。API 调用脚本、英文 prompt、配置与响应记录位于 `/home/seer/wzt/API/scannet_gpt_sam/`。完整记录见 [对比报告](InstanceRecognition/runtime/reports/GPT_SAM_COMPARISON_2026-09-17.md)。

两侧也可单独打开：[YOLOE26](InstanceRecognition/runtime/comparisons/scene0000_00_0080_0140/baseline/index.html)、[GPT](InstanceRecognition/runtime/comparisons/scene0000_00_0080_0140/gpt/index.html)。保留相对目录结构，以便加载既有背景、基线 mask 与 API JSON。

### 1.5 第 6 帧：鞋的单复数提示与按双分组

```bash
# 页面已于 2026-09-30 清理，原启动命令不再执行。
```

单帧实验，抽样序号 6 / 原始帧 67。页面默认显示最新实验：直接指定 `pair of shoes`，SAM 输出 2 张 mask，分别覆盖左右两双，没有配对/合并后处理，也没有新增 GPT 请求。另可切换原方案（4 张鞋 mask）、放宽单复数（本次 3 张鞋 mask）、在原 mask 上按双分组（2 双，保留 4 张原始 mask）。点击实例可检查 2D 覆盖范围和 3D 落点。正式 500 帧页面未改动。详见 [单帧实验记录](InstanceRecognition/runtime/reports/FRAME0006_PAIRING_2026-09-17.md)。

### 1.6 Harness 结果页与逐窗口交互记录页

每个 Harness 实验的**两个页面在同一个实验目录内**，互为同级：

```text
InstanceRecognition/runtime/experiments/<实验标识>/
├── index.html      场景理解结果：窗口末帧 + 累计物体点云 + 3D bbox
├── records.html    逐窗口 Agent 交互记录：请求、响应、工具调用与理由（中英对照）
└── records/        记录页资源（含中文译文 sidecar）
```

两页顶部都有实验切换栏，可在各轮实验之间切换，并保持当前页面类型。打开方式：

```bash
/usr/bin/google-chrome "file:///home/seer/wzt/Reconstruction/InstanceRecognition/runtime/experiments/harness-v2-gpt6-sol-50frames-10windows-20261013/index.html"
/usr/bin/google-chrome "file:///home/seer/wzt/Reconstruction/InstanceRecognition/runtime/experiments/harness-v2-gpt6-sol-50frames-10windows-20261013/records.html"
```

记录页是**生成时快照**，新跑完一轮后需要重新生成：

```bash
cd /home/seer/wzt/Reconstruction
InstanceRecognition/.venv/bin/python InstanceRecognition/src/visualization/build_harness_records.py --run <实验标识>
```

中文理由译文是独立后处理（调用翻译模型，有 token 成本），不改写英文原始日志：

```bash
InstanceRecognition/.venv/bin/python InstanceRecognition/src/visualization/translate_harness_reasons.py --run <实验标识> --batch-size 30
```

翻译之后需再次运行 `build_harness_records.py` 才能让译文生效；`--require-translated` 可在仍有待译条目时让构建失败。命名规则见 [Harness 实验页面命名规则](docs/harness-experiment-naming.md)。

## 2. 在线交互推理页：必须启动后端

源码位置：`InstanceRecognition/src/web/frontend/index.html`。网页连接本机后端；GPT 通过公司网关调用，SAM3.1 在本机 GPU 推理。

后端优先使用 `InstanceRecognition/.venv/bin/python`，不存在时使用当前环境的 `python3`；可设置 `SCENE_PYTHON=/path/to/python` 指定解释器。YOLOE / SAM3.1 的独立解释器可通过 `YOLOE_PYTHON` / `SAM31_PYTHON` 指定。
GPT/SAM 默认缓存位于 `InstanceRecognition/runtime/results_gpt/`；设置 `INSTANCE_RECOGNITION_METHOD=yoloe_sam31` 可使用原基线。权重默认从仓库根目录 `weights/` 读取；数据集默认从仓库同级的 `data/scannet_v2/` 读取（当前为 `/home/seer/wzt/data/scannet_v2`），可通过 `SCANNET_ROOT` 覆盖。

先在终端执行：

```bash
taskset -c 0 bash /home/seer/wzt/Reconstruction/InstanceRecognition/start.sh
```

当前机器的两次实测均固定 CPU 0 启动；普通启动出现过依赖导入阶段崩溃，根因尚未定位。`taskset` 固定 CPU 调度，模型仍在 GPU 上运行。详见 [当前实测报告](InstanceRecognition/runtime/reports/SMOKE_TEST_2026-09-17.md)。

保持该终端运行，然后在浏览器地址栏打开：

```text
http://127.0.0.1:8090/
```

也可以在另一个终端执行：

```bash
/usr/bin/google-chrome "http://127.0.0.1:8090/"
```

此页通过后端读取 ScanNet 场景和帧，默认调用 GPT-5.6 Terra + SAM3.1。API 密钥从 `/home/seer/wzt/API/.env` 的 `VIBEG_API_KEY` 加载；客户端和英文 prompt 位于 `/home/seer/wzt/API/scannet_gpt_sam/`。页面连接会触发 SAM 模型服务预热，GPT 按需调用；如果只是查看已经处理过的结果，优先使用第 1 节的离线页面。

注意：

- 不要直接用 `file://` 打开 `InstanceRecognition/src/web/frontend/index.html`；它依赖 `/api/...` 接口和 `/assets/...` 静态资源路由。
- 普通 `python -m http.server` 不能代替这里的 Flask 后端。
- 按启动终端中的 `Ctrl+C` 可停止服务。
- 默认地址只监听本机。`127.0.0.1` 指浏览器所在的电脑，不是任意远程服务器。
- 如果 8090 已被占用，可使用其他端口，例如：

```bash
taskset -c 0 bash /home/seer/wzt/Reconstruction/InstanceRecognition/start.sh --port 8091
```

对应打开 `http://127.0.0.1:8091/`。

## 3. 数据介绍、方法备忘录与方案讨论

### 3.1 ScanNet 数据介绍

```bash
/usr/bin/google-chrome "file:///home/seer/wzt/Reconstruction/ScanNet/index.html"
```

介绍 ScanNet 文件组织及各文件包含的信息，并以 `scene0000_00` 展示 RGB、深度、标签、三维场景等内容。此页主要是数据导览，不是实时模型推理页。

2026-09-18 已从当前机器的原始数据恢复全部 15 张示例图片，并验证 12 个图集切换选项。页面中的 20 条扫描清单已标为旧电脑记录。换机后若图片再次缺失，按 [ScanNet 图片恢复说明](ScanNet/README.md) 重新生成 `assets/`。

物体级 3D 标注真值页面：

```bash
/usr/bin/google-chrome "file:///home/seer/wzt/Reconstruction/ScanNet/objects/index.html"
```

当前包含 `scene0000_00`（67 个实例）、`scene0000_02`（71 个实例）和 `scene0001_01`（23 个实例）。按类别／原始 objectId 筛选，点击列表或三维表面选择物体，Shift 可多选；“聚焦所选”查看细节，可切换网格表面／全部顶点、实例色／原始 RGB／segment 色，以及灰色场景参照。单选后可导出 PLY、NPZ。默认隐藏 wall／floor／ceiling 的彩色显示，取消勾选即可查看。未对数据集标注重新分组；生成及校验方法见 [ScanNet 说明](ScanNet/README.md)。

### 3.2 三个视觉任务的方法备忘录

```bash
/usr/bin/google-chrome "file:///home/seer/wzt/Reconstruction/memorandum/three-vision-task-methods.html"
```

查看物体识别、实例分割和视频点云重建的候选方法及能力分类。记录反映整理时的信息，不表示每种方法都已在本地复现。

### 3.3 方案思路演进

```bash
/usr/bin/google-chrome "file:///home/seer/wzt/Reconstruction/docs/scene-reconstruction-discussions.html"
```

逐轮记录从 RGB 视频重建可交互 SimReady 场景的原问题、方案建议、判断变化和待验证问题，按从新到旧排列。最新第六轮（2026-09-18）用三张图说明 Harness 的八部分分工、马桶案例中的取证与更新，以及错误归属的版本修正；详细接口和论文参照默认折叠。第五轮记录已选择的“多帧观测＋物体记忆＋API 主动更新”方向，包含执行流程、三层记忆与归属修正三幅离线图示，以及前 30 帧验证方案。第四轮保留全局／增量分工与上游尺度、相机位姿交给开源方法的范围约定。第三轮保留马桶、柜门与组合柜四张截图及整体／部件分割问题分析。讨论中的方案不等于已经实现的功能。

## 4. 重建论文时间线

### 4.1 物体重建论文时间线

```bash
/usr/bin/google-chrome "file:///home/seer/wzt/Reconstruction/news/object-reconstruction-timeline.html"
```

### 4.2 SimReady 重建论文时间线

```bash
/usr/bin/google-chrome "file:///home/seer/wzt/Reconstruction/news/simready-reconstruction-timeline.html"
```

### 4.3 视觉表征重建论文时间线

```bash
/usr/bin/google-chrome "file:///home/seer/wzt/Reconstruction/news/visual-representation-reconstruction-timeline.html"
```

## 5. 论文阅读记录

### 5.1 阅读记录总目录

```bash
/usr/bin/google-chrome "file:///home/seer/wzt/Reconstruction/news/papers/paper_reading_record/index.html"
```

打开总目录后可以点击进入下面五篇记录，也可以直接使用各自命令。阅读记录中的本地 PDF 链接依赖 `news/papers/` 下的对应文件。

### 5.2 Articraft

```bash
/usr/bin/google-chrome "file:///home/seer/wzt/Reconstruction/news/papers/paper_reading_record/articraft.html"
```

### 5.3 Lucida

```bash
/usr/bin/google-chrome "file:///home/seer/wzt/Reconstruction/news/papers/paper_reading_record/lucida.html"
```

### 5.4 ReplicateAnyScene

```bash
/usr/bin/google-chrome "file:///home/seer/wzt/Reconstruction/news/papers/paper_reading_record/replicate_any_scene.html"
```

### 5.5 SceneSmith

```bash
/usr/bin/google-chrome "file:///home/seer/wzt/Reconstruction/news/papers/paper_reading_record/scene_smith.html"
```

### 5.6 SimRecon

```bash
/usr/bin/google-chrome "file:///home/seer/wzt/Reconstruction/news/papers/paper_reading_record/simrecon.html"
```

## 6. ScanNet++ 手写签名采集页

```bash
/usr/bin/google-chrome "file:///home/seer/wzt/Reconstruction/docs/scannetpp-signature-pad.html"
```

用鼠标手写签名并保存图片。页面本身不负责向 ScanNet++ 网站提交申请。

## 7. 源码模板：不要当作完整结果页打开

| 模板文件 | 为什么不能独立查看结果 | 应打开的完整页面 |
| --- | --- | --- |
| `InstanceRecognition/src/web/frontend/mask_pointcloud/index.html` | 模板旁没有实验生成的 `manifest.js` 和帧资产。 | 2026-10-02 起不再有现成结果页；按 1.2 指定 `--output` 生成后再打开。 |
| `InstanceRecognition/src/web/frontend/method_comparison/index.html` | 模板旁没有实验生成的 `comparison.js` 及两侧查看器。 | `InstanceRecognition/runtime/comparisons/scene0000_00_0080_0140/index.html`，使用 1.4 的命令。 |
| `Video2PointCloud/src/visualization/index.html` | 模板旁没有导出的 `assets/*.cloud.js` 点云数据。 | `Video2PointCloud/output/visualization/index.html`，使用 1.3 的命令导出后打开。 |

这些 HTML 是维护页面的源文件。仅查看已有实验时，不需要重新运行数据生成脚本，也不要为打开模板而重跑模型。

## 8. 常见问题与维护约定

- **浏览器显示旧版本**：按 `Ctrl+Shift+R` 强制刷新，或关闭标签页后重新打开。
- **结果页空白／资源缺失**：先确认打开的是 `runtime/` 或 `output/` 中的完整结果页，而不是第 7 节的模板；不要单独移动 HTML。
- **`127.0.0.1:8090` 无法访问**：确认启动终端仍在运行且没有报错，浏览器端口与启动参数一致。
- **离线结果帧数与 3D 页不完全相同**：离线识别索引会包含额外测试帧；原始 mask 点云页对应指定的 uniform-500 批次，两者不必一一等量。
- **新增网页后**：在本文补充用途、完整路径、打开命令、是否需要服务；模板和可直接查看的输出页面应明确区分。

原工作目录中曾核对 19 个 HTML 入口、离线页面直接引用的本地脚本／样式／图片，以及服务启动脚本和路由；没有为编写本文启动模型或重新生成实验结果。
