# 本仓库网页打开方法

更新日期：2026-09-16。仓库位置：`/home/wzt/wzt/CVPR_2027`。

本文收录本项目制作或整理的全部 19 个 HTML 文件：16 个可直接打开的离线页面、1 个需要后端服务的推理页面、2 个源码模板。不包含第三方依赖自带的示例或文档。

## 先选对页面

| 想查看什么 | 页面文件 | 打开方式 |
| --- | --- | --- |
| 每帧 RGB、YOLOE 检测、SAM3.1 mask，以及 episode 类别统计 | [InstanceRecognition/offline/index.html](InstanceRecognition/offline/index.html) | 离线直接打开，见 1.1。 |
| 每帧 mask 反投影后的 3D 点云 | [InstanceRecognition/runtime/mask_pointclouds/scene0000_00/amb3r_gt_raw/index.html](InstanceRecognition/runtime/mask_pointclouds/scene0000_00/amb3r_gt_raw/index.html) | 离线直接打开，见 1.2。 |
| GT 修正 AMB3R、原始 AMB3R、DA3 的点云对比 | [Video2PointCloud/output/visualization/index.html](Video2PointCloud/output/visualization/index.html) | 离线直接打开，见 1.3。 |
| 选择 ScanNet 场景／帧，并点击按钮运行模型 | [InstanceRecognition/web/index.html](InstanceRecognition/web/index.html) | 必须启动后端，再访问 HTTP 地址，见第 2 节。 |
| ScanNet 数据包含的信息及示例 | [ScanNet/index.html](ScanNet/index.html) | 离线直接打开，见 3.1。 |
| 三个视觉任务的备选方法 | [memorandum/three-vision-task-methods.html](memorandum/three-vision-task-methods.html) | 离线直接打开，见 3.2。 |
| 视频到 SimReady 的方案讨论与思路演进 | [docs/scene-reconstruction-discussions.html](docs/scene-reconstruction-discussions.html) | 离线直接打开，见 3.3。 |
| 重建论文时间线 | `news/` 下的三个时间线页面 | 离线直接打开，见第 4 节。 |
| 论文阅读记录 | [news/papers/paper_reading_record/index.html](news/papers/paper_reading_record/index.html) 及五篇详情 | 离线直接打开，见第 5 节。 |
| 手写签名采集 | [scannetpp-signature-pad.html](scannetpp-signature-pad.html) | 离线直接打开，见第 6 节。 |

对于下面的离线页面，任选一种打开方式：

- 在终端执行对应的 `google-chrome` 命令；所有命令均使用完整路径，不需要先切换目录。
- 将命令引号内的 `file:///...` 地址复制到浏览器地址栏并按回车。
- 在文件管理器中找到对应 HTML，用浏览器打开。

离线页面不需要启动网站服务或推理模型。请保留 HTML 周围的脚本、样式、数据和图片目录，不要只把单个 HTML 搬走。点击论文、GitHub 等外部链接时才需要访问对应网站。

## 1. 已有实验结果：不需要重新推理

### 1.1 YOLOE-26 与 SAM3.1 离线结果浏览器

```bash
/usr/bin/google-chrome "file:///home/wzt/wzt/CVPR_2027/InstanceRecognition/offline/index.html"
```

可以切换已处理的场景和帧，查看原始 RGB、YOLOE 检测、SAM3.1 mask、实例 ID 图，以及按置信度和检出帧数筛选的 episode 类别统计。

它显示离线索引中已经保存的结果，不会自动扫描尚未处理的帧，也不会启动新推理。若新增了推理结果但页面未收录，可以单独更新索引：

```bash
/home/wzt/.conda/envs/scene/bin/python /home/wzt/wzt/CVPR_2027/InstanceRecognition/web/build_offline_browser.py
```

### 1.2 每帧 mask 反投影到 3D 点云的检查页

```bash
/usr/bin/google-chrome "file:///home/wzt/wzt/CVPR_2027/InstanceRecognition/runtime/mask_pointclouds/scene0000_00/amb3r_gt_raw/index.html"
```

这是之前处理的第一个场景 `scene0000_00` 的 uniform-500 反投影结果。左侧查看 2D 图像和当前帧 mask，右侧查看对应的 3D 落点及灰色场景背景。

- 左键拖动：旋转。
- 右键拖动：平移。
- 滚轮：缩放。
- “重置视角”：恢复旋转、缩放和平移。

此页展示逐帧原始观测，不代表已经完成跨帧实例合并；实例 ID 只在帧内有效。几何使用 GT 修正结果，不应误解为无 GT 的纯 RGB 最终结果。

### 1.3 三类点云输出对比页

```bash
/usr/bin/google-chrome "file:///home/wzt/wzt/CVPR_2027/Video2PointCloud/output/visualization/index.html"
```

对比 GT 修正的 AMB3R-VO、原始 AMB3R-VO 和 DA3 点云，支持同步视角、预设视角、RGB／高度配色及单窗口全屏。左键旋转，右键或 Shift 拖动平移，滚轮缩放。

GT 修正版本属于借助真值的诊断结果，不能与无 GT 方法混同。打开已有页面不需要重新导出点云。

## 2. 在线交互推理页：必须启动后端

源码位置：`InstanceRecognition/web/index.html`。这里的“在线”指连接本机后端，不是云端推理。

先在终端执行：

```bash
bash /home/wzt/wzt/CVPR_2027/InstanceRecognition/start.sh
```

保持该终端运行，然后在浏览器地址栏打开：

```text
http://127.0.0.1:8090/
```

也可以在另一个终端执行：

```bash
/usr/bin/google-chrome "http://127.0.0.1:8090/"
```

此页通过后端读取 ScanNet 场景和帧，并调用 YOLOE-26 + SAM3.1。页面连接会触发模型服务预热；如果只是查看已经处理过的结果，优先使用第 1 节的离线页面。

注意：

- 不要直接用 `file://` 打开 `InstanceRecognition/web/index.html`；它依赖 `/api/...` 接口和 `/assets/...` 静态资源路由。
- 普通 `python -m http.server` 不能代替这里的 Flask 后端。
- 按启动终端中的 `Ctrl+C` 可停止服务。
- 默认地址只监听本机。`127.0.0.1` 指浏览器所在的电脑，不是任意远程服务器。
- 如果 8090 已被占用，可使用其他端口，例如：

```bash
bash /home/wzt/wzt/CVPR_2027/InstanceRecognition/start.sh --port 8091
```

对应打开 `http://127.0.0.1:8091/`。

## 3. 数据介绍、方法备忘录与方案讨论

### 3.1 ScanNet 数据介绍

```bash
/usr/bin/google-chrome "file:///home/wzt/wzt/CVPR_2027/ScanNet/index.html"
```

介绍 ScanNet 文件组织及各文件包含的信息，并以 `scene0000_00` 展示 RGB、深度、标签、三维场景等内容。此页主要是数据导览，不是实时模型推理页。

### 3.2 三个视觉任务的方法备忘录

```bash
/usr/bin/google-chrome "file:///home/wzt/wzt/CVPR_2027/memorandum/three-vision-task-methods.html"
```

查看物体识别、实例分割和视频点云重建的候选方法及能力分类。记录反映整理时的信息，不表示每种方法都已在本地复现。

### 3.3 方案思路演进

```bash
/usr/bin/google-chrome "file:///home/wzt/wzt/CVPR_2027/docs/scene-reconstruction-discussions.html"
```

逐轮记录从 RGB 视频重建可交互 SimReady 场景的原问题、方案建议、判断变化和待验证问题。第一轮讨论跨帧物体身份、位置和类别的一致性；讨论中的方案不等于已经实现的功能。

## 4. 重建论文时间线

### 4.1 物体重建论文时间线

```bash
/usr/bin/google-chrome "file:///home/wzt/wzt/CVPR_2027/news/object-reconstruction-timeline.html"
```

### 4.2 SimReady 重建论文时间线

```bash
/usr/bin/google-chrome "file:///home/wzt/wzt/CVPR_2027/news/simready-reconstruction-timeline.html"
```

### 4.3 视觉表征重建论文时间线

```bash
/usr/bin/google-chrome "file:///home/wzt/wzt/CVPR_2027/news/visual-representation-reconstruction-timeline.html"
```

## 5. 论文阅读记录

### 5.1 阅读记录总目录

```bash
/usr/bin/google-chrome "file:///home/wzt/wzt/CVPR_2027/news/papers/paper_reading_record/index.html"
```

打开总目录后可以点击进入下面五篇记录，也可以直接使用各自命令。阅读记录中的本地 PDF 链接依赖 `news/papers/` 下的对应文件。

### 5.2 Articraft

```bash
/usr/bin/google-chrome "file:///home/wzt/wzt/CVPR_2027/news/papers/paper_reading_record/articraft.html"
```

### 5.3 Lucida

```bash
/usr/bin/google-chrome "file:///home/wzt/wzt/CVPR_2027/news/papers/paper_reading_record/lucida.html"
```

### 5.4 ReplicateAnyScene

```bash
/usr/bin/google-chrome "file:///home/wzt/wzt/CVPR_2027/news/papers/paper_reading_record/replicate_any_scene.html"
```

### 5.5 SceneSmith

```bash
/usr/bin/google-chrome "file:///home/wzt/wzt/CVPR_2027/news/papers/paper_reading_record/scene_smith.html"
```

### 5.6 SimRecon

```bash
/usr/bin/google-chrome "file:///home/wzt/wzt/CVPR_2027/news/papers/paper_reading_record/simrecon.html"
```

## 6. ScanNet++ 手写签名采集页

```bash
/usr/bin/google-chrome "file:///home/wzt/wzt/CVPR_2027/scannetpp-signature-pad.html"
```

用鼠标手写签名并保存图片。页面本身不负责向 ScanNet++ 网站提交申请。

## 7. 两个源码模板：不要当作完整结果页打开

| 模板文件 | 为什么不能独立查看结果 | 应打开的完整页面 |
| --- | --- | --- |
| `InstanceRecognition/web/mask_pointcloud/index.html` | 模板旁没有实验生成的 `manifest.js` 和帧资产。 | `InstanceRecognition/runtime/mask_pointclouds/scene0000_00/amb3r_gt_raw/index.html`，使用 1.2 的命令。 |
| `Video2PointCloud/src/visualization/index.html` | 模板旁没有导出的 `assets/*.cloud.js` 点云数据。 | `Video2PointCloud/output/visualization/index.html`，使用 1.3 的命令。 |

这两份 HTML 是维护页面的源文件。仅查看已有实验时，不需要重新运行数据生成脚本，也不要为打开模板而重跑模型。

## 8. 常见问题与维护约定

- **浏览器显示旧版本**：按 `Ctrl+Shift+R` 强制刷新，或关闭标签页后重新打开。
- **结果页空白／资源缺失**：先确认打开的是 `runtime/` 或 `output/` 中的完整结果页，而不是第 7 节的模板；不要单独移动 HTML。
- **`127.0.0.1:8090` 无法访问**：确认启动终端仍在运行且没有报错，浏览器端口与启动参数一致。
- **离线结果帧数与 3D 页不完全相同**：离线识别索引会包含额外测试帧；原始 mask 点云页对应指定的 uniform-500 批次，两者不必一一等量。
- **新增网页后**：在本文补充用途、完整路径、打开命令、是否需要服务；模板和可直接查看的输出页面应明确区分。

本次已核对 19 个 HTML 入口、离线页面直接引用的本地脚本／样式／图片，以及服务启动脚本和路由；没有为编写本文启动模型或重新生成实验结果。
