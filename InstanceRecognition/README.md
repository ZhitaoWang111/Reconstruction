# ScanNet 实例识别应用

主要方案为 **YOLOE-26 Prompt-Free + SAM3.1 Multiplex**：

1. 从本地 ScanNet v2 `.sens` 文件按场景和帧编号读取 RGB JPEG；
2. YOLOE-26 在无文本提示条件下输出类别、置信度和检测框；
3. SAM3.1 只接收 YOLOE 检测框并输出逐实例 mask；
4. 网页分别展示原始 RGB、YOLOE 检测框和 SAM3.1 mask；
5. 页面打开后异步预热两个常驻模型 worker，连续推理不再重复加载权重。

`ThirdParty/` 只提供官方源码、环境和已有权重，本应用代码不会修改其中内容。运行产生的帧缓存、日志和结果统一保存在 `runtime/`。

## 启动网页

```bash
cd /home/wzt/wzt/CVPR_2027/InstanceRecognition
./start.sh
```

浏览器打开：<http://127.0.0.1:8090>

默认只监听本机。如需从局域网其他电脑访问：

```bash
./start.sh --host 0.0.0.0 --port 8090
```

## 离线查看已有结果

不启动 Flask 或模型服务，直接打开：

```bash
google-chrome "file:///home/wzt/wzt/CVPR_2027/InstanceRecognition/offline/index.html"
```

也可以在文件管理器中双击 `offline/index.html`。页面可以查看每个已处理 scene 的每个已处理帧，包括原始 RGB、YOLOE-26 检测框、SAM3.1 mask 叠加图和实例 ID 图。episode 汇总默认过滤掉 YOLOE 置信度低于 0.50 的检测，并按类别统计检出帧数、帧占比、实例数和平均/最高置信度；置信度与最少检出帧数都可以在页面中调整。

产生新推理结果后，执行下面的命令刷新离线索引：

```bash
cd /home/wzt/wzt/CVPR_2027/InstanceRecognition
/home/wzt/.conda/envs/scene/bin/python -m web.build_offline_browser
```

索引生成器会扫描所有 `runtime/results/*/frame_*/summary.json`，并在遇到缺失图片或 mask 时失败，避免网页静默漏掉不完整结果。

## 原始 mask 反投影到 AMB3R 点云

本地离线检查页：[`runtime/mask_pointclouds/scene0000_00/amb3r_gt_raw/index.html`](runtime/mask_pointclouds/scene0000_00/amb3r_gt_raw/index.html)。直接双击打开，无需启动模型或网页服务；可以选择 uniform-500 中的每帧及每张 SAM3.1 mask，查看对齐的 2D 图像与 3D 落点。

反投影代码为 [`src/backproject_masks.py`](src/backproject_masks.py)，检查页源码在 [`web/mask_pointcloud/`](web/mask_pointcloud/)。重跑命令如下，输出目录必须为空或不存在，避免覆盖已有实验：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /home/wzt/.conda/envs/scene/bin/python \
  /home/wzt/wzt/CVPR_2027/InstanceRecognition/src/backproject_masks.py \
  --output /home/wzt/wzt/CVPR_2027/InstanceRecognition/runtime/mask_pointclouds/scene0000_00/amb3r_gt_raw_rerun
```

- 几何来自 AMB3R 每帧 `392 × 518` 的点图，再应用 `pointcloud_gt_corrected_evaluation.json` 中的统一 GT Sim(3)；不使用 GT 深度，不换成逐帧 GT 位姿，不额外应用 ScanNet axisAlignment。
- mask 按 RGB 预处理链对齐：原始 `1296 × 968` → `640 × 480` → 中心裁剪 `[3, 0, 637, 480]` → `518 × 392`；mask 的两次缩放都使用最近邻，不做形态学操作。
- 所有独立 mask 保留，包括重叠、原始类别和低置信度结果。此处“无新增过滤”不代表检测器从未过滤：已有检测采用 `conf=0.25`、最多 16 个框，本步骤不改变它们。
- 逐帧 `frames/frame_XXXXXX.npz` / `.ply` 保留该帧全部 mask 覆盖的有限三维点，不做点置信度过滤或抽样。相同点属于多个 mask 时，使用 `mask_bits` 同时保存所有成员关系；第 `i` 位对应该帧 JSON 的 `instances[i]`。PLY 中重叠点标为紫色，NPZ 另存真实 RGB。
- `pointcloud_gt_corrected_mask_observations.ply` 在**原有 491,666 个点**上着色，点坐标和顺序不变。它继承原 PLY 已有的置信度筛选和抽样，仅通过每个点的原始生成帧/像素附加 mask，不向其他帧传播标签，也不是跨帧融合结果。
- `canonical_mask_memberships.npz` 与原 PLY 逐点对应，保存来源帧号、点图像素索引和 mask 位图；`projection_report.json` 记录坐标变换、预处理、帧对应及逐点重放验证。

例如，从某帧 NPZ 提取指定帧内实例的点：

```python
import numpy as np

with np.load("frame_000000.npz") as data:
    bit_index = np.flatnonzero(data["instance_ids"] == 1).item()
    belongs = (data["mask_bits"] & np.uint32(1 << bit_index)) != 0
    instance_points = data["points_world"][belongs]
```

这些 ID 只表示逐帧观测，不代表已经确定的全局物体身份。GT 修正几何也是诊断输入，不应当作纯 RGB 未知视频上的无 GT 结果。

## 默认路径

| 内容 | 路径 |
| --- | --- |
| ScanNet 场景 | `/home/wzt/HDD/datasets/scannet_v2/scans` |
| YOLOE-26 权重 | `ThirdParty/yoloe26_prompt_free/weights/yoloe-26s-seg-pf.pt` |
| SAM3.1 权重 | `../weights/sam3.1_multiplex.pt` |
| 动态帧与结果 | `runtime/` |

路径可通过 `SCANNET_ROOT`、`YOLOE_CHECKPOINT`、`SAM31_CHECKPOINT` 和 `INSTANCE_RECOGNITION_RUNTIME` 环境变量覆盖。

## 文件职责

| 文件 | 作用 |
| --- | --- |
| `src/scannet_sens.py` | 校验场景 ID，读取 `.sens`，并单帧或批量提取 RGB JPEG |
| `src/sampling.py` | 在完整时间范围内均匀选择目标数量的帧 |
| `src/yoloe_detect.py` | 第一阶段无文本检测，只保存标签与检测框，不转发原生 mask |
| `src/sam31_segment.py` | 第二阶段框提示分割，困难框使用中心正点回退 |
| `src/model_worker.py` | 在各自 Python 环境内常驻模型，并通过 JSON-lines 接收推理请求 |
| `src/model_service.py` | 管理双 worker、网页客户端心跳和自动卸载 |
| `src/pipeline.py` | 调用常驻模型服务、校验契约并缓存结果 |
| `src/batch_infer.py` | 对一个场景执行可恢复的降采样批量推理 |
| `web/server.py` | Flask 网页与 JSON/图像 API |
| `web/build_offline_browser.py` | 扫描所有完整结果并生成离线数据索引 |
| `web/index.html`、`web/styles.css`、`web/app.js` | 页面、样式和交互逻辑 |
| `offline/` | 可通过 `file://` 直接打开的静态结果浏览器 |

## 场景批量推理

以下命令对场景完整时间轴做均匀采样，将总帧数控制在 500 帧，并在同一组常驻 worker 中顺序运行 YOLOE-26 与 SAM3.1：

```bash
cd /home/wzt/wzt/CVPR_2027/InstanceRecognition
/home/wzt/.conda/envs/scene/bin/python -m src.batch_infer \
  --scene-id scene0000_00 \
  --target-frames 500
```

采样包含首、末帧。每帧结果保存在 `runtime/results/<scene_id>/frame_<index>/`，批次清单保存在 `runtime/batches/<scene_id>/target_500.json`。命令中断后可直接重跑，已经完成且通过契约校验的单帧结果会从缓存恢复；只有显式添加 `--force` 才会覆盖计算。

默认最多保留置信度最高的 16 个 YOLOE 检测，以匹配当前 SAM3.1 Multiplex 配置。SAM3.1 能改善实例轮廓，但不会纠正 YOLOE 的类别错误。

页面打开时会异步加载模型，右侧状态显示“模型预热中”或“模型已驻留”。多个标签页共享同一组 worker；只有最后一个页面关闭后才开始 5 秒释放倒计时。页面使用 5 秒心跳，若浏览器崩溃或关闭通知丢失，服务端会在 15 秒心跳超时后触发相同的卸载流程。卸载时终止两个 worker 进程，从而释放权重、CUDA 上下文和相关显存；Flask 网页服务本身继续运行。

可通过接口检查或手动释放：

```bash
curl http://127.0.0.1:8090/api/models/status
curl -X POST http://127.0.0.1:8090/api/models/unload
```

## 验证

```bash
/home/wzt/.conda/envs/scene/bin/python -m pytest -q tests
```

当前实测记录：

- 11 项读取器、采样、API 与模型生命周期测试通过；
- `scene0000_00` 从 5,578 帧均匀采样到 500 帧（包含首帧 0 和末帧 5,577），500/500 帧完成、0 失败，共得到 6,795 个 YOLOE 检测及一一对应的 SAM3.1 mask；批次耗时 802.319 秒，结果占用约 592 MiB；
- 改造前的冷启动基线 `scene0001_00 / frame 73`：YOLOE 检出 10 个实例、7 类标签，SAM3.1 生成 10 个 mask，总耗时 15.251 秒；
- 常驻模型实测 `scene0001_00 / frame 74, 75` 两次未缓存端到端推理分别耗时 1.667 秒和 1.020 秒；
- 最后一个客户端关闭后，YOLOE（666 MiB）与 SAM3.1（5500 MiB）worker 均在 5 秒防抖后退出，相关 GPU 进程消失；
- `scene0002_01` 的末帧 `7252` 可直接从 `.sens` 加载，验证了长序列尾部索引；
- 页面在 1600 × 1400 下完成视觉验收，输入、两阶段输出、缓存状态和结构化结果均可读。
