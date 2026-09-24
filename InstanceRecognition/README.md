# ScanNet 实例识别应用

默认方案为 **GPT-5.6 Terra + SAM3.1 Multiplex**：

1. 从本地 ScanNet v2 `.sens` 文件按场景和帧编号读取原始 RGB JPEG；
2. GPT 独立识别可见类别，返回英文 SAM 提示、证据、不确定性与未识别区域；
3. SAM3.1 逐类别使用文本提示生成实例 mask，保留独立二值 mask 与重叠；
4. 网页展示原始 RGB、GPT 类别判断和 SAM3.1 mask；
5. 仅预热 SAM 常驻 worker，GPT 经网关按需调用。

API 客户端、英文 prompt 与认证加载方法位于 `/home/seer/wzt/API/scannet_gpt_sam/`，密钥通过 `/home/seer/wzt/API/.env` 中的 `VIBEG_API_KEY` 加载。默认输出 `runtime/results_gpt/`，避免把旧 YOLOE 缓存当成 GPT 结果。设置 `INSTANCE_RECOGNITION_METHOD=yoloe_sam31` 可运行原基线，原有 YOLOE 结果与 80–140 对比页保留。

`ThirdParty/` 保存官方源码和独立 YOLO/SAM 复现脚本；权重统一从仓库根目录的 `weights/` 读取。运行产生的帧缓存、日志和结果统一保存在 `runtime/`。

## 增量式场景理解 Harness（初版）

源码已按职责整理为 `src/data`、`src/inference`、`src/geometry`、`src/visualization`、`src/experiments` 和 `src/harness`，原模块名保留兼容入口。见 [目录说明](src/README.md)。

[Harness 执行流程与工具](src/harness/README.md)说明每窗取证、修改单校验、点云重算及断点续跑。独立实验输出在 `runtime/experiments/harness_first30_v1/`，不覆盖既有 mask 和基线。

```bash
cd /home/seer/wzt/Reconstruction/InstanceRecognition
taskset -c 0 .venv/bin/python -m src.harness.run
```

## 启动网页

```bash
cd InstanceRecognition  # 从仓库根目录执行
taskset -c 0 ./start.sh
```

当前机器实测需要固定在 CPU 0 启动：未固定时出现过依赖导入阶段的 SIGSEGV / SIGILL，根因尚未定位；固定 CPU 0 后，两次独立的 YOLOE-26 + SAM3.1 两帧推理均通过。GPU 推理仍使用 RTX 5070 Ti。详见 [2026-09-17 实测报告](runtime/docs/SMOKE_TEST_2026-09-17.md)。

启动脚本优先使用本模块 `.venv/bin/python`，不存在时使用当前 `python3`；也可通过 `SCENE_PYTHON=/path/to/python ./start.sh` 指定解释器。
SAM3.1 worker 默认使用本模块 `.venv`；YOLOE 优先保留已有的 `ThirdParty/yoloe26_prompt_free/.venv`，不存在时共用本模块 `.venv`。没有本模块环境时，worker 回退到当前解释器。
若模型依赖安装在其他环境，分别设置 `SAM31_PYTHON` 和 `YOLOE_PYTHON`。

浏览器打开：<http://127.0.0.1:8090>

默认只监听本机。如需从局域网其他电脑访问：

```bash
taskset -c 0 ./start.sh --host 0.0.0.0 --port 8090
```

## 当前机器的推理环境

本模块使用独立的 `.venv`，本次实测为 Python 3.12.3、PyTorch 2.10.0 + CUDA 12.8、torchvision 0.25.0。
PyTorch／torchvision 配对来自 [官方版本说明](https://pytorch.org/get-started/previous-versions/#v2100)。
YOLOE 使用原来的 Ultralytics 8.4.149；SAM3 从固定版本的本地子模块安装。
旧 YOLO 示例目录中的 PyTorch 2.4.1 requirements 属于原机器的复现记录，当前 RTX 5070 Ti 使用下面的环境。

在仓库根目录执行（`uv` 在当前机器的位置为 `/home/seer/wzt/tools/uv/uv`）：

```bash
/home/seer/wzt/tools/uv/uv venv --python /usr/bin/python3 InstanceRecognition/.venv
/home/seer/wzt/tools/uv/uv pip install --python InstanceRecognition/.venv/bin/python \
  'torch @ https://download.pytorch.org/whl/cu128/torch-2.10.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl' \
  'torchvision @ https://download.pytorch.org/whl/cu128/torchvision-0.25.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl' \
  -r InstanceRecognition/requirements-runtime.txt -e InstanceRecognition/ThirdParty/SAM3 \
  --index-url https://mirrors.aliyun.com/pypi/simple
```

上述 wheel URL 对应 Linux x86_64 / Python 3.12；其他平台应按官方版本说明选择。
PyTorch wheel 使用官方来源，其余依赖通过 PyPI 镜像获取，以避开当前机器上 NVIDIA 包服务器的连接超时。
本地已有 YOLOE／SAM3.1 权重，推理无需 Hugging Face 登录。

## 离线查看 YOLOE 基线结果

不启动 Flask 或模型服务，直接打开：

```bash
google-chrome "file:///home/seer/wzt/Reconstruction/InstanceRecognition/offline/index.html"
```

也可以在文件管理器中双击 `offline/index.html`。页面可以查看每个已处理 scene 的每个已处理帧，包括原始 RGB、YOLOE-26 检测框、SAM3.1 mask 叠加图和实例 ID 图。episode 汇总默认过滤掉 YOLOE 置信度低于 0.50 的检测，并按类别统计检出帧数、帧占比、实例数和平均/最高置信度；置信度与最少检出帧数都可以在页面中调整。

产生新推理结果后，执行下面的命令刷新离线索引：

```bash
cd /home/seer/wzt/Reconstruction/InstanceRecognition
taskset -c 0 .venv/bin/python -m web.build_offline_browser
```

索引生成器会扫描所有 `runtime/results/*/frame_*/summary.json`，并在遇到缺失图片或 mask 时失败，避免网页静默漏掉不完整结果。

2026-09-17 当前机器已完成 `scene0000_00` 的 uniform-500 重跑：500/500 帧成功、0 失败，6,790 个 mask，耗时约 18 分 48 秒。离线页保留两张旧测试帧，合计 502 帧、6,822 个 mask。详见 [本次批量结果与验证记录](runtime/docs/BATCH_RESULT_2026-09-17.md)。

## GPT 类别提示与 YOLOE 基线对比

已完成原抽样序号 80–140 的 61 帧对比：[离线页面](runtime/comparisons/scene0000_00_0080_0140/index.html)。左侧复用 YOLOE26 框提示结果，右侧为 GPT-5.6 Terra 英文类别文本提示 SAM3.1。两侧共用已有 AMB3R 几何，支持统一选帧、2D mask 与 3D 落点检查。

基线 792 个 mask，新方案 479 个 mask；125 条 GPT 类别提示未分出 mask。未做人工真值评测，不能将数量差异解释为准确率差异。详见 [实验记录与复现命令](runtime/docs/GPT_SAM_COMPARISON_2026-09-17.md)。API 调用脚本和英文 prompt 位于 `/home/seer/wzt/API/scannet_gpt_sam/`。

## 每帧 mask 反投影到 AMB3R 点云

默认离线入口：[`runtime/mask_pointclouds/scene0000_00/amb3r_gt_raw/index.html`](runtime/mask_pointclouds/scene0000_00/amb3r_gt_raw/index.html)。直接打开，无需启动模型或网页服务。

2026-09-18 已启用成对类别 prompt：原标签出现 shoe/shoes 的 83 帧重新运行 GPT → SAM，其他 417 帧保留。500 帧当前展示 4,481 张 mask，共保留 4,895 张原始候选，414 张超出每帧 16 张展示预算。成对提示直接交给 SAM，不增加配对或合并后处理。详见 [本次更新报告](runtime/docs/SHOE_PAIR_REVISION_2026-09-18.md)；[2026-09-17 全场景报告](runtime/docs/GPT_SAM_FULL_SCENE_2026-09-17.md) 为历史记录。

- 正式英文 prompt：`/home/seer/wzt/API/scannet_gpt_sam/prompt_en.txt`，冻结版 `prompts/paired_units_v1.txt`；旧版 `prompts/singular_v1.txt`。
- 83 帧新标签及响应：`/home/seer/wzt/API/scannet_gpt_sam/runs/scene0000_00_shoe_pairs_v1/`。
- 合并后的 500 帧标签索引：`/home/seer/wzt/API/scannet_gpt_sam/runs/scene0000_00_uniform500_terra_pairs_v1_merged/`；`frame_provenance.json` 逐帧记录新旧 prompt 来源。
- 新 SAM 结果：`runtime/experiments/gpt_sam31_shoe_pairs_v1/results/`；完整 500 帧引用索引 `merged_results/`。
- 当前点云数据：`runtime/mask_pointclouds/scene0000_00/gpt_terra_pairs_v1_gt_raw/`；主入口引用此处的 NPZ、PLY 和 JS。
- 默认在线缓存：`runtime/results_gpt/scene0000_00/`；仅替换受影响的 83 帧，旧缓存另存于本次实验目录。
- 旧 GPT 页面：[单数版归档](runtime/mask_pointclouds/scene0000_00/gpt_terra_singular_v1_archive/index.html)。原 500 帧 API/SAM 数据仍保留。
- 原 YOLOE 页面：[归档入口](runtime/mask_pointclouds/scene0000_00/yoloe26_gt_raw/index.html)。原 `amb3r_gt_raw/assets/`、`frames/` 和 canonical 文件继续保存旧基线数据；读取当前结果应跟随 manifest 路径。

反投影复用原 AMB3R 点图与 GT Sim(3)，没有再次运行重建。背景仍为 492,555 个点；逐帧 mask 覆盖点合计 43,066,886 个，包含不同帧重复观测，不是融合后的场景点数。

- 原始 mask `1296 × 968` → `640 × 480` → 中心裁剪 `[3, 0, 637, 480]` → `518 × 392`，两次缩放均为最近邻。
- 独立 mask 的重叠通过 `mask_bits` 保留；第 i 位对应帧 manifest 的 `instances[i]`。NPZ 保存真实 RGB，PLY 的重叠点标紫。
- 不新增点置信度过滤、形态学处理、去噪、类别归并或跨帧融合。实例 ID 仅在帧内有效。
- 几何使用已有 GT 修正结果，不使用 GT 深度，也不替换成逐帧 GT 位姿。

生成和发布脚本分别为 `src/build_method_comparison.py --primary-only` 与 `src/publish_gpt_viewer.py`。完整命令记录在 `/home/seer/wzt/API/scannet_gpt_sam/README.md`。发布脚本要求 500 帧完整且校验通过，并保留 YOLOE 归档。

原 AMB3R 重建恢复入口仍为 `bash Video2PointCloud/src/amb3r/restore_mask_viewer.sh`（从仓库根目录执行）；已有 GPT 导出时会优先恢复 GPT 主页面。原始重建记录见 [恢复报告](runtime/docs/MASK_POINTCLOUD_RESTORE_2026-09-17.md)。

从新版 manifest 指向的某帧 NPZ 提取指定帧内实例：

```python
import numpy as np

with np.load("frame_000000.npz") as data:
    bit_index = np.flatnonzero(data["instance_ids"] == 1).item()
    belongs = (data["mask_bits"] & np.uint32(1 << bit_index)) != 0
    instance_points = data["points_world"][belongs]
```

## 默认路径

| 内容 | 路径 |
| --- | --- |
| ScanNet 场景 | `../data/scannet_v2`（相对于仓库根目录，当前为 `/home/seer/wzt/data/scannet_v2`） |
| YOLOE-26 权重 | `../weights/yoloe26_prompt_free/weights/yoloe-26s-seg-pf.pt` |
| SAM3.1 权重 | `../weights/sam3.1_multiplex.pt` |
| 动态帧与结果 | `runtime/` |

路径可通过 `SCANNET_ROOT`、`YOLOE_CHECKPOINT`、`SAM31_CHECKPOINT` 和 `INSTANCE_RECOGNITION_RUNTIME` 环境变量覆盖。

权重默认从仓库根目录的 `weights/` 读取，路径由源码位置推导，不依赖启动时的工作目录。
三个 YOLO 示例脚本也使用根目录下的 `weights/{yoloe26_prompt_free,yolo26_detect,yolo26_seg}/weights/`；
显式传入 `--model /path/to/checkpoint.pt` 可使用其他位置的权重。
SAM3 对比脚本使用 `../weights/sam3.pt`（相对于本模块根目录）。
WeDetect 为独立上游工具，运行时显式传入根目录下 `weights/WeDetect/checkpoints/wedetect_base_uni.pth` 的绝对路径；
ONNX 推理通过 `--onnx` 指定 `weights/WeDetect/wedetect_anything/onnx_models/wedetect_anything_base.onnx`，
并通过 `--text-feat` 指定上游目录中的 `wedetect_anything/prompt_free.npy`。
当前权重目录未包含配套 meta JSON；上游 CLI 支持省略 `--meta` 并使用内置 base 常量，有导出元数据时应通过 `--meta` 指定。

## 文件职责

| 文件 | 作用 |
| --- | --- |
| `src/scannet_sens.py` | 校验场景 ID，读取 `.sens`，并单帧或批量提取 RGB JPEG |
| `src/sampling.py` | 在完整时间范围内均匀选择目标数量的帧 |
| `src/gpt_labels.py` | 复用外部 API 客户端，独立识别原始 RGB 中的类别 |
| `src/gpt_sam31.py` | 类别文本提示 SAM3.1；保存所有候选与展示预算 |
| `src/build_method_comparison.py` | 复用既有几何生成 GPT 点云页或两种方案对比页 |
| `src/publish_gpt_viewer.py` | 校验完整 500 帧并替换主入口，归档 YOLOE 页面 |
| `src/import_gpt_batch.py` | 把批量结果接入默认在线缓存，无需再次推理 |
| `src/yoloe_detect.py` | 第一阶段无文本检测，只保存标签与检测框，不转发原生 mask |
| `src/sam31_segment.py` | 第二阶段框提示分割，困难框使用中心正点回退 |
| `src/model_worker.py` | 在各自 Python 环境内常驻模型，并通过 JSON-lines 接收推理请求 |
| `src/model_service.py` | 管理所选方案的 worker、网页客户端心跳和自动卸载 |
| `src/pipeline.py` | 调用常驻模型服务、校验契约并缓存结果 |
| `src/batch_infer.py` | 对一个场景执行可恢复的降采样批量推理 |
| `web/server.py` | Flask 网页与 JSON/图像 API |
| `web/build_offline_browser.py` | 扫描所有完整结果并生成离线数据索引 |
| `web/index.html`、`web/styles.css`、`web/app.js` | 页面、样式和交互逻辑 |
| `offline/` | 可通过 `file://` 直接打开的静态结果浏览器 |

## 场景批量推理

以下命令对场景完整时间轴做均匀采样，将总帧数控制在 500 帧，逐帧调用 GPT 标签与常驻 SAM3.1 文本分割：

```bash
cd /home/seer/wzt/Reconstruction/InstanceRecognition
taskset -c 0 env HF_HUB_OFFLINE=1 .venv/bin/python -u -m src.batch_infer \
  --scene-id scene0000_00 \
  --target-frames 500
```

采样包含首、末帧。默认每帧结果保存在 `runtime/results_gpt/<scene_id>/frame_<index>/`，批次清单保存在 `runtime/batches_gpt/<scene_id>/target_500.json`。指定 `--method yoloe_sam31` 时使用原 `results/` 和 `batches/`。命令中断后可直接重跑，已经完成且通过契约校验的单帧结果会从缓存恢复；只有显式添加 `--force` 才会覆盖计算。

默认 SAM 文本提示阈值为 0.5，每帧按 SAM 分数选最多 16 个 mask 用于页面展示；全部原始候选仍保留在 `raw_masks/`。包括 GPT 不确定类别和未分出 mask 的提示，不做跨类别 NMS、跨帧传播或类别归并。旧基线仍使用最多 16 个 YOLOE 检测框。

页面打开时会异步加载模型，右侧状态显示“模型预热中”或“模型已驻留”。多个标签页共享同一组 worker；只有最后一个页面关闭后才开始 5 秒释放倒计时。页面使用 5 秒心跳，若浏览器崩溃或关闭通知丢失，服务端会在 15 秒心跳超时后触发相同的卸载流程。卸载时终止所选方案的 worker 进程，从而释放权重、CUDA 上下文和相关显存；Flask 网页服务本身继续运行。

可通过接口检查或手动释放：

```bash
curl http://127.0.0.1:8090/api/models/status
curl -X POST http://127.0.0.1:8090/api/models/unload
```

## 验证

```bash
/home/wzt/.conda/envs/scene/bin/python -m pytest -q tests
```

原机器历史实测记录（下列时间和计数不代表 2026-09-17 在当前机器上的重跑）：

- 11 项读取器、采样、API 与模型生命周期测试通过；
- `scene0000_00` 从 5,578 帧均匀采样到 500 帧（包含首帧 0 和末帧 5,577），500/500 帧完成、0 失败，共得到 6,795 个 YOLOE 检测及一一对应的 SAM3.1 mask；批次耗时 802.319 秒，结果占用约 592 MiB；
- 改造前的冷启动基线 `scene0001_00 / frame 73`：YOLOE 检出 10 个实例、7 类标签，SAM3.1 生成 10 个 mask，总耗时 15.251 秒；
- 常驻模型实测 `scene0001_00 / frame 74, 75` 两次未缓存端到端推理分别耗时 1.667 秒和 1.020 秒；
- 最后一个客户端关闭后，YOLOE（666 MiB）与 SAM3.1（5500 MiB）worker 均在 5 秒防抖后退出，相关 GPU 进程消失；
- `scene0002_01` 的末帧 `7252` 可直接从 `.sens` 加载，验证了长序列尾部索引；
- 页面在 1600 × 1400 下完成视觉验收，输入、两阶段输出、缓存状态和结构化结果均可读。
