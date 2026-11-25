# Code Review — 代码仓逐模块梳理

> 本文件用于逐模块、逐文件地梳理本仓库的可执行代码，目标是让维护者能够完全掌握当前 checkout。
> 约定：**每个文件一行，带磁盘字节数与职责说明**；树中列出的是**磁盘上真实存在的文件**（不含 `__pycache__`、`.venv` 等缓存与虚拟环境）。

## 本轮范围

- **本轮只覆盖 `InstanceRecognition/` 下的核心处理代码**，即 `InstanceRecognition/src/`（真实实现）与 `InstanceRecognition/` 顶层的入口文件。
- **本轮不展开**：`Submodule/`（外部方法仓库）、`InstanceRecognition/runtime/`（实验运行产物）、`InstanceRecognition/.venv/`（虚拟环境）。
- `tests/` 本轮只列文件清单与一句话职责，不逐测试用例展开。
- 后续轮次再扩展到其他顶层模块（`Video2PointCloud/`、`ScanNet/`、`ObjectGeneration/` 等）。

## 事实基线

| 项目 | 数值 | 来源 |
|---|---|---|
| `InstanceRecognition/src/` 文件总数 | **112** | `find src -type f ! -path '*__pycache__*'` |
| 其中 Python 文件 | **80** | 同上，`-name '*.py'` |
| 其中非 Python 资源 | **32** | 相减（多为 `src/web/frontend/` 下的网页源码） |
| 测试基线 | `204 passed, 4 subtests passed` | `cd InstanceRecognition && taskset -c 0 .venv/bin/python -m pytest tests/ -q` |

`node tests/test_episode_stats.js` → `EPISODE_STATS_OK`。此前的 `5 failed` 源于两个测试文件硬编码的旧机器路径 `Path("/home/wzt/HDD/datasets/scannet_v2/scans")`，该布局也不含 `scans/` 这一层；2026-10-13 已改为从 `config.SCANNET_ROOT` 取根并在数据缺失时 `skipif`，5 项全部转绿。`test_existing_run_is_reported_as_cached` 另需显式固定 `method="yoloe_sam31"`——结果根目录按method 在 `results/` 与 `results_gpt/` 之间切换，依赖默认值会让该测试断言到另一分支。

> **2026-10-13 复核**：上表数值已按当前 checkout 重算。此前的 95 / 77 / 18 与 148 passed 是 2026-09 首轮的数字，此后新增了 `src/geometry/cluster_prune.py`、`src/geometry/leakage_eval.py`、`src/web/frontend/` 全部网页源码与相应的测试文件。

---

## 0. 读图须知：shim 与子包实现的双层结构

这是理解 `src/` 目录的**第一前提**，不理解它就会把一半文件误判为"重复实现"。

`src/` 曾有两条平行路径：

1. **历史扁平路径**：`src/sampling.py`、`src/pipeline.py` 等，直接写在 `src/` 根下。
2. **当前子包路径**：按职责拆进 `src/data/`、`src/inference/`、`src/geometry/`、`src/visualization/`、`src/experiments/`。

现在的状态是：**真实实现全部在子包里；`src/` 根下的同名 `.py` 只是兼容 shim（转发壳）**，体积极小（298–379 字节）。shim 的固定写法为：

```python
"""Compatibility entry point; implementation lives in <target>."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("<target>", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("<target>")
```

两个分支的意图：

- **作为脚本直接运行**（`python src/sampling.py`）→ `runpy` 把子包模块当 `__main__` 跑，等价于 `python -m src.data.sampling`。
- **被 import**（`import src.sampling`）→ **把本模块对象整个替换成子包模块对象**，从而保证 `src.sampling.MAX == src.data.sampling.MAX` 这类全局状态/补丁不分裂。这是该写法比"两行 `from ... import *`"更正确的原因。

### shim → 真实实现 映射表（22 条，全部实测）

| shim（`src/` 根下） | 目标模块 | 字节 |
|---|---|---|
| `sampling.py` | `src.data.sampling` | 298 |
| `scannet_sens.py` | `src.data.scannet_sens` | 310 |
| `pipeline.py` | `src.inference.pipeline` | 313 |
| `batch_infer.py` | `src.inference.batch_infer` | 322 |
| `gpt_labels.py` | `src.inference.gpt_labels` | 319 |
| `gpt_sam31.py` | `src.inference.gpt_sam31` | 316 |
| `sam31_segment.py` | `src.inference.sam31_segment` | 328 |
| `yoloe_detect.py` | `src.inference.yoloe_detect` | 325 |
| `model_service.py` | `src.inference.model_service` | 328 |
| `model_worker.py` | `src.inference.model_worker` | 325 |
| `backproject_masks.py` | `src.geometry.backproject_masks` | 337 |
| `incremental_objects.py` | `src.geometry.incremental_objects` | 343 |
| `run_incremental_objects.py` | `src.geometry.run_incremental_objects` | 355 |
| `build_incremental_viewer.py` | `src.visualization.build_incremental_viewer` | 373 |
| `build_method_comparison.py` | `src.visualization.build_method_comparison` | 370 |
| `build_pairing_example.py` | `src.visualization.build_pairing_example` | 364 |
| `publish_gpt_viewer.py` | `src.visualization.publish_gpt_viewer` | 355 |
| `assemble_shoe_pair_revision.py` | `src.experiments.assemble_shoe_pair_revision` | 376 |
| `publish_shoe_pair_revision.py` | `src.experiments.publish_shoe_pair_revision` | 373 |
| `summarize_shoe_pair_revision.py` | `src.experiments.summarize_shoe_pair_revision` | 379 |
| `test_direct_pair_prompt.py` | `src.experiments.test_direct_pair_prompt` | 364 |
| `import_gpt_batch.py` | `src.experiments.import_gpt_batch` | 343 |

**仅有的两个例外**（`src/` 根下不是 shim）：
- `src/config.py`（1738 B）—— 真实实现，所有子包共用。
- `src/__init__.py`（71 B）—— 包声明。

> 阅读建议：看功能永远先跳子包；`src/` 根下的 22 个 shim 只在"为什么老命令还能跑"这个问题下才有意义。

---

## 1. `InstanceRecognition/` 顶层

```
InstanceRecognition/
├── .gitignore                     89 B    忽略虚拟环境、runtime 产物、生成的网页资源
├── README.md                   15537 B    项目总说明：默认方案、5 步流程、硬件/环境、场景重跑基线数据
├── start.sh                      303 B    启动脚本（可执行）
├── requirements-runtime.txt      319 B    运行时依赖清单（非开发依赖）
├── src/                                   核心处理代码（见第 2 节起）
├── tests/                                 单测 + 浏览器 QA 脚本（见第 9 节）
├── ThirdParty/                            外部方法仓库（SAM3 / WeDetect / yolo26 / yoloe26 等），本轮不展开
└── runtime/                               实验运行产物（batches、experiments、web 等），本轮不展开
```

逐文件说明：

- **`README.md`**（15,537 B）
  项目主文档。内容包含：默认方案为 **GPT-5.6 Terra 出标签 + SAM3.1 Multiplex 出掩码**；YOLOE 作为可选基线；5 步处理流程；`.venv` 为 Python 3.12.3 / PyTorch 2.10.0 + CUDA 12.8 / torchvision 0.25.0；**必须用 `taskset -c 0` 固定 CPU 0 启动**，否则依赖导入阶段会 SIGSEGV/SIGILL；离线查看入口 `runtime/offline/index.html`；`scene0000_00` uniform-500 重跑成绩为 500/500 帧、6,790 mask、约 18 分 48 秒，80–140 帧对比为 61 帧、YOLOE 792 mask vs GPT 479 mask；成对 prompt 修订覆盖 83 帧。

- **`start.sh`**（303 B，`rwxrwxr-x`）
  Web 服务启动脚本。逻辑：`cd` 到模块目录 → 优先使用 `.venv/bin/python`（可用环境变量 `SCENE_PYTHON` 覆盖）→ `exec ... -m src.web.server "$@"`。透传参数给 `src.web.server`。

- **`requirements-runtime.txt`**（319 B）
  运行时依赖：`torch==2.10.0`、`torchvision==0.25.0`、`ultralytics==8.4.149`、`numpy==1.26.4`、`opencv-python==4.11.0.86`、`setuptools<81`、`Flask>=3.1,<4`、`einops`、`pycocotools`、`psutil`、`scipy`、`scikit-image`、`pytest`。**与 GPU/驱动强绑定**，改动前需确认与 PyTorch 2.10.0 + CUDA 12.8 的兼容性。

- **`.gitignore`**（89 B）
  忽略本模块下的虚拟环境、`runtime/` 产物与生成出来的网页资源。

---

## 2. `src/` 根目录

```
src/
├── README.md                     978 B   子包职责表（data/inference/geometry/visualization/experiments/harness/config）
├── __init__.py                    71 B   包声明："Core ScanNet reading and instance-recognition inference package."
├── config.py                    1738 B   ★ 真实实现：路径与运行时配置（全子包共用）
│   ── 以下 22 个文件均为兼容 shim，内容见第 0 节映射表 ──
├── sampling.py                   298 B   → src.data.sampling
├── scannet_sens.py               310 B   → src.data.scannet_sens
├── pipeline.py                   313 B   → src.inference.pipeline
├── batch_infer.py                322 B   → src.inference.batch_infer
├── gpt_labels.py                 319 B   → src.inference.gpt_labels
├── gpt_sam31.py                  316 B   → src.inference.gpt_sam31
├── sam31_segment.py              328 B   → src.inference.sam31_segment
├── yoloe_detect.py               325 B   → src.inference.yoloe_detect
├── model_service.py              328 B   → src.inference.model_service
├── model_worker.py               325 B   → src.inference.model_worker
├── backproject_masks.py          337 B   → src.geometry.backproject_masks
├── incremental_objects.py        343 B   → src.geometry.incremental_objects
├── run_incremental_objects.py    355 B   → src.geometry.run_incremental_objects
├── build_incremental_viewer.py   373 B   → src.visualization.build_incremental_viewer
├── build_method_comparison.py    370 B   → src.visualization.build_method_comparison
├── build_pairing_example.py      364 B   → src.visualization.build_pairing_example
├── publish_gpt_viewer.py         355 B   → src.visualization.publish_gpt_viewer
├── assemble_shoe_pair_revision.py 376 B  → src.experiments.assemble_shoe_pair_revision
├── publish_shoe_pair_revision.py 373 B   → src.experiments.publish_shoe_pair_revision
├── summarize_shoe_pair_revision.py 379 B → src.experiments.summarize_shoe_pair_revision
├── test_direct_pair_prompt.py    364 B   → src.experiments.test_direct_pair_prompt
├── import_gpt_batch.py           343 B   → src.experiments.import_gpt_batch
│   ── 子包 ──
├── data/                         ScanNet .sens 读取与抽帧
├── inference/                    GPT 类别 / SAM / YOLO / 常驻 worker / 批处理 / 在线流水线
├── geometry/                     mask 反投影、几何关联基线与运行入口
├── harness/                      窗口式模型审阅、取证、版本化归属修改、点云与回放
├── visualization/                离线页面构建与发布
├── experiments/                  历史鞋配对修订等一次性实验
└── web/                          本地 Web 服务与离线浏览器生成
```

- **`src/README.md`**（978 B）
  子包职责表。这是目录分工的权威定义，与上面树中的一行说明一致。**新增子包时应同步更新此文件。**

- **`src/config.py`**（1738 B）★
  docstring: *"Filesystem and runtime configuration for the local instance-recognition app."*
  全仓路径与默认参数的单一来源。实测包含：
  - 路径：`ROOT`（`src/` 的父目录 = `InstanceRecognition/`）、`WORKSPACE`（`ROOT.parent` = 仓库根）、`THIRD_PARTY`、`WEIGHTS_ROOT`、`RUNTIME_ROOT`。
  - 解释器：`LOCAL_PYTHON`（`ROOT/.venv/bin/python`）、`DEFAULT_PYTHON`（本地 venv 存在则用它，否则 `sys.executable`）、`LEGACY_YOLOE_PYTHON`（`ThirdParty/yoloe26_prompt_free/.venv/bin/python`）。
  - 数据：`SCANNET_ROOT` = 环境变量 `SCANNET_ROOT`，默认 `WORKSPACE.parent / "data/scannet_v2"`。**注意默认值指向仓库同级而非模块内的 `data/`，这是测试失败路径不匹配的根源所在。**
  - 模型：`YOLOE_PYTHON`、`SAM31_PYTHON`、`YOLOE_CHECKPOINT`、`SAM31_CHECKPOINT`、`SAM3_REPO`。
  - 常量：`DEFAULT_CONFIDENCE=0.25`、`DEFAULT_IMAGE_SIZE=640`、`DEFAULT_MAX_DETECTIONS=16`、`DEFAULT_METHOD="gpt_sam31"`、`GPT_MODEL="openai/gpt-5.6-terra"`、`GPT_API_DIR`、`GPT_SAM_THRESHOLD=0.5`。

---

## 3. `src/data/` — ScanNet 输入层

职责（`src/README.md`）：读取 ScanNet `.sens` 并抽帧。这层不依赖 torch，刻意做成"轻依赖"。

```
data/
├── __init__.py                    45 B    包声明
├── sampling.py                   678 B    确定性时序抽帧
└── scannet_sens.py              8825 B    .sens 文件读取器
```

- **`__init__.py`**（45 B）
  包声明。

- **`sampling.py`**（678 B）
  docstring: *"Deterministic temporal sampling for ScanNet frame sequences."*
  定义 `uniform_sample_indices()`。用途：把一段帧序列按均匀时间间隔抽成 N 帧（"uniform-500" 即由此而来）。**确定性**是刻意设计——同一输入必须得到同一帧集合，否则实验不可复现。

- **`scannet_sens.py`**（8825 B）
  docstring: *"Small, dependency-light reader for RGB frames inside ScanNet v2 .sens files."*
  定义 `SensHeader`、`_read_exact()`、`read_header()`、`ScanNetSensReader`。这是整条流水线的数据入口：`.sens` 是 ScanNet 的二进制容器格式，本模块只实现够用的部分（读 header、按索引取 RGB 帧），从而**避免引入 ScanNet 官方重依赖**。`_read_exact()` 这类函数表明它对截断/损坏文件做了显式处理。

---

## 4. `src/inference/` — 感知与推理层

职责（`src/README.md`）：GPT 出类别、SAM/YOLO 出掩码、常驻 worker、批处理与在线流水线。这是本模块最大的子包，也是"默认方案"的实现处。

```
inference/
├── __init__.py                    50 B    包声明
├── pipeline.py                  7374 B    在线流水线编排（GPT + SAM3.1，可选 YOLOE 基线）
├── gpt_labels.py                2319 B    调用已审计的 API 客户端做帧级标注
├── gpt_sam31.py                12152 B    ★ 默认方案：GPT 标签驱动的 SAM3.1 文本提示分支
├── sam31_segment.py            15151 B    SAM3.1 第二阶段分割（也支持从 YOLOE 框出发）
├── yoloe_detect.py              7481 B    YOLOE-26 Prompt-Free 第一阶段的检测基线
├── model_service.py            14466 B    常驻 YOLOE / SAM3.1 worker 进程的生命周期管理
├── model_worker.py              3938 B    JSON-lines 协议的单模型 worker 进程
└── batch_infer.py               6753 B    离线批处理推理
```

- **`pipeline.py`**（7374 B）
  docstring: *"Orchestrate GPT labels and SAM3.1, with an opt-in YOLOE baseline."*
  定义 `InferenceFailure`、`InstanceRecognitionPipeline`。这是 `web/server.py` 背后真正的推理编排：GPT 标签 → SAM3.1 分割，YOLOE 需显式开启。`InferenceFailure` 的存在说明失败被建模为可区分异常而非裸崩。

- **`gpt_labels.py`**（2319 B）
  docstring: *"Adapter to the audited API client stored in the user's API directory."*
  定义 `client_module()`、`label_frame()`。它**不自己实现 HTTP 客户端**，而是从 `config.GPT_API_DIR` 动态加载那份已审计的客户端，只做适配。这是"延迟导入外部凭据/客户端"的隔离点。

- **`gpt_sam31.py`**（12152 B）★
  docstring: *"SAM3.1 text-prompt branch consuming independently generated GPT labels."*
  定义 `save()`、`segment()`、`write_display_artifacts()`、`GptSam31Segmenter`、`main()`。**默认方案 `gpt_sam31` 的实现体**：把 GPT 给出的类别文本作为 SAM3.1 的 text prompt 去分割，因此类别与掩码是"独立生成再组合"的关系，而非端到端单模型。

- **`sam31_segment.py`**（15151 B）
  docstring: *"Second-stage SAM3.1 masks from first-stage YOLOE boxes."*
  定义 `parse_args()`、`safe_name()`、`normalize_xywh()`、`normalized_wh_to_xyxy()`、`box_iou()`、`select_box_candidate()`、`select_point_object()`、`render_masks()`、`run_sam31()`、`Sam31Segmenter`、`main()`。同时承担两种入口：YOLOE→SAM3.1 两阶段链路的第二段，以及被 `gpt_sam31` 复用的分割内核。`box_iou()` / `select_box_candidate()` 表明它在做"多个候选框选哪个"的显式决策，而不是盲取。

- **`yoloe_detect.py`**（7481 B）
  docstring: *"First-stage YOLOE-26 Prompt-Free detection for the local application. The checkpoint has a segmentation head, but this stage deliberately records and renders only labels, confidences..."*
  定义 `parse_args()`、`render_boxes()`、`YoloeDetector`、`main()`。**刻意只用检测头、丢弃分割头**——这是为了让 YOLOE 基线与 GPT 方案在"谁负责掩码"这一点上对齐，保证对比公平。

- **`model_service.py`**（14466 B）
  docstring: *"Lifecycle manager for persistent YOLOE and SAM3.1 worker processes."*
  定义 `ModelServiceError`、`PersistentWorker`、`ModelServiceManager`。模型加载代价高，所以做成常驻进程；本模块管启动/健康检查/复用/清理。这也是必须 `taskset -c 0` 的环境风险集中区。

- **`model_worker.py`**（3938 B）
  docstring: *"JSON-lines model worker used by the long-lived web service."*
  定义 `parse_args()`、`send()`、`main()`。被 `model_service.py` 拉起，以 JSON-lines 在 stdin/stdout 上通信。**独立进程的收益是：模型崩溃不会带走 Web 服务。**

- **`batch_infer.py`**（6753 B）
  离线批处理推理入口。把 `pipeline` 的在线逻辑套用到"整批帧/整批场景"上，产出 `runtime/batches*` 下的结果。适合长跑与断点续跑，区别于 Web 服务的逐帧请求模型。

---

## 5. `src/geometry/` — 几何提升与关联层

职责（`src/README.md`）：mask 反投影、几何关联基线及运行入口。这一层是 **Harness 的上游**：Harness 消费的"不可变观测"就是这里产出的。

```
geometry/
├── __init__.py                    49 B    包声明
├── SCENE_PROJECTION.md           4455 B    ★ 场景点云投影的接口约定（输入/可见性判定）
├── backproject_masks.py         26551 B    ★ 用 AMB3R point map 提升 SAM 掩码（无实例融合）
├── scene_projection.py           4398 B    与来源无关的掩码提升（标定度量点云）
├── build_independent_scene_source.py 5955 B  用独立度量场景点云构建 Harness 默认 source
├── project_scene_masks.py        4811 B    导出通用场景点云掩码观测（不覆盖既有 run）
├── filter_lifted_masks.py        4423 B    构建"过滤后"的独立 source（绝不覆盖原始导出）
├── lifting_filter.py             4546 B    融合前的保守连通性过滤
├── incremental_objects.py        8972 B    不可变掩码观测的保守增量关联
├── run_incremental_objects.py   10856 B    用显式场景点云 source 重放前若干帧
├── cluster_prune.py             22767 B    ★ 对称分簇泄露剪枝（Otsu 分簇 + 结构闸门 + 证据门）
├── prune_scene_source.py        17653 B    ★ 把剪枝落成派生 source（生产链路唯一调用方）
└── leakage_eval.py              11742 B    ★ ScanNet 真值深度的泄露标注口径（评估唯一来源）
```

- **`SCENE_PROJECTION.md`**（4455 B）★
  接口约定文档，必读。规定输入：Cloud NPZ（`points_world` N×3 米制，可选 `rgb`）、source manifest、Camera JSON（`frames`/`sequence_index`/`intrinsics` 3×3/`world_to_camera` 4×4/可选 `depth_m`）。可见性有两条分支：**A** 有可靠对齐深度时按容差 `0.03 m + 0.01 * reference_depth` 判定；**B** 无深度时对完整场景做光栅化、取最近深度缓冲（默认 splat 半径 1）。

- **`backproject_masks.py`**（26551 B）★
  docstring: *"Lift existing SAM masks through AMB3R point maps, without instance fusion. Point maps already contain one backprojected world point per model-input pixel. We transfer each independent..."*
  这是全仓最大的几何文件。定义 `crop_box()`、`align_mask()`、`membership_bits()`、`apply_sim3()`、`select_source_ply_pixels()`、`instance_color()`、`display_colors()`、`write_ply()`、`read_point_ply()`、`encode()`、`write_json()`、`write_cloud_asset()`、`sha256_file()`、`relative()` 等。核心语义是**只做提升、不做融合**：每个掩码独立映射到世界坐标，实例归并留给下游（`incremental_objects` / Harness）。`apply_sim3()` 提示存在"模型坐标系 → 世界坐标系"的相似变换对齐。另注：模板路径取自 `config.WEB_ROOT / "mask_pointcloud"`（`main()` 内复制三个查看器文件）。

- **`scene_projection.py`**（4398 B）
  docstring: *"Source-independent mask lifting from a calibrated, metric scene point cloud. Pinhole, undistorted RGB coordinates; +Z forward camera convention. Depth is camera-axis Z in metres..."*
  定义 `VisibilityConfig`、`project_masks()`。与 `backproject_masks.py` 的关键区别是**不依赖 AMB3R 的 point map**，改用外部的标定度量点云——这是为"换数据源"准备的第二条路径，也是 Harness 默认 source 的理论依据。

- **`build_independent_scene_source.py`**（5955 B）
  docstring: *"Build the default Harness source from an independent metric scene cloud. This adapter deliberately keeps the existing RGB/mask manifest and replaces only the per-frame point export..."*
  定义 `_rigid()`、`build_camera_json()`、`build_default()`、`main()`。**适配器**：保留既有 RGB/mask manifest 不动，只替换逐帧点导出。这样 Harness 切数据源不需要改 manifest 结构。

- **`project_scene_masks.py`**（4811 B）
  docstring: *"Export generic scene-cloud mask observations for Harness without overwriting runs."*
  定义 `build()`。导出通用观测，**语义上保证不覆盖既有 run**——这是一条贯穿本层的不变量。

- **`filter_lifted_masks.py`**（4423 B）
  docstring: *"Build a separate filtered Harness source; never overwrite original exports."*
  定义 `build()`。把过滤结果写成**独立的** source，原始导出永远保留，便于 A/B 对比。

- **`lifting_filter.py`**（4546 B）
  docstring: *"Conservative, per-mask surface connectivity filtering before instance fusion. This is not a visibility oracle: a large coherent wrong surface can survive."*
  定义 `FilterConfig`、`filter_memberships()`。注意 docstring 里的**自我限制声明**：它做的是保守的表面连通性过滤，**不是可见性判定器**，因此一块大的、连贯的错误表面会被放过。这是设计者主动标注的已知失效模式。

- **`incremental_objects.py`**（8972 B）
  docstring: *"Conservative incremental association of immutable mask observations. Surface support is measured against the current point map; absence is unknown, not negative visibility evidence..."*
  定义 `family()`、`granularity()`、`voxelize()`、`Associator`。**本层最重要的语义原则**：观测是不可变的；"没看到"记为**未知**，不作为"不存在"的证据。这条原则直接决定 Harness 的判定边界。

- **`run_incremental_objects.py`**（10856 B）
  docstring: *"Replay the first sampled frames with an explicit scene-cloud source. The default source is the independent RGB-camera visibility projection used by the Harness. The old AMB3R repla..."*
  定义 `save()`、`digest()`、`ply()`、`main()`。运行入口：显式指定 source 重放前 N 帧。旧 AMB3R 重放路径被保留但已非默认。

- **`cluster_prune.py`**（22767 B）★
  docstring: *"Symmetric cluster-level leakage pruning for mask-lifted point clouds..."*
  定义 `ClusterPruneConfig`、`ClusterRecord`、`otsu_threshold()`、`camera_z()`、`decide_instance()`、`pruned_bits()`、`_mask_for()`、`_side_record()`、`_judge()`、`load_cameras()`、`load_manifest_frames()`、`_report()`、`main()`。**本层最新的下游后处理**，处理 2026-10 定位的遮挡泄漏问题。

  三点设计要点：(1) **保留 Otsu 只作分簇手段**，把"哪一簇是污染"交给对称判据——结构闸门（中心度、2D 面积比、像素重叠）拦掉"大平面被沿深度切开"的假分峰，证据门（3D 最近点距离 >0.30 m，或出近簇包围盒 >0.90）才判剪；(2) **近簇与远簇同等对待**，两侧都可能是污染；(3) **`pruned_bits()` 逐 bit 独立判定**，不要改成帧级统一 split，也不要让帧级 outcome 覆写逐 bit 结果——每个实例点集内部含两簇，逐 bit 才有 Otsu 可分性。

  **已知边界**：泄漏正好在物体正后方（两簇投影完全重合）时闸门弃权、漏检；单帧纯几何无法区分，需跨帧一致性。**尚未接入主流程**（无生产调用方）。

- **`leakage_eval.py`**（11742 B）★
  docstring: *"Ground-truth leakage labels for evaluating mask-lifting on ScanNet..."*
  定义 `SensDepthReader`、`resolve_depth_maps()`、`frame_delta()`、`estimate_bias()`、`frame_bias()`、`label_frame()`、`FrameLabel`、`load_cameras()`。把评估口径收进**唯一一处**，此前各分析脚本各写一份且互相矛盾。

  口径：`bias = median_frame(z_cloud − z_sensor)`，`leaked ⟺ (delta − bias) > 0.30 m`。旧口径 `0.03 + 0.01·z`（≈0.045 m）与点云自身噪声（帧 MAD 0.040 m）同量级，会把普通表面误差误标为污染。

  两个易错点已写进 docstring：(1) **`sequence_index` ≠ 源帧号**（500 帧采样下 seq 62 → 源帧 693），`resolve_depth_maps()` 强制走 `source_frame_id`，缺失即抛错；(2) **偏置必须是帧级量**，用实例子集自身中位会让泄漏子集吞掉自己的信号而全零，故 `label_frame()` 的 `bias_m` 默认 0.0 而非自动估算。

---

## 6. `src/harness/` — 模型主导的增量归属审阅框架

职责（`src/README.md`）：窗口式模型审阅、取证、版本化归属修改、点云与回放。这是本模块最新、最复杂的一层，也是近期实验的主战场。

```
harness/
├── __init__.py                    90 B    "Incremental object identity harness; fixed upstream geometry, reversible decisions."
├── README.md                   13137 B    ★ Harness 完整设计说明（窗口、6 工具、修改单、产物、模型参数）
├── run.py                      14569 B    ★ 主循环：按窗口驱动模型并在成功后提交
├── client.py                   15534 B    带日志脱敏与完整响应校验的受限网关请求
├── tools.py                     7509 B    6 个工具的 schema 与实现（受 cutoff 约束）
├── state.py                     8051 B    纯函数式事务校验（原始观测永不修改）
├── observations.py             11981 B    复用导出的逐掩码点 + 不可变输入审计
├── inputs.py                    2568 B    在 agent 行动前构建并审计窗口输入
├── history.py                   4862 B    按需历史代表帧选择（清晰度 + 掩码 bbox）
├── spatial_context.py           2220 B    来自窗口前序已提交状态的只读空间提示
├── publish.py                  12076 B    用既有离线 WebGL 渲染器发布已提交窗口
├── regenerate_overlays.py       4766 B    重渲染归档掩码（不改动原始 API 证据与决策）
├── migrate_label_fields.py      4092 B    一次性 schema 迁移（含内嵌 JSON），不做推理
├── migrate_object_ids.py        5404 B    对象 ID 迁移：O1 → Object1，不做推理
├── naming-conventions.txt       5345 B    ★ 专有名词与命名规范 v2
├── prompt.txt                  13894 B    ★ Agent 英文系统 prompt
└── prompt.zh-CN.txt            12698 B    对应中文版
```

- **`__init__.py`**（90 B）
  docstring: *"Incremental object identity harness; fixed upstream geometry, reversible decisions."* —— 这句话是全层的设计契约：上游几何固定，决策可回滚。

- **`README.md`**（13137 B）★
  Harness 权威说明。要点：**每 5 帧一个窗口**；执行顺序为 加锁 → `observations.py` 建不可变观测（形如 `0006:003`）→ 组织窗口输入 → `client.py` 调网关模型 → `state.py` 校验修改单 → `publish.materialize` 重算受影响物体的体素点云 → **全部成功后**才切换 checkpoint → 最后一次性建页。修改单支持 `create` / `assign` / `defer` / `repair` / `merge`。产物包含 `index.html`、`audit.html`、`versions/`、`checkpoint.json`、`calls/`、`evidence/`、`objects/`、`summary.json`。默认模型 `openai/gpt-6-sol`，非流式、`reasoning_effort=none`、`max_completion_tokens=10000`、每窗最多 14 次 API 往返。支持 `--prepare-only` / `--publish-only` / `--skip-publish`。

- **`run.py`**（16533 B，244 行）★
  docstring: *"Run bounded model-led incremental association on existing masks and geometry."*
  定义 `process_window()`、`publish_completed()`、`main()`。**窗口循环的实现主体**，对应 README 描述的"全部成功才提交"语义。注意 171–183 行附近会**校验已有 `config.json`**——这就是恢复历史实验时必须沿用旧 `output` 别名、而不能手改历史 `config.json` 的原因（例外：截断已终止实验时会同步改写 `frames` 与 `output`，见 `docs/harness-experiment-naming.md`）。
  `ACCEPTED_LIFTING_POLICIES`（第 20 行）对 `--source` 的 `lifting_policy` 做**精确匹配**，目前为原始投影与 `cluster_pruned_v1` 两项；放宽这个集合必须是有意为之，它正是防止 `--source` 误指到别的几何的护栏。
  `summary.json` 记录 `api_successful_calls` 之外，还拆分 `api_calls_in_committed_windows` 与 `api_calls_in_abandoned_attempts`，并列出发生重跑的 `retried_windows`：一个窗口失败后以新 `run_tag` 重跑，`calls/` 里会同时留下两个 session，只报总数会把重试说成对提交结果的贡献。

- **`client.py`**（15534 B）
  docstring: *"Bounded gateway requests with redacted logs and complete-response validation."*
  定义 `GatewayNetworkError`、`IncompleteSSEError`、`image()`、`redacted()`、`read_stream()`、`responses_input()`、`responses_message()`、`responses_usage()`、`Client`。两个关键职责：**日志脱敏**（不落任何凭据）与**完整响应校验**（SSE 中断会被识别为 `IncompleteSSEError`，不会静默当成正常结束）。

- **`tools.py`**（7509 B）
  docstring: *"Evidence access and validated proposals, bounded by current cutoff."*
  定义 `tool()`、`Tools`。**对外只有 6 个工具**：`list_objects`、`inspect_observations`、`inspect_object`、`compare_geometry`、`propose_update`、`commit_update`。`compare_pixels` **不存在**于代码、schema 或调用中，是已过期的名称。"bounded by current cutoff" 意味着模型**看不到它决策时点之后的数据**，这是防止未来信息泄漏的关键设计。

- **`state.py`**（8051 B）
  docstring: *"Pure, testable transaction validation. Original observations are never mutated."*
  定义 `_strip_legacy_observation_keys()`、`dumps_sanitized()`、`save()`、`initial()`、`propose()`。纯函数式校验，便于单测；"原始观测永不修改"与 `run.py` 的成功-才提交共同构成可回滚性。

- **`observations.py`**（11981 B）
  docstring: *"Reuse exported per-mask points and preserve an immutable input audit."*
  定义 `digest()`、`model_observation()`、`Observations`。**复用**而非重算上游几何导出，同时留下可校验的输入审计摘要（`digest()`）。

- **`inputs.py`**（2568 B）
  docstring: *"Build and audit a window input before any agent actions occur."*
  定义 `build_window_content()`。体积极小但时序关键：在模型做任何动作**之前**把窗口输入固定并留证。

- **`history.py`**（4862 B）
  docstring: *"On-demand historical representatives: one full RGB with one original-mask bbox. Clarity is the variance of scipy's discrete Laplacian..."*
  定义 `_frame_metrics()`、`_valid_bbox()`、`choose_representative()`、`historical_bbox_view()`。**按需**挑选历史代表帧；清晰度用离散 Laplacian 方差度量，且只带**一个**原始掩码 bbox（控制 token 成本）。

- **`spatial_context.py`**（2220 B）
  docstring: *"Read-only spatial hints from the window's prior committed state."*
  定义 `build_spatial_context()`。**只读**——空间提示来自已提交状态，不会把未提交假设喂回模型。

- **`publish.py`**（12076 B）
  docstring: *"Publish committed windows with the existing offline WebGL point renderer."*
  定义 `materialize()`、`_file_hash()`、`_snapshot_cache_key()`、`_snapshot_cache_hit()`、`publish()`。复用既有离线渲染器，并用文件哈希做快照缓存以省掉重复渲染。

- **`regenerate_overlays.py`**（4766 B）
  docstring: *"Re-render archived masks without altering original API evidence or decisions."*
  定义 `regenerate()`。**只重画、不改证据**——归档的 API 证据与决策不可被重渲染过程污染。

- **`migrate_label_fields.py`**（4092 B）
  docstring: *"One-off schema migration for saved observations/objects, including embedded JSON. No inference is performed."*
  定义 `migrate()`、`decode()`、`main()`。一次性历史 schema 迁移，明确**不做推理**。

- **`migrate_object_ids.py`**（5404 B）
  docstring: *"Migrate saved Harness object identifiers from O1 to Object1 without inference..."*
  定义 `rename()`、`migrate()`、`sha()`、`main()`。对象 ID 从 `O1` 改名为 `Object1`，同样不做推理，并用 `sha()` 做校验。

- **`naming-conventions.txt`**（5345 B）★
  专有名词与命名规范 v2。定义 observation、`instance_id` 等术语，观测编号规则为 `seq:instance`（如 `0000:001`）。**改代码前先读这份**，否则容易在术语上与既有一致性冲突。

- **`prompt.txt`**（13894 B）★
  Agent 英文系统 prompt，开头为 *"You maintain cross-frame object identities and ownership of observed 3D surface points in a static indoor scene..."*。注意"static indoor scene"是显式前提，场景动态化会使 prompt 假设失效。

- **`prompt.zh-CN.txt`**（12698 B）
  上者的中文对应版。两份需同步维护。

---

## 7. `src/visualization/` — 页面构建脚本与发布

职责：页面**构建脚本**（`build_*.py`）。2026-10-12 起，本层的手写前端模板已全部迁出到
`src/web/frontend/`，此处**只剩 Python**——生成器与模板分离，见第 8.2 节。

```
visualization/
├── __init__.py                    54 B    包声明
├── object_bboxes.py            30663 B    ★ 观测支撑 bbox 拟合与逐版本 Harness 导出（只写派生文件）
├── build_harness_records.py     7435 B    ★ 发布不可变 Harness 记录页（逐窗口加载）
├── build_harness_result.py      8996 B    Harness 结果页：套用 RGB/overlay/category + 点云布局
├── build_method_comparison.py  14402 B    方法对比页（复用掩码查看器 + AMB3R map）
├── build_pairing_example.py    10043 B    孤立实验页：第 6 帧 prompt/分组
├── build_incremental_viewer.py   703 B    增量重放查看器（薄封装，复用既有渲染器）
├── publish_gpt_viewer.py        5482 B    在原 URL 发布完整 GPT viewer，保留基线资源
├── translate_harness_reasons.py  6226 B    把待翻译的英文 reason 译成独立中文 sidecar
├── page_naming.py               1862 B    逐实验页面命名的共享函数
└── harness_records.README.md    2151 B    交互记录页发布流程说明（文档，非前端资源）
```

> **模板去向**：`harness_result.html/css`、`all500-layout.{css,js}`、`harness_*.js`（5 个）
> 现位于 `src/web/frontend/harness_result/`；`harness_records/page.{html,js,css}` 位于
> `src/web/frontend/harness_records/`。生成器统一通过 `config.WEB_ROOT` 定位，不要再
> 拼接 `runtime/web/`。

生成器逐一说明：

- **`object_bboxes.py`**（30663 B）★
  docstring: *"Shared observed-support bbox fitting and per-version Harness export. Only derived files are written..."*
  定义 `sha()`、`surface_planes()`、`exterior_planes()`、`fit_plane_faces()`、`horizontal_planar_footprint()`、`apply_planar_footprint_filter()`、`supported_quantile_bounds()`、`oriented_bounds()`、`keep_dominant_component()`、`trim_weak_boundary_points()`、`export_version_objects()`、`_cached_version_objects()`、`export_harness_bboxes()`。全仓第二大文件。做的是"从观测支撑点集拟合可靠包围盒"这一件事，函数序列显示完整流水：平面提取 → 水平投影足迹过滤 → 分位数边界 → 有向包围盒 → 取主连通分量 → 削弱边界点。**只写派生文件**是硬约束。

- **`build_harness_records.py`**（7402 B）★
  docstring: *"Publish immutable harness logs, loading one window at a time. Chinese translations live in a separate hash-keyed JSON file..."*
  定义 `dump_js()`、`reasons()`、`rank()`、`compact()`、`main()`。交互记录页生成器：**逐窗口加载**以控内存；中文翻译存放在**以源文本哈希为键**的独立 JSON 中。

- **`build_harness_result.py`**（8996 B）
  docstring: *"Apply RGB/overlay/category + cloud layout to published Harness results."*
  定义 `build()`。给已发布的 Harness 结果套上 RGB/overlay/类别与点云布局。
  ⚠ 它会**整页覆写 `index.html`**，因此 `publish.py` 写在页面里的实验说明会被丢弃。几何披露必须在此处重新施加：剪枝 source（`cluster_pruned_v1`）会替换 `RESULT_NOTE` 为含 `cluster_prune` 的版本，锚点文本缺失时直接抛错而不是静默跳过。

- **`build_method_comparison.py`**（14389 B）
  docstring: *"Build an offline method comparison using the existing mask viewer and AMB3R maps."*
  定义 `viewer_bundle()`、`main()`。方法对比页（如 YOLOE vs GPT 的 mask 数对比）的构建者。

- **`build_pairing_example.py`**（10043 B）
  docstring: *"Build the isolated frame-6 prompt/grouping experiment using the existing 3D viewer."*
  定义 `main()`。第 6 帧 prompt/分组孤立实验页，属一次性实验产物。

- **`build_incremental_viewer.py`**（682 B）
  docstring: *"Reuse the existing point renderer and raw-mask viewer for incremental replay."*
  定义 `build()`。极薄的一层，只做复用与装配。

- **`publish_gpt_viewer.py`**（5482 B）
  docstring: *"Publish a complete GPT viewer at the existing URL while retaining baseline assets."*
  定义 `rebase_manifest()`、`write_manifest()`、`main()`。**原地发布**：URL 不变，基线资源保留，靠 manifest 重定位。

- **`translate_harness_reasons.py`**（6226 B）
  docstring: *"Translate pending English reason fields into an independent Chinese sidecar..."*
  定义 `main()`。只翻**待翻译**的条目，写入独立 sidecar，不覆盖英文原文。

- **`page_naming.py`**（1862 B）
  docstring: *"Shared names for per-experiment result, records and translation artifacts."*
  定义 `canonical_run_id()`、`records_page_name()`、`records_asset_name()`。**页面命名的单一来源**，与仓库 `docs/harness-experiment-naming.md` 的规则配套。页面与资源都落在**实验目录内**，内名固定为 `records.html` 与 `records/`——实验标识由目录名承载，不做二次推导。

模板与资源说明（**非成品页**）：

- **`harness_records/README.md`**（2151 B）
  交互记录页发布流程：`build_harness_records.py --run <run>` → `translate_harness_reasons.py --run <run> --batch-size 30` → 以 `--require-translated` 重建。翻译以源文本 SHA-256 为键存入 `translations.zh-CN.json`，审计记录在 `translation_audit/`。**本目录是模板源文件**，各实验目录下 `records/` 里的是被忽略的生成物。

- **`harness_records/page.html` / `page.js` / `page.css`**（模板源文件）
  交互记录页的模板源文件，被 `build_harness_records.py` 读取后填充数据生成实际页面。
  **2026-10-12 起位于 `src/web/frontend/harness_records/`。**

  ⚠️ **git 状态（2026-10-13 决定）**：`src/web/frontend/` **整体不纳入版本控制**，已在
  `InstanceRecognition/.gitignore` 中显式声明该目录。这三个文件与其余前端源码一样，
  **只存在于工作区**。

  此前 `page.js` / `page.css` 在旧路径下是 intent-to-add（index 里为空 blob
  `e69de29bb2...`），`page.html` 未跟踪；`harness_result.html` 等 6 个文件则曾被跟踪。
  搬迁后，这些旧 index 条目已用 `git rm --cached` 全部清除，`git status` 不再把它们
  报为误删除。

  **代价需知晓**：`git clean -fd` 或 `git checkout` 会永久删除这些文件（未被跟踪、
  且其中 2 个带未提交改动：`harness_result.html` 删 6 行、`harness_object_groups.js`
  删 68 行）。需要时从 `git show HEAD:<旧路径>` 可恢复被跟踪过的 6 个的历史版本，
  但 `runtime/web/` 迁移过来的 19 个从未入库，无历史可恢复。

- **`harness_result/` 下的 8 个前端资源**（现位于 `src/web/frontend/harness_result/`）
  由 `build_harness_result.py` 使用，全部随该生成器一起读取并复制到实验输出目录：
  - `harness_result.html`（5878 B）/ `harness_result.css`（6112 B）——结果页模板与样式
  - `all500-layout.js`（2548 B）/ `all500-layout.css`（2660 B）——all500 查看器布局
  - `harness_bbox_display.js`（4179 B）——3D 包围盒显示
  - `harness_cloud_pick.js`（5077 B）——基于精确深度测试的物体拾取
  - `harness_input_display.js`（611 B）——显示**已保存的 API 证据**，而非重建结果
  - `harness_object_groups.js`（5456 B）——物体列表与分组视图

  其中 `harness_input_display.js` 另被 `src/harness/regenerate_overlays.py` 复制使用。

---

## 8. `src/experiments/` 与 `src/web/`

### 8.1 `src/experiments/` — 一次性实验

职责（`src/README.md`）：历史鞋配对修订等一次性实验。**这些是一次性脚本，不是通用库。**

```
experiments/
├── __init__.py                    52 B    包声明
├── assemble_shoe_pair_revision.py 3610 B  用鞋专用修订 + 未变帧装配出版本化 500 帧 run
├── import_gpt_batch.py            4591 B  把已验证的批处理结果接入默认流水线缓存（不推理）
├── publish_shoe_pair_revision.py  4444 B  发布已验证的鞋专用修订，只替换受影响的在线缓存
├── summarize_shoe_pair_revision.py 3423 B 汇总观测到的 prompt/mask 变化（不声称准确率）
└── test_direct_pair_prompt.py     3678 B  孤立第 6 帧 SAM prompt 干预（无 GPT 调用、无分组）
```

- **`assemble_shoe_pair_revision.py`**（3610 B）
  docstring: *"Assemble a versioned 500-frame run from a shoe-only revision and unchanged frames."*
  定义 `load()`、`write()`、`digest()`、`main()`。把"仅鞋的修订"与"未改动的帧"拼成一个版本化的 500 帧 run。

- **`import_gpt_batch.py`**（4591 B）
  docstring: *"Expose verified batch results through the default pipeline cache without inference."*
  定义 `main()`。**不推理**——只把已验证的批处理结果接线到默认流水线缓存，属于结果搬运。

- **`publish_shoe_pair_revision.py`**（4444 B）
  docstring: *"Publish the validated shoe-only revision and replace only affected online caches."*
  定义 `main()`。**最小替换**：只更新受影响的在线缓存，不动其余部分。

- **`summarize_shoe_pair_revision.py`**（3423 B）
  docstring: *"Summarize observed prompt/mask changes without claiming accuracy."*
  定义 `main()`。注意其措辞：只汇总**观测到的**变化，**不声称准确率**——这是刻意的表述边界。

- **`test_direct_pair_prompt.py`**（3678 B）
  docstring: *"Isolated frame-6 SAM prompt intervention; no GPT call or mask grouping."*
  定义 `main()`。孤立干预实验，刻意排除 GPT 与分组环节以隔离变量。

### 8.2 `src/web/` — 本地服务、离线浏览器与网页源码

```
web/
├── __init__.py                    65 B    包声明
├── server.py                    8543 B    ★ 本地 ScanNet 帧浏览器 + GPT/SAM3.1 推理服务
├── build_offline_browser.py     5481 B    为已完成的推理结果构建 file:// 兼容浏览器
└── frontend/                              ← 全部手写网页源码（25 个文件，非 Python）
    ├── index.html  app.js  styles.css     Flask 推理页
    ├── mask_pointcloud/                   点云查看器模板（index/styles/viewer）
    ├── method_comparison/                 方案对比页模板（app/bridge/embedded/styles/index）
    ├── incremental_objects/               增量回放页模板（index/incremental.js/css）
    ├── harness_result/                    Harness 结果页模板（8 个，说明见第 7 节）
    └── harness_records/                   Harness 记录页模板（page.html/js/css）
```

- **`server.py`**（8543 B）★
  docstring: *"Local ScanNet frame browser with GPT + SAM3.1 inference."*
  定义 `create_app()`、`public_summary()`、`parse_args()`。`start.sh` 的最终目标（`python -m src.web.server`）。用 Flask（`requirements-runtime.txt` 中的 `Flask>=3.1,<4`）。真实推理经 `inference/model_service.py` 的常驻 worker 完成，而非在 Web 进程内加载模型。

  页面经 `config.WEB_ROOT` 定位，**不要**写死 `runtime/web`：`RUNTIME_ROOT` 可被
  `INSTANCE_RECOGNITION_RUNTIME` 覆盖，把网页源码放在运行时目录下会导致覆盖后整站 404。

- **`build_offline_browser.py`**（5481 B）
  docstring: *"Build a file:// compatible browser for completed inference results."*
  定义 `relative_url()`、`require_file()`、`build_index()`、`write_data()`、`parse_args()`、`main()`。产出可 `file://` 直接打开的离线浏览器——这是"不需要起服务也能看结果"的关键，`relative_url()` 的存在表明所有链接都被改成相对路径。

- **`frontend/`**（25 个 `.html/.css/.js`，目录约定见 `src/README.md`）
  **是代码，不是运行产物**，因此与 `server.py` 同包，不放在 `runtime/` 下。三个查看器
  模板（`mask_pointcloud`、`method_comparison`、`incremental_objects`）由生成器复制到
  实验输出目录，并替换其中的占位符——它们**不是给人直接打开的成品页**。

---

## 9. `tests/` — 文件清单

`tests/` 共 33 个文件（32 个 `.py` + 1 个 `.js`，另有 `__pycache__/`）。命名约定：`test_*.py` 是断言型单测（pytest），`check_*.py` 是真实浏览器 QA 脚本（需人工/浏览器参与，不做断言）。

```
tests/
├── test_sampling.py              823 B    均匀抽帧的确定性
├── test_scannet_sens.py         1166 B    .sens 读取（数据缺失时 skipif）
├── test_backproject_masks.py    7503 B    掩码反投影与辅助函数
├── test_gpt_pipeline.py         4290 B    GPT + SAM3.1 在线流水线
├── test_model_service.py        3111 B    常驻 worker 生命周期
├── test_incremental_objects.py  3417 B    增量关联语义（含"缺席即未知"）
├── test_lifting_filter.py       5802 B    融合前的保守连通性过滤
├── test_scene_projection.py     6317 B    场景点云投影与可见性分支
├── test_scene_projection_export.py 2002 B 场景投影导出
├── test_independent_scene_source.py 3629 B 独立场景 source 构建
├── test_cluster_prune.py        8218 B    对称分簇剪枝（闸门/证据门/逐 bit 独立性/已知边界）
├── test_leakage_eval.py         5474 B    泄露标注口径（帧级偏置、源帧号映射）
├── test_history.py              3060 B    历史代表帧选择
├── test_lazy_history.py         3341 B    历史代表帧的按需/惰性行为
├── test_spatial_context.py      1386 B    只读空间提示
├── test_object_bboxes.py        5016 B    包围盒拟合与版本化导出
├── test_harness.py             12325 B    Harness 状态/工具/事务校验
├── test_harness_client.py      11110 B    网关客户端（脱敏、完整响应校验）
├── test_harness_run.py          7374 B    窗口循环与提交语义
├── test_visualization_cache.py  2705 B    可视化快照缓存
├── test_experiment_pages.py     1776 B    实验页面命名与生成
├── test_prune_scene_source.py  12489 B    剪枝 source 构建（掩码解析、只清位、同级断言）
├── test_harness_policy_gate.py  3160 B    剪枝策略的 gate/标签/披露校验
├── test_offline_browser.py      1143 B    离线浏览器构建
├── test_server.py               2500 B    Web 服务（数据缺失时 skipif）
├── test_episode_stats.js        1272 B    前端 episode 统计脚本（node 运行）
├── check_gpt_batch.py           3719 B    QA：批处理结果页
├── check_mask_pointcloud_viewer.py 30555 B QA：掩码点云查看器（最大）
├── check_method_comparison.py   7422 B    QA：方法对比页
├── check_pairing_example.py     4693 B    QA：配对示例页
├── check_incremental_viewer.py  5553 B    QA：增量查看器
├── check_shoe_pair_revision.py  4012 B    QA：鞋配对修订页
└── check_harness_viewer.py      4057 B    QA：Harness 查看器页
```

**当前测试基线**：`204 passed, 4 subtests passed`。`test_scannet_sens.py` 与 `test_server.py` 从 `config.SCANNET_ROOT` 取数据根，数据集不存在时整体 `skipif`，不再产生固定失败。`test_episode_stats.js` 单独运行为 `EPISODE_STATS_OK`。

> 测试需固定 CPU：`.venv/bin/python -m pytest` 不加 `taskset -c 0` 会在依赖导入阶段崩溃
> （SIGSEGV/SIGILL，见 `README.md` 的实测报告）。

---

## 10. 阅读路径建议

要按"从数据进来到页面出去"的顺序建立全局图景，建议：

1. `src/config.py` + `src/README.md` —— 明确路径与目录分工。
2. `src/data/`（2 个实现文件）—— 数据入口，最小、最好读。
3. `src/inference/` —— 先 `pipeline.py`（编排）再 `gpt_sam31.py`（默认方案）与相关 worker。
4. `src/geometry/` —— 先读 `SCENE_PROJECTION.md`（接口约定），再 `incremental_objects.py`（核心语义：缺席即未知）。
5. `src/harness/` —— 先读 `README.md`、`naming-conventions.txt`、`prompt.txt`，再 `run.py` → `tools.py` → `state.py`。
6. `src/visualization/` —— 先 `page_naming.py`（命名约定）再各 `build_*.py`。这一层已是**纯 Python**；它使用的模板全部在 `src/web/frontend/`，经 `config.WEB_ROOT` 定位。
7. `src/web/frontend/` —— 全部手写网页源码，按页面分目录（Flask 页、三个查看器、harness 结果与记录页）。这是"页面上看到的东西"的最终来源。
8. `src/web/server.py` —— 最后回到"用户实际怎么用"。

若想直接看重放路径而非逐目录浏览，跳到 **第 11 节「实例识别任务的启动与执行调用链」**：那里按"服务启动 / 模型加载卸载 / 单帧推理"三条链路，给出了依次运行的文件与函数（含行号）。

## 11. 实例识别任务的启动与执行调用链

本节回答一个具体问题：**用我们的 pipeline 启动一次实例识别任务，依次会跑哪些文件、哪些函数。** 分三条链路记录（三条链路各自的**职责与相互关系**先看 11.0），全部为源码实测的行号，未运行验证的部分单独标注。

补充一个前提：按 `Code_Review.md` 第 0 节，代码里 import 常走 `src/` 根下的 shim。下表出现的 `src.model_service`、`src.pipeline`、`src.scannet_sens` 都是 shim，**实际执行体在对应子包**，行号按子包文件给出。

### 11.0 三条链路在 pipeline 中各自承担什么

这一节要回答的不是「谁先谁后」，而是**这三条链路各自在 pipeline 里担了一份什么活**。区分它们的标准是「**处理的对象是什么、产出是什么**」——三者处理的东西完全不同，所以不能混着看。

| 链路 | 处理对象 | 承担的职责 | 产出 | 触发者 |
|---|---|---|---|---|
| **A 服务启动** | 进程与对象图 | 组装整个服务：造出 reader / model_service / pipeline 三个对象，注册全部 HTTP 路由 | 一个可被调用的 Flask 服务（**此时还没有模型，也还不会识别**） | `bash start.sh`，一次 |
| **B 模型生命周期** | GPU 上的模型子进程 | 管显存占用：按需加载模型、跨请求复用、空闲后回收 | worker 子进程的存活状态（占 / 不占显存） | 浏览器页面 |
| **C 单帧推理** | 一帧 RGB 图 | 执行识别本体：取帧 → GPT 出类别 → SAM3.1 出掩码 → 契约校验 → 落盘 | `summary.json` + 各 stage 产物 | 每次 `POST /api/infer` |

换个角度看，这三条链路正好对应**三个对象的三段生命周期**：

| 对象 | 谁创建 | 谁驱动 | 谁使用 |
|---|---|---|---|
| `ScanNetSensReader` | A | — | C（`frame_path()` 取帧） |
| `ModelServiceManager` | A | **B**（load / unload / heartbeat） | **C**（`infer_sam31_text()` 等） |
| `InstanceRecognitionPipeline` | A | — | **C**（`run()` 总编排） |

从这两张表能直接读出三者的分工与关系：

**A 负责定义 pipeline 的「能力面」。** 它把服务对外能做的操作全部摆出来——也就是 `server.py:31-135` 那十几个路由。这里有一个好用的判断点：**A 结束时，除 `/api/infer` 之外的接口已经全部可用**，因为 `/api/scenes`（列场景）、`/api/frame/...`（取帧）、`/api/artifact/...`（取历史产物）、`/api/health`（查依赖）都不需要模型。换句话说，**A 跑完时 pipeline 已经能浏览数据、能读历史结果，只是还不能算新的一帧。**

**B 负责为 C 摊销模型加载成本。** 它自己不算任何一帧，唯一目的是把「每次推理都要先等模型加载」变成「加载一次、多帧复用、闲置再卸」。默认方法 `gpt_sam31` 下只注册一个 worker——`sam31_text`（`model_service.py:228-233`），这也是它唯一需要管的东西。

**C 就是 pipeline 的目的本身。** 它是唯一真正产出识别结果的链路；A 铺好的接口和 B 管着的模型，都是为了让这一帧能算完。

再落到时间关系上——这一点最容易误读，所以单独画出来：**A 严格前置，B 与 C 交错，且 B 不是 C 的前提。**

```
A 服务启动（只跑一次，B、C 都以它为起点）
│
├─ B（可反复）      开页面 → 加载模型 ──────────┐
│                                             │ 模型驻留显存
└─ C（穿插其间）           ├─ /api/infer ─────┤ 复用 B 已加载的模型
                          ├─ /api/infer ─────┤
                          关页面 → 空闲 20s ──┘ → 卸载
```

**为什么说 B 不是 C 的前提：** `PersistentWorker.request()`（`model_service.py:80`）进入临界区后**无条件**先调 `_start_locked()`（`:44`）：

```python
def request(self, command: str, **payload) -> dict:
    with self._lock:
        self._start_locked()      # 已活着就复用；没有或已死，就地拉起
```

而 `_start_locked()` 第一句就是 `if self._process is not None and self._process.poll() is None: return`；`_safe_load()`（`:293`）的注释也把同一件事写明：`# The status endpoint exposes the worker error; infer can retry/restart.`。所以 **C 本身自带「现用现备」的能力**，B 的作用只是把模型加载从「第一次点推理」提前到「打开页面」。

> 这条是静态读源码 + 读测试得出的：`tests/test_server.py` 对 `/api/infer` 只断言缺 `frame_index` 返 400，**没有「必须先 open_client」的断言**。⚠️ **未实跑验证**（验证需真正起服务、加载模型、占显存）。

B 与 C 之间确实有一处耦合，但那是**并发保护，不是顺序依赖**：`_infer()`（`:318`）进出分别 `_active_requests += 1` / `-= 1`，而 `unload_all()`（`:330`）首句是 `if self._active_requests: return`——这对守卫要保证的是卸载线程不会把推理途中的模型拆掉。

### 11.1 链路 A：服务启动与接口装配（一次性，`bash start.sh`）

| # | 文件 | 函数 / 位置 | 做什么 |
|---|---|---|---|
| A1 | `start.sh` | 全体（bash） | 切到 `InstanceRecognition/`；选解释器 `SCENE_PYTHON` > `.venv/bin/python` > `python3`；`exec ... -m src.web.server "$@"` |
| A2 | `src/web/server.py` | `__main__`（:204） | 入口，调用 `parse_args()` 与 `create_app()` |
| A3 | `src/web/server.py` | `parse_args()`（:196） | 解析 `--host`（默认 `127.0.0.1`）、`--port`（默认 **8090**）、`--debug` |
| A4 | `src/inference/model_service.py` | `ModelServiceManager.__init__()`（:180） | 校验 `method` ∈ {`gpt_sam31`,`yoloe_sam31`}；**只组装 worker 命令行，不启动进程**；起 `model-lifecycle-monitor` 线程（:241-244）；`atexit.register(shutdown)`。默认 `gpt_sam31` 只注册 `sam31_text` 一个 worker；`yoloe_sam31` 注册 `yoloe` + `sam31` |
| A5 | `src/data/scannet_sens.py` | `ScanNetSensReader.__init__()`（:88） | 记录 `scans_root` 与 `cache_root`，仅此而已（不预读）。由 `server.py:26` 构造 |
| A6 | `src/inference/pipeline.py` | `InstanceRecognitionPipeline.__init__()`（:20） | 校验 method；建 `threading.Lock()` 作推理串行锁。由 `server.py:28` 构造 |
| A7 | `src/web/server.py` | `create_app()`（:18）注册路由 | 把 reader / model_service / pipeline 装配成 Flask app |
| A8 | `src/web/server.py` | `application.run(...)`（:205-215） | `threaded=True`、**`use_reloader=False`**（避免 fork 第二个进程重复加载依赖，即 SIGSEGV/SIGILL 风险点） |
| A9 | `src/web/server.py` | `finally`（:216） | 进程退出时必定 `application.extensions["model_service"].shutdown()` 回收 worker 子进程 |

**结论：链路 A 全程不加载模型、不跑推理。** 服务起来后处于待命状态。

### 11.2 链路 B：模型生命周期管理（由前端驱动，可反复）

| 触发 | 文件 | 函数 | 行为 |
|---|---|---|---|
| 页面打开 | `src/inference/model_service.py` | `open_client()`（:249）→ `preload_async()`（:273）→ `_safe_load()`（:293） | 后台线程对每个 worker 发 `load`，**首次真正 `subprocess.Popen` 拉起模型进程** |
| 页面存活 | 同上 | `heartbeat()`（:258） | 刷新 `_clients[client_id]` 时间戳 |
| 页面关闭 | 同上 | `close_client()`（:266） | 移除客户端，记录归零时刻 |
| 每秒轮询 | 同上 | `_monitor_loop()`（:363） | 心跳超时（`heartbeat_timeout=15.0s`）摘客户端；**客户端归零 + 无在途请求 + 再等 `unload_grace=5.0s`** → `unload_all()`（:330）释放显存 |
| 手动 | `src/web/server.py` | `/api/models/status`（:77）、`/api/models/unload`（:81） | 查状态 / 立即卸载 |

前端侧（`src/web/frontend/app.js`）顺序：`/api/models/status` → `/api/client/open` → 点推理时 `/api/infer` → `/api/client/heartbeat`。

### 11.3 链路 C：单帧推理执行（每次 POST `/api/infer`）

以下为默认方法 `gpt_sam31` 的完整链路。

| # | 文件 | 函数 / 位置 | 做什么 |
|---|---|---|---|
| C1 | `src/web/frontend/app.js` | 前端 | POST `/api/infer`，body 为 `{scene_id, frame_index, force}` |
| C2 | `src/web/server.py` | `infer()`（:112） | 校验 `frame_index` 可转 int；调 `pipeline.run(scene_id, frame_index, force)`（:120） |
| C3 | `src/inference/pipeline.py` | `run()`（:51） | 单帧总编排 |
| C4 | `src/inference/pipeline.py` | `_assert_dependencies()`（:40） | 默认分支查 `SAM31_PYTHON`、`SAM31_CHECKPOINT`、`sam3/model_builder.py`、`GPT_API_DIR/label_frames.py`、`GPT_API_DIR/prompt_en.txt`；缺哪个在报错里点名 |
| C5 | `src/data/scannet_sens.py` | `frame_path()`（:142） | 取 RGB 帧。缓存 `runtime/frames/<scene>/frame_%06d.jpg` 命中即返回；否则 `header()`（:106）→ `_read_color_bytes()`（:123）解析 `.sens` → 校验 JPEG format/尺寸 → 写 `.jpg.tmp` 后 `os.replace` 原子落盘（双检锁） |
| C6 | `src/inference/pipeline.py` | `result_root()`（:36） | 算出 `runtime/results_gpt/<scene>/frame_%06d`（`yoloe_sam31` 则为 `results/`） |
| C7 | `src/inference/pipeline.py` | 缓存分支（:55-64） | `summary.json` 存在且未 `force` → 直接返回并标 `cached: true`；随后在 `_inference_lock`（:66）内**再查一次**（双检锁，防并发重复跑） |
| C8 | `src/inference/pipeline.py` | `_run_gpt()`（:105） | `gpt_sam31` 分支主体 |
| C9 | `src/inference/gpt_labels.py` | `client_module()`（:8） | `@lru_cache(maxsize=1)`；用 `importlib.util.spec_from_file_location` 从 `config.GPT_API_DIR/label_frames.py` **动态加载**已审计客户端，不 import 项目内副本 |
| C10 | `src/inference/gpt_labels.py` | `client.settings()`（:18） | 读 `VIBEG_API_KEY`（缺则 RuntimeError）与 `VIBEG_OPENAI_BASE_URL`；读 `prompt_en.txt` 并算 prompt sha256 |
| C11 | `src/inference/gpt_labels.py` | 配置校验（:28-31） | 写 `request_config.json`；若已存在且内容不同且未 `force` → `ValueError('Cached GPT request configuration changed')`。这是在防"缓存结果与当前配置不符" |
| C12 | `src/inference/gpt_labels.py` | `client.label_one(...)`（:33） | **真正的网络调用**（GPT，`model=config.GPT_MODEL`、`reasoning_effort=none`、`image_detail=high`、`max_completion_tokens=4096`）；结果写 `gpt/labels.json` |
| C13 | `src/inference/pipeline.py` | `model_service.infer_sam31_text(...)`（:107） | 把 GPT 类别当文本提示交给 SAM3.1 |
| C14 | `src/inference/model_service.py` | `infer_sam31_text()`（:300）→ `_infer()`（:318） | `_infer` 先 `_active_requests += 1` 并清 `_zero_clients_since`（**防止卸载线程在推理中途卸模型**），再转发给 worker |
| C15 | `src/inference/model_service.py` | `PersistentWorker.request()`（:80） | 首次调用内部触发 `_start_locked()`（:44）→ `subprocess.Popen(["...python","-m","src.model_worker","--stage","sam31_text",...])`；写一行 JSON 到 stdin，从队列等回包（`request_timeout=300.0s`）；非 JSON 行写进日志并跳过（容错第三方库污染） |
| C16 | `src/inference/model_worker.py` | `main()`（:31） | worker 主循环。启动时 `sys.stdout = sys.stderr`（:36）**保证 stdout 只走协议**；按行读 stdin 分发 `load` / `infer` / `shutdown` |
| C17 | `src/inference/model_worker.py` | `load_model()`（:38） | 惰性加载，`model is not None` 则复用之；`sam31_text` 分支构造 `GptSam31Segmenter` |
| C18 | `src/inference/gpt_sam31.py` | `GptSam31Segmenter.__init__()`（:109） | 内部持有一个 `Sam31Segmenter(checkpoint, threshold)`，**与批处理共用同一套分割逻辑** |
| C19 | `src/inference/gpt_sam31.py` | `GptSam31Segmenter.infer()`（:115） | 读 `labels.json`；**双重一致性校验**：`labels['source']` 与传入 source 路径一致、`labels['source_sha256']` 与当前帧文件实际 sha256 一致（防"标签与图不是同一帧"） |
| C20 | `src/inference/gpt_sam31.py` | `segment(...)`（:118） | 调 `self.segmenter.predictor` 真跑分割，`max_masks=16` |
| C21 | `src/inference/gpt_sam31.py` | `write_display_artifacts()` + `save()`（:122-124） | 落 `annotated.jpg`、`instance_id_map.png`，写 `sam31/predictions.json` |
| C22 | `src/inference/pipeline.py` | `_validate()`（:126） | 契约断言：`instance_count == len(instances)`；`policy.yoloe_input_consumed is False`；`gpt/labels.json`、`sam31/predictions.json`、标注图、instance map、**每个 raw candidate 的 mask 文件**都必须已落盘 |
| C23 | `src/inference/pipeline.py` | 落盘（:122-125） | 写 `summary.json.tmp` → `os.replace` 原子替换 |
| C24 | `src/web/server.py` | `public_summary()`（:150）→ `jsonify` | 把内部 summary 转成前端契约；artifact URL 形如 `/api/artifact/<scene>/<frame>/<stage>/<filename>` |
| C25 | `src/web/frontend/app.js` | 前端 | 渲染原图、标注图、instance map、实例列表 |
| C26 | 取产物 | `src/web/server.py` | `artifact()`（:123）白名单校验：`stage` ∈ {yoloe, sam31, gpt}，`filename` ∈ 5 个允许项，否则 `ValueError` → 400 |

### 11.4 另一条分支：`yoloe_sam31`（方法二）

与链路 C 的差异只在 C8 之后：`pipeline.run()` 不动用 `_run_gpt`，而是

`model_service.infer_yoloe()`（:303）→ worker `--stage yoloe` → `model_worker.load_model()` 构造 `src/inference/yoloe_detect.py::YoloeDetector`（:111 有 `infer`）→ 写 `yoloe/detections.json`；

再 `model_service.infer_sam31()`（:308）→ worker `--stage sam31` → `src/inference/sam31_segment.py::Sam31Segmenter`（:274 有 `infer`）→ 写 `sam31/predictions.json`。

其 `_validate()` 分支额外断言：`yoloe.native_masks_forwarded is False`、`sam31.yoloe_masks_consumed is False`、且两边 `instance_count` 相等。这些断言的作用是**防止 YOLOE 自带的分割头泄漏进 SAM3.1**，否则两阶段对比失去意义。

### 11.5 两个容易踩错的点

1. **在线路径不做时序抽帧。** `src/data/sampling.py::uniform_sample_indices()`（即 "uniform-500" 的来源）**不在本链路任何一步上**。`gpt_labels.py:32` 的注释写得很直白：*"Online requests use the raw frame ID as their local cache key; no sampling is implied."* —— 你在页面上点第 300 帧，处理的就是 `.sens` 里的第 300 帧。采样只出现在 `src/inference/batch_infer.py` 那条批处理链路上。
2. **在线识别是逐帧独立的。** 每次 `pipeline.run()` 的 `contract.input` 都是 `"ScanNet RGB frame only"`，帧与帧之间没有任何状态传递。**跨帧物体身份不属于本服务的职责** —— 那正是 `src/harness/`（窗口式、可回滚的归属修订）存在的原因。不要把 `start.sh` 与 Harness 混为一谈。

> 说明：以上文件与函数、行号均逐一读过源码确认；但链路 A–C **未经实际启动服务运行验证**（起服务需加载模型、占用显存），运行时序（如"约 20 秒后卸载"）是从 `heartbeat_timeout=15.0` + `unload_grace=5.0` 的代码逻辑推得，非实测计时。

## 12. 未决事项 / 待办

- `src/` 根下 22 个 shim 的**去留**尚未决定：删除它们会破坏既有命令行用法与文档中的调用方式，需先确认无外部脚本依赖。
- **`src/geometry/cluster_prune.py` 尚未接入主流程**：目前没有任何生产调用方，只在离线评估中验证过（150 帧 / 980 实例，精确率 78.2%、召回率 80.2%）。下一步需决定接在 `build_independent_scene_source` 之后，还是接在 Harness 输入构建处。
- **`src/web/frontend/` 按决定不纳入版本控制**：已在 `InstanceRecognition/.gitignore` 显式忽略。25 个网页源码文件只存在于工作区，`git clean -fd` 会删除它们；其中 `runtime/web/` 迁移来的 19 个从未入库，无历史版本可恢复。
- `runtime/debug/mask-lifting-*` 是 2026-10-09 生成的一次性调试页，其生成器 `build_mask_lifting_page.py` 已随旧剪枝方案删除，**产物已无任何引用**（含 377 MB 的 `mask-lifting-data/`）。删除或归档待定。
- `Submodule/` 与 `InstanceRecognition/runtime/` 本轮未展开，待后续轮次。
- 其余顶层模块（`Video2PointCloud/`、`ScanNet/`、`ObjectGeneration/` 等）待后续轮次补充。
