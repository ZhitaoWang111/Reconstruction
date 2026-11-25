# Reconstruction / InstanceRecognition 交接说明

本文件供新对话窗口接手本项目使用。**先读本文件，再按第 6 节顺序核对源码。**

文档描述的是设计意图；实际行为必须以当前 checkout 的代码、配置和运行产物为准。发现不一致时，先报告差异，不要自行改写历史日志或产物。

---

## 1. 项目在做什么

从 RGB 视频重建可交互的 SimReady 场景。仓库拆成几个子模块：场景理解（`InstanceRecognition/`）、物体生成（`ObjectGeneration/`）、视频到点云（`Video2PointCloud/`）。本文件聚焦 `InstanceRecognition/` 的 **Harness**：让模型逐窗口审阅已有的 mask 观测，跨帧决定每个观测属于哪个物体。

Harness 不重新跑分割。它的输入是上游（GPT-5.6 Terra）在抽样帧上产出的 mask，加上已反投影到世界坐标的点云。

---

## 2. 当前状态（2026-10-13）

已跑通 5 轮 Harness 实验，结果页与交互记录页都在实验目录内：

| 实验标识 | 帧 / 窗口 | 物体 | 观测 | 点云 source |
|---|---|---|---|---|
| `harness-v1-gpt56-terra-30frames-20260921` | 30 / 6 | 27 | 188 | `amb3r_gt_raw` |
| `harness-v2-gpt6-sol-30frames-20260924` | 30 / 6 | 33 | 300 | `amb3r_gt_raw` |
| `harness-v2-gpt6-sol-500frames-20260924` | 500 / 100 | 115 | 4481 | `amb3r_gt_raw` |
| `harness-v2-gpt6-sol-150frames-30windows-20260929` | 150 / 30 | 72 | 1361 | `scene_cloud_visibility_v1` |
| `harness-v2-gpt6-sol-50frames-10windows-20261013` | 50 / 10 | 42 | 523 | `scene_cloud_visibility_v1_pruned` |

模型：`openai/gpt-6-sol`，`reasoning_effort: none`，`window: 5`，`timeout: 300s`。

最近一轮（2026-10-13）用**剪枝后的 mask 成员关系**（`cluster_pruned_v1`），目的是移除遮挡泄漏。它按用户决定在第 10 个窗口后终止，目录名与 `config.json` 已同步为实际的 50 帧 / 10 窗口范围。

测试基线：`cd InstanceRecognition && taskset -c 0 .venv/bin/python -m pytest tests/ -q` → **206 passed**。

`taskset -c 0` 不可省略：不加它会在依赖导入阶段崩溃（SIGSEGV/SIGILL，实测复现过）。

### 页面位置

每个实验的两个页面在**同一个实验目录内**，互为同级：

```text
InstanceRecognition/runtime/experiments/<实验标识>/
├── index.html      场景理解结果（窗口末帧 + 累计物体点云 + 3D bbox）
├── records.html    逐窗口 Agent 交互记录（请求、响应、工具调用、理由中译）
└── records/        记录页资源（含中文译文 sidecar）
```

两页顶部都有实验切换栏。打开方式见 [`web_command.md`](../web_command.md) 1.6 节。

---

## 3. 关键设计约定

写下来是因为它们反直觉，容易在改动中破坏。

- **模型看不到决策时点之后的数据。** `Tools` 以当前窗口的 cutoff 为界（`tools.py`）。这是防止未来信息泄漏的核心设计，不要为了"提供更多上下文"而放宽。
- **只有 `commit_update` 才写入新版本。** `propose_update` 只产生待提交的 proposal；失败的 proposal 不应修改已提交状态。
- **当前观测用整帧 RGB + mask overlay**，overlay 里的编号是帧内 `instance_id`，颜色只区分当前帧，**不代表跨帧身份**。
- **历史 Object 不随初始请求发送图像**，要靠 `inspect_object` 按需取。初始只给物体档案和全局 3D bbox。
- **`reason` 用英文**，不要求模型输出中文；中译是独立后处理，不改写原始日志。
- **剪枝只清位，不删行。** `cluster_prune` 清除 `mask_bits` 的位，行数、行序、`scene_point_ids` 全部保持，原值存在 `raw_mask_bits`。成员判据是 `(bits & (1<<i)) != 0`，所以位清掉即生效。
- **`--source` 的 `lifting_policy` 是精确匹配的**（`run.ACCEPTED_LIFTING_POLICIES`）。放宽这个集合必须是有意为之——它正是防止 `--source` 误指到别的几何的护栏。
- **`gpt-6-astra` 与 `gpt-6-sol` 的协议不同**：Sol 走非流式 Chat Completions（`stream: false`），Astra 走 Responses 分支且要求至少 `low` reasoning。不要把 Astra 的参数套到 Sol 上。
- **`compare_pixels` 不存在。** 见第 5 节的工具清单；其他材料若列出该名称，视为过期描述。

---

## 4. 代码结构和调用链

### Harness 主体

```text
InstanceRecognition/src/harness/
├── run.py                 CLI、窗口循环、checkpoint、网络恢复、完成后发布
├── client.py              网关调用、协议选择、超时与网络重试、日志脱敏
├── inputs.py              当前窗口与历史 Object 的输入组装
├── observations.py        观测、图像、bbox 与几何摘要
├── prompt.txt             当前英文系统 prompt（唯一生效的那份）
├── prompt.zh-CN.txt       说明与旧版参考，不是发给模型的 prompt
├── tools.py               工具 schema 与调用
├── state.py               proposal 校验、版本提交、状态变更
├── publish.py             状态物化、点云生成、基础结果页发布
├── naming-conventions.txt 观测与 Object 命名规则
└── README.md              实现说明与启动命令
```

### 几何：从 mask 到点云

```text
InstanceRecognition/src/geometry/
├── SCENE_PROJECTION.md                 接口约定，必读
├── scene_projection.py                 可见性分支
├── backproject_masks.py                掩码反投影（全仓最大的几何文件）
├── build_independent_scene_source.py   构建独立场景 source
├── filter_lifted_masks.py              派生"过滤后"的 source
├── cluster_prune.py                    对称分簇剪枝判据
├── prune_scene_source.py               把剪枝落成派生 source（生产链路）
├── leakage_eval.py                     真值深度的泄露标注口径（仅用于评估）
└── incremental_objects.py              不可变观测的保守增量关联
```

几何来源已经换了两次，读代码时注意区分：当前 Harness 用**独立完整场景点云 + RGB 相机可见性投影**，不再以 AMB3R 逐像素反投影点图作为主要来源。

### 可视化与页面生成

```text
InstanceRecognition/src/visualization/
├── build_harness_result.py   结果页（注意它会整页覆写 index.html）
├── build_harness_records.py  交互记录页
├── page_naming.py            页面与资源命名的单一来源
├── object_bboxes.py          3D bbox 拟合
└── translate_harness_reasons.py  reason 中译（调用翻译模型）

InstanceRecognition/src/web/frontend/   全部手写网页源码
```

`build_harness_result.py` 在 `publish()` 之后**整页覆写 `index.html`**，因此几何披露必须在那里重新施加，否则页面会把剪枝后的源描述成普通可见性投影。

### 六个工具

以 `tools.py` 的声明为准，并与运行 request payload 的 tools 字段核对：

```text
list_objects  inspect_observations  inspect_object
compare_geometry  propose_update  commit_update
```

**统计调用次数不能数 request 里的 schema**：每次请求都携带完整 6 个工具声明。要统计实际调用，读 response 里的 `tool_calls`。

---

## 5. 运行和建页

### 新跑一轮

```bash
cd /home/seer/wzt/Reconstruction/InstanceRecognition
taskset -c 0 .venv/bin/python -u -m src.harness.run \
  --source runtime/mask_pointclouds/scene0000_00/scene_cloud_visibility_v1_pruned \
  --frames 50 --window 5 \
  --output runtime/experiments/harness-v2-gpt6-sol-50frames-10windows-YYYYMMDD \
  --model openai/gpt-6-sol --reasoning-effort none \
  --timeout 300 --network-retries 5 --window-network-retries 2
```

- `--output` 必须显式指定且目录不存在（新实验不覆盖旧结果）。
- 启动前检查持锁进程与同名目录；`run.lock` 是锁文件本身，**不代表有进程持锁**，用 `fuser`/`pgrep` 确认，不要直接删锁。
- 关键配置写进 `config.json` 后即参与断点续跑的比对，改语义配置需要换输出目录。
- 命名规则见 `docs/harness-experiment-naming.md`。

### 只重建页面（不调用模型）

```bash
# 结果页
taskset -c 0 .venv/bin/python -u -m src.harness.run --publish-only \
  --output runtime/experiments/<实验标识>

# 交互记录页
cd /home/seer/wzt/Reconstruction
InstanceRecognition/.venv/bin/python InstanceRecognition/src/visualization/build_harness_records.py \
  --run <实验标识>
```

翻译 reason 是独立后处理（会调用翻译模型），译文是 sidecar，不改写英文原文；翻译后需再次运行 `build_harness_records.py` 才生效。

---

## 6. 给新窗口的建议工作顺序

**源码核对是必经步骤**，不能只读文档就下结论。

1. **先读代码，再读文档。** 至少沿调用链检查：

   ```text
   InstanceRecognition/src/harness/{run,client,inputs,observations,tools,state,publish}.py
   InstanceRecognition/src/harness/prompt.txt
   InstanceRecognition/src/geometry/{scene_projection,prune_scene_source,cluster_prune}.py
   InstanceRecognition/src/visualization/{build_harness_result,build_harness_records}.py
   ```

   重点：输入如何组装、请求走哪个协议、工具结果如何回填、proposal 如何提交、点云如何生成、页面从哪些 JS 数据构建。

2. **再读文档**：本文件、`src/harness/README.md`、`docs/harness-experiment-naming.md`、`docs/harness-artifact-layout.md`、`src/geometry/SCENE_PROJECTION.md`、`docs/scene-reconstruction-design-v2.html`。

3. **检查任务是否已完成**：读运行目录的 `config.json`、`summary.json`、`checkpoint.json`，不要先重跑。

4. **汇报要区分**"已实现""已测试""已运行验证""仅准备未验证"，并给出源码路径或产物依据。

5. **不要**因为看到旧实验目录或 `run.lock` 就自动重跑、删除或清理结果。

---

## 7. 相关文档

| 文档 | 内容 |
|---|---|
| [`web_command.md`](../web_command.md) | 全部页面入口与打开方式 |
| [`docs/harness-experiment-naming.md`](harness-experiment-naming.md) | 实验标识与页面命名规则 |
| [`docs/harness-artifact-layout.md`](harness-artifact-layout.md) | 各阶段产物落在哪里 |
| [`docs/scannet-data.md`](scannet-data.md) | 本机 ScanNet 数据现状 |
| [`docs/cloud-upload-checklist.md`](cloud-upload-checklist.md) | 尚未上传的数据与权重清单 |
| [`Code_Review.md`](../Code_Review.md) | 全仓代码走读 |
| [`docs/scene-reconstruction-discussions.html`](scene-reconstruction-discussions.html) | 方案讨论与问题复盘（含失败案例） |
