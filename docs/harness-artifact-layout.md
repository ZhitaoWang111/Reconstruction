# Harness 实验产物布局

一份实验目录里放着**四种性质不同的东西**：复现依据（配置与哈希）、不可变版本栈、
原始调用日志、以及派生的页面。本文按**产出阶段**梳理每个文件由谁写、什么语义、
什么时候可信。

约定：`<out>` 指实验目录
`InstanceRecognition/runtime/experiments/<experiment-id>/`。
实验标识规则见 [`harness-experiment-naming.md`](harness-experiment-naming.md)。

---

## 0. 一图看全

```
<out>/
│
├─ ① 准备阶段 ───────────────────────────────────────
│   config.json  run_options/  code_versions/  run.lock
│   evidence_policy.json  observations.json  input_audit.json
│   evidence/*.jpg                (1559 张，实际看过的图)
│   window_inputs/*_context.json  (30 个，逐窗口输入快照)
│
├─ ② 窗口循环 ───────────────────────────────────────
│   versions/NNNN/{state,transaction,state,timing}.json + clouds/
│   checkpoint.json
│   calls/            (736 项：request/response/tool/meta/error)
│   network_recovery/ (仅发生网络故障时)
│
└─ ③ 发布阶段（全部窗口成功后一次性执行）────────────
    manifest.js  index.html  audit.html/.js  incremental.{js,css}
    states/NNNN.{json,npz,js}
    objects/object_NNNN.ply
    object_bboxes.js  result.css  result_data.js  viewer.js
    样式与前端：all500-layout.*  harness_*.js  styles.css
    visualization_cache/{snapshots,bboxes}/
    summary.json
```

三阶段的写入点：`src/harness/inputs.py` + `observations.py`（①）、
`run.py` + `publish.materialize`（②）、`publish.publish`（③）。

---

## ① 准备阶段 —— 确定"用什么输入、什么配置"

在窗口循环开始前完成。**这一阶段的产物是后续一切的可追溯依据。**

| 文件 | 写入者 | 语义 |
|---|---|---|
| `config.json` | `run.py:168` | 复现契约：`model` / `frames` / `window` / `voxel` / `timeout` / `prompt_sha256` / `manifest_sha256` / `geometry_policy` / `input_policy` |
| `run_options/<ns>.json` | `run.py:169` | 本次启动的**运维**设置（重试次数、建页策略）。与 `config.json` 分开，因为改重试参数允许断点续跑，改模型/输入则不允许 |
| `code_versions/<ns>.json` | `run.py:154` | 启动时 `src/harness/*.py` 的逐个 SHA-256 |
| `run.lock` | `run.py:153` | `flock` 独占锁，进程崩溃自动释放。**文件存在≠仍有进程持锁**，重启前用 `fuser`/`pgrep` 确认 |
| `evidence_policy.json` | `observations.py:40` | 证据格式版本（当前 v5）。**与 `observations.json` 是二选一的互斥检查**：有 policy 就比对它，有 observations 而无 policy 说明是旧格式目录，直接拒绝 |
| `observations.json` | `observations.py:84` | 全部观测的索引：观测 ID、类别、bbox、点云摘要、掩码路径 |
| `input_audit.json` | `observations.py:84` | 每个输入文件的 SHA-256。续跑时比对，**不一致即拒绝复用目录** |
| `evidence/*.jpg` | `observations.py:190` | 实际渲染并发送给模型的图，命名 `<seq>_<instance>.jpg` |
| `window_inputs/*_context.json` | `inputs.py:32` | 每窗口组装给模型的完整上下文快照，命名 `{首个帧序号:04d}_context.json` |

> **`run_options/` 在现有实验里不存在**（已核 150/500/30 帧三个目录）。不是缺陷：
> 该写入语句晚于那几轮运行才加入，`save()` 会自动建目录，所以缺目录只表示
> "这轮跑的时候还没有这个功能"。**新运行的实验才会有。**

**观测 ID 格式 `<seq:04d>:<instance_id:03d>`**（如 `0004:001`）——`seq` 是抽样序号，
不是源帧号。这个 ID 在后续所有记录里都是主键。

---

## ② 窗口循环 —— 不可变版本栈

每 5 帧一个窗口，模型审阅后提交一个版本。**核心原则：已提交的版本永不修改。**

```
versions/0001/
├── state.json         # 提交后的完整状态（objects / assignments / events / pending）
├── transaction.json   # 触发它的修改单：id + proposal + affected
├── timing.json        # 本版本耗时
└── clouds/            # 受影响物体的重算体素点云 Object*.npz（仅变化的）
```

`checkpoint.json` 指向**最后一个成功提交**的版本号。崩溃后续跑从这里恢复。

**`state.json` 的数据模型**（这是全实验的核心）：

```json
{
  "version": 1, "cutoff": 4,
  "objects":     [{"id":1, "object_label":"office chair", "frames":[0,1,2,3],
                   "voxel_count":1635, "status":"multi_frame",
                   "observations":["0000:001","0001:003", ...]}],
  "assignments": [{"id":"0004:001", "instance_id":1, "object_id":2,
                   "status":"assigned", "reason":"Same desk across viewpoints.",
                   "evidence":["0000:002", ...]}],
  "events":      [{"action":"create", "object_id":"Object1",
                   "object_label":"office chair", "evidence":[...], "reason":"..."}],
  "pending": ..., "conflict_entries": ..., "cutoff": 4, "version": 1
}
```

三个语义要点：

1. `observations` 是**不可变**的；归属变了就改 `assignments`，不改观测本身。
2. `assignments[].status` 可以是 `assigned` / `defer` / `repair`。**暂缓（defer）表示
   "没看到，记为未知"，不是"不存在"**——这条是 Harness 的判定边界。
3. `reason` 与 `object_label` 用**英文**。中文是后处理翻译，不回写原始日志。

### 调用日志 `calls/`

命名 `w{窗口:04d}_{调用ID}_call{序号:02d}_{类型}.json`：

| 类型 | 内容 |
|---|---|
| `request` | 发往网关的完整请求（含脱敏后的工具 schema） |
| `response` | 模型回答 |
| `toolNN` | 第 NN 次工具调用的结果 |
| `meta` | 用量与耗时——`summary.json` 的 token 统计从这里汇总 |
| `error` | 错误类别、HTTP 状态、耗时、等待时长 |

**注意**：`request` 里每次都携带完整的 6 个工具 schema，所以数 schema 出现次数**不等于**
调用次数。要统计调用次数必须读 `response` 里的 `tool_calls`。本轮实测：270 次调用
（`inspect_observations` 110、`inspect_object` 73、`propose_update` 56、`commit_update` 30、
`compare_geometry` 1、`list_objects` 0）。

`network_recovery/` 只在发生网络故障时出现（`run.py:33`），记录恢复批次的尝试编号；
正常跑完的实验里没有这个目录。以下计数取自 `harness-v2-gpt6-sol-150frames-30windows-20260929`
实测，用于量级参考：

| 目录 | 数量 |
|---|---:|
| `versions/` | 30 |
| `window_inputs/` | 30 |
| `states/` | 90（30 版本 × 3 扩展名） |
| `objects/` | 72 |
| `calls/` | 736 |
| `evidence/` | 1559 |
| `code_versions/` | 8（每次启动一个） |

### 中途终止的实验（`harness-v2-gpt6-sol-50frames-10windows-20261013`）

在第 10 个窗口提交后终止，实测计数：

| 目录 | 数量 |
|---|---:|
| `versions/` | 10 |
| `window_inputs/` | 10 |
| `states/` | 30 |
| `objects/` | 42 |
| `calls/` | 482 |
| `evidence/` | 586（523 观测图 + 50 帧图 + 13 `history_bbox_*`） |
| `code_versions/` | 3（首个进程 + 两次续跑） |
| `run_options/` | 3 |

工具调用实测 120 次：`inspect_observations` 70、`inspect_object` 25、`propose_update` 15、
`commit_update` 10。**其中 18 次属于被放弃的窗口尝试**（`w0020` 与 `w0045` 各失败一次后以新
`run_tag` 重跑），只有 102 次进入了提交结果。`summary.json` 的
`api_calls_in_committed_windows` / `api_calls_in_abandoned_attempts` 记录这个拆分，
`retried_windows` 列出发生重跑的窗口。

改名到 50 帧 / 10 窗口时同步截断了 `input_audit.json`、`observations.json` 与 `evidence/`，
并删除了未提交第 11 窗口的 `window_inputs/0050_context.json` 与 `calls/w0050_*` 残件；
`evidence/` 中 `history_bbox_*` 按内容哈希命名，需按日志引用判定归属而非按文件名前缀。

---

## ③ 发布阶段 —— 全部窗口成功后执行一次

**窗口运行期间不生成网页**（`run.py:210`）。这样中途失败时已有页面不会刷新成半成品。
用 `--skip-publish` 可推迟到之后用 `--publish-only` 单独建页。

### 页面与脚本

| 文件 | 语义 |
|---|---|
| `index.html` | 三维结果页：RGB/overlay 切换、场景点云、Object 列表、bbox 显示开关 |
| `audit.html` + `audit.js` | 基础审计页：全部改动、原因与图像证据 |
| `incremental.js` / `incremental.css` | 结果页交互逻辑 |
| `manifest.js` | `window.MASK_PROJECTION_MANIFEST`，**重定基**到 `<out>` 的相对路径 |
| `styles.css` / `result.css` / `result_data.js` / `viewer.js` | 从 `src/web/frontend/` 复制的模板 |

> 页面模板**不在**本目录的源码里——它们是 `src/web/frontend/` 下的手写前端，
> 由 `build_incremental_viewer.build()` 复制过来。样式与脚本均**只读**，不要在此处修改，
> 否则下次导出会丢失修复。

### 派生数据

| 路径 | 语义 |
|---|---|
| `states/NNNN.{json,npz,js}` | 逐版本的**回放格式**。`json` 给审计、`npz` 给点云、`js` 给页面 `registerIncrementalState()` 注册 |
| `objects/object_NNNN.ply` | 最终候选物体的点云，**仅在最后一个版本导出**（见 `publish.py:104`） |
| `object_bboxes.js` | 观测支撑包围盒的导出（`object_bboxes.py:560`） |
| `visualization_cache/snapshots/NNNN.json` | 快照缓存，命中即跳过重算——**这是可安全删除的纯性能缓存** |
| `visualization_cache/bboxes/NNNN.json` | 同上，bbox 侧 |
| `summary.json` | 完成状态与统计。`completed` / `frames` / `windows` / `objects` / `observations` / `pending` / `api_successful_calls` / `usage` |

**`summary.json` 里两个不能误读的字段**：

- `accuracy_evaluated: false` —— 统计量**不代表识别准确率**，只说明运行状态。
  150 帧那轮是 72 个 Object / 1361 条观测 / 88 条 pending，但这**不是**精度指标。
- `cost_currency: null` + `cost_note` —— 网关价格未配置，只记 token，**不编造成本**。

---

## ④ 实验目录内的页面产物

| 内容 | 位置 | 说明 |
|---|---|---|
| 逐窗口交互记录页 | `<id>/records.html` | 与结果页同级；跨实验导航由 JS 在加载时按目录名生成 |
| 记录页资源 | `<id>/records/` | `records.js`、`window-*.js`、翻译 sidecar 等 |
| 翻译后处理 | 同上目录内的哈希键 JSON | `translate_harness_reasons.py` 产出；**不回写** `calls/` |

结果页与记录页**都在实验目录内**。此前记录页放在 `docs/` 是为了跨实验集中导航，但导航实际由 `docs/assets/experiment-navigation.js` 按目录名动态生成，并不依赖记录页位于 `docs/`；拆开放反而让一个实验的产物散在两处。唯一留在 `docs/` 的是跨实验共享的 `experiment-navigation.js/.css` 与源码模板。

---

## ⑤ 什么可以安全删除

| 路径 | 可删？ | 理由 |
|---|---|---|
| `visualization_cache/` | ✅ | 纯性能缓存，删掉只是下次重建变慢 |
| `evidence/` | ⚠️ | 页面会立刻坏；但 `calls/` 里的原始响应仍在 |
| `states/`、`objects/`、`object_bboxes.js` | ⚠️ | 派生产物，可由 `versions/` 用 `--publish-only` 重建 |
| `versions/`、`checkpoint.json`、`calls/` | ❌ | **原始记录，删了就不可恢复** |
| `config.json`、`input_audit.json`、`code_versions/` | ❌ | 复现与追溯的唯一依据 |

---

## ⑥ 相关文档

- 实验标识命名规则：[`harness-experiment-naming.md`](harness-experiment-naming.md)
- 执行流程与工具说明：`InstanceRecognition/src/harness/README.md`
- 输入契约（点云 npz 与相机 JSON）：`InstanceRecognition/src/geometry/SCENE_PROJECTION.md`
- 逐文件梳理：仓库根 `Code_Review.md` 第 6 节
