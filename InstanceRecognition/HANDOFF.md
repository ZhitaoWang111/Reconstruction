# 交接：实例识别的 harness 框架

写于 2026-10-19。上一轮窗口完成了 150 帧实验，并重写了流水线说明页。
本文只回答一件事：**接下来要动 harness 框架，该读什么、从哪看起、哪些是坑。**

---

## 一、先读这三份，按顺序

| 顺序 | 文件 | 为什么 |
|---|---|---|
| 1 | `docs/pipeline.html` | **全局图景**。三个模块（SAM/GPT 出 mask → mask 投影到 3D → harness 框架）、每步的输入输出、目录结构、契约表、运行命令。浏览器直接打开，`file://` 可用。 |
| 2 | `README.md` | 四类目录的职责划分、环境、运行命令、两个页面的定位。 |
| 3 | `DEVNOTES.md` | 踩过的坑。**第 1、4、8 节必读**：AMB3R 溯源、环境陷阱、独立 SAM 环境。 |

`docs/pipeline.html` 是唯一一份描述**当前**流水线的文档。仓库根目录下的
`docs/scene-reconstruction-design.html` 与 `-v2.html` 是 9 月的历史设计稿，
只描述 harness 单模块，且已有多处与代码不符（重试次数、reason 语言、代表图策略、
`mask_id` vs `instance_id`、行号全漂移）。**只在查历史意图时参考，不要当规格。**

---

## 二、代码地图

```text
InstanceRecognition/
├── src/harness/          1294 行，本次任务的战场
│   ├── run.py            CLI、窗口循环、续跑、lift policy 白名单
│   ├── inputs.py         组装一个窗口的模型输入（35 行，密度极高）
│   ├── observations.py   读 npz、拆 mask、出 overlay、写 input_audit
│   ├── spatial_context.py 全局 OBB + 最近 3 个候选物体
│   ├── tools.py          六个工具的实装、representative 缓存
│   ├── state.py          纯函数事务校验（create/assign/relabel/defer/repair/merge）
│   ├── client.py         网关调用、重试、图片编码
│   ├── history.py        代表观测选择（清晰度打分）、历史 bbox 图
│   ├── publish.py        页面发布
│   └── prompt.txt        系统提示词
├── src/geometry/         952 行，上游
│   ├── assemble_perception_source.py  装配 + 三道哈希校验
│   ├── amb3r_mask_source.py           投影到 3D（唯一合法的 lift policy）
│   ├── backproject_masks.py           纯几何 helper
│   └── incremental_objects.py         voxelize / 增量点云
├── src/inference/        547 行，更上游（GPT + SAM3.1）
├── pagegen/              886 行，页面生成
└── src/visualization/    733 行
```

阅读顺序建议：`run.py` → `inputs.py` → `state.py` → `tools.py` → `observations.py`。
`inputs.py` 只有 35 行但决定了模型看到的一切，值得逐字读。

---

## 三、最新一轮实验：150 帧

**这是当前唯一有效的端到端实验，harness 相关的所有判断都应以它为准。**

```text
run     artifacts/experiments/harness-v5-amb3r-150frames-20261016-1550/
页面    site/experiments/harness-v5-amb3r-150frames-20261016-1550/index.html
```

| 项 | 值 |
|---|---|
| 帧数 / 窗口 | 150 帧 / 30 窗口（window=5，帧 0–149） |
| 状态 | `completed: true`，30 窗口全部提交，无重试窗口 |
| 物体数 | **72** |
| 观测数 | 1397（150 帧 × 约 9.3 条/帧） |
| `pending` | **55**（约 3.9%，是待定观测，不是错误） |
| API 调用 | **213** 次（上限 14/窗口，最忙的窗口用了 13 次） |
| token | prompt 15,512,477 + completion 118,006 = **15,630,483** |
| 耗时 | 约 2.9 小时 |
| 模型 | `openai/gpt-6-sol`，`reasoning_effort: none` |
| 几何策略 | `amb3r_per_frame_v1` |
| 输入策略 | `history_on_demand_rgb_bbox_v2_spatial_top3_global_obb` |
| source | `artifacts/sources/scene0000_00/scene_amb3r_v1` |

手工复现：

```bash
cd /home/seer/wzt/Reconstruction/InstanceRecognition
taskset -c 0 .venv/bin/python -m src.harness.run \
  --source artifacts/sources/scene0000_00/scene_amb3r_v1 \
  --frames 150 --window 5 --max-calls 14 --voxel .02
```

`taskset -c 0` 不是可选项。凡触碰 torch 的调用（SAM 阶段）不加它会 segfault；
harness 本身不需要 torch，但统一加上更省心。**长实验由用户自己启动，不要代跑。**

### 这个 run 里值得先看的东西

1. `summary.json` — 上面那张表的原始来源，含 token 与失败窗口统计。
2. `checkpoint.json` — `{"version": 30, "cutoff": 149}` 续跑锚点。
3. `versions/0001…0030/` — 每个窗口提交后的完整状态快照。**要看模型怎么一步步建出 72 个物体，从 `versions/0001/state.json` 顺着读。**
4. `calls/` — 每次 API 调用的 request / response / meta / toolNN 记录。调 prompt 时最常翻这里。
5. `window_inputs/0000_context.json` … — 每窗口实际发出去的输入副本，含 `historical_representatives` 审计。
6. `input_audit.json` — 输入文件哈希。**变了就拒绝续跑**，这是刻意的。
7. 页面 `audit.html` — 窗口归属与纠错；`index.html` — 三维回放；`records.html` — Agent 交互记录。

### 另外两个 run 是历史产物

`harness-v3-sol-20frames-*` 与 `harness-v4-mesh-sol-20frames-*` 各 20 帧 / 4 窗口，
用的是**已删除**的 lift policy（`cluster_pruned_v1` / `mesh_first_surface_v1`）。
页面还能打开，但 source 已无法由本树重建 —— `run.py` 的白名单会直接拒绝。
不要把这两个的数字和 v5 混着比较。

---

## 四、几个必须知道的约定

1. **`instance_id` 是帧内编号**，不是物体身份。跨帧身份只在 harness 里产生（`Object1`、`Object2`…）。
   代码里有一个 `_strip_legacy_observation_keys` 会全局清掉旧的 `mask_id` 拼写，写新产物时别绕开它。
2. **`mask_bits` 是 uint32 位图**，不是单个编号。两个 mask 重叠时一位点会同时有两位为 1。
   换成单编号会丢归属歧义。
3. **`lifting_policy` 是硬闸门**。装配产物故意不带这个字段，只有投影器会写。
   缺它 harness 直接拒绝启动 —— 这是为了防止有人跳过投影、让模型读到没有三维证据的观测。
4. **`pending` 不是错误**。`defer` / `repair` 是观测状态，占几个百分点是正常的。
5. **图片数只与帧数有关**：每帧恒定 2 张（原始 RGB + 全帧 overlay），窗口 5 帧 = 10 张。
   历史变长只增加文字档案，不发图；要图必须模型主动调 `inspect_object`。
6. **`selection`（选帧指标）不进模型输入**，只落本地审计（`public_representative()` 里剥掉了）。
7. **两个解释器**：轻量 `.venv`（numpy/pytest/PIL，no torch）跑 harness/几何/页面/测试；
   conda `scene-understanding`（torch 2.10 + sam3）只在 SAM 阶段用，见 DEVNOTES §8。

---

## 五、当前测试与状态

```bash
cd /home/seer/wzt/Reconstruction/InstanceRecognition
taskset -c 0 .venv/bin/python -m pytest tests/ -q     # 205 passed
```

20 个测试文件，覆盖 harness 决策循环、几何、页面构建。

已知的不一致（都是小问题，别被绊住）：

- `README.md` 第 182 行左右写的仍是 `199 passed`，实际 205。
- 工作区有未提交改动；`git status` 里会出现 `D InstanceRecognition/ThirdParty/...`
  这类路径已经不存在的删除记录（旧树清理的残留）。
- `ObjectGeneration/` 仍硬编码 `InstanceRecognition/.venv` 与 `ThirdParty/SAM3`。
  用户明确说过「暂时不管」。

---

## 六、如果要动 harness，先确认这几件事

- **不要重跑 150 帧实验**。213 次调用、1560 万 token、近 3 小时。改完先用
  `--prepare-only` 或 `--frames 4 --window 2` 验证路径与输入形状。
- **`--publish-only` 只重建页面**，不碰模型，改页面样式时用它。
- 改 `prompt.txt` 会被记进 `prompt_sha256`；续跑时会校验，变了就得新开 run 目录。
- 加窗口/改窗口大小会改变模型看到的历史，属于实验变量，不是纯实现改动。

---

## 七、安全与凭据

网关密钥在 `/home/seer/wzt/Gateway/key.json`，环境变量文件 `/home/seer/wzt/API/.env`。
`client.py` 对落盘的响应做 `redacted(payload, key)`，**新增任何写响应的地方都要过这个函数**。
探测凭据时只打印长度，不打内容。
