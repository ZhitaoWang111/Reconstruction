# 交接：场景理解（InstanceRecognition）

写于 2026-11-04。上一轮窗口把 harness 的 prompt 从 P3 迭代到 P6，跑完了一次 500 帧的全量实验，
并顺带做了交给下游资产生成模块的交接文件。

本文只回答一件事：**接手这个任务，该读什么、代码在哪、哪些坑已经踩过。**
不包含工作目标——那由提出需求的人定。

---

## 一、这个任务在做什么

把一段室内扫描（ScanNet 的 `scene0000_00`，一套公寓，5578 帧原始画面）变成**一份物体清单**：
房里有哪几件东西、每件东西占据哪些三维点、在哪几帧里出现过。

难点不在识别单个物体，而在**跨时间把同一个物体的观测连起来**。同一条 mask 在相邻帧里
长得几乎一样，但相隔 100 帧、隔着半个房间，判断「这还是刚才那把椅子吗」就变成了一次
没有答案的推理。整条流水线是为了让一个 LLM 逐窗口做这个判断。

目标读者：一个接手改这套系统的人（或 agent）。

---

## 二、四段流水线

```
① GPT 标注 + SAM3.1 分割      → 每帧若干 mask，带类别猜测
② assemble_perception_source  → 把两边的产物对齐到一份 manifest
③ amb3r_mask_source           → 每个 mask 的像素反投影成 3D 点
④ harness                     → 逐窗口把观测归到持久物体上
```

本文关心的是 **④**，也就是 `src/harness/`。前三段已经稳定，改动前先读 `docs/pipeline.html`。

**④ 的循环**：每 5 帧一个窗口 → 把这一窗的观测、图片、归档清单发给模型 → 模型可以调用
若干工具取证 → 提交一份完整提案（每条观测归给某个物体 / 新建 / 搁置 / 排除）→ 宿主校验、
写盘、开下一个窗口。

---

## 三、先读这四份，按顺序

| 顺序 | 文件 | 为什么 |
|---|---|---|
| 1 | `docs/pipeline.html` | **全局图景**。四段流水线、每步的输入输出、目录结构、契约。浏览器打开，`file://` 可用。篇幅大，先看 harness 那几节。 |
| 2 | `README.md` | 四类目录的职责、环境、运行命令、两个页面的定位。 |
| 3 | `DEVNOTES.md` | 踩过的坑。**第 10、11 节是这一轮新增的**，第 1、4、8 节是环境类必读。 |
| 4 | `docs/prompts.html` | prompt 的七个版本全文（英中对照）+ 每一版为什么改。**第 07 节是现在的版本**。 |

`docs/prompts.html` 的第 01–06 节是历史，记的是 P3 的问题和改法，
读它有助于理解现在的写法为什么是这样，但**不要当成当前规格**——当前规格只有
`src/harness/prompt.txt` 本身。

---

## 四、代码地图

```text
InstanceRecognition/
├── src/harness/              1608 行，主战场
│   ├── run.py          314   CLI、窗口循环、续跑、预算警告、提交写盘
│   ├── client.py       340   网关调用、重试、响应落盘（含 redacted）
│   ├── state.py        208   纯函数校验器：提案 → 下一状态，所有规则都在这里
│   ├── observations.py 206   读 npz、拆 mask、画 overlay、写 input_audit
│   ├── tools.py        163   五个工具的实现，含 reid 门禁的接入
│   ├── reid_gate.py    128   跨窗再识别的软门禁（按距离）
│   ├── history.py      104   代表观测的选取与画框
│   ├── paths.py         42   相对/绝对路径解析
│   ├── spatial_context.py 42 每帧给 Top-3 的邻近物体提示
│   ├── inputs.py        35   组装一个窗口的模型输入（行数少，密度高）
│   └── publish.py       25   把状态落成页面数据
├── src/handoff/              990 行，给下游的交接文件（另一个任务，先不用管）
├── tests/                    23 个文件，235 项测试
└── artifacts/
    ├── sources/scene0000_00/ 三个 source：amb3r（在用）、mesh_first_surface（旧）、gpt_sam31（上游）
    └── experiments/          每次实验一个目录
```

**建议的阅读顺序**：`run.py` 的 `process_window` → `state.py` 的 `propose` → `tools.py` →
`inputs.py`。这四个决定了模型看到什么、能做什么、什么被拒绝。

`state.py` 是**唯一的规则事实源**。哪些字段被允许、什么被拒绝、拒绝时说什么话，全在这里。
prompt 是对它的自然语言描述；两者不一致时，代码是对的。

---

## 五、五个概念，必须分清

**1. 三个层级，别混。**

| 层级 | 是什么 | 例子 |
|---|---|---|
| 窗口 | 一次提交单元，5 帧，提交即结束 | 窗口 12 = 帧 55–59 |
| 工具 | 模型调用什么 | `inspect_object`、`propose_update` |
| 动作 | 提案里的条目 | `assign`、`create`、`defer`、`exclude`、`merge`、`relabel` |

窗口编号 = 帧号 ÷ 5 + 1（帧 55–59 是窗口 12），和日志前缀 `w0055_` 不是一回事。

**2. 六种动作分两类。**

作用于观测（`assign` / `defer` / `exclude`，带 `observations` 列表）——
每条可见观测必须恰好落在这三者之一，否则整个提案被拒。

作用于物体（`create` / `merge` / `relabel`，不带 `observations`）——
`create` **不是第四种状态**，它建物体但不记录任何观测；促成它的那条观测还得另起一条 `assign`。

**3. 五个工具，按副作用分档。**

```
取证  inspect_observations / inspect_object / compare_geometry   只读，且登记可引用证据
校验  propose_update                                             只改内存
提交  commit_update                                              唯一写盘
```

`list_objects` 已于 2026-11 删除（返回的就是窗口首条消息里已有的内容）。

**4. 观测 id 与证据。**

id 是 `seq:instance`，比如 `0045:001`。**窗口自己的观测 id 天然可引用，归档观测的 id 必须
先 `inspect_object` 才算看过。** 这条规则绊倒过很多次：归档清单里已经不列任何 id 了，
但仍然有模型按格式编 id。

**5. `pending` 不是错误。**

`defer` 和 `exclude` 都算 pending（`object_id is None`）。占几个百分点是正常的。
但这个列表**只增不减**，见 DEVNOTES 第 11 节——它目前是上下文膨胀的主因。

---

## 六、prompt 的现状

当前 `prompt.txt` 是 P6，12,041 字符，sha256 前缀 `0e950a5`，六节：

```
WHAT THE WINDOW CONTAINS               窗口里有什么
ONE COMMIT PER WINDOW                  每个窗口只提交一次
ACTIONS ON OBSERVATIONS AND ON OBJECTS 六种动作，按作用对象分两类
HOW TO GET MORE INFORMATION            如何获取更多证据
SUBMITTING A PROPOSAL                  提交提案
ORDER OF WORK                          工作顺序
```

历史版本都留档在 `src/harness/prompt-*.txt`，哈希见 `docs/prompts.html` 第 00 节。

**改 prompt 必须开新 run 目录。** `run.py` 把 `prompt_sha256` 写进 `config.json`，
续跑时逐字段比对，不一致直接拒绝。这是为了防止前 12 个窗口用旧 prompt、后 18 个用新 prompt，
事后无法区分。

---

## 七、实验记录

| run | 帧 | prompt | 结果 |
|---|---|---|---|
| v5 | 150 | P3 | 基底，后来的对照都基于它 |
| v6 | 150 | P3 | 完整跑完，defer 535 条零回收 |
| v7 | 150 | P4 | 两次重启（`"type"` vs `"action"` 错误、524） |
| v8 | 150 | P5 | 窗口 12 超预算失败 |
| v9 | 150 | P5 | 跑完，62 物体，错误率 51% |
| v10 | 500 | P6 | 窗口 10 超预算失败（预算警告措辞问题） |
| **v11** | **500** | **P6** | **跑完，114 物体，4632 观测，零作废窗口** |

**v11 是当前最好的基线。** 目录：`artifacts/experiments/harness-v11-amb3r-500frames-20261026-1400/`。

每个 run 目录里值得看的：

```
config.json          参数与 prompt 哈希
checkpoint.json      续跑锚点
versions/NNNN/       每个窗口提交后的完整状态 + 点云
calls/               每次 API 调用的 request / response / tool 记录
window_inputs/       每个窗口实际发出去的输入副本（含代表观测的选取依据）
summary.json         统计
```

**`calls/*_tool*.json` 是调 prompt 时最常翻的地方**——每次工具调用的参数和返回都在里面，
错误率高的时候按文件名分类统计就能看出模式。

---

## 八、页面

| 页面 | 路径 | 看什么 |
|---|---|---|
| 流水线说明 | `docs/pipeline.html` | 全局图景 |
| prompt 版本 | `docs/prompts.html` | 每一版的全文与改动理由 |
| 结果页 | `site/experiments/<run>/index.html` | 3D 回放、物体清单 |
| 审计页 | `site/experiments/<run>/audit.html` | 窗口归属与纠错 |
| 交互记录 | `site/experiments/<run>/records.html` | 模型每次调用的完整往返 |
| 真值实例 | `site/gt_instances/index.html` | ScanNet 标注的 3D 视图，用来判断结果对不对 |
| 交接文件 | `site/handoff/scene0000_00/gt/index.html` | 给下游的物体清单预览（另一个任务） |

`records.html` 不在 run 结束时自动生成，要单独跑：

```bash
.venv/bin/python pagegen/build_records.py --run <实验名>
```

v11 的 `records.html` **还没生成**。

---

## 九、环境与运行

```bash
cd /home/seer/wzt/Reconstruction/InstanceRecognition
taskset -c 0 .venv/bin/python -m pytest tests/ -q        # 235 passed
```

两个解释器，别用错：

- `.venv`：轻量（numpy / pytest / PIL / scipy，**无 torch**），跑 harness、几何、页面、测试
- conda `scene-understanding`：torch + sam3，只在 ① 阶段用

`taskset -c 0` 不是可选项。碰 torch 的调用不加会 segfault；harness 本身不需要 torch，
但统一加上更省心。

**长实验由提出需求的人启动，不要代跑。** 500 帧约 100 个窗口、1155 次调用、25 分钟到数小时。

---

## 十、已经踩过的坑（按损失排序）

**1. 报错太笼统，模型无法自纠。** 一个窗口 26 轮预算，模型 8 次提交完全相同的提案，
因为报错只说「需要理由和可访问的证据」，没说哪一条。现在 `state.py` 的校验按故障拆开，
各自指名。**改任何校验时，报错必须包含「是哪一条、错在哪、怎么改」。**

**2. 归档清单里出现观测 id，模型就会引用。** 而不先 `inspect_object` 是引用不了的。
`observation_ids` 和 `representative_observation_id` 都已删除。**任何新加的字段，
先问：模型会不会把它当成可引用的证据？**

**3. 排序键选错会安静地毁掉结果。** 关键帧选择曾把「框是否触边」当第一排序键，
结果一个只有 24 个可见像素、物体被吧台完全挡住的帧排到了 79 万像素的帧前面。
**可见像素数才是遮挡的直接度量。**

**4. 最后一轮只调一个工具会丢掉整个窗口。** 预算耗尽时如果模型只 `propose_update`
不 `commit_update`，提案有效但没写盘，`run.py` 判定窗口失败。现在预算警告明说「两次调用，都需要」。

**5. 静默忽略字段。** `create` 曾接受 `observations` 但什么都不做，覆盖检查于是报
「这些观测没有状态」，模型看不出真实原因。现在多余字段直接拒绝并说明。

**6. 同一个判据被写进两处，其中一处写错。** prompt 曾说「窗口末尾付不起的扣住会丢掉整个窗口」，
而代码里 `gate_relaxed` 在剩两轮时直接放行。**prompt 里所有关于宿主行为的断言，
都要能在代码里指到那一行。**

---

## 十一、已知未解

不写解决方案，只列事实，供判断工作量：

- **`pending` 只增不减**：v11 末窗 550 条、149 KB，每窗重发。末窗 prompt token 111k，
  初窗 33.6k，涨 3.3 倍。（DEVNOTES 第 11 节有完整实测和四条候选修法）
- **defer 从未被回收**：532 条 defer，被后续窗口重新认领的 **0 条**。
  150 帧和 500 帧两个规模都一样。prompt 里「以后还能被认领」这句话从未兑现。
- **错误率停在 51%，不随规模下降**：v9 与 v11 相同。主要是 `missing-coverage`（34 次，
  中位只缺 1 条）和 `changed-twice`（29 次，多为 assign 与 defer 并存）。
- **跨窗再识别门禁在实际运行中几乎不触发**：v5–v11 累计 146 次 propose，只有 1 次。
  它按距离判断（阈值 `max(1m, 20×自尺度)`），拦不住「相邻但不同」的物体——
  v9 曾把衣柜的 37 条观测全并进窗帘。
- **物体数量没有验收标准**：v11 是 114 个，v7 是 39 个，差了三倍。哪个更接近真实
  还没有定论，因为一直没有拿 GT 对过。

---

## 十二、安全与凭据

网关密钥在 `/home/seer/wzt/Gateway/key.json`，环境变量文件在 `/home/seer/wzt/API/.env`。
`client.py` 对落盘的响应做 `redacted(payload, key)`，**新增任何写响应的地方都要过这个函数**。
探测凭据时只打印长度，不打内容。
