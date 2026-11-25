# Harness 实验页面命名规则

生效日期：2026-09-30。

## 实验标识

```text
harness-{方案版本}-{Agent模型}-{帧数}frames[-{窗口数}windows]-{运行日期}[-r{重跑序号}]
```

- 使用小写英文字母、数字和连字符，不混用下划线。
- 方案版本用 `v1`、`v2` 等，不能把“该实验第几次运行”当成方案版本。
- 模型指 Harness 决策 Agent，不是上游类别识别模型。文件名中 `gpt56-terra` 对应 `openai/gpt-5.6-terra`，`gpt6-sol` 对应 `openai/gpt-6-sol`；完整 API 模型名以实验 `config.json` 为准。
- `frames` 明确表示帧数；按窗口指定范围时可补充 `windows`，例如 `150frames-30windows`，不能把 30 个窗口写成 30 帧。实际窗口大小以 `config.json.window` 为准。
- 日期为首次运行日期（本项目使用 Asia/Shanghai），格式为 `YYYYMMDD`。重新生成页面不改变该日期。
- 同一天、同配置的独立重跑追加 `-r2`、`-r3` 等；恢复同一实验的 checkpoint 继续使用原目录。
- 改模型、输入或方法后建立新目录。`run.py` 要求显式传入 `--output`，不再默认写入固定的历史实验目录。

## 成对页面和资源

设实验标识为 `<experiment-id>`：

| 内容 | 规则 |
|---|---|
| 实验目录 | `/home/seer/wzt/Reconstruction/InstanceRecognition/runtime/experiments/<experiment-id>/` |
| 场景理解结果 | 实验目录下的 `index.html` |
| 逐窗口交互记录 | 同一实验目录下的 `records.html`（与结果页同级） |
| 交互记录资源 | 同一实验目录下的 `records/`（`page.css`、`page.js`、`records.js`、`translations.js`、`window-*.js`、翻译 sidecar） |
| 辅助审计页 | 实验目录下的 `audit.html`；不是上面独立的逐窗口交互记录页 |

结果页与交互记录页**同在一个实验目录内**，两者靠 `index.html` ↔ `records.html` 互链，页面内的数据引用全部相对实验目录解析（`calls/`、`states/`、`records/`）。因此归档或删除某个实验是单目录的原子操作。

目录内的名称是**固定的**，不重复实验标识：标识已由目录名承载，两者再做一次独立推导只会让配对关系多一处可漂移的地方。跨实验导航由 `docs/assets/experiment-navigation.js` 在页面加载时按目录名生成，记录页因此不再需要位于 `docs/`。

`build_harness_records.py --run <experiment-id>` 自动确定记录页和资源目录（实验目录下的 `records.html` 与 `records/`）。`--run` 为必填项，不再默认选择某个历史实验。只有维护特殊历史页面时才使用 `--page` 显式覆盖文件名。

`translate_harness_reasons.py --run <experiment-id>` 使用相同的资源目录；特殊页面使用 `--assets` 指定目录。翻译不是建页的隐含步骤，不改写英文原始日志。

## 本次迁移

| 原实验目录 | 新实验标识 | 首次日志日期 |
|---|---|---|
| `harness_first30_v1` | `harness-v1-gpt56-terra-30frames-20260921` | 2026-09-21 |
| `harness_first30_v2_sol_20260924` | `harness-v2-gpt6-sol-30frames-20260924` | 2026-09-24 |
| `harness_all500_v1` | `harness-v2-gpt6-sol-500frames-20260924` | 2026-09-24 |
| `harness_first30windows_v2_sol_20260929` | `harness-v2-gpt6-sol-150frames-30windows-20260929` | 2026-09-29 |

日期通过各实验最早请求日志中的会话时间戳核对。500 帧实验采用 v2 方案；旧目录后缀 `_v1` 不沿用作方案版本。

实验目录及 4 个独立交互记录 HTML 已迁移；页面资源、切换栏、文档入口和生成器已同步。旧实验目录仅保留指向新目录的符号链接，为历史日志和 config 中保存的旧绝对路径提供兼容；不是另一份实验数据。后续新页面和文档使用新路径，不要把符号链接作为新实验目录或当作另一轮结果统计。

原 `harness-tools.html` 同时保存了 `harness_first30_v1` 和 `harness_window01_v3` 两轮记录。改名后的 V1 页面默认选择前者。

> **后续变更（2026-10-13）**：`harness_window01_v3` 实验目录已清理（其复盘内容保留在 `scene-reconstruction-discussions.html` 第十轮的文字中），旧 `harness-tools.html` 也不在磁盘上。

没有重跑模型，没有改写 `calls/`、`versions/`、`config.json` 或历史源码快照。数据中保留的旧路径是原始记录，不视为待替换的页面链接。

页面入口见 [本仓库网页打开方法](/home/seer/wzt/Reconstruction/web_command.md) 1.6 节（结果页与交互记录页成对）。

## 中途终止的实验

标识中的帧数/窗口数记录**实际提交的范围**，不是启动时的计划。计划 150 帧 / 30 窗口的实验若按决定在第 10 个窗口后终止，应当改名为 `...-50frames-10windows-YYYYMMDD`：否则目录名会声称一个从未提交的范围。

改名的同时需要同步这些位置，缺一项就会让"已跑完 150 帧"的假象留在数据里：

| 位置 | 改法 |
|---|---|
| `config.json` | `frames` 改为实际提交的帧数，`output` 改为新目录名 |
| `input_audit.json` | 旧的 150 帧审计按原顺序截断到 50 帧的前缀；它与重新计算的审计必须逐字节相同 |
| `observations.json` | 只保留 `seq < frames` 的条目 |
| `evidence/` | 删除 `seq >= frames` 的观测图与 `frame_NNNN.jpg` |
| `window_inputs/`, `calls/` | 删除未提交窗口的残件（如 `0050_context.json`、`w0050_*`） |

`summary.json` 由 `--publish-only` 之外的路径生成，进程被终止时不会存在；需按提交状态补写，并注明终止范围与原计划。`run.py` 的 `publish_completed()` 要求 `checkpoint['cutoff'] == config['frames'] - 1`，改名后该条件才成立。

例：`harness-v2-gpt6-sol-150frames-30windows-20261013` 在第 10 个窗口（抽样 0–49）提交后终止，改名为 `harness-v2-gpt6-sol-50frames-10windows-20261013`。该轮使用剪枝后的 mask 源（`cluster_pruned_v1`）。

## 使用示例

以下启动命令是新实验示例，执行前确认目录不存在；本次命名迁移不执行推理任务。

```bash
cd /home/seer/wzt/Reconstruction/InstanceRecognition
.venv/bin/python -u -m src.harness.run \
  --frames 150 --window 5 \
  --output runtime/experiments/harness-v2-gpt6-sol-150frames-30windows-20260930 \
  --model openai/gpt-6-sol --reasoning-effort none \
  --timeout 300 --network-retries 5 --window-network-retries 2
```

已有实验只重建记录页，不调用 Agent：

```bash
cd /home/seer/wzt/Reconstruction
InstanceRecognition/.venv/bin/python InstanceRecognition/src/visualization/build_harness_records.py \
  --run harness-v2-gpt6-sol-150frames-30windows-20260929
```

需要翻译时另行执行（会调用翻译模型）：

```bash
InstanceRecognition/.venv/bin/python InstanceRecognition/src/visualization/translate_harness_reasons.py \
  --run harness-v2-gpt6-sol-150frames-30windows-20260929 --batch-size 30
```
