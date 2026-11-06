# InstanceRecognition

增量式场景理解 Harness。2026-10 从旧树重构而来，只做**结构性**调整，不改行为：核心功能代码与两个页面的构建代码移植过来，产物与页面分开存放。

旧树已于 2026-10-19 移除，只保留一份不含代码与产物的存根在 `../InstanceRecognition_backup/`。

## 四类东西分开存放

```text
InstanceRecognition/
├── src/            ① 功能代码
├── pagegen/        ② 页面构建代码（含模板）
├── artifacts/      ③ Pipeline 产物
└── site/           ④ 生成的页面
```

| 类别 | 内容 | 可重定向 |
|---|---|---|
| `src/` | Harness 决策循环、几何、共享可视化 helper | — |
| `pagegen/` | 结果页 / 记录页构建器 + 手写模板 | — |
| `artifacts/` | 每次运行的 `calls/`、`versions/`、`evidence/`、`config.json`、`summary.json` | `IR_ARTIFACTS_ROOT` |
| `site/` | `index.html`、`records.html`、`states/`、`objects/`、页面 JS | `IR_SITE_ROOT` |

**产物与页面分开的意义**：`site/` 里的东西全部可由 `artifacts/` 重新生成，所以一个实验的调用日志可以单独归档或删除，不影响它的页面；反之亦然。此前这两类是混在同一个实验目录里的。

## 代码结构

```text
src/config.py                路径的单一来源（四个根 + 外部依赖）
src/harness/                 run, publish, client, observations, state, tools,
                             inputs, history, spatial_context, paths, prompt.txt
src/geometry/                amb3r_mask_source（唯一的 lifting）,
                             assemble_perception_source（感知产物 → liftable source）,
                             backproject_masks（共享 helper）, incremental_objects
src/inference/               gpt_sam31, label_frames, gpt_labels（感知链路）
src/visualization/           共享 helper：page_naming, object_bboxes, publish_gpt_viewer

pagegen/build_site.py        页面装配（manifest、states/、objects/、审计页）
pagegen/build_result.py      结果页
pagegen/build_records.py     交互记录页
pagegen/build_viewer.py      渲染器 bundle 装配
pagegen/build_mask_project.py mask 投影检查页
pagegen/templates/           result/  records/  mask_project/  shared/
```

`src/visualization/` 里的两个模块不是页面代码，因此留在了功能侧：`object_bboxes.export_version_objects` 在**决策热路径**上被 `harness/spatial_context.py` 调用；`publish_gpt_viewer.rebase_manifest` 被几何模块调用。

## mask 反投影：方向为什么重要

**旧做法（已删除）**：把整份场景点云投影到每个像素，再查该像素的 mask 归属。这个方向是病态的——一条相机射线上的**所有**点都落到同一个像素，而 2D mask 对深度不可知。实测 frame 0：GT 点云投影后 **42.4% 的点穿到了可见表面之后**，中位穿深 0.54 m。物体背后的书包继承了沙发的标签，这就是遮挡泄漏。更糟的是 `cameras_rgb.json` 没有 `depth_m`，可见性判据退化成**点云自引用**（每点的参考深度就是它自己那个像素），于是恒判可见。

**现在的做法** `amb3r_per_frame_v1`：

```
对每一帧：
  1. 读取 AMB3R 的逐帧点图 pts（392×518×3）       ← 网络预测的深度，不是 GT 网格
  2. GPT/SAM3 mask（1296×968 实例图）→ resize 640×480 → center crop 518×392
  3. mask 命中且 conf ≥ 0.3 的每个像素：取该像素的 3D 位置，应用 Sim(3)
```

方向仍是**从像素出发**：一个像素最多得到一个点，所以没有多射线歧义，遮挡者的表面从未被查询。与已删除的 `mesh_first_surface_v1` 的区别只在深度的来源——那套把 GT 网格光栅化出每像素首表面，这套直接用 AMB3R 预测的深度。

选 AMB3R 而不是 GT 网格的理由：GT 网格有重建空洞（黑色椅背、细结构、扫描角度差的区域），那些像素上反投影会落到**相机实际看到的表面之后**的东西上；AMB3R 在这些位置仍给出网络预测的深度。

**已知代价**：`sim3_oracle_scale` 是**对着 GT 相机位姿拟合**的（`alignment_scale = 0.8875`，五厘米 F-score 从 0.099 涨到 0.562），所以这份几何**不能**用作 AMB3R 的独立精度评价，只能作为 mask 反投影的深度来源。另外 AMB3R 的 mask source 只含被 mask 声称的像素（约 12 万/帧），**不含全场景点云**，所以页面背景必须取自 Sim(3) 拟合所用的那份规范点云——`amb3r_mask_source` 会把这份点云编码成 `assets/background.js` 随 source 一起写出。

## 环境

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

依赖只有 **numpy、Pillow、scipy、pytest**（约 254 MB）。感知链路（GPT 出 label、SAM3.1 出 mask）在独立的重依赖环境里跑，见 `requirements-runtime.txt`；本树不含 torch / ultralytics / SAM3 / Flask。

**大构建仍需要 `taskset -c 0`。** 测试（199 项，1.2 s）不固定 CPU 也稳定通过，但完整 publish 构建不加就会 SIGSEGV（实测复现）。旧树的同类现象根因在 torch/ultralytics 一侧，但这个轻量环境里也会触发，所以构建命令一律带上。

素材位置由 `src/config.py` 推导，与旧树同深度，因此 `weights/`、`data/scannet_v2/`、`API/scannet_gpt_sam/` 都沿用仓库级布局，无需额外配置。

## 运行

完整链路是三步：感知阶段产出 label 与 mask，assembler 把两半组装成 liftable source，再 lift 成 Harness 能吃的 source。

**① 感知**（GPT 出 label，SAM3.1 出 mask；需重依赖环境，见 `requirements-runtime.txt`）：

```bash
taskset -c 0 $SAM31_PYTHON -u -m src.inference.label_frames --run gpt_sam31_uniform500
taskset -c 0 $SAM31_PYTHON -u -m src.inference.gpt_sam31   --run gpt_sam31_uniform500
```

产物落在 `artifacts/experiments/<run>/perception/{labels,masks}/`。`label_frames` 只调网关、不碰 torch，用本树 .venv 也可；`gpt_sam31` 需要 SAM3.1。

**② 组装 source**（把两个独立目录合成一份带 RGB 与 mask 路径的 source）：

```bash
.venv/bin/python -u -m src.geometry.assemble_perception_source \
  --run gpt_sam31_uniform500 \
  --output artifacts/sources/scene0000_00/gpt_sam31_uniform500_v1 \
  --frames-dir artifacts/frames --count 20
```

这一步的**全部价值是校验**：每个 mask 记录了它所用 label 文件的 SHA-256，assembler 据此确认两半确实配套——配错的话点云看起来完全正常，只有类别是错的。它同时把上游记录的那条（已被重构作废的）RGB 路径换成 `--frames-dir` 下的真实位置。输出**不含** `lifting_policy`，因为此时还没有几何。

**③ lift 成 Harness source**（mask 像素 → AMB3R 深度 → 世界点）：

```bash
taskset -c 0 .venv/bin/python -u -m src.geometry.amb3r_mask_source \
  --npz  /home/seer/wzt/Reconstruction/Video2PointCloud/output/amb3r/vo_uniform500/scene_scene0000_00_uniform500_metric_head_results.npz \
  --sim3 /home/seer/wzt/Reconstruction/Video2PointCloud/output/amb3r/vo_uniform500/pointcloud_gt_corrected_evaluation.json \
  --source artifacts/sources/scene0000_00/gpt_sam31_uniform500_v1 \
  --output artifacts/sources/scene0000_00/scene_amb3r_v1 \
  --frames-dir artifacts/frames --conf 0.3
```

`--conf` 是 AMB3R 的置信度阈值（默认 0.3）：低于它的像素即使被 mask 命中也不产点。输出目录必须为空。这一步写出 `manifest.json`（含 `lifting_policy`）、逐帧 npz、逐帧 JS、`assets/background.js`（从 Sim(3) 所用的那张规范点云编码）与 `projection_report.json`。

**④ 跑 Harness**（source 落在默认位置时可省 `--source`）：

```bash
taskset -c 0 .venv/bin/python -u -m src.harness.run \
  --output harness-v5-amb3r-20frames-YYYYMMDD-HHMM \
  --frames 20 --window 5 \
  --model openai/gpt-6-sol --reasoning-effort none
```

`--run-dir` / `--site-dir` 可分别覆盖两条输出树。`--prepare-only` 只准备证据、不调模型，适合冒烟。

只重建页面（不调模型）：

```bash
# 结果页 + 审计页：读 run 的 config.json 里记的 site_dir，或显式指定
.venv/bin/python -u -m src.harness.run --publish-only \
  --output <run-dir> --site-dir <site-dir>

# 交互记录页
.venv/bin/python pagegen/build_records.py \
  --run <实验名> --run-dir <run-dir> --site-dir <site-dir>
```

理由以模型输出的英文原文呈现，本树不做翻译（此前的中译后处理已移除）。

## 三个页面的定位

结果页（`index.html`）和交互记录页（`records.html`）生成在 `site/experiments/<实验名>/` 下，互为同级：

- **结果页**：窗口末帧 + 累计物体点云 + 3D bbox 的三维回放；底部链接到同目录的 `audit.html`（窗口归属与纠错记录）。
- **记录页**：逐窗口的模型调用流水（工具、参数、结果、英文理由）；资源在 `<实验名>/records/`。

第三个页面是文档页，不随实验生成，直接放在 `docs/` 下用浏览器打开：`pipeline.html`（四段流水线的全局图景）、`prompts.html`（prompt 每一版的全文与改动理由）、`input.html`（一个窗口模型收到的全部输入）、`debug/debug-log.html`（debug 记录：代码改动的问题与方向，以及某次运行里观察到的具体问题；图像证据放 `docs/debug/evidence_img/`）。

两页顶部的实验切换栏来自仓库共享的 `docs/assets/experiment-navigation.js`，按目录名匹配。**注意它只认 `harness-` 前缀的实验名**——目录名不匹配时导航会静默不显示（脚本 regex 不命中即 return），不是报错。

该脚本必须登记**实验目录相对仓库根的路径**（`path` 字段）。基准以脚本自身位置（`docs/assets/`）锚定仓库根，而不是从页面 URL 反推，因为页面与它引用的产物不在同一个父目录下。自 2026-10-19 起所有实验都在本树 `site/experiments/` 下。新实验登记时 `path` 不能省。

## 与旧树的关系

旧树已于 2026-10-19 删除（`src/`、`tests/`、`.venv/`、`runtime/` 共 27 GB），本树不再有任何只读依赖。

- **`observations.DEFAULT_SOURCE` 只指向本树** `artifacts/sources/scene0000_00/scene_amb3r_v1`，不回退到任何旧 policy 的 source——缺失时显式报错，而不是静默换一套几何。
- **`resolve_recorded_path()` 只按产物树解析**。它曾回退到旧树根以兼容老 run 的相对路径记录；旧树删除后该回退只会指向不存在的目录，已移除。
- **v3 / v4 两个实验页面的数据随旧树一起删除**，页面文件仍在但内容缺失；它们使用的 geometry policy 也已在代码中移除。

## 首轮实验（本树产出）

`harness-v4-mesh-sol-20frames-20261014-1558`（**历史 run**，`mesh_first_surface_v1`）：20 帧 / 4 窗口，`openai/gpt-6-sol`，source 是本树网格首表面的 lifted source。4/4 窗口提交，20/20 帧，**28 个物体**，38 次有效调用，无重试窗口，`pending` 1。该 policy 与建它的 `mesh_mask_source` 已被删除，页面仍可查看，但 source 不再能由本树重建。

逐窗口耗时（对比可见网关侧波动很大）：

| 窗口 | 耗时 | 调用 |
|---|---|---|
| 1 (帧 0-4) | 288 s | 6 |
| 2 (帧 5-9) | 1359 s | 13 |
| 3 (帧 10-14) | 1204 s | 14 |
| 4 (帧 15-19) | 433 s | 5 |

第 2、3 窗口各遇到数次读超时重试，是耗时主因；窗口 4 因网关账户余额耗尽（HTTP 402 `budget_exceeded`）中断过一次，充值后续跑完成，产物无损坏。`usage`：prompt 3,447,711 / completion 15,532 / 合计 3,463,243。

校验结果：240 个页面/资源/日志 URL 全部 HTTP 200；两棵树的隔离成立——`artifacts/` 侧 0 个 HTML，`site/` 侧无 `calls/`；Chrome headless 下两页渲染正常，页面几何说明为"真值网格几何；每帧将 mask 像素反投影到该像素的首表面"。

`harness-v3-sol-20frames-20261014-1313`（**历史 run**，`cluster_pruned_v1`）：20 帧 / 4 窗口，source 指向旧树的 `scene_cloud_visibility_v1_pruned`（只读）。4/4 窗口提交，20/20 帧，27 个物体，35 次有效调用，`pending` 24。其页面几何说明为点云投影口径。该 policy 同样已删除，仅存页面。

导航栏已登记本次实验（`V3 · 前 20 帧 / 4 窗口 · 剪枝 mask · 重构树 · GPT-6 Sol · 2026-10-14`）。未登记的实验仍会被自动追加，但 label 只会是目录名本身，且因为拿不到 `path`，跨树跳转会回落到当前页面的基准——所以正式实验都应在 `docs/assets/experiment-navigation.js` 的 `experiments` 数组里登记 `path`。

## 测试

```bash
.venv/bin/python -m pytest tests/ -q     # 199 passed
```

覆盖 Harness 决策循环与页面构建两条链路。`src/web/frontend/` 那棵旧网页源码树、`src/data/` 的 ScanNet reader、`src/inference/` 的推理侧（model_service / pipeline / SAM3）、以及旧树的 22 个兼容 shim 都不在本树的移植范围内。
