# ThirdParty 论文与开源情况记录

本目录存放本地拉取的第三方论文代码仓与对应论文 PDF，用于评估「论文声称的能力」与「实际开源的交付物」之间的差距。

本文档记录两篇 SimReady / 组合式场景重建论文的开源程度检查结果：

| 目录 | 论文 | 论文 PDF |
| --- | --- | --- |
| `ReplicateAnyScene/` | ReplicateAnyScene: Zero-Shot Video-to-3D Composition via Textual-Visual-Spatial Alignment | `ReplicateAnyScene/ReplicateAnyScene.pdf` |
| `SimRecon/` | SimRecon: SimReady Compositional Scene Reconstruction from Real Videos | `SimRecon/SimRecon.pdf` |

> 说明：本文档的结论全部来自**只读静态检查**（`git`、`ls`、`grep`、`find`、`pdftotext`）与两份 PDF 正文。
> **没有运行任何代码、没有下载任何权重、没有做过端到端复现。**
> 因此「代码已开源」一律只代表**仓库里存在对应文件**，不代表**在任意机器上可以跑通**。

---

## 一、论文速览对照

| 项目 | ReplicateAnyScene (RAS) | SimRecon |
| --- | --- | --- |
| arXiv | 2604.10789（2026-04-12） | 2603.02133 |
| 会议 | 未在仓库 README 标注 | CVPR 2026 **Highlight** |
| 单位 | Tsinghua University、Zhejiang University | Tsinghua University、Galbot |
| 作者 | Mingyu Dong, **Chong Xia**, Mingyuan Jia, Weichen Lyu, Long Xu, Zheng Zhu, Yueqi Duan | **Chong Xia**, Kai Zhu, Zizhuo Wang, Fangfu Liu, Zhizheng Zhang, Yueqi Duan |
| 定位 | SimRecon 的 robust / zero-shot **follow-up work** | 提出 perception-generation-simulation 三段式管线 |
| 核心贡献 | 五阶段级联 + **C3DR benchmark** | **AVO**（视角优化）+ **SGS**（场景图合成）两个桥接模块 |
| 评测数据 | 自建 C3DR（50 个场景，来源 Replica / ScanNet / ScanNet++ / YouTube） | ScanNet 的 20 个真实场景 |
| 输入 | 仅 raw RGB video frames | 仅 raw RGB videos（不用 depth / normal / semantics） |

两篇论文作者组高度重叠（Chong Xia、Yueqi Duan 同时在列），RAS 在 SimRecon 仓库的 News 里被明确称为 follow-up 工作；两者都在 GitHub 账号 `xiac20` 下开源。

---

## 二、仓库基本信息（已核对）

| 检查项 | `ReplicateAnyScene/` | `SimRecon/` |
| --- | --- | --- |
| remote | `https://github.com/xiac20/ReplicateAnyScene.git` | `https://github.com/xiac20/SimRecon.git` |
| 本地 HEAD | `4a2186ab488e3f7acdfef77080e67a6452e41b83`（Merge PR #4 readme-improvement） | `01e85926f0e19e89a7edbaebf095dcf01b37c8d1`（Update README.md） |
| 跟踪文件总数 | **21** | **1785** |
| LICENSE 文件 | ✅ `LICENSE`，MIT License，`Copyright (c) 2026 xcThu` | ❌ **根目录无 LICENSE 文件**（仅 README 第 21 行有 MIT badge） |
| 自研代码规模 | `main.py` 179 行 + `src/` 8 个文件 1663 行 | 主流程约 5100 行（`optimize_by_avo.py` 1124、`infer_scene_graph.py` 728、`project_instances_to_frames.py` 775、`train_semantic.py` 943、`merge_scene_graphs.py` 490 等） |
| 第三方依赖方式 | `.gitmodules` + 3 个 **gitlink 子模块**（`sam3` / `vggt` / `sam-3d-objects`），本地目录**为空未初始化** | **无 `.gitmodules`**，`submodules/` 下 4 个目录是 **vendored 源码快照**（普通 tree，非 gitlink） |
| 依赖版本固定 | 子模块用 commit 固定：`sam-3d-objects@81a82373`、`sam3@bfbed072`、`vggt@44b3afbd` | 快照冻结在仓内，**无可见的上游版本锚点** |
| 权重 / 数据 | 不在仓内（`.gitignore` 排除 `/models`、`/outputs`） | 不在仓内（无 `.pth` / `.ckpt` / `.pt` / `.ply`） |
| 环境声明 | `environments/default.yml`（CUDA 12.1） | `requirements.txt` + README 指定 python 3.9 / torch 2.1.0+cu118 |

---

## 三、ReplicateAnyScene 开源程度：按论文五阶段逐项核对

论文管线是 textual-visual-spatial 五阶段级联，仓库只交付其中一部分。依据 `main.py` 内的注释逐条核对：

| 阶段 | 论文名称 | 开源状态 | 证据 |
| --- | --- | --- | --- |
| Stage 1 | Progressive Object Discovery | ❌ **未开源** | `main.py:22–24`：`# The code of this stage is not publicly available for now.` / `# You can refer to ./assets/example/hallway.json to manually specify object categories` |
| Stage 2 | Spatial-Guided Visual Deduplication | ✅ **完整开源** | VGGT 预测 → SAM3 wall/floor 分割 → `align_to_room_coordinate_system` → room 坐标对齐 → SAM3 video tracking → `self_category_deduplicate` / `cross_category_deduplicate`（`src/sg_deduplication.py` 329 行） |
| Stage 3 | Optimal-View 3D Asset Generation | ✅ **完整开源** | `get_optimal_view_frame_id` + `generate_3d_asset_in_subprocess`（`src/instance_generation.py` 174 行） |
| Stage 4 | Iterative Visual-Spatial Alignment | ❌ **未开源** | `main.py:130–131`：`# This part of the code is not publicly available for now.`（论文里的 MASt3R 对应 + Umeyama 求解部分**没有代码**） |
| Stage 5 | Semantic-Aware Scene Refinement | ⚠️ **部分开源** | `main.py:133–136`：`# The code for this stage is not publicly available for now.` / `# We provide the code for refining objects in "supported by floor", "attached to wall" and "embedded in wall" relationships.` 仅落地 `refine_supported_by_floor_object` / `refine_attached_to_wall_object` / `refine_embedded_in_wall_object`（`src/sp_refinement.py` 215 行） |

仓库 README 自身在第 32 行的表述与上述一致：
`[04/14/2026] We release the code for stage 2 and 3, as well as partial code for stage 1 and 5.`

### 关键发现（RAS）

1. **交付范围 = 5 阶段中的 2 个完整 + 1 个部分。** 缺口的 Stage 1 与 Stage 4 正好是论文强调的「开放词汇类别发现」与「render-match-optimize 迭代对齐」两处核心创新点。作者给的替代路径是**手工填写** `--category_path`（`assets/example/hallway.json` 只有 319 字节）。
2. **子模块 URL 是 SSH 别名，普通用户 `git submodule update --init` 会失败。** `.gitmodules` 里三个 url 都是
   `org-16943930@github.com:facebookresearch/...`，这是内部 SSH 别名（`~/.ssh/config` 里的 Host 映射），不是可公开解析的地址。按 README 第 52–54 行的克隆步骤执行会在第 3 步直接报错，需要手工替换为 `https://github.com/...`。
3. **上游仓库本身是公开的**（`facebookresearch/sam3`、`vggt`、`sam-3d-objects`），所以这个坑是**配置失误**而非刻意闭源。
4. **C3DR benchmark 未见发布链接。** PDF 正文只在方法/实验部分描述 C3DR 的构建方式（50 场景、三层标注、人工选资产与人工布局对齐），grep 未发现下载地址或"will be released"承诺；仓库里也没有数据目录。当前 C3DR **不可获得**。
5. **权重全部走 HuggingFace，需自行下载 3 套**：`facebook/VGGT-1B`、`facebook/sam3`、`facebook/sam-3d-objects`（README 第 104–106 行）。
6. **硬件门槛写在 README 里**：`--max_frames` 默认 160，注释标明这是面向 **48GB VRAM** GPU 的取值。

---

## 四、SimRecon 开源程度：按模块逐项核对

SimRecon 体量大得多（1785 个跟踪文件），但「代码开源」与「管线可复现」的边界需要分开看。

### 4.1 已随仓交付的部分

| 模块 | 对应文件 | 说明 |
| --- | --- | --- |
| 2DGS 训练 | `train_2dgs.py` (301 行) | 场景级语义重建前端 |
| 语义训练 | `train_semantic.py` (943 行) | instance-level semantic 训练 |
| **AVO**（Active Viewpoint Optimization） | `optimize_by_avo.py` (1124 行) | README News `[03/03/2026]` 发布 |
| **SGS**（Scene Graph Synthesizer） | `infer_scene_graph.py` (728 行) + `merge_scene_graphs.py` (490 行) | README News `[03/18/2026]` 发布 |
| 覆盖率采样 / 实例投影 | `coverage_sampling.py` (432 行)、`project_instances_to_frames.py` (775 行) | 用 VGGT 采样最大体素覆盖率视角 |
| 渲染与工具 | `gaussian_renderer/`、`scene/`、`utils/`、`spatial_track/`、`vis_utils/`、`raytracing/` | — |
| 第三方快照 | `submodules/diff-surfel-rasterization` (1547 文件)、`simple-knn`、`raytracing`、`vggt` | vendored，非子模块 |
| CropFormer 快照 | `semantic_modules/CropFormer/` (123 文件) | qqlu/Entity CropFormer 的拷贝 |

`infer_scene_graph.py` 默认 VLM 是 `Qwen/Qwen3-VL-8B-Thinking`（第 34 行）。

### 4.2 未随仓交付、需要用户自己补的部分

这些都是 **README 明确写出**的（属于作者主动声明的范围外，不是隐藏缺失）：

| 缺失项 | README 位置 | 现状 | 用户需要做什么 |
| --- | --- | --- | --- |
| 3D 资产生成 | 第 165 行 | 论文用的是 **Rodin**（hyper3d.ai，闭源商业服务），仓库**没有任何资产生成代码** | 去 Rodin 官网生成，或改用 SAM3D / TRELLIS |
| 位姿 / 朝向估计 | 第 185 行 | 未提供 | 自行从 instance 3DGS 算 bbox 中心与尺寸，再用 **FoundationPose** 估朝向。README 原话："relatively straightforward so users can easily implement this part themselves" |
| 仿真器内装配 | 第 186 行 | 未提供 | "users can input the assets manually" —— 手工按参数摆放，再按论文的逐层物理仿真激活流程执行 |
| CropFormer 权重 | 第 93–97 行 | `ckpts/` 目录不存在 | 从 HuggingFace `qqlu1992/Adobe_EntitySeg` 手工下载 `CropFormer_hornet_3x_03823a.pth` |
| 预处理数据集 | 第 100 行 | 不在仓内 | 走 Google Drive 链接，或按文档自行用 COLMAP / HLOC 预处理 |

**关键验证**：全仓 `.py / .md / .txt / .sh` 中 grep `isaacsim|omni.isaac|pybullet|mujoco|physx|habitat` → **零命中**。也就是说论文里"在 Isaac Sim 中完成仿真场景构建"这一段，**仓库里一行代码都没有**。

### 4.3 README 声明与实际交付的落差

- README 第 30 行：`[03/18/2026] We release the code of our Scene Graph Synthesizer (SGS) module, making our entire codebase fully open-sourced.`
- README 第 31 行紧接着自我修正：`[03/18/2026] To accommodate diverse choices in 3D asset generation models and physical simulators, we leave their specific implementations open for user customization.`

前一句是「整个代码库完全开源」，后一句实际意思是「资产生成和物理模拟器这两段我们不做，留给你自己选」。**仿真与资产这两块恰好是 SimReady 论文的最终交付物**，因此"fully open-sourced"应理解为**感知—场景图这一段 fully open-sourced**。

### 关键发现（SimRecon）

1. **MIT badge 但无 LICENSE 文件。** README 第 21 行挂了 `License-MIT` badge，仓库根目录却没有任何 `LICENSE` 文件。全仓能找到的 LICENSE 只有第三方组件的：`submodules/raytracing/LICENSE`、`LICENSE_NVIDIA`、`submodules/simple-knn/LICENSE.md`、`submodules/diff-surfel-rasterization/LICENSE.md`。**没有授权正文 = 法律上默认保留所有权利**，这与 badge 的声明不符。使用前需要确认。
2. **`submodules/` 是 vendored 源码快照，不是子模块。** `git ls-tree HEAD submodules/` 返回 4 个 `040000 tree`（普通目录），而非 `160000 commit` gitlink；仓库也没有 `.gitmodules`。好处是 clone 即完整、不依赖外部网络；代价是**上游版本无法追溯**，无法判断快照对应哪个 commit，也无法做 upstream diff。
3. **VGGT 被裁剪后内联**：`submodules/vggt/` 只有 41 个文件（`dependency/`、`heads/` 等），不是完整上游仓库。
4. **CropFormer 也是内联快照**（`semantic_modules/CropFormer/` 123 文件），而非依赖声明。**快照里的 LICENSE 与上游不一致时，责任在分发方**。
5. **AVO / SGS 是真的开源了**，代码量和脚本完整度对得上 README 的发布节奏（03/03 AVO、03/18 SGS），README 第 114–179 行给出了完整命令行序列。
6. **作者主动承认了感知精度上限**：第 160–162 行说明 AVO 结果不理想时可调 5000 轮 / depth 约束系数等超参，并声明"生成的 instance labels 不准确，仅供参考，由 SGS 模块的 VLM 进一步判定"。

---

## 五、复现可行性代价对照

### ReplicateAnyScene

| 维度 | 代价 |
| --- | --- |
| 代码完整性 | ⚠️ 5 阶段缺 2.5 阶段。Stage 1 需手工写类别 JSON，Stage 4 的对齐逻辑**没有任何参考实现** |
| 依赖获取 | ⚠️ 3 个子模块需先手工改 `.gitmodules`（SSH 别名 → https），再 `git submodule update --init --recursive` |
| 权重 | 3 套 HuggingFace 模型（VGGT-1B / sam3 / sam-3d-objects），SAM3D 需 `pip install -e '.[dev],.[p3d],.[inference]'` 三步 + NVIDIA 源 |
| 硬件 | README 标注默认配置面向 **48GB VRAM** |
| 评测 | ❌ C3DR benchmark 无获取渠道，**无法复现论文主表** |
| 结论 | **能跑通 demo（hallway 样例），不能复现论文** |

### SimRecon

| 维度 | 代价 |
| --- | --- |
| 代码完整性 | ✅ 感知 + AVO + SGS 完整；❌ 资产生成、位姿估计、仿真器装配全缺 |
| 依赖获取 | ⚠️ CropFormer 需编译（`sh make.sh`）+ 外部 detectron2 + mmcv，权重手工下载 |
| 关键外部服务 | ⚠️ 论文用 **Rodin（闭源商业服务）** 生成资产；换 SAM3D/TRELLIS 需自行验证效果一致性 |
| 需自研代码 | ❌ FoundationPose 位姿估计 + 仿真器装配脚本（论文核心贡献之一）**需从零实现** |
| 数据 | Google Drive 预处理数据集（需外网） |
| 硬件 | 论文报告 AVO 单卡 RTX A6000 每物体约 30 秒、整场景平均约 21 分钟（不含 2DGS 训练与资产生成） |
| 结论 | **能跑出 2DGS + 语义 + 场景图，不能开箱得到 SimReady 场景** |

---

## 六、一句话总结

- **ReplicateAnyScene**：开源程度**中等**。代码规模小（21 文件 / 1663 行），五阶段里 Stage 2、3 完整开源，Stage 1、4 完全未开源、Stage 5 部分开源（作者在 `main.py` 注释里明确写出）；子模块 URL 是私有 SSH 别名需手工修；**C3DR benchmark 无发布渠道**。许可证 MIT，文件齐全。
- **SimRecon**：开源程度**较高但边界需自行补全**。1785 文件、主流程约 5100 行，AVO / SGS / 2DGS / 语义推理确实开源；但**资产生成、位姿估计、仿真器装配三段完全不在仓内**（README 明确告知），且**有 MIT badge 却无 LICENSE 文件**。
- 共同点：两篇论文都用闭源模型做关键环节（RAS 的 Stage 1/4 与 C3DR；SimRecon 的 Rodin 与 Isaac Sim），因此**"代码开源" ≠ "可复现"**，实际复现都需要大量补写工作。

---

## 七、核对范围与未做的事

**本次做过的事（全部只读）**

- `git remote -v` / `git log -1` / `git ls-files | wc -l` / `git ls-tree` / `git status` — 确认仓库来源、HEAD、文件数与 tree/gitlink 类型
- `find`、`ls`、`grep`、`sed`、`cat`、`wc` — 确认 LICENSE、`.gitmodules`、`submodules/` 形态、`.gitignore`、README News 与环境声明
- `pdftotext` + grep — 确认两篇 PDF 中是否声明过代码/数据发布（RAS 未提 C3DR 下载；SimRecon PDF 仅给出项目页链接）
- `main.py` 与 `README.md` 逐段阅读，核对每个阶段的注释与声明

**本次未做的事（不要当作已验证）**

- ❌ 未安装任何依赖、未创建 conda 环境
- ❌ 未下载任何模型权重（VGGT / SAM3 / SAM3D / CropFormer）
- ❌ 未运行 `main.py`、`train_2dgs.py`、`optimize_by_avo.py` 等任何脚本
- ❌ 未做端到端复现，**所有"能跑通"的判断均为静态推断**
- ❌ 未访问 GitHub 网页 / HuggingFace / Google Drive 核实远端资源当前是否可用（结论基于本地 clone 状态与 README 文本）

**风险提示**

- 两篇论文的时间戳为 2026 年，属于很新的工作，**上游仓库可能随时更新，本文档结论会过期**。
- SimRecon 无 LICENSE 文件 → 使用前请先与作者确认授权，不要仅凭 badge 判断。
- RAS 的 SSH 别名子模块 URL 说明该仓库可能混入了作者内部环境配置，克隆时注意替换。
- 目录版本控制状态：`Reconstruction` 仓库（HEAD `7779996`）当前 `git status` 显示 `?? ThirdParty/`，即本目录**尚未被跟踪**；根 `.gitmodules` 只覆盖 5 个其他第三方仓（SAM3 / WeDetect / amb3r / depth-anything-3 / map-anything），**不含 RAS 与 SimRecon**。若要提交，`ThirdParty/` 及其内部两个嵌套仓需按普通目录处理（内部 `.git` 会形成嵌套仓库，请先决定是移除内部 `.git` 还是改登记为 submodule）。

**相关文件**

- 论文阅读记录（本仓已有的中文摘要页）：`news/papers/paper_reading_record/replicate_any_scene.html`、`news/papers/paper_reading_record/simrecon.html`
