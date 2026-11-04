# 开发笔记（写给下一次接手的自己）

`README.md` 是对外的结构说明。这份是对内的：**约定、陷阱、以及为什么当初这么选**。
凡是能从代码或 git 读出来的，不在这里重复。

---

## 1. AMB3R 的溯源问题（最重要，别忘）

### 数据流实况

mask 反投影**只用逐帧深度**，全程不碰全局点云：

```
mask 像素 → pts[frame, h, w] → local_pts → _apply_sim3() → world_pts
```

原始逐帧 npz（`Video2PointCloud/output/amb3r/vo_uniform500/scene_scene0000_00_uniform500_metric_head_results.npz`，1.7 GB）：

```
pts    (500, 392, 518, 3) float32    逐帧点图 —— lift 的唯一来源
conf   (500, 392, 518)    float32    置信度，阈值 0.3
pose   (500, 4, 4)        float32    AMB3R 自己的相机位姿
kf_idx (186,)             int64      关键帧下标
metric_scale ()           float32    度量头的尺度估计（见下方「待确认」）
```

### 全局点云的两个用途

1. **页面背景**（`pagegen/build_mask_project.py` 生成 `background_amb3r.js`，纯视觉）。注意 lift 自身也会把同一份点云编码成 source 内的 `assets/background.js`，给另一个查看器用；两处是**各自独立编码**的，互不读取。
2. **拟合 Sim(3)** —— 全局点云在这里的角色是「对齐锚点」，不是「点的来源」

第 2 点才是关键：**不用它的点，但用了从它算出来的变换。**

### 为什么这是个坑

`sim3_oracle_scale` 是**对着 GT 拟合的**。两份结果对比（`pointcloud_gt_corrected_evaluation.json`）：

| 指标 | `se3_native_scale`（自尺度） | `sim3_oracle_scale`（当前链路） |
|---|---|---|
| `alignment_scale` | 1.0 | **0.8875** |
| `trajectory_ate_rmse_m` | **0.257** | **0.091** |
| `chamfer_l1_m` | 0.192 | 0.066 |
| `fscore_05cm` | **0.099** | **0.562** |

五厘米 F-score 从 0.099 涨到 0.562 —— 这个「效果好」里有一大半是 GT 的功劳，不是 AMB3R 自己的。

### 结论与选型

- **只要一份还行的深度来反投影 mask**（本树唯一用途）→ 现状可用。
- **要让 AMB3R 作为独立几何源被评价** → 现状**不行**，必须去掉 GT 锚定，走 `se3_native_scale`，或直接用 `pts` + `pose` 自洽重建。已发布的页面文案不得暗示独立精度。
- **AMB3R 与已删除的 GT mesh 链路分辨率不同，点数不可直接对比。** mesh 侧是 968×1296（逐 mask 像素一个点，frame 0 共 750,014 点）；AMB3R 侧是 392×518（frame 0 共 123,720 点），点数天生少约 4 倍 —— 那是**分辨率差异，不是质量差异**。（注：代码里 `AMBR3R_SIZE = (518, 392)` 是 **w, h** 顺序，而 npz 的 `grid_shape_hw` 是 **h, w** —— 这两个约定混用是这块最容易写错的点。）
  逐实例（frame 0，office chair）：mesh 141,064 像素 → 123,769 点；AMB3R 对齐到自己的网格后 23,055 像素 → 22,153 点（**96% 命中**，其余是 AMB3R 置信度低于阈值 0.3 的像素，即 `claimed & (conf >= 0.3)` 这个筛子滤掉的）。

### 待确认

npz 里有 `metric_scale` 标量，是 AMB3R 度量头**自己**的尺度估计。这可能是摆脱 GT 锚定的天然入口 —— 但**尚未验证**它和 `sim3_oracle_scale` 是什么关系。真要用 AMB3R 做独立源，先查这个。

---

## 2. mask_project 页面（`site/mask_project/`）

### 页面只有一个源，一份背景

| 源 | 背景 | 文件 |
|---|---|---|
| `amb3r` | Sim(3) 修正后的规范 AMB3R 点云（492,555 点） | `background_amb3r.js` |

**为什么必须外借背景**：AMB3R 的 mask source 只含被 mask 声称的像素（约 12 万/帧），**没有**全场景点云，自任背景不成立。背景只能取 Sim(3) 拟合所用的那份规范点云——它和 lifted 点在同一坐标系里。GT mesh 在这里是错的：它恰恰是 AMB3R 要绕开的、有孔洞的几何。

`amb3r_background_cloud()` 从 source manifest 的 `sim3_source` 反查 `sim3_ply`；**键缺失、evaluation 文件缺失、`sim3_ply` 缺失、PLY 缺失，四种情况都返回 `None`**，调用方打警告并降级为空场景，不回退到错的几何。所以这里用 `.get()` 而非 `[]`：`[]` 会在键缺失时抛 `KeyError`，把异常扔在返回 `None` 之前，降级分支永远走不到。

### 踩过的坑（都别再犯）

- **注册键按源命名空间隔离**（`frameKey()` / `background:<source>`）。现在只有一个源，但机制保留：加第二个源时若改用裸键，一个源的 payload 会答复另一个源的查询。源切换器已删，**`?source=` 参数也不再支持**。
- **相对 URL 要以 page root 为基准**，不是源的子目录。`rgb_url` / `mask_url` 曾多算一层 `../`，表现为红字 `无法加载 ../../../../...jpg`。
- **WebGL 里 `useProgram` 是前置条件**。丢了它，`drawArrays` 静默什么都不画（`uniformMatrix4fv: location is not from the associated program`）。
- **背景不能写深度**（`depthMask(false)`）。它和 lifted 点在同一批表面上，写深度就会遮住要点的那批点。
- **2D 底图必须走和 mask 相同的 center crop**（`rgb_crop`），否则偏约 6 px。
- **`getImageData` 在 `file://` 下对跨目录图片是禁的**（tainted canvas），所以像素归属 `claimed_bits_u16` 在**构建期服务端**算好，页面只 `drawImage`。
- **`grid_shape_hw` 要发点网格（AMB3R 是 392×518，即 h×w），不是 RGB 尺寸（968×1296）**，另发 `rgb_shape_hw` 与 `rgb_crop`。反过来，代码里的 `AMBR3R_SIZE` / `AMBR3R_INTERMEDIATE` 是 **w, h** 顺序 —— 这两套约定在同一个文件里并存，是这块最容易写错的地方。

### 便捷参数

`?source=amb3r` 直接进某个源；`?frame=N`、`?mask=ID` 同理。

---

## 3. GPT → SAM3.1 重写（2026-10-15）

用户要求删除迁入实现重写，并明确只保留 **GPT 出 label → SAM 出 mask**。
旧实现已备份到：
`/home/seer/wzt/backups/InstanceRecognition_Refactoring-gpt-sam-before-rewrite-20261015-162748.tar.gz`。
备份含原感知代码、相关测试、配置、依赖文件、Harness client 和本笔记，不含凭据或实验数据。

当前结构：

| 文件 | 职责 |
|---|---|
| `src/gateway.py` | 只读共享凭据，环境变量覆盖 `config.GATEWAY_ENV_FILE` 指定的文件；Harness 直接使用它 |
| `src/inference/label_frames.py` | 批次 RGB → GPT 类别 JSON；校验、有限重试、逐帧续跑 |
| `src/inference/sam31_segment.py` | SAM3.1 公共会话 API 与文本提示；生成独立二值 mask |
| `src/inference/gpt_sam31.py` | 标签与 RGB 配对，复用一个模型，逐帧发布 mask、预测 JSON 和预览 |
| `src/inference/io.py` | JSON 原子写、文件哈希、实验目录定位 |
| `src/inference/prompts/prompt_en.txt` | GPT 标签规则；保留整物体、成对物体和不确定性语义 |

删除了 `gpt_labels.py`、`runtime_env.py`、`artifacts.py`，旧单帧入口和闲置模型分支配置不再保留。没有检测器分支、box/point prompt、模型服务或自动分支选择。两个入口都通过 `--run` 归入同一个实验；不保留旧的 `--reuse-run`、`--wait-for-labels` 和分散输出参数。

### 运行

先生成标签：

```bash
.venv/bin/python -m src.inference.label_frames \
  --run perception-demo \
  --batch ../InstanceRecognition/runtime/batches/scene0000_00/target_500.json \
  --frames-dir artifacts/frames \
  --first 0 --count 20 --workers 3
```

`--run` 是实验名，不能是路径；`--batch` 提供 `scene_id` 和 `frame_indices`；`--frames-dir` 是场景图片目录的父目录；`--first 0 --count 20` 选择批次前 20 帧；`--workers 3` 同时处理 3 个独立请求。`--model` 默认取 `config.GPT_MODEL`；`--prepare-only` 只写配置，不调用网关。输入、模型或 prompt 改变时必须使用新实验名。

标签全部完成后生成 mask：

```bash
conda activate scene-understanding
taskset -c 0 python -m src.inference.gpt_sam31 \
  --run perception-demo --threshold 0.5 --max-masks 16
```

`--threshold 0.5` 是 SAM 输出概率阈值；`--max-masks 16` 保留最多 16 个得分最高的候选，所有原始候选的二值 mask 仍保存。可用 `--checkpoint` 指定权重。该命令必须由已安装 SAM3.1 的解释器运行；代码不会自动切换解释器或改 `sys.path`。当前使用独立的 `scene-understanding` conda 环境，安装和验证记录见第 8 节；`requirements-runtime.txt` 记录直接依赖。

产物固定为：

```text
artifacts/experiments/<run>/perception/
├── labels/  config.json、labels/、responses/、errors/、summary.json
└── masks/   frame_<原始帧号>/sam31/、summary.json
```

每帧 `sam31/` 包含 `predictions.json`、`raw_masks/`、`annotated.jpg`、`instance_id_map.png`。保留下游读取的 `mask_file`、`class_name`、`sam_prompt`、`instance_id`、`raw_candidates` 等字段。二值 mask 保留重叠，单 ID 图仅供显示。

### 续跑和验证边界

- GPT 检查 RGB、模型和 prompt；SAM 检查 RGB、标签、权重路径/大小/mtime、阈值与预算。缓存的 mask/预览文件还逐一核对哈希。权重使用 stat 指纹，不宣称是完整权重 SHA256。
- SAM 在临时目录生成完整帧，成功后才发布；失败帧不会留下可被误认为成功的目录。已完成帧可复用。
- 离线集成测试使用模拟网关和 SAM 公共接口，但实际运行两个入口、适配代码、PNG/JSON 写入和几何侧 mask 读取；覆盖重叠、预算、缓存损坏、失败续跑、会话关闭及本地输出路径。
- 本次执行 `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/ -q -p no:cacheprovider`：**222 passed，0.77 s**。两个入口分别在轻量解释器、SAM 解释器下的 `--help` 均成功；重写模块静态扫描无未使用导入、无检测器分支配置。
- 重写当时未执行真实网关或 SAM 推理；后续新环境已完成真实 SAM 权重加载与单图分割（第 8 节）。真实 GPT 请求及整个 AMB3R → GPT → SAM → Harness 链路仍未端到端验证。

## 4. 环境陷阱

- **大构建要 `taskset -c 0`**。网格光栅化与完整 publish 不加会 SIGSEGV（实测复现）。测试不用。
- **headless Chrome 截图**必须用新的 GL flag：
  ```
  --use-gl=angle --use-angle=swiftshader --enable-unsafe-swiftshader --allow-file-access-from-files
  ```
  `--use-gl=swiftshader` **已失效**（允许列表只剩 `egl-angle`），会导致 `CONTEXT_LOST_WEBGL`，3D 面板整块空白 + 角上一个破图图标 —— 那是截图命令的锅，**不是页面的 bug**，别去改页面代码。
- **`file://` 下父页面读不到 `http://` iframe 的 `contentDocument`**，跨源探针拿不到像素，只能分别截图。

## 5. 实验目录已恢复本地存放（2026-10-15）

`artifacts/experiments/` 现在是真实目录。V3/V4 已从 `experiments.backup/` 移回原位，backup 目录已不再存在；旧项目的实验没有移动或删除。

`artifacts/frames` 已于 2026-10-15 按用户要求从旧树移动到本树，现为真实目录，不再是软链接。原 `../InstanceRecognition/runtime/frames` 路径不再存在；mesh source manifest 的 20 个 `original_rgb_image` 引用及运行示例已更新为本树路径，历史实验调用日志保持原始记录；共 502 个文件、142530821 字节，同文件系统移动后逐文件 inode、大小及 mtime 均一致。AMB3R source manifest 中原来借用 experiments 软链接的 208 个路径已改为直接引用旧树上游文件，所以恢复本地目录后仍能读取原 mask。

新感知 CLI 通过实验名定位 `artifacts/experiments/<run>/perception/`，并拒绝通过软链接逃出 artifacts 根目录。不要再次把 experiments 整个目录链接到旧项目。

## 6. 构建 mask_project 页面

页面只剩一个源（AMB3R 逐帧深度），源切换器与 mesh 侧已删除。

```bash
cd InstanceRecognition
rm -rf site/mask_project/{amb3r,meta.js,meta.json,background_amb3r.js}
python3 -m pagegen.build_mask_project \
  --amb3r-source artifacts/sources/scene0000_00/scene_amb3r_v1 \
  --output site/mask_project --count 20
```

overwrite guard 容忍模板文件与 `background_amb3r.js`（可重建）。`meta.js` 的 `sources.amb3r.background` 登记背景；`amb3r_background_cloud()` 从 source manifest 的 `sim3_source` 反查 `sim3_ply`，**键缺失或文件不存在都降级为「无背景 + 警告」**——所以取键用 `.get()`，不是 `[]`。

## 7. 不进 git 的目录

`artifacts/`、`site/`、`pagegen/templates/` 全被 `.gitignore` 忽略（沿用旧树对 `src/web/frontend/` 的处理）。后果：

- **`git clean -fd` 会把 `pagegen/templates/` 删掉** —— 手写模板，删了就得重写。
- 这些目录里的改动**理由不进 git 历史**，只能靠注释、本文档、以及 `memory/` 留存。


## 8. 独立 SAM conda 环境（2026-10-15）

已创建 `scene-understanding`：`/home/seer/miniconda3/envs/scene-understanding`。
用途为 GPT 标签、SAM3.1 JPEG 文本分割、当前几何/Harness 代码及仓库测试；AMB3R 保持独立环境。
没有克隆旧 venv，也不再 editable 引用旧树 SAM 代码。原始图片已移动到本树 `artifacts/frames/`；继续共享权重文件，不复制模型权重。

| 项目 | 安装版本/处理 |
|---|---|
| Python / PyTorch / CUDA wheel | 3.12.15 / 2.10.0+cu128 / 12.8 |
| torchvision / NumPy / SciPy | 0.25.0+cu128 / 1.26.4 / 1.17.1 |
| SAM3 | 独立 wheel，`0.1.0+scene1`；上游源码和补丁来源见下文 |
| timm / huggingface-hub | 1.0.29 / 1.31.0 |
| setuptools | 固定 80.10.2，SAM 仍使用 `pkg_resources` |
| pytest | 仅用于验证当前仓库及 SAM 兼容补丁 |

未安装 ultralytics、YOLO 系列、OpenCV、scikit-image、Jupyter、notebook、TensorBoard、PyTorch3D、spconv 或 Open3D；没有安装 SAM 的 dev/notebooks/train extras。`einops`、`pycocotools`、`psutil` 是 SAM 模型实际 import 的依赖，不能只按其不完整的基础声明省掉。Torch 自带的 NVIDIA CUDA 运行库保留，无需为 SAM 另装 CUDA toolkit。

### 使用

在本项目根目录执行：

```bash
source /home/seer/miniconda3/etc/profile.d/conda.sh
conda activate scene-understanding
python -m src.inference.label_frames --help
python -m src.harness.run --help
taskset -c 0 python -m src.inference.gpt_sam31 --run perception-demo
```

`perception-demo` 必须已有第 3 节生成的 GPT 标签；输出为本树 `artifacts/experiments/perception-demo/perception/masks/`。`taskset -c 0` 延续本机 CPU 亲和性约束；代码不会自动切换到旧环境。已给 conda 环境设置 `PYTHONNOUSERSITE=1` 和新解释器的 `SAM31_PYTHON`；未设置后者时配置使用当前 Python，不再回退旧树。

### 可复现安装与来源

- `environments/scene-understanding/environment.yml`：conda 解释器和构建基础包。
- `requirements-runtime.txt`：直接依赖；`environments/scene-understanding/constraints.txt`：已验证的传递依赖固定版本。
- `environments/scene-understanding/sam3-init-state.patch`：唯一 SAM 本地兼容补丁。
- `environments/scene-understanding/sam3-provenance.json`：上游 commit、补丁及 wheel 哈希。
- `artifacts/environment/scene-understanding/`：conda explicit 清单、最终已安装包列表、源码文件哈希、构建/测试日志、源码快照、离线 wheels 和分割证据。

默认 conda 源当时 DNS 解析失败，使用清华 Anaconda main 镜像创建基础环境。pip 依赖从本机已有的、Python 3.12/Linux x86_64 标签匹配的 uv wheel 缓存重打包后离线安装，未复制旧环境的 site-packages。离线缓存约 7 GB，不属于环境依赖；保留供重建。

以下命令用于环境不存在时重建，已有环境不需再次执行：

```bash
conda env create -f environments/scene-understanding/environment.yml
conda activate scene-understanding
python -m pip install --no-index \
  --find-links artifacts/environment/scene-understanding/wheels \
  -r environments/scene-understanding/constraints.txt 'sam3==0.1.0+scene1'
python -m pip check
```

`--no-index` 限定本地缓存；`constraints.txt` 在这条离线重建命令中作为完整的固定版本安装列表。联网安装普通依赖可改用 `python -m pip install -r requirements-runtime.txt`，随后仍需安装上述修补后的 SAM wheel。

SAM 源码快照来自旧树的 `ThirdParty/SAM3`（2026-10-19 起该目录已移到仓库根的 `Submodule/SAM3`；commit `660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`），但运行时已独立。首次真实测试发现公共 `start_session` 总传入 `offload_state_to_cpu`，而 multiplex 的 `init_state` 不接受。补丁仅在模型不支持且值为 False 时省略该参数，显式请求不支持的 True 会报错；保留普通 SAM3 和 `**kwargs` 接口行为。没有改旧树源文件、模型算法或权重。版本加 `+scene1` 以免混淆。

保留的 `artifacts/environment/scene-understanding/sam3-source/` 已应用补丁；重建 SAM wheel 可执行：

```bash
python -m pip wheel --no-index --no-deps --no-build-isolation \
  --wheel-dir artifacts/environment/scene-understanding/wheels \
  artifacts/environment/scene-understanding/sam3-source
```

### 实测结果与边界

- 新环境运行 `python -m pytest tests/ -q -p no:cacheprovider`：**222 passed**。
- 单独执行 `python -m pytest environments/scene-understanding/verify_sam_compat.py -q -p no:cacheprovider`：**5 passed**。单独执行避免向轻量离线测试进程导入 torch。
- `python -m pip check`：通过；SAM 包来自新 conda 环境的 site-packages。
- RTX 5070 Ti CUDA 张量运算通过，真实加载 `weights/sam3.1_multiplex.pt` 并分割 `scene0000_00/frame_000000.jpg`，手动文本提示 `office chair`：**1 个 mask，140952 像素**。图像检查定位到左侧办公椅。模型加载约 11.94 s，含加载总计约 14.03 s；单图首次验证，不是性能基准。
- 推理没有 import cv2/skimage；没有调用 GPT 网关。SAM 验证通过不等于完整场景理解链路验证通过。
- 上游 builder 先把完整权重传给局部 tracker，会打印很长的 missing/unexpected keys；随后完整模型再次加载同一权重，无最终模型 missing/unexpected keys 提示。本次保留原始日志，没有修改上游加载逻辑或隐藏警告。

实际证据：`artifacts/environment/scene-understanding/smoke-v2/report.json`、`predictions.json`、`annotated.jpg`、`raw_masks/`；首次失败记录为 `smoke.log`。环境自检产物放在 `artifacts/environment/`，业务实验继续固定在 `artifacts/experiments/`。

复跑真实 SAM 单图验证（`--output` 必须是尚不存在的新目录）：

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
TRITON_CACHE_DIR="$PWD/artifacts/environment/scene-understanding/triton-cache" \
taskset -c 0 python environments/scene-understanding/verify_sam.py \
  --image artifacts/frames/scene0000_00/frame_000000.jpg \
  --prompt 'office chair' \
  --output artifacts/environment/scene-understanding/smoke-repeat
```

`--prompt` 是环境自检的手工提示，不代替正式 GPT 标签阶段；`HF_HUB_OFFLINE=1` 保证使用已有权重。复跑不需要 OpenCV、检测器或视频解码依赖。

## 9. GPT/SAM → 几何 → Harness：assembler 接手记录（2026-10-16）

当前推进到 `src/geometry/assemble_perception_source.py` 及其测试
`tests/test_assemble_perception_source.py`。assembler 已实现并有两帧实际组装产物；完整 AMB3R → GPT/SAM → Harness 链路仍未验证接通。接手时只阅读代码、核对已有产物并复跑测试，没有重新调用 GPT/SAM，也没有执行新的 lifting 或 Harness 实验。

### 已实现的职责

```text
artifacts/experiments/<run>/perception/
├── labels/  GPT 标签、批次清单、RGB 哈希
└── masks/   SAM predictions.json、原始二值 mask
       │
       ▼
src/geometry/assemble_perception_source.py
├── 核对 sequence_index、source_frame_id
├── 核对 SAM 记录的标签 SHA256 与实际标签文件
├── 核对批次、标签、SAM 中的 RGB SHA256 与当前图片
├── 从 artifacts/frames/<scene>/ 重新定位已迁移的 RGB
├── 要求实例计数一致、每帧至多 32 个实例、序号从 0 连续
└── 输出 source 清单、背景显示资产和组装报告
       │
       ▼
amb3r_mask_source.py                        生成逐帧三维点、mask_bits 与逐帧 rgb
       │
       ▼
Harness                                    读取几何并进行跨帧关联
```

assembler 用相对路径原地引用 RGB、预测 JSON 和 mask，不复制感知数据。输出为 `manifest.json`、`manifest.js`、`assets/background.js`、`assembly_report.json`。背景 PLY 会编码为查看器可加载的 JS，manifest 的 bounds 对应背景范围。

**assembler 不做 lifting，不产生逐帧 NPZ，也不设置 `lifting_policy`。** 报告中的 `policy=gpt_sam31_perception_v1` 仅说明组装策略，不能当作 Harness 接受的几何策略。默认背景仍来自相邻项目的 `Video2PointCloud/output/amb3r/vo_uniform500/pointcloud_gt_corrected_rgb.ply`，是 GT 修正后的 AMB3R 点云，不代表独立、无 GT 的 AMB3R 几何验证。

### 当前实际产物及验证

- 组装产物 `artifacts/sources/scene0000_00/gpt_sam31_uniform500_v1/`：**20 帧、207 个 mask、492555 点背景**。以其中 `assembly_report.json` 为准；名称带 uniform500 不表示已组装完整 500 帧（感知侧确有 500 帧，需要时用 `--count` 扩）。
- lift 产物 `artifacts/sources/scene0000_00/scene_amb3r_v1/`：`amb3r_per_frame_v1`，20 帧、207 mask、逐帧 npz、`assets/background.js`、`projection_report.json`。manifest 中所有路径（逐帧 RGB、预测 JSON、mask、annotated）**均可从输出目录解析**，已逐帧核验。
- Harness 端到端：`--frames 4 --window 2 --prepare-only` 通过，**46 个观测 / 4 帧**。
- 本次测试：`amb3r_mask_source` 7 项、`assemble_perception_source` 16 项、全仓 **199 passed**。测试使用小型模拟产物，不调用 GPU 或网关。

### lifting 与感知链路的两个真实缺陷（已修）

这两个都不报错，只是让 source **看起来完好但用不了**，只有真跑一遍才会暴露：

1. **帧路径从不 rebase。** `amb3r_mask_source` 把上游帧条目逐字复制进自己的 manifest，于是 `mask_file` / `predictions_json` 是相对**上游**目录的路径，从新 output 解析全部落空（Harness 报 `KeyError`，页面报缺文件）。已改为逐键相对 output 重算。注意 `original_rgb_image` 不能这样做——上游记的是重构前的位置，那个目录已经不存在，所以它按 `--frames-dir` 重新定位（与 assembler 同一约定）。
2. **颜色键名与语义都不对。** npz 里写的是 `colors`，而 `observations.py` 读 `z['rgb']`——直接 `KeyError`。而且即便改名，原值是实例 hue 的**合成色**：这个颜色会随观测一起送给模型，等于给出图像并不支持的证据。已改为按 mask 的同一套 resize→crop 从 RGB 帧采样**真实像素色**（crop 偏移错一列就会让每个点带上旁边像素的颜色）。

顺带修掉：`crop_box` 那行的 `align_mask.__doc__ and [...] or [...]` 是永真表达式（两支完全相同），改成实际值。

### lifting policy 收敛为单一

`ACCEPTED_LIFTING_POLICIES` 现在只含 `amb3r_per_frame_v1`。**删除的模块**：`mesh_mask_source`、`mesh_visibility`、`prune_scene_source`、`cluster_prune`、`run_incremental_objects`、`camera_calibration` 及对应测试。

- 已发布的 harness-v3/v4 **页面仍可查看**，但其 source 不再能由本树重建（policy 已不在白名单，`run.py` 会直接拒绝）。
- `read_mesh` 与 `run_incremental_objects.ply` 都随模块删除；后者原本被 `build_site.py` 用来导出物体 PLY，现已内联进 `build_site.py`。
- `camera_calibration` 的唯一消费者是两个被删的模块，故一并删除；`artifacts/geometry_inputs/` 下的 `cameras_rgb.json` 保留为历史产物。
- `mesh_mask_source._instance_masks` 被 `test_perception_pipeline` 借用，已作为 `backproject_masks.instance_masks` 保留（它验证的是几何读独立 mask 而非显示用 ID map，与本次清理无关）。

### 复跑测试

```bash
taskset -c 0 .venv/bin/python -m pytest tests/ -q          # 199 passed
taskset -c 0 .venv/bin/python -m pytest tests/test_amb3r_mask_source.py tests/test_assemble_perception_source.py -q
```

---

## 10. 跨窗再识别失效与 reid_gate 软门禁（2026-10-20）

### 症状：一个桶在 5.6 米外被续上了旧身份

`harness-v5-amb3r-150frames-20261016-1550` 的 **W25（帧 120–124，base_version 24）**
把 8 条观测 assign 给了 `Object25`（wastebasket）。理由原文：

> These masks cover the white lidded wastebasket at the left edge; repeated
> overlapping masks in the same frames are alternate segmentation views of that bin.

**这句话只回答了一半问题。** 它证明了"这几张 mask 是同一个桶"，没证明"这个桶是
Object25"。事实是：Object25 此前的 39 条观测全在 `(3.68, 1.28, 0.23)`，
这 8 条在 `(6.5–6.8, 5.8–6.0, 0.3–0.5)`，**相距 5.6 m，中间 45 帧零观测，隔着房间**。
同窗口 `Object54` 犯了同样的错（3.55 m），另在 W16 有一次 `Object29` 的
2.01 m 跳跃（床头两侧两个柜子被合并）。

### 关键实测数字（改阈值、改逻辑前先看这里）

全部 1,169 次已提交 assign 的「观测质心 → 目标物体历史最近观测质心」距离：

```
中位 0.033 m   p90 0.213 m   p95 0.337 m   p99 1.973 m
> 3 m: 11 条（全是真错）    > 5 m: 8 条（全是 W25 的 Object25）
```

排序后 **第 14 名 1.97 m 与第 15 名 1.08 m 之间有 0.9 m 空档**。阈值落在这个区间内，
拦什么完全确定，不靠运气。用 W1–W15（无错窗口）标定得 `p95×3 = 0.97 m`，
拿到 W16–W30 上评估，14 条真实错误全中、零误拦 —— 这个数就是这么来的，不是拍的。

**不要用绝对值卡所有物体。** 地毯（Object26）自身观测间距 p95 是 0.253 m，
正常归属能到 1.08 m；挂画（Object40）是 0.658 m。绝对阈值会误伤它们。
所以阈值取 `max(1.0 m, 20 × 该物体自身观测间距 p95)`。
实测中倍数项从未生效（出错物体的自身尺度都在 3–9 cm），**1.0 m 是主参数，
20 倍是给未见场景留的安全阀**。

### 污染会自我放大 —— 已跑坏的 run 不能当基线

W25 提交后，`export_version_objects` 把两个相隔 5.6 m 的点簇一起拟合，Object25 的
OBB 从 `0.21×0.33×0.24` 变成 `4.74×3.80×0.63`；Object54 的 `size_y` 从 0.40 涨到 3.92。
此后 W26–W30 每窗有 **1/3 的 `spatial_candidates` 含 Object25**（W25 时是 0/8）。

后果：**任何基于「已提交点云」的几何判据，在污染落盘后会自我认可**（观测就在自己里面，
距离算出来是 0）。所以门禁用 `self.state['assignments']`（窗口起点的 base_version），
绝不能用提案自己的结果。同理，v5 那个目录**不能再用来验证门禁**。

### 为什么模型当时看不到问题

三层，逐层剥：

1. **`spatial_candidates` 被 top-3 截断。** `build_spatial_context` 对每条观测算了
   全部 68 个物体的距离（W25 共 3,944 个），只保留最近 3 个 —— **丢弃 95.6%**。
   Object25 对那 8 条观测的距离排在第 4 名之后，**58 条观测里一次都没出现**。
   模型看到的候选是 Object60（门）、Object61（门垫）、Object51（墙柜）。
2. **prompt 亲手把这个信号废掉了。** 原文同时写着
   "This is a directional proximity hint, not identity confidence" 和
   "Do not rely only on a nearest spatial candidate"。合起来的效果是：
   候选既不能证明"是"，也不能证明"不是" —— 模型选一个**不在候选里**的物体，完全合规。
3. **档案里没有"多久没见"。** `tools.listing()` 只给 id / label / created_at /
   count / observation_ids / representative。位置在另一个块（`global_object_geometry`）。
   而 `observation_count: 39`、`created_at: 19` 给人的暗示是"熟物体、确定"，
   **恰好与真相相反**。

### 采用的方案：软门禁（不是否决）

`src/harness/reid_gate.py`，三个纯函数，不读文件、不依赖项目内其他模块：

- `self_scale(centroids)` — 每条观测到最近同伴的距离，取 p95（<6 条回退到 floor）
- `threshold_for(centroids)` — `max(1.0, 20 × self_scale)`
- `verify_reidentification(actions, prior_assignments, centroids)` — 只筛 `assign`

接线在 `Tools.reidentification_checks` + `propose_update` / `commit_update` 两处。
触发时返回 `{'status':'verification_required', 'proposal_id':..., 'checks':[...]}`，
**`self.pending` 保留不清空**。

三个容易做错的地方：

- **必须保留 pending。** `run.py` 的安全网是
  `final = step==max_calls-2 and tools.pending is None`；打回时若清掉 pending，
  预算将尽时会误催模型"立刻交提案"。
- **放行判定要用「历史观测 ∩ evidence」，不能用 evidence 全集。**
  `build_window_content` 在窗口开始前就把**当前窗口**所有观测塞进了 `evidence`。
  不限定历史，门禁永远放行。`inspect_object` 会把它选中的代表观测写进 `evidence`
  （`tools.py` 的 `self.evidence.add(oid)`），所以判定为
  `history[object_id] & self.evidence`。
- **必须能降级。** `run.py` 预算警告发出时置 `tools.gate_relaxed=True`，
  门禁从此刻起直接放行。宁可漏一次核对，也不能让整个窗口因为叫图而超预算
  （超预算是 `raise RuntimeError`，整窗失败）。

### 重放结果（v5 run，29 个已提交窗口）

```
W16  action#10  Object29   2.01 m  (阈值 1.0,    历史 49 条)
W25  action#7   Object25   5.56 m  (阈值 1.0,    历史 39 条)
W25  action#8   Object54   3.55 m  (阈值 1.408,  历史  9 条)
```

三个都是真错，零误拦。W18 那条地毯 1.08 m 被正确放行（阈值被自身尺度抬到 5.06 m）。
注意 Object54 的阈值是 1.408 而非预估的 1.84 —— 因为用它是**污染前**的历史，
那时比后来更紧凑。这正是"必须用 base_version"的实证。

### prompt 改动（因此老目录不能再续跑）

`src/harness/prompt.txt` 的 `TRANSACTION PROTOCOL` 末尾加了一段，说明
`verification_required` 不是错误、提案保留、需 `inspect_object` 后再提、结论开放。
哈希 `c32e5fd…` → `e9459de…`。`run.py` 比对 `config.json` 里的 `prompt_sha256`，
**改 prompt 就必须开新 run 目录**，否则会得到半新半旧的数据。

那段里最针对的一句是："a history of dozens of observations does not itself
confirm that the current window shows the same thing" —— 直接冲着
`observation_count: 39` 那个误导性暗示去的。

### 已知盲区（不要以为门禁解决了全部）

1. **只拦得住"跳变"，拦不住"旁边那个"。** 同房间两个桶相距 0.3 m，
   误合并时距离远低于 1 m 下限，门禁完全看不见。这是最可能的下一形态。
2. **靠质心，不靠形状。** 细长物体被错误合并时质心可能落在中间，距离反而不大。
3. **要根治得补 `split` 动作。** 现在动作集是 `create/relabel/assign/merge/defer/repair`，
   **没有 split**。W25 模型看到两个桶都需要归属，可选动作只有三个：塞给旧物体（错）、
   新建两个同名物体（像制造重复）、搁置（浪费证据）。**选第一个是最像正确的**——
   这个错误一半是动作集缺口逼出来的。
4. **`defer` 之后没有"重新认领"的路径。** 被 defer 的观测只能一直挂着。

### 测试与验证

```bash
taskset -c 0 .venv/bin/python -m pytest tests/ -q          # 226 passed
taskset -c 0 .venv/bin/python -m pytest tests/test_reid_gate.py tests/test_reid_gate_tools.py -q
```

- `tests/test_reid_gate.py` — 14 个纯几何测试，手写数据（符合本项目的测试风格，
  不引外部 fixture）。含 `test_verdict_is_stable_across_the_threshold_band`，
  守的是"空档还在"，不是 1.0 这个数。
- `tests/test_reid_gate_tools.py` — 7 个接线测试，走 `Tools.call`。
  注意 `state.py` 要求**每个可见观测都必须有归属**，所以测试里每个窗口只能放
  一条待归观测，否则会触发 "Every visible observation needs assigned/defer/repair status"。

写测试时踩的坑：`inspected_objects` 字段是多余的 —— `inspect_object` 已经把代表观测
写进 `evidence`，判定直接查 `history & evidence` 即可，不要发明第二套账本。

### 下一轮该看什么

v6 run（`harness-v6-amb3r-150frames-20261020-*`）跑完后，**不要预期 W25 会重犯同样的错**
（模型有随机性，prompt 也变了）。该看的是：

1. **触发率** —— `calls/*_tool*.json` 里 `verification_required` 的条数。
   期望 1–2% 的 assign。远高于此说明阈值过紧。
2. **改判率** —— 被打回的 action，模型看完 `inspect_object` 后是否改变了决定。
   **这是判断整套方案有没有价值的唯一指标。** 如果看完全部原样重提，
   说明门禁只是让人多点两次鼠标，要回去重设计而不是调阈值。
3. **有没有窗口因门禁超预算而失败。** `gate_relaxed` 理论上保证不会，
   但要确认每个窗口都写出了 `versions/NNNN`。

---

## 11. `pending` 只增不减，每个窗口都重发一遍（2026-10-23 记录，**2026-11-04 已解决**）

> **结论先行（2026-11-04）**：`Tools.listing()` 不再输出 `pending` 键。`defer` 仍是合法的
> 逐观测状态，`state.json` 里一条记录都没删，只是不再每窗广播。
> 在 v11 末版状态上实测：`listing` 从 165 KB 降到 **15.9 KB（-90%）**。
> 改的是 `src/harness/tools.py` 的 `listing()`，加 prompt 里 `DEFER` 那段的一句承诺。
> 下面保留原始记录，因为「为什么可以删」的判断依据在这里。

**症状**：harness 每个窗口的首条 user 消息里带着 `tools.listing()`，其中 `pending` 是
`object_id is None` 的全部观测记录。这份清单只进不出，到 150 帧跑完时已经占掉输入的大头。

### 实测（`harness-v6-amb3r-150frames-20261021-0358`，150 帧 / 30 窗）

| 窗口 | pending 条数 | JSON 字节 | objects 字节 |
|---|---|---|---|
| v01 | 0 | 2 | — |
| v05 | 43 | 24,085 | — |
| v10 | 178 | 108,929 | — |
| v15 | 322 | 187,532 | — |
| v20 | 407 | 234,185 | — |
| v25 | 554 | 324,027 | — |
| v30 | 664 | **377,285** | 3,490 |

终局 664 条 = 535 条 `defer` + 129 条 `exclude`（旧名 `repair`）。
每一条都带完整的 `reason` 与 `evidence` 列表。相比之下 prompt 全文只有约 10 KB。

### 三个独立的问题，别混着当一个问题修

1. **名字与内容不符。** `listing()` 里 `pending` 的定义是 `object_id is None`，
   但 `exclude` 是一条**已经做完的决定**，永远不会再变。它和 `defer` 一起放进 "pending"，
   语义上就是错的。
2. **体量失控。** 664 条里 535 条是历史 deferral，模型在每个窗口都要重新读一遍。
   按 150 帧的曲线外推，更长序列会把首条消息撑到 prompt 的几十倍。
3. **deferral 从未被回收。** 这是最值得注意的一条：535 条 defer 里，被后续窗口
   重新认领的是 **0 条**。prompt 一直用「未归属的观测以后还能被认领」给 defer 辩护，
   在这个规模下这句话从未兑现过。不是模型不想认领，是它在一份 37 万字节的清单里
   找不到任何一条。

### 可能的修法（都没做）

- **拆分**：`pending` 只留 `status == 'defer'`，`exclude` 单独一个键，或干脆不进首条消息。
- **瘦身**：首条消息只给 `(obs_id, status, decided_at)`，`reason`/`evidence` 靠
  `inspect_observations` 去取。这一条单独做就能砍掉九成体积，且不改任何语义。
- **按需**：只列本窗口 `spatial_candidates` 命中过的历史 deferral，而不是全部。
- **回收**：让 deferral 有一个过期或可检索的通道，否则「以后还能认领」只是说法。

### 已经做完、不要重复做的事

`list_objects` 工具已于 2026-10-23 从 `tools.py` 的 `TOOLS` 和 `call()` 中删除
（它返回的就是这份 `listing()`，而 state 在提交前不变，所以窗口内调它恒等于重读已有内容）。
文件在 `tools.py` 顶部留了注释。**`Tools.listing()` 本身没有删**，`inputs.py` 与
`inspect_object` 仍在用它——所以下面的问题与工具是否存在无关，它出在每窗必发的输入里。
