# ObjectGeneration：使用 SAM 3D Objects 生成可编辑 Isaac Sim 资产

本目录封装了一套从单张图片生成独立三维物体，并转换为 Isaac Sim 可加载、可编辑、可选物理仿真的 USD/USDA 资产的流程。

```text
RGB 图片 + 文本 prompt / 浏览器点击选物
          │
          ▼
SAM3.1 文本分割 / 点击分割并人工确认 → 二值 mask（也可直接提供已有 mask）
          │
          ▼
SAM 3D Objects
          │
          ├── mesh.glb   三角网格，带顶点颜色
          └── splat.ply  Gaussian Splat 结果
          │
          ▼
mesh_to_usd.py
          │
          └── object.usda  可编辑 USD 几何与显示材质
          │
          ▼
configure_usd_physics.py（可选）
          │
          └── 带刚体、质量、碰撞和物理材质的 USD 资产
          │
          ▼
Isaac Sim 加载、渲染与物理验证
```

> 当前入口支持三种选择物体的方式：`--image` + `--prompt` 自动文本分割；`--image` + `--mask` 使用已有 mask；只给 `--image`、同时省略 `--prompt` 和 `--mask` 时，进入浏览器点击分割，人工确认预览后继续。`--prompt` 和 `--mask` 不能同时使用。

如需一条命令完成生成、转换、物理配置和 Isaac Sim 验证，参见 [7.1 端到端命令](#71-一条命令完成第-3456-阶段)。

GPT-6 + Blender 外观精修的架构、开发阶段与验收要求，参见 [外观精修开发方案](docs/appearance_refinement_plan.md)（完整流水线仍在开发）。

开发进度和已可执行的 M0 能力探针、M1 基线、M2 照片投射、M3 反馈与确认命令，参见 [外观精修开发进度](docs/appearance_development_status.md)。M2 生成独立候选；M4 已支持保留物理参数的独立贴图打包、迁移后 Isaac 显示与掉落验证。M3 已支持受限相机反馈；当前页面仅展示候选对比、六视图及中文说明，暂不提供人工修正或确认操作。M5 已提供独立精修入口及端到端可选精修阶段，参见第 8 节；原资产不覆盖。

---

## 1. 目录结构

```text
ObjectGeneration/
├── src/objgen/                      # 核心代码：一个包，全部可按 python -m 运行
│   ├── config.py                    # 路径的唯一来源（项目根、venv、外部依赖）
│   ├── layout.py                    # 运行目录约定（原 artifact_layout.py）
│   ├── handoff_io.py                # 读交接文件、选物体、输出目录与 objects.json 汇总
│   ├── run_pipeline.py              # 顺序调度、环境切换、日志与总报告
│   ├── generate/                    # 分割与 SAM3D：generate, sam3d_smoke, sam31_mask, sam31_click(+html), sam3d_offload
│   ├── dualcraft/                   # DualCraft 路线：输入准备、导入、URDF→物理、关节检查
│   ├── to_usd/                      # mesh_ops, mesh_to_usd, configure_usd_physics
│   ├── verify/                      # Isaac Sim 加载渲染检查、掉落与碰撞检查
│   ├── planning/                    # 尺寸估计、物理/外观估计、生成器路由
│   ├── handoff/                     # build_assembly_handoff：生成结果 → 下游交接
│   ├── appearance/                  # 外观精修支线（照片投射、GPT 反馈、交付）
│   └── tools/                       # 网格预览、DualCraft 结果拉取与审阅
├── scripts/                         # 只有启动脚本
│   ├── run_pipeline.sh              # 主线入口 → objgen.run_pipeline
│   ├── run_generate.sh              # 交接文件 / 单图 + mask → SAM3D 网格
│   ├── run_sam3d.sh                 # 只跑 SAM3D 生成（旧的单图入口）
│   ├── refine_appearance.sh         # 外观精修入口
│   └── server/                      # 在 DualCraft 服务器上单文件运行的脚本（不 import 本包）
├── pagegen/                         # 页面构建器与模板
├── tests/                           # unittest
├── configs/ assets/ docs/
└── output/{分类}/{物体名}/{启动时间戳}/   # 产物（不入库）
```

第三方源码仓不在本目录下，统一放在仓库根的 `../Submodule/`（见根仓 `.gitmodules`）：
`sam-3d-objects`、`dual-craft`、`mcp-for-blender`、`pytorch3d`、`dinov2` 等。`.runtime_config/` 只留 pip 解出来的运行时依赖（MoGe、utils3d、gsplat、torch271、wheels、sam3d 配置）。

所有模块统一以 `PYTHONPATH=src python -m objgen.<模块>` 运行；`scripts/` 下的四个 `.sh` 就是这样调用的。


上游源码位于仓库根的 `Submodule/sam-3d-objects/`，当前配置使用的 checkpoint 位于外部目录 `/home/seer/wzt/weights/sam3d/`，配置文件已经通过 `.runtime_config/sam3d/pipeline.yaml` 进行本机路径适配。

---

## 2. 环境与显存注意事项

当前本机配置：

- Python 3.11 环境：`.venv-sam3d/`
- PyTorch 2.7.x + CUDA 12.8
- GPU：RTX 5070 Ti，16 GB 显存
- CUDA 工具链：`/home/seer/wzt/tools/cuda-12.8`
- 上游 SAM3D 版本：`f91db411c50efee93d8db7aeb323885650f6f722`

官方配置的显存要求高于当前机器，因此推荐始终使用 `--low-vram`。`scripts/run_sam3d.sh` 已经统一设置了本机需要的兼容参数，包括：

- OpenBLAS/OpenMP 单线程；
- PyTorch expandable CUDA memory segments；
- PyTorch SDPA attention backend；
- 默认 CPU affinity；
- SAM3D 阶段模型的 CPU offload 支持。

如果普通的 `--low-vram` 仍然无法加载，可以进一步尝试 `--sequential-condition`，它会将 condition encoder 与 generator 分阶段启用：

```bash
./scripts/run_sam3d.sh \
  --low-vram \
  --sequential-condition \
  --image /absolute/path/image.png \
  --mask /absolute/path/mask.png \
  --output output
```

默认加载方式在当前 16 GB 显存机器上曾出现 DINO 初始化阶段 CUDA OOM，因此不建议作为常规推理方式。

---

## 3. 调用 SAM3D 生成物体

### 3.0 从交接文件生成（`run_generate.sh`）

读取场景理解交出的 handoff 目录，用每个物体的关键帧 RGB + mask 生成 SAM3D 网格。模型只加载一次，逐个物体生成；单个物体失败只记录、不中断整批。

```bash
# 交接文件：默认跑全部非结构物体，也可用 --objects 指定
./scripts/run_generate.sh --low-vram --sequential-condition \
  --handoff ../InstanceRecognition/artifacts/handoff/scene0000_00/gt \
  --objects obj_0018,obj_0032,obj_0033

# 单图 + mask：自动包装成只含一个物体的 handoff，走同一条路径
./scripts/run_generate.sh --low-vram --sequential-condition \
  --image photo.jpg --mask mask.png --label bottle
```

常用参数：`--view primary|secondary|third` 选关键帧（默认 primary）；`--dry-run` 只打印计划不加载模型；`--force` 忽略已有结果重跑；`--output` 改输出根目录。

输出与 handoff 一一对应，目录只由 `scene_id / handoff 名 / object_id / 视角` 决定，没有时间戳：

```text
output/scene0000_00/gt/
├── manifest.json        # schema assets/v1、来源 handoff 及其哈希、生成器配置、各状态计数
├── objects.json         # handoff 中每个物体一条：success / failed / skipped / pending
└── objects/obj_0032/primary/sam3d/
    ├── mesh.glb  splat.ply
    ├── pose.json        # SAM3D 布局估计；rotation 可用，scale/translation 非米制
    └── report.json      # 输入 rgb/mask 的 sha256、输出文件 sha256、耗时、显存
```

重跑时，输入、配置哈希、seed 都没变且输出文件哈希仍与报告一致的物体直接复用；任一项变了就清掉旧产物重新生成。`objects.json` 由各物体的 `report.json` 重新汇总，中途中断后再跑一次即可补齐。网格保持 SAM3D 的归一化尺寸，转 USD、加物理仍用下面的阶段 4、5。单图模式的包装 handoff 放在 `output/.handoffs/<图片名_哈希>/single_image/`。

查看结果：`.venv-sam3d/bin/python pagegen/build_probe_page.py output/scene0000_00/gt`，然后打开 `site/probe/index.html`。

### 3.1 输入文件

推荐输入一张 RGB 图片和要生成物体的文本 prompt，无需手动调用 SAM3.1 或传递 mask 路径：

```bash
./scripts/run_sam3d.sh \
  --low-vram --sequential-condition \
  --image assets/xuebi/xuebi.jpg \
  --prompt 'bottle' \
  --output output \
  --seed 42
```

各参数含义：

| 参数 | 含义 |
| --- | --- |
| `--low-vram` | 启用 SAM3D 低显存模式，将暂时不用的模型阶段移到 CPU，运行当前阶段时再移回 GPU，以降低显存占用。 |
| `--sequential-condition` | 在低显存模式下进一步分开运行条件编码器和生成器，减少两者同时占用显存；必须与 `--low-vram` 一起使用。 |
| `--image assets/xuebi/xuebi.jpg` | 指定输入图片。SAM3.1 在这张图片中分割目标，SAM3D 使用同一图片和生成的 mask 重建物体；相对路径以执行命令时的工作目录为基准。 |
| `--prompt 'bottle'` | 指定交给 SAM3.1 的物体文本描述，示例表示“瓶子”。程序自动生成并传递 mask，无需再指定 `--mask`；两个参数不能同时使用。 |
| `--output output` | 指定输出根目录；程序自动追加 RGB 文件名（不含扩展名）和启动时间戳，例如 `output/xuebi/20261009_162504_317778/`。每次启动创建新目录，不覆盖上次运行。 |
| `--seed 42` | 设置 SAM3D 三维生成阶段的随机种子，默认值为 `42`，用于尽量复现生成结果；不控制 SAM3.1 分割阶段的随机性，也不保证跨环境逐位一致。 |

推荐使用英文物体名称或简短描述，例如 `bottle`、`red shopping basket`、`table`。流程先在独立 Python 进程运行 SAM3.1，保存 mask 后退出并释放该进程的 GPU 资源，再加载 SAM3D。不会调用 GPT 或在线分割 API。

若同一 prompt 返回多个物体，默认选取**置信度最高的非空实例**，不合并多个实例。所有候选 mask 都保存在 `sam3d/sam31/candidates/`，分数和边界框见 `sam3d/sam31/segmentation.json`；可以通过 `--sam31-instance-index 1` 改选第二个候选（按置信度降序，索引从 0 开始）。同分时按模型返回顺序排列。置信度不保证符合空间指代，多个同类物体时应检查 `sam3d/sam31/mask_overlay.png`。

默认阈值为 `--sam31-threshold 0.5`。没有有效分割或索引超出范围时会停止，不启动 SAM3D；错误写入 `sam3d/report.json` 和 `sam3d/sam31/segmentation.json`，详细日志在 `sam3d/sam31/run.log`。可调整 prompt 或阈值后重新执行。

本机默认复用以下资源，可通过命令行覆盖：

| 参数 | 默认路径 |
| --- | --- |
| `--sam31-python` | `../InstanceRecognition/.venv/bin/python` |
| `--sam31-repo` | `../Submodule/SAM3` |
| `--sam31-checkpoint` | `../weights/sam3.1_multiplex.pt` |

默认路径相对于本项目位置计算，不依赖启动目录。SAM3.1 使用其独立环境，SAM3D 继续使用 `.venv-sam3d`。当前 SAM3.1 checkout 的公共 `start_session` 与 multiplex 参数不匹配，因此 worker 沿用 InstanceRecognition 的 `init_state` / session 注册方式。

**点击选择物体：省略 `--prompt` 和 `--mask`。** 例如图片中有多个瓶子，可以直接点击目标瓶子：

```bash
./scripts/run_sam3d.sh \
  --image assets/Beverages_And_snacks/5.jpg \
  --low-vram --sequential-condition \
  --output output --seed 42
```

参数含义：`--image` 指定待点击的 RGB 图片；两个低显存开关、输出根目录和随机种子的含义与上表一致。这里不需要新增交互开关，也不需要提供 mask 路径。

1. 程序尝试打开本机默认浏览器，并在终端打印 `SAM31_INTERACTIVE_URL http://127.0.0.1:端口/会话路径/`。若浏览器未自动打开，复制完整链接到本机浏览器；等待页面提示模型加载完成。
2. **左键点击目标内部添加正点**；SAM3.1 自动执行分割，绿色区域表示选中的物体。一个点不准确时可继续补点；**右键点击误选区域添加负点**，也可以在侧栏切换正点/负点模式。
3. 可以撤销最后一点、清空重选、切换查看原图与绿色叠加预览。每次修改点集都会重新分割，当前结果出来前不能确认。
4. 检查绿色区域后，点击 **“确认分割，继续生成”**。确认后保存 mask，关闭 SAM3.1 进程并释放其 GPU 资源，然后自动运行 SAM3D；使用 `run_pipeline.sh` 时还会继续第 4、5、6 阶段。
5. 点击 **“取消本次任务”**，或在确认前关闭/刷新页面，会取消本次任务；不会运行 SAM3D，入口返回非零退出码并保留日志。浏览器关闭通知若未送达，可在终端按 `Ctrl+C` 结束；模型加载或分割正在执行时，取消可能需要等本次计算结束。

点击模式选择的是同一个物体，可用多个正点和负点修正，不会把每个点分别生成一个资产。页面缩放不会改变原图坐标。`--sam31-threshold` 和 `--sam31-instance-index` 只控制文本模式的候选选择，在点击模式中不使用。SAM3.1 的 Python、源码和权重路径参数对两种分割模式都有效。

页面服务只监听本机 `127.0.0.1`，无需联网或安装额外 Web 框架。若希望自己打开链接，可在启动命令前加 `SAM31_NO_BROWSER=1`；这只关闭自动打开浏览器，仍然需要人工点击与确认。

已有 mask 时也可继续输入：

- 一张 RGB 图片；
- 一张与图片高宽完全一致的灰度或二值 mask。

mask 的判断规则是：

```text
mask 像素值 > 0：物体区域
mask 像素值 = 0：背景区域
```

当前脚本会检查图片和 mask 的空间尺寸是否一致。mask 应尽量只包含目标物体，不要包含大面积背景、地面阴影或其他物体；物体内部需要保留的孔洞也应在 mask 中正确表达。

### 3.2 已有 mask 和其他运行方式

在本目录执行：

```bash
./scripts/run_sam3d.sh \
  --low-vram \
  --image /absolute/path/to/image.png \
  --mask /absolute/path/to/mask.png \
  --output /absolute/path/to/output \
  --seed 42
```

例如：

```bash
./scripts/run_sam3d.sh \
  --low-vram \
  --image assets/xuebi/xuebi.jpg \
  --mask output/xuebi/mask.png \
  --output output \
  --seed 42
```

如果使用仓库内置的 kidsroom 示例，可以省略 `--image` 和 `--mask`：

```bash
./scripts/run_sam3d.sh \
  --low-vram \
  --output output
```

仅测试模型加载、不执行推理：

```bash
./scripts/run_sam3d.sh \
  --low-vram \
  --load-only \
  --output output
```

`--load-only` 只检查 SAM3D 加载，会跳过 SAM3.1（包括点击页面）和图片/mask 校验。自定义 `--image` 正常推理时，省略 `--prompt` 和 `--mask` 就进入点击模式；完全省略图片及两种提示时仍使用内置 kidsroom 示例。

### 3.3 运行目录与三个阶段的产物

点击模式的 `sam3d/sam31/interaction.json` 记录归一化点坐标（`x/y` 范围 `[0,1]`，`label=1` 正点、`0` 负点）及操作历史。确认后才写入 `mask.png`、`mask_overlay.png`；`segmentation.json` 记录 `mode: points`、`user_confirmed: true`、最终点集、确认时间及 mask 哈希。取消时记录 `status: cancelled`，不输出可供后续生成使用的 mask。

每次启动都保留 `assets` 下的分类与物体路径，再添加启动时间戳。例如 `assets/Beverages_And_snacks/1.jpg` 对应 `output/Beverages_And_snacks/1/{启动时间戳}/`；多视角目录 `assets/Beverages_And_snacks/1/0.jpg` 也对应这一物体目录。`assets/SKU/0/0.jpg` 或 `90.jpg` 对应 `output/SKU/0/{启动时间戳}/`，角度照片名不作为物体名。时间戳采用北京时间 `YYYYMMDD_HHMMSS_微秒`，记录在 `input/input.json`。`assets` 外的图片沿用 `output/{图片名}/{启动时间戳}/`。原有输出目录和历史示例路径保持原位置，新运行使用新规则。

注意：`--output` 现在表示输出根目录，使用 `--output output` 即可；不要再写旧式的 `--output output/xuebi`，否则会得到 `output/xuebi/xuebi/{时间戳}/`。

程序启动时会打印 `RUN_DIR /absolute/path/...`。后续各阶段使用这一个目录，**不会重新生成时间戳**。四个文件夹在启动时就会建立，尚未执行的阶段保持为空：

```text
output/xuebi/20261009_162504_317778/
├── input/                         # 输入快照
│   ├── rgb.jpg                    # 原图副本，保留原文件扩展名
│   ├── input.json                 # 原始路径、prompt、seed、配置参数和输入哈希
│   └── mask.png                   # 仅 --mask 模式复制用户提供的 mask
├── sam3d/                         # 第 3 阶段：分割与三维生成
│   ├── sam31/                     # 文本模式或点击模式生成
│   │   ├── mask.png
│   │   ├── mask_overlay.png
│   │   ├── candidates/            # 仅文本模式：候选实例
│   │   ├── interaction.json       # 仅点击模式：点集、操作历史和确认时间
│   │   ├── segmentation.json
│   │   └── run.log
│   ├── mesh.glb
│   ├── splat.ply
│   ├── report.json
│   └── preview.png                # 可选，运行预览脚本后生成
├── usd/                           # 第 4 阶段：纯显示 USD
│   ├── object.usda
│   └── object.conversion.json
└── physics/                       # 第 5 阶段：独立的物理 USD
    ├── object.usda
    ├── object.physics.json
    └── object.conversion.json
```

`sam3d/report.json` 记录分割耗时、mask 来源、选定实例、生成结果及失败阶段。自动分割结果属于第 3 阶段；`input/` 保存原始输入。`--load-only` 仍创建四个文件夹，但仅记录输入参数和加载报告，不复制或校验图片/mask。

下面命令中的 `RUN_DIR` 应替换为本次终端打印的实际目录，例如：

```bash
RUN_DIR=output/xuebi/20261009_162504_317778
```

查看分割结果、生成四视图预览：

```bash
xdg-open "$RUN_DIR/sam3d/sam31/mask_overlay.png"
OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  taskset -c 0 .venv-sam3d/bin/python -m objgen.tools.preview_mesh \
  "$RUN_DIR/sam3d/mesh.glb" --output "$RUN_DIR/sam3d/preview.png"
xdg-open "$RUN_DIR/sam3d/preview.png"
```

在 Blender 中交互查看任意 `.glb`，可使用以下通用命令：

```bash
/home/seer/wzt/Blender/.local/opt/blender/blender --python-expr "import bpy; bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(); bpy.ops.import_scene.gltf(filepath='/绝对路径/模型.glb')"
```

将 `/绝对路径/模型.glb` 替换为实际文件的绝对路径，例如本次运行目录下的 `sam3d/mesh.glb`。模板使用本机 Blender 的完整安装路径，无需将 `blender` 加入 `PATH`。该命令启动 Blender、清除启动场景中的默认对象，再导入模型，便于旋转和缩放查看。

`mesh.glb` 是带顶点颜色的三角网格。当前入口使用上游 `Inference` API，并固定关闭 mesh 后处理、纹理烘焙和布局后处理，启用 vertex color 输出：

```python
result = pipeline(image, mask, seed=seed)
mesh = result["glb"]
mesh.export("mesh.glb")
result["gaussian"][0].save_ply("splat.ply")
```

因此，后续 USD 转换使用 `mesh.glb`，不是 `splat.ply`。

---

## 4. 将 GLB 转换为 USDA

### 4.1 基本转换

```bash
.venv-sam3d/bin/python -m objgen.to_usd.mesh_to_usd \
  "$RUN_DIR" \
  --prim-name Object \
  --asset-name 'Generated Object'
```

在当前终端中，先执行以下命令为 `RUN_DIR` 赋值，再运行上面的转换命令：

```bash
RUN_DIR="/home/seer/wzt/Reconstruction/ObjectGeneration/output/3/20261009_165540_430296"
```

赋值时变量名不加 `$`，使用时写作 `"$RUN_DIR"`；等号两侧不要加空格。换用其他结果时，将路径替换为对应的运行目录。

参数配置说明（`.venv-sam3d/bin/python` 使用本项目的 Python 环境，`objgen.to_usd.mesh_to_usd`（`src/objgen/to_usd/mesh_to_usd.py`） 是转换入口）：

| 参数 | 含义 | 可选项或示例 |
| --- | --- | --- |
| `"$RUN_DIR"` | 必填位置参数，指定本次运行目录；也可以传入 GLB 文件路径。 | 例如 `output/3/20261009_165540_430296`，或该目录下的 `sam3d/mesh.glb`。 |
| `--prim-name Object` | USD 根节点名称，例如生成 `/Object`；不是输出文件名。 | 自定义合法的 USD 标识符，例如 `Object`、`WhiteBottle`、`Table`；不要包含空格、斜杠或以数字开头。省略时脚本默认 `Xuebi`。 |
| `--asset-name 'Generated Object'` | 写入资产元数据的人类可读名称，不改变节点路径。 | 自定义文本，例如 `'White Bottle'`；包含空格时加引号。省略时使用根节点名，`Xuebi` 特例使用 `Xuebi bottle`。 |
| `--output`（可选追加） | 显式指定输出文件；按新运行目录输入时通常不需要填写。 | 例如 `--output "$RUN_DIR/usd/object.usda"`；可用 `.usda` 或 `.usd` 后缀。默认输出到 `usd/object.usda`。 |

传入运行目录时，自动读取 `sam3d/mesh.glb`，写入 `usd/object.usda` 和 `usd/object.conversion.json`。也可直接传 GLB 路径并用 `--output` 显式指定文件。`.usda` 是可读的 ASCII USD。

### 4.2 指定现实尺寸

SAM3D 从单张图片推断几何，但通常不能可靠提供现实世界绝对尺寸。建议在转换时显式指定物体高度：

```bash
.venv-sam3d/bin/python -m objgen.to_usd.mesh_to_usd \
  "$RUN_DIR" \
  --prim-name Object \
  --asset-name 'Generated Object' \
  --height 0.30
```

在当前终端中，先执行以下命令为 `RUN_DIR` 赋值，再运行上面的转换命令：

```bash
RUN_DIR="/home/seer/wzt/Reconstruction/ObjectGeneration/output/3/20261009_165540_430296"
```

赋值时变量名不加 `$`，使用时写作 `"$RUN_DIR"`；等号两侧不要加空格。换用其他结果时，将路径替换为对应的运行目录。

参数配置说明：

| 参数 | 含义 | 可选项或示例 |
| --- | --- | --- |
| `"$RUN_DIR"` | 指定本次运行目录，自动定位 `sam3d/mesh.glb`。 | 例如 `output/3/20261009_165540_430296`；也可传入 GLB 路径。 |
| `--prim-name Object` | USD 根节点名称。 | 例如 `WhiteBottle`，生成 `/WhiteBottle`；命名要求见 4.1。 |
| `--asset-name 'Generated Object'` | 资产的可读名称。 | 例如 `--asset-name 'White Bottle'`；可省略。 |
| `--height 0.30` | 指定最终高度，单位为米，其他两个方向按相同比例缩放。 | 必须为有限正数，例如 `--height 0.20` 表示高 20 cm；按实物高度填写。不要与 `--scale` 同时指定。 |

`--height 0.30` 表示将最终物体高度设为 0.30 米。脚本会统一缩放模型，保持长宽比例。

也可以直接提供统一缩放倍数：

```bash
.venv-sam3d/bin/python -m objgen.to_usd.mesh_to_usd \
  "$RUN_DIR" \
  --prim-name Object \
  --scale 1.5
```

在当前终端中，先执行以下命令为 `RUN_DIR` 赋值，再运行上面的转换命令：

```bash
RUN_DIR="/home/seer/wzt/Reconstruction/ObjectGeneration/output/3/20261009_165540_430296"
```

赋值时变量名不加 `$`，使用时写作 `"$RUN_DIR"`；等号两侧不要加空格。换用其他结果时，将路径替换为对应的运行目录。

参数配置说明：

| 参数 | 含义 | 可选项或示例 |
| --- | --- | --- |
| `"$RUN_DIR"` | 指定本次运行目录或输入 GLB 路径。 | 例如 `output/3/20261009_165540_430296`。 |
| `--prim-name Object` | USD 根节点名称。 | 例如 `Object`、`WhiteBottle`、`Table`；省略时默认 `Xuebi`。 |
| `--scale 1.5` | 相对于原始 GLB 的统一缩放倍数，无单位。 | 必须为有限正数，例如 `1.5` 表示三个方向均放大为 1.5 倍；默认 `1.0`，保留原始尺度。不要与 `--height` 同时指定。 |

`--height` 和 `--scale` 不能同时使用。

### 4.3 转换脚本执行的坐标与材质处理

运行 4.1 或 4.2 的转换命令时，`mesh_to_usd.py` 会自动设置下表中的属性，**本节不需要单独执行命令**。“固定”表示当前没有对应的命令行选项。

| 物体或场景属性 | 脚本实际设置 | 含义与作用 | 调整方式或示例 |
| --- | --- | --- | --- |
| 向上轴（`upAxis`） | `Z`；顶点坐标从 `(x, y, z)` 转为 `(x, -z, y)`。 | 将 SAM3D 导出的 Y-up GLB 转成 Z-up，法线也同步旋转。 | 固定；不会自动判断任意来源 GLB 的朝向。 |
| 长度单位（`metersPerUnit`） | `1.0`。 | USD 中 1 个长度单位表示 1 米；设置单位本身不代表已确定实物尺寸。 | 固定为米；实际尺寸通过下面的高度或缩放参数调整。 |
| 模型尺寸 | 对三个方向使用相同缩放倍数，默认 `1.0`。 | 保持原有长宽高比例，不单独拉伸某个方向。 | `--height 0.20` 将最终高度设为 20 cm；或 `--scale 1.5` 整体放大到 1.5 倍。 |
| 水平位置 | X、Y 方向的包围盒中心均移到 `0`。 | 模型在 XY 平面居中；这里对齐的是包围盒中心，不是物理质心。 | 固定自动计算。 |
| 底部位置 | 最低顶点的 Z 坐标移到 `0`。 | 物体底部对齐地面高度，后续便于摆放。 | 固定自动计算。 |
| 顶点位置（`points`） | 写入旋转、缩放和居中后的顶点坐标。 | 坐标变换直接应用到网格顶点，不另外创建平移或缩放的 Xform 操作。 | 受 `--height` 或 `--scale` 影响；原始 GLB 文件不修改。 |
| 三角面拓扑 | 每个面的顶点数为 `3`，沿用读入网格的三角面索引。 | 保留原有三角网格结构；这一步不做网格简化。 | 固定从 GLB 读取。 |
| 顶点法线（`normals`） | 使用 `trimesh` 提供的顶点法线，旋转后写入；插值方式为 `vertex`。 | 为表面明暗和光照计算提供方向信息。 | 固定自动处理。 |
| 曲面细分（`subdivisionScheme`） | `none`。 | 按已有三角网格显示，不启用 USD 曲面细分。 | 固定。 |
| 双面显示（`doubleSided`） | `true`。 | 允许从三角面的正反两侧显示表面；不等于为物体增加厚度。 | 固定。 |
| 局部包围盒（`extent`） | 转换后全部顶点的最小、最大 XYZ 坐标。 | 描述网格空间范围，供包围盒计算和视图取景等使用。 | 固定自动计算。 |
| 顶点颜色（`displayColor`） | 从 GLB 读取每个顶点的 RGB，将 0–255 归一化并按 sRGB → 线性颜色转换；插值方式为 `vertex`。 | 保留 SAM3D 生成的瓶身颜色、标签等外观。 | 固定从输入网格读取；没有单独设置颜色的命令行参数。 |
| 显示材质类型 | 创建 `UsdPreviewSurface`。 | 提供基础表面着色，供支持该材质的渲染器显示。 | 固定；不自动识别并分配玻璃、塑料或金属材质。 |
| 材质基础颜色（`diffuseColor`） | 通过 `UsdPrimvarReader_float3` 连接网格的 `displayColor`。 | 使用顶点颜色驱动材质，而非使用单一颜色或外部纹理图片。 | 固定连接，不需要额外提供材质文件。 |
| 粗糙度（`roughness`） | `0.45`。 | 控制表面反光的集中或分散程度；较低值通常更光亮，较高值更粗糙。 | 固定；当前没有命令行选项。 |
| 金属度（`metallic`） | `0.0`。 | 将整个物体按非金属表面着色。 | 固定；当前没有命令行选项。 |
| 不透明度 | 材质 `opacity=1.0`；网格 `displayOpacity=1.0`，插值方式为 `constant`。 | 整个模型不透明；不使用输入顶点颜色的 alpha 来还原透明瓶身。 | 固定；当前没有命令行选项。 |
| 显示材质绑定 | 将新建显示材质绑定到根节点下的 `Mesh`。 | 使该网格实际使用上述材质和顶点颜色。 | 固定自动绑定。 |
| 根节点名称与类型 | 创建 `/<prim-name>`，类型为 `Xform`。 | 作为整个资产的父节点，便于整体引用和变换；不是输出文件名。 | 例如 `--prim-name WhiteBottle` 创建 `/WhiteBottle`；默认 `Xuebi`。 |
| 默认节点（`defaultPrim`） | 指向上述根节点。 | 外部场景引用该 USD 时，可直接使用这个资产入口。 | 跟随 `--prim-name`。 |
| 资产名称（`assetInfo:name`） | 使用 `--asset-name` 指定的文本。 | 保存人类可读的资产名称，不改变网格或节点路径。 | 例如 `--asset-name 'White Bottle'`；省略时使用根节点名，`Xuebi` 特例使用 `Xuebi bottle`。 |

保存后，脚本会重新打开 USD，检查面索引和面顶点数等拓扑是否有效，以及显示材质是否绑定成功，并将尺寸、顶点数、面数和检查结果写入 `usd/object.conversion.json`。这里的拓扑检查不等于闭合性检查或物理仿真验证。刚体、质量、惯量、碰撞体和物理材质由第 5 阶段添加。

生成的典型层级为：

```text
/Object
├── Mesh
└── Looks
    └── Visual
        ├── Surface
        └── Color
```

其中 `/Object` 是 default prim，`/Object/Mesh` 是实际几何体。

当前实现使用 `displayColor` 顶点颜色，不是 UV 纹理贴图。因此它适合保留 SAM3D 输出的整体颜色和外观，但没有自动生成可独立编辑的 PNG/JPG 纹理贴图。

### 4.4 显示材质与物理材质的区别

**显示材质决定物体看起来是什么样，物理材质决定物体接触其他物体时的摩擦和回弹表现。** 两者分别由渲染器和物理引擎使用，可以同时绑定到同一个网格。

| 对比项 | 显示材质（第 4 阶段） | 物理材质（第 5 阶段） |
| --- | --- | --- |
| 主要作用 | 控制颜色、明暗、反光和透明等视觉效果。 | 控制接触时的静摩擦、动摩擦和碰撞恢复系数。 |
| 本仓库实现 | `UsdPreviewSurface`，基础颜色连接网格的 `displayColor` 顶点颜色。 | `UsdPhysics.MaterialAPI`，通过 `materialPurpose="physics"` 单独绑定。 |
| 当前参数 | 颜色来自 SAM3D；粗糙度 `0.45`、金属度 `0.0`、不透明度 `1.0`。 | 静摩擦 `0.6`、动摩擦 `0.5`、恢复系数 `0.0`。 |
| 是否需要额外提供文件 | 不需要，脚本读取 GLB 顶点颜色并自动创建材质。 | 不需要，脚本自动创建材质并写入上述初值。 |
| 能否通过当前命令行调整 | 暂无颜色、粗糙度、金属度或透明度的命令行选项。 | 暂无摩擦或恢复系数的命令行选项；`--mass` 调整的是刚体质量。 |
| 修改后的影响 | 改变渲染外观，不会自动改变物体质量、摩擦或碰撞形状。 | 改变接触表现，不会自动改变颜色、标签或反光。 |

例如，把瓶子显示材质的粗糙度调低，会让它看起来更光亮，但不会让它在桌面上更容易滑动；要改变滑动表现，需要调整物理材质的摩擦参数。把外观改成金属，也不会自动增加瓶子的质量。

质量、质心、惯量、动态刚体开关和碰撞近似属于第 5 阶段的其他物理属性，不是这里所说的物理材质参数。碰撞近似决定接触形状，物理材质决定接触时的摩擦和回弹；两者需要配合使用。

第 4 阶段运行转换命令时会自动完成 4.3 的处理并创建显示材质，无需另行执行。第 5 阶段在保留该显示材质的基础上，增加物理材质与刚体属性，另存到 `physics/`。透明塑料、玻璃等外观不会由当前脚本自动还原，可后续在 Isaac Sim 中编辑显示材质，或扩展脚本参数。

---

## 5. 为 USDA 添加 Isaac Sim 物理属性

仅运行 `mesh_to_usd.py` 后，资产已经可以在 Isaac Sim 中加载和渲染，但还不是动态刚体。若需要掉落、碰撞或被机器人操作，执行：

```bash
.venv-sam3d/bin/python -m objgen.to_usd.configure_usd_physics \
  "$RUN_DIR" \
  --mass 1 \
  --collision-approximation convexHull
```

参数配置说明（`.venv-sam3d/bin/python` 使用本项目的 Python 环境，`objgen.to_usd.configure_usd_physics`（`src/objgen/to_usd/configure_usd_physics.py`） 是物理配置入口）：

| 参数 | 含义 | 可选项或示例 |
| --- | --- | --- |
| `"$RUN_DIR"` | 必填位置参数，指定运行目录，自动读取 `usd/object.usda`；也可直接传入 USD 文件路径。 | 例如 `output/3/20261009_165540_430296`。 |
| `--mass 1` | 设置整个物体的质量，单位为千克；用于计算质心和惯量相关质量属性。 | 必须为有限正数，例如 `--mass 0.5` 表示 500 g；默认 `1.0` kg。 |
| `--collision-approximation convexHull` | 选择网格碰撞体的近似方式。 | 仅支持 `convexHull`（单个凸包，默认）和 `convexDecomposition`（多个凸包组合，适合凹陷和开放空间）。 |
| `--solver-position-iterations` | 可选，设置刚体的位置求解迭代次数，提高接触约束求解精度。 | 整数 `[1,255]`，例如 `64`；省略时不写入覆盖值，本机 PhysX 默认 `16`。 |
| `--solver-velocity-iterations` | 可选，设置刚体的速度求解迭代次数。 | 整数 `[0,255]`，例如 `8`；省略时不写入覆盖值，本机 PhysX 默认 `1`。 |
| `--output`（可选追加） | 指定带物理属性的 USD 输出文件，必须与输入文件不同。 | 例如 `--output "$RUN_DIR/physics/object.usda"`；传入运行目录时默认使用此路径。 |

脚本读取 `usd/object.usda`，另存到 `physics/object.usda`，保留第 4 阶段的纯显示 USD 和原始转换报告。第 5 阶段的报告也单独写入 `physics/`。传入其他 USD 路径时可用 `--output` 指定目标；旧式路径默认另存为 `<原文件名>_physics.usd[a]`，不再原地修改。

这个脚本会为资产添加：

- 动态刚体；
- 质量；
- 质心；
- 对角惯量和主轴；
- 网格碰撞；
- 物理材质；
- 静摩擦、动摩擦和恢复系数。

“脚本会添加”不代表每项都开放了命令行配置。当前可手动指定、自动计算和固定设置的属性如下：

| 属性 | 能否通过命令指定 | 当前设置方式 |
| --- | --- | --- |
| 质量 | 可以 | `--mass 0.5`，单位 kg；默认 `1.0` kg。 |
| 碰撞近似 | 可以 | `--collision-approximation convexHull` 或 `convexDecomposition`；默认 `convexHull`。 |
| 接触求解迭代 | 可以 | `--solver-position-iterations 64 --solver-velocity-iterations 8`；省略时使用 PhysX 默认值。提高迭代次数会增加计算量，不改变质量、几何或摩擦系数。 |
| 最大凸包数量 | 可以，仅凸分解使用 | `--max-convex-hulls 128`；默认 `128`。 |
| 凸分解体素分辨率 | 可以，仅凸分解使用 | `--voxel-resolution 500000`；默认 `500000`。 |
| 动态刚体 | 不可以 | 固定启用，`kinematic=false`。 |
| 质心 | 不可以直接指定 | 根据闭合网格、均匀密度假设自动计算。 |
| 对角惯量和主轴 | 不可以直接指定 | 根据网格与指定质量自动计算。 |
| 网格碰撞开关 | 不可以 | 固定启用。 |
| 静摩擦 | 不可以 | 固定 `0.6`。 |
| 动摩擦 | 不可以 | 固定 `0.5`。 |
| 恢复系数 | 不可以 | 固定 `0.0`。 |

对于同一个网格，修改质量会让惯量按比例变化，但在均匀密度假设下，质心位置和惯量主轴方向不会因此改变。未开放命令行配置的属性仍可在 USD 或 Isaac Sim 中编辑。

当前默认刚体与物理材质参数如下；其中 `mass` 和 `kinematic` 属于刚体属性，摩擦和恢复系数属于物理材质：

```text
mass              = 用户指定
static friction   = 0.6
dynamic friction  = 0.5
restitution       = 0.0
kinematic         = false
```

质心和惯量由闭合网格按均匀密度假设估计，质量由 `--mass` 指定。这些参数是仿真初值，不代表真实物体测量值。

### 碰撞近似选择

简单、接近实心的物体可以使用凸包：

```bash
--collision-approximation convexHull
```

这是追加到物理配置命令中的参数片段，不能单独执行。`convexHull` 使用单个凸包；另一可选值是 `convexDecomposition`。凸包会填平凹陷，不能保留购物框内部或桌腿间的开放空间。

有凹陷、孔洞或多个细长结构的物体建议使用凸分解：

```bash
.venv-sam3d/bin/python -m objgen.to_usd.configure_usd_physics \
  "$RUN_DIR" \
  --mass 10 \
  --collision-approximation convexDecomposition
```

参数配置说明：

| 参数 | 含义 | 可选项或示例 |
| --- | --- | --- |
| `"$RUN_DIR"` | 指定本次运行目录或输入 USD 文件路径。 | 例如 `output/3/20261009_165540_430296`。 |
| `--mass 10` | 设置整个物体质量，单位为千克。 | 例如 `--mass 20` 表示 20 kg；必须为有限正数，按目标物体填写。 |
| `--collision-approximation convexDecomposition` | 使用多个凸包组合近似网格，使碰撞体能够表达部分凹陷和开放空间。 | 可选 `convexDecomposition` 或 `convexHull`；此处选择前者。 |
| `--max-convex-hulls`（可选追加） | 凸分解允许生成的最大凸包数量，仅在 `convexDecomposition` 模式下使用。 | 正整数，例如 `--max-convex-hulls 128`；默认 `128`。 |
| `--voxel-resolution`（可选追加） | 凸分解的体素分辨率参数，仅在 `convexDecomposition` 模式下使用。 | 整数，例如 `--voxel-resolution 500000`；默认 `500000`，脚本要求至少 `10000`。 |

静摩擦 `0.6`、动摩擦 `0.5`、恢复系数 `0.0` 目前由脚本固定设置，没有对应的命令行选项。

例如桌子、购物框等结构，通常比单个凸包更适合 `convexDecomposition`。显示网格会保持原样，碰撞体则使用 PhysX 近似。

脚本会额外生成：

```text
physics/object.physics.json
physics/object.conversion.json
```

---

## 6. 在 Isaac Sim 中验证

### 6.1 加载、渲染和保存查看场景

先进入 Isaac Sim 环境：

```bash
source /home/seer/wzt/robot_sim/scripts/activate_env.sh
```

然后执行：

```bash
python -u src/objgen/verify/check_usd_isaacsim.py \
  "$RUN_DIR/physics/object.usda" \
  --output "$RUN_DIR/physics" \
  --physics-preview
```

该脚本会在 headless Isaac Sim 中：

- 打开 USDA；
- 检查 Mesh 和显示材质；
- 计算世界包围盒；
- 添加相机、灯光和可选地面碰撞体；
- 输出 Isaac Sim 截图；
- 保存一个带查看环境的 USDA 场景。

典型输出：

```text
<RUN_DIR>/physics/object_isaacsim.png
<RUN_DIR>/physics/object_view.usda
<RUN_DIR>/physics/isaacsim_verification.json
```

### 6.2 物理掉落与碰撞

```bash
python -u src/objgen/verify/check_usd_physics_isaacsim.py \
  "$RUN_DIR/physics/object.usda" \
  --output "$RUN_DIR/physics"
```

默认读取 USDA 资产中已有的质量，检查它是否为有效正数，并验证物理引擎实际使用的质量与资产一致；不需要重复填写质量，也不会修改或写回资产。

| 参数 | 含义 | 可选项或示例 |
| --- | --- | --- |
| `"$RUN_DIR/physics/object.usda"` | 待验证的物理资产路径。 | 第 5 步生成的 `physics/object.usda`。 |
| `--output` | 验证报告的保存目录。 | `"$RUN_DIR/physics"`，报告名为 `isaacsim_physics_verification.json`。 |
| `--mass` | 可选的额外质量校验，单位 kg；省略时读取资产质量，没有默认的 1 kg 限制。不会覆盖资产质量。 | 例如 `--mass 0.5`；必须是有限正数，与资产质量不一致时检查失败。 |
| `--steps` | 仿真步数，用于检查较长时间后的稳定性，物理时间步仍为 `1/120 s`。 | 整数，至少 `600`；默认 `600`（5 秒），例如 `1200`（10 秒）。不放宽速度或穿透阈值。 |
| `--clearance-probe` | 可选的底部空间检查，加入边长 8 cm 的测试立方体。 | 开关参数；默认关闭，需要时添加。 |

如果物体存在底部空腔或需要检查碰撞体是否错误填满内部空间，可以加上：

```bash
python -u src/objgen/verify/check_usd_physics_isaacsim.py \
  "$RUN_DIR/physics/object.usda" \
  --output "$RUN_DIR/physics" \
  --clearance-probe
```

该验证默认执行 600 个物理步，检查：

- 资产质量是否为有效正数、运行时质量是否与资产一致，以及可选的 `--mass` 预期质量校验；
- 物体是否正常下落；
- 是否穿透地面；
- 最终线速度和角速度；
- 碰撞近似和物理材质；
- 可选的物体底部空间。

验证报告记录中间位置、速度、最终速度模长及每项判定。Python 检查失败时也写入 `passed: false` 和错误原因；Isaac Sim 原生启动崩溃可能来不及生成报告。不能只凭进程退出码判断成功，仍需检查报告和成功标记。

---

## 7. 一套完整的新物体流程

### 7.1 一条命令完成第 3、4、5、6 阶段

在项目根目录运行，无需手动切换 Python 环境或设置 `RUN_DIR`：

```bash
./scripts/run_pipeline.sh \
  --image assets/Beverages_And_snacks/3.jpg \
  --prompt 'white bottle' \
  --low-vram --sequential-condition \
  --height 0.14 \
  --mass 0.5 \
  --prim-name Bottle3 \
  --asset-name 'Bottle3' \
  --collision-approximation convexHull \
  --output output \
  --seed 42
```

| 参数 | 含义 | 可选项或示例 |
| --- | --- | --- |
| `--image` | 输入 RGB 图片，必填。 | 例如 `assets/Beverages_And_snacks/3.jpg`。 |
| `--prompt` / `--mask` | 文本分割或已有 mask，互斥但均可省略；均省略时进入点击分割并等待人工确认。 | `--prompt 'white bottle'`；也可替换为 `--mask /path/to/mask.png`。 |
| `--low-vram` | SAM3D 按阶段将暂不用的模型移到 CPU，降低显存占用。 | 开关参数，默认关闭；本机建议开启。 |
| `--sequential-condition` | 顺序运行条件编码器和生成器，进一步降低显存峰值。 | 开关参数，默认关闭；依赖 `--low-vram`。 |
| `--height` / `--scale` | 设置最终高度（米）或统一缩放系数，二者互斥。 | 例如 `--height 0.14` 表示 14 cm，或 `--scale 0.2`；均须为有限正数。都省略时使用 `--scale 1`，不代表已校准现实尺寸。 |
| `--mass` | 第 5 阶段设置的物体质量，单位 kg；第 6 阶段自动从资产读取。 | 有限正数，默认 `1.0`；例如 `0.5` 表示 500 g。 |
| `--solver-position-iterations` / `--solver-velocity-iterations` | 可选，将接触求解迭代次数传给第 5 阶段并保存到资产。 | 位置 `[1,255]`，速度 `[0,255]`；例如 `--solver-position-iterations 64 --solver-velocity-iterations 8`。默认均省略，沿用 PhysX 默认值。 |
| `--prim-name` | USD 根 prim 名称。 | 默认 `Object`；例如 `Bottle3`。使用英文字母、数字或下划线，不能以数字开头。 |
| `--asset-name` | 资产显示名称。 | 例如 `'Bottle3'`；默认与 `--prim-name` 相同。 |
| `--collision-approximation` | 第 5 阶段的碰撞近似。 | `convexHull`（默认）或 `convexDecomposition`。 |
| `--max-convex-hulls` | 凸分解允许的最大凸包数量，仅凸分解使用。 | 正整数，默认 `128`；例如 `64`。 |
| `--voxel-resolution` | 凸分解体素分辨率，仅凸分解使用。 | 至少 `10000`，默认 `500000`；例如 `200000`。 |
| `--output` | 输出根目录，保留 `assets` 下的分类/物体路径，再添加启动时间戳。 | 默认项目的 `output`；例如 `--output output`。 |
| `--seed` | SAM3D 随机种子。 | 整数，默认 `42`；例如 `123`。 |
| `--clearance-probe` | 第 6.2 步额外检查 8 cm 立方体能否留在资产底部空间。 | 开关参数，默认关闭；只在资产确实需要此检查时添加，普通实心瓶子不需要。 |
| `--sam31-threshold` | SAM3.1 文本模式的候选实例置信度阈值。 | `[0, 1]`，默认 `0.5`；例如 `0.3`。 |
| `--sam31-instance-index` | 文本模式候选实例按分数降序排列后的索引。 | 非负整数，默认 `0`；例如 `1` 选择第二个实例。 |
| `--sam31-python` | SAM3.1 环境的 Python 路径，文本模式与点击模式使用。 | 默认 `../InstanceRecognition/.venv/bin/python`（相对于项目位置）。 |
| `--sam31-repo` | SAM3.1 源码目录，文本模式与点击模式使用。 | 默认 `../Submodule/SAM3`（相对于项目位置）。 |
| `--sam31-checkpoint` | SAM3.1 权重文件，文本模式与点击模式使用。 | 默认 `../weights/sam3.1_multiplex.pt`（相对于项目位置）。 |
| `--config` | SAM3D pipeline 配置。 | 默认项目中的 `.runtime_config/sam3d/pipeline.yaml`。 |
| `--isaac-env` | 用于启动 Isaac Sim 子进程的环境激活脚本。 | 默认 `/home/seer/wzt/robot_sim/scripts/activate_env.sh`；可指定另一份激活后能运行 `python` 和 Isaac Sim 的脚本。 |

要用**点击选择 + 完整流水线**，从原命令中去掉 `--prompt` 即可，例如：

```bash
./scripts/run_pipeline.sh \
  --image assets/Beverages_And_snacks/5.jpg \
  --low-vram --sequential-condition \
  --height 0.14 --mass 0.5 \
  --prim-name Bottle5 --asset-name Bottle5 \
  --collision-approximation convexHull \
  --output output --seed 42
```

参数含义和可选项与上表一致：示例输入为 `5.jpg`，目标高度 `0.14 m`、质量 `0.5 kg`、凸包碰撞；同时省略 `--prompt` 和 `--mask` 触发点击模式。无需手动设置 `RUN_DIR`。浏览器中的点击、补点和确认操作见第 3.1 节；确认之后自动执行剩余阶段。

统一入口 `scripts/run_pipeline.sh` / `src/objgen/run_pipeline.py` 会依次执行：

1. **第 3 阶段**：调用 `run_sam3d.sh`，完成 SAM3.1 分割（文本自动分割，或点击分割并等待人工确认），再运行 SAM3D；已有 mask 时直接使用。
2. **第 4 阶段**：调用 `mesh_to_usd.py`，转换坐标、设置尺寸和显示材质，生成视觉 USDA。
3. **第 5 阶段**：调用 `configure_usd_physics.py`，添加质量、碰撞与物理材质，生成物理 USDA。
4. **第 6.1 步**：在独立 Isaac Sim 进程中加载、渲染，并保存带地面的查看场景。
5. **第 6.2 步**：在另一个 Isaac Sim 进程中执行 600 步掉落与碰撞检查，质量直接读取第 5 阶段产物。

各模型及 Isaac Sim 进程顺序运行，上一进程退出后才启动下一进程。执行前会检查输入、配置、环境脚本及 Isaac Sim 包是否存在；这不替代实际运行验证。

本机的第 3–5 阶段都使用 `.venv-sam3d`，统一入口默认将这些进程绑定到 CPU 0，并设置 OpenBLAS/OpenMP 单线程兼容参数。第 4、5 阶段也在 Python 启动前完成绑定，避免遗漏 `run_sam3d.sh` 已有的 CPU 兼容设置。环境变量 `SAM3D_CPU_AFFINITY` 可覆盖默认值（例如 `0` 或 `0-1`），设为空字符串可关闭绑定；本机建议保留默认值。第 6 阶段使用独立的 Isaac Sim 环境，在激活环境后按 `SIM_CPU_AFFINITY` 绑定 CPU；本机激活脚本默认 `16-23`，不复用 SAM3D 的 CPU 0。其他激活脚本未设置该变量时不绑定。原生崩溃时日志会尽可能包含 Python 栈，统一入口的失败提示会标明 `SIGILL` 等信号名称及日志路径。

产物仍保存在四个文件夹中，不额外增加阶段目录：

```text
output/3/{启动时间戳}/
├── input/
│   ├── rgb.jpg、input.json
│   └── pipeline.json                  # 总状态、参数、各阶段命令、耗时与失败原因
├── sam3d/
│   ├── mesh.glb、splat.ply、report.json、sam31/
│   └── run.log                        # 第 3 阶段日志
├── usd/
│   └── object.usda、object.conversion.json、run.log
└── physics/
    ├── object.usda、object.physics.json、object.conversion.json
    ├── object_isaacsim.png、object_view.usda
    ├── isaacsim_verification.json、isaacsim_physics_verification.json
    └── configure.log、isaacsim_render.log、isaacsim_physics.log
```

终端会打印本次 `RUN_DIR`；只有所有阶段通过后才打印 `PIPELINE_SUCCESS` 和最终 `physics/object.usda` 路径。任一阶段失败，总入口返回非零退出码并停止后续阶段，已生成的产物、总报告和该阶段日志会保留。Isaac Sim 子进程即使错误地返回退出码 0，总入口仍要求本次生成的成功报告、成功标记和必要产物都有效，才能继续。

每次调用都新建运行目录，不会自动续跑或覆盖旧结果。若要只重做某个阶段，仍可使用第 3–6 节的独立命令。

### 7.2 分步执行同一流程

以下示例假设：

```text
图片：assets/my_object/my_object.png
prompt：bottle（替换为目标物体描述）
输出：output/my_object/{启动时间戳}/
目标高度：0.30 m
质量：1 kg
USD prim：MyObject
```

### 第一步：SAM3D 推理

```bash
./scripts/run_sam3d.sh \
  --low-vram --sequential-condition \
  --image assets/my_object/my_object.png \
  --prompt 'bottle' \
  --output output \
  --seed 42
```

记下终端打印的 `RUN_DIR`，将下面的时间戳替换为该次运行的实际值：

```bash
RUN_DIR=output/my_object/20261009_162504_317778
```

### 第二步：GLB 转 USDA

```bash
.venv-sam3d/bin/python -m objgen.to_usd.mesh_to_usd \
  "$RUN_DIR" \
  --prim-name MyObject \
  --asset-name 'My Generated Object' \
  --height 0.30
```

### 第三步：添加物理

```bash
.venv-sam3d/bin/python -m objgen.to_usd.configure_usd_physics \
  "$RUN_DIR" \
  --mass 1 \
  --collision-approximation convexHull
```

### 第四步：Isaac Sim 加载和渲染

```bash
source /home/seer/wzt/robot_sim/scripts/activate_env.sh

python -u src/objgen/verify/check_usd_isaacsim.py \
  "$RUN_DIR/physics/object.usda" \
  --output "$RUN_DIR/physics" \
  --physics-preview
```

### 第五步：Isaac Sim 物理验证

```bash
python -u src/objgen/verify/check_usd_physics_isaacsim.py \
  "$RUN_DIR/physics/object.usda" \
  --output "$RUN_DIR/physics"
```

---

## 8. USDA、USD 和场景文件

- `object.usda`：可读的 ASCII USD 资产，适合检查和手工编辑；
- `object.usd`：通常用于二进制 USD/USDC 资产，适合部署和较快加载；
- `object_view.usda`：带相机、灯光、地面等查看环境的场景，不一定是最终资产本身；
- `mesh.glb`：SAM3D 生成的原始网格；
- `splat.ply`：Gaussian Splat 输出，不是当前 USD 转换脚本的输入。

Isaac Sim 可以直接加载 `.usda` 和 `.usd`。如果需要将文本 USDA 转成二进制 USD，可使用 Pixar USD 的 stage export（`Usd.Stage.Open(...).Export("object.usd")`）。

---

## 9. 当前已验证的本地结果

2026-10-10 完成物体 1–12 的补验与版本整理：每个物体均至少有一个第 3–6 阶段全部通过的版本。物体 4 仅保留 `output/4/20261010_102841_880319/`，物体 10 仅保留 `output/10/20261010_112939_616882/`；各自未验证版本已按用户要求删除，删除记录在保留版本的 `input/version_cleanup.json`。下文提到的 `output/4/20261010_103501_776018/` 是历史失败记录，该目录现已移除。

- 物体 1：`output/1/20261010_125553_873629/`。延长至 1200 步仍有接触速度残差；最终将资产的 PhysX 位置/速度求解迭代设为 `64/8`，在原有 600 步、线速度 `<0.05 m/s` 和角速度 `<0.10 rad/s` 的标准下通过，最终角速度约 `0.00118 rad/s`。质量保持 `0.5 kg`，几何、惯量和 `convexHull` 碰撞近似不变；已重新渲染。原资产、失败报告和调整记录保留在该轮目录内，详见 `input/solver_adjustment.json`。
- 物体 11：`output/11/20261010_124649_600250/`。补跑渲染及 600 步掉落检查通过，资产未修改。物理补跑在 Python 启动前绑定 CPU `16-23`，本轮未再崩溃；这不证明底层原生崩溃的根因已完全排除。统一入口现遵循激活脚本的 `SIM_CPU_AFFINITY` 设置。

两轮 `input/pipeline.json` 已更新为 `success`，各次补跑命令、日志与原始报告备份保留。补跑使用了 180 秒外部超时限制；这项限制仅用于本次恢复命令，统一入口尚未新增全局超时。

2026-10-10 已验证点击模式的真实模型和端到端衔接：输入 `assets/Beverages_And_snacks/5.jpg`，不提供 prompt/mask，通过本地 HTTP 测试提交 4 个正点和 1 个负点，检查叠加图后发送确认，第 3、4、5、6.1、6.2 阶段全部成功。产物目录为 `output/5/20261010_110120_711695/`，最终资产为 `physics/object.usda`（高度 `0.14 m`、质量 `0.5 kg`）；Isaac Sim 渲染及 600 步掉落检查均为 `passed: true`。总报告见 `input/pipeline.json`，点击与确认记录见 `sam3d/sam31/interaction.json`。

此次是自动化 HTTP 交互测试，实际调用了 SAM3.1，并由编码助手检查了工具显示的预览图片；确认请求由测试代码发送，不是使用者在浏览器点击。当前工具环境没有可用浏览器连接，因此浏览器中的实际鼠标操作、自动弹窗与页面视觉效果尚未实测。验证范围记录在 `input/interaction_qa.json`。测试同时确认了未读取预览、过期确认请求会被拒绝，确认前没有 mask 导出或 SAM3D 生成。

2026-10-10 修复了统一入口第 4、5 阶段遗漏 CPU 绑定的问题。`output/4/20261010_102841_880319/` 原来在第 5 阶段因 Python 原生 `SIGILL` 退出；补齐 CPU 0 绑定后，沿用原始网格补跑物理配置、渲染和 600 步掉落碰撞检查，均通过，质量仍为 `0.5 kg`。总报告为 `input/pipeline.json`，最初的失败报告保留为 `input/pipeline.before_cpu_fix.json`，诊断证据见 `input/runtime_diagnosis.json`。未绑定的独立依赖导入检查也复现过原生崩溃，绑定 CPU 0 的 50 次导入检查及 15 项自动测试通过；具体底层解释器或硬件触发机制尚未完全确定。

排查期间另一次重新生成的结果 `output/4/20261010_103501_776018/` 已通过第 5 阶段，但第 6.2 步最终角速度略高于稳定性阈值；该次报告仍保留失败状态，没有放宽物理检查判据。这与原来的 `SIGILL` 是不同问题。

2026-10-10 已实测统一入口 `scripts/run_pipeline.sh`：输入 `assets/Beverages_And_snacks/3.jpg`、prompt `white bottle`、高度 `0.14 m`、质量 `0.5 kg`、`convexHull`，第 3、4、5、6.1、6.2 步全部通过，总耗时约 109 秒。结果目录为 `output/3/20261010_101920_596399/`，总报告见 `input/pipeline.json`，最终资产为 `physics/object.usda`；Isaac Sim 渲染和 600 步掉落碰撞报告均为 `passed: true`。该记录验证了本机本次输入的完整流程，不代表任意图片和碰撞设置都能通过。

新目录结构已在 `output/xuebi/20261009_162504_317778/` 完成实际验证：SAM3.1 自动分割和 SAM3D 生成成功（约 66.93 秒），随后完成第 4 阶段 13 cm 高度的 USD 转换及第 5 阶段 1 kg 刚体配置。四个文件夹齐全；重新打开 USD 检查确认 `usd/object.usda` 没有刚体，`physics/object.usda` 有刚体，第 4 阶段文件及转换报告保持不变。记录见 `sam3d/report.json` 和 `physics/layout_verification.json`。此次未重新执行 Isaac Sim 仿真。

2026-10-09 已实测新增的 `SAM3.1 → SAM3D` 自动流程：使用 `assets/xuebi/xuebi.jpg` 和 `--prompt 'bottle' --low-vram --sequential-condition`，输出位于 `output/xuebi_sam31_prompt/`。SAM3.1 返回一个有效实例，分数约 0.977，分割子进程约 15.88 秒；包含分割、模型加载和导出的完整流程约 96.38 秒，生成 146856 个顶点、293704 个三角面的闭合网格。SAM3D 阶段峰值已分配显存约 6.34 GiB。此次验证针对自动分割和三维生成，没有重新执行 Isaac Sim 仿真。

自动输入检查可在本机运行：

```bash
OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  taskset -c 0 .venv-sam3d/bin/python -m unittest discover -s tests -v
```

24 项检查覆盖候选排序与空 mask 过滤、输入参数冲突、分割失败/取消/未确认时不启动 SAM3D、当前预览确认与过期结果拦截、清空及负点修正、点击适配层的推理上下文和初始帧缓存、HTTP 交互、流水线调度和失败报告、输入快照、目录隔离，以及物理配置保留第 4 阶段资产。浏览器脚本另外通过 JavaScript 语法检查。

以下为保留在原位置的历史示例，其原始文档和报告记录的是旧目录结构。新运行统一采用上文的时间戳结构，旧命令中的输出路径应按第 3–7 节调整：

```text
output/sam3d-low-vram/          # 官方 kidsroom 示例
output/xuebi_sam31_prompt/       # SAM3.1 prompt 自动分割后生成的雪碧瓶
output/xuebi/                   # 雪碧瓶
output/table/                   # 桌子
```

已有推理记录显示，使用 `--low-vram` 可以在本机完成单物体生成，并得到闭合网格。例如：

- kidsroom 示例：约 189k 顶点、378k 三角面；
- 雪碧瓶：约 132k 顶点、264k 三角面；
- 桌子：约 105k 顶点、210k 三角面；
- 购物框和桌子资产已经进一步配置物理并在 Isaac Sim 中加载、渲染和进行掉落验证。

验证文件通常包括：

```text
report.json
verification.json
object.conversion.json
object.physics.json
isaacsim_verification.json
isaacsim_physics_verification.json
```

这些文件分别用于区分：

- SAM3D 推理是否成功；
- GLB/USD 文件和网格拓扑是否有效；
- USD 尺寸、材质和物理配置；
- Isaac Sim 是否实际加载并完成仿真验证。

---

## 10. 使用边界和已知限制

1. **单张图片重建不是三维扫描。** 背面和被遮挡区域由模型推断，不能保证与真实物体一致。
2. **绝对尺寸需要人工指定。** `--height` 是仿真标定值，不是 SAM3D 自动测量结果。
3. **当前主要保留顶点颜色。** 没有自动输出独立 UV 纹理贴图。
4. **网格可能较密。** SAM3D 输出可能达到十万级顶点；大规模场景或实时碰撞可在 USD 转换前后增加网格简化流程。
5. **显示网格与碰撞网格不同。** `convexHull` 或 `convexDecomposition` 只提供物理近似。
6. **mask 质量会直接影响重建。** mask 包含背景、阴影或其他物体时，SAM3D 可能将这些内容一起重建。
7. **质量、摩擦和惯量是仿真初值。** 当前物理脚本按闭合网格和均匀质量分布估计，不能替代真实测量。
8. **显存和系统内存仍是主要资源约束。** 当前机器应优先使用 `--low-vram`，不同输入的峰值显存和运行时间可能不同。


## 11. 外观精修与独立资产交付（M4 / M5）

已有物体可以一条命令完成照片投射、受限 GPT 反馈、物理属性保留、贴图打包以及迁移后的 Isaac 显示和掉落验证：

```bash
./scripts/refine_appearance.sh \
  --run-dir output/6/20261010_110616_147015 \
  --config configs/appearance/default.yaml
```

| 参数 | 含义 | 可选项 / 示例 |
|---|---|---|
| `--run-dir` | 已完成阶段 3～5 的物体目录，不重新调用 SAM3D | `output/6/20261010_110616_147015` |
| `--config` | 模型、Blender、贴图分辨率、迭代预算配置 | 默认 `configs/appearance/default.yaml` |
| `--revision` | 复用通过技术检查的候选，跳过重新烘焙 | `v_20261012_160952_923814`；省略则新建 |
| `--skip-feedback` | 跳过 GPT API，只做照片投射及完整交付验证 | 开关；默认启用 GPT 反馈 |
| `--resume` | 从指定 M5 任务记录恢复，严格校验输入、配置及代码指纹 | `output/6/20261010_110616_147015/usd/appearance/m5_20261012_162949_655517` |
| `--isaac-env` | Isaac 环境激活脚本 | 默认 `/home/seer/wzt/robot_sim/scripts/activate_env.sh` |

恢复时保留首次使用的 `--revision`、`--skip-feedback` 等参数。已成功交付的资产在哈希检查后直接复用；未完成的交付重新建立独立包。中断/失败的 API 反馈不会自动重放扣费，需另起任务指定已有 `--revision --skip-feedback`。代码、输入或配置已变化时拒绝恢复，需新建任务。失败的贴图候选不会仅因文件存在就被复用。

新物体从生成开始的端到端命令：

```bash
./scripts/run_pipeline.sh \
  --image assets/Beverages_And_snacks/5.jpg \
  --prompt 'white bottle' \
  --low-vram --sequential-condition \
  --height 0.14 --mass 0.5 \
  --prim-name Bottle5 --asset-name Bottle5 \
  --collision-approximation convexHull \
  --output output --seed 42 \
  --appearance-config configs/appearance/default.yaml
```

| 参数 | 含义 | 可选项 / 示例 |
|---|---|---|
| `--image` | 输入 RGB 照片，运行目录按分类和物体名自动分配 | `assets/Beverages_And_snacks/5.jpg` |
| `--prompt` | SAM3.1 文本分割提示 | `'white bottle'`；可改用 `--mask`，两者均省略则点击分割 |
| `--low-vram` / `--sequential-condition` | 低显存及条件编码顺序执行 | 开关；后者要求前者 |
| `--height` | 现实高度，单位米 | `0.14`；可改用互斥的 `--scale 1.0` |
| `--mass` | 阶段 5 设置的质量，单位 kg，精修交付保留此值 | `0.5` |
| `--prim-name` / `--asset-name` | USD 根节点标识和资产显示名 | `Bottle5` |
| `--collision-approximation` | 碰撞近似 | `convexHull` 或 `convexDecomposition` |
| `--output` | 输出根目录 | `output` |
| `--seed` | 生成随机种子 | `42` |
| `--appearance-config` | 启用外观精修和完整交付验证 | `configs/appearance/default.yaml`；省略保持原流程 |
| `--appearance-skip-feedback` | 启用精修时跳过 GPT API 反馈 | 开关，需同时提供 `--appearance-config` |

不需要人工设置 RUN_DIR。执行顺序为原阶段 **3→4→5→6**，再执行 **7_appearance**（照片投射→可选 GPT 反馈→M4 交付验证），以原物理资产为基线保留质量、惯量、摩擦等属性。`PIPELINE_SUCCESS` 打印最终精修包里的 `asset/object.usda`；任一步失败都会报错，不静默回退。

产物与版本选择：

- `usd/appearance/v_<时间戳>/`：候选、贴图、对比与六视图。
- `usd/appearance/m5_<时间戳>/report.json`：任务状态、恢复契约、最终资产路径。
- `physics/appearance/<revision>/m4_<时间戳>/asset/`：完整可迁移资产包，复制时保留整个目录。
- `usd/appearance/delivery_selection.json`：通过技术验证的版本选择，`human_approved=false`；页面仅展示，不要求人工操作，不冒充人工验收。
- `usd/appearance/pipeline.json`：最近成功的独立精修任务摘要；新端到端任务由 `input/pipeline.json` 关联本次摘要。

查看效果（将 revision 替换为终端打印的候选版本）：

```bash
.venv-appearance/bin/python -m scripts.appearance.review \
  --run-dir output/6/20261010_110616_147015 \
  --revision v_20261012_160952_923814
```

`--run-dir` 指定物体目录；`--revision` 可重复指定以比较多个版本；`--port 18873` 可指定本机端口，默认自动分配。打开终端打印的 URL 即可查看中文图解页面，页面不修改资产。


### 多视角照片：按角度累计精修

先用 `0.jpg` 运行第 3–6 阶段生成基础资产，再对该轮结果运行：

```bash
.venv-appearance/bin/python -u -m scripts.appearance.multiview \
  --run-dir "$RUN_DIR" \
  --views-dir assets/SKU/0 \
  --prompt 'bottle'
```

| 参数 | 含义与示例 |
| --- | --- |
| `--run-dir` | 已生成基础物理资产的运行目录；沿用该轮 `RUN_DIR`，不会重新生成网格。 |
| `--views-dir` | 多视角照片目录。按 `0 → 90 → 180 → 270 → up → down` 处理存在的照片；必须有 `0`，同一角度只能有一张。支持 jpg/jpeg/png/webp。 |
| `--prompt` | 各视角 SAM3.1 分割目标，默认 `bottle`。 |
| `--config` | 外观/API/Blender 配置，默认 `configs/appearance/default.yaml`。 |
| `--isaac-env` | Isaac Sim 环境脚本，默认使用本机 robot_sim 的激活脚本。 |

角度约定：0 正面、90 右侧、180 背面、270 左侧；up/down 为俯视/仰视。水平角固定，再拟合每张照片的俯仰、滚转、位置和缩放。当前采用正交投影近似；照片角度应与约定一致。

每加入一个视角就建立一个累计候选，保存全部已有视角的参考/模型对比，并调用 GPT 检查。可见性由射线遮挡检测决定；重叠面仅在新照片更正对表面时改用新照片，接近的评分保留较早视角。每张照片占独立纹理图块，不会直接把后视图覆盖到正面贴图上。未覆盖的面保留原始顶点颜色；透明材质和照片光照分离尚未处理，接缝仍可能可见。

照片、分割与逐步 GPT 检查保存在 `input/multiview_<时间戳>/`；累计候选保存在 `usd/appearance/v_*/`。每个候选的 `renders/multiview_comparison.png` 按角度排列，每对左侧为原照片，右侧为累计模型。最终通过独立打包、移位加载、贴图消融与掉落测试后，资产路径写入 `usd/appearance/pipeline.json` 的 `asset` 字段。高度和质量沿用基础资产，不由精修命令改变。严重方向或标签问题会停止并记录原因，不会当作人工验收通过。
