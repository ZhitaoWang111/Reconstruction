# 外观精修开发进度

更新：2026-10-12。M0、M1、M2、M3、M4、M5 已实测；用户暂时接受 M2 可见面效果，背面和多视角一致性留待多视角证据。常规小问题直接修复，仅在需要讨论方案或缺少必要信息时暂停。


## M4 / M5 最新集成验证

当前独立入口为 `./scripts/refine_appearance.sh --run-dir <物体目录>`，默认照片投射→GPT/MCP 受限反馈→独立物理资产包→迁移后 Isaac 显示、贴图采样和掉落验证。无需人工修正或确认页面操作。新物体的原 `run_pipeline.sh` 命令增加 `--appearance-config configs/appearance/default.yaml` 即启用精修；不加则保持原行为。所有参数、恢复方法和场景命令见 README 第 8 节。

实测证据（2026-10-12）：

- 完整已有物体精修任务：`output/6/20261010_110616_147015/usd/appearance/m5_20261012_163056_134796/report.json`，`status=success`。
- 该任务从原照片重新生成 `v_20261012_163056_162679`，通过自动对齐、贴图烘焙及 USD 检查，GPT/MCP 反馈使用 5897 token。
- M4：`physics/appearance/v_20261012_163056_162679/m4_20261012_163137_195161/report.json`，`passed=true`。迁移时原候选与交付包隐藏，结束已恢复；纹理品红对照改变 45190 像素，显示截图已检查，原质量与运行质量均为 0.5 kg，掉落通过。
- 相同任务通过 `--resume` 恢复成功，复用已验证候选和交付包，没有新增 API 调用或 Isaac 执行。`--revision --skip-feedback` 分支也真实交付通过（任务 `m5_20261012_162949_655517`）。
- 新组合场景：`output/scenes/objects_1_12_appearance_m5_20261012/objects_1_12.usda`。仅 6 号采用当时选中的精修版本，其余 11 个采用原验证资产；完整纹理包复制进场景，原场景未覆盖。`isaacsim_scene_verification.json` 的 `passed=true`，12 刚体、12 网格，质量/地面接触/格位检查通过。场景清单绑定其构建时的版本，后续 selection 更新不会改写旧场景。
- 共 30 项测试通过：原流水线、可选外观阶段的路由/成功与失败传播、纹理篡改拒绝、恢复指纹、反馈预算、确认历史函数、相机矩阵及资产打包。

新资产从 SAM3D 开始的新增参数分支已完成轻量子进程编排测试，**本轮未重复实际调用 SAM3D 跑完整 3→7 新生成任务**；真实验证覆盖已有 run-dir 从重新投射到最终 Isaac 交付。未执行 M6 全 12 物体精修或多视角一致性验收。

M5 采用 `delivery_selection.json` 独立记录技术验证版本，明确 `human_approved=false`，不伪造人工确认或覆盖历史 `selection.json`。交付和场景消费前检查候选、物理基线和全部交付依赖指纹；失败显式退出，不静默降级。已选择资产被修改会报错。中断/失败的 API 反馈不自动重放，需显式另起任务；脚本或配置变动后不能沿用旧恢复契约。

## 当前页面模式：仅展示效果

按用户最新要求，页面保留候选版本切换、四列对比、轮廓放大图、六视图和中文说明，移除人工相机修正及接受/拒绝/回退操作。HTTP POST 操作接口已停用，返回 405；浏览页面不会修改候选或版本选择。启动命令仍使用 `scripts.appearance.review`，参数不变，历史确认工具函数保留但页面不调用。下方 M3 原开发记录中的人工操作描述仅作为历史实现记录，当前以此展示模式为准。

## M3 相机修正、受限反馈与人工确认

```bash
.venv-appearance/bin/python -m scripts.appearance.feedback \
  --run-dir output/6/20261010_110616_147015 \
  --revision v_20261012_153211_917757
```

| 参数 | 含义 | 可选项 / 示例 |
|---|---|---|
| `--run-dir` | 已有物体运行目录，不重新生成几何 | `output/6/20261010_110616_147015` |
| `--revision` | 通过技术检查的起始候选名称 | `v_20261012_153211_917757` |
| `--config` | API、Blender 和预算配置 | 默认 `configs/appearance/default.yaml` |

GPT 只允许调用 `adjust_camera` 或 `stop_evaluation`，不允许执行自定义 Python、修改几何或接受资产。固定本地模板通过专用 Blender MCP 19877 渲染；原 9876 会话保持独立。每次反馈保存新的 `m3_<时间戳>/report.json`，包括工具请求、理由、usage、预算预留和停止原因。候选仍使用新的 `v_<时间戳>`，不覆盖旧候选。

配置项：`max_iterations: 3` 为最大轮数；`max_api_calls: 4` 为请求上限（失败请求也占用）；`max_total_tokens: 60000` 为保守预留预算，每次预留 12000 输入 token 加 `max_completion_tokens: 2048`；`max_wall_seconds: 1200` 限制任务时间，超时取消工具/子进程（另有清理时间）；`request_timeout_seconds: 90` 为单次 API 超时。图片最长边压到 768，历史仅保留上一条工具请求和相应 `tool_call_id` 结果。usage 缺失或超过预留即停止新调用，不无限重试。token 预留是估算，服务端实际单次计数可能超过预留，不能视为网关硬配额。

连续两轮轮廓 IoU 未改善就停止，保留轮廓指标最佳候选及所有成功候选供人工比较；轮廓指标不代表纹理质量。API/工具异常、预算耗尽和中断都进入待确认；中断后重新运行确认命令即可，反馈请求不会自动重放。当前不支持自动续接中断的模型循环。

```bash
.venv-appearance/bin/python -m scripts.appearance.review \
  --run-dir output/6/20261010_110616_147015 \
  --revision v_20261012_153211_917757 \
  --revision v_20261012_160952_923814
```

| 参数 | 含义 | 可选项 / 示例 |
|---|---|---|
| `--run-dir` | 原运行目录 | 同上 |
| `--revision` | 可重复指定，切换比较候选 | 上例比较原 M2 与相机修正链路候选 |
| `--port` | localhost 服务端口 | 默认 `0` 自动分配；例如 `18873` |
| `--config` | 修正后生成新候选时的 Blender 配置 | 默认配置文件 |

终端打印 `APPEARANCE_REVIEW_URL`，在本机浏览器打开。页面展示原照片/原顶点色/新贴图/轮廓及六视图；可接受、拒绝、明确回退原版，或调整相机生成新候选。页面关闭、服务停止不会自动接受。接受写入 revision 的 `review.json` 和 `usd/appearance/selection.json`，绑定源输入及全部阶段文件 SHA256；`verified_selection()` 会拒绝被修改、被撤销的确认。拒绝不会静默回退或删除候选；回退原版是独立决定。M5 尚未读取 selection 切换主流水线/场景。

相机使用正交模型和轮廓自动拟合，输出 IoU、归一化对称轮廓距离和青色轮廓图。IoU 小于 0.90 时进入 `needs_alignment`，不执行贴图烘焙；可以将该 revision 传给 review 后修正。相机界面范围：仰角 -60～60°、滚转 -30～30°、yaw -180～180°、相对高度缩放 0.7～1.5、横纵中心 0.2～0.8。相机矩阵要求有限、正交、右手系；图像比例和几何中心不能改变。对称瓶体 yaw 仍需人工结合标签判断；本轮未实现透视相机或点击 2D/3D 对应点求解。

实测：

- `m3_20261012_160735_250832/report.json`：真实 Blender MCP 渲染 + GPT-6 工具反馈成功，usage **5854 token**。模型认为 IoU 0.91789 和正面投射足够，调用停止工具；未自动接受、未为了迭代而强行修改。
- 最新复验 `m3_20261012_161207_422338/report.json`：修正链路候选经 MCP + GPT-6 再次反馈成功，5824 token，同样停止并等待人工确认。
- `v_20261012_160952_923814/report.json`：从已有相机显式修正入口重新跑对齐→烘焙→USD→对比，技术检查通过，保留原几何；它用于验证修正执行链路，并不声称外观优于原候选。
- 17 项测试通过，包括矩阵/输入边界、确认后篡改、拒绝和回退语义、对齐失败拒绝接受、预算预留与工具白名单。
- 确认服务页面、状态、三张图像、错误 token 拒绝、待确认不写 selection 已通过 HTTP 检查。当前自动化环境没有可连接浏览器，未完成真实浏览器按钮/布局视觉验收；人工页面交互需用户实际使用验证。
- 首次 MCP 探针失败留存于 `m3_20261012_160637_401092`，后续成功报告保留。原 M2/M4 产物未覆盖。

范围保持单照片可见侧。用户此前对 M2 的“暂时通过”仍为历史接受意见；本次新候选不冒充已获人工确认。下一步 M5 将把确认版本接入主流水线与场景选择。

## M4 独立资产交付：通过

最终交付目录：`output/6/20261010_110616_147015/physics/appearance/v_20261012_153211_917757/m4_20261012_155144_805698/`。

```bash
.venv-appearance/bin/python -m scripts.appearance.delivery \
  --run-dir output/6/20261010_110616_147015 \
  --revision v_20261012_153211_917757
```

| 参数 | 含义 | 可选项 / 示例 |
|---|---|---|
| `--run-dir` | 已有物体目录，必须包含原物理 USDA | `output/6/20261010_110616_147015` |
| `--revision` | 已完成技术检查的 M2 revision；校验各阶段产物哈希和原输入 | `v_20261012_153211_917757` |
| `--isaac-env` | Isaac Sim 环境激活脚本 | 默认 `/home/seer/wzt/robot_sim/scripts/activate_env.sh` |

每次新建 `physics/appearance/<revision>/m4_<时间戳>/`，不修改原物理资产或自动切换场景版本。本命令执行技术验证，不替代 M3 的用户确认流程。

交付内容：

- **`asset/` 是可复制的完整资产包**：`object.usda`、`textures/`、`package.json`；使用时将整个文件夹复制，不能只复制 USDA。
- `render/object_isaacsim.png` 和 `render/object_view.usda`：Isaac 显示截图和带地面查看场景；查看场景旁另带纹理依赖，保存后重新将 Isaac 扁平化写出的绝对纹理路径改为相对路径，并检查依赖全部在场景目录内。
- `ablation_render/`：用纯品红贴图替换后的对照渲染。
- `drop/isaacsim_physics_verification.json`：新的掉落报告。
- 顶层 `report.json`、各阶段日志、`relocation_journal.json`：绑定本次候选、原物理资产、工具和交付文件 SHA256。

实测证据：

1. 不重新计算物理参数，精确复制原质量、质心、惯量、主轴、碰撞、摩擦及 PhysX 配置；重新打开后验证物理属性完全一致，几何数组和变换完全一致。
2. 扁平化 USD 组合，收集 asset 类型属性中的纹理，改为内容哈希命名的相对路径；USD dependency closure 全部位于包内。当前不支持 UDIM、动画 asset 路径和嵌套 MDL/其他 USD 资源，会明确报错，不静默漏包。
3. 包复制到 `/tmp/appearance_m4_relocated_*/asset` 后，暂时改名隐藏源 M2 revision 和交付包；仅加载迁移副本，完成 Isaac Sim 5.1 渲染和掉落。结束或异常时在 finally 恢复目录，保留恢复日志；强制杀进程时可按 journal 恢复。
4. 原贴图与品红对照使用相同场景和视角，**45,127 像素**发生超过 20/255 的变化，证明贴图实际被采样。截图已视觉检查，标签正确显示。
5. 600 步掉落测试全部通过：下落超过 10cm、无深度穿地、末态接近地面、线速度和角速度达到阈值。资产和运行时质量均为 **0.5 kg**，未传入新的质量覆盖参数。
6. 新增三项包测试：迁移后本地依赖解析及物理保持、缺失纹理拒绝、几何不一致拒绝；加上既有七项，共十项通过。

用户已暂时接受 M2 可见面效果；本次 M4 通过只证明该候选可显示、可仿真、可迁移，不证明背面纹理准确或多视角一致。未替换原 12 物体场景。下一步为 M3 自动对齐、反馈及确认流程，随后 M5 流水线接入。

## M2 照片投射闭环：用户暂时通过可见面效果

样例版本：`output/6/20261010_110616_147015/usd/appearance/v_20261012_153211_917757/`。

```bash
.venv-appearance/bin/python -m scripts.appearance.texturing \
  --run-dir output/6/20261010_110616_147015
```

| 参数 | 含义 | 可选项 / 示例 |
|---|---|---|
| `--run-dir` | 已有物体的四目录运行路径，不重跑 SAM3D | `output/6/20261010_110616_147015` |
| `--config` | 配置文件，默认 `configs/appearance/default.yaml` | `texture_size` 为 1024、2048、4096；默认 2048 |
| `--camera` | 可选 JSON 相机字段覆盖，捕获到候选中 | `{"height_m": 0.1483, "offset": [0.498, 0.509]}`；`basis` 为 right/up/toward 三行，`center` 单位米 |

每次新建不可变 revision；本入口暂不提供 M2 恢复参数，不接受后不影响原版。单运行目录加锁，失败保留阶段日志和报告。M1 的旧 revision 在工具代码变化后不能原样恢复，应新建基线。

已实测：

- 6 号物体相机轮廓 IoU **0.9179**；实际三角面栅格化验证，低于 0.90 停在 `needs_alignment`，可用相机 JSON 修正后新建候选。正交模型、固定 yaw 仍有歧义，不代表 M3 自动对齐完成。
- 原图 mask 内缩、表面朝向和 BVH 射线可见性控制投射。默认 2K 基础色，sRGB；照片覆盖图为线性数据。
- 正面采用照片投影 UV chart。实验发现圆柱展开在重建肩部重叠，改为照片可见面投影；未观测面用 USD 材质子集直接保留原顶点颜色，避免圆柱烘焙损失原细节。没有镜像正面文字到背面。
- 同相机无光照对比中，“康师傅 / 冰红茶 / 冰力十足”明显比原顶点颜色更清楚；六视图检查仍可见侧边过渡和原始背面模糊。
- 重开 USDA 后逐项验证 points、face counts、face indices、displayColor 完全一致：229,190 顶点、458,364 面、高度 0.14 米；面角 UV 数 1,375,092，材质绑定与相对纹理路径可解析。
- 原资产输入哈希在完成时再次校验。七项契约/坐标自动测试通过。

主要产物：`object.usda`、`work.blend`、`camera.json`、`uv.npy`、`observed_faces.npy`、`textures/basecolor.png`、`textures/photo_coverage.png`、`usd_validation.json`、`report.json`，以及 `renders/comparison.png`、`renders/alignment.png`、`renders/aligned_{baseline,refined}.png` 和六视图。

打开工程（保留贴图文件夹）：

```bash
/home/seer/wzt/Blender/.local/opt/blender/blender \
  output/6/20261010_110616_147015/usd/appearance/v_20261012_153211_917757/work.blend
```

报告 `status=awaiting_review`、`technical_checks_passed=true`、`human_approved=false`、`isaac_verified=false`。这是 M2 原报告中的历史状态；用户随后暂时通过可见面效果，背面/多视角一致性延期。M4 已单独验证交付包，M2 原报告保持不可变。尚未生成 selection 或替换原物理资产。M3 提供模型反馈与确认页面。M2 本次未新增 API 调用。

局限：照片高光和阴影保留在基础色中；单张图片无法恢复真实背面；当前投射只验收了瓶体样例，盒/袋未验证。没有主张物理透明度、去光照或全自动外观准确率。

失败实验保留：`v_20261012_152759_933743`（UV 越界检查）、`v_20261012_152925_499814`（密集网格 Smart UV 引发 Blender 崩溃）；随后改用确定性投影，后续候选正常完成。中间候选仅供诊断，以本节指定的最终版本为准。

## M1 最新验收：通过

报告：`output/6/20261010_110616_147015/usd/appearance/v_20261012_151712_303828/report.json`，`status=baseline_ready`，`passed=true`。

- 已修正 ValidateTopology 的 Vt.IntArray 参数类型。
- 从 manifest 解析捕获的 RGB/mask，校验快照哈希；完成时再次确认原始资产文件未变。
- 规范网格指纹：229,190 顶点、458,364 面、高度 0.14 米；Blender 建模后逐项校验顶点和面索引保持一致。
- 保存原图裁剪、mask、隔离背景图、六视图、相机记录、work.blend；六视图已经视觉检查。
- 主动暂停 reference 阶段后恢复成功；完整恢复时三个阶段全部复用。
- 配置变化拒绝复用，损坏产物不能作为成功证据。
- 五项自动测试通过：原子写失败、损坏重跑、并发锁、输入快照和路径、失败阶段复用保护。

恢复命令：

```bash
.venv-appearance/bin/python -m scripts.appearance.baseline \
  --run-dir output/6/20261010_110616_147015 \
  --resume v_20261012_151712_303828
```

`--resume` 指定 revision ID，要求输入/配置/工具指纹匹配；省略则创建新版本。`--stop-after geometry|reference` 可在检查点主动暂停。基线完成不代表 M2 贴图精修已经实现。

## M1 首次失败记录（已修正，保留诊断历史）

已编写 `contracts.py`（输入快照解析、SHA256、配置校验、原子 JSON、运行锁、revision 与产物指纹）、`extract_geometry.py`（规范网格和几何指纹）、`baseline_blender.py`（原规范网格建模及六视图基线）、`baseline.py`（裁剪图、阶段报告及指纹恢复入口）。这些功能尚未整体验收通过，不能视为 M1 完成。

实测命令：

```bash
.venv-appearance/bin/python -m scripts.appearance.baseline \
  --run-dir output/6/20261010_110616_147015 \
  --stop-after reference
```

`--stop-after reference` 计划用于在几何和参考图片阶段完成后暂停，再验证恢复；本次未到达该检查点。

失败目录：`output/6/20261010_110616_147015/usd/appearance/v_20261012_151547_997975/`。具体错误见 geometry.log：`UsdGeom.Mesh.ValidateTopology` 不接受 NumPy ndarray，要求 USD `VtArray<int>`。建议显式转换为 `Vt.IntArray` 后继续。当前原始资产未修改；六视图、中断恢复、源码变化失效和并发锁测试待执行。

## 已实现

- 独立 `.venv-appearance`，安装本地 MCP checkout，不修改 SAM3D Python 环境。
- `configs/appearance/default.yaml` 与依赖版本快照。
- `src/objgen/appearance/api_client.py`：Chat Completions 图片输入、工具调用、完成状态检查及脱敏错误。
- `src/objgen/appearance/mcp_client.py`：本地 stdio MCP 会话和模板执行结果检查。
- `src/objgen/appearance/blender_bootstrap.py`：独立进程完整注册插件，禁用自动启动及更新回调，手动绑定 19877，不安装全局插件或保存偏好设置。
- `src/objgen/appearance/preflight.py`：照片→GPT-6 工具请求→Blender 渲染→模型反馈探针，保存失败报告并清理自建进程。

运行方式（能力探针，尚不是精修入口）：

```bash
.venv-appearance/bin/python -m scripts.appearance.preflight \
  --run-dir output/6/20261010_110616_147015
```

`--run-dir` 为已有物体运行目录；可通过 `--config` 指定配置文件，默认 `configs/appearance/default.yaml`。该探针不重新运行 SAM3D，不修改原 USD 或物理资产，在 `usd/appearance/m0_<时间戳>/` 保存报告。

## 最新实测：M0 通过

报告：`output/6/20261010_110616_147015/usd/appearance/m0_20261012_151043_923725/report.json`，`passed=true`。

- GPT-6 接收原始照片，返回指定 render_asset 工具调用。
- 独立 Blender 4.5.14 LTS 完成插件握手、导入 GLB、通过 MCP 执行模板、保存 probe.blend 和渲染 512×512 baseline.png。
- GPT-6 接收渲染图，指出标签文字模糊失真、瓶盖颜色及细节与照片不一致；最终 finish_reason=stop。
- 两次 API 请求合计 37,894 token（网关 usage，含缓存输入），不是费用估算。
- 自建 Blender 已退出，原有 9876 会话保留。
- 异常组具体原因展开已实现。
- M1—M6 尚未开发，M0 成功不代表外观精修成功。

## 首次实测：插件注册失败，已修正

报告：`output/6/20261010_110616_147015/usd/appearance/m0_20261012_150854_296633/report.json`。

- GPT-6 图片输入和指定工具调用：成功，返回模型标签 `gpt-6-astra`，finish_reason 为 `tool_calls`。
- MCP 启动及独立端口 19877 连接：成功；现有 Blender 9876 会话保持独立。
- 插件握手及执行：失败，`Scene` 缺少 `blendermcp_use_polyhaven` 属性。
- 渲染、工程保存及模型看渲染反馈：未完成。
- M0：不通过。M1—M6：未开发。

原因：bootstrap 只实例化上游 socket server，没有执行 `addon.register()`，而命令路由读取 register 创建的 Scene 属性。原设计意图是避免全局安装和自动更新，但绕过注册不满足上游接口要求。

经用户确认，已在自建 Blender 进程内执行完整插件注册，同时覆盖自动启动与更新回调，仅手动绑定专用端口；未安装到用户 addons、未保存用户 preferences、未修改现有会话。

后续需增加明确的图像尺寸策略和请求预算限制。首次失败的报告保留用于诊断；当前状态以最新通过报告为准。

## 2026-10-12 SKU/0 多视角累计精修

新增 `src/objgen/appearance/multiview.py` 入口：自动按文件名排序、拍照输入快照、逐视角 SAM3.1 分割、固定方向的轮廓拟合、累计可见面纹理来源选择、逐步 GPT 全部已有视角检查，以及最终 M4/M5 物理资产交付。基础网格、尺寸和质量不因照片纹理步骤改变。对应实现为 `multiview_geometry.py` 和 `multiview_blender.py`；展示页增加中文多视角对比解释。

SKU/0 照片只有 0/90/180/270，没有 up/down。本轮基础目录为 `output/0/20261012_170949_216484`，尺寸约 0.04827×0.04734×0.14 m，质量 0.5 kg。初始默认求解器掉落测试的角速度残余约 0.103 rad/s 未通过，延长到 1200 步仍未通过；改用已有参数 velocity iterations=8、position iterations=64 后，600 步实测全部通过，末态角速度约 0.0348 rad/s。不降低测试阈值，不改变质量。

第一轮侧视角分割受未限制的 CPU 环境影响，SAM3.1 初始化报错；新入口已继承本机 Haswell/OpenBLAS 和 CPU 0 约束。失败输入/候选保留，重新建立独立任务。

限制：按照片角度固定 yaw，采用正交近似；照片光照保留，透明瓶身当前仍是照片颜色材质。采用独立照片图块和逐面最佳入射选择，未观测面保留原始顶点颜色，暂未进行接缝颜色融合。up/down 排序和相机入口已有代码，当前只有四个水平视角的真实证据，不能宣称已实测顶部/底部。

本轮最终真实验证完成：

- 逐视角任务：`output/0/20261012_170949_216484/input/multiview_20261012_171525_656096/report.json`，顺序 0/90/180/270，4 次 GPT 检查均 pass。
- 最终累计候选：`usd/appearance/v_20261012_171732_409573`；四视角轮廓 IoU 分别 0.95698/0.95668/0.94983/0.96116。
- 最终交付：`physics/appearance/v_20261012_171732_409573/m4_20261012_171809_013502/asset/object.usda`（均相对本轮目录）。M4 验证全部通过，贴图消融改变 37,278 像素，实际质量 0.5 kg；`selected_asset()` 重新校验候选和交付全部哈希通过。
- M5 任务：`usd/appearance/m5_20261012_171808_796922/report.json` success；原始 pipeline 的失败记录保留，恢复说明为 `input/pipeline_recovery.json`，不把第一次失败改写成成功。
- 自动测试共 34 项通过，覆盖原有流程回归、照片顺序、缺少正面、重复角度、水平及上下方向的相机基。up/down 仍无真实照片实测。

### SKU/0 第一版透明瓶身

独立产物位于本轮目录 `physics/transparent/v1/`，资产为 `asset/object.usda`。新增 `src/objgen/appearance/transparent_bottle.py`：按此直立瓶的高度将三角面划分为标签（z=0.025–0.076 m）、红盖（z≥0.121 m）和其余瓶身。标签保留四视角贴图，瓶盖改为纯红塑料，瓶身去掉照片背景，使用 UsdPreviewSurface 浅色透明效果（opacity=0.18，roughness=0.12，ior=1.47）。这是 alpha 透明近似，没有瓶壁厚度、液体或真实 transmission/refraction 模型；不能仅凭 ior 参数声称物理折射正确。

实测：网格数组和物理属性逐项保持一致；Isaac Sim 显示通过，红/绿/蓝背景可透过瓶身，标签和瓶盖仍遮挡背景；另一个 opacity=1 的同材质控制资产用于渲染对比，差异像素记录在 `report.json`。600 步掉落全部通过，实际质量 0.5 kg。`transparency_comparison.png` 是不透明控制/透明版本对比，`render/object_view.usda` 为带地面的可打开场景，纹理路径已本地化。当前独立交付，不改写原多视角交付选择；标签边界为人工数值高度分段，局部反光和原始几何凹凸仍较明显。
