# 参考实现的可借鉴经验：LiteReality-Agent 与 scenesmith

记录两套参考仓库里"组装放置 agent"部分的做法、代码位置，以及对我们场景组装设计的启发。

- 读码日期：2026-11-11。只读了代码和文档，**两套都没有运行过**。
- 仓库位置：`/home/seer/wzt/Reconstruction/Submodule/`
  - `LiteReality-Agent`，提交 `28ffe1f`
  - `scenesmith`，提交 `67cc408`
- 下文路径都相对各自仓库根目录，`文件:行号` 指函数或常量的定义行。
- "对我们的启发"是我的判断，不是两套代码的原话；方案本身还没定。
- 我们这边的现状见 `scene-assembly-design.html` 和 `scene-assembly-pilot-walkthrough.html`。

## 0. 一览

| # | 经验 | 来源 | 对应我们批 1 的问题 |
|---|---|---|---|
| L1 | 先在盒子上修布局，再生成和放置资产 | LiteReality | 占地不符、资产被压扁 |
| L2 | 确定性修复是"合法动作搜索"，不会比输入更差 | LiteReality | 无自动修复 |
| L3 | 模型只回一个离散动作，厘米数由几何算 | LiteReality | agent 直接调 7 个数 |
| L4 | 每个提议过接受门：违规严格下降才保留，拒绝原因回传 | LiteReality | 警告不拦截提交 |
| L5 | 相对扫描的累计改动上限 | LiteReality | 无上限 |
| L6 | 是否穿插由真实网格判定，另有"合理重叠"允许名单 | LiteReality | 单向测量，无允许名单 |
| L7 | 每个物体声明支撑，再用底面接触核对 | LiteReality | 支撑只是一个字段 |
| L8 | 检查结果自带修复向量 | LiteReality | 警告只有数值 |
| L9 | 渲染只以"渲染 \| 同机位照片"成对返回 | LiteReality | 已有，可对照 |
| L10 | 提示词给目标和验收门，不规定过程 | LiteReality | 提示词写法 |
| L11 | 每次运行都画一张修复前后对比图 | LiteReality | 只有版本链 |
| S1 | designer / critic / planner 三角色，critic 只读 | scenesmith | agent 自己验收自己 |
| S2 | 按评分回滚到上一个检查点 | scenesmith | 无回滚 |
| S3 | `snap_to_object`：一次调用完成解穿插、定朝向、推到接触 | scenesmith | 床头柜 × 床穿模 |
| S4 | 朝向用工具核对，不凭透视图判断 | scenesmith | 包的朝向选错 |
| S5 | 支撑面提取 + 面内二维坐标放置，三项硬校验 | scenesmith | 纸巾盒、枕头的落点 |
| S6 | 穿插用碰撞体有符号距离，1 mm 阈值，物体与自己的支撑物放宽 2 cm | scenesmith | 单向、2 cm 警告 |
| S7 | 每件家具摆完做"去穿插投影 + 重力仿真" | scenesmith | 无物理 settle |
| S8 | 观察图带标注：坐标、朝向箭头、包围盒、编号 | scenesmith | 没有俯视图 |
| S9 | 上下文控制：旧图剥离、轮次摘要、重复调用拦截 | scenesmith | 网关缓存只写不读 |
| S10 | 资产入库前由 VLM 定"哪面朝上、哪面是正面" | scenesmith | 朝上面交给组装 agent 判断 |

---

## 1. LiteReality-Agent

输入是 Apple RoomPlan 给的墙体和带朝向的物体盒（centre / size / yaw），目标是可仿真的房间。和放置有关的代码在三处：`pipeline/scene_init/layout/`（盒子阶段的布局修复）、`room_ops/compile/build_room.py`（资产进盒）、`pipeline/room_qc/`（组装后的检查和修正）。以下路径省略前缀 `src/litereality_agent/`。

### L1 先在盒子上修布局，再生成和放置资产

- **做法：** 布局修复排在"合并盒子之后、裁图之前"。理由是盒子上的修正没有成本；等资产按错的尺寸生成后再压扁，或盒子在裁图后移动，代价都更大。
- **资产进盒：** 资产的 X/Y/Z 分别缩放到盒子的宽/深/高，朝向取盒子的 yaw，不做长宽比交换。
- **代码位置：**
  - 说明文档：`pipeline/scene_init/layout/README.md`
  - 入口：`pipeline/scene_init/layout/stage.py:105`（`run_layout`），调用处 `pipeline/scene_init/flow.py:147`
  - 修好的盒子写回给组装器：`pipeline/scene_init/layout/stage.py:72`（`_publish_shell`）
  - 资产进盒：`room_ops/compile/build_room.py:757`（`_place_in_box`）
  - 放到台面上：`room_ops/compile/build_room.py:1126`（`place_on`），台面高度来自 `:1115`（`top_surface`）
- **对我们的启发：** 我们没有 RoomPlan 的盒子，但可以从扫描点云求带朝向的盒子（最小面积矩形），资产按两个占地轴分别缩放去贴。批 1 里床的轴对齐范围是 1.98×2.32，真实带朝向占地是 2.02×1.53，正是这一步缺失造成的。

### L2 确定性修复是"合法动作搜索"

- **做法：** 每个违规提出几个人能看懂的动作，按侵入程度从小到大：贴墙（snap）、同墙两件沿墙滑开（dewrap）、沿墙滑动（slide）、按最短分离向量推开（separate）、裁掉过深的盒子（shrink）。每个动作在副本上试并打分。
- **评分：** 字典序 `(违规数 + 被破坏的贴墙关系 + 移动过远的件数, 总穿插深度, 总位移)`。穿插深度排第二，是为了让"需要两步的修复"能迈出第一步。
- **保证：** 用束宽 3 的搜索，允许中途经过更差的状态，但返回的是"见过的最好状态"，输入也在其中，所以不会比输入更差。
- **删除默认关闭：** 删重复检测是唯一"失败了看不出来"的动作，需要 `LR_LAYOUT_DROP=1` 才启用。
- **代码位置：**
  - 设计说明：`pipeline/scene_init/layout/repair.py:1`（模块文档）
  - 评分：`repair.py:164`（`score`）
  - 候选动作：`repair.py:382`（`_candidates`），贴墙 `:227`，角落贴合 `:291`，裁深 `:354`，两件各裁一点 `:331`
  - 搜索主循环：`repair.py:592`（`repair`）
  - 移动时带上同组物体（桌子带椅子）：`repair.py:154`（`groups`）、`:212`（`_shift`）
  - 检查项和阈值：`adjust.py:425`（`check`），阈值 `adjust.py:74-86`
- **对我们的启发：** 我们没有任何自动修复。可以先做最小的两个动作：沿最短分离向量推开、落到支撑面，并采用"在副本上试、变好才保留"的框架。

### L3 模型只回一个离散动作，厘米数由几何算

- **做法：** 确定性修复解决不了的违规才问模型，每个物体问一次。输入是该物体的几何、相邻物体、相关墙体、之前被拒的尝试，以及两张画了盒子的原始照片。
- **输出：** 一个 JSON，动作只能是 `resize / translate / attach / split / none`。`attach` 只给墙的名字，贴多少、角落里裁多少由几何算。`resize` 之后由代码把物体重新贴回它原来靠的墙。
- **源码注释的原话意思：** 模型擅长判断"这个衣柜属于那个角落"，不该由它算厘米。
- **代码位置：**
  - 上下文组装：`pipeline/scene_init/layout/agent.py:51`（`_context`）
  - 提示词：`agent.py:106`（`PROMPT`）
  - 单次提问：`agent.py:150`（`propose`，用 `claude -p`，只允许 Read 工具）
  - `resize` 后重新贴墙：`agent.py:180`（`_reflush`）
  - 证据图（盒子投影到原始帧，按面积选 2 张，帧间隔至少 3）：`evidence.py:14`（`prepare`）
- **对我们的启发：** 现在 agent 直接给 `(x, y, z, yaw, up, scale_xy, scale_z)`。可以加关系级动作，例如"贴着某物体的某一侧""放在某物体的台面上"，由工具算出位置。

### L4 每个提议过接受门，拒绝原因回传

- **做法：** 提议先过范围检查，再在副本上执行。只有 `(违规数, 总穿插深度)` 严格下降、且没有破坏**其他物体**的贴墙关系时才保留。
- **循环：** 最多 6 轮。每轮对仍有问题的物体各问一次；被拒的提议和拒绝原因写进 `layout_feedback`，下一轮作为 `previous_attempts` 给模型。连续两轮没进展就停。
- **管线门：** 循环结束后重新从磁盘读盒子再检查一遍，还有错误就抛异常中止管线。
- **代码位置：**
  - 范围检查：`agent.py:205`（`_bounded`）
  - 接受门：`agent.py:229`（`apply_proposals`）
  - 多轮循环和停止条件：`converge.py:39`（`solve`）
  - 管线门：`validation.py:75`（`require_valid_layout`），实际检查在 `:20`（`inspect_layout`）
- **对我们的启发：** 对应设计页 §12 "警告是否升级为否决"。批 1 里 8 次提交有 6 次带警告，其中 4 次用错误理由放行了穿插。这里的做法是：不变好就不落盘，并把原因告诉模型。

### L5 相对扫描的累计改动上限

- **做法：** 保存一份扫描基线（带源文件哈希）。任何时候，物体相对基线的水平位移不得超过 0.6 m，垂直不得超过 0.08 m，尺寸变化不得超过 15%（桌、椅、沙发、床）或 35%（其他）。超出的布局即使没有碰撞也算失败。
- **代码位置：**
  - `converge.py:10`（`fidelity_errors`）
  - 基线保存和哈希校验：`stage.py:126-136`
  - 单次提议的上限：`agent.py:35-36`（`MAX_SCALE`、`MAX_SHIFT`）
  - 超过 0.22 m 的移动计为缺陷：`repair.py:67`（`REVIEW_SHIFT`）
- **对我们的启发：** 如果把穿插改成否决，必须同时加这类上限，否则为了不穿模会把物体推离照片里的位置。我们的基线可以用扫描点云的范围。

### L6 是否穿插由真实网格判定，另有允许名单

- **做法：** 接触的"是/否"用真实三角网格判定（FCL）。包围盒的重叠只作为复查列表，不拦截，因为盒子分不清"椅子塞在桌下"和"椅子穿过桌子"。
- **深度不取自网格：** 生成的网格不是水密的，FCL 给出的穿插深度不可信（源码举例：贴墙的桌子报出 1.16 m）。
- **允许名单：** 一张"本来就该重叠"的类别对表（如水槽嵌在台面里、椅子在桌下），检查和修复共用，避免修复器去"修"正确的重叠。
- **代码位置：**
  - 网格接触：`agent/tools/check_collisions/source/collision_mesh.py:137`（`mesh_contacts`）
  - 全量检查：`collision_mesh.py:182`（`check_all`）
  - 允许名单：`pipeline/scene_init/layout/graph.py:88`（`EXPECTED_CONTAINMENT`）、`:161`（`expected_pair`）
  - 盒子的最短分离向量：`graph.py:311`（`obb_mtv`）
  - 总门里只有网格碰撞拦截：`pipeline/room_qc/validate.py:195`（`validate`）
  - 规则说明：`doc/QC/collision_check.md`
- **对我们的启发：** 我们现在用 `contains` 加表面采样做单向测量，报 4–5 cm 而实际 7–8 cm。应改成双向，并用资产自带的碰撞体。允许名单要有（枕头压床、被子盖床）。

### L7 每个物体声明支撑，再用底面接触核对

- **做法：** 每个物体必须声明 `rests_on` 或 `attached_to`。检查分两层：
  1. 支撑图：目标存在、没有环、能追溯到地面/墙/天花板。
  2. 接触：从物体底面采样点向下，只对支撑物**朝上的三角面**求交；间隙在 1.5 cm 内、下陷在 2 cm 内算接触。还报告接触点凸包面积，以及物体中心是否落在凸包内。
- **柔性物体单独处理：** 枕头、被子、地毯等类别改用整个下表面采样。
- **代码位置：**
  - 支撑图：`pipeline/room_qc/support_audit.py:34`（`support_graph`）
  - 接触核对：`support_audit.py:121`（`vertical_contacts`）
  - 柔性类别：`support_audit.py:31`
  - 汇总：`support_audit.py:261`（`audit`）
  - 规则说明：`doc/QC/supporting_relationship.md`
- **注意：** 该文档还写了"支撑物移动时按从根到叶的顺序带着上面的物体走"，但我在 `room_qc/` 的代码里没找到对应实现。
- **对我们的启发：** 我们的 `rest_z` 已经是向下射线落到支撑物上，但没有事后核对，也没有"中心是否在接触面内"的稳定性指标。批 1 的枕头属于柔性类别，需要单独的判定方式。

### L8 检查结果自带修复向量

- **做法：** `check_collisions` 只检测和建议，不修改。每条结果带上能解除它的具体移动量（世界坐标 Δx、Δy），以及等价的"移另一个"和"缩小"方案。由调用方看图决定移动还是缩小。
- **独立的确定性修复器：** 按"可移动性"决定谁让：贴墙的安装类家具不动，自由摆放类（椅、桌、沙发）让开；只动 XY；单次最多 0.30 m；失败时说明是哪一种卡住。
- **代码位置：**
  - 工具：`agent/tools/check_collisions/tool.py`（网格模式 `:262`，阈值 `:41-44`）
  - 修复器：`pipeline/room_qc/fix.py:131`（`plan_from_mesh`）、`:266`（`plan`），阈值 `fix.py:41-51`
  - 网格版入口：`pipeline/room_qc/correct.py`
- **对我们的启发：** 我们的警告只有"穿插 0.04 m"。加上"沿 (dx, dy) 移 4 cm 可解除"之后，agent 少一步自己估算。

### L9 渲染只以"渲染 | 同机位照片"成对返回

- **做法：** 渲染工具从不单独返回渲染图，每张都是同一相机位姿下的"渲染 | 照片"。三个层级：整间房、单面墙、单个物体。渲染前强制重新编译，避免 agent 看到旧状态。评判由另一个工具 `critic` 做，不混进渲染里。
- **代码位置：** `agent/tools/render/tool.py`、`agent/tools/render/source/compose.py`；说明在 `doc/Tools/rendering_and_compare.md`
- **对我们的启发：** 我们的 `view_pose` 已经是这个形式。可以对照的是"渲染必须反映当前状态"这条不变式。

### L10 提示词给目标和验收门，不规定过程

- **做法：** 作者 agent 有四套提示词。前三套规定了读什么、按什么顺序、多少次调用必须保存一次。第四套 `open` 改成只给目标、证据包、测量工具和最终验收门。
- **源码注释给的理由：** 一次性运行的模型只拿到原始扫描和一页意图，自己先看完所有照片、从深度图量高度、做了穿插和接触自检，结果在布局和固定物上比照本宣科的版本更接近照片。规定节奏原本是为了解决"模型先做便宜的调用、最后才编辑"，这属于 harness 的问题，用结果门解决比用过程规则好。
- **证据包里的测量工具：** 像素反查三维点、两张照片三角化、某占地范围内的水平面高度分布、某高度的点云切片。
- **代码位置：**
  - 四套提示词：`agent/author.py:53`（节奏规则）、`:243`（simulation）、`:371`（open）
  - 预算：`author.py:461`（300 步）
  - 测量工具：`evidence_kit/measure.py:49`（`probe_depth`）、`:83`（`triangulate`）、`:107`（`pcd_slice`）、`:136`（`height_profile`）
- **对我们的启发：** 我们设计里的 `measure` 工具还没做。`height_profile` 这类"给一个占地范围，返回里面有哪些水平面及其高度"的工具，正好能回答"床头柜台面多高"，批 1 里 agent 是靠纸巾盒底面反推的。

### L11 每次运行都画一张修复前后对比图

- **做法：** 布局修复每次都输出一张俯视图：修复前和修复后并排，原位置用虚线叠在下面，每处修正画一个箭头。理由是"违规数为零"分不清"柜子贴回了墙"和"桌子被推了半米"。
- **代码位置：** `pipeline/scene_init/layout/report.py`，调用处 `stage.py:34`（`_visualize`）
- **对我们的启发：** 我们的版本链只有 JSON。带扫描点云轮廓的俯视图既能给 agent 看，也能给人复查。

### 没接上的代码（只作思路参考）

- `pipeline/scene_init/layout/agent_loop.py` 和 `toolcli.py` 是另一种形态：给 agent 一组命令行动词（`summary / object / check / attach / move / resize / drop / undo`），每个修改动作都打印修改前后的评分。但它们引用的 `physical_engine` 模块在仓库里不存在，`solve_v14` 也没有调用方。
- `agent.py:302`（`solve_v10`）里有"只缓存被接受的提议"的做法，理由是同一个问题问两次答案不同，评分会随运行波动。这个函数同样没有调用方。

---

## 2. scenesmith

输入是一段文字，没有真实照片，所以它解决的是"摆得合理"而不是"摆得和照片一样"。可借鉴的是工具设计、验收结构和物理处理。以下路径省略前缀 `scenesmith/`（包目录）。

### S1 designer / critic / planner 三角色

- **做法：**
  - designer 有全部修改工具和观察工具。
  - critic 是独立会话，只有只读工具（观察、读场景状态、朝向核对）。物理检查结果直接写进它每次的指令里，不需要它自己调用。
  - planner 只有四个工具：请求初始设计、请求评审、请求修改、回滚。它把 critic 的意见原文转给 designer。
- **critic 的规则：** 有穿插时真实度最高 4 分；存在严重问题时只许提严重问题，不许夹带优化建议；工具结论优先于看图。
- **代码位置：**
  - 框架：`agent_utils/base_stateful_agent.py:94`（类），planner 工具 `:663`，评审实现 `:762`
  - 家具 critic 的只读工具：`furniture_agents/stateful_furniture_agent.py:114`（`_create_critic_tools`）
  - 提示词：`prompts/data/furniture_agent/`（`stateful_planner_agent.yaml`、`stateful_critic_agent.yaml`、`stateful_critic_runner_instruction.yaml`、`designer_critique_instruction_stateful.yaml`）
- **对我们的启发：** 批 1 里提议、判断警告、提交都是同一个会话。可以先试成本最低的一种：批末另开一个不看 designer 推理的会话，只给渲染对比和检查结果。

### S2 按评分回滚到上一个检查点

- **做法：** 每次评审后保存场景状态和评分，保留最近两份。任一项评分下降 ≥ 2，或总分下降 ≥ 1，planner 被建议回滚；流程结束时代码再自动检查一次，变差就恢复上一份。
- **配置：** 家具最多 3 轮评审、全部 ≥ 9 分提前结束；小物件最多 2 轮、≥ 8 分提前结束。
- **代码位置：**
  - 判定：`agent_utils/base_stateful_agent.py:400`（`_should_reset_to_checkpoint`）
  - 回滚：`:442`（`_perform_checkpoint_reset`），结束时的自动检查 `:479`
  - 场景状态序列化：`agent_utils/room.py:1570`（`to_state_dict`）、`:1596`（`restore_from_state_dict`）
  - 配置：`configurations/furniture_agent/base_furniture_agent.yaml:244-252`，`configurations/manipuland_agent/base_manipuland_agent.yaml:303-311`
- **对我们的启发：** 我们的版本链本来就是不可变快照，回滚很便宜。缺的是"什么情况下回滚"的判据；我们有留出帧 IoU 和穿插量，比主观打分更适合做判据。

### S3 `snap_to_object`：一次调用完成解穿插、定朝向、推到接触

- **做法：** designer 先把物体放在大致位置，然后调用 `snap_to_object(物体, 目标, orientation)`。工具依次做三件事：
  1. 如果已经穿插，先保守推开（按最大尺寸当作圆，保证之后旋转不会再穿）。
  2. 按 `toward / away / none` 设置朝向。
  3. 沿方向以 1 cm 步长移动，用碰撞体（凸分解）检测，直到接触，留 1 cm 缝。
- **为什么逐步检测：** 用顶点投影会被高处的顶点（如桌面）挡住，椅子就推不进桌下；用凸分解后的碰撞体逐步检测可以。
- **代码位置：**
  - 工具定义：`furniture_agents/tools/scene_tools.py:112`，实现 `:578`
  - 定朝向：`scene_tools.py:411`（`_compute_and_apply_orientation`）
  - 先推开：`furniture_agents/tools/snapping_helpers.py:520`（`resolve_collision_if_penetrating`）
  - 逐步推到接触：`snapping_helpers.py:320`（`snap_with_iterative_collision_check`）
  - 算法选择：`snapping_helpers.py:782`
  - 参数：`configurations/furniture_agent/base_furniture_agent.yaml:329` 起的 `snap_to_object` 段（步长 0.01，留缝 0.01）
  - 提示词里的用法规定：`prompts/data/furniture_agent/designer_agent.yaml` 的 "FURNITURE PLACEMENT PROCEDURE" 一节
- **对我们的启发：** 这是和批 1 穿模最直接对应的一条。床头柜 × 床、纸巾盒 × 床都是"应该贴着但不穿"的关系，agent 说关系、工具推到接触即可。我们的版本不需要定朝向那一步，只需要"先推开，再沿指定方向推到接触"。

### S4 朝向用工具核对，不凭透视图判断

- **做法：** `check_facing_tool(A, B, toward/away)` 返回是否朝向正确，以及需要的绝对 yaw。designer 和 critic 的提示词都写明：透视图会造成错觉，朝向必须用工具核对，工具结论优先。
- **代码位置：** `furniture_agents/tools/scene_tools.py:73`（定义）、`:175`（实现）；critic 提示词 `prompts/data/furniture_agent/stateful_critic_agent.yaml` 的 STEP 3
- **对我们的启发：** 批 1 里包的朝向只凭一张关键帧选定，留出帧上另一个候选更好。我们没有"正面"的定义，但可以做等价的事：朝向类判断必须在多于一个视角上核对，并把各候选在各视角的轮廓吻合度作为数值给出。

### S5 支撑面提取 + 面内二维坐标放置

- **做法：** 小物件 agent 一次只处理一件家具。先从家具网格提取水平支撑面：按法向聚类、按高度拆分、拟合平面、向上射线求净空（取第 10 百分位，避开隔板）。每个面有 id、面积、高度、净空。
- **放置：** 模型只给 `(面 id, 面内 x, 面内 y, 旋转)`，工具换算成世界位姿。三项校验不过就报错，不落盘：
  1. 落点在面的凸包内。
  2. 物体凸包占地在面内（顶面允许 5% 悬出，隔层不允许）。
  3. 物体高度不超过该面净空。
- **代码位置：**
  - 提取入口：`agent_utils/support_surface_extraction.py:1004`（`extract_support_surfaces_from_mesh`）
  - 法向聚类 `:169`，按高度拆分 `:275`，净空 `:587`
  - 面的数据结构和坐标换算：`agent_utils/room.py:168`（`SupportSurface`）、`:243`（`to_world_pose`）、`:301`（`contains_point_2d`）
  - 工具：`manipuland_agents/tools/manipuland_tools.py:516`（`list_support_surfaces`）、`:557`（`place_manipuland_on_surface`）
  - 放置实现和三项校验：`manipuland_tools.py:988`，占地校验 `:392`
  - 每件家具一个新会话：`manipuland_agents/stateful_manipuland_agent.py:681`（`add_manipulands`）
- **对我们的启发：** 我们"支撑高度继承"是靠提示词让 agent 推理的。把支撑面做成显式对象后，"纸巾盒放在床头柜上"变成选一个面加一个面内坐标，高度不再需要 agent 推。前提是资产网格的台面要足够平，SAM3D 资产需要先验证。

### S6 穿插用碰撞体有符号距离

- **做法：** 用 Drake 对所有物体的碰撞体求成对有符号距离，只取穿插的。阈值 1 mm。地面穿插放宽到 5 cm。**当前物体与它所在家具**的接触放宽到 2 cm，其他一律不放宽。检查时所有物体都设为自由体，这样固定的家具之间也会被查到。
- **按 agent 类型过滤：** 每个 agent 只看到自己能改的物体的违规。
- **代码位置：**
  - 计算：`agent_utils/physics_validation.py:155`（`compute_scene_collisions`），距离查询 `:223`
  - 放宽规则：`physics_validation.py:330`（`_should_skip_collision_pair`）
  - 汇总成文字：`agent_utils/physics_tools.py:157`（`check_physics_violations`）
  - 阈值：`configurations/manipuland_agent/base_manipuland_agent.yaml:314` 起的 `physics_validation` 段
- **对我们的启发：** "物体和它自己的支撑物之间放宽，其他不放宽"是一条可以直接采用的规则，比按类别写允许名单简单。

### S7 每件家具摆完做"去穿插投影 + 重力仿真"

- **做法：** 两个阶段：
  1. 投影：用带最小距离约束的优化把穿插的物体挪开，旋转固定。
  2. 前向仿真 5 秒，让物体在重力下落稳。掉到地上的小物件删除。
- **分而治之：** 每件家具摆完后，只取"墙 + 地面 + 这件家具 + 它上面的物体"做一次，家具固定；全部结束后再做一次全场景的。任何一步失败都退回原场景。
- **一个负面结果：** 让 agent 主动调用的 `resolve_penetrations` 工具被注释掉了，注释写的是实验中成功率为 0%。
- **代码位置：**
  - 投影：`agent_utils/physical_feasibility.py:761`（`apply_non_penetration_projection`），单个面上的版本 `:1030`
  - 仿真：`physical_feasibility.py:1327`（`apply_forward_simulation`）
  - 两阶段组合：`:1548`；单件家具版本 `:1668`
  - 调用处：`manipuland_agents/stateful_manipuland_agent.py:842`
  - 被禁用的工具：`manipuland_agents/tools/manipuland_tools.py:984`
  - 配置：`configurations/manipuland_agent/base_manipuland_agent.yaml:463` 起
- **对我们的启发：** 我们的 Isaac 导出已经带地面和物理场景，但没有跑过 settle。可以按"每个支撑物一组"分批 settle，并记录 settle 前后的位移作为质量指标：位移大说明摆放不稳或有穿插。重建任务里不能删物体，位移过大应标记给人看。

### S8 观察图带标注

- **做法：** `observe_scene` 返回一组图。两个阶段的配置都是 1 张俯视加 4 张侧视（家具的提示词写的是 9 张，对应另一个可选布局 `grid_3x3`）。小物件阶段只渲染当前家具、它上面的物体和少量上下文家具。图上叠加：坐标刻度、物体编号标签、包围盒、表示正面方向的箭头；挡住视线的墙自动隐藏；多个支撑面用不同颜色。
- **代码位置：**
  - 家具观察：`furniture_agents/tools/vision_tools.py:132`
  - 小物件观察（只渲染相关物体）：`manipuland_agents/tools/vision_tools.py:108`
  - 标注绘制：`agent_utils/blender/annotations.py`
  - 视图布局：`configurations/furniture_agent/base_furniture_agent.yaml:25-39`；标注开关 `:56-64`
- **对我们的启发：** 我们设计里的 `view_topdown` 还没做。俯视图上画扫描点云轮廓、资产占地和坐标刻度后，18° 的房间旋转和占地不符一眼可见。

### S9 上下文控制

- **做法：**
  - 轮次裁剪：只保留最近 1 轮的完整内容（含图），更早的轮次去掉图片，并可用模型摘要成文字。
  - 轮内裁剪：同一轮里只保留最近 2 次观察的图片。
  - 重复调用拦截：在最近若干次调用的窗口里，同一个调用重复超过上限就返回错误，提示先刷新场景状态。
- **代码位置：**
  - 轮次裁剪：`agent_utils/turn_trimming_session.py:336`（`TurnTrimmingSession`）
  - 轮内裁剪：`agent_utils/intra_turn_image_filter.py:29`
  - 重复调用拦截：`agent_utils/loop_detector.py:18`
  - 配置：`configurations/manipuland_agent/base_manipuland_agent.yaml:5`（`session_memory`）、`:297`（`loop_detection`）；家具的在同名文件 `:5`、`:294`
- **对我们的启发：** 我们的网关缓存只写不读，每一轮都按全价重发历史。批 1 用了 31 轮、31 万输入 token，物体多了以后图片累积会更明显。"旧图剥离"值得先做。

### S10 资产入库前由 VLM 定朝上面和正面

- **做法：** 每个生成的资产先渲染多视图（顶、底、若干侧面，每张带编号和坐标轴），让 VLM 输出：哪个局部轴朝上、哪张图是正面、材质、质量。然后把网格转到统一姿态（上 = +Z，正面 = +Y，底面在 z=0）。放置阶段只需要 yaw。
- **代码位置：**
  - 分析：`agent_utils/mesh_physics_analyzer.py:149`（`analyze_mesh_orientation_and_material`）
  - 规范化：`agent_utils/mesh_canonicalization.py:20`（`canonicalize_mesh`）
  - 提示词：`prompts/data/mesh_physics/generated.yaml`
- **对我们的启发：** 我们把"哪面朝上"留给组装 agent，在每个物体上都要看一遍资产转台图。如果上游生成阶段统一做一次并写进交接文件，组装阶段的候选就从"6 个朝上面 × 若干 yaw"收缩到只剩 yaw。这要改 ObjectGeneration 的交接内容，属于跨模块的决定。

---

## 3. 不适用于我们的部分

| 做法 | 来源 | 不适用的原因 |
|---|---|---|
| 删物体、或重新生成更小的资产来解决碰撞 | scenesmith | 我们是重建，物体集合由扫描决定 |
| 摆放时加随机扰动（"自然风格"） | scenesmith | 位置应贴照片 |
| 真实度、完整度、提示词符合度等主观打分 | scenesmith | 我们有照片和留出帧，可以直接量 |
| 通行性、门窗净空检查 | scenesmith | 目前没有门窗数据，也不是重建的目标 |
| 贴墙、角落贴合、墙体穿越检查 | LiteReality | 我们只有地面高度和轴对齐矩阵，没有墙体线段；如果以后补墙体，这部分可以再看 |
| 添加扫描里没有的固定物和小物件 | LiteReality | 超出当前范围 |

---

## 4. 没确认的地方

- **LiteReality 神经网格的朝向：** 没找到对 TRELLIS 网格做位姿估计的步骤。程序化资产靠提示词规定统一朝向（`models/object_generation/category_specs.py:22`），神经网格在盒内的朝向怎么保证我没查清。
- **LiteReality 文档与代码可能不同步：** `doc/QC/supporting_relationship.md` 说检查只对地面测量，但 `support_audit.py` 已经有物体对物体的接触核对；文档描述的"支撑链带动"也没找到实现。以代码为准。
- **两套都没有运行过：** 以上都是读码得出的，阈值和流程没有用实际运行核对。scenesmith 依赖 Drake、Blender 服务和 OpenAI 接口，直接跑起来需要另配环境。
