# 场景组装 Agent 交接说明

本文件供新对话窗口接手场景组装任务使用。先读本文件，再按第 6 节顺序核对源码。资产交接文件已升级到 v2（2026-11-11），第 2 节的字段说明以 v2 为准。

---

## 1. 任务目标

把 ObjectGeneration 生成的 52 个物体 USD 资产，按扫描点云提供的位置信息放入 Isaac Sim 场景，并通过 agent 对照原始关键帧渲染逐步修正每个物体的位姿，最终输出可供机器人仿真使用的 `scene.usdc`。

这是一个 **harness agent**：它驱动 Isaac Sim（通过 omniverse-mcp），把原始关键帧图和仿真渲染图并排呈现给视觉模型，模型调用工具逐物体修正位姿，直到对比图满意为止。

---

## 2. 输入：两份交接文件

场景组装同时读两份交接文件，按 `object_id` 关联。资产交接文件只放物体生成阶段产生的信息，不再转抄场景交接文件的字段和文件。

| | 场景交接（handoff/v1） | 资产交接（assembly_handoff/v2） |
|---|---|---|
| 产出方 | InstanceRecognition | ObjectGeneration |
| 路径 | `/home/seer/wzt/Reconstruction/InstanceRecognition/artifacts/handoff/scene0000_00/gt/` | `/home/seer/wzt/Reconstruction/ObjectGeneration/output/scene0000_00/gt/assembly_handoff/` |
| 物体 | 67（含 15 个结构物体） | 67 条记录：52 个 success、15 个 skipped |
| 内容 | label、点云、centroid / bounds、关键帧 RGB / mask、相机内外参、floor_z | USD 资产、生成状态、生成帧、SAM3D 位姿、资产尺寸与可信度、物理参数 |

### 2.1 场景交接 handoff/v1（只读，字段以其 manifest 为准）

- `manifest.json` 的 `camera`：内参 `fx=fy=1170.188, cx=647.75, cy=483.75`，图像 1296×968，无畸变；外参约定写在 `camera.extrinsics`。
- `objects.json[].key_frames[]`：`seq`、`role`、`rgb`、`mask`、`camera_to_world`（OpenCV，+X 右、+Y 下、+Z 前，4×4 行主序）。旧字段 `camera_to_world_gt` 是同一个矩阵（逐元素相等），上游注释说保留它只为兼容。
- `objects.json[]`：`label`、`cloud`、`centroid`、`bounds`、`point_count` 等，坐标系 `scannet_scan`，Z-up，米。
- `manifest.json` 的 `floor_z = 0.0544`。

### 2.2 资产交接 assembly_handoff/v2

```
assembly_handoff/
├── manifest.json            # 版本、上游哈希、约定说明
├── objects.json             # 每个生成记录一条（含未生成的）
└── usds/<oid>/
    ├── physics.usdc         # 下游加载这个：视觉 + 刚体 + 碰撞
    ├── physics.physics.json # 实际写入的质量、质心、惯量、碰撞面数
    ├── visual.usdc          # 仅视觉
    └── visual.conversion.json  # dimensions_m、缩放；11-11 之后新建的资产还会记录 GLB→USD 变换（glb_to_usd），现有 52 个没有，偏移量见 objects.json 的 pose.usd_offset_m
```

**manifest.json 要点：**

```json
{
  "schema_version": "assembly_handoff/v2",
  "source_handoff": {
    "path": ".../InstanceRecognition/artifacts/handoff/scene0000_00/gt",
    "manifest_sha256": "54684a5b…", "objects_sha256": "e20cf71f…",
    "matches_generation_input": false,
    "generation_input": {"manifest_sha256": "9c53cbcf…", "objects_sha256": "8c838bb4…"},
    "generation_inputs_unchanged": "52/52 assets' source RGB and mask still match handoff/v1 byte for byte"
  },
  "source_generation": {"path": "...", "generator": {"config_sha256": "…", "seed": 42}, "manifest_sha256": "…"},
  "join": "objects.json[].object_id == handoff/v1 objects.json[].object_id",
  "asset_frame": "USD Z-up, metres; mesh XY bounding box centred on the origin, lowest point at z=0",
  "conventions": {"pose": {…}, "size": {…}}
}
```

`source_handoff` 的哈希是交接文件构建时读到的上游文件。上游在生成之后又更新过一次（11-10 加了内参、外参约定和 floor_z），所以 `matches_generation_input=false`。但每个资产的生成输入（RGB 和 mask）都逐字节核对过，52/52 仍与当前上游一致，生成结果有效。

**objects.json 单条（success）：**

```json
{
  "object_id": "obj_0002",
  "status": "success",
  "generation": {
    "view": "primary", "seq": 454,
    "rgb": "keyframes/obj_0002/0454_rgb.jpg",
    "mask": "keyframes/obj_0002/0454_mask.png",
    "rgb_sha256": "ea19…", "mask_sha256": "c41c…",
    "inputs_match_handoff": true
  },
  "usd": "usds/obj_0002/physics.usdc",
  "size": {
    "dimensions_m": [2.1999, 0.5004, 1.087],
    "height_m": 1.087, "scale_factor": 2.1991, "source": "scan height",
    "confidence": "high", "flags": [],
    "axis_ratio": {"height": 0.999, "footprint_long": 0.894, "footprint_short": 1.019},
    "candidates_m": {"scan_height": 1.087, "footprint": 1.216, "pca_secondary": 1.104}
  },
  "pose": {
    "rotation": [[…3x3…]], "translation": [0.386, 0.250, 2.277], "scale": 1.485,
    "usd_offset_m": [-0.0003, -0.0013, -0.5446],
    "raw": {"rotation_wxyz": […], "translation": […], "scale": 3.266}
  },
  "physics": {
    "body_type": "dynamic", "mass_kg": 45, "roughness": 0.4, "metallic": 0.0,
    "collision_approximation": "convexDecomposition", "confidence": "model",
    "estimated_body_type": "dynamic", "demotion_reason": null, "reasoning": "…"
  }
}
```

未生成的记录只有 `object_id`、`status`（`skipped` / `failed`）、`reason`，以及 `generation.view`。scene0000_00 的 15 条 skipped 都是结构物体（floor / wall / ceiling / doorframe / window）。

`generation.rgb` / `mask` 是**场景交接目录下的相对路径**，资产交接里不再复制图片。

### 2.3 字段说明

**generation**：生成用的是哪一帧。目前全部是 `primary` 视角，`seq` 对应 handoff/v1 里该物体 `key_frames[]` 中 `seq` 相同的那一帧，从那一帧取外参。`inputs_match_handoff` 表示生成时用的 RGB / mask 与当前上游文件的哈希一致。

**size**：
- `dimensions_m`：资产在自身坐标系的包围盒，X、Y、Z（Z 是高）。从 `visual.conversion.json` 读，USD 实测一致（52/52，误差 < 1 mm）。
- `confidence`：
  - `high`：用扫描 Z 高度定尺度，资产三个轴（高、长边、短边）都与扫描一致（比值在 0.8–1.25）。
  - `medium`：没有矛盾，但有未核实的地方，见 `flags`。例如扫描不全（资产比扫描大）、点数 < 100、扫描底部离地 0.1–0.3 m（可能下半截没扫到，尺寸偏小）、某个轴比扫描小 0.5–0.8 倍。
  - `low`：Z 高度不可用、退回 footprint 定尺度，或某个轴不到扫描范围的一半。
  - scene0000_00 统计：high 19 / medium 27 / low 6。low 的 6 个是 kitchen counter obj_0003（SAM3D 只生成了薄板）、guitar case obj_0066（资产平躺）、scale obj_0011、tv obj_0015、guitar obj_0047、obj_0060。
- `axis_ratio`：资产 / 扫描的尺寸比。footprint 两个轴都按长短排序，与朝向无关。扫描只覆盖看到的部分，所以比值 > 1 是常见的。
- `source` / `candidates_m`：尺度来源和三个候选高度，沿用 `sizes.json`。

**pose**：SAM3D 对生成帧的单图位姿估计，已经换算到 **USD 资产坐标系**（即 USD 文件里存的点，已完成 Z-up 旋转、按 scale_factor 缩放、XY 居中、底部落到 z=0）：

```python
p_cam   = pose["scale"] * R @ p_usd + t      # R = pose["rotation"], t = pose["translation"]
p_scene = C2W @ [p_cam, 1]                    # C2W = handoff/v1 生成帧的 camera_to_world
```

- `p_cam` 在 OpenCV 相机坐标系（+X 右、+Y 下、+Z 前），与 handoff/v1 外参约定一致，可直接用内参投影到生成帧图像上。
- **rotation 可用；scale 与 translation 不是米制**（SAM3D 深度有未知尺度）。在 scene0000_00 上，扫描深度 / SAM3D 深度的中位比值是 0.96，IQR 0.86–1.15，只能作为位置的粗略参考，放置位置仍以扫描为准。
- 世界坐标系下资产的朝向：`R_scene = C2W[:3,:3] @ R`。`R_scene[:,2]` 是资产 +Z（向上）在世界里的方向。
- 换算推导：SAM3D 给出 q(wxyz)、t、s，含义是 `x_cam_p3d = s · R(q)ᵀ x_obj + t`（PyTorch3D 相机，+X 左、+Y 上）。USD 点 `p_usd = scale_factor · x_obj − usd_offset_m`。所以 `rotation = diag(−1,−1,1) R(q)ᵀ`，`scale = s / scale_factor`，`translation = diag(−1,−1,1)(t + scale · R(q)ᵀ usd_offset_m)`。`raw` 保留原始值。
- 验证：用这个公式把 USD 网格投影到生成帧，轮廓与 handoff/v1 mask 的 IoU 中位 0.64，42/52 > 0.5。与从原始 GLB 直接计算的结果相对误差 1.2e-6。
- **可靠性**：资产 +Z 与世界 Z 偏差中位 7.9°，39/52 < 15°。6 个物体整体倒置约 170–180°（curtain obj_0006、refrigerator、laundry basket、microwave、obj_0061、mirror）。pillow 71°、guitar case 118°。偏航角未经核实：直立的 44 个里有 15 个与扫描配准最优相差 180°，仅靠几何分不出正反，要看图判断。

**physics**：`body_type` 是实际写进 USD 的类型（dynamic 37 / kinematic 3 / static 12）。`estimated_body_type` 是模型估计的类型。两者不同时，`demotion_reason` 说明原因（例如非水密网格降为 static，本场景 0 个）。`mass_kg` 在 static 时为 null。`reasoning` 是模型给出的理由，带 `[manual]` 前缀的是人工修正过的。

### 2.4 USD 资产

- defaultPrim `/ObjN`（obj_0002 → `/Obj2`），Z-up，`metersPerUnit=1`。
- `/ObjN/Mesh`：视觉网格，4 万面，顶点色（displayColor 存线性值），碰撞已关闭。
- `/ObjN/CollisionMesh`：不可见的碰撞体，convexHull 或 convexDecomposition。
- 根节点：dynamic 带 RigidBodyAPI + MassAPI（含质心和惯量）；kinematic 带 RigidBodyAPI（kinematicEnabled）+ 质量；static 不带。

### 2.5 从旧版（v1）迁移

| v1 字段 | v2 去向 |
|---|---|
| `label`、`cloud`、`centroid`、`bounds`、`point_count`、`key_frames` | 从 handoff/v1 按 `object_id` 读取 |
| `clouds/`、`keyframes/` 目录 | 已删除，用 handoff/v1 下同名的相对路径 |
| `key_frames[].camera_to_world_gt` | handoff/v1 的 `camera_to_world`（同一矩阵） |
| `scale.{scale_factor,height_m,source}` | `size.{scale_factor,height_m,source}` |
| `physics.kinematic`、`physics.rigid_body_enabled` | 由 `physics.body_type` 推出 |
| `physics.kinematic_reason` | `physics.demotion_reason` |

---

## 3. 初始放置

USD 资产在自身坐标系里 XY 包围盒居中、底面在 z=0。从扫描得到放置参考（来自 handoff/v1）：

```python
lo, hi = np.array(bounds["minimum"]), np.array(bounds["maximum"])
translate = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]
```

- 用包围盒中心而不是 `centroid`：centroid 是点的均值，有 8 个物体与包围盒中心的 XY 差 0.15–0.31 m。
- `lo[2]` 是扫描到的最低点。下半截没扫到的物体会悬空（如 nightstand obj_0050、stool obj_0031，这两个的 `size.flags` 里有说明），贴地或贴支撑面要靠 agent 判断。
- 初始朝向可以取 SAM3D 位姿：`R_scene = C2W @ R`，取其偏航角作 `rz`；倒置的物体要人工翻转。这只是候选，不是结论。

## 4. omniverse-mcp

**已安装，无需额外配置。**

- Bridge 扩展已链接到 Isaac Sim 用户扩展目录（见 `/home/seer/wzt/robot_sim/docs/OMNIVERSE_MCP.md`）
- 启动 Isaac Sim：`bash /home/seer/wzt/robot_sim/scripts/launch_isaacsim_mcp.sh`
- MCP 服务器脚本：`/home/seer/wzt/robot_sim/scripts/start_omniverse_mcp.sh`
- 连接地址：`127.0.0.1:8211`
- Python 环境：`/home/seer/wzt/robot_sim/.venv`
- 验证：`.venv/bin/python scripts/check_omniverse_mcp.py`（在 robot_sim 目录下）

**关键工具（共 41 个，其中与本任务直接相关的）：**

| 工具 | 用途 |
|---|---|
| `execute_script` | 加载物体 USD 引用到场景（add_reference + xformOp） |
| `set_prim_transform(prim_path, position, rotation, scale)` | 调整位姿；rotation 3元素=XYZ欧拉角度，4元素=[w,x,y,z]四元数 |
| `set_camera(position, target)` | 设置视口相机；position和target都是[x,y,z] |
| `capture_viewport(width, height, include_image)` | 渲染截图，返回图像 |
| `inspect_prim(prim_path)` | 从14个角度拍物体轨道截图 |
| `get_prim_bounds(prim_path)` | 获取当前包围盒（用于验证落地） |
| `sim_control(action)` | play/pause/stop/step 控制物理仿真 |
| `save_scene(file_path)` | 写最终 scene.usdc |
| `new_scene` | 创建空场景 |
| `get_scene_tree` | 检查当前场景层级 |

**相机内参（scene0000_00，未降采样；现在也在 handoff/v1 的 manifest.camera 里）：**

| 参数 | 值 |
|---|---|
| 图像分辨率 | 1296 × 968 px |
| fx = fy | 1170.188 |
| cx | 647.75 |
| cy | 483.75 |
| 来源 | `/home/seer/wzt/data/scannet_v2/scene0000_00/scene0000_00.txt` |

关键帧图就是原始 ScanNet 分辨率，未降采样。

Isaac Sim 渲染时对应的水平 FOV：`2 * atan(1296 / 2 / 1170.188) ≈ 57.4°`。

内参以 handoff/v1 的 `manifest.json` → `camera` 为准（每个场景只有一套）。

**从 camera_to_world 提取 set_camera 参数：**
```python
import numpy as np
M = np.array(camera_to_world)      # 4x4，handoff/v1 key_frames[]
position = M[:3, 3].tolist()       # 相机位置
forward = M[:3, 2]                 # 相机前向（+Z列）；ScanNet 用 OpenCV 约定，+Z 朝前
target = (M[:3, 3] + forward).tolist()  # look-at 目标点
# → set_camera(position=position, target=target)
# 同时设置 FOV ≈ 57.4° 保证与原图一致
```

注意：ScanNet `camera_to_world` 使用 OpenCV 约定（相机 +Z 朝前，+Y 朝下），col2 = 世界空间前向，不取反。
验证：scene0000_00 seq=454，`cos(col2, 相机→obj_0002) = 0.949`，几乎正对桌子，符合预期。

---

## 5. Agent 设计

### 工作流

```
load_scene(objects_json)            # 一次性加载全部 52 个物体
    ↓
按 Z 高度分窗口（~15 物体/窗口）
    ↓
对每个窗口：
    render_from_keyframe(oid)       # sim渲染 vs 原始关键帧 并排
    模型判断 → 调用 adjust_object  # 调整 dx/dy/dz/rz
    accept_object / flag_issue
    ↓
verify_physics(dynamic_objects)     # play 120步，检查穿地/飘起
    ↓
commit_scene → scene.usdc
```

### 建议工具集

```
load_scene(objects_json_path)
  → execute_script: 逐个 add_reference，设 xformOp:translate

render_from_keyframe(object_id, kf_index=0)
  → set_camera(pos, target) + capture_viewport
  → 返回 [原始 keyframe 图, sim 渲染图] 并排

adjust_object(object_id, dx=0, dy=0, dz=0, rz_deg=0)
  → set_prim_transform + 立即重渲染返回新对比图

inspect_object_3d(object_id)
  → inspect_prim → 14 角度截图

verify_physics(object_ids)
  → sim_control(play) → 120步 → get_prim_bounds 检查位移
  → sim_control(stop) + 恢复初始位置

accept_object(object_id, notes="")
flag_issue(object_id, reason)
commit_scene(output_path)
  → save_scene
```

### 状态文件：`assembly_state.json`

```json
[{
  "object_id": "obj_0002",
  "baseline_transform": {"translate": [4.49, 6.94, 0.07], "rotation": [0,0,0]},
  "current_transform": {"translate": [4.49, 6.94, 0.07], "rotation": [0,0,15]},
  "adjustment_history": [
    {"round": 1, "delta": {"rz_deg": 15}, "reason": "front face toward primary keyframe camera"}
  ],
  "status": "accepted"
}]
```

中断恢复：读 assembly_state.json，`status == "accepted"` 的物体跳过重渲染。

---

## 6. 建议开发顺序

1. **`mcp_client.py`**：封装 MCP stdio 连接，暴露 `call(tool_name, args)` 和图像返回处理
2. **`load_scene.py`**：读两份交接文件按 object_id 关联 → execute_script 加载 status=success 的 USD，输出初始 assembly_state.json
3. **`render.py`**：`render_from_keyframe` 和 `adjust_object` 的实现；验证 set_camera 从 camera_to_world 正确定位
4. **`agent.py`**：harness 主循环，窗口策略，工具 schema，模型调用
5. **`verify_physics.py`**：物理落地验证
6. **`commit.py`**：最终 save_scene

先跑通单个物体（obj_0002 table）的完整闭环再展开到全部 52 个。

---

## 7. 相关路径速查

| 内容 | 路径 |
|---|---|
| 资产交接（v2） | `/home/seer/wzt/Reconstruction/ObjectGeneration/output/scene0000_00/gt/assembly_handoff/` |
| 场景交接（handoff/v1） | `/home/seer/wzt/Reconstruction/InstanceRecognition/artifacts/handoff/scene0000_00/gt/` |
| 资产交接构建脚本 | `/home/seer/wzt/Reconstruction/ObjectGeneration/scripts/tools/build_assembly_handoff.py` |
| omniverse-mcp bridge | `/home/seer/wzt/robot_sim/third_party/omniverse-mcp/` |
| MCP Python 环境 | `/home/seer/wzt/robot_sim/.venv` |
| MCP 启动脚本 | `/home/seer/wzt/robot_sim/scripts/start_omniverse_mcp.sh` |
| Isaac Sim 启动 | `/home/seer/wzt/robot_sim/scripts/launch_isaacsim_mcp.sh` |
| omniverse-mcp 文档 | `/home/seer/wzt/robot_sim/docs/OMNIVERSE_MCP.md` |
| 既有 MCP 客户端参考 | `/home/seer/wzt/Blender/scripts/mcp_client.py` |
| set_prim_transform handler | `/home/seer/wzt/robot_sim/third_party/omniverse-mcp/extension/isaacsim.mcp.bridge-1.0.0/isaacsim/mcp/bridge/handlers/_utils.py:172` |
