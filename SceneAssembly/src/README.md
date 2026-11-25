# SceneAssembly/src

| 目录 | 内容 | 运行环境 |
|---|---|---|
| `legacy/` | 测试实验（2026-11-11，批 1，r1 / r2）用的第一版 harness，**冻结，不再改**。自带一份当时的 `handoff.py`、`usdmesh.py`、`usdcam.py`，单独可跑：`.venv/bin/python src/legacy/run_experiment.py`、`src/legacy/export_usd.py` | `.venv`；`isaac_render.py` 用 Isaac |
| `inputs/` | M0 数据接入：物体与资产交接 `handoff.py`、墙体交接 `structure.py`、USD 网格读取 `usdmesh.py` | `.venv` |
| `isaac/` | M1 Isaac 常驻服务：`server.py`（服务端）、`client.py`（harness 调用接口 `IsaacService`）、`camera.py`（相机换算，两边共用） | `server.py` 用 `~/isaacsim/python.sh`，其余 `.venv` |

新代码以 `src/` 为导入根：`from inputs import handoff, structure`、`from isaac.client import IsaacService`。
后续模块各开一个包（M2 位姿与状态、M4 验证、M5 观测、M6 批次、M7 agent 工具……），不再往 `legacy/` 里加东西。
`legacy/` 里的模块用扁平导入（`import handoff`），要把 `src/legacy` 放到路径最前面。
| `initial/` | M3 初始位姿：`sam3d.py`（SAM3D 旋转 → 朝上轴 / yaw / 翻转）、`icp.py`（有界配准，从 legacy 复制）、`plan.py`（`InitialPoses`：计算、缓存、写入状态、文字卡） | `.venv` |
| `verify/` | M4 验证与拦截：`rules.py`（阈值）、`measure.py`（几何测量）、`check.py`（前后比较、撤回、物理判定） | `.venv` + Isaac |
| `eval/` | M9 评估：`frames.py`（留出帧）、`score.py`（轮廓 IoU）、`baselines.py`（B0/B1/M3）、`run.py`（汇总、JSON + HTML 报告） | `.venv` |
| `run/` | M8 编排：`main.py`（11 批连续、断点恢复、预算、全局物理）、`export.py`（导出 scene.usda，从 legacy 复制改写） | `.venv` + Isaac + 网关 |
| `agent/` | M7 agent：`llm.py`（网关 Responses 调用，gpt-6-astra）、`tools.py`（10 个 strict 工具、提示词）、`loop.py`（工具实现、开局消息、对话循环） | `.venv` + Isaac + 网关 |
| `batch/` | M6 批次：`plan.py`（切批）、`session.py`（开批、顺序、拉入、视角限制、批末检查与重开、断点） | `.venv` + Isaac |
| `observe/` | M5 观测：`boxes.py`（扫描点投影出框）、`render.py`（四面图离线渲染，从 legacy 复制）、`observer.py`（文字卡、四面图、带框关键帧、拼接图、开局材料） | `.venv` + Isaac |
| `state/` | M2 位姿与状态：`pose.py`（Pose、Support）、`resolve.py`（位姿+支撑→4×4，落 z、贴墙、打回）、`store.py`（SceneState：改动、带动、撤回、版本快照）、`assets.py` | `.venv` |
