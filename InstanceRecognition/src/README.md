# 源码目录（2026-09-21 整理）

| 目录 | 职责 |
| --- | --- |
| `data/` | ScanNet `.sens` 读取、抽帧 |
| `inference/` | GPT 类别、SAM/YOLO、常驻 worker、批处理与在线流水线 |
| `geometry/` | mask 反投影、原有几何关联基线及其运行入口 |
| `visualization/` | 离线页面构建与发布 |
| `experiments/` | 历史鞋配对修订、批次导入等一次性实验 |
| `harness/` | 窗口式模型审阅、取证、版本化归属修改、点云与回放 |
| `config.py` | 共用配置，继续位于包根 |

根目录同名 Python 文件是兼容入口，实际实现已搬到子包。旧的 `python -m src.batch_infer`、服务 worker 命令、`from src.pipeline import ...` 仍有效。兼容入口导入同一模块对象，避免 patch/全局状态分裂。新代码直接使用子包路径。

从 `InstanceRecognition/` 运行命令；模型推理环境与纯几何环境保持分离。已有结果不随目录整理迁移。
