# YOLOE-26 + SAM3 / SAM3.1 复现结果

本次只把 YOLOE-26 Prompt-Free 保存的类别、置信度和检测框交给 SAM，明确不读取 YOLOE 自带 mask。所有结构化输出都通过 `verify_outputs.py` 校验：检测数与 mask 数一致，逐实例 mask 非空、二值且与输入图同分辨率。

| 样例 | 后端 | YOLOE 检测 / 标签 | SAM 提示 | 本机计时 |
| --- | --- | --- | --- | --- |
| `bus.jpg` | SAM3 image | 11 / 7 | 11 正框 | 加载 5.252 s；编码 0.155 s；提示 1.574 s |
| `bus.jpg` | SAM3.1 Multiplex | 11 / 7 | 8 正框 + 3 中心点回退 | 加载 8.863 s；会话与提示 1.435 s |
| ScanNet frame 0162 | SAM3 image | 14 / 12 | 14 正框 | 加载 5.220 s；编码 0.157 s；提示 1.657 s |
| ScanNet frame 0162 | SAM3.1 Multiplex | 14 / 12 | 11 正框 + 3 中心点回退 | 加载 8.806 s；会话与提示 1.755 s |

这些是两个样例上的功能与定性测试，不是准确率或速度基准。SAM 只改善实例轮廓，不会修正 YOLOE 的类别错误；例如原有的 `lotus`、`boardwalk` 等标签仍会保留。SAM3.1 在这里按单帧使用，因此也没有测试视频跟踪能力。
