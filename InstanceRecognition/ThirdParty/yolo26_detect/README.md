# YOLO26 Detect 本地复现

这里复现的是 Ultralytics YOLO26 的检测模型 `yolo26s.pt`。输入只有图像，不提供文本提示；模型在预训练的 COCO 80 类闭集词表内输出类别、置信度和二维检测框，不输出 mask。

## 运行

复用相邻 YOLOE-26 复现目录中的 Python 环境：

```bash
bash run_comparison_examples.sh
```

脚本固定使用与已有对比相同的两张图像和参数：

- Ultralytics 示例 `bus.jpg`
- ScanNet `scene0000_00` 第 162 帧
- `imgsz=640`
- `conf=0.25`
- `device=0`

每个输出目录包含：

- `annotated.jpg`：类别、置信度和检测框可视化
- `predictions.json`：运行环境、参数、时延和逐框结构化结果

`RESULTS.md` 记录本机实测结果。总对比图位于上一级目录的 `yoloe26-vs-yolo26-seg-comparison.png`。
