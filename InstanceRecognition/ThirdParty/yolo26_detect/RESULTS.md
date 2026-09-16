# YOLO26 Detect 同图实测

## 结论

`yolo26s.pt` 已在本机 RTX 4090 上成功运行。模型只接收图像，不需要文本提示；它从 COCO 80 类闭集词表中输出类别、置信度和二维检测框，不输出 mask。

## 公平比较设置

- 模型规模：YOLO26 Detect 与 YOLO26-seg 均为 Small；
- 输入：完全相同的官方 `bus.jpg` 和 ScanNet `scene0000_00` 第 162 帧；
- 参数：`imgsz=640`、`conf=0.25`、Ultralytics 默认预测路径；
- 环境：Ultralytics 8.4.149、PyTorch 2.4.1、RTX 4090。

## 实测结果

| 输入 | 检测框数 | 唯一类别 | 单次模型推理时间 | 定性观察 |
| --- | ---: | --- | ---: | --- |
| 官方 `bus.jpg` | 5 | bus、person | 3.001 ms | 检出 1 辆公交车和 4 个人；与同尺寸 YOLO26-seg 的实例数和类别相同。 |
| ScanNet frame 0162 | 2 | book、chair | 3.031 ms | chair 与 book 的位置合理；没有检出 YOLO26-seg 在同图给出的 mouse 和 remote。 |

时间来自单次 `result.speed["inference"]`，没有预热或重复统计，不能作为严格速度排名。

## 详细预测

### 官方 `bus.jpg`

| 类别 | 置信度 |
| --- | ---: |
| person | 0.923 |
| bus | 0.923 |
| person | 0.898 |
| person | 0.846 |
| person | 0.833 |

### ScanNet frame 0162

| 类别 | 置信度 |
| --- | ---: |
| chair | 0.726 |
| book | 0.400 |

## 校验值与产物

```text
646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b  weights/yolo26s.pt
```

- `outputs/official_bus_s/`：公交车样例；
- `outputs/scannet_scene0000_00_frame0162_s/`：ScanNet RGB 帧样例。

每个输出目录均包含 `annotated.jpg` 和 `predictions.json`。结构化结果记录逐框类别、置信度与 `box_xyxy`，不包含 mask。
