# YOLO26-seg 与 YOLOE-26 Prompt-Free 同图实测

## 结论

`yolo26s-seg.pt` 已在本机 RTX 4090 上成功运行。它只接收图像，并为每个预测实例同时返回 COCO 类别、检测框和 mask。

在本次两张图的定性测试中，YOLO26-seg 的标签明显更保守：没有出现 YOLOE-26 Prompt-Free 的 `boardwalk`、`lotus`、`homework`、`remove` 等大词表误匹配，但它只能从 COCO 的 80 个固定类别中选择，不能识别 `desk`、`cabinet`、`curtain` 等不在其闭集词表中的室内类别。

## 公平比较设置

- 模型规模：两者均为 Small；
- 输入：完全相同的两个原始文件；
- 参数：`imgsz=640`、`conf=0.25`、默认 NMS 路径；
- 环境：Ultralytics 8.4.149、PyTorch 2.4.1、RTX 4090；
- 输出：均启用 `retina_masks=True`，保存原图尺寸的逐实例 mask。

## 实测结果

| 输入 | 模型 | 实例数 | 唯一类别 | 单次模型推理时间 | 定性观察 |
| --- | --- | ---: | --- | ---: | --- |
| 官方 `bus.jpg` | YOLO26s-seg | 5 | bus、person | 3.627 ms | 识别 1 辆公交车和 4 个人；标签保守且与可见主体一致，但没有检出左侧交通标志。 |
| 官方 `bus.jpg` | YOLOE-26s Prompt-Free | 11 | boardwalk、bus、glasses、helmet、lotus、person、stop sign | 4.979 ms | 输出了更多细分类别，但出现 boardwalk、lotus、helmet 等明显误标签；图中圆形交通标志被称为 stop sign 也不可靠。 |
| ScanNet frame 0162 | YOLO26s-seg | 4 | book、chair、mouse、remote | 3.554 ms | chair 明确合理；其余小物体与图像区域基本对应，但受低分辨率和模糊影响，不应视为人工真值。 |
| ScanNet frame 0162 | YOLOE-26s Prompt-Free | 14 | bathroom mirror、bureau、chair、cornice、frame、homework、jack、laundry room、office desk、remote、remove、speaker | 4.181 ms | chair、office desk 等部分合理，但出现 homework、remove、laundry room 等明显错误语义，并生成较多大面积错误 mask。 |

表中的推理时间来自各自现有 `predictions.json` 的单次 `result.speed["inference"]`，没有预热和重复统计，不能作为严格速度排名。

## YOLO26-seg 详细预测

### 官方 `bus.jpg`

| 类别 | 置信度 |
| --- | ---: |
| bus | 0.920 |
| person | 0.912 |
| person | 0.900 |
| person | 0.867 |
| person | 0.817 |

### ScanNet frame 0162

| 类别 | 置信度 |
| --- | ---: |
| chair | 0.854 |
| book | 0.451 |
| remote | 0.372 |
| mouse | 0.339 |

## 校验值与产物

```text
3da1d83e31caec96f9300eb4064f4f62882c133c7c264d63dfe61a7c197837a4  weights/yolo26s-seg.pt
```

- `outputs/official_bus_s/`：公交车样例；
- `outputs/scannet_scene0000_00_frame0162_s/`：ScanNet RGB 帧样例。

每个目录均包含 `annotated.jpg`、`predictions.json`、`instance_id_map.png` 和逐实例 `masks/*.png`。
