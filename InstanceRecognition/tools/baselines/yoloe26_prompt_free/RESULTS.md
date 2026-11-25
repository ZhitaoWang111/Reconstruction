# 首轮复现结果

## 结论

YOLOE-26 Prompt-Free 已在本机 RTX 4090 上成功运行。它确实只接收图像，并为每个预测实例同时返回类别、框和 mask；但是当前样例也显示，大词表自动匹配存在明显误分类，因此这里只能确认功能链路复现成功，不能据此声称语义结果可靠。

## 运行记录

| 输入 | 权重 | 实例数 | 唯一类别数 | 单次模型推理时间 | 质量观察 |
| --- | --- | ---: | ---: | ---: | --- |
| Ultralytics `bus.jpg` | `yoloe-26s-seg-pf.pt` | 11 | 7 | 5.0 ms | 正确识别 bus、person、stop sign；出现 boardwalk、lotus、helmet 等错误标签。 |
| Ultralytics `bus.jpg` | `yoloe-26n-seg-pf.pt` | 12 | 8 | 6.8 ms | 正确识别人和 stop sign；公交车被称为 recreational vehicle，并出现 city square、cowbell、water gun 等错误标签。 |
| ScanNet `scene0000_00` frame 0162 | `yoloe-26s-seg-pf.pt` | 14 | 12 | 4.2 ms | chair、office desk 等部分合理，但也出现 laundry room、homework、remove 等明显错误语义。 |

时间来自单次 `result.speed["inference"]`，未经过预热和重复统计，不能作为 Nano 与 Small 的速度排名。

## 输入与权重校验值

```text
c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63  inputs/official_bus.jpg
2829871406fcdb7f36b44af4e534960402626b7c27b3f8b9b42dee6a05fbff6e  weights/yoloe-26n-seg-pf.pt
9f0cefea64c48103a917dbc8ea4baf581aaf6ffa286675861dafd6fd04826f50  weights/yoloe-26s-seg-pf.pt
```

## 产物位置

- `outputs/official_bus_s/`：Small 权重的官方样例结果；
- `outputs/official_bus_n/`：Nano 权重的官方样例结果；
- `outputs/scannet_scene0000_00_frame0162_s/`：Small 权重的 ScanNet RGB 帧结果。

每个目录均包含 `annotated.jpg`、`predictions.json`、`instance_id_map.png` 和逐实例 `masks/*.png`。
