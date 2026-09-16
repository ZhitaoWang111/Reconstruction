# YOLO26-seg 本地复现

该目录验证标准 YOLO26 实例分割模型的输入输出契约：

- 输入：仅一张图像，不提供文本或视觉提示；
- 输出：COCO 闭集类别、检测框、逐实例二值 mask 和叠加可视化；
- 默认模型：`yolo26s-seg.pt`，与此前 YOLOE-26 Prompt-Free 测试同为 Small 规模。

## 公平比较设置

两种模型使用相同输入、GPU 和推理参数：

- Ultralytics 官方 `bus.jpg`；
- ScanNet `scene0000_00` 的第 162 帧；
- `imgsz=640`、`conf=0.25`、`device=0`；
- 均保留原图分辨率的实例 mask。

## 运行

复用相邻 YOLOE-26 目录中已经锁定依赖的虚拟环境，避免重复占用磁盘：

```bash
cd /home/wzt/wzt/CVPR_2027/InstanceRecognition/ThirdParty/yolo26_seg
chmod +x run_comparison_examples.sh
./run_comparison_examples.sh
```

运行自己的图片：

```bash
../yoloe26_prompt_free/.venv/bin/python infer_yolo26_seg.py \
  --source /absolute/path/to/image.jpg \
  --output-dir outputs/my_image
```

每个输出目录包含 `annotated.jpg`、`predictions.json`、
`instance_id_map.png` 和逐实例 `masks/*.png`。

本机同图测试结果和与 YOLOE-26 Prompt-Free 的对比见 [RESULTS.md](RESULTS.md)。

代码与权重的使用需遵守 Ultralytics 的 AGPL-3.0 或 Enterprise 许可条款。
