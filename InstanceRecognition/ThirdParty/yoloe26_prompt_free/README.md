# YOLOE-26 Prompt-Free 本地复现

这个目录验证以下输入输出契约：

- 输入：仅一张图像，不提供类别文本、参考框或点提示。
- 输出：自动识别的类别、检测框、逐实例二值 mask 和叠加可视化。
- 默认模型：`yoloe-26s-seg-pf.pt`，即 YOLOE-26 Small 的 prompt-free 专用权重。

## 当前环境

- Python 3.8.10
- PyTorch 2.4.1 + CUDA 12.1
- Ultralytics 8.4.149
- GPU：NVIDIA GeForce RTX 4090

## 运行官方样例

```bash
cd /home/wzt/wzt/CVPR_2027/InstanceRecognition/ThirdParty/yoloe26_prompt_free
./run_official_example.sh
```

## 运行自己的图像

```bash
.venv/bin/python infer_prompt_free.py \
  --source /absolute/path/to/image.jpg \
  --output-dir outputs/my_image
```

首次运行会把官方权重下载到 `weights/`。输出目录包含：

- `annotated.jpg`：类别、框和 mask 的叠加可视化；
- `predictions.json`：每个实例的类别、置信度、框坐标、mask 像素数和文件名；
- `masks/*.png`：与输入图像同分辨率的逐实例二值 mask；
- `instance_id_map.png`：16 位实例 ID 图，背景为 0，第 N 个实例的像素值为 N。

可以单独复核已有产物：

```bash
.venv/bin/python verify_outputs.py outputs/official_bus_s
```

首轮实测与质量边界见 [RESULTS.md](RESULTS.md)。

## 重新创建环境

```bash
uv venv --python /usr/bin/python3 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

代码和权重的使用需遵守 Ultralytics 的 AGPL-3.0 或 Enterprise 许可条款。
