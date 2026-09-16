# WeDetect-Anything 本地复现记录

## 复现范围

本次使用官方 WeDetect-Anything Base 流水线：

1. WeDetect-Base-Uni 从图像生成类别无关候选框和区域 embedding；
2. 区域 embedding 与仓库自带的 4,585 类中文词表特征匹配；
3. 无需用户输入文本，输出中文类别、置信度和检测框。

当前官方 WeDetect-Anything 不输出分割 mask，因此它满足“只给图，自动识别物体类别”的需求，但不满足实例分割需求。

## 固定版本与环境

- 官方仓库提交：`dd302dba0069ace1b05816bafbc3fa1dbd6aa68c`；
- Python 3.8.10；
- PyTorch 2.4.1+cu121；
- torchvision 0.19.1+cu121；
- ONNX 1.17.0；
- ONNX Runtime GPU 1.19.2；
- GPU：NVIDIA GeForce RTX 4090；
- checkpoint：`wedetect_base_uni.pth`，433,889,662 bytes；
- checkpoint SHA-256：`e595578aeeb5f2a36cca383c23279cf7ea72e0bfb7c546dc39420359bf2d2c82`。

官方推荐 PyTorch 2.5.1+cu124；本机为避免重复下载 CUDA 运行库，复用了已有 PyTorch 2.4.1+cu121。ONNX 导出、模型检查及 CUDAExecutionProvider 推理均已通过。

## 一键重跑

```bash
cd /home/wzt/wzt/CVPR_2027/InstanceRecognition/ThirdParty/WeDetect
./run_local_examples.sh
```

如果仅重建最小环境：

```bash
uv venv --python /usr/bin/python3 --system-site-packages .venv
uv pip install --python .venv/bin/python -r requirements-local.txt
```

权重来自公开、无需申请的 Hugging Face 仓库：

```bash
mkdir -p checkpoints
wget -O checkpoints/wedetect_base_uni.pth \
  https://huggingface.co/fushh7/WeDetect/resolve/main/wedetect_base_uni.pth
```

## 推理设置

- Base 模型，640×640 ONNX；
- `score_thr=0.1`；
- `iou_thr=0.7`；
- `keep_top_k=100`；
- `match_thr=0.4`；
- CUDAExecutionProvider，CPU 作为回退。

## 实测结果

| 输入 | 候选框 | 获得标签 | 标签摘要 | 定性观察 |
| --- | ---: | ---: | --- | --- |
| 官方滑雪样例 `examples/1.jpg` | 8 | 8 | 滑雪靴、手套、滑雪板、人、护目镜/墨镜、头盔 | 主体与部件类别均与图像吻合，官方样例复现成功。 |
| Ultralytics `bus.jpg` | 21 | 17 | 公交车、人、眼镜、鞋、牛仔裤、外套、车轮、夹克衫 | 公交车与人物正确；模型同时把服饰和人体部件作为独立对象，框较密集。 |
| ScanNet `scene0000_00` frame 0162 | 10 | 6 | 桌子、椅子、书桌、手机、书、窗帘 | 桌椅、书与窗帘较合理；“手机”对应右侧模糊小物体，缺少人工真值支持。 |

这些结果只证明当前权重和推理链路可运行。两张对比图不是评测集，不能据此推断总体准确率。

## 产物

- `wedetect_anything/onnx_models/wedetect_anything_base.onnx`；
- `wedetect_anything/onnx_models/wedetect_anything_base_meta.json`；
- `wedetect_anything/outputs/official_example_1.jpg`；
- `wedetect_anything/outputs/official_bus.jpg`；
- `wedetect_anything/outputs/scannet_scene0000_00_frame0162.jpg`；
- 每张图对应的 `.log` 文件。

代码和模型采用 GPL-3.0 许可。
