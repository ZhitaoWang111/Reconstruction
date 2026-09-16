# YOLOE-26 + SAM3 / SAM3.1

这条两阶段流水线只接收图像：

1. YOLOE-26 Prompt-Free 给出类别、置信度和检测框；
2. 忽略 YOLOE 自带的 mask；
3. 分别将每个 YOLOE 框作为正几何提示交给 SAM3 或 SAM3.1；SAM3.1 对不返回候选的困难框使用框中心正点提示；
4. 当 SAM 返回多个候选时，选择与 YOLOE 输入框 IoU 最大的 mask。

SAM3 使用官方图像模型。SAM3.1 使用官方 Multiplex 视频模型，并把静态图像作为单帧会话处理。因此 SAM3.1 结果只是单帧适配测试，不代表其主要的视频多物体跟踪优势。

SAM3.1 官方当前 `main` 分支的统一 session 包装器会向 Multiplex `init_state` 多传一个不支持的参数。本地脚本只在调用侧兼容该签名，没有修改 `../SAM3` 中的官方代码；构建时还关闭了实数 RoPE，以匹配下载权重中的复数 `freqs_cis` 参数。

## 运行

```bash
bash run_comparison_examples.sh
```

脚本使用：

- 官方仓库：`../SAM3`
- Python 环境：`/home/wzt/.conda/envs/scene`
- SAM3 权重：`../../../weights/sam3.pt`
- SAM3.1 权重：`../../../weights/sam3.1_multiplex.pt`
- 与已有方法完全相同的 `bus.jpg` 和 ScanNet frame 0162
- YOLOE-26s Prompt-Free 已保存的逐实例框和标签

输出目录保存 `annotated.jpg`、`predictions.json`、`instance_id_map.png` 和逐实例二值 mask。
其中 `predictions.json` 的 `sam_prompt_type` 会标记每个实例使用 `positive_box` 还是 `center_point_fallback`。
