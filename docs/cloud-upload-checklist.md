# 数据与权重：待上传私有网盘

盘点日期：2026-09-16；**2026-10-13 重新核对路径与体积**。状态：**尚未上传**。网盘链接：**待补充**。

本清单记录哪些文件只在本机、未进入 GitHub，以及恢复时的布局要求。

> **2026-10-13 复核说明**：初版清单里大量路径已过期（权重原先散在 `InstanceRecognition/ThirdParty/*/weights/`、`Video2PointCloud/checkpoints/`，`/home/wzt/HDD/models/` 已不存在）。下表的路径与体积均为当前 checkout 的实测值。

## 1. 模型权重

权重现已**集中到仓库根目录 `weights/`**：

| 状态 | 本地路径 | 约占用 | 内容 |
| --- | --- | ---: | --- |
| 待上传 | `weights/sam3.pt`、`weights/sam3.1_multiplex.pt` | 6.5 GiB | SAM3 / SAM3.1 主权重 |
| 待上传 | `weights/yoloe26_prompt_free/weights/` | 44 MiB | YOLOE-26 n/s Prompt-Free 权重 |
| 待上传 | `weights/yolo26_seg/weights/` | 23 MiB | YOLO26-seg |
| 待上传 | `weights/yolo26_detect/weights/` | 20 MiB | YOLO26 Detect |
| 待上传 | `weights/WeDetect/` | 824 MiB | WeDetect PyTorch 权重与导出的 ONNX |

`weights/` 合计约 **7.4 GiB**。主流水线与各示例的默认路径已按此布局调整。

模型的访问权限与再分发条件沿用原来源要求，**不把私有网盘链接公开写进 GitHub**。

## 2. 数据

| 状态 | 本地路径 | 约占用 | 内容与恢复方式 |
| --- | --- | ---: | --- |
| 待备份，需授权 | `/home/seer/wzt/data/scannet_v2/` | 8.7 GiB | 已下载的 ScanNet 场景（5 个）。原布局是 `<root>/<scene_id>/<scene_id>.sens`，**没有额外的 `scans/` 层**。可通过 `SCANNET_ROOT` 指向其他位置；详见 [ScanNet 数据说明](scannet-data.md) |

ScanNet 为需申请访问的数据，**不要公开再分发下载包**；备份与共享范围应符合获批用途及原数据条款。

## 3. 实验产物（体积大、可重建，但重建成本高）

| 状态 | 本地路径 | 约占用 | 内容 |
| --- | --- | ---: | --- |
| 待上传 | `InstanceRecognition/runtime/` | **20 GiB** | 见下 |
| 待上传 | `Video2PointCloud/data/` | 209 MiB | 抽帧 RGB、深度、输入 NPZ 与采样清单 |
| 待上传 | `Video2PointCloud/output/` | 3.5 GiB | 点云、评估 JSON、NPZ、中间结果与可视化（部分 Markdown 报告已在 Git 中） |

`InstanceRecognition/runtime/` 的主要构成：

| 子目录 | 约占用 | 是否需要恢复 |
| --- | ---: | --- |
| `mask_pointclouds/` | 12 GiB | **需要**——Harness 的点云来源 |
| `experiments/` | 6.1 GiB | **需要**——5 轮实验的页面与日志 |
| `results/`、`results_gpt/` | 661 MiB | 上游推理缓存，可重跑 |
| `debug/`、`comparisons/` | 743 MiB | 一次性调试页与对比页 |
| `frames/` | 137 MiB | Harness 结果页显示的 RGB |
| 其余（`batches`、`logs`、`verification`、`diagnostics` 等） | 约 30 MiB | 过程记录 |

GT 修正点云的评估 JSON、坐标变换、采样清单、像素对应等应与点云一起保留，不只上传单独的 PLY。

## 4. 文档配套资源

| 状态 | 本地路径 | 约占用 | 说明 |
| --- | --- | ---: | --- |
| 待上传 | `ScanNet/assets/` | 49 MiB | 数据介绍网页的示例图 |
| 待上传 | `docs/assets/` | 128 MiB | 方案讨论截图、全局 bbox 数据；保留目录结构 |
| 待上传 | `news/papers/` | 90 MiB | 阅读用论文 PDF；HTML 阅读记录已提交，PDF 从官方来源获取或仅作个人备份 |

签名图片 `docs/qin.png`、`docs/zhitaowang.png` 和 `scannetpp-terms-of-use.pdf` 属于个人申请材料，
**不进入 GitHub，也不纳入共享数据包**；如需保管，只作独立私有备份。

## 5. 不必打包的目录

- `.git/`、各第三方 `.git/`：主仓库已版本化，上游代码按子模块固定提交重新获取。
- `.venv/`、`__pycache__/`、`.cache/`、`.pytest_cache/`：可重建，不作为模型权重备份。
- `.omx/`：本地代理状态、历史、临时验证材料，不公开上传。
- 密钥、令牌、`.env` 和账号会话：不进入 GitHub 或共享网盘包。

WeDetect 的 4 个本地新增辅助文件已保存到 `third_party_overlays/WeDetect/`，不依赖网盘恢复。
依赖安装参考各模块现有 README 和 requirements；此代码快照未声称已提供所有环境的完整锁文件。

## 6. 后续上传时填写

- [ ] 按“权重 / 数据 / 运行结果 / 文档资源”分包，保留原相对路径。
- [ ] 生成实际文件列表、字节数和 SHA-256 校验，尤其记录外部符号链接的目标。
- [ ] 将包上传到有权限控制的私有网盘；检查空间容量与原资源访问条件。
- [ ] 私下记录下载链接与提取信息；公开仓库只记录包名、版本和非敏感校验值。
- [ ] 下载到临时目录验证校验和、路径与离线网页，再标记“已上传／已恢复验证”。

本次只重新核对了路径与体积，**没有执行网盘上传**，也没有删除本地数据或权重。
