# 数据与权重：待上传私有网盘

盘点日期：2026-09-16。状态：**尚未上传**。网盘链接：**待补充**。

GitHub 只保存代码、文本报告、网页源码和第三方版本引用；以下文件仍留在本机，未删除或搬走。
体积是本次 `du -sh` 的近似磁盘占用，不是打包压缩后的大小；目录存在重叠时不要重复计数。
仓库相对路径以 `/home/wzt/wzt/CVPR_2027` 为根，恢复时保持同样的相对布局。

## 1. 模型权重

| 状态 | 本地路径 | 约占用 | 恢复内容 |
| --- | --- | --- | --- |
| 待上传 | `weights/` | 6.5 GiB | `sam3.pt`、`sam3.1_multiplex.pt` |
| 待上传 | `InstanceRecognition/ThirdParty/yoloe26_prompt_free/weights/` | 44 MiB | YOLOE-26 n/s Prompt-Free 权重 |
| 待上传 | `InstanceRecognition/ThirdParty/yolo26_seg/weights/` | 23 MiB | YOLO26-seg |
| 待上传 | `InstanceRecognition/ThirdParty/yolo26_detect/weights/` | 20 MiB | YOLO26 Detect |
| 待上传 | `InstanceRecognition/ThirdParty/WeDetect/checkpoints/` | 414 MiB | WeDetect PyTorch 权重 |
| 待上传 | `InstanceRecognition/ThirdParty/WeDetect/wedetect_anything/onnx_models/` | 410 MiB | 导出的 ONNX 与配套元数据 |
| 待上传 | `Video2PointCloud/checkpoints/amb3r/` | 3.9 GiB | AMB3R 权重 |
| 待上传 | `/home/wzt/HDD/models/da3nested-gl-1.1-b2359bd/` | 6.3 GiB | DA3 模型目录，含配置与权重 |
| 待上传 | `/home/wzt/HDD/models/map-anything-a1d87e9/` | 4.6 GiB | MapAnything 模型目录，含配置与权重 |

注意：`Video2PointCloud/checkpoints/da3` 和 `checkpoints/mapanything` 是指向上述 HDD 目录的符号链接。
上传时应备份链接目标的实际文件，不能只上传链接；换机器后重新建立链接或按模块参数指定新位置。
模型的访问权限与再分发条件沿用原来源要求，不把私有网盘链接公开写进 GitHub。

## 2. 数据与实验产物

| 状态 | 本地路径 | 约占用 | 内容与恢复方式 |
| --- | --- | --- | --- |
| 待备份，需授权 | `/home/wzt/HDD/datasets/scannet_v2/` | 32 GiB | 已下载的 ScanNet 场景；原目录恢复或设置 `SCANNET_ROOT` 指向 `scans/` |
| 待上传 | `Video2PointCloud/data/` | 226 MiB | 抽帧 RGB、深度、输入 NPZ 与采样清单 |
| 待上传 | `InstanceRecognition/runtime/` | 5.8 GiB | 帧缓存、批次记录、YOLOE/SAM 结果、逐 mask 点云及生成的离线点云页 |
| 待上传 | `Video2PointCloud/output/` | 7.3 GiB | 点云、评估 JSON、NPZ、中间结果、可视化与已有校验文件；少量 Markdown 报告已在 Git 中 |
| 待上传 | `InstanceRecognition/offline/data.js` | 1.7 MiB | 离线场景／帧索引；也可在恢复 `runtime/results/` 后重新生成 |
| 待上传 | `InstanceRecognition/ThirdParty/yoloe26_prompt_free/inputs/` | 小量图片 | 官方示例输入 |
| 待上传 | `InstanceRecognition/ThirdParty/{yoloe26_prompt_free,yolo26_seg,yolo26_detect,yoloe26_sam3}/outputs/` | 以打包清单为准 | 各方法比较结果；花括号表示四个实际目录 |
| 待上传 | `InstanceRecognition/ThirdParty/WeDetect/wedetect_anything/outputs/` | 408 KiB | WeDetect 示例输出 |
| 待上传 | `InstanceRecognition/ThirdParty/yoloe26-vs-yolo26-seg-comparison.png` | 单张图片 | 方法对比图 |

ScanNet 为需申请访问的数据，**不要公开再分发下载包**；备份和共享范围应符合获批用途及原数据条款。
GT 修正点云的评估 JSON、坐标变换、采样清单、像素对应等应与点云一起保留，不只上传单独的 PLY。

## 3. 文档配套资源

| 状态 | 本地路径 | 约占用 | 说明 |
| --- | --- | --- | --- |
| 待上传 | `ScanNet/assets/` | 1.3 MiB | 数据介绍网页示例图 |
| 待上传 | `docs/assets/` | 892 KiB | 方案讨论截图；保留对应目录结构 |
| 待上传 | `news/papers/*.pdf`、`news/papers/CN/*.pdf` | `news/papers/` 总计约 65 MiB | 阅读用论文 PDF；HTML 阅读记录已提交，PDF 从官方来源获取或仅作个人备份 |
| 可选备份 | `tmp/109.png`、`tmp/110.png`、`tmp/111.png` | 三张截图 | 已与 `docs/assets/scene-reconstruction/round-02/` 留有同内容副本，不必重复上传 |

签名图片 `docs/qin.png`、`docs/zhitaowang.png` 和 `scannetpp-terms-of-use.pdf` 属于个人申请材料，
**不进入 GitHub，也不纳入共享数据包**；如需保管，只作独立私有备份。

## 4. 不必打包的目录

- `.git/`、各第三方 `.git/`：主仓库已版本化，上游代码按子模块固定提交重新获取。
- `.venv/`、`__pycache__/`、`.cache/`、`.pytest_cache/`：可重建，不作为模型权重备份。
- `.omx/`：本地代理状态、历史、临时验证材料，不公开上传。
- 密钥、令牌、`.env` 和账号会话：不进入 GitHub 或共享网盘包。

WeDetect 的 4 个本地新增辅助文件已经保存到 `third_party_overlays/WeDetect/`，不依赖网盘恢复。
依赖安装参考各模块现有 README 和 requirements；此代码快照未声称已提供所有环境的完整锁文件。

## 5. 后续上传时填写

- [ ] 按“权重 / 数据 / 运行结果 / 文档资源”分包，保留原相对路径。
- [ ] 生成实际文件列表、字节数和 SHA-256 校验，尤其记录外部符号链接的目标。
- [ ] 将包上传到有权限控制的私有网盘；检查空间容量与原资源访问条件。
- [ ] 私下记录下载链接与提取信息，公开仓库只记录包名、版本和非敏感校验值。
- [ ] 下载到临时目录验证校验和、路径和离线网页，再标记“已上传／已恢复验证”。

本次只创建 Git 代码快照和本清单，没有执行网盘上传，也没有删除本地数据或权重。
