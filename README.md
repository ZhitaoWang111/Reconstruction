# Reconstruction

从 RGB 视频探索场景重建、物体实例识别与可交互 SimReady 场景生成。
本仓库是本地 `CVPR_2027` 工作目录的**代码与文档版本**，不是包含全部数据和模型的即开即用发布包。

## 目录

| 目录 | 内容 |
| --- | --- |
| `InstanceRecognition/src/` | YOLOE-26 + SAM3.1 推理、采样、模型服务、mask 到点云的映射 |
| `InstanceRecognition/web/` | 在线网页、离线浏览器生成器、点云查看器模板 |
| `InstanceRecognition/offline/` | 离线结果浏览器代码；`data.js` 为未提交的生成数据 |
| `InstanceRecognition/tests/` | 单元测试与浏览器检查 |
| `Video2PointCloud/src/` | AMB3R、DA3、MapAnything 的运行、评估和可视化封装 |
| `ScanNet/` | 数据介绍网页与示例资源生成脚本 |
| `docs/`、`memorandum/`、`news/` | 方案讨论、制作原则、方法与论文阅读记录 |
| `third_party_overlays/` | 第三方仓库之外备份的本地复现辅助脚本 |

## 获取代码

```bash
git clone --recurse-submodules https://github.com/ZhitaoWang111/Reconstruction.git
cd Reconstruction
python3 scripts/restore_third_party_overlays.py --check
python3 scripts/restore_third_party_overlays.py
```

已有非递归克隆可先执行 `git submodule update --init --recursive`。
5 个顶层第三方仓库用 gitlink 固定到当前实际使用的提交；DA3 内的 SALAD 子模块由上游递归恢复。
本仓库没有复制或重新发布这些上游仓库的完整历史、数据与模型。它们各自的许可证仍适用。

WeDetect 的 4 个本地新增复现文件不在上游提交里，单独保存于
[`third_party_overlays/WeDetect/`](third_party_overlays/WeDetect/)。恢复脚本仅向对应子模块补充这些文件，
若同名文件内容不同会拒绝覆盖，不修改上游已有文件。

## 数据、模型与页面资源

数据、权重、推理结果、图片和论文 PDF **没有上传到 GitHub**。
待上传私有网盘的目录、体积、恢复位置见 [`docs/cloud-upload-checklist.md`](docs/cloud-upload-checklist.md)。
当前尚未上传网盘，没有可用网盘下载链接。

运行前应恢复所需文件并安装各模块的环境。现有脚本中保留了本机路径和实验命令，
不能把“代码已上传”理解为在任意机器上已复现实验。
实例识别路径可通过 `SCANNET_ROOT`、`YOLOE_PYTHON`、`SAM31_PYTHON`、
`YOLOE_CHECKPOINT`、`SAM31_CHECKPOINT`、`INSTANCE_RECOGNITION_RUNTIME` 覆盖；
其他模块请参阅各自 README 与命令行参数。

- [实例识别说明](InstanceRecognition/README.md)
- [点云重建说明](Video2PointCloud/README.md)
- [所有网页的打开方法](web_command.md)
- [方案讨论](docs/scene-reconstruction-discussions.html)

网页源码保留原有资源相对路径。未恢复资源时，数据展示页的图片、点云和离线索引不可用；
纯文字记录可直接阅读。可恢复 `InstanceRecognition/offline/data.js`，
也可在恢复推理结果后按模块说明重新生成它。

## 实验边界

当前 mask 点云输出保留逐帧观测及重叠关系，尚不能当作跨帧全局物体身份。
GT 修正的 AMB3R 点云是 GT-assisted 诊断结果，不代表纯 RGB 输入天然具备可靠绝对尺度。

## 验证

在具备现有 Python 依赖的环境中：

```bash
python3 -m pytest InstanceRecognition/tests -q
node InstanceRecognition/tests/test_episode_stats.js
python3 scripts/restore_third_party_overlays.py --check
```

浏览器端到端检查与真实模型推理还需要恢复相应数据、浏览器或权重。
签名图片、申请表、账号凭证不属于代码发布内容，不应上传公开仓库。
