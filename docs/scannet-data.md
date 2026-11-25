# 当前机器的 ScanNet v2 数据

数据目录：`/home/seer/wzt/data/scannet_v2`。
项目默认读取仓库同级的 `data/scannet_v2/`，可通过 `SCANNET_ROOT` 覆盖。

## 当前布局与读取检查（2026-09-17）

用户已将场景直接保存到数据根目录下，当前路径为：

```text
/home/seer/wzt/data/scannet_v2/scene0000_00/scene0000_00.sens
```

这里没有额外的 `scans/` 层。原下载时使用的 `scans/` 路径和后台日志命令不再适用于当前布局。

已检查第一个场景的 `.sens`：文件为 3,699,853,377 字节，格式版本为 4，
全部 5,578 帧记录边界完整，RGB 为 1296×968，深度为 640×480。
此检查验证文件结构，不等于官方校验和比对或逐帧图像内容验收。

场景目录还包含 GT 网格、场景元数据、实例聚合／分段 JSON 和二维标签 ZIP。
其他场景目录的存在不表示已逐一验证完整性。

## 与项目衔接

实例识别后端和 `ScanNet/prepare_example_assets.py` 使用上述数据根目录，均支持环境变量覆盖：

```bash
export SCANNET_ROOT=/home/seer/wzt/data/scannet_v2
```

YOLOE-26 + SAM3.1 的环境、启动与验证记录见 [实例识别说明](../InstanceRecognition/README.md)。

## 官方下载来源

用户已确认拥有 ScanNet 下载权限。
[官方数据说明](https://github.com/ScanNet/ScanNet#scannet-data)解释了各文件的含义与访问申请流程。
[官方下载脚本](https://kaldir.vc.in.tum.de/scannet/download-scannet.py)不依赖 Hugging Face 登录；
默认选择 v2，对训练／验证场景复用 v1 的 `.sens` 文件。

现有部分 Python 测试仍依赖旧机器的场景集合；恢复单个场景不等于恢复原有全部测试数据。
