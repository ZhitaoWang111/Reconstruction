#!/bin/bash
# GenIA 服务器安装脚本（venv 版，适配阿里云 DSW RTX PRO 5000 Blackwell）
# 用法：bash GenIA_server_install.sh
# 目标：/mnt/data/usr/wzt/submodules/GenIA

set -e

# ========== 配置 ==========
TARGET_DIR="/mnt/data/usr/wzt/submodules/GenIA"
VENV_DIR="$TARGET_DIR/.venv"
CUDA_VERSION="12.8"
CUDA_HOME="/usr/local/cuda-$CUDA_VERSION"
# Blackwell 架构算力 12.0
export GENIA_TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=4
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}

# 网络配置：使用国内镜像
export HF_ENDPOINT=https://hf-mirror.com
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128/

echo "== 配置检查 =="
echo "目标目录: $TARGET_DIR"
echo "CUDA: $CUDA_HOME"
echo "算力: $GENIA_TORCH_CUDA_ARCH_LIST"
nvidia-smi --query-gpu=index,name,compute_cap --format=csv
[ -d "$CUDA_HOME" ] || { echo "错误：CUDA toolkit 不存在于 $CUDA_HOME"; exit 1; }

# ========== 克隆代码 ==========
echo "== 克隆 GenIA 仓库 =="
if [ -d "$TARGET_DIR" ]; then
    echo "$TARGET_DIR 已存在，进入目录"
    cd "$TARGET_DIR"
else
    git clone https://github.com/facebookresearch/GenIA "$TARGET_DIR"
    cd "$TARGET_DIR"
fi

# ========== 初始化子模块 ==========
echo "== 初始化子模块 =="
git submodule update --init submodules/sam-3d-objects submodules/map-anything

# 打补丁到 sam-3d-objects
if git -C submodules/sam-3d-objects apply --reverse --check ../sam-3d-objects.patch 2>/dev/null; then
    echo "sam-3d-objects 补丁已应用"
else
    git -C submodules/sam-3d-objects apply ../sam-3d-objects.patch
    echo "已应用 sam-3d-objects 补丁"
fi

# actionmesh 用 https 而非 git@，绕过 SSH
git -c url."https://github.com/".insteadOf="git@github.com:" \
    submodule update --init --recursive submodules/actionmesh

# ========== 创建 venv 虚拟环境 ==========
echo "== 创建 Python venv =="
if [ -d "$VENV_DIR" ]; then
    echo "venv 已存在: $VENV_DIR"
else
    python3 -m venv "$VENV_DIR"
    echo "已创建 venv: $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel

# ========== 安装 PyTorch ==========
echo "== 安装 PyTorch 2.7.1+cu128 =="
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128

# ========== 安装依赖 ==========
echo "== 安装 requirements.txt =="
pip install -r requirements.txt

# ========== 编译 CUDA 扩展（按 sm_120 编译）==========
echo "== 编译 CUDA 扩展（算力 $GENIA_TORCH_CUDA_ARCH_LIST）=="
pip install --no-build-isolation diso==0.1.4
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9"
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3
pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0
pip install --no-build-isolation --no-deps git+https://github.com/rahul-goel/fused-ssim.git

# ========== 预编译 wheel（可能没 sm_120，先装着试）==========
echo "== 安装预编译包 =="
pip install --no-deps xformers==0.0.31 || echo "xformers 安装失败，稍后可能需要源码编译"
pip install --no-deps spconv-cu126==2.3.8 cumm-cu126==0.7.11 || echo "spconv 安装失败，稍后可能需要源码编译"

# ========== 安装子模块包 ==========
echo "== 安装 MoGe、utils3d、map-anything、actionmesh =="
pip install --no-deps git+https://github.com/microsoft/MoGe.git@07444410f1e33f402353b99d6ccd26bd31e469e8
pip install --no-deps git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183
pip install --no-deps -e submodules/map-anything
pip install --no-deps uniception==0.1.6
pip install --no-deps -e submodules/actionmesh

# ========== 安装 GenIA 自身 ==========
echo "== 安装 GenIA 包 =="
pip install --no-deps -e .

echo ""
echo "========================================"
echo "安装完成！"
echo "========================================"
echo "虚拟环境: $VENV_DIR"
echo ""
echo "下一步："
echo "  1. 激活环境: source $VENV_DIR/bin/activate"
echo "  2. 登录 HF（SAM 3D Objects 需要接受协议后才能下载）:"
echo "     huggingface-cli login --token <你的 HF token>"
echo "  3. 下载权重: python -m genia.core.download_weights"
echo "  4. 运行 demo: python demo/run_demo.py image"
echo ""
