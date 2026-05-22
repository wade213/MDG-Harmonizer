#!/bin/bash
# AutoDL 云端环境初始化脚本
# 上传 hcdm_autodl.tar.gz 后在云端运行: bash scripts/setup_cloud.sh

set -e

echo "=== MDG-Harmonizer AutoDL 环境初始化 ==="

# 1) 检查目录结构
if [ ! -f "run.py" ] || [ ! -d "models" ]; then
    echo "错误: 请在 HCDM-master 项目根目录下运行此脚本"
    exit 1
fi

# 2) 创建虚拟环境
echo "[1/4] 创建虚拟环境..."
python3 -m venv .venv
source .venv/bin/activate

# 3) 安装项目依赖（AutoDL 镜像已有 PyTorch + CUDA）
echo "[2/4] 安装项目依赖..."
pip install numpy pandas tqdm scipy opencv-python-headless Pillow \
    'tensorboardX>=1.14' thop timm scikit-image lpips -q

# 4) 创建数据集目录结构（从 composite_degraded_images 硬链接出 train/test）
echo "[3/4] 创建数据集目录结构..."
python tools/setup_diharmony4_datasets.py

# 5) 验证环境
echo "[4/4] 验证环境..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

from data.dataset import HarmonizationTrainDataset
d = HarmonizationTrainDataset([
    'D-Hday2night/composite_images_train/',
    'D-HCOCO/composite_images_train/',
    'D-HFlickr/composite_images_train/',
])
print(f'Mixed dataset: {len(d)} images')
print('All OK!')
"

echo ""
echo "=== 环境初始化完成 ==="
echo ""
echo "训练命令:"
echo "  source .venv/bin/activate"
echo "  python run.py -p train -c config/ablation_A_full_train.json -gpu 0"
echo ""
echo "测试命令:"
echo "  python run.py -p test -c config/ablation_A_full_test.json -gpu 0"
