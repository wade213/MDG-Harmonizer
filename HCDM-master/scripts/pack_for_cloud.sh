#!/bin/bash
# 打包项目用于 AutoDL 云端 GPU 训练
# 在 HCDM-master/ 目录下运行: bash scripts/pack_for_cloud.sh
#
# 打包内容: 代码 + 预训练权重(1D_embed) + D-Hday2night 数据集
# D-HCOCO 和 D-HFlickr 需单独通过 AutoDL 数据盘上传

set -e

cd "$(dirname "$0")/.."

OUT="hcdm_autodl.tar.gz"

echo "=== 打包项目到 ${OUT} ==="

tar czf "${OUT}" \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.claude' \
  --exclude='.git' \
  --exclude='pretrained_model/checkpoint/2d_map' \
  --exclude='D-HCOCO' \
  --exclude='D-HFlickr' \
  --exclude='D-HAdobe5k' \
  --exclude='HAdobe5k' \
  --exclude='HFlickr' \
  --exclude='D-Hday2night/composite_images_train' \
  --exclude='D-Hday2night/composite_images_test' \
  --exclude='results_cloud' \
  --exclude='hcdm_*.tar.gz' \
  models/ core/ data/ evaluation/ scripts/ tools/ \
  config/ run.py requirements.txt \
  pretrained_model/ \
  D-Hday2night/ \
  TestData/Hday2night/ \
  Hday2night_test.txt Hday2night_train.txt \
  IHD_degraded_test.txt IHD_degraded_train.txt \
  IHD_test.txt IHD_train.txt

SIZE=$(du -sh "${OUT}" 2>/dev/null | cut -f1 || echo "unknown size")
echo "打包完成: ${OUT} (${SIZE})"
echo ""
echo "下一步:"
echo "  1. 上传 ${OUT} 到 AutoDL 数据盘（网页上传或 JupyterLab 文件管理器）"
echo "  2. D-HCOCO (5.7GB) 和 D-HFlickr (1.3GB) 需单独上传"
echo "  3. 云端解压后运行: bash scripts/setup_cloud.sh"
