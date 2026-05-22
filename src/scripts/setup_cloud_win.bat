@echo off
rem UCloud Windows 云端环境初始化
rem 解压 hcdm_cloud.tar.gz 后在此目录运行: scripts\setup_cloud_win.bat

echo === HCDM Windows 云端初始化 ===

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到 Python，请先安装 Python 3.10
    echo 下载: https://www.python.org/downloads/
    pause
    exit /b 1
)

if not exist "run.py" (
    echo 错误: 请在项目根目录下运行此脚本
    pause
    exit /b 1
)

echo [1/3] 创建虚拟环境...
python -m venv .venv
call .venv\Scripts\activate.bat

echo [2/3] 安装依赖...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas tqdm scipy opencv-python Pillow "tensorboardX>=1.14" thop timm scikit-image

echo [3/3] 验证环境...
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); import models.network_mdg; print('MDGNetwork: OK')"

echo.
echo === 环境就绪 ===
echo 训练: python run.py -p train -c config\mdg_decoder_finetune_train.json -gpu 0
pause
