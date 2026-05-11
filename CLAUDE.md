# MDG-Harmonizer 项目

## 项目背景

- **基础**: 在 HCDM (Image Harmonization in Complex Degradation Scenes) 仓库基础上做的本科毕业设计
- **框架名**: MDG-Harmonizer
- **目标**: 在 HCDM 基础上提升前景区域指标 + 加速推理

## 硬件与训练约束

- **GPU**: NVIDIA RTX 3050 Ti (4-6 GB VRAM)
- **必须使用**: fp16 AMP + GradScaler + 梯度累积，不开 EMA
- **训练分辨率**: 256x256
- **训练策略**: 冻结 HCDM 预训练 U-Net 主干，仅训练新加模块 (CDP-Net, AFM)
- **数据集**: 仅用 D-Hday2night (311 train / 133 test)

## 三个核心模块

| 缩写 | 全称 | 替换对象 | 参数预算 |
|------|------|---------|----------|
| CDP-Net | Compact Degradation Prior Network | DegradationPrior (8维→32维) | < 1M |
| AFM | Adaptive Feature Modulation | FiLMLayer | < 0.5M/实例 |
| FB-Loss | Foreground-Boundary Aware Loss | 原损失函数 | 不增参 |

加速推理: DPM-Solver++ (25步) 替换 DDPM (1000步)

## 运行环境

- **项目根目录**: `D:\HCDM-master\HCDM-master\`
- **虚拟环境**: `D:\HCDM-master\HCDM-master\.venv\`
- **执行 Python**: `.\.venv\Scripts\python.exe`
- **pip install**: `.\.venv\Scripts\pip.exe install xxx`

## 严禁修改

- `.venv/` 下任何文件
- `experiments/` 下历史实验记录
- 原 `models/degradation_prior.py` 的 DegradationPrior 与 FiLMLayer (保留作 ablation baseline)
- `models/loss.py` / `losses.py` 中已有损失类

## 实验对比基线

消融实验: Full / w/o CDP-Net / w/o AFM / w/o FB-Loss
报告指标: MSE / PSNR / SSIM / fMSE / fPSNR / fSSIM
