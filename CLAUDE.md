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

## 修改安全约束（严格遵守，每次操作前自查）

### 只能修改/新增的文件

| 允许范围 | 说明 |
|----------|------|
| `models/cdp_net.py` | CDP-Net 退化先验编码器 |
| `models/afm.py` | AFM 自适应调制模块 |
| `models/fb_loss.py` | FB-Loss 损失函数 |
| `models/dpm_solver.py` | DPM-Solver++ 采样器 |
| `models/network_mdg.py` | MDG 主网络包装 |
| `models/network_modified.py` | baseline DPM 接口（仅改 restoration_dpm） |
| `models/model_mdg.py` | MDG 训练器 |
| `models/guided_diffusion_modules/unet_mdg.py` | MDG UNet |
| `models/__init__.py` | 模块导出 |
| `config/*.json` | 配置文件 |
| `run.py` | 训练入口 |
| `scripts/*.py` | 脚本 |
| `tools/*.py` | 工具（含新建可视化脚本） |
| `evaluation/*.py` | 评估脚本 |
| `CLAUDE.md` | 本文件 |
| `paper/`、`figs/` | 论文和图表目录（待新建） |

### 严禁修改/删除

- `.venv/` — 虚拟环境，动辄 4.6GB 重装
- `experiments/` — 历史实验记录和 checkpoint
- `models/degradation_prior.py` — ablation baseline
- `models/loss.py`、`models/losses.py`、`models/network_modified_backup.py` — 原 HCDM baseline
- `pretrained_model/` — 预训练权重
- `Hday2night/`、`TestData/` — 数据集
- `core/` — 基础框架代码
- `.gitignore`、`.git/` — git 相关

### 操作红线和安全规则

1. **绝不 `rm -rf`** 任何目录，尤其是 `experiments/`、`.venv/`、`pretrained_model/`
2. **绝不 `git reset --hard`** 或 `git push --force`
3. **修改前先备份**：涉及 >20 行的改动，先用 `cp` 备份原文件
4. **只改目标方法**：修 DPM bug 只能改 `restoration_dpm` 相关方法，不动其他 forward/training 逻辑
5. **改完跑单元测试**：修改后立即 `python -m models.xxx` 验证
6. **不确定时先问**：任何超出上述"允许范围"的修改，必须先确认

## 实验对比基线

消融实验: Full / w/o CDP-Net / w/o AFM / w/o FB-Loss
报告指标: MSE / PSNR / SSIM / fMSE / fPSNR / fSSIM
