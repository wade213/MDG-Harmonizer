# Code Contribution

本项目基于 HCDM 开源实现进行改进。原 HCDM 代码主要用于 baseline、数据加载、扩散模型训练框架和评估流程。本文新增和修改的主要内容如下。

## 新增文件

| 文件 | 作用 |
|------|------|
| `models/cdp_net.py` | CDP-Net，结构化退化先验编码器 |
| `models/afm.py` | AFM，自适应特征调制模块 |
| `models/fb_loss.py` | FB-Loss，前景边界感知损失函数 |
| `models/network_mdg.py` | MDG 主网络，整合 CDP-Net、AFM 和扩散模型 |
| `models/model_mdg.py` | MDG 训练器，支持冻结 backbone、梯度累积、AMP 和 loss 日志 |
| `models/guided_diffusion_modules/unet_mdg.py` | MDG UNet，在 bottleneck 加入 AFM |
| `models/dpm_solver.py` | DPM-Solver++ 采样器（实验性，推荐使用 DDPM 200步） |
| `config/ablation_A_full_train.json` | 完整模型训练配置（CDP + AFM + FB-Loss） |
| `config/ablation_B_no_cdp_train.json` | 消融 B 训练配置（去除 CDP-Net） |
| `config/ablation_C_no_afm_train.json` | 消融 C 训练配置（去除 AFM） |
| `config/ablation_D_no_fb_train.json` | 消融 D 训练配置（去除 FB-Loss） |
| `scripts/compute_baseline_metrics.py` | PSNR/SSIM/MAE/fPSNR 指标计算脚本 |
| `scripts/infer_single.py` | 单图推理脚本 |
| `tools/setup_diharmony4_datasets.py` | D-iHarmony4 数据集目录结构创建工具 |

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `run.py` | 增加 `--sampler` / `--ddim-steps` 参数 |
| `data/dataset.py` | 支持 list data_root（三数据集混合训练）、文件名路径推导 |
| `models/model_rihd.py` | `_run_restoration` 支持 DDPM/DDIM/DPM 采样切换 |
| `models/__init__.py` | 注册 MDGTrainer |

## 未修改文件

以下文件来自原 HCDM，作为 baseline 未做修改：
- `core/` — 训练框架
- `models/degradation_prior.py` — 原版退化先验（消融对照用）
- `models/network_modified_backup.py` — 原版扩散网络
- `models/loss.py`、`models/losses.py` — 原版损失函数
- `evaluation/` — 原版评估代码

## 说明

本文没有将 HCDM 原始方法作为原创内容，而是将其作为 baseline，在此基础上进行模块化改进和实验验证。所有新增代码和修改均有明确记录。
