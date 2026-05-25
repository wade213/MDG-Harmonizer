# MDG-Harmonizer 项目

## 项目背景

- **基础**: 在 HCDM (Image Harmonization in Complex Degradation Scenes) 仓库基础上做的本科毕业设计
- **框架名**: MDG-Harmonizer
- **目标**: 在 HCDM 基础上提升前景区域指标 + 加速推理

## 数据集（重要：区分两个版本）

| 版本 | 本地路径 | 实际内容 | 数量 |
|------|---------|---------|------|
| iHarmony4 (错) | `Hday2night/` | 原始合成图，**非**论文所用 | 311 train / 133 test |
| **D-iHarmony4 (对)** | **`D-Hday2night/`** | 论文所用退化版 | 311 train / 133 test |
| D-iHarmony4 (对) | **`D-HCOCO/`** | 退化版 | 38545 train / 4283 test |
| D-iHarmony4 (对) | **`D-HFlickr/`** | 退化版 | 7449 train / 828 test |

**训练策略**: 三数据集混合训练（38545+7449+311=46305 张），测试时分别在各数据集上出指标。

**目录结构**: 每个 D-* 数据集下有 `composite_images_train/` 和 `composite_images_test/`（硬链接从 `composite_degraded_images/` 分出），由 `tools/setup_diharmony4_datasets.py` 创建。

**关键发现**: `Hday2night/` 训练集与 `D-Hday2night/` 同名文件但像素不同（20/20 验证），确认是 iHarmony4 而非 D-iHarmony4。之前的消融训练用错了数据集。

子集规模: D-HCOCO (38545/4283) > D-HAdobe5k (19437/2160) > D-HFlickr (7449/828) > D-Hday2night (311/133)

## 硬件与训练约束

- **本地 GPU**: NVIDIA RTX 3050 Ti (4-6 GB VRAM) — 必须用 fp16 + 梯度累积
- **训练分辨率**: 256x256
- **训练策略**: 冻结 HCDM 预训练 U-Net 主干，仅训练新加模块
- **推理加速**: DDPM 200步（5× 加速，PSNR 仅降 0.8 dB）。DDIM 与 repaint 策略不兼容（η=0 确定性模式边缘无法融合），放弃

## 三个核心模块

| 缩写 | 全称 | 替换对象 | 参数预算 |
|------|------|---------|----------|
| CDP-Net | Compact Degradation Prior Network | DegradationPrior (8维→32维) | < 1M |
| AFM | Adaptive Feature Modulation | FiLMLayer | < 0.5M/实例 |
| FB-Loss | Foreground-Boundary Aware Loss | 原损失函数 | 不增参 |

## 实验结果

### Baseline（正确数据集，D-iHarmony4）

| 模型 | 数据集 | PSNR | SSIM | MAE | 备注 |
|------|--------|------|------|-----|------|
| 原 HCDM (1D_embed/770) | D-Hday2night (133 张) | 36.85 | — | 1.19 | 与论文报的 36.89 一致 |
| 原 HCDM (2D_map/550) | D-Hday2night (133 张) | 23.40 | 0.920 | 5.54 | 未充分训练的对照模型 |

### MDG-Harmonizer 消融实验（D-iHarmony4 三数据集混合训练，DDPM 200步）

| 消融 | 模块 | PSNR | SSIM | MAE | fPSNR | 备注 |
|------|------|------|------|-----|-------|------|
| D_no_fb | CDP + AFM | **36.40** | 0.973 | 1.54 | 21.26 | 31 epoch |
| C_no_afm | CDP + FB-Loss | 35.86 | 0.971 | 1.72 | 20.72 | 38 epoch (cut at 30) |
| A_full | CDP + AFM + FB-Loss | 35.81 | 0.972 | 1.59 | 20.67 | 30+10+10=50 epoch |
| B_no_cdp | AFM + FB-Loss | 35.67 | 0.971 | 1.66 | 20.51 | 15+16=31 epoch |

**训练配置**: 30-50 epoch, batch=16, fp32, lr=3e-5, freeze_backbone, AutoDL RTX 5090/vGPU

**A_full 训练历史**: 首次 30 epoch (260520_231626) → batch=8 续训 10 epoch (260523_142959) → batch=16 续训 10 epoch (260523_204007) = 有效 50 epoch

### 解冻微调实验（基于 A_full 50epoch）

| 方案 | 训练数据 | epoch | PSNR | 结论 |
|------|---------|-------|------|------|
| 冻结 (50ep) | 46K 混合 | 50 | 35.81 | 稳定基线 |
| last2 decoder | 311张 | 5 | 35.67 | 近似冻结 |
| last2 decoder | 2000张 | 5 | 35.50 | 略降 |
| last6 decoder | 311张 | 10 | 34.63 | 灾难性遗忘 |

**结论**: 解冻 decoder 均使指标下降，HCDM 预训练 U-Net 与旧退化管线深度耦合，少量微调不足以解绑。

### M-DPR 掩码感知退化提示路由（工作二，DDPM 200步）

| 实验 | 配置 | PSNR | 可训参数 |
|------|------|------|---------|
| MDG-D (基线) | CDP + AFM | 36.40 | ~0.6M |
| **Router only** | descriptor_only | 35.94 | **~1K** |
| Hybrid | CDP + Router | 35.80 | ~0.6M |
| Uniform | 平均权重 | 35.38 | ~1K |

**结论**: M-DPR 以 1K 可训参数达到接近主干性能（差 0.46 dB），自适应路由 (35.94) > 均匀权重 (35.38)，验证退化描述子 + Prompt Router 可独立工作。

### 推理加速对比

| 方法 | 步数 | PSNR | vs 1000步 | 加速比 |
|------|------|------|----------|--------|
| DDPM | 1000 | 36.56 | — | 1× |
| DDPM | 200 | 35.81 | -0.8 dB | **5×** |
| DDPM | 50 | 31.02 | -5.5 dB | 20× |

## 关键调试经验

- **AMP fp16 NaN**: lr=5e-4 + fp16 时 CDP-Net 梯度爆炸导致权重全 NaN，loss 变为 0。修复：关 AMP 用 fp32，lr 降到 3e-5
- **DDIM 不兼容 repaint**: DDIM η=0 确定性模式不注入后验噪声，边缘无法融合。η=1 可工作但等于 DDPM，无加速效果。放弃 DDIM
- **loss_per_epoch.npy 全 0**: pandas ChainedAssignmentError 导致 LogTracker 记录 NaN 被吞成 0，实际 loss 正常（~1.0）。不影响训练
- **batch=48 + AMP 在 PRO 6000 96GB 上 OOM**: 加噪图(6ch) + 梯度图追踪消耗远超预期，batch=16 + fp32 稳定

## 自创/修改文件清单

```
models/cdp_net.py                               — CDP-Net 退化先验编码器
models/afm.py                                   — AFM 自适应调制模块
models/fb_loss.py                               — FB-Loss 损失函数
models/network_mdg.py                           — MDG 主网络包装 (MDGNetwork)
models/model_mdg.py                             — MDG 训练器 (MDGTrainer, 继承 RIHD)
models/guided_diffusion_modules/unet_mdg.py     — MDG UNet (MDGUNet, 加 AFM)
models/__init__.py                              — 修改：支持 MDGTrainer
models/dpm_solver.py                            — DPM-Solver++ 采样器
data/dataset.py                                 — 修改：支持 list data_root + 文件名路径推导
run.py                                          — 修改：加 --sampler / --ddim-steps 参数
scripts/compute_baseline_metrics.py             — 指标计算脚本 (MAE/MSE/PSNR/SSIM/fMAE/fPSNR/fSSIM)
tools/setup_diharmony4_datasets.py              — 数据集目录结构创建脚本
config/ablation_*_train.json / *_test.json      — 消融实验配置（data_root 改为三数据集列表）
config/mdg_decoder_finetune_*.json              — decoder finetune 配置
config/prompt_router_*_train.json               — M-DPR 退化提示路由实验配置（工作二）
models/degradation_descriptor.py                — 8 维退化描述子（工作二核心）
models/degradation_prompt.py                    — DegradationPromptBank + MaskAwarePromptRouter
models/network_prompt_router_mdg.py             — PromptRouterMDGNetwork（三种模式）
scripts/infer_single.py                         — 修改：支持动态网络 + prompt 权重输出
```

## 运行环境

- **项目根目录**: `D:\MDG-Harmonizer\src\`
- **虚拟环境**: `D:\MDG-Harmonizer\src\.venv\`
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
| `data/dataset.py` | 数据集加载（支持 list data_root + 文件名路径推导） |
| `evaluation/*.py` | 评估脚本 |
| `CLAUDE.md` | 本文件 |
| `paper/`、`figs/` | 论文和图表目录（待新建） |

### 严禁修改/删除

- `.venv/` — 虚拟环境，动辄 4.6GB 重装
- `experiments/` — 历史实验记录和 checkpoint（旧 MDG 消融结果已清理，仅保留 baseline）
- `models/degradation_prior.py` — ablation baseline
- `models/loss.py`、`models/losses.py`、`models/network_modified_backup.py` — 原 HCDM baseline
- `pretrained_model/` — 预训练权重
- `Hday2night/`、`TestData/`、`D-HCOCO/`、`D-Hday2night/`、`D-HFlickr/` — 数据集
- `core/` — 基础框架代码
- `.gitignore`、`.git/` — git 相关

### 操作红线和安全规则

1. **绝不 `rm -rf`** 任何目录，尤其是 `experiments/`、`.venv/`、`pretrained_model/`
2. **绝不 `git reset --hard`** 或 `git push --force`
3. **修改前先备份**：涉及 >20 行的改动，先用 `cp` 备份原文件
4. **只改目标方法**：修 DPM bug 只能改 `restoration_dpm` 相关方法，不动其他 forward/training 逻辑
5. **改完跑单元测试**：修改后立即 `python -m models.xxx` 验证
6. **不确定时先问**：任何超出上述"允许范围"的修改，必须先确认

## 待做

1. 系统搭建：图像 harmonization 演示系统（Gradio，含退化分析可视化）
2. 论文撰写（`paper/` 目录）
3. D-HCOCO、D-HFlickr 测试集上出指标（目前仅 D-Hday2night）
4. Prompt 权重可视化脚本
