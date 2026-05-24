# MDG-Harmonizer

**本科毕业设计项目** — 在 HCDM 扩散模型框架上做图像 harmonization（图像协调），通过三个轻量模块实现参数高效迁移，在冻结预训练底座的条件下以仅 1% 可训参数（~0.6M）取得接近 HCDM baseline（63M 全训）的结果。

## 三个核心模块

| 模块 | 全称 | 替换对象 | 参数 |
|------|------|---------|------|
| **CDP-Net** | Compact Degradation Prior Network | DegradationPrior（8维）→ 64维卷积编码 + 解耦头 | < 1M |
| **AFM** | Adaptive Feature Modulation | FiLMLayer → Cross-Attention 空间自适应调制 | < 0.5M |
| **FB-Loss** | Foreground-Boundary Aware Loss | 新增：前景/边界/感知/频率 5 项损失 | 0 |

## 核心思路

HCDM 需要 63M 参数全训练、770 epoch。我们冻结预训练 U-Net 底座，只训练三个插件模块（~0.6M 参数，仅占 1%），30 epoch 即接近 baseline 水平。

## 实验结果 (D-Hday2night 133张)

### 消融实验（DDPM 200步，三数据集混合训练）

| 消融 | 模块 | PSNR | SSIM | MAE | fPSNR |
|------|------|------|------|-----|-------|
| **D** | CDP + AFM（无 FB-Loss） | **36.40** | 0.973 | 1.54 | 21.26 |
| C | CDP + FB-Loss（无 AFM） | 35.86 | 0.971 | 1.72 | 20.72 |
| A | CDP + AFM + FB-Loss | 35.81 | 0.972 | 1.59 | 20.67 |
| B | AFM + FB-Loss（无 CDP） | 35.67 | 0.971 | 1.66 | 20.51 |
| HCDM baseline | 原版全训练 (DDPM 1000步) | 36.85 | — | 1.19 | — |

### 解冻微调实验（A_full 50epoch 基础上）

| 方案 | 数据 | epoch | PSNR |
|------|------|-------|------|
| 冻结 (50ep) | 46K | 50 | 35.81 |
| last2 decoder | 311张 | 5 | 35.67 |
| last2 decoder | 2000张 | 5 | 35.50 |
| last6 decoder | 311张 | 10 | 34.63 |

### 推理加速对比

| 方法 | 步数 | PSNR | vs 1000步 | 加速比 |
|------|------|------|----------|--------|
| DDPM | 1000 | 36.56 | — | 1x |
| DDPM | 200 | 35.81 | -0.8 dB | **5x** |
| DDPM | 50 | 31.02 | -5.5 dB | 20x |

## 项目结构

```
src/
├── models/                    ← 核心模型代码
│   ├── cdp_net.py             ← CDP-Net 退化编码器（自写）
│   ├── afm.py                 ← AFM 自适应调制（自写）
│   ├── fb_loss.py             ← FB-Loss 损失函数（自写）
│   ├── network_mdg.py         ← MDGNetwork 扩散网络（自写）
│   ├── model_mdg.py           ← MDGTrainer 训练器（自写）
│   ├── guided_diffusion_modules/unet_mdg.py ← MDG UNet（自写）
│   ├── degradation_prior.py   ← 原 HCDM baseline（勿改）
│   ├── network_modified.py    ← 原 HCDM 网络（勿改）
│   └── model_rihd.py          ← 原 HCDM 训练器
├── data/dataset.py            ← 数据集加载（修改）
├── config/                    ← 实验配置
│   ├── ablation_A_full_*.json ← Full MDG（CDP+AFM+FB）
│   ├── ablation_B_no_cdp_*.json
│   ├── ablation_C_no_afm_*.json
│   └── ablation_D_no_fb_*.json
├── scripts/
│   ├── compute_baseline_metrics.py  ← PSNR/SSIM 指标计算
│   ├── infer_single.py       ← 单图推理
│   ├── pack_for_cloud.sh     ← 云端打包
│   └── setup_cloud.sh        ← 云端环境初始化
├── tools/setup_diharmony4_datasets.py ← 数据集目录创建
├── core/                     ← 框架代码（勿改）
├── run.py                    ← 训练/测试入口
└── pretrained_model/         ← 预训练权重（需单独下载）
```

## 快速开始

### 环境

```bash
pip install -r requirements.txt
```

### 数据集准备

本项目使用 **D-iHarmony4** 数据集（退化版），非原版 iHarmony4。数据集目录结构：

```
D-HCOCO/
├── composite_degraded_images/   ← 所有退化合成图
├── composite_images_train/      ← 硬链接分出（脚本自动创建）
├── composite_images_test/
├── masks/
├── real_images/
├── HCOCO_train.txt
└── HCOCO_test.txt
```

```bash
# 下载 D-HCOCO / D-HFlickr / D-Hday2night 后运行：
python tools/setup_diharmony4_datasets.py
```

### 训练

```bash
# 消融 A (Full MDG)
python run.py -p train -c config/ablation_A_full_train.json -gpu 0
```

**注意**：配置已设为 fp32 + lr=3e-5（稳定方案）。不要在低显存 GPU 上开 AMP。

### 测试

```bash
python run.py -p test -c config/ablation_A_full_test.json -gpu 0

# 计算指标
python -W ignore scripts/compute_baseline_metrics.py \
    --run experiments/test_<timestamp> \
    --mask-root TestData/Hday2night/masks
```

### 单图推理

```bash
python scripts/infer_single.py \
    --checkpoint <你的checkpoint.pth> \
    --input <输入图.jpg> \
    --mask <mask图.png> \
    --output <输出.jpg> \
    --steps 200 \
    --gpu
```

## 训练稳定性注意事项

- 关闭 AMP（fp16），使用 fp32 训练
- 学习率设为 3e-5（5e-4 会导致 CDP-Net 梯度爆炸 → NaN）
- 冻结 backbone 时确保 `freeze_backbone: true`
- 先用小数据集验证 loss 正常（~1.0），再跑完整训练
- DDIM 与 repaint 策略不兼容，推荐 **DDPM 200 步**加速推理

## 引用

基于 HCDM：*Image Harmonization in Complex Degradation Scenes*

## 文档索引

- [CONTRIBUTION.md](../CONTRIBUTION.md) — 代码贡献声明（新增/修改文件清单）
- [ACKNOWLEDGEMENT.md](../ACKNOWLEDGEMENT.md) — HCDM 来源声明
- [docs/method.md](../docs/method.md) — 三个模块详细说明 + HCDM vs MDG 对比
- [docs/original_HCDM_README.md](../docs/original_HCDM_README.md) — 原 HCDM 项目说明（保留）
- [CLAUDE.md](../CLAUDE.md) — 项目开发记录与调试经验

## 联系方式

GitHub: https://github.com/wade213/MDG-Harmonizer
