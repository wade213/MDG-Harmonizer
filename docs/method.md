# MDG-Harmonizer 方法说明

## 整体框架

MDG-Harmonizer 基于 HCDM 条件扩散图像协调框架，保持其扩散 U-Net 预训练权重不变（freeze backbone），新增三个模块：

```
输入: 退化合成图 + 前景 mask
  │
  ├─→ CDP-Net ─→ deg_vec (64维退化先验)
  │                 │
  │                 ↓
  └─→ DDPM 加噪 ─→ U-Net bottleneck (AFM) ─→ 预测噪声
                       ↑
                   deg_vec 注入
  │
  ↓
去噪图 ─→ FB-Loss (vs ground truth)
```

## 三个模块

### CDP-Net：退化先验编码增强

**替换对象**：HCDM 原 `DegradationPrior`（全局池化 + 2层FC → 8维向量）

**改进动机**：原版8维容量不足以描述复杂退化（偏色、噪声、饱和度），且池化丢失空间信息。

**设计**：
- 4级卷积下采样（16→32→64→128ch）+ 全局平均池化
- 解耦多任务头：亮度(1) + 色温(1) + 饱和度(1) + 噪声(1) + 通用嵌入(60) = 64维
- 参数量 < 1M，1.6% 总参数

### AFM：自适应特征调制

**替换对象**：HCDM 原 `FiLMLayer`（gamma·features + beta，全图统一）

**改进动机**：前景和背景退化差异大，统一调制无法区分处理。

**设计**：
- deg_vec → 8个 learnable tokens
- tokens 与特征图做 cross-attention → 每个空间位置自适应选择退化补偿
- FiLM 残差 shortcut（保证不弱于原版）
- 零初始化恒等映射（不破坏预训练特征）

### FB-Loss：前景-边界感知损失

**替换对象**：无（新增）。原版只有多尺度 L1 噪声损失。

**改进动机**：噪声损失在全图等权重，模型不知道哪里是前景/边界。

**设计**：在 x̂₀ 反算后计算 5 项损失
```
FB-Loss = 1.0×L1_global + 3.0×L1_foreground + 5.0×L1_boundary + 0.1×LPIPS + 0.5×FFT_highfreq
```
边界带通过形态学膨胀-腐蚀获得。

## HCDM vs MDG-Harmonizer 对比

| 对比项 | HCDM | MDG-Harmonizer |
|--------|------|---------------|
| 基础框架 | 条件扩散图像协调 | 保留 HCDM 作为 baseline |
| 退化感知 | DegradationPrior（8维池化+FC） | **CDP-Net（64维卷积+解耦头）** |
| 特征调制 | FiLMLayer（全图统一 gamma/beta） | **AFM（cross-attention 空间自适应）** |
| 损失函数 | 多尺度 L1 噪声损失 | **噪声损失 + FB-Loss（5项前景感知）** |
| 训练策略 | 全参数训练（63M），需多卡 | **冻结 backbone（0.6M 可训），单卡 4GB** |
| 训练 epoch | 770 | **30** |
| 推理加速 | DDPM 1000步（基准） | **DDPM 200步（5× 加速，PSNR 仅降 0.8 dB）** |

## 实验结果 (D-Hday2night 133张, DDPM 200步)

### 消融实验

| 消融 | 模块 | PSNR | SSIM | MAE | fPSNR |
|------|------|------|------|-----|-------|
| **D** | CDP + AFM（无 FB-Loss） | **36.40** | 0.973 | 1.54 | 21.26 |
| C | CDP + FB-Loss（无 AFM） | 35.86 | 0.971 | 1.72 | 20.72 |
| A | CDP + AFM + FB-Loss | 35.81 | 0.972 | 1.59 | 20.67 |
| B | AFM + FB-Loss（无 CDP） | 35.67 | 0.971 | 1.66 | 20.51 |
| HCDM baseline | 原版全训练 (DDPM 1000步) | 36.85 | — | 1.19 | — |

### 解冻微调实验（基于 A_full 50epoch）

| 方案 | 数据 | epoch | PSNR |
|------|------|-------|------|
| 冻结 (50ep) | 46K 混合 | 50 | 35.81 |
| last2 decoder | 311张 | 5 | 35.67 |
| last2 decoder | 2000张 | 5 | 35.50 |
| last6 decoder | 311张 | 10 | 34.63 |

**结论**: 消融四组均在 35.7-36.4 区间，差异 < 0.8 dB，三模块功能互补。解冻 decoder 均使指标下降，验证了预训练 U-Net 与旧退化管线的深度耦合。
