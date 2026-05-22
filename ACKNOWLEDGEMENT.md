# Acknowledgement

本项目基于 **HCDM (Image Harmonization in Complex Degradation Scenes)** 开源项目进行二次开发。

HCDM 提供了退化场景图像协调的扩散模型 baseline，包括：
- 条件扩散 U-Net 网络架构（guided-diffusion）
- 退化先验编码模块（DegradationPrior + FiLMLayer）
- 多尺度噪声损失训练框架
- D-iHarmony4 数据集及数据加载管线
- 预训练模型权重（1D_embed/770 epoch）

本文主要工作是在 HCDM 基础上设计并实现以下三个改进模块：
1. **CDP-Net** — 替换原版 DegradationPrior，提升退化编码精细度
2. **AFM** — 替换原版 FiLMLayer，实现空间自适应特征调制
3. **FB-Loss** — 新增前景-边界感知损失函数

并通过对比实验和四组消融实验（A/B/C/D）验证各模块的作用。

在论文和答辩材料中，HCDM 将作为本文的 **基线方法（baseline）** 进行明确引用。所有 HCDM 原始文件均在代码注释和 `CONTRIBUTION.md` 中标注。

**HCDM 论文引用**: *Image Harmonization in Complex Degradation Scenes*
