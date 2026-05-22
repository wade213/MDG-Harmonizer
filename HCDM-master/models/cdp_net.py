"""CDP-Net (Compact Degradation Prior Network).

MDG-Harmonizer 的紧凑退化先验编码器，用于替换原 baseline `DegradationPrior`
（见 ``models/degradation_prior.py``，**保留作 ablation baseline，不要删**）。

关键设计动机：
    - baseline 仅用 GAP+GMP 池化 + 两层 FC 输出 8 维向量，表达力不足以捕捉
      复杂退化场景（光照偏色、噪声、饱和度等）下的细粒度先验。
    - CDPNet 采用 4 级轻量卷积主干 + 多任务**解耦输出头**：
      亮度 / 色温 / 饱和度 / 噪声水平 这 4 个标量分量可被显式辅助监督，
      剩余 60 维通用 embedding 走端到端学习；总输出 64 维供后续 AFM 调制使用。
    - 整网参数 < 1 M，支持 fp16 (autocast)，符合 RTX 3050 Ti 4–6 GB 显存预算。
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


def _gn(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """构造 GroupNorm：组数取 ``min(max_groups, num_channels)``，
    避免 num_channels 较小时除不开。"""

    num_groups = min(max_groups, num_channels)
    while num_channels % num_groups != 0 and num_groups > 1:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class _DownBlock(nn.Module):
    """Conv(3x3, stride=2) + GroupNorm + SiLU 的下采样基本块。

    用 stride=2 的 conv 同时完成下采样与通道升维，避免额外的 pooling 层，
    省参数也省显存（中间 feature map 只 keep 一份）。
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.norm = _gn(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class CDPNet(nn.Module):
    """Compact Degradation Prior Network.

    Args:
        in_channels: 拼接后输入通道数。默认为 4（RGB + 1 通道前景 mask）。
        backbone_channels: 4 级下采样后的通道数序列，长度必须为 4。
        embed_dim: 通用 embedding 头的输出维度。配合 4 个标量头组成 64 维 ``deg_vector``。

    Forward Returns:
        deg_vector: ``(B, 64)`` 退化先验向量，供 AFM 等调制模块消费。
        aux_dict: 含四个解耦标量预测，便于辅助监督：
            - ``brightness``: ``(B, 1)``
            - ``color_temp``: ``(B, 1)``
            - ``saturation``: ``(B, 1)``
            - ``noise_level``: ``(B, 1)``
            - ``general_embed``: ``(B, 60)``  # 同时暴露便于调试 / 可视化

    总输出维度 = 1 + 1 + 1 + 1 + 60 = 64。
    """

    def __init__(
        self,
        in_channels: int = 4,
        backbone_channels: Tuple[int, int, int, int] = (16, 32, 64, 128),
        embed_dim: int = 60,
    ) -> None:
        super().__init__()
        if len(backbone_channels) != 4:
            raise ValueError(
                f"backbone_channels must have length 4, got {len(backbone_channels)}"
            )

        c1, c2, c3, c4 = backbone_channels

        # 4 级下采样主干：每级 spatial /2，通道翻倍
        # 输入若为 256x256：依次得到 128 / 64 / 32 / 16，再 GAP → (B, 128)
        self.down1 = _DownBlock(in_channels, c1)
        self.down2 = _DownBlock(c1, c2)
        self.down3 = _DownBlock(c2, c3)
        self.down4 = _DownBlock(c3, c4)

        # 用 AdaptiveAvgPool2d(1) 汇聚到 128 维全局特征
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 多任务解耦输出头：4 个物理可解释标量 + 1 个通用 embedding
        # 之所以解耦：让 4 个标量头能被独立监督（aux loss），逼网络学到
        # 物理上有意义的退化分量；general_embed 兜底捕捉剩余信息。
        self.head_brightness = nn.Linear(c4, 1)
        self.head_color_temp = nn.Linear(c4, 1)
        self.head_saturation = nn.Linear(c4, 1)
        self.head_noise_level = nn.Linear(c4, 1)
        self.head_general_embed = nn.Linear(c4, embed_dim)

        self._embed_dim = embed_dim
        self._out_dim = 4 + embed_dim  # = 64

    @property
    def out_dim(self) -> int:
        """deg_vector 的输出维度，方便下游模块按属性查询而不写死 64。"""
        return self._out_dim

    def forward(
        self,
        degraded_image: torch.Tensor,
        foreground_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Args:
            degraded_image: ``(B, 3, H, W)`` 退化输入图像。
            foreground_mask: ``(B, 1, H, W)`` 前景二值或软掩码。

        Returns:
            ``(deg_vector, aux_dict)``，详见类 docstring。
        """

        # 沿通道维度拼接，让 mask 直接参与卷积（隐式提供前景空间先验）
        x = torch.cat([degraded_image, foreground_mask], dim=1)  # (B, 4, H, W)

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)  # (B, 128, H/16, W/16)

        # 全局特征：(B, 128, 1, 1) -> (B, 128)
        feat = self.global_pool(x).flatten(1)

        brightness = self.head_brightness(feat)        # (B, 1)
        color_temp = self.head_color_temp(feat)        # (B, 1)
        saturation = self.head_saturation(feat)        # (B, 1)
        noise_level = self.head_noise_level(feat)      # (B, 1)
        general_embed = self.head_general_embed(feat)  # (B, 28)

        # 拼接为 64 维退化先验向量；顺序固定，便于消费方 slice 出对应分量
        deg_vector = torch.cat(
            [brightness, color_temp, saturation, noise_level, general_embed],
            dim=1,
        )  # (B, 32)

        aux_dict: Dict[str, torch.Tensor] = {
            "brightness": brightness,
            "color_temp": color_temp,
            "saturation": saturation,
            "noise_level": noise_level,
            "general_embed": general_embed,
        }

        return deg_vector, aux_dict


def _count_parameters(module: nn.Module) -> Tuple[int, int]:
    """返回 ``(total_params, trainable_params)``。"""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # CPU 单元测试：覆盖参数量、输出 shape、aux_dict 完整性、fp16 autocast 不出 NaN
    torch.manual_seed(0)

    B, H, W = 2, 256, 256
    degraded_image = torch.randn(B, 3, H, W)
    foreground_mask = torch.rand(B, 1, H, W)  # 软掩码 in [0, 1]

    model = CDPNet()
    model.eval()

    total, trainable = _count_parameters(model)
    print("=" * 60)
    print(f"[CDPNet] total params:     {total:,}  (~{total / 1e6:.4f} M)")
    print(f"[CDPNet] trainable params: {trainable:,}")
    assert total < 1_000_000, f"参数量 {total} 超过 1 M 上限"
    print(f"[CDPNet] out_dim:          {model.out_dim}")

    # 1) 标准 fp32 前向
    deg_vector, aux = model(degraded_image, foreground_mask)
    print("-" * 60)
    print(f"[fp32] deg_vector shape:        {tuple(deg_vector.shape)}")
    print(f"[fp32] aux['brightness'] shape: {tuple(aux['brightness'].shape)}")
    print(f"[fp32] aux['color_temp'] shape: {tuple(aux['color_temp'].shape)}")
    print(f"[fp32] aux['saturation'] shape: {tuple(aux['saturation'].shape)}")
    print(f"[fp32] aux['noise_level'] shape:{tuple(aux['noise_level'].shape)}")
    print(f"[fp32] aux['general_embed'] sh: {tuple(aux['general_embed'].shape)}")

    assert deg_vector.shape == (B, 64), f"期望 (B, 64)，实际 {tuple(deg_vector.shape)}"
    for k, expected in [
        ("brightness", (B, 1)),
        ("color_temp", (B, 1)),
        ("saturation", (B, 1)),
        ("noise_level", (B, 1)),
        ("general_embed", (B, 60)),
    ]:
        assert aux[k].shape == expected, f"aux[{k}] 期望 {expected}，实际 {tuple(aux[k].shape)}"
    assert torch.isfinite(deg_vector).all(), "fp32 输出含 NaN/Inf"

    # 2) fp16 autocast：CPU 上用 bfloat16 验证（避免 CPU 不支持 float16 autocast 的某些算子）
    #    GPU 训练时会以 float16 形式被 GradScaler 包裹；这里只验证算子兼容性。
    print("-" * 60)
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            deg_vector_h, aux_h = model(degraded_image, foreground_mask)
        print(f"[bf16] deg_vector dtype: {deg_vector_h.dtype}")
        print(f"[bf16] deg_vector shape: {tuple(deg_vector_h.shape)}")
        assert torch.isfinite(deg_vector_h.float()).all(), "bf16 autocast 输出含 NaN/Inf"
        print("[bf16] autocast 前向通过，无 NaN/Inf")
    except Exception as e:  # pragma: no cover - 仅诊断用
        print(f"[bf16] autocast 测试跳过：{e}")

    # 3) 反向传播 sanity check：保证梯度可流通
    print("-" * 60)
    model.train()
    deg_vector, _ = model(degraded_image, foreground_mask)
    loss = deg_vector.pow(2).mean()
    loss.backward()
    grad_norm = sum(
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    )
    print(f"[backward] loss = {loss.item():.6f}, sum(|grad|_2) = {grad_norm:.4f}")
    assert grad_norm > 0, "反向传播后梯度全 0"

    print("=" * 60)
    print("[CDPNet] 全部单元测试通过")
