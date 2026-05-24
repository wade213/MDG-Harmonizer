"""Mask-aware Degradation Descriptor — 掩码感知退化描述子。

从 composite image + mask 计算 8 个可解释的退化描述子：
    亮度、色偏、饱和度、对比度、边界伪影、模糊、噪声、纹理

不引入可训练参数，纯统计量。可直接用于 Prompt Router，也可在系统界面展示。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DegradationDescriptor(nn.Module):
    """计算 8 维退化描述子向量。

    Args:
        eps: 数值稳定性 epsilon。
    """

    VALID_KEYS = (
        "brightness",
        "color_shift",
        "saturation",
        "contrast",
        "boundary",
        "blur",
        "noise",
        "texture",
    )

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)

    def _masked_mean_std(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """mask 区域内的均值和标准差 per-channel。"""
        m = mask.float()
        area = m.sum(dim=(2, 3), keepdim=True) + self.eps
        mean = (x * m).sum(dim=(2, 3), keepdim=True) / area
        sq = ((x - mean) ** 2) * m
        std = (sq.sum(dim=(2, 3), keepdim=True) / area).sqrt()
        return mean, std

    def _boundary_mask(self, mask: torch.Tensor, kernel: int = 7) -> torch.Tensor:
        k, pad = kernel, kernel // 2
        dilated = F.max_pool2d(mask, k, 1, pad)
        eroded = -F.max_pool2d(-mask, k, 1, pad)
        return (dilated - eroded).clamp(0, 1)

    def _bg_ring_mask(self, mask: torch.Tensor, ring_width: int = 8) -> torch.Tensor:
        k = ring_width * 2 + 1
        pad = ring_width
        dilated = F.max_pool2d(mask, k, 1, pad)
        return (dilated - mask).clamp(0, 1)

    def _laplacian_var(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Laplacian 方差，衡量模糊程度。"""
        kernel = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=x.dtype, device=x.device)
        lap = F.conv2d(x, kernel.expand(x.size(1), 1, 3, 3), padding=1, groups=x.size(1))
        _, std = self._masked_mean_std(lap, mask)
        return std.mean(dim=(1, 2, 3))

    def _high_freq_energy(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """高频能量比例，衡量噪声/纹理。"""
        b, c, h, w = x.shape
        fft = torch.fft.rfft2(x.float(), norm="ortho")
        mag = fft.abs()
        rh, rw = max(1, int(h * 0.125)), max(1, int(w * 0.125))
        lo_mask = torch.zeros_like(mag)
        lo_mask[:, :, :rh, :rw] = 1.0
        lo_mask[:, :, -rh:, :rw] = 1.0
        hi_energy = (mag * (1 - lo_mask)).sum(dim=(1, 2, 3))
        total = mag.sum(dim=(1, 2, 3)) + self.eps
        return hi_energy / total

    def forward(self, y_cond: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """计算 8 维退化描述子。

        Args:
            y_cond: (B, 3, H, W), 合成图，值域 [-1, 1]。
            mask:   (B, 1, H, W)，前景掩码 [0, 1]。

        Returns:
            descriptor: (B, 8)，归一化到 [0, 1] 的描述子。
            desc_dict:  {key: (B,) tensor}，各自的本征单位。
        """
        B = y_cond.shape[0]
        mask_bin = (mask > 0.5).float()

        fg_mask = mask_bin
        bg_mask = self._bg_ring_mask(mask_bin)
        bd_mask = self._boundary_mask(mask_bin)

        # Region statistics
        fg_mean, fg_std = self._masked_mean_std(y_cond, fg_mask)     # (B,3,1,1)
        bg_mean, bg_std = self._masked_mean_std(y_cond, bg_mask)

        desc: dict[str, torch.Tensor] = {}

        # 1. Brightness mismatch: |FG - BG| luminance diff
        lum_fg = fg_mean.mean(dim=1)  # (B, 1)
        lum_bg = bg_mean.mean(dim=1)
        desc["brightness"] = (lum_fg - lum_bg).abs().view(B) / (lum_bg.abs() + self.eps).view(B)

        # 2. Color shift: channel-wise mean difference
        desc["color_shift"] = (fg_mean - bg_mean).abs().mean(dim=(1, 2, 3))

        # 3. Saturation mismatch: std diff
        desc["saturation"] = (fg_std - bg_std).abs().mean(dim=(1, 2, 3))

        # 4. Contrast mismatch: std ratio
        fg_std_mean = fg_std.mean(dim=(1, 2, 3))
        bg_std_mean = bg_std.mean(dim=(1, 2, 3))
        desc["contrast"] = ((fg_std_mean - bg_std_mean) / (bg_std_mean + self.eps)).abs()

        # 5. Boundary artifact: gradient magnitude at boundary
        grad_h = (y_cond[:, :, :, 2:] - y_cond[:, :, :, :-2]).abs().mean(dim=1, keepdim=True)
        grad_h = F.pad(grad_h, (1, 1, 0, 0))  # pad W dim
        grad_v = (y_cond[:, :, 2:, :] - y_cond[:, :, :-2, :]).abs().mean(dim=1, keepdim=True)
        grad_v = F.pad(grad_v, (0, 0, 1, 1))  # pad H dim
        grad = grad_h + grad_v
        bd_grad, _ = self._masked_mean_std(grad, bd_mask)
        bg_grad, _ = self._masked_mean_std(grad, bg_mask)
        desc["boundary"] = (bd_grad - bg_grad).abs().mean(dim=(1, 2, 3))

        # 6. Blur mismatch: Laplacian var diff
        desc["blur"] = self._laplacian_var(y_cond, fg_mask) - self._laplacian_var(y_cond, bg_mask)
        desc["blur"] = desc["blur"].abs()

        # 7. Noise mismatch: high-freq energy diff
        desc["noise"] = self._high_freq_energy(y_cond, fg_mask) - self._high_freq_energy(y_cond, bg_mask)
        desc["noise"] = desc["noise"].abs()

        # 8. Texture: combined high-freq + std
        desc["texture"] = (fg_std.mean(dim=(1, 2, 3)) * self._high_freq_energy(y_cond, fg_mask)).abs()

        # Stack and normalize to [0, 1]
        raw = torch.stack([desc[k] for k in self.VALID_KEYS], dim=1)  # (B, 8)
        # Soft normalization: tanh to clip extreme values
        descriptor = torch.tanh(raw)

        return descriptor, desc


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    desc_module = DegradationDescriptor()
    desc_module.eval()

    x = torch.randn(4, 3, 256, 256).clamp(-1, 1)
    m = (torch.rand(4, 1, 256, 256) > 0.3).float()

    descriptor, detail = desc_module(x, m)
    print(f"descriptor: {tuple(descriptor.shape)} (B, 8)")
    print(f"  range: [{descriptor.min():.4f}, {descriptor.max():.4f}]")
    print(f"  NaN: {torch.isnan(descriptor).any()}")
    print()
    for k, v in detail.items():
        v_mean = v.mean().item()
        print(f"  {k:>15s}: {v_mean:.4f}")
    print("Passed.")
