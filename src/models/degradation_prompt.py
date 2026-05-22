"""Degradation Prompt Bank — 可学习退化提示库。

与 CDP-Net 的区别：
    CDP-Net: 卷积编码器，从 (image, mask) 显式提取退化特征
    Prompt Bank: K 个可学习原型向量，Selector 根据 CDP 输出自适应加权

核心思路：CDP-Net 输出退化向量 d ∈ R^64，Prompt Selector 根据 d 为 K 个
可学习 prompt 分配权重，加权得到 prompt_vec，再与 d 融合后送入 AFM。
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DegradationPromptBank(nn.Module):
    """退化提示库。

    Args:
        num_prompts:  可学习退化提示数量 K
        prompt_dim:   每个提示的维度 D（须与 CDP-Net 输出维度一致）
        hidden_dim:   Selector 和 Fusion 的隐藏层维度
        temperature:  softmax 温度
        uniform:      True 时跳过 Selector，直接平均所有 prompt（消融用）
    """

    def __init__(
        self,
        num_prompts: int = 8,
        prompt_dim: int = 64,
        hidden_dim: int = 128,
        temperature: float = 1.0,
        uniform: bool = False,
    ) -> None:
        super().__init__()

        self.num_prompts = num_prompts
        self.prompt_dim = prompt_dim
        self.temperature = temperature
        self._uniform = uniform

        # K 个可学习退化提示
        self.prompts = nn.Parameter(torch.randn(num_prompts, prompt_dim) * 0.02)

        # Selector: deg_vec → prompt weights
        if not uniform:
            self.selector = nn.Sequential(
                nn.LayerNorm(prompt_dim),
                nn.Linear(prompt_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_prompts),
            )

        # Fusion: [deg_vec; prompt_vec] → delta
        self.fusion = nn.Sequential(
            nn.Linear(prompt_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, prompt_dim),
        )

        # 残差强度，初始很小避免破坏原 MDG 输出
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    @property
    def use_uniform(self) -> bool:
        return self._uniform

    def forward(self, deg_vec: torch.Tensor) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """根据退化向量选择并融合退化提示。

        Args:
            deg_vec: CDP-Net 输出 (B, D)

        Returns:
            fused_deg: 融合后的退化向量 (B, D)
            aux: {prompt_weights, prompt_vec} 供日志
        """
        B = deg_vec.shape[0]

        if self._uniform:
            # 消融：平均权重
            weights = torch.ones(B, self.num_prompts, device=deg_vec.device) / self.num_prompts
        else:
            logits = self.selector(deg_vec) / self.temperature
            weights = torch.softmax(logits, dim=-1)  # (B, K)

        prompt_vec = weights @ self.prompts  # (B, D)

        delta = self.fusion(torch.cat([deg_vec, prompt_vec], dim=-1))
        fused_deg = deg_vec + torch.tanh(self.res_scale) * delta

        aux = {
            "prompt_weights": weights,
            "prompt_vec": prompt_vec,
        }
        return fused_deg, aux

    def extra_repr(self) -> str:
        return f"num_prompts={self.num_prompts}, dim={self.prompt_dim}, uniform={self._uniform}"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    bank = DegradationPromptBank(num_prompts=8, prompt_dim=64)
    bank.eval()

    x = torch.randn(4, 64)
    fused, aux = bank(x)

    print(f"num_prompts: {bank.num_prompts}")
    print(f"trainable params: {sum(p.numel() for p in bank.parameters() if p.requires_grad):,}")
    print(f"deg_vec:     {tuple(x.shape)}")
    print(f"fused_deg:   {tuple(fused.shape)}")
    print(f"has_nan:     {torch.isnan(fused).any()}")
    print(f"weights:     {aux['prompt_weights'].shape}  mean={aux['prompt_weights'].mean():.3f}")

    # Uniform 消融
    bank_u = DegradationPromptBank(num_prompts=8, prompt_dim=64, uniform=True)
    fused_u, aux_u = bank_u(x)
    print(f"uniform weights: min={aux_u['prompt_weights'].min():.4f}  max={aux_u['prompt_weights'].max():.4f}")
    print("Passed.")
