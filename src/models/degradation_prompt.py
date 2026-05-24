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


# ---------------------------------------------------------------------------
# Mask-Aware Prompt Router — 基于图像退化描述子的提示路由
# ---------------------------------------------------------------------------

class MaskAwarePromptRouter(nn.Module):
    """掩码感知退化提示路由器（M-DPR：Mask-aware Degradation Prompt Routing）。

    工作流程:
        1. DegradationDescriptor 从 (image, mask) 提取 8 维描述子 r
        2. MLP Router 输出 8 个 prompt 的权重 w = softmax(MLP(r)/T)
        3. 8 个可学习 prompt prototype P ∈ R^{8×D} 加权求和 → p = wP
        4. 返回退化先验 p 及所有权重/描述子详情（供系统展示）

    不依赖 CDP-Net，可独立工作。

    Args:
        prompt_dim:         提示向量维度 D。
        router_hidden:      Router MLP 隐藏层维度。
        use_mlp_router:     False 时用规则路由（无参数），True 时用 MLP。
        temperature:        softmax 温度。
    """

    def __init__(
        self,
        prompt_dim: int = 64,
        router_hidden: int = 32,
        use_mlp_router: bool = True,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()

        self.prompt_dim = prompt_dim
        self.temperature = temperature
        self._use_mlp_router = use_mlp_router

        try:
            from .degradation_descriptor import DegradationDescriptor
        except ImportError:
            from degradation_descriptor import DegradationDescriptor  # type: ignore[no-redef]

        self._Descriptor = DegradationDescriptor

        self.descriptor = self._Descriptor()

        # 8 个退化原型 prompt
        self.prompts = nn.Parameter(torch.randn(8, prompt_dim) * 0.02)

        # MLP Router: 8维描述子 → 8维权重
        if use_mlp_router:
            self.router = nn.Sequential(
                nn.Linear(8, router_hidden),
                nn.GELU(),
                nn.Linear(router_hidden, 8),
            )

    @property
    def use_mlp(self) -> bool:
        return self._use_mlp_router

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self, y_cond: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """根据图像退化信息路由选择 prompt。

        Args:
            y_cond: 合成图 (B, 3, H, W)。
            mask:   前景掩码 (B, 1, H, W)。

        Returns:
            prompt_vec: 退化先验向量 (B, D)。
            aux: {
                "prompt_weights": (B, 8) 权重,
                "descriptor":     (B, 8) 退化描述子,
                "descriptor_labels": [str] 8 个标签,
            }
        """
        descriptor, _ = self.descriptor(y_cond, mask)  # (B, 8)

        if self._use_mlp_router:
            logits = self.router(descriptor) / self.temperature
            weights = torch.softmax(logits, dim=-1)  # (B, 8)
        else:
            # 规则路由：描述子归一化后直接 softmax
            weights = torch.softmax(descriptor / self.temperature, dim=-1)

        prompt_vec = weights @ self.prompts  # (B, D)

        aux = {
            "prompt_weights": weights,
            "descriptor": descriptor,
            "descriptor_labels": list(self._Descriptor.VALID_KEYS),
        }
        return prompt_vec, aux

    def extra_repr(self) -> str:
        return f"dim={self.prompt_dim}, mlp={self._use_mlp_router}, params={self.num_params:,}"


# ---------------------------------------------------------------------------
# Smoke test: MaskAwarePromptRouter
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # re-run full tests with the previous bank test
    torch.manual_seed(0)

    # Original prompt bank test
    print("=== DegradationPromptBank ===")
    bank = DegradationPromptBank(num_prompts=8, prompt_dim=64)
    bank.eval()
    x = torch.randn(4, 64)
    fused, aux = bank(x)
    print(f"  num_prompts: {bank.num_prompts}")
    print(f"  params: {sum(p.numel() for p in bank.parameters() if p.requires_grad):,}")
    print(f"  fused_deg: {tuple(fused.shape)}, has_nan: {torch.isnan(fused).any()}")

    print()
    print("=== MaskAwarePromptRouter ===")
    router = MaskAwarePromptRouter(prompt_dim=64, use_mlp_router=True)
    router.eval()

    img = torch.randn(4, 3, 256, 256).clamp(-1, 1)
    m = (torch.rand(4, 1, 256, 256) > 0.3).float()
    p_vec, aux = router(img, m)

    print(f"  params: {router.num_params:,}")
    print(f"  prompt_vec: {tuple(p_vec.shape)}, has_nan: {torch.isnan(p_vec).any()}")
    print(f"  weights: {tuple(aux['prompt_weights'].shape)}")
    print(f"  descriptor: {tuple(aux['descriptor'].shape)}")
    print(f"  labels: {aux['descriptor_labels']}")
    print("  weights mean per prompt:")
    for i, label in enumerate(aux["descriptor_labels"]):
        w_mean = aux["prompt_weights"][:, i].mean().item()
        print(f"    {label:>15s}: {w_mean:.4f}")

    # Rule-based router
    router_r = MaskAwarePromptRouter(prompt_dim=64, use_mlp_router=False)
    p_vec_r, aux_r = router_r(img, m)
    print(f"  rule-router weights: {aux_r['prompt_weights'].mean(dim=0).tolist()}")
    print("Passed.")
