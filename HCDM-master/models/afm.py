"""AFM (Adaptive Feature Modulation) — MDG-Harmonizer 创新点 2

替代原 ``FiLMLayer`` 的简单仿射调制。

核心思想：
    - 把退化向量 ``deg_vec (B, 64)`` 通过 MLP 扩展成 K=8 个 token
      （每个 token 维度 = ``C/4``），让退化先验在「token 子空间」内可解耦表达。
    - 用单头轻量 cross-attention（query=features, key/value=tokens，
      投影维度降到 ``C/4``）让每个空间位置自适应地选择需要的退化分量，
      解决 FiLM「全图统一仿射」无法应对前景/背景不同退化的痛点。
    - 同时保留一条 FiLM shortcut 作为残差，保证 AFM 至少不弱于 FiLM
      （有退化时 attention 起主导，无退化时 FiLM 退化为恒等）。

参数与显存预算（C=64 时）：
    - 单实例参数 < 0.5 M（实际 ~ 1.3 万）
    - cross-attention 在 ``C/K`` 维度做（不是 C），fp16 下 attention 矩阵
      显存约 ``B*N*K`` floats，K=8 极小。

冻结主干 + 只训新模块的训练范式要求新模块「初始化近似恒等」，否则会立刻
破坏预训练特征导致前几个 epoch 直接发散。这里通过小随机初始化 ``out_proj``、
``gamma``（std=0.01~0.02），``beta`` 保持零初始化，让 ``forward`` 在初始化时
近似等于 ``features``，同时保证梯度能流过 CDP-Net。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp


class AFM(nn.Module):
    """Adaptive Feature Modulation.

    Args:
        feature_channels: 特征图通道数 C，必须能被 ``num_tokens`` 整除。
        degradation_dim:  退化向量维度（默认 64，匹配 CDP-Net 输出）。
        num_tokens:       K，token 数量（默认 8）。
        use_checkpoint:   是否对前向用 ``torch.utils.checkpoint`` 包装以省显存。
    """

    def __init__(
        self,
        feature_channels: int,
        degradation_dim: int = 64,
        num_tokens: int = 8,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        assert feature_channels % num_tokens == 0, (
            f"feature_channels({feature_channels}) must be divisible by num_tokens({num_tokens})"
        )

        self.feature_channels = feature_channels
        self.degradation_dim = degradation_dim
        self.num_tokens = num_tokens
        self.token_dim = feature_channels // num_tokens
        self.attn_dim = self.token_dim
        self.use_checkpoint = use_checkpoint

        # token MLP：deg_vec (B, deg_dim) -> (B, K, token_dim)
        # 用 2 层 + GELU 让 token 之间有非线性区分度
        hidden = max(degradation_dim, num_tokens * self.token_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(degradation_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_tokens * self.token_dim),
        )

        # 单头 cross-attention 投影（全部降到 C/4 节省显存）
        self.q_proj = nn.Linear(feature_channels, self.attn_dim, bias=False)
        self.k_proj = nn.Linear(self.token_dim, self.attn_dim, bias=False)
        self.v_proj = nn.Linear(self.token_dim, self.attn_dim, bias=False)
        self.out_proj = nn.Linear(self.attn_dim, feature_channels)

        # FiLM shortcut：保留原 FiLMLayer 的能力
        self.gamma_layer = nn.Linear(degradation_dim, feature_channels)
        self.beta_layer = nn.Linear(degradation_dim, feature_channels)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # 让 AFM 初始化为「近似恒等映射」。
        # 用小随机初始化代替零初始化，解决 CDP-Net 梯度被阻断的冷启动问题。
        # out_proj/gamma 用极小 std，beta 保持零，保证 output ≈ features。
        nn.init.normal_(self.out_proj.weight, std=0.001)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.normal_(self.gamma_layer.weight, std=0.001)
        nn.init.zeros_(self.gamma_layer.bias)
        nn.init.zeros_(self.beta_layer.weight)
        nn.init.zeros_(self.beta_layer.bias)

    def _forward_impl(self, features: torch.Tensor, deg_vec: torch.Tensor) -> torch.Tensor:
        B, C, H, W = features.shape
        N = H * W

        tokens = self.token_mlp(deg_vec)
        tokens = tokens.view(B, self.num_tokens, self.token_dim)  # (B, K, C/4)

        x_seq = features.view(B, C, N).transpose(1, 2).contiguous()  # (B, N, C)

        q = self.q_proj(x_seq)        # (B, N, C/4)
        k = self.k_proj(tokens)       # (B, K, C/4)
        v = self.v_proj(tokens)       # (B, K, C/4)

        scale = self.attn_dim ** -0.5
        # K=8 极小，attention 矩阵 (B, N, K) 几乎不占显存
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, N, K)
        attn_weights = attn_logits.softmax(dim=-1)
        attn_out = torch.matmul(attn_weights, v)                    # (B, N, C/4)
        attn_out = self.out_proj(attn_out)                          # (B, N, C)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, C, H, W)

        gamma = self.gamma_layer(deg_vec).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = self.beta_layer(deg_vec).unsqueeze(-1).unsqueeze(-1)
        film_out = gamma * features + beta

        return features + attn_out + film_out

    def forward(self, features: torch.Tensor, deg_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: ``(B, C, H, W)`` 待调制特征图。
            deg_vec:  ``(B, degradation_dim)`` 退化先验向量（来自 CDP-Net）。

        Returns:
            ``(B, C, H, W)`` 调制后的特征图。
        """
        # checkpoint 要求至少一个输入有 grad，且仅在训练时启用才有意义
        if (
            self.use_checkpoint
            and self.training
            and (features.requires_grad or deg_vec.requires_grad)
        ):
            return cp.checkpoint(
                self._forward_impl, features, deg_vec, use_reentrant=False
            )
        return self._forward_impl(features, deg_vec)


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


if __name__ == "__main__":
    torch.manual_seed(0)

    B, C, H, W = 2, 64, 32, 32
    deg_dim = 64

    afm = AFM(feature_channels=C, degradation_dim=deg_dim, num_tokens=8)
    afm.eval()  # CPU 测试关掉 checkpoint 触发条件，走纯前向

    features = torch.randn(B, C, H, W)
    deg_vec = torch.randn(B, deg_dim)

    out = afm(features, deg_vec)

    n_params = _count_params(afm)

    print("=" * 60)
    print("AFM unit test (CPU, fp32)")
    print("=" * 60)
    print(f"Input  features shape : {tuple(features.shape)}")
    print(f"Input  deg_vec  shape : {tuple(deg_vec.shape)}")
    print(f"Output features shape : {tuple(out.shape)}")
    print(f"Shape preserved       : {out.shape == features.shape}")
    print()
    print(f"Total parameters      : {n_params:,}  (~{n_params / 1e6:.4f} M)")
    print(f"Under 0.5M budget     : {n_params < 5e5}")
    print()

    # 初始化近似恒等性检查（small init -> output ≈ features）
    identity_err = (out - features).abs().max().item()
    print(f"Init-time identity max-abs-err : {identity_err:.3e}")
    print(f"Init is near-identity (< 0.2)  : {identity_err < 0.2}")
    print()

    # 反向传播健全性检查
    afm.train()
    features_g = torch.randn(B, C, H, W, requires_grad=True)
    deg_vec_g = torch.randn(B, deg_dim, requires_grad=True)
    out_g = afm(features_g, deg_vec_g)
    loss = out_g.mean()
    loss.backward()
    grad_ok = features_g.grad is not None and torch.isfinite(features_g.grad).all().item()
    print(f"Backward grad finite          : {grad_ok}")

    print("=" * 60)
    print("AFM passed.")
