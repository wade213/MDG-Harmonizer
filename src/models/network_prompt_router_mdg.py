"""M-DPR Prompt Router MDG Network — 掩码感知退化提示路由网络。

工作二核心网络。支持三种模式：
    descriptor_only: 独立工作，不用 CDP-Net（工作二独立版本）
    cdp_only:        只用 CDP-Net（工作一版本）
    hybrid:          CDP-Net + Prompt Router 融合（最终系统版本）

完全继承 MDGNetwork，仅覆写 ``_compute_deg_vec`` 和 ``_freeze_backbone``。
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .network_mdg import MDGNetwork


class PromptRouterMDGNetwork(MDGNetwork):
    """M-DPR 退化提示路由网络。

    新增参数:
        prompt_mode:    "descriptor_only" | "cdp_only" | "hybrid"
        prompt_dim:     提示向量维度（默认 64）
        router_hidden:  Router MLP 隐藏层维度
        use_mlp_router: 是否用 MLP router（False = 规则路由）
        prompt_alpha:   hybrid 模式下 prompt 的融合比例初始值
    """

    def __init__(
        self,
        *args,
        prompt_mode: str = "hybrid",
        prompt_dim: int = 64,
        router_hidden: int = 32,
        use_mlp_router: bool = True,
        prompt_alpha: float = 0.1,
        **kwargs,
    ) -> None:
        self._prompt_mode = prompt_mode
        self._prompt_dim = prompt_dim
        self._router_hidden = router_hidden
        self._use_mlp_router = use_mlp_router
        self._prompt_alpha_val = prompt_alpha

        # Placeholder for prompt_router (created after parent init)
        self.prompt_router = None

        super().__init__(*args, **kwargs)

        # Create prompt router after parent has set up everything
        if self._prompt_mode != "cdp_only":
            from .degradation_prompt import MaskAwarePromptRouter

            self.prompt_router = MaskAwarePromptRouter(
                prompt_dim=prompt_dim,
                router_hidden=router_hidden,
                use_mlp_router=use_mlp_router,
            )

        # Learnable fusion weight (hybrid mode only)
        if self._prompt_mode == "hybrid":
            self.prompt_alpha = nn.Parameter(torch.tensor(prompt_alpha))

    # ------------------------------------------------------------------
    # 退化先验提取：三种模式
    # ------------------------------------------------------------------
    def _compute_deg_vec(self, y_cond, mask):
        d_cdp = super()._compute_deg_vec(y_cond, mask)

        if self._prompt_mode == "cdp_only":
            self._last_prompt_aux = {}
            return d_cdp

        # Compute prompt router output
        if self.prompt_router is not None and mask is not None:
            d_prompt, aux = self.prompt_router(y_cond, mask)
            self._last_prompt_aux = aux
        else:
            d_prompt = d_cdp.new_zeros(d_cdp.shape[0], self._prompt_dim) if d_cdp is not None else None
            self._last_prompt_aux = {}

        if d_cdp is None:
            return d_prompt

        if self._prompt_mode == "descriptor_only":
            return d_prompt
        elif self._prompt_mode == "hybrid":
            alpha = torch.tanh(self.prompt_alpha)
            return d_cdp + alpha * d_prompt
        else:
            return d_cdp

    # ------------------------------------------------------------------
    # 冻结策略：freeze_backbone=True 时冻结 CDP，只训 prompt_router
    # ------------------------------------------------------------------
    def _freeze_backbone(self, unfreeze_decoder_last_n: int = 0) -> None:
        super()._freeze_backbone(unfreeze_decoder_last_n)

        # CDP-Net 冻结
        cdp = getattr(self, "cdp_net", None)
        if cdp is not None:
            for p in cdp.parameters():
                p.requires_grad = False

        # Prompt router 始终可训
        pr = getattr(self, "prompt_router", None)
        if pr is not None:
            for p in pr.parameters():
                p.requires_grad = True

        # 融合系数可训
        alpha = getattr(self, "prompt_alpha", None)
        if alpha is not None:
            alpha.requires_grad = True


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)

    H = 32
    unet_kwargs = dict(
        image_size=H, in_channel=6, inner_channel=64, out_channel=3,
        res_blocks=2, attn_res=[16], channel_mults=(1, 2, 4, 8),
        use_checkpoint=False, use_fp16=False,
    )
    beta_schedule = {
        "train": {"schedule": "linear", "n_timestep": 8, "linear_start": 1e-6, "linear_end": 1e-2},
        "test":  {"schedule": "linear", "n_timestep": 8, "linear_start": 1e-6, "linear_end": 1e-2},
    }
    cdp_args = {"in_channels": 4, "backbone_channels": (16, 32, 64, 128), "embed_dim": 60}
    afm_args = {"num_tokens": 4, "use_checkpoint": False}
    fb_args = {"use_lpips": False, "weights": (1.0, 3.0, 5.0, 0.1, 0.5)}
    loss_w = {"noise": 1.0, "fb": 0.5}

    for mode in ["cdp_only", "descriptor_only", "hybrid"]:
        net = PromptRouterMDGNetwork(
            unet=unet_kwargs, beta_schedule=beta_schedule,
            cdp=cdp_args, afm=afm_args, fb_loss=fb_args,
            loss_weights=loss_w, freeze_backbone=True, deg_dim=64,
            prompt_mode=mode, prompt_dim=64,
        )
        net.set_loss(F.l1_loss)
        net.set_new_noise_schedule(device=torch.device("cpu"), phase="train")

        # Fix zero out-convs
        with torch.no_grad():
            for n, p in net.denoise_fn.named_parameters():
                if (n.startswith("out.") or n.startswith("out_list.")) and p.abs().sum() < 1e-6:
                    p.normal_(0, 0.01)

        x = torch.randn(1, 3, H, H)
        m = torch.rand(1, 1, H, H)
        loss = net(x, y_cond=x, mask=m)
        n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)

        print(f"[{mode:>16s}] loss={loss.item():.4f}, trainable={n_train:,}")
        if net._last_prompt_aux:
            w = net._last_prompt_aux.get("prompt_weights")
            if w is not None:
                print(f"  weights: {w.mean(dim=0).tolist()}")

    print("PromptRouterMDGNetwork passed.")
