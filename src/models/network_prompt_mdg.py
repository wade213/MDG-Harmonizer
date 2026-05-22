"""Prompt-MDG Network：在 MDGNetwork 基础上加入退化提示学习。

与 MDGNetwork 的区别仅在于 ``_compute_deg_vec``:
    MDG:  CDP-Net → deg_vec
    PromptMDG: CDP-Net → deg_vec → Prompt Bank → fused_deg_vec

其余逻辑（denoise_fn, AFM, FB-Loss, restoration, training forward）完全继承。

用法：
    训练时加载 MDG checkpoint 初始化 CDP/AFM/FB-Loss 权重，
    冻结 backbone+CDP，只训 Prompt Bank + AFM。
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

try:
    from .degradation_prompt import DegradationPromptBank
    from .network_mdg import MDGNetwork
except ImportError:
    from degradation_prompt import DegradationPromptBank
    from network_mdg import MDGNetwork


class PromptMDGNetwork(MDGNetwork):
    """MDGNetwork + 退化提示学习。

    新增参数：
        use_prompt:        bool，是否启用 Prompt Bank（消融开关）
        prompt_num:        int，K 个可学习退化提示
        prompt_dim:        int，提示维度（默认 = deg_dim）
        prompt_hidden:     int，Selector/Fusion 隐藏层维度
        prompt_temperature: float，softmax 温度
        prompt_uniform:    bool，True 时平均权重（消融用）
    """

    def __init__(
        self,
        unet: Dict,
        beta_schedule: Dict,
        cdp: Optional[Dict] = None,
        afm: Optional[Dict] = None,
        fb_loss: Optional[Dict] = None,
        loss_weights: Optional[Dict] = None,
        freeze_backbone: bool = False,
        unfreeze_decoder_last_n: int = 0,
        deg_dim: int = 64,
        cdp_zero_vec: bool = False,
        disable_afm: bool = False,
        # ---- Prompt 参数 ----
        use_prompt: bool = True,
        prompt_num: int = 8,
        prompt_dim: int = 64,
        prompt_hidden: int = 128,
        prompt_temperature: float = 1.0,
        prompt_uniform: bool = False,
        **kwargs,
    ) -> None:
        self._use_prompt = bool(use_prompt)
        self.prompt_bank = None  # 在父类 __init__ 之前占位，_freeze_backbone 会检查

        super().__init__(
            unet=unet,
            beta_schedule=beta_schedule,
            cdp=cdp,
            afm=afm,
            fb_loss=fb_loss,
            loss_weights=loss_weights,
            freeze_backbone=freeze_backbone,
            unfreeze_decoder_last_n=unfreeze_decoder_last_n,
            deg_dim=deg_dim,
            cdp_zero_vec=cdp_zero_vec,
            disable_afm=disable_afm,
            **kwargs,
        )

        # Prompt Bank（在父类 CDP-Net 之后创建）
        self.prompt_bank = (
            DegradationPromptBank(
                num_prompts=prompt_num,
                prompt_dim=prompt_dim,
                hidden_dim=prompt_hidden,
                temperature=prompt_temperature,
                uniform=prompt_uniform,
            )
            if self._use_prompt
            else None
        )

    # ------------------------------------------------------------------
    # 退化先验提取：CDP → Prompt Bank
    # ------------------------------------------------------------------
    def _compute_deg_vec(self, y_cond, mask):
        deg_vec = super()._compute_deg_vec(y_cond, mask)

        if deg_vec is not None and self.prompt_bank is not None:
            fused, aux = self.prompt_bank(deg_vec)
            self._last_prompt_aux = aux
            return fused

        self._last_prompt_aux = {}
        return deg_vec

    # ------------------------------------------------------------------
    # 冻结策略：Prompt Bank 始终可训
    # ------------------------------------------------------------------
    def _freeze_backbone(self, unfreeze_decoder_last_n: int = 0) -> None:
        super()._freeze_backbone(unfreeze_decoder_last_n)

        # Prompt Bank 可训练（init 时可能尚未创建，用 getattr）
        bank = getattr(self, "prompt_bank", None)
        if bank is not None:
            for p in bank.parameters():
                p.requires_grad = True


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)

    H = 32
    unet_kwargs = dict(
        image_size=H,
        in_channel=6,
        inner_channel=64,
        out_channel=3,
        res_blocks=2,
        attn_res=[16],
        channel_mults=(1, 2, 4, 8),
        use_checkpoint=False,
        use_fp16=False,
    )
    beta_schedule = {
        "train": {"schedule": "linear", "n_timestep": 8, "linear_start": 1e-6, "linear_end": 1e-2},
        "test":  {"schedule": "linear", "n_timestep": 8, "linear_start": 1e-6, "linear_end": 1e-2},
    }

    # With prompt
    net_prompt = PromptMDGNetwork(
        unet=unet_kwargs,
        beta_schedule=beta_schedule,
        cdp={"in_channels": 4, "backbone_channels": (16, 32, 64, 128), "embed_dim": 60},
        afm={"num_tokens": 4, "use_checkpoint": False},
        fb_loss={"use_lpips": False, "weights": (1.0, 3.0, 5.0, 0.1, 0.5)},
        loss_weights={"noise": 1.0, "fb": 0.5},
        freeze_backbone=True,
        deg_dim=64,
        use_prompt=True,
        prompt_num=8,
    )
    net_prompt.set_loss(F.l1_loss)
    net_prompt.set_new_noise_schedule(device=torch.device("cpu"), phase="train")

    # Fix zero out-convs for smoke test
    with torch.no_grad():
        for name, p in net_prompt.denoise_fn.named_parameters():
            if (name.startswith("out.") or name.startswith("out_list.")) and p.detach().abs().sum().item() < 1e-6:
                p.normal_(mean=0.0, std=0.01)

    B = 1
    y_0 = torch.randn(B, 3, H, H)
    y_cond = torch.randn(B, 3, H, H)
    mask = (torch.rand(B, 1, H, H) > 0.5).float()

    # Forward
    loss = net_prompt(y_0, y_cond=y_cond, mask=mask)
    print(f"Prompt-MDG forward loss: {loss.item():.4f}")
    print(f"Loss breakdown: {net_prompt._last_loss_breakdown}")
    print(f"Prompt aux: {list(net_prompt._last_prompt_aux.keys())}")

    # Restoration
    net_prompt.eval()
    with torch.no_grad():
        out, _ = net_prompt.restoration(y_cond=y_cond, y_0=y_0, mask=mask, sample_num=2)
    print(f"Output shape: {tuple(out.shape)}")

    # With uniform
    net_uniform = PromptMDGNetwork(
        unet=unet_kwargs,
        beta_schedule=beta_schedule,
        cdp={"in_channels": 4, "backbone_channels": (16, 32, 64, 128), "embed_dim": 60},
        afm={"num_tokens": 4, "use_checkpoint": False},
        fb_loss={"use_lpips": False, "weights": (1.0, 3.0, 5.0, 0.1, 0.5)},
        loss_weights={"noise": 1.0, "fb": 0.5},
        freeze_backbone=True,
        deg_dim=64,
        use_prompt=True,
        prompt_num=8,
        prompt_uniform=True,
    )
    net_uniform.set_loss(F.l1_loss)
    net_uniform.set_new_noise_schedule(device=torch.device("cpu"), phase="train")
    print(f"Uniform prompt bank: {net_uniform.prompt_bank.use_uniform}")

    n_total = sum(p.numel() for p in net_prompt.parameters())
    n_train = sum(p.numel() for p in net_prompt.parameters() if p.requires_grad)
    print(f"Params: total={n_total:,}, trainable={n_train:,} (~{n_train/n_total*100:.2f}%)")
    print("PromptMDGNetwork passed.")
