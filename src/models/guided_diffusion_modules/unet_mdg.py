"""MDG U-Net: 在原 ``unet_modified.UNet`` 基础上注入 AFM 退化先验调制。

设计取舍：
    - 通过 *继承* 原 UNet 而非整文件 copy，避免 600+ 行重复代码与后续维护漂移；
      整个改造的核心只是 ``__init__`` 末尾构造一个 bottleneck AFM，再把
      ``forward`` 复制一份并在 ``middle_block`` 之后插入一行 AFM 调用。
    - 注入位置选 **bottleneck**（middle_block 之后）的原因：
        1. 该处空间最小、通道最大（C = inner_channel * channel_mults[-1]，
           在 HCDM 默认配置下是 64 * 8 = 512），所有编码器信息汇聚于此，
           AFM 一次注入可影响全部解码路径。
        2. 只放一个 AFM 实例，参数预算最省（~ 0.48 M），冻结主干后
           可训练参数总量仍 < 1 M，符合 MDG-Harmonizer 轻量化卖点。
        3. ``out_proj/gamma/beta`` 零初始化保证「初始即恒等」，加载预训练
           权重后立刻恢复 baseline 行为；之后训练才逐步学到调制信号。

参考：
    - 原始 forward 见 ``models/guided_diffusion_modules/unet_modified.py`` 中
      ``UNet.forward``（约 547-584 行）。
"""

from __future__ import annotations

from typing import List, Optional

import torch

from .nn import gamma_embedding
from .unet_modified import UNet as BaseUNet
from ..afm import AFM


class MDGUNet(BaseUNet):
    """MDG-Harmonizer 专用 U-Net，bottleneck 处接 AFM 退化先验调制。

    新增构造参数：
        deg_dim:   退化向量维度，必须与 ``CDPNet.out_dim`` 一致（默认 32）。
        afm_num_tokens: AFM 内部的 token 数 K。
        afm_use_checkpoint: AFM 是否启用 ``torch.utils.checkpoint`` 节省显存。

    其余参数完全透传给父类 ``UNet``，默认行为与 baseline 完全一致。
    """

    def __init__(
        self,
        *args,
        deg_dim: int = 32,
        afm_num_tokens: int = 8,
        afm_use_checkpoint: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # bottleneck 通道数：父类的 middle_block 沿用最后一级编码器输出，
        # 即 inner_channel * channel_mults[-1]。父类已在 self 上保存属性。
        bottleneck_ch = int(self.inner_channel * self.channel_mults[-1])
        if bottleneck_ch % afm_num_tokens != 0:
            raise ValueError(
                f"bottleneck channels ({bottleneck_ch}) must be divisible "
                f"by afm_num_tokens ({afm_num_tokens})"
            )

        self.afm_bottleneck = AFM(
            feature_channels=bottleneck_ch,
            degradation_dim=deg_dim,
            num_tokens=afm_num_tokens,
            use_checkpoint=afm_use_checkpoint,
        )
        self.deg_dim = deg_dim

    def forward(
        self,
        x: torch.Tensor,
        gammas: torch.Tensor,
        deg_vec: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """与父类 ``UNet.forward`` 保持完全一致的多尺度输出列表，
        额外接收一个可选 ``deg_vec``。

        Args:
            x:        ``(B, in_channel, H, W)`` 输入（cond + noisy 拼接）。
            gammas:   ``(B,)`` 或 ``(B, 1)`` 时间步 gamma 累积值。
            deg_vec:  ``(B, deg_dim)`` 退化先验向量。``None`` 时跳过 AFM
                      调制，等价于 baseline U-Net（用于 ablation）。

        Returns:
            一个长度为 4 的 list：``[res_lvl3, res_lvl2, res_lvl1, out]``，
            每项形状从 (B, out_channel, H/8, W/8) 递增到 (B, out_channel, H, W)，
            与 baseline ``UNet.forward`` 完全相同（多尺度 deep supervision）。
        """
        hs: List[torch.Tensor] = []
        gammas = gammas.view(-1)
        emb = self.cond_embed(gamma_embedding(gammas, self.inner_channel))

        output: List[torch.Tensor] = []

        h = x.type(torch.float32)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)

        h = self.middle_block(h, emb)

        # ================== MDG 关键改动：bottleneck AFM 注入 ==================
        # deg_vec=None 时跳过，行为与 baseline 完全一致（用于消融实验）。
        if deg_vec is not None:
            # 与 h 当前 dtype（fp32 in this codebase）保持一致，避免 autocast
            # 导致 AFM 内部 fp16 与外部 fp32 不匹配。
            h = self.afm_bottleneck(h, deg_vec.to(h.dtype))
        # =====================================================================

        output_count = 1
        out_indx = 0
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)

            if output_count < 10 and output_count % 3 == 2:
                h = h.type(x.dtype)
                res = self.out_list[out_indx](h)
                output.append(res)
                out_indx += 1
            output_count += 1

        h = h.type(x.dtype)
        output.append(self.out(h))
        return output


if __name__ == "__main__":
    # CPU 冒烟测试：构造一个迷你 MDGUNet，验证：
    #   1. 构造成功，AFM 正确接到 bottleneck
    #   2. 不传 deg_vec 时输出与 baseline UNet 一致（AFM 路径被跳过）
    #   3. 传 deg_vec 时输出 shape 仍正确（AFM 是 in-place 通道维度恒等）
    #   4. 因为 AFM 零初始化，传入 deg_vec 与不传应该输出**完全相同**
    import torch as _torch

    _torch.manual_seed(0)
    # 父类 nn.normalization 要求 num_channels 能被 32 整除（GroupNorm32），
    # 因此最浅层通道也须 ≥ 32；用 inner_channel=64 与正式 config 一致，
    # 缩小 image_size 让 CPU 测试在数秒内完成。
    B, H, W = 1, 32, 32
    in_ch, out_ch = 6, 3
    inner_ch = 64
    channel_mults = (1, 2, 4, 8)

    # 父类的 deep-supervision 输出位置硬编码为 ``output_count % 3 == 2``，
    # 隐含假设 res_blocks=2（每个 level 对应 3 个 output_block）。res_blocks=1
    # 会让通道数与 ``out_list`` 不匹配。这里保持 res_blocks=2 与正式 config 同。
    model = MDGUNet(
        image_size=H,
        in_channel=in_ch,
        inner_channel=inner_ch,
        out_channel=out_ch,
        res_blocks=2,
        attn_res=[16],  # 与正式 config 同
        channel_mults=channel_mults,
        deg_dim=64,
        afm_num_tokens=8,
        afm_use_checkpoint=False,
        use_checkpoint=False,
        use_fp16=False,
    )
    model.eval()

    x = _torch.randn(B, in_ch, H, W)
    gammas = _torch.rand(B)
    deg_vec = _torch.randn(B, 32)

    # 1) 不传 deg_vec：等价 baseline 路径
    out_baseline = model(x, gammas, deg_vec=None)
    # 2) 传 deg_vec（AFM 零初始化 -> 恒等）
    out_with_deg = model(x, gammas, deg_vec=deg_vec)

    print("=" * 60)
    print("MDGUNet smoke test (CPU, fp32)")
    print("=" * 60)
    print(f"Multi-scale outputs (should be 4): {len(out_baseline)}")
    for i, o in enumerate(out_baseline):
        print(f"  output[{i}] shape: {tuple(o.shape)}")

    # 由于 AFM 零初始化恒等，两路结果应当数值完全相同
    max_diff = max(
        (a - b).abs().max().item()
        for a, b in zip(out_baseline, out_with_deg)
    )
    print(f"\nAFM identity check: max-abs diff between deg=None and deg=vec: {max_diff:.3e}")
    print(f"  (must be ~ 0 because AFM zero-inits to identity at start)")
    assert max_diff < 1e-5, "AFM not identity at init -> 会破坏冻结主干训练"

    n_params = sum(p.numel() for p in model.parameters())
    n_afm = sum(p.numel() for p in model.afm_bottleneck.parameters())
    print(f"\nTotal MDGUNet params: {n_params:,} (~{n_params / 1e6:.3f} M)")
    print(f"AFM-only params:     {n_afm:,} (~{n_afm / 1e6:.4f} M)")
    print()
    print("MDGUNet passed.")
