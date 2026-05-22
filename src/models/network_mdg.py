"""MDG-Harmonizer Network 包装层。

在 baseline ``network_modified_backup.Network`` 基础上加三件事：
    1. 在前向之前用 ``CDPNet(y_cond, mask)`` 抽取退化先验向量 ``deg_vec``，
       透传给 ``MDGUNet``（U-Net bottleneck 处的 AFM 消费它）。
    2. 训练时，可选在多尺度噪声 L1 损失之外，叠加 ``FBLoss`` 在
       ``predict_start_from_noise`` 反算出的 ``y_0_hat`` 上做前景/边界/感知/
       高频损失（论文创新点 3）。
    3. ``freeze_backbone=True`` 时，把 ``denoise_fn`` 的所有参数 ``requires_grad``
       置 False，但**保留 AFM** 子模块为可训练（AFM 是新加的不在预训练权重里）；
       同时 CDPNet/FBLoss(LPIPS 已 frozen) 也保持可训练。这是 4 GB 显存下能
       跑通的核心策略。

与 baseline 的接口保持兼容：
    - ``forward(y_0, y_cond, mask, noise)`` 返回单个 scalar loss（兼容
      ``model_rihd.RIHD.train_step`` 现有 ``loss.backward()``）。
    - ``restoration(y_cond, y_t, y_0, mask, sample_num)`` 返回 ``(y_t, ret_arr)``
      仍为 baseline 同款 inference 接口。

注意：本类**不**继承 baseline ``Network``。`Network.__init__` 内部硬编码了
``module_name`` 到 unet 文件的分发，不便于 override；这里直接重写完整逻辑，
但所有数学（beta schedule、q_sample、p_sample、predict_start_from_noise）都与
baseline 保持完全一致，避免引入不易察觉的训练偏差。
"""

from __future__ import annotations

import math
from functools import partial
from inspect import isfunction
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize
from tqdm import tqdm

from core.base_network import BaseNetwork

from .afm import AFM
from .cdp_net import CDPNet
from .fb_loss import FBLoss
from .guided_diffusion_modules.unet_mdg import MDGUNet


# ----------------------------- 辅助函数 -----------------------------
def _exists(x):
    return x is not None


def _default(val, d):
    if _exists(val):
        return val
    return d() if isfunction(d) else d


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape=(1, 1, 1, 1)) -> torch.Tensor:
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def _resize_tensor(input_tensor: torch.Tensor) -> List[torch.Tensor]:
    """与 baseline ``resize_tensor`` 完全一致，按 4 个尺度从小到大返回 list。"""
    width = input_tensor.shape[2]
    height = input_tensor.shape[3]
    output_tensor = input_tensor
    output_tensor_list = [output_tensor]
    for _ in range(3):
        width = width // 2
        height = height // 2
        torch_resize_fun = Resize([width, height])
        output_tensor = torch_resize_fun(output_tensor)
        output_tensor_list.insert(0, output_tensor)
    return output_tensor_list


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def _make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':
        betas = 1.0 / np.linspace(n_timestep, 1, n_timestep, dtype=np.float64)
    elif schedule == 'cosine':
        timesteps = (torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s)
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# ----------------------------- 主类 -----------------------------
class MDGNetwork(BaseNetwork):
    """MDG-Harmonizer Diffusion Network。

    Args:
        unet:           dict，传给 ``MDGUNet`` 的所有 kwargs（不含 deg_dim）。
        beta_schedule:  dict（与 baseline 同），含 ``train`` / ``test`` 两个 key。
        cdp:            dict，CDPNet 构造参数（``backbone_channels``、``embed_dim`` 等）。
        afm:            dict，传给 MDGUNet 的 AFM kwargs（``num_tokens`` 等）。
        fb_loss:        dict，FBLoss 构造参数。``None`` 时禁用 FBLoss。
        loss_weights:   dict，``{noise: float, fb: float}``，控制两路损失的权重。
        freeze_backbone: bool，是否冻结主干（仅 AFM + CDPNet + FBLoss 可训）。
        deg_dim:        int，CDPNet 的输出维度，必须 = MDGUNet AFM 的 deg_dim。
        cdp_zero_vec:   bool，**ablation 专用**。True 时跳过 ``self.cdp_net`` 的
                        实际调用，用 ``torch.zeros(B, deg_dim)`` 替代输出，从而把
                        AFM 退化为「FiLM-only / 无内容感知」模式。论文消融 B 用。
        disable_afm:    bool，**ablation 专用**。True 时强制把 ``deg_vec`` 设为
                        ``None``，``MDGUNet`` 端整个 AFM bottleneck 被跳过，等价
                        baseline UNet。论文消融 C 用。
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
        deg_dim: int = 32,
        cdp_zero_vec: bool = False,
        disable_afm: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # ------- 1) CDP-Net（退化先验编码器） -------
        cdp = cdp or {}
        self.cdp_net = CDPNet(**cdp)
        if self.cdp_net.out_dim != deg_dim:
            raise ValueError(
                f"deg_dim ({deg_dim}) must equal CDPNet.out_dim ({self.cdp_net.out_dim})"
            )

        # ------- 2) MDG U-Net（带 bottleneck AFM） -------
        afm = afm or {}
        unet_kwargs = dict(unet)
        unet_kwargs["deg_dim"] = deg_dim
        if "num_tokens" in afm:
            unet_kwargs["afm_num_tokens"] = afm["num_tokens"]
        if "use_checkpoint" in afm:
            unet_kwargs["afm_use_checkpoint"] = afm["use_checkpoint"]
        self.denoise_fn = MDGUNet(**unet_kwargs)

        # ------- 3) FB-Loss（可选） -------
        if fb_loss is not None:
            self.fb_loss = FBLoss(**fb_loss)
        else:
            self.fb_loss = None

        # ------- 4) loss 权重 -------
        loss_weights = loss_weights or {}
        self.w_noise = float(loss_weights.get("noise", 1.0))
        self.w_fb = float(loss_weights.get("fb", 0.0))

        # ------- 5) 冻结策略 -------
        self.beta_schedule = beta_schedule
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self._freeze_backbone(unfreeze_decoder_last_n)

        # ------- 6) ablation 开关（仅消融实验使用，主训练保持 False） -------
        # 同时配合 cdp_zero_vec / disable_afm 配置可精准复现：
        #   B_no_cdp  : cdp_zero_vec=True, disable_afm=False
        #   C_no_afm  : cdp_zero_vec=False, disable_afm=True
        # 单独保留两个字段方便后续做更细粒度（例如 zero-vec + AFM-on）的对照。
        self._cdp_zero_vec = bool(cdp_zero_vec)
        self._disable_afm = bool(disable_afm)
        self._deg_dim = int(deg_dim)

        # 训练日志：最后一次 forward 的 loss 分量，供外部 logger 读取
        self._last_loss_breakdown: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # 初始化覆写：保留 AFM 的「零初始化恒等」性
    # ------------------------------------------------------------------
    def init_weights(self) -> None:
        """``BaseNetwork.init_weights`` 会对全网 Conv/Linear 做 kaiming/...
        初始化，这会**摧毁** AFM 的零初始化恒等性（``out_proj/gamma/beta``
        三个零层会被覆写成随机值），导致冻结主干 + 插入 AFM 的训练在第 1
        步就把预训练特征污染掉。

        因此先调父类做默认初始化，之后再把所有 ``AFM`` 子模块的关键层
        手动复位为零。CDPNet 仍按默认 init 处理（其结构无零初始化要求）。
        """
        super().init_weights()
        for module in self.modules():
            if isinstance(module, AFM):
                module._reset_parameters()

    # ------------------------------------------------------------------
    # 冻结控制
    # ------------------------------------------------------------------
    def _freeze_backbone(self, unfreeze_decoder_last_n: int = 0) -> None:
        """冻结 ``denoise_fn`` 中除 AFM + 输出层之外的所有参数。

        4 GB VRAM 下连原 baseline 单 batch 训练都 OOM；MDG-Harmonizer 的核心
        是「冻结 ~63 M 主干，仅训 ~0.6 M 新模块」，使 optimizer state +
        activation 均能塞进显存。CDPNet 与 FBLoss 自身参数（LPIPS 已 frozen）
        不在此处理范围，本身就是可训练的。

        输出层 (out / out_list) 始终可训练：它们是零初始化卷积，不解冻则
        无 AFM 时（消融 C）前向输出恒为零，无法训练。

        Args:
            unfreeze_decoder_last_n: 解冻 output_blocks 最后 N 个 block。
                0 = 默认行为（全部冻结）；6 = 解冻后 6 个（ch=64/128 层）。
        """
        # 1) 整个 denoise_fn 先全部冻结
        for p in self.denoise_fn.parameters():
            p.requires_grad = False
        # 2) 把 AFM 子模块的参数恢复为可训练
        for module in self.denoise_fn.modules():
            if isinstance(module, AFM):
                for p in module.parameters():
                    p.requires_grad = True
        # 3) 输出层始终可训练（消融 C 无 AFM 时保证梯度流）
        for p in self.denoise_fn.out.parameters():
            p.requires_grad = True
        for m in self.denoise_fn.out_list:
            for p in m.parameters():
                p.requires_grad = True
        # 4) 解冻解码器最后 N 个 block（云端微调用）
        if unfreeze_decoder_last_n > 0:
            blocks = self.denoise_fn.output_blocks
            start = len(blocks) - unfreeze_decoder_last_n
            for idx in range(start, len(blocks)):
                for p in blocks[idx].parameters():
                    p.requires_grad = True
        # 3) 修复 frozen ResBlock / AttentionBlock 的 checkpoint 兼容性问题。
        #    自定义 ``CheckpointFunction.backward`` 调 ``torch.autograd.grad``
        #    会要求所有传入的 params 都 ``requires_grad=True``；冻结后会报
        #    ``RuntimeError: One of the differentiated Tensors does not require grad``。
        #    通过给冻结模块装一个 forward 钩子，**调用 checkpoint 时只传可训
        #    params**（一般为空 list），从而保留 checkpoint 的显存收益。
        self._patch_frozen_checkpoint()

    def _patch_frozen_checkpoint(self) -> None:
        """给 frozen ResBlock / AttentionBlock 替换 forward，避免两个互相
        关联的 bug：

        1. **frozen params 触发 ``CheckpointFunction.backward`` 报错**
           原 ``nn.CheckpointFunction.backward`` 调 ``torch.autograd.grad``
           会要求所有传入的 params 都 ``requires_grad=True``；冻结后会报
           ``RuntimeError: One of the differentiated Tensors does not require grad``。
        2. **legacy checkpoint 与 AMP autocast 不兼容**
           原 ``CheckpointFunction.backward`` 会重新跑一遍 forward，但**没有
           保留 autocast 上下文**：fp16 输入 + fp32 权重 → ``Input type
           (Half) and bias type (float) should be the same``。

        修复：直接换用 PyTorch 现代 ``torch.utils.checkpoint.checkpoint``
        (``use_reentrant=False``)。它会自动保留 autocast 上下文、不依赖显式
        params 列表，且在所有 input/param 都不 requires_grad 时优雅退化。

        参考：``models/guided_diffusion_modules/unet_modified.py`` 中
        ``ResBlock.forward`` / ``AttentionBlock.forward`` 的原实现。
        """
        from .guided_diffusion_modules.unet_modified import ResBlock, AttentionBlock
        import torch.utils.checkpoint as _cp

        def _safe_resblock_forward(self_m, x, emb):
            if self_m.use_checkpoint:
                return _cp.checkpoint(self_m._forward, x, emb, use_reentrant=False)
            return self_m._forward(x, emb)

        def _safe_attn_forward(self_m, x):
            # AttentionBlock 原 forward 硬编码使用 checkpoint=True；这里沿用
            # 同样的「省显存」策略，仅替换为现代 API。
            return _cp.checkpoint(self_m._forward, x, use_reentrant=False)

        n_res, n_attn = 0, 0
        for m in self.denoise_fn.modules():
            if isinstance(m, ResBlock):
                m.forward = _safe_resblock_forward.__get__(m, type(m))
                n_res += 1
            elif isinstance(m, AttentionBlock):
                m.forward = _safe_attn_forward.__get__(m, type(m))
                n_attn += 1
        # 没有 logger 句柄时静默；外部 trainer 可读 self._patch_stats 反查
        self._patch_stats = {"resblock_patched": n_res, "attention_patched": n_attn}

    def trainable_parameters(self) -> List[torch.nn.Parameter]:
        """返回所有 ``requires_grad=True`` 的参数（供 optimizer 使用）。

        外部 ``model_rihd.RIHD`` 默认拿 ``self.netG.parameters()`` 给 optimizer。
        即便冻结的参数 requires_grad=False，绝大多数 optimizer 也会跳过它们；
        但显式暴露此接口可在自定义 trainer 中收紧（例如想看可训练比例）。
        """
        return [p for p in self.parameters() if p.requires_grad]

    # ------------------------------------------------------------------
    # loss 接口（兼容 baseline 的 set_loss）
    # ------------------------------------------------------------------
    def set_loss(self, loss_fn) -> None:
        """RIHD 通过 ``set_loss`` 注入 ``mse_loss`` 等基础噪声损失函数。

        与 baseline 一致，``loss_fn(pred_noise, gt_noise)`` 用于多尺度噪声损失。
        """
        self.loss_fn = loss_fn

    # ------------------------------------------------------------------
    # diffusion 调度（数学与 baseline 完全一致）
    # ------------------------------------------------------------------
    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train') -> None:
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = _make_beta_schedule(**self.beta_schedule[phase])
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1.0 - betas

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1.0, gammas[:-1])

        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1.0 / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1.0 / gammas - 1)))

        posterior_variance = betas * (1.0 - gammas_prev) / (1.0 - gammas)
        self.register_buffer(
            'posterior_log_variance_clipped',
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            'posterior_mean_coef1',
            to_torch(betas * np.sqrt(gammas_prev) / (1.0 - gammas)),
        )
        self.register_buffer(
            'posterior_mean_coef2',
            to_torch((1.0 - gammas_prev) * np.sqrt(alphas) / (1.0 - gammas)),
        )

    # ------------------------------------------------------------------
    # 数学：从 noise 反算 x_0_hat / 后验
    # ------------------------------------------------------------------
    def predict_start_from_noise(self, y_t, t, noise):
        return (
            _extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t
            - _extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def q_posterior(self, y_0_hat, y_t, t):
        posterior_mean = (
            _extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat
            + _extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = _extract(
            self.posterior_log_variance_clipped, t, y_t.shape
        )
        return posterior_mean, posterior_log_variance_clipped

    def q_sample(self, y_0, sample_gammas, noise=None):
        noise = _default(noise, lambda: torch.randn_like(y_0))
        return sample_gammas.sqrt() * y_0 + (1 - sample_gammas).sqrt() * noise

    # ------------------------------------------------------------------
    # 退化先验提取（统一入口，方便 ablation 一处修改）
    # ------------------------------------------------------------------
    def _compute_deg_vec(
        self,
        y_cond: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """统一的 ``deg_vec`` 提取入口。优先级：

        1. ``disable_afm=True``：直接返回 ``None``，``MDGUNet.forward`` 内会跳
           过整个 AFM bottleneck（消融 C，等价 baseline UNet）。
        2. ``mask is None``：原本就没有 fg 信息可供 CDPNet 编码，返回 ``None``。
        3. ``cdp_zero_vec=True``：构造 ``(B, deg_dim)`` 全零张量，**不**调用
           ``self.cdp_net``。这等价于让 AFM 看到「内容无关的 deg_vec」，从而
           退化为只剩可学习 token 的 FiLM-only 调制（消融 B）。
        4. 否则：按主训练路径调用 ``self.cdp_net(y_cond, mask)``。
        """
        if self._disable_afm:
            return None
        if mask is None:
            return None
        if self._cdp_zero_vec:
            b = y_cond.shape[0]
            return torch.zeros(
                b, self._deg_dim, device=y_cond.device, dtype=y_cond.dtype
            )
        deg_vec, _ = self.cdp_net(y_cond, mask)
        return deg_vec

    # ------------------------------------------------------------------
    # 推理路径：把 deg_vec 一路透传到 denoise_fn
    # ------------------------------------------------------------------
    def p_mean_variance(
        self,
        y_t: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool,
        y_cond: Optional[torch.Tensor] = None,
        deg_vec: Optional[torch.Tensor] = None,
    ):
        noise_level = _extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        predicted_noise = self.denoise_fn(
            torch.cat([y_cond, y_t], dim=1), noise_level, deg_vec=deg_vec
        )
        y_0_hat = self.predict_start_from_noise(
            y_t, t=t, noise=predicted_noise[-1]  # -1 = full-res，与 baseline 一致
        )

        if clip_denoised:
            y_0_hat.clamp_(-1.0, 1.0)

        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t
        )
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(
        self,
        y_t: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
        y_cond: Optional[torch.Tensor] = None,
        deg_vec: Optional[torch.Tensor] = None,
    ):
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond, deg_vec=deg_vec
        )
        noise = torch.randn_like(y_t) if any(t > 0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def restoration(
        self,
        y_cond: torch.Tensor,
        y_t: Optional[torch.Tensor] = None,
        y_0: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        sample_num: int = 8,
    ):
        """与 baseline 接口完全一致；多算一步 CDPNet 把 deg_vec 缓存到本次循环。

        在 1000 步 DDPM 采样下，CDPNet 只跑一次（外面），节省时间。
        """
        b, *_ = y_cond.shape
        assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
        sample_inter = self.num_timesteps // sample_num

        # CDPNet 只算一次：整个 reverse process 共享同一个 deg_vec。
        # 经 _compute_deg_vec 后已根据 ablation 开关返回 None / zeros / 真实编码。
        deg_vec = self._compute_deg_vec(y_cond, mask)

        y_t = _default(y_t, lambda: torch.randn_like(y_cond))
        ret_arr = y_t
        for i in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc='sampling loop time step',
            total=self.num_timesteps,
        ):
            t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)
            y_t = self.p_sample(y_t, t, y_cond=y_cond, deg_vec=deg_vec)
            if mask is not None:
                y_t = y_0 * (1.0 - mask) + mask * y_t
            if i % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_t], dim=0)
        return y_t, ret_arr

    @torch.no_grad()
    def restoration_ddim(
        self,
        y_cond: torch.Tensor,
        y_t: Optional[torch.Tensor] = None,
        y_0: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        sample_num: int = 8,
        ddim_steps: int = 25,
        eta: float = 0.0,
    ):
        """DDIM 风格多步跳转采样。

        每步：预测噪声 → 反推 x_0 → 用 DDIM 公式算 x_s。
        eta=0 确定性（最快），eta=1 加后验噪声。
        """
        b = y_cond.shape[0]
        device = y_cond.device

        deg_vec = self._compute_deg_vec(y_cond, mask)
        x = _default(y_t, lambda: torch.randn_like(y_cond))

        # 子采样时刻表（降序）
        stride = max(1, self.num_timesteps // ddim_steps)
        all_steps = list(range(self.num_timesteps - 1, -1, -1))
        subsampled = [all_steps[i] for i in range(0, len(all_steps), stride)]
        seen = set()
        subsampled = [t for t in subsampled if t not in seen and not seen.add(t)]
        if subsampled[-1] != 0:
            subsampled.append(0)

        sample_inter = max(len(subsampled) // sample_num, 1)

        ret_arr = x
        for idx, t_val in enumerate(subsampled[:-1]):
            s_val = subsampled[idx + 1]

            # 模型预测噪声
            t = torch.full((b,), t_val, device=device, dtype=torch.long)
            noise_level = _extract(self.gammas, t, x_shape=(1, 1)).to(device)
            unet_input = torch.cat([y_cond, x], dim=1)
            eps = self.denoise_fn(unet_input, noise_level, deg_vec=deg_vec)[-1]

            # 反推 x_0: x_t = √γ_t · x_0 + √(1-γ_t) · ε
            a_t = self.gammas[t_val]
            a_s = self.gammas[s_val]
            x_0 = (x - (1.0 - a_t).sqrt() * eps) / a_t.sqrt().clamp(min=1e-8)
            x_0.clamp_(-1.0, 1.0)

            # DDIM 跳转: x_s = √α̅_s · x_0 + √(1-α̅_s - σ²) · ε + σ · noise
            # 其中 c² = (1-α̅_s)/(1-α̅_t) · (1-α̅_t/α̅_s),  σ = η · c
            c_sq = ((1.0 - a_s) / (1.0 - a_t).clamp(min=1e-20)) * (
                1.0 - a_t / a_s.clamp(min=1e-20)
            )
            c_sq = c_sq.clamp(min=0.0)
            sigma = eta * c_sq.sqrt()
            sigma_sq = sigma * sigma
            d_sq = (1.0 - a_s - sigma_sq).clamp(min=0.0)

            if s_val > 0:
                noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
                x = a_s.sqrt() * x_0 + d_sq.sqrt() * eps + sigma * noise
            else:
                x = x_0

            if mask is not None:
                x = y_0 * (1.0 - mask) + mask * x
            if idx % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, x], dim=0)

        return x, ret_arr

    @torch.no_grad()
    def restoration_dpm(
        self,
        y_cond: torch.Tensor,
        y_t: Optional[torch.Tensor] = None,
        y_0: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        sample_num: int = 8,
        dpm_steps: int = 25,
        dpm_order: int = 2,
    ):
        """DPM-Solver++ 快速采样。

        与 ``restoration()`` (DDPM) 接口一致，仅采样器不同。
        使用 ``--sampler dpm --ddim-steps 25`` 启用。

        Note:
            DDIM 使用 ``--sampler ddim``（调用 ``restoration_ddim``），
            与 DPM-Solver++ 是不同的方法，不要混用。
        """
        from .dpm_solver import dpm_solver_restoration

        b, *_ = y_cond.shape
        device = y_cond.device
        y_t = _default(y_t, lambda: torch.randn_like(y_cond))

        deg_vec = self._compute_deg_vec(y_cond, mask)

        # 构造 denoise_fn 闭包适配 dpm_solver_restoration
        def model_fn(x, noise_level, deg_vec=None):
            return self.denoise_fn(x, noise_level, deg_vec=deg_vec)

        y_out, ret_arr = dpm_solver_restoration(
            denoise_fn=model_fn,
            y_cond=y_cond,
            alphas_cumprod=self.gammas.cpu().numpy(),
            mask=mask,
            y_t=y_t,
            deg_vec=deg_vec,
            y_0=y_0,
            steps=dpm_steps,
            order=dpm_order,
            progress=True,
        )

        # sample_num 兼容：从 ret_arr 中采样
        if sample_num > 0 and ret_arr.shape[0] > 1:
            n_total = ret_arr.shape[0]
            indices = torch.linspace(0, n_total - 1, min(sample_num, n_total)).long()
            ret_arr = ret_arr[indices]

        return y_out, ret_arr

    # ------------------------------------------------------------------
    # 训练前向：返回 scalar loss（与 baseline 接口兼容）
    # ------------------------------------------------------------------
    def forward(
        self,
        y_0: torch.Tensor,
        y_cond: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, *_ = y_0.shape
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()
        gamma_t1 = _extract(self.gammas, t - 1, x_shape=(1, 1))
        sqrt_gamma_t2 = _extract(self.gammas, t, x_shape=(1, 1))
        sample_gammas = (sqrt_gamma_t2 - gamma_t1) * torch.rand(
            (b, 1), device=y_0.device
        ) + gamma_t1
        sample_gammas = sample_gammas.view(b, -1)

        noise = _default(noise, lambda: torch.randn_like(y_0))

        y_noisy = self.q_sample(
            y_0=y_0,
            sample_gammas=sample_gammas.view(-1, 1, 1, 1),
            noise=noise,
        )

        # ============== 退化先验提取（CDPNet） ==============
        # CDPNet 接受 y_cond（合成图，含退化）+ mask；不依赖 y_0，所以推理也能用。
        # ablation 开关在 _compute_deg_vec 里统一处理。
        deg_vec = self._compute_deg_vec(y_cond, mask)

        noise_resized_list = _resize_tensor(noise)
        if mask is not None:
            mask_resized_list = _resize_tensor(mask)
            unet_input = torch.cat([y_cond, y_noisy * mask + (1.0 - mask) * y_0], dim=1)
        else:
            mask_resized_list = [torch.ones_like(s) for s in noise_resized_list]
            unet_input = torch.cat([y_cond, y_noisy], dim=1)

        noise_hat_list = self.denoise_fn(unet_input, sample_gammas, deg_vec=deg_vec)

        # ============== 噪声多尺度 L1（与 baseline 完全一致） ==============
        noise_loss = noise_hat_list[0].new_zeros(())
        for i in range(len(noise_hat_list)):
            noise_loss = noise_loss + self.loss_fn(
                mask_resized_list[i] * noise_resized_list[i],
                mask_resized_list[i] * noise_hat_list[i],
            )

        # ============== 可选：FB-Loss 在 image domain ==============
        fb_total = noise_loss.new_zeros(())
        fb_dict: Dict[str, torch.Tensor] = {}
        if self.fb_loss is not None and self.w_fb > 0 and mask is not None:
            # 用 full-res noise 反算 x_0_hat（baseline `predict_start_from_noise` 同款）
            y_0_hat = self.predict_start_from_noise(
                y_t=y_noisy, t=t, noise=noise_hat_list[-1]
            )
            y_0_hat = y_0_hat.clamp(-1.0, 1.0)
            fb_total, fb_dict = self.fb_loss(y_0_hat, y_0, mask)

        total = self.w_noise * noise_loss + self.w_fb * fb_total

        # 缓存损失分量，外部 trainer 可读 self._last_loss_breakdown 写 tensorboard
        breakdown = {
            "total": total.detach().item(),
            "noise": noise_loss.detach().item(),
        }
        if self.fb_loss is not None and self.w_fb > 0 and mask is not None:
            breakdown["fb_total"] = fb_total.detach().item()
            breakdown.update({f"fb_{k}": v.detach().item() for k, v in fb_dict.items()})
        self._last_loss_breakdown = breakdown

        return total


# ----------------------------- 自检 -----------------------------
if __name__ == "__main__":
    """CPU smoke test：构造 mini MDGNetwork，验证：
        1. 构造成功；CDPNet/AFM/FB-Loss 都接上
        2. freeze_backbone=True 后只剩 CDPNet + AFM + (FB-Loss 中 LPIPS 之外) 可训
        3. forward 返回 scalar loss、能 backward 且 grad 仅流到可训练参数
        4. restoration（迷你 num_timesteps）可跑通，shape 与 y_cond 相同
    """
    import torch as _torch

    _torch.manual_seed(0)

    # mini config（CPU 上几秒跑完）
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

    net = MDGNetwork(
        unet=unet_kwargs,
        beta_schedule=beta_schedule,
        cdp={"in_channels": 4, "backbone_channels": (16, 32, 64, 128), "embed_dim": 28},
        afm={"num_tokens": 4, "use_checkpoint": False},
        fb_loss={"use_lpips": False, "weights": (1.0, 3.0, 5.0, 0.1, 0.5)},
        loss_weights={"noise": 1.0, "fb": 0.5},
        freeze_backbone=True,
        deg_dim=32,
    )
    net.set_loss(F.l1_loss)
    net.set_new_noise_schedule(device=_torch.device("cpu"), phase="train")

    # ----------------------------------------------------------------------
    # 关键修正：原 ``UNet`` 的最终 ``self.out`` / ``out_list[*]`` 通过
    # ``zero_module`` 零初始化最终卷积层。在「冻结主干 + 随机初始化」的
    # smoke test 场景里，这会导致 ``dL/d(activation_before_zero_conv)``
    # = ``dL/d(noise_hat) @ zero_conv.W`` = 0，从而切断到 AFM 的梯度回传。
    #
    # 真实训练里我们 *加载预训练权重*，这些层早已被训成非零，不存在该问题。
    # 这里手动把零初始化层用小标准差正态扰动，**仅用于** smoke test，
    # 让梯度流验证能跑过。
    # ----------------------------------------------------------------------
    with _torch.no_grad():
        for name, p in net.denoise_fn.named_parameters():
            if (name.startswith("out.") or name.startswith("out_list.")) and p.detach().abs().sum().item() < 1e-6:
                p.normal_(mean=0.0, std=0.01)

    print("=" * 68)
    print("MDGNetwork smoke test (CPU, fp32, freeze + simulated-pretrained out-convs)")
    print("=" * 68)

    n_total = sum(p.numel() for p in net.parameters())
    n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total params:     {n_total:,} (~{n_total / 1e6:.2f} M)")
    print(f"Trainable params: {n_trainable:,} (~{n_trainable / 1e6:.3f} M)")
    print(f"Trainable ratio:  {100 * n_trainable / n_total:.2f}%")
    assert n_trainable < n_total, "freeze_backbone seemingly did nothing"
    assert n_trainable / n_total < 0.05, "trainable params should be < 5% of total"

    # forward
    B = 1
    y_0 = _torch.randn(B, 3, H, H)
    y_cond = _torch.randn(B, 3, H, H)
    mask = (_torch.rand(B, 1, H, H) > 0.5).float()

    loss = net(y_0, y_cond=y_cond, mask=mask)
    print(f"\nforward loss: {loss.item():.4f}")
    print(f"loss breakdown: {net._last_loss_breakdown}")

    # backward step 1
    loss.backward()
    afm_nonzero_1 = sum(
        1 for _, p in net.denoise_fn.afm_bottleneck.named_parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().sum().item() > 0
    )
    cdp_nonzero_1 = sum(
        1 for _, p in net.cdp_net.named_parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().sum().item() > 0
    )
    print(f"[step 1] AFM nonzero-grad params:    {afm_nonzero_1}")
    print(f"[step 1] CDPNet nonzero-grad params: {cdp_nonzero_1}  (期望 > 0：small-init 保证梯度流通)")
    assert afm_nonzero_1 >= 4, "step 1: AFM out_proj/film 路径应拿到梯度"
    assert cdp_nonzero_1 >= 1, "step 1: small-init 后 CDPNet 应在首步就拿到梯度"

    # restoration（极迷你 num_timesteps=8 加快 CPU 测试）
    net.eval()
    with _torch.no_grad():
        out, ret = net.restoration(y_cond=y_cond, y_0=y_0, mask=mask, sample_num=4)
    print(f"\nrestoration output shape: {tuple(out.shape)}")
    assert out.shape == y_cond.shape
    print()
    print("MDGNetwork passed.")
