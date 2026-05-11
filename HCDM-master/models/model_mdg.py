"""MDG-Harmonizer Trainer：``RIHD`` 的子类，叠加四件 4 GB 显存训练必需的能力。

不直接修改 ``model_rihd.RIHD`` 的原因：
    - baseline 是消融实验的对照组，必须保持原样可复跑（``RIHD`` 的训练
      路径不能引入任何新行为）。
    - 通过子类化只增加 *opt-in* 的能力：所有新功能都默认关闭，开关由
      新增 config key 控制，与 baseline 完全互不干扰。

新增能力（全部 opt-in）：
    1. **fp16 AMP**：``use_amp=True`` 时，前向 + 损失放进 ``autocast``，
       反向走 ``GradScaler``。3050 Ti 4 GB 上把 activation 显存砍 ~ 40%。
    2. **梯度累积**：``gradient_accumulation_steps>1`` 时按 micro-batch 累
       积梯度后再 ``optimizer.step()``，让有效 batch_size 等价于
       ``batch_size * gradient_accumulation_steps``，显存占用却保持不变。
    3. **预训练 label 覆写**：``pretrained_label="Network"`` 让加载逻辑
       去找 ``<resume>_Network.pth`` 而不是 ``<resume>_MDGNetwork.pth``，
       从而直接复用 baseline 已发布的预训练权重，不需要复制重命名 .pth 文件。
    4. **loss breakdown 日志**：当 ``self.netG._last_loss_breakdown`` 存在
       且 ``log_loss_breakdown=True`` 时，把 noise / fb / fft / boundary
       等子项分别写到 tensorboard 与 logger，便于调参。
"""

from __future__ import annotations

from typing import Optional

import torch
import tqdm

from .model_rihd import RIHD


class MDGTrainer(RIHD):
    """MDG-Harmonizer 训练器，``RIHD`` 的薄包装。

    新增构造参数（**全部 opt-in，默认值保持 baseline 行为**）：
        use_amp (bool):                  是否启用 fp16 autocast + GradScaler。
        gradient_accumulation_steps (int):  梯度累积步数；1 = 不累积。
        pretrained_label (str | None):   覆盖加载预训练时的 network_label。
        log_loss_breakdown (bool):       是否把 ``netG._last_loss_breakdown``
                                          写到 tensorboard + logger。
    """

    def __init__(
        self,
        *args,
        use_amp: bool = False,
        gradient_accumulation_steps: int = 1,
        pretrained_label: Optional[str] = None,
        log_loss_breakdown: bool = False,
        **kwargs,
    ) -> None:
        # 在父类 __init__ 之前先把新增字段存好，因为父类 __init__ 中会
        # 调用 self.load_everything()——我们的覆写需要在那里读取
        # ``self._pretrained_label``。
        self._use_amp = bool(use_amp)
        self._grad_accum_steps = max(1, int(gradient_accumulation_steps))
        self._pretrained_label = pretrained_label
        self._log_loss_breakdown = bool(log_loss_breakdown)

        super().__init__(*args, **kwargs)

        # GradScaler 仅在 CUDA + AMP 时构造；CPU 走 fp32 兜底
        self._scaler: Optional[torch.cuda.amp.GradScaler] = None
        if self._use_amp and torch.cuda.is_available():
            self._scaler = torch.cuda.amp.GradScaler()
            self.logger.info("[MDGTrainer] AMP enabled (fp16 autocast + GradScaler)")
        if self._grad_accum_steps > 1:
            self.logger.info(
                f"[MDGTrainer] Gradient accumulation enabled: "
                f"effective_batch = batch_size * {self._grad_accum_steps}"
            )
        if self._pretrained_label:
            self.logger.info(
                f"[MDGTrainer] Loading pretrained with label override: "
                f"'{self._pretrained_label}' (instead of class name)"
            )

    # ------------------------------------------------------------------
    # 加载逻辑：允许用 pretrained_label 覆写 class name
    # ------------------------------------------------------------------
    def load_everything(self) -> None:
        """与 baseline 同结构，仅把 ``netG_label`` 替换成可配置的标签。

        典型用法：MDGNetwork 想直接复用 ``Network`` 类训出来的预训练权重
        （它们的 ``denoise_fn`` 子模块同名同结构），把 ``pretrained_label``
        设为 ``"Network"`` 即可，``strict=False`` 会跳过新增的
        ``cdp_net``、``afm_bottleneck``、``fb_loss`` 三类未在预训练里出现的 key。
        """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__

        # 用户显式指定 pretrained_label 时优先；否则保持原行为
        load_label = self._pretrained_label or netG_label
        self.load_network(network=self.netG, network_label=load_label, strict=False)

        if self.ema_scheduler is not None:
            self.load_network(
                network=self.netG_EMA,
                network_label=load_label + '_ema',
                strict=False,
            )

        # 关键：当 pretrained_label 显式覆盖时，意味着我们是从 baseline 的
        # 「Network」预训练权重起步做 MDG fine-tune，optimizer 维度（63 M
        # baseline vs 0.6 M trainable）完全不匹配，强行 resume_training 会
        # AssertionError 或 size mismatch。直接跳过，从 epoch=0 开始。
        # 仅当用户在恢复**自己之前训过的 MDG checkpoint** 时（pretrained_label
        # 留空），才走完整的 resume_training。
        if self._pretrained_label is None:
            self.resume_training([self.optG], self.schedulers)
        else:
            self.logger.info(
                "[MDGTrainer] Skipping resume_training because pretrained_label is set "
                "(starting fresh epoch counter; only network weights loaded)."
            )

    # ------------------------------------------------------------------
    # 训练循环：AMP + 梯度累积 + loss breakdown 日志
    # ------------------------------------------------------------------
    def train_step(self):
        """复刻 ``RIHD.train_step`` 的所有副作用，但加入 AMP + grad accum。

        关键不变量：
            - 每 ``log_iter`` 次记录 metric（与 baseline 一致）。
            - 每 ``ema_iter`` 次更新 EMA（与 baseline 一致）。
            - ``self.iter`` 仍按 ``+= batch_size`` 推进，不因为梯度累积而变。
            - scheduler.step() 在 epoch 末尾走一次，与 baseline 一致。
        """
        self.netG.train()
        self.train_metrics.reset()

        accum_step = 0
        self.optG.zero_grad(set_to_none=True)

        for train_data in tqdm.tqdm(self.phase_loader):
            self.set_input(train_data)

            # ------------- forward + loss（可选 autocast）-------------
            if self._use_amp and torch.cuda.is_available():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    loss = self.netG(self.gt_image, self.cond_image, mask=self.mask)
            else:
                loss = self.netG(self.gt_image, self.cond_image, mask=self.mask)

            # 梯度累积：把 loss 除以累积步数，让 N 个 micro-batch 反向梯度的
            # 求和等价于一次大 batch 的平均梯度。
            scaled_loss = loss / self._grad_accum_steps

            # ------------- backward + step -------------
            if self._scaler is not None:
                self._scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            accum_step += 1
            do_step = accum_step % self._grad_accum_steps == 0

            if do_step:
                if self._scaler is not None:
                    self._scaler.step(self.optG)
                    self._scaler.update()
                else:
                    self.optG.step()
                self.optG.zero_grad(set_to_none=True)

            # ------------- bookkeeping（对齐 baseline）-------------
            self.iter += self.batch_size
            self.writer.set_iter(self.epoch, self.iter, phase='train')
            self.train_metrics.update(self.loss_fn.__name__, loss.item())

            if self.iter % self.opt['train']['log_iter'] == 0:
                for key, value in self.train_metrics.result().items():
                    self.logger.info('{:5s}: {}\t'.format(str(key), value))
                    self.writer.add_scalar(key, value)
                # MDG: 额外把 loss 分量打到 tensorboard
                if self._log_loss_breakdown:
                    self._maybe_log_loss_breakdown()
                for key, value in self.get_current_visuals().items():
                    self.writer.add_images(key, value)

            if self.ema_scheduler is not None:
                if (
                    self.iter % self.ema_scheduler['ema_iter'] == 0
                    and self.iter > self.ema_scheduler['ema_start']
                ):
                    self.logger.info('Update the EMA  model at the iter {:.0f}'.format(self.iter))
                    self.EMA.update_model_average(self.netG_EMA, self.netG)

        # 处理 epoch 末尾「最后一个 micro-batch 还没 step」的情况
        if accum_step % self._grad_accum_steps != 0:
            if self._scaler is not None:
                self._scaler.step(self.optG)
                self._scaler.update()
            else:
                self.optG.step()
            self.optG.zero_grad(set_to_none=True)

        for scheduler in self.schedulers:
            scheduler.step()
        return self.train_metrics.result()

    # ------------------------------------------------------------------
    # 辅助：把 netG._last_loss_breakdown 写到日志
    # ------------------------------------------------------------------
    def _maybe_log_loss_breakdown(self) -> None:
        net = self.netG.module if self.opt['distributed'] else self.netG
        breakdown = getattr(net, '_last_loss_breakdown', None)
        if not breakdown:
            return
        for sub_key, sub_val in breakdown.items():
            tag = f'train/breakdown_{sub_key}'
            self.writer.add_scalar(tag, float(sub_val))
            self.logger.info(f'{tag:35s}: {sub_val:.6f}')
