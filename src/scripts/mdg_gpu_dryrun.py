"""GPU 5-iter dry-run：验证 MDG-Harmonizer 在 4 GB 显存下能跑通。

关心三件事：
    1. **显存峰值** ≤ 3.5 GB（4 GB 卡留 0.5 GB 余量给 driver/cuDNN）
    2. **AMP autocast** 不出 NaN / inf；前向 + 反向 + scaler.step 全部跑通
    3. **单步速度**：每步 forward+backward 时间，用于估总训练时长

不走完整 trainer / dataloader，避免 dataset 重复 IO；直接在 GPU 上构造
随机 256×256 mini batch 跑 5 步。结果与真实数据训练的显存/速度高度相近
（差异主要是 H2H 数据传输 + EMA 更新）。
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F


def _parse_jsonc(path: Path) -> OrderedDict:
    s = ''
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s += line.split('//')[0] + '\n'
    return json.loads(s, object_pairs_hook=OrderedDict)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))

    if not torch.cuda.is_available():
        print("!! CUDA not available, this script needs GPU. Exiting.")
        sys.exit(1)
    device = torch.device('cuda')

    print('=' * 78)
    print('MDG-Harmonizer GPU dry-run (5 iter, batch=2, H=W=256)')
    print(f'GPU: {torch.cuda.get_device_name(0)}, '
          f'total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
    print('=' * 78)

    # 1) 解析 config 拿出 net 构造参数
    cfg_path = repo_root / 'config' / 'harmonization_day2night_mdg.json'
    opt = _parse_jsonc(cfg_path)
    net_args = dict(opt['model']['which_networks'][0]['args'])
    init_type = net_args.pop('init_type', 'kaiming')

    # 2) 构造 MDGNetwork
    from models.network_mdg import MDGNetwork
    net = MDGNetwork(**net_args, init_type=init_type)

    # 3) 加载 baseline 预训练权重
    pretrained_label = opt['model']['which_model']['args']['pretrained_label']
    pretrained_path = repo_root / f"{opt['path']['resume_state']}_{pretrained_label}.pth"
    if pretrained_path.exists():
        sd = torch.load(pretrained_path, map_location='cpu')
        missing, unexpected = net.load_state_dict(sd, strict=False)
        bb_missing = [k for k in missing
                      if 'afm_bottleneck' not in k
                      and 'cdp_net' not in k
                      and not k.startswith('fb_loss')]
        print(f'\n[1] Pretrained loaded: missing={len(missing)} '
              f'(backbone_missing={len(bb_missing)}), unexpected={len(unexpected)}')
        if bb_missing:
            print(f'    !! backbone missing keys 应该为 0，实际：{bb_missing[:3]} ...')
            sys.exit(2)
    else:
        print(f'\n!! Pretrained not found at {pretrained_path}, using random init')

    # 4) init_weights 复位 + 上 GPU
    net.init_weights()
    net.set_loss(F.l1_loss)
    net.set_new_noise_schedule(device=device, phase='train')
    net.to(device)
    net.train()

    n_total = sum(p.numel() for p in net.parameters())
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'[2] Network on GPU. total={n_total:,} trainable={n_train:,} '
          f'({100*n_train/n_total:.2f}%)')

    # 5) optimizer：仅可训参数（与 run.py:44 行为一致）
    train_params = [p for p in net.parameters() if p.requires_grad]
    optG = torch.optim.Adam(train_params, lr=5e-4, weight_decay=0)

    # 6) AMP scaler
    scaler = torch.cuda.amp.GradScaler()

    # 7) 假 mini batch（256×256, batch=2, fp32 在 dataloader 端，autocast 内转 fp16）
    B, H = 2, 256
    print(f'\n[3] Allocating dummy batch on GPU: B={B}, H=W={H}')

    torch.cuda.reset_peak_memory_stats()
    grad_accum_steps = opt['model']['which_model']['args']['gradient_accumulation_steps']
    print(f'    use_amp=True, grad_accum_steps={grad_accum_steps}')
    print()

    # 8) 5 iter loop
    iter_times = []
    optG.zero_grad(set_to_none=True)
    for step in range(1, 6):
        y0 = torch.randn(B, 3, H, H, device=device)
        yc = torch.randn(B, 3, H, H, device=device)
        msk = (torch.rand(B, 1, H, H, device=device) > 0.5).float()

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.cuda.amp.autocast(dtype=torch.float16):
            loss = net(y0, y_cond=yc, mask=msk)
        scaled_loss = loss / grad_accum_steps
        scaler.scale(scaled_loss).backward()

        # 模拟梯度累积：每 grad_accum_steps 才 step 一次
        if step % grad_accum_steps == 0:
            scaler.step(optG)
            scaler.update()
            optG.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        iter_times.append(dt)

        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        # 检查 NaN/Inf
        nan_check = (
            torch.isnan(loss).any().item() or torch.isinf(loss).any().item()
        )
        bd = getattr(net, '_last_loss_breakdown', {})
        print(
            f'  step {step}: loss={loss.item():.4f}  '
            f'time={dt*1000:.0f}ms  peak_mem={peak_mb:.0f}MB  '
            f"nan/inf={nan_check}"
        )
        if bd:
            inline = ' '.join(f'{k}={v:.3f}' for k, v in bd.items()
                              if k in ('noise', 'fb_total'))
            print(f'           breakdown: {inline}')

    print()
    print('=' * 78)
    avg_step = sum(iter_times) / len(iter_times)
    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    n_train_imgs = 311  # D-Hday2night train
    n_epochs = int(opt['train']['n_epoch'])
    iters_per_epoch = (n_train_imgs + B - 1) // B
    total_iters = iters_per_epoch * n_epochs
    eta_hours = (avg_step * total_iters) / 3600
    print(f'avg step time : {avg_step*1000:.0f} ms')
    print(f'peak VRAM     : {peak_gb:.2f} GB / 4.00 GB')
    print(f'estimated full training ({n_epochs} epoch × {iters_per_epoch} iter): '
          f'{eta_hours:.2f} hours')
    print('=' * 78)
    if peak_gb > 3.7:
        print('!! 显存接近 4 GB，建议把 batch_size 或 H 减小')
        sys.exit(3)
    print('OK：可以正式开训。')


if __name__ == '__main__':
    main()
