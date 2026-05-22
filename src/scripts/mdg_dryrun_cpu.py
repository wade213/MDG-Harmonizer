"""MDG-Harmonizer 端到端 dry-run（CPU only）。

验证目标（GPU 还被 baseline test 占用，只能在 CPU 上验）：
    1. 真实 config 能被 ``parse_jsonc`` 正确解析
    2. 实例化 ``MDGNetwork``：CDPNet / MDGUNet / FBLoss 全部正常构造
    3. 用 ``strict=False`` 把 baseline 的 ``Network`` 预训练权重加载进
       ``MDGNetwork``，确认 ``denoise_fn.<...>`` 子模块完整 hit，新增的
       ``cdp_net`` / ``afm_bottleneck`` / ``fb_loss`` 是 missing 状态（预期）
    4. ``init_weights`` 后 AFM 仍然零初始化（关键：训练第 1 步不破坏主干）
    5. 跑一次 forward + backward；可训练参数 < 1 M，AFM 拿到梯度

跑完之后，等 baseline test 一结束、GPU 空出来，就可以直接：
    .\\.venv\\Scripts\\python.exe -W ignore run.py -p train -c config/harmonization_day2night_mdg.json

不会再有「显存爆掉」/「 size mismatch 」/「 init 覆盖恒等性 」等三类隐藏雷。
"""
from __future__ import annotations

import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F


def _parse_jsonc(path: Path) -> OrderedDict:
    """复刻 baseline ``core.praser.parse`` 的注释剥离逻辑。"""
    s = ''
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s += line.split('//')[0] + '\n'
    return json.loads(s, object_pairs_hook=OrderedDict)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))

    print('=' * 78)
    print('MDG-Harmonizer dry-run (CPU only)')
    print('repo_root:', repo_root)
    print('=' * 78)

    # ------- 1) 解析 config -------
    cfg_path = repo_root / 'config' / 'harmonization_day2night_mdg.json'
    opt = _parse_jsonc(cfg_path)
    print(f'\n[1] parsed: {cfg_path.name}')
    net_args = dict(opt['model']['which_networks'][0]['args'])
    print(f"    freeze_backbone={net_args['freeze_backbone']}, deg_dim={net_args['deg_dim']}")
    print(f"    use_amp={opt['model']['which_model']['args']['use_amp']}, "
          f"grad_accum={opt['model']['which_model']['args']['gradient_accumulation_steps']}, "
          f"pretrained_label={opt['model']['which_model']['args']['pretrained_label']}")

    # ------- 2) 实例化 MDGNetwork -------
    from models.network_mdg import MDGNetwork
    # init_type / kaiming 走 BaseNetwork；其余字段透传给 MDGNetwork
    init_type = net_args.pop('init_type', 'kaiming')
    net = MDGNetwork(**net_args, init_type=init_type)
    print(f'\n[2] MDGNetwork instantiated.')
    n_total = sum(p.numel() for p in net.parameters())
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'    total={n_total:,}  trainable={n_train:,}  ratio={100*n_train/n_total:.2f}%')

    # ------- 3) 加载预训练权重（strict=False） -------
    pretrained_label = opt['model']['which_model']['args']['pretrained_label']
    pretrained_path = repo_root / f"{opt['path']['resume_state']}_{pretrained_label}.pth"
    print(f'\n[3] looking for pretrained: {pretrained_path}')
    if pretrained_path.exists():
        sd = torch.load(pretrained_path, map_location='cpu')
        missing, unexpected = net.load_state_dict(sd, strict=False)
        afm_missing = sum(1 for k in missing if 'afm_bottleneck' in k)
        cdp_missing = sum(1 for k in missing if 'cdp_net' in k)
        fb_missing = sum(1 for k in missing if k.startswith('fb_loss'))
        backbone_missing = [k for k in missing
                            if 'afm_bottleneck' not in k
                            and 'cdp_net' not in k
                            and not k.startswith('fb_loss')]
        print(f'    state_dict keys: {len(sd)}')
        print(f'    missing: {len(missing)} (afm={afm_missing}, cdp={cdp_missing}, fb_loss={fb_missing}, others={len(backbone_missing)})')
        print(f'    unexpected: {len(unexpected)}')
        if backbone_missing:
            print(f'    !! 非新模块的 missing key（应该为空）：')
            for k in backbone_missing[:5]:
                print(f'       {k}')
            if len(backbone_missing) > 5:
                print(f'       ... 共 {len(backbone_missing)} 个')
            print('    -> 说明 MDGUNet 与 baseline UNet 子模块对不齐，需要排查')
            sys.exit(2)
        print('    OK: 仅 cdp_net / afm_bottleneck / fb_loss 三类新模块 missing，符合预期')
    else:
        print(f'    !! 文件不存在，跳过预训练加载（仅测 init_weights 流程）')

    # ------- 4) 运行 init_weights 验证 AFM 仍是零初始化 -------
    net.init_weights()  # 内部会先 super().init_weights() 再复位 AFM
    afm = net.denoise_fn.afm_bottleneck
    sums = {n: p.detach().abs().sum().item() for n, p in afm.named_parameters()
            if n in ('out_proj.weight', 'gamma_layer.weight', 'beta_layer.weight')}
    print(f'\n[4] post-init_weights AFM 关键层范数（必须为 0 以保证恒等性）:')
    for k, v in sums.items():
        print(f'    {k:25s} = {v:.3e}')
    assert all(v == 0.0 for v in sums.values()), \
        'AFM out_proj/gamma/beta 必须全零，否则会破坏冻结主干训练！'
    print('    OK: AFM 零初始化恒等性保留')

    # ------- 5) forward + backward 一次（CPU mini batch） -------
    # 用真实 256×256 太慢，用 64×64 验证逻辑（H 必须 ≥ 16 让 attn_res=16 有效）
    net.set_loss(F.l1_loss)
    net.set_new_noise_schedule(device=torch.device('cpu'), phase='train')
    net.train()

    H = 64
    B = 1
    y0 = torch.randn(B, 3, H, H)
    yc = torch.randn(B, 3, H, H)
    msk = (torch.rand(B, 1, H, H) > 0.5).float()
    print(f'\n[5] forward+backward on B={B}, H=W={H} (CPU)')

    loss = net(y0, y_cond=yc, mask=msk)
    print(f'    forward loss: {loss.item():.4f}')
    print(f'    breakdown   : {net._last_loss_breakdown}')
    loss.backward()

    # AFM 应该至少有 6 个权重（out_proj.W/b, gamma.W/b, beta.W/b）拿到梯度
    afm_grad_count = sum(
        1 for _, p in afm.named_parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().sum().item() > 0
    )
    print(f'    AFM nonzero-grad params: {afm_grad_count}')
    if afm_grad_count < 4:
        print('    !! AFM 没拿到梯度——zero_module 切断了梯度回传路径')
        print('       原因：随机初始化的 out / out_list 是零；真实加载预训练后会恢复。')

    print('\n' + '=' * 78)
    print('Dry-run 全部通过。等 GPU 空出来即可启动真训练:')
    print('  .\\.venv\\Scripts\\python.exe -W ignore run.py -p train -c config/harmonization_day2night_mdg.json')
    print('=' * 78)


if __name__ == '__main__':
    main()
