"""DPM-Solver++ speed benchmark for HCDM Day2Night Harmonization.

Compares DDPM-1000 vs DPM-Solver++-25 on the first 5 D-Hday2night test images.

Usage:
    python test_dpm_speed.py
    python test_dpm_speed.py --checkpoint experiments/train_mdg_harmonizer_day2night_260506_172332/checkpoint/30
    python test_dpm_speed.py --steps 25 --num-images 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def psnr(pred: torch.Tensor, gt: torch.Tensor, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio (higher is better)."""
    mse = F.mse_loss(pred, gt).item()
    if mse == 0:
        return float('inf')
    return float(20 * np.log10(max_val) - 10 * np.log10(mse))


def load_image(path: str, size: int = 256) -> torch.Tensor:
    """Load and normalise image to [-1, 1]."""
    img = Image.open(path).convert('RGB')
    img = transforms.Resize((size, size))(img)
    img = transforms.ToTensor()(img)
    img = img * 2.0 - 1.0  # [-1, 1]
    return img


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='DPM-Solver++ speed benchmark')
    parser.add_argument('--checkpoint', type=str,
                        default='experiments/train_mdg_harmonizer_day2night_260506_172332/checkpoint/30',
                        help='Path to checkpoint (without extension, e.g. checkpoint/30)')
    parser.add_argument('--config', type=str,
                        default='config/harmonization_day2night_mdg.json',
                        help='Config JSON for model construction')
    parser.add_argument('--data-root', type=str,
                        default='./TestData/Hday2night/composite_images_test/',
                        help='Test data root')
    parser.add_argument('--num-images', type=int, default=5,
                        help='Number of test images')
    parser.add_argument('--steps', type=int, default=25,
                        help='DPM-Solver steps')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    # --- Device ---
    if not torch.cuda.is_available():
        print('ERROR: CUDA not available. This benchmark requires a GPU.')
        sys.exit(1)
    device = torch.device('cuda')
    torch.manual_seed(args.seed)

    # --- Parse config ---
    with open(args.config, 'r') as f:
        json_str = ''
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
        opt = json.loads(json_str)

    # --- Build network ---
    print('Building MDGNetwork ...')
    from models.network_mdg import MDGNetwork

    net_cfg = opt['model']['which_networks'][0]['args']
    netG = MDGNetwork(**net_cfg).to(device)
    netG.eval()

    # --- Load checkpoint ---
    pth_path = args.checkpoint + '_MDGNetwork.pth'
    state_path = args.checkpoint + '.state'

    if not os.path.exists(pth_path):
        print(f'ERROR: Checkpoint not found at {pth_path}')
        print('Available checkpoints:')
        ckpt_dir = os.path.dirname(args.checkpoint)
        if os.path.isdir(ckpt_dir):
            for f in sorted(os.listdir(ckpt_dir)):
                print(f'  {f}')
        sys.exit(1)

    print(f'Loading checkpoint: {pth_path}')
    ckpt = torch.load(pth_path, map_location=device)
    netG.load_state_dict(ckpt, strict=False)
    print('Checkpoint loaded (strict=False).')

    # Also load .state for optimiser/scheduler (not needed for inference)
    if os.path.exists(state_path):
        print(f'Found companion .state file (skipped for inference).')

    # --- Set noise schedule (test config) ---
    netG.set_new_noise_schedule(device=device, phase='test')
    gammas_np = netG.gammas.cpu().numpy()  # alphas_cumprod

    # --- Load test images ---
    print(f'Loading {args.num_images} test images from {args.data_root} ...')
    data_root = Path(args.data_root)

    # Hday2night test structure:
    #   TestData/Hday2night/composite_images_test/  (composites)
    #   TestData/Hday2night/real_images/              (ground truth)
    #   TestData/Hday2night/masks/                    (foreground masks)
    # data_root may point to composite_images_test/ or Hday2night/
    parent = data_root.parent if data_root.name in ('composite_images_test', 'composite_images') else data_root
    # Determine composite image directory
    comp_candidates = [data_root / 'composite_images_test', data_root / 'composite_images', data_root]
    comp_dir = next((d for d in comp_candidates if d.is_dir() and any(d.glob('*'))), data_root)
    real_dir = parent / 'real_images'
    mask_dir = parent / 'masks'
    # Also try sibling paths in case parent != data_root
    for d in [data_root / 'real_images', data_root / 'masks']:
        if d.is_dir():
            if d.name == 'real_images':
                real_dir = d
            elif d.name == 'masks':
                mask_dir = d

    # Find image files
    comp_files = sorted([str(p) for p in comp_dir.glob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])

    # Limit
    comp_files = comp_files[:max(args.num_images, 1)]
    print(f'  Found {len(comp_files)} test images.')

    # --- Benchmark ---
    ddpm_times = []
    dpm_times = []
    psnr_ddpm_list = []
    psnr_dpm_list = []

    print(f'\n{"="*60}')
    print(f'Benchmark: DDPM-{netG.num_timesteps} vs DPM-Solver++-{args.steps}')
    print(f'{"="*60}')

    from models.dpm_solver import dpm_solver_restoration

    for idx, comp_path in enumerate(comp_files[:args.num_images]):
        print(f'\n--- Image {idx+1}/{args.num_images}: {os.path.basename(comp_path)} ---')

        # Load composite image
        comp_img = load_image(comp_path).unsqueeze(0).to(device)

        # Try to load real image (ground truth) for PSNR
        real_path = None
        if real_dir.exists():
            basename = os.path.basename(comp_path)
            # Hday2night naming: composite_images/xxx.jpg → real_images/xxx.jpg
            candidate = real_dir / basename
            if candidate.exists():
                real_path = str(candidate)

        real_img = None
        if real_path:
            real_img = load_image(real_path).unsqueeze(0).to(device)

        # Load mask if available
        mask = None
        if mask_dir.exists():
            basename = os.path.basename(comp_path)
            mask_candidate = mask_dir / basename
            if mask_candidate.exists():
                mask = load_image(str(mask_candidate)).unsqueeze(0).to(device)
                # Convert RGB mask to single-channel
                if mask.shape[1] == 3:
                    mask = mask[:, 0:1, :, :]

        # Generate random noise (fixed for fair comparison)
        noise = torch.randn_like(comp_img)

        # ---- DDPM ----
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            out_ddpm, _ = netG.restoration(
                comp_img, y_t=noise.clone(),
                y_0=real_img if real_img is not None else comp_img,
                mask=mask, sample_num=2,
            )

        torch.cuda.synchronize()
        ddpm_time = time.perf_counter() - t0
        ddpm_times.append(ddpm_time)
        print(f'  DDPM-{netG.num_timesteps}: {ddpm_time:.2f}s')

        # ---- DPM-Solver++ ----
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            out_dpm, _ = netG.restoration_dpm(
                comp_img, y_t=noise.clone(),
                y_0=real_img if real_img is not None else comp_img,
                mask=mask, sample_num=2,
                dpm_steps=args.steps,
            )

        torch.cuda.synchronize()
        dpm_time = time.perf_counter() - t0
        dpm_times.append(dpm_time)
        print(f'  DPM-Solver++-{args.steps}: {dpm_time:.2f}s')
        print(f'  Speedup: {ddpm_time / max(dpm_time, 0.001):.1f}×')

        # ---- PSNR comparison ----
        if real_img is not None:
            psnr_ddpm = psnr(out_ddpm.clamp(-1, 1), real_img)
            psnr_dpm = psnr(out_dpm.clamp(-1, 1), real_img)
            psnr_ddpm_list.append(psnr_ddpm)
            psnr_dpm_list.append(psnr_dpm)
            print(f'  PSNR: DDPM={psnr_ddpm:.2f} dB, DPM={psnr_dpm:.2f} dB, Δ={psnr_dpm - psnr_ddpm:.2f} dB')

    # === Summary ===
    print(f'\n{"="*60}')
    print('SUMMARY')
    print(f'{"="*60}')
    print(f'{"Method":<25} {"Time (s)":>10} {"Speedup":>10}')
    print(f'{"-"*45}')
    avg_ddpm = np.mean(ddpm_times)
    avg_dpm = np.mean(dpm_times)
    speedup = avg_ddpm / max(avg_dpm, 0.001)
    print(f'{"DDPM-" + str(netG.num_timesteps):<25} {avg_ddpm:>10.2f} {"1.0×":>10}')
    print(f'{"DPM-Solver++-" + str(args.steps):<25} {avg_dpm:>10.2f} {speedup:>9.1f}×')

    if psnr_ddpm_list:
        avg_psnr_ddpm = np.mean(psnr_ddpm_list)
        avg_psnr_dpm = np.mean(psnr_dpm_list)
        delta = avg_psnr_dpm - avg_psnr_ddpm
        print(f'\n{"Method":<25} {"PSNR (dB)":>10}')
        print(f'{"-"*35}')
        print(f'{"DDPM":<25} {avg_psnr_ddpm:>10.2f}')
        print(f'{"DPM-Solver++":<25} {avg_psnr_dpm:>10.2f}')
        print(f'{"Δ (DPM - DDPM)":<25} {delta:>+10.2f}')

    # Check acceptance criteria (avoid emoji on Windows GBK)
    print(f'\n--- Acceptance Criteria ---')
    speed_ok = speedup >= 20
    print(f'  Speedup >= 20x: {"PASS" if speed_ok else "FAIL"} ({speedup:.1f}x)')
    if psnr_ddpm_list:
        delta = avg_psnr_dpm - avg_psnr_ddpm
        psnr_ok = abs(delta) < 0.5
        print(f'  PSNR drop < 0.5 dB: {"PASS" if psnr_ok else "FAIL"} ({delta:+.2f} dB)')


if __name__ == '__main__':
    main()
