"""Test DDIM-style subsampled sampling with real masks."""
import json, sys, time, re
import numpy as np
import torch
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

def load_jsonc(path):
    txt = ""
    with open(path) as f:
        for line in f:
            txt += line.split("//")[0] + "\n"
    return json.loads(txt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

cfg = load_jsonc(REPO / "config" / "harmonization_day2night_mdg.json")
from models.network_mdg import MDGNetwork
net_cfg = cfg["model"]["which_networks"][0]["args"]
netG = MDGNetwork(**net_cfg).to(device)
netG.eval()

ckpt_path = REPO / "experiments/train_mdg_harmonizer_day2night_260511_065422/checkpoint/30_MDGNetwork.pth"
ckpt = torch.load(ckpt_path, map_location=device)
netG.load_state_dict(ckpt, strict=False)
netG.set_new_noise_schedule(device=device, phase="test")

from torchvision import transforms
from PIL import Image

test_dir = REPO / "TestData/Hday2night/composite_images_test"
gt_dir = REPO / "TestData/Hday2night/real_images"
mask_dir = REPO / "TestData/Hday2night/masks"

comp_files = sorted(test_dir.glob("*.jpg"))[:5]
print(f"Testing on {len(comp_files)} images\n")

def load_mask_for_stem(stem, mask_dir):
    """Match mask: composite stem 'd1048_xxx_1_1' -> mask 'd1048_xxx_1.png'"""
    base = re.sub(r'_\d+$', '', stem.rsplit("_", 1)[0])
    for m_path in mask_dir.glob(f"{base}*.png"):
        return m_path
    return None

results = {}
for comp_path in comp_files:
    stem = comp_path.stem
    img = Image.open(str(comp_path)).convert("RGB")
    img = transforms.Resize((256, 256))(img)
    img_t = transforms.ToTensor()(img) * 2.0 - 1.0
    y_cond = img_t.unsqueeze(0).to(device)

    # Ground truth
    gt_base = re.sub(r'_\d+_\d+$', '', stem)
    gt_path = gt_dir / f"{gt_base}.jpg"
    gt = None
    if gt_path.exists():
        gt_img = Image.open(str(gt_path)).convert("RGB")
        gt_img = transforms.Resize((256, 256))(gt_img)
        gt = transforms.ToTensor()(gt_img).unsqueeze(0).to(device)

    # Mask
    mask_path = load_mask_for_stem(stem, mask_dir)
    if mask_path:
        m = Image.open(str(mask_path)).convert("L")
        m = transforms.Resize((256, 256))(m)
        mask = transforms.ToTensor()(m).unsqueeze(0).to(device)
    else:
        mask = None
        print(f"  WARNING: no mask for {stem}, skipping")
        continue

    torch.manual_seed(42)
    noise = torch.randn_like(y_cond)

    print(f"--- {stem} (mask: {mask_path.name}) ---")
    for steps in [1000, 25, 50, 100]:
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        with torch.no_grad():
            if steps == 1000:
                out, _ = netG.restoration(y_cond, y_t=noise.clone(), y_0=y_cond, mask=mask, sample_num=8)
            else:
                out, _ = netG.restoration_ddim(
                    y_cond, y_t=noise.clone(), y_0=y_cond, mask=mask,
                    ddim_steps=steps, eta=0.0
                )
        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = time.perf_counter() - t0

        out_clip = out.clamp(-1, 1)
        if gt is not None:
            mse = ((out_clip - gt) ** 2).mean().item()
            psnr = 10 * np.log10(2.0 / mse) if mse > 0 else 99.0
        else:
            psnr = -1

        label = "DDPM1k" if steps == 1000 else f"DDIM{steps}"
        print(f"  {label:8s}: PSNR={psnr:.2f} dB  time={elapsed:.1f}s  range=[{out.min():.3f},{out.max():.3f}]")
        results.setdefault(steps, []).append(psnr)

    print()

print("=== Average PSNR ===")
for steps in sorted(results):
    label = "DDPM1k" if steps == 1000 else f"DDIM{steps}"
    avg = np.mean(results[steps])
    print(f"  {label:8s}: {avg:.2f} dB (n={len(results[steps])})")
print("Done.")
