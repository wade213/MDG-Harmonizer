"""Quick 3-image × 3-setting ablation preview for Figure 4-1."""
import sys, os, json, glob
sys.path.insert(0, ".")

import torch
import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision import transforms

def _resolve(name):
    mod = __import__(name[0], fromlist=[name[1]])
    return getattr(mod, name[1])

def load_model(config_path, ckpt_path, device, test_steps=200):
    with open(config_path) as f:
        cfg = json.load(f)
    net_spec = cfg["model"]["which_networks"][0]
    net_args = dict(net_spec["args"])
    NetCls = _resolve(net_spec["name"])
    # Build with checkpoint-compatible schedule first
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_n = ckpt["gammas"].shape[0]
    net_args["beta_schedule"]["test"]["n_timestep"] = ckpt_n
    net_args["beta_schedule"]["train"]["n_timestep"] = ckpt_n
    net = NetCls(**net_args)
    net.set_new_noise_schedule(device=device, phase="test")
    net.load_state_dict(ckpt, strict=False)
    # Rebuild with desired test steps
    net.beta_schedule["test"]["n_timestep"] = test_steps
    net.set_new_noise_schedule(device=device, phase="test")
    net.to(device).eval()
    return net

def infer(net, comp_pil, mask_pil, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    comp = comp_pil.convert("RGB").resize((256, 256), Image.BICUBIC)
    comp_t = transform(comp).unsqueeze(0).to(device)
    mask = mask_pil.convert("1").resize((256, 256), Image.NEAREST)
    mask_t = TF.to_tensor(mask).unsqueeze(0).to(device)
    with torch.no_grad():
        out, _ = net.restoration(comp_t, y_t=comp_t, y_0=comp_t, mask=mask_t, sample_num=2)
    unnorm = lambda t: (t * 0.5 + 0.5).clamp(0, 1)
    return TF.to_pil_image(unnorm(out[0]).cpu())

# --- Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_ROOT = "D-Hday2night"
OUT_DIR = "figs/fig41_preview"
os.makedirs(OUT_DIR, exist_ok=True)

# Pick 3 representative samples from metric analysis
PICKS = ["d90000004-27_1_5", "d90000012-181_1_3", "d18959-20120615-084228_1_6"]
comp_files = [f"{TEST_ROOT}/composite_images_test/{p}.jpg" for p in PICKS]
print(f"Selected: {PICKS}")

# Model configs: (name, config, checkpoint)
MODELS = [
    ("B_no_cdp",  "config/fig41_test_B_no_cdp.json",  "experiments/train_mdg_ablation_B_no_cdp_260521_102604/checkpoint/15_MDGNetwork.pth"),
    ("C_no_afm",  "config/fig41_test_C_no_afm.json",  "experiments/train_mdg_ablation_C_no_afm_260523_142638/checkpoint/30_MDGNetwork.pth"),
    ("D_no_fb",   "config/fig41_test_D_no_fb.json",   "experiments/train_mdg_ablation_D_no_fb_260523_224814/checkpoint/30_MDGNetwork.pth"),
]

# Load all models
nets = {}
for name, cfg, ckpt in MODELS:
    print(f"Loading {name}...")
    nets[name] = load_model(cfg, ckpt, device)
print("All models loaded.")

# Run inference
results = []  # list of (comp_pil, gt_pil, mask_pil, {name: out_pil})
for comp_path in comp_files:
    fname = os.path.basename(comp_path)
    stem = fname.replace(".jpg", "")
    # Derive mask and GT paths
    # composite: d1048-20120628-200951_1_1.jpg -> mask: d1048-20120628-200951_1.png, gt: d1048-20120628-200951.jpg
    parts = stem.rsplit("_", 1)
    mask_base = parts[0]
    gt_base = mask_base.rsplit("_", 1)[0]
    mask_path = f"{TEST_ROOT}/masks/{mask_base}.png"
    gt_path = f"{TEST_ROOT}/real_images/{gt_base}.jpg"

    comp_pil = Image.open(comp_path)
    mask_pil = Image.open(mask_path) if os.path.exists(mask_path) else Image.new("1", comp_pil.size)
    gt_pil = Image.open(gt_path) if os.path.exists(gt_path) else Image.new("RGB", comp_pil.size)

    outs = {}
    for name, _, _ in MODELS:
        print(f"  {fname} × {name}...")
        outs[name] = infer(nets[name], comp_pil, mask_pil, device)

    results.append((comp_pil.resize((256,256)), gt_pil.resize((256,256)), mask_pil.resize((256,256)), outs, fname))

# PSNR computation
def compute_psnr(img1, img2):
    import numpy as np
    a = np.array(img1, dtype=np.float64)
    b = np.array(img2, dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    if mse < 1e-10:
        return 99.0
    return 10 * np.log10(255.0**2 / mse)

# Compute PSNR for each output vs GT
psnr_data = []
for comp, gt, mask, outs, fname in results:
    row_psnr = {}
    for name in ["B_no_cdp", "C_no_afm", "D_no_fb"]:
        row_psnr[name] = compute_psnr(np.array(gt), np.array(outs[name]))
    psnr_data.append(row_psnr)

# Compose figure: rows=samples, cols=[Composite, GT, B, C, D]
COL_LABELS = ["Composite", "GT", "B: no CDP", "C: no AFM", "D: CDP+AFM"]
n_rows = len(results)
n_cols = 5
CELL = 256
gap = 6

canvas_w = n_cols * CELL + (n_cols - 1) * gap
canvas_h = n_rows * CELL + (n_rows - 1) * gap + 44
canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

from PIL import ImageDraw, ImageFont
import numpy as np
draw = ImageDraw.Draw(canvas)

try:
    font = ImageFont.truetype("arial.ttf", 16)
    font_sm = ImageFont.truetype("arial.ttf", 13)
except:
    font = ImageFont.load_default()
    font_sm = font

for j, label in enumerate(COL_LABELS):
    x = j * (CELL + gap) + CELL // 2
    draw.text((x, 12), label, fill=(0, 0, 0), font=font, anchor="mt")

# Images with PSNR overlay
MODEL_KEYS = ["B_no_cdp", "C_no_afm", "D_no_fb"]
for i, (comp, gt, mask, outs, fname) in enumerate(results):
    y_off = 44 + i * (CELL + gap)
    row_imgs = [comp, gt, outs["B_no_cdp"], outs["C_no_afm"], outs["D_no_fb"]]
    for j, img in enumerate(row_imgs):
        x = j * (CELL + gap)
        canvas.paste(img, (x, y_off))
        # PSNR label on B/C/D columns
        if j >= 2:
            key = MODEL_KEYS[j - 2]
            psnr_val = psnr_data[i][key]
            label = f"PSNR {psnr_val:.1f}"
            # Background rectangle for readability
            bbox = draw.textbbox((x + 4, y_off + 4), label, font=font_sm)
            draw.rectangle([bbox[0]-2, bbox[1]-1, bbox[2]+2, bbox[3]+1], fill=(0, 0, 0))
            draw.text((x + 4, y_off + 4), label, fill=(255, 255, 255), font=font_sm)

out_path = f"{OUT_DIR}/fig41_preview.png"
canvas.save(out_path, quality=95)
print(f"\nSaved to {out_path}")
print(f"Grid: {n_rows} rows × {n_cols} cols, {canvas_w}×{canvas_h}px")
# Print PSNR summary
for i, (comp, gt, mask, outs, fname) in enumerate(results):
    ps = psnr_data[i]
    print(f"  {fname}: B={ps['B_no_cdp']:.2f}  C={ps['C_no_afm']:.2f}  D={ps['D_no_fb']:.2f}  (D-B={ps['D_no_fb']-ps['B_no_cdp']:+.2f}  D-C={ps['D_no_fb']-ps['C_no_afm']:+.2f})")
