"""单图推理脚本。用法:
    .venv/Scripts/python.exe scripts/infer_single.py \
        --checkpoint experiments/train_mdg_ablation_A_full_<ts>/checkpoint/30_MDGNetwork.pth \
        --input TestData/Hday2night/composite_images_test/d1048-20120628-200951_1_1.jpg \
        --mask TestData/Hday2night/masks/d1048-20120628-200951_1.png \
        --output output.jpg \
        --steps 200
"""
import argparse
import torch
from PIL import Image
from torchvision.transforms import functional as tf
from torchvision import transforms

from models.network_mdg import MDGNetwork
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/ablation_A_full_train.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--mask")
    parser.add_argument("--output", default="output.jpg")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载配置和模型
    with open(args.config) as f:
        cfg = json.load(f)
    net_args = cfg["model"]["which_networks"][0]["args"]

    net = MDGNetwork(**net_args)
    net.set_new_noise_schedule(device=device, phase="test")

    ckpt = torch.load(args.checkpoint, map_location=device)
    missing, unexpected = net.load_state_dict(ckpt, strict=False)
    print(f"Loaded checkpoint: {len(missing)} missing, {len(unexpected)} unexpected")

    net = net.to(device)
    net.eval()

    # 加载图片
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    comp = Image.open(args.input).convert("RGB")
    comp_t = transform(comp).unsqueeze(0).to(device)

    if args.mask:
        mask = Image.open(args.mask).convert("1")
        mask_t = tf.to_tensor(tf.resize(mask, [256, 256])).unsqueeze(0).to(device)
    else:
        mask_t = torch.ones(1, 1, 256, 256, device=device)

    print(f"Input: {comp_t.shape}, Mask: {mask_t.shape}")

    # 推理
    with torch.no_grad():
        output, ret_arr = net.restoration(
            comp_t,
            y_t=comp_t,  # 从合成图开始（harmonization）
            y_0=comp_t,   # repaint 用背景替换
            mask=mask_t,
            sample_num=2,
        )

    # 反归一化保存
    def unnorm(t: torch.Tensor) -> torch.Tensor:
        return (t * 0.5 + 0.5).clamp(0, 1)

    out_img = unnorm(output[0]).cpu()
    out_pil = tf.to_pil_image(out_img)
    out_pil.save(args.output)
    print(f"Saved to {args.output}")

    # 同时保存输入图对照
    in_img = unnorm(comp_t[0]).cpu()
    in_path = args.output.rsplit(".", 1)[0] + "_input.jpg"
    tf.to_pil_image(in_img).save(in_path)
    print(f"Input saved to {in_path}")


if __name__ == "__main__":
    main()
