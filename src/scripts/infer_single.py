"""单图推理脚本（支持 MDG、PromptRouterMDG 等动态网络）。用法:
    .venv/Scripts/python.exe scripts/infer_single.py \
        --config config/ablation_A_full_train.json \
        --checkpoint experiments/.../checkpoint/30_MDGNetwork.pth \
        --input TestData/Hday2night/composite_images_test/d1048...jpg \
        --mask TestData/Hday2night/masks/d1048...png \
        --output output.jpg \
        --steps 200 \
        --gpu

使用 Prompt Router 网络时，会额外输出 prompt_weights.json。
"""
import argparse
import json
import torch
from PIL import Image
from torchvision.transforms import functional as tf
from torchvision import transforms


def _resolve_network_class(name: list[str]):
    """动态加载网络类。name = [module_file, class_name]"""
    mod = __import__(name[0], fromlist=[name[1]])
    return getattr(mod, name[1])


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

    # 加载配置
    with open(args.config) as f:
        cfg = json.load(f)

    # 动态创建网络（不再硬编码 MDGNetwork）
    net_spec = cfg["model"]["which_networks"][0]
    net_args = dict(net_spec["args"])
    NetworkClass = _resolve_network_class(net_spec["name"])
    net = NetworkClass(**net_args)
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
            y_t=comp_t,
            y_0=comp_t,
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

    in_img = unnorm(comp_t[0]).cpu()
    in_path = args.output.rsplit(".", 1)[0] + "_input.jpg"
    tf.to_pil_image(in_img).save(in_path)
    print(f"Input saved to {in_path}")

    # Prompt Router 额外输出
    prompt_aux = getattr(net, "_last_prompt_aux", None)
    if prompt_aux and "prompt_weights" in prompt_aux:
        w = prompt_aux["prompt_weights"][0].cpu().tolist()
        labels = prompt_aux.get("descriptor_labels", [f"P{i}" for i in range(len(w))])
        prompt_out = {labels[i]: w[i] for i in range(len(w))}
        prompt_path = args.output.rsplit(".", 1)[0] + "_prompt.json"
        with open(prompt_path, "w") as f:
            json.dump(prompt_out, f, indent=2)
        print(f"Prompt weights saved to {prompt_path}")
        print(f"  {prompt_out}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    main()
