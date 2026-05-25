r"""MDG-Harmonizer 图像协调演示系统。

支持 MDG-D（工作一）和 M-DPR（工作二）三种模式推理，
并展示退化提示权重可视化。

用法:
    cd src && python ../system/app_gradio.py
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as tf
from torchvision import transforms

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DESCRIPTOR_LABELS = [
    "brightness", "color_shift", "saturation", "contrast",
    "boundary", "blur", "noise", "texture",
]

MODEL_PRESETS = {
    "MDG-D (CDP + AFM)": {
        "config": "config/ablation_D_no_fb_test.json",
        "checkpoint": "experiments/train_mdg_ablation_D_no_fb_260524_105447/checkpoint/30_MDGNetwork.pth",
        "description": "工作一最佳消融：CDP-Net + AFM，冻结 backbone，~0.6M 可训参数",
    },
    "M-DPR Router only": {
        "config": "config/ablation_A_full_test.json",
        "checkpoint": "experiments/train_prompt_router_m-dpr_descriptor_260525_163654/checkpoint/5_PromptRouterMDGNetwork.pth",
        "network": ["models.network_prompt_router_mdg", "PromptRouterMDGNetwork"],
        "network_kwargs": {"prompt_mode": "descriptor_only", "prompt_dim": 64, "freeze_backbone": True},
        "description": "工作二独立版本：仅退化描述子 + Prompt Router，~1K 可训参数",
    },
    "M-DPR Hybrid": {
        "config": "config/ablation_A_full_test.json",
        "checkpoint": "experiments/train_prompt_router_m-dpr_hybrid_260525_163654/checkpoint/5_PromptRouterMDGNetwork.pth",
        "network": ["models.network_prompt_router_mdg", "PromptRouterMDGNetwork"],
        "network_kwargs": {"prompt_mode": "hybrid", "prompt_dim": 64, "freeze_backbone": True},
        "description": "工作一 + 工作二融合：CDP-Net + Prompt Router",
    },
}

# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def _resolve_network_class(name: list[str]):
    mod = __import__(name[0], fromlist=[name[1]])
    return getattr(mod, name[1])


def run_inference(
    composite_img: np.ndarray,
    mask_img: np.ndarray,
    model_name: str,
    steps: int,
    progress=gr.Progress(),
) -> tuple[Image.Image, Image.Image, plt.Figure, str]:
    """执行单图推理，返回 (result, composite, prompt_fig, info_text)。"""

    preset = MODEL_PRESETS[model_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load network ---
    config_path = preset["config"]
    checkpoint_path = preset["checkpoint"]

    with open(config_path) as f:
        cfg = json.load(f)

    # Override resume_state
    cfg["path"]["resume_state"] = checkpoint_path
    cfg["model"]["which_networks"][0]["args"]["beta_schedule"]["test"]["n_timestep"] = steps

    net_spec = preset.get("network")
    if net_spec is not None:
        cfg["model"]["which_networks"][0]["name"] = net_spec
        for k, v in preset.get("network_kwargs", {}).items():
            cfg["model"]["which_networks"][0]["args"][k] = v

    net_args = cfg["model"]["which_networks"][0]["args"]
    NetworkClass = _resolve_network_class(cfg["model"]["which_networks"][0]["name"])
    net = NetworkClass(**net_args)
    net.set_new_noise_schedule(device=device, phase="test")

    ckpt = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = net.load_state_dict(ckpt, strict=False)
    net = net.to(device)
    net.eval()

    # --- Prepare inputs ---
    transform_fn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Convert numpy/PIL to tensor
    comp_np = composite_img if isinstance(composite_img, np.ndarray) else np.array(composite_img)
    comp_pil = Image.fromarray(comp_np).convert("RGB")
    comp_t = transform_fn(comp_pil).unsqueeze(0).to(device)

    mask_np = mask_img if isinstance(mask_img, np.ndarray) else np.array(mask_img)
    if mask_np.ndim == 3 and mask_np.shape[2] > 1:
        mask_np = mask_np[:, :, 0]
    mask_pil = Image.fromarray(mask_np).convert("1")
    mask_t = tf.to_tensor(tf.resize(mask_pil, [256, 256])).unsqueeze(0).to(device)

    # --- Inference ---
    with torch.no_grad():
        output, _ = net.restoration(comp_t, y_t=comp_t, y_0=comp_t, mask=mask_t, sample_num=2)

    def unnorm(t: torch.Tensor) -> torch.Tensor:
        return (t * 0.5 + 0.5).clamp(0, 1)

    out_img = unnorm(output[0]).cpu()
    result_pil = tf.to_pil_image(out_img)

    # --- Prompt weights ---
    prompt_aux = getattr(net, "_last_prompt_aux", {})
    prompt_weights = prompt_aux.get("prompt_weights")
    descriptor = prompt_aux.get("descriptor")
    labels = prompt_aux.get("descriptor_labels", DESCRIPTOR_LABELS)

    if prompt_weights is not None and prompt_weights.numel() > 0:
        w = prompt_weights[0].cpu().numpy()
        fig = _make_bar_chart(w, labels)
        info = _make_info_text(w, labels, preset, steps, len(missing))
    else:
        fig = _make_placeholder_chart()
        info = f"**{model_name}**\n{preset['description']}\n步数: {steps} | 设备: {device}\n(该模式无 prompt 权重输出)"

    return result_pil, comp_pil, fig, info


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
matplotlib.use("Agg")

def _make_bar_chart(weights: np.ndarray, labels: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = plt.cm.Blues(0.3 + 0.6 * weights / (weights.max() + 1e-6))
    bars = ax.barh(labels, weights, color=colors, edgecolor="steelblue")
    ax.set_xlabel("Prompt Weight")
    ax.set_title("Degradation Prompt Weights")
    ax.set_xlim(0, max(weights.max() * 1.2, 0.5))
    for bar, val in zip(bars, weights):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    return fig


def _make_placeholder_chart() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, "No prompt weights", ha="center", va="center", fontsize=14, color="gray")
    ax.set_title("Degradation Prompt Weights")
    plt.tight_layout()
    return fig


def _make_info_text(weights: np.ndarray, labels: list[str], preset: dict, steps: int, missing: int) -> str:
    lines = [
        f"**{preset.get('description', '')}**",
        f"",
        f"采样步数: {steps}  |  参数: {sum(weights):.2f}  (top-3: {', '.join(labels[i] for i in np.argsort(-weights)[:3])})",
        f"",
    ]
    for i, (label, val) in enumerate(zip(labels, weights)):
        lines.append(f"- {label}: {val:.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def create_ui():
    with gr.Blocks(title="MDG-Harmonizer") as demo:
        gr.Markdown("# MDG-Harmonizer 图像协调演示系统")
        gr.Markdown(
            "上传合成图和前景掩码，选择模型模式和采样步数，点击运行进行图像协调推理。"
            "使用 M-DPR 模式时可查看退化提示权重可视化。"
        )

        with gr.Row():
            with gr.Column(scale=1):
                composite_in = gr.Image(label="Composite Image", type="numpy", height=256)
                mask_in = gr.Image(label="Foreground Mask", type="numpy", height=256)
                model_choice = gr.Dropdown(
                    choices=list(MODEL_PRESETS.keys()),
                    value=list(MODEL_PRESETS.keys())[0],
                    label="Model",
                )
                steps_slider = gr.Slider(50, 200, value=200, step=50, label="DDPM Steps")
                run_btn = gr.Button("Run", variant="primary")

            with gr.Column(scale=1):
                result_out = gr.Image(label="Harmonized Result", type="pil")
                info_out = gr.Markdown("")

        with gr.Row():
            prompt_chart = gr.Plot(label="Prompt Weights")

        # Examples
        gr.Examples(
            examples=[
                ["system/examples/composite.jpg", "system/examples/mask.png"],
            ],
            inputs=[composite_in, mask_in],
            label="Example (请替换为你的样例图)",
        )

        run_btn.click(
            fn=run_inference,
            inputs=[composite_in, mask_in, model_choice, steps_slider],
            outputs=[result_out, composite_in, prompt_chart, info_out],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = create_ui()
    demo.launch(server_port=args.port, share=args.share)
