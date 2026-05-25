r"""MDG-Harmonizer 退化感知图像协调与提示分析系统。

支持 MDG-D（工作一）和 M-DPR（工作二）三种模式推理，
并展示退化提示权重可视化与诊断分析。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as tf
from torchvision import transforms

import gradio as gr

_PROJECT_ROOT = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_PROJECT_ROOT))

matplotlib.use("Agg")

# ─── 常量 ──────────────────────────────────────────────
DESCRIPTOR_LABELS_EN = [
    "brightness", "color_shift", "saturation", "contrast",
    "boundary", "blur", "noise", "texture",
]

DESCRIPTOR_LABELS_ZH = {
    "brightness": "亮度不一致",
    "color_shift": "色彩偏移",
    "saturation": "饱和度不一致",
    "contrast": "对比度不一致",
    "boundary": "边界伪影",
    "blur": "模糊差异",
    "noise": "噪声差异",
    "texture": "纹理差异",
}

DESCRIPTOR_EXPLANATIONS = {
    "brightness": "前景与背景存在明显亮度差异，模型倾向于优先调整明暗关系。",
    "color_shift": "前景和背景色调分布不一致，需要进行颜色迁移与校正。",
    "saturation": "前景与背景饱和度不匹配，可能导致目标显得过艳或过灰。",
    "contrast": "局部对比度不一致，可能影响前景与背景的融合感。",
    "boundary": "前景边界区域存在不自然过渡，需要加强边缘协调。",
    "blur": "前景和背景清晰度不一致，可能出现过锐或过糊的问题。",
    "noise": "前景与背景噪声强度不一致，需要匹配成像质量。",
    "texture": "前景和背景纹理统计差异较大，需要增强局部风格一致性。",
}

MODEL_PRESETS = {
    "MDG-D (CDP + AFM)": {
        "config": "config/ablation_D_no_fb_test.json",
        "checkpoint": "experiments/train_mdg_ablation_D_no_fb_260524_105447/checkpoint/30_MDGNetwork.pth",
        "tag": "工作一基线模型",
        "path_desc": "Composite + Mask → CDP-Net → AFM → Harmonization",
        "explain": "用于展示 CDP-AFM 轻量退化感知适配算法，当前最优消融 (PSNR 36.40)。",
    },
    "M-DPR Router only": {
        "config": "config/ablation_A_full_test.json",
        "checkpoint": "experiments/train_prompt_router_m-dpr_descriptor_260525_163654/checkpoint/5_PromptRouterMDGNetwork.pth",
        "network": ["models.network_prompt_router_mdg", "PromptRouterMDGNetwork"],
        "network_kwargs": {"prompt_mode": "descriptor_only", "prompt_dim": 64, "freeze_backbone": True},
        "tag": "工作二独立模型",
        "path_desc": "Composite+Mask → Degradation Descriptor → Prompt Router → Harmonization",
        "explain": "用于展示 M-DPR 不依赖 CDP-Net 的独立退化提示构建能力 (~1K 可训参数)。",
    },
    "M-DPR Hybrid": {
        "config": "config/ablation_A_full_test.json",
        "checkpoint": "experiments/train_prompt_router_m-dpr_hybrid_260525_163654/checkpoint/5_PromptRouterMDGNetwork.pth",
        "network": ["models.network_prompt_router_mdg", "PromptRouterMDGNetwork"],
        "network_kwargs": {"prompt_mode": "hybrid", "prompt_dim": 64, "freeze_backbone": True},
        "tag": "最终融合模型",
        "path_desc": "CDP-Net Prior + Prompt Router Prior → Prior Fusion → Harmonization",
        "explain": "用于展示工作一和工作二的互补关系，适合系统最终演示。",
    },
}

_MODEL_CACHE = {}

# ─── 模型说明卡片 ──────────────────────────────────────────
def get_model_card(model_name: str) -> str:
    preset = MODEL_PRESETS[model_name]
    return f"""### 当前模型：{model_name}

**类型：** {preset["tag"]}

**算法路径：** `{preset["path_desc"]}`

**说明：** {preset["explain"]}

**Checkpoint：** `{preset["checkpoint"]}`
"""

# ─── 推理核心 ────────────────────────────────────────────
def _resolve_network_class(name: list[str]):
    mod = __import__(name[0], fromlist=[name[1]])
    return getattr(mod, name[1])

def _load_model(model_name: str, steps: int, device: torch.device):
    cache_key = (model_name, steps, str(device))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    preset = MODEL_PRESETS[model_name]
    with open(preset["config"]) as f:
        cfg = json.load(f)

    cfg["path"]["resume_state"] = preset["checkpoint"]
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

    ckpt = torch.load(preset["checkpoint"], map_location=device)
    _missing, _unexpected = net.load_state_dict(ckpt, strict=False)
    net = net.to(device)
    net.eval()

    _MODEL_CACHE[cache_key] = net
    return net

def run_inference(
    composite_img: np.ndarray,
    mask_img: np.ndarray,
    model_name: str,
    steps: int,
    progress=gr.Progress(),
) -> tuple[Image.Image, Image.Image, Image.Image, plt.Figure, str]:
    preset = MODEL_PRESETS[model_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = _load_model(model_name, steps, device)

    transform_fn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Composite
    comp_np = composite_img if isinstance(composite_img, np.ndarray) else np.array(composite_img)
    comp_pil = Image.fromarray(comp_np).convert("RGB").resize((256, 256), Image.BICUBIC)
    comp_t = transform_fn(comp_pil).unsqueeze(0).to(device)

    # Mask → binary visualization
    mask_np = mask_img if isinstance(mask_img, np.ndarray) else np.array(mask_img)
    if mask_np.ndim == 3 and mask_np.shape[2] > 1:
        mask_np = mask_np[:, :, 0]
    mask_pil = Image.fromarray(mask_np).convert("1").resize((256, 256), Image.NEAREST)
    mask_t = tf.to_tensor(mask_pil).unsqueeze(0).to(device)
    mask_vis = (np.array(mask_pil) > 0).astype(np.uint8) * 255
    mask_vis_pil = Image.fromarray(mask_vis).convert("RGB")

    # Inference
    with torch.no_grad():
        output, _ = net.restoration(comp_t, y_t=comp_t, y_0=comp_t, mask=mask_t, sample_num=2)

    def unnorm(t: torch.Tensor) -> torch.Tensor:
        return (t * 0.5 + 0.5).clamp(0, 1)

    result_pil = tf.to_pil_image(unnorm(output[0]).cpu())

    # Prompt weights
    prompt_aux = getattr(net, "_last_prompt_aux", {})
    prompt_weights = prompt_aux.get("prompt_weights")
    labels = prompt_aux.get("descriptor_labels", DESCRIPTOR_LABELS_EN)

    if prompt_weights is not None and prompt_weights.numel() > 0:
        w = prompt_weights[0].cpu().numpy()
        fig = _make_bar_chart(w, labels)
        info = _make_info_text(w, labels, preset, steps)
    else:
        fig = _make_placeholder_chart()
        info = f"### {model_name}\n\n{preset['tag']}  \n{preset['explain']}  \n步数: {steps} | 设备: {device}\n\n> 该模式无 prompt 权重输出"

    return comp_pil, mask_vis_pil, result_pil, fig, info

# ─── 可视化 ──────────────────────────────────────────────
def _make_bar_chart(weights: np.ndarray, labels: list[str]) -> plt.Figure:
    zh_labels = [DESCRIPTOR_LABELS_ZH.get(x, x) for x in labels]
    order = np.argsort(weights)
    w_sorted = weights[order]
    l_sorted = [zh_labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = ["#93c5fd"] * len(w_sorted)
    for idx in range(max(0, len(colors) - 3), len(colors)):
        colors[idx] = "#2563eb"

    bars = ax.barh(l_sorted, w_sorted, color=colors)
    ax.set_xlabel("Prompt Weight")
    ax.set_title("退化提示权重分析", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", linestyle="--", alpha=0.25)

    for bar, val in zip(bars, w_sorted):
        ax.text(min(val + 0.015, 0.96), bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig

def _make_placeholder_chart() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.text(0.5, 0.5, "No prompt weights (CDP-AFM mode)", ha="center", va="center", fontsize=14, color="gray")
    ax.set_title("退化提示权重分析", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig

def _make_info_text(weights: np.ndarray, labels: list[str], preset: dict, steps: int) -> str:
    order = np.argsort(-weights)
    top3 = order[:3]
    top_names = [DESCRIPTOR_LABELS_ZH.get(labels[i], labels[i]) for i in top3]

    explain_lines = []
    for i in top3:
        en_key = labels[i]
        zh = DESCRIPTOR_LABELS_ZH.get(en_key, en_key)
        exp = DESCRIPTOR_EXPLANATIONS.get(en_key, "检测到该类别退化，模型将针对性适配。")
        explain_lines.append(f"- **{zh}**：{weights[i]:.3f}。{exp}")

    return f"""### 退化诊断结果

**模型说明：** {preset.get("explain", "")}

**采样步数：** {steps}

**主要退化类型：** {', '.join(top_names)}

**Top-3 详细解释：**

{chr(10).join(explain_lines)}

> 权重越高，表示系统判断该类退化越突出，模型会优先对该方向进行补偿。
"""

# ─── UI ──────────────────────────────────────────────────
APP_CSS = """
.gradio-container { max-width: 1280px !important; margin: auto !important; font-family: 'Inter', 'Microsoft YaHei', sans-serif; }
.hero { padding: 28px 32px; border-radius: 22px; border: 1px solid #e4ecff; box-shadow: 0 10px 30px rgba(30,64,175,0.08); margin-bottom: 18px; }
.hero h1 { margin: 0 0 8px 0; font-size: 34px; font-weight: 800; color: #172554; }
.hero p { margin: 0; font-size: 16px; color: #475569; }
.badge { display: inline-block; padding: 6px 12px; margin: 12px 8px 0 0; border-radius: 999px; background: #dbeafe; color: #1e40af; font-size: 13px; font-weight: 600; }
.section-title { font-size: 18px; font-weight: 700; color: #0f172a; margin-bottom: 10px; }
"""

def create_ui():
    default_model = list(MODEL_PRESETS.keys())[0]

    with gr.Blocks(title="MDG-Harmonizer") as demo:
        preview_path = Path(__file__).parent / "preview.jpg"
        with gr.Row():
            if preview_path.exists():
                gr.Image(str(preview_path), label="", show_label=False, container=False, height=240)

        gr.Markdown("""
        <div class="hero">
          <h1>MDG-Harmonizer</h1>
          <p>退化感知图像协调与提示分析系统：支持工作一 CDP-AFM、工作二 M-DPR 以及融合模型推理。</p>
          <span class="badge">CDP-AFM 退化感知适配</span>
          <span class="badge">M-DPR 掩码感知 Prompt 路由</span>
          <span class="badge">Prompt 权重可视化</span>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown('<div class="section-title">1. 输入与模型设置</div>')
                with gr.Row():
                    composite_in = gr.Image(label="Composite Image", type="numpy", height=260)
                    mask_in = gr.Image(label="Foreground Mask", type="numpy", height=260)
                model_choice = gr.Dropdown(choices=list(MODEL_PRESETS.keys()), value=default_model, label="模型模式")
                steps_slider = gr.Slider(50, 200, value=200, step=50, label="DDPM 采样步数")
                run_btn = gr.Button("开始图像协调", variant="primary", size="lg")

            with gr.Column(scale=3):
                gr.Markdown('<div class="section-title">2. 当前模型说明</div>')
                model_card = gr.Markdown(get_model_card(default_model))

        gr.Markdown('<div class="section-title">3. 图像协调结果</div>')
        with gr.Row():
            comp_view = gr.Image(label="Composite Input", type="pil", height=300)
            mask_view = gr.Image(label="Mask", type="pil", height=300)
            result_out = gr.Image(label="Harmonized Result", type="pil", height=300)

        gr.Markdown('<div class="section-title">4. 退化提示分析</div>')
        with gr.Row():
            prompt_chart = gr.Plot(label="Prompt Weights")
            info_out = gr.Markdown("运行 M-DPR 模式后，将在这里显示退化诊断结果。")

        model_choice.change(get_model_card, inputs=model_choice, outputs=model_card)
        run_btn.click(
            fn=run_inference,
            inputs=[composite_in, mask_in, model_choice, steps_slider],
            outputs=[comp_view, mask_view, result_out, prompt_chart, info_out],
        )

    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    create_ui().launch(server_port=args.port, share=args.share, css=APP_CSS)
