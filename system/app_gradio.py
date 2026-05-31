r"""MDG-Harmonizer 退化感知图像协调与提示分析系统。

支持 MDG-D（工作一）和 M-DPR（工作二）三种模式推理，
展示退化提示权重可视化与诊断分析。
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

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_BG_PATH = _ASSETS_DIR / "bj.jpg"
_BG_URL = f"/gradio_api/file={_BG_PATH.as_posix()}"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_PROJECT_ROOT))

matplotlib.use("Agg")
# Chinese font support
import matplotlib.font_manager as fm
_SYS_FONTS = [f.name for f in fm.fontManager.ttflist]
_CN_FONT = next((n for n in _SYS_FONTS if "Microsoft YaHei" in n or "SimHei" in n or "WenQuanYi" in n), None)
if _CN_FONT:
    matplotlib.rc("font", family=_CN_FONT, size=10)
# Fallback for minus sign
matplotlib.rc("axes", unicode_minus=False)

# ─── Constants ──────────────────────────────────────────
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
        "config": "config/ablation_A_full_test.json",
        "checkpoint": "experiments/train_mdg_ablation_D_no_fb_260523_224814/checkpoint/30_MDGNetwork.pth",
        "tag": "工作一最优消融",
        "path_desc": "Composite + Mask -> CDP-Net -> AFM -> Harmonization",
        "explain": "CDP-AFM 轻量退化感知适配算法，当前最优消融 (PSNR 36.40)。",
        "icon": "🟢",
    },
    "M-DPR Router only": {
        "config": "config/ablation_A_full_test.json",
        "checkpoint": "experiments/train_prompt_router_m-dpr_descriptor_260524_145005/checkpoint/5_PromptRouterMDGNetwork.pth",
        "network": ["models.network_prompt_router_mdg", "PromptRouterMDGNetwork"],
        "network_kwargs": {"prompt_mode": "descriptor_only", "prompt_dim": 64, "freeze_backbone": True},
        "tag": "工作二独立模型",
        "path_desc": "Composite+Mask -> Degradation Descriptor -> Prompt Router -> Harmonization",
        "explain": "M-DPR 不依赖 CDP-Net 的独立退化提示构建能力 (~1K 可训参数)。",
        "icon": "🟣",
    },
    "M-DPR Hybrid": {
        "config": "config/ablation_A_full_test.json",
        "checkpoint": "experiments/train_prompt_router_m-dpr_hybrid_260524_145338/checkpoint/5_PromptRouterMDGNetwork.pth",
        "network": ["models.network_prompt_router_mdg", "PromptRouterMDGNetwork"],
        "network_kwargs": {"prompt_mode": "hybrid", "prompt_dim": 64, "freeze_backbone": True},
        "tag": "最终融合模型",
        "path_desc": "CDP-Net Prior + Prompt Router Prior -> Prior Fusion -> Harmonization",
        "explain": "工作一和工作二的互补融合，适合作为系统默认展示模型。",
        "icon": "🔵",
    },
}

_MODEL_CACHE = {}

# ─── Model Card ─────────────────────────────────────────
def get_model_card(model_name: str) -> str:
    p = MODEL_PRESETS[model_name]
    return f"""### {p['icon']} {model_name}

**模型类型：** {p['tag']}

**算法路径：** `{p['path_desc']}`

**说明：** {p['explain']}

**Checkpoint：** `{p['checkpoint']}`
"""

# ─── Inference ──────────────────────────────────────────
def _resolve_network_class(name: list[str]):
    mod = __import__(name[0], fromlist=[name[1]])
    return getattr(mod, name[1])

def _load_model(model_name: str, steps: int, device: torch.device):
    key = (model_name, int(steps), str(device))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    preset = MODEL_PRESETS[model_name]
    with open(str(_PROJECT_ROOT / preset["config"])) as f:
        cfg = json.load(f)
    cfg["path"]["resume_state"] = str(_PROJECT_ROOT / preset["checkpoint"])

    ns = preset.get("network")
    if ns is not None:
        cfg["model"]["which_networks"][0]["name"] = ns
        for k, v in preset.get("network_kwargs", {}).items():
            cfg["model"]["which_networks"][0]["args"][k] = v

    net_args = cfg["model"]["which_networks"][0]["args"]
    NetCls = _resolve_network_class(cfg["model"]["which_networks"][0]["name"])

    # Load checkpoint first to get actual buffer shapes
    ckpt = torch.load(str(_PROJECT_ROOT / preset["checkpoint"]), map_location=device)
    # Force n_timestep to match checkpoint
    ckpt_n = ckpt["gammas"].shape[0]
    net_args["beta_schedule"]["test"]["n_timestep"] = ckpt_n
    net_args["beta_schedule"]["train"]["n_timestep"] = ckpt_n

    net = NetCls(**net_args)
    net.set_new_noise_schedule(device=device, phase="test")
    missing, _ = net.load_state_dict(ckpt, strict=False)
    net = net.to(device)
    net.eval()
    # Rebuild with user's desired step count
    net.beta_schedule["test"]["n_timestep"] = int(steps)
    net.set_new_noise_schedule(device=device, phase="test")

    _MODEL_CACHE[key] = (net, len(missing))
    return _MODEL_CACHE[key]

def run_inference(
    comp_img: np.ndarray, mask_img: np.ndarray,
    model_name: str, steps: int,
    progress=gr.Progress(),
) -> tuple:
    preset = MODEL_PRESETS[model_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    progress(0.05, desc="加载模型...")
    net, n_missing = _load_model(model_name, steps, device)

    transform_fn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    progress(0.20, desc="预处理输入...")
    comp_np = comp_img if isinstance(comp_img, np.ndarray) else np.array(comp_img)
    comp_pil = Image.fromarray(comp_np).convert("RGB").resize((256, 256), Image.BICUBIC)
    comp_t = transform_fn(comp_pil).unsqueeze(0).to(device)

    mask_np = mask_img if isinstance(mask_img, np.ndarray) else np.array(mask_img)
    if mask_np.ndim == 3 and mask_np.shape[2] > 1:
        mask_np = mask_np[:, :, 0]
    mask_pil = Image.fromarray(mask_np).convert("1").resize((256, 256), Image.NEAREST)
    mask_t = tf.to_tensor(mask_pil).unsqueeze(0).to(device)
    mask_vis = (np.array(mask_pil) > 0).astype(np.uint8) * 255
    mask_vis_pil = Image.fromarray(mask_vis).convert("RGB")

    progress(0.40, desc="扩散推理中...")
    with torch.no_grad():
        output, _ = net.restoration(comp_t, y_t=comp_t, y_0=comp_t, mask=mask_t, sample_num=2)

    def unnorm(t): return (t * 0.5 + 0.5).clamp(0, 1)

    progress(0.80, desc="生成可视化...")
    result_pil = tf.to_pil_image(unnorm(output[0]).cpu())

    aux = getattr(net, "_last_prompt_aux", {})
    pw = aux.get("prompt_weights")
    labels = aux.get("descriptor_labels", DESCRIPTOR_LABELS_EN)

    if pw is not None and pw.numel() > 0:
        w = pw[0].cpu().numpy()
        fig = _make_bar_chart(w, labels)
        info = _make_info_text(w, labels, preset, steps, n_missing)
    else:
        fig = _make_placeholder_chart()
        info = f"### {model_name}\n\n{preset['icon']} {preset['tag']}  \n{preset['explain']}  \n步数: {steps} | 设备: {device}\n\n> 该模式无 prompt 权重输出"

    progress(1.0, desc="完成")
    return comp_pil, mask_vis_pil, result_pil, fig, info

def _safe_run(*args) -> tuple:
    """错误处理包装，把报错显示在界面上。"""
    try:
        return run_inference(*args)
    except Exception as e:
        import traceback
        err_fig = plt.Figure()
        plt.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", fontsize=10, color="red")
        err_msg = f"### 运行错误\n\n```\n{traceback.format_exc()}\n```"
        # Return placeholders
        blank = Image.new("RGB", (256, 256), (240, 240, 240))
        return blank, blank, blank, err_fig, err_msg

# ─── Visualization ──────────────────────────────────────
def _make_bar_chart(weights: np.ndarray, labels: list[str]) -> plt.Figure:
    zh = [DESCRIPTOR_LABELS_ZH.get(x, x) for x in labels]
    order = np.argsort(weights)
    ws, ls = weights[order], [zh[i] for i in order]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = ["#bfdbfe"] * len(ws)
    for i in range(max(0, len(colors) - 3), len(colors)):
        colors[i] = "#2563eb"
    ax.barh(ls, ws, color=colors)
    ax.set_xlabel("Prompt Weight")
    ax.set_title("退化提示权重分析", fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(1.0, float(ws.max()) * 1.15))
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    for bar, val in zip(ax.patches, ws):
        ax.text(min(float(val) + 0.015, ax.get_xlim()[1] * 0.96),
                bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig

def _make_placeholder_chart() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.text(0.5, 0.5, "无 prompt 权重 (CDP-AFM 模式)", ha="center", va="center", fontsize=14, color="gray")
    ax.set_title("退化提示权重分析", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig

def _make_info_text(weights, labels, preset, steps, missing):
    order = np.argsort(-weights)
    top3 = order[:3]
    top_names = "、".join(DESCRIPTOR_LABELS_ZH.get(labels[i], labels[i]) for i in top3)
    lines = []
    for i in top3:
        en = labels[i]
        zh_n = DESCRIPTOR_LABELS_ZH.get(en, en)
        lines.append(f"- **{zh_n}**：{weights[i]:.3f}。{DESCRIPTOR_EXPLANATIONS.get(en, '')}")
    return f"""### 退化诊断结果

**模型说明：** {preset.get('explain', '')}

**采样步数：** {steps}

**主要退化类型：** {top_names}

**Top-3 解释：**

{chr(10).join(lines)}

> 权重越高，表示系统判断该类退化越突出。
> 加载信息: missing keys = {missing}
"""

# ─── UI ─────────────────────────────────────────────────
APP_CSS = f"""
body {{
    background: linear-gradient(rgba(248, 250, 252, 0.85), rgba(239, 246, 255, 0.92)),
                #f8fafc !important;
}}
.gradio-container {{
    max-width: 1280px !important; margin: auto !important;
    background: transparent !important;
    font-family: 'Inter', 'Microsoft YaHei', 'PingFang SC', sans-serif;
}}
.hero {{
    padding: 34px 38px; border-radius: 28px; margin-bottom: 22px;
    background: linear-gradient(135deg, rgba(15,23,42,0.88), rgba(30,64,175,0.78)), url('{_BG_URL}') center/cover no-repeat;
    color: white; box-shadow: 0 24px 70px rgba(15,23,42,0.25);
    position: relative; overflow: hidden;
}}
.hero::after {{
    content: ""; position: absolute; right: -80px; top: -80px;
    width: 260px; height: 260px; border-radius: 999px;
    background: rgba(96,165,250,0.28); filter: blur(6px);
}}
.hero h1, .hero h1 *, .hero strong {{
    margin: 0 0 10px 0;
    font-size: 42px !important;
    font-weight: 900 !important;
    letter-spacing: -0.6px;
    color: #ffffff !important;
    text-shadow: 0 3px 14px rgba(0, 0, 0, 0.55);
}}
.hero p, .hero p *, .hero span {{
    font-size: 17px !important;
    font-weight: 600 !important;
    color: #f8fafc !important;
    line-height: 1.75 !important;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.45);
}}
.badge {{
    display: inline-block; margin: 16px 8px 0 0; padding: 7px 13px;
    border-radius: 999px; background: rgba(255,255,255,0.14);
    border: 1px solid rgba(255,255,255,0.22); color: rgba(255,255,255,0.92);
    font-size: 13px; font-weight: 650; backdrop-filter: blur(10px);
}}
.glass-card {{
    padding: 18px; border-radius: 24px;
    background: rgba(255,255,255,0.76); border: 1px solid rgba(255,255,255,0.70);
    box-shadow: 0 16px 45px rgba(15,23,42,0.10); backdrop-filter: blur(16px);
}}
.section-title {{ font-size: 18px; font-weight: 800; color: #0f172a; margin: 8px 0 12px 0; }}
.subtle-text {{ color: #64748b; font-size: 13px; line-height: 1.6; }}
.image-frame {{ border-radius: 20px; overflow: hidden; }}
footer {{ display: none !important; }}
button.primary, #run_btn {{
    border-radius: 16px !important; min-height: 46px !important; font-weight: 800 !important;
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    border: none !important; box-shadow: 0 14px 28px rgba(37,99,235,0.22) !important;
}}
button.primary:hover, #run_btn:hover {{
    transform: translateY(-1px); box-shadow: 0 18px 36px rgba(37,99,235,0.30) !important;
}}

.hero,
.hero * {{
    color: #ffffff !important;
}}

.hero h1 {{
    font-size: 42px !important;
    font-weight: 900 !important;
    letter-spacing: -0.6px !important;
    text-shadow: 0 3px 14px rgba(0, 0, 0, 0.55) !important;
}}

.hero p,
.hero span {{
    color: #f8fafc !important;
    font-size: 17px !important;
    font-weight: 600 !important;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.45) !important;
}}
"""

def create_ui():
    default_model = list(MODEL_PRESETS.keys())[0]

    with gr.Blocks(title="MDG-Harmonizer") as demo:
        # Hero
        gr.Markdown("""
        <div class="hero">
          <h1>MDG-Harmonizer</h1>
          <p>退化感知图像协调与提示分析系统，支持 CDP-AFM、M-DPR 与融合模型推理。</p>
          <span class="badge">CDP-AFM 退化感知适配</span>
          <span class="badge">M-DPR 掩码感知 Prompt 路由</span>
          <span class="badge">Prompt 权重可解释分析</span>
        </div>
        """)

        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                gr.Markdown('<div class="section-title">1. 输入与推理设置</div>')
                gr.Markdown('<div class="subtle-text">上传合成图与前景掩码，选择模型后即可完成图像协调推理。</div>')
                with gr.Row():
                    composite_in = gr.Image(label="Composite Image", type="numpy", height=260, elem_classes=["image-frame"])
                    mask_in = gr.Image(label="Foreground Mask", type="numpy", height=260, elem_classes=["image-frame"])
                model_choice = gr.Dropdown(choices=list(MODEL_PRESETS.keys()), value=default_model, label="模型模式")
                steps_slider = gr.Radio(choices=[50, 100, 200], value=200, label="DDPM 采样步数", info="50 更快，200 质量更稳")
                run_btn = gr.Button("开始协调 / Run Harmonization", variant="primary", size="lg", elem_id="run_btn")

            with gr.Column(scale=4):
                gr.Markdown('<div class="section-title">2. 当前模型说明</div>')
                model_card = gr.Markdown(get_model_card(default_model))

        gr.Markdown('<div class="section-title">3. 图像协调结果</div>')
        with gr.Row():
            comp_view = gr.Image(label="Composite Input", type="pil", height=300)
            mask_view = gr.Image(label="Mask Preview", type="pil", height=300)
            result_out = gr.Image(label="Harmonized Result", type="pil", height=300)

        gr.Markdown('<div class="section-title">4. 退化提示分析</div>')
        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                prompt_chart = gr.Plot(label="Prompt Weights")
            with gr.Column(scale=4):
                info_out = gr.Markdown("运行 M-DPR 模式后，将在这里显示退化诊断结果。")

        model_choice.change(get_model_card, inputs=model_choice, outputs=model_card)

        run_btn.click(
            fn=_safe_run,
            inputs=[composite_in, mask_in, model_choice, steps_slider],
            outputs=[comp_view, mask_view, result_out, prompt_chart, info_out],
        )

    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    create_ui().launch(server_port=args.port, share=args.share, css=APP_CSS, allowed_paths=[str(_ASSETS_DIR)])
