"""从 baseline test 的 Out_*/In_* 图像对中计算聚合指标。

baseline 的 ``RIHD.test`` 只把每张的 mae 写到 tensorboard，没有最终汇总打印；
本脚本扫描 ``experiments/<run>/results/test/0/`` 下所有图像对，重新计算
**标准 image harmonization 指标**（与 HCDM 论文表格可比）：
    - **MAE**     全图平均绝对误差，[0, 255] 像素尺度
    - **MSE**     均方误差
    - **PSNR**    峰值信噪比 (dB)
    - **SSIM**    结构相似度（用 skimage 实现）
    - 同时输出 fMAE/fPSNR（前景掩码内）—— 仅当能定位到 mask 文件时

用法：
    python -W ignore scripts/compute_baseline_metrics.py \\
        --run experiments/test_harmonization_allinone_260506_123100 \\
        [--mask-root TestData/Hday2night/masks]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def _load_rgb(path: Path) -> np.ndarray:
    """读图为 ``(H, W, 3)`` uint8。失败返回 None。"""
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def _load_mask(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    m = Image.open(path).convert("L")
    arr = np.asarray(m)
    return (arr > 127).astype(np.float32)  # 二值化：0/1


def _resize_to(arr: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """对齐 mask 与 Out/In 的尺寸（test 流程会把图统一缩到 256×256）。"""
    if arr.shape[:2] == target_hw:
        return arr
    img = Image.fromarray(arr if arr.dtype == np.uint8 else (arr * 255).astype(np.uint8))
    img = img.resize((target_hw[1], target_hw[0]), Image.NEAREST)
    out = np.asarray(img)
    if arr.dtype != np.uint8:
        return (out > 127).astype(np.float32)
    return out


def _safe_psnr(mse: float, max_val: float = 255.0) -> float:
    if mse <= 1e-10:
        return float("inf")
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)


def _try_ssim(out: np.ndarray, gt: np.ndarray) -> Optional[float]:
    try:
        from skimage.metrics import structural_similarity as ssim
    except Exception:
        return None
    return float(ssim(out, gt, channel_axis=2, data_range=255))


def _resolve_mask_path(stem: str, mask_root: Path) -> Optional[Path]:
    """匹配多种常见 mask 文件名约定，返回首个存在的。"""
    import re
    candidates = [
        mask_root / f"{stem}.png",
        mask_root / f"{stem}.jpg",
        mask_root / f"{stem}_mask.png",
        mask_root / f"{stem}_mask.jpg",
    ]
    # 去掉末尾 _N 后缀再试（如 d1048_xxx_1_1 → d1048_xxx_1）
    stripped = re.sub(r'_\d+$', '', stem)
    if stripped != stem:
        candidates.extend([
            mask_root / f"{stripped}.png",
            mask_root / f"{stripped}.jpg",
        ])
    for c in candidates:
        if c.exists():
            return c
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="实验目录（含 results/test/0）")
    parser.add_argument(
        "--mask-root",
        default="TestData/Hday2night/masks",
        help="测试集 mask 根目录（用于算 fMAE/fPSNR）",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="把聚合指标写到 JSON（默认：<run>/baseline_metrics.json）",
    )
    parser.add_argument("--limit", type=int, default=0, help="仅取前 N 张（debug）")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = (repo_root / run_dir).resolve()
    results_dir = run_dir / "results" / "test" / "0"
    if not results_dir.exists():
        print(f"!! results dir not found: {results_dir}")
        sys.exit(2)

    mask_root = Path(args.mask_root)
    if not mask_root.is_absolute():
        mask_root = (repo_root / mask_root).resolve()

    # 配对 Out_<stem>.jpg ↔ In_<stem>.jpg
    out_files = sorted(results_dir.glob("Out_*.jpg"))
    if args.limit:
        out_files = out_files[: args.limit]

    pairs: List[Tuple[Path, Path, str]] = []
    for of in out_files:
        stem = of.stem[len("Out_"):]
        inf = results_dir / f"In_{stem}.jpg"
        if inf.exists():
            pairs.append((of, inf, stem))

    print(f"[{results_dir.name}] paired {len(pairs)} (Out, In) images. mask_root={mask_root}")

    mae_list: List[float] = []
    mse_list: List[float] = []
    psnr_list: List[float] = []
    ssim_list: List[float] = []
    fmae_list: List[float] = []
    fpsnr_list: List[float] = []
    fssim_list: List[float] = []
    n_with_mask = 0
    skim_unavailable = False

    for i, (of, inf, stem) in enumerate(pairs):
        out_rgb = _load_rgb(of).astype(np.float32)
        gt_rgb = _load_rgb(inf).astype(np.float32)
        if out_rgb.shape != gt_rgb.shape:
            print(f"  shape mismatch on {stem}: {out_rgb.shape} vs {gt_rgb.shape}, skipping")
            continue

        diff = out_rgb - gt_rgb
        mae = float(np.abs(diff).mean())
        mse = float(np.square(diff).mean())
        psnr = _safe_psnr(mse)

        ssim_v = _try_ssim(out_rgb.astype(np.uint8), gt_rgb.astype(np.uint8))
        if ssim_v is None:
            skim_unavailable = True

        mae_list.append(mae)
        mse_list.append(mse)
        psnr_list.append(psnr)
        if ssim_v is not None:
            ssim_list.append(ssim_v)

        # fMAE / fPSNR（仅当 mask 可定位）
        mp = _resolve_mask_path(stem, mask_root)
        if mp is not None:
            mask = _load_mask(mp)
            if mask is not None:
                mask = _resize_to(mask, out_rgb.shape[:2])  # type: ignore[arg-type]
                m3 = mask[..., None]
                m_sum = m3.sum() * 3  # 3 通道总像素数
                if m_sum > 0:
                    fmae = float((np.abs(diff) * m3).sum() / m_sum)
                    fmse = float((np.square(diff) * m3).sum() / m_sum)
                    fmae_list.append(fmae)
                    fpsnr_list.append(_safe_psnr(fmse))
                    # fSSIM: crop to mask bounding box
                    ys, xs = np.where(mask > 0.5)
                    if len(ys) > 0:
                        y0, y1 = max(ys.min() - 4, 0), min(ys.max() + 5, mask.shape[0])
                        x0, x1 = max(xs.min() - 4, 0), min(xs.max() + 5, mask.shape[1])
                        out_crop = out_rgb.astype(np.uint8)[y0:y1, x0:x1]
                        gt_crop = gt_rgb.astype(np.uint8)[y0:y1, x0:x1]
                        fssim_v = _try_ssim(out_crop, gt_crop)
                        if fssim_v is not None:
                            fssim_list.append(fssim_v)
                    n_with_mask += 1

        if (i + 1) % 25 == 0:
            print(f"  processed {i+1}/{len(pairs)}: MAE={mae:.3f} PSNR={psnr:.3f}")

    # 聚合
    summary: Dict[str, float] = {
        "num_images": len(mae_list),
        "MAE": float(np.mean(mae_list)) if mae_list else float("nan"),
        "MSE": float(np.mean(mse_list)) if mse_list else float("nan"),
        "PSNR": float(np.mean(psnr_list)) if psnr_list else float("nan"),
    }
    if ssim_list:
        summary["SSIM"] = float(np.mean(ssim_list))
    if fmae_list:
        summary["fMAE"] = float(np.mean(fmae_list))
        summary["fPSNR"] = float(np.mean(fpsnr_list))
        summary["num_images_with_mask"] = n_with_mask
    if fssim_list:
        summary["fSSIM"] = float(np.mean(fssim_list))

    print()
    print("=" * 60)
    print("Baseline test aggregated metrics")
    print("=" * 60)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:25s} {v:.4f}")
        else:
            print(f"  {k:25s} {v}")
    if skim_unavailable:
        print("  (skimage not available, SSIM skipped)")
    if not fmae_list:
        print("  (no masks resolved, fMAE/fPSNR/fSSIM skipped)")

    out_json = (
        Path(args.out_json) if args.out_json else (run_dir / "baseline_metrics.json")
    )
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote: {out_json}")


if __name__ == "__main__":
    main()
