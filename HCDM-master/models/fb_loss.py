"""FB-Loss: Foreground-Boundary Aware composite loss for MDG-Harmonizer.

Combines five terms tailored for image harmonization where the foreground
(composite) region matters far more than the background:

    1. ``L_l1``       global per-pixel L1 (low-frequency content)
    2. ``L_fg``       L1 restricted to the foreground mask
                      (foreground is what gets visually harmonized)
    3. ``L_boundary`` L1 on the dilated-minus-eroded mask band
                      (transition zone is where seams appear)
    4. ``L_lpips``    perceptual distance via a frozen LPIPS-AlexNet
                      (the network adds NO trainable parameters)
    5. ``L_fft``      L1 over the high-frequency rFFT magnitudes
                      (recovers texture / edges that L1 over-smooths)

The boundary band is computed with stacked ``F.max_pool2d`` (dilate = max-pool
on the mask, erode = -max-pool on -mask). This avoids pulling in heavy
dependencies like Kornia.

All loss terms internally up-cast to fp32 and run with autocast disabled, so
the module is safe under ``torch.cuda.amp.autocast`` mixed-precision training.
"""

from __future__ import annotations

import contextlib
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lpips as _lpips_pkg

    _HAS_LPIPS = True
except Exception:  # pragma: no cover - environment specific
    _lpips_pkg = None
    _HAS_LPIPS = False


class FBLoss(nn.Module):
    """Foreground-Boundary aware composite loss.

    Args:
        weights: 5 floats in the order ``(l1, fg, boundary, lpips, fft)``.
            Defaults to ``(1.0, 2.0, 3.0, 0.1, 0.5)``.
        boundary_kernel: odd kernel size for the morphological dilate/erode
            that defines the boundary band. Larger -> wider transition band.
        fft_low_freq_ratio: fraction of the spectrum (per axis) to mask out as
            "low frequency". 0.125 keeps the highest ~87.5% of the spectrum.
        input_range: ``"tanh"`` if pred/gt are in [-1, 1] (typical for
            diffusion / DDPM), or ``"sigmoid"`` if in [0, 1]. Used only to
            re-map inputs into the [-1, 1] range LPIPS expects.
        use_lpips: if False, the LPIPS branch is skipped (still returns 0 in
            the loss dict). Useful for ablations / when network is offline.
        eps: numerical epsilon for masked-mean denominators.
    """

    DEFAULT_WEIGHTS: Tuple[float, ...] = (1.0, 2.0, 3.0, 0.1, 0.5)
    LOSS_KEYS: Tuple[str, ...] = ("l1", "fg", "boundary", "lpips", "fft")

    def __init__(
        self,
        weights: Optional[Sequence[float]] = None,
        boundary_kernel: int = 7,
        fft_low_freq_ratio: float = 0.125,
        input_range: str = "tanh",
        use_lpips: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if weights is None:
            weights = self.DEFAULT_WEIGHTS
        if len(weights) != 5:
            raise ValueError(f"weights must have 5 entries, got {len(weights)}")
        # Buffer (not Parameter) so it follows .to(device) but never trains.
        self.register_buffer(
            "weights", torch.tensor(list(weights), dtype=torch.float32)
        )

        if boundary_kernel % 2 == 0 or boundary_kernel < 3:
            raise ValueError(
                f"boundary_kernel must be an odd integer >= 3, got {boundary_kernel}"
            )
        self.boundary_kernel = int(boundary_kernel)

        if not (0.0 < fft_low_freq_ratio < 0.5):
            raise ValueError(
                "fft_low_freq_ratio must be in (0, 0.5), got "
                f"{fft_low_freq_ratio}"
            )
        self.fft_low_freq_ratio = float(fft_low_freq_ratio)

        if input_range not in ("tanh", "sigmoid"):
            raise ValueError(f"input_range must be 'tanh' or 'sigmoid', got {input_range}")
        self.input_range = input_range
        self.eps = float(eps)

        # LPIPS is loaded lazily only when requested, kept frozen in eval mode.
        if use_lpips:
            if not _HAS_LPIPS:
                raise ImportError(
                    "lpips package is required for FBLoss(use_lpips=True). "
                    "Install via: .venv/Scripts/pip.exe install lpips"
                )
            # ``verbose=False`` keeps the load quiet; ``net='alex'`` is the
            # smallest backbone (~6M frozen params, no grad).
            self.lpips_net = _lpips_pkg.LPIPS(net="alex", verbose=False)
            self.lpips_net.eval()
            for p in self.lpips_net.parameters():
                p.requires_grad = False
        else:
            self.lpips_net = None

    @staticmethod
    def _amp_off_ctx():
        """Disable autocast inside the loss; FFT and LPIPS need fp32.

        ``torch.is_autocast_enabled()`` is GPU-side; on CPU this is always
        False and we use a no-op context manager.
        """
        if torch.is_autocast_enabled():
            return torch.cuda.amp.autocast(enabled=False)
        return contextlib.nullcontext()

    def _to_lpips_range(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_range == "sigmoid":
            return x.clamp(0.0, 1.0).mul(2.0).sub(1.0)
        return x.clamp(-1.0, 1.0)

    def _morph_band(self, mask: torch.Tensor) -> torch.Tensor:
        """Return ``dilated - eroded`` boundary band in [0, 1].

        Erosion is implemented as ``-max_pool(-mask)`` so we don't have to
        write a custom op; for a soft mask this still gives a sensible
        transition band and exactly matches min-pooling for a binary mask.
        """
        k, pad = self.boundary_kernel, self.boundary_kernel // 2
        dilated = F.max_pool2d(mask, kernel_size=k, stride=1, padding=pad)
        eroded = -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=pad)
        return (dilated - eroded).clamp(0.0, 1.0)

    def _masked_l1(
        self, pred: torch.Tensor, gt: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        """L1 on (pred, gt) averaged only over the weighted (active) area.

        ``weight`` has shape ``(B,1,H,W)`` and broadcasts over channels;
        normalising by ``weight.sum() * C`` keeps this term on the same scale
        as the global L1 even when the mask covers very few pixels.
        """
        diff = (pred - gt).abs() * weight  # broadcasts across C
        denom = weight.sum() * pred.size(1) + self.eps
        return diff.sum() / denom

    def _fft_high_freq_l1(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """L1 between rFFT magnitudes after zeroing out the low-frequency block.

        For ``rfft2`` over an ``(H, W)`` image the output has shape
        ``(H, W//2+1)``. Low-frequency components live at the four spectral
        corners along the H-axis (positive low freqs at ``[:rh]``, negative
        low freqs wrap around to ``[-rh:]``) and the leftmost columns
        ``[:rw]`` along the half-spectrum W-axis. We zero a single corner
        block on each of those bands to suppress the low-frequency content
        the global L1 already supervises.
        """
        # rFFT requires fp32 for stable autograd; do this regardless of dtype.
        pred_fft = torch.fft.rfft2(pred.float(), norm="ortho")
        gt_fft = torch.fft.rfft2(gt.float(), norm="ortho")
        diff = (pred_fft - gt_fft).abs()  # complex -> real magnitude
        b, c, h, wh = diff.shape
        rh = max(1, int(round(h * self.fft_low_freq_ratio)))
        rw = max(1, int(round(wh * self.fft_low_freq_ratio)))
        hf_mask = torch.ones_like(diff)
        hf_mask[:, :, :rh, :rw] = 0.0          # +low-freq corner along H
        hf_mask[:, :, -rh:, :rw] = 0.0         # -low-freq corner along H (wrap-around)
        active = hf_mask.sum() + self.eps
        return (diff * hf_mask).sum() / active

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the composite loss.

        Args:
            pred: predicted image, ``(B, 3, H, W)``.
            gt:   ground-truth image, ``(B, 3, H, W)``.
            mask: foreground mask, ``(B, 1, H, W)`` in ``[0, 1]``.

        Returns:
            ``(total_loss, loss_dict)`` where ``loss_dict`` holds the five
            *unweighted* sub-losses keyed by ``LOSS_KEYS`` (handy for
            tensorboard).
        """
        if pred.shape != gt.shape:
            raise ValueError(f"pred {tuple(pred.shape)} vs gt {tuple(gt.shape)} mismatch")
        if mask.shape[0] != pred.shape[0] or mask.shape[-2:] != pred.shape[-2:]:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} incompatible with pred "
                f"{tuple(pred.shape)}"
            )
        if mask.size(1) != 1:
            # Allow either 1-channel mask or already-broadcast 3-channel mask.
            mask = mask[:, :1]

        with self._amp_off_ctx():
            pred_f = pred.float()
            gt_f = gt.float()
            mask_f = mask.float().clamp(0.0, 1.0)

            l1 = F.l1_loss(pred_f, gt_f)
            fg = self._masked_l1(pred_f, gt_f, mask_f)
            boundary = self._masked_l1(pred_f, gt_f, self._morph_band(mask_f))

            if self.lpips_net is not None:
                # LPIPS expects (B,3,H,W) in [-1, 1].
                lp = self.lpips_net(
                    self._to_lpips_range(pred_f),
                    self._to_lpips_range(gt_f),
                ).mean()
            else:
                lp = pred_f.new_zeros(())

            fft = self._fft_high_freq_l1(pred_f, gt_f)

        loss_dict: Dict[str, torch.Tensor] = {
            "l1": l1,
            "fg": fg,
            "boundary": boundary,
            "lpips": lp,
            "fft": fft,
        }
        w = self.weights.to(pred_f.device)
        total = (
            w[0] * l1
            + w[1] * fg
            + w[2] * boundary
            + w[3] * lp
            + w[4] * fft
        )
        return total, loss_dict


# ---------------------------------------------------------------------------
# Unit test (CPU only): random tensors, sanity-check magnitudes & gradients.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cpu")

    b, c, h, w = 2, 3, 256, 256
    # Use a mild perturbation so loss values are realistic, not maxed out.
    gt = torch.randn(b, c, h, w, device=device).tanh()        # in (-1, 1)
    pred = (gt + 0.1 * torch.randn_like(gt)).clamp(-1.0, 1.0)
    mask = (torch.rand(b, 1, h, w, device=device) > 0.6).float()

    fb = FBLoss().to(device)
    fb.eval()  # LPIPS prefers eval; no BN buffers to update anyway.

    n_train = sum(p.numel() for p in fb.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in fb.parameters())
    n_lpips = (
        sum(p.numel() for p in fb.lpips_net.parameters())
        if fb.lpips_net is not None
        else 0
    )
    print(
        f"FBLoss params -> trainable: {n_train}  "
        f"total: {n_total}  (LPIPS frozen: {n_lpips})"
    )
    assert n_train == 0, "FBLoss must add no trainable parameters."

    total, parts = fb(pred, gt, mask)
    print(f"Total weighted loss: {total.item():.6f}")
    print(f"Default weights:     {fb.weights.tolist()}")
    print("Unweighted sub-losses:")
    for k in FBLoss.LOSS_KEYS:
        v = parts[k]
        print(f"  {k:>9s}: {v.item():.6f}")
        assert torch.isfinite(v), f"{k} is non-finite!"

    # Sanity 1: identical inputs should give a (nearly) zero loss.
    total_same, parts_same = fb(gt, gt, mask)
    print(f"\nSanity (pred == gt) total: {total_same.item():.6f}")
    for k in FBLoss.LOSS_KEYS:
        print(f"  {k:>9s}: {parts_same[k].item():.6f}")

    # Sanity 2: gradient flows back to pred.
    pred_g = pred.clone().requires_grad_(True)
    total_g, _ = fb(pred_g, gt, mask)
    total_g.backward()
    assert pred_g.grad is not None and torch.isfinite(pred_g.grad).all()
    print(
        f"\nGrad OK: |grad|.mean()={pred_g.grad.abs().mean().item():.6f}  "
        f"|grad|.max()={pred_g.grad.abs().max().item():.6f}"
    )

    # Sanity 3: empty mask must not produce NaN.
    empty_mask = torch.zeros_like(mask)
    total_em, parts_em = fb(pred, gt, empty_mask)
    assert torch.isfinite(total_em), "empty-mask path produced NaN/Inf!"
    print(
        f"\nEmpty-mask total: {total_em.item():.6f}  "
        f"fg={parts_em['fg'].item():.6f}  boundary={parts_em['boundary'].item():.6f}"
    )
