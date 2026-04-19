from __future__ import annotations

from typing import Dict

import torch

from .losses import make_border_weight_map, seam_consistency_loss, extended_seam_loss, ssim_loss, weighted_l1_loss


def compute_first_hop_image_loss(
    x_pred: torch.Tensor,
    x_gt: torch.Tensor,
    border_width: int = 14,
    border_weight: float = 2.0,
    w_l1: float = 1.0,
    w_ssim: float = 0.25,
    w_seam: float = 0.10,
    seam_patch_size: int = 14,
    use_extended_seam: bool = False,
    seam_zone_width: int = 3,
) -> Dict[str, torch.Tensor]:
    """Compute hop0 image auxiliary loss on cropped image tensors."""
    if x_pred.shape != x_gt.shape:
        raise ValueError(f"x_pred/x_gt shape mismatch: {tuple(x_pred.shape)} vs {tuple(x_gt.shape)}")
    if x_pred.dim() != 4 or x_pred.shape[1] != 1:
        raise ValueError(f"Expected single-channel image tensors [B,1,H,W], got {tuple(x_pred.shape)}")

    border_map = make_border_weight_map(
        height=x_gt.shape[-2],
        width=x_gt.shape[-1],
        border_width=int(border_width),
        border_weight=float(border_weight),
        device=x_gt.device,
        dtype=x_gt.dtype,
    )

    loss_l1 = weighted_l1_loss(x_pred, x_gt, border_map)
    loss_ssim, ssim_stats = ssim_loss(x_pred, x_gt, return_stats=True)
    if use_extended_seam:
        loss_seam = extended_seam_loss(
            x_pred, x_gt,
            patch_size=int(seam_patch_size),
            zone_width=int(seam_zone_width),
        )
    else:
        loss_seam = seam_consistency_loss(x_pred, patch_size=int(seam_patch_size))
    total = float(w_l1) * loss_l1 + float(w_ssim) * loss_ssim + float(w_seam) * loss_seam

    return {
        "total": total,
        "l1": loss_l1,
        "ssim": loss_ssim,
        "seam": loss_seam,
        "ssim_raw_mean": ssim_stats["ssim_raw_mean"],
        "ssim_raw_min": ssim_stats["ssim_raw_min"],
        "ssim_raw_max": ssim_stats["ssim_raw_max"],
        "ssim_over1_frac": ssim_stats["ssim_over1_frac"],
        "ssim_below0_frac": ssim_stats["ssim_below0_frac"],
        "ssim_clamped_mean": ssim_stats["ssim_clamped_mean"],
    }
