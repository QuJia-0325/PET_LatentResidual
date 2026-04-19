from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def gaussian_window(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)


def ssim_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 5,
    data_range: float = 2.0,
    eps_var: float = 0.0,
    eps_den: float = 1.0e-6,
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
    return_stats: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if x.shape[1] != 1 or y.shape[1] != 1:
        raise ValueError("ssim_loss expects single-channel tensors")
    # Compute SSIM statistics in fp32 for numerical stability under AMP.
    half_range = float(data_range) / 2.0
    x32 = torch.nan_to_num(x.float(), nan=0.0, posinf=half_range, neginf=-half_range)
    y32 = torch.nan_to_num(y.float(), nan=0.0, posinf=half_range, neginf=-half_range)
    pad = window_size // 2
    window = gaussian_window(window_size, 1.5, x32.device, x32.dtype)
    mu_x = F.conv2d(x32, window, padding=pad)
    mu_y = F.conv2d(y32, window, padding=pad)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x = (F.conv2d(x32 * x32, window, padding=pad) - mu_x2).clamp_min(float(eps_var))
    sigma_y = (F.conv2d(y32 * y32, window, padding=pad) - mu_y2).clamp_min(float(eps_var))
    sigma_xy = F.conv2d(x32 * y32, window, padding=pad) - mu_xy
    c1 = (0.01 * float(data_range)) ** 2
    c2 = (0.03 * float(data_range)) ** 2
    ssim_n = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    ssim_d = ((mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)).clamp_min(float(eps_den))
    ssim_raw = ssim_n / ssim_d
    ssim_raw = torch.nan_to_num(
        ssim_raw,
        nan=float(clamp_min),
        posinf=float(clamp_max),
        neginf=float(clamp_min),
    )
    ssim_map = torch.clamp(ssim_raw, min=float(clamp_min), max=float(clamp_max))
    loss = 1.0 - ssim_map.mean()
    if not return_stats:
        return loss
    stats = {
        "ssim_raw_mean": ssim_raw.mean().detach(),
        "ssim_raw_min": ssim_raw.min().detach(),
        "ssim_raw_max": ssim_raw.max().detach(),
        "ssim_over1_frac": (ssim_raw > 1.0).float().mean().detach(),
        "ssim_below0_frac": (ssim_raw < 0.0).float().mean().detach(),
        "ssim_clamped_mean": ssim_map.mean().detach(),
    }
    return loss, stats


def make_border_weight_map(
    height: int,
    width: int,
    border_width: int,
    border_weight: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    weight = torch.ones((1, 1, height, width), device=device, dtype=dtype)
    bw = max(int(border_width), 0)
    if bw <= 0 or border_weight <= 1.0:
        return weight
    weight[:, :, :bw, :] = border_weight
    weight[:, :, -bw:, :] = border_weight
    weight[:, :, :, :bw] = border_weight
    weight[:, :, :, -bw:] = border_weight
    return weight


def weighted_l1_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"weighted_l1_loss expects pred/target same shape, got {tuple(pred.shape)} vs {tuple(target.shape)}")
    diff = (pred - target).abs()
    if weight.dim() != diff.dim():
        raise ValueError(f"weight dim mismatch: expected {diff.dim()}D, got {weight.dim()}D")
    if weight.shape == diff.shape:
        w = weight
    else:
        try:
            w = weight.expand_as(diff)
        except RuntimeError as e:
            raise ValueError(
                f"weight shape {tuple(weight.shape)} is not broadcastable to diff shape {tuple(diff.shape)}"
            ) from e
    # Weighted mean over all broadcasted elements (batch-size invariant).
    return (diff * w).sum() / w.sum().clamp(min=1e-8)


def seam_consistency_loss(pred: torch.Tensor, patch_size: int = 14) -> torch.Tensor:
    loss_seam = pred.new_zeros(())
    loss_grad = pred.new_zeros(())
    seam_positions = list(range(patch_size, pred.shape[-1], patch_size))
    for x in seam_positions:
        if x < pred.shape[-1]:
            loss_seam = loss_seam + F.l1_loss(pred[:, :, :, x - 1], pred[:, :, :, x])
            grad_left = pred[:, :, :, x - 1] - pred[:, :, :, max(0, x - 2)]
            grad_right = pred[:, :, :, min(pred.shape[-1] - 1, x + 1)] - pred[:, :, :, x]
            loss_grad = loss_grad + F.l1_loss(grad_left, grad_right)
    for y in seam_positions:
        if y < pred.shape[-2]:
            loss_seam = loss_seam + F.l1_loss(pred[:, :, y - 1, :], pred[:, :, y, :])
            grad_top = pred[:, :, y - 1, :] - pred[:, :, max(0, y - 2), :]
            grad_bottom = pred[:, :, min(pred.shape[-2] - 1, y + 1), :] - pred[:, :, y, :]
            loss_grad = loss_grad + F.l1_loss(grad_top, grad_bottom)
    num_seams = max(len(seam_positions) * 2, 1)
    return (loss_seam + 0.5 * loss_grad) / num_seams


def extended_seam_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    patch_size: int = 14,
    zone_width: int = 3,
) -> torch.Tensor:
    """Extended seam loss with ±zone_width pixel zone and second-order smoothness.

    Improvements over seam_consistency_loss:
    1. Penalizes ±zone_width pixels around each boundary (not just ±1)
    2. Uses distance-decaying weights within the zone
    3. Adds second-order derivative continuity (curvature matching)
    4. Compares against GT in the seam zone (not just self-consistency)

    Vectorized implementation (no Python loops over pixels).
    """
    H, W = pred.shape[-2:]
    loss = pred.new_zeros(())
    count = 0

    # Build seam column/row indices
    seam_x = torch.arange(patch_size, W, patch_size, device=pred.device)
    seam_y = torch.arange(patch_size, H, patch_size, device=pred.device)
    offsets = torch.arange(-zone_width, zone_width + 1, device=pred.device)
    # Distance-decaying weights: 1/(1+|offset|)
    weights = 1.0 / (1.0 + offsets.abs().float())

    # --- Horizontal seams (vertical boundaries at seam_x) ---
    for i, off in enumerate(offsets.tolist()):
        cols = seam_x + off
        valid = (cols >= 1) & (cols < W - 1)
        if not valid.any():
            continue
        cols_v = cols[valid]
        w = float(weights[i].item())
        # First-order gradient matching
        pred_grad = pred[:, :, :, cols_v + 1] - pred[:, :, :, cols_v - 1]
        gt_grad = gt[:, :, :, cols_v + 1] - gt[:, :, :, cols_v - 1]
        loss = loss + w * F.l1_loss(pred_grad, gt_grad)
        count += 1
        # Second-order curvature matching
        valid2 = (cols_v >= 2) & (cols_v < W - 2)
        if valid2.any():
            cols_v2 = cols_v[valid2]
            pred_curv = pred[:, :, :, cols_v2 + 1] + pred[:, :, :, cols_v2 - 1] - 2 * pred[:, :, :, cols_v2]
            gt_curv = gt[:, :, :, cols_v2 + 1] + gt[:, :, :, cols_v2 - 1] - 2 * gt[:, :, :, cols_v2]
            loss = loss + 0.5 * w * F.l1_loss(pred_curv, gt_curv)

    # --- Vertical seams (horizontal boundaries at seam_y) ---
    for i, off in enumerate(offsets.tolist()):
        rows = seam_y + off
        valid = (rows >= 1) & (rows < H - 1)
        if not valid.any():
            continue
        rows_v = rows[valid]
        w = float(weights[i].item())
        pred_grad = pred[:, :, rows_v + 1, :] - pred[:, :, rows_v - 1, :]
        gt_grad = gt[:, :, rows_v + 1, :] - gt[:, :, rows_v - 1, :]
        loss = loss + w * F.l1_loss(pred_grad, gt_grad)
        count += 1
        valid2 = (rows_v >= 2) & (rows_v < H - 2)
        if valid2.any():
            rows_v2 = rows_v[valid2]
            pred_curv = pred[:, :, rows_v2 + 1, :] + pred[:, :, rows_v2 - 1, :] - 2 * pred[:, :, rows_v2, :]
            gt_curv = gt[:, :, rows_v2 + 1, :] + gt[:, :, rows_v2 - 1, :] - 2 * gt[:, :, rows_v2, :]
            loss = loss + 0.5 * w * F.l1_loss(pred_curv, gt_curv)

    return loss / max(count, 1)


def residual_l2_penalty(residual: torch.Tensor) -> torch.Tensor:
    return (residual ** 2).mean()
