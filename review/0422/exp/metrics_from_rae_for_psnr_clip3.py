#!/usr/bin/env python3
"""Unified metrics for evaluation (PSNR, SSIM)"""
import torch
import numpy as np


def calc_psnr_clip3(pred, gt):
    """
    PSNR with clip_max=3 (99.9% quantile) for evaluation.

    Args:
        pred: torch.Tensor in [-1, 1] range
        gt: torch.Tensor in [-1, 1] range

    Returns:
        PSNR in dB
    """
    # 反归一化到 [0, 10] SUV 空间
    pred_suv = (pred + 1.0) * 5.0
    gt_suv = (gt + 1.0) * 5.0

    # Clip 到 [0, 3]
    pred_suv = pred_suv.clamp(0, 3.0)
    gt_suv = gt_suv.clamp(0, 3.0)

    # 计算 PSNR (data_range=3.0)
    mse = ((pred_suv - gt_suv) ** 2).mean().item()
    if mse == 0:
        return float('inf')
    return 20 * np.log10(3.0) - 10 * np.log10(mse)
