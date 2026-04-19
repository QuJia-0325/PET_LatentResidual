#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
from tqdm import tqdm
import yaml

from pet_lr.data_first_hop import Hop0OnlyViewDataset, PETFirstHopAligned4HopDataset
from pet_lr.losses_first_hop import compute_first_hop_image_loss
from pet_lr.model_first_hop import PETFlowDiTFirstHop
from pet_lr.rollout_first_hop import (
    get_linear_schedule_value,
    rollout_multistep_losses_first_hop,
    sample_chain_first_hop,
)
from pet_lr.path_guard import DEFAULT_OUTPUT_ROOT, ensure_repo_local_outputs_absent, resolve_data_disk_dir


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # Warn-only avoids hard crashes for rare nondeterministic ops while still surfacing them.
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def get_warmup_cosine_lr(
    step: int,
    max_steps: int,
    base_lr: float,
    warmup_steps: int,
    min_lr: float,
) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return float(base_lr) * float(step) / float(max(warmup_steps, 1))
    if max_steps <= warmup_steps:
        return float(base_lr)
    progress = float(step - warmup_steps) / float(max(max_steps - warmup_steps, 1))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr) + (float(base_lr) - float(min_lr)) * cosine


def _l2_grad_norm(params: list[torch.nn.Parameter]) -> float:
    sq = 0.0
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach()
        sq += float(g.float().pow(2).sum().item())
    return math.sqrt(max(sq, 0.0))


def _numel(params: list[torch.nn.Parameter]) -> int:
    return int(sum(int(p.numel()) for p in params))


def _format_alignment_summary(summary: Dict) -> str:
    if not summary:
        return "{}"
    parts = []
    for tp, stats in summary.items():
        matched = float(stats.get("psnr_clip3_matched_mean", 0.0))
        gap_roll1 = float(stats.get("psnr_gap_vs_roll1_db", 0.0))
        n_eval = int(float(stats.get("n_eval", 0)))
        parts.append(f"{tp}:matched={matched:.2f}dB,gap_roll1={gap_roll1:.2f}dB,n={n_eval}")
    return " | ".join(parts)


def _resolve_schedule_steps(
    total_steps: int,
    section: Dict,
    steps_key: str,
    ratio_key: str,
    default_steps: int,
) -> int:
    if steps_key in section and section[steps_key] is not None:
        return max(0, int(section[steps_key]))
    if ratio_key in section and section[ratio_key] is not None:
        ratio = float(section[ratio_key])
        if ratio < 0.0:
            raise ValueError(f"{ratio_key} must be >= 0, got {ratio}")
        return max(0, int(round(ratio * float(total_steps))))
    return max(0, int(default_steps))


def _resolve_pair_sample_probs(cfg: Dict, num_pairs: int) -> torch.Tensor | None:
    transport_cfg = cfg.get("transport", {})
    probs_cfg = transport_cfg.get("pair_sample_probs", None)
    if probs_cfg is None:
        return None
    probs = [float(x) for x in probs_cfg]
    if len(probs) != num_pairs:
        raise ValueError(
            f"transport.pair_sample_probs length ({len(probs)}) must equal number of pair hops ({num_pairs})"
        )
    p = torch.tensor(probs, dtype=torch.float64)
    if not torch.isfinite(p).all():
        raise ValueError("transport.pair_sample_probs must be finite (no NaN/Inf)")
    if torch.any(p < 0):
        raise ValueError("transport.pair_sample_probs must be non-negative")
    s = float(p.sum().item())
    if s <= 0.0:
        raise ValueError("transport.pair_sample_probs sum must be > 0")
    return p / s


def _resolve_pair_loss_weights(cfg: Dict, num_pairs: int, hop_idx: torch.Tensor) -> torch.Tensor | None:
    transport_cfg = cfg.get("transport", {})
    weighting = str(transport_cfg.get("pair_weighting", "none")).lower()
    if weighting != "manual":
        return None
    weights_cfg = transport_cfg.get("pair_loss_weights", None)
    if weights_cfg is None:
        return None
    weights = [float(x) for x in weights_cfg]
    if len(weights) != num_pairs:
        raise ValueError(
            f"transport.pair_loss_weights length ({len(weights)}) must equal number of pair hops ({num_pairs})"
        )
    w = torch.tensor(weights, dtype=torch.float32, device=hop_idx.device)
    if not torch.isfinite(w).all():
        raise ValueError("transport.pair_loss_weights must be finite (no NaN/Inf)")
    if torch.any(w < 0):
        raise ValueError("transport.pair_loss_weights must be non-negative")
    return w[hop_idx.long()]


def build_dataloaders(cfg: Dict) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    train_cfg = cfg.get("training", {})
    print("[startup] building datasets/dataloaders...", flush=True)
    if not bool(data_cfg.get("verify_alignment", True)):
        raise RuntimeError(
            "docs/main.md requires data.verify_alignment=true. "
            "Disabling alignment checks is not allowed for first-hop training."
        )
    alignment_audit_json = str(data_cfg.get("alignment_audit_json", "")).strip()
    if not alignment_audit_json:
        raise RuntimeError(
            "data.alignment_audit_json is required. "
            "Generate it with check_alignment_224_clip3.py (calc_psnr_clip3)."
        )
    if not os.path.exists(alignment_audit_json):
        raise RuntimeError(f"alignment_audit_json does not exist: {alignment_audit_json}")
    latent_dir = data_cfg["latent_dir"]
    common = dict(
        raw_data_dir=data_cfg["raw_data_dir"],
        clamp_max=float(data_cfg.get("clamp_max", 10.0)),
        t_map=data_cfg["t_map"],
        rollout_timepoints=data_cfg.get("rollout_timepoints", ["D50", "D20", "D10", "D4", "NORMAL"]),
        verify_alignment=bool(data_cfg.get("verify_alignment", True)),
        alignment_check_num_samples=int(data_cfg.get("alignment_check_num_samples", 16)),
        alignment_audit_json=alignment_audit_json,
        image_size=int(data_cfg.get("image_size", 224)),
    )
    train_set = PETFirstHopAligned4HopDataset(
        latent_path=os.path.join(latent_dir, "latents_train.pt"),
        split="train",
        include_x_rollout_first=bool(data_cfg.get("train_include_x_rollout_first", True)),
        include_full_x_rollout=bool(data_cfg.get("train_include_full_x_rollout", False)),
        **common,
    )
    print(f"[startup] train dataset ready: {len(train_set)} pair-samples", flush=True)
    val_set = PETFirstHopAligned4HopDataset(
        latent_path=os.path.join(latent_dir, "latents_val.pt"),
        split="val",
        include_x_rollout_first=bool(data_cfg.get("val_include_x_rollout_first", True)),
        include_full_x_rollout=bool(data_cfg.get("val_include_full_x_rollout", True)),
        **common,
    )
    print(f"[startup] val dataset ready: {len(val_set)} pair-samples", flush=True)

    if train_set.alignment_summary:
        print("[alignment][train]", _format_alignment_summary(train_set.alignment_summary), flush=True)
    if val_set.alignment_summary:
        print("[alignment][val]", _format_alignment_summary(val_set.alignment_summary), flush=True)

    hop0_train = Hop0OnlyViewDataset(train_set)
    hop0_val = Hop0OnlyViewDataset(val_set)

    main_batch_size = int(data_cfg.get("batch_size", 4))
    hop0_batch_size = int(data_cfg.get("hop0_aux_batch_size", max(1, math.ceil(main_batch_size * 0.25))))
    num_workers = int(data_cfg.get("num_workers", 0))
    max_steps = int(train_cfg.get("max_steps", 0))
    if max_steps <= 0:
        raise ValueError(f"training.max_steps must be > 0 before dataloader construction, got {max_steps}")

    pair_probs = _resolve_pair_sample_probs(cfg, num_pairs=len(train_set.PAIRS))
    if pair_probs is None:
        main_sampler = None
        main_shuffle = True
    else:
        pair_counts = torch.zeros(len(train_set.PAIRS), dtype=torch.double)
        for pair_idx, _ in train_set.index:
            pair_counts[pair_idx] += 1.0
        sample_weights = torch.empty(len(train_set.index), dtype=torch.double)
        for i, (pair_idx, _) in enumerate(train_set.index):
            denom = max(float(pair_counts[pair_idx].item()), 1.0)
            sample_weights[i] = pair_probs[pair_idx] / denom
        main_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_set.index),
            replacement=True,
        )
        main_shuffle = False
        print("[transport] pair_sample_probs(normalized)=", [float(x) for x in pair_probs.tolist()])

    main_train_loader = DataLoader(
        train_set,
        batch_size=main_batch_size,
        shuffle=main_shuffle,
        sampler=main_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    # docs/main.md requires every training step to receive an explicit hop0 batch.
    # Use replacement sampling sized to the planned train budget so normal epoch
    # boundaries never silently disable hop0 supervision.
    hop0_sampler = RandomSampler(
        hop0_train,
        replacement=True,
        num_samples=max_steps * hop0_batch_size,
    )
    hop0_train_loader = DataLoader(
        hop0_train,
        batch_size=hop0_batch_size,
        shuffle=False,
        sampler=hop0_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    main_val_loader = DataLoader(
        val_set,
        batch_size=main_batch_size,
        shuffle=bool(data_cfg.get("val_shuffle", False)),
        num_workers=num_workers,
        pin_memory=True,
    )
    hop0_val_loader = DataLoader(
        hop0_val,
        batch_size=hop0_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    print(
        f"[startup] dataloaders ready: main_bs={main_batch_size}, hop0_aux_bs={hop0_batch_size}, workers={num_workers}",
        flush=True,
    )
    return main_train_loader, hop0_train_loader, main_val_loader, hop0_val_loader


def compute_pair_losses(
    model: PETFlowDiTFirstHop,
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg: Dict,
) -> Dict[str, torch.Tensor]:
    pair_cfg = cfg["loss"].get("pair", {})
    transport_cfg = cfg.get("transport", {})
    v_weight = float(transport_cfg.get("velocity_loss_weight", pair_cfg.get("velocity_weight", 1.0)))
    endpoint_weight = float(transport_cfg.get("endpoint_loss_weight", pair_cfg.get("endpoint_weight", 1.0)))
    endpoint_pair_weighting = bool(transport_cfg.get("endpoint_pair_weighting", False))
    endpoint_dt_normalize = bool(transport_cfg.get("endpoint_dt_normalize", False))

    dt = (batch["t_dst"] - batch["t_src"]).view(-1, 1, 1, 1).clamp_min(1e-8)
    v_target_raw = (batch["z_dst"] - batch["z_src"]) / dt
    if model.target_normalize:
        sigma = model.sigma_for_hop(batch["hop_idx"]).clamp_min(1e-8)
        v_target = v_target_raw / sigma
    else:
        v_target = v_target_raw

    vel_err = (out["v_total_raw"] - v_target).pow(2).mean(dim=(1, 2, 3))
    if endpoint_dt_normalize:
        end_err = ((out["z_pred"] - batch["z_dst"]) / dt).pow(2).mean(dim=(1, 2, 3))
    else:
        end_err = (out["z_pred"] - batch["z_dst"]).pow(2).mean(dim=(1, 2, 3))
    vel_err_unweighted = vel_err.mean()
    end_err_unweighted = end_err.mean()
    sample_weights = _resolve_pair_loss_weights(cfg, num_pairs=int(model.num_hops), hop_idx=batch["hop_idx"])
    if sample_weights is None:
        loss_velocity = vel_err_unweighted
    else:
        # Match RAE MeanFlow semantics: pair weights scale velocity term before batch mean.
        loss_velocity = (vel_err * sample_weights).mean()
    if sample_weights is None or not endpoint_pair_weighting:
        loss_endpoint = end_err_unweighted
    else:
        loss_endpoint = (end_err * sample_weights).mean()

    # Optional: rebalance velocity term when it is consistently smaller than endpoint term.
    # This keeps v supervision from being numerically dominated by endpoint MSE.
    rebalance_cfg = pair_cfg.get("velocity_rebalance", {})
    rebalance_enabled = bool(rebalance_cfg.get("enabled", False))
    rebalance_scale = 1.0
    if rebalance_enabled:
        eps = float(rebalance_cfg.get("eps", 1.0e-12))
        clip_min = float(rebalance_cfg.get("clip_min", 1.0))
        clip_max = float(rebalance_cfg.get("clip_max", 10.0))
        mode = str(rebalance_cfg.get("mode", "sqrt_ratio")).lower()
        ratio = (loss_endpoint.detach() + eps) / (loss_velocity.detach() + eps)
        if mode == "ratio":
            scale_t = ratio
        else:
            scale_t = torch.sqrt(ratio)
        scale_t = torch.clamp(scale_t, min=clip_min, max=clip_max)
        rebalance_scale = float(scale_t.item())

    velocity_weight_eff = v_weight * rebalance_scale
    total = velocity_weight_eff * loss_velocity + endpoint_weight * loss_endpoint
    return {
        "total": total,
        "velocity": loss_velocity,
        "endpoint": loss_endpoint,
        "velocity_unweighted": vel_err_unweighted,
        "endpoint_unweighted": end_err_unweighted,
        "velocity_weight_eff": torch.tensor(velocity_weight_eff, device=loss_velocity.device, dtype=loss_velocity.dtype),
        "endpoint_weight_eff": torch.tensor(endpoint_weight, device=loss_velocity.device, dtype=loss_velocity.dtype),
        "velocity_rebalance_scale": torch.tensor(rebalance_scale, device=loss_velocity.device, dtype=loss_velocity.dtype),
    }


def resolve_rollout_straight_through(roll_cfg: Dict, alpha: float) -> bool:
    base = bool(roll_cfg.get("straight_through", True))
    mode = str(roll_cfg.get("straight_through_mode", "fixed")).lower()
    if mode in ("fixed", "always"):
        return base
    if mode in ("never", "disabled", "off"):
        return False
    if mode in ("alpha_gated", "alpha_gate", "gated"):
        if not base:
            return False
        alpha_min = float(roll_cfg.get("straight_through_alpha_min", 1.0))
        return float(alpha) >= alpha_min
    raise ValueError(
        "training.rollout.straight_through_mode must be one of "
        "{fixed, always, never, disabled, off, alpha_gated, alpha_gate, gated}"
    )


def resolve_rollout_lambda(base_lambda: float, alpha: float, roll_cfg: Dict) -> tuple[float, float]:
    mode = str(roll_cfg.get("lambda_scale_mode", "none")).lower()
    if mode in ("none", "fixed"):
        return float(base_lambda), 1.0
    if mode == "alpha":
        scale = float(alpha)
        return float(base_lambda) * scale, scale
    if mode in ("alpha_floor", "alpha_min"):
        alpha_floor = float(roll_cfg.get("lambda_scale_alpha_floor", 0.0))
        scale = max(float(alpha), alpha_floor)
        return float(base_lambda) * scale, scale
    raise ValueError("training.rollout.lambda_scale_mode must be one of {none, fixed, alpha, alpha_floor, alpha_min}")


def compute_rollout_losses(
    model: PETFlowDiTFirstHop,
    main_batch: Dict[str, torch.Tensor],
    cfg: Dict,
    global_step: int,
    rollout_times: list[float],
) -> Dict[str, torch.Tensor]:
    roll_cfg = cfg["training"].get("rollout", {})
    num_steps = max(len(rollout_times) - 1, 0)
    step_weights_cfg = roll_cfg.get("step_weights", None)
    if step_weights_cfg is None:
        step_weights = [1.0] * num_steps
    else:
        step_weights = list(step_weights_cfg)
        if len(step_weights) != num_steps:
            raise ValueError(
                f"rollout.step_weights length ({len(step_weights)}) "
                f"must equal rollout steps ({num_steps})"
            )
    if not bool(roll_cfg.get("enabled", True)):
        z = main_batch["z_src"].new_zeros(())
        return {
            "loss_total": z,
            "step_losses": [z for _ in range(num_steps)],
            "lambda_roll": torch.tensor(0.0, device=z.device),
            "alpha_mix": torch.tensor(0.0, device=z.device),
        }

    alpha_mix = get_linear_schedule_value(
        global_step=global_step,
        warmup_steps=int(roll_cfg.get("warmup_steps", 500)),
        ramp_steps=int(roll_cfg.get("ramp_steps", 1000)),
        start=float(roll_cfg.get("alpha_start", 0.0)),
        end=float(roll_cfg.get("alpha_end", 1.0)),
    )
    lambda_end_default = float(roll_cfg.get("endpoint_loss_weight", 1.0))
    lambda_roll = get_linear_schedule_value(
        global_step=global_step,
        warmup_steps=int(roll_cfg.get("warmup_steps", 500)),
        ramp_steps=int(roll_cfg.get("ramp_steps", 1000)),
        start=float(roll_cfg.get("lambda_start", 0.0)),
        end=float(roll_cfg.get("lambda_end", lambda_end_default)),
    )
    lambda_roll_eff, lambda_roll_scale = resolve_rollout_lambda(
        base_lambda=float(lambda_roll),
        alpha=float(alpha_mix),
        roll_cfg=roll_cfg,
    )
    straight_through = resolve_rollout_straight_through(roll_cfg=roll_cfg, alpha=float(alpha_mix))

    out = rollout_multistep_losses_first_hop(
        model=model,
        z_rollout=main_batch["z_rollout"],
        rollout_times=rollout_times,
        x_rollout_first=main_batch.get("x_rollout_first"),
        alpha=alpha_mix,
        straight_through=straight_through,
        loss_type=str(roll_cfg.get("loss_type", "mse")),
        step_weights=step_weights,
    )
    step_losses = out["step_losses"]
    return {
        "loss_total": out["loss_total"],
        "step_losses": step_losses,
        "lambda_roll": torch.tensor(lambda_roll_eff, device=out["loss_total"].device),
        "lambda_roll_base": torch.tensor(lambda_roll, device=out["loss_total"].device),
        "lambda_roll_scale": torch.tensor(lambda_roll_scale, device=out["loss_total"].device),
        "straight_through": torch.tensor(1.0 if straight_through else 0.0, device=out["loss_total"].device),
        "alpha_mix": torch.tensor(alpha_mix, device=out["loss_total"].device),
    }


def compute_foc_losses(
    model: PETFlowDiTFirstHop,
    hop0_batch: Dict[str, torch.Tensor],
    cfg: Dict,
    global_step: int,
    rollout_times: list[float],
) -> Dict[str, torch.Tensor]:
    """FOC-lite: First-Order ODE Calibration for hop0.

    Compares full-step prediction with two half-step predictions on hop0
    (D50→D20). If the velocity field is accurate, both should agree.
    Penalizing the discrepancy reduces the numerical integration error
    of the first hop — the dominant bottleneck in the cascade.

    Reference: Consistency Flow Matching (Yang et al., arXiv 2407.02398, 2024)

    Full step:  z_full = z_D50 + v(z_D50, t_D50, t_D20) × dt
    Half steps: z_mid  = z_D50 + v(z_D50, t_D50, t_mid) × dt/2
                z_half = z_mid + v(z_mid, t_mid, t_D20) × dt/2
    Loss:       L_foc  = MSE(z_full, sg(z_half))   [sg = stop-gradient]
    """
    foc_cfg = cfg["training"].get("foc", {})
    device = hop0_batch["z_src"].device

    def _zero():
        z = hop0_batch["z_src"].new_zeros(())
        return {
            "loss_total": z,
            "loss_consistency": z,
            "lambda_foc": torch.tensor(0.0, device=device),
            "z_full_norm": z,
            "z_half_norm": z,
            "gap_norm": z,
        }

    if not bool(foc_cfg.get("enabled", False)):
        return _zero()

    # Schedule
    lambda_foc = get_linear_schedule_value(
        global_step=global_step,
        warmup_steps=int(foc_cfg.get("warmup_steps", 500)),
        ramp_steps=int(foc_cfg.get("ramp_steps", 2000)),
        start=float(foc_cfg.get("lambda_start", 0.0)),
        end=float(foc_cfg.get("lambda_max", 0.08)),
    )

    if len(rollout_times) < 2:
        return _zero()

    t_src_val = float(rollout_times[0])  # D50 = 2.0
    t_dst_val = float(rollout_times[1])  # D20 = 5.0
    t_mid_val = (t_src_val + t_dst_val) / 2.0  # midpoint = 3.5

    bsz = hop0_batch["z_src"].shape[0]
    z_src = hop0_batch["z_src"]
    x_src = hop0_batch.get("x_src")  # pixel conditioning for hop0
    hop_idx = torch.zeros(bsz, device=device, dtype=torch.long)

    # Time tensors
    t_src = torch.full((bsz,), t_src_val, device=device)
    t_dst = torch.full((bsz,), t_dst_val, device=device)
    t_mid = torch.full((bsz,), t_mid_val, device=device)

    # --- Full step: D50 → D20 ---
    out_full = model.predict_latent_step(
        z_src=z_src, t_src=t_src, t_dst=t_dst,
        hop_idx=hop_idx, x_src_img=x_src,
    )
    z_full = out_full["z_pred"]

    # --- Half step 1: D50 → mid ---
    out_half1 = model.predict_latent_step(
        z_src=z_src, t_src=t_src, t_dst=t_mid,
        hop_idx=hop_idx, x_src_img=x_src,
    )
    z_mid = out_half1["z_pred"]

    # --- Half step 2: mid → D20 ---
    # Note: at mid-point, pixel conditioning still uses x_D50 (same hop0 source).
    out_half2 = model.predict_latent_step(
        z_src=z_mid, t_src=t_mid, t_dst=t_dst,
        hop_idx=hop_idx, x_src_img=x_src,
    )
    z_half = out_half2["z_pred"]

    # --- Consistency loss ---
    detach_target = bool(foc_cfg.get("detach_target", True))
    loss_type = str(foc_cfg.get("loss_type", "mse")).lower()

    target = z_half.detach() if detach_target else z_half
    if loss_type == "l1":
        loss_consistency = F.l1_loss(z_full, target)
    else:
        loss_consistency = F.mse_loss(z_full, target)

    # Diagnostic norms
    with torch.no_grad():
        z_full_norm = z_full.detach().abs().mean()
        z_half_norm = z_half.detach().abs().mean()
        gap_norm = (z_full.detach() - z_half.detach()).abs().mean()

    return {
        "loss_total": loss_consistency,
        "loss_consistency": loss_consistency,
        "lambda_foc": torch.tensor(lambda_foc, device=device),
        "z_full_norm": z_full_norm,
        "z_half_norm": z_half_norm,
        "gap_norm": gap_norm,
    }


def compute_hop0_image_losses(
    model: PETFlowDiTFirstHop,
    hop0_batch: Dict[str, torch.Tensor],
    cfg: Dict,
) -> Dict[str, torch.Tensor]:
    img_cfg = cfg["loss"].get("image_aux", {})
    out = model.predict_latent_step(
        z_src=hop0_batch["z_src"],
        t_src=hop0_batch["t_src"],
        t_dst=hop0_batch["t_dst"],
        hop_idx=hop0_batch["hop_idx"],
        x_src_img=hop0_batch["x_src"],
    )
    x_pred = model.decode_crop(out["z_pred"], crop_size=int(cfg["data"].get("image_size", 224)))
    loss_dict = compute_first_hop_image_loss(
        x_pred=x_pred,
        x_gt=hop0_batch["x_dst"],
        border_width=int(img_cfg.get("border_width", 14)),
        border_weight=float(img_cfg.get("border_weight", 2.0)),
        w_l1=float(img_cfg.get("l1_weight", 1.0)),
        w_ssim=float(img_cfg.get("ssim_weight", 0.25)),
        w_seam=float(img_cfg.get("seam_weight", 0.10)),
        seam_patch_size=int(img_cfg.get("seam_patch_size", 14)),
        use_extended_seam=bool(img_cfg.get("use_extended_seam", False)),
        seam_zone_width=int(img_cfg.get("seam_zone_width", 3)),
    )
    return {
        "total": loss_dict["total"],
        "l1": loss_dict["l1"],
        "ssim": loss_dict["ssim"],
        "seam": loss_dict["seam"],
        "ssim_raw_mean": loss_dict["ssim_raw_mean"],
        "ssim_raw_min": loss_dict["ssim_raw_min"],
        "ssim_raw_max": loss_dict["ssim_raw_max"],
        "ssim_over1_frac": loss_dict["ssim_over1_frac"],
        "ssim_below0_frac": loss_dict["ssim_below0_frac"],
        "ssim_clamped_mean": loss_dict["ssim_clamped_mean"],
    }


def _resolve_eval_window(
    total_batches: int,
    max_batches_cfg: int,
    global_step: int,
    eval_interval: int,
    window_mode: str,
) -> tuple[int, int]:
    if total_batches <= 0:
        return 0, 0
    if max_batches_cfg <= 0 or max_batches_cfg >= total_batches:
        return 0, total_batches

    take = int(max_batches_cfg)
    mode = str(window_mode).lower()
    if mode == "head":
        return 0, take
    if mode == "rolling":
        eval_idx = max((int(global_step) - 1) // max(int(eval_interval), 1), 0)
        num_windows = max(int(math.ceil(float(total_batches) / float(take))), 1)
        start = (eval_idx % num_windows) * take
        if start >= total_batches:
            start = 0
        return start, take
    raise ValueError(f"training.val_window_mode must be one of {{head, rolling}}, got {window_mode!r}")


def _iter_loader_window(loader: DataLoader, start: int, take: int):
    if take <= 0:
        return
    total = len(loader)
    if total <= 0:
        return
    if start < 0 or start >= total:
        raise ValueError(f"Invalid eval window start={start} for loader of len={total}")

    remaining = int(take)
    it = iter(loader)
    for _ in range(start):
        try:
            next(it)
        except StopIteration:
            it = iter(loader)
            break

    while remaining > 0:
        try:
            batch = next(it)
        except StopIteration:
            break
        yield batch
        remaining -= 1


@torch.no_grad()
def evaluate(
    model: PETFlowDiTFirstHop,
    main_val_loader: DataLoader,
    hop0_val_loader: DataLoader,
    cfg: Dict,
    device: torch.device,
    rollout_times: list[float],
    global_step: int,
) -> Dict[str, float]:
    model.eval()
    max_main_batches_cfg = int(cfg["training"].get("max_val_batches", 32))
    max_hop0_batches_cfg = int(cfg["training"].get("max_hop0_val_batches", max_main_batches_cfg))
    window_mode = str(cfg["training"].get("val_window_mode", "rolling")).lower()
    eval_interval = int(cfg["training"].get("eval_interval", 250))
    main_start, max_main_batches = _resolve_eval_window(
        total_batches=len(main_val_loader),
        max_batches_cfg=max_main_batches_cfg,
        global_step=global_step,
        eval_interval=eval_interval,
        window_mode=window_mode,
    )
    hop0_start, max_hop0_batches = _resolve_eval_window(
        total_batches=len(hop0_val_loader),
        max_batches_cfg=max_hop0_batches_cfg,
        global_step=global_step,
        eval_interval=eval_interval,
        window_mode=window_mode,
    )
    roll_cfg = cfg["training"].get("rollout", {})
    eval_chain_on_all_samples = bool(cfg["training"].get("eval_chain_on_all_samples", True))
    eval_chain_unique_slices = bool(cfg["training"].get("eval_chain_unique_slices", True))
    eval_chain_max_unique_slices = int(cfg["training"].get("eval_chain_max_unique_slices", 0))
    if eval_chain_max_unique_slices < 0:
        raise ValueError(
            f"training.eval_chain_max_unique_slices must be >= 0, got {eval_chain_max_unique_slices}"
        )
    num_steps = max(len(rollout_times) - 1, 0)
    step_weights_cfg = roll_cfg.get("step_weights", None)
    if step_weights_cfg is None:
        step_weights = [1.0] * num_steps
    else:
        step_weights = list(step_weights_cfg)
        if len(step_weights) != num_steps:
            raise ValueError(
                f"rollout.step_weights length ({len(step_weights)}) "
                f"must equal rollout steps ({num_steps})"
            )

    sums = {
        "val_pair_total": 0.0,
        "val_pair_velocity": 0.0,
        "val_pair_endpoint": 0.0,
        "val_rollout_total": 0.0,
        "val_hop0_img_total": 0.0,
        "val_hop0_img_l1": 0.0,
        "val_hop0_img_ssim": 0.0,
        "val_hop0_img_seam": 0.0,
        "val_foc_total": 0.0,
        "val_foc_gap": 0.0,
        "val_chain_d20_mse": 0.0,
        "val_chain_d10_mse": 0.0,
        "val_chain_d4_mse": 0.0,
        "val_chain_normal_mse": 0.0,
        "val_chain_d20_raw_mse": 0.0,
        "val_chain_d10_raw_mse": 0.0,
        "val_chain_d4_raw_mse": 0.0,
        "val_chain_normal_raw_mse": 0.0,
    }
    for i in range(num_steps):
        sums[f"val_rollout_step_{i}"] = 0.0

    main_count = 0
    chain_samples = 0
    chain_unique_slice_ids: set[int] = set()
    for batch in _iter_loader_window(main_val_loader, start=main_start, take=max_main_batches):
        slice_idx_cpu = None
        if "slice_idx" in batch and torch.is_tensor(batch["slice_idx"]):
            slice_idx_cpu = batch["slice_idx"].detach().clone().cpu().long()
        batch = move_batch_to_device(batch, device)
        out = model.predict_latent_step(
            z_src=batch["z_src"],
            t_src=batch["t_src"],
            t_dst=batch["t_dst"],
            hop_idx=batch["hop_idx"],
            x_src_img=batch["x_src"],
        )
        pair_losses = compute_pair_losses(model, out, batch, cfg)
        sums["val_pair_total"] += float(pair_losses["total"].item())
        sums["val_pair_velocity"] += float(pair_losses["velocity"].item())
        sums["val_pair_endpoint"] += float(pair_losses["endpoint"].item())

        if bool(roll_cfg.get("enabled", True)):
            eval_alpha = roll_cfg.get("eval_alpha", roll_cfg.get("alpha_end", 1.0))
            eval_straight_through = resolve_rollout_straight_through(roll_cfg=roll_cfg, alpha=float(eval_alpha))
            roll_out = rollout_multistep_losses_first_hop(
                model=model,
                z_rollout=batch["z_rollout"],
                rollout_times=rollout_times,
                x_rollout_first=batch.get("x_rollout_first"),
                alpha=float(eval_alpha),
                straight_through=eval_straight_through,
                loss_type=str(roll_cfg.get("loss_type", "mse")),
                step_weights=step_weights,
            )
            sums["val_rollout_total"] += float(roll_out["loss_total"].item())
            for i, step_loss in enumerate(roll_out["step_losses"]):
                sums[f"val_rollout_step_{i}"] += float(step_loss.item())

        if "x_rollout" in batch:
            if slice_idx_cpu is None:
                raise RuntimeError("Validation batch must contain slice_idx for chain metric de-duplication")
            slice_idx_batch = slice_idx_cpu

            if eval_chain_on_all_samples:
                if eval_chain_unique_slices:
                    keep: list[int] = []
                    for pos, sid in enumerate(slice_idx_batch.tolist()):
                        sid_i = int(sid)
                        if sid_i in chain_unique_slice_ids:
                            continue
                        chain_unique_slice_ids.add(sid_i)
                        keep.append(pos)
                        if eval_chain_max_unique_slices > 0 and len(chain_unique_slice_ids) >= eval_chain_max_unique_slices:
                            break
                    if keep:
                        keep_idx = torch.tensor(keep, device=batch["z_rollout"].device, dtype=torch.long)
                        z_d50 = batch["z_rollout"].index_select(0, keep_idx)[:, 0]
                        x_d50 = batch["x_rollout"].index_select(0, keep_idx)[:, 0]
                        x_roll = batch["x_rollout"].index_select(0, keep_idx)
                    else:
                        z_d50 = batch["z_rollout"][0:0, 0]
                        x_d50 = batch["x_rollout"][0:0, 0]
                        x_roll = batch["x_rollout"][0:0]
                else:
                    chain_unique_slice_ids.update(int(s) for s in slice_idx_batch.tolist())
                    z_d50 = batch["z_rollout"][:, 0]
                    x_d50 = batch["x_rollout"][:, 0]
                    x_roll = batch["x_rollout"]
            else:
                mask = batch["hop_idx"] == 0
                mask_cpu = mask.detach().cpu()
                if eval_chain_unique_slices:
                    keep_hop0: list[int] = []
                    for pos, sid in enumerate(slice_idx_batch[mask_cpu].tolist()):
                        sid_i = int(sid)
                        if sid_i in chain_unique_slice_ids:
                            continue
                        chain_unique_slice_ids.add(sid_i)
                        keep_hop0.append(pos)
                        if eval_chain_max_unique_slices > 0 and len(chain_unique_slice_ids) >= eval_chain_max_unique_slices:
                            break
                    if keep_hop0:
                        true_pos = torch.where(mask)[0]
                        keep_idx = true_pos[torch.tensor(keep_hop0, device=true_pos.device, dtype=torch.long)]
                        z_d50 = batch["z_src"].index_select(0, keep_idx)
                        x_d50 = batch["x_src"].index_select(0, keep_idx)
                        x_roll = batch["x_rollout"].index_select(0, keep_idx)
                    else:
                        z_d50 = batch["z_src"][0:0]
                        x_d50 = batch["x_src"][0:0]
                        x_roll = batch["x_rollout"][0:0]
                else:
                    chain_unique_slice_ids.update(int(s) for s in slice_idx_batch[mask_cpu].tolist())
                    z_d50 = batch["z_src"][mask]
                    x_d50 = batch["x_src"][mask]
                    x_roll = batch["x_rollout"][mask]
            if z_d50.shape[0] > 0:
                z_chain = sample_chain_first_hop(model, z_d50, x_d50, rollout_times=rollout_times)
                if len(z_chain) < 5 or int(x_roll.shape[1]) < 5:
                    raise ValueError(
                        "first-hop evaluation expects at least 5 rollout points "
                        "(D50, D20, D10, D4, NORMAL)"
                    )
                _cs = int(cfg["data"].get("image_size", 224))
                x_d20_pred = model.decode_crop(z_chain[1], crop_size=_cs)
                x_d10_pred = model.decode_crop(z_chain[2], crop_size=_cs)
                x_d4_pred = model.decode_crop(z_chain[3], crop_size=_cs)
                x_n_pred = model.decode_crop(z_chain[-1], crop_size=_cs)
                n = int(z_d50.shape[0])
                sums["val_chain_d20_mse"] += float(F.mse_loss(x_d20_pred, x_roll[:, 1]).item()) * n
                sums["val_chain_d10_mse"] += float(F.mse_loss(x_d10_pred, x_roll[:, 2]).item()) * n
                sums["val_chain_d4_mse"] += float(F.mse_loss(x_d4_pred, x_roll[:, 3]).item()) * n
                sums["val_chain_normal_mse"] += float(F.mse_loss(x_n_pred, x_roll[:, -1]).item()) * n
                # Raw decode metrics (skip refiner) for causal attribution
                if model.seam_refiner_enabled:
                    x_d20_raw = model.decode_crop(z_chain[1], crop_size=_cs, apply_refiner=False)
                    x_d10_raw = model.decode_crop(z_chain[2], crop_size=_cs, apply_refiner=False)
                    x_d4_raw = model.decode_crop(z_chain[3], crop_size=_cs, apply_refiner=False)
                    x_n_raw = model.decode_crop(z_chain[-1], crop_size=_cs, apply_refiner=False)
                    sums["val_chain_d20_raw_mse"] += float(F.mse_loss(x_d20_raw, x_roll[:, 1]).item()) * n
                    sums["val_chain_d10_raw_mse"] += float(F.mse_loss(x_d10_raw, x_roll[:, 2]).item()) * n
                    sums["val_chain_d4_raw_mse"] += float(F.mse_loss(x_d4_raw, x_roll[:, 3]).item()) * n
                    sums["val_chain_normal_raw_mse"] += float(F.mse_loss(x_n_raw, x_roll[:, -1]).item()) * n
                chain_samples += n

        main_count += 1

    hop0_count = 0
    for batch in _iter_loader_window(hop0_val_loader, start=hop0_start, take=max_hop0_batches):
        batch = move_batch_to_device(batch, device)
        img_losses = compute_hop0_image_losses(model, batch, cfg)
        sums["val_hop0_img_total"] += float(img_losses["total"].item())
        sums["val_hop0_img_l1"] += float(img_losses["l1"].item())
        sums["val_hop0_img_ssim"] += float(img_losses["ssim"].item())
        sums["val_hop0_img_seam"] += float(img_losses["seam"].item())
        # FOC-lite eval on hop0 batches
        foc_losses = compute_foc_losses(
            model=model, hop0_batch=batch, cfg=cfg,
            global_step=global_step, rollout_times=rollout_times,
        )
        sums["val_foc_total"] += float(foc_losses["loss_total"].item())
        sums["val_foc_gap"] += float(foc_losses["gap_norm"].item())
        hop0_count += 1

    require_chain_metrics = bool(cfg["training"].get("require_chain_metrics", True))
    if require_chain_metrics and chain_samples <= 0:
        raise RuntimeError(
            "No chain samples were evaluated. "
            "Please ensure val_include_full_x_rollout=true and max_val_batches > 0."
        )

    model.train()
    out = {}
    for k, v in sums.items():
        if k.startswith("val_hop0") or k.startswith("val_foc"):
            out[k] = v / max(hop0_count, 1)
        elif k.startswith("val_chain"):
            out[k] = v / max(chain_samples, 1)
        else:
            out[k] = v / max(main_count, 1)
    if not model.seam_refiner_enabled:
        for k in (
            "val_chain_d20_raw_mse",
            "val_chain_d10_raw_mse",
            "val_chain_d4_raw_mse",
            "val_chain_normal_raw_mse",
        ):
            out[k] = float("nan")
    out["val_chain_tail_mse"] = (
        out["val_chain_d10_mse"] + out["val_chain_d4_mse"] + out["val_chain_normal_mse"]
    ) / 3.0
    out["val_chain_samples"] = float(chain_samples)
    out["val_chain_unique_slices"] = float(len(chain_unique_slice_ids))
    out["val_main_batches_evaluated"] = float(main_count)
    out["val_hop0_batches_evaluated"] = float(hop0_count)
    out["val_main_window_start_batch"] = float(main_start)
    out["val_hop0_window_start_batch"] = float(hop0_start)
    return out


def resolve_best_selection_score(metrics: Dict[str, float], train_cfg: Dict) -> tuple[float, str]:
    best_metric_name = str(train_cfg.get("best_metric", "val_rollout_total"))
    if best_metric_name in ("val_multi_objective", "multi_objective"):
        terms_cfg = train_cfg.get("best_metric_terms", None)
        if terms_cfg is None:
            terms_cfg = [
                {"name": "val_chain_d20_mse", "weight": 1.0},
                {"name": "val_chain_d10_mse", "weight": 0.30},
                {"name": "val_chain_d4_mse", "weight": 0.30},
                {"name": "val_chain_normal_mse", "weight": 0.45},
            ]
        if not isinstance(terms_cfg, list) or len(terms_cfg) == 0:
            raise ValueError("training.best_metric_terms must be a non-empty list when best_metric=val_multi_objective")

        score = 0.0
        display_terms = []
        for i, term in enumerate(terms_cfg):
            if isinstance(term, str):
                metric_name = term
                weight = 1.0
            elif isinstance(term, dict):
                metric_name = str(term.get("name", "")).strip()
                weight = float(term.get("weight", 1.0))
            else:
                raise ValueError(
                    f"training.best_metric_terms[{i}] must be string or dict(name, weight), got {type(term)}"
                )
            if metric_name == "":
                raise ValueError(f"training.best_metric_terms[{i}].name is empty")
            if metric_name not in metrics:
                raise KeyError(
                    f"training.best_metric_terms[{i}] references unknown metric '{metric_name}'. "
                    f"Available keys: {sorted(metrics.keys())}"
                )
            score += weight * float(metrics[metric_name])
            display_terms.append(f"{weight:g}*{metric_name}")
        return score, "val_multi_objective(" + " + ".join(display_terms) + ")"

    if best_metric_name not in metrics:
        raise KeyError(
            f"training.best_metric references unknown metric '{best_metric_name}'. "
            f"Available keys: {sorted(metrics.keys())}"
        )
    return float(metrics[best_metric_name]), best_metric_name


def build_best_metric_signature(train_cfg: Dict) -> str:
    best_metric_name = str(train_cfg.get("best_metric", "val_rollout_total")).strip()
    if best_metric_name in ("val_multi_objective", "multi_objective"):
        terms_cfg = train_cfg.get("best_metric_terms", None)
        if terms_cfg is None:
            terms_cfg = [
                {"name": "val_chain_d20_mse", "weight": 1.0},
                {"name": "val_chain_d10_mse", "weight": 0.30},
                {"name": "val_chain_d4_mse", "weight": 0.30},
                {"name": "val_chain_normal_mse", "weight": 0.45},
            ]
        normalized_terms = []
        for i, term in enumerate(terms_cfg):
            if isinstance(term, str):
                metric_name = term.strip()
                weight = 1.0
            elif isinstance(term, dict):
                metric_name = str(term.get("name", "")).strip()
                weight = float(term.get("weight", 1.0))
            else:
                raise ValueError(
                    f"training.best_metric_terms[{i}] must be string or dict(name, weight), got {type(term)}"
                )
            if metric_name == "":
                raise ValueError(f"training.best_metric_terms[{i}].name is empty")
            normalized_terms.append({"name": metric_name, "weight": weight})
        payload = {"best_metric": "val_multi_objective", "terms": normalized_terms}
    else:
        payload = {"best_metric": best_metric_name}
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def save_checkpoint(
    model: PETFlowDiTFirstHop,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    step: int,
    output_dir: Path,
    name: str,
    rollout_timepoints: list[str],
    best_val: float | None = None,
    best_metric_name: str | None = None,
    best_metric_signature: str | None = None,
    best_d1_val: float | None = None,
    best_d1_metric_name: str | None = None,
    best_d1_guard_metric_name: str | None = None,
    best_d1_guard_best: float | None = None,
) -> None:
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["cuda"] = [s.cpu() for s in torch.cuda.get_rng_state_all()]
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "target_normalize": model.target_normalize,
        "rollout_path": rollout_timepoints,
        "first_hop_pixel_enabled": True,
        "best_val": float(best_val) if best_val is not None else None,
        "best_metric_name": best_metric_name,
        "best_metric_signature": best_metric_signature,
        "best_d1_val": float(best_d1_val) if best_d1_val is not None else None,
        "best_d1_metric_name": best_d1_metric_name,
        "best_d1_guard_metric_name": best_d1_guard_metric_name,
        "best_d1_guard_best": float(best_d1_guard_best) if best_d1_guard_best is not None else None,
        "rng_state": rng_state,
    }
    torch.save(ckpt, output_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train first-hop 224 latent transport")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--resume", default="", help="Optional checkpoint path to resume from")
    args = parser.parse_args()

    print(f"[startup] loading config: {args.config}", flush=True)
    cfg = load_config(args.config)
    train_cfg_boot = cfg.get("training", {})
    deterministic = bool(train_cfg_boot.get("deterministic", False))
    set_seed(int(cfg.get("seed", 42)), deterministic=deterministic)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if not bool(train_cfg_boot.get("freeze_rae", True)):
        raise RuntimeError(
            "docs/main.md requires training.freeze_rae=true for first-hop training. "
            "Decoder unfreezing is not allowed in this trainer."
        )

    repo_root = Path(__file__).resolve().parent
    ensure_repo_local_outputs_absent(repo_root)
    output_root = resolve_data_disk_dir(
        cfg.get("output_dir", str(DEFAULT_OUTPUT_ROOT)),
        arg_name="output_dir",
    )
    run_name = cfg.get("run_name", datetime.now().strftime("%m%d_%H%M%S"))
    output_dir = output_root / run_name
    resume_path = str(args.resume).strip()
    if resume_path.lower() == "auto":
        resume_path = str(output_dir / "last.pt")
    resume_enabled = bool(resume_path)
    if resume_enabled and not os.path.exists(resume_path):
        raise FileNotFoundError(f"--resume checkpoint not found: {resume_path}")
    require_fresh_output_dir = bool(train_cfg_boot.get("require_fresh_output_dir", False))
    if output_dir.exists() and require_fresh_output_dir and any(output_dir.iterdir()) and not resume_enabled:
        raise RuntimeError(
            f"Output dir already exists and is not empty: {output_dir}. "
            "Please change run_name or clean the directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.yaml"
    if resume_enabled and config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            existing_cfg = yaml.safe_load(f)
        if existing_cfg != cfg:
            allow_cfg_mismatch = bool(train_cfg_boot.get("resume_allow_config_mismatch", False))
            if not allow_cfg_mismatch:
                raise RuntimeError(
                    "Resume config mismatch with existing run config.yaml. "
                    "Set training.resume_allow_config_mismatch=true to bypass explicitly."
                )
            resume_cfg_path = output_dir / f"config.resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
            with open(resume_cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            print(f"[resume][warn] config mismatch allowed, wrote {resume_cfg_path}", flush=True)
    else:
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[startup] run dir: {output_dir}", flush=True)
    if resume_enabled:
        print(f"[startup] resume from: {resume_path}", flush=True)

    main_train_loader, hop0_train_loader, main_val_loader, hop0_val_loader = build_dataloaders(cfg)
    if len(main_train_loader) <= 0:
        raise RuntimeError("main_train_loader is empty")
    if len(hop0_train_loader) <= 0:
        raise RuntimeError("hop0_train_loader is empty")
    if len(main_val_loader) <= 0:
        raise RuntimeError("main_val_loader is empty")
    if len(hop0_val_loader) <= 0:
        raise RuntimeError("hop0_val_loader is empty")
    max_val_batches_cfg = int(train_cfg_boot.get("max_val_batches", 32))
    eval_chain_max_unique_cfg = int(train_cfg_boot.get("eval_chain_max_unique_slices", 0))
    is_truncated_by_batches = max_val_batches_cfg > 0 and max_val_batches_cfg < len(main_val_loader)
    is_truncated_by_unique = eval_chain_max_unique_cfg > 0
    if (
        bool(train_cfg_boot.get("require_chain_metrics", True))
        and bool(train_cfg_boot.get("eval_chain_on_all_samples", False))
        and bool(cfg.get("data", {}).get("val_shuffle", False))
        and (is_truncated_by_batches or is_truncated_by_unique)
    ):
        raise RuntimeError(
            "For reliable chain-based selection, set data.val_shuffle=false when "
            "training.eval_chain_on_all_samples=true and validation is truncated "
            "(max_val_batches < len(val_loader) or eval_chain_max_unique_slices > 0)."
        )
    roll_cfg = cfg.get("training", {}).get("rollout", {})
    rollout_tps = roll_cfg.get("path", cfg["data"].get("rollout_timepoints", ["D50", "D20", "D10", "D4", "NORMAL"]))
    data_rollout_tps = cfg["data"].get("rollout_timepoints", rollout_tps)
    if list(rollout_tps) != list(data_rollout_tps):
        raise ValueError(
            "training.rollout.path must match data.rollout_timepoints for first-hop trainer: "
            f"{list(rollout_tps)} vs {list(data_rollout_tps)}"
        )
    print("[rollout] path=", list(rollout_tps))
    rollout_times = [float(cfg["data"]["t_map"][tp]) for tp in rollout_tps]

    model = PETFlowDiTFirstHop(cfg, device=device).to(device)
    print("[startup] model initialized", flush=True)
    model.assert_decoder_frozen()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found")

    opt_cfg = cfg.get("optimizer", {})
    base_lr = float(opt_cfg.get("lr", 8.0e-5))
    global_weight_decay = float(opt_cfg.get("weight_decay", 0.01))
    backbone_weight_decay = float(opt_cfg.get("backbone_weight_decay", global_weight_decay))
    first_hop_weight_decay = float(opt_cfg.get("first_hop_weight_decay", global_weight_decay))
    if global_weight_decay < 0.0 or backbone_weight_decay < 0.0 or first_hop_weight_decay < 0.0:
        raise ValueError("optimizer weight_decay values must be >= 0")
    backbone_lr_mult = float(opt_cfg.get("backbone_lr_mult", 1.0))
    first_hop_lr_mult = float(opt_cfg.get("first_hop_lr_mult", 1.0))
    if backbone_lr_mult <= 0.0 or first_hop_lr_mult <= 0.0:
        raise ValueError("optimizer lr multipliers must be > 0")

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    backbone_ids = {id(p) for p in backbone_params}
    first_hop_params = [p for p in trainable_params if id(p) not in backbone_ids]

    param_groups = []
    if backbone_params:
        param_groups.append(
            {
                "name": "backbone",
                "params": backbone_params,
                "lr": base_lr * backbone_lr_mult,
                "lr_scale": backbone_lr_mult,
                "weight_decay": backbone_weight_decay,
            }
        )
    if first_hop_params:
        param_groups.append(
            {
                "name": "first_hop",
                "params": first_hop_params,
                "lr": base_lr * first_hop_lr_mult,
                "lr_scale": first_hop_lr_mult,
                "weight_decay": first_hop_weight_decay,
            }
        )
    if not param_groups:
        raise RuntimeError("No optimizer param groups were created")

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=global_weight_decay,
        betas=tuple(opt_cfg.get("betas", [0.9, 0.95])),
    )

    train_cfg = cfg["training"]
    best_metric_signature = build_best_metric_signature(train_cfg)
    max_steps = int(train_cfg.get("max_steps", 2000))
    if max_steps <= 0:
        raise ValueError(f"training.max_steps must be > 0, got {max_steps}")

    roll_cfg_runtime = train_cfg.get("rollout", {})
    image_aux_cfg = train_cfg.get("image_aux", {})

    rollout_warmup_steps = _resolve_schedule_steps(
        total_steps=max_steps,
        section=roll_cfg_runtime,
        steps_key="warmup_steps",
        ratio_key="warmup_ratio",
        default_steps=500,
    )
    rollout_ramp_steps = _resolve_schedule_steps(
        total_steps=max_steps,
        section=roll_cfg_runtime,
        steps_key="ramp_steps",
        ratio_key="ramp_ratio",
        default_steps=1000,
    )
    image_warmup_steps = _resolve_schedule_steps(
        total_steps=max_steps,
        section=image_aux_cfg,
        steps_key="warmup_steps",
        ratio_key="warmup_ratio",
        default_steps=200,
    )
    image_ramp_steps = _resolve_schedule_steps(
        total_steps=max_steps,
        section=image_aux_cfg,
        steps_key="ramp_steps",
        ratio_key="ramp_ratio",
        default_steps=600,
    )

    # Persist resolved values into runtime cfg so helper functions use one source of truth.
    roll_cfg_runtime["warmup_steps"] = rollout_warmup_steps
    roll_cfg_runtime["ramp_steps"] = rollout_ramp_steps
    image_aux_cfg["warmup_steps"] = image_warmup_steps
    image_aux_cfg["ramp_steps"] = image_ramp_steps

    foc_cfg_runtime = train_cfg.get("foc", {})
    foc_warmup_steps = _resolve_schedule_steps(
        total_steps=max_steps,
        section=foc_cfg_runtime,
        steps_key="warmup_steps",
        ratio_key="warmup_ratio",
        default_steps=500,
    )
    foc_ramp_steps = _resolve_schedule_steps(
        total_steps=max_steps,
        section=foc_cfg_runtime,
        steps_key="ramp_steps",
        ratio_key="ramp_ratio",
        default_steps=2000,
    )
    foc_cfg_runtime["warmup_steps"] = foc_warmup_steps
    foc_cfg_runtime["ramp_steps"] = foc_ramp_steps

    align_cfg_runtime = cfg.get("first_hop", {}).get("alignment", {})
    if bool(align_cfg_runtime.get("enabled", False)):
        align_warmup = _resolve_schedule_steps(
            total_steps=max_steps,
            section=align_cfg_runtime,
            steps_key="warmup_steps",
            ratio_key="warmup_ratio",
            default_steps=500,
        )
        align_ramp = _resolve_schedule_steps(
            total_steps=max_steps,
            section=align_cfg_runtime,
            steps_key="ramp_steps",
            ratio_key="ramp_ratio",
            default_steps=2000,
        )
        align_cfg_runtime["warmup_steps"] = align_warmup
        align_cfg_runtime["ramp_steps"] = align_ramp

    lr_cfg = cfg.get("lr_schedule", {})
    lr_sched_enabled = bool(lr_cfg.get("enabled", True))
    lr_warmup_steps = _resolve_schedule_steps(
        total_steps=max_steps,
        section=lr_cfg,
        steps_key="warmup_steps",
        ratio_key="warmup_ratio",
        default_steps=4000,
    )
    lr_cfg["warmup_steps"] = lr_warmup_steps
    lr_min = float(lr_cfg.get("min_lr", 2.0e-6))
    if lr_min <= 0:
        raise ValueError(f"lr_schedule.min_lr must be > 0, got {lr_min}")
    if lr_min > base_lr:
        raise ValueError(f"lr_schedule.min_lr ({lr_min}) must be <= optimizer.lr ({base_lr})")
    print(
        f"[optimizer] AdamW base_lr={base_lr:.3e}, wd_global={global_weight_decay:.3e}, "
        f"wd_backbone={backbone_weight_decay:.3e}, wd_firsthop={first_hop_weight_decay:.3e}, "
        f"betas={tuple(opt_cfg.get('betas', [0.9, 0.95]))}, "
        f"backbone_lr_mult={backbone_lr_mult:.2f}, first_hop_lr_mult={first_hop_lr_mult:.2f}",
        flush=True,
    )
    print(
        f"[optimizer] params: backbone={_numel(backbone_params):,}, first_hop={_numel(first_hop_params):,}",
        flush=True,
    )
    if lr_sched_enabled:
        print(
            f"[lr_schedule] warmup+cosine enabled: warmup_steps={lr_warmup_steps}, min_lr={lr_min:.3e}",
            flush=True,
        )
    else:
        print("[lr_schedule] disabled: constant lr", flush=True)
    print(
        f"[schedule] rollout warmup={rollout_warmup_steps} ramp={rollout_ramp_steps}; "
        f"image warmup={image_warmup_steps} ramp={image_ramp_steps}",
        flush=True,
    )

    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)
    log_interval = int(train_cfg.get("log_interval", 25))
    eval_interval = int(train_cfg.get("eval_interval", 250))
    save_interval = int(train_cfg.get("save_interval", 500))
    if log_interval <= 0:
        raise ValueError(f"training.log_interval must be > 0, got {log_interval}")
    if eval_interval <= 0:
        raise ValueError(f"training.eval_interval must be > 0, got {eval_interval}")
    if save_interval <= 0:
        raise ValueError(f"training.save_interval must be > 0, got {save_interval}")
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    log_grad_norms = bool(train_cfg.get("log_grad_norms", False))
    print_train_line_with_pbar = bool(train_cfg.get("print_train_line_with_pbar", False))
    metrics_jsonl = bool(train_cfg.get("metrics_jsonl", True))

    coverage_target = train_cfg.get("hop0_coverage_target", [0.25, 0.50])
    if not isinstance(coverage_target, (list, tuple)) or len(coverage_target) != 2:
        raise ValueError(
            f"training.hop0_coverage_target must be [min,max], got {coverage_target}"
        )
    coverage_min = float(coverage_target[0])
    coverage_max = float(coverage_target[1])
    if not (0.0 <= coverage_min <= coverage_max <= 1.0):
        raise ValueError(
            f"training.hop0_coverage_target must satisfy 0<=min<=max<=1, got [{coverage_min}, {coverage_max}]"
        )
    coverage_bad_streak = 0

    img_enabled = bool(image_aux_cfg.get("enabled", True))

    best_val = float("inf")
    best_metric_name_for_ckpt = str(train_cfg.get("best_metric", "val_rollout_total"))
    # Optional D1-oriented checkpoint lane (seam-focused selection with optional transport guard).
    best_d1_metric_name = str(train_cfg.get("best_metric_d1", "")).strip()
    best_d1_enabled = best_d1_metric_name != ""
    best_d1_filename = str(train_cfg.get("best_metric_d1_filename", "best_d1.pt")).strip() or "best_d1.pt"
    best_d1_min_step = int(train_cfg.get("best_metric_d1_min_step", 0))
    if best_d1_min_step < 0:
        raise ValueError(f"training.best_metric_d1_min_step must be >= 0, got {best_d1_min_step}")
    best_d1_guard_metric_name = str(train_cfg.get("best_metric_d1_guard_metric", "")).strip()
    best_d1_guard_rel_tol = float(train_cfg.get("best_metric_d1_guard_rel_tol", 0.0))
    if best_d1_guard_rel_tol < 0.0:
        raise ValueError(
            f"training.best_metric_d1_guard_rel_tol must be >= 0, got {best_d1_guard_rel_tol}"
        )
    best_d1_val = float("inf")
    best_d1_guard_best = float("inf")
    if best_d1_enabled:
        print(
            f"[d1_best] enabled: metric={best_d1_metric_name}, file={best_d1_filename}, "
            f"min_step={best_d1_min_step}, guard_metric={best_d1_guard_metric_name or 'none'}, "
            f"guard_rel_tol={best_d1_guard_rel_tol:.4f}",
            flush=True,
        )
    start_step = 0
    img_loss_zero_streak = 0
    dead_branch_streak = 0
    dead_branch_observed_steps = 0
    dead_watch_steps = int(train_cfg.get("dead_branch_watch_steps", max_steps))
    if dead_watch_steps < 0:
        raise ValueError(f"training.dead_branch_watch_steps must be >= 0, got {dead_watch_steps}")

    use_pbar = bool(train_cfg.get("progress_bar", True))
    pbar = tqdm(total=max_steps, desc="train", dynamic_ncols=True) if use_pbar else None
    metrics_fp = None
    metrics_jsonl_mode = str(train_cfg.get("metrics_jsonl_mode", "append")).lower()
    if metrics_jsonl_mode not in ("append", "write"):
        raise ValueError(
            f"training.metrics_jsonl_mode must be 'append' or 'write', got {metrics_jsonl_mode}"
        )
    if resume_enabled and metrics_jsonl_mode == "write":
        print("[startup] resume enabled: force metrics_jsonl_mode=append", flush=True)
        metrics_jsonl_mode = "append"
    if metrics_jsonl:
        metrics_open_mode = "a" if metrics_jsonl_mode == "append" else "w"
        metrics_fp = open(output_dir / "metrics.jsonl", metrics_open_mode, encoding="utf-8")

    loss_balance_watch_enabled = bool(train_cfg.get("loss_balance_watch_enabled", True))
    loss_balance_watch_warn = bool(train_cfg.get("loss_balance_watch_warn", True))
    loss_balance_watch_enforce = bool(train_cfg.get("loss_balance_watch_enforce", False))
    loss_balance_watch_warmup_steps = int(train_cfg.get("loss_balance_watch_warmup_steps", 500))
    loss_balance_watch_patience = int(train_cfg.get("loss_balance_watch_patience", 200))
    loss_balance_dominance_threshold = float(train_cfg.get("loss_balance_dominance_threshold", 0.85))
    if not (0.0 < loss_balance_dominance_threshold < 1.0):
        raise ValueError(
            f"training.loss_balance_dominance_threshold must be in (0,1), got {loss_balance_dominance_threshold}"
        )
    if loss_balance_watch_patience <= 0:
        raise ValueError(f"training.loss_balance_watch_patience must be > 0, got {loss_balance_watch_patience}")
    loss_balance_streak = {"pair": 0, "roll": 0, "img": 0}

    if resume_enabled:
        ckpt = torch.load(resume_path, map_location="cpu")
        if "model" not in ckpt:
            raise KeyError(f"Resume checkpoint missing 'model': {resume_path}")
        # Allow missing keys ONLY for known new-module prefixes.
        # Unexpected keys or unknown missing keys → hard fail.
        _KNOWN_NEW_MODULE_PREFIXES = ("seam_refiner.", "alignment_projector.")
        load_result = model.load_state_dict(ckpt["model"], strict=False)
        if load_result.unexpected_keys:
            raise RuntimeError(
                f"Resume checkpoint has unexpected keys (possible architecture mismatch): "
                f"{load_result.unexpected_keys}"
            )
        if load_result.missing_keys:
            allowed_missing = [
                k for k in load_result.missing_keys
                if any(k.startswith(pfx) for pfx in _KNOWN_NEW_MODULE_PREFIXES)
            ]
            unknown_missing = [k for k in load_result.missing_keys if k not in allowed_missing]
            if unknown_missing:
                raise RuntimeError(
                    f"Resume checkpoint is missing {len(unknown_missing)} keys that are NOT known "
                    f"new modules (architecture drift?): {unknown_missing[:10]}"
                    f"{'...' if len(unknown_missing) > 10 else ''}"
                )
            if allowed_missing:
                print(
                    f"[resume][info] {len(allowed_missing)} new-module keys initialized from scratch: "
                    f"{allowed_missing[:10]}{'...' if len(allowed_missing) > 10 else ''}",
                    flush=True,
                )
                # Architecture changed: skip optimizer/scaler restore to avoid state mismatch
                print(
                    "[resume][warn] architecture has new modules — skipping optimizer/scaler restore "
                    "and resetting best_val/step to avoid stale state.",
                    flush=True,
                )
                _arch_changed = True
            else:
                _arch_changed = False
        else:
            _arch_changed = False
        if not _arch_changed:
            if "optimizer" in ckpt and ckpt["optimizer"] is not None:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scaler" in ckpt and ckpt["scaler"] is not None:
                scaler.load_state_dict(ckpt["scaler"])

        strict_resume_compat = bool(train_cfg.get("strict_resume_compat", True))
        allow_metric_mismatch = bool(train_cfg.get("resume_allow_metric_mismatch", False))
        ckpt_metric_signature = ckpt.get("best_metric_signature", None)
        best_state_compatible = True
        if ckpt_metric_signature is None:
            msg = (
                "Resume checkpoint missing best_metric_signature; cannot guarantee objective continuity. "
                "Set training.resume_allow_metric_mismatch=true to bypass explicitly."
            )
            if not allow_metric_mismatch:
                raise RuntimeError(msg)
            print(f"[resume][warn] {msg}", flush=True)
            best_state_compatible = False
        elif str(ckpt_metric_signature) != str(best_metric_signature):
            msg = (
                "Resume best-metric signature mismatch between checkpoint and current config. "
                "Set training.resume_allow_metric_mismatch=true to bypass explicitly."
            )
            if not allow_metric_mismatch:
                raise RuntimeError(msg)
            print(f"[resume][warn] {msg}", flush=True)
            best_state_compatible = False

        if strict_resume_compat:
            allow_missing_compat = bool(train_cfg.get("resume_allow_missing_compat_metadata", False))
            required_meta = ["target_normalize", "rollout_path", "first_hop_pixel_enabled"]
            missing_meta = [k for k in required_meta if k not in ckpt or ckpt.get(k) is None]
            if missing_meta:
                msg = (
                    "Resume checkpoint missing/invalid strict compatibility metadata: "
                    f"{missing_meta}. Set training.resume_allow_missing_compat_metadata=true to bypass explicitly."
                )
                if not allow_missing_compat:
                    raise RuntimeError(msg)
                print(f"[resume][warn] {msg}", flush=True)
                best_state_compatible = False

            ckpt_target_normalize = ckpt.get("target_normalize", None)
            if ckpt_target_normalize is not None and bool(ckpt_target_normalize) != bool(model.target_normalize):
                raise RuntimeError(
                    "Resume target_normalize mismatch between checkpoint and current model config"
                )
            ckpt_rollout_path = ckpt.get("rollout_path", None)
            if ckpt_rollout_path is not None and list(ckpt_rollout_path) != list(rollout_tps):
                raise RuntimeError(
                    f"Resume rollout_path mismatch: ckpt={list(ckpt_rollout_path)} vs cfg={list(rollout_tps)}"
                )
            ckpt_first_hop_pixel = ckpt.get("first_hop_pixel_enabled", None)
            if ckpt_first_hop_pixel is not None and not bool(ckpt_first_hop_pixel):
                raise RuntimeError("Resume checkpoint indicates first_hop_pixel_enabled=false, incompatible with trainer")

        start_step = int(ckpt.get("step", 0))
        best_val_ckpt = ckpt.get("best_val", None)
        if _arch_changed:
            # Warm-start mode: new modules exist, treat as fresh training from step 0
            start_step = 0
            best_val = float("inf")
            best_metric_name_for_ckpt = str(train_cfg.get("best_metric", "val_rollout_total"))
            if best_d1_enabled:
                best_d1_val = float("inf")
                best_d1_guard_best = float("inf")
            # Do NOT restore rng_state — fresh randomness for new architecture
            print(
                f"[resume][warm-start] architecture changed → reset step=0, best_val=inf, "
                f"fresh optimizer/scaler/rng (model weights warm-started from checkpoint)",
                flush=True,
            )
        else:
            if best_state_compatible and best_val_ckpt is not None:
                best_val = float(best_val_ckpt)
                best_metric_name_for_ckpt = str(ckpt.get("best_metric_name", best_metric_name_for_ckpt))
            else:
                best_val = float("inf")
                best_metric_name_for_ckpt = str(train_cfg.get("best_metric", "val_rollout_total"))
            if best_d1_enabled:
                best_d1_val_ckpt = ckpt.get("best_d1_val", None)
                if best_d1_val_ckpt is not None:
                    best_d1_val = float(best_d1_val_ckpt)
                best_d1_guard_best_ckpt = ckpt.get("best_d1_guard_best", None)
                if best_d1_guard_best_ckpt is not None:
                    best_d1_guard_best = float(best_d1_guard_best_ckpt)

            rng_state = ckpt.get("rng_state", None)
            if isinstance(rng_state, dict):
                try:
                    if "python" in rng_state:
                        random.setstate(rng_state["python"])
                    if "numpy" in rng_state:
                        np.random.set_state(rng_state["numpy"])
                    if "torch" in rng_state:
                        torch.set_rng_state(rng_state["torch"])
                    if "cuda" in rng_state and torch.cuda.is_available():
                        cuda_state = rng_state["cuda"]
                        if isinstance(cuda_state, (list, tuple)) and len(cuda_state) > 0:
                            torch.cuda.set_rng_state_all(cuda_state)
                except Exception as e:
                    print(f"[resume][warn] failed to restore rng_state: {e}", flush=True)
        print(
            f"[resume] loaded step={start_step}, best_val={best_val:.6f}, best_metric_name={best_metric_name_for_ckpt}",
            flush=True,
        )
        if best_d1_enabled:
            print(
                f"[resume] loaded best_d1_metric={best_d1_metric_name}, "
                f"best_d1_val={best_d1_val:.6f}, best_d1_guard_best={best_d1_guard_best:.6f}",
                flush=True,
            )
        if start_step >= max_steps:
            print(
                f"[resume] checkpoint step ({start_step}) >= max_steps ({max_steps}), no further training needed.",
                flush=True,
            )
            if metrics_fp is not None:
                metrics_fp.close()
            if pbar is not None:
                pbar.close()
            return

    if pbar is not None and start_step > 0:
        pbar.update(start_step)

    main_iter = iter(main_train_loader)
    hop0_iter = iter(hop0_train_loader)
    model.train()

    for step in range(start_step + 1, max_steps + 1):
        try:
            main_batch = next(main_iter)
        except StopIteration:
            main_iter = iter(main_train_loader)
            main_batch = next(main_iter)
        try:
            hop0_batch = next(hop0_iter)
        except StopIteration:
            raise RuntimeError(
                "hop0_train_loader exhausted before training finished. "
                "This violates docs/main.md: every step must receive a hop0 auxiliary batch."
            )

        main_batch = move_batch_to_device(main_batch, device)
        hop0_batch = move_batch_to_device(hop0_batch, device)

        if not torch.all(hop0_batch["hop_idx"] == 0):
            raise RuntimeError("hop0 auxiliary batch contains non-hop0 samples")

        hop0_aux_ratio = float(hop0_batch["hop_idx"].numel()) / float(max(main_batch["hop_idx"].numel(), 1))
        hop0_main_ratio = float((main_batch["hop_idx"].long() == 0).float().mean().item())
        hop0_coverage = hop0_aux_ratio
        if hop0_coverage < coverage_min or hop0_coverage > coverage_max:
            coverage_bad_streak += 1
        else:
            coverage_bad_streak = 0
        if coverage_bad_streak >= 100:
            raise RuntimeError(
                f"hop0_coverage out of target range [{coverage_min:.2f}, {coverage_max:.2f}] for 100 steps"
            )

        lambda_img = get_linear_schedule_value(
            global_step=step,
            warmup_steps=int(image_aux_cfg.get("warmup_steps", 200)),
            ramp_steps=int(image_aux_cfg.get("ramp_steps", 600)),
            start=float(image_aux_cfg.get("lambda_start", 0.0)),
            end=float(image_aux_cfg.get("lambda_max", 0.05)),
        )

        if lr_sched_enabled:
            lr_now = get_warmup_cosine_lr(
                step=step,
                max_steps=max_steps,
                base_lr=base_lr,
                warmup_steps=lr_warmup_steps,
                min_lr=lr_min,
            )
        else:
            lr_now = base_lr
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now * float(pg.get("lr_scale", 1.0))

        with autocast(enabled=amp_enabled):
            main_out = model.predict_latent_step(
                z_src=main_batch["z_src"],
                t_src=main_batch["t_src"],
                t_dst=main_batch["t_dst"],
                hop_idx=main_batch["hop_idx"],
                x_src_img=main_batch["x_src"],
            )
            hop0_mask_main = main_batch["hop_idx"].long() == 0
            hop0_in_main = bool(torch.any(hop0_mask_main).item())
            if hop0_in_main:
                pix_delta_abs_hop0_t = (
                    (main_out["z_in"][hop0_mask_main] - main_batch["z_src"][hop0_mask_main]).abs().mean()
                )
                v_hop_abs_hop0_t = main_out["v_hop"][hop0_mask_main].abs().mean()
            else:
                zero = main_out["z_pred"].new_zeros(())
                pix_delta_abs_hop0_t = zero
                v_hop_abs_hop0_t = zero
            pair_losses = compute_pair_losses(model, main_out, main_batch, cfg)

            rollout_losses = compute_rollout_losses(
                model=model,
                main_batch=main_batch,
                cfg=cfg,
                global_step=step,
                rollout_times=rollout_times,
            )

            if img_enabled:
                img_losses = compute_hop0_image_losses(model, hop0_batch, cfg)
                loss_img = img_losses["total"]
            else:
                zero = pair_losses["total"].new_zeros(())
                nan = torch.full((), float("nan"), device=zero.device, dtype=zero.dtype)
                img_losses = {
                    "total": zero,
                    "l1": zero,
                    "ssim": zero,
                    "seam": zero,
                    "ssim_raw_mean": nan,
                    "ssim_raw_min": nan,
                    "ssim_raw_max": nan,
                    "ssim_over1_frac": nan,
                    "ssim_below0_frac": nan,
                    "ssim_clamped_mean": nan,
                }
                loss_img = zero

            # --- FOC-lite: hop0 sub-stepping consistency ---
            foc_losses = compute_foc_losses(
                model=model,
                hop0_batch=hop0_batch,
                cfg=cfg,
                global_step=step,
                rollout_times=rollout_times,
            )

            # --- iREPA alignment loss (Scheme A) ---
            align_cfg = cfg.get("first_hop", {}).get("alignment", {})
            align_enabled = bool(align_cfg.get("enabled", False))
            loss_align = main_out["z_pred"].new_zeros(())
            lambda_align = 0.0
            if align_enabled and main_out.get("align_proj") is not None:
                lambda_align = get_linear_schedule_value(
                    global_step=step,
                    warmup_steps=int(align_cfg.get("warmup_steps", 500)),
                    ramp_steps=int(align_cfg.get("ramp_steps", 2000)),
                    start=float(align_cfg.get("lambda_start", 0.0)),
                    end=float(align_cfg.get("lambda_max", 0.10)),
                )
                # Reference: GT target latent [B, C, H, W]
                z_ref = main_batch["z_dst"].detach()
                align_proj = main_out["align_proj"]  # [B, C_out, H, W]
                # The projector output channels should match the latent channels
                # (both default to 768). If not, truncate the reference to match.
                if z_ref.shape[1] != align_proj.shape[1]:
                    z_ref = z_ref[:, :align_proj.shape[1]]
                loss_align = F.mse_loss(align_proj.float(), z_ref.float())

            total_loss = (
                pair_losses["total"]
                + rollout_losses["lambda_roll"] * rollout_losses["loss_total"]
                + float(lambda_img) * loss_img
                + foc_losses["lambda_foc"] * foc_losses["loss_total"]
                + float(lambda_align) * loss_align
            )

            reg_cfg = cfg["loss"].get("regularizer", {})
            if bool(reg_cfg.get("enabled", False)):
                w_lambda_hop = float(reg_cfg.get("lambda_hop_l2_weight", 0.0))
                w_gate = float(reg_cfg.get("gate_pix_l2_weight", 0.0))
                total_loss = (
                    total_loss
                    + w_lambda_hop * (model.hop_residual_head.lambda_hop_value().pow(2).mean())
                    + w_gate * (model.gate_pix_value().pow(2).mean())
                )

            pair_weighted = pair_losses["total"]
            roll_weighted = rollout_losses["lambda_roll"] * rollout_losses["loss_total"]
            img_weighted = pair_weighted.new_tensor(float(lambda_img)) * loss_img
            foc_weighted = foc_losses["lambda_foc"] * foc_losses["loss_total"]
            weighted_total = pair_weighted + roll_weighted + img_weighted + foc_weighted
            frac_denom = weighted_total.abs().clamp_min(1.0e-12)
            pair_frac_t = (pair_weighted / frac_denom).clamp(min=-10.0, max=10.0)
            roll_frac_t = (roll_weighted / frac_denom).clamp(min=-10.0, max=10.0)
            img_frac_t = (img_weighted / frac_denom).clamp(min=-10.0, max=10.0)

            if loss_balance_watch_enabled and step >= loss_balance_watch_warmup_steps:
                frac_values = {
                    "pair": float(pair_frac_t.detach().item()),
                    "roll": float(roll_frac_t.detach().item()),
                    "img": float(img_frac_t.detach().item()),
                }
                for k, v in frac_values.items():
                    if v >= loss_balance_dominance_threshold:
                        loss_balance_streak[k] += 1
                    else:
                        loss_balance_streak[k] = 0
                    if loss_balance_streak[k] == loss_balance_watch_patience and loss_balance_watch_warn:
                        print(
                            f"[loss_balance][warn] {k} dominates weighted loss "
                            f"(frac={v:.3f} >= {loss_balance_dominance_threshold:.3f}) "
                            f"for {loss_balance_streak[k]} consecutive steps",
                            flush=True,
                        )
                    if loss_balance_watch_enforce and loss_balance_streak[k] >= loss_balance_watch_patience:
                        raise RuntimeError(
                            f"[loss_balance][fail] {k} dominates weighted loss "
                            f"(frac={v:.3f} >= {loss_balance_dominance_threshold:.3f}) "
                            f"for {loss_balance_streak[k]} consecutive steps"
                        )

        if not torch.isfinite(total_loss):
            pair_v = pair_losses["total"].detach()
            roll_v = rollout_losses["loss_total"].detach()
            img_v = img_losses["total"].detach()
            img_l1_v = img_losses["l1"].detach()
            img_ssim_v = img_losses["ssim"].detach()
            img_seam_v = img_losses["seam"].detach()
            ssim_raw_min_v = img_losses["ssim_raw_min"].detach()
            ssim_raw_max_v = img_losses["ssim_raw_max"].detach()
            ssim_raw_mean_v = img_losses["ssim_raw_mean"].detach()
            print(
                "[nonfinite] "
                f"step={step} "
                f"total={float(total_loss.detach().item())} "
                f"pair={float(pair_v.item())} roll={float(roll_v.item())} img={float(img_v.item())} "
                f"img_l1={float(img_l1_v.item())} img_ssim={float(img_ssim_v.item())} img_seam={float(img_seam_v.item())} "
                f"ssim_raw_mean={float(ssim_raw_mean_v.item())} "
                f"ssim_raw_min={float(ssim_raw_min_v.item())} ssim_raw_max={float(ssim_raw_max_v.item())} "
                f"finite_pair={bool(torch.isfinite(pair_v).all().item())} "
                f"finite_roll={bool(torch.isfinite(roll_v).all().item())} "
                f"finite_img={bool(torch.isfinite(img_v).all().item())} "
                f"finite_img_ssim={bool(torch.isfinite(img_ssim_v).all().item())}",
                flush=True,
            )
            raise RuntimeError("Non-finite detected in total loss")

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        grad_total_preclip = 0.0
        grad_backbone = 0.0
        grad_first_hop = 0.0
        clip_grad_report = 0.0
        grad_finite_check = bool(train_cfg.get("grad_finite_check", True))
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            if log_grad_norms:
                grad_total_preclip = _l2_grad_norm(trainable_params)
                grad_backbone = _l2_grad_norm(backbone_params) if backbone_params else 0.0
                grad_first_hop = _l2_grad_norm(first_hop_params) if first_hop_params else 0.0
            clip_ret = torch.nn.utils.clip_grad_norm_(
                trainable_params,
                grad_clip,
                error_if_nonfinite=grad_finite_check,
            )
            if log_grad_norms:
                clip_grad_report = float(clip_ret.item()) if torch.is_tensor(clip_ret) else float(clip_ret)
        elif grad_finite_check:
            for p in trainable_params:
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise RuntimeError("Non-finite gradient detected after unscale")
        scaler.step(optimizer)
        scaler.update()

        gate_val = model.gate_pix_value().detach()
        gate_abs = float(gate_val.abs().item())
        gate_raw = float(model.g_pix_raw.detach().item())
        lambda_eff = model.hop_residual_head.lambda_hop_value().detach()
        lambda_raw = model.hop_residual_head.lambda_hop_raw.detach()
        lambda_h0 = float(lambda_eff[0].item())
        lambda_h0_raw = float(lambda_raw[0].item())
        lambda_max_abs = float(lambda_eff.abs().max().item())
        gate_floor = float(getattr(model, "pixel_gate_floor", 0.0))
        lambda_floor = float(getattr(model.hop_residual_head, "lambda_hop_floor", 0.0))
        gate_eff = max(gate_abs - gate_floor, 0.0)
        lambda_eff_h0 = max(abs(lambda_h0) - lambda_floor, 0.0)
        pix_delta_abs_hop0 = float(pix_delta_abs_hop0_t.detach().item())
        v_hop_abs_hop0 = float(v_hop_abs_hop0_t.detach().item())
        if step <= int(train_cfg.get("gate_explosion_watch_steps", 500)):
            if gate_abs > float(train_cfg.get("gate_abs_max", 2.0)):
                raise RuntimeError(f"gate_pix exploded early: |gate_pix|={gate_abs:.4f}")
            if abs(lambda_h0) > float(train_cfg.get("lambda_hop0_abs_max", 2.0)):
                raise RuntimeError(f"lambda_hop_0 exploded early: {lambda_h0:.4f}")

        if img_enabled:
            img_val = float(img_losses["total"].detach().item())
            if not math.isfinite(img_val):
                raise RuntimeError(f"hop0 image loss is non-finite at step={step}: {img_val}")
            img_zero_eps = float(train_cfg.get("img_loss_zero_eps", 1.0e-8))
            if abs(img_val) <= img_zero_eps:
                img_loss_zero_streak += 1
            else:
                img_loss_zero_streak = 0
            img_zero_patience = int(train_cfg.get("img_loss_zero_patience", 200))
            if img_loss_zero_streak >= img_zero_patience:
                raise RuntimeError(
                    f"hop0 image loss stayed near-zero (|loss| <= {img_zero_eps}) "
                    f"for {img_loss_zero_streak} consecutive steps"
                )

        if step <= dead_watch_steps and hop0_in_main:
            dead_branch_observed_steps += 1
            dead_gate_eps = float(train_cfg.get("dead_branch_gate_eps", 1.0e-8))
            dead_lambda_eps = float(train_cfg.get("dead_branch_lambda_eps", 1.0e-8))
            dead_pix_eps = float(train_cfg.get("dead_branch_pix_eps", 1.0e-8))
            dead_gate_rel_eps = float(train_cfg.get("dead_branch_gate_rel_eps", 0.0))
            dead_lambda_rel_eps = float(train_cfg.get("dead_branch_lambda_rel_eps", 0.0))
            gate_thr = max(dead_gate_eps, dead_gate_rel_eps * max(gate_floor, 1.0e-12))
            lambda_thr = max(dead_lambda_eps, dead_lambda_rel_eps * max(lambda_floor, 1.0e-12))
            if gate_eff <= gate_thr and lambda_eff_h0 <= lambda_thr and pix_delta_abs_hop0 <= dead_pix_eps:
                dead_branch_streak += 1
            else:
                dead_branch_streak = 0
            dead_patience = int(train_cfg.get("dead_branch_patience", 100))
            if dead_branch_streak >= dead_patience:
                raise RuntimeError(
                    "first-hop hop0 branch appears dead: gate_pix/lambda_hop_0/pix_delta_hop0 stayed near-zero "
                    f"for {dead_branch_streak} consecutive observed hop0 steps"
                )

        if step % log_interval == 0:
            step_losses = rollout_losses["step_losses"]
            roll0 = float(step_losses[0].item()) if len(step_losses) > 0 else 0.0
            pix_delta_abs = float(main_out["pix_delta_abs"].item())
            v_hop_abs = float(main_out["v_hop_abs"].item())
            vel_raw = float(pair_losses.get("velocity_unweighted", pair_losses["velocity"]).item())
            end_raw = float(pair_losses.get("endpoint_unweighted", pair_losses["endpoint"]).item())
            vel_w_eff = float(pair_losses.get("velocity_weight_eff", torch.tensor(1.0, device=device)).item())
            end_w_eff = float(pair_losses.get("endpoint_weight_eff", torch.tensor(1.0, device=device)).item())
            vel_reb = float(pair_losses.get("velocity_rebalance_scale", torch.tensor(1.0, device=device)).item())
            pair_w = float(pair_weighted.detach().item())
            roll_w = float(roll_weighted.detach().item())
            img_w = float(img_weighted.detach().item())
            pair_frac = float(pair_frac_t.detach().item())
            roll_frac = float(roll_frac_t.detach().item())
            img_frac = float(img_frac_t.detach().item())
            lambda_roll_base = float(rollout_losses.get("lambda_roll_base", rollout_losses["lambda_roll"]).item())
            lambda_roll_scale = float(rollout_losses.get("lambda_roll_scale", torch.tensor(1.0, device=device)).item())
            rollout_st = float(rollout_losses.get("straight_through", torch.tensor(1.0, device=device)).item())
            img_l1 = float(img_losses["l1"].item())
            img_ssim = float(img_losses["ssim"].item())
            img_seam = float(img_losses["seam"].item())
            ssim_raw_mean = float(img_losses["ssim_raw_mean"].item())
            ssim_raw_min = float(img_losses["ssim_raw_min"].item())
            ssim_raw_max = float(img_losses["ssim_raw_max"].item())
            ssim_over1_frac = float(img_losses["ssim_over1_frac"].item())
            ssim_below0_frac = float(img_losses["ssim_below0_frac"].item())
            ssim_clamped_mean = float(img_losses["ssim_clamped_mean"].item())
            metrics_payload = {
                "event": "train",
                "step": int(step),
                "loss": float(total_loss.item()),
                "pair": float(pair_losses["total"].item()),
                "vel": float(pair_losses["velocity"].item()),
                "vel_raw": vel_raw,
                "end": float(pair_losses["endpoint"].item()),
                "end_raw": end_raw,
                "vel_w": vel_w_eff,
                "end_w": end_w_eff,
                "vel_reb": vel_reb,
                "roll": float(rollout_losses["loss_total"].item()),
                "roll0": roll0,
                "img": float(img_losses["total"].item()),
                "img_l1": img_l1,
                "img_ssim": img_ssim,
                "img_seam": img_seam,
                "ssim_raw_mean": ssim_raw_mean,
                "ssim_raw_min": ssim_raw_min,
                "ssim_raw_max": ssim_raw_max,
                "ssim_over1_frac": ssim_over1_frac,
                "ssim_below0_frac": ssim_below0_frac,
                "ssim_clamped_mean": ssim_clamped_mean,
                "lambda_roll": float(rollout_losses["lambda_roll"].item()),
                "lambda_roll_base": lambda_roll_base,
                "lambda_roll_scale": lambda_roll_scale,
                "rollout_straight_through": rollout_st,
                "lambda_img": float(lambda_img),
                "alpha": float(rollout_losses["alpha_mix"].item()),
                "pair_w": pair_w,
                "roll_w": roll_w,
                "img_w": img_w,
                "pair_frac": pair_frac,
                "roll_frac": roll_frac,
                "img_frac": img_frac,
                "hop0_coverage": hop0_coverage,
                "hop0_main_ratio": hop0_main_ratio,
                "hop0_aux_ratio": hop0_aux_ratio,
                "gate_pix": float(model.gate_pix_value().item()),
                "gate_pix_raw": gate_raw,
                "lambda_hop_0": lambda_h0,
                "lambda_hop_0_raw": lambda_h0_raw,
                "gate_eff": gate_eff,
                "lambda_eff_h0": lambda_eff_h0,
                "pix_delta_abs": pix_delta_abs,
                "pix_delta_abs_hop0": pix_delta_abs_hop0,
                "v_hop_abs": v_hop_abs,
                "v_hop_abs_hop0": v_hop_abs_hop0,
                "hop0_in_main_batch": int(hop0_in_main),
                "dead_branch_streak": int(dead_branch_streak),
                "dead_branch_observed_steps": int(dead_branch_observed_steps),
                "lr_firsthop": float(lr_now * first_hop_lr_mult),
                "lr_backbone": float(lr_now * backbone_lr_mult),
                "foc": float(foc_losses["loss_total"].item()),
                "foc_gap": float(foc_losses["gap_norm"].item()),
                "lambda_foc": float(foc_losses["lambda_foc"].item()),
                "align": float(loss_align.item()),
                "lambda_align": float(lambda_align),
            }
            if metrics_fp is not None:
                metrics_fp.write(json.dumps(metrics_payload, ensure_ascii=True) + "\n")
                metrics_fp.flush()
            grad_suffix = ""
            if log_grad_norms:
                grad_suffix = (
                    f" grad_total={grad_total_preclip:.4e} "
                    f"grad_backbone={grad_backbone:.4e} "
                    f"grad_firsthop={grad_first_hop:.4e} "
                    f"grad_clip_pre={clip_grad_report:.4e}"
                )
            if pbar is not None:
                pbar.set_postfix(
                    loss=f"{float(total_loss.item()):.4e}",
                    pair=f"{float(pair_losses['total'].item()):.3e}",
                    roll=f"{float(rollout_losses['loss_total'].item()):.4e}",
                    img=f"{float(img_losses['total'].item()):.3e}",
                    pf=f"{pair_frac:.2f}",
                    rf=f"{roll_frac:.2f}",
                    ifc=f"{img_frac:.2f}",
                    i_ssim=f"{img_ssim:.3e}",
                    smax=f"{ssim_raw_max:.3f}",
                    lr=f"{(lr_now * first_hop_lr_mult):.2e}",
                    gate=f"{float(model.gate_pix_value().item()):.3f}",
                    geff=f"{gate_eff:.1e}",
                    leff=f"{lambda_eff_h0:.1e}",
                    pix=f"{pix_delta_abs:.1e}",
                    pix0=f"{pix_delta_abs_hop0:.1e}",
                    vh=f"{v_hop_abs:.1e}",
                    vh0=f"{v_hop_abs_hop0:.1e}",
                    reb=f"{vel_reb:.2f}",
                    foc=f"{float(foc_losses['loss_total'].item()):.2e}",
                    fgap=f"{float(foc_losses['gap_norm'].item()):.2e}",
                )
                if print_train_line_with_pbar:
                    print(
                        f"[train] step={step:05d} "
                        f"loss={float(total_loss.item()):.6f} "
                        f"pair={float(pair_losses['total'].item()):.6f} "
                        f"vel={float(pair_losses['velocity'].item()):.6f} "
                        f"vel_raw={vel_raw:.6f} "
                        f"end={float(pair_losses['endpoint'].item()):.6f} "
                        f"end_raw={end_raw:.6f} "
                        f"vel_w={vel_w_eff:.3f} "
                        f"end_w={end_w_eff:.3f} "
                        f"vel_reb={vel_reb:.3f} "
                        f"roll={float(rollout_losses['loss_total'].item()):.6f} "
                        f"roll0={roll0:.6f} "
                        f"img={float(img_losses['total'].item()):.6f} "
                        f"img_l1={img_l1:.6f} "
                        f"img_ssim={img_ssim:.6f} "
                        f"img_seam={img_seam:.6f} "
                        f"ssim_raw_mean={ssim_raw_mean:.6f} "
                        f"ssim_raw_min={ssim_raw_min:.6f} "
                        f"ssim_raw_max={ssim_raw_max:.6f} "
                        f"ssim_over1={ssim_over1_frac:.6f} "
                        f"ssim_below0={ssim_below0_frac:.6f} "
                        f"ssim_clamped_mean={ssim_clamped_mean:.6f} "
                        f"lambda_roll={float(rollout_losses['lambda_roll'].item()):.4f} "
                        f"lambda_roll_base={lambda_roll_base:.4f} "
                        f"lambda_roll_scale={lambda_roll_scale:.4f} "
                        f"st={int(round(rollout_st))} "
                        f"lambda_img={float(lambda_img):.4f} "
                        f"alpha={float(rollout_losses['alpha_mix'].item()):.4f} "
                        f"pair_w={pair_w:.6f} roll_w={roll_w:.6f} img_w={img_w:.6f} "
                        f"pair_frac={pair_frac:.3f} roll_frac={roll_frac:.3f} img_frac={img_frac:.3f} "
                        f"hop0_coverage={hop0_coverage:.3f} "
                        f"hop0_main_ratio={hop0_main_ratio:.3f} "
                        f"gate_pix={float(model.gate_pix_value().item()):.4f} "
                        f"gate_pix_raw={gate_raw:.4f} "
                        f"lambda_hop_0={lambda_h0:.4f} "
                        f"lambda_hop_0_raw={lambda_h0_raw:.4f} "
                        f"gate_eff={gate_eff:.6f} "
                        f"lambda_eff_h0={lambda_eff_h0:.6f} "
                        f"pix_delta_abs={pix_delta_abs:.6f} "
                        f"pix_delta_abs_hop0={pix_delta_abs_hop0:.6f} "
                        f"v_hop_abs={v_hop_abs:.6f}"
                        f" v_hop_abs_hop0={v_hop_abs_hop0:.6f}"
                        f"{grad_suffix}"
                    )
            else:
                print(
                    f"[train] step={step:05d} "
                    f"loss={float(total_loss.item()):.6f} "
                    f"pair={float(pair_losses['total'].item()):.6f} "
                    f"vel={float(pair_losses['velocity'].item()):.6f} "
                    f"vel_raw={vel_raw:.6f} "
                    f"end={float(pair_losses['endpoint'].item()):.6f} "
                    f"end_raw={end_raw:.6f} "
                    f"vel_w={vel_w_eff:.3f} "
                    f"end_w={end_w_eff:.3f} "
                    f"vel_reb={vel_reb:.3f} "
                    f"roll={float(rollout_losses['loss_total'].item()):.6f} "
                    f"roll0={roll0:.6f} "
                    f"img={float(img_losses['total'].item()):.6f} "
                    f"img_l1={img_l1:.6f} "
                    f"img_ssim={img_ssim:.6f} "
                    f"img_seam={img_seam:.6f} "
                    f"ssim_raw_mean={ssim_raw_mean:.6f} "
                    f"ssim_raw_min={ssim_raw_min:.6f} "
                    f"ssim_raw_max={ssim_raw_max:.6f} "
                    f"ssim_over1={ssim_over1_frac:.6f} "
                    f"ssim_below0={ssim_below0_frac:.6f} "
                    f"ssim_clamped_mean={ssim_clamped_mean:.6f} "
                    f"lambda_roll={float(rollout_losses['lambda_roll'].item()):.4f} "
                    f"lambda_roll_base={lambda_roll_base:.4f} "
                    f"lambda_roll_scale={lambda_roll_scale:.4f} "
                    f"st={int(round(rollout_st))} "
                    f"lambda_img={float(lambda_img):.4f} "
                    f"alpha={float(rollout_losses['alpha_mix'].item()):.4f} "
                    f"pair_w={pair_w:.6f} roll_w={roll_w:.6f} img_w={img_w:.6f} "
                    f"pair_frac={pair_frac:.3f} roll_frac={roll_frac:.3f} img_frac={img_frac:.3f} "
                    f"hop0_coverage={hop0_coverage:.3f} "
                    f"hop0_main_ratio={hop0_main_ratio:.3f} "
                    f"gate_pix={float(model.gate_pix_value().item()):.4f} "
                    f"gate_pix_raw={gate_raw:.4f} "
                    f"lambda_hop_0={lambda_h0:.4f} "
                    f"lambda_hop_0_raw={lambda_h0_raw:.4f} "
                    f"gate_eff={gate_eff:.6f} "
                    f"lambda_eff_h0={lambda_eff_h0:.6f} "
                    f"pix_delta_abs={pix_delta_abs:.6f} "
                    f"pix_delta_abs_hop0={pix_delta_abs_hop0:.6f} "
                    f"v_hop_abs={v_hop_abs:.6f}"
                    f" v_hop_abs_hop0={v_hop_abs_hop0:.6f}"
                    f"{grad_suffix}"
                )
        if step % save_interval == 0:
            save_checkpoint(
                model,
                optimizer,
                scaler,
                step,
                output_dir,
                f"step_{step:06d}.pt",
                rollout_tps,
                best_val=best_val,
                best_metric_name=best_metric_name_for_ckpt,
                best_metric_signature=best_metric_signature,
                best_d1_val=best_d1_val if best_d1_enabled else None,
                best_d1_metric_name=best_d1_metric_name if best_d1_enabled else None,
                best_d1_guard_metric_name=best_d1_guard_metric_name if best_d1_enabled else None,
                best_d1_guard_best=best_d1_guard_best if best_d1_enabled else None,
            )

        if step % eval_interval == 0:
            metrics = evaluate(
                model=model,
                main_val_loader=main_val_loader,
                hop0_val_loader=hop0_val_loader,
                cfg=cfg,
                device=device,
                rollout_times=rollout_times,
                global_step=step,
            )
            key, key_name = resolve_best_selection_score(metrics, train_cfg)
            best_metric_name_for_ckpt = key_name
            metrics["val_select_score"] = float(key)
            if best_d1_enabled:
                if step < best_d1_min_step:
                    metrics["val_d1_skipped_before_min_step"] = float(best_d1_min_step)
                else:
                    if best_d1_metric_name not in metrics:
                        raise KeyError(
                            f"training.best_metric_d1 references unknown metric '{best_d1_metric_name}'. "
                            f"Available keys: {sorted(metrics.keys())}"
                        )
                    d1_score = float(metrics[best_d1_metric_name])
                    d1_guard_ok = True
                    d1_guard_limit = float("nan")
                    if best_d1_guard_metric_name:
                        if best_d1_guard_metric_name not in metrics:
                            raise KeyError(
                                "training.best_metric_d1_guard_metric references unknown metric "
                                f"'{best_d1_guard_metric_name}'. Available keys: {sorted(metrics.keys())}"
                            )
                        d1_guard_val = float(metrics[best_d1_guard_metric_name])
                        if math.isfinite(best_d1_guard_best):
                            d1_guard_limit = best_d1_guard_best * (1.0 + best_d1_guard_rel_tol)
                            d1_guard_ok = d1_guard_val <= d1_guard_limit
                        metrics["val_d1_guard_metric"] = d1_guard_val
                        metrics["val_d1_guard_limit"] = d1_guard_limit
                        metrics["val_d1_guard_ok"] = 1.0 if d1_guard_ok else 0.0
                        best_d1_guard_best = min(best_d1_guard_best, d1_guard_val)
                    else:
                        metrics["val_d1_guard_ok"] = 1.0
                    metrics["val_d1_select_score"] = d1_score
                    if d1_guard_ok and d1_score < best_d1_val:
                        best_d1_val = d1_score
                        save_checkpoint(
                            model,
                            optimizer,
                            scaler,
                            step,
                            output_dir,
                            best_d1_filename,
                            rollout_tps,
                            best_val=best_val,
                            best_metric_name=best_metric_name_for_ckpt,
                            best_metric_signature=best_metric_signature,
                            best_d1_val=best_d1_val,
                            best_d1_metric_name=best_d1_metric_name,
                            best_d1_guard_metric_name=best_d1_guard_metric_name if best_d1_guard_metric_name else None,
                            best_d1_guard_best=best_d1_guard_best if math.isfinite(best_d1_guard_best) else None,
                        )
                        print(
                            f"[val] new d1-best {best_d1_metric_name}={best_d1_val:.6f} "
                            f"(file={best_d1_filename}) at step={step}",
                            flush=True,
                        )
            if metrics_fp is not None:
                val_payload = {"event": "val", "step": int(step)}
                for k, v in metrics.items():
                    val_payload[k] = float(v)
                metrics_fp.write(json.dumps(val_payload, ensure_ascii=True) + "\n")
                metrics_fp.flush()
            summary = " ".join([f"{k}={v:.6f}" for k, v in metrics.items()])
            print(f"[val] step={step:05d} {summary}")

            if key < best_val:
                best_val = key
                save_checkpoint(
                    model,
                    optimizer,
                    scaler,
                    step,
                    output_dir,
                    "best.pt",
                    rollout_tps,
                    best_val=best_val,
                    best_metric_name=best_metric_name_for_ckpt,
                    best_metric_signature=best_metric_signature,
                    best_d1_val=best_d1_val if best_d1_enabled else None,
                    best_d1_metric_name=best_d1_metric_name if best_d1_enabled else None,
                    best_d1_guard_metric_name=best_d1_guard_metric_name if best_d1_enabled else None,
                    best_d1_guard_best=best_d1_guard_best if best_d1_enabled else None,
                )
                print(f"[val] new best {key_name}={best_val:.6f} at step={step}")
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()
    if metrics_fp is not None:
        metrics_fp.close()
    save_checkpoint(
        model,
        optimizer,
        scaler,
        max_steps,
        output_dir,
        "last.pt",
        rollout_tps,
        best_val=best_val,
        best_metric_name=best_metric_name_for_ckpt,
        best_metric_signature=best_metric_signature,
        best_d1_val=best_d1_val if best_d1_enabled else None,
        best_d1_metric_name=best_d1_metric_name if best_d1_enabled else None,
        best_d1_guard_metric_name=best_d1_guard_metric_name if best_d1_enabled else None,
        best_d1_guard_best=best_d1_guard_best if best_d1_enabled else None,
    )
    print(f"Training done. Outputs at: {output_dir}")


if __name__ == "__main__":
    main()
