#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import yaml

from pet_lr.data import PETLatentResidual4HopDataset
from pet_lr.losses import (
    make_border_weight_map,
    residual_l2_penalty,
    seam_consistency_loss,
    ssim_loss,
    weighted_l1_loss,
)
from pet_lr.model import LatentResidualRefinementModel
from pet_lr.rollout import get_rollout_alpha, rollout_latent_chain


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(cfg: Dict) -> tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    latent_dir = data_cfg["latent_dir"]
    common = dict(
        raw_data_dir=data_cfg["raw_data_dir"],
        clamp_max=float(data_cfg.get("clamp_max", 10.0)),
        t_map=data_cfg["t_map"],
        rollout_timepoints=data_cfg.get("rollout_timepoints", ["D50", "D20", "D10", "D4", "NORMAL"]),
    )
    train_set = PETLatentResidual4HopDataset(
        latent_path=os.path.join(latent_dir, "latents_train.pt"),
        split="train",
        **common,
    )
    val_set = PETLatentResidual4HopDataset(
        latent_path=os.path.join(latent_dir, "latents_val.pt"),
        split="val",
        **common,
    )
    batch_size = int(data_cfg.get("batch_size", 4))
    num_workers = int(data_cfg.get("num_workers", 0))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def compute_losses(
    model: LatentResidualRefinementModel,
    batch: Dict[str, torch.Tensor],
    cfg: Dict,
    global_step: int,
    rollout_times: list[float],
) -> Dict[str, torch.Tensor]:
    out = model.forward_step(
        z_src=batch["z_src"],
        t_src=batch["t_src"],
        t_dst=batch["t_dst"],
        hop_idx=batch["hop_idx"],
    )
    loss_cfg = cfg["loss"]
    border_map = make_border_weight_map(
        height=batch["x_dst"].shape[-2],
        width=batch["x_dst"].shape[-1],
        border_width=int(loss_cfg.get("border_width", 14)),
        border_weight=float(loss_cfg.get("border_weight", 2.0)),
        device=batch["x_dst"].device,
        dtype=batch["x_dst"].dtype,
    )

    loss_l1 = weighted_l1_loss(out["x_refined"], batch["x_dst"], border_map)
    loss_ssim = ssim_loss(out["x_refined"], batch["x_dst"])
    loss_seam = seam_consistency_loss(out["x_refined"], patch_size=14)
    loss_residual = residual_l2_penalty(out["residual"])

    total = (
        float(loss_cfg.get("l1_weight", 1.0)) * loss_l1
        + float(loss_cfg.get("ssim_weight", 0.25)) * loss_ssim
        + float(loss_cfg.get("seam_weight", 0.10)) * loss_seam
        + float(loss_cfg.get("residual_l2_weight", 0.01)) * loss_residual
    )

    rollout_cfg = cfg["training"]["rollout"]
    loss_roll = batch["x_dst"].new_zeros(())
    if rollout_cfg.get("enabled", True):
        alpha = get_rollout_alpha(
            global_step=global_step,
            warmup_steps=int(rollout_cfg.get("warmup_steps", 8000)),
            ramp_steps=int(rollout_cfg.get("ramp_steps", 12000)),
            alpha_max=float(rollout_cfg.get("alpha_max", 0.35)),
        )
        z_chain = rollout_latent_chain(
            model=model,
            z_rollout=batch["z_rollout"],
            rollout_times=rollout_times,
            alpha=alpha,
            straight_through=bool(rollout_cfg.get("straight_through", True)),
        )
        final_hop = batch["hop_idx"].new_full((batch["hop_idx"].shape[0],), len(rollout_times) - 2)
        t_src_final = batch["t_src"].new_full((batch["t_src"].shape[0],), rollout_times[-2])
        t_dst_final = batch["t_dst"].new_full((batch["t_dst"].shape[0],), rollout_times[-1])
        final_out = model.forward_step(
            z_src=z_chain[-2],
            t_src=t_src_final,
            t_dst=t_dst_final,
            hop_idx=final_hop,
        )
        loss_roll = weighted_l1_loss(final_out["x_refined"], batch["x_rollout"][:, -1], border_map)
        total = total + float(loss_cfg.get("rollout_endpoint_weight", 0.5)) * loss_roll
    else:
        alpha = 0.0

    draft_l1 = weighted_l1_loss(out["x_dst_draft"], batch["x_dst"], border_map)
    return {
        "loss": total,
        "loss_l1": loss_l1.detach(),
        "loss_ssim": loss_ssim.detach(),
        "loss_seam": loss_seam.detach(),
        "loss_residual": loss_residual.detach(),
        "loss_rollout": loss_roll.detach(),
        "draft_l1": draft_l1.detach(),
        "alpha": torch.tensor(alpha, device=batch["x_dst"].device),
    }


@torch.no_grad()
def evaluate(
    model: LatentResidualRefinementModel,
    val_loader: DataLoader,
    cfg: Dict,
    device: torch.device,
    max_batches: int,
    rollout_times: list[float],
) -> Dict[str, float]:
    model.eval()
    sums = {
        "loss": 0.0,
        "loss_l1": 0.0,
        "loss_ssim": 0.0,
        "loss_seam": 0.0,
        "loss_residual": 0.0,
        "loss_rollout": 0.0,
        "draft_l1": 0.0,
    }
    count = 0
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        losses = compute_losses(model, batch, cfg, global_step=10**9, rollout_times=rollout_times)
        for key in sums:
            sums[key] += float(losses[key].item())
        count += 1
    model.train()
    return {k: (v / max(count, 1)) for k, v in sums.items()}


def save_checkpoint(model, optimizer, scaler, step: int, output_dir: str, name: str) -> None:
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }
    torch.save(ckpt, os.path.join(output_dir, name))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PET latent residual v2.1 experiment")
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cudnn.benchmark = True

    output_root = Path(cfg.get("output_dir", "./outputs"))
    run_name = cfg.get("run_name", datetime.now().strftime("%m%d_%H%M%S"))
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    train_loader, val_loader = build_dataloaders(cfg)
    rollout_times = [float(cfg["data"]["t_map"][tp]) for tp in cfg["data"]["rollout_timepoints"]]

    model = LatentResidualRefinementModel(cfg, device=device).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found. Check freeze settings.")

    opt_cfg = cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(opt_cfg.get("lr", 2e-4)),
        weight_decay=float(opt_cfg.get("weight_decay", 0.01)),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.95])),
    )

    amp_enabled = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)
    max_steps = int(cfg["training"].get("max_steps", 2000))
    log_interval = int(cfg["training"].get("log_interval", 25))
    eval_interval = int(cfg["training"].get("eval_interval", 250))
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))
    max_val_batches = int(cfg["training"].get("max_val_batches", 32))

    best_val = float("inf")
    train_iter = iter(train_loader)
    model.train()

    for step in range(1, max_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = move_batch_to_device(batch, device)

        with autocast(enabled=amp_enabled):
            losses = compute_losses(model, batch, cfg, global_step=step, rollout_times=rollout_times)
            loss = losses["loss"]

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % log_interval == 0:
            print(
                f"[train] step={step} "
                f"loss={losses['loss'].item():.4f} "
                f"draft_l1={losses['draft_l1'].item():.4f} "
                f"refine_l1={losses['loss_l1'].item():.4f} "
                f"roll={losses['loss_rollout'].item():.4f} "
                f"alpha={losses['alpha'].item():.3f}"
            )

        if step % eval_interval == 0 or step == max_steps:
            metrics = evaluate(model, val_loader, cfg, device, max_val_batches=max_val_batches, rollout_times=rollout_times)
            print(
                f"[val] step={step} "
                f"loss={metrics['loss']:.4f} "
                f"draft_l1={metrics['draft_l1']:.4f} "
                f"refine_l1={metrics['loss_l1']:.4f} "
                f"roll={metrics['loss_rollout']:.4f}"
            )
            save_checkpoint(model, optimizer, scaler, step, str(output_dir), f"ckpt_step_{step}.pt")
            if metrics["loss"] < best_val:
                best_val = metrics["loss"]
                save_checkpoint(model, optimizer, scaler, step, str(output_dir), "best.pt")

    print(f"Training complete. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
