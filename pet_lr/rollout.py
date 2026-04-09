from __future__ import annotations

from typing import List, Sequence

import torch


def get_rollout_alpha(global_step: int, warmup_steps: int, ramp_steps: int, alpha_max: float) -> float:
    if global_step < warmup_steps:
        return 0.0
    progress = min((global_step - warmup_steps) / max(ramp_steps, 1), 1.0)
    return alpha_max * progress


def mix_latent(
    z_gt: torch.Tensor,
    z_pred: torch.Tensor,
    alpha: float,
    straight_through: bool = True,
) -> torch.Tensor:
    if alpha <= 0:
        return z_pred + (z_gt - z_pred).detach() if straight_through else z_gt
    if alpha >= 1:
        return z_pred
    mixed = (1.0 - alpha) * z_gt + alpha * z_pred
    if straight_through:
        return z_pred + (mixed - z_pred).detach()
    return mixed


def rollout_latent_chain(
    model,
    z_rollout: torch.Tensor,
    rollout_times: Sequence[float],
    alpha: float,
    straight_through: bool = True,
) -> List[torch.Tensor]:
    if z_rollout.dim() != 5:
        raise ValueError(f"Expected z_rollout [B, T, C, H, W], got {tuple(z_rollout.shape)}")
    preds: List[torch.Tensor] = [z_rollout[:, 0]]
    z_curr = z_rollout[:, 0]
    device = z_rollout.device
    for hop_idx in range(len(rollout_times) - 1):
        t_src = torch.full((z_rollout.shape[0],), float(rollout_times[hop_idx]), device=device)
        t_dst = torch.full((z_rollout.shape[0],), float(rollout_times[hop_idx + 1]), device=device)
        hop = torch.full((z_rollout.shape[0],), hop_idx, device=device, dtype=torch.long)
        z_pred = model.predict_latent_step(z_curr, t_src, t_dst, hop)
        preds.append(z_pred)
        if hop_idx < len(rollout_times) - 2:
            z_curr = mix_latent(
                z_gt=z_rollout[:, hop_idx + 1],
                z_pred=z_pred,
                alpha=alpha,
                straight_through=straight_through,
            )
        else:
            z_curr = z_pred
    return preds
