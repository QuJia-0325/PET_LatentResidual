from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F


def get_linear_schedule_value(
    global_step: int,
    warmup_steps: int,
    ramp_steps: int,
    start: float,
    end: float,
) -> float:
    if global_step < warmup_steps:
        return float(start)
    progress = min((global_step - warmup_steps) / max(ramp_steps, 1), 1.0)
    return float(start) + (float(end) - float(start)) * float(progress)


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
        # STE: forward value = mixed (uses GT for stability),
        # backward gradient flows only through z_pred (detach blocks GT path).
        return z_pred + (mixed - z_pred).detach()
    return mixed


def _latent_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    lt = loss_type.lower()
    if lt == "mse":
        return F.mse_loss(pred, target)
    if lt == "l1":
        return F.l1_loss(pred, target)
    raise ValueError(f"Unsupported rollout loss_type: {loss_type}")


def rollout_multistep_losses_first_hop(
    model,
    z_rollout: torch.Tensor,
    rollout_times: Sequence[float],
    x_rollout_first: torch.Tensor | None,
    alpha: float,
    straight_through: bool = True,
    loss_type: str = "mse",
    step_weights: Sequence[float] | None = None,
) -> Dict[str, object]:
    if z_rollout.dim() != 5:
        raise ValueError(f"Expected z_rollout [B,T,C,H,W], got {tuple(z_rollout.shape)}")
    if len(rollout_times) != z_rollout.shape[1]:
        raise ValueError(
            f"rollout_times length ({len(rollout_times)}) must equal z_rollout T ({z_rollout.shape[1]})"
        )

    num_steps = len(rollout_times) - 1
    if step_weights is None:
        step_weights = [1.0] * num_steps
    if len(step_weights) != num_steps:
        raise ValueError(f"step_weights length ({len(step_weights)}) must equal rollout steps ({num_steps})")

    preds: List[torch.Tensor] = [z_rollout[:, 0]]
    step_losses: List[torch.Tensor] = []
    device = z_rollout.device
    z_curr = z_rollout[:, 0]

    for hop_idx in range(num_steps):
        t_src = torch.full((z_rollout.shape[0],), float(rollout_times[hop_idx]), device=device)
        t_dst = torch.full((z_rollout.shape[0],), float(rollout_times[hop_idx + 1]), device=device)
        hop = torch.full((z_rollout.shape[0],), hop_idx, device=device, dtype=torch.long)

        x_src_img = x_rollout_first if hop_idx == 0 else None
        out = model.predict_latent_step(
            z_src=z_curr,
            t_src=t_src,
            t_dst=t_dst,
            hop_idx=hop,
            x_src_img=x_src_img,
        )
        z_pred = out["z_pred"]
        z_gt = z_rollout[:, hop_idx + 1]
        step_loss = _latent_loss(z_pred, z_gt, loss_type=loss_type)
        step_losses.append(step_loss)
        preds.append(z_pred)

        if hop_idx < num_steps - 1:
            z_curr = mix_latent(
                z_gt=z_gt,
                z_pred=z_pred,
                alpha=alpha,
                straight_through=straight_through,
            )
        else:
            z_curr = z_pred

    w = torch.tensor(step_weights, dtype=step_losses[0].dtype, device=step_losses[0].device)
    stacked = torch.stack(step_losses, dim=0)
    total = (stacked * w).sum() / w.sum().clamp_min(1e-8)
    return {
        "z_preds": preds,
        "step_losses": step_losses,
        "loss_total": total,
        "z_final": preds[-1],
    }


@torch.no_grad()
def sample_one_step_first_hop(
    model,
    z_start: torch.Tensor,
    t_start: float,
    t_end: float,
    hop_idx: int,
    x_src_img: torch.Tensor | None = None,
) -> torch.Tensor:
    b = z_start.shape[0]
    device = z_start.device
    t_src = torch.full((b,), float(t_start), device=device)
    t_dst = torch.full((b,), float(t_end), device=device)
    hop = torch.full((b,), int(hop_idx), device=device, dtype=torch.long)
    out = model.predict_latent_step(z_src=z_start, t_src=t_src, t_dst=t_dst, hop_idx=hop, x_src_img=x_src_img)
    return out["z_pred"]


@torch.no_grad()
def sample_chain_first_hop(
    model,
    z_d50: torch.Tensor,
    x_d50: torch.Tensor,
    rollout_times: Sequence[float],
) -> List[torch.Tensor]:
    if len(rollout_times) < 2:
        raise ValueError("rollout_times must contain at least 2 timestamps")
    preds = [z_d50]
    z_curr = z_d50
    for hop_idx in range(len(rollout_times) - 1):
        x_src = x_d50 if hop_idx == 0 else None
        z_next = sample_one_step_first_hop(
            model=model,
            z_start=z_curr,
            t_start=rollout_times[hop_idx],
            t_end=rollout_times[hop_idx + 1],
            hop_idx=hop_idx,
            x_src_img=x_src,
        )
        preds.append(z_next)
        z_curr = z_next
    return preds

