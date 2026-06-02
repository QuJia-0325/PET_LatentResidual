#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from pet_lr.data_first_hop import PETFirstHopAligned4HopDataset
from pet_lr.losses import extended_seam_loss, seam_consistency_loss
from pet_lr.model_first_hop import PETFlowDiTFirstHop
from pet_lr.rollout_first_hop import sample_chain_first_hop
from pet_lr.path_guard import ensure_repo_local_outputs_absent, resolve_data_disk_dir

# Canonical PSNR implementation (window to 3 first, then PSNR) from RAE repo.
RAE_CODE_ROOT = "/home/qujiaxiang/project/RAE/code/RAE"
if RAE_CODE_ROOT not in sys.path:
    sys.path.insert(0, RAE_CODE_ROOT)
from src.utils.metrics import calc_psnr_clip3  # noqa: E402


TIMEPOINTS = ["D50", "D20", "D10", "D4", "NORMAL"]
METRICS = ["psnr", "ssim", "ms_ssim", "nrmse", "rmse", "mae", "seam", "ext_seam"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate 224 first-hop latent transport with canonical calc_psnr_clip3 and "
            "write reviewable JSON/CSV outputs."
        )
    )
    p.add_argument(
        "--config",
        default="/home/qujiaxiang/project/PET_LatentResidual/configs/pet_flow/pet_flow_first_hop_224_alignchecked_detfix.yaml",
        help="Training config used to build dataset/model",
    )
    p.add_argument(
        "--checkpoint",
        default="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_smoke_alignchecked_detfix/best.pt",
        help="Checkpoint path from train_first_hop.py",
    )
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument(
        "--max-slices",
        type=int,
        default=512,
        help="Number of slices to evaluate (<=0 means full split)",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--out-dir",
        default="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_eval_clip3",
        help="Directory for JSON/CSV artifacts",
    )
    p.add_argument(
        "--decode-mode",
        choices=["default", "raw", "both"],
        default="default",
        help="Decode mode: 'default'=use model default (refined if refiner enabled), "
             "'raw'=force skip refiner, 'both'=output both raw and refined PSNR",
    )
    return p.parse_args()


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_indices(num_slices: int, max_slices: int) -> torch.Tensor:
    if max_slices <= 0 or max_slices >= num_slices:
        return torch.arange(num_slices, dtype=torch.long)
    idx = torch.linspace(0, num_slices - 1, steps=max_slices).round().long()
    idx = torch.unique(idx, sorted=True)
    return idx


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0.0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": float(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }



def to_clip3_suv(x: torch.Tensor) -> torch.Tensor:
    """Convert normalized PET tensor [-1, 1] to clipped SUV [0, 3]."""
    return ((x.float() + 1.0) * 5.0).clamp(0.0, 3.0)


def gaussian_window(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum().clamp_min(1e-12)
    return (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)


def ssim_components(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 5,
    data_range: float = 3.0,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape != y.shape or x.dim() != 4 or x.shape[1] != 1:
        raise ValueError(f"Expected [B,1,H,W] tensors with same shape, got {tuple(x.shape)} and {tuple(y.shape)}")
    pad = window_size // 2
    window = gaussian_window(window_size, 1.5, x.device, x.dtype)
    mu_x = F.conv2d(x, window, padding=pad)
    mu_y = F.conv2d(y, window, padding=pad)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x = (F.conv2d(x * x, window, padding=pad) - mu_x2).clamp_min(0.0)
    sigma_y = (F.conv2d(y * y, window, padding=pad) - mu_y2).clamp_min(0.0)
    sigma_xy = F.conv2d(x * y, window, padding=pad) - mu_xy
    c1 = (0.01 * float(data_range)) ** 2
    c2 = (0.03 * float(data_range)) ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
    ).clamp_min(eps)
    cs_map = (2 * sigma_xy + c2) / (sigma_x + sigma_y + c2).clamp_min(eps)
    ssim = torch.nan_to_num(ssim_map, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0).flatten(1).mean(dim=1)
    cs = torch.nan_to_num(cs_map, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0).flatten(1).mean(dim=1)
    return ssim, cs


def ms_ssim_value(x: torch.Tensor, y: torch.Tensor, data_range: float = 3.0) -> torch.Tensor:
    weights = x.new_tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    values = []
    cur_x = x
    cur_y = y
    for level in range(int(weights.numel())):
        ssim, cs = ssim_components(cur_x, cur_y, data_range=data_range)
        values.append(ssim if level == int(weights.numel()) - 1 else cs)
        if level != int(weights.numel()) - 1:
            cur_x = F.avg_pool2d(cur_x, kernel_size=2, stride=2)
            cur_y = F.avg_pool2d(cur_y, kernel_size=2, stride=2)
    stacked = torch.stack(values, dim=1).clamp_min(1.0e-6)
    return torch.prod(stacked ** weights.view(1, -1), dim=1)


def batch_clip3_metrics(x_pred: torch.Tensor, x_gt: torch.Tensor) -> Dict[str, torch.Tensor]:
    x_pred_suv = to_clip3_suv(x_pred)
    x_gt_suv = to_clip3_suv(x_gt)
    diff = x_pred_suv - x_gt_suv
    mse = diff.square().flatten(1).mean(dim=1)
    rmse = torch.sqrt(mse.clamp_min(0.0))
    mae = diff.abs().flatten(1).mean(dim=1)
    ssim, _ = ssim_components(x_pred_suv, x_gt_suv, data_range=3.0)
    return {
        "ssim": ssim,
        "ms_ssim": ms_ssim_value(x_pred_suv, x_gt_suv, data_range=3.0),
        "nrmse": rmse / 3.0,
        "rmse": rmse,
        "mae": mae,
    }


def load_model(cfg: Dict, ckpt_path: str, device: torch.device) -> PETFlowDiTFirstHop:
    model = PETFlowDiTFirstHop(cfg, device=device).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    # Verify pixel_forcing semantic consistency between checkpoint and config
    if isinstance(ckpt, dict):
        ckpt_pixel = ckpt.get("first_hop_pixel_enabled", None)
        current_pixel = not model.pixel_forcing_disabled
        if ckpt_pixel is not None and bool(ckpt_pixel) != current_pixel:
            raise RuntimeError(
                f"Eval pixel_forcing semantic mismatch: checkpoint has "
                f"first_hop_pixel_enabled={ckpt_pixel}, but config has "
                f"pixel_forcing_disabled={model.pixel_forcing_disabled}"
            )
    model.eval()
    return model


def build_dataset(cfg: Dict, split: str) -> PETFirstHopAligned4HopDataset:
    data_cfg = cfg["data"]
    latent_path = os.path.join(data_cfg["latent_dir"], f"latents_{split}.pt")
    alignment_audit_json = str(data_cfg.get("alignment_audit_json", "")).strip() or None
    return PETFirstHopAligned4HopDataset(
        latent_path=latent_path,
        raw_data_dir=data_cfg["raw_data_dir"],
        split=split,
        clamp_max=float(data_cfg.get("clamp_max", 10.0)),
        t_map=data_cfg["t_map"],
        rollout_timepoints=data_cfg.get("rollout_timepoints", TIMEPOINTS),
        verify_alignment=bool(data_cfg.get("verify_alignment", True)),
        alignment_check_num_samples=int(data_cfg.get("alignment_check_num_samples", 16)),
        alignment_audit_json=alignment_audit_json,
        include_x_rollout_first=True,
        include_full_x_rollout=True,
        image_size=int(data_cfg.get("image_size", 224)),
    )


@torch.no_grad()
def evaluate(
    model: PETFlowDiTFirstHop,
    dataset: PETFirstHopAligned4HopDataset,
    rollout_times: List[float],
    batch_size: int,
    eval_indices: torch.Tensor,
    image_size: int,
    device: torch.device,
    decode_mode: str = "default",
    seam_patch_size: int = 14,
    seam_zone_width: int = 3,
) -> tuple[
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, Dict[str, float]]],
    List[Dict[str, object]],
]:
    # Determine decode modes to evaluate
    modes = []
    if decode_mode in ("default", "both"):
        modes.append(("", None))  # suffix="", apply_refiner=None (model default)
    if decode_mode in ("raw", "both"):
        modes.append(("_raw", False))  # suffix="_raw", apply_refiner=False

    per_tp: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for tp in dataset.rollout_timepoints:
        per_tp[tp] = {}
        for suffix, _ in modes:
            per_tp[tp][suffix] = {metric: [] for metric in METRICS}
    row_map: Dict[int, Dict[str, object]] = {}

    total = int(eval_indices.numel())
    for start in tqdm(range(0, total, batch_size), desc="eval"):
        idx = eval_indices[start : start + batch_size]
        z_d50 = dataset.latents["D50"][idx].to(device)
        x_d50 = dataset.images["D50"][idx].to(device)
        z_chain = sample_chain_first_hop(model, z_d50, x_d50, rollout_times=rollout_times)

        for tp_i, tp in enumerate(dataset.rollout_timepoints):
            x_gt = dataset.images[tp][idx].float().cpu()
            for suffix, refiner_flag in modes:
                # skip_first_tp: don't apply refiner on D50 (tp_i==0) passthrough
                actual_refiner_flag = refiner_flag
                if tp_i == 0 and getattr(model, "seam_refiner_skip_first_tp", False) and refiner_flag is None:
                    actual_refiner_flag = False
                x_pred = model.decode_crop(
                    z_chain[tp_i], crop_size=image_size, apply_refiner=actual_refiner_flag,
                ).detach().cpu()
                clip_metrics = batch_clip3_metrics(x_pred, x_gt)
                x_pred_suv = to_clip3_suv(x_pred)
                x_gt_suv = to_clip3_suv(x_gt)
                for b in range(x_pred.shape[0]):
                    psnr_v = float(calc_psnr_clip3(x_pred[b : b + 1], x_gt[b : b + 1]))
                    seam_v = float(seam_consistency_loss(
                        x_pred_suv[b : b + 1], patch_size=int(seam_patch_size),
                    ).item())
                    ext_seam_v = float(extended_seam_loss(
                        x_pred_suv[b : b + 1],
                        x_gt_suv[b : b + 1],
                        patch_size=int(seam_patch_size),
                        zone_width=int(seam_zone_width),
                    ).item())
                    values = {
                        "psnr": psnr_v,
                        "ssim": float(clip_metrics["ssim"][b].item()),
                        "ms_ssim": float(clip_metrics["ms_ssim"][b].item()),
                        "nrmse": float(clip_metrics["nrmse"][b].item()),
                        "rmse": float(clip_metrics["rmse"][b].item()),
                        "mae": float(clip_metrics["mae"][b].item()),
                        "seam": seam_v,
                        "ext_seam": ext_seam_v,
                    }
                    for metric, value in values.items():
                        per_tp[tp][suffix][metric].append(value)
                    slice_idx = int(idx[b].item())
                    row = row_map.setdefault(slice_idx, {"slice_idx": slice_idx})
                    for metric, value in values.items():
                        row[f"{metric}{suffix}_{tp}"] = value

    summary = {}
    seam_summary = {}
    ext_seam_summary = {}
    metrics_summary = {}
    for tp in dataset.rollout_timepoints:
        for suffix, _ in modes:
            key = f"{tp}{suffix}" if suffix else tp
            metrics_summary[key] = {metric: summarize(per_tp[tp][suffix][metric]) for metric in METRICS}
            summary[key] = metrics_summary[key]["psnr"]
            seam_summary[key] = metrics_summary[key]["seam"]
            ext_seam_summary[key] = metrics_summary[key]["ext_seam"]
    rows = [row_map[k] for k in sorted(row_map.keys())]
    return summary, seam_summary, ext_seam_summary, metrics_summary, rows


def write_csv(path: Path, split: str, rows: List[Dict[str, object]], rollout_tps: List[str]) -> None:
    # Dynamically detect columns from all rows to handle all decode modes robustly
    metric_keys = set()
    for row in rows:
        for k in row.keys():
            if any(k.startswith(prefix) for prefix in METRICS):
                metric_keys.add(k)
    metric_keys_sorted = sorted(metric_keys)
    if not metric_keys_sorted:
        # Fallback for default mode
        metric_keys_sorted = [f"psnr_{tp}" for tp in rollout_tps]
    fieldnames = ["split", "slice_idx"] + metric_keys_sorted
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out_row = {"split": split, "slice_idx": row["slice_idx"]}
            for k in metric_keys_sorted:
                out_row[k] = row.get(k, "")
            writer.writerow(out_row)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    device = torch.device(args.device)
    ensure_repo_local_outputs_absent(Path(__file__).resolve().parent)
    out_dir = resolve_data_disk_dir(args.out_dir, arg_name="--out-dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    rollout_tps = cfg["data"].get("rollout_timepoints", TIMEPOINTS)
    rollout_times = [float(cfg["data"]["t_map"][tp]) for tp in rollout_tps]
    image_size = int(cfg["data"].get("image_size", 224))
    img_loss_cfg = cfg.get("loss", {}).get("image_aux", {})
    seam_patch_size = int(img_loss_cfg.get("seam_patch_size", 14))
    seam_zone_width = int(img_loss_cfg.get("seam_zone_width", 3))

    dataset = build_dataset(cfg, split=args.split)
    eval_indices = select_indices(dataset.num_slices, args.max_slices)
    model = load_model(cfg, ckpt_path=args.checkpoint, device=device)

    summary, seam_summary, ext_seam_summary, metrics_summary, rows = evaluate(
        model=model,
        dataset=dataset,
        rollout_times=rollout_times,
        batch_size=int(args.batch_size),
        eval_indices=eval_indices,
        image_size=image_size,
        device=device,
        decode_mode=args.decode_mode,
        seam_patch_size=seam_patch_size,
        seam_zone_width=seam_zone_width,
    )

    payload = {
        "meta": {
            "psnr_metric": "src.utils.metrics.calc_psnr_clip3",
            "config": args.config,
            "checkpoint": args.checkpoint,
            "split": args.split,
            "max_slices": int(args.max_slices),
            "num_eval_slices": int(eval_indices.numel()),
            "batch_size": int(args.batch_size),
            "device": str(device),
            "rollout_timepoints": rollout_tps,
            "decode_mode": args.decode_mode,
            "seam_patch_size": seam_patch_size,
            "seam_zone_width": seam_zone_width,
            "metric_domain": "clip3_suv_[0,3] for SSIM/MS-SSIM/NRMSE/RMSE/MAE/seam/ext-seam; PSNR via calc_psnr_clip3",
            "nrmse_definition": "RMSE / 3.0 on clip3 SUV domain",
            "ssim_definition": "single-scale Gaussian SSIM on clip3 SUV domain, data_range=3.0",
            "ms_ssim_definition": "5-scale MS-SSIM on clip3 SUV domain, data_range=3.0",
        },
        "summary_psnr_clip3": summary,
        "summary_seam_consistency": seam_summary,
        "summary_extended_seam": ext_seam_summary,
        "summary_metrics_clip3": metrics_summary,
        "per_slice": rows,
    }

    json_path = out_dir / f"first_hop_224_{args.split}_clip3_eval.json"
    csv_path = out_dir / f"first_hop_224_{args.split}_clip3_eval.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_csv(csv_path, split=args.split, rows=rows, rollout_tps=rollout_tps)

    print(json.dumps(payload["meta"], indent=2))
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
