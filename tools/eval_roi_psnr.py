#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pet_lr.data_first_hop import PETFirstHopAligned4HopDataset
from pet_lr.model_first_hop import PETFlowDiTFirstHop
from pet_lr.rollout_first_hop import sample_chain_first_hop

RAE_CODE_ROOT = "/home/qujiaxiang/project/RAE/code/RAE"
if RAE_CODE_ROOT not in sys.path:
    sys.path.insert(0, RAE_CODE_ROOT)
from src.utils.metrics import calc_psnr_clip3  # noqa: E402

TIMEPOINTS = ["D50", "D20", "D10", "D4", "NORMAL"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chain rollout ROI/high-SUV PSNR diagnostics.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--split", choices=["val"], default="val")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-slices", type=int, default=0, help="<=0 means full split")
    p.add_argument("--decode-mode", choices=["default", "raw", "both"], default="both")
    return p.parse_args()


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def load_model(cfg: Dict, ckpt_path: str, device: torch.device) -> tuple[PETFlowDiTFirstHop, Dict]:
    model = PETFlowDiTFirstHop(cfg, device=device).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    if isinstance(ckpt, dict):
        ckpt_pixel = ckpt.get("first_hop_pixel_enabled", None)
        current_pixel = not model.pixel_forcing_disabled
        if ckpt_pixel is not None and bool(ckpt_pixel) != current_pixel:
            raise RuntimeError(
                "Eval pixel_forcing semantic mismatch: "
                f"checkpoint first_hop_pixel_enabled={ckpt_pixel}, "
                f"config pixel_forcing_disabled={model.pixel_forcing_disabled}"
            )
    model.eval()
    return model, ckpt if isinstance(ckpt, dict) else {}


def select_indices(num_slices: int, max_slices: int) -> torch.Tensor:
    if max_slices <= 0 or max_slices >= num_slices:
        return torch.arange(num_slices, dtype=torch.long)
    idx = torch.linspace(0, num_slices - 1, steps=max_slices).round().long()
    return torch.unique(idx, sorted=True)


def _decode_modes(mode: str) -> list[tuple[str, bool | None]]:
    modes: list[tuple[str, bool | None]] = []
    if mode in ("default", "both"):
        modes.append(("", None))
    if mode in ("raw", "both"):
        modes.append(("_raw", False))
    return modes


def _new_acc() -> Dict[str, float]:
    return {"sum": 0.0, "sum_sq": 0.0, "n": 0.0, "min": float("inf"), "max": float("-inf")}


def _add(acc: Dict[str, float], values: torch.Tensor) -> None:
    values = values.detach().float().flatten()
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return
    acc["sum"] += float(finite.sum().item())
    acc["sum_sq"] += float((finite * finite).sum().item())
    acc["n"] += float(finite.numel())
    acc["min"] = min(acc["min"], float(finite.min().item()))
    acc["max"] = max(acc["max"], float(finite.max().item()))


def _summary(acc: Dict[str, float]) -> Dict[str, float]:
    n = int(acc["n"])
    if n <= 0:
        return {"n": 0.0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    mean = acc["sum"] / float(n)
    var = max(acc["sum_sq"] / float(n) - mean * mean, 0.0)
    return {"n": float(n), "mean": float(mean), "std": float(math.sqrt(var)), "min": float(acc["min"]), "max": float(acc["max"])}


def _to_suv(x_norm: torch.Tensor) -> torch.Tensor:
    return (x_norm.float() + 1.0) * 5.0


def _psnr_from_mse_peak(mse: torch.Tensor, peak: torch.Tensor) -> torch.Tensor:
    mse = mse.clamp_min(1e-12)
    peak = peak.clamp_min(1e-6)
    return 20.0 * torch.log10(peak) - 10.0 * torch.log10(mse)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    num = (x * mask).flatten(1).sum(dim=1)
    den = mask.flatten(1).sum(dim=1).clamp_min(1.0)
    return num / den


def _masked_psnr(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    err = (pred - gt).pow(2)
    mse = _masked_mean(err, mask)
    masked_gt = gt.masked_fill(mask <= 0, float("-inf"))
    peak = masked_gt.flatten(1).max(dim=1).values
    peak = torch.where(torch.isfinite(peak), peak, gt.flatten(1).max(dim=1).values)
    return _psnr_from_mse_peak(mse, peak)


def _gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    dx = F.pad(x[..., 1:] - x[..., :-1], (0, 1, 0, 0))
    dy = F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx.pow(2) + dy.pow(2) + 1e-12)


def compute_roi_metrics(x_pred_norm: torch.Tensor, x_gt_norm: torch.Tensor) -> Dict[str, torch.Tensor]:
    pred = _to_suv(x_pred_norm)
    gt = _to_suv(x_gt_norm)
    out: Dict[str, torch.Tensor] = {}
    err = (pred - gt).pow(2)
    mse = err.flatten(1).mean(dim=1)
    peak = gt.flatten(1).max(dim=1).values
    out["psnr_unclipped"] = _psnr_from_mse_peak(mse, peak)
    out["mse_unclipped"] = mse

    for top_pct, name in [(0.10, "top10suv"), (0.05, "top5suv"), (0.01, "top1suv")]:
        thresh = torch.quantile(gt.flatten(1), 1.0 - top_pct, dim=1, keepdim=True).view(-1, 1, 1, 1)
        mask = (gt >= thresh).float()
        out[f"psnr_{name}"] = _masked_psnr(pred, gt, mask)
        out[f"suvmean_err_{name}"] = _masked_mean(pred, mask) - _masked_mean(gt, mask)

    grad = _gradient_magnitude(gt)
    g_thresh = torch.quantile(grad.flatten(1), 0.90, dim=1, keepdim=True).view(-1, 1, 1, 1)
    g_mask = (grad >= g_thresh).float()
    out["psnr_highgrad"] = _masked_psnr(pred, gt, g_mask)

    out["suvmax_err"] = pred.flatten(1).max(dim=1).values - gt.flatten(1).max(dim=1).values
    gt_pos_mask = (gt > 0.5).float()
    out["suvmean_err_gt_gt0p5"] = _masked_mean(pred, gt_pos_mask) - _masked_mean(gt, gt_pos_mask)
    return out


@torch.no_grad()
def evaluate(
    model: PETFlowDiTFirstHop,
    dataset: PETFirstHopAligned4HopDataset,
    rollout_times: List[float],
    eval_indices: torch.Tensor,
    batch_size: int,
    image_size: int,
    device: torch.device,
    decode_mode: str,
) -> tuple[Dict[str, Dict[str, Dict[str, float]]], list[dict]]:
    modes = _decode_modes(decode_mode)
    metric_acc: Dict[str, Dict[str, Dict[str, float]]] = {}
    rows: list[dict] = []
    for tp in dataset.rollout_timepoints:
        metric_acc[tp] = {}
        for suffix, _ in modes:
            key = suffix or "default"
            metric_acc[tp][key] = {}

    total = int(eval_indices.numel())
    for start in tqdm(range(0, total, batch_size), desc=f"roi:{total}", disable=not sys.stderr.isatty()):
        idx = eval_indices[start : start + batch_size]
        z_d50 = dataset.latents["D50"][idx].to(device, non_blocking=True)
        x_d50 = dataset.images["D50"][idx].to(device, non_blocking=True)
        z_chain = sample_chain_first_hop(model, z_d50, x_d50, rollout_times=rollout_times)
        batch_rows = [{"slice_idx": int(i)} for i in idx.tolist()]
        for tp_i, tp in enumerate(dataset.rollout_timepoints):
            x_gt = dataset.images[tp][idx].to(device, non_blocking=True).float()
            for suffix, refiner_flag in modes:
                actual_refiner_flag = refiner_flag
                if tp_i == 0 and getattr(model, "seam_refiner_skip_first_tp", False) and refiner_flag is None:
                    actual_refiner_flag = False
                x_pred = model.decode_crop(z_chain[tp_i], crop_size=image_size, apply_refiner=actual_refiner_flag).float()
                metrics = compute_roi_metrics(x_pred, x_gt)
                psnr_clip3_vals = []
                for b in range(x_pred.shape[0]):
                    psnr_clip3_vals.append(float(calc_psnr_clip3(x_pred[b : b + 1].detach().cpu(), x_gt[b : b + 1].detach().cpu())))
                metrics["psnr_clip3"] = torch.tensor(psnr_clip3_vals, dtype=torch.float32, device=x_pred.device)
                mode_key = suffix or "default"
                for metric_name, values in metrics.items():
                    acc = metric_acc[tp][mode_key].setdefault(metric_name, _new_acc())
                    _add(acc, values.detach().cpu())
                    for b, row in enumerate(batch_rows):
                        col = f"{metric_name}{suffix}_{tp}" if suffix else f"{metric_name}_{tp}"
                        row[col] = float(values[b].detach().cpu().item())
        rows.extend(batch_rows)

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for tp, modes_dict in metric_acc.items():
        summary[tp] = {}
        for mode_key, metric_dict in modes_dict.items():
            summary[tp][mode_key] = {metric: _summary(acc) for metric, acc in metric_dict.items()}
    return summary, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({k for row in rows for k in row.keys() if k != "slice_idx"})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["slice_idx"] + keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag.strip() or Path(args.checkpoint).parent.name + "_" + Path(args.checkpoint).stem

    dataset = build_dataset(cfg, args.split)
    eval_indices = select_indices(dataset.num_slices, int(args.max_slices))
    rollout_tps = cfg["data"].get("rollout_timepoints", TIMEPOINTS)
    rollout_times = [float(cfg["data"]["t_map"][tp]) for tp in rollout_tps]
    image_size = int(cfg["data"].get("image_size", 224))
    t0 = time.time()
    model, ckpt_meta = load_model(cfg, args.checkpoint, device)
    summary, rows = evaluate(
        model=model,
        dataset=dataset,
        rollout_times=rollout_times,
        eval_indices=eval_indices,
        batch_size=int(args.batch_size),
        image_size=image_size,
        device=device,
        decode_mode=args.decode_mode,
    )
    runtime_sec = time.time() - t0
    payload = {
        "tag": tag,
        "meta": {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "checkpoint_step": ckpt_meta.get("step") if isinstance(ckpt_meta, dict) else None,
            "checkpoint_best_val": ckpt_meta.get("best_val") if isinstance(ckpt_meta, dict) else None,
            "split": args.split,
            "max_slices": int(args.max_slices),
            "num_eval_slices": int(eval_indices.numel()),
            "batch_size": int(args.batch_size),
            "device": str(device),
            "decode_mode": args.decode_mode,
            "runtime_sec": float(runtime_sec),
            "protocol": "chain rollout ROI/high-SUV diagnostics; SUV=(x_norm+1)*5; ROI masks from GT percentiles",
            "rollout_timepoints": rollout_tps,
        },
        "summary": summary,
    }
    json_path = out_dir / "roi_psnr_summary.json"
    csv_path = out_dir / "roi_psnr_per_slice.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
