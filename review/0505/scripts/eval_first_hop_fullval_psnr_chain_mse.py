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
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pet_lr.data_first_hop import PETFirstHopAligned4HopDataset
from pet_lr.model_first_hop import PETFlowDiTFirstHop
from pet_lr.rollout_first_hop import sample_chain_first_hop
from pet_lr.path_guard import resolve_data_disk_dir

RAE_CODE_ROOT = "/home/qujiaxiang/project/RAE/code/RAE"
if RAE_CODE_ROOT not in sys.path:
    sys.path.insert(0, RAE_CODE_ROOT)
from src.utils.metrics import calc_psnr_clip3  # noqa: E402

TIMEPOINTS = ["D50", "D20", "D10", "D4", "NORMAL"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Full-val first-hop eval with canonical calc_psnr_clip3 and "
            "training-compatible decoded chain MSE, including chain_normal."
        )
    )
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--split", choices=["val"], default="val")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-slices", type=int, default=0, help="<=0 means full split")
    p.add_argument("--decode-mode", choices=["default", "raw", "both"], default="both")
    p.add_argument(
        "--out-dir",
        default="/data_2/qujiaxiang/outputs/PET_LatentResidual/eval_0505_v6_v61_fullval",
    )
    return p.parse_args()


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_indices(num_slices: int, max_slices: int) -> torch.Tensor:
    if max_slices <= 0 or max_slices >= num_slices:
        return torch.arange(num_slices, dtype=torch.long)
    idx = torch.linspace(0, num_slices - 1, steps=max_slices).round().long()
    return torch.unique(idx, sorted=True)


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
    return {
        "n": float(n),
        "mean": float(mean),
        "std": float(math.sqrt(var)),
        "min": float(acc["min"]),
        "max": float(acc["max"]),
    }


def _decode_modes(mode: str) -> list[tuple[str, bool | None]]:
    modes: list[tuple[str, bool | None]] = []
    if mode in ("default", "both"):
        modes.append(("", None))
    if mode in ("raw", "both"):
        modes.append(("_raw", False))
    return modes


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
) -> tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], list[dict]]:
    modes = _decode_modes(decode_mode)
    psnr_acc: Dict[str, Dict[str, float]] = {}
    mse_acc: Dict[str, Dict[str, float]] = {}
    for tp in dataset.rollout_timepoints:
        for suffix, _ in modes:
            key = f"{tp}{suffix}" if suffix else tp
            psnr_acc[key] = _new_acc()
            mse_acc[key] = _new_acc()

    rows: list[dict] = []
    total = int(eval_indices.numel())
    for start in tqdm(range(0, total, batch_size), desc=f"eval:{total}"):
        idx = eval_indices[start : start + batch_size]
        z_d50 = dataset.latents["D50"][idx].to(device, non_blocking=True)
        x_d50 = dataset.images["D50"][idx].to(device, non_blocking=True)
        z_chain = sample_chain_first_hop(model, z_d50, x_d50, rollout_times=rollout_times)

        batch_rows = [{"slice_idx": int(i)} for i in idx.tolist()]
        for tp_i, tp in enumerate(dataset.rollout_timepoints):
            x_gt = dataset.images[tp][idx].to(device, non_blocking=True).float()
            for suffix, refiner_flag in modes:
                apply_refiner = refiner_flag
                if tp_i == 0 and getattr(model, "seam_refiner_skip_first_tp", False) and refiner_flag is None:
                    apply_refiner = False
                x_pred = model.decode_crop(z_chain[tp_i], crop_size=image_size, apply_refiner=apply_refiner).float()
                per_sample_mse = (x_pred - x_gt).pow(2).flatten(start_dim=1).mean(dim=1)
                key = f"{tp}{suffix}" if suffix else tp
                _add(mse_acc[key], per_sample_mse.cpu())
                psnr_vals = []
                for b in range(x_pred.shape[0]):
                    psnr = float(calc_psnr_clip3(x_pred[b : b + 1].detach().cpu(), x_gt[b : b + 1].detach().cpu()))
                    psnr_vals.append(psnr)
                _add(psnr_acc[key], torch.tensor(psnr_vals, dtype=torch.float32))
                for b, row in enumerate(batch_rows):
                    row[f"mse{suffix}_{tp}"] = float(per_sample_mse[b].detach().cpu().item())
                    row[f"psnr{suffix}_{tp}"] = float(psnr_vals[b])
        rows.extend(batch_rows)

    return ({k: _summary(v) for k, v in psnr_acc.items()}, {k: _summary(v) for k, v in mse_acc.items()}, rows)


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
    out_dir = resolve_data_disk_dir(args.out_dir, arg_name="--out-dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(cfg, args.split)
    eval_indices = select_indices(dataset.num_slices, int(args.max_slices))
    image_size = int(cfg["data"].get("image_size", 224))
    rollout_tps = cfg["data"].get("rollout_timepoints", TIMEPOINTS)
    rollout_times = [float(cfg["data"]["t_map"][tp]) for tp in rollout_tps]

    t0 = time.time()
    model, ckpt_meta = load_model(cfg, args.checkpoint, device)
    psnr_summary, mse_summary, rows = evaluate(
        model=model,
        dataset=dataset,
        rollout_times=rollout_times,
        eval_indices=eval_indices,
        batch_size=int(args.batch_size),
        image_size=image_size,
        device=device,
        decode_mode=args.decode_mode,
    )
    dt = time.time() - t0

    def mean_or_nan(key: str, metric: Dict[str, Dict[str, float]]) -> float:
        return float(metric.get(key, {}).get("mean", float("nan")))

    payload = {
        "tag": args.tag,
        "meta": {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "checkpoint_step": ckpt_meta.get("step"),
            "checkpoint_best_val": ckpt_meta.get("best_val"),
            "checkpoint_best_metric_name": ckpt_meta.get("best_metric_name"),
            "split": args.split,
            "max_slices": int(args.max_slices),
            "num_eval_slices": int(eval_indices.numel()),
            "batch_size": int(args.batch_size),
            "device": str(device),
            "decode_mode": args.decode_mode,
            "runtime_sec": float(dt),
            "psnr_metric": "src.utils.metrics.calc_psnr_clip3",
            "chain_mse_definition": "mean((decode_crop(z_chain[t]) - x_rollout[t])^2), matching train_first_hop.py val_chain_*_mse",
            "rollout_timepoints": rollout_tps,
        },
        "summary_psnr_clip3": psnr_summary,
        "summary_chain_mse": mse_summary,
        "headline": {
            "normal_psnr": mean_or_nan("NORMAL", psnr_summary),
            "normal_chain_mse": mean_or_nan("NORMAL", mse_summary),
            "normal_psnr_raw": mean_or_nan("NORMAL_raw", psnr_summary),
            "normal_chain_mse_raw": mean_or_nan("NORMAL_raw", mse_summary),
            "tail_chain_mse": float(
                (mean_or_nan("D10", mse_summary) + mean_or_nan("D4", mse_summary) + mean_or_nan("NORMAL", mse_summary)) / 3.0
            ),
            "transport_psnr_avg": float(
                (mean_or_nan("D20", psnr_summary) + mean_or_nan("D10", psnr_summary) + mean_or_nan("D4", psnr_summary) + mean_or_nan("NORMAL", psnr_summary)) / 4.0
            ),
        },
    }

    json_path = out_dir / f"{args.tag}_fullval_psnr_chain_mse.json"
    csv_path = out_dir / f"{args.tag}_fullval_psnr_chain_mse_per_slice.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)

    print(json.dumps({"tag": args.tag, **payload["headline"], "runtime_sec": dt}, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved JSON: {json_path}", flush=True)
    print(f"Saved CSV:  {csv_path}", flush=True)


if __name__ == "__main__":
    main()
