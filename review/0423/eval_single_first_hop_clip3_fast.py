#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

from pet_lr.data_first_hop import PETFirstHopAligned4HopDataset
from pet_lr.model_first_hop import PETFlowDiTFirstHop
from pet_lr.rollout_first_hop import sample_chain_first_hop
from pet_lr.path_guard import ensure_repo_local_outputs_absent

TIMEPOINTS = ["D50", "D20", "D10", "D4", "NORMAL"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast full-val eval for a single checkpoint (clip3 PSNR).")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tag", required=True, help="Result tag, e.g. backbone_lr1_best")
    p.add_argument("--split", choices=["val"], default="val")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--out-dir",
        default="/data_2/qujiaxiang/outputs/PET_LatentResidual/clip3_eval/0423_backbone_lr1_fast",
    )
    return p.parse_args()


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataset(cfg: Dict, split: str) -> PETFirstHopAligned4HopDataset:
    data_cfg = cfg["data"]
    latent_path = str(Path(data_cfg["latent_dir"]) / f"latents_{split}.pt")
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


def load_model(cfg: Dict, ckpt_path: str, device: torch.device) -> PETFlowDiTFirstHop:
    model = PETFlowDiTFirstHop(cfg, device=device).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    if isinstance(ckpt, dict):
        ckpt_pixel = ckpt.get("first_hop_pixel_enabled", None)
        current_pixel = not model.pixel_forcing_disabled
        if ckpt_pixel is not None and bool(ckpt_pixel) != current_pixel:
            raise RuntimeError(
                f"Eval pixel semantic mismatch: checkpoint={ckpt_pixel}, "
                f"config pixel_forcing_disabled={model.pixel_forcing_disabled}"
            )
    model.eval()
    return model


def psnr_clip3_per_sample(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    pred_suv = (pred + 1.0) * 5.0
    gt_suv = (gt + 1.0) * 5.0
    pred_suv = pred_suv.clamp(0.0, 3.0)
    gt_suv = gt_suv.clamp(0.0, 3.0)
    mse = (pred_suv - gt_suv).pow(2).flatten(start_dim=1).mean(dim=1)
    eps = torch.finfo(mse.dtype).tiny
    psnr = 20.0 * math.log10(3.0) - 10.0 * torch.log10(mse.clamp_min(eps))
    psnr = torch.where(mse <= 0.0, torch.full_like(psnr, float("inf")), psnr)
    return psnr


def summarize_from_acc(acc: Dict[str, float]) -> Dict[str, float]:
    n = int(acc["n"])
    if n <= 0:
        return {"n": 0.0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    mean = acc["sum"] / float(n)
    var = max(acc["sum_sq"] / float(n) - mean * mean, 0.0)
    std = math.sqrt(var)
    return {
        "n": float(n),
        "mean": float(mean),
        "std": float(std),
        "min": float(acc["min"]),
        "max": float(acc["max"]),
    }


@torch.no_grad()
def eval_checkpoint(
    model: PETFlowDiTFirstHop,
    dataset: PETFirstHopAligned4HopDataset,
    rollout_times: List[float],
    image_size: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, object]:
    acc: Dict[str, Dict[str, float]] = {
        tp: {"sum": 0.0, "sum_sq": 0.0, "n": 0, "min": float("inf"), "max": float("-inf")}
        for tp in dataset.rollout_timepoints
    }
    n_total = dataset.num_slices
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        idx = torch.arange(start, end, dtype=torch.long)
        z_d50 = dataset.latents["D50"][idx].to(device, non_blocking=True)
        x_d50 = dataset.images["D50"][idx].to(device, non_blocking=True)
        z_chain = sample_chain_first_hop(model, z_d50, x_d50, rollout_times=rollout_times)
        for tp_i, tp in enumerate(dataset.rollout_timepoints):
            x_gt = dataset.images[tp][idx].to(device, non_blocking=True)
            apply_refiner = None
            if tp_i == 0 and getattr(model, "seam_refiner_skip_first_tp", False):
                apply_refiner = False
            x_pred = model.decode_crop(z_chain[tp_i], crop_size=image_size, apply_refiner=apply_refiner)
            psnr = psnr_clip3_per_sample(x_pred.float(), x_gt.float())
            finite = psnr[torch.isfinite(psnr)]
            if finite.numel() == 0:
                continue
            s = float(finite.sum().item())
            sq = float((finite * finite).sum().item())
            mn = float(finite.min().item())
            mx = float(finite.max().item())
            n = int(finite.numel())
            acc_tp = acc[tp]
            acc_tp["sum"] += s
            acc_tp["sum_sq"] += sq
            acc_tp["n"] += n
            acc_tp["min"] = min(acc_tp["min"], mn)
            acc_tp["max"] = max(acc_tp["max"], mx)
        if (start // batch_size) % 80 == 0:
            print(f"[eval] processed {end}/{n_total} slices", flush=True)

    summary_psnr = {tp: summarize_from_acc(acc[tp]) for tp in dataset.rollout_timepoints}
    d20 = summary_psnr["D20"]["mean"]
    d10 = summary_psnr["D10"]["mean"]
    d4 = summary_psnr["D4"]["mean"]
    normal = summary_psnr["NORMAL"]["mean"]
    d50 = summary_psnr["D50"]["mean"]
    transport_avg = (d20 + d10 + d4 + normal) / 4.0
    all_avg = (d50 + d20 + d10 + d4 + normal) / 5.0
    return {
        "summary_psnr_clip3": summary_psnr,
        "summary_transport_avg": float(transport_avg),
        "summary_all_avg": float(all_avg),
    }


def write_tp_csv(path: Path, summary_psnr: Dict[str, Dict[str, float]]) -> None:
    fieldnames = ["timepoint", "n", "mean", "std", "min", "max"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for tp in TIMEPOINTS:
            row = {"timepoint": tp, **summary_psnr[tp]}
            w.writerow(row)


def main() -> None:
    args = parse_args()
    ensure_repo_local_outputs_absent(REPO_ROOT)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(args.config)
    device = torch.device(args.device)
    dataset = build_dataset(cfg, split=args.split)
    rollout_tps = cfg["data"].get("rollout_timepoints", TIMEPOINTS)
    rollout_times = [float(cfg["data"]["t_map"][tp]) for tp in rollout_tps]
    image_size = int(cfg["data"].get("image_size", 224))

    t0 = time.time()
    model = load_model(cfg, args.checkpoint, device=device)
    out = eval_checkpoint(
        model=model,
        dataset=dataset,
        rollout_times=rollout_times,
        image_size=image_size,
        batch_size=int(args.batch_size),
        device=device,
    )
    dt = time.time() - t0

    payload = {
        "tag": args.tag,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "batch_size": int(args.batch_size),
        "device": str(device),
        "runtime_sec": float(dt),
        "psnr_metric": "clip3_per_sample_formula_equivalent",
        **out,
    }

    json_path = out_dir / f"{args.tag}_fullval_clip3_fast.json"
    csv_path = out_dir / f"{args.tag}_fullval_clip3_fast.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    write_tp_csv(csv_path, out["summary_psnr_clip3"])

    print(json.dumps({
        "tag": args.tag,
        "runtime_sec": float(dt),
        "transport_avg": out["summary_transport_avg"],
        "all_avg": out["summary_all_avg"],
    }, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved JSON: {json_path}", flush=True)
    print(f"Saved CSV:  {csv_path}", flush=True)


if __name__ == "__main__":
    main()
