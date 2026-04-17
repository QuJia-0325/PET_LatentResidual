#!/usr/bin/env python3
"""E1: Hop0 Error Budget Decomposition (eval only, no training).

Decomposes the D50→D20 PSNR gap into three additive components:
  1. Decoder ceiling gap:  PSNR(decode(z_gt_D20), x_gt_D20) vs perfect
  2. Transport-only gap:   MSE(z_pred_D20, z_gt_D20) in latent space
  3. Off-manifold amplif.: PSNR(decode(z_pred), decode(z_gt)) vs PSNR(decode(z_pred), x_gt)

Usage:
    python scripts/diagnose_error_budget.py \
        --config configs/pet_flow/pet_flow_first_hop_224_50k_formal_v3_chainstable.yaml \
        --checkpoint /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_formal_v3_chainstable/best.pt \
        --split val --max-slices 512 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from tqdm import tqdm

from pet_lr.data_first_hop import PETFirstHopAligned4HopDataset
from pet_lr.model_first_hop import PETFlowDiTFirstHop
from pet_lr.rollout_first_hop import sample_chain_first_hop
from pet_lr.path_guard import resolve_data_disk_dir

RAE_CODE_ROOT = "/home/qujiaxiang/project/RAE/code/RAE"
if RAE_CODE_ROOT not in sys.path:
    sys.path.insert(0, RAE_CODE_ROOT)
from src.utils.metrics import calc_psnr_clip3  # noqa: E402


TIMEPOINTS = ["D50", "D20", "D10", "D4", "NORMAL"]


def parse_args():
    p = argparse.ArgumentParser(description="E1: Hop0 Error Budget Decomposition")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--max-slices", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", default="/data_2/qujiaxiang/outputs/PET_LatentResidual/diagnostics/error_budget")
    return p.parse_args()


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_indices(n: int, max_n: int) -> torch.Tensor:
    if max_n <= 0 or max_n >= n:
        return torch.arange(n, dtype=torch.long)
    idx = torch.linspace(0, n - 1, steps=max_n).round().long()
    return torch.unique(idx, sorted=True)


def summarize(vals: List[float]) -> Dict[str, float]:
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max())}


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    device = torch.device(args.device)
    out_dir = Path(resolve_data_disk_dir(args.out_dir, arg_name="--out-dir"))
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg["data"]
    image_size = int(data_cfg.get("image_size", 224))
    latent_path = os.path.join(data_cfg["latent_dir"], f"latents_{args.split}.pt")
    alignment_audit_json = str(data_cfg.get("alignment_audit_json", "")).strip() or None

    dataset = PETFirstHopAligned4HopDataset(
        latent_path=latent_path,
        raw_data_dir=data_cfg["raw_data_dir"],
        split=args.split,
        clamp_max=float(data_cfg.get("clamp_max", 10.0)),
        t_map=data_cfg["t_map"],
        rollout_timepoints=data_cfg.get("rollout_timepoints", TIMEPOINTS),
        verify_alignment=True,
        alignment_check_num_samples=int(data_cfg.get("alignment_check_num_samples", 16)),
        alignment_audit_json=alignment_audit_json,
        include_x_rollout_first=True,
        include_full_x_rollout=True,
        image_size=image_size,
    )

    # Load model
    model = PETFlowDiTFirstHop(cfg, device=device).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    rollout_tps = data_cfg.get("rollout_timepoints", TIMEPOINTS)
    rollout_times = [float(data_cfg["t_map"][tp]) for tp in rollout_tps]
    eval_idx = select_indices(dataset.num_slices, args.max_slices)
    n_eval = int(eval_idx.numel())

    # Collectors per timepoint
    psnr_decoder_ceiling: Dict[str, List[float]] = {tp: [] for tp in rollout_tps}
    psnr_pred_vs_gt_img: Dict[str, List[float]] = {tp: [] for tp in rollout_tps}
    psnr_pred_vs_raw: Dict[str, List[float]] = {tp: [] for tp in rollout_tps}
    latent_mse_per_tp: Dict[str, List[float]] = {tp: [] for tp in rollout_tps}

    print(f"[E1] Evaluating {n_eval} slices on {device}...", flush=True)
    bs = args.batch_size

    for start in tqdm(range(0, n_eval, bs), desc="E1"):
        idx = eval_idx[start:start + bs]
        bsz = idx.numel()

        z_d50 = dataset.latents["D50"][idx].to(device)
        x_d50 = dataset.images["D50"][idx].to(device)

        # --- Transport prediction (full chain) ---
        z_chain = sample_chain_first_hop(model, z_d50, x_d50, rollout_times=rollout_times)

        for tp_i, tp in enumerate(rollout_tps):
            z_gt = dataset.latents[tp][idx].to(device)
            x_raw = dataset.images[tp][idx].float().cpu()

            # 1. Decoder ceiling: decode GT latent → compare with raw image
            x_from_gt = model.decode_crop(z_gt, crop_size=image_size).detach().cpu()

            # 2. Transport prediction: decode predicted latent
            z_pred = z_chain[tp_i]
            x_from_pred = model.decode_crop(z_pred, crop_size=image_size).detach().cpu()

            # 3. Decode GT for off-manifold comparison
            x_from_gt_cpu = x_from_gt

            for b in range(bsz):
                # Decoder ceiling PSNR: how good can decode(z_gt) be?
                psnr_ceil = float(calc_psnr_clip3(x_from_gt[b:b+1], x_raw[b:b+1]))
                psnr_decoder_ceiling[tp].append(psnr_ceil)

                # Pred vs raw: end-to-end PSNR
                psnr_e2e = float(calc_psnr_clip3(x_from_pred[b:b+1], x_raw[b:b+1]))
                psnr_pred_vs_raw[tp].append(psnr_e2e)

                # Pred vs GT decode: off-manifold amplification
                psnr_pred_gt = float(calc_psnr_clip3(x_from_pred[b:b+1], x_from_gt_cpu[b:b+1]))
                psnr_pred_vs_gt_img[tp].append(psnr_pred_gt)

                # Latent MSE
                lmse = float(((z_pred[b] - z_gt[b]).pow(2)).mean().item())
                latent_mse_per_tp[tp].append(lmse)

    # --- Aggregate ---
    results = {}
    for tp in rollout_tps:
        ceil_stats = summarize(psnr_decoder_ceiling[tp])
        e2e_stats = summarize(psnr_pred_vs_raw[tp])
        pred_gt_stats = summarize(psnr_pred_vs_gt_img[tp])
        lmse_stats = summarize(latent_mse_per_tp[tp])

        gap_total = ceil_stats["mean"] - e2e_stats["mean"]  # Total gap = ceiling - actual
        gap_transport = ceil_stats["mean"] - pred_gt_stats["mean"]  # Decoder cancels, isolates latent error
        gap_decoder = pred_gt_stats["mean"] - e2e_stats["mean"]  # Residual from decode(z_gt) ≠ x_raw

        results[tp] = {
            "decoder_ceiling_psnr": ceil_stats,
            "end_to_end_psnr": e2e_stats,
            "pred_vs_gt_decode_psnr": pred_gt_stats,
            "latent_mse": lmse_stats,
            "gap_total_dB": round(gap_total, 4),
            "gap_transport_dB": round(gap_transport, 4),
            "gap_decoder_dB": round(gap_decoder, 4),
            "transport_fraction": round(gap_transport / max(gap_total, 1e-8), 4) if gap_total > 0 else 0.0,
        }

    payload = {
        "meta": {
            "script": "scripts/diagnose_error_budget.py",
            "config": args.config,
            "checkpoint": args.checkpoint,
            "split": args.split,
            "n_eval": n_eval,
            "psnr_metric": "src.utils.metrics.calc_psnr_clip3",
        },
        "results": results,
    }

    out_path = out_dir / f"error_budget_{args.split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 80)
    print("E1: Error Budget Decomposition")
    print("=" * 80)
    print(f"{'TP':<8} {'Ceiling':>10} {'E2E':>10} {'Gap_Total':>10} {'Gap_Transp':>11} {'Gap_Dec':>10} {'%Transport':>11}")
    print("-" * 80)
    for tp in rollout_tps:
        r = results[tp]
        print(
            f"{tp:<8} "
            f"{r['decoder_ceiling_psnr']['mean']:>10.4f} "
            f"{r['end_to_end_psnr']['mean']:>10.4f} "
            f"{r['gap_total_dB']:>10.4f} "
            f"{r['gap_transport_dB']:>11.4f} "
            f"{r['gap_decoder_dB']:>10.4f} "
            f"{r['transport_fraction']*100:>10.1f}%"
        )
    print("=" * 80)

    # Gate check for FOC-lite
    d20 = results.get("D20", {})
    tfrac = d20.get("transport_fraction", 0)
    print(f"\n[GATE] D20 transport_fraction = {tfrac*100:.1f}%")
    if tfrac >= 0.30:
        print("[GATE] PASS: ODE integration error accounts for ≥30% of hop0 gap.")
        print("[GATE] FOC-lite hypothesis is plausible — proceed to E2.")
    else:
        print("[GATE] FAIL: ODE integration error accounts for <30% of hop0 gap.")
        print("[GATE] FOC-lite should be deprioritized. Focus on decoder-side or encoder interventions.")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
