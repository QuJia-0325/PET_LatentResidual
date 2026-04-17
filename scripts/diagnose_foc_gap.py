#!/usr/bin/env python3
"""E2: Pre-training FOC Gap Measurement (eval only, no training).

Measures the full-step vs half-step discrepancy on hop0 (D50→D20) using the
EXISTING checkpoint, BEFORE any FOC-lite training. This tells us whether the
velocity field has significant ODE integration error that FOC-lite could correct.

If the gap is negligible, FOC-lite's hypothesis is falsified and it should be
deprioritized.

Usage:
    python scripts/diagnose_foc_gap.py \
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
from pet_lr.path_guard import resolve_data_disk_dir

RAE_CODE_ROOT = "/home/qujiaxiang/project/RAE/code/RAE"
if RAE_CODE_ROOT not in sys.path:
    sys.path.insert(0, RAE_CODE_ROOT)
from src.utils.metrics import calc_psnr_clip3  # noqa: E402


TIMEPOINTS = ["D50", "D20", "D10", "D4", "NORMAL"]


def parse_args():
    p = argparse.ArgumentParser(description="E2: Pre-training FOC Gap Measurement")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--max-slices", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", default="/data_2/qujiaxiang/outputs/PET_LatentResidual/diagnostics/foc_gap")
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

    model = PETFlowDiTFirstHop(cfg, device=device).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    rollout_tps = data_cfg.get("rollout_timepoints", TIMEPOINTS)
    rollout_times = [float(data_cfg["t_map"][tp]) for tp in rollout_tps]

    t_src_val = rollout_times[0]  # D50 = 2.0
    t_dst_val = rollout_times[1]  # D20 = 5.0
    t_mid_val = (t_src_val + t_dst_val) / 2.0  # 3.5

    eval_idx = select_indices(dataset.num_slices, args.max_slices)
    n_eval = int(eval_idx.numel())
    bs = args.batch_size

    # Collectors
    gap_abs_list: List[float] = []
    gap_rel_list: List[float] = []
    full_norm_list: List[float] = []
    half_norm_list: List[float] = []
    psnr_full_list: List[float] = []
    psnr_half_list: List[float] = []
    psnr_gt_list: List[float] = []
    per_sample_gap: List[float] = []
    per_sample_psnr_deficit: List[float] = []

    print(f"[E2] Measuring FOC gap on {n_eval} slices...", flush=True)

    for start in tqdm(range(0, n_eval, bs), desc="E2"):
        idx = eval_idx[start:start + bs]
        bsz = idx.numel()

        z_src = dataset.latents["D50"][idx].to(device)
        x_src = dataset.images["D50"][idx].to(device)
        z_gt_d20 = dataset.latents["D20"][idx].to(device)
        x_gt_d20 = dataset.images["D20"][idx].float().cpu()

        hop_idx = torch.zeros(bsz, device=device, dtype=torch.long)
        t_src = torch.full((bsz,), t_src_val, device=device)
        t_dst = torch.full((bsz,), t_dst_val, device=device)
        t_mid = torch.full((bsz,), t_mid_val, device=device)

        # Full step: D50 → D20
        out_full = model.predict_latent_step(
            z_src=z_src, t_src=t_src, t_dst=t_dst,
            hop_idx=hop_idx, x_src_img=x_src,
        )
        z_full = out_full["z_pred"]

        # Half step 1: D50 → mid
        out_h1 = model.predict_latent_step(
            z_src=z_src, t_src=t_src, t_dst=t_mid,
            hop_idx=hop_idx, x_src_img=x_src,
        )
        z_mid = out_h1["z_pred"]

        # Half step 2: mid → D20
        out_h2 = model.predict_latent_step(
            z_src=z_mid, t_src=t_mid, t_dst=t_dst,
            hop_idx=hop_idx, x_src_img=x_src,
        )
        z_half = out_h2["z_pred"]

        # Decode for PSNR comparison
        x_full = model.decode_crop(z_full, crop_size=image_size).detach().cpu()
        x_half = model.decode_crop(z_half, crop_size=image_size).detach().cpu()

        for b in range(bsz):
            # Latent-space gap
            gap = (z_full[b] - z_half[b]).abs().mean().item()
            fnorm = z_full[b].abs().mean().item()
            hnorm = z_half[b].abs().mean().item()
            rel = gap / max(fnorm, 1e-8)

            gap_abs_list.append(gap)
            gap_rel_list.append(rel)
            full_norm_list.append(fnorm)
            half_norm_list.append(hnorm)

            # PSNR comparison
            psnr_f = float(calc_psnr_clip3(x_full[b:b+1], x_gt_d20[b:b+1]))
            psnr_h = float(calc_psnr_clip3(x_half[b:b+1], x_gt_d20[b:b+1]))
            psnr_full_list.append(psnr_f)
            psnr_half_list.append(psnr_h)

            # For correlation analysis: does gap correlate with PSNR deficit?
            # Ceiling PSNR for this slice
            x_from_gt = model.decode_crop(z_gt_d20[b:b+1], crop_size=image_size).detach().cpu()
            psnr_ceil = float(calc_psnr_clip3(x_from_gt, x_gt_d20[b:b+1]))
            psnr_gt_list.append(psnr_ceil)
            deficit = psnr_ceil - psnr_f  # how much worse is transport pred vs ceiling
            per_sample_gap.append(gap)
            per_sample_psnr_deficit.append(deficit)

    # Correlation: gap vs psnr deficit (Reviewer-B falsification criterion #1)
    gap_arr = np.asarray(per_sample_gap, dtype=np.float64)
    deficit_arr = np.asarray(per_sample_psnr_deficit, dtype=np.float64)
    if gap_arr.std() > 1e-12 and deficit_arr.std() > 1e-12:
        corr = float(np.corrcoef(gap_arr, deficit_arr)[0, 1])
    else:
        corr = 0.0

    results = {
        "gap_abs": summarize(gap_abs_list),
        "gap_rel": summarize(gap_rel_list),
        "z_full_norm": summarize(full_norm_list),
        "z_half_norm": summarize(half_norm_list),
        "psnr_full_step": summarize(psnr_full_list),
        "psnr_half_step": summarize(psnr_half_list),
        "psnr_decoder_ceiling_d20": summarize(psnr_gt_list),
        "gap_vs_deficit_correlation": round(corr, 4),
        "half_minus_full_psnr_mean": round(
            float(np.mean(psnr_half_list)) - float(np.mean(psnr_full_list)), 4
        ),
    }

    payload = {
        "meta": {
            "script": "scripts/diagnose_foc_gap.py",
            "config": args.config,
            "checkpoint": args.checkpoint,
            "split": args.split,
            "n_eval": n_eval,
            "t_src": t_src_val,
            "t_dst": t_dst_val,
            "t_mid": t_mid_val,
            "psnr_metric": "src.utils.metrics.calc_psnr_clip3",
        },
        "results": results,
    }

    out_path = out_dir / f"foc_gap_{args.split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 70)
    print("E2: FOC Gap Measurement (hop0 D50→D20)")
    print("=" * 70)
    print(f"  Slices evaluated:       {n_eval}")
    print(f"  t_src={t_src_val}, t_mid={t_mid_val}, t_dst={t_dst_val}")
    print(f"  |z_full| mean:          {results['z_full_norm']['mean']:.6f}")
    print(f"  |z_half| mean:          {results['z_half_norm']['mean']:.6f}")
    print(f"  |z_full - z_half| mean: {results['gap_abs']['mean']:.6f}")
    print(f"  Relative gap:           {results['gap_rel']['mean']*100:.2f}%")
    print(f"  PSNR full-step:         {results['psnr_full_step']['mean']:.4f} dB")
    print(f"  PSNR half-step:         {results['psnr_half_step']['mean']:.4f} dB")
    print(f"  PSNR decoder ceiling:   {results['psnr_decoder_ceiling_d20']['mean']:.4f} dB")
    print(f"  Half - Full PSNR:       {results['half_minus_full_psnr_mean']:+.4f} dB")
    print(f"  Gap↔Deficit corr (r):   {results['gap_vs_deficit_correlation']:.4f}")
    print("-" * 70)

    gap_pct = results["gap_rel"]["mean"] * 100
    print(f"\n[GATE] Relative gap = {gap_pct:.2f}%")
    if gap_pct >= 5.0:
        print("[GATE] PASS: ODE integration error is non-trivial (≥5%).")
        print("[GATE] FOC-lite has empirical basis — proceed to training (E3).")
    else:
        print("[GATE] FAIL: ODE integration error is negligible (<5%).")
        print("[GATE] FOC-lite's hypothesis is weak. Consider deprioritizing.")

    half_vs_full = results["half_minus_full_psnr_mean"]
    if half_vs_full > 0.05:
        print(f"[BONUS] Half-step PSNR > Full-step by {half_vs_full:+.4f} dB →")
        print("        Two half-steps are already more accurate than one full-step.")
        print("        FOC-lite has strong justification: it teaches the model this accuracy.")
    elif half_vs_full < -0.05:
        print(f"[WARN] Half-step PSNR < Full-step by {half_vs_full:+.4f} dB →")
        print("       Two half-steps are WORSE (error accumulation from 2× bad v).")
        print("       FOC-lite's stop-gradient target may be harmful. Reconsider detach direction.")
    else:
        print(f"[INFO] Half-step ≈ Full-step ({half_vs_full:+.4f} dB). Marginal integration error.")

    corr_val = results["gap_vs_deficit_correlation"]
    if corr_val >= 0.3:
        print(f"[CAUSAL] Gap↔Deficit r={corr_val:.3f} ≥ 0.3 → integration error correlates with PSNR loss.")
    else:
        print(f"[CAUSAL] Gap↔Deficit r={corr_val:.3f} < 0.3 → weak/no correlation. "
              "Reviewer-B falsification criterion #1 applies.")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
