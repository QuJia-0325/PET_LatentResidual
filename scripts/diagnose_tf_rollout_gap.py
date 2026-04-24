#!/usr/bin/env python3
"""Path A: Teacher-Forcing vs Rollout Gap Diagnostic (eval only, no training).

Purpose
-------
E1 (review/0416/conclusion.md) established that 96-99% of the 10 dB oracle
gap is in the *transport* (latent-prediction) component — not the decoder
and not the ODE integrator. Remaining question: is that transport error
caused by **exposure bias** (velocity sees clean z_gt at train time but
rolled-out z_pred at eval time) or by **intrinsic velocity capacity**?

Design
------
For each hop k in {0, 1, 2, 3}, produce two 1-step predictions and compare:

  Arm TF (teacher-forced input):
      z_pred_TF[k+1] = model.predict_latent_step(z_gt[t_k] -> t_{k+1})

  Arm RO (rollout input):
      z_pred_RO[k+1] = model.predict_latent_step(z_pred_RO[t_k] -> t_{k+1})
      (where z_pred_RO[t_0] = z_gt[t_0] = z_d50, so at k=0 TF == RO.)

Metrics per hop:
  - latent_mse_TF, latent_mse_RO
  - PSNR_TF  = PSNR(decode(z_pred_TF), x_gt[t_{k+1}])
  - PSNR_RO  = PSNR(decode(z_pred_RO), x_gt[t_{k+1}])
  - exposure_gap_dB = PSNR_TF - PSNR_RO    (>0 => exposure bias present)
  - ceiling_gap_dB  = PSNR_ceiling - PSNR_TF  (intrinsic per-step capacity gap)

Verdict matrix (printed at end):
  exposure_gap_dB   ceiling_gap_dB    Interpretation
  ---------------   --------------    --------------
  >= 0.5 dB         ~~                EXPOSURE BIAS dominates. Fix via scheduled
                                      sampling / DAgger / rollout-consistency.
  < 0.2 dB          >= 5 dB           VELOCITY CAPACITY bound. Fix needs richer
                                      velocity head, different loss, or pivot.
  < 0.2 dB          < 2 dB            Near-ceiling already. 10 dB gap is DECODER
                                      non-linearity / off-manifold (contradicts E1).
  0.2-0.5 dB        variable          MIXED cause; both regularization & capacity
                                      help, but neither alone closes the gap.

Usage
-----
    PYTHONPATH=/home/qujiaxiang/project/PET_LatentResidual \
    /home/qujiaxiang/.conda/envs/rae/bin/python scripts/diagnose_tf_rollout_gap.py \
        --config   configs/pet_flow/pet_flow_first_hop_224_50k_imgaux_boost.yaml \
        --checkpoint /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_imgaux_boost/best.pt \
        --split val --max-slices 0 --batch-size 8 --device cuda:0

Outputs
-------
  <out-dir>/tf_rollout_gap_<split>.json           (aggregated)
  <out-dir>/tf_rollout_gap_<split>_per_slice.csv  (raw per-slice metrics)
"""
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
import yaml
from tqdm import tqdm

from pet_lr.data_first_hop import PETFirstHopAligned4HopDataset
from pet_lr.model_first_hop import PETFlowDiTFirstHop
from pet_lr.rollout_first_hop import sample_one_step_first_hop
from pet_lr.path_guard import resolve_data_disk_dir

RAE_CODE_ROOT = "/home/qujiaxiang/project/RAE/code/RAE"
if RAE_CODE_ROOT not in sys.path:
    sys.path.insert(0, RAE_CODE_ROOT)
from src.utils.metrics import calc_psnr_clip3  # noqa: E402


TIMEPOINTS = ["D50", "D20", "D10", "D4", "NORMAL"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Path A: Teacher-Forcing vs Rollout Gap Diagnostic"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val")
    p.add_argument(
        "--max-slices",
        type=int,
        default=0,
        help="0 = evaluate all slices in the split.",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--out-dir",
        default="/data_2/qujiaxiang/outputs/PET_LatentResidual/diagnostics/tf_rollout_gap",
    )
    p.add_argument(
        "--exposure-threshold",
        type=float,
        default=0.5,
        help="exposure_gap_dB threshold (mean over hops 1..3) for the "
        "EXPOSURE_BIAS verdict. Below --exposure-near-zero => 'near zero'.",
    )
    p.add_argument("--exposure-near-zero", type=float, default=0.2)
    p.add_argument("--capacity-threshold", type=float, default=5.0)
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
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def paired_bootstrap_ci(
    vals_a: List[float],
    vals_b: List[float],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Paired bootstrap 95% CI on mean(a - b)."""
    a = np.asarray(vals_a, dtype=np.float64)
    b = np.asarray(vals_b, dtype=np.float64)
    if a.size == 0 or a.size != b.size:
        return {"mean_diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    n = a.size
    diffs = a - b
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = diffs[idx].mean()
    lo = float(np.quantile(boots, (1 - ci) / 2))
    hi = float(np.quantile(boots, 1 - (1 - ci) / 2))
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": lo,
        "ci_high": hi,
    }


@torch.no_grad()
def main() -> None:
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
    num_hops = len(rollout_tps) - 1  # 4

    eval_idx = select_indices(dataset.num_slices, args.max_slices)
    n_eval = int(eval_idx.numel())
    print(
        f"[Path A] TF-vs-Rollout diagnostic on n={n_eval} slices | "
        f"device={device} | config={Path(args.config).name}",
        flush=True,
    )

    # Collectors per hop (hop index = source timepoint index, predicting timepoint k+1).
    # Hops are indexed 0..3; hop 0 is D50->D20 (TF == RO by construction).
    latent_mse_tf: Dict[int, List[float]] = {k: [] for k in range(num_hops)}
    latent_mse_ro: Dict[int, List[float]] = {k: [] for k in range(num_hops)}
    psnr_tf: Dict[int, List[float]] = {k: [] for k in range(num_hops)}
    psnr_ro: Dict[int, List[float]] = {k: [] for k in range(num_hops)}
    psnr_ceiling: Dict[int, List[float]] = {k: [] for k in range(num_hops)}

    bs = args.batch_size
    for start in tqdm(range(0, n_eval, bs), desc="PathA"):
        idx = eval_idx[start : start + bs]
        bsz = idx.numel()

        # Source image (needed only for hop 0 pixel forcing, if enabled).
        x_d50 = dataset.images["D50"][idx].to(device)

        # All GT latents for this minibatch, pre-loaded once.
        z_gt_all: List[torch.Tensor] = [
            dataset.latents[tp][idx].to(device) for tp in rollout_tps
        ]
        # GT images (raw) on CPU for PSNR.
        x_gt_all_cpu: List[torch.Tensor] = [
            dataset.images[tp][idx].float().cpu() for tp in rollout_tps
        ]

        # Rollout: run full chain, teacher-forcing NOTHING.
        # z_ro[k] = the model's own prediction fed as input to hop k.
        z_ro: List[torch.Tensor] = [z_gt_all[0]]  # hop 0 input == z_d50 (GT)
        for hop_idx in range(num_hops):
            x_src_img = x_d50 if hop_idx == 0 else None
            z_next_ro = sample_one_step_first_hop(
                model=model,
                z_start=z_ro[hop_idx],
                t_start=rollout_times[hop_idx],
                t_end=rollout_times[hop_idx + 1],
                hop_idx=hop_idx,
                x_src_img=x_src_img,
            )
            z_ro.append(z_next_ro)

        # Teacher-forced single-step for each hop.
        for hop_idx in range(num_hops):
            x_src_img = x_d50 if hop_idx == 0 else None
            # TF: input is z_gt at t_k
            z_pred_tf = sample_one_step_first_hop(
                model=model,
                z_start=z_gt_all[hop_idx],
                t_start=rollout_times[hop_idx],
                t_end=rollout_times[hop_idx + 1],
                hop_idx=hop_idx,
                x_src_img=x_src_img,
            )
            # RO: input is z_ro[hop_idx] (already computed above, = z_pred_RO at t_k).
            # For hop 0, z_ro[0] == z_gt_all[0], so z_pred_ro == z_pred_tf (sanity).
            z_pred_ro = z_ro[hop_idx + 1]

            z_target = z_gt_all[hop_idx + 1]

            # Decode both predictions and ceiling reference.
            x_from_tf = model.decode_crop(z_pred_tf, crop_size=image_size).detach().cpu()
            x_from_ro = model.decode_crop(z_pred_ro, crop_size=image_size).detach().cpu()
            x_from_gt = model.decode_crop(z_target, crop_size=image_size).detach().cpu()

            x_raw = x_gt_all_cpu[hop_idx + 1]

            for b in range(bsz):
                # Latent MSE
                lmse_tf = float(((z_pred_tf[b] - z_target[b]).pow(2)).mean().item())
                lmse_ro = float(((z_pred_ro[b] - z_target[b]).pow(2)).mean().item())
                # Decoded PSNR
                p_tf = float(calc_psnr_clip3(x_from_tf[b : b + 1], x_raw[b : b + 1]))
                p_ro = float(calc_psnr_clip3(x_from_ro[b : b + 1], x_raw[b : b + 1]))
                p_ceil = float(calc_psnr_clip3(x_from_gt[b : b + 1], x_raw[b : b + 1]))

                latent_mse_tf[hop_idx].append(lmse_tf)
                latent_mse_ro[hop_idx].append(lmse_ro)
                psnr_tf[hop_idx].append(p_tf)
                psnr_ro[hop_idx].append(p_ro)
                psnr_ceiling[hop_idx].append(p_ceil)

    # --- Aggregate per hop ---
    results: Dict[str, Dict] = {}
    for hop_idx in range(num_hops):
        tp_src = rollout_tps[hop_idx]
        tp_dst = rollout_tps[hop_idx + 1]
        key = f"hop{hop_idx}_{tp_src}_to_{tp_dst}"

        tf_s = summarize(psnr_tf[hop_idx])
        ro_s = summarize(psnr_ro[hop_idx])
        ceil_s = summarize(psnr_ceiling[hop_idx])
        lmse_tf_s = summarize(latent_mse_tf[hop_idx])
        lmse_ro_s = summarize(latent_mse_ro[hop_idx])

        boot_exposure = paired_bootstrap_ci(psnr_tf[hop_idx], psnr_ro[hop_idx])
        boot_ceiling = paired_bootstrap_ci(psnr_ceiling[hop_idx], psnr_tf[hop_idx])

        results[key] = {
            "hop_idx": hop_idx,
            "src_tp": tp_src,
            "dst_tp": tp_dst,
            "psnr_tf": tf_s,
            "psnr_rollout": ro_s,
            "psnr_ceiling": ceil_s,
            "latent_mse_tf": lmse_tf_s,
            "latent_mse_rollout": lmse_ro_s,
            "exposure_gap_dB": round(tf_s["mean"] - ro_s["mean"], 4),
            "ceiling_gap_dB": round(ceil_s["mean"] - tf_s["mean"], 4),
            "exposure_gap_ci95": boot_exposure,
            "ceiling_gap_ci95": boot_ceiling,
            "latent_mse_ratio_ro_over_tf": round(
                lmse_ro_s["mean"] / max(lmse_tf_s["mean"], 1e-12), 4
            ),
        }

    # Mean over hops 1..3 (hop 0 is trivially TF==RO so excluded from verdict).
    downstream_hops = [k for k in range(num_hops) if k >= 1]
    if downstream_hops:
        mean_exposure = float(
            np.mean(
                [
                    results[f"hop{k}_{rollout_tps[k]}_to_{rollout_tps[k+1]}"][
                        "exposure_gap_dB"
                    ]
                    for k in downstream_hops
                ]
            )
        )
        mean_ceiling = float(
            np.mean(
                [
                    results[f"hop{k}_{rollout_tps[k]}_to_{rollout_tps[k+1]}"][
                        "ceiling_gap_dB"
                    ]
                    for k in downstream_hops
                ]
            )
        )
    else:
        mean_exposure = float("nan")
        mean_ceiling = float("nan")

    # Verdict.
    if mean_exposure >= args.exposure_threshold:
        verdict = "EXPOSURE_BIAS_DOMINATES"
        advice = (
            "Fix: scheduled sampling / DAgger / rollout-consistency loss. "
            "chainstable-style rollout loss already targets this; inspect its "
            "hyperparameters (alpha ramp, step weights) and the mid-chain "
            "contribution."
        )
    elif mean_exposure < args.exposure_near_zero and mean_ceiling >= args.capacity_threshold:
        verdict = "VELOCITY_CAPACITY_BOUND"
        advice = (
            "Fix needs richer velocity head, alternative loss surfaces "
            "(e.g. direct z-regression, diffusion with CFG), or paradigm pivot. "
            "36.2 dB plateau likely structural."
        )
    elif mean_exposure < args.exposure_near_zero and mean_ceiling < 2.0:
        verdict = "NEAR_CEILING"
        advice = (
            "Transport is near decoder ceiling; any remaining gap is off-manifold "
            "decoder non-linearity. Contradicts E1's 96%% transport fraction — "
            "re-check data pipeline."
        )
    else:
        verdict = "MIXED_CAUSE"
        advice = (
            "Both exposure bias and velocity capacity contribute. Neither alone "
            "closes the gap; combine scheduled sampling with a richer velocity "
            "head and re-evaluate."
        )

    payload = {
        "meta": {
            "script": "scripts/diagnose_tf_rollout_gap.py",
            "config": args.config,
            "checkpoint": args.checkpoint,
            "split": args.split,
            "n_eval": n_eval,
            "batch_size": args.batch_size,
            "device": str(device),
            "psnr_metric": "src.utils.metrics.calc_psnr_clip3",
            "rollout_timepoints": rollout_tps,
            "thresholds": {
                "exposure_threshold": args.exposure_threshold,
                "exposure_near_zero": args.exposure_near_zero,
                "capacity_threshold": args.capacity_threshold,
            },
        },
        "per_hop": results,
        "downstream_summary": {
            "hops": downstream_hops,
            "mean_exposure_gap_dB": round(mean_exposure, 4),
            "mean_ceiling_gap_dB": round(mean_ceiling, 4),
        },
        "verdict": verdict,
        "advice": advice,
    }

    out_path = out_dir / f"tf_rollout_gap_{args.split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # Per-slice CSV for downstream stats.
    csv_path = out_dir / f"tf_rollout_gap_{args.split}_per_slice.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["slice_idx"]
        for k in range(num_hops):
            tag = f"hop{k}_{rollout_tps[k]}_to_{rollout_tps[k+1]}"
            header.extend(
                [
                    f"{tag}__psnr_tf",
                    f"{tag}__psnr_rollout",
                    f"{tag}__psnr_ceiling",
                    f"{tag}__latent_mse_tf",
                    f"{tag}__latent_mse_rollout",
                ]
            )
        w.writerow(header)
        for i in range(n_eval):
            row = [int(eval_idx[i].item())]
            for k in range(num_hops):
                row.extend(
                    [
                        f"{psnr_tf[k][i]:.6f}",
                        f"{psnr_ro[k][i]:.6f}",
                        f"{psnr_ceiling[k][i]:.6f}",
                        f"{latent_mse_tf[k][i]:.8e}",
                        f"{latent_mse_ro[k][i]:.8e}",
                    ]
                )
            w.writerow(row)

    # Print summary.
    print("\n" + "=" * 96)
    print("Path A: Teacher-Forcing vs Rollout Gap Diagnostic")
    print("=" * 96)
    print(
        f"{'Hop':<18} {'PSNR_TF':>10} {'PSNR_RO':>10} {'PSNR_Ceil':>10} "
        f"{'ExpoGap':>9} {'CeilGap':>9} {'LatMSE_RO/TF':>14}"
    )
    print("-" * 96)
    for hop_idx in range(num_hops):
        key = f"hop{hop_idx}_{rollout_tps[hop_idx]}_to_{rollout_tps[hop_idx+1]}"
        r = results[key]
        print(
            f"{rollout_tps[hop_idx]+'->'+rollout_tps[hop_idx+1]:<18} "
            f"{r['psnr_tf']['mean']:>10.4f} "
            f"{r['psnr_rollout']['mean']:>10.4f} "
            f"{r['psnr_ceiling']['mean']:>10.4f} "
            f"{r['exposure_gap_dB']:>9.4f} "
            f"{r['ceiling_gap_dB']:>9.4f} "
            f"{r['latent_mse_ratio_ro_over_tf']:>14.4f}"
        )
    print("=" * 96)
    print(
        f"[Downstream (hops 1..3)] mean exposure_gap = "
        f"{mean_exposure:.4f} dB | mean ceiling_gap = {mean_ceiling:.4f} dB"
    )
    print(f"[VERDICT] {verdict}")
    print(f"[ADVICE ] {advice}")
    print(f"[Artifact] JSON => {out_path}")
    print(f"[Artifact] CSV  => {csv_path}")


if __name__ == "__main__":
    main()
