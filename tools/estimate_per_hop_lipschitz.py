#!/usr/bin/env python3
"""Per-hop Lipschitz measurement for V7/V8/V6_NOISE checkpoints.

Source: review/plan/ARCHITECTURE_ANALYSIS_20260501.md §16.4.4 / §19.4 / §19.6.
This script is the executable form of the long-pending "10-minute Lipschitz
check" that decides whether V7's step_weights are emergently optimal under
the multi-hop Grönwall closed-form (review/0516/STEP_WEIGHTS_THEORY_REFERENCE.md §6).

Outputs JSON:
    {
      "checkpoint": <path>,
      "config": <path>,
      "n_samples": int,
      "eps": float,
      "L_per_hop": [L0, L1, L2, L3],
      "deviation_pct": [...],
      "closed_form_w_with_measured_L": [w0, w1, w2, w3],
      "current_step_weights": [...],
      "alignment_distance_pct": float
    }

Usage (on the GPU server, conda env `rae`):
    python tools/estimate_per_hop_lipschitz.py \
        --config review/0505/local/runs/V7/config.resolved.yaml \
        --checkpoint /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V7/run/first_hop_224_v7_gronwall_raw/step_160000.pt \
        --out review/0517/lipschitz/V7_lipschitz.json \
        --n-samples 64 --eps 1e-3

NOTE: this is a SKELETON. The model API call (`predict_latent_step` /
equivalent) needs to be wired to the actual PETFlowDiTFirstHop interface in
pet_lr/. Search for "TODO" tags below.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pet_lr.data_first_hop import PETFirstHopAligned4HopDataset
from pet_lr.model_first_hop import PETFlowDiTFirstHop


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--split", choices=["val"], default="val")
    p.add_argument("--seed", type=int, default=20260517)
    return p.parse_args()


def load_v7_sigma_dt():
    """σ_j · dt_j for the four chain steps; from ARCHITECTURE_ANALYSIS §18.3.4."""
    return [(0.00963 * 3), (0.00295 * 5), (0.00078 * 15), (0.000141 * 75)]


def closed_form_weights(beta: List[float], L: List[float], sigma_dt: List[float]) -> List[float]:
    """Multi-hop Grönwall closed-form (ARCHITECTURE_ANALYSIS §18.3.3) with
    sigma·dt magnitude compensation (V7 config header §3).

        w_j  =  ( Σ_{k=j+1}^{K} β_k · Π_{i=j+1}^{k-1} L_i² )  /  (σ_j · dt_j)²
    """
    K = len(beta)
    w = []
    for j in range(K):
        s = 0.0
        for k in range(j + 1, K + 1):
            prod = 1.0
            for i in range(j + 1, k):
                prod *= L[i] ** 2 if i < K else 1.0
            s += beta[k - 1] * prod  # beta is 0-indexed for hops 1..K
        w.append(s / (sigma_dt[j] ** 2))
    # normalize so hop1 = 2.0 (anchor matches V6/V7 convention)
    anchor = w[1] / 2.0
    return [wi / anchor for wi in w]


def summarize(values: torch.Tensor) -> Dict[str, float]:
    values = values.detach().float().flatten()
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return {"n": 0.0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": float(finite.numel()),
        "mean": float(finite.mean().item()),
        "std": float(finite.std(unbiased=False).item()),
        "min": float(finite.min().item()),
        "max": float(finite.max().item()),
    }


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
        rollout_timepoints=data_cfg.get("rollout_timepoints", ["D50", "D20", "D10", "D4", "NORMAL"]),
        verify_alignment=bool(data_cfg.get("verify_alignment", True)),
        alignment_check_num_samples=int(data_cfg.get("alignment_check_num_samples", 16)),
        alignment_audit_json=alignment_audit_json,
        include_x_rollout_first=True,
        include_full_x_rollout=False,
        image_size=int(data_cfg.get("image_size", 224)),
        image_timepoints=["D50", "D20"],
        latent_mmap=bool(data_cfg.get("latent_mmap", False)),
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
                "Lipschitz eval pixel_forcing semantic mismatch: "
                f"checkpoint first_hop_pixel_enabled={ckpt_pixel}, "
                f"config pixel_forcing_disabled={model.pixel_forcing_disabled}"
            )
    model.eval()
    return model, ckpt if isinstance(ckpt, dict) else {}


def sample_hop_inputs(
    dataset: PETFirstHopAligned4HopDataset,
    n_samples: int,
) -> tuple[list[torch.Tensor], torch.Tensor | None, list[int]]:
    n = int(dataset.num_slices)
    k = min(int(n_samples), n)
    # Deterministic coverage across the validation volume; avoids cherry-picked contiguous slices.
    idx = torch.linspace(0, n - 1, steps=k).round().long()
    tps = list(dataset.rollout_timepoints)
    if len(tps) != 5:
        raise ValueError(f"Expected five rollout timepoints, got {tps}")
    z_per_hop = [dataset.latents[tps[hop]][idx].clone() for hop in range(4)]
    x_hop0 = dataset.images["D50"][idx].clone() if "D50" in dataset.images else None
    return z_per_hop, x_hop0, [int(i) for i in idx.tolist()]


@torch.no_grad()
def estimate_lipschitz(
    model: PETFlowDiTFirstHop,
    cfg: Dict,
    dataset: PETFirstHopAligned4HopDataset,
    n_samples: int,
    eps: float,
    device: torch.device,
    seed: int,
) -> tuple[list[float], dict[str, dict[str, float]], list[int]]:
    z_per_hop, x_hop0, slice_indices = sample_hop_inputs(dataset, n_samples=n_samples)
    tps = list(dataset.rollout_timepoints)
    t_map = cfg["data"]["t_map"]
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    l_means: list[float] = []
    per_hop_stats: dict[str, dict[str, float]] = {}

    for hop in range(4):
        z = z_per_hop[hop].to(device, non_blocking=True)
        delta = torch.randn(z.shape, generator=generator, dtype=z.dtype) * float(eps)
        delta = delta.to(device, non_blocking=True)
        bsz = int(z.shape[0])
        t_src = torch.full((bsz,), float(t_map[tps[hop]]), device=device)
        t_dst = torch.full((bsz,), float(t_map[tps[hop + 1]]), device=device)
        hop_idx = torch.full((bsz,), hop, device=device, dtype=torch.long)
        x_src = x_hop0.to(device, non_blocking=True) if hop == 0 and x_hop0 is not None else None

        out0 = model.predict_latent_step(
            z_src=z,
            t_src=t_src,
            t_dst=t_dst,
            hop_idx=hop_idx,
            x_src_img=x_src,
        )
        out1 = model.predict_latent_step(
            z_src=z + delta,
            t_src=t_src,
            t_dst=t_dst,
            hop_idx=hop_idx,
            x_src_img=x_src,
        )
        f0 = out0["z_pred"] if isinstance(out0, dict) else out0
        f1 = out1["z_pred"] if isinstance(out1, dict) else out1
        ratios = (f1 - f0).flatten(1).norm(dim=1) / delta.flatten(1).norm(dim=1).clamp_min(1e-12)
        stats = summarize(ratios.detach().cpu())
        l_means.append(float(stats["mean"]))
        per_hop_stats[f"hop{hop}_{tps[hop]}_to_{tps[hop + 1]}"] = stats

    return l_means, per_hop_stats, slice_indices


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ----- read β from best_metric_terms -----
    terms = cfg["training"]["best_metric_terms"]
    name_to_w = {t["name"]: float(t["weight"]) for t in terms}
    beta = [
        name_to_w["val_chain_d20_mse"],
        name_to_w["val_chain_d10_mse"],
        name_to_w["val_chain_d4_mse"],
        name_to_w["val_chain_normal_mse"],
    ]

    # ----- read current step_weights -----
    current_sw = list(cfg["training"]["rollout"]["step_weights"])
    sigma_dt = load_v7_sigma_dt()

    device = torch.device(args.device)
    dataset = build_dataset(cfg, args.split)
    model, ckpt_meta = load_model(cfg, args.checkpoint, device)
    L_per_hop, L_stats, slice_indices = estimate_lipschitz(
        model=model,
        cfg=cfg,
        dataset=dataset,
        n_samples=int(args.n_samples),
        eps=float(args.eps),
        device=device,
        seed=int(args.seed),
    )

    # ----- closed-form with measured L -----
    w_closed = closed_form_weights(beta, L_per_hop, sigma_dt)

    # ----- alignment to current step_weights -----
    def normalize(v):
        s = sum(v)
        return [x / s for x in v]

    dist_pct = (
        sum(abs(a - b) for a, b in zip(normalize(w_closed), normalize(current_sw)))
        / 2.0
        * 100.0
    )  # total-variation-like distance over normalized weight vectors, in %

    payload = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "n_samples": args.n_samples,
        "actual_n_samples": len(slice_indices),
        "eps": args.eps,
        "split": args.split,
        "seed": args.seed,
        "beta": beta,
        "sigma_dt": sigma_dt,
        "L_per_hop": L_per_hop,
        "L_per_hop_stats": L_stats,
        "deviation_pct": [float((l - 1.0) * 100.0) for l in L_per_hop],
        "closed_form_w_with_measured_L": w_closed,
        "current_step_weights": current_sw,
        "alignment_distance_pct": dist_pct,
        "sample_slice_indices": slice_indices,
        "checkpoint_step": ckpt_meta.get("step") if isinstance(ckpt_meta, dict) else None,
        "checkpoint_best_val": ckpt_meta.get("best_val") if isinstance(ckpt_meta, dict) else None,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
