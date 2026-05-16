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
from pathlib import Path
from typing import List

import torch
import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--device", default="cuda:0")
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

    # ----- TODO: build model + load checkpoint -----
    # from pet_lr.model_first_hop import PETFlowDiTFirstHop
    # model = PETFlowDiTFirstHop(**cfg["model"]).to(args.device)
    # sd = torch.load(args.checkpoint, map_location="cpu")
    # model.load_state_dict(sd["model"], strict=True)
    # model.eval()

    # ----- TODO: sample z_at_hop[k] from val loader -----
    # from pet_lr.data_first_hop import PETFirstHopAligned4HopDataset
    # ds = PETFirstHopAligned4HopDataset(...)
    # loader = torch.utils.data.DataLoader(ds, batch_size=args.n_samples, shuffle=True)
    # batch = next(iter(loader))  # contains z_D50, z_D20, z_D10, z_D4 (the chain inputs)
    # z_per_hop = [batch["z_D50"], batch["z_D20"], batch["z_D10"], batch["z_D4"]]

    # ----- TODO: per-hop Lipschitz estimate -----
    # L_per_hop = []
    # with torch.no_grad():
    #     for k in range(4):
    #         z = z_per_hop[k].to(args.device)
    #         delta = torch.randn_like(z) * args.eps
    #         # NOTE: confirm the actual PETFlowDiTFirstHop call signature; this is a sketch.
    #         out0 = model.predict_latent_step(z, hop_idx=k)
    #         out1 = model.predict_latent_step(z + delta, hop_idx=k)
    #         F0 = out0["z_pred"] if isinstance(out0, dict) else out0
    #         F1 = out1["z_pred"] if isinstance(out1, dict) else out1
    #         num = (F1 - F0).flatten(1).norm(dim=1)
    #         den = delta.flatten(1).norm(dim=1)
    #         L_k = (num / den.clamp_min(1e-12)).mean().item()
    #         L_per_hop.append(L_k)

    # ----- STUB to keep the script runnable on a laptop (delete on server) -----
    L_per_hop = [1.00, 1.00, 1.00, 1.00]
    note = (
        "STUB output: L_per_hop hard-coded to 1.0 because model/data wiring is TODO. "
        "When wired on the GPU server, the values come from the actual model. The "
        "rest of the math below (closed-form, alignment) reacts correctly to any L."
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
        "eps": args.eps,
        "beta": beta,
        "sigma_dt": sigma_dt,
        "L_per_hop": L_per_hop,
        "closed_form_w_with_measured_L": w_closed,
        "current_step_weights": current_sw,
        "alignment_distance_pct": dist_pct,
        "stub_note": note,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
