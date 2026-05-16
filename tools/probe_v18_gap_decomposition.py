#!/usr/bin/env python3
"""V18 gap-decomposition probe.

Round 4 peer review (agent1 + agent2 + agent3) all independently demanded
that we decompose the "11.2 dB transport gap" into:

  (a) decoder reconstruction ceiling on GT latent:
        psnr_ceil      = PSNR(decode(z_GT_D20),  x_D20_gt)

  (b) actual transport-then-decode performance:
        psnr_transport = PSNR(decode(z_pred^V7), x_D20_gt)

  (c) latent-space transport error:
        latent_l2_rel  = ||z_pred^V7 - z_GT||_2 / ||z_GT||_2

The actually-attackable gap for V18 (which only modifies the decoder) is
  attackable_gap = psnr_ceil - psnr_transport
which is at most 11.2 dB but could easily be <1 dB.

This probe runs ~30-60 min on 1 GPU, n=7403 (full val), and writes:
  review/0517/V18_decoder_lora/GAP_DECOMP_REPORT.md  (claude-readable summary)
  review/0517/V18_decoder_lora/GAP_DECOMP_PER_SLICE.csv  (per-slice raw)
  review/0517/V18_decoder_lora/GAP_DECOMP_SUMMARY.json   (machine-readable)

Pre-registered decision rule (from Round 4 agent3 + claude integration):

| attackable_gap (D20) | action |
|---|---|
| < 2 dB | V18 wrong direction → launch V21 (conv head on frozen V7) instead |
| 2-5 dB | launch V18 with rank=8, KL fix, success threshold downgraded to +0.20 dB |
| >= 5 dB | launch V18 with rank=32, KL fix, original success threshold +0.30 dB |

Usage on GPU server:
  cd /home/qujiaxiang/project/PET_LatentResidual
  git pull --ff-only gitee foc_lite_hop0
  CUDA_VISIBLE_DEVICES=0 python tools/probe_v18_gap_decomposition.py \
    --config review/0511/log_snapshots_20260516_163900/configs/V7_config.resolved.yaml \
    --checkpoint /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V7/run/first_hop_224_v7_gronwall_raw/best.pt \
    --out-dir review/0517/V18_decoder_lora \
    --batch-size 8 --max-slices 0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pet_lr.data_first_hop import Hop0OnlyViewDataset, PETFirstHopAligned4HopDataset
from pet_lr.model_first_hop import PETFlowDiTFirstHop
from pet_lr.path_guard import ensure_repo_local_outputs_absent, resolve_data_disk_dir
from pet_lr.rollout_first_hop import sample_chain_first_hop

# Canonical PSNR (matches eval_first_hop_224_clip3.py exactly)
RAE_CODE_ROOT = "/home/qujiaxiang/project/RAE/code/RAE"
if RAE_CODE_ROOT not in sys.path:
    sys.path.insert(0, RAE_CODE_ROOT)
from src.utils.metrics import calc_psnr_clip3  # noqa: E402

TARGET_HOP_IDX = 1  # D20 is rollout_timepoints[1] in V7 (D50, D20, D10, D4, NORMAL)
TARGET_STAGE_NAME = "D20"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-slices", type=int, default=0, help="0 = all val slices")
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_2d(x: torch.Tensor) -> torch.Tensor:
    """Collapse channel dim to 1 if needed (PET is single-channel)."""
    if x.dim() == 4 and x.shape[1] > 1:
        x = x[:, 0:1]
    return x


def build_dataset(cfg: Dict, split: str) -> Hop0OnlyViewDataset:
    """Build a full-rollout val dataset with one sample per slice.

    PETFirstHopAligned4HopDataset indexes all four training pairs by default
    (`4 * num_slices`). The gap decomposition is a D50->D20 diagnostic, so the
    evaluation unit must be the hop0 slice, not all four pair rows.
    """
    data_cfg = cfg["data"]
    latent_path = os.path.join(data_cfg["latent_dir"], f"latents_{split}.pt")
    alignment_audit_json = str(data_cfg.get("alignment_audit_json", "")).strip() or None
    base = PETFirstHopAligned4HopDataset(
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
        include_full_x_rollout=True,
        image_size=int(data_cfg.get("image_size", 224)),
        latent_mmap=bool(data_cfg.get("latent_mmap", False)),
    )
    return Hop0OnlyViewDataset(base)


def maybe_subset_dataset(ds: Hop0OnlyViewDataset, max_slices: int):
    if max_slices <= 0 or max_slices >= len(ds):
        return ds
    idx = torch.linspace(0, len(ds) - 1, steps=max_slices).round().long()
    idx = torch.unique(idx, sorted=True).tolist()
    return torch.utils.data.Subset(ds, idx)


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Path-guard (mimics eval_first_hop_224_clip3.py)
    repo_root = Path(__file__).resolve().parent.parent
    ensure_repo_local_outputs_absent(repo_root)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[probe] device={device}, config={args.config}, ckpt={args.checkpoint}")

    # ---- build model + load V7 best.pt ----
    model = PETFlowDiTFirstHop(cfg, device).to(device)
    sd = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd["model"], strict=False)
    print(f"[probe] state_dict load: missing={len(missing)}, unexpected={len(unexpected)}")
    if unexpected:
        print(f"[probe] WARN unexpected keys (first 5): {unexpected[:5]}")
    model.eval()

    # ---- sanity check: confirm decoder Linear is NOT already LoRA-wrapped ----
    try:
        last_block = model.rae.decoder.decoder_layers[-1]
        q = last_block.attention.attention.query
        q_type = type(q).__name__
        print(f"[probe] SANITY: type(rae.decoder.decoder_layers[-1].attention.attention.query) = {q_type}")
        if q_type != "Linear":
            print(
                f"[probe] !!! WARN: decoder Linear is already wrapped as {q_type}. "
                f"V18 wrap-order bug risk; needs merge_lora() step before V18 LoRA. "
                f"DO NOT launch V18 until pet_lr/decoder_lora.py is updated."
            )
    except AttributeError as e:
        print(f"[probe] !!! FATAL: cannot find rae.decoder.decoder_layers[-1].*.query: {e}")
        print(f"[probe] !!! Update V18 yaml target_root to the real path before launch.")
        sys.exit(1)

    # ---- key audit: check if V7 best.pt has any decoder LoRA params ----
    decoder_lora_keys = [k for k in sd["model"].keys() if "decoder" in k and ("lora_A" in k or "lora_B" in k)]
    print(f"[probe] KEY AUDIT: V7 best.pt contains {len(decoder_lora_keys)} decoder LoRA keys")
    if decoder_lora_keys:
        print(f"[probe] !!! WARN: V7 ckpt has decoder LoRA: {decoder_lora_keys[:5]}")
        print(f"[probe] !!! Stage 2 may have used decoder LoRA. V18 wrap logic needs revision.")

    # ---- build dataset & loader ----
    ds = maybe_subset_dataset(build_dataset(cfg, args.split), args.max_slices)
    print(f"[probe] dataset {args.split}: n={len(ds)} samples")

    # We only need hop-0 samples for D20 GT pair AND access to z_rollout[1] for z_GT_D20.
    # Use the full dataset; per-slice averaging at the end.
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, drop_last=False,
    )

    rollout_times = [float(cfg["data"]["t_map"][tp]) for tp in cfg["data"]["rollout_timepoints"]]
    image_size = int(cfg["data"].get("image_size", 224))

    rows = []
    n_skipped = 0
    t0 = time.time()
    with torch.no_grad():
        for batch in tqdm(loader, desc="probe", disable=not sys.stderr.isatty()):
            slice_idx = batch["slice_idx"].cpu().numpy()
            hop_idx_per_sample = batch["hop_idx"].cpu().numpy()

            # We need z_rollout (GT latents) and x_rollout (GT images) at hop position 1 = D20
            if "z_rollout" not in batch or "x_rollout" not in batch:
                n_skipped += len(slice_idx)
                continue

            z_rollout = batch["z_rollout"].to(device)  # [B, T, C, H, W]
            x_rollout = batch["x_rollout"].to(device)  # [B, T, 1, H, W]

            z_GT_D20 = z_rollout[:, TARGET_HOP_IDX]    # [B, C, H, W]
            x_GT_D20 = x_rollout[:, TARGET_HOP_IDX]    # [B, 1, H, W]
            x_D50 = x_rollout[:, 0]                    # [B, 1, H, W]
            z_D50 = z_rollout[:, 0]                    # [B, C, H, W]
            x_D50 = _to_2d(x_D50)
            x_GT_D20 = _to_2d(x_GT_D20)

            # ---- (a) decoder ceiling: decode GT latent, compare to GT image
            x_decoded_from_GT = model.decode_crop(z_GT_D20, crop_size=image_size)
            x_decoded_from_GT = _to_2d(x_decoded_from_GT)

            # ---- (b) V7 chain rollout: D50 -> D20
            # sample_chain_first_hop returns list of latents [z_D50, z_D20_pred, z_D10_pred, ...]
            chain_preds = sample_chain_first_hop(
                model=model,
                z_d50=z_D50,
                x_d50=x_D50,
                rollout_times=rollout_times,
            )
            z_pred_D20 = chain_preds[TARGET_HOP_IDX]   # [B, C, H, W]
            x_decoded_from_pred = model.decode_crop(z_pred_D20, crop_size=image_size)
            x_decoded_from_pred = _to_2d(x_decoded_from_pred)

            # ---- per-slice metrics
            for i in range(z_GT_D20.shape[0]):
                psnr_ceil = float(calc_psnr_clip3(x_decoded_from_GT[i:i+1], x_GT_D20[i:i+1]).item())
                psnr_transport = float(calc_psnr_clip3(x_decoded_from_pred[i:i+1], x_GT_D20[i:i+1]).item())
                latent_diff = (z_pred_D20[i] - z_GT_D20[i]).float()
                latent_gt = z_GT_D20[i].float()
                l2_diff = float(latent_diff.norm().item())
                l2_gt = float(latent_gt.norm().item())
                latent_l2_rel = l2_diff / max(l2_gt, 1e-12)
                # Also compute decoder-only error on GT-decoded vs V7-decoded (decoder-invariant proxy)
                psnr_transport_vs_ceil = float(
                    calc_psnr_clip3(x_decoded_from_pred[i:i+1], x_decoded_from_GT[i:i+1]).item()
                )
                rows.append({
                    "slice_idx": int(slice_idx[i]),
                    "hop_idx_in_batch": int(hop_idx_per_sample[i]),
                    "psnr_ceil": psnr_ceil,
                    "psnr_transport": psnr_transport,
                    "attackable_gap_dB": psnr_ceil - psnr_transport,
                    "psnr_transport_vs_ceil": psnr_transport_vs_ceil,
                    "latent_l2_rel": latent_l2_rel,
                })

    elapsed = time.time() - t0
    print(f"[probe] processed {len(rows)} slices in {elapsed:.1f}s (skipped {n_skipped})")

    # ---- write per-slice CSV ----
    csv_path = out_dir / "GAP_DECOMP_PER_SLICE.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[probe] wrote {csv_path}")

    # ---- compute summary stats ----
    def stats(key):
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        return {
            "n": int(len(vals)),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "p05": float(np.percentile(vals, 5)),
            "p95": float(np.percentile(vals, 95)),
        }

    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "n_slices": len(rows),
        "target_stage": TARGET_STAGE_NAME,
        "elapsed_sec": elapsed,
        "psnr_ceil": stats("psnr_ceil"),
        "psnr_transport": stats("psnr_transport"),
        "attackable_gap_dB": stats("attackable_gap_dB"),
        "psnr_transport_vs_ceil": stats("psnr_transport_vs_ceil"),
        "latent_l2_rel": stats("latent_l2_rel"),
    }
    json_path = out_dir / "GAP_DECOMP_SUMMARY.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[probe] wrote {json_path}")

    # ---- claude-readable decision report ----
    mean_gap = summary["attackable_gap_dB"]["mean"]
    median_gap = summary["attackable_gap_dB"]["median"]

    if mean_gap < 2.0:
        decision = "V18_WRONG_DIRECTION_LAUNCH_V21"
        explanation = (
            f"attackable_gap mean = {mean_gap:.3f} dB < 2 dB. V18 (which only modifies the decoder)\n"
            f"cannot recover more than ~{mean_gap:.2f} dB even with perfect decoder LoRA. The 11.2 dB\n"
            f"original gap is overwhelmingly TRANSPORT error (z_pred vs z_GT), not decoder error.\n"
            f"\n"
            f"Recommended action: launch V21 (conv head on frozen V7 output) instead.\n"
            f"V21 design: freeze V7 entirely + train ConvDecoderHead on top of decode(z_pred^V7).\n"
            f"RAE already has src/stage1/decoders/conv_head.py. Code cost ~1 day; train ~2 days.\n"
        )
    elif mean_gap >= 5.0:
        decision = "V18_LAUNCH_RANK32_KLFIX"
        explanation = (
            f"attackable_gap mean = {mean_gap:.3f} dB >= 5 dB. V18 has real runway.\n"
            f"\n"
            f"Recommended: launch V18 with rank=32 (not rank=8 — decoder is already PET-full-tuned,\n"
            f"need bigger adapter), lambda_kl=0.05 (not 0.5 — see Round 4 agent1 calibration bug),\n"
            f"use_pred_latent=true for KL (focus regularization on predicted-path drift).\n"
            f"Original success threshold +0.30 dB still applies.\n"
        )
    else:
        decision = "V18_LAUNCH_RANK8_KLFIX_DOWNGRADE_THRESHOLD"
        explanation = (
            f"attackable_gap mean = {mean_gap:.3f} dB in [2, 5) dB. Marginal V18 case.\n"
            f"\n"
            f"Recommended: launch V18 with rank=8 (cheap probe), lambda_kl=0.05, use_pred_latent=true.\n"
            f"Downgrade primary success threshold to +0.20 dB (not +0.30 dB).\n"
            f"If V18 hits +0.20 dB, consider V18-rank32 follow-up. If not, pivot to V21.\n"
        )

    report = f"""# V18 Gap Decomposition Report

- generated_at: {time.strftime("%Y-%m-%d %H:%M:%S %Z")}
- config: {args.config}
- checkpoint: {args.checkpoint}
- n_slices: {summary['n_slices']}
- target stage: {TARGET_STAGE_NAME}
- elapsed: {elapsed:.1f}s

## Sanity Checks

- Decoder Linear type at last block (must be `Linear`, not `LinearWithLoRA`):
  see `[probe] SANITY:` line in stdout above.
- V7 best.pt decoder LoRA keys (must be empty):
  see `[probe] KEY AUDIT:` line in stdout above.

## Primary Numbers (n={summary['n_slices']})

| metric | mean | median | std | min | max | p05 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `psnr_ceil` PSNR(decode(z_GT_D20), x_D20) | {summary['psnr_ceil']['mean']:.4f} | {summary['psnr_ceil']['median']:.4f} | {summary['psnr_ceil']['std']:.4f} | {summary['psnr_ceil']['min']:.4f} | {summary['psnr_ceil']['max']:.4f} | {summary['psnr_ceil']['p05']:.4f} | {summary['psnr_ceil']['p95']:.4f} |
| `psnr_transport` PSNR(decode(z_pred^V7), x_D20) | {summary['psnr_transport']['mean']:.4f} | {summary['psnr_transport']['median']:.4f} | {summary['psnr_transport']['std']:.4f} | {summary['psnr_transport']['min']:.4f} | {summary['psnr_transport']['max']:.4f} | {summary['psnr_transport']['p05']:.4f} | {summary['psnr_transport']['p95']:.4f} |
| **attackable_gap_dB** | **{summary['attackable_gap_dB']['mean']:.4f}** | **{summary['attackable_gap_dB']['median']:.4f}** | {summary['attackable_gap_dB']['std']:.4f} | {summary['attackable_gap_dB']['min']:.4f} | {summary['attackable_gap_dB']['max']:.4f} | {summary['attackable_gap_dB']['p05']:.4f} | {summary['attackable_gap_dB']['p95']:.4f} |
| `psnr_transport_vs_ceil` PSNR(decode(z_pred^V7), decode(z_GT_D20)) | {summary['psnr_transport_vs_ceil']['mean']:.4f} | {summary['psnr_transport_vs_ceil']['median']:.4f} | {summary['psnr_transport_vs_ceil']['std']:.4f} | {summary['psnr_transport_vs_ceil']['min']:.4f} | {summary['psnr_transport_vs_ceil']['max']:.4f} | {summary['psnr_transport_vs_ceil']['p05']:.4f} | {summary['psnr_transport_vs_ceil']['p95']:.4f} |
| `latent_l2_rel` ‖z_pred − z_GT‖₂ / ‖z_GT‖₂ | {summary['latent_l2_rel']['mean']:.4f} | {summary['latent_l2_rel']['median']:.4f} | {summary['latent_l2_rel']['std']:.4f} | {summary['latent_l2_rel']['min']:.4f} | {summary['latent_l2_rel']['max']:.4f} | {summary['latent_l2_rel']['p05']:.4f} | {summary['latent_l2_rel']['p95']:.4f} |

## Pre-Registered Decision Rule (Round 4 agent3 + integration)

| attackable_gap (mean) | action |
|---|---|
| < 2 dB | V18 wrong direction → launch V21 (conv head on frozen V7) |
| 2-5 dB | V18 rank=8 + KL fix + downgrade success threshold to +0.20 dB |
| ≥ 5 dB | V18 rank=32 + KL fix + original success threshold +0.30 dB |

## Verdict

**Decision: `{decision}`**

{explanation}

## Next Action

See `review/0517/CODEX_RUNBOOK_V18_20260517.md` §4 Day 1 implementation steps.
The pre-registered decision above replaces the launch-vs-no-launch question
that was open in Round 3.
"""

    report_path = out_dir / "GAP_DECOMP_REPORT.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"[probe] wrote {report_path}")
    print(f"[probe] DECISION: {decision}")
    print(f"[probe] attackable_gap mean = {mean_gap:.3f} dB, median = {median_gap:.3f} dB")


if __name__ == "__main__":
    main()
