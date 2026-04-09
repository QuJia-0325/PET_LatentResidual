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
from tqdm import tqdm

# Reuse canonical RAE + metrics implementation from the RAE repo.
RAE_CODE_ROOT = "/home/qujiaxiang/project/RAE/code/RAE"
if RAE_CODE_ROOT not in sys.path:
    sys.path.insert(0, RAE_CODE_ROOT)

from src.stage1.rae import RAE
from src.utils.lora import inject_lora_into_dinov2_attention
from src.utils.metrics import calc_psnr_clip3


TIMEPOINTS = ["D50", "D20", "D10", "D4", "NORMAL"]
LORA_KEYWORDS = [
    "attention.qkv",
    "attention.projection",
    "attention.attention.query",
    "attention.attention.key",
    "attention.attention.value",
    "attention.output.dense",
    ".mlp.fc1",
    ".mlp.fc2",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Audit latent<->pixel alignment at 224 using canonical calc_psnr_clip3. "
            "Outputs reviewable JSON/CSV."
        )
    )
    p.add_argument(
        "--latents-dir",
        default="/data_2/qujiaxiang/lowdose_pet_ct/latents_224",
        help="Directory containing latents_train.pt and latents_val.pt",
    )
    p.add_argument(
        "--raw-data-dir",
        default="/data_2/qujiaxiang",
        help="Directory containing preprocessed_data_*.pt",
    )
    p.add_argument(
        "--rae-ckpt",
        default="/data_2/qujiaxiang/outputs/pet_lora_dinov2_pt_224/best_model.pt",
        help="Stage-1 224 RAE checkpoint",
    )
    p.add_argument(
        "--out-dir",
        default="/data_2/qujiaxiang/outputs/PET_LatentResidual/alignment_224_audit",
        help="Output directory for JSON/CSV reports",
    )
    p.add_argument(
        "--sample-per-split",
        type=int,
        default=512,
        help="Number of slices per split/timepoint for decode-based PSNR checks (<=0 means full split)",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--device",
        default="cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clamp-max", type=float, default=10.0)
    return p.parse_args()


def normalize_image(x: torch.Tensor, clamp_max: float) -> torch.Tensor:
    x = x.clamp(0, clamp_max)
    return (x / (clamp_max / 2.0)) - 1.0


def load_images(split: str, raw_data_dir: str, clamp_max: float) -> Dict[str, torch.Tensor]:
    imgs: Dict[str, torch.Tensor] = {}
    for tp in ["D50", "D20", "D10", "D4"]:
        blob = torch.load(os.path.join(raw_data_dir, f"preprocessed_data_{tp}.pt"), map_location="cpu")
        x = normalize_image(blob[split]["x_T"], clamp_max).float()
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
        imgs[tp] = x

    normal_blob = torch.load(os.path.join(raw_data_dir, "preprocessed_data_D50.pt"), map_location="cpu")
    x0 = normalize_image(normal_blob[split]["x_0"], clamp_max).float()
    x0 = F.interpolate(x0, size=(224, 224), mode="bicubic", align_corners=False)
    imgs["NORMAL"] = x0
    return imgs


def load_rae(ckpt_path: str, device: str) -> RAE:
    rae = RAE(
        encoder_cls="Dinov2withNorm",
        encoder_config_path="/home/qujiaxiang/project/RAE/dinov2",
        encoder_input_size=224,
        encoder_params={"dinov2_path": "/home/qujiaxiang/project/RAE/dinov2", "normalize": True},
        encoder_mean=[0.0, 0.0, 0.0],
        encoder_std=[1.0, 1.0, 1.0],
        decoder_config_path="/home/qujiaxiang/project/RAE/vit-mae",
        decoder_patch_size=14,
        pretrained_decoder_path=None,
        noise_tau=0.0,
        reshape_to_2d=True,
    )
    inject_lora_into_dinov2_attention(
        rae.encoder.encoder,
        rank=16,
        alpha=16,
        dropout=0.0,
        target_keywords=LORA_KEYWORDS,
    )
    state = torch.load(ckpt_path, map_location="cpu")["model"]
    rae.load_state_dict(state, strict=False)
    return rae.to(device).eval()


def select_indices(n: int, sample_n: int) -> torch.Tensor:
    if sample_n <= 0 or sample_n >= n:
        return torch.arange(n, dtype=torch.long)
    idx = torch.linspace(0, n - 1, steps=sample_n).round().long()
    idx = torch.unique(idx, sorted=True)
    if idx.numel() < sample_n:
        # Fill missing slots deterministically.
        missing = sample_n - idx.numel()
        extra = torch.arange(n, dtype=torch.long)[:missing]
        idx = torch.unique(torch.cat([idx, extra]), sorted=True)
    return idx


def zscore(x: torch.Tensor) -> torch.Tensor:
    mu = x.mean()
    std = x.std(unbiased=False).clamp_min(1e-8)
    return (x - mu) / std


def corr(a: torch.Tensor, b: torch.Tensor) -> float:
    az = zscore(a)
    bz = zscore(b)
    return float((az * bz).mean().item())


def summarize(xs: List[float]) -> Dict[str, float]:
    arr = np.asarray(xs, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size > 0 else float("nan"),
        "std": float(arr.std()) if arr.size > 0 else float("nan"),
        "min": float(arr.min()) if arr.size > 0 else float("nan"),
        "max": float(arr.max()) if arr.size > 0 else float("nan"),
    }


@torch.no_grad()
def decode_latents(rae: RAE, z: torch.Tensor, device: str, batch_size: int) -> torch.Tensor:
    outs: List[torch.Tensor] = []
    for i in tqdm(range(0, z.shape[0], batch_size), desc="decode", leave=False):
        zb = z[i : i + batch_size].to(device)
        xb = rae.decode(zb)[:, 0:1].detach().cpu()
        outs.append(xb)
    return torch.cat(outs, dim=0)


def psnr_list(pred: torch.Tensor, gt: torch.Tensor) -> List[float]:
    out: List[float] = []
    for i in range(pred.shape[0]):
        out.append(float(calc_psnr_clip3(pred[i : i + 1], gt[i : i + 1])))
    return out


def audit_split(
    split: str,
    latents_dir: str,
    raw_data_dir: str,
    rae: RAE,
    device: str,
    sample_n: int,
    batch_size: int,
    clamp_max: float,
    seed: int,
) -> Dict[str, Dict[str, object]]:
    lat_path = os.path.join(latents_dir, f"latents_{split}.pt")
    latents: Dict[str, torch.Tensor] = torch.load(lat_path, map_location="cpu")
    imgs = load_images(split=split, raw_data_dir=raw_data_dir, clamp_max=clamp_max)

    results: Dict[str, Dict[str, object]] = {}
    g = torch.Generator().manual_seed(seed)

    for tp in TIMEPOINTS:
        z = latents[tp].float()
        x = imgs[tp].float()
        if z.shape[0] != x.shape[0]:
            raise RuntimeError(f"Slice count mismatch at {split}/{tp}: latent={z.shape[0]}, image={x.shape[0]}")

        n_total = int(z.shape[0])
        idx = select_indices(n_total, sample_n)
        z_idx = z[idx]
        x_idx = x[idx]
        n_eval = int(idx.numel())

        # Scalar correlation sanity.
        z_scalar = z_idx.mean(dim=(1, 2, 3))
        x_scalar = x_idx.mean(dim=(1, 2, 3))
        x_scalar_roll = x_scalar.roll(1, dims=0)
        perm = torch.randperm(n_eval, generator=g)
        x_scalar_rand = x_scalar[perm]

        scalar_stats = {
            "corr_matched": corr(z_scalar, x_scalar),
            "corr_roll1": corr(z_scalar, x_scalar_roll),
            "corr_random": corr(z_scalar, x_scalar_rand),
        }

        # Decode-based alignment check with required PSNR metric.
        pred = decode_latents(rae=rae, z=z_idx, device=device, batch_size=batch_size)
        gt_matched = x_idx
        gt_roll = gt_matched.roll(1, dims=0)
        gt_rand = gt_matched[perm]

        psnr_matched = psnr_list(pred, gt_matched)
        psnr_roll = psnr_list(pred, gt_roll)
        psnr_rand = psnr_list(pred, gt_rand)

        psnr_stats = {
            "matched": summarize(psnr_matched),
            "roll1": summarize(psnr_roll),
            "random": summarize(psnr_rand),
        }

        gap_roll = psnr_stats["matched"]["mean"] - psnr_stats["roll1"]["mean"]
        gap_rand = psnr_stats["matched"]["mean"] - psnr_stats["random"]["mean"]

        results[tp] = {
            "split": split,
            "timepoint": tp,
            "n_total": n_total,
            "n_eval": n_eval,
            "scalar": scalar_stats,
            "psnr_clip3": psnr_stats,
            "psnr_gap_vs_roll1_db": float(gap_roll),
            "psnr_gap_vs_random_db": float(gap_rand),
            "alignment_signal_positive": bool(gap_roll > 0.0 and gap_rand > 0.0),
            "psnr_metric": "src.utils.metrics.calc_psnr_clip3",
        }

    return results


def flatten_rows(all_results: Dict[str, Dict[str, Dict[str, object]]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for split, split_res in all_results.items():
        for tp, r in split_res.items():
            row = {
                "split": split,
                "timepoint": tp,
                "n_total": r["n_total"],
                "n_eval": r["n_eval"],
                "psnr_metric": r["psnr_metric"],
                "corr_matched": r["scalar"]["corr_matched"],
                "corr_roll1": r["scalar"]["corr_roll1"],
                "corr_random": r["scalar"]["corr_random"],
                "psnr_matched_mean": r["psnr_clip3"]["matched"]["mean"],
                "psnr_matched_std": r["psnr_clip3"]["matched"]["std"],
                "psnr_roll1_mean": r["psnr_clip3"]["roll1"]["mean"],
                "psnr_roll1_std": r["psnr_clip3"]["roll1"]["std"],
                "psnr_random_mean": r["psnr_clip3"]["random"]["mean"],
                "psnr_random_std": r["psnr_clip3"]["random"]["std"],
                "psnr_gap_vs_roll1_db": r["psnr_gap_vs_roll1_db"],
                "psnr_gap_vs_random_db": r["psnr_gap_vs_random_db"],
                "alignment_signal_positive": r["alignment_signal_positive"],
            }
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rae = load_rae(args.rae_ckpt, args.device)
    all_results: Dict[str, Dict[str, Dict[str, object]]] = {}

    for split in ["train", "val"]:
        print(f"[audit] split={split}")
        all_results[split] = audit_split(
            split=split,
            latents_dir=args.latents_dir,
            raw_data_dir=args.raw_data_dir,
            rae=rae,
            device=args.device,
            sample_n=args.sample_per_split,
            batch_size=args.batch_size,
            clamp_max=args.clamp_max,
            seed=args.seed,
        )

    rows = flatten_rows(all_results)
    json_path = out_dir / "alignment_224_clip3_audit.json"
    csv_path = out_dir / "alignment_224_clip3_audit.csv"

    payload = {
        "meta": {
            "psnr_metric": "src.utils.metrics.calc_psnr_clip3",
            "latents_dir": args.latents_dir,
            "raw_data_dir": args.raw_data_dir,
            "rae_ckpt": args.rae_ckpt,
            "device": args.device,
            "sample_per_split": args.sample_per_split,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "timepoints": TIMEPOINTS,
        },
        "results": all_results,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(payload["meta"], indent=2))
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
