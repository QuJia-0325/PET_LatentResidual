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

PET_ROOT = "/home/qujiaxiang/project/PET_LatentResidual"
RAE_CODE_ROOT = "/home/qujiaxiang/project/RAE/code/RAE"
if PET_ROOT not in sys.path:
    sys.path.insert(0, PET_ROOT)
if RAE_CODE_ROOT not in sys.path:
    sys.path.insert(0, RAE_CODE_ROOT)

from src.stage1.rae import RAE  # noqa: E402
from src.utils.lora import inject_lora_into_dinov2_attention  # noqa: E402
from src.utils.metrics import calc_psnr_clip3  # noqa: E402


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
        description="Evaluate GT latent decode ceiling on val split with calc_psnr_clip3."
    )
    p.add_argument(
        "--latents-path",
        default="/data_2/qujiaxiang/lowdose_pet_ct/latents_224/latents_val.pt",
        help="Path to latents_val.pt",
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
        "--split",
        default="val",
        choices=["train", "val"],
        help="Split to evaluate (default val)",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Directory to save summary JSON + per-slice CSV",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--clamp-max",
        type=float,
        default=10.0,
        help="Raw PET clamp max before normalization to [-1, 1]",
    )
    return p.parse_args()


def normalize_image(x: torch.Tensor, clamp_max: float) -> torch.Tensor:
    x = x.clamp(0, clamp_max)
    return (x / (clamp_max / 2.0)) - 1.0


def load_images(split: str, raw_data_dir: str, clamp_max: float) -> Dict[str, torch.Tensor]:
    images: Dict[str, torch.Tensor] = {}
    for tp in ["D50", "D20", "D10", "D4"]:
        blob = torch.load(os.path.join(raw_data_dir, f"preprocessed_data_{tp}.pt"), map_location="cpu")
        x = normalize_image(blob[split]["x_T"], clamp_max=clamp_max).float()
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
        images[tp] = x

    normal_blob = torch.load(os.path.join(raw_data_dir, "preprocessed_data_D50.pt"), map_location="cpu")
    x0 = normalize_image(normal_blob[split]["x_0"], clamp_max=clamp_max).float()
    x0 = F.interpolate(x0, size=(224, 224), mode="bicubic", align_corners=False)
    images["NORMAL"] = x0
    return images


def load_rae(ckpt_path: str, device: torch.device) -> RAE:
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


@torch.no_grad()
def decode_latents(rae: RAE, z: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    preds: List[torch.Tensor] = []
    for i in tqdm(range(0, z.shape[0], batch_size), desc="decode", leave=False):
        zb = z[i : i + batch_size].to(device)
        xb = rae.decode(zb)[:, 0:1].detach().cpu()
        preds.append(xb)
    return torch.cat(preds, dim=0)


def summarize(vals: List[float]) -> Dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": float(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    rae = load_rae(args.rae_ckpt, device=device)
    latents = torch.load(args.latents_path, map_location="cpu")
    images = load_images(split=args.split, raw_data_dir=args.raw_data_dir, clamp_max=float(args.clamp_max))

    summary: Dict[str, Dict[str, float]] = {}
    rows: List[Dict[str, float]] = []

    for tp in TIMEPOINTS:
        z = latents[tp].float()
        x = images[tp].float()
        if z.shape[0] != x.shape[0]:
            raise RuntimeError(f"Slice count mismatch for {tp}: latent={z.shape[0]}, image={x.shape[0]}")

        pred = decode_latents(rae=rae, z=z, batch_size=int(args.batch_size), device=device)
        vals: List[float] = []
        for i in range(pred.shape[0]):
            v = float(calc_psnr_clip3(pred[i : i + 1], x[i : i + 1]))
            vals.append(v)

            if tp == TIMEPOINTS[0]:
                rows.append({"slice_idx": float(i)})
            rows[i][f"psnr_{tp}"] = v

        summary[tp] = summarize(vals)
        print(f"[{tp}] mean={summary[tp]['mean']:.6f} std={summary[tp]['std']:.6f} n={int(summary[tp]['n'])}")

    summary_path = out_dir / f"gt_latent_decoder_ceiling_clip3_{args.split}_summary.json"
    per_slice_path = out_dir / f"gt_latent_decoder_ceiling_clip3_{args.split}_per_slice.csv"

    payload = {
        "meta": {
            "split": args.split,
            "latents_path": args.latents_path,
            "raw_data_dir": args.raw_data_dir,
            "rae_ckpt": args.rae_ckpt,
            "device": str(device),
            "batch_size": int(args.batch_size),
            "clamp_max": float(args.clamp_max),
            "timepoints": TIMEPOINTS,
            "psnr_metric": "src.utils.metrics.calc_psnr_clip3",
            "eval_type": "GT_latent_direct_decode",
        },
        "summary_psnr_clip3": summary,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    fieldnames = ["slice_idx"] + [f"psnr_{tp}" for tp in TIMEPOINTS]
    with open(per_slice_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(payload["meta"], indent=2))
    print("summary_psnr_clip3_mean:")
    for tp in TIMEPOINTS:
        print(f"  {tp}: {summary[tp]['mean']:.9f}")
    print(f"Saved summary JSON: {summary_path}")
    print(f"Saved per-slice CSV: {per_slice_path}")


if __name__ == "__main__":
    main()
