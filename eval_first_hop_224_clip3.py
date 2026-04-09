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
import yaml
from tqdm import tqdm

from pet_lr.data_first_hop import PETFirstHopAligned4HopDataset
from pet_lr.model_first_hop import PETFlowDiTFirstHop
from pet_lr.rollout_first_hop import sample_chain_first_hop

# Canonical PSNR implementation (window to 3 first, then PSNR) from RAE repo.
RAE_CODE_ROOT = "/home/qujiaxiang/project/RAE/code/RAE"
if RAE_CODE_ROOT not in sys.path:
    sys.path.insert(0, RAE_CODE_ROOT)
from src.utils.metrics import calc_psnr_clip3  # noqa: E402


TIMEPOINTS = ["D50", "D20", "D10", "D4", "NORMAL"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate 224 first-hop latent transport with canonical calc_psnr_clip3 and "
            "write reviewable JSON/CSV outputs."
        )
    )
    p.add_argument(
        "--config",
        default="/home/qujiaxiang/project/PET_LatentResidual/configs/pet_flow/pet_flow_first_hop_224_alignchecked_detfix.yaml",
        help="Training config used to build dataset/model",
    )
    p.add_argument(
        "--checkpoint",
        default="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_smoke_alignchecked_detfix/best.pt",
        help="Checkpoint path from train_first_hop.py",
    )
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument(
        "--max-slices",
        type=int,
        default=512,
        help="Number of slices to evaluate (<=0 means full split)",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--out-dir",
        default="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_eval_clip3",
        help="Directory for JSON/CSV artifacts",
    )
    return p.parse_args()


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_indices(num_slices: int, max_slices: int) -> torch.Tensor:
    if max_slices <= 0 or max_slices >= num_slices:
        return torch.arange(num_slices, dtype=torch.long)
    idx = torch.linspace(0, num_slices - 1, steps=max_slices).round().long()
    idx = torch.unique(idx, sorted=True)
    return idx


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0.0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": float(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def load_model(cfg: Dict, ckpt_path: str, device: torch.device) -> PETFlowDiTFirstHop:
    model = PETFlowDiTFirstHop(cfg, device=device).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


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


@torch.no_grad()
def evaluate(
    model: PETFlowDiTFirstHop,
    dataset: PETFirstHopAligned4HopDataset,
    rollout_times: List[float],
    batch_size: int,
    eval_indices: torch.Tensor,
    image_size: int,
    device: torch.device,
) -> tuple[Dict[str, Dict[str, float]], List[Dict[str, object]]]:
    per_tp: Dict[str, List[float]] = {tp: [] for tp in dataset.rollout_timepoints}
    row_map: Dict[int, Dict[str, object]] = {}

    total = int(eval_indices.numel())
    for start in tqdm(range(0, total, batch_size), desc="eval"):
        idx = eval_indices[start : start + batch_size]
        z_d50 = dataset.latents["D50"][idx].to(device)
        x_d50 = dataset.images["D50"][idx].to(device)
        z_chain = sample_chain_first_hop(model, z_d50, x_d50, rollout_times=rollout_times)

        for tp_i, tp in enumerate(dataset.rollout_timepoints):
            x_pred = model.decode_crop(z_chain[tp_i], crop_size=image_size).detach().cpu()
            x_gt = dataset.images[tp][idx].float().cpu()

            for b in range(x_pred.shape[0]):
                psnr_v = float(calc_psnr_clip3(x_pred[b : b + 1], x_gt[b : b + 1]))
                per_tp[tp].append(psnr_v)

                slice_idx = int(idx[b].item())
                row = row_map.setdefault(slice_idx, {"slice_idx": slice_idx})
                row[f"psnr_{tp}"] = psnr_v

    summary = {tp: summarize(per_tp[tp]) for tp in dataset.rollout_timepoints}
    rows = [row_map[k] for k in sorted(row_map.keys())]
    return summary, rows


def write_csv(path: Path, split: str, rows: List[Dict[str, object]], rollout_tps: List[str]) -> None:
    fieldnames = ["split", "slice_idx"] + [f"psnr_{tp}" for tp in rollout_tps]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out_row = {"split": split, "slice_idx": row["slice_idx"]}
            for tp in rollout_tps:
                out_row[f"psnr_{tp}"] = row.get(f"psnr_{tp}", "")
            writer.writerow(out_row)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rollout_tps = cfg["data"].get("rollout_timepoints", TIMEPOINTS)
    rollout_times = [float(cfg["data"]["t_map"][tp]) for tp in rollout_tps]
    image_size = int(cfg["data"].get("image_size", 224))

    dataset = build_dataset(cfg, split=args.split)
    eval_indices = select_indices(dataset.num_slices, args.max_slices)
    model = load_model(cfg, ckpt_path=args.checkpoint, device=device)

    summary, rows = evaluate(
        model=model,
        dataset=dataset,
        rollout_times=rollout_times,
        batch_size=int(args.batch_size),
        eval_indices=eval_indices,
        image_size=image_size,
        device=device,
    )

    payload = {
        "meta": {
            "psnr_metric": "src.utils.metrics.calc_psnr_clip3",
            "config": args.config,
            "checkpoint": args.checkpoint,
            "split": args.split,
            "max_slices": int(args.max_slices),
            "num_eval_slices": int(eval_indices.numel()),
            "batch_size": int(args.batch_size),
            "device": str(device),
            "rollout_timepoints": rollout_tps,
        },
        "summary_psnr_clip3": summary,
        "per_slice": rows,
    }

    json_path = out_dir / f"first_hop_224_{args.split}_clip3_eval.json"
    csv_path = out_dir / f"first_hop_224_{args.split}_clip3_eval.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_csv(csv_path, split=args.split, rows=rows, rollout_tps=rollout_tps)

    print(json.dumps(payload["meta"], indent=2))
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
