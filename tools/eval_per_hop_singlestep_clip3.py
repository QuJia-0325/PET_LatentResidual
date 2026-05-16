#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pet_lr.data_first_hop import PETFirstHopAligned4HopDataset
from pet_lr.model_first_hop import PETFlowDiTFirstHop

RAE_CODE_ROOT = "/home/qujiaxiang/project/RAE/code/RAE"
if RAE_CODE_ROOT not in sys.path:
    sys.path.insert(0, RAE_CODE_ROOT)
from src.utils.metrics import calc_psnr_clip3  # noqa: E402

TIMEPOINTS = ["D50", "D20", "D10", "D4", "NORMAL"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-hop GT-input single-step PSNR_clip3 eval.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--split", choices=["val"], default="val")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-slices", type=int, default=0, help="<=0 means full split")
    p.add_argument("--decode-mode", choices=["default", "raw", "both"], default="both")
    return p.parse_args()


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
                "Eval pixel_forcing semantic mismatch: "
                f"checkpoint first_hop_pixel_enabled={ckpt_pixel}, "
                f"config pixel_forcing_disabled={model.pixel_forcing_disabled}"
            )
    model.eval()
    return model, ckpt if isinstance(ckpt, dict) else {}


def select_indices(num_slices: int, max_slices: int) -> torch.Tensor:
    if max_slices <= 0 or max_slices >= num_slices:
        return torch.arange(num_slices, dtype=torch.long)
    idx = torch.linspace(0, num_slices - 1, steps=max_slices).round().long()
    return torch.unique(idx, sorted=True)


def _decode_modes(mode: str) -> list[tuple[str, bool | None]]:
    modes: list[tuple[str, bool | None]] = []
    if mode in ("default", "both"):
        modes.append(("", None))
    if mode in ("raw", "both"):
        modes.append(("_raw", False))
    return modes


def _new_acc() -> Dict[str, float]:
    return {"sum": 0.0, "sum_sq": 0.0, "n": 0.0, "min": float("inf"), "max": float("-inf")}


def _add(acc: Dict[str, float], values: torch.Tensor) -> None:
    values = values.detach().float().flatten()
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return
    acc["sum"] += float(finite.sum().item())
    acc["sum_sq"] += float((finite * finite).sum().item())
    acc["n"] += float(finite.numel())
    acc["min"] = min(acc["min"], float(finite.min().item()))
    acc["max"] = max(acc["max"], float(finite.max().item()))


def _summary(acc: Dict[str, float]) -> Dict[str, float]:
    n = int(acc["n"])
    if n <= 0:
        return {"n": 0.0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    mean = acc["sum"] / float(n)
    var = max(acc["sum_sq"] / float(n) - mean * mean, 0.0)
    return {"n": float(n), "mean": float(mean), "std": float(math.sqrt(var)), "min": float(acc["min"]), "max": float(acc["max"])}


@torch.no_grad()
def evaluate(
    model: PETFlowDiTFirstHop,
    dataset: PETFirstHopAligned4HopDataset,
    eval_indices: torch.Tensor,
    batch_size: int,
    image_size: int,
    device: torch.device,
    decode_mode: str,
) -> tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], list[dict]]:
    modes = _decode_modes(decode_mode)
    tps = list(dataset.rollout_timepoints)
    t_map = dataset.t_map
    psnr_acc: Dict[str, Dict[str, float]] = {}
    mse_acc: Dict[str, Dict[str, float]] = {}
    for hop in range(4):
        for suffix, _ in modes:
            key = f"hop{hop}_{tps[hop]}_to_{tps[hop + 1]}{suffix}"
            psnr_acc[key] = _new_acc()
            mse_acc[key] = _new_acc()

    rows: list[dict] = []
    total = int(eval_indices.numel())
    for start in tqdm(range(0, total, batch_size), desc=f"singlestep:{total}", disable=not sys.stderr.isatty()):
        idx = eval_indices[start : start + batch_size]
        batch_rows = [{"slice_idx": int(i)} for i in idx.tolist()]
        for hop in range(4):
            src_tp, dst_tp = tps[hop], tps[hop + 1]
            z_src = dataset.latents[src_tp][idx].to(device, non_blocking=True)
            x_gt = dataset.images[dst_tp][idx].to(device, non_blocking=True).float()
            bsz = int(z_src.shape[0])
            t_src = torch.full((bsz,), float(t_map[src_tp]), device=device)
            t_dst = torch.full((bsz,), float(t_map[dst_tp]), device=device)
            hop_idx = torch.full((bsz,), hop, device=device, dtype=torch.long)
            x_src = dataset.images["D50"][idx].to(device, non_blocking=True).float() if hop == 0 else None
            out = model.predict_latent_step(z_src=z_src, t_src=t_src, t_dst=t_dst, hop_idx=hop_idx, x_src_img=x_src)
            z_pred = out["z_pred"] if isinstance(out, dict) else out
            for suffix, refiner_flag in modes:
                x_pred = model.decode_crop(z_pred, crop_size=image_size, apply_refiner=refiner_flag).float()
                per_sample_mse = (x_pred - x_gt).pow(2).flatten(start_dim=1).mean(dim=1)
                key = f"hop{hop}_{src_tp}_to_{dst_tp}{suffix}"
                _add(mse_acc[key], per_sample_mse.cpu())
                psnr_vals = []
                for b in range(bsz):
                    psnr_vals.append(float(calc_psnr_clip3(x_pred[b : b + 1].detach().cpu(), x_gt[b : b + 1].detach().cpu())))
                _add(psnr_acc[key], torch.tensor(psnr_vals, dtype=torch.float32))
                for b, row in enumerate(batch_rows):
                    prefix = f"{key}"
                    row[f"psnr_{prefix}"] = float(psnr_vals[b])
                    row[f"mse_{prefix}"] = float(per_sample_mse[b].detach().cpu().item())
        rows.extend(batch_rows)
    return ({k: _summary(v) for k, v in psnr_acc.items()}, {k: _summary(v) for k, v in mse_acc.items()}, rows)


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({k for row in rows for k in row.keys() if k != "slice_idx"})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["slice_idx"] + keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag.strip() or Path(args.checkpoint).parent.name + "_" + Path(args.checkpoint).stem

    dataset = build_dataset(cfg, args.split)
    eval_indices = select_indices(dataset.num_slices, int(args.max_slices))
    image_size = int(cfg["data"].get("image_size", 224))
    t0 = time.time()
    model, ckpt_meta = load_model(cfg, args.checkpoint, device)
    psnr_summary, mse_summary, rows = evaluate(
        model=model,
        dataset=dataset,
        eval_indices=eval_indices,
        batch_size=int(args.batch_size),
        image_size=image_size,
        device=device,
        decode_mode=args.decode_mode,
    )
    runtime_sec = time.time() - t0
    payload = {
        "tag": tag,
        "meta": {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "checkpoint_step": ckpt_meta.get("step") if isinstance(ckpt_meta, dict) else None,
            "checkpoint_best_val": ckpt_meta.get("best_val") if isinstance(ckpt_meta, dict) else None,
            "split": args.split,
            "max_slices": int(args.max_slices),
            "num_eval_slices": int(eval_indices.numel()),
            "batch_size": int(args.batch_size),
            "device": str(device),
            "decode_mode": args.decode_mode,
            "runtime_sec": float(runtime_sec),
            "psnr_metric": "src.utils.metrics.calc_psnr_clip3",
            "protocol": "single-step GT z_k -> z_{k+1}; no cascade; hop0 receives GT D50 image only",
            "rollout_timepoints": list(dataset.rollout_timepoints),
        },
        "summary_psnr_clip3": psnr_summary,
        "summary_mse": mse_summary,
    }
    json_path = out_dir / "per_hop_singlestep_psnr.json"
    csv_path = out_dir / "per_hop_singlestep_per_slice.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
