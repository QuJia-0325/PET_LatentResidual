#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RAE_CODE_ROOT = "/home/qujiaxiang/project/RAE/code/RAE"
if RAE_CODE_ROOT not in sys.path:
    sys.path.insert(0, RAE_CODE_ROOT)

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from pet_lr.data_first_hop import PETFirstHopAligned4HopDataset
from pet_lr.model_first_hop import PETFlowDiTFirstHop
from pet_lr.rollout_first_hop import sample_one_step_first_hop
from pet_lr.path_guard import ensure_repo_local_outputs_absent
from src.utils.metrics import calc_psnr_clip3


TIMEPOINTS = ["D50", "D20", "D10", "D4", "NORMAL"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize first-hop rollout chain with clip3 PSNR and SUV[0,3] display."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tag", required=True, help="Short run tag used in output filenames.")
    p.add_argument("--split", choices=["val"], default="val")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--vis-indices", default="100,300,500,700,900,1100")
    p.add_argument("--diff-vmax", type=float, default=0.3)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--out-dir",
        default="/home/qujiaxiang/project/PET_LatentResidual/review/0425/visuals",
    )
    return p.parse_args()


def parse_indices(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataset(cfg: Dict, split: str) -> PETFirstHopAligned4HopDataset:
    data_cfg = cfg["data"]
    latent_path = str(Path(data_cfg["latent_dir"]) / f"latents_{split}.pt")
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


def load_model(cfg: Dict, ckpt_path: str, device: torch.device) -> PETFlowDiTFirstHop:
    model = PETFlowDiTFirstHop(cfg, device=device).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    if isinstance(ckpt, dict):
        ckpt_pixel = ckpt.get("first_hop_pixel_enabled", None)
        current_pixel = not model.pixel_forcing_disabled
        if ckpt_pixel is not None and bool(ckpt_pixel) != current_pixel:
            raise RuntimeError(
                f"Eval pixel semantic mismatch: checkpoint={ckpt_pixel}, "
                f"config pixel_forcing_disabled={model.pixel_forcing_disabled}"
            )
    model.eval()
    return model


def to_suv_clip3(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 5.0).clamp(0.0, 3.0)


@torch.no_grad()
def rollout_to_normal_from_source(
    model: PETFlowDiTFirstHop,
    z_src: torch.Tensor,
    source_idx: int,
    rollout_times: List[float],
    x_d50: torch.Tensor | None,
) -> torch.Tensor:
    z_curr = z_src
    for hop_idx in range(source_idx, len(rollout_times) - 1):
        x_src_img = x_d50 if hop_idx == 0 else None
        z_curr = sample_one_step_first_hop(
            model=model,
            z_start=z_curr,
            t_start=rollout_times[hop_idx],
            t_end=rollout_times[hop_idx + 1],
            hop_idx=hop_idx,
            x_src_img=x_src_img,
        )
    return z_curr


@torch.no_grad()
def visualize_slice(
    model: PETFlowDiTFirstHop,
    dataset: PETFirstHopAligned4HopDataset,
    rollout_times: List[float],
    image_size: int,
    slice_idx: int,
    out_path: Path,
    diff_vmax: float,
    dpi: int,
    device: torch.device,
    tag: str,
) -> Dict[str, float]:
    source_tps = dataset.rollout_timepoints[:-1]
    gt_normal = dataset.images["NORMAL"][slice_idx : slice_idx + 1].to(device)
    x_d50 = dataset.images["D50"][slice_idx : slice_idx + 1].to(device)

    fig, axes = plt.subplots(len(source_tps), 4, figsize=(12, 3 * len(source_tps)))
    if len(source_tps) == 1:
        axes = np.expand_dims(axes, axis=0)

    psnr_vals: Dict[str, float] = {}
    for row, tp in enumerate(source_tps):
        source_idx = row
        z_src = dataset.latents[tp][slice_idx : slice_idx + 1].to(device)

        z_pred_normal = rollout_to_normal_from_source(
            model=model,
            z_src=z_src,
            source_idx=source_idx,
            rollout_times=rollout_times,
            x_d50=x_d50,
        )

        x_input = model.decode_crop(z_src, crop_size=image_size)
        x_pred_normal = model.decode_crop(z_pred_normal, crop_size=image_size)
        psnr = float(calc_psnr_clip3(x_pred_normal[:, 0:1], gt_normal[:, 0:1]))
        psnr_vals[f"{tp}_to_NORMAL"] = psnr

        input_suv = to_suv_clip3(x_input[:, 0:1]).squeeze().detach().cpu().numpy()
        pred_suv = to_suv_clip3(x_pred_normal[:, 0:1]).squeeze().detach().cpu().numpy()
        gt_suv = to_suv_clip3(gt_normal[:, 0:1]).squeeze().detach().cpu().numpy()
        diff = np.abs(pred_suv - gt_suv)

        axes[row, 0].imshow(input_suv, cmap="gray", vmin=0.0, vmax=3.0)
        axes[row, 0].set_title(f"{tp} Input")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(pred_suv, cmap="gray", vmin=0.0, vmax=3.0)
        axes[row, 1].set_title(f"{tp}→NORMAL Pred\nPSNR={psnr:.2f} dB")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(gt_suv, cmap="gray", vmin=0.0, vmax=3.0)
        axes[row, 2].set_title("GT NORMAL")
        axes[row, 2].axis("off")

        axes[row, 3].imshow(diff, cmap="hot", vmin=0.0, vmax=diff_vmax)
        axes[row, 3].set_title("|Pred - GT|")
        axes[row, 3].axis("off")

    fig.suptitle(
        f"{tag} | slice={slice_idx} | source→NORMAL rollout | display=SUV clip[0,3] | diff vmax={diff_vmax}",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return psnr_vals


def main() -> None:
    args = parse_args()
    ensure_repo_local_outputs_absent(REPO_ROOT)

    cfg = load_yaml(args.config)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(cfg, split=args.split)
    rollout_tps = cfg["data"].get("rollout_timepoints", TIMEPOINTS)
    rollout_times = [float(cfg["data"]["t_map"][tp]) for tp in rollout_tps]
    image_size = int(cfg["data"].get("image_size", 224))
    vis_indices = parse_indices(args.vis_indices)

    model = load_model(cfg, args.checkpoint, device=device)

    summary_rows = []
    for slice_idx in vis_indices:
        if slice_idx < 0 or slice_idx >= dataset.num_slices:
            continue
        out_path = out_dir / f"{args.tag}_slice_{slice_idx:04d}.png"
        psnr_vals = visualize_slice(
            model=model,
            dataset=dataset,
            rollout_times=rollout_times,
            image_size=image_size,
            slice_idx=slice_idx,
            out_path=out_path,
            diff_vmax=float(args.diff_vmax),
            dpi=int(args.dpi),
            device=device,
            tag=args.tag,
        )
        row = {"slice_idx": slice_idx, **psnr_vals}
        summary_rows.append(row)
        print(f"saved: {out_path}", flush=True)

    summary_path = out_dir / f"{args.tag}_visual_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        import json

        json.dump(
            {
                "tag": args.tag,
                "config": args.config,
                "checkpoint": args.checkpoint,
                "device": str(device),
                "vis_indices": vis_indices,
                "display": "SUV clip[0,3]",
                "diff_vmax": float(args.diff_vmax),
                "psnr_metric": "src.utils.metrics.calc_psnr_clip3",
                "rows": summary_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"saved: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
