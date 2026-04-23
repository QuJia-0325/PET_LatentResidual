#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

from pet_lr.data_first_hop import PETFirstHopAligned4HopDataset
from pet_lr.model_first_hop import PETFlowDiTFirstHop
from pet_lr.rollout_first_hop import sample_chain_first_hop
from pet_lr.path_guard import ensure_repo_local_outputs_absent


TIMEPOINTS = ["D50", "D20", "D10", "D4", "NORMAL"]


@dataclass(frozen=True)
class EvalTarget:
    scheme: str
    ckpt_kind: str
    config_path: str
    checkpoint_path: str

    @property
    def tag(self) -> str:
        return f"{self.scheme}_{self.ckpt_kind}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fast full-val rerank for C/N1/v2 (best/last) using clip3 per-sample PSNR, "
            "protocol-equivalent to eval_first_hop_224_clip3.py for ranking metrics."
        )
    )
    p.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device. Use CUDA_VISIBLE_DEVICES to pin physical GPU.",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--split",
        choices=["val"],
        default="val",
        help="Rerank split (fixed to val for this task).",
    )
    p.add_argument(
        "--out-dir",
        default="/home/qujiaxiang/project/PET_LatentResidual/review/0423/artifacts",
        help="Directory for summary JSON/CSV outputs.",
    )
    p.add_argument(
        "--schemes",
        default="C,N1,v2",
        help="Comma-separated subset of schemes to run, e.g. C,N1,v2 or N1,v2",
    )
    return p.parse_args()


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_targets() -> List[EvalTarget]:
    return [
        EvalTarget(
            scheme="C",
            ckpt_kind="best",
            config_path="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_imgaux_boost_gpu2_nopbar_20260418_010035/config.yaml",
            checkpoint_path="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_imgaux_boost_gpu2_nopbar_20260418_010035/best.pt",
        ),
        EvalTarget(
            scheme="C",
            ckpt_kind="last",
            config_path="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_imgaux_boost_gpu2_nopbar_20260418_010035/config.yaml",
            checkpoint_path="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_imgaux_boost_gpu2_nopbar_20260418_010035/last.pt",
        ),
        EvalTarget(
            scheme="N1",
            ckpt_kind="best",
            config_path="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_pixenc_ablation/config.yaml",
            checkpoint_path="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_pixenc_ablation/best.pt",
        ),
        EvalTarget(
            scheme="N1",
            ckpt_kind="last",
            config_path="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_pixenc_ablation/config.yaml",
            checkpoint_path="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_pixenc_ablation/last.pt",
        ),
        EvalTarget(
            scheme="v2",
            ckpt_kind="best",
            config_path="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_schemec_v2/config.yaml",
            checkpoint_path="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_schemec_v2/best.pt",
        ),
        EvalTarget(
            scheme="v2",
            ckpt_kind="last",
            config_path="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_schemec_v2/config.yaml",
            checkpoint_path="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_schemec_v2/last.pt",
        ),
    ]


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


def psnr_clip3_per_sample(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    pred_suv = (pred + 1.0) * 5.0
    gt_suv = (gt + 1.0) * 5.0
    pred_suv = pred_suv.clamp(0.0, 3.0)
    gt_suv = gt_suv.clamp(0.0, 3.0)
    mse = (pred_suv - gt_suv).pow(2).flatten(start_dim=1).mean(dim=1)
    eps = torch.finfo(mse.dtype).tiny
    psnr = 20.0 * math.log10(3.0) - 10.0 * torch.log10(mse.clamp_min(eps))
    psnr = torch.where(mse <= 0.0, torch.full_like(psnr, float("inf")), psnr)
    return psnr


def summarize_from_acc(acc: Dict[str, float]) -> Dict[str, float]:
    n = int(acc["n"])
    if n <= 0:
        return {"n": 0.0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    mean = acc["sum"] / float(n)
    var = max(acc["sum_sq"] / float(n) - mean * mean, 0.0)
    std = math.sqrt(var)
    return {
        "n": float(n),
        "mean": float(mean),
        "std": float(std),
        "min": float(acc["min"]),
        "max": float(acc["max"]),
    }


@torch.no_grad()
def eval_checkpoint(
    model: PETFlowDiTFirstHop,
    dataset: PETFirstHopAligned4HopDataset,
    rollout_times: List[float],
    image_size: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, object]:
    acc: Dict[str, Dict[str, float]] = {
        tp: {"sum": 0.0, "sum_sq": 0.0, "n": 0, "min": float("inf"), "max": float("-inf")}
        for tp in dataset.rollout_timepoints
    }
    n_total = dataset.num_slices
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        idx = torch.arange(start, end, dtype=torch.long)
        z_d50 = dataset.latents["D50"][idx].to(device, non_blocking=True)
        x_d50 = dataset.images["D50"][idx].to(device, non_blocking=True)
        z_chain = sample_chain_first_hop(model, z_d50, x_d50, rollout_times=rollout_times)
        for tp_i, tp in enumerate(dataset.rollout_timepoints):
            x_gt = dataset.images[tp][idx].to(device, non_blocking=True)
            apply_refiner = None
            if tp_i == 0 and getattr(model, "seam_refiner_skip_first_tp", False):
                apply_refiner = False
            x_pred = model.decode_crop(z_chain[tp_i], crop_size=image_size, apply_refiner=apply_refiner)
            psnr = psnr_clip3_per_sample(x_pred.float(), x_gt.float())
            finite = psnr[torch.isfinite(psnr)]
            if finite.numel() == 0:
                continue
            s = float(finite.sum().item())
            sq = float((finite * finite).sum().item())
            mn = float(finite.min().item())
            mx = float(finite.max().item())
            n = int(finite.numel())
            acc_tp = acc[tp]
            acc_tp["sum"] += s
            acc_tp["sum_sq"] += sq
            acc_tp["n"] += n
            acc_tp["min"] = min(acc_tp["min"], mn)
            acc_tp["max"] = max(acc_tp["max"], mx)
        if (start // batch_size) % 50 == 0:
            print(f"[eval] processed {end}/{n_total} slices", flush=True)

    summary_psnr = {tp: summarize_from_acc(acc[tp]) for tp in dataset.rollout_timepoints}
    d20 = summary_psnr["D20"]["mean"]
    d10 = summary_psnr["D10"]["mean"]
    d4 = summary_psnr["D4"]["mean"]
    normal = summary_psnr["NORMAL"]["mean"]
    d50 = summary_psnr["D50"]["mean"]
    transport_avg = (d20 + d10 + d4 + normal) / 4.0
    all_avg = (d50 + d20 + d10 + d4 + normal) / 5.0
    return {
        "summary_psnr_clip3": summary_psnr,
        "summary_transport_avg": float(transport_avg),
        "summary_all_avg": float(all_avg),
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = ["scheme", "checkpoint_kind", "D50", "D20", "D10", "D4", "NORMAL", "transport_avg", "all_avg", "runtime_sec"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    args = parse_args()
    ensure_repo_local_outputs_absent(Path(__file__).resolve().parents[2])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    targets = build_targets()
    selected = {x.strip() for x in str(args.schemes).split(",") if x.strip()}
    if selected:
        targets = [t for t in targets if t.scheme in selected]
    if not targets:
        raise RuntimeError(f"No targets selected from --schemes={args.schemes!r}")
    cfg_map = {t.tag: load_yaml(t.config_path) for t in targets}

    # Build dataset once using C config (shared data protocol).
    base_cfg = cfg_map[targets[0].tag]
    dataset = build_dataset(base_cfg, split=args.split)
    rollout_tps = base_cfg["data"].get("rollout_timepoints", TIMEPOINTS)
    rollout_times = [float(base_cfg["data"]["t_map"][tp]) for tp in rollout_tps]
    image_size = int(base_cfg["data"].get("image_size", 224))

    result_items = []
    csv_rows = []
    for t in targets:
        cfg = cfg_map[t.tag]
        # Guard for protocol consistency on evaluation axes.
        tps = cfg["data"].get("rollout_timepoints", TIMEPOINTS)
        if list(tps) != list(rollout_tps):
            raise RuntimeError(f"rollout_timepoints mismatch for {t.tag}: {tps} vs {rollout_tps}")
        t_map = cfg["data"]["t_map"]
        for tp in rollout_tps:
            if float(t_map[tp]) != float(base_cfg["data"]["t_map"][tp]):
                raise RuntimeError(f"t_map mismatch for {t.tag} at {tp}")

        print(f"\n=== Evaluating {t.tag} ===", flush=True)
        print(f"config={t.config_path}", flush=True)
        print(f"ckpt={t.checkpoint_path}", flush=True)
        t0 = time.time()
        model = load_model(cfg, t.checkpoint_path, device=device)
        out = eval_checkpoint(
            model=model,
            dataset=dataset,
            rollout_times=rollout_times,
            image_size=image_size,
            batch_size=int(args.batch_size),
            device=device,
        )
        dt = time.time() - t0
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        item = {
            "scheme": t.scheme,
            "checkpoint_kind": t.ckpt_kind,
            "config": t.config_path,
            "checkpoint": t.checkpoint_path,
            "runtime_sec": float(dt),
            **out,
        }
        result_items.append(item)
        s = out["summary_psnr_clip3"]
        csv_rows.append(
            {
                "scheme": t.scheme,
                "checkpoint_kind": t.ckpt_kind,
                "D50": s["D50"]["mean"],
                "D20": s["D20"]["mean"],
                "D10": s["D10"]["mean"],
                "D4": s["D4"]["mean"],
                "NORMAL": s["NORMAL"]["mean"],
                "transport_avg": out["summary_transport_avg"],
                "all_avg": out["summary_all_avg"],
                "runtime_sec": float(dt),
            }
        )
        print(
            f"[done] {t.tag}: transport_avg={out['summary_transport_avg']:.6f}, "
            f"all_avg={out['summary_all_avg']:.6f}, runtime={dt/60.0:.1f}min",
            flush=True,
        )

    by_transport = sorted(
        [
            {
                "tag": f"{x['scheme']}_{x['checkpoint_kind']}",
                "transport_avg": float(x["summary_transport_avg"]),
                "all_avg": float(x["summary_all_avg"]),
            }
            for x in result_items
        ],
        key=lambda z: z["transport_avg"],
        reverse=True,
    )

    payload = {
        "psnr_metric": "clip3_per_sample_mean (equivalent formula to calc_psnr_clip3)",
        "split": args.split,
        "num_eval_slices": int(dataset.num_slices),
        "batch_size": int(args.batch_size),
        "device": str(device),
        "rollout_timepoints": rollout_tps,
        "results": result_items,
        "ranking_by_transport_avg_desc": by_transport,
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"fullval_rerank_c_n1_v2_clip3_fast_{ts}.json"
    csv_path = out_dir / f"fullval_rerank_c_n1_v2_clip3_fast_{ts}.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    write_csv(csv_path, csv_rows)
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")


if __name__ == "__main__":
    main()
