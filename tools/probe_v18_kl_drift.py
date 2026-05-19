#!/usr/bin/env python3
"""Stage B V18 KL drift probe.

Measures decoder GT-manifold drift by comparing canonical PSNR_clip3 of
`decode(z_GT)` against target pixels for V7.best, V18.best, and V18.last.
This intentionally skips the transport rollout path.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pet_lr.data_first_hop import Hop0OnlyViewDataset, PETFirstHopAligned4HopDataset
from pet_lr.model_first_hop import PETFlowDiTFirstHop
from pet_lr.path_guard import ensure_repo_local_outputs_absent

RAE_CODE_ROOT = "/home/qujiaxiang/project/RAE/code/RAE"
if RAE_CODE_ROOT not in sys.path:
    sys.path.insert(0, RAE_CODE_ROOT)
from src.utils.metrics import calc_psnr_clip3  # noqa: E402

TIMEPOINTS = ["D50", "D20", "D10", "D4", "NORMAL"]
TARGET_TIMEPOINTS = ["D20", "D10", "D4", "NORMAL"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V18 decoder KL drift probe on GT latents.")
    p.add_argument("--config", required=True)
    p.add_argument("--v18-config", default="review/0517/V18_decoder_lora/V18_decoder_lora.yaml")
    p.add_argument("--v18-cap-config", default=None, help="Defaults to --v18-config when omitted")
    p.add_argument("--v7-ckpt", required=True)
    p.add_argument("--v18-best-ckpt", required=True)
    p.add_argument("--v18-step170k-ckpt", default=None)
    p.add_argument("--v18-last-ckpt", required=True)
    p.add_argument("--v18-cap-ckpt", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-slices", type=int, default=0, help="0 = all val slices")
    p.add_argument("--split", choices=["val"], default="val")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataset(cfg: Dict, split: str) -> Hop0OnlyViewDataset:
    data_cfg = cfg["data"]
    latent_path = os.path.join(data_cfg["latent_dir"], f"latents_{split}.pt")
    alignment_audit_json = str(data_cfg.get("alignment_audit_json", "")).strip() or None
    base = PETFirstHopAligned4HopDataset(
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
        latent_mmap=bool(data_cfg.get("latent_mmap", False)),
    )
    return Hop0OnlyViewDataset(base)


def maybe_subset_dataset(ds: Hop0OnlyViewDataset, max_slices: int):
    if max_slices <= 0 or max_slices >= len(ds):
        return ds
    idx = torch.linspace(0, len(ds) - 1, steps=max_slices).round().long()
    idx = torch.unique(idx, sorted=True).tolist()
    return torch.utils.data.Subset(ds, idx)


def to_2d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4 and x.shape[1] > 1:
        x = x[:, 0:1]
    return x


def infer_ckpt_step(ckpt: Dict) -> int | None:
    for key in ("step", "global_step", "iter"):
        if key in ckpt:
            try:
                return int(ckpt[key])
            except Exception:
                pass
    return None


def load_model(cfg: Dict, ckpt_path: str, device: torch.device) -> tuple[PETFlowDiTFirstHop, Dict]:
    model = PETFlowDiTFirstHop(cfg, device).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"[kl_drift] load {ckpt_path}: missing={len(missing)} unexpected={len(unexpected)} step={infer_ckpt_step(ckpt) if isinstance(ckpt, dict) else None}",
        flush=True,
    )
    if missing:
        print(f"[kl_drift] WARN missing first5={missing[:5]}", flush=True)
    if unexpected:
        print(f"[kl_drift] WARN unexpected first5={unexpected[:5]}", flush=True)
    model.eval()
    return model, ckpt if isinstance(ckpt, dict) else {}


def new_acc() -> Dict[str, float]:
    return {"sum": 0.0, "sum_sq": 0.0, "n": 0.0, "min": float("inf"), "max": float("-inf")}


def add(acc: Dict[str, float], values: Iterable[float]) -> None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return
    arr = np.asarray(vals, dtype=np.float64)
    acc["sum"] += float(arr.sum())
    acc["sum_sq"] += float((arr * arr).sum())
    acc["n"] += float(arr.size)
    acc["min"] = min(acc["min"], float(arr.min()))
    acc["max"] = max(acc["max"], float(arr.max()))


def summarize(acc: Dict[str, float]) -> Dict[str, float]:
    n = int(acc["n"])
    if n <= 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    mean = acc["sum"] / n
    var = max(acc["sum_sq"] / n - mean * mean, 0.0)
    return {"n": n, "mean": float(mean), "std": float(math.sqrt(var)), "min": float(acc["min"]), "max": float(acc["max"])}


def verdict_from_drift(drift: float) -> str:
    if drift < 0.05:
        return "NEGLIGIBLE"
    if drift < 1.0:
        return "MODERATE"
    return "SIGNIFICANT"


def evaluate_ckpt(
    *,
    name: str,
    cfg: Dict,
    ckpt_path: str,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    image_size: int,
) -> tuple[Dict, List[Dict]]:
    model, ckpt = load_model(cfg, ckpt_path, device)
    ckpt_step = infer_ckpt_step(ckpt)
    acc = {tp: new_acc() for tp in TARGET_TIMEPOINTS}
    rows: List[Dict] = []
    tp_to_idx = {tp: TIMEPOINTS.index(tp) for tp in TARGET_TIMEPOINTS}

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"kl_drift:{name}", disable=not sys.stderr.isatty()):
            slice_idx = batch["slice_idx"].cpu().tolist()
            z_rollout = batch["z_rollout"].to(device, non_blocking=True)
            x_rollout = batch["x_rollout"].to(device, non_blocking=True)

            for tp, tp_idx in tp_to_idx.items():
                z_gt = z_rollout[:, tp_idx]
                x_gt = to_2d(x_rollout[:, tp_idx]).float()
                x_hat = to_2d(model.decode_crop(z_gt, crop_size=image_size)).float()
                vals = []
                for b in range(x_hat.shape[0]):
                    psnr = float(calc_psnr_clip3(x_hat[b : b + 1].detach().cpu(), x_gt[b : b + 1].detach().cpu()))
                    vals.append(psnr)
                    rows.append({"ckpt": name, "step": ckpt_step if ckpt_step is not None else "", "slice_idx": int(slice_idx[b]), "timepoint": tp, "psnr_clip3": psnr})
                add(acc[tp], vals)

    summary = {
        "name": name,
        "checkpoint": ckpt_path,
        "step": ckpt_step,
        "psnr_clip3": {tp: summarize(acc[tp]) for tp in TARGET_TIMEPOINTS},
    }
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary, rows


def make_report(summary: Dict, out_dir: Path, elapsed: float) -> str:
    ckpts = summary["checkpoints"]
    ckpt_order = summary.get("meta", {}).get("ckpt_order") or list(ckpts.keys())

    def mean(name: str, tp: str) -> float:
        return float(ckpts[name]["psnr_clip3"][tp]["mean"])

    def diff(lhs: str, rhs: str, tp: str) -> float:
        return mean(lhs, tp) - mean(rhs, tp)

    lines = []
    lines.append("# V18 KL Drift Probe Report")
    lines.append("")
    lines.append("## Protocol")
    lines.append("- Evaluator: `tools/probe_v18_kl_drift.py`")
    lines.append("- Metric: `src.utils.metrics.calc_psnr_clip3`")
    lines.append(f"- Split: `{summary['meta']['split']}`, n={summary['meta']['num_eval_slices']}")
    lines.append("- Computation: direct `decode_crop(z_GT)` only; transport rollout is skipped.")
    lines.append(f"- Runtime: {elapsed:.1f}s")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| ckpt | step | D20 | D10 | D4 | NORMAL |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name in ckpt_order:
        if name not in ckpts:
            continue
        step = ckpts[name].get("step")
        step_s = "" if step is None else str(step)
        vals = [mean(name, tp) for tp in TARGET_TIMEPOINTS]
        lines.append(f"| {name} | {step_s} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | {vals[3]:.4f} |")
    lines.append("")

    if "V7.best" in ckpts:
        lines.append("## KL Drift: V7.best - Checkpoint")
        lines.append("")
        lines.append("Positive values mean the compared checkpoint underperforms V7 on GT latents.")
        lines.append("")
        lines.append("| comparison | D20 | D10 | D4 | NORMAL |")
        lines.append("|---|---:|---:|---:|---:|")
        for name in ckpt_order:
            if name == "V7.best" or name not in ckpts:
                continue
            vals = [diff("V7.best", name, tp) for tp in TARGET_TIMEPOINTS]
            lines.append(f"| V7.best - {name} | {vals[0]:+.4f} | {vals[1]:+.4f} | {vals[2]:+.4f} | {vals[3]:+.4f} |")
        lines.append("")

    if "V18.step170k" in ckpts and "V18-cap.last" in ckpts:
        vals = [diff("V18-cap.last", "V18.step170k", tp) for tp in TARGET_TIMEPOINTS]
        normal_delta = vals[-1]
        if abs(normal_delta) < 0.02:
            a3_verdict = "capacity-only ~= V18.step170k on NORMAL; LoRA capacity is sufficient for the GT-manifold gain."
        elif normal_delta < -0.05:
            a3_verdict = "capacity-only is clearly lower than V18.step170k on NORMAL; KL-dependent optimization effects remain plausible."
        elif normal_delta > 0.05:
            a3_verdict = "capacity-only is clearly higher than V18.step170k on NORMAL; KL may be unnecessary or mildly harmful for GT-manifold decode."
        else:
            a3_verdict = "capacity-only differs from V18.step170k by an intermediate amount; interpret as mixed or marginal."
        lines.append("## A3 Capacity-Only Matched-Step Delta")
        lines.append("")
        lines.append("Primary delta is `V18-cap.last(170K) - V18.step170k` on direct `decode(z_GT)` PSNR_clip3.")
        lines.append("")
        lines.append("| comparison | D20 | D10 | D4 | NORMAL |")
        lines.append("|---|---:|---:|---:|---:|")
        lines.append(f"| V18-cap.last - V18.step170k | {vals[0]:+.4f} | {vals[1]:+.4f} | {vals[2]:+.4f} | {vals[3]:+.4f} |")
        lines.append("")
        lines.append(f"- A3 primary verdict: {a3_verdict}")
        lines.append("")

    lines.append("## Verdict")
    lines.append("")
    if "V18.best" in ckpts:
        normal_best_drift = diff("V7.best", "V18.best", "NORMAL")
        verdict_best = verdict_from_drift(abs(normal_best_drift))
        lines.append(f"- V18.best NORMAL KL drift: `{normal_best_drift:+.4f} dB` -> `{verdict_best}` by abs drift")
    if "V18.last" in ckpts:
        normal_last_drift = diff("V7.best", "V18.last", "NORMAL")
        verdict_last = verdict_from_drift(abs(normal_last_drift))
        lines.append(f"- V18.last NORMAL KL drift: `{normal_last_drift:+.4f} dB` -> `{verdict_last}` by abs drift")
    lines.append("")
    signed_v18 = []
    for name in ("V18.best", "V18.last"):
        if name in ckpts and "V7.best" in ckpts:
            signed_v18.extend(diff("V7.best", name, tp) for tp in TARGET_TIMEPOINTS)
    if signed_v18 and all(v < 0.0 for v in signed_v18):
        interp = (
            "All signed drifts are negative: `decode_V18(z_GT)` scores higher PSNR than `decode_V7(z_GT)` "
            "on every reported timepoint. The magnitude is non-negligible, but the direction is an "
            "improvement relative to V7 on the GT latent manifold, not a harmful degradation."
        )
    elif signed_v18 and all(verdict_from_drift(abs(v)) == "NEGLIGIBLE" for v in signed_v18):
        interp = (
            "KL drift is negligible. V18 decoder remains close to the V7 decoder on the GT latent manifold; "
            "the weak V18 transport result is therefore unlikely to be caused by decoder drift."
        )
    else:
        interp = (
            "KL drift magnitude is non-negligible for at least one V18 checkpoint. This means the decoder "
            "changed materially relative to V7 on GT latents and should be considered in Stage C together "
            "with the sign of the change."
        )
    lines.append("## Interpretation")
    lines.append("")
    lines.append(f"- {interp}")
    lines.append("- This report does not launch or recommend a new training run by itself; it is Stage C input only.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `KL_DRIFT_SUMMARY.json`")
    lines.append("- `KL_DRIFT_PER_SLICE.csv`")
    lines.append("- `probe.log`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ensure_repo_local_outputs_absent(REPO_ROOT)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_v7 = load_yaml(args.config)
    cfg_v18 = load_yaml(args.v18_config)
    cfg_v18_cap = load_yaml(args.v18_cap_config) if args.v18_cap_config else cfg_v18
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[kl_drift] device={device}", flush=True)
    print(f"[kl_drift] config_v7={args.config}", flush=True)
    print(f"[kl_drift] config_v18={args.v18_config}", flush=True)
    if args.v18_cap_ckpt:
        print(f"[kl_drift] config_v18_cap={args.v18_cap_config or args.v18_config}", flush=True)

    ds = maybe_subset_dataset(build_dataset(cfg_v7, args.split), args.max_slices)
    image_size = int(cfg_v7["data"].get("image_size", 224))
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"[kl_drift] dataset split={args.split} n={len(ds)} batch_size={args.batch_size}", flush=True)

    ckpt_specs = [
        ("V7.best", cfg_v7, args.v7_ckpt),
        ("V18.best", cfg_v18, args.v18_best_ckpt),
    ]
    if args.v18_step170k_ckpt:
        ckpt_specs.append(("V18.step170k", cfg_v18, args.v18_step170k_ckpt))
    ckpt_specs.append(("V18.last", cfg_v18, args.v18_last_ckpt))
    if args.v18_cap_ckpt:
        ckpt_specs.append(("V18-cap.last", cfg_v18_cap, args.v18_cap_ckpt))

    t0 = time.time()
    summaries: Dict[str, Dict] = {}
    all_rows: List[Dict] = []
    for name, cfg, ckpt in ckpt_specs:
        summary, rows = evaluate_ckpt(
            name=name,
            cfg=cfg,
            ckpt_path=ckpt,
            loader=loader,
            device=device,
            image_size=image_size,
        )
        summaries[name] = summary
        all_rows.extend(rows)
    elapsed = time.time() - t0

    csv_path = out_dir / "KL_DRIFT_PER_SLICE.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["ckpt", "step", "slice_idx", "timepoint", "psnr_clip3"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(all_rows)

    def drift_payload(ref: str, cur: str) -> Dict[str, float]:
        return {
            tp: float(summaries[ref]["psnr_clip3"][tp]["mean"] - summaries[cur]["psnr_clip3"][tp]["mean"])
            for tp in TARGET_TIMEPOINTS
        }

    meta = {
        "config_v7": args.config,
        "config_v18": args.v18_config,
        "config_v18_cap": args.v18_cap_config or args.v18_config,
        "split": args.split,
        "num_eval_slices": len(ds),
        "num_rows": len(all_rows),
        "batch_size": args.batch_size,
        "device": str(device),
        "ckpt_order": [name for name, _, _ in ckpt_specs],
        "psnr_metric": "src.utils.metrics.calc_psnr_clip3",
        "definition": "PSNR_clip3(decode_crop(z_GT), x_target) with transport rollout skipped",
        "runtime_sec": elapsed,
    }
    drift_v7_minus = {
        name: drift_payload("V7.best", name)
        for name in summaries
        if name != "V7.best"
    }
    a3_delta = None
    if "V18.step170k" in summaries and "V18-cap.last" in summaries:
        a3_delta = {
            tp: float(summaries["V18-cap.last"]["psnr_clip3"][tp]["mean"] - summaries["V18.step170k"]["psnr_clip3"][tp]["mean"])
            for tp in TARGET_TIMEPOINTS
        }
    payload = {
        "meta": meta,
        "checkpoints": summaries,
        "kl_drift_v7_minus": drift_v7_minus,
        "kl_drift_v7_minus_v18_best": drift_payload("V7.best", "V18.best"),
        "kl_drift_v7_minus_v18_last": drift_payload("V7.best", "V18.last"),
        "a3_capacity_delta_cap_minus_v18_step170k": a3_delta,
        "verdict": {
            "v18_best_normal_abs_drift": abs(drift_payload("V7.best", "V18.best")["NORMAL"]),
            "v18_best_normal_verdict": verdict_from_drift(abs(drift_payload("V7.best", "V18.best")["NORMAL"])),
            "v18_last_normal_abs_drift": abs(drift_payload("V7.best", "V18.last")["NORMAL"]),
            "v18_last_normal_verdict": verdict_from_drift(abs(drift_payload("V7.best", "V18.last")["NORMAL"])),
            "a3_capacity_normal_delta_cap_minus_v18_step170k": None if a3_delta is None else a3_delta["NORMAL"],
        },
    }

    json_path = out_dir / "KL_DRIFT_SUMMARY.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report = make_report(payload, out_dir, elapsed)
    report_path = out_dir / "KL_DRIFT_REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"[kl_drift] wrote {csv_path}", flush=True)
    print(f"[kl_drift] wrote {json_path}", flush=True)
    print(f"[kl_drift] wrote {report_path}", flush=True)
    print(json.dumps(payload["verdict"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
