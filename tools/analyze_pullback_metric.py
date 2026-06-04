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
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pet_lr.data_first_hop import PETFirstHopAligned4HopDataset  # noqa: E402
from pet_lr.model_first_hop import PETFlowDiTFirstHop  # noqa: E402

TIMEPOINTS = ["D50", "D20", "D10", "D4", "NORMAL"]
HOP_BY_DST = {
    "D20": (0, "D50", "D20"),
    "D10": (1, "D20", "D10"),
    "D4": (2, "D10", "D4"),
    "NORMAL": (3, "D4", "NORMAL"),
}
DEFAULT_MODELS = {
    "V13": {
        "config": "/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0516_runs/V13_true_image_aux_ablation/run/first_hop_224_v13_true_image_aux_off/config.yaml",
        "checkpoint": "/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0516_runs/V13_true_image_aux_ablation/run/first_hop_224_v13_true_image_aux_off/best.pt",
    },
    "A4": {
        "config": "review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08.yaml",
        "checkpoint": "/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0521_runs/A4_image_aux_lambda_08/first_hop_224_a4_image_aux_lambda_08/best.pt",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only S2 pilot for decoder pullback metric M=J^T J."
    )
    parser.add_argument("--stage", choices=["b", "a", "c"], required=True)
    parser.add_argument("--out-dir", default="review/0603/server")
    parser.add_argument("--split", choices=["val"], default="val")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n", type=int, default=16, help="Total samples across requested destination timepoints")
    parser.add_argument("--timepoints", nargs="+", default=["D20", "NORMAL"], choices=list(HOP_BY_DST))
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--batch-size", type=int, default=1, help="Reserved; S2.b uses per-sample autograd")
    parser.add_argument("--decode-mode", choices=["default", "raw"], default="default")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--subspace-iters", type=int, default=2)
    parser.add_argument("--hutchinson", type=int, default=16)
    parser.add_argument("--anisotropy-ratio-threshold", type=float, default=100.0)
    parser.add_argument("--patch-grid-ratio-threshold", type=float, default=1.20)
    parser.add_argument("--v13-config", default=DEFAULT_MODELS["V13"]["config"])
    parser.add_argument("--v13-checkpoint", default=DEFAULT_MODELS["V13"]["checkpoint"])
    parser.add_argument("--a4-config", default=DEFAULT_MODELS["A4"]["config"])
    parser.add_argument("--a4-checkpoint", default=DEFAULT_MODELS["A4"]["checkpoint"])
    parser.add_argument("--force", action="store_true", help="Allow later stages without gate artifacts; intended only for debugging")
    return parser.parse_args()


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_config(path: str) -> str:
    candidate = Path(path)
    if candidate.exists():
        return str(candidate)
    if str(path).startswith("/data_2/"):
        fallback = REPO_ROOT / "review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08.yaml"
        if fallback.exists():
            return str(fallback)
    raise FileNotFoundError(f"Config not found: {path}")


def ensure_paths(paths: Iterable[str]) -> None:
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def build_dataset(cfg: Dict[str, Any], split: str) -> PETFirstHopAligned4HopDataset:
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
        include_full_x_rollout=False,
        image_size=int(data_cfg.get("image_size", 224)),
        image_timepoints=["D50", "D20"],
        latent_mmap=True,
    )


def load_model(cfg: Dict[str, Any], checkpoint: str, device: torch.device) -> tuple[PETFlowDiTFirstHop, Dict[str, Any]]:
    model = PETFlowDiTFirstHop(cfg, device=device).to(device)
    ckpt = torch.load(checkpoint, map_location="cpu")
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
    for param in model.parameters():
        param.requires_grad_(False)
    return model, ckpt if isinstance(ckpt, dict) else {}


def select_samples(num_slices: int, destinations: Sequence[str], total_n: int, seed: int) -> list[dict[str, Any]]:
    if total_n <= 0:
        raise ValueError("--n must be positive")
    if not destinations:
        raise ValueError("--timepoints must not be empty")
    base = total_n // len(destinations)
    remainder = total_n % len(destinations)
    rng = np.random.default_rng(seed)
    samples: list[dict[str, Any]] = []
    for tp_idx, dst in enumerate(destinations):
        count = base + (1 if tp_idx < remainder else 0)
        if count <= 0:
            continue
        choices = np.asarray(sorted(rng.choice(num_slices, size=count, replace=False).tolist()), dtype=np.int64)
        hop_idx, src, real_dst = HOP_BY_DST[dst]
        for slice_idx in choices.tolist():
            samples.append({"slice_idx": int(slice_idx), "hop_idx": int(hop_idx), "src": src, "dst": real_dst})
    return samples


def to_clip3_suv(x: torch.Tensor) -> torch.Tensor:
    return ((x.float() + 1.0) * 5.0).clamp(0.0, 3.0)


def decode_clip3(model: PETFlowDiTFirstHop, z: torch.Tensor, image_size: int, decode_mode: str) -> torch.Tensor:
    apply_refiner = False if decode_mode == "raw" else None
    return to_clip3_suv(model.decode_crop(z, crop_size=image_size, apply_refiner=apply_refiner))


def predict_delta(
    model: PETFlowDiTFirstHop,
    dataset: PETFirstHopAligned4HopDataset,
    sample: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    hop_idx_int = int(sample["hop_idx"])
    src = str(sample["src"])
    dst = str(sample["dst"])
    slice_idx = int(sample["slice_idx"])
    z_src = dataset.latents[src][slice_idx : slice_idx + 1].to(device).float()
    z_gt = dataset.latents[dst][slice_idx : slice_idx + 1].to(device).float()
    t_src = torch.full((1,), float(dataset.t_map[src]), device=device)
    t_dst = torch.full((1,), float(dataset.t_map[dst]), device=device)
    hop_idx = torch.full((1,), hop_idx_int, device=device, dtype=torch.long)
    x_src = dataset.images["D50"][slice_idx : slice_idx + 1].to(device).float() if hop_idx_int == 0 else None
    with torch.no_grad():
        out = model.predict_latent_step(z_src=z_src, t_src=t_src, t_dst=t_dst, hop_idx=hop_idx, x_src_img=x_src)
        z_pred = out["z_pred"] if isinstance(out, dict) else out
    return z_gt.detach(), (z_pred - z_gt).detach()


def jvp_decoder_delta(
    model: PETFlowDiTFirstHop,
    z_gt: torch.Tensor,
    delta: torch.Tensor,
    image_size: int,
    decode_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    def fn(z: torch.Tensor) -> torch.Tensor:
        return decode_clip3(model, z, image_size=image_size, decode_mode=decode_mode)

    try:
        y, jvp = torch.autograd.functional.jvp(fn, (z_gt,), (delta,), create_graph=False, strict=False)
    except RuntimeError as exc:
        message = str(exc)
        if "BatchNorm" not in message and "functionalize" not in message:
            raise
        y = fn(z_gt)
        eps = 1.0e-3 / delta.flatten(1).norm(dim=1).clamp_min(1.0e-12).view(-1, 1, 1, 1)
        jvp = (fn(z_gt + eps * delta) - y) / eps
    return y.detach(), jvp.detach()


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    x0 = x - x.mean()
    y0 = y - y.mean()
    denom = float(np.sqrt((x0 * x0).sum() * (y0 * y0).sum()))
    if denom <= 0.0:
        return float("nan")
    return float((x0 * y0).sum() / denom)


def rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=np.float64)
    sorted_vals = a[order]
    i = 0
    while i < a.size:
        j = i + 1
        while j < a.size and sorted_vals[j] == sorted_vals[i]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x), rankdata(y))


def correlation_block(rows: Sequence[dict[str, Any]], filter_key: str | None = None, filter_value: str | None = None) -> Dict[str, float]:
    selected = [r for r in rows if filter_key is None or str(r[filter_key]) == str(filter_value)]
    q = np.asarray([float(r["q_pullback_mse_sum"]) for r in selected], dtype=np.float64)
    e = np.asarray([float(r["e_true_mse_sum"]) for r in selected], dtype=np.float64)
    l = np.asarray([float(r["l_latent_sum"]) for r in selected], dtype=np.float64)
    log_q = np.log10(np.clip(q, 1.0e-30, None))
    log_e = np.log10(np.clip(e, 1.0e-30, None))
    log_l = np.log10(np.clip(l, 1.0e-30, None))
    return {
        "n": float(len(selected)),
        "pearson_q_e": pearson(q, e),
        "pearson_l_e": pearson(l, e),
        "spearman_q_e": spearman(q, e),
        "spearman_l_e": spearman(l, e),
        "pearson_log_q_log_e": pearson(log_q, log_e),
        "pearson_log_l_log_e": pearson(log_l, log_e),
        "spearman_advantage_q_minus_l": spearman(q, e) - spearman(l, e),
        "pearson_log_advantage_q_minus_l": pearson(log_q, log_e) - pearson(log_l, log_e),
    }


def run_stage_b(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v13_config = resolve_config(args.v13_config)
    a4_config = resolve_config(args.a4_config)
    ensure_paths([v13_config, args.v13_checkpoint, a4_config, args.a4_checkpoint])
    cfg = load_yaml(a4_config)
    dataset = build_dataset(cfg, args.split)
    samples = select_samples(dataset.num_slices, args.timepoints, int(args.n), int(args.seed))
    image_size = int(cfg["data"].get("image_size", 224))

    model_specs = [
        ("V13", v13_config, args.v13_checkpoint),
        ("A4", a4_config, args.a4_checkpoint),
    ]
    rows: list[dict[str, Any]] = []
    t_start = time.time()
    for model_name, config_path, checkpoint_path in model_specs:
        print(f"[model] loading {model_name}: {checkpoint_path}", flush=True)
        model_cfg = load_yaml(config_path)
        model, ckpt_meta = load_model(model_cfg, checkpoint_path, device)
        print(f"[model] loaded {model_name}: step={ckpt_meta.get('step') if isinstance(ckpt_meta, dict) else None}", flush=True)
        for sample_idx, sample in enumerate(samples):
            z_gt, delta = predict_delta(model, dataset, sample, device)
            y_gt, j_delta = jvp_decoder_delta(model, z_gt, delta, image_size=image_size, decode_mode=args.decode_mode)
            with torch.no_grad():
                y_pred = decode_clip3(model, z_gt + delta, image_size=image_size, decode_mode=args.decode_mode)
            q = float(j_delta.square().sum().item())
            q_mean = float(j_delta.square().mean().item())
            e = float((y_pred - y_gt).square().sum().item())
            e_mean = float((y_pred - y_gt).square().mean().item())
            l = float(delta.square().sum().item())
            l_mean = float(delta.square().mean().item())
            rows.append(
                {
                    "model": model_name,
                    "sample_id": int(sample_idx),
                    "slice_idx": int(sample["slice_idx"]),
                    "hop_idx": int(sample["hop_idx"]),
                    "src": str(sample["src"]),
                    "dst": str(sample["dst"]),
                    "q_pullback_mse_sum": q,
                    "q_pullback_mse_mean": q_mean,
                    "e_true_mse_sum": e,
                    "e_true_mse_mean": e_mean,
                    "l_latent_sum": l,
                    "l_latent_mean": l_mean,
                    "sqrt_q_over_sqrt_e": math.sqrt(q / e) if e > 0 else float("nan"),
                    "delta_norm": math.sqrt(l),
                    "jdelta_norm": math.sqrt(q),
                    "true_pixel_delta_norm": math.sqrt(e),
                }
            )
            print(
                f"[S2.b] {model_name} {sample_idx + 1:02d}/{len(samples)} "
                f"{sample['dst']} slice={sample['slice_idx']} q={q_mean:.4e} e={e_mean:.4e} l={l_mean:.4e}",
                flush=True,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    correlations: Dict[str, Any] = {"overall": correlation_block(rows)}
    for model_name in ["V13", "A4"]:
        correlations[f"model={model_name}"] = correlation_block(rows, "model", model_name)
    for dst in args.timepoints:
        correlations[f"dst={dst}"] = correlation_block(rows, "dst", dst)

    gate = correlations["overall"]
    spearman_adv = float(gate["spearman_advantage_q_minus_l"])
    pearson_log_adv = float(gate["pearson_log_advantage_q_minus_l"])
    pass_gate = bool(
        math.isfinite(spearman_adv)
        and spearman_adv >= 0.15
        and math.isfinite(float(gate["pearson_log_q_log_e"]))
        and float(gate["pearson_log_q_log_e"]) > 0.5
    )
    verdict = {
        "stage": "S2.b-pilot",
        "pass_gate": pass_gate,
        "decision": "continue_to_S2a" if pass_gate else "stop_M_quantitative_narrative",
        "rule": "PASS if overall Spearman(q,e)-Spearman(l,e) >= 0.15 and log-log Pearson(q,e) > 0.5",
        "spearman_advantage_q_minus_l": spearman_adv,
        "pearson_log_advantage_q_minus_l": pearson_log_adv,
    }
    payload = {
        "meta": {
            "stage": "b",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "runtime_sec": float(time.time() - t_start),
            "device": str(device),
            "split": args.split,
            "n_requested": int(args.n),
            "num_samples": len(samples),
            "num_rows": len(rows),
            "timepoints": list(args.timepoints),
            "decode_mode": args.decode_mode,
            "image_domain": "clip3 SUV [0,3]",
            "protocol": "single-step GT source latent; q=||J_G delta||^2 by matrix-free jvp; e=||G(z+delta)-G(z)||^2; l=||delta||^2",
            "v13_config": v13_config,
            "v13_checkpoint": args.v13_checkpoint,
            "a4_config": a4_config,
            "a4_checkpoint": args.a4_checkpoint,
        },
        "verdict": verdict,
        "correlations": correlations,
        "rows": rows,
    }
    write_stage_b_outputs(out_dir, payload)
    return payload


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return str(value)
        return f"{float(value):.{digits}f}"
    return str(value)


def write_stage_b_outputs(out_dir: Path, payload: Dict[str, Any]) -> None:
    json_path = out_dir / "S2b_pilot_correlation_20260604.json"
    csv_path = out_dir / "S2b_pilot_correlation_20260604.csv"
    md_path = out_dir / "S2b_pilot_correlation_20260604.md"
    verdict_path = out_dir / "S2_PILOT_VERDICT_20260604.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(csv_path, payload["rows"])

    gate = payload["correlations"]["overall"]
    verdict = payload["verdict"]
    lines = [
        "# S2.b Pilot Correlation Gate",
        "",
        f"日期: {payload['meta']['created_at']}",
        "",
        "## Verdict",
        "",
        f"- Gate: {'PASS' if verdict['pass_gate'] else 'FAIL'}",
        f"- Decision: `{verdict['decision']}`",
        f"- Rule: {verdict['rule']}",
        f"- Spearman advantage `corr(q,e)-corr(l,e)`: {fmt(verdict['spearman_advantage_q_minus_l'], 4)}",
        f"- Log-Pearson advantage `corr(log q,log e)-corr(log l,log e)`: {fmt(verdict['pearson_log_advantage_q_minus_l'], 4)}",
        "",
        "## Overall Correlations",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key, value in gate.items():
        lines.append(f"| `{key}` | {fmt(value, 6)} |")
    lines.extend(["", "## Group Correlations", "", "| group | n | Pearson(q,e) | Pearson(l,e) | Spearman(q,e) | Spearman(l,e) | Spearman adv |", "|---|---:|---:|---:|---:|---:|---:|"])
    for group, block in payload["correlations"].items():
        lines.append(
            f"| `{group}` | {fmt(block['n'], 0)} | {fmt(block['pearson_q_e'], 4)} | {fmt(block['pearson_l_e'], 4)} | "
            f"{fmt(block['spearman_q_e'], 4)} | {fmt(block['spearman_l_e'], 4)} | {fmt(block['spearman_advantage_q_minus_l'], 4)} |"
        )
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            f"- Rows: `{payload['meta']['num_rows']}` = 2 models × {payload['meta']['num_samples']} samples.",
            "- `q = δᵀMδ = ||J_G δ||²` computed by matrix-free JVP in clip3 SUV [0,3] domain.",
            "- `e = ||G(z_gt + δ) - G(z_gt)||²` is the true nonlinear decode-domain error in the same clip3 domain.",
            "- `l = ||δ||²` is the latent-space baseline.",
            "- Single-step protocol uses GT source latent; hop0 receives GT D50 image conditioning; NORMAL uses GT D4 source latent.",
            "",
            "## Artifacts",
            "",
            f"- JSON: `{json_path}`",
            f"- CSV: `{csv_path}`",
            f"- Report: `{md_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    final_lines = [
        "# S2 Pilot Verdict",
        "",
        f"日期: {payload['meta']['created_at']}",
        "",
        "## Gate Status",
        "",
        f"- S2.b-pilot: {'PASS' if verdict['pass_gate'] else 'FAIL'} — `{verdict['decision']}`.",
    ]
    if verdict["pass_gate"]:
        final_lines.extend(
            [
                "- Next action: run S2.a-pilot top-spectrum / Hutchinson trace before making any M anisotropy claim.",
                "- Paper wording for now: M pullback is supported by the S2.b correlation gate, pending spectrum validation.",
            ]
        )
    else:
        final_lines.extend(
            [
                "- Next action: stop S2.a/S2.c under the pre-registered gate.",
                "- Paper wording: do not use M=JᵀJ as a quantitative mechanism; keep seam as empirical observation or qualitative hypothesis only.",
            ]
        )
    final_lines.extend(["", "## Evidence", "", f"- S2.b report: `{md_path}`", f"- S2.b JSON: `{json_path}`"])
    verdict_path.write_text("\n".join(final_lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "report": str(md_path), "verdict": str(verdict_path), "gate": verdict}, indent=2, ensure_ascii=False))


def check_s2b_gate(out_dir: Path, force: bool = False) -> Dict[str, Any]:
    gate_json = out_dir / "S2b_pilot_correlation_20260604.json"
    if not gate_json.exists():
        if force:
            return {}
        raise SystemExit("S2.b gate artifact missing; run --stage b first.")
    payload = json.loads(gate_json.read_text(encoding="utf-8"))
    if not bool(payload.get("verdict", {}).get("pass_gate", False)) and not force:
        raise SystemExit("S2.b gate failed; S2.a/S2.c must not run under task protocol.")
    return payload


def check_s2a_gate(out_dir: Path, force: bool = False) -> Dict[str, Any]:
    gate_json = out_dir / "S2a_pilot_spectrum_20260604.json"
    if not gate_json.exists():
        if force:
            return {}
        raise SystemExit("S2.a gate artifact missing; run --stage a first.")
    payload = json.loads(gate_json.read_text(encoding="utf-8"))
    if not bool(payload.get("verdict", {}).get("anisotropy_pass", False)) and not force:
        raise SystemExit("S2.a did not show sufficient anisotropy; S2.c must not run under task protocol.")
    return payload


def get_destination_latent(dataset: PETFirstHopAligned4HopDataset, sample: dict[str, Any], device: torch.device) -> torch.Tensor:
    dst = str(sample["dst"])
    slice_idx = int(sample["slice_idx"])
    return dataset.latents[dst][slice_idx : slice_idx + 1].to(device).float()


def decoder_mvp(
    model: PETFlowDiTFirstHop,
    z_gt: torch.Tensor,
    vector: torch.Tensor,
    image_size: int,
    decode_mode: str,
) -> tuple[torch.Tensor, float]:
    def fn(z: torch.Tensor) -> torch.Tensor:
        return decode_clip3(model, z, image_size=image_size, decode_mode=decode_mode)

    z_req = z_gt.detach().clone().requires_grad_(True)
    v = vector.detach().to(z_req.device, dtype=z_req.dtype)
    with torch.enable_grad():
        y = fn(z_req)
        _, jv = torch.autograd.functional.jvp(fn, (z_req,), (v,), create_graph=False, strict=False)
        scalar = (y * jv.detach()).sum()
        grad = torch.autograd.grad(scalar, z_req, retain_graph=False, create_graph=False)[0]
    vt_m_v = float((v * grad.detach()).sum().item())
    return grad.detach(), vt_m_v


def orthonormalize_columns(mat: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(mat.float(), mode="reduced")
    return q


def estimate_top_spectrum(
    model: PETFlowDiTFirstHop,
    z_gt: torch.Tensor,
    image_size: int,
    decode_mode: str,
    top_k: int,
    subspace_iters: int,
    hutchinson: int,
    seed: int,
) -> Dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if hutchinson <= 0:
        raise ValueError("hutchinson must be positive")
    shape = tuple(z_gt.shape)
    latent_dim = int(z_gt.numel())
    generator = torch.Generator(device=z_gt.device)
    generator.manual_seed(int(seed))
    rank = int(top_k)
    q = torch.randn((latent_dim, rank), device=z_gt.device, generator=generator, dtype=torch.float32)
    q = orthonormalize_columns(q)
    mv_calls = 0

    for _ in range(int(subspace_iters)):
        y_cols = []
        for col in range(rank):
            mv, _ = decoder_mvp(model, z_gt, q[:, col].reshape(shape), image_size, decode_mode)
            y_cols.append(mv.flatten())
            mv_calls += 1
        q = orthonormalize_columns(torch.stack(y_cols, dim=1))

    mq_cols = []
    for col in range(rank):
        mv, _ = decoder_mvp(model, z_gt, q[:, col].reshape(shape), image_size, decode_mode)
        mq_cols.append(mv.flatten())
        mv_calls += 1
    mq = torch.stack(mq_cols, dim=1)
    small = q.T @ mq
    small = 0.5 * (small + small.T)
    eigvals, eigvecs = torch.linalg.eigh(small)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order][:top_k].clamp_min(0.0)
    eigvecs = eigvecs[:, order][:, :top_k]
    ritz = q @ eigvecs

    trace_vals = []
    for _ in range(int(hutchinson)):
        signs = torch.randint(0, 2, shape, device=z_gt.device, generator=generator, dtype=torch.int8).float()
        signs = signs.mul_(2.0).sub_(1.0)
        _, vt_m_v = decoder_mvp(model, z_gt, signs, image_size, decode_mode)
        trace_vals.append(vt_m_v)
        mv_calls += 1
    trace_arr = np.asarray(trace_vals, dtype=np.float64)
    trace_mean = float(trace_arr.mean())
    trace_std = float(trace_arr.std())
    eig_np = eigvals.detach().cpu().numpy().astype(np.float64)
    sigma_np = np.sqrt(np.clip(eig_np, 0.0, None))
    top1 = float(eig_np[0] / trace_mean) if trace_mean > 0 else float("nan")
    topk = float(eig_np.sum() / trace_mean) if trace_mean > 0 else float("nan")
    return {
        "latent_dim": latent_dim,
        "eigvals": eig_np.tolist(),
        "singular_values": sigma_np.tolist(),
        "trace_hutchinson_mean": trace_mean,
        "trace_hutchinson_std": trace_std,
        "top1_energy_concentration": top1,
        "topk_energy_concentration": topk,
        "uniform_top1_baseline": 1.0 / float(latent_dim),
        "uniform_topk_baseline": float(top_k) / float(latent_dim),
        "top1_over_uniform": top1 * float(latent_dim) if math.isfinite(top1) else float("nan"),
        "topk_over_uniform": topk / (float(top_k) / float(latent_dim)) if math.isfinite(topk) else float("nan"),
        "mv_calls": mv_calls,
        "ritz_vectors": ritz.detach(),
    }


def build_stage_a_rows(args: argparse.Namespace) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    device = torch.device(args.device)
    a4_config = resolve_config(args.a4_config)
    ensure_paths([a4_config, args.a4_checkpoint])
    cfg = load_yaml(a4_config)
    dataset = build_dataset(cfg, args.split)
    samples = select_samples(dataset.num_slices, args.timepoints, int(args.n), int(args.seed))
    image_size = int(cfg["data"].get("image_size", 224))
    model, ckpt_meta = load_model(load_yaml(a4_config), args.a4_checkpoint, device)
    rows: list[Dict[str, Any]] = []
    vectors_by_sample: Dict[str, torch.Tensor] = {}
    for sample_idx, sample in enumerate(samples):
        z_gt = get_destination_latent(dataset, sample, device)
        spec = estimate_top_spectrum(
            model=model,
            z_gt=z_gt,
            image_size=image_size,
            decode_mode=args.decode_mode,
            top_k=int(args.top_k),
            subspace_iters=int(args.subspace_iters),
            hutchinson=int(args.hutchinson),
            seed=int(args.seed) + sample_idx * 1009,
        )
        vectors_by_sample[str(sample_idx)] = spec.pop("ritz_vectors")
        row: Dict[str, Any] = {
            "sample_id": int(sample_idx),
            "slice_idx": int(sample["slice_idx"]),
            "hop_idx": int(sample["hop_idx"]),
            "src": str(sample["src"]),
            "dst": str(sample["dst"]),
            "trace_hutchinson_mean": spec["trace_hutchinson_mean"],
            "trace_hutchinson_std": spec["trace_hutchinson_std"],
            "top1_energy_concentration": spec["top1_energy_concentration"],
            "topk_energy_concentration": spec["topk_energy_concentration"],
            "top1_over_uniform": spec["top1_over_uniform"],
            "topk_over_uniform": spec["topk_over_uniform"],
            "uniform_top1_baseline": spec["uniform_top1_baseline"],
            "uniform_topk_baseline": spec["uniform_topk_baseline"],
            "mv_calls": spec["mv_calls"],
        }
        for i, value in enumerate(spec["singular_values"], start=1):
            row[f"sigma_{i}"] = float(value)
        for i, value in enumerate(spec["eigvals"], start=1):
            row[f"eig_{i}"] = float(value)
        rows.append(row)
        print(
            f"[S2.a] {sample_idx + 1:02d}/{len(samples)} {sample['dst']} slice={sample['slice_idx']} "
            f"top1/tr={row['top1_energy_concentration']:.4e} topk/tr={row['topk_energy_concentration']:.4e} "
            f"topk/uniform={row['topk_over_uniform']:.1f}",
            flush=True,
        )
    meta = {"cfg": cfg, "dataset": dataset, "model": model, "image_size": image_size, "samples": samples, "vectors_by_sample": vectors_by_sample, "checkpoint_step": ckpt_meta.get("step") if isinstance(ckpt_meta, dict) else None}
    return rows, meta


def run_stage_a(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    check_s2b_gate(out_dir, force=bool(args.force))
    t0 = time.time()
    rows, _meta_runtime = build_stage_a_rows(args)
    topk_ratios = np.asarray([float(r["topk_over_uniform"]) for r in rows], dtype=np.float64)
    top1_ratios = np.asarray([float(r["top1_over_uniform"]) for r in rows], dtype=np.float64)
    topk_conc = np.asarray([float(r["topk_energy_concentration"]) for r in rows], dtype=np.float64)
    threshold = float(args.anisotropy_ratio_threshold)
    anisotropy_pass = bool(np.nanmedian(topk_ratios) >= threshold and np.nanmedian(top1_ratios) >= threshold)
    verdict = {
        "stage": "S2.a-pilot",
        "anisotropy_pass": anisotropy_pass,
        "decision": "continue_to_S2c" if anisotropy_pass else "stop_patchgrid_causal_claim",
        "rule": f"PASS if median top1/topk energy over uniform baseline are both >= {threshold:g}x",
        "median_top1_over_uniform": float(np.nanmedian(top1_ratios)),
        "median_topk_over_uniform": float(np.nanmedian(topk_ratios)),
        "median_topk_energy_concentration": float(np.nanmedian(topk_conc)),
    }
    payload = {
        "meta": {
            "stage": "a",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "runtime_sec": float(time.time() - t0),
            "device": str(args.device),
            "n_requested": int(args.n),
            "num_samples": len(rows),
            "timepoints": list(args.timepoints),
            "decode_mode": args.decode_mode,
            "top_k": int(args.top_k),
            "subspace_iters": int(args.subspace_iters),
            "hutchinson": int(args.hutchinson),
            "a4_config": resolve_config(args.a4_config),
            "a4_checkpoint": args.a4_checkpoint,
        },
        "verdict": verdict,
        "rows": rows,
    }
    write_stage_a_outputs(out_dir, payload)
    return payload


def write_stage_a_outputs(out_dir: Path, payload: Dict[str, Any]) -> None:
    json_path = out_dir / "S2a_pilot_spectrum_20260604.json"
    csv_path = out_dir / "S2a_pilot_spectrum_20260604.csv"
    md_path = out_dir / "S2a_pilot_spectrum_20260604.md"
    verdict_path = out_dir / "S2_PILOT_VERDICT_20260604.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(csv_path, payload["rows"])
    verdict = payload["verdict"]
    lines = [
        "# S2.a Pilot Decoder Top Spectrum",
        "",
        f"日期: {payload['meta']['created_at']}",
        "",
        "## Verdict",
        "",
        f"- Gate: {'PASS' if verdict['anisotropy_pass'] else 'FAIL'}",
        f"- Decision: `{verdict['decision']}`",
        f"- Rule: {verdict['rule']}",
        f"- Median top-1 over uniform: {fmt(verdict['median_top1_over_uniform'], 2)}×",
        f"- Median top-{payload['meta']['top_k']} over uniform: {fmt(verdict['median_topk_over_uniform'], 2)}×",
        f"- Median top-{payload['meta']['top_k']} energy concentration: {fmt(verdict['median_topk_energy_concentration'], 6)}",
        "",
        "## Per-Sample Summary",
        "",
        "| sample | dst | slice | trace | top1/tr | topk/tr | top1/uniform | topk/uniform | σ1 | σ8 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        sigma8 = row.get("sigma_8", float("nan"))
        lines.append(
            f"| {row['sample_id']} | `{row['dst']}` | {row['slice_idx']} | {fmt(row['trace_hutchinson_mean'], 4)} | "
            f"{fmt(row['top1_energy_concentration'], 6)} | {fmt(row['topk_energy_concentration'], 6)} | "
            f"{fmt(row['top1_over_uniform'], 1)} | {fmt(row['topk_over_uniform'], 1)} | {fmt(row.get('sigma_1', float('nan')), 5)} | {fmt(sigma8, 5)} |"
        )
    lines.extend(["", "## Artifacts", "", f"- JSON: `{json_path}`", f"- CSV: `{csv_path}`", f"- Report: `{md_path}`"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    previous = verdict_path.read_text(encoding="utf-8") if verdict_path.exists() else "# S2 Pilot Verdict\n\n"
    addition = [
        "",
        "## S2.a Update",
        "",
        f"- S2.a-pilot: {'PASS' if verdict['anisotropy_pass'] else 'FAIL'} — `{verdict['decision']}`.",
        f"- Median top-{payload['meta']['top_k']} over uniform: {fmt(verdict['median_topk_over_uniform'], 2)}×.",
        f"- Evidence: `{md_path}`",
    ]
    verdict_path.write_text(previous.rstrip() + "\n" + "\n".join(addition) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "report": str(md_path), "verdict": str(verdict_path), "gate": verdict}, indent=2, ensure_ascii=False))


def patch_grid_mask(size: int = 224, patch: int = 14, band: int = 1, device: torch.device | None = None) -> torch.Tensor:
    mask = torch.zeros((size, size), dtype=torch.bool, device=device)
    for boundary in range(patch, size, patch):
        lo = max(0, boundary - band)
        hi = min(size, boundary + band + 1)
        mask[lo:hi, :] = True
        mask[:, lo:hi] = True
    return mask.view(1, 1, size, size)


def jvp_response(model: PETFlowDiTFirstHop, z_gt: torch.Tensor, vector: torch.Tensor, image_size: int, decode_mode: str) -> torch.Tensor:
    def fn(z: torch.Tensor) -> torch.Tensor:
        return decode_clip3(model, z, image_size=image_size, decode_mode=decode_mode)
    _, jv = torch.autograd.functional.jvp(fn, (z_gt.detach(),), (vector.detach(),), create_graph=False, strict=False)
    return jv.detach()


def run_stage_c(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    check_s2b_gate(out_dir, force=bool(args.force))
    check_s2a_gate(out_dir, force=bool(args.force))
    t0 = time.time()
    rows_a, meta_runtime = build_stage_a_rows(args)
    model = meta_runtime["model"]
    dataset = meta_runtime["dataset"]
    samples = meta_runtime["samples"]
    vectors_by_sample = meta_runtime["vectors_by_sample"]
    image_size = int(meta_runtime["image_size"])
    device = torch.device(args.device)
    mask = patch_grid_mask(size=image_size, patch=14, band=1, device=device).float()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed) + 99173)
    rows: list[Dict[str, Any]] = []
    for sample_idx, sample in enumerate(samples):
        z_gt = get_destination_latent(dataset, sample, device)
        vectors = vectors_by_sample[str(sample_idx)]
        shape = tuple(z_gt.shape)
        for direction_idx in range(int(args.top_k)):
            vec = vectors[:, direction_idx].reshape(shape)
            resp = jvp_response(model, z_gt, vec, image_size, args.decode_mode)
            energy = resp.square()
            top_ratio = float((energy * mask).sum().item() / energy.sum().clamp_min(1.0e-12).item())
            rand = torch.randn(shape, device=device, generator=generator)
            rand = rand / rand.flatten().norm().clamp_min(1.0e-12)
            rand_resp = jvp_response(model, z_gt, rand, image_size, args.decode_mode)
            rand_energy = rand_resp.square()
            rand_ratio = float((rand_energy * mask).sum().item() / rand_energy.sum().clamp_min(1.0e-12).item())
            rows.append({
                "sample_id": int(sample_idx),
                "slice_idx": int(sample["slice_idx"]),
                "dst": str(sample["dst"]),
                "direction_idx": int(direction_idx),
                "top_patchgrid_energy_frac": top_ratio,
                "random_patchgrid_energy_frac": rand_ratio,
                "top_minus_random": top_ratio - rand_ratio,
                "top_over_random": top_ratio / rand_ratio if rand_ratio > 0 else float("nan"),
            })
        print(f"[S2.c] {sample_idx + 1:02d}/{len(samples)} {sample['dst']} slice={sample['slice_idx']}", flush=True)
    top = np.asarray([float(r["top_patchgrid_energy_frac"]) for r in rows], dtype=np.float64)
    rnd = np.asarray([float(r["random_patchgrid_energy_frac"]) for r in rows], dtype=np.float64)
    ratio = float(top.mean() / rnd.mean()) if rnd.mean() > 0 else float("nan")
    pass_gate = bool(math.isfinite(ratio) and ratio >= float(args.patch_grid_ratio_threshold))
    verdict = {
        "stage": "S2.c-pilot",
        "patchgrid_pass": pass_gate,
        "decision": "top_M_patchgrid_alignment_supported" if pass_gate else "patchgrid_causal_claim_not_supported",
        "rule": f"PASS if mean top-M patch-grid energy / random control >= {float(args.patch_grid_ratio_threshold):.2f}",
        "mean_top_patchgrid_energy_frac": float(top.mean()),
        "mean_random_patchgrid_energy_frac": float(rnd.mean()),
        "top_over_random_mean_ratio": ratio,
    }
    payload = {
        "meta": {
            "stage": "c",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "runtime_sec": float(time.time() - t0),
            "device": str(args.device),
            "num_rows": len(rows),
            "num_samples": len(samples),
            "top_k": int(args.top_k),
            "decode_mode": args.decode_mode,
        },
        "verdict": verdict,
        "rows": rows,
    }
    write_stage_c_outputs(out_dir, payload)
    return payload


def write_stage_c_outputs(out_dir: Path, payload: Dict[str, Any]) -> None:
    json_path = out_dir / "S2c_pilot_patchgrid_20260604.json"
    csv_path = out_dir / "S2c_pilot_patchgrid_20260604.csv"
    md_path = out_dir / "S2c_pilot_patchgrid_20260604.md"
    verdict_path = out_dir / "S2_PILOT_VERDICT_20260604.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(csv_path, payload["rows"])
    verdict = payload["verdict"]
    lines = [
        "# S2.c Pilot Patch-Grid Alignment",
        "",
        f"日期: {payload['meta']['created_at']}",
        "",
        "## Verdict",
        "",
        f"- Gate: {'PASS' if verdict['patchgrid_pass'] else 'FAIL'}",
        f"- Decision: `{verdict['decision']}`",
        f"- Rule: {verdict['rule']}",
        f"- Mean top-M patch-grid energy fraction: {fmt(verdict['mean_top_patchgrid_energy_frac'], 6)}",
        f"- Mean random patch-grid energy fraction: {fmt(verdict['mean_random_patchgrid_energy_frac'], 6)}",
        f"- Top/random ratio: {fmt(verdict['top_over_random_mean_ratio'], 4)}",
        "",
        "## Artifacts",
        "",
        f"- JSON: `{json_path}`",
        f"- CSV: `{csv_path}`",
        f"- Report: `{md_path}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    previous = verdict_path.read_text(encoding="utf-8") if verdict_path.exists() else "# S2 Pilot Verdict\n\n"
    addition = [
        "",
        "## S2.c Update",
        "",
        f"- S2.c-pilot: {'PASS' if verdict['patchgrid_pass'] else 'FAIL'} — `{verdict['decision']}`.",
        f"- Top/random patch-grid energy ratio: {fmt(verdict['top_over_random_mean_ratio'], 4)}.",
        f"- Evidence: `{md_path}`",
    ]
    verdict_path.write_text(previous.rstrip() + "\n" + "\n".join(addition) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "report": str(md_path), "verdict": str(verdict_path), "gate": verdict}, indent=2, ensure_ascii=False))


def run_unimplemented_stage(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    gate_json = out_dir / "S2b_pilot_correlation_20260604.json"
    if not args.force:
        if not gate_json.exists():
            raise SystemExit("S2.b gate artifact missing; run --stage b first.")
        payload = json.loads(gate_json.read_text(encoding="utf-8"))
        if not bool(payload.get("verdict", {}).get("pass_gate", False)):
            raise SystemExit("S2.b gate failed; S2.a/S2.c must not run under task protocol.")
    raise SystemExit(
        "S2.a/S2.c implementation is intentionally deferred until S2.b passes in this run. "
        "This prevents spending GPU on gated stages before the correlation evidence is known."
    )


def main() -> None:
    args = parse_args()
    if args.stage == "b":
        run_stage_b(args)
    elif args.stage == "a":
        run_stage_a(args)
    elif args.stage == "c":
        run_stage_c(args)
    else:
        run_unimplemented_stage(args)


if __name__ == "__main__":
    main()
