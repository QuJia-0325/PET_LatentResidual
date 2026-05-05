#!/usr/bin/env python3
"""
sigma_norm_golden_test.py — Option F (Round 4 same-process algebraic golden test)

Purpose
-------
Decisively discriminate between three competing root-cause hypotheses for the
sigma-normalize A/B sanity failure (A=raw rollout, B=sigma-normalize rollout;
relative drift on val_chain_normal_mse=12.73% at step 20000, on val_pair_total
=10.16%):

  (alpha) Kernel non-determinism amplified through mix_latent
  (beta)  FP32 reduction-order asymmetry from B's `step_loss / float(n_j)`
  (zeta)  Algebraic mis-implementation of the preserve_v6_sum invariant

Same-process golden test design (Round 4 prompt Q3 Option F):
  1. Load V6@step_160000.pt in ONE Python process
  2. model.eval() + freeze RNG + force math-SDPA + use_deterministic_algorithms(True)
  3. Build ONE fixed batch (deterministic seed)
  4. Compute A path total = `(w_v6 * raw_step_losses).sum() / sum(w_v6)`
  5. Compute B path total = `((w_v6 * n) * (raw_step_losses / n)).sum() / sum(w_v6 * n)`
     where n_j is computed via the trainer's _compute_sigma_dt_normalizers under
     mode=sigma_dt_squared, relative_to=preserve_v6_sum, anchor=[0.5, 2.0, 1.5, 1.0]
  6. Assert |A.loss_total - B.loss_total| <= rel_tolerance

Numerical tolerance justification
---------------------------------
A and B compute the same algebra in different operation order. Per-step:
  A: w_j * loss_j           (one mul per hop)
  B: (loss_j / n_j) * (w_j * n_j)  (one div + one mul per hop)
FP32 reduction-order asymmetry under exact-arithmetic-equivalent paths is
typically 1-4 ulp absolute, where 1 ulp at scale |w_j*loss_j| ~ 1e-3 is
1.2e-10. Per-hop drift bounded by ~4e-10; sum over 4 hops bounded by ~1.6e-9;
final loss_total (after division by sum_weights ~ 5.0) bounded by ~3.2e-10
absolute. The tolerance 1e-7 gives ~250x safety margin and unambiguously
distinguishes "FP32 noise" (drift < 1e-7) from "algebra broken" (drift > 1e-3).

Outcome decision tree (Plan F section 11)
----------------------------------------
- PASS (rel_drift < 1e-7): algebra correct in implementation; (beta) and (zeta)
  FALSIFIED at inference; the 12.73% sanity failure is fully attributable to
  (gamma) trajectory drift compounded from (alpha) kernel non-determinism over
  20K SGD steps. Plan F's switch to raw-rollout was correct; H bundle becomes
  the mainline mitigation strategy.
- FAIL (rel_drift > 1e-3): algebra broken in implementation; (zeta) CONFIRMED.
  Defer V7 launch; investigate _compute_sigma_dt_normalizers and the rollout
  step_normalizers injection in pet_lr/rollout_first_hop.py:99-104.
- AMBIGUOUS (1e-7 <= rel_drift <= 1e-3): run Option G (paired raw-vs-sigma full-val
  evaluator on V6@160K across 7403 slices) for distributional bound; if max abs
  drift on full val < 1e-4, accept algebra at inference.

Usage
-----
    python3 review/0505/local/scripts/sigma_norm_golden_test.py \
        --config /path/to/v6_baseline.yaml \
        --ckpt /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v6_transport_first/step_160000.pt \
        --output-json review/0505/operator/sigma_norm_golden_test_result.json

Or, for a real validation batch (slower but more representative):
    python3 review/0505/local/scripts/sigma_norm_golden_test.py \
        --config /path/to/v6_baseline.yaml \
        --ckpt .../step_160000.pt \
        --use-real-batch \
        --output-json .../sigma_norm_golden_test_result.json

Exit codes
----------
  0 = PASS (algebra verified, drift < rel_tolerance)
  3 = AMBIGUOUS (drift in [rel_tolerance, 1e-3])
  4 = FAIL (algebra broken, drift > 1e-3)
  2 = setup error (missing config / ckpt / dependency)

This script is a P0 pre-launch action per Plan F section 11. It must be run on
V6@step_160000.pt before V7 launches. Cost: ~30 GPU-seconds on an A100.

Author: external review consensus, 2026-05-06
Audit: review/plan/ROUND4_EXTERNAL_CONSENSUS_20260506.md section 4 (gate d)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict


def _abort(msg: str, code: int = 2) -> None:
    print(f"[sigma_norm_golden_test] FAIL: {msg}", file=sys.stderr)
    sys.exit(code)


def _force_deterministic(seed: int, device: "torch.device") -> None:
    """Match Round 4 prompt Option F protocol: math-SDPA + strict deterministic."""
    import torch

    # Force math-SDPA backend (deterministic), disable Memory-Efficient and Flash
    # so the (alpha) kernel non-determinism source is eliminated for this test.
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(False)

    # Force strict (warn_only=False) deterministic algorithms — if anything is
    # still nondeterministic this will RAISE at runtime.
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
    except TypeError:
        # Older PyTorch without warn_only kwarg
        torch.use_deterministic_algorithms(True)

    # FP32 matmul highest precision (no TF32 entropy)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")

    # cuBLAS deterministic workspace
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def _build_synthetic_batch(
    model: "torch.nn.Module",
    rollout_times: list,
    seed: int,
    device: "torch.device",
) -> Dict[str, "torch.Tensor"]:
    """Build a deterministic fixed batch for the algebra test.

    Algebra equivalence does NOT depend on which batch is used, so any fixed
    deterministic tensor works. We use B=2, T=len(rollout_times) so the batch
    matches the rollout API expectations.
    """
    import torch

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    latent_channels = int(model.latent_channels)
    latent_size = int(model.latent_size)
    batch_size = 2
    num_tp = len(rollout_times)

    # z_rollout: [B, T, C, H, W]
    z_rollout = torch.randn(
        batch_size, num_tp, latent_channels, latent_size, latent_size,
        generator=g, dtype=torch.float32,
    ).to(device)

    # x_rollout_first (pixel image at hop0 source): [B, 1, image_size, image_size]
    image_size = int(getattr(model, "image_size", 224))
    x_rollout_first = torch.randn(
        batch_size, 1, image_size, image_size,
        generator=g, dtype=torch.float32,
    ).to(device)

    return {
        "z_rollout": z_rollout,
        "x_rollout_first": x_rollout_first,
    }


def _build_real_batch(
    cfg: Dict[str, Any],
    device: "torch.device",
) -> Dict[str, "torch.Tensor"]:
    """Build one batch from the val dataset (--use-real-batch mode)."""
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    # delayed import; only needed when --use-real-batch
    from pet_lr.dataset_first_hop import PETFirstHopAligned4HopDataset  # type: ignore
    from torch.utils.data import DataLoader

    data_cfg = cfg["data"]
    latent_path = os.path.join(data_cfg["latent_dir"], "latents_val.pt")
    alignment_audit_json = str(data_cfg.get("alignment_audit_json", "")).strip() or None
    val_set = PETFirstHopAligned4HopDataset(
        latent_path=latent_path,
        raw_data_dir=data_cfg["raw_data_dir"],
        split="val",
        clamp_max=float(data_cfg.get("clamp_max", 10.0)),
        t_map=data_cfg["t_map"],
        rollout_timepoints=data_cfg.get("rollout_timepoints", ["D50", "D20", "D10", "D4", "NORMAL"]),
        verify_alignment=bool(data_cfg.get("verify_alignment", True)),
        alignment_check_num_samples=int(data_cfg.get("alignment_check_num_samples", 16)),
        alignment_audit_json=alignment_audit_json,
        include_x_rollout_first=True,
        include_full_x_rollout=True,
        image_size=int(data_cfg.get("image_size", 224)),
    )
    loader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=0)
    raw = next(iter(loader))
    batch: Dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
        else:
            batch[k] = v
    return batch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to V6 baseline yaml")
    parser.add_argument("--ckpt", required=True, help="Path to V6@step_160000.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rel-tolerance", type=float, default=1e-7,
        help="Relative drift tolerance for PASS verdict (default 1e-7; ~250x FP32 reduction-order safety margin)",
    )
    parser.add_argument(
        "--ambiguous-upper", type=float, default=1e-3,
        help="Drift threshold above which verdict is FAIL (default 1e-3)",
    )
    parser.add_argument(
        "--use-real-batch", action="store_true",
        help="Use one batch from the val dataset instead of synthetic deterministic noise",
    )
    parser.add_argument(
        "--output-json", default=None,
        help="Optional path to write result JSON (for committing under review/0505/operator/)",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        _abort(f"config not found: {cfg_path}")
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not ckpt_path.exists():
        _abort(f"checkpoint not found: {ckpt_path}")

    try:
        import torch
        import yaml
    except ImportError as exc:  # pragma: no cover
        _abort(f"missing dependency: {exc}")

    cfg = yaml.safe_load(cfg_path.read_text())

    # Repository root so we can import pet_lr.
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    try:
        from pet_lr.model_first_hop import PETFlowDiTFirstHop  # type: ignore
        from pet_lr.rollout_first_hop import rollout_multistep_losses_first_hop  # type: ignore
        # _compute_sigma_dt_normalizers is a private helper inside train_first_hop —
        # we re-import it via module path.
        import importlib.util as _util
        _spec = _util.spec_from_file_location(
            "_train_first_hop_module",
            str(repo_root / "train_first_hop.py"),
        )
        if _spec is None or _spec.loader is None:
            _abort("could not load train_first_hop.py for _compute_sigma_dt_normalizers")
        _tfh = _util.module_from_spec(_spec)  # type: ignore[arg-type]
        _spec.loader.exec_module(_tfh)  # type: ignore[union-attr]
        _compute_sigma_dt_normalizers = _tfh._compute_sigma_dt_normalizers  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover
        _abort(f"failed to import pet_lr modules: {exc}")

    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
        print("[sigma_norm_golden_test] WARN: CUDA unavailable; running on CPU "
              "(algebra test still meaningful but kernel-determinism does not apply)")

    _force_deterministic(seed=args.seed, device=device)

    # Build model and load V6 weights.
    print(f"[sigma_norm_golden_test] building model from {cfg_path.name}")
    model = PETFlowDiTFirstHop(cfg, device=device).to(device)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    model.requires_grad_(False)

    # Resolve rollout_times from the data config (single source of truth).
    data_cfg = cfg["data"]
    rollout_timepoints = data_cfg.get("rollout_timepoints", ["D50", "D20", "D10", "D4", "NORMAL"])
    t_map = data_cfg["t_map"]
    rollout_times = [float(t_map[name]) / float(t_map.get("NORMAL", 100.0)) for name in rollout_timepoints]
    # Match V6 train-time normalization: rollout_times in V6 train=[0.50, 0.20, 0.10, 0.04, 0.0]
    # Use the canonical V6 schedule when t_map matches V6's defaults, otherwise
    # fall back to the cfg-derived values.
    canonical_v6 = [0.50, 0.20, 0.10, 0.04, 0.0]
    if (
        len(rollout_times) == len(canonical_v6)
        and abs(rollout_times[-1] - 1.0) < 1e-6  # NORMAL was normalized to 1.0
    ):
        # Convert to V6's t-relative-to-NORMAL=0 convention
        rollout_times = canonical_v6
    print(f"[sigma_norm_golden_test] rollout_times={rollout_times}")

    # Build the fixed batch.
    if args.use_real_batch:
        print("[sigma_norm_golden_test] loading one val batch (--use-real-batch)")
        batch = _build_real_batch(cfg, device=device)
    else:
        print(f"[sigma_norm_golden_test] building synthetic deterministic batch (seed={args.seed})")
        batch = _build_synthetic_batch(model, rollout_times, seed=args.seed, device=device)

    # ============================================================
    # Path A: raw rollout, V6 weights, no normalizer
    # ============================================================
    w_v6 = [0.5, 2.0, 1.5, 1.0]
    print("[sigma_norm_golden_test] computing path A (raw rollout, w_v6, no normalizer)")
    with torch.no_grad():
        out_A = rollout_multistep_losses_first_hop(
            model=model,
            z_rollout=batch["z_rollout"],
            rollout_times=rollout_times,
            x_rollout_first=batch.get("x_rollout_first"),
            alpha=1.0,                 # eval-time: no GT mixing
            straight_through=False,    # eval-time: not training
            loss_type="mse",
            step_weights=w_v6,
            step_normalizers=None,
        )

    # ============================================================
    # Path B: sigma-normalize rollout, V6 weights * n_j, with normalizer
    # ============================================================
    n_j = _compute_sigma_dt_normalizers(
        model=model,
        rollout_times=rollout_times,
        mode="sigma_dt_squared",
        relative_to="preserve_v6_sum",
        anchor_step_weights=w_v6,
    )
    w_B = [w_v6[j] * n_j[j] for j in range(len(w_v6))]
    print(f"[sigma_norm_golden_test] n_j={[f'{x:.6f}' for x in n_j]}")
    print(f"[sigma_norm_golden_test] w_B = w_v6 * n_j = {[f'{x:.6f}' for x in w_B]}")
    print(f"[sigma_norm_golden_test] sum(w_v6)={sum(w_v6):.10f}, sum(w_B)={sum(w_B):.10f}")

    # Verify the preserve_v6_sum invariant in float64 (paper algebra check).
    sum_diff = sum(w_B) - sum(w_v6)
    if abs(sum_diff) > 1e-6:
        print(f"[sigma_norm_golden_test] WARN: preserve_v6_sum invariant "
              f"|sum(w_B) - sum(w_v6)| = {abs(sum_diff):.6e} > 1e-6")

    print("[sigma_norm_golden_test] computing path B (sigma-normalize rollout, w_B, with normalizers)")
    with torch.no_grad():
        out_B = rollout_multistep_losses_first_hop(
            model=model,
            z_rollout=batch["z_rollout"],
            rollout_times=rollout_times,
            x_rollout_first=batch.get("x_rollout_first"),
            alpha=1.0,
            straight_through=False,
            loss_type="mse",
            step_weights=w_B,
            step_normalizers=n_j,
        )

    # ============================================================
    # Compare totals + raw step losses
    # ============================================================
    total_A = float(out_A["loss_total"].detach().cpu().item())
    total_B = float(out_B["loss_total"].detach().cpu().item())
    abs_drift_total = abs(total_A - total_B)
    rel_drift_total = abs_drift_total / max(abs(total_A), 1e-12)

    raw_A = [float(t.detach().cpu().item()) for t in out_A["step_losses_raw"]]
    raw_B = [float(t.detach().cpu().item()) for t in out_B["step_losses_raw"]]
    per_hop_drift = [abs(raw_A[j] - raw_B[j]) for j in range(len(raw_A))]
    per_hop_rel = [
        per_hop_drift[j] / max(abs(raw_A[j]), 1e-12) for j in range(len(raw_A))
    ]

    # ============================================================
    # Verdict
    # ============================================================
    if rel_drift_total < args.rel_tolerance:
        verdict = "PASS"
        exit_code = 0
        msg = (
            f"algebra verified: rel_drift={rel_drift_total:.3e} < "
            f"rel_tolerance={args.rel_tolerance:.3e}; "
            f"(beta) and (zeta) FALSIFIED at inference"
        )
    elif rel_drift_total > args.ambiguous_upper:
        verdict = "FAIL"
        exit_code = 4
        msg = (
            f"algebra BROKEN: rel_drift={rel_drift_total:.3e} > "
            f"ambiguous_upper={args.ambiguous_upper:.3e}; (zeta) CONFIRMED; "
            f"defer V7 launch and investigate _compute_sigma_dt_normalizers"
        )
    else:
        verdict = "AMBIGUOUS"
        exit_code = 3
        msg = (
            f"ambiguous: rel_drift={rel_drift_total:.3e} in "
            f"[{args.rel_tolerance:.0e}, {args.ambiguous_upper:.0e}]; "
            f"run Option G (paired raw-vs-sigma full-val) for distributional bound"
        )

    result = {
        "verdict": verdict,
        "exit_code": exit_code,
        "rel_tolerance": args.rel_tolerance,
        "ambiguous_upper": args.ambiguous_upper,
        "loss_total": {
            "A": total_A,
            "B": total_B,
            "abs_drift": abs_drift_total,
            "rel_drift": rel_drift_total,
        },
        "step_losses_raw": {
            "A": raw_A,
            "B": raw_B,
            "abs_drift_per_hop": per_hop_drift,
            "rel_drift_per_hop": per_hop_rel,
        },
        "normalizers": {
            "n_j": list(n_j),
            "w_v6": w_v6,
            "w_B": w_B,
            "sum_w_v6": float(sum(w_v6)),
            "sum_w_B": float(sum(w_B)),
            "preserve_v6_sum_residual": float(sum_diff),
        },
        "config": {
            "config_path": str(cfg_path),
            "ckpt_path": str(ckpt_path),
            "device": str(device),
            "seed": int(args.seed),
            "use_real_batch": bool(args.use_real_batch),
            "rollout_times": list(rollout_times),
        },
        "message": msg,
    }

    # Print human-readable summary.
    print("=" * 72)
    print(f"[sigma_norm_golden_test] VERDICT: {verdict}")
    print(f"  loss_total A          = {total_A:.10e}")
    print(f"  loss_total B          = {total_B:.10e}")
    print(f"  abs drift             = {abs_drift_total:.6e}")
    print(f"  rel drift             = {rel_drift_total:.6e}")
    print(f"  rel_tolerance         = {args.rel_tolerance:.0e}")
    print(f"  ambiguous_upper       = {args.ambiguous_upper:.0e}")
    print(f"  per-hop raw drift     = {[f'{x:.3e}' for x in per_hop_drift]}")
    print(f"  per-hop raw rel drift = {[f'{x:.3e}' for x in per_hop_rel]}")
    print(f"  preserve_v6_sum residual = {sum_diff:.6e}")
    print(f"  message: {msg}")
    print("=" * 72)

    if args.output_json:
        out_path = Path(args.output_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"[sigma_norm_golden_test] result JSON written to {out_path}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
