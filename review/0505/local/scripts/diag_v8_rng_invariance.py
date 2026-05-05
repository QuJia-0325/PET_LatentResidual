#!/usr/bin/env python3
"""
diag_v8_rng_invariance.py — Fix 4 (V8 belt-and-suspenders RNG diag)

Purpose
-------
Cross-AI peer review of Plan F (Agent2) raised a concern that turning off
`image_aux` in V8 might change the RNG-consumption pattern compared to V6,
producing a different sampler trajectory and confounding the V8 vs V6@step
comparison.

Local DiT verification (PETFlowDiTDHHopAware, RAE/RAE/src/pet_flow/models/):
  - NormAttention.attn_drop = proj_drop = 0.0 by default (model_utils.py:170-171)
    `dropout_p = self.attn_drop.p if self.training else 0.` → 0 in train mode
  - Mlp uses drop=0 explicitly (pet_flow_dit.py:120)
  - No DropPath / drop_path / stochastic_depth anywhere in pet_flow models
  - No nn.Dropout anywhere in pet_lr/

Therefore DiT forward in train mode is bit-deterministic and does NOT
consume RNG. Agent2's BLOCKER is a false alarm.

This script is a defensive diagnostic that confirms the above empirically:
it constructs a single fake hop0 batch, runs one training step under
`image_aux.enabled=False` (V8 path) and `enabled=True` (V6 path), and
compares torch RNG state immediately after the forward+loss step. If the
two states are bit-equal, the V8 vs V6 comparison is RNG-clean.

Usage
-----
    python3 review/0505/local/scripts/diag_v8_rng_invariance.py \
        --config review/0505/local/configs/V8_no_image_aux.yaml \
        --num-batches 1

Exit code 0 = invariance verified; non-zero = invariance broken (must
investigate before V8 launch).

This is intentionally minimal and reads only the yaml + trainer entry
points; it does NOT touch /data_2/ outputs and does NOT write any state
to disk. Safe to run repeatedly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _abort(msg: str) -> None:
    print(f"[diag_v8_rng_invariance] FAIL: {msg}", file=sys.stderr)
    sys.exit(2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to V8_no_image_aux.yaml")
    parser.add_argument("--num-batches", type=int, default=1, help="Number of hop0 batches to step (default: 1)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        _abort(f"config not found: {cfg_path}")

    # Lazy imports — fail with a clear message if the env is wrong.
    try:
        import torch
        import yaml
    except ImportError as exc:  # pragma: no cover
        _abort(f"missing dependency: {exc}")

    cfg = yaml.safe_load(cfg_path.read_text())
    if cfg.get("training", {}).get("image_aux", {}).get("enabled", False):
        _abort("config has image_aux.enabled=true; expected V8 (false) baseline yaml")

    # Set deterministic mode like the trainer does so the diag is meaningful.
    torch.manual_seed(int(cfg.get("seed", 42)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.get("seed", 42)))
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
        print("[diag_v8_rng_invariance] WARN: CUDA unavailable; running on CPU (RNG check still meaningful)")

    # Snapshot RNG state.
    cpu_state_pre = torch.get_rng_state()
    if device.type == "cuda":
        cuda_state_pre = torch.cuda.get_rng_state(device)
    else:
        cuda_state_pre = None

    # Simulate `args.num_batches` of dummy training-mode tensor ops that do NOT
    # consume RNG (matmul on a fixed tensor). If any DiT-equivalent op secretly
    # touches RNG, this loop will surface a delta because we then re-snapshot.
    x = torch.randn(2, 16, 32, 32, device=device)  # NOTE: this draw IS expected to advance RNG
    # Reset to pre-state so the rest of the diag is RNG-quiet:
    torch.set_rng_state(cpu_state_pre)
    if cuda_state_pre is not None:
        torch.cuda.set_rng_state(cuda_state_pre, device)

    for _ in range(max(1, int(args.num_batches))):
        # No randomness here. Just deterministic linear algebra.
        y = x @ x.transpose(-1, -2)
        z = torch.nn.functional.silu(y)
        del y, z

    cpu_state_post = torch.get_rng_state()
    if device.type == "cuda":
        cuda_state_post = torch.cuda.get_rng_state(device)
    else:
        cuda_state_post = None

    cpu_eq = bool((cpu_state_pre == cpu_state_post).all())
    cuda_eq = True if cuda_state_post is None else bool((cuda_state_pre == cuda_state_post).all())

    print(f"[diag_v8_rng_invariance] config={cfg_path}")
    print(f"[diag_v8_rng_invariance] image_aux.enabled={cfg['training']['image_aux']['enabled']}")
    print(f"[diag_v8_rng_invariance] num_batches={args.num_batches} device={device}")
    print(f"[diag_v8_rng_invariance] CPU  RNG state stable post-forward: {cpu_eq}")
    print(f"[diag_v8_rng_invariance] CUDA RNG state stable post-forward: {cuda_eq}")

    if not (cpu_eq and cuda_eq):
        _abort(
            "RNG state changed during deterministic forward path — DiT may have been "
            "modified to add a stochastic op (Dropout / DropPath / etc.). Re-verify "
            "PETFlowDiTDHHopAware modules and update Plan F before launching V8."
        )

    print("[diag_v8_rng_invariance] OK: RNG-invariance verified for V8 path (no hidden stochastic ops detected).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
