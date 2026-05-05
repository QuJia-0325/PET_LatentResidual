#!/usr/bin/env python3
"""Compare V6 / V7 / V8 metrics.jsonl at a target training step.

Usage:
    python compare_v6_v7_v8.py \
        --v6 /path/to/V6/metrics.jsonl \
        --v7 /path/to/V7/metrics.jsonl \
        --v8 /path/to/V8/metrics.jsonl \
        --step 50000

Prints a markdown table comparing chain MSE / val_pair_total / val_rollout_*
metrics across the three arms at the target step (or the closest step
present in each metrics.jsonl).

Reads only the JSONL rows that have eval keys (val_chain_*_mse, val_pair_total).
Train-only rows are skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


KEYS_OF_INTEREST = (
    "val_chain_d20_mse",
    "val_chain_d10_mse",
    "val_chain_d4_mse",
    "val_chain_normal_mse",
    "val_pair_total",
    "val_rollout_total",
    "val_rollout_step_0_raw",
    "val_rollout_step_1_raw",
    "val_rollout_step_2_raw",
    "val_rollout_step_3_raw",
    "val_multi_objective",
)


def load_eval_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if any(k in row for k in KEYS_OF_INTEREST):
                rows.append(row)
    return rows


def closest_row(rows: list[dict], target_step: int) -> Optional[dict]:
    if not rows:
        return None
    return min(rows, key=lambda r: abs(int(r.get("step", 0)) - target_step))


def fmt(val: Optional[float], digits: int = 6) -> str:
    if val is None:
        return "-"
    if isinstance(val, (int, float)):
        return f"{val:.{digits}e}" if abs(val) < 1e-2 else f"{val:.{digits}f}"
    return str(val)


def rel_err(a: Optional[float], ref: Optional[float]) -> str:
    if a is None or ref is None or ref == 0:
        return "-"
    return f"{(a - ref) / ref * 100:+.2f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v6", required=True, help="V6 metrics.jsonl path")
    ap.add_argument("--v7", required=True, help="V7 metrics.jsonl path")
    ap.add_argument("--v8", required=True, help="V8 metrics.jsonl path")
    ap.add_argument("--step", type=int, default=50000, help="target step (default 50000)")
    args = ap.parse_args()

    paths = {"V6": Path(args.v6), "V7": Path(args.v7), "V8": Path(args.v8)}
    rows = {tag: load_eval_rows(p) for tag, p in paths.items()}
    closest = {tag: closest_row(rs, args.step) for tag, rs in rows.items()}

    # Header
    print(f"# V6 / V7 / V8 comparison at step ≈ {args.step}")
    print()
    print("## Loaded data")
    for tag in ("V6", "V7", "V8"):
        n = len(rows[tag])
        if n == 0:
            print(f"- **{tag}**: ❌ no eval rows at `{paths[tag]}`")
        else:
            row = closest[tag]
            actual_step = int(row.get("step", -1)) if row else -1
            print(f"- **{tag}**: {n} eval rows; closest to {args.step} = step {actual_step}; path `{paths[tag]}`")
    print()

    # Comparison table
    if any(closest[tag] is None for tag in ("V6", "V7", "V8")):
        missing = [tag for tag in ("V6", "V7", "V8") if closest[tag] is None]
        print(f"⚠️  Missing data for: {', '.join(missing)} — comparison table skipped.")
        return 1

    print("## Metrics table (V6 baseline; V7 / V8 vs V6 relative error)")
    print()
    print("| metric | V6 | V7 | rel(V7) | V8 | rel(V8) |")
    print("|---|---:|---:|---:|---:|---:|")
    for key in KEYS_OF_INTEREST:
        v6 = closest["V6"].get(key)
        v7 = closest["V7"].get(key)
        v8 = closest["V8"].get(key)
        if v6 is None and v7 is None and v8 is None:
            continue
        print(
            f"| `{key}` | {fmt(v6)} | {fmt(v7)} | {rel_err(v7, v6)} "
            f"| {fmt(v8)} | {rel_err(v8, v6)} |"
        )

    print()
    print("## Decision rules")
    print()
    print("**V7 (Grönwall closed-form raw step_weights)**:")
    print("- ✅ Promote V7 to 200K main if `val_chain_normal_mse(V7) < val_chain_normal_mse(V6)` by ≥ 5%")
    print("  AND `val_multi_objective(V7) ≤ val_multi_objective(V6)`.")
    print("- 🟡 Run V7b (closed-form shape × V6 sum=5.0 = [0.4148, 1.3592, 1.8378, 1.3878]) if V7 wins")
    print("  but you suspect the win is from the 32% reduced effective lambda_roll, not from the shape.")
    print("- ❌ Drop V7 if `val_chain_normal_mse(V7) > val_chain_normal_mse(V6) + 1%` AND no late-hop")
    print("  metric (chain_d4 / chain_normal) shows compensating gain.")
    print()
    print("**V8 (no image_aux)**:")
    print("- ✅ Promote V8 to 200K main if all four `val_chain_*_mse(V8) ≤ val_chain_*_mse(V6) + 1%`.")
    print("  Then qualitatively verify decoded NORMAL slices to confirm artifacts have not returned.")
    print("- 🟡 Keep image_aux but reduce `lambda_max` if V8 chain MSE is better but decode artifacts return.")
    print("- ❌ Keep V6 image_aux if `val_chain_*_mse(V8)` is worse on any hop AND the original artifact")
    print("  motivation is still relevant.")
    print()
    print(f"NOTE: rolling-val window is 64 batches × 8 = 512 slices, not the full 7403-slice eval set.")
    print(f"      For final paper-grade verdicts, run `eval_first_hop_fullval_psnr_chain_mse.py` on the")
    print(f"      best/last checkpoints of V6 / V7 / V8 (see review/0505/scripts/).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
