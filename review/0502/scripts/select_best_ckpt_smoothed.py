#!/usr/bin/env python3
"""Select best ckpt via local-neighborhood smoothing of val_select_score.

Background (see review/0502/ROLLING_WINDOW_TRADEOFF.md §5.1):
  best.pt is selected on single-window val_select_score (CV ~25% under
  V6 yaml max_val_batches=64). Over ~500 evals, extreme-value bias on the
  min sample is ~3σ → best.pt row is biased low by ~61% from population
  mean.

Mitigation (Method D — zero disk cost):
  At paper time, for each *saved* ckpt step S ∈ {save_interval, 2·save_interval,
  ...}, smooth val_select_score over K eval rows on each side of S, then pick
  S* = argmin smoothed_score(S). Because adjacent rolling windows are disjoint,
  K-side smoothing averages over (2K+1) disjoint sample subsets → noise drops
  by sqrt(2K+1). With K=10 (21 evals × 64 batches × 8 bs ≈ 10752 samples ≈
  37% of val set), single-window CV ~25% → smoothed CV ~5.5%; extreme-value
  bias on min over 6 saved ckpts drops from -61% to -7%.

This script does NOT touch ckpt files. It only reads metrics.jsonl and prints
a recommendation. Use the recommended ckpt with
  eval_first_hop_224_clip3.py --max-slices 0
to obtain the paper-table number.

Usage:
    python review/0502/scripts/select_best_ckpt_smoothed.py \
        --metrics /data_2/qujiaxiang/outputs/PET_LatentResidual/A_main/run-.../metrics.jsonl \
        --ckpt-dir /data_2/qujiaxiang/outputs/PET_LatentResidual/A_main/run-.../ \
        --neighborhood 10
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

DEFAULT_METRIC_KEYS = ("val_select_score", "val_chain_normal_mse")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--metrics", required=True, type=Path,
                   help="Path to metrics.jsonl produced by training")
    p.add_argument("--ckpt-dir", required=True, type=Path,
                   help="Directory containing ckpt_step_*.pt / best.pt / ckpt_last.pt")
    p.add_argument("--neighborhood", type=int, default=10,
                   help="K: number of eval rows to include on each side of "
                        "each candidate step. Default 10 (i.e., 21 rows total). "
                        "Higher K = more smoothing but blurs over schedule "
                        "transitions; lower K = noisier. Don't go below 5.")
    p.add_argument("--metric-key", default=None,
                   help="Override which metric to minimize. Default: "
                        "val_select_score if present, else val_chain_normal_mse.")
    p.add_argument("--include-best-pt", action="store_true",
                   help="Also evaluate best.pt's step alongside the saved-grid "
                        "ckpts (requires torch to load the .pt header).")
    p.add_argument("--include-last-pt", action="store_true",
                   help="Also evaluate ckpt_last.pt's step alongside the "
                        "saved-grid ckpts.")
    return p.parse_args()


def load_metrics(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def detect_metric_key(val_rows: list[dict], override: str | None) -> str:
    if override is not None:
        if not any(override in r for r in val_rows):
            raise SystemExit(f"--metric-key {override!r} not present in any eval row")
        return override
    for k in DEFAULT_METRIC_KEYS:
        if all(k in r for r in val_rows[: min(5, len(val_rows))]):
            return k
    raise SystemExit(
        f"Could not auto-detect metric key. Tried {DEFAULT_METRIC_KEYS}. "
        f"Pass --metric-key explicitly. Available keys (first row): "
        f"{sorted(val_rows[0].keys())[:20]}..."
    )


def list_saved_steps(ckpt_dir: Path,
                     include_best: bool,
                     include_last: bool) -> list[int]:
    steps: set[int] = set()

    for cf in ckpt_dir.glob("ckpt_step_*.pt"):
        try:
            steps.add(int(cf.stem.split("_")[-1]))
        except ValueError:
            continue

    def _try_load_step(name: str) -> int | None:
        path = ckpt_dir / name
        if not path.exists():
            return None
        try:
            import torch  # heavy import; only needed when --include-best-pt / --include-last-pt
        except ImportError:
            print(f"[warn] torch not available; cannot extract step from {name}")
            return None
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            print(f"[warn] failed to load {name}: {exc}")
            return None
        s = ckpt.get("global_step", None)
        return int(s) if s is not None else None

    if include_best:
        s = _try_load_step("best.pt")
        if s is not None:
            steps.add(s)
            print(f"[info] best.pt at step {s}")
    if include_last:
        s = _try_load_step("ckpt_last.pt")
        if s is not None:
            steps.add(s)
            print(f"[info] ckpt_last.pt at step {s}")

    return sorted(steps)


def smooth_at_step(val_rows: list[dict],
                   step: int,
                   metric_key: str,
                   k_side: int) -> tuple[float, float, int, float] | None:
    """Return (smoothed_mean, smoothed_std, n, raw_at_closest_step) or None."""
    have_metric = [r for r in val_rows if metric_key in r]
    if not have_metric:
        return None

    # Pick the (2K+1) eval rows whose global_step is closest to `step`
    nearby = sorted(have_metric, key=lambda r: abs(r.get("global_step", 0) - step))
    nearby = nearby[: 2 * k_side + 1]
    scores = [float(r[metric_key]) for r in nearby]
    if not scores:
        return None

    smoothed = statistics.mean(scores)
    std = statistics.stdev(scores) if len(scores) > 1 else float("nan")

    # Raw single-window value at the step closest to `step`
    raw_row = min(have_metric, key=lambda r: abs(r.get("global_step", 0) - step))
    raw = float(raw_row[metric_key])
    return smoothed, std, len(scores), raw


def main() -> None:
    args = parse_args()

    rows = load_metrics(args.metrics)
    val_rows = [r for r in rows if any(k in r for k in DEFAULT_METRIC_KEYS)]
    if not val_rows:
        raise SystemExit(
            f"No eval rows with {DEFAULT_METRIC_KEYS} found in {args.metrics}"
        )
    print(f"[info] loaded {len(rows)} rows ({len(val_rows)} have val metrics)")

    metric_key = detect_metric_key(val_rows, args.metric_key)
    print(f"[info] metric: {metric_key}")
    print(f"[info] neighborhood K={args.neighborhood} → averaging "
          f"{2*args.neighborhood + 1} eval rows per candidate")

    saved_steps = list_saved_steps(
        args.ckpt_dir,
        include_best=args.include_best_pt,
        include_last=args.include_last_pt,
    )
    if not saved_steps:
        raise SystemExit(
            f"No saved ckpts found in {args.ckpt_dir} matching ckpt_step_*.pt; "
            f"--include-best-pt / --include-last-pt also yielded none"
        )
    print(f"[info] candidate ckpt steps: {saved_steps}")
    print()

    # Smooth each candidate
    table: list[dict] = []
    for s in saved_steps:
        result = smooth_at_step(val_rows, s, metric_key, args.neighborhood)
        if result is None:
            continue
        smoothed, std, n, raw = result
        table.append({
            "step": s,
            "raw": raw,
            "smoothed": smoothed,
            "std": std,
            "n": n,
            "cv_smoothed": (std / smoothed) if (smoothed != 0 and not math.isnan(std)) else float("nan"),
        })

    if not table:
        raise SystemExit("No saved ckpt has any nearby eval rows in metrics.jsonl")

    # Sort by smoothed
    table_sorted = sorted(table, key=lambda d: d["smoothed"])

    # Print table
    print(f"{'step':>8} | {'raw':>12} | {'smoothed':>12} | "
          f"{'std':>12} | {'CV':>6} | {'n':>3} | {'rank':>5}")
    print("-" * 80)
    rank_by_step: dict[int, int] = {row["step"]: i + 1 for i, row in enumerate(table_sorted)}
    for row in table:
        rank = rank_by_step[row["step"]]
        cv_str = f"{row['cv_smoothed']*100:>4.1f}%" if not math.isnan(row["cv_smoothed"]) else "  N/A"
        print(f"{row['step']:>8} | {row['raw']:>12.6e} | {row['smoothed']:>12.6e} | "
              f"{row['std']:>12.6e} | {cv_str:>6} | {row['n']:>3} | #{rank}")

    print()
    best = table_sorted[0]
    second = table_sorted[1] if len(table_sorted) > 1 else None

    # Robustness: how separated is the winner from the runner-up?
    if second is not None:
        gap_rel = (second["smoothed"] - best["smoothed"]) / abs(best["smoothed"])
        # If the gap is smaller than 1σ of the winner's smoothed estimate, the
        # selection is statistically tied with the runner-up
        gap_in_sigma = (second["smoothed"] - best["smoothed"]) / best["std"] if best["std"] > 0 else float("inf")
    else:
        gap_rel = float("nan")
        gap_in_sigma = float("nan")

    print(f"=> Recommended ckpt: step {best['step']}")
    print(f"   smoothed score = {best['smoothed']:.6e} (over {best['n']} nearby evals)")
    if second is not None:
        print(f"   runner-up: step {second['step']}, smoothed {second['smoothed']:.6e} "
              f"(+{gap_rel*100:.1f}%, gap = {gap_in_sigma:.2f}σ)")
        if gap_in_sigma < 1.0:
            print(f"   ⚠️  WARNING: gap < 1σ → winner is statistically tied with runner-up. "
                  f"Run full-val on BOTH and use mean ± std as paper number.")
    print()
    print(f"   Next step: load this ckpt and run full-val:")
    ckpt_name = f"ckpt_step_{best['step']}.pt"
    if not (args.ckpt_dir / ckpt_name).exists():
        ckpt_name = f"<the .pt file at step {best['step']}>"
    print(f"     python eval_first_hop_224_clip3.py \\")
    print(f"       --config <yaml> \\")
    print(f"       --checkpoint {args.ckpt_dir / ckpt_name} \\")
    print(f"       --split val --max-slices 0")


if __name__ == "__main__":
    main()
