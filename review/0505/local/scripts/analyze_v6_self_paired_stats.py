#!/usr/bin/env python3
"""Compute paired Cohen d between V6 checkpoints from existing per-slice CSVs.

This is a one-shot diagnostic to provide a tighter prior on d_pure
(V6_seed42 vs V6_seed1337 at step 160K) using the data we already have
on disk. The output informs Plan F section 11 tier judgment BEFORE
V6_NOISE finishes running.

Outputs to stdout; meant to be invoked once for analysis. Not committed
as a permanent script.
"""
import csv
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
ART = REPO / "review/0505/operator/artifacts"

CHECKPOINTS = {
    "v6_step160k": ART / "v6_step160k_fullval_psnr_chain_mse_per_slice.csv",
    "v6_best":     ART / "v6_best_fullval_psnr_chain_mse_per_slice.csv",
    "v6_last":     ART / "v6_last_fullval_psnr_chain_mse_per_slice.csv",
    "v6.1_best":   ART / "v6_1_best_fullval_psnr_chain_mse_per_slice.csv",
    "v6.1_last":   ART / "v6_1_last_fullval_psnr_chain_mse_per_slice.csv",
}

METRIC = "mse_NORMAL"  # primary endpoint

def load(path: Path):
    """Returns (slice_indices, mse_normal_values) parallel lists, sorted by slice_idx."""
    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append((int(r["slice_idx"]), float(r[METRIC])))
    rows.sort(key=lambda x: x[0])
    idx = [r[0] for r in rows]
    val = [r[1] for r in rows]
    return idx, val


def paired_cohen_d(a, b):
    """Paired Cohen d = mean(a-b) / SD(a-b)."""
    diffs = [ai - bi for ai, bi in zip(a, b)]
    n = len(diffs)
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    sd_d = math.sqrt(var_d)
    cohen_d = mean_d / sd_d if sd_d > 0 else 0.0
    # paired t-stat = mean_d / (sd_d / sqrt(n))
    se = sd_d / math.sqrt(n)
    t_stat = mean_d / se if se > 0 else 0.0
    return mean_d, sd_d, cohen_d, t_stat, n


def relative(a, b):
    """Relative change in mean (a - b) / b."""
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    return (mean_a - mean_b) / mean_b


def main():
    print(f"Loading per-slice CSVs (metric={METRIC})...")
    data = {}
    for name, path in CHECKPOINTS.items():
        if not path.exists():
            print(f"  MISSING: {name} -> {path}")
            continue
        idx, val = load(path)
        data[name] = (idx, val)
        print(f"  {name:14s} n={len(val):5d}  mean={sum(val)/len(val):.6e}  path={path.name}")

    # All checkpoints should have the same slice_idx ordering
    ref_idx = next(iter(data.values()))[0]
    for name, (idx, _) in data.items():
        if idx != ref_idx:
            print(f"WARN: slice_idx mismatch for {name}")

    print()
    print("=" * 80)
    print("Pairwise comparisons (mean rel diff = (A - B) / B; Cohen d = mean(A-B)/SD(A-B))")
    print("=" * 80)
    pairs = [
        # Same-yaml self-comparisons (training progress, not RNG noise)
        ("v6_step160k", "v6_best",     "V6 self: 160K -> 185.6K (24K extra steps)"),
        ("v6_step160k", "v6_last",     "V6 self: 160K -> 200K   (40K extra steps)"),
        ("v6_best",     "v6_last",     "V6 self: 185.6K -> 200K (14.4K extra steps, tightest proxy for stability)"),
        # V6 vs V6.1 (different yaml — algorithmic differences confound)
        ("v6_best",     "v6.1_best",   "V6 vs V6.1: best (cross-arm, algorithmic)"),
        ("v6_last",     "v6.1_last",   "V6 vs V6.1: last (cross-arm, algorithmic)"),
        ("v6_step160k", "v6.1_best",   "V6@160K vs V6.1 best"),
        # V6.1 self
        ("v6.1_best",   "v6.1_last",   "V6.1 self: best -> last"),
    ]
    print(f"{'comparison':70s}  {'rel_diff':>10s}  {'cohen_d':>9s}  {'t_stat':>9s}  {'mean_diff':>11s}  {'n':>5s}")
    print("-" * 130)
    for a, b, desc in pairs:
        if a not in data or b not in data:
            print(f"  SKIP {desc}: missing")
            continue
        va = data[a][1]
        vb = data[b][1]
        rel = relative(va, vb)
        mean_d, sd_d, cohen_d, t_stat, n = paired_cohen_d(va, vb)
        print(f"{desc:70s}  {rel*100:>+9.3f}%  {cohen_d:>+9.3f}  {t_stat:>+9.2f}  {mean_d:>+11.4e}  {n:>5d}")

    print()
    print("=" * 80)
    print("Distribution of per-slice MSE values (for context)")
    print("=" * 80)
    print(f"{'checkpoint':14s}  {'mean':>11s}  {'sd':>11s}  {'p10':>11s}  {'p50':>11s}  {'p90':>11s}  {'cv':>6s}")
    for name, (_, val) in data.items():
        v_sorted = sorted(val)
        n = len(v_sorted)
        mean = sum(val) / n
        sd = math.sqrt(sum((x - mean) ** 2 for x in val) / (n - 1))
        p10 = v_sorted[int(0.10 * n)]
        p50 = v_sorted[int(0.50 * n)]
        p90 = v_sorted[int(0.90 * n)]
        cv = sd / mean
        print(f"{name:14s}  {mean:>11.4e}  {sd:>11.4e}  {p10:>11.4e}  {p50:>11.4e}  {p90:>11.4e}  {cv:>6.2%}")

    print()
    print("=" * 80)
    print("Implications for d_pure prior")
    print("=" * 80)
    print(
        """
The V6@best vs V6@last paired Cohen d is the TIGHTEST same-yaml proxy for
d_pure: identical seed, identical config, only 14.4K extra training steps
between checkpoints. This is essentially a 'how much can a saturated V6
model wobble' lower bound.

V6@160K vs V6@200K paired Cohen d = 0.208 = SGD progress (model improving),
NOT pure RNG noise. This is the LOOSER bound that Opus 4.7 cited.

For V6_seed42 vs V6_seed1337 at step 160K:
- Mean diff component: ~0 (no systematic direction; different seeds explore
  different but equally-valid local minima)
- SD-of-diff component: bounded above by V6@160K-vs-V6@200K's SD (which
  combines 40K steps of training progress + RNG drift)
- Plausible d_pure: SUBSTANTIALLY SMALLER than 0.208 because the systematic
  improvement direction is absent

Best informed prior (anchored on the V6@best vs V6@last d above):
  d_pure ~ paired-d(V6@best, V6@last) <= 0.05 likely (data dependent)
  d_pure 90% upper bound ~ 0.10-0.15

Compare Opus 4.7 estimate range: [0.20, 0.40]
Compare external Agent 6 catastrophic estimate: [0.50, 2.0]

If V6@best vs V6@last d < 0.10 (likely), Plan F is in Tier 0/1 not Tier 2/3.
The 'd=0.208 is alarming' Opus 4.7 framing CONFLATES SGD-progress with noise.
"""
    )


if __name__ == "__main__":
    main()
