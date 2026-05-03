#!/usr/bin/env python3
"""Judge paired-diff between two runs over a step window with CI95 + autocorr-corrected N_eff.

Background (see review/0502/ROLLING_WINDOW_TRADEOFF.md §2.3 and §3.4):
  Single-window val_chain_normal_mse has CV ~25% under V6 yaml
  (max_val_batches=64, val_window_mode=rolling). best-vs-best comparison
  between two runs has 3σ detection floor ~76% (window noise + extreme-value
  bias) → effectively useless for ablation magnitude judgments.

  But because val_loader has shuffle=False and rolling window is fully
  determined by (global_step, eval_interval, total_batches, max_val_batches),
  TWO RUNS WITH IDENTICAL EVAL CONFIG SEE THE EXACT SAME 512 VAL SAMPLES
  AT EVERY SHARED global_step → window noise CANCELS in same-step paired
  diff → residual is pure model-difference signal + much smaller eval-mse
  estimator noise (CV ~5% theoretical, not directly measured).

This script:
  - Reads two metrics.jsonl files (treated as A and B).
  - Verifies the eval-config prerequisites are identical (guard 5 below).
  - Computes paired diff (B - A) / A at every shared global_step in window.
  - Reports MEAN ± SE ± CI95, autocorrelation-corrected effective N.
  - Applies a LOCKED 10% decision rule (configurable via --threshold) using
    mean ± 2·SE bands → "no confound" / "confound" / "borderline".

Usage:
    python review/0502/scripts/paired_diff_judge.py \
        --metrics-a path/to/A_main/metrics.jsonl \
        --config-a  review/0502/configs/A_control.yaml \
        --metrics-b path/to/A_pair_uniform_spot/metrics.jsonl \
        --config-b  review/0502/configs/A_pair_uniform_spot.yaml \
        --metric-key val_chain_normal_mse \
        --step-min 40000 \
        --step-max 60000 \
        --threshold 0.10

Guards (all are hard-fails — mis-applied paired diff is silently catastrophic):
  1. metric-key MUST be present in BOTH files at >= 5 shared steps
  2. autocorrelation-corrected N_eff MUST be >= 5 (else statement of mean is
     not statistically defensible)
  3. window MUST be specified explicitly (no default range — forces user to
     think about which phase of training they're judging)
  4. mean ± 2·SE bands MUST not straddle 0 AND threshold simultaneously (else
     verdict is "borderline" not "no confound")
  5. Both yaml configs MUST share seed / val_shuffle / val_window_mode /
     eval_interval / max_val_batches / data.batch_size — otherwise paired
     diff is nominal not real.

Status: Risk 4 spot check rule (LOCKED 2026-05-03 with --threshold 0.10).
        See POST_V6_NEXT_STEPS.md §6.4 for usage protocol.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

# Local helper: schema-tolerant step extraction. See `_metrics_compat.py`
# for the rationale (trainer writes `step`; legacy fixtures may use
# `global_step`).
from _metrics_compat import get_row_step

# Eval-config keys that MUST match between the two runs (guard 5).
PAIRED_REQUIRED_KEYS = (
    ("seed", lambda cfg: cfg.get("seed")),
    ("data.batch_size", lambda cfg: cfg.get("data", {}).get("batch_size")),
    ("data.val_shuffle", lambda cfg: cfg.get("data", {}).get("val_shuffle", False)),
    ("training.eval_interval", lambda cfg: cfg.get("training", {}).get("eval_interval")),
    ("training.max_val_batches", lambda cfg: cfg.get("training", {}).get("max_val_batches")),
    ("training.val_window_mode", lambda cfg: cfg.get("training", {}).get("val_window_mode", "rolling")),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--metrics-a", required=True, type=Path,
                   help="Path to A run's metrics.jsonl (the control / reference)")
    p.add_argument("--config-a", required=True, type=Path,
                   help="Path to A run's yaml config (used by guard 5)")
    p.add_argument("--metrics-b", required=True, type=Path,
                   help="Path to B run's metrics.jsonl (the comparison)")
    p.add_argument("--config-b", required=True, type=Path,
                   help="Path to B run's yaml config (used by guard 5)")
    p.add_argument("--metric-key", default="val_chain_normal_mse",
                   help="Which metric to compute paired diff on. Default "
                        "val_chain_normal_mse (the paper number). Use "
                        "val_select_score for selection-aware judgment.")
    p.add_argument("--step-min", type=int, required=True,
                   help="Inclusive lower bound on global_step (post-warmup, "
                        "pre-Phase-III ramp recommended e.g., 40000)")
    p.add_argument("--step-max", type=int, required=True,
                   help="Inclusive upper bound on global_step (must be <= the "
                        "shorter run's max_steps)")
    p.add_argument("--threshold", type=float, default=0.10,
                   help="Decision threshold for |mean rel diff|. Default 0.10. "
                        "For Risk 4 (defensive confound check), 0.10 is LOCKED. "
                        "DO NOT raise post-hoc.")
    p.add_argument("--label-a", default="A", help="Display label for A run")
    p.add_argument("--label-b", default="B", help="Display label for B run")
    p.add_argument("--output-md", type=Path, default=None,
                   help="Optional: write a markdown report to this path "
                        "(suitable for pasting into paper supplementary)")
    p.add_argument("--allow-config-mismatch", action="store_true",
                   help="DANGEROUS: skip guard 5 (config equality check). "
                        "Only use if you've manually verified equivalence.")
    return p.parse_args()


def load_yaml(path: Path) -> dict:
    """Minimal YAML loader. Uses PyYAML if available, else fails clearly."""
    try:
        import yaml  # type: ignore
    except ImportError:
        sys.stderr.write(
            "ERROR: PyYAML not installed. `pip install pyyaml` and retry.\n"
        )
        sys.exit(2)
    with path.open() as f:
        return yaml.safe_load(f) or {}


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


def index_by_step(rows: list[dict], metric_key: str) -> dict[int, float]:
    """Return {step: metric_value} for rows that have BOTH a usable step key
    (`step` or `global_step` per `_metrics_compat.get_row_step`) AND the
    requested metric key."""
    out: dict[int, float] = {}
    for r in rows:
        step = get_row_step(r)
        if step is None:
            continue
        if metric_key not in r:
            continue
        try:
            val = float(r[metric_key])
        except (TypeError, ValueError):
            continue
        # Skip non-positive values (cannot compute relative diff)
        if val <= 0.0:
            continue
        out[step] = val
    return out


def check_config_equality(cfg_a: dict, cfg_b: dict, label_a: str, label_b: str) -> list[str]:
    """Guard 5: verify both configs share the eval-determining keys."""
    mismatches: list[str] = []
    for key_name, getter in PAIRED_REQUIRED_KEYS:
        va = getter(cfg_a)
        vb = getter(cfg_b)
        if va != vb:
            mismatches.append(f"  {key_name}: {label_a}={va!r}  {label_b}={vb!r}")
    return mismatches


def lag1_autocorrelation(series: list[float]) -> float:
    """Pearson lag-1 autocorrelation. Returns 0.0 if N < 3 or zero variance."""
    n = len(series)
    if n < 3:
        return 0.0
    mean = sum(series) / n
    centered = [x - mean for x in series]
    num = sum(centered[i] * centered[i + 1] for i in range(n - 1))
    den = sum(c * c for c in centered)
    if den <= 0.0:
        return 0.0
    return num / den


def effective_sample_size(series: list[float]) -> tuple[float, float]:
    """Autocorrelation-corrected effective N. Returns (N_eff, rho1).

    Uses the standard correction
        N_eff = N × (1 - rho1) / (1 + rho1)
    capped to [1, N]. This is conservative for AR(1)-like series (which
    rolling-window paired diffs approximately are).
    """
    n = len(series)
    if n <= 1:
        return float(n), 0.0
    rho1 = lag1_autocorrelation(series)
    rho1_capped = max(-0.999, min(0.999, rho1))
    factor = (1.0 - rho1_capped) / (1.0 + rho1_capped)
    n_eff = max(1.0, min(float(n), float(n) * factor))
    return n_eff, rho1


def summarize(deltas: list[float]) -> dict:
    n = len(deltas)
    mean = sum(deltas) / n
    if n >= 2:
        var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    n_eff, rho1 = effective_sample_size(deltas)
    se = std / math.sqrt(max(n_eff, 1.0))
    ci95_half = 1.96 * se
    return {
        "n": n,
        "n_eff": n_eff,
        "rho1": rho1,
        "mean": mean,
        "std": std,
        "se": se,
        "ci95_lo": mean - ci95_half,
        "ci95_hi": mean + ci95_half,
        "min": min(deltas),
        "max": max(deltas),
    }


def render_verdict(stats: dict, threshold: float) -> tuple[str, str]:
    """Apply LOCKED decision rule. Returns (verdict_code, human_text).

    Bands use mean ± 2·SE (≈ 95% CI of mean) compared against ±threshold:
      |mean| + 2·SE < threshold        → 'no_confound'
      |mean| - 2·SE > threshold        → 'confound'
      else                             → 'borderline'
    """
    mean = stats["mean"]
    two_se = 2.0 * stats["se"]
    abs_mean = abs(mean)

    if abs_mean + two_se < threshold:
        verdict = "no_confound"
        text = (
            f"NO CONFOUND DETECTED. |mean|+2·SE = "
            f"{abs_mean + two_se:.4f} < threshold {threshold:.2f}. "
            f"Pair-channel effect is bounded below {threshold * 100:.0f}% rel diff "
            f"with high statistical confidence."
        )
    elif abs_mean - two_se > threshold:
        sign = "+" if mean > 0 else "-"
        verdict = "confound"
        text = (
            f"CONFOUND DETECTED, direction {sign}, "
            f"magnitude ≈ {abs_mean * 100:.2f}% rel diff. "
            f"|mean|-2·SE = {abs_mean - two_se:.4f} > threshold {threshold:.2f}. "
            f"Re-frame paper to disclose pair-channel sensitivity."
        )
    else:
        verdict = "borderline"
        text = (
            f"BORDERLINE. |mean| = {abs_mean:.4f}, 2·SE = {two_se:.4f}, "
            f"threshold = {threshold:.2f}. CI95 of mean straddles the threshold. "
            f"Recommended next: full-val on best.pt of both runs via "
            f"eval_first_hop_224_clip3.py --max-slices 0; if that "
            f"still borderline, report as paper supplementary."
        )
    return verdict, text


def render_report_md(
    args: argparse.Namespace,
    cfg_a: dict,
    cfg_b: dict,
    common_steps: list[int],
    deltas: list[float],
    stats: dict,
    verdict: str,
    verdict_text: str,
) -> str:
    lines: list[str] = []
    lines.append(f"# Paired diff judge — {args.label_a} vs {args.label_b}")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- A: `{args.metrics_a}` (config `{args.config_a}`)")
    lines.append(f"- B: `{args.metrics_b}` (config `{args.config_b}`)")
    lines.append(f"- Metric key: `{args.metric_key}`")
    lines.append(f"- Step window: `[{args.step_min}, {args.step_max}]`")
    lines.append(f"- Threshold (LOCKED): `{args.threshold:.2f}`")
    lines.append("")
    lines.append("## Statistics")
    lines.append("")
    lines.append("| Quantity | Value |")
    lines.append("|---|---|")
    lines.append(f"| Paired observations N | {stats['n']} |")
    lines.append(f"| Lag-1 autocorrelation ρ₁ | {stats['rho1']:.3f} |")
    lines.append(f"| Effective N_eff | {stats['n_eff']:.1f} |")
    lines.append(f"| Mean rel diff | {stats['mean']:+.4f} ({stats['mean'] * 100:+.2f}%) |")
    lines.append(f"| Std (across observations) | {stats['std']:.4f} |")
    lines.append(f"| SE of mean (autocorr-corrected) | {stats['se']:.4f} |")
    lines.append(f"| CI95 of mean | [{stats['ci95_lo']:+.4f}, {stats['ci95_hi']:+.4f}] |")
    lines.append(f"| Range over observations | [{stats['min']:+.4f}, {stats['max']:+.4f}] |")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**`{verdict}`** — {verdict_text}")
    lines.append("")
    lines.append("## Per-step paired diff (first 20 rows + last 5)")
    lines.append("")
    lines.append("| step | A | B | rel_diff = (B-A)/A |")
    lines.append("|---|---|---|---|")
    a_idx = index_by_step(load_metrics(args.metrics_a), args.metric_key)
    b_idx = index_by_step(load_metrics(args.metrics_b), args.metric_key)
    rows_to_show = list(zip(common_steps[:20], deltas[:20]))
    if len(common_steps) > 25:
        lines.extend(_render_step_rows(rows_to_show, a_idx, b_idx))
        lines.append("| ... | ... | ... | ... |")
        rows_to_show = list(zip(common_steps[-5:], deltas[-5:]))
    lines.extend(_render_step_rows(rows_to_show, a_idx, b_idx))
    return "\n".join(lines) + "\n"


def _render_step_rows(rows, a_idx, b_idx) -> list[str]:
    out = []
    for step, d in rows:
        a = a_idx.get(step)
        b = b_idx.get(step)
        if a is None or b is None:
            continue
        out.append(f"| {step} | {a:.6e} | {b:.6e} | {d:+.4f} |")
    return out


def main() -> int:
    args = parse_args()

    # ---- Guard 3: window must be explicit and well-formed ----
    if args.step_max <= args.step_min:
        sys.stderr.write(
            f"ERROR (guard 3): --step-max ({args.step_max}) must be > "
            f"--step-min ({args.step_min}). No default range — pick a phase "
            f"deliberately.\n"
        )
        return 3

    # ---- Guard 5: config equality (unless explicitly opted out) ----
    if args.allow_config_mismatch:
        cfg_a = {}
        cfg_b = {}
        sys.stderr.write(
            "WARNING (guard 5 bypassed): config equality not checked. "
            "You are responsible for verifying the two runs share "
            "seed/val_shuffle/val_window_mode/eval_interval/max_val_batches/"
            "batch_size manually.\n"
        )
    else:
        cfg_a = load_yaml(args.config_a)
        cfg_b = load_yaml(args.config_b)
        mismatches = check_config_equality(cfg_a, cfg_b, args.label_a, args.label_b)
        if mismatches:
            sys.stderr.write(
                "ERROR (guard 5): the two yaml configs disagree on eval-determining "
                "keys. Paired diff would be nominal, not real.\n"
            )
            sys.stderr.write("\n".join(mismatches) + "\n")
            sys.stderr.write(
                "If you really want to bypass this (e.g., you hand-verified "
                "equivalence), pass --allow-config-mismatch.\n"
            )
            return 5

    # ---- Load and join ----
    a_idx = index_by_step(load_metrics(args.metrics_a), args.metric_key)
    b_idx = index_by_step(load_metrics(args.metrics_b), args.metric_key)
    common_all = sorted(set(a_idx) & set(b_idx))
    common = [s for s in common_all if args.step_min <= s <= args.step_max]

    # ---- Guard 1: minimum shared observations ----
    if len(common) < 5:
        sys.stderr.write(
            f"ERROR (guard 1): only {len(common)} shared steps in window "
            f"[{args.step_min}, {args.step_max}] for metric '{args.metric_key}'. "
            f"Need >= 5. Either widen the window, lower eval_interval, or "
            f"check the metric key spelling.\n"
        )
        return 1

    deltas = [(b_idx[s] - a_idx[s]) / a_idx[s] for s in common]
    stats = summarize(deltas)

    # ---- Guard 2: minimum effective sample size ----
    if stats["n_eff"] < 5.0:
        sys.stderr.write(
            f"ERROR (guard 2): autocorrelation-corrected N_eff = "
            f"{stats['n_eff']:.1f} < 5. Lag-1 ρ₁ = {stats['rho1']:.3f}. "
            f"Sample mean is not statistically defensible. Widen the window "
            f"to span more independent training intervals.\n"
        )
        return 2

    # ---- Apply LOCKED decision rule (guard 4 is implicit in render_verdict) ----
    verdict, verdict_text = render_verdict(stats, args.threshold)

    # ---- Print to stdout ----
    print(f"=== Paired diff judge: {args.label_a} vs {args.label_b} ===")
    print(f"  metric:    {args.metric_key}")
    print(f"  window:    [{args.step_min}, {args.step_max}]")
    print(f"  threshold: {args.threshold:.2f}  (LOCKED for Risk 4 spot check)")
    print()
    print(f"  N (paired obs):        {stats['n']}")
    print(f"  ρ₁ (lag-1 autocorr):   {stats['rho1']:+.3f}")
    print(f"  N_eff (autocorr-corr): {stats['n_eff']:.1f}")
    print(f"  mean rel diff:         {stats['mean']:+.4f} ({stats['mean'] * 100:+.2f}%)")
    print(f"  std (across obs):      {stats['std']:.4f}")
    print(f"  SE of mean:            {stats['se']:.4f}")
    print(f"  CI95 of mean:          [{stats['ci95_lo']:+.4f}, {stats['ci95_hi']:+.4f}]")
    print(f"  obs range:             [{stats['min']:+.4f}, {stats['max']:+.4f}]")
    print()
    print(f"  VERDICT: {verdict}")
    print(f"  {verdict_text}")
    print()
    if verdict == "no_confound":
        print(f"  Action: proceed to paper write-up. Cite this report as Risk 4 spot check evidence.")
    elif verdict == "confound":
        print(f"  Action: re-frame ablation in paper. Disclose pair-channel sensitivity.")
        print(f"          Recommended additional: full-val on best.pt of both runs.")
    else:
        print(f"  Action: run full-val on best.pt of both runs via")
        print(f"          eval_first_hop_224_clip3.py --max-slices 0;")
        print(f"          if still borderline, report as paper supplementary.")

    # ---- Optional markdown report ----
    if args.output_md is not None:
        md = render_report_md(args, cfg_a, cfg_b, common, deltas, stats, verdict, verdict_text)
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(md)
        print()
        print(f"  Markdown report written to: {args.output_md}")

    # Exit non-zero on confound or borderline so CI / scripts can catch it.
    if verdict == "confound":
        return 10
    if verdict == "borderline":
        return 11
    return 0


if __name__ == "__main__":
    sys.exit(main())
