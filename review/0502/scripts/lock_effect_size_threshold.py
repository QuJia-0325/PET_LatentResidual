#!/usr/bin/env python3
"""Lock the σ-normalize ablation effect-size threshold X (per §6.6 of POST_V6_NEXT_STEPS.md).

This script automates §6.6.2 Step 3-4 of the blinded analysis pre-registration:

    Step 3: Read A_main metrics.jsonl, compute paired_CV_A from val_select_score
            in [step_min, step_max] window.
    Step 4: Apply LOCKED formula  X = max(0.10, 3 × paired_CV_A).
            Write EFFECT_SIZE_LOCKED.md (containing paired_CV_A, X, metrics file
            content hash, training run's git commit hash, formula citation).
            Optionally git commit + push.

LOCKED parameters (these MUST NOT be configurable post-hoc — modifying them is
scientific fraud per §6.6.7):

    floor       0.10
    slope       3.0
    metric_key  val_select_score
    window      step ∈ [step_min, step_max]   (caller specifies, but cannot
                                                widen post-hoc to game results)

The script enforces the LOCKED parameters: caller cannot pass --floor, --slope,
or --metric-key. Window is required and recommended to be [40000, 60000] per
§6.6.1, but caller can specify a different window only if they document why
in EFFECT_SIZE_LOCKED.md (deviation note).

Order-violation guards (these prevent the most common pre-registration breakage):

    Guard 1: working tree must be clean (no uncommitted changes that could
             retroactively pollute the lock-in commit).
    Guard 2: EFFECT_SIZE_LOCKED.md must NOT already exist (else lock has already
             happened — running again = retroactive modification = fraud).
    Guard 3: NO C_uniform full-val artifact may exist when this script runs.
             We detect this by scanning for files matching --c-uniform-output-dir
             that suggest a Layer 4 full-val has already been computed. If found,
             exit 7 (sequence violation: C results were peeked at before X locked).
    Guard 4: A_main metrics.jsonl must contain ≥ 5 rows in the [step_min, step_max]
             window with val_select_score present.
    Guard 5: A_main config yaml must declare the exact training_run_dir we read
             metrics from (sanity check that we're locking against the right run).

Exit codes:
    0   Success: EFFECT_SIZE_LOCKED.md written and (optionally) committed+pushed.
    1   Guard 4 failed (insufficient observations).
    2   Guard 1 failed (dirty working tree).
    3   Guard 2 failed (lock file already exists).
    4   Guard 5 failed (config / run-dir mismatch).
    5   PyYAML missing.
    7   Guard 3 failed (C_uniform results detected — sequence violation).
    8   Git operation failed.

Usage:
    # 1. Sanity preview (no commit)
    python review/0502/scripts/lock_effect_size_threshold.py \
        --metrics-a /data_2/.../A_main/run-.../metrics.jsonl \
        --config-a  review/0502/configs/A_control.yaml \
        --output    review/0502/EFFECT_SIZE_LOCKED.md \
        --step-min  40000 \
        --step-max  60000 \
        --no-commit

    # 2. Real lock-in (commit + push)
    python review/0502/scripts/lock_effect_size_threshold.py \
        --metrics-a /data_2/.../A_main/run-.../metrics.jsonl \
        --config-a  review/0502/configs/A_control.yaml \
        --output    review/0502/EFFECT_SIZE_LOCKED.md \
        --step-min  40000 \
        --step-max  60000 \
        --c-uniform-output-dir /data_2/.../C_uniform/

Status: Implements §6.6.2 Step 3-4 (LOCKED 2026-05-03 in commit 867b5c0).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Local helper: schema-tolerant step extraction. See `_metrics_compat.py` for
# the rationale (trainer writes `step`; legacy fixtures may use `global_step`).
from _metrics_compat import get_row_step

# ---- LOCKED parameters per §6.6.1 — DO NOT EXPOSE AS CLI ARGS ----
LOCKED_FLOOR = 0.10
LOCKED_SLOPE = 3.0
LOCKED_METRIC_KEY = "val_select_score"
LOCKED_RECOMMENDED_WINDOW = (40000, 60000)

# ---- Pre-registration protocol version (S12 from REV1_TOOLING_PLAN §10.2) ----
# This constant is checked by `run_ablation.sh` `require_lock_pass()` (C5b)
# before C_uniform launches; only `"R1b_locked"` permits the launch. The
# value transitions across the commit chain:
#     C1   sets   "v0_pending_R1a"      (this file: pre-R1a tooling baseline)
#     R1a  sets   "R1a_pending_data"    (skeleton committed; A_main / A_seed=43 not yet at 120K)
#     R1b  sets   "R1b_locked"          (numeric paired_SD_AA + X embedded; C may launch)
# Any other value at C-launch time → hard-gate refuses.
LOCKED_PROTOCOL_VERSION = "v0_pending_R1a"
LOCKED_FORMULA_CITATION = (
    "X = max(0.10, 3 × paired_CV_A)  per POST_V6_NEXT_STEPS.md §6.6.1"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--metrics-a", required=True, type=Path,
                   help="Path to A_main metrics.jsonl")
    p.add_argument("--config-a", required=True, type=Path,
                   help="Path to A_main yaml (used by guard 5 sanity check)")
    p.add_argument("--output", required=True, type=Path,
                   help="Path to write EFFECT_SIZE_LOCKED.md (typically "
                        "review/0502/EFFECT_SIZE_LOCKED.md)")
    p.add_argument("--step-min", type=int, default=LOCKED_RECOMMENDED_WINDOW[0],
                   help=f"Inclusive lower step bound. LOCKED default = "
                        f"{LOCKED_RECOMMENDED_WINDOW[0]}. If you change this "
                        f"you MUST document why in the deviation note.")
    p.add_argument("--step-max", type=int, default=LOCKED_RECOMMENDED_WINDOW[1],
                   help=f"Inclusive upper step bound. LOCKED default = "
                        f"{LOCKED_RECOMMENDED_WINDOW[1]}. Same deviation rule.")
    p.add_argument("--c-uniform-output-dir", type=Path, default=None,
                   help="Path to C_uniform run output dir (or its eval output "
                        "dir). If provided, guard 3 scans for 'full_val' or "
                        "Method-D-selected ckpt artifacts that would indicate "
                        "C results were peeked at before X was locked.")
    p.add_argument("--deviation-note", type=str, default=None,
                   help="Required if --step-min or --step-max differ from the "
                        "LOCKED default window. Free-text rationale that will "
                        "be embedded in EFFECT_SIZE_LOCKED.md.")
    p.add_argument("--no-commit", action="store_true",
                   help="Don't auto-commit. Useful for previewing the X value "
                        "before locking in. Files are still written to --output.")
    p.add_argument("--no-push", action="store_true",
                   help="Commit but don't push. Default is to push to gitee "
                        "remote so pre-registration timestamp is publicly verifiable.")
    p.add_argument("--remote", default="gitee",
                   help="Git remote to push to. Default 'gitee'.")
    p.add_argument("--allow-dirty", action="store_true",
                   help="DANGEROUS: bypass guard 1 (clean working tree). Only "
                        "for development testing; never use in production.")
    return p.parse_args()


def load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError:
        sys.stderr.write(
            "ERROR: PyYAML not installed. `pip install pyyaml` and retry.\n"
        )
        sys.exit(5)
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


def load_metrics_with_hash(path: Path) -> tuple[list[dict], str]:
    """Read file as bytes ONCE, hash those bytes, then parse from them.

    Fixes the Cl-Q6 SHA256-race bug (Round-1 review): previously
    `compute_paired_cv()` called `load_metrics()` (file open at T0) and
    `file_sha256()` (separate file open at T1); if the trainer appended
    rows in [T0, T1] the hash would cover bytes never seen by the parser.
    This helper guarantees the returned rows and hash refer to the exact
    same byte stream.
    """
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    rows: list[dict] = []
    for line in raw.splitlines():
        s = line.decode("utf-8", errors="replace").strip()
        if not s:
            continue
        try:
            rows.append(json.loads(s))
        except json.JSONDecodeError:
            continue
    return rows, sha


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git(args: list[str], cwd: Path | None = None) -> str:
    res = subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        sys.stderr.write(
            f"git {' '.join(args)} failed (exit {res.returncode}):\n{res.stderr}\n"
        )
        raise RuntimeError(f"git command failed: git {' '.join(args)}")
    return res.stdout.strip()


def find_repo_root(path: Path) -> Path:
    """Find git repo root by walking up from path."""
    cur = path.resolve()
    while cur != cur.parent:
        if (cur / ".git").exists():
            return cur
        cur = cur.parent
    raise RuntimeError(f"No .git found above {path}")


def check_clean_tree(repo_root: Path) -> tuple[bool, str]:
    out = git(["status", "--porcelain"], cwd=repo_root)
    return (out == ""), out


def detect_c_uniform_artifacts(c_dir: Path | None) -> list[Path]:
    """Guard 3: detect any artifact that suggests C_uniform was already eval'd.

    Looks for:
      - eval_*.json or *full_val* files in the C_uniform output dir
      - Method-D-selected ckpt eval outputs
    """
    if c_dir is None:
        return []
    if not c_dir.exists():
        return []
    suspect: list[Path] = []
    patterns = ("*full_val*", "*eval*.json", "*method_d*", "*selected_ckpt*")
    for pat in patterns:
        suspect.extend(c_dir.rglob(pat))
    # Filter out training metrics.jsonl itself (that's allowed to exist;
    # it's only forbidden to have run a SEPARATE full-val on top of it)
    suspect = [p for p in suspect if p.name != "metrics.jsonl"]
    return suspect


def compute_paired_cv(
    metrics_path: Path,
    step_min: int,
    step_max: int,
) -> tuple[float, float, float, int, list[float], str]:
    """Returns (paired_CV, mean, std, N, raw_values, metrics_sha256).

    The metrics SHA256 is returned from the SAME byte read used to populate
    `raw_values`, eliminating the Cl-Q6 race window between parse-time and
    hash-time.
    """
    rows, sha = load_metrics_with_hash(metrics_path)
    series: list[float] = []
    for r in rows:
        step = get_row_step(r)
        if step is None:
            continue
        if LOCKED_METRIC_KEY not in r:
            continue
        try:
            val = float(r[LOCKED_METRIC_KEY])
        except (TypeError, ValueError):
            continue
        if step_min <= step <= step_max and val > 0.0:
            series.append(val)

    n = len(series)
    if n < 5:
        return 0.0, 0.0, 0.0, n, series, sha
    mean = sum(series) / n
    var = sum((v - mean) ** 2 for v in series) / (n - 1)
    std = math.sqrt(var)
    cv = std / mean if mean > 0 else 0.0
    return cv, mean, std, n, series, sha


def render_lock_md(
    args: argparse.Namespace,
    paired_cv: float,
    mean: float,
    std: float,
    n: int,
    x_value: float,
    metrics_hash: str,
    train_git_commit: str,
    deviation_note: str | None,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    using_default_window = (
        args.step_min == LOCKED_RECOMMENDED_WINDOW[0]
        and args.step_max == LOCKED_RECOMMENDED_WINDOW[1]
    )

    lines: list[str] = []
    lines.append("# EFFECT_SIZE_LOCKED.md — σ-normalize ablation X threshold")
    lines.append("")
    lines.append(
        "**STATUS: LOCKED.** This file is the pre-registration artifact "
        "for [POST_V6_NEXT_STEPS.md §6.6](review/0502/POST_V6_NEXT_STEPS.md). "
        "Modifying any number below = scientific fraud."
    )
    lines.append("")
    lines.append("## Lock-in metadata")
    lines.append("")
    lines.append(f"- **Lock timestamp**: {now}")
    lines.append(f"- **Generated by**: `review/0502/scripts/lock_effect_size_threshold.py`")
    lines.append(f"- **A_main metrics path**: `{args.metrics_a}`")
    lines.append(f"- **A_main metrics SHA256**: `{metrics_hash}`")
    lines.append(f"- **A_main config path**: `{args.config_a}`")
    lines.append(f"- **Repo HEAD at lock time**: `{train_git_commit}`")
    lines.append(f"- **LOCKED_PROTOCOL_VERSION**: `{LOCKED_PROTOCOL_VERSION}`")
    lines.append("")
    lines.append("## Computation")
    lines.append("")
    lines.append(f"- **Formula (LOCKED at commit 867b5c0)**: {LOCKED_FORMULA_CITATION}")
    lines.append(f"- **Metric key**: `{LOCKED_METRIC_KEY}`")
    lines.append(
        f"- **Window**: `step ∈ [{args.step_min}, {args.step_max}]`"
        + ("" if using_default_window
           else f"  ⚠️ **DEVIATION** from LOCKED default "
                f"`[{LOCKED_RECOMMENDED_WINDOW[0]}, "
                f"{LOCKED_RECOMMENDED_WINDOW[1]}]`")
    )
    lines.append(f"- **N (observations in window)**: {n}")
    lines.append(f"- **mean(val_select_score)**: {mean:.6e}")
    lines.append(f"- **std(val_select_score)**: {std:.6e}")
    lines.append(f"- **paired_CV_A** = std / mean = `{paired_cv:.4f}` ({paired_cv * 100:.2f}%)")
    lines.append(
        f"- **3 × paired_CV_A** = `{3.0 * paired_cv:.4f}` ({3.0 * paired_cv * 100:.2f}%)"
    )
    lines.append(f"- **floor (LOCKED)** = `{LOCKED_FLOOR:.4f}` ({LOCKED_FLOOR * 100:.0f}%)")
    lines.append("")
    lines.append("## Result")
    lines.append("")
    lines.append(f"## **X = max(floor, 3 × paired_CV_A) = `{x_value:.4f}` ({x_value * 100:.2f}%)**")
    lines.append("")
    lines.append("## Decision rule (LOCKED, per §6.6.3)")
    lines.append("")
    lines.append(
        f"After Method D + full-val on **C_uniform** top-2 ckpts, compute "
        f"`rel_diff = (mean(C_top2) - mean(A_top2)) / mean(A_top2)`."
    )
    lines.append("")
    lines.append("| `rel_diff`                           | Conclusion / Action                                                                                                                                                                                                                  |")
    lines.append("|---|---|")
    lines.append(
        f"| `rel_diff < {x_value:.4f}` ({x_value * 100:.2f}%)             | **C ≈ A.** σ-norm is the primary lever. Paper claim: \"step-weight shape contributes little beyond scale compensation under the current pair-heavy V6 setup.\"                                          |"
    )
    lines.append(
        f"| `{x_value:.4f} ≤ rel_diff ≤ {2 * x_value:.4f}` ({x_value * 100:.2f}–{2 * x_value * 100:.2f}%)        | **Grey zone.** Run 200K continuation on A_main best.pt + C_uniform best.pt; re-evaluate with the same X. If still grey after 200K: report both numbers, no claim.                                                                       |"
    )
    lines.append(
        f"| `rel_diff > {2 * x_value:.4f}` ({2 * x_value * 100:.2f}%)            | **C ≠ A.** \"After σ-normalization, V6 middle-heavy weighting still helps → genuine hop-shape effect.\" Reframe paper as 'we validate V6 with rigor', σ-norm becomes controlled-comparison methodology.                                  |"
    )
    lines.append("")
    if deviation_note:
        lines.append("## Deviation note")
        lines.append("")
        lines.append(deviation_note)
        lines.append("")
    lines.append("## Raw observations")
    lines.append("")
    lines.append(
        "First / last 5 of the in-window val_select_score series (full series "
        "reproducible from `metrics.jsonl @ {sha256}` and the LOCKED window).".format(
            sha256=metrics_hash[:16]
        )
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()

    # ---- Refuse CLI overrides of LOCKED parameters via undocumented args ----
    # (argparse doesn't expose these; this is just a sanity assertion.)
    assert LOCKED_METRIC_KEY == "val_select_score"
    assert LOCKED_FLOOR == 0.10
    assert LOCKED_SLOPE == 3.0

    # ---- Window deviation: require a deviation note ----
    using_default_window = (
        args.step_min == LOCKED_RECOMMENDED_WINDOW[0]
        and args.step_max == LOCKED_RECOMMENDED_WINDOW[1]
    )
    if not using_default_window and not args.deviation_note:
        sys.stderr.write(
            f"ERROR: --step-min/--step-max differ from LOCKED default "
            f"[{LOCKED_RECOMMENDED_WINDOW[0]}, {LOCKED_RECOMMENDED_WINDOW[1]}]; "
            f"you must pass --deviation-note explaining why. This goes into "
            f"EFFECT_SIZE_LOCKED.md and becomes part of the public pre-reg.\n"
        )
        return 4

    # ---- Resolve paths ----
    output = args.output.resolve()
    metrics_a = args.metrics_a.resolve()
    config_a = args.config_a.resolve()
    repo_root = find_repo_root(output)

    # ---- Guard 2: lock file must not already exist ----
    if output.exists():
        sys.stderr.write(
            f"ERROR (guard 2): {output} already exists. Lock has already been "
            f"performed. Re-running this script = retroactive modification = "
            f"scientific fraud per §6.6.7. If you really need to deviate, "
            f"add a deviation note to the existing file MANUALLY in a new "
            f"git commit; do NOT use this script.\n"
        )
        return 3

    # ---- Guard 1: clean working tree ----
    if not args.allow_dirty:
        clean, dirty_files = check_clean_tree(repo_root)
        if not clean:
            sys.stderr.write(
                f"ERROR (guard 1): working tree is dirty. Lock-in commit must "
                f"be atomic — uncommitted changes in the tree could pollute the "
                f"pre-registration timestamp. Commit or stash first.\n\n"
                f"Dirty files:\n{dirty_files}\n\n"
                f"To bypass (DEV ONLY, never in production), pass --allow-dirty.\n"
            )
            return 2

    # ---- Guard 3: no C_uniform artifacts ----
    suspect = detect_c_uniform_artifacts(args.c_uniform_output_dir)
    if suspect:
        sys.stderr.write(
            f"ERROR (guard 3): SEQUENCE VIOLATION. C_uniform artifacts already "
            f"exist in {args.c_uniform_output_dir} — implies C results were "
            f"already evaluated before X was locked. This destroys the blinded "
            f"analysis pre-registration.\n\n"
            f"Suspect files:\n"
        )
        for p in suspect[:20]:
            sys.stderr.write(f"  {p}\n")
        if len(suspect) > 20:
            sys.stderr.write(f"  ... and {len(suspect) - 20} more\n")
        sys.stderr.write(
            "\nRecovery options:\n"
            "  1. If those artifacts pre-date the A_main run (they're left over "
            "     from an earlier experiment), move/delete them and retry.\n"
            "  2. If they were really computed on the current C_uniform run "
            "     before reading this script, the §6.6 pre-reg is broken; you "
            "     must add a deviation note to POST_V6_NEXT_STEPS.md §6.6 and "
            "     accept reduced reviewer-defensibility.\n"
        )
        return 7

    # ---- Guard 5 (config side per Op-flag-5): config-side best_metric ----
    # Operator round-2.5 reply (`OPERATOR_REPLY_pre_C1_20260503.md` Op-flag-5)
    # clarified that production configs declare `best_metric: val_multi_objective`
    # (config-side) while the metrics row's selection-aware key is
    # `val_select_score` (data-side computed). Both must be checked. The data-
    # side check is performed inside `compute_paired_cv()` via the LOCKED_METRIC_KEY
    # filter (rows missing `val_select_score` are skipped → if 0 rows survive,
    # Guard 4 catches it as N<5).
    cfg_a = load_yaml(config_a)
    cfg_metric_key = cfg_a.get("training", {}).get("best_metric")
    if cfg_metric_key not in (None, "val_multi_objective", "val_select_score"):
        sys.stderr.write(
            f"ERROR (guard 5 config-side): A_main config best_metric="
            f"'{cfg_metric_key}' is unexpected. §6.6 was designed assuming "
            f"val_multi_objective → val_select_score key (see "
            f"POST_V6_NEXT_STEPS.md §0 + OPERATOR_REPLY_pre_C1_20260503.md "
            f"Op-flag-5). Aborting to avoid locking against the wrong run.\n"
        )
        return 4

    # ---- Compute paired_CV_A (also returns SHA256 of the exact byte stream
    # parsed for the series, eliminating the Cl-Q6 race window) ----
    paired_cv, mean, std, n, series, metrics_hash = compute_paired_cv(
        metrics_a, args.step_min, args.step_max
    )

    # ---- Guard 4: minimum N ----
    if n < 5:
        sys.stderr.write(
            f"ERROR (guard 4): only {n} val_select_score observations in "
            f"window [{args.step_min}, {args.step_max}] of {metrics_a}. "
            f"Need ≥ 5. Either A_main hasn't reached step_max yet, or the "
            f"metric key is missing. Wait for training to progress further.\n"
        )
        return 1

    # ---- Guard 5 (data-side per Op-flag-5): val_select_score must appear
    # ----                                       in actual metrics rows ----
    # If `compute_paired_cv` returned N≥5, val_select_score is present by
    # construction (it filters rows on LOCKED_METRIC_KEY). This is an
    # explicit re-check + clearer error so a config-vs-data mismatch is not
    # silently swallowed by Guard 4's "N<5" message.
    rows_for_check, _ = load_metrics_with_hash(metrics_a)
    has_select = any(LOCKED_METRIC_KEY in r for r in rows_for_check)
    if not has_select:
        sys.stderr.write(
            f"ERROR (guard 5 data-side): metrics file {metrics_a} contains no "
            f"row with key '{LOCKED_METRIC_KEY}'. Per Op-flag-5 the data-side "
            f"selection key is computed by the trainer (Method-D selector); "
            f"its absence means either the run uses a different selector or "
            f"the metrics file is from a stale schema. Aborting.\n"
        )
        return 4

    # ---- Apply LOCKED formula ----
    x_value = max(LOCKED_FLOOR, LOCKED_SLOPE * paired_cv)

    # ---- Compute artifacts ----
    # metrics_hash already obtained from compute_paired_cv (race-free).
    try:
        repo_head = git(["rev-parse", "HEAD"], cwd=repo_root)
    except RuntimeError:
        repo_head = "<unknown — git rev-parse failed>"
    md = render_lock_md(
        args=args,
        paired_cv=paired_cv,
        mean=mean,
        std=std,
        n=n,
        x_value=x_value,
        metrics_hash=metrics_hash,
        train_git_commit=repo_head,
        deviation_note=args.deviation_note,
    )

    # Append raw observations (first 5 + last 5)
    md_lines = md.rstrip("\n").split("\n")
    md_lines.append("| Index | step | val_select_score |")
    md_lines.append("|---|---|---|")
    rows, _ = load_metrics_with_hash(metrics_a)
    in_window: list[tuple[int, float]] = []
    for r in rows:
        step = get_row_step(r)
        if step is None or LOCKED_METRIC_KEY not in r:
            continue
        try:
            val = float(r[LOCKED_METRIC_KEY])
        except (TypeError, ValueError):
            continue
        if args.step_min <= step <= args.step_max and val > 0.0:
            in_window.append((step, val))
    show_rows = in_window[:5] + in_window[-5:] if len(in_window) > 10 else in_window
    for i, (step, val) in enumerate(show_rows):
        md_lines.append(f"| {i} | {step} | {val:.6e} |")
    if len(in_window) > 10:
        md_lines.insert(len(md_lines) - 5, f"| ... | ... | (showing 5+5 of {len(in_window)} obs) |")
    md = "\n".join(md_lines) + "\n"

    # ---- Write file ----
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(md)
    print(f"Written: {output}")
    print()
    print(f"  paired_CV_A = {paired_cv:.4f} ({paired_cv * 100:.2f}%)")
    print(f"  3 × paired_CV_A = {3.0 * paired_cv:.4f}")
    print(f"  floor = {LOCKED_FLOOR}")
    print(f"  X = max(floor, 3·CV) = {x_value:.4f} ({x_value * 100:.2f}%)")
    print(f"  metrics SHA256 = {metrics_hash}")
    print()

    # ---- Commit + push (unless --no-commit) ----
    if args.no_commit:
        print("--no-commit set: file written but NOT committed.")
        print("To finalize the lock-in, commit this file and push.")
        return 0

    rel_output = output.relative_to(repo_root)
    commit_msg = (
        f"review/0502: EFFECT_SIZE_LOCKED.md — X={x_value:.4f} "
        f"(paired_CV_A={paired_cv:.4f}, n={n})\n\n"
        f"Per POST_V6_NEXT_STEPS.md §6.6.2 Step 4. Locks the σ-normalize ablation\n"
        f"effect-size threshold X before any C_uniform full-val eval. This commit\n"
        f"is the pre-registration timestamp evidence.\n\n"
        f"  formula:        max(0.10, 3 × paired_CV_A)\n"
        f"  paired_CV_A:    {paired_cv:.4f} ({paired_cv * 100:.2f}%)\n"
        f"  floor:          {LOCKED_FLOOR:.2f}\n"
        f"  X:              {x_value:.4f} ({x_value * 100:.2f}%)\n"
        f"  window:         step ∈ [{args.step_min}, {args.step_max}]\n"
        f"  metric:         {LOCKED_METRIC_KEY}\n"
        f"  N:              {n}\n"
        f"  metrics SHA256: {metrics_hash}\n"
    )
    try:
        git(["add", str(rel_output)], cwd=repo_root)
        # Use stdin to safely handle commit message
        proc = subprocess.run(
            ["git", "commit", "-F", "-"],
            cwd=repo_root,
            input=commit_msg,
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            sys.stderr.write(f"git commit failed: {proc.stderr}\n")
            return 8
        commit_hash = git(["rev-parse", "HEAD"], cwd=repo_root)
        print(f"Committed: {commit_hash}")
    except RuntimeError:
        return 8

    if args.no_push:
        print("--no-push set: committed locally but not pushed.")
        print("To finalize the public pre-registration, push to remote.")
        return 0

    try:
        # Detect current branch
        branch = git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
        proc = subprocess.run(
            ["git", "push", args.remote, branch],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            sys.stderr.write(
                f"git push {args.remote} {branch} failed:\n{proc.stderr}\n"
                f"You committed locally; push manually to finalize.\n"
            )
            return 8
        print(f"Pushed: {args.remote}/{branch}")
    except RuntimeError:
        return 8

    print()
    print("EFFECT_SIZE_LOCKED.md is now public. X is LOCKED.")
    print("You may now proceed to §6.6.2 Step 5 (Method D + full-val on C_uniform).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
