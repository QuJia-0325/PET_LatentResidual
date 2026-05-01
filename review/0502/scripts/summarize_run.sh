#!/usr/bin/env bash
# review/0502/scripts/summarize_run.sh
#
# Quick best-checkpoint summary from a metrics.jsonl file (pure-python — no jq).
# Pulls best/last of:
#   val_chain_d20_mse, val_chain_d10_mse, val_chain_d4_mse, val_chain_normal_mse
#   val_pair_total, val_rollout_total, val_multi_objective
#   val_rollout_step_*_raw  (raw σ-norm-independent per-hop diagnostics)
#
# Usage:
#   bash review/0502/scripts/summarize_run.sh <run_dir>
#       where <run_dir> is either:
#         - review/0502/runs/<tag>/                (auto-discovers metrics.jsonl)
#         - <output_dir>/<run_name>/               (the actual training output dir)
#         - direct path to metrics.jsonl

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <run_dir_or_metrics.jsonl>" >&2
    exit 2
fi

ARG="$1"
JSONL=""
PYTHON="${PYTHON:-python}"

if [[ -f "${ARG}" ]] && [[ "${ARG}" == *metrics.jsonl ]]; then
    JSONL="${ARG}"
elif [[ -d "${ARG}" ]]; then
    # Try common locations (handles both the new run_ablation.sh layout and direct output dirs).
    for cand in \
        "${ARG}/metrics.jsonl" \
        "${ARG}/run/"*"/metrics.jsonl" \
        "${ARG}/"*"/metrics.jsonl"; do
        if [[ -f "${cand}" ]]; then
            JSONL="${cand}"
            break
        fi
    done
    # If a config.resolved.yaml exists, try to look up the actual training output dir.
    if [[ -z "${JSONL}" ]] && [[ -f "${ARG}/config.resolved.yaml" ]]; then
        out=$(awk -F': *' '/^output_dir:/ { gsub(/[ \t"]/,"",$2); print $2; exit }' "${ARG}/config.resolved.yaml")
        run=$(awk -F': *' '/^run_name:/  { gsub(/[ \t"]/,"",$2); print $2; exit }' "${ARG}/config.resolved.yaml")
        if [[ -n "${out}" && -n "${run}" && -f "${out}/${run}/metrics.jsonl" ]]; then
            JSONL="${out}/${run}/metrics.jsonl"
        fi
    fi
fi

if [[ -z "${JSONL}" ]]; then
    echo "ERROR: could not locate metrics.jsonl from '${ARG}'" >&2
    exit 1
fi

echo "============================================================"
echo "metrics file: ${JSONL}"
echo "  total events: $(wc -l < "${JSONL}" | tr -d ' ')"
echo "============================================================"

JSONL="${JSONL}" "${PYTHON}" - <<'PYEOF'
import json, os, sys

path = os.environ["JSONL"]
rows = []
with open(path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue

if not rows:
    print("ERROR: no parseable JSON rows in metrics.jsonl", file=sys.stderr)
    sys.exit(1)

def step_of(r):
    return r.get("step", r.get("global_step", "?"))

primary = [
    "val_chain_d20_mse",
    "val_chain_d10_mse",
    "val_chain_d4_mse",
    "val_chain_normal_mse",
    "val_multi_objective",
    "val_rollout_total",
    "val_pair_total",
]

print("=== Best (= min) per metric ===")
for k in primary:
    matched = [(step_of(r), r[k]) for r in rows if k in r and r[k] is not None]
    if not matched:
        print(f"  {k:<25} <absent>")
        continue
    s, v = min(matched, key=lambda x: x[1])
    print(f"  {k:<25} best={v:>14.6e}  @ step {s}")

print()
print("=== Raw per-hop val_rollout_step_*_raw (last eval) ===")
last_with_raw = None
for r in reversed(rows):
    if any(k.startswith("val_rollout_step_") and k.endswith("_raw") for k in r):
        last_with_raw = r
        break
if last_with_raw is None:
    print("  <no raw step losses found — patches may not be applied>")
else:
    s = step_of(last_with_raw)
    raws = sorted(
        (k, v) for k, v in last_with_raw.items()
        if k.startswith("val_rollout_step_") and k.endswith("_raw")
    )
    print(f"  step = {s}")
    for k, v in raws:
        print(f"  {k:<28} = {v:.6e}")

print()
print("=== Last event (key fields) ===")
last = rows[-1]
keep = ["step", "global_step", "val_chain_normal_mse", "val_rollout_total", "val_pair_total"]
out = {k: last[k] for k in keep if k in last}
print(json.dumps(out, indent=2))
PYEOF
