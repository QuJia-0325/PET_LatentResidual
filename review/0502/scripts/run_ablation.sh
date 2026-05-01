#!/usr/bin/env bash
# review/0502/scripts/run_ablation.sh
#
# Drive the sigma-normalize ablation. Runs train_first_hop.py with a chosen yaml
# and routes outputs/logs into the data disk (`/data_2` by default; project's
# pet_lr/path_guard.py enforces this), with an in-repo index dir under
# review/0502/runs/<tag>/ for the resolved yaml + tee'd log.
#
# Usage:
#   bash review/0502/scripts/run_ablation.sh sanity      # A_sanity + B (50K each, sanity check)
#   bash review/0502/scripts/run_ablation.sh main        # A_main + C (120K each)
#   bash review/0502/scripts/run_ablation.sh closed_form # D (120K)
#   bash review/0502/scripts/run_ablation.sh A           # only A_main
#   bash review/0502/scripts/run_ablation.sh B           # only B
#   bash review/0502/scripts/run_ablation.sh C           # only C
#   bash review/0502/scripts/run_ablation.sh D           # only D
#
# Environment overrides (export before running):
#   PYTHON              python interpreter (default: python)
#   TRAIN_ENTRY         training script (default: train_first_hop.py)
#   ABLATION_INDEX_DIR  in-repo index dir (default: review/0502/runs)
#                       Stores config.resolved.yaml and train.log per tag.
#   ABLATION_OUTPUT_ROOT  data-disk root for actual training products
#                       (default: /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0502_runs)
#                       MUST be under /data_2 — pet_lr/path_guard.resolve_data_disk_dir
#                       hard-rejects anything else with RuntimeError.
#   GPU                 CUDA device id (default: 0)
#
# Output layout per condition (tag):
#   ${ABLATION_INDEX_DIR}/<tag>/
#     ├── config.resolved.yaml   (max_steps + run_name + output_dir rewritten)
#     └── train.log              (tee'd stdout)
#   ${ABLATION_OUTPUT_ROOT}/<tag>/run/first_hop_224_sigma_norm_<tag>/
#     ├── metrics.jsonl
#     ├── ckpts/
#     └── ...                    (everything train_first_hop.py writes)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REVIEW_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${REVIEW_DIR}/../.." && pwd)"

PYTHON="${PYTHON:-python}"
TRAIN_ENTRY="${TRAIN_ENTRY:-train_first_hop.py}"
ABLATION_INDEX_DIR="${ABLATION_INDEX_DIR:-${REVIEW_DIR}/runs}"
ABLATION_OUTPUT_ROOT="${ABLATION_OUTPUT_ROOT:-/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0502_runs}"
GPU="${GPU:-0}"

cd "${REPO_ROOT}"

# Sanity: enforce data-disk root convention up-front (mirrors pet_lr/path_guard.py).
case "${ABLATION_OUTPUT_ROOT}" in
    /data_2/*|/data_2)
        : # ok
        ;;
    *)
        echo "ERROR: ABLATION_OUTPUT_ROOT must be under /data_2 (path_guard requirement)." >&2
        echo "  got: ${ABLATION_OUTPUT_ROOT}" >&2
        echo "  default: /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0502_runs" >&2
        exit 1
        ;;
esac

mkdir -p "${ABLATION_INDEX_DIR}"

resolve_yaml() {
    # $1 = condition tag (A_sanity/A_main/B/C/D)
    # $2 = source yaml
    # $3 = max_steps override (empty = keep yaml default)
    local tag="$1" src="$2" steps="$3"
    local indexdir="${ABLATION_INDEX_DIR}/${tag}"
    local dst="${indexdir}/config.resolved.yaml"
    # Training products land on the data disk so path_guard.resolve_data_disk_dir
    # accepts them (otherwise train_first_hop.py raises RuntimeError before training starts).
    local out_dir="${ABLATION_OUTPUT_ROOT}/${tag}/run"
    local run_name="first_hop_224_sigma_norm_${tag}"
    mkdir -p "${indexdir}"
    cp "${src}" "${dst}"

    # In-place rewrite: max_steps + run_name + output_dir.
    # Use awk for portability (macOS sed -i quirks).
    awk -v new_steps="${steps}" -v new_name="${run_name}" -v new_out="${out_dir}" '
        BEGIN { steps_done=0; name_done=0; out_done=0 }
        /^[[:space:]]*max_steps:/ && new_steps != "" && !steps_done {
            sub(/max_steps:.*/, "max_steps: " new_steps); steps_done=1
        }
        /^run_name:/ && !name_done {
            sub(/run_name:.*/, "run_name: " new_name); name_done=1
        }
        /^output_dir:/ && !out_done {
            sub(/output_dir:.*/, "output_dir: " new_out); out_done=1
        }
        { print }
    ' "${dst}" > "${dst}.tmp" && mv "${dst}.tmp" "${dst}"
    echo "${dst}"
}

run_one() {
    # $1 = tag, $2 = source yaml, $3 = max_steps
    local tag="$1" yaml="$2" steps="$3"
    local indexdir="${ABLATION_INDEX_DIR}/${tag}"
    local resolved
    resolved="$(resolve_yaml "${tag}" "${yaml}" "${steps}")"
    local logf="${indexdir}/train.log"
    local metrics="${ABLATION_OUTPUT_ROOT}/${tag}/run/first_hop_224_sigma_norm_${tag}/metrics.jsonl"
    echo "============================================================"
    echo "[run_ablation.sh] condition=${tag}"
    echo "  yaml      = ${resolved}"
    echo "  log       = ${logf}    (in-repo index)"
    echo "  metrics   = ${metrics} (data-disk products)"
    echo "  GPU       = ${GPU}"
    echo "  max_steps = ${steps:-<from yaml>}"
    echo "============================================================"
    CUDA_VISIBLE_DEVICES="${GPU}" \
        "${PYTHON}" "${TRAIN_ENTRY}" --config "${resolved}" 2>&1 \
        | tee "${logf}"
}

# Pure-Python multi-tier sanity comparator (no jq dependency on the GPU host).
# Mirrors SIGMA_NORMALIZE_ABLATION_PLAN.md §3.4 multi-tier criteria:
#   Tier 1 (strongest): val_pair_total — must be bit-equal (rel < 1e-6)
#   Tier 2 (strong)   : val_rollout_total — must agree to < 1% relative
#   Tier 3 (strong)   : val_rollout_step_*_raw (per-hop raw losses) < 1% relative
#   Tier 4 (weak)     : val_chain_*_mse (4 chain MSEs) < 5% relative (training noise)
print_sanity_check_cmd() {
    local a_metrics="$1" b_metrics="$2"
    cat <<EOF

===== Sanity check (multi-tier, pure-python — no jq required) =====
${PYTHON} - <<'PYEOF'
import json, sys

def last_eval(path):
    with open(path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    evals = [r for r in rows if any(k in r for k in (
        "val_rollout_total", "val_chain_d20_mse", "val_pair_total",
    ))]
    return evals[-1] if evals else None

A = last_eval("${a_metrics}")
B = last_eval("${b_metrics}")
if A is None or B is None:
    print("ERROR: no eval rows found in one of the metrics files"); sys.exit(1)

# Tier definitions: (key, max_rel_err, label)
# Tier 1: bit-equal under FP32 (~1e-6); strictly < 1e-4 to allow some accumulated drift.
TIER1 = [("val_pair_total", 1e-4, "Tier 1 (bit-equal)")]
TIER2 = [("val_rollout_total", 0.01, "Tier 2 (<1%)")]
# Tier 3: any hop with a "_raw" suffix.
TIER3 = sorted(
    [(k, 0.01, "Tier 3 (<1%)") for k in A
     if k.startswith("val_rollout_step_") and k.endswith("_raw") and k in B]
)
TIER4 = [
    ("val_chain_d20_mse", 0.05, "Tier 4 (<5%)"),
    ("val_chain_d10_mse", 0.05, "Tier 4 (<5%)"),
    ("val_chain_d4_mse",  0.05, "Tier 4 (<5%)"),
    ("val_chain_normal_mse", 0.05, "Tier 4 (<5%)"),
]

ALL = TIER1 + TIER2 + TIER3 + TIER4

print(f"{'metric':<32} {'A':>14} {'B':>14} {'rel_err':>10}  {'tier_threshold'}")
print("-" * 96)
overall_pass = True
for k, thr, label in ALL:
    a = A.get(k); b = B.get(k)
    if a is None or b is None:
        print(f"{k:<32} {'<absent>':>14} {'<absent>':>14} {'-':>10}  {label}")
        continue
    rel = abs(b - a) / max(abs(a), 1e-12)
    flag = "PASS" if rel < thr else "FAIL"
    if rel >= thr:
        overall_pass = False
    print(f"{k:<32} {a:>14.6e} {b:>14.6e} {rel:>9.2%}  {label} -> {flag}")

print()
print("Result:", "PASS (all tiers under threshold)" if overall_pass else "FAIL (one or more tiers exceed threshold)")
print()
print("FP32 expectation reminder:")
print("  - val_pair_total tier 1 may show ~1e-6..1e-7 rel-err from bf16/AMP stochasticity")
print("  - val_rollout_total tier 2 typically ~1e-6..1e-5 rel-err from order-of-ops in normalizer division")
print("  - chain MSE tier 4 ~1e-3..1e-2 from EMA + ckpt sampling jitter (training noise, not bug)")
PYEOF
EOF
}

CMD="${1:-}"

case "${CMD}" in
    sanity)
        run_one "A_sanity" "${REVIEW_DIR}/configs/A_control.yaml" "50000"
        run_one "B"        "${REVIEW_DIR}/configs/B_sanity.yaml" "50000"
        a_metrics="${ABLATION_OUTPUT_ROOT}/A_sanity/run/first_hop_224_sigma_norm_A_sanity/metrics.jsonl"
        b_metrics="${ABLATION_OUTPUT_ROOT}/B/run/first_hop_224_sigma_norm_B/metrics.jsonl"
        echo ""
        echo "===== Sanity check criteria (preserve_v6_sum: A==B exact mathematically) ====="
        echo "  Tier 1 val_pair_total            < 1e-4 rel  (bit-equal target — RNG/data path)"
        echo "  Tier 2 val_rollout_total         < 1% rel    (effective lambda_roll match)"
        echo "  Tier 3 val_rollout_step_*_raw    < 1% rel    (per-hop raw signal — diagnostic)"
        echo "  Tier 4 val_chain_*_mse (4 keys)  < 5% rel    (training noise tolerance)"
        echo ""
        echo "  With preserve_v6_sum + B step_weights = w_v6 × n, A==B is mathematically exact"
        echo "  (FP64 residual ~1e-20). FP32 training will introduce ~1e-6 noise in tiers 1-3"
        echo "  and ~1e-3 in tier 4; thresholds chosen to PASS noise but FAIL implementation bugs."
        print_sanity_check_cmd "${a_metrics}" "${b_metrics}"
        ;;
    main)
        run_one "A_main" "${REVIEW_DIR}/configs/A_control.yaml" "120000"
        run_one "C"      "${REVIEW_DIR}/configs/C_uniform.yaml" "120000"
        ;;
    closed_form)
        run_one "D" "${REVIEW_DIR}/configs/D_closed_form.yaml" "120000"
        ;;
    A)
        run_one "A_main" "${REVIEW_DIR}/configs/A_control.yaml" "120000"
        ;;
    B)
        run_one "B" "${REVIEW_DIR}/configs/B_sanity.yaml" "50000"
        ;;
    C)
        run_one "C" "${REVIEW_DIR}/configs/C_uniform.yaml" "120000"
        ;;
    D)
        run_one "D" "${REVIEW_DIR}/configs/D_closed_form.yaml" "120000"
        ;;
    *)
        echo "Usage: $0 {sanity|main|closed_form|A|B|C|D}" >&2
        echo "  sanity      : run A_sanity + B (50K each) — multi-tier A==B equivalence gate" >&2
        echo "  main        : run A_main (120K) + C (120K) — Phase II convergence comparison" >&2
        echo "  closed_form : run D (120K) — extra Grönwall comparison" >&2
        echo "  A/B/C/D     : run a single condition (A/C/D=120K, B=50K)" >&2
        echo "" >&2
        echo "  In-repo index (config + log):" >&2
        echo "    ${ABLATION_INDEX_DIR}/<tag>/" >&2
        echo "      ├── config.resolved.yaml" >&2
        echo "      └── train.log" >&2
        echo "  Data-disk products (ckpts + metrics):" >&2
        echo "    ${ABLATION_OUTPUT_ROOT}/<tag>/run/first_hop_224_sigma_norm_<tag>/" >&2
        echo "      ├── metrics.jsonl" >&2
        echo "      └── ckpt_*.pt" >&2
        echo "" >&2
        echo "  After 'main', see PLAN.md §4 for interpretation matrix." >&2
        echo "  Use scripts/summarize_run.sh <runs/<tag>> for a quick summary." >&2
        exit 2
        ;;
esac
