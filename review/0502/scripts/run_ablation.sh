#!/usr/bin/env bash
# review/0502/scripts/run_ablation.sh
#
# Drive the sigma-normalize ablation. Runs train_first_hop.py with a chosen yaml
# and routes outputs/logs into the data disk (`/data_2` by default; project's
# pet_lr/path_guard.py enforces this), with an in-repo index dir under
# review/0502/runs/<tag>/ for the resolved yaml + tee'd log.
#
# Usage:
#   bash review/0502/scripts/run_ablation.sh sanity      # A_sanity + B (50K each, writes gate sentinel on PASS)
#   bash review/0502/scripts/run_ablation.sh A           # only A_main (can run before sanity PASS)
#   bash review/0502/scripts/run_ablation.sh C           # only C (requires sanity-pass sentinel)
#   bash review/0502/scripts/run_ablation.sh main        # A_main + C (requires sanity-pass sentinel)
#   bash review/0502/scripts/run_ablation.sh closed_form # D (requires sanity-pass sentinel)
#   bash review/0502/scripts/run_ablation.sh B           # only B
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

# v3.1 (per agent1 review): match run_sanity_light.sh determinism setup so the
# Tier 1 bit-equal target is actually achievable. Without this var, CUDA cuBLAS
# GEMM workspace selection is non-deterministic across runs even with
# torch.use_deterministic_algorithms(True), defeating preserve_v6_sum A==B.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

# Sentinel file written after a successful sanity gate. `main` / `C` / `D` will
# refuse to launch unless this exists or ALLOW_UNGATED=1 is set.
SANITY_PASS_SENTINEL="${SANITY_PASS_SENTINEL:-${ABLATION_INDEX_DIR}/.sanity_pass}"

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
#   Tier 1 (strongest): val_pair_total — must be bit-equal (rel < 1e-4)
#   Tier 2 (strong)   : val_rollout_total — must agree to < 1% relative
#   Tier 3 (strong)   : val_rollout_step_*_raw (per-hop raw losses) < 1% relative
#   Tier 4 (weak)     : val_chain_*_mse (4 chain MSEs) < 5% relative (training noise)
#
# v3 fix (per agent3/agent4 review): this function now actually executes the
# comparator and exits non-zero on FAIL — earlier print_sanity_check_cmd only
# printed a heredoc to stdout, so the sanity gate had no effect on exit code.
run_sanity_comparator() {
    local a_metrics="$1" b_metrics="$2"
    echo ""
    echo "===== Sanity check (multi-tier, pure-python — no jq required) ====="
    "${PYTHON}" - "${a_metrics}" "${b_metrics}" <<'PYEOF'
import json, sys

a_path, b_path = sys.argv[1], sys.argv[2]

def last_eval(path):
    with open(path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    evals = [r for r in rows if any(k in r for k in (
        "val_rollout_total", "val_chain_d20_mse", "val_pair_total",
    ))]
    return evals[-1] if evals else None

A = last_eval(a_path)
B = last_eval(b_path)
if A is None or B is None:
    print("ERROR: no eval rows found in one of the metrics files")
    sys.exit(1)

# Tier definitions: (key, max_rel_err, label, required)
# v3.1 fix (agent2 review): missing keys now FAIL the gate — previously
# `if a is None or b is None: continue` left overall_pass unchanged, which
# meant a comparator run on light-sanity metrics (chain keys absent) silently
# returned PASS. We now mark required keys explicitly and miss → FAIL.
TIER1 = [("val_pair_total", 1e-4, "Tier 1 (bit-equal)", True)]
TIER2 = [("val_rollout_total", 0.01, "Tier 2 (<1%)", True)]
TIER3 = sorted(
    [(k, 0.01, "Tier 3 (<1%)", True) for k in A
     if k.startswith("val_rollout_step_") and k.endswith("_raw") and k in B]
)
TIER4 = [
    ("val_chain_d20_mse", 0.05, "Tier 4 (<5%)", True),
    ("val_chain_d10_mse", 0.05, "Tier 4 (<5%)", True),
    ("val_chain_d4_mse",  0.05, "Tier 4 (<5%)", True),
    ("val_chain_normal_mse", 0.05, "Tier 4 (<5%)", True),
]

ALL = TIER1 + TIER2 + TIER3 + TIER4

# Sanity: Tier 3 must have at least 4 raw step keys for a 4-hop rollout
raw_step_keys = [k for k, *_ in TIER3]
if len(raw_step_keys) < 4:
    print(f"ERROR: expected 4 val_rollout_step_*_raw keys, found {len(raw_step_keys)}: {raw_step_keys}")
    print("       This usually means the run used val_include_full_x_rollout=false (light sanity).")
    print("       run_ablation.sh sanity requires the full eval path; check yaml.")
    sys.exit(1)

print(f"{'metric':<32} {'A':>14} {'B':>14} {'rel_err':>10}  {'tier_threshold'}")
print("-" * 96)
overall_pass = True
missing_required = []
for k, thr, label, required in ALL:
    a = A.get(k); b = B.get(k)
    if a is None or b is None:
        marker = "<absent>"
        if required:
            overall_pass = False
            missing_required.append(k)
            print(f"{k:<32} {marker:>14} {marker:>14} {'-':>10}  {label} -> FAIL (missing)")
        else:
            print(f"{k:<32} {marker:>14} {marker:>14} {'-':>10}  {label} (skipped, optional)")
        continue
    rel = abs(b - a) / max(abs(a), 1e-12)
    flag = "PASS" if rel < thr else "FAIL"
    if rel >= thr:
        overall_pass = False
    print(f"{k:<32} {a:>14.6e} {b:>14.6e} {rel:>9.2%}  {label} -> {flag}")

print()
if missing_required:
    print(f"Missing required keys (gate FAIL): {missing_required}")
print("Result:", "PASS (all tiers under threshold)" if overall_pass else "FAIL (one or more tiers exceed threshold or missing)")
print()
print("FP32 expectation reminder:")
print("  - val_pair_total tier 1 may show ~1e-6..1e-7 rel-err from bf16/AMP stochasticity")
print("  - val_rollout_total tier 2 typically ~1e-6..1e-5 rel-err from order-of-ops in normalizer division")
print("  - chain MSE tier 4 ~1e-3..1e-2 from EMA + ckpt sampling jitter (training noise, not bug)")
sys.exit(0 if overall_pass else 1)
PYEOF
}

CMD="${1:-}"

# v3.1 (agent2 review) + v3.1 follow-up (agent6): require_sanity_pass enforces
# that main/C/D launches only happen after a successful `run_ablation.sh sanity`
# Tier 1-4 PASS. Beyond simple file-existence, we now also verify that:
#   - sanity_steps >= 20000 (Phase I chain MSE noise tolerance)
#   - the recorded a_metrics/b_metrics paths still exist (sanity output
#     wasn't manually deleted between sanity and main/C/D)
# This catches the "old sentinel from a 10K smoke / since-deleted run" failure
# mode without requiring full code-hash binding (which is acknowledged optional
# polish for paper-time).
# Override the entire gate with ALLOW_UNGATED=1 (only for known-good debug rebuilds).
#
# Minimum sanity_steps a sentinel must record to be considered "main/C/D-grade";
# tunable for retroactive smoke-only debug rebuilds via SANITY_PASS_MIN_STEPS=...
SANITY_PASS_MIN_STEPS="${SANITY_PASS_MIN_STEPS:-20000}"

require_sanity_pass() {
    local what="$1"
    if [[ "${ALLOW_UNGATED:-0}" == "1" ]]; then
        echo "[run_ablation.sh] ALLOW_UNGATED=1 — launching ${what} without sanity-pass sentinel." >&2
        return 0
    fi
    if [[ ! -f "${SANITY_PASS_SENTINEL}" ]]; then
        echo "" >&2
        echo "ERROR: ${what} requires a sanity-pass sentinel at:" >&2
        echo "  ${SANITY_PASS_SENTINEL}" >&2
        echo "" >&2
        echo "Run a Tier 1-4 sanity gate first:" >&2
        echo "  bash review/0502/scripts/run_ablation.sh sanity        # default 50K×2" >&2
        echo "  SANITY_STEPS=20000 bash review/0502/scripts/run_ablation.sh sanity   # mini full sanity" >&2
        echo "" >&2
        echo "Or, only when you have an out-of-band reason (debug rebuild after a known-good run):" >&2
        echo "  ALLOW_UNGATED=1 bash review/0502/scripts/run_ablation.sh ${CMD}" >&2
        echo "" >&2
        exit 3
    fi
    # v3.1 follow-up (agent6): freshness checks. Read sentinel keys and assert.
    local sentinel_steps sentinel_a sentinel_b
    sentinel_steps="$(awk -F= '/^sanity_steps=/ { print $2; exit }' "${SANITY_PASS_SENTINEL}" 2>/dev/null)"
    sentinel_a="$(awk -F= '/^a_metrics=/ { sub(/^a_metrics=/, ""); print; exit }' "${SANITY_PASS_SENTINEL}" 2>/dev/null)"
    sentinel_b="$(awk -F= '/^b_metrics=/ { sub(/^b_metrics=/, ""); print; exit }' "${SANITY_PASS_SENTINEL}" 2>/dev/null)"
    if [[ -z "${sentinel_steps}" || ! "${sentinel_steps}" =~ ^[0-9]+$ ]]; then
        echo "" >&2
        echo "ERROR: sentinel ${SANITY_PASS_SENTINEL} is malformed (no sanity_steps=N line)." >&2
        echo "  Re-run: bash review/0502/scripts/run_ablation.sh sanity" >&2
        echo "" >&2
        exit 3
    fi
    if (( sentinel_steps < SANITY_PASS_MIN_STEPS )); then
        echo "" >&2
        echo "ERROR: sentinel sanity_steps=${sentinel_steps} < SANITY_PASS_MIN_STEPS=${SANITY_PASS_MIN_STEPS}." >&2
        echo "  This sentinel was written by a smoke run, not a Tier-4 gate." >&2
        echo "  Re-run with SANITY_STEPS=${SANITY_PASS_MIN_STEPS} (or larger):" >&2
        echo "    SANITY_FRESH=1 SANITY_STEPS=${SANITY_PASS_MIN_STEPS} bash review/0502/scripts/run_ablation.sh sanity" >&2
        echo "  Or, only for debug rebuilds, set SANITY_PASS_MIN_STEPS=${sentinel_steps} or ALLOW_UNGATED=1." >&2
        echo "" >&2
        exit 3
    fi
    if [[ -n "${sentinel_a}" && ! -f "${sentinel_a}" ]]; then
        echo "" >&2
        echo "ERROR: sentinel a_metrics path no longer exists: ${sentinel_a}" >&2
        echo "  The sanity output was deleted; the gate cannot be trusted." >&2
        echo "  Re-run: SANITY_FRESH=1 bash review/0502/scripts/run_ablation.sh sanity" >&2
        echo "" >&2
        exit 3
    fi
    if [[ -n "${sentinel_b}" && ! -f "${sentinel_b}" ]]; then
        echo "" >&2
        echo "ERROR: sentinel b_metrics path no longer exists: ${sentinel_b}" >&2
        echo "  The sanity output was deleted; the gate cannot be trusted." >&2
        echo "  Re-run: SANITY_FRESH=1 bash review/0502/scripts/run_ablation.sh sanity" >&2
        echo "" >&2
        exit 3
    fi
    echo "[run_ablation.sh] sanity-pass sentinel found (${SANITY_PASS_SENTINEL}):" >&2
    sed 's/^/  /' "${SANITY_PASS_SENTINEL}" >&2
    echo "[run_ablation.sh]   freshness OK: sanity_steps=${sentinel_steps} ≥ ${SANITY_PASS_MIN_STEPS}, metrics paths exist." >&2
}

case "${CMD}" in
    sanity)
        # SANITY_STEPS (preferred) or MAX_STEPS (compat with run_sanity_light.sh)
        # lets you override the per-condition step budget.
        # Default 50000 matches PLAN §3.3 stage 1; set 20000 for a mini full-eval gate
        # (10000 was tried but Phase I chain MSE CV ~60% causes Tier 4 false-fail >30%).
        SANITY_STEPS="${SANITY_STEPS:-${MAX_STEPS:-50000}}"
        if [[ ! "${SANITY_STEPS}" =~ ^[0-9]+$ ]]; then
            echo "ERROR: SANITY_STEPS must be a positive integer, got: ${SANITY_STEPS}" >&2
            exit 5
        fi
        if (( SANITY_STEPS <= 0 )); then
            echo "ERROR: SANITY_STEPS must be a positive integer, got: ${SANITY_STEPS}" >&2
            exit 5
        fi
        # v3.1 (agent3 review): A_control.yaml + B_sanity.yaml have
        # require_fresh_output_dir: true. The first sanity invocation populates
        # ${ABLATION_OUTPUT_ROOT}/{A_sanity,B}/run/...; any subsequent rerun
        # (FAIL retry, different SANITY_STEPS, different code path)
        # will be blocked by train_first_hop.py L1303. Sanity is *by design*
        # ephemeral (smoke test for A==B equivalence) — allow an explicit reset
        # via SANITY_FRESH=1 instead of forcing the operator to rm -rf manually.
        for _stale_tag in A_sanity B; do
            _stale_dir="${ABLATION_OUTPUT_ROOT}/${_stale_tag}"
            if [[ -d "${_stale_dir}" && -n "$(ls -A "${_stale_dir}" 2>/dev/null)" ]]; then
                if [[ "${SANITY_FRESH:-0}" == "1" ]]; then
                    echo "[run_ablation.sh] SANITY_FRESH=1: removing previous sanity output ${_stale_dir}" >&2
                    rm -rf "${_stale_dir}"
                else
                    # v3.1 follow-up (agent6): if a stale sentinel still points
                    # to this output dir we are about to refuse re-creating,
                    # invalidate it so subsequent main/C/D launches don't trust
                    # an unrepeatable sanity. Even if SANITY_FRESH wasn't set,
                    # the operator's intent ("I tried to rerun sanity") means
                    # the previous gate should no longer authorize anything.
                    if [[ -f "${SANITY_PASS_SENTINEL}" ]]; then
                        echo "[run_ablation.sh] invalidating stale sentinel ${SANITY_PASS_SENTINEL}" >&2
                        rm -f "${SANITY_PASS_SENTINEL}"
                    fi
                    echo "" >&2
                    echo "ERROR: previous sanity output exists at ${_stale_dir}" >&2
                    echo "  (A_control.yaml/B_sanity.yaml set require_fresh_output_dir: true," >&2
                    echo "   so train_first_hop.py will refuse to start over a non-empty dir)." >&2
                    echo "" >&2
                    echo "To rerun sanity (different SANITY_STEPS, retry after FAIL, code change), set:" >&2
                    echo "  SANITY_FRESH=1 SANITY_STEPS=${SANITY_STEPS} bash review/0502/scripts/run_ablation.sh sanity" >&2
                    echo "" >&2
                    exit 4
                fi
            fi
        done
        unset _stale_tag _stale_dir
        run_one "A_sanity" "${REVIEW_DIR}/configs/A_control.yaml" "${SANITY_STEPS}"
        run_one "B"        "${REVIEW_DIR}/configs/B_sanity.yaml" "${SANITY_STEPS}"
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
        # v3: actually execute the comparator (returns non-zero on FAIL)
        if run_sanity_comparator "${a_metrics}" "${b_metrics}"; then
            if [[ "${SANITY_STEPS}" -lt 20000 ]]; then
                rm -f "${SANITY_PASS_SENTINEL}"
                echo ""
                echo "[run_ablation.sh] Smoke sanity PASS at SANITY_STEPS=${SANITY_STEPS}, but no sentinel was written." >&2
                echo "[run_ablation.sh] Run SANITY_STEPS=20000 (or default 50000) to unlock main/C/D." >&2
                exit 0
            fi
            # v3.1 (agent2 review): write sentinel so main/C/D launches refuse
            # to run before a fresh sanity pass. Encode the metrics paths so
            # we can later prove which sanity run gated us.
            mkdir -p "$(dirname "${SANITY_PASS_SENTINEL}")"
            {
                echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
                echo "sanity_steps=${SANITY_STEPS}"
                echo "a_metrics=${a_metrics}"
                echo "b_metrics=${b_metrics}"
            } > "${SANITY_PASS_SENTINEL}"
            echo ""
            echo "[run_ablation.sh] Wrote sanity-pass sentinel: ${SANITY_PASS_SENTINEL}"
        else
            rm -f "${SANITY_PASS_SENTINEL}"
            echo ""
            echo "[run_ablation.sh] Sanity FAILED — sentinel cleared. main/C/D will refuse to launch." >&2
            exit 1
        fi
        ;;
    main)
        # The combined shortcut is gated before launching anything. Operators
        # who want to overlap A_main with sanity should run the explicit `A`
        # command first, then run `C` after the sentinel exists.
        require_sanity_pass "main"
        run_one "A_main" "${REVIEW_DIR}/configs/A_control.yaml" "120000"
        run_one "C"      "${REVIEW_DIR}/configs/C_uniform.yaml" "120000"
        ;;
    closed_form)
        require_sanity_pass "closed_form (D)"
        run_one "D" "${REVIEW_DIR}/configs/D_closed_form.yaml" "120000"
        ;;
    A)
        # A_main does not depend on the sigma_dt code path being correct
        # (it uses the V6 reference step_weights), so it can run in parallel
        # with sanity. No sentinel required.
        run_one "A_main" "${REVIEW_DIR}/configs/A_control.yaml" "120000"
        ;;
    B)
        run_one "B" "${REVIEW_DIR}/configs/B_sanity.yaml" "50000"
        ;;
    C)
        require_sanity_pass "C (single)"
        run_one "C" "${REVIEW_DIR}/configs/C_uniform.yaml" "120000"
        ;;
    D)
        require_sanity_pass "D (single)"
        run_one "D" "${REVIEW_DIR}/configs/D_closed_form.yaml" "120000"
        ;;
    pair_uniform)
        # Risk 4 spot check: A baseline with pair_loss_weights=[1,1,1,1] (vs A_main [2.5,1,1,1])
        # 60K seed=42. Run AFTER A_main + C_uniform main ablation completes.
        # Used to disambiguate whether pair_weight[0]=2.5 confounds the rollout-shape ablation.
        require_sanity_pass "pair_uniform (Risk 4 spot)"
        run_one "A_pair_uniform_spot" "${REVIEW_DIR}/configs/A_pair_uniform_spot.yaml" "60000"
        ;;
    *)
        echo "Usage: $0 {sanity|main|closed_form|A|B|C|D|pair_uniform}" >&2
        echo "  sanity      : run A_sanity + B (default 50K each, set SANITY_STEPS=N to override)" >&2
        echo "                — multi-tier A==B equivalence gate; exits non-zero on FAIL" >&2
        echo "                — writes ${SANITY_PASS_SENTINEL} on PASS" >&2
        echo "  main        : run A_main + C (120K each) — requires sanity-pass sentinel before launching" >&2
        echo "  closed_form : run D (120K) — requires sanity-pass sentinel" >&2
        echo "  A           : run A_main (120K, single) — no gate (V6 reference path)" >&2
        echo "  B           : run B (50K, single) — no gate (sanity-only condition)" >&2
        echo "  C           : run C (120K, single) — requires sanity-pass sentinel" >&2
        echo "  D           : run D (120K, single) — requires sanity-pass sentinel" >&2
        echo "  pair_uniform: run A_pair_uniform_spot (60K, Risk 4 spot check) — requires sentinel" >&2
        echo "" >&2
        echo "Environment overrides:" >&2
        echo "  GPU=N             pin CUDA_VISIBLE_DEVICES (default: 0)" >&2
        echo "  SANITY_STEPS=N    override per-condition step budget for the sanity command" >&2
        echo "                    (default 50000; recommended 20000 for mini full sanity;" >&2
        echo "                    10000 is debug-only smoke — chain Tier 4 false-fail >30%)" >&2
        echo "  MAX_STEPS=N       legacy alias of SANITY_STEPS (kept for run_sanity_light.sh parity)" >&2
        echo "  SANITY_FRESH=1    rm -rf previous sanity output dirs before rerun" >&2
        echo "                    (only touches ${ABLATION_OUTPUT_ROOT}/{A_sanity,B}; main/C/D unaffected)" >&2
        echo "  SANITY_PASS_MIN_STEPS=N" >&2
        echo "                    minimum sanity_steps a sentinel must record to gate main/C/D" >&2
        echo "                    (default 20000; lower only for debug rebuilds)" >&2
        echo "  ALLOW_UNGATED=1   bypass sanity-pass sentinel for main/C/D (debug rebuilds only)" >&2
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
        echo "  Use scripts/summarize_run.sh <runs/<tag>> for a quick per-metric scan" >&2
        echo "  (NOT a paper-table; for the paper use POST_V6_NEXT_STEPS.md §1.2 best.pt-row snippet)." >&2
        exit 2
        ;;
esac
