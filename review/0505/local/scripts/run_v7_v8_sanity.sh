#!/usr/bin/env bash
# review/0505/local/scripts/run_v7_v8_sanity.sh
#
# Drive the V7 (Grönwall closed-form raw step_weights, σ-normalize OFF),
# V8 (no image_aux), and V6_NOISE (V6-seed1337 noise-baseline replica) ablation arms.
#
# IMPORTANT: this script's filename retains "sanity" for historical reasons.
# Per Plan F (review/plan/PLAN_F_V7_V8_FROM_SCRATCH_150K_20260505.md), the
# actual horizon is **160000 steps from-scratch per arm** — NOT a 50K sanity.
# (Endpoint moved 150K → 160K via Fix F: V6 save_interval=20000 means
# V6@step_150000.pt does not exist on disk; V6@step_160000.pt is the closest
# valid paired anchor on the rollout plateau where lambda_roll=4.0 clamped.)
# The default SANITY_STEPS env value below is therefore 160000.
#
# V6_NOISE arm (Q10-A, added 2026-05-05): a V6-seed42 replica with seed=1337,
# everything else bit-equivalent (incl. step_weights, image_aux, schedule).
# Used to compute d_pure = paired_cohen_d(V6-seed42@step_160000, V6-seed1337@step_160000)
# as a clean RNG-only noise floor; PRIMARY decision threshold becomes
# d ≥ max(0.10, 1.8 × d_pure). See ANALYSIS.md §5 Stage 0.
#
# Usage:
#   bash review/0505/local/scripts/run_v7_v8_sanity.sh V7        # V7 only
#   bash review/0505/local/scripts/run_v7_v8_sanity.sh V8        # V8 (a.k.a. "消融 arm") only
#   bash review/0505/local/scripts/run_v7_v8_sanity.sh V6_NOISE  # V6-seed1337 only (Q10-A)
#   bash review/0505/local/scripts/run_v7_v8_sanity.sh both      # V7 then V8 (sequential)
#   bash review/0505/local/scripts/run_v7_v8_sanity.sh parallel  # V7 (GPU_V7) + V8/消融 (GPU_V8) concurrent
#   bash review/0505/local/scripts/run_v7_v8_sanity.sh all       # V7 then V8 then V6_NOISE (sequential)
#   bash review/0505/local/scripts/run_v7_v8_sanity.sh compare   # post-hoc summary table (no training)
#
# Environment overrides (export before running):
#   PYTHON              python interpreter (default: python)
#   TRAIN_ENTRY         training script (default: train_first_hop.py)
#   ABLATION_INDEX_DIR  in-repo index dir (default: review/0505/local/runs)
#                       Stores config.resolved.yaml and train.log per arm.
#   ABLATION_OUTPUT_ROOT  data-disk root (default: /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs)
#                       MUST be under /data_2 — pet_lr/path_guard.py enforces this.
#   GPU                 CUDA device id for sequential modes (default: 0)
#   GPU_V7              CUDA device id for V7 in `parallel` mode (default: 2)
#   GPU_V8              CUDA device id for V8/消融 in `parallel` mode (default: 3)
#   SANITY_STEPS        per-arm step budget (default: 160000 — Plan F endpoint).
#                       MUST match yaml's `max_steps` to prevent silent downgrade.
#
# `parallel` mode notes:
#   - V7 and V8 write to disjoint output_dirs (${ABLATION_OUTPUT_ROOT}/V7/run
#     vs .../V8/run), so concurrent training cannot collide on outputs.
#   - require_fresh_output_dir is enforced PER ARM before launch — if either
#     arm's output already exists and SANITY_FRESH=1 is not set, the script
#     aborts before starting either job.
#   - Logs go to ${ABLATION_INDEX_DIR}/V7/train.log and .../V8/train.log
#     respectively. Use `tail -f` on either file to monitor.
#   - Each job's exit status is reported separately at the end. Non-zero on
#     either side causes the script to exit non-zero.
#
# Output layout per arm (tag = V7 | V8 | V6_NOISE):
#   ${ABLATION_INDEX_DIR}/<tag>/
#     ├── config.resolved.yaml   (max_steps + run_name + output_dir rewritten)
#     └── train.log              (tee'd stdout)
#   ${ABLATION_OUTPUT_ROOT}/<tag>/run/<run_name>/
#     ├── metrics.jsonl
#     ├── ckpts/
#     └── ...                    (everything train_first_hop.py writes)
#
# Comparison vs V6:
#   V7 / V8 are compared against V6-seed42's full-val score at step = 160000
#   (computed from V6-seed42@step_160000.pt, which exists on disk under V6's
#   save_interval=20000 grid). V6_NOISE is compared against V6-seed42 at the
#   same step to compute d_pure (the noise-floor estimate). See ANALYSIS.md §3
#   for the rationale (V6-seed42 is already trained; we re-measure noise-only
#   variance via a seed-twin replica, not via re-training V6 a third time).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${LOCAL_DIR}/../../.." && pwd)"

PYTHON="${PYTHON:-python}"
TRAIN_ENTRY="${TRAIN_ENTRY:-train_first_hop.py}"
ABLATION_INDEX_DIR="${ABLATION_INDEX_DIR:-${LOCAL_DIR}/runs}"
ABLATION_OUTPUT_ROOT="${ABLATION_OUTPUT_ROOT:-/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs}"
GPU="${GPU:-0}"
GPU_V7="${GPU_V7:-2}"   # parallel-mode GPU for V7 (Grönwall raw)
GPU_V8="${GPU_V8:-3}"   # parallel-mode GPU for V8 (消融 arm: no image_aux)
SANITY_STEPS="${SANITY_STEPS:-160000}"  # Plan F endpoint (160K from-scratch, post Fix F); matches yaml max_steps.

# Match run_ablation.sh determinism setup: CUBLAS_WORKSPACE_CONFIG must be set
# for cuBLAS GEMM workspace selection to be deterministic across runs even with
# torch.use_deterministic_algorithms(True).
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

cd "${REPO_ROOT}"

# Sanity: data-disk root convention (mirrors pet_lr/path_guard.py).
case "${ABLATION_OUTPUT_ROOT}" in
    /data_2/*|/data_2)
        : # ok
        ;;
    *)
        echo "ERROR: ABLATION_OUTPUT_ROOT must be under /data_2 (path_guard requirement)." >&2
        echo "  got: ${ABLATION_OUTPUT_ROOT}" >&2
        exit 1
        ;;
esac

# Sanity: SANITY_STEPS is a positive integer.
if ! [[ "${SANITY_STEPS}" =~ ^[0-9]+$ ]] || (( SANITY_STEPS <= 0 )); then
    echo "ERROR: SANITY_STEPS must be a positive integer, got: ${SANITY_STEPS}" >&2
    exit 1
fi

mkdir -p "${ABLATION_INDEX_DIR}"

# resolve_yaml: copy a source yaml, rewrite max_steps + run_name + output_dir.
resolve_yaml() {
    local tag="$1" src="$2" steps="$3"
    local indexdir="${ABLATION_INDEX_DIR}/${tag}"
    local dst="${indexdir}/config.resolved.yaml"
    local out_dir="${ABLATION_OUTPUT_ROOT}/${tag}/run"
    # The yaml-internal run_name keeps its semantic prefix (V7 / V8).
    # We do NOT rename it; the tag-based output_dir is what disambiguates.
    mkdir -p "${indexdir}"
    cp "${src}" "${dst}"

    awk -v new_steps="${steps}" -v new_out="${out_dir}" '
        BEGIN { steps_done=0; out_done=0 }
        /^[[:space:]]*max_steps:/ && new_steps != "" && !steps_done {
            sub(/max_steps:.*/, "max_steps: " new_steps); steps_done=1
        }
        /^output_dir:/ && !out_done {
            sub(/output_dir:.*/, "output_dir: " new_out); out_done=1
        }
        { print }
    ' "${dst}" > "${dst}.tmp" && mv "${dst}.tmp" "${dst}"
    echo "  resolved → ${dst}"
}

# prep_one: resolve yaml + enforce fresh output dir. Idempotent. Safe to call
# sequentially before parallel-launching multiple training jobs.
prep_one() {
    local tag="$1" src="$2" steps="$3"
    echo "===== prep: ${tag} (steps=${steps}) ====="
    resolve_yaml "${tag}" "${src}" "${steps}"

    # Refuse to overwrite a previous run output (mirrors A/B sanity behavior).
    local out_dir="${ABLATION_OUTPUT_ROOT}/${tag}"
    if [[ -d "${out_dir}" && -n "$(ls -A "${out_dir}" 2>/dev/null)" ]]; then
        if [[ "${SANITY_FRESH:-0}" == "1" ]]; then
            echo "[run_v7_v8_sanity.sh] SANITY_FRESH=1: removing ${out_dir}" >&2
            rm -rf "${out_dir}"
        else
            echo "" >&2
            echo "ERROR: previous output exists at ${out_dir}" >&2
            echo "  (V7/V8 yaml has require_fresh_output_dir: true)" >&2
            echo "" >&2
            echo "To rerun, set:" >&2
            echo "  SANITY_FRESH=1 SANITY_STEPS=${steps} bash review/0505/local/scripts/run_v7_v8_sanity.sh ${CMD}" >&2
            echo "" >&2
            exit 4
        fi
    fi
}

# train_one: launch python training. Assumes prep_one has already resolved the yaml.
# Optional 4th arg overrides GPU (used by parallel mode).
train_one() {
    local tag="$1" src="$2" steps="$3" gpu_override="${4:-}"
    local indexdir="${ABLATION_INDEX_DIR}/${tag}"
    local resolved="${indexdir}/config.resolved.yaml"
    local logfile="${indexdir}/train.log"
    local gpu="${gpu_override:-${GPU}}"

    echo "===== train: ${tag} (steps=${steps}, gpu=${gpu}) ====="
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${TRAIN_ENTRY}" \
        --config "${resolved}" 2>&1 | tee "${logfile}"
    echo "===== ${tag}: done ====="
}

# run_one: prep + train (sequential, single GPU = $GPU). Optional 4th gpu arg.
run_one() {
    local tag="$1" src="$2" steps="$3" gpu_override="${4:-}"
    echo ""
    echo "===== run_v7_v8_sanity.sh: ${tag} (steps=${steps}) ====="
    prep_one "${tag}" "${src}" "${steps}"
    train_one "${tag}" "${src}" "${steps}" "${gpu_override}"
}

CMD="${1:-}"

case "${CMD}" in
    V7)
        run_one "V7" "${LOCAL_DIR}/configs/V7_gronwall_raw.yaml" "${SANITY_STEPS}"
        ;;
    V8)
        run_one "V8" "${LOCAL_DIR}/configs/V8_no_image_aux.yaml" "${SANITY_STEPS}"
        ;;
    V6_NOISE)
        run_one "V6_NOISE" "${LOCAL_DIR}/configs/V6_seed1337.yaml" "${SANITY_STEPS}"
        ;;
    both)
        run_one "V7" "${LOCAL_DIR}/configs/V7_gronwall_raw.yaml" "${SANITY_STEPS}"
        run_one "V8" "${LOCAL_DIR}/configs/V8_no_image_aux.yaml" "${SANITY_STEPS}"
        ;;
    parallel)
        # V7 (Grönwall raw) on GPU_V7; V8 (消融 arm: no image_aux) on GPU_V8.
        # Pre-flight both BEFORE launching either, so freshness errors fail-fast.
        echo ""
        echo "===== run_v7_v8_sanity.sh: parallel (V7@GPU${GPU_V7}, V8/消融@GPU${GPU_V8}) ====="
        prep_one "V7" "${LOCAL_DIR}/configs/V7_gronwall_raw.yaml" "${SANITY_STEPS}"
        prep_one "V8" "${LOCAL_DIR}/configs/V8_no_image_aux.yaml" "${SANITY_STEPS}"

        # Launch both python trainings concurrently. Each writes its own train.log.
        # set -e is locally relaxed for the wait stage so we can collect both rcs.
        set +e
        train_one "V7" "${LOCAL_DIR}/configs/V7_gronwall_raw.yaml" "${SANITY_STEPS}" "${GPU_V7}" &
        pid_v7=$!
        train_one "V8" "${LOCAL_DIR}/configs/V8_no_image_aux.yaml" "${SANITY_STEPS}" "${GPU_V8}" &
        pid_v8=$!

        echo "[parallel] launched V7 pid=${pid_v7} on GPU${GPU_V7}, V8 pid=${pid_v8} on GPU${GPU_V8}"
        echo "[parallel] follow logs:"
        echo "  tail -f ${ABLATION_INDEX_DIR}/V7/train.log"
        echo "  tail -f ${ABLATION_INDEX_DIR}/V8/train.log"

        wait "${pid_v7}"; rc_v7=$?
        wait "${pid_v8}"; rc_v8=$?
        set -e

        echo ""
        echo "===== parallel: V7 rc=${rc_v7}, V8/消融 rc=${rc_v8} ====="
        if (( rc_v7 != 0 )) || (( rc_v8 != 0 )); then
            echo "ERROR: at least one parallel arm failed" >&2
            exit 1
        fi
        echo "[parallel] both arms completed successfully"
        ;;
    all)
        run_one "V7" "${LOCAL_DIR}/configs/V7_gronwall_raw.yaml" "${SANITY_STEPS}"
        run_one "V8" "${LOCAL_DIR}/configs/V8_no_image_aux.yaml" "${SANITY_STEPS}"
        run_one "V6_NOISE" "${LOCAL_DIR}/configs/V6_seed1337.yaml" "${SANITY_STEPS}"
        ;;
    compare)
        # Post-hoc comparison: read metrics.jsonl from V6 / V7 / V8 at step ≈ SANITY_STEPS,
        # print chain MSE table. Does NOT train anything.
        # NOTE: V6_NOISE is for d_pure computation (separate evaluator), not part of this compare.
        V6_METRICS="${V6_METRICS:-/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0501_runs/V6/run/first_hop_224_v6_transport_first/metrics.jsonl}"
        V7_METRICS="${ABLATION_OUTPUT_ROOT}/V7/run/first_hop_224_v7_gronwall_raw/metrics.jsonl"
        V8_METRICS="${ABLATION_OUTPUT_ROOT}/V8/run/first_hop_224_v8_no_image_aux/metrics.jsonl"
        "${PYTHON}" "${SCRIPT_DIR}/compare_v6_v7_v8.py" \
            --v6 "${V6_METRICS}" \
            --v7 "${V7_METRICS}" \
            --v8 "${V8_METRICS}" \
            --step "${SANITY_STEPS}"
        ;;
    "")
        echo "Usage: $0 {V7|V8|V6_NOISE|both|parallel|all|compare}" >&2
        exit 2
        ;;
    *)
        echo "Unknown command: ${CMD}" >&2
        echo "Usage: $0 {V7|V8|V6_NOISE|both|parallel|all|compare}" >&2
        exit 2
        ;;
esac
