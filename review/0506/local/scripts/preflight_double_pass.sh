#!/usr/bin/env bash
# review/0506/local/scripts/preflight_double_pass.sh
#
# Run Round 4 external review pre-launch Action #4 (V6@seed42 same-seed
# double-pass at 5K steps) or Action #5 (V7@seed42 5K spot-check).
#
# Each invocation runs ONE pass (1 or 2). Pass-1 and pass-2 must run on
# the SAME GPU (cuBLAS heuristic + cuDNN nondeterminism are GPU-chip-
# specific). They are sequential — do NOT launch both passes in parallel
# on one GPU.
#
# After both passes finish, run the eval script on each step_5000.pt and
# compare val_pair_total / val_chain_normal_mse drift between passes.
#
# Usage:
#   GPU=2 bash review/0506/local/scripts/preflight_double_pass.sh V6 1
#   GPU=2 bash review/0506/local/scripts/preflight_double_pass.sh V6 2
#   GPU=3 bash review/0506/local/scripts/preflight_double_pass.sh V7 1
#   GPU=3 bash review/0506/local/scripts/preflight_double_pass.sh V7 2
#
# Environment overrides:
#   GPU                       CUDA device id (default: 2)
#   PYTHON                    python binary (default: python)
#   TRAIN_ENTRY               training script (default: train_first_hop.py)
#   PREFLIGHT_OUTPUT_ROOT     data-disk output root (default:
#                             /data_2/qujiaxiang/outputs/PET_LatentResidual/preflight_0506)

set -euo pipefail

ARM="${1:?arm required: V6 or V7}"
PASS="${2:?pass required: 1 or 2}"

if [[ "${PASS}" != "1" && "${PASS}" != "2" ]]; then
    echo "ERROR: pass must be 1 or 2, got: ${PASS}" >&2
    exit 1
fi

case "${ARM}" in
    V6)
        SRC="configs/pet_flow/pet_flow_first_hop_224_v6_transport_first.yaml"
        PREFIX="first_hop_224_v6_base_5k"
        ;;
    V7)
        SRC="review/0505/local/configs/V7_gronwall_raw.yaml"
        PREFIX="first_hop_224_v7_gronwall_5k"
        ;;
    *)
        echo "ERROR: arm must be V6 or V7, got: ${ARM}" >&2
        exit 1
        ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -f "${SRC}" ]]; then
    echo "ERROR: source yaml not found: ${SRC}" >&2
    exit 1
fi

PYTHON="${PYTHON:-python}"
TRAIN_ENTRY="${TRAIN_ENTRY:-train_first_hop.py}"
PREFLIGHT_OUTPUT_ROOT="${PREFLIGHT_OUTPUT_ROOT:-/data_2/qujiaxiang/outputs/PET_LatentResidual/preflight_0506}"
GPU="${GPU:-2}"

# Sanity: data-disk root convention (mirrors pet_lr/path_guard.py).
case "${PREFLIGHT_OUTPUT_ROOT}" in
    /data_2/*|/data_2)
        : # ok
        ;;
    *)
        echo "ERROR: PREFLIGHT_OUTPUT_ROOT must be under /data_2 (path_guard requirement)." >&2
        echo "  got: ${PREFLIGHT_OUTPUT_ROOT}" >&2
        exit 1
        ;;
esac

OUT_DIR="${PREFLIGHT_OUTPUT_ROOT}/${ARM}_BASE_5K_pass${PASS}"
RESOLVED_DIR="review/0506/local/preflight"
RESOLVED="${RESOLVED_DIR}/${ARM}_BASE_5K_pass${PASS}.resolved.yaml"
LOG_DIR="review/0506/local/logs"
LOGFILE="${LOG_DIR}/${ARM}_5K_pass${PASS}_gpu${GPU}.log"

mkdir -p "${RESOLVED_DIR}" "${LOG_DIR}"

# Resolve yaml: copy, then patch max_steps / save_interval / run_name / output_dir.
# Mirrors the awk pattern from review/0505/local/scripts/run_v7_v8_sanity.sh::resolve_yaml.
cp "${SRC}" "${RESOLVED}"
awk -v new_steps=5000 -v new_save=5000 -v new_run="${PREFIX}_pass${PASS}" -v new_out="${OUT_DIR}" '
    BEGIN { steps_done=0; save_done=0; run_done=0; out_done=0 }
    /^[[:space:]]*max_steps:/ && !steps_done {
        sub(/max_steps:.*/, "max_steps: " new_steps); steps_done=1
    }
    /^[[:space:]]*save_interval:/ && !save_done {
        sub(/save_interval:.*/, "save_interval: " new_save); save_done=1
    }
    /^run_name:/ && !run_done {
        sub(/run_name:.*/, "run_name: " new_run); run_done=1
    }
    /^output_dir:/ && !out_done {
        sub(/output_dir:.*/, "output_dir: " new_out); out_done=1
    }
    { print }
' "${RESOLVED}" > "${RESOLVED}.tmp" && mv "${RESOLVED}.tmp" "${RESOLVED}"

# Refuse if output exists already (mirrors require_fresh_output_dir).
if [[ -d "${OUT_DIR}" && -n "$(ls -A "${OUT_DIR}" 2>/dev/null)" ]]; then
    echo "ERROR: output exists at ${OUT_DIR}" >&2
    echo "  set PREFLIGHT_FRESH=1 to remove and retry." >&2
    if [[ "${PREFLIGHT_FRESH:-0}" == "1" ]]; then
        echo "[preflight_double_pass] PREFLIGHT_FRESH=1: removing ${OUT_DIR}" >&2
        rm -rf "${OUT_DIR}"
    else
        exit 4
    fi
fi

# cuBLAS workspace must be set for deterministic GEMM.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONPATH="${PYTHONPATH:-${REPO_ROOT}}"

echo "===== preflight: ${ARM} pass${PASS} (5K, GPU${GPU}) ====="
echo "  src yaml      = ${SRC}"
echo "  resolved yaml = ${RESOLVED}"
echo "  output_dir    = ${OUT_DIR}"
echo "  log file      = ${LOGFILE}"
echo "  CUDA_VISIBLE_DEVICES=${GPU}  CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG}"
echo ""

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${TRAIN_ENTRY}" \
    --config "${RESOLVED}" 2>&1 | tee "${LOGFILE}"

echo ""
echo "===== ${ARM} pass${PASS}: done ====="
echo "  step_5000.pt expected at: ${OUT_DIR}/${PREFIX}_pass${PASS}/step_5000.pt"
echo "  next: run eval_first_hop_224_clip3.py against this checkpoint."
