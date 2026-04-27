#!/bin/bash
# ================================================================
# Experiment 5: Null-Control Full-Val Evaluation
# ================================================================
# 在 03_null_control.sh 完成后，对 null-control 的 best/last/latest-step
# 做同口径 full-val，避免再次用 rolling-val 做最终结论。
#
# 用法: bash review/0427/run_me/05_fullval_null_control.sh [GPU_ID]
# ================================================================
set -euo pipefail

GPU_ID="${1:-0}"
PYTHON="/home/qujiaxiang/.conda/envs/rae/bin/python"

NULL_DIR="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v5_null_control"
NULL_CONFIG="configs/pet_flow/pet_flow_first_hop_224_v5_null_control.yaml"
OUT_BASE="/data_2/qujiaxiang/outputs/PET_LatentResidual/eval_0427_fullval_null_control"

eval_one() {
    local label="$1"
    local ckpt="$2"
    local out_dir="$3"

    if [ ! -f "${ckpt}" ]; then
        echo "[SKIP] ${label}: checkpoint 不存在: ${ckpt}"
        return 0
    fi

    echo "[RUN] ${label}"
    echo "      ckpt: ${ckpt}"
    echo "      out : ${out_dir}"
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} -u eval_first_hop_224_clip3.py \
        --config "${NULL_CONFIG}" \
        --checkpoint "${ckpt}" \
        --split val --max-slices 0 --batch-size 8 --device cuda:0 \
        --decode-mode both \
        --out-dir "${out_dir}"
    echo "[DONE] ${label}"
}

latest_step_ckpt() {
    local dir="$1"
    find "${dir}" -maxdepth 1 -type f -name 'step_*.pt' -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | awk 'NR==1 {print $2}'
}

echo "=== Experiment 5: Null-Control Full-Val ==="
echo "dir: ${NULL_DIR}"

eval_one "null-control best full-val" "${NULL_DIR}/best.pt" "${OUT_BASE}/null_best"
eval_one "null-control last full-val" "${NULL_DIR}/last.pt" "${OUT_BASE}/null_last"

NULL_LATEST="$(latest_step_ckpt "${NULL_DIR}")"
if [ -n "${NULL_LATEST}" ]; then
    STEP_NAME=$(basename "${NULL_LATEST}" .pt)
    eval_one "null-control ${STEP_NAME} full-val" "${NULL_LATEST}" "${OUT_BASE}/null_${STEP_NAME}"
else
    echo "[SKIP] null-control latest-step: ${NULL_DIR}/step_*.pt 不存在"
fi

echo ""
echo "=== Null-Control Full-Val 结果目录 ==="
echo "  ${OUT_BASE}/"
ls -la "${OUT_BASE}/" 2>/dev/null || echo "  (目录为空)"
