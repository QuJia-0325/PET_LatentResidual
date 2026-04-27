#!/bin/bash
# ================================================================
# Experiment 1: Full-Val Evaluation (消除 rolling-val window 混淆)
# ================================================================
# 对 V3 baseline 与 V5 best/last/latest-step 做 full-val eval
# 使用相同的 eval 脚本；每个 checkpoint 使用各自训练 config 构建模型/数据
# --max-slices 0 = 全量 val set
#
# 用法: bash review/0427/run_me/01_fullval_eval.sh [GPU_ID]
# 预计耗时: ~3×20min = ~1h
# ================================================================
set -euo pipefail

GPU_ID="${1:-0}"
PYTHON="/home/qujiaxiang/.conda/envs/rae/bin/python"

V3_DIR="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_200k_transport_v3"
V3_BEST="${V3_DIR}/best.pt"
V3_LAST="${V3_DIR}/last.pt"
V5_DIR="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v5_rollout_heavy"
# V5 config（用于构建 dataset/model）
V5_CONFIG="configs/pet_flow/pet_flow_first_hop_224_v5_rollout_heavy.yaml"
# V3 config（用于 V3 baseline eval）
V3_CONFIG="configs/pet_flow/pet_flow_first_hop_224_200k_transport_v3.yaml"

OUT_BASE="/data_2/qujiaxiang/outputs/PET_LatentResidual/eval_0427_fullval"

echo "=== Experiment 1: Full-Val Evaluation ==="
echo "消除 rolling-val window 噪声，统一评估协议"
echo ""

eval_one() {
    local label="$1"
    local config="$2"
    local ckpt="$3"
    local out_dir="$4"

    if [ ! -f "${ckpt}" ]; then
        echo "[SKIP] ${label}: checkpoint 不存在: ${ckpt}"
        return 0
    fi

    echo "[RUN] ${label}"
    echo "      ckpt: ${ckpt}"
    echo "      out : ${out_dir}"
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} -u eval_first_hop_224_clip3.py \
        --config "${config}" \
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

# --- V3 baseline ---
eval_one "V3 best full-val" "${V3_CONFIG}" "${V3_BEST}" "${OUT_BASE}/v3_best"
eval_one "V3 last full-val" "${V3_CONFIG}" "${V3_LAST}" "${OUT_BASE}/v3_last"

# --- V5 best/last/latest-step ---
eval_one "V5 best full-val" "${V5_CONFIG}" "${V5_DIR}/best.pt" "${OUT_BASE}/v5_best"
eval_one "V5 last full-val" "${V5_CONFIG}" "${V5_DIR}/last.pt" "${OUT_BASE}/v5_last"

V5_LATEST="$(latest_step_ckpt "${V5_DIR}")"
if [ -n "${V5_LATEST}" ]; then
    STEP_NAME=$(basename ${V5_LATEST} .pt)
    eval_one "V5 ${STEP_NAME} full-val" "${V5_CONFIG}" "${V5_LATEST}" "${OUT_BASE}/v5_${STEP_NAME}"
else
    echo "[SKIP] V5 latest-step: ${V5_DIR}/step_*.pt 不存在"
fi

echo ""
echo "=== Full-Val 结果目录 ==="
echo "  ${OUT_BASE}/"
ls -la ${OUT_BASE}/ 2>/dev/null || echo "  (目录为空)"
echo ""
echo "=== 对比方法 ==="
echo "  JSON: <out-dir>/first_hop_224_val_clip3_eval.json"
echo "  关键字段: summary_psnr_clip3.NORMAL.mean, summary_psnr_clip3.D20.mean"
echo "  decode-mode=both 时同时检查 NORMAL_raw / D20_raw，避免 seam refiner 混淆"
