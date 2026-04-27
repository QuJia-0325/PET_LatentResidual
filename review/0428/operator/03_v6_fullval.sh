#!/bin/bash
# ================================================================
# V6 Full-Val 评估（在 pilot 通过后或 200K 完成后执行）
# ================================================================
# 对 V6 best.pt 做全量 val 评估，与 V3 baseline 对比
#
# 用法: bash review/0428/operator/03_v6_fullval.sh [GPU_ID]
# 预计耗时: ~40min
# ================================================================
set -euo pipefail

GPU_ID="${1:-0}"
PYTHON="/home/qujiaxiang/.conda/envs/rae/bin/python"

V3_DIR="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_200k_transport_v3"
V6_DIR="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v6_transport_first"
V3_CONFIG="configs/pet_flow/pet_flow_first_hop_224_200k_transport_v3.yaml"
V6_CONFIG="configs/pet_flow/pet_flow_first_hop_224_v6_transport_first.yaml"

OUT_BASE="/data_2/qujiaxiang/outputs/PET_LatentResidual/eval_0428_v6_fullval"
mkdir -p "${OUT_BASE}"

echo "=== V6 Full-Val Evaluation ==="

# V3 baseline（如果 0427 已经跑过，跳过）
V3_EVAL_EXISTING="/data_2/qujiaxiang/outputs/PET_LatentResidual/eval_0427_fullval/v3_best"
if [ -d "${V3_EVAL_EXISTING}" ] && [ -f "${V3_EVAL_EXISTING}/first_hop_224_val_clip3_eval.json" ]; then
    echo "[V3 baseline] 复用 0427 fullval 结果: ${V3_EVAL_EXISTING}"
else
    echo "[V3 baseline] 运行全量 eval..."
    CUDA_VISIBLE_DEVICES="${GPU_ID}" ${PYTHON} eval_first_hop_224_clip3.py \
        --config "${V3_CONFIG}" \
        --checkpoint "${V3_DIR}/best.pt" \
        --output-dir "${OUT_BASE}/v3_best" \
        --max-slices 0
fi

# V6 best
if [ -f "${V6_DIR}/best.pt" ]; then
    echo "[V6 best] 运行全量 eval..."
    CUDA_VISIBLE_DEVICES="${GPU_ID}" ${PYTHON} eval_first_hop_224_clip3.py \
        --config "${V6_CONFIG}" \
        --checkpoint "${V6_DIR}/best.pt" \
        --output-dir "${OUT_BASE}/v6_best" \
        --max-slices 0
else
    echo "[V6 best] best.pt 不存在，跳过"
fi

# V6 last
if [ -f "${V6_DIR}/last.pt" ]; then
    echo "[V6 last] 运行全量 eval..."
    CUDA_VISIBLE_DEVICES="${GPU_ID}" ${PYTHON} eval_first_hop_224_clip3.py \
        --config "${V6_CONFIG}" \
        --checkpoint "${V6_DIR}/last.pt" \
        --output-dir "${OUT_BASE}/v6_last" \
        --max-slices 0
else
    echo "[V6 last] last.pt 不存在，跳过"
fi

echo ""
echo "=== 完成。结果在 ${OUT_BASE}/ ==="
echo "对比: diff <(jq .summary_psnr_clip3 ${OUT_BASE}/v3_best/*.json) <(jq .summary_psnr_clip3 ${OUT_BASE}/v6_best/*.json)"
