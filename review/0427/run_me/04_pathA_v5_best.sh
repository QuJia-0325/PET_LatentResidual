#!/bin/bash
# ================================================================
# Experiment 4: Path A 诊断 on V5 best (如果有)
# ================================================================
# 对 V5 best.pt 做 Path A 诊断，与 V3 baseline 的 ExpoGap 对比
#
# 用法: bash review/0427/run_me/04_pathA_v5_best.sh [GPU_ID]
# 预计耗时: ~30min
# ================================================================
set -euo pipefail

GPU_ID="${1:-0}"
PYTHON="/home/qujiaxiang/.conda/envs/rae/bin/python"

V5_DIR="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v5_rollout_heavy"
V5_CONFIG="configs/pet_flow/pet_flow_first_hop_224_v5_rollout_heavy.yaml"

if [ ! -f "${V5_DIR}/best.pt" ]; then
    echo "ERROR: ${V5_DIR}/best.pt 不存在"
    exit 1
fi

OUT_DIR="/data_2/qujiaxiang/outputs/PET_LatentResidual/diagnostics/pathA_v5_rollout_heavy_best"

echo "=== Experiment 4: Path A on V5 best ==="

CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} -u scripts/diagnose_tf_rollout_gap.py \
    --config "${V5_CONFIG}" \
    --checkpoint "${V5_DIR}/best.pt" \
    --split val --max-slices 0 --batch-size 8 --device cuda:0 \
    --out-dir "${OUT_DIR}"

echo ""
echo "=== Path A V5 结果 ==="
cat "${OUT_DIR}/tf_rollout_gap_val.json" 2>/dev/null || echo "  (JSON 不存在)"
