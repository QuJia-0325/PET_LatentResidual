#!/bin/bash
# ================================================================
# Experiment 2: Path A 诊断 (V3 baseline @ step 86800)
# ================================================================
# 对 V3 baseline 的 resume 起点做 Path A 诊断
# 获得 ExpoGap_86800 作为公平对比基线
# 之前的 Path A 数据来自 Scheme C (step ~46K)，不是 200K v3
#
# 用法: bash review/0427/run_me/02_pathA_baseline.sh [GPU_ID]
# 预计耗时: ~30min
# ================================================================
set -euo pipefail

GPU_ID="${1:-0}"
PYTHON="/home/qujiaxiang/.conda/envs/rae/bin/python"

V3_BEST="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_200k_transport_v3/best.pt"
V3_CONFIG="configs/pet_flow/pet_flow_first_hop_224_200k_transport_v3.yaml"
OUT_DIR="/data_2/qujiaxiang/outputs/PET_LatentResidual/diagnostics/pathA_v3_200k_best"

echo "=== Experiment 2: Path A on V3 200K best ==="
echo "获取 ExpoGap baseline (step 86800)"
echo ""

CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} -u scripts/diagnose_tf_rollout_gap.py \
    --config "${V3_CONFIG}" \
    --checkpoint "${V3_BEST}" \
    --split val --max-slices 0 --batch-size 8 --device cuda:0 \
    --out-dir "${OUT_DIR}"

echo ""
echo "=== Path A 结果 ==="
echo "  ${OUT_DIR}/"
cat "${OUT_DIR}/tf_rollout_gap_val.json" 2>/dev/null || echo "  (JSON 文件不存在)"
