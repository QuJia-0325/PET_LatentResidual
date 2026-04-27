#!/bin/bash
# ================================================================
# Experiment 1: Full-Val Evaluation (消除 rolling-val window 混淆)
# ================================================================
# 对 V3 baseline, V5 best, V5 latest 三个 ckpt 做 full-val eval
# 使用相同的 eval 脚本和相同的 config (V5 config)
# --max-slices 0 = 全量 val set
#
# 用法: bash review/0427/run_me/01_fullval_eval.sh [GPU_ID]
# 预计耗时: ~3×20min = ~1h
# ================================================================
set -euo pipefail

GPU_ID="${1:-0}"
PYTHON="/home/qujiaxiang/.conda/envs/rae/bin/python"

# Checkpoint 路径
V3_BEST="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_200k_transport_v3/best.pt"
V5_DIR="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v5_rollout_heavy"
# V5 config（用于构建 dataset/model）
V5_CONFIG="configs/pet_flow/pet_flow_first_hop_224_v5_rollout_heavy.yaml"
# V3 config（用于 V3 baseline eval）
V3_CONFIG="configs/pet_flow/pet_flow_first_hop_224_200k_transport_v3.yaml"

OUT_BASE="/data_2/qujiaxiang/outputs/PET_LatentResidual/eval_0427_fullval"

echo "=== Experiment 1: Full-Val Evaluation ==="
echo "消除 rolling-val window 噪声，统一评估协议"
echo ""

# --- V3 baseline ---
echo "[1/3] V3 baseline full-val ..."
CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} -u eval_first_hop_224_clip3.py \
    --config "${V3_CONFIG}" \
    --checkpoint "${V3_BEST}" \
    --split val --max-slices 0 --batch-size 8 --device cuda:0 \
    --decode-mode both \
    --out-dir "${OUT_BASE}/v3_baseline"
echo "[1/3] Done."

# --- V5 best.pt ---
if [ -f "${V5_DIR}/best.pt" ]; then
    echo "[2/3] V5 best.pt full-val ..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} -u eval_first_hop_224_clip3.py \
        --config "${V5_CONFIG}" \
        --checkpoint "${V5_DIR}/best.pt" \
        --split val --max-slices 0 --batch-size 8 --device cuda:0 \
        --decode-mode both \
        --out-dir "${OUT_BASE}/v5_best"
    echo "[2/3] Done."
else
    echo "[2/3] SKIP: ${V5_DIR}/best.pt 不存在"
fi

# --- V5 latest checkpoint (step_10000 or last saved) ---
V5_LATEST=$(ls -t ${V5_DIR}/step_*.pt 2>/dev/null | head -1 || echo "")
if [ -n "${V5_LATEST}" ]; then
    STEP_NAME=$(basename ${V5_LATEST} .pt)
    echo "[3/3] V5 ${STEP_NAME} full-val ..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} -u eval_first_hop_224_clip3.py \
        --config "${V5_CONFIG}" \
        --checkpoint "${V5_LATEST}" \
        --split val --max-slices 0 --batch-size 8 --device cuda:0 \
        --decode-mode both \
        --out-dir "${OUT_BASE}/v5_${STEP_NAME}"
    echo "[3/3] Done."
else
    echo "[3/3] SKIP: ${V5_DIR}/step_*.pt 不存在"
fi

echo ""
echo "=== Full-Val 结果目录 ==="
echo "  ${OUT_BASE}/"
ls -la ${OUT_BASE}/ 2>/dev/null || echo "  (目录为空)"
echo ""
echo "=== 对比方法 ==="
echo "  比较三个目录下的 JSON 文件中的 chain PSNR 和 per-hop MSE"
echo "  关键指标: chain_normal_psnr, chain_d20_psnr, transport_avg_psnr"
