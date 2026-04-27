#!/bin/bash
# ================================================================
# V5 归因实验闭环（0427 遗留任务）
# ================================================================
# 如果 0427 的 01/03/05 脚本尚未执行或结果不完整，在此统一补齐
# V5 归因不阻塞 V6 启动，但结果影响 V6 是否继续到 full 200K
#
# 用法: bash review/0428/operator/04_v5_attribution.sh [GPU_ID]
# ================================================================
set -euo pipefail

GPU_ID="${1:-0}"
PYTHON="/home/qujiaxiang/.conda/envs/rae/bin/python"

V3_DIR="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_200k_transport_v3"
V5_DIR="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v5_rollout_heavy"
NC_DIR="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v5_null_control"
V3_CONFIG="configs/pet_flow/pet_flow_first_hop_224_200k_transport_v3.yaml"
V5_CONFIG="configs/pet_flow/pet_flow_first_hop_224_v5_rollout_heavy.yaml"
NC_CONFIG="configs/pet_flow/pet_flow_first_hop_224_v5_null_control.yaml"
OUT_BASE="/data_2/qujiaxiang/outputs/PET_LatentResidual/eval_0428_v5_attribution"
mkdir -p "${OUT_BASE}"

echo "=== V5 归因实验闭环 ==="
echo ""

# --- 检查 0427 遗留状态 ---
echo "--- 检查 0427 遗留实验状态 ---"

# 1. V3 fullval
V3_FULLVAL="/data_2/qujiaxiang/outputs/PET_LatentResidual/eval_0427_fullval/v3_best"
if [ -f "${V3_FULLVAL}/first_hop_224_val_clip3_eval.json" ]; then
    echo "[OK] V3 fullval 已完成: ${V3_FULLVAL}"
else
    echo "[TODO] V3 fullval 未完成，执行中..."
    CUDA_VISIBLE_DEVICES="${GPU_ID}" ${PYTHON} eval_first_hop_224_clip3.py \
        --config "${V3_CONFIG}" \
        --checkpoint "${V3_DIR}/best.pt" \
        --out-dir "${OUT_BASE}/v3_best" \
        --max-slices 0
fi

# 2. V5 fullval
V5_FULLVAL="/data_2/qujiaxiang/outputs/PET_LatentResidual/eval_0427_fullval/v5_best"
if [ -f "${V5_FULLVAL}/first_hop_224_val_clip3_eval.json" ]; then
    echo "[OK] V5 fullval 已完成: ${V5_FULLVAL}"
else
    if [ -f "${V5_DIR}/best.pt" ]; then
        echo "[TODO] V5 fullval 未完成，执行中..."
        CUDA_VISIBLE_DEVICES="${GPU_ID}" ${PYTHON} eval_first_hop_224_clip3.py \
            --config "${V5_CONFIG}" \
            --checkpoint "${V5_DIR}/best.pt" \
            --out-dir "${OUT_BASE}/v5_best" \
            --max-slices 0
    else
        echo "[SKIP] V5 best.pt 不存在"
    fi
fi

# 3. Null-control 训练
if [ -d "${NC_DIR}" ] && { [ -f "${NC_DIR}/best.pt" ] || [ -f "${NC_DIR}/last.pt" ]; }; then
    echo "[OK] Null-control 训练目录存在: ${NC_DIR}"
    # 检查是否还在训练
    NC_JSONL="${NC_DIR}/metrics.jsonl"
    if [ -f "${NC_JSONL}" ]; then
        LAST_STEP=$(tail -1 "${NC_JSONL}" | python3 -c "import json,sys; print(json.loads(sys.stdin.read()).get('step','?'))" 2>/dev/null || echo "?")
        echo "  last step: ${LAST_STEP}"
    fi
    # Null-control fullval
    # 兼容两个可能的输出路径（0427 脚本用 eval_0427_fullval_null_control/null_best）
    NC_FULLVAL_A="/data_2/qujiaxiang/outputs/PET_LatentResidual/eval_0427_fullval_null_control/null_best"
    NC_FULLVAL_B="${OUT_BASE}/null_control_best"
    if [ -f "${NC_FULLVAL_A}/first_hop_224_val_clip3_eval.json" ]; then
        echo "[OK] Null-control fullval 已完成: ${NC_FULLVAL_A}"
    elif [ -f "${NC_FULLVAL_B}/first_hop_224_val_clip3_eval.json" ]; then
        echo "[OK] Null-control fullval 已完成: ${NC_FULLVAL_B}"
    else
        if [ -f "${NC_DIR}/best.pt" ]; then
            echo "[TODO] Null-control fullval 未完成，执行中..."
            CUDA_VISIBLE_DEVICES="${GPU_ID}" ${PYTHON} eval_first_hop_224_clip3.py \
                --config "${NC_CONFIG}" \
                --checkpoint "${NC_DIR}/best.pt" \
                --out-dir "${NC_FULLVAL_B}" \
                --max-slices 0
        fi
    fi
else
    echo "[TODO] Null-control 训练未启动。如需启动:"
    echo "  bash review/0427/operator/03_null_control.sh ${GPU_ID}"
fi

echo ""
echo "=== 归因分析完成。比较结果: ==="
echo "  V3 baseline: ${V3_FULLVAL} 或 ${OUT_BASE}/v3_best"
echo "  V5 best:     ${V5_FULLVAL} 或 ${OUT_BASE}/v5_best"
echo "  Null-control: ${OUT_BASE}/null_control_best"
