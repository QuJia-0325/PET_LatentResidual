#!/bin/bash
# V5 Rollout-Heavy 启动脚本（温和版）
# 用法: bash scripts/launch_v5_rollout_heavy.sh [GPU_ID]
# 自动：查 200K v3 best.pt 步号 → 设 max_steps → 启动训练
set -euo pipefail

GPU_ID="${1:-0}"
V3_DIR="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_200k_transport_v3"
V3_BEST="${V3_DIR}/best.pt"
CONFIG="configs/pet_flow/pet_flow_first_hop_224_v5_rollout_heavy.yaml"
NEW_STEPS=50000

echo "=== V5 Rollout-Heavy Launch (温和版) ==="
echo "[1/6] 检查 200K v3 best.pt ..."
if [ ! -f "${V3_BEST}" ]; then
    echo "ERROR: ${V3_BEST} 不存在"
    exit 1
fi

# 读取 resume step 和 val
CKPT_INFO=$(python -c "
import torch
c = torch.load('${V3_BEST}', map_location='cpu')
step = c.get('step', 0)
val = c.get('best_val', -1)
print(f'{step} {val:.6f}')
")
RESUME_STEP=$(echo $CKPT_INFO | awk '{print $1}')
RESUME_VAL=$(echo $CKPT_INFO | awk '{print $2}')
MAX_STEPS=$((RESUME_STEP + NEW_STEPS))

echo "[2/6] Resume step=${RESUME_STEP}, val=${RESUME_VAL}"
echo "       max_steps=${MAX_STEPS} (= ${RESUME_STEP} + ${NEW_STEPS})"

# 安全检查
if [ "$RESUME_STEP" -lt 10000 ]; then
    echo "ERROR: resume step ${RESUME_STEP} 太小，200K v3 可能还没开始?"
    exit 1
fi

echo "[3/6] 写入 max_steps 到 config ..."
sed -i "s/^  max_steps: .*/  max_steps: ${MAX_STEPS}/" "${CONFIG}"
echo "       已更新 config: max_steps=${MAX_STEPS}"

echo "[4/6] 确认 config 关键参数 ..."
echo "  rollout lambda:   $(grep 'lambda_start: 1.50' ${CONFIG} | head -1 || echo 'NOT FOUND')"
echo "  image_aux lambda:  $(grep 'lambda_start: 0.08' ${CONFIG} | head -1 || echo 'NOT FOUND')"
echo "  step_weights:      $(grep -A4 'step_weights:' ${CONFIG} | tail -4 | tr -d ' -' | tr '\n' ',' || echo 'NOT FOUND')"
echo "  SF enabled:        $(grep 'enabled: false' ${CONFIG} | head -1 || echo 'NOT FOUND')"
echo "  max_steps:         $(grep 'max_steps' ${CONFIG} | head -1)"

LOG_DIR="review/0426/logs_train"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/v5_rollout_heavy_gpu${GPU_ID}.log"

echo "[5/6] V5 参数摘要"
echo ""
echo "  ┌─────────────────────────────────────────────┐"
echo "  │ V5 Rollout-Heavy (温和版)                    │"
echo "  │                                             │"
echo "  │ rollout λ:     0.25 → 1.50  (6×)           │"
echo "  │ image_aux λ:   0.12 → 0.08  (÷1.5)         │"
echo "  │ step_weights:  [0.8, 1.0, 1.5, 2.5]        │"
echo "  │ SF:            disabled                     │"
echo "  │                                             │"
echo "  │ resume:        step ${RESUME_STEP}          │"
echo "  │ new steps:     ${NEW_STEPS}                 │"
echo "  │ max_steps:     ${MAX_STEPS}                 │"
echo "  └─────────────────────────────────────────────┘"
echo ""

echo "[6/6] 启动训练 (GPU=${GPU_ID}) ..."
echo "       log: ${LOG_FILE}"

CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python -u train_first_hop.py \
    --config "${CONFIG}" \
    --resume "${V3_BEST}" \
    > "${LOG_FILE}" 2>&1 &

PID=$!
echo ""
echo "训练已启动 (PID=${PID})"
echo ""
echo "=== 监控命令 ==="
echo "  tail -f ${LOG_FILE}"
echo "  grep 'roll_frac' ${LOG_FILE} | tail -5"
echo "  grep '\\[val\\]' ${LOG_FILE} | tail -5"
echo ""
echo "=== Go/No-Go 检查 ==="
echo "  +5K 步:  grep 'val_pair_total' → 应 < 0.00005 (相变警报)"
echo "  +10K 步: grep 'val_chain_normal_mse' → 应 < ${RESUME_VAL} × 1.05"
echo "  +25K 步: val_chain_normal_mse 应有改善趋势"
echo "  +50K 步: full-val eval + Path A redo"
echo ""
echo "=== 止损条件 ==="
echo "  val_pair_total > 0.00005 @ +5K  → 立即停"
echo "  val_select > baseline × 1.05 @ +10K → 止损"
echo "  roll_frac < 20% → λ_roll 不够，考虑升级"
