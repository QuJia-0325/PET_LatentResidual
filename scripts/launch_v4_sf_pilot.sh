#!/bin/bash
# V4 SF Pilot 启动脚本
# 用法: bash scripts/launch_v4_sf_pilot.sh [GPU_ID]
# 自动：查 200K v3 best.pt 步号 → 算 max_steps → 改 config → 启动训练
set -euo pipefail

GPU_ID="${1:-0}"
V3_DIR="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_200k_transport_v3"
V3_BEST="${V3_DIR}/best.pt"
CONFIG="configs/pet_flow/pet_flow_first_hop_224_v4_sf_pilot.yaml"
PILOT_STEPS=10000

echo "=== V4 SF Pilot Launch ==="
echo "[1/5] 检查 200K v3 best.pt ..."
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
MAX_STEPS=$((RESUME_STEP + PILOT_STEPS))

echo "[2/5] Resume step=${RESUME_STEP}, val=${RESUME_VAL}"
echo "       max_steps=${MAX_STEPS} (= ${RESUME_STEP} + ${PILOT_STEPS})"

# 安全检查
if [ "$RESUME_STEP" -lt 10000 ]; then
    echo "ERROR: resume step ${RESUME_STEP} 太小，200K v3 可能还没开始?"
    exit 1
fi

echo "[3/5] 写入 max_steps 到 config ..."
# 用 sed 替换 max_steps 行
sed -i "s/^  max_steps: .*/  max_steps: ${MAX_STEPS}/" "${CONFIG}"
echo "       已更新 config: max_steps=${MAX_STEPS}"

echo "[4/5] 确认 config 关键参数 ..."
echo "       schedule_origin: $(grep 'schedule_origin' ${CONFIG} | head -1)"
echo "       warmup_steps: $(grep 'warmup_steps' ${CONFIG} | head -1)"
echo "       ramp_steps: $(grep 'ramp_steps' ${CONFIG} | head -1)"
echo "       max_steps: $(grep 'max_steps' ${CONFIG} | head -1)"

LOG_DIR="review/0426/logs_train"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/v4_sf_pilot_gpu${GPU_ID}.log"

echo "[5/5] 启动训练 (GPU=${GPU_ID}) ..."
echo "       log: ${LOG_FILE}"
echo "       resume: ${V3_BEST}"
echo ""
echo "--- 预期 SF 调度 ---"
echo "  step ${RESUME_STEP} ~ $((RESUME_STEP + 2000)): sf_alpha=0 (warmup, 纯 GT)"
echo "  step $((RESUME_STEP + 2000)) ~ $((RESUME_STEP + 5000)): sf_alpha 0→1 (ramp)"
echo "  step $((RESUME_STEP + 5000)) ~ ${MAX_STEPS}: sf_alpha=1 (稳态 SF)"
echo ""

CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python -u train_first_hop.py \
    --config "${CONFIG}" \
    --resume "${V3_BEST}" \
    > "${LOG_FILE}" 2>&1 &

PID=$!
echo "训练已启动 (PID=${PID})"
echo ""
echo "=== 监控命令 ==="
echo "  tail -f ${LOG_FILE}"
echo "  grep 'sf_alpha' ${LOG_FILE} | tail -5"
echo "  grep '\\[val\\]' ${LOG_FILE} | tail -5"
echo ""
echo "=== Go/No-Go 检查 ==="
echo "  +1K步: sf_alpha 应为 0.000, val 不应退化 >35%"
echo "  +5K步: sf_alpha 应为 1.000, sf_gap 应 > 0"
echo "  +10K步: val_select_score 应 ≤ ${RESUME_VAL} × 1.05"
