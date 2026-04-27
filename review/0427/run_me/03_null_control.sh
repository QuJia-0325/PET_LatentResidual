#!/bin/bash
# ================================================================
# Experiment 3: Null-Control (baseline resume 继续训 50K，不改 loss)
# ================================================================
# 目的: 分离 "V5 loss 改动的影响" vs "继续训练 + LR schedule 的影响"
#
# 如果 null-control 也退化 → 问题在 resume/LR，不在 V5 的 loss 改动
# 如果 null-control 持平或改善 → 问题确实在 V5 的 loss 改动
#
# 配置: 使用 200K v3 原始 config，仅改 max_steps 和 resume 兼容性
#       不改 rollout λ / image_aux λ / step_weights
#
# 用法: bash review/0427/run_me/03_null_control.sh [GPU_ID]
# 预计耗时: ~30h (50K 步)
# ================================================================
set -euo pipefail

GPU_ID="${1:-0}"
PYTHON="/home/qujiaxiang/.conda/envs/rae/bin/python"

V3_DIR="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_200k_transport_v3"
V3_BEST="${V3_DIR}/best.pt"
CONFIG="configs/pet_flow/pet_flow_first_hop_224_v5_null_control.yaml"

echo "=== Experiment 3: Null-Control ==="

# 检查 config 是否存在
if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: ${CONFIG} 不存在。请先创建 null-control config。"
    echo "  方法: 复制 200K v3 config，仅改以下项:"
    echo "    run_name: first_hop_224_v5_null_control"
    echo "    max_steps: <resume_step + 50000>"
    echo "    strict_resume_compat: false"
    echo "    resume_allow_*: true"
    echo "    progress_bar: false"
    echo "    log_grad_norms: true"
    echo "    其余参数完全不变"
    exit 1
fi

# 读取 resume step
CKPT_INFO=$(${PYTHON} -c "
import torch
c = torch.load('${V3_BEST}', map_location='cpu')
step = c.get('step', 0)
val = c.get('best_val', -1)
print(f'{step} {val:.6f}')
")
RESUME_STEP=$(echo $CKPT_INFO | awk '{print $1}')
RESUME_VAL=$(echo $CKPT_INFO | awk '{print $2}')
MAX_STEPS=$((RESUME_STEP + 50000))

echo "[1/3] Resume step=${RESUME_STEP}, val=${RESUME_VAL}"
echo "      max_steps=${MAX_STEPS}"

# 写入 max_steps
sed -i "s/^  max_steps: .*/  max_steps: ${MAX_STEPS}/" "${CONFIG}"
echo "[2/3] 已更新 config: max_steps=${MAX_STEPS}"

LOG_DIR="review/0427/logs_train"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/v5_null_control_gpu${GPU_ID}.log"

echo "[3/3] 启动 null-control (GPU=${GPU_ID}) ..."
echo "  config: ${CONFIG}"
echo "  resume: ${V3_BEST}"
echo "  log: ${LOG_FILE}"
echo ""

CUDA_VISIBLE_DEVICES=${GPU_ID} nohup ${PYTHON} -u train_first_hop.py \
    --config "${CONFIG}" \
    --resume "${V3_BEST}" \
    > "${LOG_FILE}" 2>&1 &

PID=$!
echo "训练已启动 (PID=${PID})"
echo ""
echo "=== 监控 ==="
echo "  tail -f ${LOG_FILE}"
echo "  grep '\\[val\\]' ${LOG_FILE} | tail -5"
echo ""
echo "=== 与 V5-main 对比 ==="
echo "  null-control 用 baseline loss 权重继续训练"
echo "  如果 null-control 也退化 → resume/LR 是主因"
echo "  如果 null-control 持平 → V5 loss 改动是主因"
