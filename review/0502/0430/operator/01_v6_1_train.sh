#!/bin/bash
# ================================================================
# V6.1 Transport-First + Rollout Floor: 200K from scratch
# ================================================================
# 与 V6 唯一区别：rollout.lambda_start = 0.05（V6 = 0.0）
# 让 Phase I（0-50K）就有 chain composition 梯度信号
#
# 在 V5 NC 完成释放 GPU 后启动
# V6（GPU 1）继续运行，V6.1（GPU 3）并行作为 hedge
#
# 用法: bash review/0430/operator/01_v6_1_train.sh [GPU_ID]
# 预计耗时: ~5-6 天（200K 步）
# ================================================================
set -euo pipefail

GPU_ID="${1:-3}"
PYTHON="/home/qujiaxiang/.conda/envs/rae/bin/python"
CONFIG="configs/pet_flow/pet_flow_first_hop_224_v6_1_rollout_floor.yaml"
V6_1_DIR="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v6_1_rollout_floor"

echo "=== V6.1 Transport-First + Rollout Floor: 200K from scratch ==="
echo "Config: ${CONFIG}"
echo "Output: ${V6_1_DIR}"
echo "GPU: ${GPU_ID}"
echo ""
echo "与 V6 唯一区别: rollout.lambda_start = 0.05 (V6 = 0.0)"
echo ""

if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: ${CONFIG} 不存在"
    exit 1
fi

if [ -d "${V6_1_DIR}" ]; then
    echo "ERROR: ${V6_1_DIR} 已存在。V6.1 是 from-scratch 训练。"
    echo "如果确认要重新开始: rm -rf ${V6_1_DIR}"
    exit 1
fi

echo "--- 启动 V6.1 ---"
CUDA_VISIBLE_DEVICES="${GPU_ID}" ${PYTHON} train_first_hop.py \
    --config "${CONFIG}" \
    2>&1 | tee "${V6_1_DIR}_train.log"
