#!/bin/bash
# ================================================================
# V6 Transport-First: 200K from scratch（Pilot → Full 递进）
# ================================================================
# Pilot 策略：
#   Step 1: 直接用正式权重启动 200K（pair_weight=15, lambda_end=4.0）
#   Step 2: 在 +1K/+10K/+50K 做 Go/No-Go 检查（看日志即可）
#   Step 3: 50K 通过后继续训练，不重启
#   Step 4: +130K/+150K 做 Phase II/III 验证
#
# Go/No-Go 检查点（看 metrics JSONL）：
#   +1K:   pair_loss < 5e-4, 无 NaN
#   +10K:  pair_frac > 70%
#   +50K:  img_frac < 30%
#   +130K: img_frac < 30%, img_raw 无连续上升
#   +150K: transport >= 60%, val_chain_normal_mse 趋势改善
#
# 用法: bash review/0428/operator/01_v6_train.sh [GPU_ID]
# 预计耗时: ~5-6 天（200K 步）
# ================================================================
set -euo pipefail

GPU_ID="${1:-0}"
PYTHON="/home/qujiaxiang/.conda/envs/rae/bin/python"
CONFIG="configs/pet_flow/pet_flow_first_hop_224_v6_transport_first.yaml"
V6_DIR="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v6_transport_first"

echo "=== V6 Transport-First: 200K from scratch ==="
echo "Config: ${CONFIG}"
echo "Output: ${V6_DIR}"
echo "GPU: ${GPU_ID}"
echo ""

# 检查 config 存在
if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: ${CONFIG} 不存在"
    exit 1
fi

# 检查 output dir 不存在（from scratch，require_fresh_output_dir: true）
if [ -d "${V6_DIR}" ]; then
    echo "ERROR: ${V6_DIR} 已存在。V6 是 from-scratch 训练，不应有残留目录。"
    echo "如果确认要重新开始，请手动删除: rm -rf ${V6_DIR}"
    exit 1
fi

echo "--- 启动 V6 训练 ---"
echo "注意事项："
echo "  1. 训练启动后，在 +1K 步检查 metrics JSONL 的 pair_loss 和 NaN"
echo "  2. 使用 02_v6_monitor.sh 自动检查 Go/No-Go 门控"
echo "  3. 如果需要中断，kill 进程即可；下次用 resume 继续"
echo ""

CUDA_VISIBLE_DEVICES="${GPU_ID}" ${PYTHON} train_first_hop.py \
    --config "${CONFIG}" \
    2>&1 | tee "${V6_DIR}_train.log"
