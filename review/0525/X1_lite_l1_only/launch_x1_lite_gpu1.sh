#!/usr/bin/env bash
set -euo pipefail

cd /home/qujiaxiang/project/PET_LatentResidual
export PYTHONPATH=/home/qujiaxiang/project/PET_LatentResidual:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=1

LOG_FILE=review/0525/X1_lite_l1_only/X1_lite_train_20260525_222455_gpu1.log
PID_FILE=review/0525/X1_lite_l1_only/X1_lite.pid
GPU_FILE=review/0525/X1_lite_l1_only/X1_lite.gpu

echo "$$" > "$PID_FILE"
echo "1" > "$GPU_FILE"

exec /home/qujiaxiang/.conda/envs/rae/bin/python train_first_hop.py \
  --config review/0525/X1_lite_l1_only/X1_lite_l1_only.yaml \
  > "$LOG_FILE" 2>&1
