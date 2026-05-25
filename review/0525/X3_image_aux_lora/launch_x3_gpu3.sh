#!/usr/bin/env bash
set -euo pipefail

cd /home/qujiaxiang/project/PET_LatentResidual
export PYTHONPATH=/home/qujiaxiang/project/PET_LatentResidual:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=3

LOG_FILE=review/0525/X3_image_aux_lora/X3_train_20260525_222455_gpu3.log
PID_FILE=review/0525/X3_image_aux_lora/X3.pid
GPU_FILE=review/0525/X3_image_aux_lora/X3.gpu
V7_CKPT=/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V7/run/first_hop_224_v7_gronwall_raw/best.pt

test -f "$V7_CKPT"
echo "$$" > "$PID_FILE"
echo "3" > "$GPU_FILE"

exec /home/qujiaxiang/.conda/envs/rae/bin/python train_first_hop.py \
  --config review/0525/X3_image_aux_lora/X3_image_aux_lora.yaml \
  --resume "$V7_CKPT" \
  > "$LOG_FILE" 2>&1
