#!/usr/bin/env bash
# V9 NORMAL-emphasis launch script — review/0516
# Generated 2026-05-16. Run from the repo root on the GPU server.
#
# Pre-launch checklist (manual):
#   1. Confirm /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0516_runs/V9_normal_emphasis
#      does not exist (require_fresh_output_dir=true will refuse otherwise).
#   2. Confirm `git rev-parse HEAD` matches the commit referenced in
#      review/0516/V9_normal_emphasis/V9_PREREGISTRATION.md.
#   3. Confirm GPU is free with `nvidia-smi`.
#   4. Confirm conda env: /home/qujiaxiang/.conda/envs/rae/bin/python --version
#
# Estimated wall-clock: ~7 days (matches V7 on RTX A6000).
set -euo pipefail

REPO_ROOT="/home/qujiaxiang/project/PET_LatentResidual"
CONFIG="${REPO_ROOT}/review/0516/V9_normal_emphasis/V9_normal_emphasis.yaml"
LOG_DIR="${REPO_ROOT}/review/0516/V9_normal_emphasis/logs"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/V9_train_${TS}.log"
GPU="${1:-0}"   # default GPU0; override: ./run_v9_train.sh 2

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

echo "[launch] $(date -Iseconds) GPU=${GPU} config=${CONFIG}" | tee -a "${LOG}"
echo "[launch] git HEAD=$(git rev-parse HEAD)" | tee -a "${LOG}"
echo "[launch] log=${LOG}" | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU}" \
  /home/qujiaxiang/.conda/envs/rae/bin/python \
  train_first_hop.py \
  --config "${CONFIG}" \
  2>&1 | tee -a "${LOG}"

echo "[done] $(date -Iseconds)" | tee -a "${LOG}"
