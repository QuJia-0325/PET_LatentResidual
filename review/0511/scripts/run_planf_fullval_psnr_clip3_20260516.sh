#!/usr/bin/env bash
set -euo pipefail

cd /home/qujiaxiang/project/PET_LatentResidual

PY=/home/qujiaxiang/.conda/envs/rae/bin/python
EVAL=review/0505/operator/scripts/eval_first_hop_fullval_psnr_chain_mse.py
OUT=/data_2/qujiaxiang/outputs/PET_LatentResidual/eval_0511_planf_fullval_psnr_clip3
LOG_DIR=review/0511/logs_eval
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
LOG=${LOG_DIR}/planf_v7_v8_v6noise_best_last_fullval_psnr_clip3_gpu${GPU:-1}_${STAMP}.log

mkdir -p "$OUT" "$LOG_DIR"

run_eval() {
  local tag="$1"
  local config="$2"
  local ckpt="$3"
  echo "===== ${tag} start $(date '+%Y-%m-%d %H:%M:%S %Z') ====="
  echo "config=${config}"
  echo "checkpoint=${ckpt}"
  CUDA_VISIBLE_DEVICES=${GPU:-1} PYTHONPATH=/home/qujiaxiang/project/PET_LatentResidual \
    "$PY" -u "$EVAL" \
      --config "$config" \
      --checkpoint "$ckpt" \
      --tag "$tag" \
      --split val \
      --max-slices 0 \
      --batch-size ${BATCH_SIZE:-8} \
      --device cuda:0 \
      --decode-mode both \
      --out-dir "$OUT"
  echo "===== ${tag} done $(date '+%Y-%m-%d %H:%M:%S %Z') ====="
}

{
  echo "[launch] $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "[env] GPU=${GPU:-1} BATCH_SIZE=${BATCH_SIZE:-8} OUT=${OUT}"
  echo "[metric] psnr_metric=src.utils.metrics.calc_psnr_clip3"

  run_eval planf_v7_best \
    review/0505/local/runs/V7/config.resolved.yaml \
    /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V7/run/first_hop_224_v7_gronwall_raw/best.pt

  run_eval planf_v7_last \
    review/0505/local/runs/V7/config.resolved.yaml \
    /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V7/run/first_hop_224_v7_gronwall_raw/last.pt

  run_eval planf_v8_best \
    review/0505/local/runs/V8/config.resolved.yaml \
    /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V8/run/first_hop_224_v8_no_image_aux/best.pt

  run_eval planf_v8_last \
    review/0505/local/runs/V8/config.resolved.yaml \
    /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V8/run/first_hop_224_v8_no_image_aux/last.pt

  run_eval planf_v6noise_best \
    review/0505/local/runs/V6_NOISE/config.resolved.yaml \
    /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V6_NOISE/run/first_hop_224_v6_seed1337/best.pt

  run_eval planf_v6noise_last \
    review/0505/local/runs/V6_NOISE/config.resolved.yaml \
    /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V6_NOISE/run/first_hop_224_v6_seed1337/last.pt

  echo "[all_done] $(date '+%Y-%m-%d %H:%M:%S %Z')"
} 2>&1 | tee "$LOG"
