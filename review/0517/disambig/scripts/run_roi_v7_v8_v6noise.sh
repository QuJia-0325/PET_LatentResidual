#!/usr/bin/env bash
set -euo pipefail
cd /home/qujiaxiang/project/PET_LatentResidual
OUT=review/0517/disambig/roi_psnr
LOG_DIR=review/0517/disambig/logs
mkdir -p "$OUT" "$LOG_DIR"
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
LOG=${LOG_DIR}/roi_v7_v8_v6noise_gpu${GPU:-1}_${STAMP}.log
run_eval() {
  local exp="$1"
  local kind="$2"
  local cfg="review/0511/log_snapshots_20260516_163900/configs/${exp}_config.resolved.yaml"
  local base=""
  case "$exp" in
    V7) base=/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V7/run/first_hop_224_v7_gronwall_raw ;;
    V8) base=/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V8/run/first_hop_224_v8_no_image_aux ;;
    V6_NOISE) base=/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V6_NOISE/run/first_hop_224_v6_seed1337 ;;
  esac
  local ckpt="${base}/${kind}.pt"
  local tag="${exp}_${kind}"
  echo "===== ${tag} start $(date '+%Y-%m-%d %H:%M:%S %Z') ====="
  CUDA_VISIBLE_DEVICES=${GPU:-1} /home/qujiaxiang/.conda/envs/rae/bin/python -u tools/eval_roi_psnr.py \
    --config "$cfg" \
    --checkpoint "$ckpt" \
    --tag "$tag" \
    --max-slices ${MAX_SLICES:-0} \
    --decode-mode both \
    --batch-size ${BATCH_SIZE:-8} \
    --device cuda:0 \
    --out-dir "$OUT/${tag}"
  echo "===== ${tag} done $(date '+%Y-%m-%d %H:%M:%S %Z') ====="
}
{
  echo "[launch] $(date '+%Y-%m-%d %H:%M:%S %Z') GPU=${GPU:-1} MAX_SLICES=${MAX_SLICES:-0}"
  run_eval V7 last
  run_eval V8 last
  run_eval V6_NOISE last
  echo "[all_done] $(date '+%Y-%m-%d %H:%M:%S %Z')"
} 2>&1 | tee "$LOG"
