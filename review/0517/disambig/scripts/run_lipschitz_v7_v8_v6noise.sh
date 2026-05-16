#!/usr/bin/env bash
set -euo pipefail
cd /home/qujiaxiang/project/PET_LatentResidual
OUT=review/0517/disambig/lipschitz
LOG_DIR=review/0517/disambig/logs
mkdir -p "$OUT" "$LOG_DIR"
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
LOG=${LOG_DIR}/lipschitz_v7_v8_v6noise_gpu${GPU:-1}_${STAMP}.log
{
  echo "[launch] $(date '+%Y-%m-%d %H:%M:%S %Z') GPU=${GPU:-1}"
  for EXP in V7 V8 V6_NOISE; do
    CFG=review/0511/log_snapshots_20260516_163900/configs/${EXP}_config.resolved.yaml
    case $EXP in
      V7) CKPT=/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V7/run/first_hop_224_v7_gronwall_raw/last.pt ;;
      V8) CKPT=/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V8/run/first_hop_224_v8_no_image_aux/last.pt ;;
      V6_NOISE) CKPT=/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V6_NOISE/run/first_hop_224_v6_seed1337/last.pt ;;
    esac
    echo "===== ${EXP} start $(date '+%Y-%m-%d %H:%M:%S %Z') ====="
    CUDA_VISIBLE_DEVICES=${GPU:-1} /home/qujiaxiang/.conda/envs/rae/bin/python -u tools/estimate_per_hop_lipschitz.py \
      --config "$CFG" \
      --checkpoint "$CKPT" \
      --out "$OUT/${EXP}_lipschitz.json" \
      --n-samples ${N_SAMPLES:-64} \
      --eps ${EPS:-1e-3} \
      --device cuda:0
    echo "===== ${EXP} done $(date '+%Y-%m-%d %H:%M:%S %Z') ====="
  done
  echo "[all_done] $(date '+%Y-%m-%d %H:%M:%S %Z')"
} 2>&1 | tee "$LOG"
