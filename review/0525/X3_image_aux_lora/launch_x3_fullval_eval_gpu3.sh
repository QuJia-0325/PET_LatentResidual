#!/usr/bin/env bash
set -euo pipefail
cd /home/qujiaxiang/project/PET_LatentResidual
export PYTHONPATH=/home/qujiaxiang/project/PET_LatentResidual:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=3
PYTHON=/home/qujiaxiang/.conda/envs/rae/bin/python
X3_OUT=/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0525_runs/X3_image_aux_lora/first_hop_224_x3_image_aux_lora
OUT_BASE=/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0525_eval/x3_eval
LOCAL_LOG_DIR=review/0525/X3_image_aux_lora/fullval_eval/logs
LOCAL_ART_DIR=review/0525/X3_image_aux_lora/fullval_eval/artifacts
mkdir -p "$OUT_BASE" "$LOCAL_LOG_DIR" "$LOCAL_ART_DIR"
for tag in best last; do
  LOG="$LOCAL_LOG_DIR/x3_${tag}_eval_20260526_gpu3.log"
  "$PYTHON" review/0505/operator/scripts/eval_first_hop_fullval_psnr_chain_mse.py \
    --config "$X3_OUT/config.yaml" \
    --checkpoint "$X3_OUT/${tag}.pt" \
    --tag "x3_image_aux_lora_${tag}" \
    --split val --max-slices 0 --batch-size 8 --decode-mode both \
    --out-dir "$OUT_BASE" \
    2>&1 | tee "$LOG"
done
cp "$OUT_BASE"/x3_image_aux_lora_*.json "$LOCAL_ART_DIR"/
cp "$OUT_BASE"/x3_image_aux_lora_*.csv "$LOCAL_ART_DIR"/
echo "X3 full-val eval done"
