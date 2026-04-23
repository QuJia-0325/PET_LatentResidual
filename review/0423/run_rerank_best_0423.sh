#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=2
export TQDM_DISABLE=1
PY=/home/qujiaxiang/.conda/envs/rae/bin/python
ROOT=/home/qujiaxiang/project/PET_LatentResidual
OUT=/data_2/qujiaxiang/outputs/PET_LatentResidual/clip3_eval/0423_fixed_rerank
cd "$ROOT"

$PY -u eval_first_hop_224_clip3.py \
  --config /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_imgaux_boost_gpu2_nopbar_20260418_010035/config.yaml \
  --checkpoint /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_imgaux_boost_gpu2_nopbar_20260418_010035/best.pt \
  --split val --max-slices 0 --batch-size 8 --device cuda:0 \
  --out-dir "$OUT/c_best_fullval_20260423" |& tee review/0423/logs_eval/c_best_fullval_20260423.log

$PY -u eval_first_hop_224_clip3.py \
  --config /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_pixenc_ablation/config.yaml \
  --checkpoint /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_pixenc_ablation/best.pt \
  --split val --max-slices 0 --batch-size 8 --device cuda:0 \
  --out-dir "$OUT/n1_best_fullval_20260423" |& tee review/0423/logs_eval/n1_best_fullval_20260423.log

$PY -u eval_first_hop_224_clip3.py \
  --config /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_schemec_v2/config.yaml \
  --checkpoint /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_schemec_v2/best.pt \
  --split val --max-slices 0 --batch-size 8 --device cuda:0 \
  --out-dir "$OUT/v2_best_fullval_20260423" |& tee review/0423/logs_eval/v2_best_fullval_20260423.log
