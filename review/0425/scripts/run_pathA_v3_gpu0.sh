#!/usr/bin/env bash
set -euo pipefail
cd /home/qujiaxiang/project/PET_LatentResidual
PY=/home/qujiaxiang/.conda/envs/rae/bin/python
export PYTHONPATH=/home/qujiaxiang/project/PET_LatentResidual:${PYTHONPATH:-}
SCHEME_C_DIR=/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_imgaux_boost_gpu2_nopbar_20260418_010035
LOG=review/0425/logs_diag/pathA_C_multi_ckpt_gpu0.log
: > "$LOG"
for CKPT in step_010000.pt step_030000.pt best.pt; do
  echo "[START] $(date '+%F %T') $CKPT" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=0 TQDM_DISABLE=1 "$PY" -u scripts/diagnose_tf_rollout_gap.py \
    --config configs/pet_flow/pet_flow_first_hop_224_50k_imgaux_boost.yaml \
    --checkpoint "${SCHEME_C_DIR}/${CKPT}" \
    --split val --max-slices 0 --batch-size 8 --device cuda:0 \
    --out-dir "/data_2/qujiaxiang/outputs/PET_LatentResidual/diagnostics/pathA_C_${CKPT%.pt}" \
    2>&1 | tee -a "$LOG"
  echo "[DONE ] $(date '+%F %T') $CKPT" | tee -a "$LOG"
  echo "" | tee -a "$LOG"
done
