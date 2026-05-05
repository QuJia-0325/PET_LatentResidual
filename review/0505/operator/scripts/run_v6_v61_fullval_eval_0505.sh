#!/bin/bash
set -euo pipefail
cd /home/qujiaxiang/project/PET_LatentResidual
PY=/home/qujiaxiang/.conda/envs/rae/bin/python
EVAL=review/0505/scripts/eval_first_hop_fullval_psnr_chain_mse.py
OUT=/data_2/qujiaxiang/outputs/PET_LatentResidual/eval_0505_v6_v61_fullval
mkdir -p "$OUT" review/0505/logs_eval

run_v6() {
  export CUDA_VISIBLE_DEVICES=2
  echo "[V6] start $(date)"
  "$PY" -u "$EVAL" \
    --config configs/pet_flow/pet_flow_first_hop_224_v6_transport_first.yaml \
    --checkpoint /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v6_transport_first/best.pt \
    --tag v6_best \
    --split val --max-slices 0 --batch-size 8 --device cuda:0 --decode-mode both \
    --out-dir "$OUT"
  "$PY" -u "$EVAL" \
    --config configs/pet_flow/pet_flow_first_hop_224_v6_transport_first.yaml \
    --checkpoint /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v6_transport_first/last.pt \
    --tag v6_last \
    --split val --max-slices 0 --batch-size 8 --device cuda:0 --decode-mode both \
    --out-dir "$OUT"
  echo "[V6] done $(date)"
}

run_v61() {
  export CUDA_VISIBLE_DEVICES=3
  echo "[V6.1] start $(date)"
  "$PY" -u "$EVAL" \
    --config configs/pet_flow/pet_flow_first_hop_224_v6_1_rollout_floor.yaml \
    --checkpoint /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v6_1_rollout_floor/best.pt \
    --tag v6_1_best \
    --split val --max-slices 0 --batch-size 8 --device cuda:0 --decode-mode both \
    --out-dir "$OUT"
  "$PY" -u "$EVAL" \
    --config configs/pet_flow/pet_flow_first_hop_224_v6_1_rollout_floor.yaml \
    --checkpoint /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v6_1_rollout_floor/last.pt \
    --tag v6_1_last \
    --split val --max-slices 0 --batch-size 8 --device cuda:0 --decode-mode both \
    --out-dir "$OUT"
  echo "[V6.1] done $(date)"
}

case "${1:-}" in
  v6) run_v6 ;;
  v61) run_v61 ;;
  *) echo "Usage: $0 {v6|v61}" >&2; exit 2 ;;
esac
