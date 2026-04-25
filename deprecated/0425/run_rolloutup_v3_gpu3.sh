#!/usr/bin/env bash
set -euo pipefail
cd /home/qujiaxiang/project/PET_LatentResidual
CUDA_VISIBLE_DEVICES=3 TQDM_DISABLE=1 /home/qujiaxiang/.conda/envs/rae/bin/python -u train_first_hop.py   --config configs/pet_flow/pet_flow_first_hop_224_50k_rollout_up_from_C.yaml   --resume /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_imgaux_boost_gpu2_nopbar_20260418_010035/best.pt
