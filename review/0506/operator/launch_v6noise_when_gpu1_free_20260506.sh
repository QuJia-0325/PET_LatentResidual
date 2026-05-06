#!/usr/bin/env bash
set -euo pipefail
REPO=/home/qujiaxiang/project/PET_LatentResidual
PYTHON=/home/qujiaxiang/.conda/envs/rae/bin/python
LOG=review/0506/operator/logs/v6noise_wait_gpu1_20260506.log
cd "$REPO"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
log() { echo "[$(date -Is)] $*"; }
log "V6_NOISE GPU1 watcher started at HEAD $(git rev-parse --short HEAD)"
while true; do
  gpu_line="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '$1 ~ /^ *1$/ {gsub(/ /,"",$2); gsub(/ /,"",$3); print $2, $3}')"
  read -r mem util <<< "${gpu_line:-999999 100}"
  mem_avail_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  if [[ "$mem" -le 500 && "$util" -le 20 && "$mem_avail_kb" -ge 178257920 ]]; then
    log "resource gate passed for GPU1: mem=${mem}MiB util=${util}% MemAvailable=${mem_avail_kb}KB"
    break
  fi
  log "waiting GPU1: mem=${mem}MiB util=${util}% MemAvailable=${mem_avail_kb}KB"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits | grep 'GPU-f0aebf61-7ab8-1357-a063-da24f1642c07' || true
  sleep 600
done
if [[ -d /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V6_NOISE && -n "$(ls -A /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V6_NOISE 2>/dev/null)" ]]; then
  log "ERROR: V6_NOISE output already exists; refusing overwrite"
  exit 4
fi
log "launching V6_NOISE main training on GPU1"
export PYTHON="$PYTHON"
export PYTHONPATH="$REPO"
SANITY_STEPS=160000 GPU=1 bash review/0505/local/scripts/run_v7_v8_sanity.sh V6_NOISE
log "V6_NOISE command finished"
