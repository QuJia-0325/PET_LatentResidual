#!/usr/bin/env bash
set -euo pipefail

REPO=/home/qujiaxiang/project/PET_LatentResidual
PYTHON=/home/qujiaxiang/.conda/envs/rae/bin/python
cd "$REPO"

LOG_DIR=review/0506/operator/logs
mkdir -p "$LOG_DIR" review/0506/operator
SUP_LOG="$LOG_DIR/planf_supervisor_20260506.log"
DECISION=review/0506/operator/preflight_decision_20260506.md
MAIN_LAUNCH=review/0506/operator/main_train_launch_20260506.md

log() { echo "[$(date -Is)] $*" | tee -a "$SUP_LOG"; }
fail_decision() {
  local msg="$1"
  log "BLOCK: $msg"
  cat > "$DECISION" <<EOF
# Plan F Phase 1 Pre-flight Decision (2026-05-06)

Decision: **BLOCKED**

Reason: ${msg}

See log: \\`${SUP_LOG}\\`
EOF
}

latest_val_line() {
  local f="$1"
  grep '^\[val\] step=' "$f" | tail -1 || true
}

field_float() {
  local line="$1" key="$2"
  python - "$line" "$key" <<'PY'
import re, sys
line, key = sys.argv[1], sys.argv[2]
m = re.search(rf'(?:^| ){re.escape(key)}=([-+0-9.eE]+)', line)
if not m:
    raise SystemExit(1)
print(m.group(1))
PY
}

wait_session_done() {
  local session="$1"
  while tmux has-session -t "$session" 2>/dev/null; do
    log "waiting tmux session ${session} ..."
    sleep 300
  done
  log "tmux session ${session} finished"
}

log "supervisor started at HEAD $(git rev-parse --short HEAD)"
log "waiting for pre-flight sessions pf_v6_0506 and pf_v7_0506"
wait_session_done pf_v6_0506
wait_session_done pf_v7_0506

# Gate artifacts.
if [[ ! -f review/0506/operator/sigma_norm_golden_test_result.json ]]; then
  fail_decision "missing sigma_norm_golden_test_result.json"
  exit 10
fi
if [[ ! -f review/0506/operator/v8_rng_invariance_result.json ]]; then
  fail_decision "missing v8_rng_invariance_result.json"
  exit 11
fi

SIG_VERDICT="$($PYTHON - <<'PY'
import json
p='review/0506/operator/sigma_norm_golden_test_result.json'
print(json.load(open(p)).get('verdict',''))
PY
)"
V8_RNG="$($PYTHON - <<'PY'
import json
p='review/0506/operator/v8_rng_invariance_result.json'
print(str(json.load(open(p)).get('rng_byte_identical', False)).lower())
PY
)"
log "gate results: sigma=${SIG_VERDICT}, v8_rng_byte_identical=${V8_RNG}"
if [[ "$SIG_VERDICT" != "PASS" ]]; then
  fail_decision "Option F sigma golden test is ${SIG_VERDICT}; do not launch V7"
  exit 12
fi
if [[ "$V8_RNG" != "true" ]]; then
  fail_decision "V8 RNG invariance failed; do not launch V8"
  exit 13
fi

# Check all 5K pass logs finished.
declare -A LOGS=(
  [v6p1]=review/0506/local/logs/V6_5K_pass1_gpu2.log
  [v6p2]=review/0506/local/logs/V6_5K_pass2_gpu2.log
  [v7p1]=review/0506/local/logs/V7_5K_pass1_gpu3.log
  [v7p2]=review/0506/local/logs/V7_5K_pass2_gpu3.log
)
for k in "${!LOGS[@]}"; do
  f="${LOGS[$k]}"
  if [[ ! -f "$f" ]]; then
    fail_decision "missing preflight log: $f"
    exit 20
  fi
  if ! grep -q '===== .* pass[12]: done =====' "$f"; then
    fail_decision "preflight log does not show normal completion: $f"
    exit 21
  fi
  latest_line="$(latest_val_line "$f")"
  if [[ -z "$latest_line" ]]; then
    fail_decision "no [val] line found in preflight log: $f"
    exit 22
  fi
  log "${k} latest val: ${latest_line}"
done

v6p1_line="$(latest_val_line "${LOGS[v6p1]}")"
v6p2_line="$(latest_val_line "${LOGS[v6p2]}")"
v7p1_line="$(latest_val_line "${LOGS[v7p1]}")"
v7p2_line="$(latest_val_line "${LOGS[v7p2]}")"

v6p1_pair="$(field_float "$v6p1_line" val_pair_total)"
v6p2_pair="$(field_float "$v6p2_line" val_pair_total)"
v7p1_pair="$(field_float "$v7p1_line" val_pair_total)"
v7p2_pair="$(field_float "$v7p2_line" val_pair_total)"
v6p1_chain="$(field_float "$v6p1_line" val_chain_normal_mse)"
v6p2_chain="$(field_float "$v6p2_line" val_chain_normal_mse)"
v7p1_chain="$(field_float "$v7p1_line" val_chain_normal_mse)"
v7p2_chain="$(field_float "$v7p2_line" val_chain_normal_mse)"

read -r drift_v6 drift_v7 drift_v6_chain drift_v7_chain verdict <<< "$($PYTHON - <<PY
v6p1=float('$v6p1_pair'); v6p2=float('$v6p2_pair')
v7p1=float('$v7p1_pair'); v7p2=float('$v7p2_pair')
v6c1=float('$v6p1_chain'); v6c2=float('$v6p2_chain')
v7c1=float('$v7p1_chain'); v7c2=float('$v7p2_chain')
dv6=abs(v6p2-v6p1)/max(abs(v6p1),1e-12)
dv7=abs(v7p2-v7p1)/max(abs(v7p1),1e-12)
dc6=abs(v6c2-v6c1)/max(abs(v6c1),1e-12)
dc7=abs(v7c2-v7c1)/max(abs(v7c1),1e-12)
verdict='PASS' if (dv6 <= 0.01 and dv7 <= dv6) else 'BLOCKED'
print(dv6, dv7, dc6, dc7, verdict)
PY
)"

cat > "$DECISION" <<EOF
# Plan F Phase 1 Pre-flight Decision (2026-05-06)

Decision: **${verdict}**

## Gate Results

- Option F sigma golden test: ${SIG_VERDICT}
- V8 RNG byte-identical: ${V8_RNG}

## 5K Double-pass Metrics

| arm/pass | val_pair_total | val_chain_normal_mse | latest val line |
|---|---:|---:|---|
| V6 pass1 | ${v6p1_pair} | ${v6p1_chain} | \\`${v6p1_line}\\` |
| V6 pass2 | ${v6p2_pair} | ${v6p2_chain} | \\`${v6p2_line}\\` |
| V7 pass1 | ${v7p1_pair} | ${v7p1_chain} | \\`${v7p1_line}\\` |
| V7 pass2 | ${v7p2_pair} | ${v7p2_chain} | \\`${v7p2_line}\\` |

## Drift

- drift_v6_pair = ${drift_v6}
- drift_v7_pair = ${drift_v7}
- drift_v6_chain_normal = ${drift_v6_chain}
- drift_v7_chain_normal = ${drift_v7_chain}

## Decision Rule

Proceed iff Option F PASS, V8 RNG true, drift_v6_pair <= 1.0%, and drift_v7_pair <= drift_v6_pair.

Supervisor log: \\`${SUP_LOG}\\`
EOF

log "preflight decision=${verdict}; drift_v6=${drift_v6}; drift_v7=${drift_v7}"
if [[ "$verdict" != "PASS" ]]; then
  log "preflight blocked; main training will not launch"
  exit 30
fi

# Wait for GPU1/2/3 and RAM before main launch.
log "waiting for GPU1/2/3 and RAM gates before main training"
while true; do
  gpu_csv="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)"
  mem_avail_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  ok_gpu="$($PYTHON - <<PY
csv='''${gpu_csv}'''
ok=True
bad=[]
for line in csv.strip().splitlines():
    idx, mem, util = [x.strip() for x in line.split(',')]
    if idx in {'1','2','3'}:
        if int(mem) > 500 or int(util) > 20:
            ok=False; bad.append(f'GPU{idx}:mem={mem}MiB,util={util}%')
print('OK' if ok else '; '.join(bad))
PY
)"
  # Conservative: 420 GiB available for three loader processes. This avoids starting
  # main training while old GPU0/GPU1 jobs or page-cache pressure can force swap/OOM.
  if [[ "$ok_gpu" == "OK" && "$mem_avail_kb" -ge 440401920 ]]; then
    log "resource gate passed: GPUs 1/2/3 free, MemAvailable=${mem_avail_kb}KB"
    break
  fi
  log "resource gate waiting: gpu=${ok_gpu}; MemAvailable=${mem_avail_kb}KB"
  sleep 600
done

# Refuse overwrite unless operator explicitly removed old dirs.
for d in \
  /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V7 \
  /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V8 \
  /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V6_NOISE; do
  if [[ -d "$d" && -n "$(ls -A "$d" 2>/dev/null)" ]]; then
    fail_decision "main output already exists: $d"
    exit 40
  fi
done

if tmux has-session -t planf_v7v8_0506 2>/dev/null || tmux has-session -t planf_v6noise_0506 2>/dev/null; then
  fail_decision "main tmux session already exists"
  exit 41
fi

log "launching Phase 2 main training: V7@GPU2, V8@GPU3, V6_NOISE@GPU1"
tmux new-session -d -s planf_v7v8_0506 "cd $REPO && export PYTHON=$PYTHON && export PYTHONPATH=$REPO && SANITY_STEPS=160000 GPU_V7=2 GPU_V8=3 bash review/0505/local/scripts/run_v7_v8_sanity.sh parallel"
tmux new-session -d -s planf_v6noise_0506 "cd $REPO && export PYTHON=$PYTHON && export PYTHONPATH=$REPO && SANITY_STEPS=160000 GPU=1 bash review/0505/local/scripts/run_v7_v8_sanity.sh V6_NOISE"
cat > "$MAIN_LAUNCH" <<EOF
# Plan F Phase 2 Main Training Launch (2026-05-06)

Launched by supervisor after pre-flight PASS.

- V7: GPU2 via tmux \\`planf_v7v8_0506\\`
- V8: GPU3 via tmux \\`planf_v7v8_0506\\`
- V6_NOISE: GPU1 via tmux \\`planf_v6noise_0506\\`
- Steps: 160000 per arm
- Python: \\`${PYTHON}\\`
- Script: \\`review/0505/local/scripts/run_v7_v8_sanity.sh\\`

Logs:

- \\`review/0505/local/runs/V7/train.log\\`
- \\`review/0505/local/runs/V8/train.log\\`
- \\`review/0505/local/runs/V6_NOISE/train.log\\`

Supervisor log: \\`${SUP_LOG}\\`
EOF
log "main training launched"
