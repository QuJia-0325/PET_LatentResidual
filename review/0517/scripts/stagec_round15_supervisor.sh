#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/qujiaxiang/project/PET_LatentResidual"
cd "$ROOT"

PY="/home/qujiaxiang/.conda/envs/rae/bin/python"
V13_GPU="${V13_GPU:-0}"
V14_GPU="${V14_GPU:-2}"
SKIP_V14="${SKIP_V14:-0}"
SMOKE_DIR="review/0516/V13_true_image_aux_ablation/smoke_runs"
V13_CFG="review/0516/V13_true_image_aux_ablation/V13_true_image_aux_off.yaml"
V14_CFG="review/0516/V14_true_d_pure/V14_v7_seed1337.yaml"
V13_PID_FILE="review/0516/V13_true_image_aux_ablation/V13_train.pid"
V14_PID_FILE="review/0516/V14_true_d_pure/V14_train.pid"

ts() {
  date -Is
}

log() {
  echo "[$(ts)] $*"
}

latest_file() {
  local pattern="$1"
  ls -1t $pattern 2>/dev/null | head -1 || true
}

gpu_mem_used_mb() {
  local gpu="$1"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$((gpu + 1))p" | awk '{print int($1)}'
}

gpu_mem_free_mb() {
  local gpu="$1"
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n "$((gpu + 1))p" | awk '{print int($1)}'
}

wait_for_pid_exit() {
  local pid="$1"
  while kill -0 "$pid" >/dev/null 2>&1; do
    sleep 30
  done
}

latest_smoke_pid_file="$(latest_file "$SMOKE_DIR"/V13_smoke_*.pid)"
if [[ -z "$latest_smoke_pid_file" ]]; then
  log "FAIL: no V13 smoke pid file found under $SMOKE_DIR"
  exit 1
fi

smoke_base="${latest_smoke_pid_file%.pid}"
smoke_log="${smoke_base}.log"
smoke_yaml="${smoke_base}.yaml"
smoke_pid="$(tr -d '\n' < "$latest_smoke_pid_file")"

if [[ ! -f "$smoke_log" || ! -f "$smoke_yaml" ]]; then
  log "FAIL: smoke artifacts missing: log=$smoke_log yaml=$smoke_yaml"
  exit 1
fi

log "tracking smoke pid=$smoke_pid yaml=$smoke_yaml"
wait_for_pid_exit "$smoke_pid"
log "smoke pid exited; validating outputs"

smoke_run_dir="$("$PY" - <<'PY' "$smoke_yaml"
import sys, yaml
from pathlib import Path
cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
print(Path(cfg["output_dir"]) / cfg["run_name"])
PY
)"
smoke_metrics="$smoke_run_dir/metrics.jsonl"

"$PY" - <<'PY' "$smoke_metrics" "$smoke_log"
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

metrics_path = Path(sys.argv[1])
log_path = Path(sys.argv[2])
if not metrics_path.exists():
    raise SystemExit(f"FAIL: missing metrics.jsonl: {metrics_path}")
if not log_path.exists():
    raise SystemExit(f"FAIL: missing smoke log: {log_path}")

log_text = log_path.read_text(encoding="utf-8", errors="replace")
if re.search(r"Traceback|RuntimeError|CUDA error|OOM", log_text):
    raise SystemExit("FAIL: smoke log contains fatal error markers")

start_line = next((line for line in log_text.splitlines() if line.startswith("[launcher] ")), None)
if start_line is None:
    raise SystemExit("FAIL: missing launcher line in smoke log")
match = re.search(r"start=(\S+)", start_line)
if match is None:
    raise SystemExit("FAIL: launcher line missing ISO start time")
start_dt = datetime.fromisoformat(match.group(1))
elapsed_sec = (datetime.now(start_dt.tzinfo) - start_dt).total_seconds()
if elapsed_sec > 900:
    raise SystemExit(f"FAIL: smoke elapsed {elapsed_sec:.1f}s > 900s")

rows = []
with metrics_path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

train_rows = [r for r in rows if r.get("event") == "train"]
if len(train_rows) < 15:
    raise SystemExit(f"FAIL: only {len(train_rows)} train rows in metrics.jsonl")

bad_img = [r.get("step") for r in train_rows if abs(float(r.get("img", 0.0))) > 1e-12]
if bad_img:
    raise SystemExit(f"FAIL: img != 0 found in train rows, first steps: {bad_img[:5]}")

critical = ["loss", "pair", "roll", "img", "lambda_roll", "lambda_img", "gate_pix", "lambda_hop_0"]
for row in train_rows:
    step = row.get("step")
    for key in critical:
        val = row.get(key)
        if val is None or not math.isfinite(float(val)):
            raise SystemExit(f"FAIL: non-finite {key} at step {step}")

print(f"PASS: smoke validated with {len(train_rows)} train rows, elapsed {elapsed_sec:.1f}s")
PY

v13_ts="$(date +%Y%m%d_%H%M%S)"
v13_session="v13_full_gpu${V13_GPU}_${v13_ts}"
v13_log="review/0516/V13_true_image_aux_ablation/V13_train_${v13_ts}.log"
log "launching V13 full on gpu=$V13_GPU session=$v13_session"
tmux new-session -d -s "$v13_session" "bash -lc 'cd $ROOT && export CUDA_VISIBLE_DEVICES=$V13_GPU && echo \"[launcher] session=$v13_session gpu=$V13_GPU start=\$(date -Is)\" > \"$v13_log\" && exec \"$PY\" train_first_hop.py --config \"$V13_CFG\" >> \"$v13_log\" 2>&1'"
sleep 2
v13_pid="$(ps -eo pid,cmd | awk '/train_first_hop.py --config review\\/0516\\/V13_true_image_aux_ablation\\/V13_true_image_aux_off.yaml/ && !/awk/ {print $1; exit}')"
if [[ -z "$v13_pid" ]]; then
  log "FAIL: could not find V13 full pid after launch"
  exit 1
fi
echo "$v13_pid" > "$V13_PID_FILE"
log "V13 full pid=$v13_pid log=$v13_log"

sleep 300
if ! ps -p "$v13_pid" >/dev/null 2>&1; then
  log "FAIL: V13 full died within 5 min"
  tail -80 "$v13_log" || true
  exit 1
fi
if [[ "$(gpu_mem_used_mb "$V13_GPU")" -le 1000 ]]; then
  log "FAIL: V13 full gpu=$V13_GPU memory usage <= 1000 MB after 5 min"
  exit 1
fi
if grep -qE 'resume from:|loaded step=' "$v13_log"; then
  log "FAIL: V13 full unexpectedly resumed from checkpoint"
  exit 1
fi
log "V13 full launch checks passed"

if [[ "$SKIP_V14" == "1" ]]; then
  log "SKIP_V14=1, stop after launching V13 full"
  exit 0
fi

log "waiting 30 min before V14 launch for staggered IO health check"
for _ in $(seq 1 30); do
  if ! ps -p "$v13_pid" >/dev/null 2>&1; then
    log "FAIL: V13 full died during staggered wait"
    exit 1
  fi
  sleep 60
done

a3_pid_file="$(latest_file review/0517/V18_capacity_only/*.pid)"
if [[ -z "$a3_pid_file" ]]; then
  log "FAIL: no A3 pid file found"
  exit 1
fi
a3_pid="$(tr -d '\n' < "$a3_pid_file")"
if ! ps -p "$a3_pid" >/dev/null 2>&1; then
  log "FAIL: A3 pid not alive at V14 launch gate: pid=$a3_pid file=$a3_pid_file"
  exit 1
fi
if ! ps -p "$v13_pid" >/dev/null 2>&1; then
  log "FAIL: V13 pid not alive at V14 launch gate: pid=$v13_pid"
  exit 1
fi

v14_free_mb="$(gpu_mem_free_mb "$V14_GPU")"
if [[ "$v14_free_mb" -le 12000 ]]; then
  log "FAIL: V14 target gpu=$V14_GPU free memory only ${v14_free_mb}MB"
  exit 1
fi

log "capturing IO health before V14 launch"
iostat_out="$(iostat -x 1 5)"
printf '%s\n' "$iostat_out"
max_util="$(printf '%s\n' "$iostat_out" | awk '
  BEGIN { max = 0 }
  $1 != \"Device\" && $1 != \"avg-cpu:\" && $NF ~ /^[0-9.]+$/ { if ($NF + 0 > max) max = $NF + 0 }
  END { print max + 0 }
')"
if awk "BEGIN { exit !($max_util < 80.0) }"; then
  log "IO health OK: max util=$max_util"
else
  log "FAIL: IO max util too high before V14 launch: $max_util"
  exit 1
fi

v14_ts="$(date +%Y%m%d_%H%M%S)"
v14_session="v14_full_gpu${V14_GPU}_${v14_ts}"
v14_log="review/0516/V14_true_d_pure/V14_train_${v14_ts}.log"
log "launching V14 full on gpu=$V14_GPU session=$v14_session"
tmux new-session -d -s "$v14_session" "bash -lc 'cd $ROOT && export CUDA_VISIBLE_DEVICES=$V14_GPU && echo \"[launcher] session=$v14_session gpu=$V14_GPU start=\$(date -Is)\" > \"$v14_log\" && exec \"$PY\" train_first_hop.py --config \"$V14_CFG\" >> \"$v14_log\" 2>&1'"
sleep 2
v14_pid="$(ps -eo pid,cmd | awk '/train_first_hop.py --config review\\/0516\\/V14_true_d_pure\\/V14_v7_seed1337.yaml/ && !/awk/ {print $1; exit}')"
if [[ -z "$v14_pid" ]]; then
  log "FAIL: could not find V14 full pid after launch"
  exit 1
fi
echo "$v14_pid" > "$V14_PID_FILE"
log "V14 full pid=$v14_pid log=$v14_log"

sleep 300
if ! ps -p "$v14_pid" >/dev/null 2>&1; then
  log "FAIL: V14 full died within 5 min"
  tail -80 "$v14_log" || true
  exit 1
fi
if [[ "$(gpu_mem_used_mb "$V14_GPU")" -le 1000 ]]; then
  log "FAIL: V14 full gpu=$V14_GPU memory usage <= 1000 MB after 5 min"
  exit 1
fi
if grep -qE 'resume from:|loaded step=' "$v14_log"; then
  log "FAIL: V14 full unexpectedly resumed from checkpoint"
  exit 1
fi

v14_run_dir="$("$PY" - <<'PY' "$V14_CFG"
import sys, yaml
from pathlib import Path
cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
print(Path(cfg["output_dir"]) / cfg["run_name"])
PY
)"
v14_cfg_copy="$v14_run_dir/config.yaml"
if [[ ! -f "$v14_cfg_copy" ]]; then
  log "FAIL: V14 run config.yaml missing at $v14_cfg_copy"
  exit 1
fi
"$PY" - <<'PY' "$v14_cfg_copy"
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
assert int(cfg["seed"]) == 1337, cfg["seed"]
assert bool(cfg["training"]["image_aux"]["enabled"]) is True
print("PASS: V14 config.yaml confirms seed=1337 and image_aux.enabled=true")
PY

log "SUCCESS: V13 full and V14 full launched under Round 15 staggered protocol"
