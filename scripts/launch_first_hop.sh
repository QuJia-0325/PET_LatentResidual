#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config.yaml> [additional train_first_hop args...]" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
CONFIG_PATH=$(python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$1")
shift

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 2
fi

PYTHON_BIN=${PYTHON_BIN:-/home/qujiaxiang/.conda/envs/rae/bin/python}
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python binary not executable: $PYTHON_BIN" >&2
  exit 2
fi

RUN_DIR=$("$PYTHON_BIN" - "$CONFIG_PATH" "$REPO_ROOT" <<'PY'
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1]).resolve()
repo_root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repo_root))
from pet_lr.path_guard import DEFAULT_OUTPUT_ROOT, resolve_data_disk_dir

with config_path.open('r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
run_name = str(cfg.get('run_name', '')).strip()
if not run_name:
    raise SystemExit('config must set explicit run_name so launch_first_hop.sh can resolve a stable run directory')
output_root = resolve_data_disk_dir(cfg.get('output_dir', str(DEFAULT_OUTPUT_ROOT)), arg_name='output_dir')
print(output_root / run_name)
PY
)

mkdir -p "$RUN_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
STDOUT_LOG="$RUN_DIR/train_${STAMP}.stdout.log"
STDERR_LOG="$RUN_DIR/train_${STAMP}.stderr.log"
LAUNCH_META="$RUN_DIR/launch_${STAMP}.txt"
EXIT_META="$RUN_DIR/exit_code_${STAMP}.txt"
ln -sfn "$(basename "$STDOUT_LOG")" "$RUN_DIR/latest_stdout.log"
ln -sfn "$(basename "$STDERR_LOG")" "$RUN_DIR/latest_stderr.log"
ln -sfn "$(basename "$LAUNCH_META")" "$RUN_DIR/latest_launch.txt"

{
  echo "timestamp: $(date --iso-8601=seconds)"
  echo "repo_root: $REPO_ROOT"
  echo "config: $CONFIG_PATH"
  echo "run_dir: $RUN_DIR"
  echo "git_branch: $(git -C "$REPO_ROOT" branch --show-current)"
  echo "git_commit: $(git -C "$REPO_ROOT" rev-parse HEAD)"
  printf 'command:'
  printf ' %q' "$PYTHON_BIN" "$REPO_ROOT/train_first_hop.py" --config "$CONFIG_PATH" "$@"
  printf '\n'
} > "$LAUNCH_META"

echo "[launcher] run_dir: $RUN_DIR"
echo "[launcher] stdout: $STDOUT_LOG"
echo "[launcher] stderr: $STDERR_LOG"
echo "[launcher] meta:   $LAUNCH_META"

set +e
(
  cd "$REPO_ROOT"
  "$PYTHON_BIN" "$REPO_ROOT/train_first_hop.py" --config "$CONFIG_PATH" "$@" \
    > >(tee -a "$STDOUT_LOG") \
    2> >(tee -a "$STDERR_LOG" >&2)
)
RC=$?
set -e

printf '%s\n' "$RC" > "$EXIT_META"
if [[ $RC -eq 0 ]]; then
  echo "[launcher] exit_code=0"
else
  echo "[launcher] exit_code=$RC" >&2
fi
exit $RC
