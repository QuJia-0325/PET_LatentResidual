#!/usr/bin/env bash
# Memory-light A/B sanity check for sigma-normalize rollout.
# Runs A_sanity_light then B_sanity_light serially on one GPU.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REVIEW_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${REVIEW_DIR}/../.." && pwd)"

PYTHON="${PYTHON:-/home/qujiaxiang/.conda/envs/rae/bin/python}"
TRAIN_ENTRY="${TRAIN_ENTRY:-train_first_hop.py}"
GPU="${GPU:-2}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
ABLATION_INDEX_DIR="${ABLATION_INDEX_DIR:-${REVIEW_DIR}/runs/sanity_light_${RUN_ID}}"
ABLATION_OUTPUT_ROOT="${ABLATION_OUTPUT_ROOT:-/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0502_runs_light/${RUN_ID}}"
MAX_STEPS="${MAX_STEPS:-50000}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

cd "${REPO_ROOT}"
mkdir -p "${ABLATION_INDEX_DIR}"

resolve_yaml() {
    local tag="$1" src="$2"
    local indexdir="${ABLATION_INDEX_DIR}/${tag}"
    local dst="${indexdir}/config.resolved.yaml"
    local out_dir="${ABLATION_OUTPUT_ROOT}/${tag}/run"
    local run_name="first_hop_224_sigma_norm_${tag}_${RUN_ID}"
    mkdir -p "${indexdir}"
    cp "${src}" "${dst}"
    awk -v new_steps="${MAX_STEPS}" -v new_name="${run_name}" -v new_out="${out_dir}" '
        BEGIN { steps_done=0; name_done=0; out_done=0 }
        /^[[:space:]]*max_steps:/ && !steps_done { sub(/max_steps:.*/, "max_steps: " new_steps); steps_done=1 }
        /^run_name:/ && !name_done { sub(/run_name:.*/, "run_name: " new_name); name_done=1 }
        /^output_dir:/ && !out_done { sub(/output_dir:.*/, "output_dir: " new_out); out_done=1 }
        { print }
    ' "${dst}" > "${dst}.tmp" && mv "${dst}.tmp" "${dst}"
    echo "${dst}"
}

run_one() {
    local tag="$1" src="$2"
    local resolved logf
    resolved="$(resolve_yaml "${tag}" "${src}")"
    logf="${ABLATION_INDEX_DIR}/${tag}/train.log"
    echo "============================================================"
    echo "[run_sanity_light] condition=${tag}"
    echo "  config    = ${resolved}"
    echo "  log       = ${logf}"
    echo "  output    = ${ABLATION_OUTPUT_ROOT}/${tag}/run"
    echo "  GPU       = ${GPU}"
    echo "  max_steps = ${MAX_STEPS}"
    echo "============================================================"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${TRAIN_ENTRY}" --config "${resolved}" 2>&1 | tee "${logf}"
}

run_one "A_sanity_light" "${REVIEW_DIR}/configs/A_sanity_light.yaml"
run_one "B_sanity_light" "${REVIEW_DIR}/configs/B_sanity_light.yaml"

echo ""
echo "===== Light sanity comparator ====="
"${PYTHON}" - <<PYEOF
import json, pathlib, sys
base = pathlib.Path('${ABLATION_OUTPUT_ROOT}')
paths = {
    'A': base / 'A_sanity_light/run/first_hop_224_sigma_norm_A_sanity_light_${RUN_ID}/metrics.jsonl',
    'B': base / 'B_sanity_light/run/first_hop_224_sigma_norm_B_sanity_light_${RUN_ID}/metrics.jsonl',
}

def last_eval(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                if obj.get('event') == 'val' or 'val_rollout_total' in obj:
                    rows.append(obj)
    if not rows:
        raise RuntimeError(f'no val rows in {path}')
    return rows[-1]

A = last_eval(paths['A'])
B = last_eval(paths['B'])
keys = ['val_pair_total', 'val_rollout_total']
keys += sorted(k for k in A if k.startswith('val_rollout_step_') and k.endswith('_raw') and k in B)
thresholds = {'val_pair_total': 1e-4, 'val_rollout_total': 0.01}
print(f"{'metric':<32} {'A':>14} {'B':>14} {'rel_err':>10} threshold")
print('-' * 90)
ok = True
for k in keys:
    a = float(A[k]); b = float(B[k])
    rel = abs(b - a) / max(abs(a), 1e-12)
    thr = thresholds.get(k, 0.01)
    flag = rel < thr
    ok = ok and flag
    print(f"{k:<32} {a:>14.6e} {b:>14.6e} {rel:>9.2%} {thr:g} {'PASS' if flag else 'FAIL'}")
print('Result:', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
PYEOF
