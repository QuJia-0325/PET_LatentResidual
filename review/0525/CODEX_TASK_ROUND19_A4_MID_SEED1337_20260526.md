# Codex Task — Round 19 A4-mid Seed Replicate (seed=1337)

- date: 2026-05-26
- branch: foc_lite_hop0
- status: **READY FOR CODEX EXECUTION**
- upstream: [REVIEW_INTEGRATION_round19_next_slot_after_X3_20260526.md](./REVIEW_INTEGRATION_round19_next_slot_after_X3_20260526.md)
- purpose: use the freed X3 slot for exactly one seed replicate of A4-mid to de-risk the paper headline
- hard constraint: single run only; no seed sweep, no lambda sweep, no V18/X3 continuation

---

## 0. Five-Line Summary for Codex

1. Launch exactly one run: **A4-mid-seed1337**.
2. Clone `review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08.yaml` and change only `output_dir`, `run_name`, and `seed`.
3. Keep `lambda_img=0.08`, full image_aux components, max_steps=160000, LR total_steps_override=200000, no LoRA, no KL, no resume.
4. Use one free GPU, preferably the freed X3 GPU, but explicitly avoid the X1-lite GPU if sidecar exists.
5. After training, run canonical full-val eval for best.pt and last.pt, then write one report comparing A4-mid-seed1337 vs A4-mid, V7, and V14.

---

## 1. Scientific Meaning

A4-mid (`lambda_img=0.08`, seed=42) is currently the paper headline:

| run | NORMAL PSNR_clip3 | delta vs V7 |
|---|---:|---:|
| V7.best | 36.780951 | 0 |
| A4-mid.best | 36.893917 | +0.112966 |

But A4-mid is one seed. V14 showed that V7 at `lambda_img=0.04` is seed-stable (`V14 - V7 ~= -0.0004 dB`), but that does not prove the stronger `lambda_img=0.08` setting is seed-stable.

This run answers: **does the A4-mid headline survive seed=1337?**

Interpretation:
- A4-mid-seed1337 within 0.02 dB of A4-mid: headline robust under seed=1337.
- A4-mid-seed1337 > V7 + 0.05 dB but >0.02 below A4-mid: effect real but seed-sensitive; report conservative range.
- A4-mid-seed1337 near V7: A4-mid was likely seed-lucky; downgrade headline.

---

## 2. Not-Do List

Do not launch:
- any additional seed (seed2024 / seed7 / seed sweep)
- any new lambda point
- X1-v2-balanced before X1-lite outcome
- X3-extend / X3-v2 / V19 / V18-clean / decoder-rank variants
- architecture/data pivots

Do not modify:
- `train_first_hop.py`
- existing V7 / V13 / V14 / A4 / X1 / X3 configs or checkpoints
- image_aux subweights (`l1_weight`, `ssim_weight`, `seam_weight`)

---

## 3. Create YAML

```bash
cd /home/qujiaxiang/project/PET_LatentResidual
git pull --ff-only origin foc_lite_hop0

PYTHON=/home/qujiaxiang/.conda/envs/rae/bin/python
export PYTHONPATH=/home/qujiaxiang/project/PET_LatentResidual:${PYTHONPATH:-}

mkdir -p review/0525/A4_mid_seed1337
cp review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08.yaml \
  review/0525/A4_mid_seed1337/A4_mid_seed1337.yaml

"$PYTHON" <<'PY'
from pathlib import Path
import yaml
p = Path('review/0525/A4_mid_seed1337/A4_mid_seed1337.yaml')
y = yaml.safe_load(p.read_text())
y['output_dir'] = '/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0525_runs/A4_mid_seed1337'
y['run_name'] = 'first_hop_224_a4_mid_seed1337'
y['seed'] = 1337
p.write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True))
print('A4-mid-seed1337 yaml written')
PY

# Verify exact 3-field diff vs A4-mid
"$PYTHON" <<'PY'
import yaml
base = yaml.safe_load(open('review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08.yaml'))
new = yaml.safe_load(open('review/0525/A4_mid_seed1337/A4_mid_seed1337.yaml'))

def flat(d, p=''):
    out = {}
    for k, v in d.items():
        kk = f'{p}.{k}' if p else k
        if isinstance(v, dict): out.update(flat(v, kk))
        else: out[kk] = v
    return out

fb, fn = flat(base), flat(new)
diff = {k: (fb.get(k), fn.get(k)) for k in set(fb) | set(fn) if fb.get(k) != fn.get(k)}
allowed = {'output_dir', 'run_name', 'seed'}
unexpected = set(diff) - allowed
if unexpected:
    print('FAIL unexpected diffs:', unexpected)
    for k, v in sorted(diff.items()): print(k, v)
    raise SystemExit(1)
for k in allowed:
    assert k in diff, f'missing expected diff {k}'
assert new['training']['image_aux']['lambda_start'] == 0.08
assert new['training']['image_aux']['lambda_max'] == 0.08
assert new['loss']['image_aux']['l1_weight'] == 1.0
assert new['loss']['image_aux']['ssim_weight'] == 0.25
assert new['loss']['image_aux']['seam_weight'] == 0.1
assert new['training']['max_steps'] == 160000
assert new['lr_schedule']['total_steps_override'] == 200000
assert 'decoder_lora' not in new['training'] or not bool(new['training'].get('decoder_lora', {}).get('enabled', False))
assert 'decoder_kl_pullback' not in new['loss'] or float(new['loss'].get('decoder_kl_pullback', {}).get('lambda_kl', 0.0)) == 0.0
print('A4-mid-seed1337 exact diff verified:', sorted(diff.keys()))
PY

git add review/0525/A4_mid_seed1337/A4_mid_seed1337.yaml
git commit -m "Round 19 A4-mid seed1337 yaml (exact seed replicate of lambda=0.08)"
```

---

## 4. Launch

```bash
# Exclude X1-lite GPU if sidecar exists
X1_GPU=""
if [ -f review/0525/X1_lite_l1_only/X1_lite.gpu ]; then
  X1_GPU=$(cat review/0525/X1_lite_l1_only/X1_lite.gpu)
fi

if [ -n "$X1_GPU" ]; then
  FREE_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -v exclude=$X1_GPU -F',' '$1+0 != exclude {print $1, $2}' \
    | sort -k2 -rn | head -1 | awk '{print $1}')
else
  FREE_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -rn | head -1 | cut -d, -f1 | tr -d ' ')
fi

[ -z "$FREE_GPU" ] && { echo "FAIL: no free GPU"; exit 1; }
export CUDA_VISIBLE_DEVICES=$FREE_GPU

TS=$(date +%Y%m%d_%H%M%S)
LOG=review/0525/A4_mid_seed1337/A4_mid_seed1337_train_${TS}.log

nohup "$PYTHON" train_first_hop.py \
  --config review/0525/A4_mid_seed1337/A4_mid_seed1337.yaml \
  > "$LOG" 2>&1 &
PID=$!
echo "$PID" > review/0525/A4_mid_seed1337/A4_mid_seed1337.pid
echo "$FREE_GPU" > review/0525/A4_mid_seed1337/A4_mid_seed1337.gpu
echo "A4-mid-seed1337 launched: PID=$PID GPU=$FREE_GPU LOG=$LOG"

sleep 300
ps -p "$PID" > /dev/null || { echo "FAIL: process died"; tail -100 "$LOG"; exit 1; }
grep -q 'lambda_img=0\.0800' "$LOG" || echo "WARN: lambda_img not logged yet"
grep -qE 'OOM|out of memory|NaN' "$LOG" && { echo "FAIL: OOM/NaN in log"; exit 1; }
echo "A4-mid-seed1337 alive at +5min"

git add review/0525/A4_mid_seed1337/A4_mid_seed1337.pid review/0525/A4_mid_seed1337/A4_mid_seed1337.gpu review/0525/A4_mid_seed1337/A4_mid_seed1337_train_*.log
git commit -m "Round 19 A4-mid seed1337 launch log + sidecars"
git push origin foc_lite_hop0
```

---

## 5. Monitor

| time | check | action |
|---|---|---|
| +5 min | process alive, no OOM/NaN, lambda_img=0.0800 if logged | hard fail if process dead/OOM/NaN |
| +1 h | at least one `[val]` line | note only |
| +12 h | step >= 5000 and checkpoint exists | note only |
| +24 h | rolling val recorded | note only; no auto-kill |
| completion | step=160000, Training done | run eval |

No auto-kill on metric quality. This is a seed robustness replicate; even a bad result is scientifically useful.

---

## 6. Full-Val Eval

```bash
OUT=/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0525_runs/A4_mid_seed1337/first_hop_224_a4_mid_seed1337
EVAL_OUT=/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0525_eval/a4_mid_seed1337
REPO_EVAL=review/0525/A4_mid_seed1337/fullval_eval
mkdir -p "$EVAL_OUT" "$REPO_EVAL/artifacts" "$REPO_EVAL/logs"

for ckpt in best last; do
  "$PYTHON" review/0505/operator/scripts/eval_first_hop_fullval_psnr_chain_mse.py \
    --config "$OUT/config.yaml" \
    --checkpoint "$OUT/${ckpt}.pt" \
    --tag "a4_mid_seed1337_${ckpt}" \
    --split val \
    --max-slices 0 \
    --batch-size 8 \
    --decode-mode both \
    --out-dir "$EVAL_OUT" \
    2>&1 | tee "$REPO_EVAL/logs/a4_mid_seed1337_${ckpt}_eval_$(date +%Y%m%d_%H%M%S).log"
done

cp "$EVAL_OUT"/a4_mid_seed1337_*.json "$REPO_EVAL/artifacts/"
cp "$EVAL_OUT"/a4_mid_seed1337_*_per_slice.csv "$REPO_EVAL/artifacts/"
```

---

## 7. Report Template

Write `review/0525/A4_mid_seed1337/A4_MID_SEED1337_REPORT_20260526.md`:

```markdown
# A4-mid Seed=1337 Full-Val Report

| run | seed | NORMAL | delta vs V7 | delta vs A4-mid(seed42) |
|---|---:|---:|---:|---:|
| V7.best | 42 | 36.7810 | 0 | -0.1130 |
| V14.best | 1337 | 36.7806 | -0.0004 | -0.1133 |
| A4-mid.best | 42 | 36.8939 | +0.1130 | 0 |
| A4-mid-seed1337.best | 1337 | ? | ? | ? |
| A4-mid-seed1337.last | 1337 | ? | ? | ? |

## Verdict

- within 0.02 dB of A4-mid: headline robust under seed=1337
- > V7 + 0.05 but >0.02 below A4-mid: effect real but seed-sensitive; report conservative range
- near V7: A4-mid likely seed-lucky; downgrade headline

## Stop Rule

No additional seed replicate. No seed2024/seed7. No lambda sweep. No X3/V18 continuation from this slot.
```

Commit:
```bash
git add review/0525/A4_mid_seed1337/fullval_eval/ review/0525/A4_mid_seed1337/A4_MID_SEED1337_REPORT_20260526.md
git commit -m "Round 19 A4-mid seed1337 canonical full-val eval + report"
git push origin foc_lite_hop0
```

---

## 8. Self-Check

```bash
#!/usr/bin/env bash
set -u
ERR=0
check() { local label="$1"; shift; if "$@"; then echo "PASS: $label"; else echo "FAIL: $label"; ERR=$((ERR+1)); fi; }

check "A4 seed1337 yaml exists" test -f review/0525/A4_mid_seed1337/A4_mid_seed1337.yaml
check "seed=1337" "$PYTHON" -c "import yaml; y=yaml.safe_load(open('review/0525/A4_mid_seed1337/A4_mid_seed1337.yaml')); assert y['seed']==1337"
check "lambda_start=0.08" "$PYTHON" -c "import yaml; y=yaml.safe_load(open('review/0525/A4_mid_seed1337/A4_mid_seed1337.yaml')); assert y['training']['image_aux']['lambda_start']==0.08"
check "lambda_max=0.08" "$PYTHON" -c "import yaml; y=yaml.safe_load(open('review/0525/A4_mid_seed1337/A4_mid_seed1337.yaml')); assert y['training']['image_aux']['lambda_max']==0.08"
check "ssim unchanged" "$PYTHON" -c "import yaml; y=yaml.safe_load(open('review/0525/A4_mid_seed1337/A4_mid_seed1337.yaml')); assert y['loss']['image_aux']['ssim_weight']==0.25"
check "seam unchanged" "$PYTHON" -c "import yaml; y=yaml.safe_load(open('review/0525/A4_mid_seed1337/A4_mid_seed1337.yaml')); assert y['loss']['image_aux']['seam_weight']==0.1"
check "max_steps=160000" "$PYTHON" -c "import yaml; y=yaml.safe_load(open('review/0525/A4_mid_seed1337/A4_mid_seed1337.yaml')); assert y['training']['max_steps']==160000"
check "total_steps_override=200000" "$PYTHON" -c "import yaml; y=yaml.safe_load(open('review/0525/A4_mid_seed1337/A4_mid_seed1337.yaml')); assert y['lr_schedule']['total_steps_override']==200000"

# forbid extra variants
N=$(find review/0525 -name 'A4_mid_seed*.yaml' 2>/dev/null | wc -l | tr -d ' ')
if [ "$N" -gt 1 ]; then echo "FAIL extra A4-mid seed variants found: $N"; ERR=$((ERR+1)); else echo "PASS no extra A4-mid seed variants"; fi

[ $ERR -gt 0 ] && exit 1
echo "SELF-CHECK PASSED"
```
