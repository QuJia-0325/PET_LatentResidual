# REVIEW INTEGRATION — Round 18-Prep (X1-lite + X3 Code Review)

- date: 2026-05-25
- scope: integrate 3 independent Round 18-Prep deep code reviews on `CODEX_TASK_ROUND18_X1_X3_20260525.md`
- sources:
  - PEER_REVIEW_round18_prep_x1_x3_code_review_copilot_20260525.md (GitHub Copilot, with --resume HIGH catch)
  - PEER_REVIEW_ROUND18_PREP_X1_X3_REVIEWER_B_20260525.md (Reviewer B)
  - PEER_REVIEW_round18_prep_x1_x3_reviewer_D_20260525.md (Reviewer D)

---

## 1) Unanimous verdict

**3/3 reviewers: MODIFY-BEFORE-PUSH.** Strategic direction (X1-lite + X3 from Round 18 4/4 consensus) is unaffected. The blockers are all in `CODEX_TASK_ROUND18_X1_X3_20260525.md` execution scripts and a few yaml field handling decisions.

**Critical finding**: claude self-review (CI-1 ~ CI-5) **missed 2 HIGH bugs** that reviewers caught. This is B93-class "self-review tunnel vision": claude verified what it thought to check (yaml fields, trainer code paths) but missed the launcher script semantics that codex actually executes.

---

## 2) HIGH blockers (must fix before push)

### H1 — X3 launch command omits `--resume` ⚠️ NEW (Copilot found, code-verified)

**Location**: §3 B.2 — `nohup "$PYTHON" train_first_hop.py --config ... > "$X3_LOG" 2>&1 &`

**Verified bug**: `train_first_hop.py:1381` reads `resume_path = str(args.resume).strip()`. **YAML `training.resume_from` is never consumed by the launcher**. Without CLI `--resume`, `resume_enabled=False`, X3 trains **from scratch** to 170K with LoRA + lambda_img=0.08.

**Cross-check**: A3 task md (`CODEX_TASK_STAGE_C_A3_V18_CAPACITY_ONLY_20260518.md:114`) correctly passed `--resume /data_2/.../V7/.../best.pt`. A3 log line 4 confirms `[startup] resume from:` and line 70 confirms `[resume] loaded step=160000`. X3 missing this destroys A3-matched protocol.

**Severity**: HIGH. If unfixed, X3 burns 1-2d GPU producing a result with no scientific meaning.

**Fix**:
```bash
nohup "$PYTHON" train_first_hop.py \
  --config review/0525/X3_image_aux_lora/X3_image_aux_lora.yaml \
  --resume /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V7/run/first_hop_224_v7_gronwall_raw/best.pt \
  > "$X3_LOG" 2>&1 &
```

Plus add resume verification to health check:
```bash
grep -q '\[startup\] resume from:' "$X3_LOG" || { echo "FAIL X3 not resumed"; exit 1; }
grep -q '\[resume\] loaded step=160000' "$X3_LOG" || { echo "FAIL X3 wrong resume step"; exit 1; }
```

### H2 — X1 PID extraction reads from wrong source ⚠️ NEW (3/3 reviewers found)

**Location**: §3 B.2 — `X1_PID=$(grep -oE 'PID=[0-9]+' "$X1_LOG" | head -1 | cut -d= -f2)`

**Verified bug**: §A.1 prints `echo "X1-lite launched: PID=$X1_PID GPU=..."` to **shell stdout**, not to `$X1_LOG` (which is trainer stdout via `nohup > $X1_LOG`). A3/V18 real logs confirm trainer logs do not contain `PID=` launcher echo. → X1_PID will be empty in fresh codex step → `ps -p ""` fails → X3 launch aborts even when X1 is healthy.

**Severity**: HIGH. Aborts entire X3 launch path.

**Fix**: persist PID via sidecar file:
```bash
# §A.1 add after launch:
echo "$X1_PID" > review/0525/X1_lite_l1_only/X1_lite.pid
echo "$FREE_GPU" > review/0525/X1_lite_l1_only/X1_lite.gpu

# §B.2 read sidecars instead of grep:
X1_PID=$(cat review/0525/X1_lite_l1_only/X1_lite.pid)
X1_GPU=$(cat review/0525/X1_lite_l1_only/X1_lite.gpu)
ps -p "$X1_PID" > /dev/null || { echo "FAIL X1 dead"; exit 1; }
```

### H3 — GPU selection doesn't explicitly exclude X1's GPU (3/3 confirmed)

**Location**: §3 B.2 — `FREE_GPU2=$(nvidia-smi ... | sort -k2 -rn | head -1)`

**Issue**: Sorting by global free memory can pick X1's GPU during dataloader loading transient. My self-review CI-1 framed this right but didn't fix.

**Severity**: HIGH (3/3 reviewers agree it's avoidable).

**Fix**:
```bash
X1_GPU=$(cat review/0525/X1_lite_l1_only/X1_lite.gpu)
FREE_GPU2=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
  | awk -v exclude=$X1_GPU -F',' '$1+0 != exclude {print $1, $2}' \
  | sort -k2 -rn | head -1 | awk '{print $1}')
[ -z "$FREE_GPU2" ] && { echo "FAIL no second free GPU (excluded X1's GPU $X1_GPU)"; exit 1; }
[ "$FREE_GPU2" = "$X1_GPU" ] && { echo "FAIL GPU race: chose same GPU as X1"; exit 1; }
export CUDA_VISIBLE_DEVICES=$FREE_GPU2
```

### H4 — Self-check anti-check pipeline does not fail reliably (Copilot found)

**Location**: §6 self-check — `anti_check "..." bash -c "[[ $(...) -gt N ]]"`

**Issue**: bash -c with command substitution in subshell loses exit code propagation. anti_check function's failure detection unreliable.

**Severity**: HIGH (silent self-check passing while real condition fails).

**Fix**: replace `bash -c` pattern with direct test in shell:
```bash
anti_check_count() {
    local label="$1"; local pattern="$2"; local maxn="$3"
    local n=$(find review/0525 -name "$pattern" 2>/dev/null | wc -l)
    if [ "$n" -gt "$maxn" ]; then
        echo "FAIL: $label (found $n, max $maxn)"; ERR=$((ERR+1))
    else
        echo "PASS: $label (found $n)"
    fi
}
anti_check_count "no extra A4 variant" 'A4_image_aux_lambda_*.yaml' 2
```

---

## 3) MED issues (recommended fix)

### M1 — X3 `decoder_kl_pullback.enabled: true` left as-is (3/3 confirmed)

**Issue**: `pet_lr/model_first_hop.py:286-324` shows model construction reads `enabled` to decide whether to build the frozen RAE reference (extra VRAM + module copy). With `lambda_kl=0.0` the loss is skipped (line 2229 `if lambda_kl > 0.0:`) but the frozen RAE is still constructed.

**Fix**: add 7th yaml diff to X3:
```yaml
loss.decoder_kl_pullback.enabled: true → false
```

Save VRAM, cleaner semantics, doesn't change paired comparison to A3 (A3 has `enabled: true + lambda_kl=0`, but since X3's only declared variable vs A3 is image_aux λ, also disabling enabled is harmless cleanup).

### M2 — X3 vs A3 "matched" claim has a save_interval mismatch (Reviewer D found)

**Issue**: A3 changed V18's `training.save_interval: 10000 → 5000` to produce `step_165000.pt` for matched-step comparison. claude's X3 task md keeps V18's default 10000. So X3 vs A3 "differs only in image_aux λ" is **not strictly true**: also differs in save cadence.

**Severity**: MED. Functionally doesn't change training, only checkpoint emission cadence. But if comparison protocol expects `step_165000.pt` from X3, it won't exist.

**Fix**: add 8th yaml diff:
```yaml
training.save_interval: 10000 → 5000
```

### M3 — X1-lite `img_ssim` health check expects 0 but SSIM still computed (Copilot found)

**Location**: §A.1 — `grep -E 'img_ssim=[0-9.]+' "$X1_LOG" | head -3` (comment says "应该为 0 或非常小")

**Issue**: `pet_lr/losses_first_hop.py` always computes `loss_ssim` regardless of `w_ssim`. Only `total = w_l1*l1 + w_ssim*ssim + w_seam*seam` is weighted. The `img_ssim` log field shows raw SSIM loss (nonzero). User would see normal SSIM values and think the ablation failed.

**Fix**: change health check to compare `img_total ≈ img_l1` (within numerical tolerance) instead of expecting `img_ssim ≈ 0`:
```bash
# X1-lite verify: img total should ≈ img_l1 (since ssim/seam weights = 0)
grep -E 'img=[0-9.eE+-]+ img_l1=[0-9.eE+-]+' "$X1_LOG" | head -3
# Or python check: parse one train line, assert abs(img - img_l1) < 1e-6
```

### M4 — 30K early-stop threshold not grounded (3/3 confirmed)

**Issue**: My self-review CI-3 proposed "+24h step 30K, diff > 5e-5 NORMAL MSE → early stop". Reviewers verified: V7 and A4-mid at step 30K rolling val NORMAL MSE are essentially identical (~0.000930 vs ~0.000932 — within noise). So 5e-5 threshold would either never trigger (too loose) or auto-kill A4-mid (too tight, false positive).

**Fix**: remove the auto-kill protocol from monitor table. Replace with **diagnostic-only**:
```
| +24h (step ~30K) | log rolling val_chain_normal_mse + img_frac; if val_chain_normal_mse > 0.0015 (50% above V7 baseline) → manual user review, do NOT auto-kill |
| +48h (step ~50K) | first full-val available; compare full-val NORMAL MSE to V7 step-50K baseline; if > +10% → manual user review |
```

No auto-kill on rolling val (too noisy). Auto-kill only on hard infrastructure errors (OOM/NaN/process dead).

---

## 4) LOW / accepted (not blocking)

### L1 — CI-2 LR `total_steps_override=200000` with `max_steps=170000`

Verified by all 3 reviewers: this matches V18+A3 protocol (both used same configuration successfully). X3 inherits and is correct.

### L2 — CI-4 SSIM/seam compute when w=0

3/3 reviewers verified: gradient correctly zeroed (multiplied by 0), wall-clock overhead 2-5% (negligible), no side effects (no BN/dropout state mutation in SSIM/seam computations). Accept as-is.

### L3 — loss_balance_watch fires warn but not kill

3/3 reviewers verified: `loss_balance_watch_enforce: false` in V18 yaml (inherited by X3). Watchdog logs warn but doesn't kill. Accept.

---

## 5) Design-level concerns (reviewer-flagged, not blocking)

### D1 — X3 10K short-training cannot test V18 LoRA at 40K-training depth

3/3 reviewers note: X3 mirrors A3 protocol (10K LoRA training from V7.best). A3 had LoRA contribute ~0 vs V18.step170k. So X3 outcome ≈ A4-mid is the **expected** result if LoRA needs 40K training to bind (like V18.last vs V18.best).

**Recommended paper language** (Reviewer D suggests):
> "X3 measures additive effect within the A3-matched 10K LoRA-adaptation window. If X3 ≤ A4-mid, this rules out short-window additivity but does NOT rule out long-training (40K+) LoRA additivity. Long-training cannot be tested within this paper's compute budget."

This is honesty boilerplate, not an experiment change.

### D2 — X1-lite outcome ambiguity (l1-only between V7 and A4-mid)

If X1-lite NORMAL ≈ 36.83 (interior of V7-A4-mid range), paper needs to attribute partial contributions to L1 vs SSIM/seam. Round 18 stop rule disallows X1 v2 (ssim-only).

**Recommended Round 19 trigger**: if X1-lite lands in interior, **explicitly request user permission** for X1-v2 instead of auto-launching. Document in codex task md NOT-DO #11 update.

---

## 6) New biases B93-B95 (consolidated)

| ID | bias | severity | source |
|---|---|---|---|
| B93 | self-review tunnel vision (claude verifies yaml fields, misses launcher script semantics) | HIGH | Copilot + Reviewer D |
| B94 | matched-comparison overclaim (X3 vs A3 framed as 1-variable but has multiple diffs) | MED | Reviewer D |
| B95 | health-check theater (grep patterns that look thorough but read wrong source) | MED | 3/3 |

**Root cause**: claude wrote task md describing **intent** but did not simulate codex execution path step-by-step. Each launcher script line should be mentally executed against actual A3/V18/A4 log output before commit.

---

## 7) Final integrated verdict

**MODIFY-BEFORE-PUSH** with 4 HIGH + 4 MED fixes.

Fix sequence:
1. **H1** (add --resume to X3 launch)
2. **H2** (PID sidecar)
3. **H3** (GPU exclusion via sidecar)
4. **H4** (anti-check pipeline)
5. **M1** (enabled=false in X3 yaml — change 6-field diff to 7-field)
6. **M2** (save_interval=5000 in X3 yaml — change 7-field to 8-field)
7. **M3** (img_ssim health check fix)
8. **M4** (remove auto-kill at step 30K; replace with diagnostic-only)

Total estimated fix time: ~20 minutes mechanical edits, no design rework. After fix, push to codex.

---

## 8) Next steps

1. claude immediately applies 4 HIGH + 4 MED fixes to `CODEX_TASK_ROUND18_X1_X3_20260525.md`.
2. claude verifies fixes via grep + re-run yaml diff verification.
3. Commit + push to gitee.
4. codex pulls and executes (no additional review round; H1-H4 are mechanical, fixes don't change Round 18 strategy).
5. Round 19 trigger condition: ≥ 2 of {X1-lite full-val eval done, X3 full-val eval done, paper outline skeleton complete} — but in case X1-lite lands in interior (D2), trigger Round 19 immediately to discuss X1-v2 permission.
