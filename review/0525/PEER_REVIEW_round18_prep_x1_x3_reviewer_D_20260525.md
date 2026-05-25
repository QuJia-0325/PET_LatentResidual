# PEER REVIEW (Reviewer D) — Round18 Prep X1-lite + X3 Code Review

- date: 2026-05-25
- reviewer: D
- scope: pre-push execution readiness of `CODEX_TASK_ROUND18_X1_X3_20260525.md` (code-grounded)
- verdict: **MODIFY-BEFORE-PUSH**

---

## 1) Candidate Issues (CI-1..CI-5)

### CI-1 GPU race condition
- verdict: **EXTEND**
- finding:
  - Concern is directionally valid, but the bigger concrete risk is script robustness, not only runtime memory race.
  - In §3 B.2, X3 launch gate uses:
    - `X1_PID=$(grep -oE 'PID=[0-9]+' "$X1_LOG" ...)`
    - `ps -p "$X1_PID" ...`
  - `X1_LOG` is trainer stdout; it does not contain the launcher echo `PID=...` line. This can make `X1_PID` empty and fail the health gate before X3 launch.
  - Additionally, GPU selection picks global max-free GPU again without excluding the X1 GPU explicitly.
- evidence:
  - A4/V18 real logs show trainer logs do not include `PID=...` launcher text.
  - A4 smoke startup can take >200s before first train step; this confirms startup phase is long and dynamic.
- severity: **HIGH (script correctness) + MED (allocation race risk)**

### CI-2 LR schedule total_steps_override
- verdict: **VERIFY (with caveat)**
- finding:
  - `total_steps_override=200000` is present in both V18 and A3, so X3 keeping this is correct for matched LR phase.
  - No hidden mismatch found for `decoder_lr_mult`, `decoder_weight_decay`, `freeze_rae` between V18 and A3.
- caveat:
  - X3-vs-A3 “matched-step” claim is slightly weakened by another field mismatch (see Q2): `training.save_interval`.
- severity: **LOW**

### CI-3 early monitoring insufficiency
- verdict: **VERIFY (threshold not yet grounded)**
- finding:
  - Task monitoring table is mostly liveness/step based; no explicit outcome-based stop rule.
  - A4 has concrete step-30000 points (rolling `val_chain_normal_mse ≈ 1.161e-3`, full-val `≈ 9.318e-4`), so trajectory checkpoints are available.
  - Proposed hard threshold `5e-5` vs V7@30k is not currently verifiable from tracked repo artifacts (V7 step-30k metrics file not found in repo paths reviewed).
- severity: **MED**

### CI-4 SSIM/seam with zero weights still computed
- verdict: **VERIFY**
- finding:
  - `compute_first_hop_image_loss` always computes L1/SSIM/seam, then applies weighted sum.
  - With `w_ssim=0`/`w_seam=0`, optimization semantics are still l1-only (grad contributions from these terms are zero), but forward/backward compute overhead remains.
  - No evidence of stateful side effects from these loss functions (pure tensor ops, no BN/dropout state mutation).
- severity: **LOW**

### CI-5 KL enabled=true with lambda_kl=0
- verdict: **EXTEND**
- finding:
  - Trainer KL loss compute is correctly gated by `if lambda_kl > 0.0`, so KL term is not computed.
  - But model init path still checks only `enabled` and will build a frozen RAE reference when `enabled=true`, even if `lambda_kl=0`.
  - This adds unnecessary memory/init overhead and can increase launch fragility.
- severity: **MED**

---

## 2) Deep Questions (Q1..Q9)

### Q1 X1-lite 6-field diff sufficiency
- decision: **APPROVE**
- rationale:
  - Mechanism target is precise: keep A4-mid lambda intensity and disable SSIM/seam only.
  - No required extra field change detected for this intended contrast.

### Q2 X3 6-field diff sufficiency
- decision: **MODIFY**
- rationale:
  - Flattened diff check shows V18->A3 differs by **5** fields (includes `training.save_interval: 10000 -> 5000`), not 4.
  - If X3 is framed as strict A3-protocol-matched except lambda_img, `save_interval` should be aligned or explicitly declared as intentional unmatched field.

### Q3 X1-lite vs A4-mid matchedness
- decision: **APPROVE**
- rationale:
  - Verified V7->A4-mid is exactly 4-field diff (output_dir/run_name/lambda_start/lambda_max).
  - X1-lite design as “A4-mid + disable SSIM/seam” remains a clean mechanism contrast.

### Q4 X3 vs A3 matchedness
- decision: **MODIFY**
- rationale:
  - “Matched” is currently overstated unless `save_interval` alignment is handled.
  - Recommended: either set X3 `save_interval=5000` or explicitly state `save_interval` is the only non-functional logging/checkpoint cadence difference.

### Q5 watchdog kill risk
- decision: **APPROVE (with logging caution)**
- rationale:
  - `loss_balance_watch_enforce=false` in V18/A3/A4-style configs means warn-only, not kill.
  - Warn volume may still affect operator interpretation but not job survival.

### Q6 early-stop protocol design
- decision: **MODIFY**
- rationale:
  - Outcome-based gate should be added, but `5e-5` threshold currently lacks grounded baseline in tracked V7@30k artifacts.
  - Better to use a two-stage guard: rolling check at 30k (soft), full-val check at 50k (hard).

### Q7 10K X3 duration and LoRA marginal underestimation risk
- decision: **MODIFY**
- rationale:
  - Risk is real; 10K may understate LoRA additive effect.
  - Keep 10K for Round18 scope, but pre-register wording that X3 is a short-horizon additive probe, not final upper bound.

### Q8 NOT-DO completeness
- decision: **MODIFY**
- rationale:
  - Add explicit “do not change `loss.image_aux.l1_weight`” for X1-lite.
  - Add explicit “do not infer X3 outcome as V18 rescue.” (already present in spirit; keep as hard guard)
  - Add script-level guard against same-GPU reuse for X3 launch.

### Q9 new bias checks (B93+)
- decision: **APPROVE**
- rationale:
  - Echo-chamber/anchor/time-pressure bias risks exist and are correctly identified.

---

## 3) New Issues Found (beyond CI list)

### N1 (HIGH) — B.2 health-gate PID extraction is brittle and can fail launch
- location: §3 B.2 launch snippet
- issue: reads PID from trainer log where PID line does not exist.
- impact: false failure before X3 launch.

### N2 (MED) — matched-comparison wording inconsistent with actual diff count
- location: §1.3, §3 B.0, and related “matched-step paired” statements
- issue: V18->A3 practical diff includes `save_interval`.
- impact: comparability narrative becomes vulnerable in review.

### N3 (MED) — second launch GPU selection does not exclude X1 GPU
- location: §3 B.2
- impact: possible same-GPU co-location/OOM risk under transient free-memory ordering.

---

## 4) Main Verdict

- decision: **MODIFY-BEFORE-PUSH**
- reason:
  - At least one pre-execution blocker exists (N1 high: PID gate bug).
  - Additional medium risks (N2/N3/CI-5 overhead) should be patched now to avoid preventable restart/interpretation churn.

---

## 5) Must-Fix List (before push)

| location | original wording / behavior | required fix | severity |
|---|---|---|---|
| §3 B.2 health check | Parse `X1_PID` from `X1_LOG` using `grep 'PID='` | Persist PID to a sidecar file at X1 launch (e.g., `X1_lite.pid`) and read from it; fail fast if missing | HIGH |
| §3 B.2 GPU selection | Select global max-free GPU again | Exclude X1 GPU explicitly (`FREE_GPU2 != FREE_GPU`) or abort if only one viable GPU | MED |
| X3 matched protocol claim | Treat X3-vs-A3 as strict paired while leaving cadence mismatch implicit | Either set X3 `training.save_interval=5000` to match A3, or explicitly state this is a non-functional unmatched field | MED |
| X3 KL config | Keep `decoder_kl_pullback.enabled=true` with `lambda_kl=0` | Set `enabled=false` for X3 to avoid unnecessary frozen-RAE construction overhead | MED |
| §2/§3 monitoring | Liveness-only progress checks | Add explicit outcome guard: rolling checkpoint at 30k (soft) + full-val checkpoint at 50k (hard) with predeclared stop action | MED |

---

## 6) D-Class Design Concerns

1. X1-lite intermediate outcome band (between V7 and A4-mid) should be pre-mapped to manuscript phrasing now to avoid post-hoc interpretation.
2. X3 short-horizon (10K) limitation must be explicitly disclosed as an exploratory additive probe, not final LoRA ceiling.
3. Single-seed constraints remain; avoid overclaiming mechanism certainty from one run per arm.

---

## 7) Bias Check (B93+)

- claude self-review anchoring risk: **present**
- design-perfect anchor risk: **present**
- time-pressure bias risk: **present**
- mitigation for this round: keep strict pre-push script correctness and comparability wording fixes; defer strategic expansion decisions to Round19.
