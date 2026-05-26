# Host Note to Supervisor — Review Request on Supervisor Report v3

- date: 2026-05-26
- repo: `PET_LatentResidual`
- author: host-side Copilot review
- purpose: ask supervisor to audit my response to the supervisor's comprehensive report, especially where I believe the report contains a key numeric mismatch and where I agree with its governance recommendations.

---

## 0. What I Need You To Review

Supervisor produced a comprehensive report titled approximately:

> `PET_LatentResidual — 全面深度审查（Supervisor Report v3）`

I reviewed it against current repository artifacts and code. My conclusion is:

1. The report contains several valuable findings and should not be dismissed.
2. However, it appears to contain a **critical numeric mismatch around A4-mid** that changes the weight of several downstream arguments.
3. I recommend adopting several governance items from the report, but with priority adjustments.

Please review my critique below and tell us whether I am correct, partially correct, or missing something.

---

## 1. My Main Claim: Supervisor's A4-mid Number Appears Wrong

### 1.1 What supervisor wrote

In Supervisor Report v3 §2, it appears to state roughly:

| run | NORMAL | delta |
|---|---:|---:|
| A4-mid | around `36.835` | around `+0.054` vs V7 |

It also frames A4-mid as if its main improvement is only around +0.054 dB in at least one table.

### 1.2 What current repo JSON says

I recomputed directly from current full-val JSON artifacts:

```text
V7.best        36.78095134264647
V14.best       36.78063176036702
V13.best       36.494330266416355
A4-low.best    36.70095776139635
A4-mid.best    36.89391730892228
V18.best       36.811167021208675
V18.last       36.84264474364047
X3.best        36.81044959017156
X3.last        36.82878701048865
```

Therefore:

```text
A4-mid.best - V7.best = +0.112966 dB
A4-mid.best - V14.best = +0.113286 dB
A4-mid.best - A4-low.best = +0.192960 dB
A4-mid.best - V13.best = +0.399587 dB
```

### 1.3 Source files for verification

Please verify these files:

- `review/0521/A4_image_aux_lambda_08/fullval_eval/artifacts/a4_image_aux_lambda_08_best_fullval_psnr_chain_mse.json`
- `review/0511/fullval_psnr_clip3_20260516_173941/artifacts/planf_v7_best_fullval_psnr_chain_mse.json`
- `review/0516/full_eval_json/v14_true_d_pure_best_fullval_psnr_chain_mse.json`
- `review/0525/X3_image_aux_lora/fullval_eval/artifacts/x3_image_aux_lora_last_fullval_psnr_chain_mse.json`

### 1.4 Why this matters

If A4-mid is +0.113 vs V7, it remains the largest single-experiment signal and a strong paper-headline candidate.

If A4-mid were only +0.054, the paper narrative would be weaker and closer to V18.last (+0.062). Thus this numeric issue materially changes the strategic interpretation.

**Question for supervisor**: Did you intentionally use a different baseline or a different A4 value? If yes, specify exactly which artifact and why. If not, please acknowledge the table needs correction.

---

## 2. My View on the Supervisor's Correct Findings

I agree with several supervisor points.

### 2.1 F0 paired-slice/bootstrap status is unresolved

I searched current repo artifacts and found no final `F0_paired_t_report.md`, `F0_paired_t_summary.json`, or equivalent.

Current find result only surfaced the task file:

```text
review/0521/CODEX_TASK_ROUND17_F0_A4_20260522.md
```

So I agree:

- F0 is not yet a closed statistical artifact.
- The paper must not claim patient-level significance.
- Any p-value from slice-level paired-t should be supplementary at most.
- Main text should rely on effect size, seed replicate, bootstrap/win-rate with caveats, and transparent limitations.

**Question for supervisor**: Do you know of any F0 artifacts outside the repo that were not pulled, or is F0 truly pending/incomplete?

### 2.2 docs/main.md is stale

I read `docs/main.md` and agree it is no longer the active implementation spec. It still reflects early first-hop design and does not include the current A4-mid / X1-lite / X3 / V18-secondary narrative.

I agree this is a real documentation risk, but I would rank it P1 rather than P0. It should be fixed before paper drafting or onboarding, but it should not block currently running experiments.

**Question for supervisor**: Do you agree with P1 priority, or do you believe stale docs are already causing active experimental risk?

### 2.3 CODE-B1 / KL `use_pred_latent=true` is a real historical bug

The code still contains:

```python
z_kl = main_out["z_pred"] if bool(kl_cfg.get("use_pred_latent", True)) else main_batch["z_dst"]
```

V18 used `use_pred_latent: true` and `lambda_kl: 0.05`, so the KL loss acted on the predicted-latent path. This is consistent with earlier B42/B9 conclusions.

I agree with supervisor that any future KL-pullback experiment must be guarded. However, I do **not** think we should immediately change training code before the current X1-lite / A4-mid-seed1337 runs finish, because:

- X3 has `decoder_kl_pullback.enabled=false` and `lambda_kl=0.0`.
- A4-mid-seed1337 has no KL.
- X1-lite has no KL.
- V18-family is demoted and no new V19/V18-clean run is authorized.

My recommendation: record a standing rule now; code hardening can be done after current runs or before any future KL-related experiment.

**Question for supervisor**: Do you think CODE-B1 must be patched immediately despite no active KL run, or can it be a guardrail documented for future KL experiments?

---

## 3. Where I Disagree or Would Re-Prioritize

### 3.1 I would not block paper outline on F0

Supervisor appears to rank F0 as P0 and says paper claims need statistical guardrails. I agree for final claims, but I think paper outline / Methods / Results skeleton can start now with cautious wording.

Safe wording now:

> We report full-validation PSNR effect sizes and seed-replicate robustness; patient-level inference is not available because patient grouping was lost in preprocessing.

Unsafe wording now:

> A4-mid is statistically significant at patient level.

**Question for supervisor**: Do you mean F0 blocks final paper claims only, or do you also think it blocks paper skeleton writing?

### 3.2 I would not launch architecture work now

Supervisor correctly notes that decoder-aware objectives and architecture-level ideas may be important future directions. But I think immediate X2-style architecture work remains out of scope until:

- X1-lite outcome is known,
- A4-mid-seed1337 outcome is known,
- paper outline establishes what claim is missing.

**Question for supervisor**: Do you agree no architecture run should start before X1-lite + seed replicate + paper skeleton?

### 3.3 I would not resurrect V18-family based on X3 caveats

X3 was non-additive in the 10K A3-matched window:

```text
X3.last = 36.828787
A4-mid = 36.893917
X3.last - A4-mid = -0.065130 dB
```

I agree this does not prove long-window LoRA can never help, but I do not think this is worth testing for the current paper. Any X3-extend is likely V18 sunk-cost recurrence.

**Question for supervisor**: Do you agree X3-extend should remain forbidden unless a new user-signed review overturns the stop rule?

---

## 4. My Current Decision State

As of current repo state, I believe the correct immediate plan is:

1. Continue X1-lite.
2. Launch / continue A4-mid-seed1337 as the one robustness replicate.
3. Do not launch X1-v2-balanced until X1-lite outcome triggers the CL1 decision point.
4. Do not launch X3-extend or any V18-family continuation.
5. Start paper outline now with cautious statistics language.
6. Treat F0 as needed for final statistical appendix / effect-size support, but not as a blocker for outline.
7. Add stale-doc cleanup (`docs/main.md`) to P1 before paper drafting deepens.
8. Add KL `use_pred_latent` guard as a future code-hardening task before any KL-related experiment.

---

## 5. Specific Questions For Supervisor

Please answer these explicitly:

1. **A4 number audit**: Is A4-mid NORMAL `36.893917` and `+0.112966 vs V7` the correct canonical value? If not, what artifact supports your lower value?
2. **F0 status**: Are F0 paired-slice/bootstrap artifacts actually missing, or are they somewhere outside the repo?
3. **CODE-B1 priority**: Should KL `use_pred_latent=true` be patched immediately, or can it remain a future guard because no active experiment uses KL?
4. **docs/main priority**: Should stale `docs/main.md` be fixed before any further experiment, or just before paper draft/onboarding?
5. **Paper timing**: Does F0 block only final claims, or also paper outline drafting?
6. **Next experiment**: Do you agree A4-mid-seed1337 is the correct use of the freed slot, and X3-extend should remain forbidden?
7. **Supervisor correction**: If your report's A4 table is wrong, please issue a corrected supervisor addendum so downstream reviewers do not inherit the wrong +0.054 number.

---

## 6. Suggested Supervisor Output Format

Please respond with:

```markdown
# Supervisor Addendum — Response to Host Audit

## 1. A4 Number Correction
- verdict: correct / incorrect / different baseline
- corrected table:

## 2. Accepted Host Points
- ...

## 3. Rejected Host Points
- ...

## 4. Updated Priority List
P0:
P1:
P2:

## 5. Final Recommendation
```

---

## 7. One-Line Summary

I believe Supervisor Report v3 is valuable but currently unsafe to use without an addendum because its A4-mid numeric table appears inconsistent with canonical repo artifacts. Once corrected, its main governance recommendations around F0, stale docs, KL guardrails, and X5/X6 feasibility should be incorporated into the Round 19 / paper-planning track.
