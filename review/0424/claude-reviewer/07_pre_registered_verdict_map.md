# Pre-Registered Verdict Map — How I Will Score the Incoming Data

**Date**: 2026-04-24  
**Author**: Nightmare Panel (synthesized), acting as the reviewer  
**Purpose**: Fix the reviewer's scoring rules **before** Proposer's new data lands, so Round 7 is deterministic (data-driven, not argument-driven).  
**Companion to**: `06_reasoning_dossier_on_rebuttal.md`, `../response_to_reviewer/reviewer_signoff_r6.md`

---

## 0. Why this document exists

After 6 rounds the review has produced ~2500 lines of argumentative text. If I wait for Proposer's data and then write another ad-hoc review, we waste everything we learned about anchoring bias and shifting goalposts. This document **pre-registers**:

1. Exactly which numerical result triggers which score change.
2. Which null/positive outcomes I already commit to accepting honestly (both ways).
3. The precise pass/fail gates for each instrument (σ_seed-lite, Path A, Path C, Pix2Pix).
4. The pre-paid rebuttals I have to honor if the data lands in specific configurations.

If I deviate from this document in Round 7, Proposer should cite this file and push back.

---

## 1. The Six Artifacts That Will Close The Case

| Artifact | Source | Blocking | Max score contribution |
|---|---|---|---:|
| **A1** σ_seed-lite result | 4 seeds × imgaux_boost × 50K on fixed codebase | Everything | +0.3 to Causal |
| **A2** Path A verdict | `diagnose_tf_rollout_gap.py` on C + N1 full-val | Story + mechanism | +0.8 to Causal |
| **A3** Path C diagnostic | Decoder-FT 5K steps, last 2 upsample blocks | RAE bound narrative | +0.5 to Novelty, +0.3 to Story |
| **A4** F2 rewrite applied | CLAUDE.md + IDEA_REPORT.md language change | Writing hygiene | +0.3 to Story |
| **A5** Pix2Pix baseline | Same val split, same PSNR metric | External calibration | +1.0 to Submission |
| **A6** 200K decision record | Either abandon with σ-unit justification, or run with σ-unit stop criterion | Process discipline | ±0.5 to Causal |

Anything else Proposer submits is bonus and does not affect the score.

---

## 2. Pre-Registered Decision Tree on σ_seed-lite (A1)

Input: 4 PSNR numbers for `transport_avg` on full-val (n=7403) under identical code, identical config, identical data, seeds ∈ {42, 123, 456, 789}.

```
σ_hat = std(4 PSNRs)
σ_upper_80 ≈ 1.5 × σ_hat   (conservative operating bound, per dossier §2.4)
mean_C    = mean(4 PSNRs)
```

| Condition | Score delta | Unlocks |
|---|:---:|---|
| σ_hat ≤ 0.04 dB | +0.3 Causal | Can attempt positive-result story; 200K allowed with strict σ-unit gate |
| 0.04 < σ_hat ≤ 0.08 | +0.2 Causal | Negative-result reframe required; 200K allowed only if stop criterion uses σ_upper_80 |
| σ_hat > 0.08 | +0.1 Causal | 200K must abandon (effect unmeasurable); write pure null paper |
| σ cannot be computed (only 3 seeds, mixed codebase, or incomplete) | −0.3 Causal | Reviewer revoke of R6 acceptance; back to R5 scoring |

**Pre-paid concession**: if σ_hat > 0.08 dB and Proposer correctly kills 200K, I will add **+0.1 to Story** for methodological discipline.

**Pre-paid concession 2**: if Proposer reruns seed=42 on current codebase even though it costs 1 extra GPU-day, I will note this in the score rationale as **honest σ measurement** (intangible credibility, not a score number).

---

## 3. Pre-Registered Decision Tree on Path A (A2)

Input: `tf_rollout_gap_val.json` for imgaux_boost (C) and pixenc_ablation (N1) on full-val.

Read `downstream_summary.mean_exposure_gap_dB` and `mean_ceiling_gap_dB` from each JSON.

### 3.1 Per-checkpoint verdict table

| exposure_gap (dB) | ceiling_gap (dB) | Verdict | Score for this checkpoint |
|:---:|:---:|---|:---:|
| ≥ 0.5 | any | `EXPOSURE_BIAS_DOMINATES` | +0.3 Causal, +0.2 Story |
| < 0.2 | ≥ 5 | `VELOCITY_CAPACITY_BOUND` | +0.4 Causal (harder to fix, so stronger diagnosis) |
| < 0.2 | < 2 | `NEAR_CEILING` | +0.0 (contradicts E1; requires re-audit) |
| 0.2–0.5 | any | `MIXED_CAUSE` | +0.2 Causal, +0.1 Story |

### 3.2 Cross-checkpoint comparison (the real value of running 2 ckpts)

Compare C vs N1:

| |Δexposure_gap_C_vs_N1| | Interpretation | Score |
|:---:|---|:---:|
| ≥ 0.2 dB | Pixel forcing demonstrably reduces exposure bias → hop0 story partially rescued | +0.3 Novelty |
| 0.05–0.2 dB | Weak effect, same direction as hypothesized | +0.1 Novelty |
| < 0.05 dB | Pixel forcing does not affect exposure bias → F1 concession stands | 0 |
| Opposite sign (pixel forcing *increases* exposure bias) | Actively harmful; extra evidence for negative-result paper | +0.2 Story (honesty), +0.1 Submission |

### 3.3 Cap

Max from Path A = +0.8 Causal, +0.3 Story, +0.3 Novelty, +0.1 Submission. Cannot stack past those caps regardless of how many sub-verdicts trigger.

---

## 4. Pre-Registered Decision Tree on Path C (A3)

Input: before-FT E2E PSNR vs. after-FT (5K steps) E2E PSNR at NORMAL on full-val.

Let ΔPSNR_FT = PSNR_after - PSNR_before.

| ΔPSNR_FT | Interpretation | Score |
|:---:|---|:---|
| ≥ 3.0 dB | RAE decoder is binding; latent-only paradigm has a hard ceiling that decoder-FT can partially break. Strong negative-result evidence. | +0.5 Novelty, +0.3 Story |
| 1.0–3.0 dB | Partial RAE bound. Latent-only paradigm is expressive but imperfect. | +0.3 Novelty, +0.2 Story |
| 0.2–1.0 dB | Weak decoder sensitivity; bottleneck is upstream. | +0.1 Novelty, +0.1 Story |
| < 0.2 dB | Decoder is not the bottleneck. 10 dB gap is entirely upstream (RAE encoder null space). **Strongest possible negative-result evidence.** | +0.4 Novelty, +0.4 Story |

Note: the two extreme outcomes (≥3 dB and <0.2 dB) both boost the score because either one is a **sharp, publishable finding** about where the bottleneck lives. Middle outcomes are scientifically less informative.

---

## 5. F2 Rewrite (A4)

I will inspect `CLAUDE.md`, `IDEA_REPORT.md`, and any new `review/0424+/*.md` files for:

- [ ] Removal of "decoder amplifies tail error" framing
- [ ] Replacement with either "normalized improvement (ΔPSNR / Gap_Transport)" metric or "Gap_Transport grows monotonically so dB units bias tail" explanation
- [ ] No lingering claims tying tail-bigger-gain to decoder non-linearity

**Pass condition**: all three items check. **+0.3 Story.**

**Fail condition**: any old language survives. **0 score change + flag in next review.** This is cheap; no excuse for not fixing.

---

## 6. Pix2Pix Baseline (A5)

Minimum viable baseline:
- Same val split (the patient-level split verified in R4)
- Same image-space PSNR metric (`calc_psnr_clip3`)
- Same clamp_max (10.0)
- Runs trained to reasonable convergence (not necessarily SOTA, just honest)
- Report transport_avg on the 4 target timepoints

Results table:

| Gap (C best − Pix2Pix best) | Score delta |
|---|:---|
| Pix2Pix > C by > 2σ_seed | +0.5 Submission (honest loss; reframe-negative story strengthened) |
| Pix2Pix within 2σ_seed of C | +1.0 Submission (our method is competitive, strong workshop story) |
| C > Pix2Pix by > 2σ_seed | +1.0 Submission (our method is better, but only claim this if σ_seed is known) |
| Pix2Pix not reported | 0 Submission change |

**Pre-paid rebuttal**: I will NOT ask for a second baseline (PET-DDPM, Palette, etc.) for the workshop submission. One is enough if executed honestly. Save second baseline for any subsequent full-conference attempt.

---

## 7. 200K Decision Record (A6)

Case A — σ_seed kills 200K and Proposer writes short justification memo citing `σ_hat > 0.08`:
- **+0.5 Causal** for process discipline.
- Review scoring ceiling capped at 4.4/10 (no upside from running 200K).
- Story ceiling capped at 4.5/10 (pure negative paper).

Case B — 200K runs under σ-unit gate and gate triggers abandon at 150K:
- **+0.3 Causal** for following the gate.
- No penalty for running the experiment.

Case C — 200K runs to completion, ΔPSNR(200K vs 50K) > 3·σ_upper_80 AND CI lower bound > σ_upper_80:
- **+0.5 Causal, +0.3 Story** (positive finding, properly calibrated).
- Ceiling opens to 5.2/10 total.

Case D — 200K runs to completion with ΔPSNR > 3σ but Proposer does not report CI:
- **−0.3 Causal, −0.2 Submission** for incomplete statistical reporting.

Case E — 200K runs, claim made without σ_seed data first:
- **Revoke R6 acceptance**. Back to R5 score 3.05/10. Pre-registered.

---

## 8. Hard Floors and Ceilings

### 8.1 Floor (what can drag the score below 3.25)

1. **Any publication claim made without σ_seed** (−0.5 Story, −0.3 Causal)
2. **Reactivating SpatialAlignmentProjector, SeamRefiner, or adding new modules to "save" the story** (−0.5 Implementation, violates KILL list K2/K3)
3. **Proposer submits to MICCAI main / NeurIPS main / CVPR / TMI with current data** (−1.5 Submission)
4. **F2 language survives in any published doc** (−0.3 Story)
5. **Failure to document patient-level split in final paper** (−1.0 Submission, this is desk-reject-prevention)

### 8.2 Ceiling

Maximum reachable score with the R6 clist alone (no new unforeseen positives): **5.2 / 10**.

Breaking past 5.2 requires ONE of:
- External validation of the diagnostic framework on a non-PET task (e.g., video multi-step prediction)
- New baseline beating Pix2Pix on the same split
- A demonstrable positive result from Path B (velocity representation change), which is out of R6 scope

None of these are required for a workshop submission.

---

## 9. Things I Will NOT Re-Litigate in Round 7

To prevent goalpost-shifting, I am committing now:

- ❌ **Will not re-challenge**: the clean-ablation claim (C vs N1 = single-variable diff). Tribunal verified. Closed.
- ❌ **Will not re-challenge**: the patient-level split. R4 C-auditor verified via `RAE/preprocess_lowdose_pet_to_rae.py::split_subjects`. Closed.
- ❌ **Will not re-challenge**: whether hop0 "is" the bottleneck. F1 concession accepted; Proposer no longer claims it. Closed.
- ❌ **Will not demand**: mediation experiment (Exp 2). Nice-to-have, not blocking.
- ❌ **Will not demand**: seed-variance for N1 specifically. σ_seed on C alone is sufficient given F1 concession.

If I find myself reaching for any of these items in Round 7, I should stop and cite this section.

---

## 10. Things I Reserve The Right To Flag

These are NOT pre-committed to score changes, but I reserve flagging rights:

- 🟡 **New red flags in the data** (e.g., per-slice distribution has a fat tail indicating data bugs). Will be reported as new findings, scored on their own merit.
- 🟡 **Writing quality in the workshop submission itself** — clarity, claim hygiene, reproducibility section. Those get scored when the manuscript lands, not now.
- 🟡 **Patient-split statement in the final paper**. This *is* required for publication; if missing, I flag as a blocker irrespective of its R6 status.
- 🟡 **Consistency between artifacts** — if Path A JSON and CLAUDE.md narrative contradict, that's a new finding, not a re-litigation of existing ones.

---

## 11. Scoring Worksheet Template (for Round 7)

When Proposer submits data, I fill in:

```
=== Round 7 Score Calculation ===
Base (R6 signoff):                        3.25

+ A1 σ_seed-lite    (§2):                 +____
+ A2 Path A         (§3):                 +____
+ A3 Path C         (§4):                 +____
+ A4 F2 rewrite     (§5):                 +____
+ A5 Pix2Pix        (§6):                 +____
± A6 200K decision  (§7):                 +____
− Floor violations  (§8.1):               −____

Hard ceiling cap    (§8.2):               min(., 5.2)

=== Round 7 Total ===                      ____  / 10
```

Each line must cite its own §-rule. No freelancing.

---

## 12. Reviewer Declarations (Fairness Pre-Commitments)

Because Round 6 was a gracious mutual concession, I pre-commit to the following to keep Round 7 fair:

1. **If Proposer executes exactly the R6 clist (trap-fixed) and all gates come out in the "negative result" direction**, I will write Round 7 as **endorsement for MIDL/MICCAI workshop submission** with specific comments, not as a request for more experiments.
2. **If σ_seed > 0.08 dB** and Proposer honestly reports it as null, I will score this at **minimum 4.2/10**, because the negative finding is itself a scientific contribution when executed with this much rigor.
3. **If data collection takes longer than 14 days** due to legitimate compute issues (node failures, queue times), I will not use delay alone as a score penalty. Delay combined with silence → yes; delay with honest update → no penalty.

---

## 13. Summary Card (for fast reference)

| Axis | R6 baseline | R7 ceiling (everything lands) | R7 floor (all traps) |
|---|:---:|:---:|:---:|
| Novelty | 3.0 | 3.8 | 2.5 |
| Story | 3.8 | 4.8 | 3.0 |
| Implementation | 5.8 | 7.0 | 5.3 |
| Causal | 2.3 | 5.5 | 1.5 |
| Submission | 2.5 | 4.5 | 1.5 |
| **Weighted total** | **3.25** | **~5.1** | **~2.5** |

Most likely Round 7 landing zone: **4.0–4.6** (workshop submittable with honest negative-result framing).

---

## 14. What Proposer Should Do With This Document

- Treat it as your grading rubric. Everything I will score is specified.
- If you disagree with any threshold (e.g., my 3·σ_upper_80 for 200K), push back **now**, not in Round 7. I will consider changes, but not after data lands.
- Cite this file in your Round 7 submission so we both know we're operating on the same rules.
- If external circumstances force deviation (e.g., 4-seed σ budget impossible), flag it upfront and I'll negotiate; silent deviation triggers the R6-revocation clause (§2 row 4).

---

*End of Pre-Registered Verdict Map. Review process is now fully specified. Floor is Proposer's.*
