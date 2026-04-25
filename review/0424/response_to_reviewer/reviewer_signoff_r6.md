# Reviewer Signoff — Round 6 Acceptance

**Date**: 2026-04-24  
**Responding to**: `response_to_reviewer/proposer_final_response_r6.md`  
**Reviewer verdict**: **ACCEPTED**. Round 6 closes the review loop. Process moves to Round 7 only if execution data invalidates the new clist.

---

## 0. Overall

Proposer Round 6 fully accepts the revised clist (§1 200K, §2 σ_seed, §3 F2 rewrite, §4 new timeline) and provides concrete config + command mappings. The three retractions proposer made (σ_seed scope generalization, F2 concept confusion, "plateau" undefinedness) are exactly the three load-bearing corrections I asked for. **From my seat this is the cleanest rebuttal→signoff transition I've seen in 6 rounds.**

Score stays at the new **3.25 / 10** (Round 6 signoff is not a re-review; no new data yet). Conditional ceilings 4.4–4.9/10 remain valid if the clist executes.

---

## 1. Credit Where Due

Three moves I want to record:

1. **§3 F2 rewrite commitment** — accepting normalized improvement (`ΔPSNR / Gap_Transport`) or latent-MSE relative change as the reporting metric is a big deal. Most authors keep dB absolute values because dB-language is what reviewers expect; proposer is trading surface appeal for mechanistic honesty. Good trade.
2. **Cost-saving σ-lite acknowledgement** — accepting that σ_seed-lite (2 GPU-day) saves vs. 200K+full-σ (6.7 GPU-day) shows proposer isn't anchoring on "more experiments = more rigor" — which is a common mistake at this stage.
3. **Reframe title commitment in §5** — dropping the "pixel injection breaks bottleneck" language in writing (not just in private docs) is what separates a real reframe from a cosmetic one. If this title sticks, the workshop path is on track.

---

## 2. Three Implementation Traps — Flag Now, Not at Round 7

Proposer's Round 6 plan is correct in logic. Three operational traps would still sink it if not addressed **before kickoff**:

### Trap 1 — "Reuse seed=42 old results" is unsafe for σ_seed

Proposer §2 Phase 1 says:
> "seed=42：已有结果（旧 Scheme C best = 36.206）✓"
> "实际新增：2 runs × 50K"

This is almost certainly **contaminated**. Between when `imgaux_boost_seed42` was trained and today:
- the 0422 weight-fix may have landed in `train_first_hop.py`
- dataloader seed handling, EMA schedule, or mixed-precision settings may have drifted
- pytorch version / cudnn may have changed

If the old seed=42 checkpoint ran on a different codebase than the new seed=123/456 runs, you're measuring (σ_seed) + (σ_codebase) in one number. σ_seed estimate is inflated by an unknown codebase term, and your 200K stop criterion inherits that inflation.

**Required fix**: **rerun seed=42 on the current codebase.** Yes, this costs 1 extra GPU-day (back to 3 new runs = 3 GPU-day). Yes, it's worth it. The whole point of σ_seed is that everything except seed is identical. If you need to cut budget, prefer 3 seeds × same-code over 3 seeds × mixed-code.

Alternative if you really can't afford the extra run: **explicitly document in the paper** "seed 42 is from codebase rev X; seeds 123, 456 from rev Y" and report **within-codebase std** (n=2 on rev Y) as the primary σ_seed, with the cross-codebase seed=42 as a sanity check. Reviewers will accept this if documented; they won't accept silent mixing.

### Trap 2 — σ_seed from n=3 has a wide CI, σ-unit thresholds inherit that

I touched this in dossier §2.4 but proposer's Round 6 text uses σ_seed as a *point estimate* in the thresholds:
```
PASS: mean(200K) - mean(50K) > 3·σ_hat_seed
ABANDON: Δ < σ_hat_seed
```

With n=3, the true σ could easily be 0.5× to 1.7× of `σ_hat`. If `σ_hat = 0.07`, true σ might be anywhere 0.035–0.12.

**Required fix**: in every threshold, replace `σ_hat_seed` with **`σ_upper` = upper 95% CI of σ_seed**. Formula for n=3:
```
σ_upper ≈ σ_hat × sqrt((n-1) / χ²_{0.025, n-1}) = σ_hat × sqrt(2/0.0506) ≈ σ_hat × 6.29
```

That's way too conservative. Pragmatic compromise: use `σ_upper ≈ 1.5 × σ_hat` as the conservative bound (this matches the upper limit of the 80% CI for n=3, which is a reasonable operating point for n=3 decisions). Document this in the paper.

**Better alternative**: run n=4 seeds instead of 3. Bumps cost from 3 to 4 GPU-day but dramatically tightens the σ CI. Given proposer is already getting seed=42 rerun for free under Trap 1, the total is 4 GPU-day — same as my original Exp 1 ask. **This is my actual recommendation.** Seeds {42, 123, 456, 789} × imgaux_boost × 50K = 4 runs.

### Trap 3 — Path A checkpoint choice is ambiguous

Proposer §2 says:
> "Path A ... --checkpoint <C_best.pt>"

**Which C best.pt?** imgaux_boost_best selected by which criterion? If `val_multi_objective` was the selection metric, there's the same leaderboard-contamination risk F7 identified.

**Required fix**: explicitly use the **full-val reranked best** (the 0423 protocol), not the multi-objective best. And run Path A on **at least 2 checkpoints** — imgaux_boost (C) and pixenc_ablation (N1). Reason: if exposure_gap_dB differs meaningfully between C and N1, that alone is new evidence about what pixel forcing is actually doing. If it's the same, the hop0 null hypothesis gets *more* support.

Cost delta: 2 × 0.5 GPU-day = 1 GPU-day (already budgeted).

---

## 3. Revised Budget (with traps fixed)

| Item | Proposer R6 | Trap-fixed | Δ |
|---|---:|---:|---:|
| σ_seed | 2 GPU-d (reuse seed=42) | 4 GPU-d (4 seeds fresh) | +2 |
| Path A | 0.5 GPU-d (1 ckpt) | 1.0 GPU-d (2 ckpt) | +0.5 |
| Path C | 1.0 | 1.0 | 0 |
| 200K (conditional) | 2.7 | 2.7 | 0 |
| **Total if 200K runs** | **6.2** | **8.7** | **+2.5** |
| **Total if σ_seed kills 200K** | **3.5** | **6.0** | **+2.5** |

Net: 2.5 GPU-day more than proposer's R6 plan, but each added day eliminates a specific contamination source. For a negative-result workshop paper, that's the right trade.

If budget is truly fixed at ~6 GPU-day, drop Path C for now (Path C is independent and can run post-submission as camera-ready supplement).

---

## 4. Meta — What Ends the Review Loop

Round 6 is the formal end of the review dialogue. The next interaction between reviewer and proposer should be **data-driven**, not argument-driven:

- σ_seed-lite result → triggers either 200K launch or 200K kill.
- Path A verdict → triggers either exposure-bias work or capacity/pivot work.
- Path C result → sets the RAE bound on the narrative.

Reviewer ask of proposer: **post results here once collected**, with raw JSON/CSV pointers. No need to re-argue any of the 5 rounds. I will update scores based on data alone.

If any of Traps 1–3 are not fixed, I will note it in the data review (not hold up the data collection itself).

---

## 5. One Sentence Signoff

Rebuttal cycle closes at **3.25 / 10 (workshop-marginal)** with a credible 4.4–4.9 ceiling contingent on (σ_seed with 4 seeds on a single codebase) + (Path A on 2 checkpoints) + (F2 rewrite in-doc) + (Pix2Pix baseline) being delivered in the next 14 days.

---

## 6. Administrative

- Review file tree complete. Final state of `review/0424/`:
  - `claude-reviewer/nightmare/00..05_*.md` — 5-round nightmare review
  - `claude-reviewer/breakthrough_analysis_10dB_gap.md` — 10 dB gap analysis + Path A design
  - `claude-reviewer/06_reasoning_dossier_on_rebuttal.md` — reviewer's uncompressed reasoning
  - `response_to_reviewer/rebuttal_nightmare_r5.md` — proposer R5 rebuttal
  - `response_to_reviewer/reviewer_reply_to_rebuttal.md` — reviewer compressed reply
  - `response_to_reviewer/proposer_final_response_r6.md` — proposer R6 acceptance
  - `response_to_reviewer/reviewer_signoff_r6.md` — **this file**
- `scripts/diagnose_tf_rollout_gap.py` — Path A implementation (ready to run)

The loop is closed. Next artifact should be **experimental data**, not more review text.

---

*End of Round 6. Good luck with execution.*
