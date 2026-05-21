# Round 17 Peer Review — Reviewer C (independent)

- **reviewer name**: **C** (consistent with R15/R16)
- date: 2026-05-21
- branch: foc_lite_hop0 (commit 52406e7)
- substrate read independently (no other reviewer drafts seen):
  - PEER_REVIEW_PROMPT_round17_strategy_post_V13V14_20260521.md (the prompt)
  - review/0516/full_eval_json/v13_true_image_aux_off_best_fullval_psnr_chain_mse.json
  - review/0516/full_eval_json/v14_true_d_pure_best_fullval_psnr_chain_mse.json
  - review/0516/full_eval_json/v13_true_image_aux_off_last_fullval_psnr_chain_mse.json
  - review/0516/full_eval_json/v14_true_d_pure_last_fullval_psnr_chain_mse.json
  - review/0516/V13_V14_full_eval_analysis_20260521.md (cross-reference)
- mandate per prompt §9: this is a **strategic decision**. Give a clear path or state what minimal data is missing. Avoid the "稳妥, 再跑 2 个实验" default.

---

## TL;DR (Reviewer C verdict)

| Item | Verdict |
|---|---|
| Q1 V14 as noise estimator | **MODIFY** — V14 is a 1-sample *upper bound* on d_pure noise median, not a std. Sufficient for ranking V18 deltas vs image_aux; insufficient for "p < 0.05 V18 main-result" claim. |
| Q2 V13 cleared V7-V8 confound | **APPROVE with one qualifier** — image_aux carries ~+0.29 dB of the historical V7-V8 gap; Grönwall step_weights' residual contribution is ≤ +0.02 dB and indistinguishable from train-config drift. Update narrative; do not relaunch a Grönwall-isolation experiment. |
| Q3 V18 +0.06 dB significance | **MODIFY** — V18.last is "real signal, small, paper-ablation-only-grade". V18 should be **demoted from headline to ablation**, image_aux promoted to headline. |
| Q4 strategic option | **Hybrid B+A4** primary, **B' (image_aux headline paper)** as the framing. See §4 below. |
| Q5 paper narrative | **B' headline (image_aux primary) + C' as structural diagnosis (decoupling) + KL/V18 as negative-control ablation**. MICCAI main track realistic. |
| Q6 missing controls | **V14b second seed** is the single highest-EV outstanding experiment ONLY IF we want to keep V18 +0.06 dB as a paper-claim. If paper headlines image_aux, V14b/V14c become optional polish. |
| Q7 new biases | **3 new (B61-B63)**. **B61 (MOD)** is a sign-flip in prompt §2 table; **B62 (LOW)** is the unused seed=1337 V14↔V6_NOISE pair; **B63 (LOW)** is anchoring "real signal" via 150× single-sample ratio. |

**Main verdict (one line)**: V18 is real but tiny; image_aux is the project's actual main result; ship the paper with image_aux as headline, V18 as ablation, KL as negative control. Run V14b in parallel only as cheap polish, not as a gating dependency.

---

## 0. Mechanical verifications I ran first

### 0.1 JSON ↔ prompt §1.2 (✓ exact)

Loaded both JSON files and read `summary_psnr_clip3.{D20,D10,D4,NORMAL}.mean`:

| run | claim (prompt §1.2) | JSON value | match |
|---|---:|---:|---|
| V13.best NORMAL | 36.4943 | 36.4943 | ✓ |
| V14.best NORMAL | 36.7806 | 36.7806 | ✓ |
| V13.best D20 | 35.2172 | 35.2172 | ✓ |
| V14.best D20 | 35.4249 | 35.4249 | ✓ |
| V13 last == best | yes | last.NORMAL = best.NORMAL (Δ=0) | ✓ |
| V14 last == best | yes | last.NORMAL = best.NORMAL (Δ=0) | ✓ |

### 0.2 Independent recomputation of §1.4 facts (✓)

```
X1 d_pure 1-sample:  V14 - V7 = -0.0004 dB
X2 image_aux:        V7 - V13 = +0.2867 dB    (V14 - V13 = +0.2863 dB; agree at 4th decimal)
X3 Grönwall residual: V13 - V8 = +0.0214 dB
```

All three derived facts match prompt §1.4. ✓

### 0.3 Cross-checks the prompt did NOT do

I computed two additional deltas the prompt does not surface:

| comparison | value | note |
|---|---:|---|
| **V14 − V6_NOISE (both seed=1337)** | **+0.0369 dB** | V7-config vs V6-config **at fixed seed=1337**. This is a cleaner per-config delta than V7 (seed=42) vs V6_NOISE (seed=1337). |
| V7 − V6_NOISE (mixed seeds) | +0.0373 dB | Consistent with the per-config delta above; the seed=42-vs-1337 perturbation contributes ~+0.0004 (matches X1). |

Implication for the prompt's §2 SNR table and Q3: the V7-config-over-V6-config historical gain is ~+0.037 dB at fixed seed, which is **already smaller than the V18.best (+0.030) is large**. This sharpens the framing: V18 +0.030 is ~80% of the V7-vs-V6 historical step, which is one of the project's foundational "wins". V18 is *not* trivially small relative to historical wins, but the V18.last +0.062 is the more defensible number against any plausible noise.

---

## 1. Q1 — Is V14 sufficient as d_pure noise estimator?

**Verdict: MODIFY.**

V14 is one (config, seed=1337) sample. With one sample you cannot estimate a standard deviation; you can only assert |Δ_obs| as an order-of-magnitude bound. The prompt §1.4 X1 carefully labels this as a **bound**, but prompt §2 then computes "SNR = Δ / noise = 75-150×" treating the single sample as if it were noise. That's where the framing slips.

**The most defensible claims using only V14:**
- V18.last vs V7 = +0.062 dB **exceeds any single-seed perturbation observed in this project**. (True, but a one-sample test.)
- V18.best vs V7 = +0.030 dB **exceeds the single-seed perturbation observed in this project**. (Same caveat.)
- Image_aux contribution (V7 − V13 = +0.287 dB) **exceeds plausible noise by >2 orders of magnitude regardless of the std estimate**. (Defensible without further data.)

**Indefensible without V14b/V14c:**
- "V18.best ≠ V7 at p < 0.05." (Requires variance.)
- "V18 chain gain is 75× the noise floor std." (Treats 1 sample as std.)
- Any per-seed std estimate.

**Practical implication for the paper:**
- If image_aux is the headline (+0.29 dB), no second seed is needed; the result is robust to any plausible variance.
- If V18 +0.030 / +0.062 is in the paper as **ablation** ("decoder LoRA at rank=32 yields +0.06 dB chain over V7"), one sample is acceptable for ablation, especially with the caveat "single-seed".
- If V18 is the **main claim**, V14b at minimum (preferably V14c too) is required. Reviewers will ask.

**Recommendation**: do not promote V18 to main claim. Then V14b/V14c are nice-to-have, not gating. If user prefers to keep V14b as a polish item, launch it in parallel with paper writing (7 days no-cost since slots are idle); but do not let it block paper drafting.

---

## 2. Q2 — Has V13 cleared the V7-V8 confound?

**Verdict: APPROVE with one qualifier.**

The factorial picture is now clean enough to act on:

| config | image_aux | Grönwall sw | seed | NORMAL | data |
|---|:-:|:-:|:-:|---:|---|
| V8 | OFF | ON (Grönwall) | 42 | 36.4729 | Plan F train config |
| V13 | OFF | ON (Grönwall) | 42 | 36.4943 | V7 train config |
| V6_NOISE | ON | OFF (V6 weights) | 1337 | 36.7437 | V6 train config |
| V7 | ON | ON (Grönwall) | 42 | 36.7810 | V7 train config |
| V14 | ON | ON (Grönwall) | 1337 | 36.7806 | V7 train config |

Derivations:
- **image_aux effect** (V7 − V13, V7 vs V13 differ ONLY in `lambda_img`): +0.2867 dB. Clean single-variable estimate.
- **Seed effect at V7 config** (V14 − V7): −0.0004 dB. Clean within-config.
- **V7-config vs V6-config at fixed seed=1337** (V14 − V6_NOISE): +0.0369 dB. The "V7 over V6" structural advantage with seed held constant.
- **Grönwall step_weights residual at no-image_aux** (V13 − V8): +0.0214 dB. **Confounded** by train-config differences (V7 train config vs Plan F train config), as the prompt notes.

**What is now safe to assert**:
1. Image_aux alone moves NORMAL by ~+0.29 dB. This is the single largest design decision in the project's history. ✓
2. Grönwall step_weights' contribution to V7's headline number is ≤ +0.02 dB (and might be even less after deconfounding train config). Not a "design victory"; closer to noise. ✓
3. V7-config-vs-V6-config delivers ~+0.037 dB at fixed seed. Modest. ✓

**Qualifier**: V13-V8 has the train-config confound; do not point to V13-V8 ≈ 0.02 dB and conclude "Grönwall step_weights = +0.02 dB exactly". The correct paper phrasing is: *"After isolating image_aux (V13), the residual difference attributable jointly to Grönwall step_weights and train-config differences is ≤ 0.02 dB, well within any plausible variance threshold. We therefore do not identify Grönwall step_weights as a meaningful design factor."*

**Action**: yes, update V18_design_rationale.md / paper draft to retire the "Grönwall step_weights is the transport design victory" framing. Do NOT launch a Grönwall-isolation experiment to clean up the 0.02 dB residual — EV is too low and the conclusion ("not a meaningful factor") already holds regardless.

---

## 3. Q3 — V18 +0.06 dB significance and reframing

**Verdict: MODIFY — V18 is "real but ablation-grade".**

### 3.1 What the numbers actually say

| comparison | NORMAL Δ (dB) | relative to image_aux (V13→V7 = +0.287) | reframing |
|---|---:|---:|---|
| V18.best − V7.best | +0.030 | 11% | ablation |
| V18.last − V7.best | +0.062 | 22% | ablation, slightly stronger |
| V18.last − V18.best (35K more LoRA training) | +0.031 | 11% | training-duration effect |
| **image_aux** (V7 − V13) | **+0.287** | 100% | **main effect** |

V18's entire delta is ~1/5 of the image_aux effect. V18 should not be the paper's main contribution when image_aux is sitting right next to it doing 5× the work for free (no LoRA, no KL, just a loss-weight choice).

### 3.2 Most parsimonious mechanism for V18 +0.06 dB

After A3 (Round 16):
- KL pullback contributes ≈ 0 (matched-step verified at both direct and chain).
- LoRA capacity at rank=32 / last 2 blocks contributes ~+0.17 dB on direct GT-manifold decode.
- Only ~18% of that propagates to chain (~+0.030 dB).
- Continuing to train (165K → 200K LoRA-on-V7) adds another +0.031 dB chain.

So the cleanest reading is: **V18 ≈ 35K extra fine-tune steps on V7 with a rank-32 LoRA bottleneck, equivalent to a generic capacity-boosted continuation of V7 training.** This is real but mundane — it would happen with most reasonable extra-training schemes.

### 3.3 Concrete reframing for paper

V18 should appear as an ablation row labeled something like:
> "Continued V7 fine-tuning with a rank=32 LoRA on the last 2 decoder blocks (V18) yields +0.06 dB chain NORMAL PSNR over V7 best.pt at 200K total steps, mostly attributable to LoRA capacity rather than the KL pullback regularizer (validated via the V18-capacity-only A3 control)."

This puts V18 in proper proportion: nice-to-have, mechanistically interesting (decoupling diagnosis), not a method contribution.

---

## 4. Q4 — Strategic option

**Verdict: Hybrid B+A4 (paper draft + A4 image_aux schedule probe in parallel), framed as B' (image_aux headline).**

### 4.1 Rank of the 5 + Hybrid options

| option | EV | cost | verdict |
|---|---|---|---|
| **B' Paper as-is (image_aux headline)** | **High** — image_aux +0.29 dB is the strongest, cleanest result in the project | ~1-2 months writing | **PRIMARY** |
| **Hybrid B+A4** (B' + cheap image_aux schedule probe) | **Highest if A4 yields anything** — image_aux already proven dominant, schedule tuning could add another +0.05 dB headline gain; if not, no loss | ~7 days GPU + parallel writing | **PRIMARY (extended)** |
| A transport intervention (A1-A3, A5) | Medium-low — unknown EV, Round 16 showed A3 cannot confirm transport is THE bottleneck (W1/W2/W3) | 7-14 days each + design | **DEFER** until paper draft exists |
| C arch pivot | Low EV near-term | 4-8 weeks | **REJECT** for current paper window |
| D data pivot | Unknown | 2-4 weeks | **REJECT** for current paper window |
| E terminate | Wasteful — image_aux finding is paper-worthy | 0 | **REJECT** |
| F V19 = corrected KL (`use_pred_latent=false`) | **Near zero** | 7 days | **REJECT**. After A3, V19 would at best tie A3, no story. |

### 4.2 Why A4 specifically (out of A1-A5)

Among the transport-side intervention candidates:
- **A4 (image_aux schedule tuning)** is the only one with a *prior* — we now know image_aux is the dominant signal (V13). Tuning its schedule (longer warmup? higher lambda_max? per-hop weighting?) has highest base rate of yielding a measurable Δ, because we're now tuning a knob we've **confirmed** matters.
- A1/A2 (backbone changes) are speculative and expensive.
- A3 (multi-step refinement) is plausible but doesn't connect to the V13 finding.
- A5 (data augmentation) is open-ended.

So if we must run ONE more experiment, A4 is the right choice — informed by V13, not blind.

### 4.3 Concrete hybrid plan

| timeline | action |
|---|---|
| T=0 | Begin paper draft with B' narrative (image_aux headline). Update V18_design_rationale.md. Withdraw KL-success claims. |
| T=0 | Launch A4 on free slot (1 of 3 slots; ~7 days). Design: V7-config with image_aux schedule perturbed (e.g., warmup_ratio=0.0 → 0.1, lambda_max=0.04 → 0.06; one yaml at a time). |
| T=0 | (Optional) Launch V14b (V7-seed=2024) on second free slot if user wants robust ablation. Purely polish. |
| T=7d | A4 returns. If Δ ≥ +0.05 dB: rewrite paper to include A4. If < +0.05: footnote A4 as ablation confirming image_aux is near-saturation. |
| T=8w | Submit. |

**Critically**: A4 is *exploratory polish*, not a gating dependency. Paper drafting starts at T=0 regardless.

### 4.4 Where I differ from a default Hybrid B+A4 reading

The prompt frames A4 as "image_aux schedule tuning (image_aux 已饱和 V7, 但可能 alternative schedule 释放 ~0.05 dB?)". That hedges on saturation. V13 shows image_aux *exists*; it does not show image_aux is *saturated*. The honest framing: "we don't know whether V7's image_aux schedule is optimal." A4 answers that.

---

## 5. Q5 — Paper narrative

**Verdict: B' (image_aux headline) is the main story. C' (decoupling diagnosis) is the supporting structural claim. KL/V18 are negative-control ablations.**

### 5.1 Recommended narrative structure

**Title direction**: something like *"Image-Aware Loss Dominates Latent-Transport Quality in PET Low-Dose Reconstruction: A Structural Ablation Study"*

**Headline contributions**:
1. **Image auxiliary supervision contributes +0.29 dB chain NORMAL PSNR** in latent-transport PET reconstruction, identified via the V7-V13 single-variable comparison. This is the dominant design factor measured in this project.
2. **Decoder capacity and transport quality are structurally decoupled** in the frozen-decoder latent-transport framework: decoder LoRA at rank=32 yields +0.17 dB on direct GT-manifold decode but only ~+0.03 dB propagates to chain output, suggesting chain output is bottlenecked by transport (or by an unidentified joint factor) rather than decoder fidelity.
3. **KL pullback regularization** between LoRA and frozen decoder copies does not improve over a matched-step capacity-only control (A3), validating that decoder capacity alone explains the small V18 chain gain.

**Ablations**:
- V14: single-seed robustness (V7-config two-seed bound on |Δ| < 0.001 dB)
- V13: image_aux ablation (paired with V7)
- V18 vs A3: KL pullback vs capacity-only matched-step
- V8 vs V13: cross-config validation of image_aux dominance
- (optional) A4: image_aux schedule sensitivity

**Limitations**:
- Single-seed noise floor estimator; multi-seed variance not characterized.
- Frozen-decoder framework; co-trained decoder not evaluated.
- transport-side intervention beyond image_aux not explored in this paper.

### 5.2 Why this beats other narratives

| narrative | weakness |
|---|---|
| **A' (KL pullback aligned decoder)** | Disproven by A3. REJECT immediately. |
| **C' (decoupling main result)** | Academic but workshop-level; doesn't lead with the +0.29 dB image_aux finding. Underweights the strongest result. |
| **D' (negative result, +0.1 dB ceiling)** | Misframed — the project DID find a +0.29 dB design factor. That's not a negative result. |
| **E' (feasibility study)** | Sells short. We have a positive image_aux finding plus a structural negative finding. Better than feasibility. |
| **B' as above** | Strongest positive story + honest negative findings + structural insight. **APPROVE.** |

### 5.3 Venue

- **MICCAI 2026 main track**: realistic with B'. Image_aux finding is a concrete, reproducible design insight for PET reconstruction. Adequate ablation depth.
- **ISBI / IPMI**: backup if MICCAI doesn't land.
- **Med Image Anal / TMI** (journal): possible if the project adds 1-2 more positive results post-paper-draft. Not the right venue for the current state.

---

## 6. Q6 — Missing controls (post-V13/V14)

**Verdict: V14b is the only high-EV outstanding control. Everything else is low-EV.**

| candidate | priority | reason |
|---|---|---|
| **V14b** (V7 + seed=2024) | **MEDIUM** | Gives second seed point → 2-seed std estimate → unblocks "V18 +0.06 dB statistical significance" claim. Only matters if V18 stays in the paper as a substantive ablation; if V18 is just a 1-line footnote, V14b is optional polish. ~7 days. |
| **V14c** (V7 + seed=7) | LOW | Marginal value over V14b. Three-seed gives proper std but requires another 7 days. Defer unless reviewer (paper reviewer) explicitly asks. |
| V18 + seed=1337 | **LOW** | Tests V18 +0.06 dB seed-robustness directly. Diagnostic but expensive; subsumed by V14b for paper-grade variance. |
| **V13 + decoder LoRA** | **LOW** | Tests whether decoder capacity matters more when image_aux is off. Mechanistically interesting but not paper-critical for B' narrative. |
| V19 = corrected KL (`use_pred_latent=false`) | **REJECT** | After A3, would at best tie capacity-only. No story. Drop entirely. |
| A4 (image_aux schedule tuning) | **MEDIUM-HIGH** | The cheapest path to a positive supplemental result; informed by V13's image_aux finding. Already in §4 hybrid plan. |

### 6.1 If I had to rank the single most-valuable next experiment

**A4 image_aux schedule probe** — because it could yield a positive result that strengthens B' headline. V14b is variance-clean-up; A4 is potential additional contribution.

---

## 7. Q7 — Round 17 prompt biases

### B61 (MODERATE) — Sign flip in §2 SNR table

Prompt §2 row 4 reads:
```
V7.best − V6_NOISE.best | +0.0373 (sic: prompt has −0.0373)
```

But §1.3 (correctly) shows V6_NOISE in the "vs V7.best" column with `−0.0373`, meaning V6_NOISE is 0.0373 dB *below* V7. Therefore V7.best − V6_NOISE.best = **+0.0373** (V7 is higher), not −0.0373.

My independent recomputation: 36.7810 − 36.7437 = **+0.0373**. ✓

The prompt §2 also annotates this row with *"V7 用 seed=42, V6_NOISE 用 seed=1337, 所以这个 delta 与 d_pure 同号且同量级"* — but if "d_pure" (V14 − V7 = −0.0004) and "V7 − V6_NOISE" are claimed to be "same sign", the prompt's sign on the latter would need to be negative. The narrative annotation is downstream of the table sign error.

**Severity**: MODERATE. Doesn't affect strategic recommendation, but if a reader follows the §2 table without independent recomputation, they may interpret V7 as having *lost* to V6_NOISE, contradicting §1.3.

**Fix**: replace §2 row 4 sign to `+0.0373` and adjust the annotation. The independent observation that I think the annotation was *trying* to make is: V7 vs V6_NOISE differs in both config and seed, so the +0.037 dB cannot cleanly attribute to seed *or* to config — the V14-vs-V6_NOISE comparison at fixed seed=1337 (= +0.0369) shows the config delta is essentially the whole +0.037, and seed contribution is ≤ 0.001.

### B62 (LOW) — Unused seed=1337 V14↔V6_NOISE pair

V14 and V6_NOISE both use seed=1337. Their difference (+0.0369 dB) is a **clean per-config delta with seed held fixed**. The prompt doesn't compute this. Surfacing it would:
- Tighten the §2 SNR table (removes the seed-vs-config confound).
- Strengthen Q2 conclusion about Grönwall step_weights residual (the +0.037 dB V7-config over V6-config is mostly real per-config effect, not seed).
- Provide a small extra check that the seed perturbation truly is ≤ 0.001 dB across two adjacent configs.

**Severity**: LOW. Adds analytical clarity but doesn't change strategy.

### B63 (LOW) — "SNR = 75-150×" framing in §2

Prompt §2 computes "SNR = Δ / noise" using V14 single-sample |Δ| as denominator. This treats a single observation as if it were a standard deviation. The honest framing is "ratio of effect to a single-sample upper-bound on noise". Worth one sentence of clarification in any downstream document that cites these SNR numbers.

**Severity**: LOW. Doesn't mislead the strategic decision but encourages over-confidence if reused.

### What I did NOT find

- **No image_aux over-attribution**: V13 is a clean single-variable ablation (only `lambda_img` differs from V7). The +0.29 dB attribution is mechanistically clean.
- **No premature strategic lock-in**: the prompt lists 6 options including hybrid, gives them honest EV bounds, and explicitly asks reviewers to provide alternatives.
- **No paper-narrative anchoring**: the prompt lists 4 narrative candidates with neutral framing and asks reviewers to evaluate.

This is a clean substrate overall, on par with R16. B55-class fabrication is not present.

---

## 8. Output per §6 format

### 8.1 Per-question

| Q | verdict |
|---|---|
| Q1 V14 as noise estimator | **MODIFY** — 1-sample upper bound, not std. Suffices for image_aux + V18 ablation; insufficient if V18 is main claim. |
| Q2 V7-V8 confound cleared | **APPROVE with qualifier** — image_aux is ~+0.29 dB; Grönwall residual ≤ +0.02 dB confounded with train config. Update narrative. |
| Q3 V18 +0.06 significance | **MODIFY** — real but ablation-grade. Demote V18 to ablation; promote image_aux to headline. |
| Q4 strategic option | **Hybrid B+A4 (with B' framing)** primary; defer A/C/D/E; reject F (V19). |
| Q5 paper narrative | **B' (image_aux headline) + C' (decoupling structural) + V18/KL as negative-control ablation**. MICCAI main realistic. |
| Q6 missing controls | **A4 highest EV; V14b medium polish; V14c/V18-seed1337/V19 low or reject.** |
| Q7 prompt biases | **3 new (B61 MOD sign flip, B62 LOW unused pair, B63 LOW SNR framing).** |

### 8.2 Main strategic verdict

**Primary: Hybrid B+A4 with B' narrative framing.**
- Begin paper draft immediately (B') with image_aux as headline contribution.
- Launch A4 (image_aux schedule probe) in parallel on free slot, 7 days.
- A4 outcome is paper-additive, not paper-gating.

**Backup: pure B'** (paper-as-is, no new experiments) if user prefers minimum-overhead path.

**Strongly NOT recommended**:
- Option A (transport intervention beyond A4): too speculative, Round 16 established A3 cannot confirm transport-is-bottleneck.
- Option C (arch pivot): not appropriate for the current paper window.
- Option D (data pivot): no design, no EV estimate.
- Option F (V19 corrected KL): A3 closes this question — V19 at best ties A3.

### 8.3 V18 signal verdict

**"Real but should be reframed as ablation supporting decoupling diagnosis."**
- V18.last +0.062 dB is real (one-sample noise floor ≤ 0.001 dB; effect 60×).
- V18.best +0.030 dB is real but small.
- Mechanism: LoRA capacity + 35K extra training; KL contribution ≈ 0 (A3 confirmed).
- Paper role: ablation row + decoupling diagnosis (decoder gain ≠ chain gain, only ~18% propagates).
- Do NOT headline V18.
- Do NOT spend slot on V18-clean / V19 / rank sweep.

### 8.4 Immediate executable (no GPU)

**Update now (T=0, no GPU)**:
- V18_design_rationale.md §5: add R17 finding — image_aux is the dominant design factor (V7-V13 = +0.29 dB single-variable). Retire "Grönwall step_weights is design victory" framing.
- paper draft scaffold: B' narrative outline with image_aux as headline, V18/KL/A3 as ablations.
- claim ledger: lock in image_aux +0.287, V14 noise upper-bound ≤ 0.001, V18.last/best deltas with proper "single-seed" qualifier.

**Pause now**:
- Any further V18-family launch design (V19, V18-clean, rank sweep) — DROP.
- Architecture / data pivot deliberation — REJECT for current paper window.
- Multi-seed V14b/V14c launch unless A4 lands first and indicates polish-stage need.

**Optional launches now (don't block paper writing)**:
- A4 image_aux schedule probe (~7 days, slot 1).
- V14b if user wants robust ablation (~7 days, slot 2).
- (Other slots stay free.)

### 8.5 New biases (B61-B63)

Detailed §7. **B61** is the only one that warrants a substrate edit (sign flip in §2 table). B62/B63 are clarity improvements, not corrections.

---

## 9. One-paragraph executive summary

V13/V14 substrate is grep-verified clean: image_aux contributes +0.287 dB (single-variable, V7-V13), seed perturbation is ≤0.001 dB (V14-V7), Grönwall step_weights residual is ≤ +0.02 dB confounded with train-config drift. Image_aux is the project's actual main finding — ~5× larger than the entire V18 chain gain (+0.062 dB). V18 should be demoted from headline to ablation; the paper's main story is **image_aux dominance + decoupling diagnosis (A3) + KL negative control**. Recommended action: Hybrid B+A4 — start paper drafting with B' narrative immediately, launch A4 image_aux-schedule probe in parallel (~7 days, slot 1) as polish-not-gating. Reject V19/V18-clean/rank-sweep/transport-intervention/arch-pivot for the current paper window. V14b second-seed is medium-EV polish only if V18 stays as a substantive ablation. Round 17 prompt is clean except for one sign-flip in §2 (B61, MOD), one unused per-config delta the prompt didn't surface (V14-V6_NOISE at fixed seed=1337, +0.0369 dB), and a single-sample-as-std framing in §2 SNR table. None of these change the strategic recommendation.

---

## 10. What I did NOT review

- V13/V14 yaml or eval protocol (already signed; covered by my own grep of JSON metadata, which is consistent).
- A3 substrate (Round 16 signed).
- KL pullback design (Round 16 signed; reaffirmed dead by A3).
- Round 1-16 already-signed content.
