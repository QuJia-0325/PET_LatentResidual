# Round 17 Peer Review - Post-V13/V14 Strategy (GitHub Copilot)

- date: 2026-05-21
- reviewer: **GitHub Copilot**
- prompt: `PEER_REVIEW_PROMPT_round17_strategy_post_V13V14_20260521.md`
- scope: post-V13/V14 substrate interpretation and next project strategy
- stance: decide strategy from the completed substrate; avoid adding multi-week controls only for comfort

---

## 0. Executive Verdict

| item | verdict | reason |
|---|---|---|
| strategic main path | **Hybrid B+A4** | Start paper-as-is now, while running exactly one low-cost image_aux schedule / hop0-strength probe. A4 is the only new GPU run directly justified by V13. |
| backup path | **B paper-as-is** | If A4 is not launched or returns < `+0.05 dB`, the current result is still writeable as a rigorous feasibility / ablation / decoupling study. |
| V18 signal | **real but ablation-only / decoupling diagnosis** | V18.last `+0.0617 dB` is likely above the observed single-seed perturbation, but it is ~1/5 of image_aux and Round16 already attributes the early gain to capacity, not KL. |
| image_aux | **main positive result** | V7/V14-style image_aux beats true image_aux-off V13 by about `+0.286 dB` NORMAL; this is the dominant single-axis design result now. |
| V14 noise | **useful bound, not a distribution** | V14 says seed 1337 under V7 config did not move the mean, but one seed cannot define statistical std. Do not write `150x SNR` as formal significance. |
| extra controls | **do not block strategy** | V14b/V14c can strengthen a paper, but should not be the next decision gate. V19/corrected KL and V18 seed-control are low ROI now. |

One-line recommendation: **write the paper around image_aux dominance plus decoder/transport decoupling, and spend at most one parallel run on an image_aux schedule probe; do not revive KL or launch a seed campaign before drafting.**

---

## 0.5 Evidence Checked

Primary checked artifacts:

- `V13_V14_full_eval_analysis_20260521.md`
- `v13_true_image_aux_off_best_fullval_psnr_chain_mse.json`
- `v14_true_d_pure_best_fullval_psnr_chain_mse.json`
- `REVIEW_INTEGRATION_round16_20260519.md`
- `PLANF_FINAL_ANALYSIS_20260516.md`

Key verified numbers:

| comparison | NORMAL PSNR | delta |
|---|---:|---:|
| V13 image_aux off | 36.494330 | V7 - V13 = `+0.2866 dB` |
| V14 V7 seed1337 | 36.780632 | V14 - V7 ~= `-0.0003 dB` using V7 36.7810 |
| V8 historical image_aux-off/confounded | 36.4729 | V13 - V8 = `+0.0214 dB` |
| V18.best | 36.8112 | V18.best - V7 = `+0.0302 dB` |
| V18.last | 36.8426 | V18.last - V7 = `+0.0617 dB` |

Round16 signed conclusions carried forward:

- A3 kills current V18 KL direct-decoder narrative.
- A3@165K trainer chain metrics essentially tie V18.best@165K.
- V18 should be read as capacity / fine-tuning / decoupling evidence, not KL success.

---

## 1. Q1 - Is V14 Enough as a d_pure Noise Estimator?

**Verdict: MODIFY. V14 is enough for internal strategy, not enough for formal variance.**

V14 is a clean and useful single-axis seed perturbation: V7 config, seed changed to 1337, full-val `n=7403`, NORMAL `36.780632`, effectively tied with V7 `36.7810`. This is strong evidence that **this particular seed change does not move the canonical mean**.

The strict statement is:

> V18.best and V18.last exceed the observed V7(seed42)-to-V14(seed1337) single rerun delta.

The statement that is too strong is:

> V18 is statistically significant at `75x` or `150x` seed std.

One seed pair cannot estimate a seed distribution or standard deviation. It can show that the old fear of huge seed drift did not materialize in this one rerun. That is enough for strategic decision-making because V18 is no longer the planned headline anyway. It is not enough for a paper sentence like "150x seed noise" unless phrased as "150x this observed single-seed perturbation".

Do we need V14b/V14c now? **No, not as a gate.** They are useful only if the paper's claim depends on precise multi-seed uncertainty. The current main claim should not be V18 significance; it should be image_aux dominance and decoupling, where the V13 effect is ~`0.29 dB` and much larger than any plausible seed wobble suggested here.

---

## 2. Q2 - Did V13 Clear the V7-V8 Confound?

**Verdict: APPROVE for the main story; MODIFY the Gronwall wording.**

V13 is the clean control the project needed: same V7 setup, same seed, image_aux disabled. The result is decisive enough for the main question:

- V7 - V13 ~= `+0.2867 dB` NORMAL.
- V7 - V8 historical ~= `+0.3081 dB` NORMAL.
- V13 - V8 ~= `+0.0214 dB` NORMAL.

So the old V7-V8 gap was mostly image_aux. The residual `~0.02 dB` is small relative to image_aux and is confounded by V8's older train config / Plan F lineage, so it should not trigger another isolation run.

The wording I would use:

> The V7-V8 gap is overwhelmingly explained by image_aux. Any residual step-weight / historical-config effect is second-order at about `0.02 dB` and is not a main project lever.

Do not say:

> Gronwall step_weights are proven exactly zero.

Do update historical narrative. Any old sentence saying "Gronwall step_weights are the transport design victory" should be withdrawn or moved behind "not supported after V13".

---

## 3. Q3 - What Does the V18 `+0.0617 dB` Signal Mean?

**Verdict: real enough to keep, too small and too reinterpreted to headline.**

Given V14, V18.last's `+0.0617 dB` looks like a real mean shift under the observed single-seed perturbation. But two constraints dominate:

1. It is much smaller than image_aux: `0.0617 / 0.2867 ~= 0.22`.
2. Round16 removed the KL mechanism explanation and showed early V18 behavior is capacity-explainable.

Therefore V18 should be written as:

> a small, real decoder-capacity / long fine-tuning effect whose main value is diagnostic: direct decoder improvements and chain improvements are weakly coupled.

It should not be the paper's main result. It belongs in an ablation or mechanism section, where it supports the decoupling story:

- decoder LoRA improves direct `decode(z_GT)` by about `+0.17 dB`,
- only a small fraction reaches canonical rollout,
- KL pullback adds no measurable contribution in current V18.

I would not launch V18-seed-control before deciding strategy. If a reviewer later challenges the `+0.0617 dB` line specifically, a V18 seed-control is a revision experiment, not a current blocker.

---

## 4. Q4 - Strategic Option

**Main verdict: Hybrid B+A4. Backup: B paper-as-is.**

### Why Hybrid B+A4

The substrate now says the only robust positive lever is image_aux. A4 is the only proposed new run that follows directly from that evidence rather than from sunk cost:

- image_aux on/off is `~0.29 dB`, the largest verified lever;
- current image_aux is likely low-level / hop0-weighted and may not be saturated;
- an A4 schedule or hop0-strength probe is cheap relative to architecture pivot;
- writing can start immediately, so A4 does not delay the paper unless it succeeds.

Pre-register A4 as a single probe, not a new sweep campaign:

| item | recommendation |
|---|---|
| experiment | image_aux schedule / hop0-strength / ROI-aware variant, one run |
| success threshold | `+0.05 dB` NORMAL over V7/V14-style baseline, plus no D20 regression |
| failure policy | if < `+0.05 dB`, freeze paper as B |
| no-go | no multi-run image_aux fishing unless the first probe clears threshold |

### Why not A transport intervention as main

Transport is likely the unresolved path, but A1/A2/A3 are broad and can become a new project. They need design work, not a reflex launch. A4 is the narrow transport-adjacent action justified by the data.

### Why not C / D / E

- C architecture pivot may be the next project, but it is too large to treat as a Round17 follow-up.
- D data pivot has no evidence yet that data size is the bottleneck.
- E terminate is too pessimistic because the current package is already paper-shaped as a rigorous negative/diagnostic study.

### Why B remains the backup

Even without A4, the current package has a coherent story: true image_aux effect, true seed control, A3 negative control, V18 decoupling. That is not a blockbuster method paper, but it is a writeable ablation / feasibility paper.

---

## 5. Q5 - Paper Narrative

**Verdict: combine B' + C', with D' as the honesty guardrail.**

Best narrative:

> Image auxiliary supervision is the dominant verified design choice in PET latent transport, while decoder adaptation and KL pullback expose a structural decoupling between decoder-manifold improvements and rollout-chain quality.

This is stronger than pure feasibility and more honest than a V18-centered method claim.

Claim structure:

| role | claim |
|---|---|
| main positive result | V13 confirms image_aux contributes about `+0.2867 dB` NORMAL under the V7 configuration. |
| main diagnostic result | A3/V18 show decoder capacity improves direct GT-manifold decode but transfers weakly to chain. |
| negative control | KL pullback in current V18 has no measurable direct-decoder contribution. |
| calibration | V14 shows the V7 seed1337 rerun is nearly identical to V7 on canonical NORMAL mean, but not a full variance estimate. |
| honest ceiling | The project-level improvement is small; present it as a PET latent transport feasibility and bottleneck analysis. |

Venue fit:

| venue | fit |
|---|---|
| ISBI | best fit as-is or with no more than A4; ablation/feasibility story can land if writing is crisp. |
| MICCAI | possible if A4 adds a clean improvement or if the diagnostic framing is unusually strong with clinical/ROI metrics. |
| IPMI | possible for the structural decoupling / latent-representation analysis angle, but needs sharper theory or statistics. |
| TMI / Med Image Anal | unlikely as-is; would need stronger clinical validation, multi-seed / external data, or a method improvement beyond `~0.1 dB`. |

Extra experiments to strengthen, in order:

1. A4 image_aux schedule / ROI-aware probe.
2. no-GPU paired bootstrap / win-rate / CI tables from existing per-slice outputs.
3. ROI/SUV metrics if available without retraining.
4. V14b/V14c only if targeting a venue/reviewer likely to demand seed envelopes.

---

## 6. Q6 - Missing Controls and Priority

**Verdict: only A4 and analysis-only statistics are high EV now.**

Ranking:

| candidate | priority | decision |
|---|---|---|
| A4 image_aux schedule / hop0-strength / ROI-aware probe | **HIGH** | Worth one pre-registered run because V13 identified image_aux as the main lever. |
| paired bootstrap / win-rate / CI from existing per-slice CSVs | **HIGH, no GPU** | Do immediately for paper tables. |
| ROI/SUV clinical metrics on existing outputs | **MEDIUM-HIGH, no new training if possible** | Strengthens medical venue fit more than another tiny PSNR run. |
| V14b/V14c | **MEDIUM** | Useful for paper robustness, but should not block strategy. Run only if drafting reveals seed variance is a central claim. |
| V13 + decoder LoRA | **LOW-MEDIUM** | Interesting mechanistically, but it answers a niche interaction after image_aux is already known dominant. |
| V18 + seed=1337 | **LOW** | Only needed if V18 is promoted beyond ablation, which I do not recommend. |
| V19 corrected KL (`use_pred_latent=false`) | **REJECT** | Round16 killed the current KL story; corrected KL is a new rescue hypothesis with poor ROI. |

The smallest viable next package is:

1. Start paper draft and claim ledger now.
2. Run A4 if one GPU-week is acceptable.
3. Do analysis-only statistics regardless.
4. Defer seed expansion until a concrete submission target or reviewer risk justifies it.

---

## 7. Q7 - New Biases B69+

### B69 - Single-seed SNR inflation

The prompt says V18 is `75x` / `150x` over V14 noise. This is numerically true only if V14's `0.0004 dB` is treated as a noise scale. It is not a std; it is one observed perturbation.

Fix: say "exceeds the observed single-seed perturbation" and avoid formal SNR language unless more seeds exist.

### B70 - Image_aux overclaim risk

V7 - V13 is clean image_aux, but V14 - V13 mixes image_aux plus seed. Use V7 - V13 as the primary image_aux estimate; use V14 - V13 only as a sanity check.

Fix: headline `V7 - V13 ~= +0.2867 dB`, not `V14 - V13` alone.

### B71 - Gronwall exact-zero overreach

V13 - V8 being `0.0214 dB` does not prove step_weights are zero. It proves they are not the main lever in this historical comparison.

Fix: retire Gronwall as a main story; do not claim mathematical null effect.

### B72 - V18 sunk-cost headline bias

Because V18 took many rounds, there is a temptation to keep it central. The data say V18 is small and mechanistically reinterpreted.

Fix: V18 should be an ablation/diagnostic section unless a new result changes its scale.

### B73 - Paper-as-is defeatism

The opposite risk is treating `+0.099 dB` total optimization as a failed project and terminating too early. The ablation package is scientifically useful even if the method gain is small.

Fix: write it as a bottleneck/feasibility study, not as a failed leaderboard chase.

### B74 - Hybrid scope creep

Hybrid B+A4 can quietly become "paper plus indefinite experiments".

Fix: one A4 run, one threshold, one stop rule. If it fails, write B.

---

## 8. Output Summary by Prompt Format

### 8.1 Seven Questions

| Q | verdict |
|---|---|
| Q1 V14 noise | **MODIFY** - enough for strategy; not enough for formal std/significance. |
| Q2 V13 confound | **APPROVE main claim** - V7/V8 gap mostly image_aux; residual `~0.02 dB` not worth isolating now. |
| Q3 V18 signal | **MODIFY** - real enough, but ablation-only / decoupling diagnosis, not headline. |
| Q4 strategy | **Hybrid B+A4**, backup **B**. |
| Q5 paper narrative | **B' + C' with D' guardrail** - image_aux dominance plus structural decoupling. |
| Q6 missing controls | **A4 + no-GPU stats high EV; V14b/c optional; V19 reject.** |
| Q7 biases | **B69-B74** above. |

### 8.2 Strategic Verdict

Main: **Hybrid B+A4**.

Backup: **B paper-as-is**.

Do not choose full A/C/D/E yet. A4 is the only narrow new run justified by the finished substrate; broader transport or architecture work should become a next project or post-paper phase.

### 8.3 V18 Signal 判定

**Real but should be reframed as decoupling diagnosis.**

It should not be headlined as a main method result, and it should not be used to rescue KL. It supports the paper's diagnostic claim that decoder-manifold gains do not reliably transfer into rollout-chain gains.

### 8.4 Immediate Executable Now Without GPU

1. Update claim ledger with four buckets: supported image_aux, rejected KL, diagnostic V18/A3, pending optional A4.
2. Start paper outline around image_aux dominance and decoder/transport decoupling.
3. Remove old Gronwall-as-main-gain wording from narrative docs.
4. Generate no-GPU paired stats / bootstrap CIs / win rates for V7, V13, V14, V18 where per-slice outputs exist.
5. Draft A4 preregistration with a `+0.05 dB` threshold and stop rule; do not launch a sweep.

Pause:

1. V19 / corrected KL.
2. V18 seed-control.
3. V14b/V14c unless paper target demands a seed envelope.
4. Architecture pivot decisions until paper strategy is explicit.

---

## 9. Bottom Line

Round17 resolves the main pending question from Round16: **the strongest current result is image_aux, not V18 and not KL.** V13 makes image_aux the dominant verified design choice; V14 removes the immediate fear that one seed rerun could erase all small effects, but it does not create a formal seed distribution. V18 is probably real in the observed mean, but strategically it is a small diagnostic ablation.

The project should now stop chasing KL and stop treating V18 as the main story. Write the paper around image_aux dominance and decoder/transport decoupling, with one carefully bounded A4 probe only if it can run in parallel without delaying the draft.