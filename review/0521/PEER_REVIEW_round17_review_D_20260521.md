# Peer Review Round 17 - Review D Audit

- date: 2026-05-21
- reviewer: **Review D (GitHub Copilot)**
- scope: Post-V13/V14 strategic decision
- prompt: `PEER_REVIEW_PROMPT_round17_strategy_post_V13V14_20260521.md`
- independence note: I did not read any Round17 reviewer drafts; this review is based on the prompt substrate and upstream signed artifacts only.

---

## 0. Executive Verdict

**Strategic verdict: Hybrid B+A4-light.** Start writing the paper now under the honest paper-as-is framing, and run at most one cheap, pre-registered image_aux schedule probe in parallel if GPU time is otherwise idle. Do not make A4 a decision gate for whether the current project is worth writing.

**Backup verdict: B paper-as-is.** If the team wants to avoid another training loop, the evidence is already sufficient for a coherent, publishable but modest ablation/diagnosis paper.

**V18 signal verdict: real but should be reframed as decoupling diagnosis / ablation-only.** V18.last is above the observed V14 single-seed perturbation, but it is too small and too mechanism-confounded to headline. The KL direct-decoder story is dead by Round16; V18 now belongs in a "capacity/transport decoupling" section, not in the main success claim.

The strategic center of gravity has moved: **image_aux is the dominant clean positive result; decoder LoRA/KL is a diagnostic result; Grönwall step_weights are no longer a defensible main win.**

---

## 1. Substrate Ledger

I use the following substrate labels:

- `canonical_chain`: full-val `PSNR_clip3` / chain MSE from `eval_first_hop_fullval_psnr_chain_mse.py`, n=7403.
- `trainer_chain`: training `val_full` chain MSE records.
- `direct_probe`: direct `decode_crop(z_GT)` probe, not rollout.

Accepted facts:

| Claim | Substrate | Value | Review D reading |
|---|---|---:|---|
| V13 true image_aux off NORMAL | canonical_chain | 36.494330 dB | image_aux removal is harmful |
| V14 true d_pure NORMAL | canonical_chain | 36.780632 dB | V7 seed perturbation observed here is near zero |
| V7 baseline NORMAL | canonical_chain | 36.7810 dB | V14 essentially ties it |
| V18.best NORMAL | canonical_chain | 36.8112 dB | +0.030 dB over V7 |
| V18.last NORMAL | canonical_chain | 36.8426 dB | +0.062 dB over V7 |
| A3 capacity-only vs V18@170K direct probe | direct_probe | +0.002 dB NORMAL | capacity explains direct GT-manifold gain |

The key comparison is not "is V18 nonzero?" but "what story can survive after A3, V13, and V14 are all accepted?" My answer: image_aux survives as a clean design lever; V18 survives as a small decoupling/ablation signal; KL success does not survive.

---

## 2. Seven Questions

### Q1 - Is V14 enough as d_pure noise floor estimator?

**Verdict: MODIFY.**

V14 is enough for the strict statement: **V18.best/last exceed the observed V7 seed42->1337 perturbation under the same canonical_chain protocol.** It is not enough to estimate a seed-noise distribution or a standard deviation.

The prompt's SNR table is directionally useful but numerically overconfident. A single `|V14 - V7| ~= 0.0004 dB` should not be treated as `sigma`. It is one draw. A more honest wording is:

> Under the one completed seed perturbation, V7 and V14 are effectively tied; therefore V18's +0.03/+0.06 dB is above the observed perturbation, but multi-seed significance remains unmeasured.

I do **not** require V14b/V14c for the strategic decision. I would require them only if the final paper wants to claim precise seed-robust effect sizes for deltas in the 0.03-0.06 dB band.

### Q2 - Does V13 +0.29 dB clear the V7-V8 confound?

**Verdict: APPROVE with wording limits.**

Yes, V13 is strong enough to close the old V7-V8 confound for strategic purposes. V13 and V8 are both near 36.49/36.47 while V7/V14 are near 36.78. That makes the practical conclusion clear: **the historical V7-V8 drop was dominated by image_aux removal, not by Grönwall step_weights.**

However, I would not write "Grönwall step_weights are in the V14 noise floor." The V13-V8 difference is about 0.021 dB and is not a pure Grönwall contrast because V8 and V13 also differ in training context/config lineage. The safe claim is narrower and stronger:

> After the true V13 single-axis ablation, image_aux accounts for almost all of the old V7-V8 gap at the scale that matters for project decisions; any residual Grönwall/config effect is small enough that it should not be a headline mechanism.

Yes, immediately update Plan-F / V18 design rationale docs to demote old "Grönwall transport win" language.

### Q3 - What does V18 +0.06 dB mean under V14?

**Verdict: MODIFY.**

V18.last - V7.best = +0.0617 dB is likely a real canonical_chain mean shift relative to the observed V14 perturbation. But it is not a main result.

Three constraints keep it out of the headline:

1. The effect is about one fifth of the clean V13/V14 image_aux contrast.
2. A3 already showed the direct_probe improvement is capacity-explainable, not KL evidence.
3. V18 combines decoder LoRA, KL config, warm-start continuation, and longer training from V7; without V18 seed/control and plain continuation controls, the mechanism is not clean enough.

The best interpretation is:

> V18 gives a small positive chain delta, while A3 shows decoder GT-manifold gains transfer poorly through rollout. This supports a structural decoder/transport decoupling diagnosis, not a new transport method claim.

I would not run V18-seed-control before deciding strategy. I would only run it if the team insists on presenting V18 as a positive quantitative result rather than an ablation.

### Q4 - Strategic option recommendation

**Verdict: Hybrid B+A4-light; backup B.**

Recommended path:

- Start paper draft immediately under B.
- In parallel, run one A4 image_aux schedule probe only if it is already simple to launch and pre-registered.
- Stop after that probe unless it clearly beats a pre-declared threshold, e.g. `>= +0.05 dB NORMAL PSNR_clip3` over V7/V14 under canonical_chain.

Why not broad A transport intervention as main path: the last several rounds show that design changes are producing +0.03 to +0.10 dB class shifts, while the strongest clean effect was a supervision anchor already known to matter. A broad transport campaign may be scientifically interesting, but it is a new phase, not required to salvage the current paper.

Why not C architecture pivot: a 4-8 week architecture reset is effectively a new project. It may be the right next research direction after this paper, but it should not block documenting the current ablation story.

Why not D data pivot: no evidence yet that data quantity is the current binding constraint; it has high operational cost and weak attribution.

Why not E terminate only: too pessimistic. The current work has a real systematic ablation story: image_aux dominates, decoder capacity and transport quality decouple, and KL pullback failed in an interpretable way.

Custom F, if desired: **B + no-GPU claim hardening.** Add paired CI/bootstrap tables, ROI/clinical metric panels if already computable, and a claim ledger before any further training.

### Q5 - Paper narrative

**Verdict: MODIFY the menu; combine B' and C'.**

The most honest and publishable narrative is not purely "image_aux main result" and not purely "decoupling main result." It is:

> A systematic ablation study of PET latent transport showing that pixel-space image_aux is the dominant reliable positive lever (+0.29 dB), while decoder capacity improves direct reconstruction but only weakly transfers through the transport chain; current KL pullback does not measurably explain the gains.

This is basically B' + C' with D' as the cautionary interpretation. Avoid a triumphant SOTA tone. The contribution is diagnostic clarity and experimental hygiene, not a large absolute PSNR breakthrough.

Venue fit:

- **ISBI / MICCAI workshop / focused MICCAI submission**: plausible if the story is framed as PET low-dose latent transport ablation and failure-mode analysis.
- **IPMI**: plausible only if the decoupling/transport analysis is made more methodological and not just empirical bookkeeping.
- **TMI / MedIA**: likely needs stronger clinical metrics, multi-dataset validation, or a larger performance gain. Current +0.099 dB one-year total optimization is thin for a flagship journal unless the diagnostic angle is unusually strong.

Extra experiments to strengthen, not gate:

- V14b/V14c if the paper leans on small deltas.
- A4 if the team wants one last low-cost chance at a stronger positive result.
- ROI/SUV/lesion metric panels if available without retraining; these may matter more for medical reviewers than another 0.02 dB PSNR run.

### Q6 - Missing controls after V13/V14

**Verdict: MODIFY priorities.**

Priority ranking:

| Rank | Candidate | Priority | Rationale |
|---:|---|---|---|
| 1 | No-GPU claim ledger + paired CI/bootstrap + substrate-tagged tables | High | Highest immediate paper value; prevents old direct/chain conflation from returning |
| 2 | A4 image_aux schedule tuning | Medium-high | Only new GPU run I would tolerate now; directly follows the strongest clean lever |
| 3 | V14b/V14c | Medium | Useful for reviewer-proofing seed variance, but not needed for strategy |
| 4 | ROI/SUV/clinical metric evaluation | Medium | May strengthen medical relevance more than another tiny PSNR delta |
| 5 | V18 seed=1337 | Low-medium | Needed only if V18 is promoted beyond ablation-only |
| 6 | V13 + decoder LoRA | Low | Interesting interaction test, weak paper leverage unless current narrative changes |
| 7 | V19 corrected KL (`use_pred_latent=false`) | Low | KL story is not worth reopening before paper; keep as future-method backlog |

If the team runs only one GPU experiment, pick A4. If the team runs no GPU experiment, the paper still has a viable core.

### Q7 - New bias audit B61+

**Verdict: APPROVE bias concerns; add more.**

New bias ledger:

- **B61 - Single-seed SNR precision bias.** The prompt's 75x/150x SNR language makes a one-draw seed perturbation sound like a measured variance distribution.
- **B62 - Zero-noise anchor bias.** V14 ~= V7 may be unusually lucky; do not hard-code `0.0004 dB` as the project's universal noise floor.
- **B63 - Image_aux monopoly bias.** V13 strongly supports image_aux dominance, but V13 vs V8 still contains residual config/history differences; avoid claiming every historical delta is exactly image_aux.
- **B64 - V18 sunk-cost bias.** Because V18 took substantial engineering, there is pressure to headline it. The data do not justify that; it is ablation evidence.
- **B65 - Venue ambition bias.** TMI/MedIA framing may push overclaiming. The current evidence is more natural for a focused conference/workshop or a carefully scoped methods note.
- **B66 - Strategy-menu anchor bias.** The menu underrepresents "paper now + no-GPU claim hardening" as its own path.
- **B67 - Metric monoculture bias.** PSNR_clip3 is canonical here, but medical imaging reviewers may ask whether +0.29 dB matters clinically. ROI/SUV panels should be considered before more tiny PSNR hunts.
- **B68 - Free-GPU pressure bias.** Empty slots should not automatically create new experiments; the claim structure should decide compute, not the other way around.
- **B69 - A4 optimism bias.** Because image_aux is the clean positive result, schedule tuning feels attractive, but it may just overfit the same small anchor. Pre-register the threshold and stop rule.

---

## 3. Strategic Verdict Details

### Main: Hybrid B+A4-light

This is the best EV path because it preserves momentum toward a paper while allowing one final low-cost upside probe. The paper should not wait for A4 unless the team pre-commits that A4 is the final run and the launch cost is low.

Minimum A4 standard:

- Same canonical_chain evaluator, full val n=7403.
- Pre-register primary endpoint: NORMAL `PSNR_clip3` vs V7/V14.
- Pre-register success threshold: at least `+0.05 dB` over V7/V14, or an equivalent MSE threshold, before narrative upgrade.
- If A4 is below threshold, do not launch A4b/A4c; fold it into negative/neutral ablation.

### Backup: B paper-as-is

If the team is tired of this loop, B is defensible today. The draft can center on:

1. true image_aux effect (`canonical_chain`): +0.286 dB vs V13;
2. direct decoder capacity effect (`direct_probe`): capacity explains V18 direct decode gain;
3. chain transfer failure (`trainer_chain` + `canonical_chain`): direct decoder improvements weakly propagate;
4. KL negative control: current pullback configuration is not supported;
5. methodology: the project corrected confounds through V13/V14/A3 rather than narrating stale comparisons.

---

## 4. V18 Signal Judgment

Selected category: **real but should be reframed as decoupling diagnosis**.

I reject both extremes:

- Not "real and worth headlining": effect too small, mechanism too dirty, A3 falsifies the direct KL explanation.
- Not "borderline, need V18-seed control" for strategy: V18 does not need to be the main claim, so seed-locking V18 is not a blocker.

The paper wording should be something like:

> Decoder LoRA/KL variants produced at most a small rollout improvement (+0.03 to +0.06 dB), while matched capacity-only direct-probe controls showed that direct decoder gains were capacity-explainable. This indicates a decoupling between decoder manifold quality and transport-chain quality in the current PET latent flow setup.

Do not write:

- "KL improves PET latent transport."
- "V18 is the main performance breakthrough."
- "V18 proves decoder adaptation transfers to rollout."

---

## 5. Immediate Actions Without GPU

Do now:

1. Update V18 design rationale and any paper notes to remove KL-success language.
2. Update Plan-F / V7-V8 historical narrative: old V7-V8 was confounded; V13 is the true image_aux ablation.
3. Create a substrate-tagged claim ledger with columns: claim, metric family, substrate, evidence file, allowed wording, forbidden wording.
4. Draft the paper outline around "image_aux dominance + decoder/transport decoupling + KL negative control."
5. Add tables that keep `direct_probe`, `trainer_chain`, and `canonical_chain` separate.
6. If A4 is chosen, write its pre-registration before launch.

Pause until reviewer consensus:

- broad architecture pivot;
- data pivot;
- V19 corrected KL;
- V18 seed-control;
- decoder-rank or block sweeps.

---

## 6. Final Answer to the Strategic Question

The project should **paper-and-pivot, with one optional A4-light probe**, not continue an open-ended transport campaign and not terminate as a failed project.

The publishable unit is not "we found a big new PET transport method." It is:

> We systematically deconfounded a PET latent transport pipeline and found that image auxiliary supervision is the dominant reliable lever, while decoder capacity, KL pullback, and transport rollout quality are structurally decoupled in the current frozen-decoder latent-flow regime.

That story is smaller than the original ambition, but it is clean, useful, and much more credible than trying to stretch +0.0617 dB into a headline.