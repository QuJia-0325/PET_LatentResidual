# Peer Review Round 18 — Reviewer C Strategic Report

- date: 2026-05-25
- reviewer: Reviewer C
- scope: Round 18 strategic next step after A4 bracket
- reviewed file: `ROUND_18_A4_BRACKET_ANALYSIS_20260525.md`
- constraint: report-only review; no code or existing artifact modifications

## 0. Reviewer C Verdict

**Main verdict: rewrite the paper around image_aux now, and run a bounded X1-lite mechanism package before choosing any large new research branch.**

My recommended next step is **Hybrid: Paper-now + X1-lite + X5 feasibility check**.

- **Paper-now**: immediately demote V18 to a secondary ablation and make stronger image-space supervision the headline.
- **X1-lite**: do not run a full 3-run l1/ssim/seam grid yet. First do one mechanism probe plus one most-diagnostic ablation run at λ=0.08.
- **X5 feasibility check**: start a data-availability audit for cross-tracer/cross-dataset validation, but do not block the current paper on it.

Backup: if the user rejects any mechanism ablation as “调小参”, choose **X5 feasibility + paper draft**, and defer X2 architecture work to a follow-on project rather than using it to delay this paper.

I do **not** recommend immediately starting X2 architecture pivot. It is too large and too failure-prone for a project that now has a plausible publishable core. I also do not recommend X3 as the next default run; it risks resurrecting the V18 family before we understand why A4 worked.

## 1. Q1 — Is A4-mid +0.113 dB Real?

**APPROVE with statistical-language modification.**

A4-mid is very likely a real practical signal, not ordinary seed luck. The reasons are stronger than just “283× V14 noise”:

- A4-mid and A4-low are controlled V7 clones with the same seed and only image_aux strength changed.
- A4-mid improves all rollout endpoints, not only NORMAL: D20, D10, D4, and NORMAL move in the same direction.
- A4-low moves in the opposite direction from A4-mid, which makes the bracket internally coherent.
- best.pt and last.pt coincide at step 160000, so the reported A4-mid result is not a selected early checkpoint artifact.

However, I would not use the V14 single-seed |Δ|=0.0004 as a formal noise standard deviation. It is a useful sanity point, not a seed distribution. The safe paper language is:

> In a controlled same-seed full-validation comparison, increasing image_aux strength from 0.04 to 0.08 improves NORMAL PSNR_clip3 by +0.113 dB over V7, with consistent improvements across D20/D10/D4/NORMAL. This is single-seed and slice-level evidence, not patient-level statistical inference.

No extra ablation is needed to claim the observed full-val result. Extra work is needed only if the paper wants to claim mechanism, generality, or seed robustness.

## 2. Q2 — Which Mechanism Is Most Likely?

**MODIFY. The listed M1-M4 are plausible, but M1+M4 is the leading hypothesis.**

My ranking:

1. **M1: pixel-space gradient supplies a stronger training signal.** The response to λ is large, and A4-low being worse than V7 suggests the amount of pixel supervision matters, not merely the presence of a regularizer.
2. **M4: decoder-manifold anchoring.** This fits the broader evidence that direct decoder capacity and transport-chain performance are decoupled. image_aux may be teaching transport to stay in regions the frozen decoder renders well.
3. **M2: SSIM structure term.** Plausible, but not proven. It should not be promoted without attribution evidence.
4. **M3: seam artifact control.** Lowest prior. The gain appears across chain metrics and timepoints, not just as an artifact-boundary cleanup story.

The minimum disambiguation package should be:

- **Probe**: on fixed V7/A4-mid checkpoints, decompose image_aux gradients by l1, ssim, and seam terms; report gradient norm, gradient cosine with latent MSE gradient, and per-timepoint correlation with per-slice PSNR gain.
- **One run**: λ=0.08 **l1-only** or “no-SSIM/no-seam” ablation, keeping the same total training setup. If l1-only preserves most of A4-mid, M1/M4 dominate. If it collapses, M2/M3 become credible.

This is not a sweet-spot sweep. It is causal attribution for the paper’s main mechanism.

## 3. Q3 — Which X Direction Under “No Small Tuning”?

**Main: Hybrid X1-lite + X5 feasibility. Conditional: X3 only after X1-lite. Reject immediate X2/X4.**

My priority ranking:

1. **X1-lite mechanism package**. The full X1 proposal of three ablation runs is too grid-like. A one-run plus one-probe version is the best EV/cost path because it converts “λ=0.08 worked” into a publishable explanation.
2. **X5 feasibility check**. If a second PET tracer/dataset is actually accessible, it is the strongest upgrade path for venue quality. But data access should be checked before assigning GPU time.
3. **X3 conditional additive test**. Run only if X1-lite suggests decoder-manifold anchoring is the core mechanism. Otherwise X3 is likely a V18.v2 distraction.
4. **X2 architecture pivot**. Keep as a next-project idea. It is too ambitious for the current paper and has a high chance of consuming 1-2 months without a usable result.
5. **X4 chain redesign**. Deprioritize. It breaks comparability with V13/V14/V18/A4 and reopens the whole substrate.

Why X1-lite does not violate “不调小参”: it is not searching for a better λ or better weights. It is answering which term/mechanism made the observed result happen. That is a mechanism-control experiment, not parameter tuning.

## 4. Q4 — V18 Role In The Paper

**Choose (b): V18 as secondary ablation.**

The paper headline should move to image_aux immediately. V18 should not carry the main story, and X3 should not be used to keep V18 emotionally alive.

Recommended V18 role:

- Include V18 as evidence that decoder LoRA/KL engineering produced smaller chain gains than image_aux strength.
- State that Round 16/A3 retired the KL direct-decoder narrative.
- Use V18 to support the structural claim that decoder capacity changes do not automatically translate into transport-chain gains.

Do not fully remove V18 from the paper. It is useful as a negative/secondary ablation because it explains what the project tried and why the final story changed. But do not run further V18-family experiments unless X1-lite specifically points to decoder-manifold anchoring as the bottleneck.

## 5. Q5 — Missing Direction X6+

Two missing directions are more important than another architecture brainstorm:

### X6 — Clinically stratified evaluation

Before claiming the result is medically meaningful, analyze where the +0.113 dB comes from:

- lesion-like high-uptake regions vs background
- D20/D10/D4/NORMAL stage-specific gains
- slice intensity/SUV strata
- qualitative failure cases where A4-mid improves or worsens structure

This may not require new training, but it can prevent a PSNR-only paper from feeling shallow.

### X7 — Decoder-aware latent loss / Jacobian-weighted transport

If image_aux works because latent MSE is misaligned with pixel reconstruction, the substantive method is not “increase λ”; it is to derive a decoder-aware transport objective. A practical version could approximate the decoder pullback metric or use image_aux gradient alignment as a preconditioner. This is more paper-worthy than a raw λ sweep and less risky than replacing the whole transport architecture.

## 6. Q6 — Paper Timeline

**MICCAI 2027 is feasible if the team stops opening large design branches now.**

Recommended schedule:

- **Now**: start paper outline, methods, experiment table skeleton, and claim ledger.
- **Next 2-3 weeks**: finish X1-lite probe + one ablation run; add clinical stratified evaluation if data annotations or intensity strata are available.
- **Next 1 month**: draft main result and ablation sections with image_aux as headline and V18 as secondary ablation.
- **Only after draft skeleton is real**: decide whether X5 cross-dataset is feasible for a stronger venue version.

TMI/MedIA becomes attractive only if X5 is truly available or X7 becomes a principled method. X2 architecture pivot should be treated as a follow-on paper or a major revision path, not as the default next step before writing.

## 7. Q7 — Bias Audit B87+

| ID | bias | severity | reviewer C assessment | mitigation |
|---|---|---|---|---|
| B87 | A4-mid romanticization | HIGH | +0.113 is real enough to reframe, but not enough to claim seed/patient-level truth | Use fixed-seed full-val wording; no patient-level p-values |
| B88 | image_aux fetish | HIGH | The project can over-anchor on image_aux and dismiss architecture/data too quickly | Use X1-lite to explain, X5 to test generality |
| B89 | “No tuning” overcorrection | MED | Rejecting all ablations would harm the paper; mechanism ablation is not tuning | Define X1-lite as causal attribution, not search |
| B90 | Architecture pivot optimism | HIGH | X2 can consume months and still fail | Keep X2 as follow-on, not current-paper blocker |
| B91 | V18 pendulum swing | MED | V18 should not be headline, but removing it loses useful negative evidence | Keep as secondary ablation |
| B92 | PSNR monoculture | MED | A4 may improve PSNR without clinically meaningful gains | Add X6 stratified/qualitative analysis |
| B93 | Single-seed denominator misuse | HIGH | V14 \|Δ\| is not a σ estimate | Avoid “283× noise” as formal significance language |

## 8. Final Action Recommendation

1. **Start the paper now** with image_aux as the main result and V18 as a secondary ablation.
2. **Run X1-lite**, not full X1: one gradient/mechanism probe plus one l1-only or no-SSIM/no-seam λ=0.08 ablation.
3. **Run X6 evaluation immediately if possible** because it strengthens medical credibility without opening a new model branch.
4. **Start X5 data feasibility in parallel**, but do not block the current paper on data acquisition.
5. **Do not launch X2/X4 now**, and do not launch X3 unless X1-lite specifically motivates a decoder-manifold follow-up.

This path respects the user’s “no more small parameter tuning” constraint while still doing the minimum mechanism work needed to keep the paper from becoming “we doubled a YAML coefficient and PSNR went up.”
