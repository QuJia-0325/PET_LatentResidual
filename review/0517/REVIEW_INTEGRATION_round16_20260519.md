# REVIEW INTEGRATION — Round 16 (A3 Results Interpretation)

- date: 2026-05-19
- scope: integrate independent Round16 reviews for A3 capacity-only result interpretation
- sources:
  - PEER_REVIEW_round16_review_D_20260519.md
  - PEER_REVIEW_ROUND16_REVIEWER_A_20260519.md
  - PEER_REVIEW_round16_A3_results_copilot_20260519.md
  - PEER_REVIEW_round16_reviewer_C_20260519.md

---

## 1) Consensus (high confidence)

### 1.1 Main verdict

All reviewers converge on **A**:

> A3 is sufficient to retire the current V18 "KL explains direct decode(z_GT) gain" narrative.

Core evidence accepted by all reviewers:

1. Matched-step direct probe tie:
   - `V18-cap.last(170K) - V18.step170k = +0.0020 dB` (NORMAL)
   - D20/D10/D4 are all ~0.001 dB level.
2. B42 mechanism remains valid:
   - with `use_pred_latent=true`, KL is applied on `z_pred` path, not direct `z_GT` path.
3. A3 165K chain metrics are essentially tied to V18.best@165K:
   - `val_select_score`: 0.0009037353 vs 0.0009037494
   - `val_chain_normal_mse`: 0.0002454921 vs 0.0002454947

Interpretation consensus:

- Current V18 direct GT-manifold gain should be attributed to decoder LoRA capacity (for this configuration), not KL pullback.
- Do not use direct `decode(z_GT)` improvement as evidence of KL design success.

### 1.2 What is safe to conclude now

1. The KL direct-decoder story for current V18 should be withdrawn.
2. Capacity-only is sufficient to reproduce the observed matched-step direct gain.
3. Early matched-step chain behavior is also capacity-explainable on trainer full-val substrate.
4. Launching new long KL-specific controls is not top priority right now.

### 1.3 What must still wait for V13/V14

1. Whether small chain deltas exceed d_pure / seed noise floor (V14).
2. True image_aux single-variable contribution (V13).
3. Project-level ceiling/pivot decision.
4. Final paper posture (positive/marginal/negative framing).

---

## 2) Meaningful disagreements (medium confidence)

### 2.1 How strong is A3 beyond direct probe?

- Reviewer A and Reviewer C argue A3 impact is stronger than prompt framing, because matched-step chain metrics are also essentially tied.
- Review D and Copilot review keep a narrower framing, emphasizing direct-probe falsification first and chain caution second.

Integration resolution:

- Accept the stronger factual statement, but keep substrate labeling strict:
  - direct GT-manifold probe: strong falsification of KL narrative;
  - trainer chain MSE at matched step: near-tie supporting capacity-only explanation;
  - canonical chain significance: still pending broader context (V13/V14).

### 2.2 Should transport be declared the bottleneck now?

- Some reviews promote transport as the likely main unresolved path.
- Reviewer C warns A3 alone cannot uniquely prove transport-only bottleneck (could be joint saturation or path mismatch worlds).

Integration resolution:

- Use cautious wording: "decoder-capacity gains do not reliably propagate to chain in current regime".
- Avoid hard claim "transport is definitively the sole bottleneck" before V13/V14 and further targeted transport-side evidence.

### 2.3 V18-clean resurrection

- All reviews deprioritize V18-clean.
- Minor disagreement is whether to keep it as distant backlog vs full drop.

Integration resolution:

- Mark as **backlog-low** only, not current execution candidate.

---

## 3) Bias audit merged (B61+)

Cross-review common risks:

1. Direct-decode vs chain conflation.
2. Over-extrapolating A3 to whole-project terminal judgment.
3. Capacity-anchor overreach (turning "sufficient in this substrate" into "only cause everywhere").
4. Metric-family mixing (trainer MSE vs canonical PSNR).

Operational guardrails for Round 16+:

1. Every claim must carry substrate tag: `direct_probe`, `trainer_chain`, or `canonical_chain`.
2. No project-terminal or paper-significance claim before V13/V14.
3. No "KL success" language in any active narrative/document.

---

## 4) Integrated action list

### 4.1 Execute now (no new long GPU)

1. Update narrative docs to remove KL-success interpretation for current V18.
2. Add explicit claim ledger section to Round16 materials:
   - Supported now
   - Rejected now
   - Pending V13/V14
3. Prepare V13/V14 result-template and significance checklist in advance.

### 4.2 Hold until V13/V14 return

1. New long KL-specific training (including V18-clean).
2. Decoder rank/block sweep campaigns.
3. Project/paper final verdict.

---

## 5) Final integrated verdict

Round16 integrated verdict remains **A** with constrained scope:

> A3 is strong enough to retire the current KL direct-decoder narrative and re-label observed early V18 gains as capacity-explainable. This is a local mechanism correction, not a project-terminal judgment. Strategic decisions that depend on significance and attribution still require V13/V14.

---

## 6) Suggested Round17 trigger condition

Start Round17 integration immediately after both conditions are met:

1. V13 has complete full-val outputs.
2. V14 has complete full-val outputs and noise-floor estimate.

Round17 should answer exactly three questions:

1. Is V18-family chain delta above d_pure?
2. How much of historical gains are true image_aux effect?
3. Given 1+2, continue transport intervention or pivot narrative?
