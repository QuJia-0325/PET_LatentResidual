# Round 2 — Merged Reviewer B+D: Story Critic + Causal Critic

Date: 2026-04-24  
Mode: NIGHTMARE, Round 2 / 5+  
Reviewer Fusion: B (Story Critic) + D (Causal Critic)  
Rationale: Round 1 verdicts **Story BROKEN** and **Causal NOT IDENTIFIED** are mutually reinforcing — the story's central claim ("hop0 mechanism fixes first-hop bottleneck") cannot survive without causal evidence, and the absence of causal evidence exposes the story as unfalsifiable.

---

## Anticipated author rebuttals

The repo leadership will attempt three rebuttals in Round 3. This review pre-empts each with explicit acceptance/rejection criteria.

---

## Pre-emptive response to claim α: Causal mediation

> **Author claim α**: "The story IS correct — first-hop IS the bottleneck; the tail gains (+0.66 dB at NORMAL vs +0.18 dB at D20) are downstream propagation effects of improved hop-0 transport."

### What a causal mediation test would look like

To establish that tail gains are causally mediated by hop-0 improvements, the author would need to:

1. **Measure hop-0 quality independently**: compute `transport_PSNR(D50→D20)` vs `oracle_PSNR(D20)` — not the chained endpoint
2. **Run an intervention at hop-0 only**: compare chained NORMAL PSNR when:
   - (A) hop-0 uses learned transport, hops 1-3 use oracle latents
   - (B) hop-0 uses oracle latent, hops 1-3 use learned transport
3. **Mediation analysis**: if `PSNR_A - PSNR_baseline` >> `PSNR_B - PSNR_baseline`, then tail gains are mediated by hop-0
4. **Statistical test**: repeat with ≥3 seeds; report bootstrap CI on mediation proportion

### Strongest steelman version

> "In a 4-hop chain, a 0.01 dB error at hop-0 propagates multiplicatively through subsequent velocity predictions because each hop conditions on the previous latent. The 3.7× amplification ratio (NORMAL/D20 = 0.66/0.18) is exactly what error propagation theory predicts for a 4-step autoregressive chain where later hops have smaller baseline displacement and therefore higher relative sensitivity to upstream perturbations."

### Why it is STILL insufficient even if the test passes

1. **Observation ≠ mechanism**: Even if tail gains are mediated by hop-0, this does NOT validate the pixel forcing mechanism. Mediation shows the chain propagates error — any hop-0 improvement would produce tail gains. The question is whether **pixel forcing specifically** caused the hop-0 improvement, which requires the C↔N1 comparison to be clean and significant.

2. **C↔N1 delta is sub-noise**: The claimed hop-0 mechanism contribution (C−N1 = 0.083 dB) is smaller than:
   - Best↔last within-run variance (0.038 dB)
   - Step-to-step eval variance (~0.02–0.05 dB typical)
   - Unknown seed-to-seed variance (σ_seed unmeasured)
   
   A mediation analysis on a sub-noise effect is meaningless — you are decomposing random fluctuations.

3. **Story direction is backwards**: If hop-0 were genuinely the bottleneck AND pixel forcing fixed it, we would expect:
   - **Largest absolute gain at D20** (where the bottleneck lives)
   - **Decreasing gains downstream** (propagation dilutes)
   
   Instead we observe the opposite: D20 +0.18 → NORMAL +0.66. This is consistent with a **tail-optimized training objective** (which the chainstable config explicitly uses), NOT with a hop-0 structural fix.

### Verdict on α

**NOT ACCEPTABLE** unless:
- Mediation analysis is run with intervention design above
- AND C↔N1 effect size exceeds 3× σ_seed
- AND D20 absolute gain > NORMAL absolute gain (story-consistent direction)

---

## Pre-emptive response to claim β: Seed study coming later

> **Author claim β**: "The plateau IS the ceiling with current training budget; we need a seed study and significance test which will come in the next iteration."

### Explicit σ_seed threshold

If σ_seed > **0.04 dB**, then the C−N1 effect (0.083 dB) is within 2σ of noise — **statistically indistinguishable from zero**.

If σ_seed > **0.027 dB**, then the effect is within 3σ — **not publishable at any venue requiring significance testing**.

The 0.083 dB effect is only meaningful if σ_seed ≤ 0.02 dB — an extremely low bar that would require near-perfect reproducibility.

### Is ANY claim about hop-0 mechanism currently publishable?

**NO.**

Reasoning:
1. **No claim is publishable without uncertainty quantification**: IEEE TMI, MIA, MICCAI, NeurIPS, ICML all require either:
   - Standard deviation across seeds (minimum 3 runs)
   - Confidence interval from paired bootstrap/permutation test
   - Wilcoxon signed-rank p-value
   
2. **The repo has ZERO of these**: Every headline number (C=36.206, N1=36.123, etc.) is a single-run point estimate with no variance estimate
   
3. **"Test coming later" is never acceptable**: Variance estimation must happen BEFORE making claims, not after. A result table without error bars is raw data, not a publishable result.

### Minimum seed count and test to satisfy

| Requirement | Minimum |
|---|---|
| Seeds per config | 3 (absolute minimum), 5 preferred |
| Test | Paired bootstrap on per-sample PSNR (7,403 val slices); report 95% CI on (C − N1) |
| Threshold | CI must exclude zero; lower bound > 0 |
| Reporting | Mean ± std across seeds; CI on pairwise difference |
| Pre-registration | State hypothesis and test BEFORE running seeds |

If the bootstrap 95% CI on (C − N1) includes zero, the claim "pixel forcing helps" is dead regardless of point estimate magnitude.

### Verdict on β

**NOT ACCEPTABLE** to defer statistical analysis. The current state is:
- σ_seed: **UNKNOWN** (no multi-seed runs)
- CI on C−N1: **UNMEASURED**
- Publishability of any hop-0 claim: **NO**

---

## Pre-emptive response to claim γ: Config diffs

> **Author claim γ**: "The N1 ablation toggles more than pixel forcing — there are other hyperparameter deltas, so the 0.083 dB difference is a lower bound on pixel forcing's true contribution."

### Full config diff: `transport_v3.yaml` vs `pixenc_ablation.yaml` (N1)

| Parameter | transport_v3 (C) | pixenc_ablation (N1) | Impact |
|---|---|---|---|
| `first_hop.pixel_forcing_disabled` | absent (=false) | **true** | Primary ablation target |
| `training.best_metric_terms[0].weight` (D20) | **0.50** | 0.15 | ⚠️ Selection criterion favors D20 in C |
| `training.rollout.step_weights` | **[1.30, 1.20, 1.10, 1.00]** | [1.00, 1.10, 1.20, 1.30] | ⚠️ **INVERTED** — C prioritizes hop0, N1 prioritizes tail |
| `transport.pair_loss_weights` | **[1.20, 1.10, 1.05, 1.00]** | [1.00, 1.05, 1.10, 1.20] | ⚠️ **INVERTED** — C prioritizes hop0, N1 prioritizes tail |
| `loss.pair.velocity_rebalance.enabled` | **true** | false | ⚠️ sqrt_ratio rebalancing disabled in N1 |
| `ema` section | **present** (decay=0.9999) | **MISSING** | ⚠️ EMA disabled in N1 |
| `optimizer.first_hop_weight_decay` | 0.0 | 0.0 | Same |

### Critical finding: N1 is NOT a clean ablation

The N1 config differs from C in **at least 5 hyperparameters** beyond `pixel_forcing_disabled`:

1. **D20 selection weight reduced 3.3×** (0.50 → 0.15): N1's checkpoint selection underweights D20 quality
2. **Rollout step_weights inverted**: C trains harder on hop0, N1 trains harder on tail
3. **Pair loss weights inverted**: Same effect as above — C gets more hop0 gradient, N1 gets more tail gradient
4. **Velocity rebalance disabled**: C uses sqrt_ratio stop-gradient heuristic, N1 does not
5. **EMA disabled**: N1 reports raw weights, C reports smoothed weights — not comparable

### Does this strengthen or weaken the author's position?

**CATASTROPHICALLY WEAKENS** it.

The author cannot claim "C−N1 = pixel forcing effect" when C and N1 differ on 5+ training knobs. The observed 0.083 dB difference could be entirely explained by:
- C's hop0-prioritized loss weighting
- C's EMA smoothing
- C's velocity rebalance stop-gradient

In fact, the **inversion of step_weights and pair_loss_weights** means N1 is actively de-prioritizing hop0 during training while C prioritizes it. This alone would produce a D20 gap even if pixel forcing were a no-op.

**The N1 ablation is confounded beyond repair.** Any claim about pixel forcing contribution based on C↔N1 is methodologically invalid.

### Verdict on γ

**DAMAGING TO AUTHOR**: The config diff reveals that the "ablation" simultaneously changed:
- Loss weighting direction (hop0 → tail)
- Selection criterion (D20 weight)
- Training dynamics (velocity rebalance)
- Inference weights (EMA)

This is not an ablation — it is a different training recipe. The author's implicit claim that C−N1 isolates pixel forcing is **false**.

---

## Joint verdict (Story + Causal)

The Story and Causal critiques are now **fused** into a single indictment: **the story claims a hop0 mechanism fixes a hop0 bottleneck, but (a) the evidence shows tail gains 3.7× larger than head gains (story-inconsistent), (b) the only "ablation" (C↔N1) confounds pixel forcing with 5 other hyperparameter changes, (c) no seed variance is measured so the 0.083 dB effect is uninterpretable, and (d) 6 different configs converge to the same 36.1–36.2 dB plateau regardless of architectural choices, suggesting a structural ceiling that the hop0 mechanism cannot address.** The project is currently in a state where the central claim is unfalsifiable (no clean ablation), unsupported (effect direction wrong), and unmeasured (no σ_seed). This is not a borderline case requiring more data — the methodology is fundamentally flawed.

### Updated fused scores

| Dimension | R1 Score | R2 Score | Change | Reason |
|---|---|---|---|---|
| Story coherence | 3/10 | **2/10** | ↓1 | Config diff reveals story is not just unsupported but actively contradicted by training design |
| Claim-evidence | 2/10 | **1/10** | ↓1 | N1 ablation is confounded; no causal claim survives |
| Causal effect | 2/10 | **1/10** | ↓1 | Cannot isolate any single variable |
| Statistical rigor | 1/10 | **1/10** | — | Still zero seeds/CIs |
| Joint weighted average | — | **1.25/10** | — | (2+1+1+1)/4 |

### One condition for simultaneous PASS

Both Story and Causal verdicts could flip to PASS if and only if:

> **A single-variable ablation on pixel forcing (all other hyperparameters locked), run with ≥3 seeds, shows D20 PSNR gain > 0.5 dB with 95% CI excluding zero, AND the gain ratio D20/NORMAL > 1.0 (story-consistent direction).**

This would simultaneously establish causal effect (clean ablation + significance) and validate the story (bottleneck is at hop0, not tail).

---

## The killing experiment

A single experiment that resolves both Story and Causal concerns:

### Design: Clean pixel-forcing A/B with locked hyperparameters

| Parameter | Value (locked for both arms) |
|---|---|
| Config base | `transport_v3.yaml` |
| Seeds | {42, 123, 456} — run both arms on all 3 |
| `training.best_metric_terms[0].weight` | 0.50 (D20) |
| `training.rollout.step_weights` | [1.30, 1.20, 1.10, 1.00] |
| `transport.pair_loss_weights` | [1.20, 1.10, 1.05, 1.00] |
| `loss.pair.velocity_rebalance.enabled` | true |
| `ema.enabled` | true |
| **Only toggled knob** | `first_hop.pixel_forcing_disabled` |

### Arms

| Arm | `pixel_forcing_disabled` | Expected if story is true |
|---|---|---|
| **A (pixel forcing ON)** | false | Higher D20 PSNR, higher D20/NORMAL gain ratio |
| **B (pixel forcing OFF)** | true | Lower D20 PSNR, error propagates to tail |

### Metrics to report (per seed)

1. `val_transport_d20_psnr` (hop-0 quality)
2. `val_transport_normal_psnr` (chain endpoint quality)
3. `transport_avg_psnr` (chain average)
4. Per-hop gap: `transport_PSNR(h) − oracle_PSNR(h)` for h ∈ {D20, D10, D4, NORMAL}

### Aggregate analysis

1. **Seed variance**: σ_seed for each arm on `transport_avg_psnr`
2. **Effect size**: mean(A) − mean(B) on `val_transport_d20_psnr`
3. **Paired bootstrap CI**: 10,000 resamples on per-sample PSNR difference (7,403 slices)
4. **Gain ratio test**: (A−B)_D20 / (A−B)_NORMAL — must be > 1.0 for story-consistent result

### Stopping rule

- Complete all 6 runs (2 arms × 3 seeds)
- Report all metrics before interpreting
- No checkpoint selection — use last checkpoint for all (or mean-of-last-5)

### PASS criteria (both required)

1. **Causal PASS**: 95% bootstrap CI on (A−B) transport_avg excludes zero; lower bound > 0.05 dB
2. **Story PASS**: (A−B)_D20 > (A−B)_NORMAL (i.e., gain ratio > 1.0)

### FAIL criteria (any is terminal)

1. **Causal FAIL**: 95% CI includes zero, OR σ_seed > |mean effect|
2. **Story FAIL**: (A−B)_D20 ≤ (A−B)_NORMAL (tail gains ≥ head gains)
3. **Methodology FAIL**: Any config drift discovered between arms post-hoc

### Effort estimate

| Task | Time |
|---|---|
| Create locked ablation config (B arm) | 30 min |
| Run 6 experiments (50K steps × 6 on 1 GPU) | ~36 hours (6h each) |
| Collect per-sample PSNRs | 2 hours |
| Bootstrap analysis script | 2 hours |
| Report writing | 2 hours |
| **Total** | **~45 hours elapsed, ~6 hours active work** |

### Outcome interpretation

| Result | Interpretation | Next step |
|---|---|---|
| Both PASS | Story and mechanism validated; proceed to external baselines | Write paper |
| Causal PASS, Story FAIL | Pixel forcing helps but not via hop0 bottleneck; reframe story | Pivot narrative |
| Causal FAIL | Pixel forcing effect is noise; drop mechanism claim | Pivot to systems paper or negative result |
| Both FAIL | Project thesis is wrong; 36.2 dB is a structural ceiling | Investigate decoder fine-tuning or reduced hop count |

---

## Files inspected

| File | Lines | Purpose |
|---|---|---|
| `review/0424/nightmare/01_round1_initial_reviews.md` | 1–500 | Round 1 reviews from all 5 personas |
| `configs/pet_flow/pet_flow_first_hop_224_50k_transport_v3.yaml` | 1–270 | Scheme C (main) config |
| `configs/pet_flow/pet_flow_first_hop_224_50k_pixenc_ablation.yaml` | 1–230 | N1 ablation config |
| `configs/pet_flow/pet_flow_first_hop_224_50k_foc_lite.yaml` | 1–245 | FOC-lite variant config |

---

*End of Round 2 merged B+D review.*
