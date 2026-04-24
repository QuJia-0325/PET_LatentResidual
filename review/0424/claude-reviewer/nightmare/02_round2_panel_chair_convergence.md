# Round 2 — Panel Chair Convergence Analysis

Date: 2026-04-24  
Mode: NIGHTMARE, Round 2 / 5+  
Chair: Panel Chair (synthesizing A · B · C · D · E)

---

## 1. Top 5 Disputes (Requiring Resolution)

### Dispute 1: Is There ANY Salvageable Contribution?

| Aspect | Detail |
|--------|--------|
| **Name** | Salvageable contribution path |
| **Position A** | A: Reframe to systems/recipe paper or make A/B/C/D diagnostic framework the contribution. There's something here, just not a method contribution. |
| **Position B** | D: Nothing is identified until σ_seed is measured. Cannot discuss salvage paths when the effect magnitude (0.08 dB) is within noise. |
| **Holders** | A (diagnostic reframe) · E (workshop viable) vs D (null until variance) · B (broken narrative) |
| **Resolving evidence** | 3-seed variance run of scheme C. If σ > 0.08 dB → D wins (null). If σ < 0.04 dB → A/E path remains open. |
| **Severity** | **CRITICAL** — determines pivot vs. reframe |

### Dispute 2: First-Hop Bottleneck — Real or Retrofitted?

| Aspect | Detail |
|--------|--------|
| **Name** | Bottleneck hypothesis validity |
| **Position A** | First-hop is structurally the bottleneck (lowest D20 PSNR at 32.93 dB, propagation through chain) |
| **Position B** | The "bottleneck" claim contradicts evidence — gains at NORMAL (+0.66 dB) > gains at D20 (+0.18 dB), which is backwards |
| **Holders** | None explicitly defend Position A; B · D · E attack it |
| **Resolving evidence** | Per-hop transport-minus-oracle decomposition: `transport_PSNR(h) − oracle_PSNR(h)` for each hop. If hop0 shows largest negative gap, bottleneck is real. If monotonically improving (35.57→35.94→36.51→36.81), it's retrofitted. |
| **Severity** | **HIGH** — central narrative collapses if false |

### Dispute 3: Plateau = Under-training or Structural Ceiling?

| Aspect | Detail |
|--------|--------|
| **Name** | Plateau interpretation |
| **Position A** | Plateau at 36.2 dB is training artifact; 200K steps will break it |
| **Position B** | 5 very different configs (C/v2/N1/v3/chainstable) all land in 36.12–36.21 dB (σ=0.029 dB) → structural ceiling of latent-only transport |
| **Holders** | Implicit author position vs B · C · D · E |
| **Resolving evidence** | Pre-registered continuation: run v3 to 200K with stopping criterion; if plateau persists ±0.05 dB at 150K, 175K, 200K → ceiling confirmed |
| **Severity** | **HIGH** — determines resource allocation |

### Dispute 4: Workshop Publication Worth Effort?

| Aspect | Detail |
|--------|--------|
| **Name** | Venue viability without pivot |
| **Position A** | E: MIDL/workshop ~40% odds with honest reframe as negative/analysis paper |
| **Position B** | B: Even workshop requires resolving story-evidence contradiction; current data doesn't support any positive claim |
| **Holders** | E (workshop viable) vs B (narrative broken) · D (causality not established) |
| **Resolving evidence** | If σ_seed ≤ 0.04 dB AND honest negative-result framing passes internal mock review, workshop is viable. Otherwise, full pivot. |
| **Severity** | **MEDIUM** — affects effort allocation, not scientific validity |

### Dispute 5: Should External Baselines Be Added or Is That Scope Creep?

| Aspect | Detail |
|--------|--------|
| **Name** | Scope vs. rigor tradeoff |
| **Position A** | E: ≥2 external baselines mandatory (Pix2Pix + Palette/LDM) to interpret 36.2 dB |
| **Position B** | C: Already 25+ configs, 960 lines of spec, dead code. Adding baselines = more scope creep without addressing core causality issues. |
| **Holders** | E (baselines required) vs C (cut scope first) |
| **Resolving evidence** | None — this is a prioritization decision. Chair recommendation: baselines only after σ_seed established. If null, baselines are wasted effort. |
| **Severity** | **MEDIUM** — execution order, not direction |

---

## 2. Top 5 Consensus Findings (with Evidence)

### Consensus 1: Hop0 Mechanism Effect Is Sub-Noise

| Aspect | Detail |
|--------|--------|
| **Finding** | C − N1 = 0.083 dB, smaller than within-scheme variance (0.038 dB best↔last), no seed repeats, no CI |
| **Evidence** | D: within-scheme variance 0.038 dB; C−N1 = 0.083 dB with no significance test (Round 1, Causal Critic Q1) |
| **Unanimity** | **5/5** (A · B · C · D · E) |
| **Implication** | Cannot claim pixel forcing "helps" without 3-seed paired bootstrap. Core contribution is statistically null. |

### Consensus 2: Story-Evidence Contradiction (Tail > Head Gains)

| Aspect | Detail |
|--------|--------|
| **Finding** | Method claimed to fix first-hop bottleneck produces 3.7× more improvement at NORMAL (+0.66 dB) than at D20 (+0.18 dB) |
| **Evidence** | All 5 reviewers identified this independently; B formalized as "Claim 1 = BROKEN" |
| **Unanimity** | **5/5** |
| **Implication** | Central narrative is mechanistically backwards. Mechanism either (a) helps tail not head, or (b) helps nothing and variance explains both. |

### Consensus 3: Multi-Config Plateau = Structural Ceiling

| Aspect | Detail |
|--------|--------|
| **Finding** | 6 numbers from 5+ configs: [36.123, 36.206], std ≈ 0.029 dB |
| **Evidence** | D: "consistent with 'nothing is helping'"; B: "5 very different configs all land in 36.12–36.21 dB" |
| **Unanimity** | **5/5** |
| **Implication** | Latent-only transport may have a hard ceiling around 36.2 dB for this task. Iteration on hop0 mechanisms is unlikely to break it. |

### Consensus 4: 10 dB Oracle Gap Unexplained

| Aspect | Detail |
|--------|--------|
| **Finding** | Oracle decoder achieves ~46 dB; best transport achieves ~36 dB. 10 dB gap with no proposed mechanism to close it. |
| **Evidence** | B: "no plan in any doc"; E: "must explain 10 dB gap" as blocker |
| **Unanimity** | **5/5** |
| **Implication** | The "latent sufficiency" assumption implicit in the approach is falsified by this gap. Future work must address latent→pixel information loss. |

### Consensus 5: Missing External Baselines

| Aspect | Detail |
|--------|--------|
| **Finding** | No comparison to any published method (Pix2Pix, Palette, CycleGAN, PET-specific DDPM) |
| **Evidence** | E: "Baselines 1/10"; all reviewers note 36.2 dB is uninterpretable without reference |
| **Unanimity** | **5/5** |
| **Implication** | Even if 36.2 dB is state-of-art for this private dataset, no reader can verify. Paper is unsubmittable without ≥1 external baseline. |

---

## 3. Score Convergence Analysis

### Raw Scores Extracted from Round 1

| Dimension | A | B | C | D | E | Weight |
|-----------|---|---|---|---|---|--------|
| Novelty | 3/10 | — | — | — | — | 20% |
| Story (Coherence + Claim-Evidence) | — | 3+2 = 2.5/10 avg | — | — | — | 25% |
| Implementation (Spec-code + Eval + Scope) | — | — | (5+7+4)/3 = 5.3/10 | — | — | 15% |
| Causal (Effect + Stats + Reprod) | — | — | — | (2+1+2)/3 = 1.7/10 | — | 25% |
| Submission (Venue + Baselines + Result + Writing) | — | — | — | — | (3+1+3+1)/4 = 2.0/10 | 15% |

### Spread Analysis

| Dimension | Score | High | Low | Spread | Notes |
|-----------|-------|------|-----|--------|-------|
| Novelty | **3.0** | 3 | 3 | 0 | Single reviewer (A), no spread to analyze |
| Story | **2.5** | 3 | 2 | 1 | B's two sub-scores; coherence > claim-evidence |
| Implementation | **5.3** | 7 | 4 | 3 | Eval okay, scope poor |
| Causal | **1.7** | 2 | 1 | 1 | Uniformly catastrophic |
| Submission | **2.0** | 3 | 1 | 2 | Baselines = 1, Writing = 1 |

### Fused Score Calculation

```
Fused = 0.20(3.0) + 0.25(2.5) + 0.15(5.3) + 0.25(1.7) + 0.15(2.0)
      = 0.60 + 0.625 + 0.795 + 0.425 + 0.30
      = 2.745 / 10
```

### Final Convergence Matrix

| Metric | Value |
|--------|-------|
| Novelty (20%) | 3.0 |
| Story (25%) | 2.5 |
| Implementation (15%) | 5.3 |
| Causal (25%) | 1.7 |
| Submission (15%) | 2.0 |
| **FUSED TOTAL** | **2.75 / 10** |

**Interpretation**: Below 3/10 = reject-with-prejudice territory. Causal (1.7) is the load-bearing failure. Even if story and novelty were fixed, the lack of statistical rigor makes claims undefendable.

---

## 4. Chair's Pre-Rebuttal Briefing for Proposer (Round 3)

### The 3 Hardest Questions the Proposer MUST Answer

1. **The Inversion Question**  
   > "If hop0 mechanism targets the first-hop bottleneck, why does it produce 3.7× more improvement at NORMAL (+0.66 dB) than at D20 (+0.18 dB)? This is mechanistically backwards. Either (a) you mislabeled the bottleneck, (b) the mechanism helps something other than hop0, or (c) the improvement is noise. Which is it, and what experiment distinguishes these?"

2. **The Variance Question**  
   > "What is the seed-to-seed standard deviation of transport_avg for scheme C? You report a 0.08 dB improvement (C−N1) but never report σ. If σ ≥ 0.08 dB, your entire hop0 contribution is null. Why was this number never computed?"

3. **The Ceiling Question**  
   > "Five architecturally distinct configurations (C/v2/N1/v3/chainstable) all plateau at 36.12–36.21 dB (σ = 0.029 dB). What evidence distinguishes 'structural ceiling of latent-only transport' from 'haven't trained long enough'? A pre-registered 200K continuation with stopping criterion would resolve this — do you have one?"

### The 2 Weakest Reviewer Claims Where Proposer Has Counter-Arguments

1. **A's "Components are derivative" on the diagnostic framework**  
   A concedes the A/B/C/D diagnostic decomposition could be reframed as a contribution. If Proposer can show this decomposition generalizes beyond PET (e.g., applies to any multi-hop flow), the "derivative" label on method components becomes irrelevant — the contribution shifts to analysis methodology.

2. **E's 40% workshop odds**  
   E may be over-pessimistic. A well-framed negative-result paper ("What Doesn't Work for First-Hop Injection in Medical Image Flow Matching") at MIDL/NeurIPS workshop could be 60%+ if:
   - Statistical rigor added (3 seeds, CI)
   - Story reframed from "hop0 works" to "hop0 doesn't measurably help; here's why"
   - 10 dB gap positioned as open problem, not failure

### The 1 Face-Saving Path the Proposer Should NOT Take

**"Just run to 200K and see if plateau breaks."**

Why this path is toxic:
- Without pre-registered stopping criterion, any result is cherry-pickable
- If plateau holds → 1 week wasted, same paper problem
- If plateau breaks by 0.1 dB → still sub-noise without variance estimate
- Perpetuates the core mistake: running experiments without statistical design

The proposer will be tempted to say "we just need more compute." This is exactly what B calls "sunk-cost fallacy" and D calls "unidentified effect." Reviewers will reject any response that promises future experiments without specifying success/failure criteria upfront.

---

## 5. Chair's Ruling: Method vs. Paper vs. Evidence Problem

### Ruling: **(d) Multi-Problem**

The project suffers from all three failure modes simultaneously:

#### (a) Method Problem Component — **PARTIAL**
- Core mechanism (pixel forcing at hop0) produces 0.08 dB gain, within noise
- D: "consistent with 'nothing is helping'" — null result
- A: every component maps to published primitive (ControlNet, LoRA, LPIPS)
- However: the diagnostic framework (A/B/C/D decomposition) could be salvageable as a methodological contribution if properly formalized

#### (b) Paper Problem Component — **SEVERE**
- B: "First-hop is bottleneck" claim contradicts evidence (tail gains > head gains)
- Story integrity 5/10; claim-evidence 2/10
- Retroactive rationalization pattern: chainstable objective added to optimize tail, then claimed to validate hop0 mechanism
- Fixable with honest reframe, but requires abandoning core narrative

#### (c) Evidence Problem Component — **CRITICAL**
- D: no seed variance, no CI, no significance tests
- C: 26 config files, no canonical baseline
- E: 0 external baselines
- Every headline claim is statistically unsupported

### Severity Ranking
1. **Evidence** (most urgent) — without σ_seed, nothing else matters
2. **Paper** (blocks submission) — story-evidence contradiction kills any venue
3. **Method** (may be unsalvageable) — depends on whether pivot to analysis paper is acceptable

### Recommended Triage Order
1. **Week 1**: Compute σ_seed for scheme C (3 seeds). This determines whether method contribution is null.
2. **Week 1-2**: If σ_seed < 0.04 dB → proceed to reframe as analysis paper. If σ_seed ≥ 0.08 dB → pivot to diagnostic framework contribution or abandon.
3. **Week 3-4**: Add 1 external baseline (Pix2Pix) to contextualize 36.2 dB.
4. **Week 5-6**: Write honest negative-result paper or analysis paper.

### Final Chair Statement

This project is not a "fix the writing and submit" situation. The core experimental claim (hop0 mechanism helps) is statistically unidentified. The core narrative (first-hop is bottleneck) contradicts the evidence. Five reviewers independently converged on the same killing question, which the Proposer has not answered in any internal document.

The path forward is not iteration — it is decision:
- **If the goal is publication**: pivot to analysis/negative-result paper framing
- **If the goal is positive method result**: need fundamentally different approach (reduced hop count, hybrid latent-pixel, partial decoder fine-tuning)
- **If the goal is "find the right hyperparameter"**: this path is closed; 5 configs × same plateau = ceiling

The Proposer should not spend another GPU-week on hop0 variants without first computing σ_seed.

---

*Chair signature: Round 2 complete. Disputes clarified. Proposer briefing delivered. Awaiting Round 3 rebuttal.*
