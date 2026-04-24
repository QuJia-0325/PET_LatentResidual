# Round 1 — 5 Reviewer Initial Reviews

Date: 2026-04-24  
Mode: NIGHTMARE, Round 1 / 5+  
Reviewers: A (Novelty Skeptic) · B (Story Critic) · C (Implementation Auditor) · D (Causal Critic) · E (Submission Gatekeeper)  
All 5 run as independent Opus 4.5 subagents with full repo read access; no shared context during review.

---

## Convergent headline

| Reviewer | Verdict | Score (headline) |
|---|---|---|
| A · Novelty | INCREMENTAL | Novelty **3/10** · Positioning **4/10** |
| B · Story | **BROKEN** | Coherence **3/10** · Claim-evidence **2/10** · Integrity **5/10** |
| C · Implementation | SLOPPY | Spec-code **5/10** · Eval **7/10** · Scope **4/10** |
| D · Causal | **NOT IDENTIFIED** | Effect **2/10** · Stats **1/10** · Reprod **2/10** |
| E · Gatekeeper | **NOT READY** | Venue **3/10** · Baselines **1/10** · Result **3/10** · Writing **1/10** |

**All 5 reviewers independently converged on the same killing question**:
> "You claim first-hop is the bottleneck and your hop0 mechanism fixes it — then why does it produce ~3.7× more improvement at NORMAL (+0.66 dB) than at D20 (+0.18 dB)?"

---

## A · Novelty Skeptic — full review

### Verdict: INCREMENTAL
Novelty 3/10, Positioning clarity 4/10.

### Component-by-component
| Component | Closest prior art | Verdict |
|---|---|---|
| FirstHopPixelEncoder | ControlNet (2302.05543), T2I-Adapter (2302.08453), SPADE (CVPR'19) | **Incremental** — only delta is "hop-0 only" + softplus gate + 5% cap |
| HopResidualVelocityHead | LoRA (2106.09685), Adapter (ICML'19), Hypernet, conditional residual blocks | **Derivative** — literally `v_shared + λ·head(v_shared)`, which is LoRA form with nonlinear head |
| L_img_hop0 | LPIPS (CVPR'18), VAE-GAN / VQ-VAE-2 decoder-based perceptual losses | **Incremental** — using own decoder instead of VGG is domain choice, not contribution |
| 4-hop cascade latent transport | CDM (2106.15282), Rectified Flow (2209.03003), Flow Matching (2210.02747), Progressive Distillation | **Derivative** — multi-step flow matching applied to fixed PET dose schedule |
| CCT-224 (in test branch) | Consistency Models (2303.01469), scheduled sampling (Bengio 2015), DMD | **Marginal** — not even on master |

### Sharpest critiques
1. "First-hop bottleneck" narrative contradicts its own evidence — gains concentrate at tail
2. Every component has a direct published analog; combination produces no emergent benefit
3. Self-scored novelty 8.4/10 for CCT-224 is inflated — consistency training is published, delta is just task specialization

### Question that cannot be answered
> "If pixel forcing genuinely addresses a first-hop structural bottleneck, why does disabling it hurt the tail more than the first hop? Please isolate the causal effect of hop-0 mechanisms on hop-0 PSNR specifically, not the chain average."

### Suggested reframings
1. Pivot to **systems/recipe paper** (MICCAI workshop level) — honest about component reuse
2. Make the **A/B/C/D diagnostic framework** the main contribution — formalize the decomposition
3. Find a setting where hop-0 is **demonstrably and uniquely** necessary (gain > 1 dB at D20 specifically)

---

## B · Story Critic — full review

### Verdict: BROKEN
Coherence 3/10, Claim-evidence 2/10, Integrity 5/10.

### Per-claim audit
| # | Claim | Supporting | Contradicting | Ruling |
|---|---|---|---|---|
| 1 | First-hop is the bottleneck | `no_transport D20 = 32.93`, D20 PSNR lowest | D20 gain +0.18 vs NORMAL +0.66 | **BROKEN** |
| 2 | hop0 mechanism solves first-hop | C−N1 = +0.083 dB | 0424 investigation: "pixel forcing 无正面贡献"; 5 configs → same plateau | **WEAK / UNFALSIFIABLE** |
| 3 | Plateau = under-training | Reasonable in isolation | 5 very different configs all land in 36.12–36.21 dB | **WEAK** |
| 4 | Need 10× latent MSE reduction | Math is correct | No mechanism proposed; "excuse, not mechanism" | **UNFALSIFIABLE** |
| 5 | CCT-224 novelty 8.4/10 | Self-assessment | CCT parked on test branch; master iterates on ruled-out mechanisms | **BROKEN** |
| 6 | 10 dB oracle gap is closable | Implicit | No plan in any doc | **UNFALSIFIABLE** |

### Sharpest critiques
1. "First-hop bottleneck" is retroactive rationalization — chainstable objective was added to optimize tail, then claimed to validate the mechanism
2. 5 configs → same plateau = structural ceiling, not "haven't found the right knob"
3. Sunk-cost fallacy: 0424 investigation ruled out pixel forcing; 200K v3 still iterates on it

### Narrative crack
> "If the hop0 mechanism targets the first-hop bottleneck, why does it produce 3.7× more improvement at NORMAL than at D20?"

### Honest reframing
> "All five experimental configurations converged to the same 36.2 dB plateau regardless of architectural choices, suggesting a structural ceiling in the latent transport paradigm. The 10 dB gap to oracle remains unexplained. Next steps should investigate whether this ceiling is intrinsic to latent-only transport or addressable through fundamentally different approaches (reduced hop count, partial decoder fine-tuning, hybrid latent-pixel)."

---

## C · Implementation Auditor — full review

### Verdict: SLOPPY
Spec-code 5/10, Eval 7/10, Scope 4/10.

### Findings (by severity)
| # | Sev | File:line | Issue |
|---|---|---|---|
| 1 | **MEDIUM** | main.md vs model_first_hop.py L36 | Spec says `N=196` (14×14), code uses `latent_size=16` → 256 tokens. Spec cannot be trusted. |
| 2 | **MEDIUM** | main.md vs transport_v3.yaml L266 | Spec says `lambda_hop init = 0`, config uses `0.10` with floor 0.001. Zero-init claim false. |
| 3 | **MEDIUM** | model_first_hop.py L68-120, L123-177 | `SpatialAlignmentProjector` (iREPA) + `SeamRefiner` (~170 LoC) implemented but **not enabled** in transport_v3 configs. Dead code. |
| 4 | **LOW** | rollout_first_hop.py L145-146 | `preds[0] = z_d50` — "D50 PSNR 42.62 dB" is decode baseline, not transport output (misleading in results tables) |
| 5 | **LOW** | `configs/pet_flow/` (26 files) | v1/v2/v3/chainstable/tailaligned/… no canonical baseline |
| 6 | **LOW** | model_first_hop.py L329-335 | N1 ablation still logs `gate_pix` metric even when bypassed |

### Sharpest critiques
1. Pixel-forcing mechanism is textbook `z_in = z_src + g·f(x)`; "hop0-only restriction" is a hyperparameter, not a contribution; delta over N1 is 0.08 dB
2. 4 modules × no factorial ablation → no attribution possible
3. `velocity_rebalance` uses double-`detach()` → stop-gradient heuristic masquerading as principled "sqrt_ratio mode"

### Implementation gap question
> "N1 ablation vs scheme C: 0.08 dB gap, smaller than step-to-step variance, no significance test, no confidence interval. What does 'pixel forcing helps' mean quantitatively?"

---

## D · Causal Critic — full review

### Verdict: NOT IDENTIFIED
Effect 2/10, Stats 1/10, Reproducibility 2/10.

### Per-question
| Q | Ruling | Key data |
|---|---|---|
| Q1 Does hop0 help? | **UNIDENTIFIED** | C−N1 = 0.083 dB; within-scheme best↔last = 0.038 dB; no seeds, no σ |
| Q2 First-hop = bottleneck? | **CONFOUNDED** | The −7.31 dB "gap" mixes decoder ceiling into transport claim. Per-hop profile is monotonically improving (35.57→35.94→36.51→36.81), not a hop0 bottleneck shape |
| Q3 Checkpoint-selection | **CONFOUNDED** | val_multi_objective weights differ across schemes; N1 last > N1 best |
| Q4 Config drift | **UNIDENTIFIED** | C/v2/N1/v3 simultaneously vary 4–7 knobs; no single-variable ablation except C↔N1 (which is sub-noise) |
| Q5 Plateau | **NULL-FAVORING** | 6 numbers in [36.123, 36.206], std ≈ 0.029 dB → consistent with "nothing is helping" |
| Q6 Missing artifacts | **UNIDENTIFIED** | `tf_vs_pure_gap.json`, `seam_strata_summary.csv`, bootstrap CI, seed repeats — all missing |

### Minimum experiments to establish causality
1. **Seed-variance baseline**: 3 seeds of scheme C → report σ of transport_avg. This single number determines whether any 0.1 dB effect is discussable.
2. **Clean pixel-forcing A/B**: C vs N1, 3 seeds each, paired bootstrap CI on (C−N1) per-sample PSNR.
3. **Selection-protocol neutralization**: Every run reports multi-obj best, last, and mean-of-last-5.
4. **Decoder-normalized per-hop gap**: `transport_PSNR(h) − oracle_PSNR(h)` per hop — only way to isolate transport quality.
5. **Pre-registered single-variable grid**: one locked baseline, toggle each knob separately, ≥2 seeds each.

### Causal-killing question
> "What is the seed-to-seed standard deviation of full-val transport_avg for a single fixed config, and why is every headline claim in this repo being reported without that number?"

---

## E · Submission Gatekeeper — full review

### Verdict: NOT READY / DESK-REJECT-RISK
Venue 3/10, Baselines 1/10, Result 3/10, Writing 1/10.

### Venue odds
| Venue | Odds | Why |
|---|---|---|
| TMI / MIA | ~15% | Needs external baselines + clinical utility + significance tests |
| MICCAI short / IPMI | ~10% | Needs crisp delta; 0.08 dB won't survive |
| NeurIPS/ICML/CVPR | **<2%** | Desk-reject: single private dataset, no external baseline, sub-0.1 dB ablations, all components map to known primitives |
| MIDL / workshop | ~40% | Only if reframed honestly as negative/analysis paper |

### Must-do blockers (with effort)
1. ≥2 external baselines (Pix2Pix + Palette/LDM + published PET CycleGAN/DDPM) — **10-14d**
2. Paired bootstrap / Wilcoxon on 7,403 val slices; 95% CI on every PSNR — **1-2d**
3. Full factorial ablation on {pixel enc, residual head, img aux} — **5-7d**
4. Resolve story-evidence contradiction: run D50→NORMAL single-hop vs chained — **3-5d**
5. Explain 10 dB oracle gap — bias decomposition or architectural intervention — **5-10d**
6. Dataset/ethics statement; patient-level split (not slice-level — HIGH risk) — **1-2d**
7. Actual manuscript (8 pages) — **7-10d**
8. Code release plan — **2-3d**

**Total**: 5–8 weeks focused effort.

### Meta-reviewer first question
> "Across your 7,403 val pairs, is the 0.08 dB improvement of your full method over the 'pixel-forcing-off' ablation statistically significant, and how does it compare in absolute terms to a single off-the-shelf image-translation baseline (e.g., Pix2Pix or a published PET-denoising diffusion model) trained on the same data? If you cannot answer both in numbers on the same split, why is this paper under review?"

---

## Cross-reviewer aggregates (for Round 2 chair)

### Unanimous findings (5/5 agree)
1. Hop0 mechanism is sub-noise against no-hop0 ablation (C−N1 = 0.08 dB, no SE/CI)
2. Story-evidence contradiction: tail gains >> head gains
3. All 5+ experiments plateau at 36.12–36.21 dB → structural ceiling, not under-training
4. 10 dB gap to oracle decoder is unexplained with no mechanism proposed
5. Missing external baselines renders 36.2 dB uninterpretable

### Strong findings (4/5 agree)
6. Components are derivative (ControlNet/LoRA/perceptual-loss analogs)
7. Checkpoint-selection criterion drifts across schemes, contaminating leaderboard
8. Scope creep: 4 implemented modules, 25+ configs, 960 lines of spec, dead code

### Disagreements to probe in Round 2
- **A vs D on whether there's ANY salvageable contribution**: A suggests reframing to systems/recipe or A/B/C/D diagnostic framework; D says nothing is identified until σ_seed is measured. Could the diagnostic framework itself be the contribution even without a positive method result?
- **B vs E on venue path**: B implies honest negative-result reframe; E assigns 40% odds at MIDL workshop. Is a workshop publication with current results worth the effort, or is pivot mandatory?
- **C's "scope creep" vs E's "must add external baselines"**: adding baselines = more scope. How to reconcile "cut dead code" with "add 2 external baselines"?

### Memory update (Reviewer brain for R2)
Watch in Round 2 for:
- Author rebuttal attempts claiming "tail gains imply hop0 works because errors propagate" — demand causal mediation analysis
- Author claiming "200K will break plateau" — demand pre-specified stopping criterion
- Author claiming "statistical significance test comes later" — this is never acceptable for a submission-stage claim
