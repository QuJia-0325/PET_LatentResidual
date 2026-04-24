# Round 3 — Reviewer Tribunal Pre-Ruling

Date: 2026-04-24  
Mode: NIGHTMARE, Round 3 / 5+  
Tribunal: B (Story Critic) + D (Causal Critic) + E (Submission Gatekeeper)  
Document Type: **PRE-RULING TEMPLATE** — to be applied when Proposer's rebuttal arrives

---

## Part 1: Independent Repo Verification (Adversarial)

### 1.1 Which Config Is "C"?

**BD's claim (Round 2 merged review)**: Compared `transport_v3.yaml` vs `pixenc_ablation.yaml`.

**TRIBUNAL FINDING: BD COMPARED THE WRONG CONFIG.**

Evidence:
- [multiagent_transport_review_and_rerank_20260423.md](../0423/multiagent_transport_review_and_rerank_20260423.md) § 2.2 shows:
  - `C_best = 36.205522 dB` 
  - Source: `first_hop_224_50k_schemec_v2/best.pt`
- [fullval_rerank_c_n1_v2_clip3_fast.py](../0423/fullval_rerank_c_n1_v2_clip3_fast.py) lines 108-109 confirm:
  ```python
  config_path="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_schemec_v2/config.yaml"
  checkpoint_path="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_schemec_v2/best.pt"
  ```

**THE ACTUAL C CONFIG IS `pet_flow_first_hop_224_50k_schemec_v2.yaml`, NOT `transport_v3.yaml`.**

### 1.2 Verified Config Diff: `schemec_v2.yaml` (C) vs `pixenc_ablation.yaml` (N1)

| Parameter | schemec_v2 (C) | pixenc_ablation (N1) | Impact |
|---|---|---|---|
| `first_hop.pixel_forcing_disabled` | absent (=false) | **true** | ✅ Intended ablation target |
| `ema` section | **PRESENT** (enabled=true, decay=0.9999) | **MISSING ENTIRELY** | ⚠️ **CONFOUNDER** |
| `training.best_metric_terms[0].weight` (D20) | 0.15 | 0.15 | ✅ Same |
| `training.rollout.step_weights` | [1.00, 1.10, 1.20, 1.30] | [1.00, 1.10, 1.20, 1.30] | ✅ Same |
| `transport.pair_loss_weights` | [1.00, 1.05, 1.10, 1.20] | [1.00, 1.05, 1.10, 1.20] | ✅ Same |
| `loss.pair.velocity_rebalance.enabled` | false | false | ✅ Same |

**BD'S CLAIMED 5 CONFOUNDERS ARE FALSE.** BD mistakenly compared `transport_v3.yaml` to `pixenc_ablation.yaml`. The actual C↔N1 diff has only **2 differences**:

1. **`pixel_forcing_disabled`**: the intended ablation variable ✅
2. **`ema` section missing in N1**: still a confounder ⚠️

### 1.3 Severity of the EMA Confounder

The EMA confounder is **REAL but LESS SEVERE** than BD's claimed 5-variable confound:

- C uses EMA (decay=0.9999), meaning best.pt contains exponentially smoothed weights
- N1 has no EMA section, meaning best.pt contains raw optimizer weights
- This is **ONE additional variable** beyond `pixel_forcing_disabled`, not five

**HOWEVER**: The 0424 investigation table ([investigation_and_plan.md](../0424/investigation_and_plan.md)) reports:

| Experiment | EMA | transport_avg |
|---|---|---|
| C best | ❌ | 36.206 |
| v2 best | ✅ | 36.197 |

This suggests C was run **WITHOUT EMA** despite the config file specifying `ema.enabled=true`. If true, the confounder vanishes. **TRIBUNAL REQUIRES AUTHOR TO CLARIFY ACTUAL EMA STATE OF C RUN.**

### 1.4 Multi-Seed Evidence Check

**Config search result**:
- `pet_flow_first_hop_224_50k_formal_v3_chainstable_seed2_gpu3.yaml` EXISTS with `seed: 43`
- This is for the **chainstable** config, NOT for schemec_v2 (C) or pixenc_ablation (N1)

**Review folder search**: No files in `review/0423/` or `review/0424/` contain "seed2", "std", "bootstrap", or "CI" relating to the C↔N1 comparison.

**FINDING: NO SEED VARIANCE DATA EXISTS FOR THE CLAIMED C↔N1 ABLATION.**

The 0.083 dB effect (36.206 − 36.123) cannot be interpreted without σ_seed.

### 1.5 Config Lineage Check

Both configs derive from the same lineage:

| File | Comment/Source |
|---|---|
| `schemec_v2.yaml` | `# Scheme C v2: same as imgaux_boost but with 0422 fixes` |
| `pixenc_ablation.yaml` | No explicit comment; structure matches schemec_v2 exactly except pixel_forcing_disabled and missing ema |

**FINDING: LINEAGES ARE CONSISTENT.** BD's claim that "N1 = pixenc_ablation is based on chainstable" is FALSE. The pixenc_ablation.yaml structure is identical to schemec_v2.yaml.

### 1.6 Internal Investigation Already Concluded "Pixel Forcing 无正面贡献"

From [investigation_and_plan.md](../0424/investigation_and_plan.md) § 1:

> | 假设 | 实验 | 结果 | 结论 |
> |------|------|------|------|
> | pixel forcing 有帮助 | N1 (pixel OFF) | 36.123 dB | ❌ 无正面贡献 |

**The author's own internal analysis concluded pixel forcing provides no positive contribution.** This contradicts the external-facing narrative.

---

## Part 2: Pre-Ruling Template

The Proposer must address 4 items. For each, we specify ruling criteria.

---

### Q1: The Inversion Question

> "If hop0 mechanism targets the first-hop bottleneck, why does it produce 3.7× more improvement at NORMAL (+0.66 dB) than at D20 (+0.18 dB)?"

#### SUSTAINED (critique withdrawn) requires:

1. **Causal mediation analysis** with the following design:
   - Measure per-hop transport−oracle gap: `transport_PSNR(h) − oracle_PSNR(h)` for h ∈ {D20, D10, D4, NORMAL}
   - Show that hop0 gap (transport_D20 − oracle_D20) is largest negative gap
   - Run intervention: (A) hop0 learned + hops 1-3 oracle; (B) hop0 oracle + hops 1-3 learned
   - Show (A−baseline) >> (B−baseline), proving tail gains are mediated by hop0 improvement

2. **Story-consistent gain direction**: absolute gain at D20 > absolute gain at NORMAL

3. **Statistical significance**: 3+ seeds, bootstrap CI excluding zero on mediation proportion

#### PARTIALLY SUSTAINED (narrow the critique) if:

- Mediation analysis shows tail gains ARE propagated from hop0 improvements
- BUT gain direction remains inverted (NORMAL > D20)
- Critique narrows to: "The hop0 mechanism does propagate through the chain, but the evidence does not support 'bottleneck' language — consider reframing to 'error propagation' without bottleneck claim."

#### OVERRULED (critique stands) if:

- No mediation analysis provided
- OR mediation shows gains at hop0 ≤ gains at later hops
- OR mere assertion that "propagation explains it" without quantitative intervention

#### Tribunal's independent finding bearing on Q1:

The 0424 internal investigation ([investigation_and_plan.md](../0424/investigation_and_plan.md)) states:

> **关键数据**: 每 0.0001 latent MSE → 损失 5.4 dB (D20) 到 9.4 dB (NORMAL) image PSNR

This shows the decoder is more sensitive at NORMAL, which could explain larger NORMAL gains without validating a hop0 bottleneck. The Proposer could argue this sensitivity amplification explains the ratio — but this would WEAKEN the "hop0 is the bottleneck" claim, not strengthen it.

---

### Q2: The Variance Question

> "What is σ_seed of transport_avg for scheme C? The 0.083 dB effect (C−N1) is meaningless without it."

#### SUSTAINED requires:

1. **3+ seed runs of both C and N1** with locked hyperparameters
2. **Paired bootstrap CI on (C − N1) transport_avg** excluding zero
3. **Lower bound of CI > 0.05 dB** (half the claimed effect)

#### PARTIALLY SUSTAINED if:

- σ_seed computed and reported
- CI includes zero OR lower bound < 0.05 dB
- Critique narrows to: "Effect is detectable but marginal; contribution claims must be hedged with 'suggestive' language."

#### OVERRULED if:

- No seed study provided
- OR σ_seed > 0.08 dB (effect within 1σ of noise)
- OR "seed study coming later" claim without pre-registered design

#### Tribunal's independent finding bearing on Q2:

**No seed variance data exists.** The `chainstable_seed2` config (seed=43) exists but:
1. It's for the wrong config (chainstable, not schemec_v2 or pixenc_ablation)
2. No results from this run appear in any review file

The Proposer cannot satisfy SUSTAINED without new experiments.

---

### Q3: The Ceiling Question

> "Five configs plateau at 36.12–36.21 dB (σ=0.029). Is this a structural ceiling or undertrained?"

#### SUSTAINED requires:

1. **Pre-registered 200K continuation** with stopping criterion stated BEFORE viewing results
2. **Results showing > 0.3 dB improvement** at 200K over 50K plateau
3. **Continued improvement trend** at 150K, 175K, 200K (no re-plateau)

#### PARTIALLY SUSTAINED if:

- 200K run shows 0.1–0.3 dB improvement
- Critique narrows to: "Plateau is partially training-limited, but gains are marginal relative to the 10 dB oracle gap."

#### OVERRULED if:

- No 200K results
- OR 200K plateau at 36.2 ± 0.1 dB (within noise of 50K)
- OR claiming "300K will break it" without 200K evidence first

#### Tribunal's independent finding bearing on Q3:

The 200K experiment is planned ([investigation_and_plan.md](../0424/investigation_and_plan.md) § 3) with config `pet_flow_first_hop_224_200k_transport_v3.yaml`. BUT:
1. This uses transport_v3, not schemec_v2 (C)
2. Transport_v3 differs from schemec_v2 in 4 hyperparameters (see v3 header comment)
3. If 200K breaks plateau, it will be confounded by v3's different weighting scheme

**The 200K experiment does not cleanly address the plateau question for the C (schemec_v2) config.**

---

### Q4: N1 Is Not a Clean Ablation

> "The N1 config differs from C in multiple hyperparameters beyond pixel_forcing_disabled."

#### SUSTAINED requires:

1. **Proof that C and N1 had identical EMA state** during actual training
2. **Config diff showing only one variable changed** (pixel_forcing_disabled)
3. **OR new clean ablation run** with only pixel_forcing_disabled toggled, ≥3 seeds each arm

#### PARTIALLY SUSTAINED if:

- EMA confounder acknowledged but argued to be "low impact" with supporting evidence (e.g., v2 with EMA vs C without EMA shows < 0.01 dB difference)
- Critique narrows to: "EMA is a minor confounder; pixel forcing effect is still the dominant signal."

#### OVERRULED if:

- No clarification of actual EMA state
- OR multiple confounders remain unexplained
- OR "we'll run a clean ablation later" without committing to it in this response

#### Tribunal's independent finding bearing on Q4:

**BD's 5-confounder claim is FALSE.** The actual C↔N1 diff has only 2 variables:
1. `pixel_forcing_disabled`: intended target
2. `ema` section: present in C, missing in N1

The 0424 investigation table shows C was run with EMA=❌, which would eliminate the confounder entirely — but this contradicts the config file. **TRIBUNAL REQUIRES CLARIFICATION.**

If C was indeed run without EMA (despite the config saying enabled=true), then N1 IS a cleaner ablation than BD claimed, and this critique should be PARTIALLY SUSTAINED.

---

## Part 3: Tribunal's Position Going Into Round 3

### Reviewer Claims From Rounds 1-2 That Were OVERSTATED

| Claim | Source | Tribunal Finding |
|---|---|---|
| "N1 ablation confounds 5+ hyperparameters" | BD merged review | **OVERSTATED** — BD compared the wrong config (transport_v3 instead of schemec_v2). Actual diff is 2 variables, one of which may not be active. |
| "step_weights inverted between C and N1" | BD merged review | **FALSE** — Both schemec_v2 and pixenc_ablation use identical step_weights [1.00, 1.10, 1.20, 1.30]. |
| "pair_loss_weights inverted between C and N1" | BD merged review | **FALSE** — Both use identical pair_loss_weights [1.00, 1.05, 1.10, 1.20]. |
| "D20 selection weight 0.50 in C vs 0.15 in N1" | BD merged review | **FALSE** — Both use 0.15. The 0.50 value is in transport_v3, which was NOT the C config. |
| "velocity_rebalance enabled in C, disabled in N1" | BD merged review | **FALSE** — Both have velocity_rebalance.enabled=false. |

**BD owes the Proposer a partial apology.** The config diff section of the BD merged review is factually incorrect and should be withdrawn.

### Reviewer Claims That STAND

| Claim | Tribunal Assessment |
|---|---|
| "0.083 dB effect is sub-noise without σ_seed" | **STANDS** — No seed variance data exists. This remains the killing issue. |
| "Tail gains > head gains contradicts hop0 bottleneck narrative" | **STANDS** — No mediation analysis has been provided. |
| "5 configs plateau at 36.1–36.2 dB suggests structural ceiling" | **STANDS** — 200K results pending; if they confirm plateau, this is terminal. |
| "10 dB oracle gap unexplained" | **STANDS** — No mechanism proposed to close it. |
| "No external baselines" | **STANDS** — 36.2 dB remains uninterpretable without reference. |

### NEW Critique From Independent Verification

**NEW-1: Internal Conclusion Contradicts External Narrative**

The 0424 investigation concludes:

> | 假设 | 结论 |
> |------|------|
> | pixel forcing 有帮助 | ❌ 无正面贡献 |

Yet Round 1-2 reviews treat the hop0 mechanism as the Proposer's claimed contribution. **If the Proposer's internal analysis already concluded "no positive contribution," the external narrative must be updated accordingly.**

**NEW-2: 200K Experiment Uses Different Config**

The 200K experiment (`transport_v3`) differs from C (`schemec_v2`) in 4 hyperparameters. If 200K breaks the plateau, the cause will be confounded between:
- Longer training (200K vs 50K)
- Inverted step_weights [1.30→1.00] vs [1.00→1.30]
- Inverted pair_loss_weights
- velocity_rebalance enabled
- D20 selection weight 0.50 vs 0.15

**A clean 200K continuation should use schemec_v2 with only max_steps changed.**

---

## Summary Tribunal Position

| Question | Current Status | Path to SUSTAINED |
|---|---|---|
| Q1 Inversion | OVERRULED pending mediation analysis | Intervention experiment + story-consistent gain direction |
| Q2 Variance | OVERRULED pending σ_seed | 3+ seeds per arm, CI excluding zero |
| Q3 Ceiling | OVERRULED pending 200K | Pre-registered continuation, > 0.3 dB gain |
| Q4 N1 Cleanliness | **PARTIALLY SUSTAINED** — BD's 5-confounder claim was wrong; only EMA is a real confounder, and it may not have been active | Clarify actual EMA state of C run |

**BD's config diff error improves the Proposer's position on Q4 but does not affect Q1, Q2, or Q3.** The core statistical critique (no σ_seed) remains the killing issue.

---

*End of Round 3 Tribunal Pre-Ruling. Awaiting Proposer's rebuttal to apply rulings.*
