# Round 3 — Proposer Rebuttal + Tribunal Arbitration

Date: 2026-04-24  
Mode: NIGHTMARE, Round 3 / 5+  
Role: Proposer defense + independent Tribunal ruling

---

## Round 3 highlight: BD alliance's "5-hyperparameter confounder" claim is PARTIALLY OVERRULED

### Chair's independent diff (ground truth)

```
imgaux_boost vs pixenc_ablation → only differs in `pixel_forcing_disabled`
schemec_v2   vs pixenc_ablation → differs in `pixel_forcing_disabled` AND EMA section
transport_v3 vs pixenc_ablation → differs in 5+ knobs (BD's original target)
```

### Interpretation
- If the "C best = 36.206 dB" number came from `imgaux_boost` → **ablation is clean, BD overruled on this point**
- If from `schemec_v2` → 1 residual confounder (EMA), BD partially sustained
- BD compared `transport_v3` which is a LATER experiment branch — genuine error

### Proposer's position
"C = imgaux_boost"; clean single-variable ablation exists. (Self-reported; needs cross-check against actual experiment logs but plausible given Proposer cites `review/0424/investigation_and_plan.md` line 42.)

### Tribunal position
Independently verified there EXISTS a clean pair (imgaux_boost ↔ pixenc_ablation). Even if the reported C best came from schemec_v2, the addition of imgaux_boost comparison would still give a clean delta.

**Ruling on Q4 (N1-not-clean)**: **PARTIALLY SUSTAINED**
- SUSTAINED: BD made a factual error on config identification
- STILL STANDING: single-seed ablation with no CI remains statistically uninterpretable

---

## Rulings on the 3 big questions

### Q1 — The Inversion Question

**Proposer position**:
- (a) Mislabeled: partially accepts — "hardest" ≠ "only bottleneck"; concedes narrative overstatement
- (b) Helps tail via propagation: accepts but concedes no mediation test run
- (c) Noise: rejects (conditionally) but concedes no σ_seed

**Tribunal ruling**: **OVERRULED on (a)** — concession is acceptable but not enough; the current story in CLAUDE.md and IDEA_REPORT.md explicitly centers "first-hop is THE bottleneck" and cannot be quietly softened without manuscript rewrite.

**Ruling on (b)**: **PARTIALLY SUSTAINED** — propagation theory IS mechanistically plausible; but mediation test must be run before any mediation claim appears in a paper.

**Ruling on (c)**: **NOT RULED** — depends on Q2 variance.

### Q2 — The Variance Question

**Proposer position**: Concedes σ_seed unmeasured; no multi-seed run exists; `chainstable_seed2_gpu3.yaml` exists but never ran; commits to 3-seed × 2-config experiment (6 runs, ~4 GPU-days).

**Tribunal ruling**: **OVERRULED (Proposer concedes)** — the 0.083 dB claim is statistically null until σ_seed is measured and CI on (C − N1) excludes zero. No rebuttal possible without the data.

**Stipulated threshold**: CI lower bound > 0.03 dB for the claim to be publishable.

### Q3 — The Ceiling Question

**Proposer commitment**:
- 200K pre-registered stopping: if transport_avg at 150K within ±0.05 dB of 50K result → declare ceiling, abandon iteration
- Will NOT pivot to "just more compute" escape hatch

**Tribunal ruling**: **PARTIALLY SUSTAINED** — commitment is principled but the experiment still buys time. Tribunal demands that if at 100K the signal shows no monotone improvement pattern, the Proposer stop early rather than wait for 150K. Wasted GPU-weeks compound the rigor problem.

---

## Consolidated Proposer concessions (for Round 4 baseline)

| # | Concession | Status |
|---|---|---|
| 1 | σ_seed is unmeasured; 0.083 dB is statistically null | Conceded |
| 2 | No external baselines | Conceded |
| 3 | 10 dB oracle gap unexplained | Conceded |
| 4 | Component novelty is low (ControlNet/LoRA/perceptual) | Conceded |
| 5 | Scope creep real (4 modules, 26 configs, dead code) | Conceded |
| 6 | "First-hop bottleneck" narrative overstated | Conceded |
| 7 | Mediation analysis not run (for tail-propagation claim) | Conceded |
| 8 | BD's 5-confounder diff: partially refuted (actual diff is 1–2 variables depending on which "C" config) | Won |

---

## Proposer's 3 committed experiments (pre-registered)

### Exp 1: Seed variance × clean ablation
- Configs: `imgaux_boost` (C), `pixenc_ablation` (N1)
- Seeds: {42, 123, 456} × 2 configs = 6 runs × 50K steps × 16h ≈ 4 GPU-days
- Report: σ_seed per config; paired bootstrap 95% CI on (C − N1); per-sample n=7403
- PASS: CI lower bound > 0.03 dB

### Exp 2: Causal mediation for tail-propagation claim
- Arms: (A) learned hop0 + oracle hops1-3; (B) oracle hop0 + learned hops1-3
- Single seed, inference-only on best checkpoints from Exp 1
- ~0.5 GPU-day
- PASS: gain at NORMAL (A−baseline) > gain at NORMAL (B−baseline); direction consistent with hop0 mediation

### Exp 3: 200K extended run with hard stop
- Config: `pet_flow_first_hop_224_200k_transport_v3.yaml`
- Checkpoints every 20K
- Hard stop rule: if at 150K transport_avg within ±0.05 dB of 50K → declare ceiling, abandon
- ~2.7 GPU-days
- PASS: NOT "break plateau"; PASS = honest resolution of ceiling hypothesis either direction

**Total GPU commitment: 7.2 GPU-days before any submission claim.**

---

## Updated fused score after Round 3

| Dimension | R1→R2 | R3 update | Reasoning |
|---|---|---|---|
| Novelty | 3 | **3** | Unchanged; no new component novelty emerged |
| Story | 2.5 | **3.5** | Proposer's honest reframing path is viable; narrative fix credit |
| Implementation | 5.3 | **5.3** | Confirmed; clean ablation pair exists but scope creep real |
| Causal | 1.7 | **2.0** | Proposer committed to proper seed/CI/mediation protocol |
| Submission | 2.0 | **2.5** | Path to workshop exists if commitments honored |

**Fused (R3)**: 0.20·3 + 0.25·3.5 + 0.15·5.3 + 0.25·2.0 + 0.15·2.5 = 0.60 + 0.875 + 0.795 + 0.50 + 0.375 = **3.15 / 10**

Up from 2.75 → 3.15 (+0.4) after R3 concessions and commitments. Still below the 5.0 "workshop viable" threshold, but trajectory is positive IF the 3 committed experiments actually run and the results are honestly reported regardless of direction.

---

## Remaining open disputes for Round 4

1. **D1 — Does the "diagnostic framework" reframe provide any real publishable novelty?**
   Proposer claims A/B/C/D decomposition generalizes beyond PET. No evidence offered that it has been applied elsewhere. Round 4 should demand either (a) cross-domain demonstration or (b) acknowledge it as PET-specific.

2. **D2 — What is the honest venue path: reframe as negative result or kill and pivot?**
   - Reframe: requires accepting "method didn't work" as main finding (courage + rewrite cost)
   - Pivot: reduced-hop / hybrid / partial decoder FT (new 8-12 week effort)

3. **D3 — Is the 200K run even worth running given Proposer's own stop-criterion?**
   If 5 configs already show plateau within 0.03 dB std, Proposer's 0.05 dB tolerance means the criterion is almost certain to trigger "abandon" at 150K. Is the 200K run just a delay tactic, or a genuine falsification attempt?

4. **D4 — Patient-level vs slice-level train/val split — red flag not yet checked**
   E raised this in R1. Proposer did not address. CRITICAL for any medical-imaging submission.
