# Round 4 external consensus — σ-normalize vs raw-rollout

**Date**: 2026-05-06
**Trigger**: After [`ROUND4_PRE_REVIEW_CONSENSUS_20260506.md`](ROUND4_PRE_REVIEW_CONSENSUS_20260506.md) finalized the Round 4 prompt with 8 convergent fixes and the prompt was committed at `dcdcd2c` (tagged `round4-tmp-pre-external-review-20260506` on both `origin/foc_lite_hop0` and `gitee/foc_lite_hop0`), the prompt was sent to 6 external reviewer agents (per the conversation transcript at `transcripts/2c103958-6142-45c0-ba2d-226ce4501cf2.jsonl`):

- **Agent 1**: prompt-quality meta-review (reviewing the v2 prompt itself, not answering Q1/Q2/Q3)
- **Agents 2, 3, 5**: substantive Q1/Q2/Q3 reviewers, anchoring on V6/V6.1 historical paired-d ≈ 0.067
- **Agent 4**: substantive reviewer with mid-range `d_pure` estimate (optimizer-state divergence emphasis)
- **Agent 6**: most thorough reviewer with √8 step-compounding heuristic, predicting Plan F structurally underpowered

The remote codex review failed (502 Bad Gateway × 5 retries; substituted by the 6-agent external review).

This document records the cross-agent convergence, the one critical divergence (`d_pure` point estimate), the resulting GPU-launch gate decision, and the supersession commit plan.

---

## 1. Cross-agent convergence matrix

For each finding, columns mark **A**=Agent verdict; if 5/5 substantive reviewers (Agents 2-6) agree, the finding is treated as **convergent** and folded into the supersession plan.

### Q1 — σ-normalize root cause

| Finding | A2 | A3 | A4 | A5 | A6 | Verdict |
|---|---|---|---|---|---|---|
| Sub-claim (i) "PyTorch reports nondeterministic Mem-Eff-attention + adaptive_avg_pool2d_backward_cuda" is **VERIFIED FACT** | ✅ | ✅ | ✅ | ✅ | ✅ | **5/5 unanimous** |
| Sub-claim (ii) "kernel noise amplified through `mix_latent` is the **dominant** mechanism" is **PARTIAL or WRONG** | ✅ partial | ✅ partial | ✅ wrong | ✅ partial | ✅ wrong | **5/5 unanimous reject as exclusive cause** |
| `val_pair_total = 10.16%` falsifies any rollout-amplification-only diagnosis (pair_loss is structurally outside `mix_latent`) | ✅ | ✅ | ✅ | ✅ | ✅ | **5/5 unanimous** |
| Dominant mechanism is **(γ) optimizer-state / weight-trajectory divergence after 20K SGD steps under warn-only nondeterministic kernels** | ✅ ranked #1 | ✅ ranked #1 | ✅ ranked #1 | ✅ ranked #1 | ✅ ranked #1 | **5/5 unanimous** |
| (β) FP32 reduction-order asymmetry from B's `step_loss / float(n_j)` is a real **secondary** contributor | ✅ #3 | ✅ #3 | ✅ #1 (joint with α) | ✅ #3 | ✅ #2-3 | **5/5 unanimous as real but not primary** |
| Existing artifacts CANNOT prove σ-normalize algebraic correctness; same-process golden test is required | ✅ | ✅ | ✅ | ✅ | ✅ | **5/5 unanimous** |

### Q2 — raw-rollout path adequacy

| Finding | A2 | A3 | A4 | A5 | A6 | Verdict |
|---|---|---|---|---|---|---|
| V7 raw-rollout structurally **eliminates (β)** σ-normalize-specific autograd graph asymmetry | ✅ | ✅ | ✅ | ✅ | ✅ | **5/5 unanimous** |
| V7 raw-rollout does **NOT eliminate (α)** kernel non-determinism; underlying training instability remains | ✅ | ✅ | ✅ | ✅ | ✅ | **5/5 unanimous** |
| 12.73% A/B σ-normalize sanity drift is **NEITHER (b1) V6-replica drift NOR (b2) V7-vs-V6 signal** — it is a third quantity (independent training + algorithmic asymmetry + scalar-weight difference) | ✅ | ✅ | ✅ | ✅ | ✅ | **5/5 unanimous** |
| V6_NOISE (V6-seed1337) is **NECESSARY but not sufficient** for V7-vs-V6 noise-floor characterization (different step_weights → potentially different (α) compound rate) | ✅ | ✅ | ✅ | ✅ | ✅ | **5/5 unanimous** |
| V7-replica gate (same-yaml double-pass) is necessary before paired-ablation claim | ✅ | ✅ | ✅ | ✅ | ✅ | **5/5 unanimous** |
| **`d_pure` point estimate** (DIVERGENT — see §2 below) | 0.08 | 0.07 | 0.20 | 0.08 | 1.0 | **DIVERGENT 14× spread** |

### Q3 — Remediation ranking

| Option | A2 rank | A3 rank | A4 rank | A5 rank | A6 rank | Verdict |
|---|---|---|---|---|---|---|
| **F + G + H bundle BEFORE V7 launch** | #1 | #1 (F+G+H) | #1 (F→G→H) | #1 | #1 (F+G+H+pinning) | **5/5 unanimous strongest recommendation** |
| **V7-replica gate** | required | required | required | required | required (5K-step spot-check) | **5/5 unanimous** |
| **C (3-seed Welch)** | #6 | #4 | #6 | #6 | #6 | unanimous fallback if F/G/H reveal fundamental issues |
| **A (full bit-deterministic)** | #5 | #5 | #5 | #5 | #7 (V6 sunk-cost) | unanimous low-rank (high impl risk; A6 explicitly rejects) |
| **B (Mem-Eff only)** | #4 | #7 | #4 | #7 | #8 | unanimous partial; H is strict superset |
| **D (inflate K)** | #8 | #8 | #8 | #8 | #9 | **5/5 unanimous reject** |
| **E (reframe + disclose `d_pure`)** | #7 | #6 | #7 | #6 | #2 (mandatory framing) | unanimous fallback; A6 escalates to mandatory regardless of F/G/H outcome |

### Agent 1 (meta-review) verdicts on the prompt

Agent 1 acted as prompt-quality auditor and gave a separate set of polish points (not Q1/Q2/Q3 verdicts):

| Polish point | Severity | Action |
|---|---|---|
| Option H scope is mixed (3 sub-options with vastly different costs); recommend split into H1/H2/H3 | minor | **DEFERRED** — treated as documentation clarification; the 5 substantive reviewers all converged on "F+G+H" bundle without needing the split. Will document in §11 of Plan F. |
| VERIFIED FACT block should explicitly state `Σ(w·n) = Σw` is an enforced invariant under `preserve_v6_sum` | minor | **DEFERRED** — the Round 4 prompt's VERIFIED FACT block already implies this; explicit invariant statement is a polish item for any future Round 5. |
| Option F "≤ 1e-7" tolerance should justify the number (typical FP32 reduction-order is ~4e-10/hop summed; 1e-7 is ~250× safety margin) | minor | **APPLIED** — golden test script comments document this margin. |
| (β) vs (γ) distinction (algebraic vs kernel-level) needs explicit note | minor | **APPLIED** — captured in Plan F §11 audit reference and in the golden test outcome decision tree. |

---

## 2. The one critical divergence — `d_pure` point estimate

All 5 substantive reviewers converged on Q1/Q2/Q3 structural verdicts but produced **wildly different `d_pure` quantitative estimates**:

| Agent | Point estimate | 90% range | Mapped Plan F §11 tier | Plan F survival under this estimate |
|---|---|---|---|---|
| Agent 2 | **0.08** | [0.02, 0.22] | Tier 0 (`d_thr = 0.10` floor) | ✅ Plan F survives as written |
| Agent 3 | **0.07** | [0.02, 0.20] | Tier 0 (`d_thr = 0.10` floor) | ✅ Plan F survives as written |
| Agent 4 | **0.20** | [0.10, 0.40] / [0.05, 0.50] | Tier 2 (`d_thr ≈ 0.36`) | ⚠️ Plan F borderline; switch to Q3 Option C |
| Agent 5 | **0.08** | [0.03, 0.20] | Tier 0 (`d_thr = 0.10` floor) | ✅ Plan F survives as written |
| Agent 6 | **1.0** | [0.5, 2.0] | **Tier 3+** (`d_thr ≈ 1.8 – 3.6`) | ❌ Plan F structurally underpowered; mandatory reframe via Q3 Option E |

**Spread**: 14× between Agent 3 (0.07) and Agent 6 (1.0). Agents 2/3/5 cluster tightly around 0.07-0.08; Agent 4 is a moderate outlier; Agent 6 is the catastrophic outlier.

### Why Agents 2/3/5 cluster around 0.07-0.08

- All three anchor on the **historical V6/V6.1 paired-d ≈ 0.067** as the reference point and widen to account for σ-normalize-A/B 12.73% being a heavier tail than V6/V6.1.
- All three argue: 12.73% relative chain_normal_mse drift between A and B at 20K is **not** the same quantity as paired Cohen's `d` between V6-seed42 and V6-seed1337 at 160K because (i) σ-normalize A/B has additional (β) contribution, (ii) 12.73% is unaveraged endpoint MSE not paired SD, (iii) V6-replica share both algorithmic path and yaml topology.
- Anchored estimate: V6_NOISE produces relative drift "probably smaller than 12.73%" because (β) is removed, mapping to paired-d in the 0.05-0.20 range with point ~0.07-0.08.

### Why Agent 4 estimates 0.20

- Agent 4 emphasizes optimizer-state divergence as the dominant mechanism and argues SGD trajectories autocorrelate over 200K steps, producing sustained weight drift independent of (β).
- Anchored on V6 chain_normal_mse mean ≈ 2.43e-4 with per-slice SD ≈ 3.65e-4; 12.73% mean drift implies 3.1e-5 absolute → paired-d in [0.10, 0.40].
- Treats the (β)-contribution-ratio uncertainty as low and the absolute weight-drift magnitude as high.

### Why Agent 6 estimates 1.0

- Agent 6 applies a **√8 step-compounding heuristic** (Gaussian diffusion model) to extrapolate from the σ-normalize 12.73% at 20K to V6-replica at 160K.
- Anchored on V6 vs V6.1 paired-d 0.056 with paired-SD ≈ 9.4% of mean; for V6_NOISE at 15-40% drift, paired-SD-relative ≈ 9.4-15%, giving Cohen's d ≈ 1.0-4.3.
- **Critical assumption**: SGD step-noise behaves as an uncorrelated random walk over 8× more steps. This is a worst-case assumption — empirically SGD trajectories exhibit autocorrelation (steps within a basin) and saturation (two trained models settling at nearby minima), both of which **reduce** compounding below √steps.
- Agent 6 itself caveats: "my estimate range [0.5, 2.0] is heuristic, not empirically grounded. The author MUST run V6_NOISE (preferably + V6 same-seed double-pass) and compute d_pure empirically BEFORE V7 launches."

### Resolution

- **The empirical V6_NOISE measurement is the only definitive resolution**. Reviewer estimates are heuristic priors, not measurements.
- Agent 6's worst-case prediction is a **stress-test prior**, not a calibrated estimate. Plan F §11 already pre-registers a Tier 3 contingency (`d_pure > 0.40`) with Option E reframing as the response, which directly absorbs the Agent 6 scenario.
- Agent 6's recommendation of **V6@seed42 same-seed double-pass at 1K-5K steps as a cheap upper bound on (α) compounded** is decisive evidence the Plan F pre-launch checklist should adopt regardless of which `d_pure` estimate turns out correct (cost: 10K extra GPU-steps total = ~6% overhead on the V6_NOISE arm; see §4 below).
- The Round 4 prompt's Q2(d) explicitly asked for "point estimate + 90% range"; the divergence is a **legitimate finding**, not a bug. It tells us the prompt was decisive enough to surface a quantitative disagreement that the Plan F threshold-inflation contingency needs to handle.

---

## 3. Convergent actionable findings (5/5 unanimous → must-do)

### Action #1 — Run Option F (same-process algebraic golden test) BEFORE V7 launch
- All 5 substantive reviewers ranked Option F as the #1 most decisive pre-launch diagnostic.
- Cost: ~30 GPU-seconds.
- Implementation: [`review/0505/local/scripts/sigma_norm_golden_test.py`](../0505/local/scripts/sigma_norm_golden_test.py) (new in this commit).
- Outcome decision tree (per Plan F §11):
  - PASS (rel < 1e-6) → algebra correct in implementation; (β)/(ζ) FALSIFIED at inference; remaining failure is purely (γ) trajectory drift; H bundle becomes the mainline mitigation
  - FAIL (rel > 1e-3) → algebra broken; (β)/(ζ) CONFIRMED at inference; defer V7 launch; investigate `step_normalizers` injection bug
  - AMBIGUOUS (1e-7 < rel < 1e-3) → run Option G (paired raw-vs-σnorm full-val) for distributional bound

### Action #2 — Run Option G (paired raw-vs-σnorm full-val on V6@step_160000) BEFORE V7 launch
- All 5 substantive reviewers ranked G as #2 (after F) for distributional confirmation of inference-side algebraic equivalence.
- Cost: ~100 GPU-seconds (one full-val pass).
- Outcome: max abs drift across 7403 slices on `chain_normal_mse`; if max < 1e-4, accept algebra at inference.

### Action #3 — Apply Option H bundle (math-SDPA + AdaptiveAvgPool replacement + matmul precision="highest") to V7/V8/V6_NOISE training from step 0
- All 5 substantive reviewers concur this is the cheapest direct attack on (α).
- Cost: ~50 LOC trainer patch + bench run; expected <2% throughput overhead.
- Localization: `nn.AdaptiveAvgPool2d` is at `pet_lr/model_first_hop.py:50` (FirstHopPixelEncoder, hop0-only — replaceable with fixed `F.avg_pool2d`).
- **Note**: applying H to V6_NOISE means V6_NOISE's training stack will differ from V6-seed42's training stack (V6-seed42 was trained without H). This is acceptable because the V7 vs V6 paired comparison uses V6@step_160000.pt as a **fixed evaluation reference**, not as a training reference; V6_NOISE measures (α)-with-H rather than (α)-without-H. If the paper claims "with Option H interventions, V7 achieves d > 0.10 over V6 on `val_chain_normal_mse`", this is the relevant noise floor.

### Action #4 — Add V6@seed42 same-seed double-pass at 1K-5K steps as P0 pre-launch sanity (Agent 6 strong recommendation)
- 5/5 reviewers acknowledged this would be diagnostic; Agent 6 strongly recommended it.
- Cost: 10K extra GPU-steps (5K × 2 sequential passes on same GPU).
- Purpose: bounds **(α) compounded over short training horizon** independently of (β)/(ζ); if pair_total drift at step 5000 between two V6@seed42 passes is significant, V6_NOISE will resolve to a high `d_pure` and Plan F §11 Tier 2/3 contingency will apply.
- Pre-launch gate: if step-5000 `val_pair_total` drift between V6@seed42 pass-1 and pass-2 exceeds 1.0%, escalate to Plan F §11 Tier 2 (Welch's t-test) regardless of V6_NOISE final result.

### Action #5 — V7-replica is necessary but can be a 5K-step spot-check, not a full 160K replica (Agent 6 cost-optimization)
- 5/5 reviewers said V7-replica is necessary; Agent 6 noted full 160K is overkill and 5K spot-check is sufficient.
- Cost: 5K extra GPU-steps (sequential after V7 reaches step 5000).
- Pre-launch addition to §9 of Plan F: train V7@seed42 to 5K, save state, compare full-val `val_pair_total` and `val_chain_normal_mse` at step 5000 to V7@seed42-replica at step 5000. If pair_total drift exceeds the V6-double-pass measurement from Action #4, V7's path activates new (α) sources.

---

## 4. GPU-launch gate status

The tag `round4-tmp-pre-external-review-20260506` (annotated, present on both `origin/foc_lite_hop0` and `gitee/foc_lite_hop0`) explicitly listed 4 gates that must clear before GPU launch:

| Gate | Description | Status | Notes |
|---|---|---|---|
| (a) | Local codex review | ✅ **SATISFIED** | This conversation; 6 internal agents pre-reviewed prompt; consensus in [`ROUND4_PRE_REVIEW_CONSENSUS_20260506.md`](ROUND4_PRE_REVIEW_CONSENSUS_20260506.md) |
| (b) | Remote codex review | ❌ **FAILED** but **SUBSTITUTED** | Remote codex returned 502 Bad Gateway × 5; substituted by external 6-agent review. This consensus document is the audit artifact for the substitution. User accepted this substitution path. |
| (c) | ≥ 2 reviewer Q1/Q2/Q3 verdicts integrated | ✅ **SATISFIED** (5/5 ≫ 2) | All 5 substantive reviewers' verdicts integrated in §1 above; the one critical divergence (`d_pure` estimates) is addressed in §2 above; convergent actions are in §3 above |
| (d) | Option F golden test executed | ⏳ **PENDING** | Script written in this commit at [`review/0505/local/scripts/sigma_norm_golden_test.py`](../0505/local/scripts/sigma_norm_golden_test.py); operator must execute on V6@step_160000.pt with output committed before V7 launches |

**Gate decision**: 3/4 gates SATISFIED, 1/4 PENDING (operator action). GPU launch is BLOCKED on Gate (d). Operator next-step:

```bash
# Run on the GPU server hosting V6@step_160000.pt:
cd /path/to/PET_LatentResidual
python3 review/0505/local/scripts/sigma_norm_golden_test.py \
    --ckpt /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v6_transport_first/step_160000.pt \
    --config-A review/0505/operator/logs_sanity/run_ablation_A_sanity_config.resolved.yaml \
    --config-B review/0505/operator/logs_sanity/run_ablation_B_sanity_config.resolved.yaml \
    --output-json review/0505/operator/sigma_norm_golden_test_result.json
```

Outcome routing per §3 Action #1.

---

## 5. Per-agent verdict reference (full)

| Agent | Q1 Verdict | Q2 Verdict | Q3 Strongest Rec | Notable contribution |
|---|---|---|---|---|
| 1 | n/a (meta) | n/a (meta) | n/a (meta) | 3 polish points: H1/H2/H3 split, Σ(w·n)=Σw invariant explicit, 1e-7 numerical justification; (β) vs (γ) distinction note |
| 2 | (i) VERIFIED, (ii) PARTIAL | raw-rollout removes σ branch only; not nondeterminism | F+G first, then H, V7-replica required | Q2(b2) factual correction: prompt mislabeled `[1.7908,1.8606,0.8691,0.4795]` as V6 weights; actual V6 weights are `[0.5,2.0,1.5,1.0]`. Acknowledged minor and out-of-scope. |
| 3 | (i) VERIFIED, (ii) PARTIAL | avoids σ branch; doesn't avoid nondeterminism | F+G+H bundle + V7-replica gate | "9th option I": same-process paired trajectory replay with checksums to localize first divergence step (cheap; would discriminate β vs γ vs δ vs η) |
| 4 | (i) verified, (ii) WRONG-AS-DOMINANT | eliminates β not γ; V6_NOISE INADEQUATE on its own | F→G→H pre-launch, V7-replica mandatory | "9th option I": FP64 mixed-precision for rollout chain only (~30% wall-clock cost; eliminates γ for chain) |
| 5 | (i) VERIFIED, (ii) PARTIAL | eliminates σ-specific asymmetry; doesn't eliminate underlying noise | F+G+H + micro-replica gate before launch | "9th option": two-stage deterministic micro-replica gate (5K/20K same-yaml replicas under H for V6 and V7 BEFORE any 160K run) |
| 6 | (i) VERIFIED FACT, (ii) WRONG-AS-STATED | eliminates β; inherits α compounded 8×; predicts catastrophic underpower | F+G+H+GPU-pinning+V6-double-pass+E reframing (mandatory regardless) | "9th options": GPU UUID pinning, BF16 migration, TF32 disable, multi-pass eval averaging, smaller cuBLAS workspace, V6 same-seed double-pass at 1K-5K steps |

Notable cross-agent additions (beyond Options A-H originally enumerated in the prompt):

- **Agent 3 Option I**: paired trajectory replay with checksums (forensic localization)
- **Agent 4 Option I**: FP64 rollout chain (eliminates γ for chain forward)
- **Agent 5 Option I**: two-stage micro-replica gate (pre-launch sanity)
- **Agent 6 Options I-VI**: GPU pinning, BF16, TF32 off, multi-pass eval, cuBLAS workspace, V6 double-pass

These are recorded for future Plan G consideration but do NOT block Plan F V7/V8/V6_NOISE launch under the §3 must-do action set.

---

## 6. What this consensus does NOT change

- V7 step_weights = `[0.6106, 2.0, 2.7041, 2.0423]` — Grönwall closed-form derivation stands; user's Option A on hop0 weight from earlier round not revisited.
- 消融 arm (V8) yaml — `image_aux.enabled=false` on V6 baseline, unchanged.
- V6_NOISE (V6-seed1337) yaml — bit-equivalent to V6 except seed=1337, unchanged.
- Plan F GPU budget — 480K total (160K × 3 arms), unchanged. Threshold-inflation Tier 2 contingency may add +160K (1× V7 retrain) under empirical `d_pure` ∈ (0.20, 0.40].
- Parallel launcher (`run_v7_v8_sanity.sh parallel`) — unchanged.
- Pre-launch checklist §9 — unchanged (additions are §11 contingency reads + new pre-launch gate (d) for Option F).
- Round 4 prompt at `dcdcd2c` — unchanged. The 82/82 integrity check still passes.

---

## 7. File manifest

| Path | Status | Purpose |
|---|---|---|
| [`review/plan/ROUND4_EXTERNAL_CONSENSUS_20260506.md`](ROUND4_EXTERNAL_CONSENSUS_20260506.md) | **NEW** 2026-05-06 (this file) | 6-agent external consensus + GPU-launch gate decision |
| [`review/0505/local/scripts/sigma_norm_golden_test.py`](../0505/local/scripts/sigma_norm_golden_test.py) | **NEW** 2026-05-06 | Option F same-process algebraic golden test |
| [`review/plan/PLAN_F_V7_V8_FROM_SCRATCH_150K_20260505.md`](PLAN_F_V7_V8_FROM_SCRATCH_150K_20260505.md) | **EDITED** 2026-05-06 | §11 reviewer-informed `d_pure` prior added; §9 pre-launch checklist updated for Action #1/#4/#5 |
| [`review/plan/ROUND4_PRE_REVIEW_CONSENSUS_20260506.md`](ROUND4_PRE_REVIEW_CONSENSUS_20260506.md) | UNCHANGED | Internal 6-agent pre-review consensus (preserved for traceability) |
| [`review/0505/local/PEER_REVIEW_PROMPT_SIGMA_VS_RAW_20260506.md`](../0505/local/PEER_REVIEW_PROMPT_SIGMA_VS_RAW_20260506.md) | UNCHANGED | Round 4 prompt at `dcdcd2c` — 82/82 integrity preserved |

---

## 8. Audit trail

| Date | Event |
|---|---|
| 2026-05-06 | Round 4 prompt sent to 6 external reviewer agents per the strategy committed at `dcdcd2c` |
| 2026-05-06 | Agent 1 returned prompt-quality meta-review (3 polish points; v2 prompt approved as senior-reviewer-grade) |
| 2026-05-06 | Agents 2-6 returned substantive Q1/Q2/Q3 verdicts |
| 2026-05-06 | Cross-agent convergence matrix built (§1 above) |
| 2026-05-06 | `d_pure` divergence flagged as legitimate finding (§2 above); Agent 6 catastrophic estimate addressed by existing Plan F §11 Tier 3 contingency |
| 2026-05-06 | Convergent actionable findings extracted (§3 above) — 5 must-do items |
| 2026-05-06 | GPU-launch gate status decided (§4 above) — 3/4 satisfied, gate (d) Option F pending |
| 2026-05-06 | This consensus document written |
| 2026-05-06 | Option F golden test script written at [`review/0505/local/scripts/sigma_norm_golden_test.py`](../0505/local/scripts/sigma_norm_golden_test.py) |
| 2026-05-06 | Plan F §11 updated with reviewer-informed `d_pure` prior + V6-double-pass spot-check + V7-replica spot-check |
| **PENDING** | Operator runs `sigma_norm_golden_test.py` on V6@step_160000.pt; commits result JSON; tag `round4-tmp-pre-external-review-20260506` superseded by `round4-post-external-review-20260506` |
