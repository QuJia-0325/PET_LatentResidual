# Round 4 internal pre-review consensus — σ-normalize vs raw-rollout

**Date**: 2026-05-06
**Trigger**: Before sending [`PEER_REVIEW_PROMPT_SIGMA_VS_RAW_20260506.md`](../0505/local/PEER_REVIEW_PROMPT_SIGMA_VS_RAW_20260506.md) to external reviewers (GPT-5.5 / Claude Opus 4.7 / optional Gemini-2.5-Pro), I ran 6 internal agents against the prompt:
- **4 reviewer-role agents** (Agents 1, 3, 5, 6): answer Q1/Q2/Q3 as-if external reviewer
- **2 prompt-quality-role agents** (Agents 2, 4): audit the prompt itself for missing constraints, leading framing, omitted candidate hypotheses

This pre-review exposed defects that would have caused external reviewers to give well-reasoned but **evidence-light** verdicts (e.g., agreeing with the author's "kernel non-determinism" framing without addressing structurally-incompatible evidence). The Round 4 prompt has been amended in this same commit; this document records the consensus and pre-registers the response actions.

---

## 1. Convergent findings (≥3 agents flagged independently)

### Finding A — `val_pair_total = 10.16%` is reverse-evidence the prompt was hiding
**Flagged by**: Agents 2, 4 (explicit critical defect); Agent 6 (implicit via diagnosis structure).
**Mechanism**: `compute_pair_losses` is defined at [`train_first_hop.py:292-330`](../../train_first_hop.py#L292-L330) and verified to NOT call `rollout_multistep_losses_first_hop`, NOT pass through `mix_latent`, NOT receive `step_normalizers`. Pair-loss is a single-step velocity loss on `hop_idx ∈ {0,1,2,3}` randomly sampled per batch — structurally outside the rollout chain.
**Implication**: The 10.16% drift on `val_pair_total` between A and B (same seed, 20K steps) **falsifies** the author's "trajectory amplification through `mix_latent`" diagnosis. Whatever mechanism produced 10.16% pair_total drift must operate on the *trained model weights*, i.e., 20K steps of SGD under non-bit-deterministic forward+backward yielded measurably different parameters in A vs B.
**Prompt fix**: Critical observation block added before Q1; Q1(b) revised to require addressing pair_total constraint; Hard Constraint #6 added making pair_total reconciliation non-negotiable.

### Finding B — Same-process algebraic golden test should be a P0 prerequisite, not a Q3 option
**Flagged by**: Agents 1, 3, 4, 5, 6 (all reviewer-role agents converged on this; Agent 4 explicitly noted it was missing as Q3 option).
**Mechanism**: Load V6@step_160000.pt in one Python process, freeze RNG, force `model.eval()` + `torch.backends.cuda.enable_math_sdp(True)`, compute A path total and B path total on one fixed batch. With `pair_v_std` non-persistent + math-SDPA + eval-mode, the only remaining drift source is FP32 reduction-order. Cost: ~30 GPU-seconds. **Decisive** discriminator between hypotheses (α) kernel-noise-only / (β) FP32-reduction-order-only / (ζ) algebraic-bug.
**Cost**: <1 GPU·minute. Does not block V7 launch.
**Prompt fix**: Added as Option F in Q3 with concrete protocol; renumbered "6th option" → "9th option"; "After reviewer replies" §3 escalated to "P0 if ≥2 reviewers concur"; Plan F §11 makes Option F a **pre-launch** action regardless of reviewer verdict.

### Finding C — `pair_v_std` non-persistent buffer fact was missing from Background
**Flagged by**: Agent 6 (verified at [`pet_lr/model_first_hop.py:336`](../../pet_lr/model_first_hop.py#L336) via `register_buffer("pair_v_std", ..., persistent=False)`).
**Implication**: σ-normalize's per-hop normalizer `n_j = (σ_j·dt_j)² / weighted_mean(ρ, anchor)` is **constant** across training (does not depend on optimizer state). In **exact arithmetic**, A's `(w_v6·ℓ).sum()/Σw_v6` = B's `((w_v6·n)·(ℓ/n)).sum()/Σ(w_v6·n)`. Algebraic equivalence holds **on paper**. The remaining FP32 drift is purely implementation-side reduction-order asymmetry from B's `step_loss / float(n_j)` host-scalar division (only on B path, [`pet_lr/rollout_first_hop.py:103`](../../pet_lr/rollout_first_hop.py#L103)).
**Prompt fix**: Added as VERIFIED FACT in Background; Q1(b) extended to enumerate (β) FP32 reduction-order asymmetry as a candidate distinct from (α) kernel non-determinism; Q1(d) updated with the bit-identical expectation.

### Finding D — `1.8 × d_pure` threshold can inflate to 0.18-0.72 (under-power risk)
**Flagged by**: Agents 1, 3, 5, 6 (Agent 6 with quantitative estimates `d_pure ∈ [0.10, 0.40]` plausible).
**Mechanism**: σ-normalize sanity at 12.73% on `val_chain_normal_mse` between A and B (same seed, 20K steps) implies non-trivial run-to-run drift even on identical settings. Extrapolating to V6-seed42 vs V6-seed1337 (different seeds, same yaml, 160K steps), `d_pure` is plausibly larger than the 0.056 originally assumed under "V6/V6.1 = 0.067 includes algorithmic+selection contamination so RNG-only is smaller". If `d_pure > 0.40`, `d_thr = 1.8 × d_pure > 0.72` — Plan F's adequately-powered test becomes a no-detection-possible test.
**Prompt fix**: Q2.d expanded to require quantitative point estimate + 90% range for `d_pure`; Plan F §11 added with 4-tier contingency table (`d_pure ≤ 0.10` / `0.10–0.20` / `0.20–0.40` / `> 0.40`).

### Finding E — Per-hop monotonicity claim was strictly false
**Flagged by**: Agent 2 (rigorous reading of the failure table).
**Mechanism**: Original prompt said "error grows monotonically with hop index" but actually hop0 = 0.26%, hop1 = 0.25% (hop1 < hop0). Strict monotonicity false; the pattern is "essentially tied at sub-0.3% on hop0/hop1, then jumps to 1.88%/4.92% at hop2/hop3". Tiny but reviewers will catch it and lose trust in the prompt.
**Prompt fix**: Replaced with the corrected description.

### Finding F — A/B drift is NOT a clean upper bound on V6-vs-V6-replica RNG drift
**Flagged by**: Agents 4, 6 (explicit), Agent 3 (implicit).
**Mechanism**: A and B differ in BOTH (i) algorithmic path (raw vs σ-normalize) AND (ii) yaml step_weights ([1.7908,1.8606,0.8691,0.4795] vs [1.0,1.0,1.0,1.0] × normalizers). They share only seed and topology. Therefore the 12.73% A/B drift confounds three sources: kernel-noise + algorithmic-asymmetry + scalar-weight-difference. Using 12.73% as a direct lower bound on `d_pure` (V6-vs-V6-replica RNG only) is a category error.
**Prompt fix**: Q2(b) split into (b1) V7-vs-V6-replica drift bound and (b2) V7-vs-V6 drift bound (different `step_weights`); explicit note that A/B is **neither** of these and must be discussed as a third quantity.

### Finding G — Option H bundle (math-SDPA + AdaptiveAvgPool replacement + matmul precision) was missing
**Flagged by**: Agent 6 (strongly), Agent 4 (partially).
**Mechanism**: Three concrete determinism interventions exhaust the warn-only-flagged ops in the train log: (i) `enable_math_sdp(True)` + `enable_flash_sdp(False)` + `enable_mem_efficient_sdp(False)` eliminates Memory-Efficient-attention non-determinism; (ii) replace `nn.AdaptiveAvgPool2d` with `F.avg_pool2d` at fixed kernel size — eliminates the `adaptive_avg_pool2d_backward_cuda` warning; (iii) `torch.set_float32_matmul_precision("highest")` forces FP32 matmul. Combined overhead estimated <2%. Cost: ~50 LOC + one bench run. Together they eliminate ALL documented warn-only ops.
**Prompt fix**: Added as Option H in Q3; "After reviewer replies" §4 escalated Option H as the second-tier remediation if Q2 deems raw-rollout also unsafe.

### Finding H — Constraint #1 (every claim cites file path with line number) was over-tight for theory claims
**Flagged by**: Agent 4.
**Mechanism**: Theoretical claims like "FP32 reduction-order asymmetry produces drift O(ε·N)" cannot cite a file:line in this codebase — they need textbook citation or established-result citation. Original Constraint #1 forced reviewers to either skip such claims or fabricate citations.
**Prompt fix**: Constraint #1 split into factual (file:line cite required) vs theoretical (textbook/paper cite acceptable, must be explicit about where authority comes from).

---

## 2. Per-agent verdicts (summary)

For traceability. Full agent outputs are not stored in this commit (kept in `transcripts/2c103958-6142-45c0-ba2d-226ce4501cf2.jsonl` for audit).

| Agent | Role | Q1 verdict | Q2 verdict | Q3 strongest rec |
|---|---|---|---|---|
| 1 | reviewer | (i) VERIFIED, (ii) PARTIAL — kernel-noise + amplification plausible but unproven; recommends same-process golden test before declaring | raw-rollout structurally avoids σ-normalize-specific algebra branch but NOT kernel non-determinism; V6_NOISE adequate as lower bound, inadequate as sole guarantee for V7 | A if it launches; otherwise C+E hybrid; explicit support for Option F as P0 |
| 2 | prompt-quality | n/a | n/a | n/a — flagged 3 prompt defects: missing val_pair_total reverse-evidence, monotonicity wrong, attachment paths inconsistent |
| 3 | reviewer | (i) VERIFIED, (ii) PARTIAL — same as Agent 1 with different emphasis on Adam-state divergence | similar to Agent 1; explicit concern that V6_NOISE only changes seed, not kernels | Option F first; if pass → C; if fail → re-engineer |
| 4 | prompt-quality | n/a | n/a | n/a — flagged: missing Option F+G, Q1(b) "≥3 alternatives" too small, Constraint #4 conflicts with body framing, last_eval row CV concern, Q2(b) confound on step_weights |
| 5 | reviewer | (i) VERIFIED, (ii) PARTIAL — emphasizes optimizer-state divergence as primary | similar; recommends V7-replica + V6-NOISE together | A if affordable; F as gating test |
| 6 | reviewer (most thorough) | (i) VERIFIED, (ii) WRONG-AS-EXCLUSIVE-CAUSE — pair_v_std non-persistent fact verified; algebraic equivalence holds in exact arithmetic; FP32 reduction-order is a SECOND mechanism independent of kernel non-determinism | quantitative `d_pure ∈ [0.10, 0.40]` plausible → Plan F threshold-inflation risk needs pre-registration | E + same-process golden test + 20K-step V7-replica spot-check + SDPA/AdaptiveAvgPool determinism fixes (Option H bundle) |

---

## 3. P0 actions before sending Round 4 prompt to external reviewers

Per consensus, these should be done **before** the prompt is sent externally:

1. **Apply prompt edits** (Findings A–H above). ✅ DONE in this commit. See [`PEER_REVIEW_PROMPT_SIGMA_VS_RAW_20260506.md`](../0505/local/PEER_REVIEW_PROMPT_SIGMA_VS_RAW_20260506.md).
2. **Add threshold-inflation contingency to Plan F**. ✅ DONE in this commit. See [`PLAN_F_V7_V8_FROM_SCRATCH_150K_20260505.md §11`](PLAN_F_V7_V8_FROM_SCRATCH_150K_20260505.md).
3. **Run same-process algebraic golden test (Option F)** on V6@step_160000.pt. **NOT YET DONE**. Cost: ~30 GPU-seconds. Decisive discriminator. Result feeds back into the Round 4 prompt as evidence (rather than asking external reviewers to speculate). If golden test passes → external reviewers will likely converge on (γ) optimizer-state divergence as primary cause; if fails → external review can focus on the algebraic-bug branch.
4. **(Optional) Run paired raw-vs-σnorm full-val evaluator (Option G)** on V6@step_160000.pt across 7403 slices. Cost: ~100 LOC, 0 GPU-day retrain. Provides distributional bound on inference-side algebraic equivalence.

**Recommended sequencing**:
- **Path A (if user prefers fast external feedback)**: Send refined Round 4 prompt now; run Option F in parallel. External reviewers will likely flag missing Option F result; respond by providing it in their second pass.
- **Path B (if user prefers strongest evidence first)**: Run Option F first. Append result to Round 4 prompt as a new "Pre-flight evidence" section. Then send. External reviewers will give evidence-grounded verdicts.

---

## 4. What this consensus does NOT change

- V7 step_weights = [0.6106, 2.0, 2.7041, 2.0423] — Grönwall closed-form derivation stands. User chose Option A on hop0 weight in earlier round; not revisited here.
- 消融 arm (V8) yaml — image_aux.enabled=false on V6 baseline, unchanged.
- V6_NOISE (V6-seed1337) yaml — bit-equivalent to V6 except seed=1337, max_steps=160000, no `best_select_full_eval_interval`. Unchanged.
- Plan F GPU budget — 480K total (160K × 3 arms), unchanged. Threshold-inflation contingency adds at most +160K (1× V7 retrain) under Tier 2.
- Parallel launcher (`run_v7_v8_sanity.sh parallel`) — unchanged. GPU2/3 dual-card schedule.
- Pre-launch checklist §9 — unchanged. Add §11 contingency reading after V6_NOISE completes.

---

## 5. File manifest

| Path | Status | Purpose |
|---|---|---|
| [`review/0505/local/PEER_REVIEW_PROMPT_SIGMA_VS_RAW_20260506.md`](../0505/local/PEER_REVIEW_PROMPT_SIGMA_VS_RAW_20260506.md) | EDITED 2026-05-06 | Round 4 prompt with 8 convergent fixes applied |
| [`review/plan/PLAN_F_V7_V8_FROM_SCRATCH_150K_20260505.md`](PLAN_F_V7_V8_FROM_SCRATCH_150K_20260505.md) | EDITED 2026-05-06 | §11 threshold-inflation contingency added; audit trail entry |
| [`review/plan/ROUND4_PRE_REVIEW_CONSENSUS_20260506.md`](ROUND4_PRE_REVIEW_CONSENSUS_20260506.md) | NEW 2026-05-06 (this file) | 6-agent consensus + P0 action sequencing |
| [`review/plan/PLAN_F_PARALLEL_LAUNCH_DESIGN_20260506.md`](PLAN_F_PARALLEL_LAUNCH_DESIGN_20260506.md) | UNCHANGED | GPU2/3 parallel launcher design (separate concern) |
| [`review/0505/local/scripts/run_v7_v8_sanity.sh`](../0505/local/scripts/run_v7_v8_sanity.sh) | UNCHANGED | Launcher with parallel mode |

---

## 6. Audit trail

| Date | Event |
|---|---|
| 2026-05-06 | Round 4 prompt drafted, 50/50 integrity check passed |
| 2026-05-06 | 6 internal agents run against prompt (4 reviewer-role + 2 prompt-quality-role) |
| 2026-05-06 | 6-agent consensus extracted (Findings A–H) |
| 2026-05-06 | Round 4 prompt amended with 8 fixes; `pair_v_std` non-persistent fact verified at `pet_lr/model_first_hop.py:336`; `compute_pair_losses` no-rollout fact verified at `train_first_hop.py:292-330` |
| 2026-05-06 | Plan F §11 threshold-inflation contingency added |
| 2026-05-06 | This consensus document written |
| **TODO** | Decide Path A vs Path B sequencing (run Option F first, or send prompt first); operator action required |
