# V9 NORMAL-emphasis — pre-registration (frozen 20260516)

- generated_at: 2026-05-16 Asia/Shanghai
- branch: foc_lite_hop0
- commit_when_generated: acc7a9d
- config under test: [V9_normal_emphasis.yaml](V9_normal_emphasis.yaml)
- baseline: V7 (`step_160000.pt`), full-val PSNR_clip3 from [planf_v7_last_fullval_psnr_chain_mse.json](../../0511/fullval_psnr_clip3_20260516_173941/artifacts/planf_v7_last_fullval_psnr_chain_mse.json)
- theory reference: [review/0516/STEP_WEIGHTS_THEORY_REFERENCE.md](../STEP_WEIGHTS_THEORY_REFERENCE.md) (multi-hop Grönwall closed-form + σ·dt magnitude compensation; V7 and V9 step_weights are both instantiations of the same formula)
- seed noise floor d_pure: established in [review/0516/PLANF_FINAL_ANALYSIS_20260516.md §2.1](../PLANF_FINAL_ANALYSIS_20260516.md)
  - NORMAL MSE d_pure = 0.37%
  - NORMAL PSNR d_pure = 0.026 dB

This file is frozen BEFORE V9 is launched. Do not edit thresholds after seeing
any V9 result. If a threshold turns out to be wrong, document the lesson in a
new postmortem; do not rewrite the prior.

---

## 0. Pre-launch gate — Lipschitz measurement (cheap, MUST run first)

Before committing GPU days to V9, run the 10-minute Lipschitz check from
[STEP_WEIGHTS_THEORY_REFERENCE §6](../STEP_WEIGHTS_THEORY_REFERENCE.md) using
[tools/estimate_per_hop_lipschitz.py](../../../tools/estimate_per_hop_lipschitz.py)
on V7 `step_160000.pt`. The script wiring (model/data API) is a TODO; doing
that wiring is part of the gate.

Branch decision based on `alignment_distance_pct` (TV distance between current
V7 step_weights and closed-form with measured L_j, both normalized to sum 1):

| Lipschitz gate result | next action |
|---|---|
| `alignment_distance_pct < 10%` and `max(L_j) < 1.10` | Hypothesis `L_i ≈ 1` confirmed. V7 step_weights ARE the closed-form. Launch V9 as planned (§2-§6 below). |
| `10% ≤ alignment_distance_pct < 25%` | `L_i ≈ 1` partially holds. Launch V9 AND in parallel record measured L_j into [V9_normal_emphasis.yaml](V9_normal_emphasis.yaml) header comment for paper traceability. |
| `alignment_distance_pct ≥ 25%` OR `max(L_j) ≥ 1.15` | `L_i ≈ 1` FAILS. Do NOT launch V9 as-is. Instead derive V9' step_weights from §2.2 closed-form with the measured L_j, and re-do this pre-registration with V9' (one-axis at a time). The current V9 changes would be confounded by L_j re-discovery. |

This gate is binding. The Lipschitz measurement is the single highest-leverage
pre-V9 experiment because it costs ~10 minutes and can either validate the
entire V7/V9 closed-form derivation or invalidate the `L_i ≈ 1` premise that
both runs depend on.

---

## 1. Hypothesis

H1 (primary, NORMAL-bottleneck hypothesis):
> Increasing β_NORMAL from 1.50 to 2.50 in `best_metric_terms` (and the
> corresponding closed-form `step_weights`, re-derived in the V9 config header)
> reduces full-val NORMAL chain MSE at step 160000 by more than 3×d_pure,
> measured by paired statistics over 7403 val slices.

H0 (null): the V7→V9 NORMAL-PSNR delta is within seed noise.

Honest alternative H2 (trade-off): NORMAL improves but D20/transport-avg
regresses; recorded as "trade-off", not "success".

---

## 2. Single variable under test

Only one knob is changed between V7 and V9:

| field | V7 | V9 |
|---|---|---|
| `training.best_metric_terms[normal].weight` | 1.50 | **2.50** |
| `training.rollout.step_weights` | [0.6106, 2.0000, 2.7041, 2.0423] | **[0.5870, 2.0000, 2.8333, 2.5173]** |

Everything else (seed, image_aux, schedules, optimizer, EMA, init, dataset)
is byte-identical to V7. The closed-form re-derivation is mechanical from β
(see V9 config header for algebra) — the step_weights change is not a free
parameter, it is forced by changing β under the same Grönwall framework.

---

## 3. Pre-registered evaluation protocol

Identical to the protocol used for V7/V8/V6_NOISE in
[review/0511/fullval_psnr_clip3_20260516_173941](../../0511/fullval_psnr_clip3_20260516_173941):

- After training completes at step 160000:
  - Run `eval_first_hop_224_clip3.py` on V9 `step_160000.pt` (= V9 `last.pt`)
    AND V9 `best.pt`.
  - `--max-slices 0` (full val loader, 7403 slices).
  - `--decode-mode both`.
  - `--config review/0516/V9_normal_emphasis/V9_normal_emphasis.yaml`.
  - Output to `review/0517/fullval_psnr_clip3_<timestamp>/artifacts/`.
- Paired t-test and per-slice win rate computed against V7 last per-slice CSV
  using the same Ruby snippet from
  [review/0516/PLANF_FINAL_ANALYSIS_20260516.md §2.2](../PLANF_FINAL_ANALYSIS_20260516.md).

---

## 4. Decision thresholds (frozen)

### 4.1 Primary success criteria — all three must hold

| criterion | threshold | rationale |
|---|---|---|
| ΔNORMAL_PSNR (V9 − V7) | **≥ +0.078 dB** | 3 × d_pure (0.026 dB) — the smallest defensible effect given seed noise |
| paired t-stat NORMAL | **> 5.0** | comparable rigor to image_aux (which had t ≈ 74); 5 is the minimum we accept as "not chance" |
| per-slice NORMAL win rate | **> 60%** | rules out a small mean driven by outliers; image_aux had 91.5%, image_aux benchmark sets the upper bar but 60% is the minimum we accept |

### 4.2 Trade-off / regression guards

| guard | condition that triggers | label |
|---|---|---|
| transport-avg PSNR | V9 − V7 < 0 dB | "trade-off worse" |
| D20 PSNR | V9 − V7 < −0.10 dB | "early-cascade regression" |
| Failure of any of §4.1 with no §4.2 trigger | — | "null" (V7 unchanged as baseline) |

### 4.3 Decision matrix

| §4.1 result | §4.2 result | label | next action |
|---|---|---|---|
| all pass | none triggered | **SUCCESS** | promote V9 step_weights derivation method as canonical; consider β_NORMAL=4.0 follow-up |
| all pass | trade-off triggered | **MIXED** | keep V7 as baseline for D20-sensitive claims, V9 for NORMAL-only claims; do not promote |
| any fail | none triggered | **NULL** | record β-weighting as exhausted lever; move to architecture/conditioning ablation |
| any fail | trade-off triggered | **REGRESSION** | record as failed direction; consider OPPOSITE direction (β_NORMAL < 1.5) as a sanity follow-up |

---

## 5. Operational notes

- Run on the same GPU pool as V7/V8/V6_NOISE to keep wall-clock comparable.
- `require_fresh_output_dir: true` is set in the config; existing dirs will be
  rejected. This protects against accidental resume from a prior run.
- Estimated cost: identical to V7 wall-clock (~7 days on RTX A6000 single GPU).
- Concurrent launches: do NOT pair V9 with another ablation that touches loss
  weights, because thermal/IO contention can perturb timing and seed effects.
  V9 should run alongside non-loss-weight experiments only (e.g. architecture).

---

## 6. Pre-registered next-experiment options (conditional on V9 outcome)

To avoid post-hoc cherry-picking, list the next experiment for each label up
front.

| V9 outcome | next experiment |
|---|---|
| SUCCESS | V10 = β_NORMAL = 4.0 (probe the upper bound of NORMAL emphasis); same eval protocol |
| MIXED | V10 = σ-normalize rollout ablation ([STEP_WEIGHTS_THEORY_REFERENCE §5.2](../STEP_WEIGHTS_THEORY_REFERENCE.md), [ARCHITECTURE_ANALYSIS §20.6 action 7](../../plan/ARCHITECTURE_ANALYSIS_20260501.md)) on V7 baseline; this is the long-pending critical ablation that decouples Grönwall weighting from σ·dt magnitude compensation |
| NULL | V10 = σ-normalize rollout ablation (same as MIXED); β-axis exhausted, time to attack the other unverified assumption |
| REGRESSION | V10 = β_NORMAL = 1.0 sanity check, brief 80K-step probe to confirm direction; then σ-normalize |

This pre-registration is binding. The selected V10 will use the protocol in §3.
