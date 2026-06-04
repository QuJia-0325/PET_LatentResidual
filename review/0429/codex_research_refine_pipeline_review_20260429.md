# Codex Research-Refine-Pipeline Review: V5/V6/Claude Analysis

Date: 2026-04-29
Branch: `foc_lite_hop0`
Remote baseline reviewed: `d393229 analysis: V5 attribution + V6 Phase I + E1 error budget + figure prompts`

## 1. Scope

This note reviews the latest Claude/Gitee analysis against the current code and available live metrics. The goal is not to re-state the remote analysis, but to identify which conclusions are code-grounded, which are over-interpreted, and what experiment protocol should be used next.

Reviewed materials:

- `review/0428/reviewer/v6_phase1_viewer_analysis_20260429.md`
- `review/0429/supervisor/v6_phase1_e1_supervisor_review_20260429.md`
- `review/0429/e1_core_hypothesis_viewer_assessment_20260429.md`
- `review/0429/e1_error_budget_decomposition_explained.md`
- `review/0428/supervisor/v5_v6_supervisor_addendum_20260429.md`
- `review/0428/reviewer/v5_attribution_viewer_update_20260429.md`
- `scripts/diagnose_error_budget.py`
- `configs/pet_flow/pet_flow_first_hop_224_v6_transport_first.yaml`
- `train_first_hop.py`

Live metric snapshots inspected:

- V6: `/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v6_transport_first/metrics.jsonl`
- V5 null-control: `/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v5_null_control/metrics.jsonl`

## 2. Executive Conclusion

Claude's latest analysis is mostly reasonable and significantly more careful than earlier versions. The strongest correct points are:

- V6 Phase I is a pilot, not a full 200K-approved method.
- V6 intentionally suppresses image auxiliary dominance and front-loads pair transport supervision.
- E1 should be interpreted as a decoded-space PSNR diagnostic, not a strict linear MSE attribution.
- V5 attribution is not closed without full-val and Path-A-style controls.
- V5 and V6 should not be mixed into one claim; they test different hypotheses.

However, several conclusions are still too strong or under-specified:

- V6 Phase I chain degradation is directionally expected because `lambda_roll=0`, but the observed magnitude should remain a risk signal rather than being fully dismissed as harmless warmup behavior.
- The proposed V7 fallback floor `lambda_roll=0.05` is a heuristic. It must be calibrated using actual weighted loss fractions after rollout is enabled, not only from rough V3-scale reasoning.
- V6 stop/go decisions should not be based on one rolling-val point. Current validation appears to use rolling/sampled batches, so gates need consecutive-window or fixed-slice confirmation before killing or endorsing a run.
- E1's MSE-ratio argument is useful but still not a strict per-case MSE decomposition unless raw per-slice/per-case MSE aggregation is added or reported.

My refined position:

> The current bottleneck is not primarily the decoder ceiling. It is transported-latent trajectory consistency under open-loop rollout. Pair-only transport pressure can make local transitions numerically small while still failing to preserve chained decoded image quality. V6 is the right direction as a pilot, but the central variable is the schedule/weighting protocol that balances pair supervision and open-loop chain supervision, not merely increasing `pair_weight`.

## 3. Code-Grounded Checks

### 3.1 V6 schedule and loss composition

V6 config uses a transport-first schedule:

- `configs/pet_flow/pet_flow_first_hop_224_v6_transport_first.yaml`
- `rollout_warmup_steps: 50000`
- `rollout_ramp_steps: 25000`
- `rollout_lambda: 0.35`
- `pair_weight: 15.0`
- `image_aux.lambda_img: 0.04`
- `transport_steps.weight_d20: 2.0`
- `transport_steps.weight_d10: 1.3`

The training code confirms the effective total loss is:

```python
loss = pair_loss_weight * loss_pair_total + lambda_roll * loss_roll_total + lambda_img * loss_img
```

Relevant code locations:

- `train_first_hop.py`: `_resolve_schedule_steps` resolves ratio/step schedules.
- `train_first_hop.py`: `resolve_rollout_lambda` controls rollout weight schedule.
- `train_first_hop.py`: `compute_rollout_losses` computes rollout losses.
- `train_first_hop.py`: training loop combines pair, rollout, and image losses into total loss.
- `train_first_hop.py`: logged `pair_frac`, `roll_frac`, `img_frac` are weighted loss fractions.

Important interpretation:

- `pair_frac`, `roll_frac`, and `img_frac` are not direct gradient-norm fractions.
- They are still useful for diagnosing objective dominance.
- During V6 Phase I, `lambda_roll=0`, so rollout loss can be computed/logged but contributes zero gradient to the optimized objective.

### 3.2 V6 current live state

At the inspected snapshot, V6 was still in Phase I:

```text
last_train_step = 44000
pair_frac       = 0.982692
roll_frac       = 0.000000
img_frac        = 0.017308
lambda_roll     = 0.000000
lambda_img      = 0.040000
alpha           = 0.000000
pair_weight     = 15.000000
```

Recent validation snapshot:

```text
last_val_step          = 43600
val_pair_total         = 1.8903e-07
val_rollout_total      = 1.0444e-03
val_chain_d20_mse      = 3.8674e-04
val_chain_d10_mse      = 3.9507e-04
val_chain_d4_mse       = 4.3137e-04
val_chain_normal_mse   = 7.4363e-04
val_chain_tail_mse     = 5.2336e-04
val_select_score       = 1.8748e-03
```

Interpretation:

- V6 has successfully made pair transport the dominant optimized term.
- Pair validation loss is extremely low by this snapshot.
- Chain/open-loop validation is still poor because rollout supervision has not started.
- This supports Claude's claim that V6 Phase I tests local transport first, not final chain quality.

Risk:

- The pair loss appears saturated while chain quality is still bad. This is exactly the failure mode where local transition loss is not sufficient for open-loop composition.
- Therefore, the post-50K rollout ramp is the real experiment. Phase I alone should not be interpreted as success.

## 4. Where Claude's Analysis Is Reasonable

### 4.1 V6 should be treated as GO pilot, not full approval

This is correct. Code and live metrics both show that V6 is still in the pre-rollout or early-rollout protocol. The meaningful question is whether chain quality recovers once `lambda_roll` and `alpha` ramp after 50K.

### 4.2 V6 degradation before rollout is expected in direction

This is mostly correct. Since `lambda_roll=0` before 50K, the model is not directly optimized for open-loop rollout consistency. Evaluation uses open-loop behavior, so a train/eval distribution mismatch is deliberately introduced.

However, this should be stated as:

> Directionally expected, but magnitude-sensitive.

It is not enough to say the degradation is harmless. If chain MSE remains high after rollout weight becomes non-trivial, it means pair-heavy warmup either failed to build useful composable transport or induced representation drift that rollout cannot repair efficiently.

### 4.3 E1 is a diagnostic, not strict MSE-linear attribution

This correction is important and correct. PSNR is logarithmic, and decoder-mediated decoded-space PSNR cannot be interpreted as a linear additive decomposition.

The latest E1 wording is better because it frames the analysis as:

- decoded-space transport discrepancy;
- decoder ceiling/floor measurement;
- evidence that transport-related discrepancy is large relative to decoder reconstruction floor.

### 4.4 V5 attribution is not closed without full-val and controls

This is correct. V5 null-control remains a useful control, but rolling validation and partial snapshots are not enough to make a final attribution claim.

## 5. Problems Or Over-Strong Claims

### 5.1 V6 Phase I degradation is not automatically harmless

Claude's current framing reduces the severity too much. The degradation is expected under `lambda_roll=0`, but the current metrics still matter:

```text
V6 step 43600:
val_pair_total       ~= 1.89e-07
val_chain_normal_mse ~= 7.44e-04
val_chain_tail_mse   ~= 5.23e-04
```

This shows a large gap between local pair transport and chained decoded quality. That gap is the core bottleneck. If rollout ramp does not quickly improve it, V6's pair-first premise is insufficient.

Recommended wording:

> V6 Phase I intentionally sacrifices chain performance to learn local transport. This is acceptable only if chain metrics recover shortly after rollout/alpha ramp starts. Otherwise, the pair-first warmup is not just incomplete but actively misaligned with the final open-loop objective.

### 5.2 V7 `lambda_roll=0.05` fallback is not yet calibrated

Claude suggests `lambda_roll floor=0.05` and rejects `0.002` because `pair_weight=15` would drown it. The direction is reasonable, but the numeric value is not yet proven.

Why:

- The effect of `lambda_roll` depends on raw rollout loss scale and raw pair loss scale at that training stage.
- V6 pair loss can become extremely small on validation while train pair loss is larger and batch-dependent.
- A fixed rollout floor should be judged by observed `roll_frac`, not only the nominal lambda value.

At the inspected step, the last train weighted fractions were:

```text
pair_frac ~= 0.983
img_frac  ~= 0.017
roll_frac = 0.000 because lambda_roll = 0
```

If a floor is introduced in V7, the necessary check is:

```text
target early roll_frac should be non-zero and visible, for example 5%-20%, not hidden below logging noise.
```

Better V7 options:

- Candidate A: `lambda_roll_floor=0.05` from step 0, keep 50K warmup for alpha.
- Candidate B: shorten warmup to 20K-30K while keeping no floor.
- Candidate C: adaptive floor targeting a minimum rolling `roll_frac`, e.g. 0.05-0.10.

Do not hot-patch current V6. If V6 fails the gate, start V7 cleanly.

### 5.3 V6 gates need rolling-window caveats

The same critique applied to V5 should apply to V6. Current validation metrics are not full-val; they are rolling/sampled validation snapshots. A single point can be noisy.

Recommended gate protocol:

- Continue current V6 to at least 50K because before that rollout is intentionally disabled.
- From 50K to 60K, inspect at least 3 consecutive eval points.
- Use fixed-slice or fixed-batch diagnostic if available before declaring failure.
- Do not use one transient rolling-val point as a hard stop.

Suggested fail condition:

```text
After lambda_roll and alpha become non-trivial, if val_chain_normal_mse and val_chain_tail_mse do not trend downward across 3 consecutive evals, V6 should be considered failing its central hypothesis.
```

### 5.4 E1 still needs raw MSE-domain reporting for strict claims

The latest E1 docs correctly avoid claiming strict PSNR additivity. But if the team wants a strict MSE-domain statement, the current reporting should include raw per-case/per-slice MSE aggregation.

Potential issue in `scripts/diagnose_error_budget.py`:

- The script now includes better comments that this is a PSNR diagnostic.
- It still emits/uses terms like `transport_fraction`, which may invite overinterpretation.
- If this fraction is computed from PSNR-gap-derived quantities, it should not be treated as a rigorous MSE attribution.

Recommended fix:

- Add fields such as `decoded_transport_mse_mean`, `decoder_ceiling_mse_mean`, and per-step MSE ratios computed directly from per-sample MSE.
- Rename or qualify `transport_fraction` to `psnr_gap_transport_fraction` if it is based on PSNR gap logic.
- Keep PSNR tables for interpretability, but use MSE tables for attribution language.

### 5.5 V5 null-control interpretation should remain conservative

The current V5 null-control snapshot was:

```text
last_train_step = 129250
pair_frac       ~= 0.0168
roll_frac       ~= 0.0502
img_frac        ~= 0.9330
lambda_roll     = 0.25
lambda_img      = 0.12
alpha           = 1.0
```

Recent validation:

```text
val_chain_normal_mse ~= 2.277e-04
val_chain_tail_mse   ~= 2.542e-04
```

This supports the conclusion that V5 remains heavily image-dominated. But it does not by itself prove the method is useless, because rolling validation values are not catastrophic and need full-val comparison against V3/V6/Path-A controls.

The correct conclusion is:

> V5 is not a clean transport mechanism test because image auxiliary still dominates. It can be used as an attribution/control run, but final claims require full-val best/last/latest and matched baseline comparisons.

## 6. Refined Research Hypothesis

The strongest hypothesis supported by code and results is:

> The main limitation is not encoder-decoder capacity alone. The limitation is that local transport supervision, image reconstruction auxiliary, and open-loop rollout supervision are not aligned. A model can reduce pair transition losses while still failing to compose transitions into a high-quality NORMAL prediction.

This suggests the next method should optimize for composable trajectory consistency, not just local pair accuracy.

The current V6 is a useful test because it isolates pair transport first. But if V6 fails after rollout starts, the lesson is not simply that pair loss was too weak. It may mean:

- local pair losses are an insufficient proxy for open-loop trajectory correctness;
- rollout supervision starts too late;
- teacher-forced pair learning learns a transition field that is brittle under self-generated inputs;
- image auxiliary either dominates too much in V5 or becomes too weak to regularize decoded anatomy in V6;
- hop weighting may still not focus enough on the D50-to-D20 bottleneck after composition is considered.

## 7. Recommended Experiment Protocol

### 7.1 Current V6

Do not modify current V6 before 50K. The design intentionally has no rollout gradient before 50K.

At 50K-60K:

- Monitor `lambda_roll`, `alpha`, `roll_frac`, `pair_frac`, `img_frac`.
- Monitor `val_chain_d20_mse`, `val_chain_d10_mse`, `val_chain_d4_mse`, `val_chain_normal_mse`, `val_chain_tail_mse`.
- Look for monotonic or at least consistent downward trend in chain metrics.
- Use at least 3 consecutive validation snapshots.

If chain metrics do not recover after rollout becomes visible in weighted loss fractions, stop V6 and launch V7 cleanly.

### 7.2 V5 null-control

Let V5 null-control reach its planned endpoint if it is already close, then run:

- full-val latest;
- full-val best;
- full-val last;
- Path-A or matching control if available.

Use V5 mainly to answer attribution questions, not as a candidate final method.

### 7.3 V7 fallback

If V6 fails after rollout ramp begins, design V7 as one clean change, not a pile of patches.

Recommended V7 candidates:

- V7-A: keep V6 pair-heavy setup, add `lambda_roll_floor=0.05` from step 0.
- V7-B: keep no rollout floor, shorten warmup from 50K to 20K-30K.
- V7-C: adaptive rollout floor targeting a minimum logged `roll_frac` of 0.05-0.10.

Run V7 as short pilots first. The success criterion should be early chain recovery without returning to image-dominated training.

## 8. What To Ask Claude/Remote Reviewer To Revise

Recommended requests to remote reviewer:

1. Do not state that V6 Phase I degradation is harmless. State that it is expected but must recover after rollout ramp.
2. Add rolling-validation caveats to all V6 gate decisions.
3. Treat `lambda_roll=0.05` as a fallback candidate, not a validated value.
4. Add raw MSE-domain E1 reporting if making MSE attribution claims.
5. Separate V5 attribution conclusion from V6 method conclusion.
6. Define explicit V6 gate thresholds after rollout begins, preferably with consecutive eval requirements.

## 9. Bottom Line

I agree with the direction of Claude's latest analysis, but I would make the conclusion more conservative:

- V6 is worth continuing through the beginning of rollout ramp.
- V6 is not yet evidence that transport-first solves the problem.
- The current evidence actually sharpens the core risk: pair transport can look excellent while open-loop chain quality remains poor.
- The next decisive observation is whether chain MSE drops after `lambda_roll` and `alpha` become non-zero.
- If not, V7 should introduce early non-zero rollout pressure or shorten the no-rollout phase, while keeping image auxiliary low enough to avoid returning to V5-style image domination.
