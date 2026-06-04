# Nightmare Multi-Agent Review: V5/V6/E1/Transport Pipeline

Date: 2026-04-29
Branch: `foc_lite_hop0`
Remote baseline reviewed: `d393229 analysis: V5 attribution + V6 Phase I + E1 error budget + figure prompts`
Reviewer: Codex local review + 4 parallel subagents

## 0. Review Scope

This is a deliberately adversarial review. I treated every current conclusion as suspicious until it was checked against code, configs, logs, and available metrics.

Subagent split:

- Agent A: training implementation, loss composition, rollout schedule, resume/checkpoint behavior.
- Agent B: evaluation, diagnostics, PSNR/MSE semantics, full-val protocol.
- Agent C: research design, claim support, Claude narrative validity.
- Agent D: operations, reproducibility, launcher/API hygiene, Gitee state.

Local checks performed:

- Pulled Gitee latest before review: `d393229`.
- Inspected `train_first_hop.py`, `pet_lr/rollout_first_hop.py`, `pet_lr/model_first_hop.py`, `pet_lr/data_first_hop.py`, `pet_lr/ema.py`.
- Inspected V5/V6 configs and 0427/0428/0429 review docs.
- Parsed live metrics from:
  - `/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v6_transport_first/metrics.jsonl`
  - `/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v5_null_control/metrics.jsonl`
- Loaded checkpoint metadata for V3/V5/V6 using `/home/qujiaxiang/.conda/envs/rae/bin/python`.

## 1. Executive Verdict

Claude's latest direction is useful but still not safe as a claim-level conclusion.

The safe statement is:

> Current evidence supports a transport-side or trajectory-consistency bottleneck hypothesis. It does not yet prove the mechanism, does not prove V6 solves it, and does not close V5/null-control attribution.

The unsafe statements are:

- "V3 plateau is proven to be recipe disease."
- "V6 Phase I degradation is harmless/causally closed."
- "V6 validates the transport bottleneck."
- "transport gradient budget is now dominated by pair/rollout."
- "val_chain_normal_mse is a latent-space chain diagnostic."
- "best.pt is the best full-val model."

The largest risks are:

1. `val_chain_*_mse` is decoded image-space MSE, but some 0429 docs call it latent-space MSE.
2. `best.pt` is selected on rolling 512-slice validation, not full-val.
3. EMA checkpoint saving makes resume from `best.pt` an EMA-weight warm-start with raw optimizer state, not exact continuation.
4. V6 Phase II train/eval mismatch remains: training rollout is alpha-mixed, eval is full open-loop.
5. V6 full-val helper can evaluate stale early `best.pt` and ignore latest step checkpoints.
6. E1 still emits `transport_fraction` from PSNR dB gaps and gates on it, inviting overinterpretation.
7. Path A diagnostics are not currently reproduced from shipped scripts; the recorded run failed with `ModuleNotFoundError`.
8. Current Gitee state is not sufficient for remote reproduction because runtime YAML/log evidence is untracked and some launchers mutate tracked YAML in place.

## 2. Immediate Correction To My Previous 0429 Note

My earlier file `review/0429/codex_research_refine_pipeline_review_20260429.md` contains stale V6 schedule values.

Wrong in that file:

- `rollout_ramp_steps: 25000`
- `rollout_lambda: 0.35`

Actual V6 config:

- `configs/pet_flow/pet_flow_first_hop_224_v6_transport_first.yaml:128` sets `warmup_ratio: 0.25`, so warmup is 50K under 200K max steps.
- `configs/pet_flow/pet_flow_first_hop_224_v6_transport_first.yaml:130` sets `ramp_ratio: 0.50`, so ramp is 100K under 200K max steps.
- `configs/pet_flow/pet_flow_first_hop_224_v6_transport_first.yaml:139` sets `lambda_end: 4.0`.

Impact:

- Any analysis that says V6 ramps only 25K or uses final rollout lambda 0.35 is invalid.
- The real V6 test window is 50K-150K, not 50K-75K only.
- A 60K/75K gate can be used as early warning, but full Phase II behavior is not complete until 150K.

## 3. Live State Snapshot

At local inspection time:

V6 live metrics:

```text
train rows = 886
val rows   = 110
last train step = 44300
pair_frac = 0.6701
roll_frac = 0.0000
img_frac  = 0.3299
lambda_roll = 0.0000
lambda_img  = 0.0400
alpha       = 0.0000
pair_loss_weight = 15.0
last val step = 44000
val_pair_total = 1.9493e-07
val_rollout_total = 8.9078e-04
val_chain_d20_mse = 3.1459e-04
val_chain_d10_mse = 3.2355e-04
val_chain_d4_mse = 3.4719e-04
val_chain_normal_mse = 5.8003e-04
val_chain_tail_mse = 4.1692e-04
```

Interpretation:

- V6 is still pre-rollout: `lambda_roll=0`, `alpha=0`.
- It has not yet tested the actual open-loop rollout correction hypothesis.
- The latest train batch is not pair-dominated at every step; `img_frac` can still be about 33% even with `pair_weight=15`. The mean over 0-44K is pair-heavy, but per-batch values vary.

V5 null-control live metrics:

```text
last train step = 129450
pair_frac = 0.0536
roll_frac = 0.0713
img_frac  = 0.8751
lambda_roll = 0.25
lambda_img = 0.12
alpha = 1.0
last val step = 129200
val_chain_normal_mse = 2.277e-04
val_chain_tail_mse = 2.542e-04
```

Interpretation:

- Null-control remains mostly image-loss dominated in weighted scalar loss fraction.
- This supports the hypothesis that the V3 recipe stays image-heavy after resume.
- It does not prove that training cannot help or that V5 changes caused degradation. Full-val at matched endpoint is still needed.

## 4. P0 Findings: Must Fix Or Explicitly Qualify Before Any Claim

### P0.1 `val_chain_*_mse` is decoded image-space MSE, not latent-space MSE

Evidence:

- `train_first_hop.py:958` calls `sample_chain_first_hop` to produce latent chain predictions.
- `train_first_hop.py:965-968` decodes those latent predictions into images.
- `train_first_hop.py:970-973` computes `F.mse_loss(x_*_pred, x_roll[:, *])`.

Therefore:

- `val_chain_d20_mse`, `val_chain_d10_mse`, `val_chain_d4_mse`, and `val_chain_normal_mse` are decoded image-space MSE in normalized image space.
- They are not latent MSE.

Problem:

- `review/0429/supervisor/v6_phase1_e1_supervisor_review_20260429.md` calls `val_chain_normal_mse` a latent-space chain MSE around line 64/65.
- That wording is wrong and can falsely imply a direct latent-space gap between pair loss and chain loss.

Impact:

- The comparison `val_pair_total ~= tiny` vs `val_chain_normal_mse ~= large` is still useful, but it compares latent pair objective against decoded image-space chain quality. It does not isolate latent trajectory error.

Required fix:

- Rename docs wording to "decoded image-space chain MSE".
- Add a true latent-chain metric in evaluation: `val_chain_normal_latent_mse`, `val_chain_d20_latent_mse`, etc.
- Do not call existing `val_chain_*_mse` latent diagnostics.

### P0.2 `best.pt` is rolling-val best, not full-val best

Evidence:

- V6/V5/V3 configs set `max_val_batches: 64` and `val_window_mode: rolling`.
- `train_first_hop.py:737-760` implements rolling validation window selection.
- `train_first_hop.py:2405-2426` saves `best.pt` if the rolling-window selection score improves.
- Each rolling validation has `val_chain_samples=512`, not all 7403 val slices.

Impact:

- `best.pt` is a rolling-window checkpoint. It can be chosen because a favorable 512-slice window was seen.
- Full-val must be run post-hoc to claim checkpoint quality.
- Any doc that treats `best.pt` as "best on full-val" is overclaiming.

Concrete current example:

- V6 `best.pt` is currently step 6000.
- V6 already has `step_040000.pt`.
- Running `review/0428/operator/03_v6_fullval.sh` now would evaluate the stale early rolling best, not current V6 behavior.

Required fix:

- Report checkpoint provenance explicitly: `rolling-best step`, `latest step`, `last step`, `full-val score`.
- For V6 pilot, evaluate latest numeric step checkpoint if doing interim full-val.
- Do not use `best.pt` alone for in-progress V6 conclusions.

### P0.3 EMA checkpoint resume is not exact continuation

Evidence:

- `pet_lr/ema.py:99-105` swaps EMA parameters into the model inside `ema.average_parameters()` and restores raw parameters after the context.
- `train_first_hop.py:2306-2324`, `2407-2425`, and `2434-2452` save checkpoints inside this EMA context.
- `train_first_hop.py:1145-1158` saves `model.state_dict()`, optimizer state, and EMA state into the same checkpoint.
- `train_first_hop.py:1582` resumes by loading `ckpt["model"]` into the model.
- `train_first_hop.py:1618-1621` restores optimizer/scaler state if architecture is unchanged.

This means:

- `best.pt`, step checkpoints, and `last.pt` store EMA model weights when EMA is enabled.
- Optimizer moments are still from the raw training trajectory.
- Resume from such a checkpoint loads EMA weights but raw optimizer moments.

Impact:

- V5 rollout-heavy and V5 null-control resume from V3 `best.pt` are not exact continuation of raw training.
- They are EMA-weight warm-starts with raw optimizer state and restored EMA state.
- This weakens claims like "null-control isolates only extra training / LR schedule".

Required fix:

- Either save two model states: `model_raw` and `model_ema`, and define resume semantics explicitly.
- Or make resume from best intentionally reset optimizer and call it "EMA warm-start fine-tuning".
- For attribution experiments, document this confound directly.

### P0.4 V6 Phase II train/eval mismatch is substantial

Evidence:

- Training rollout alpha is scheduled in `train_first_hop.py:420-426`.
- Training rollout loss uses `rollout_multistep_losses_first_hop` in `train_first_hop.py:442-451`.
- Validation forces `eval_alpha` from config in `train_first_hop.py:883-890`.
- V6 config has `eval_alpha: 1.00` at `configs/pet_flow/pet_flow_first_hop_224_v6_transport_first.yaml:136`.
- V6 config has `straight_through_mode: fixed` and `straight_through: true` at lines 133-135.
- `pet_lr/rollout_first_hop.py:28-36` implements straight-through mixing: forward can be GT/mixed while gradient flows through prediction.

Impact:

- From 50K to 150K, V6 training is alpha-mixed, while validation is full open-loop.
- "Rollout starts at 50K" is not the same as "open-loop training starts at 50K".
- V6 may still fail open-loop eval even while rollout loss is improving under teacher-forced/mixed forward states.

Required fix:

- Gate V6 using open-loop validation, but describe it as deliberately harsher than training until alpha reaches 1.
- Add a diagnostic train/eval pair: rollout loss with `alpha_train` and rollout loss with `alpha=1` on the same validation windows.
- If V6 fails after 50K, consider earlier open-loop pressure or a rollout lambda floor, but do not hot-patch the current run without marking it as a new experiment.

### P0.5 Pair velocity and endpoint losses are effectively duplicated under current settings

Evidence:

- `train_first_hop.py:302-303` computes `v_target_raw = (z_dst - z_src) / dt`.
- `train_first_hop.py:310` computes velocity error on `out["v_total_raw"] - v_target`.
- `train_first_hop.py:312` computes endpoint error as `((z_pred - z_dst) / dt)^2` when `endpoint_dt_normalize=true`.
- `pet_lr/model_first_hop.py:505-511` uses `z_pred = z_src + v_total_raw * dt` when `target_normalize=false`.
- Checkpoint metadata confirms `target_normalize: False` for V3/V5/V6 inspected checkpoints.
- V6 config sets `endpoint_dt_normalize: true` at `configs/pet_flow/pet_flow_first_hop_224_v6_transport_first.yaml:190`.

Algebra:

```text
(z_pred - z_dst) / dt
= (z_src + v_total_raw * dt - z_dst) / dt
= v_total_raw - (z_dst - z_src) / dt
= velocity residual
```

Therefore:

- Velocity loss and endpoint loss become the same per-sample error up to the same pair weighting.
- `velocity_rebalance` ratio tends to 1 and is effectively a no-op.

Impact:

- Claims that `velocity_rebalance` is an active fix are overclaimed.
- Pair loss is not two independent constraints; it is mostly duplicated supervision.

Required fix:

- Decide whether endpoint should be unnormalized latent endpoint MSE or remove it as duplicate.
- If keeping both, log correlation/ratio between velocity and endpoint loss; if ratio is exactly 1, do not claim rebalance is active.

### P0.6 E1 still invites PSNR-domain overinterpretation

Evidence:

- `scripts/diagnose_error_budget.py:175-177` computes dB-domain gaps from mean PSNR values.
- `scripts/diagnose_error_budget.py:187` emits `transport_fraction` from those dB gaps.
- `scripts/diagnose_error_budget.py:225-234` gates on that `transport_fraction`.
- `review/0429/e1_error_budget_decomposition_explained.md` correctly says PSNR gaps are diagnostic, not linear MSE attribution.

Impact:

- The code output name `transport_fraction` and gate still encourage readers to say "X% of error is transport".
- That is not mathematically valid for dB PSNR differences.

Required fix:

- Rename field to `psnr_gap_transport_fraction_diagnostic`.
- Remove PASS/FAIL gate or change it to "diagnostic flag".
- Add raw per-slice MSE aggregation if making MSE attribution claims.

### P0.7 E1 evidence may be from a different checkpoint than the current V3 200K claim

Evidence from existing docs:

- `review/0416/conclusion.md` describes E1 as using `first_hop_224_50k_formal_v3_chainstable/best.pt`.
- The 0429 E1 explanation discusses "current system (V3 best)" and quotes V3 200K-like NORMAL PSNR around 36.87 dB.
- The old E1 table reportedly has NORMAL E2E around 36.742 dB, not the V3 200K full-val 36.865 dB.

Impact:

- If E1 was not rerun on V3 200K best, then using it as direct support for V6 design is weaker.
- It remains directional evidence, but not exact current-baseline evidence.

Required fix:

- Rerun E1 on `first_hop_224_200k_transport_v3/best.pt` with `--max-slices 0` or clearly label existing E1 as 50K-chainstable evidence.

### P0.8 Path A is currently not reproduced from shipped scripts

Evidence:

- `review/0427/logs_eval/p0_02_pathA_baseline_gpu2.log` records `ModuleNotFoundError: No module named 'pet_lr'`.
- `scripts/diagnose_tf_rollout_gap.py` documents that `PYTHONPATH=/home/qujiaxiang/project/PET_LatentResidual` is required.
- `review/0427/operator/02_pathA_baseline.sh` and `review/0427/operator/04_pathA_v5_best.sh` invoke the script without exporting `PYTHONPATH`.

Impact:

- Current Path A closure is missing for V3 200K and V5 best.
- Existing Path A artifacts appear to be old Scheme-C diagnostics, not the intended V3/V5 attribution baseline.

Required fix:

- Add `export PYTHONPATH=/home/qujiaxiang/project/PET_LatentResidual:${PYTHONPATH:-}` to Path A launchers.
- Rerun V3 200K best and V5 best Path A.
- Sync JSON/log outputs into the review tree.

## 5. P1 Findings: Serious Validity Risks

### P1.1 "Gradient budget" language is technically wrong

Evidence:

- `train_first_hop.py:1936-1944` computes `pair_frac`, `roll_frac`, and `img_frac` from weighted scalar losses.
- These are not gradient-norm fractions.
- They also exclude `align` and regularizer contributions from the fraction denominator.

Impact:

- Phrases like "transport gradient share" or "image_aux gradient budget" are unsupported unless actual per-loss gradient norms are measured.

Required fix:

- Use "weighted scalar loss fraction" in reports.
- If gradient budget is important, implement explicit per-loss gradient norm measurement on sampled batches.

### P1.2 Rollout `step_weights` are normalized, not absolute multipliers

Evidence:

- `pet_lr/rollout_first_hop.py:106-108` computes `(stacked * w).sum() / w.sum()`.

Impact:

- V6 `[0.5, 2.0, 1.5, 1.0]` changes relative weights to `[0.1, 0.4, 0.3, 0.2]`; it does not increase total rollout loss scale.
- V5 tail-heavy weights similarly change relative emphasis, not absolute total rollout strength.

Required fix:

- State effective normalized weights in configs/docs.
- Use `lambda_roll` for absolute rollout strength.

### P1.3 Pair loss weights are not normalized

Evidence:

- `train_first_hop.py:156` builds manual pair weights.
- `train_first_hop.py:322` multiplies per-sample velocity error by those weights before mean.

Impact:

- V6 `pair_loss_weights=[2.5,1,1,1]` changes both hop emphasis and average pair loss scale.
- Combined with `pair_weight=15`, pair scaling is not simply 15x vs V3.

Required fix:

- Report both relative and expected average scaling.
- Consider normalizing pair weights to mean 1 if the intent is pure reallocation rather than scale increase.

### P1.4 V6 full-val helper ignores latest numeric step checkpoint

Evidence:

- `review/0428/operator/03_v6_fullval.sh:38-45` evaluates `best.pt`.
- `review/0428/operator/03_v6_fullval.sh:50-57` evaluates `last.pt` only if it exists.
- It has no fallback to latest `step_*.pt`.

Current metadata:

- V6 `best.pt` is step 6000.
- V6 `step_040000.pt` exists.

Impact:

- Running the script during pilot can report an early rolling best and completely miss the current model.

Required fix:

- Add `latest_step_ckpt_numeric()` based on parsed step number, not modification time.
- Evaluate `v6_best`, `v6_latest_step`, and later `v6_last`.

### P1.5 Latest checkpoint selection by modification time is fragile

Evidence:

- `review/0427/run_me/01_fullval_eval.sh:55-58` sorts `step_*.pt` by `%T@`.
- `review/0428/operator/04_v5_attribution.sh:24-28` repeats this pattern.

Impact:

- Copying/touching/restoring a checkpoint can make the wrong step appear latest.

Required fix:

- Parse step from filename and sort numerically.

### P1.6 Full-val and training gates mix different metrics

Evidence:

- Training `val_chain_*_mse` is un-clipped decoded image MSE in normalized `[-1,1]` style image space.
- Full-val uses `calc_psnr_clip3` at `eval_first_hop_224_clip3.py:209`, which clips to SUV `[0,3]` before PSNR.

Impact:

- These are not equivalent objective metrics.
- A run may improve clip3 PSNR while worsening un-clipped decoded MSE or vice versa.

Required fix:

- Gate with one primary metric family per decision.
- For final decisions, use full-val clip3 PSNR plus full-val per-timepoint summaries.
- Use rolling decoded MSE only as online health signal.

### P1.7 NORMAL-only summaries hide hop tradeoffs

Evidence:

Accessible full-val artifacts show:

- V3 best NORMAL PSNR: 36.8650 dB.
- V5 best NORMAL PSNR: 36.9826 dB.
- V5 step_100000 NORMAL PSNR: 37.0396 dB.
- But V5 best worsens D20 by about -0.046 dB and D10 by about -0.026 dB versus V3.

Impact:

- A headline "NORMAL improved" can hide first-hop/mid-hop regression.
- Since the project narrative says D50 to D20 is the main bottleneck, D20 regression matters.

Required fix:

- Always report D20, D10, D4, NORMAL together.
- Define claim-specific success: first-hop improvement, tail improvement, or final NORMAL improvement.

### P1.8 Diagnostic scripts lack checkpoint/config semantic guards

Evidence:

- `eval_first_hop_224_clip3.py:108-117` checks checkpoint pixel-forcing metadata against config.
- `scripts/diagnose_error_budget.py:105-109` loads checkpoint without equivalent semantic metadata checks.
- `scripts/diagnose_foc_gap.py:104-108` also loads without equivalent checks.
- Path A script follows similar direct loading.

Impact:

- Wrong config/checkpoint pair can silently produce plausible but invalid diagnostics.

Required fix:

- Share a common `load_model_with_semantic_checks()` helper across eval and diagnostics.
- Check `first_hop_pixel_enabled`, `target_normalize`, `rollout_path`, and possibly best metric signature.

### P1.9 FOC/E2 half-step diagnostic re-applies hop0 pixel forcing

Evidence:

- `scripts/diagnose_foc_gap.py:149-152` applies `x_src_img=x_src` on full D50 to D20 step.
- `scripts/diagnose_foc_gap.py:156-159` applies it on half step 1.
- `scripts/diagnose_foc_gap.py:163-166` applies it again on half step 2 from midpoint to D20.
- Training `compute_foc_losses` has the same pattern around `train_first_hop.py:650-669`.

Impact:

- FOC gap measures both time subdivision and repeated pixel forcing, not pure ODE/integration consistency.
- If FOC looks bad, it may be because pixel forcing is injected twice.

Required fix:

- Add variant where half-step 2 uses `x_src_img=None`.
- Report both variants if keeping current behavior.

### P1.10 V5/null-control attribution is not closed

Evidence:

- Null-control is still running and has not reached planned endpoint in local live metrics.
- Null-control full-val outputs were not found under `/data_2/...` for null-control.
- V5/NC attribution docs themselves say full-val and Path A are pending.

Impact:

- Current V5 evidence is suggestive only.
- It cannot prove recipe disease, LR exclusion, or V5 loss-change causality.

Required fix:

- Finish NC to matched budget or explicitly mark partial.
- Run full-val on NC latest/best/last.
- Rerun Path A for V3 200K best and V5 best.

## 6. P2 Findings: Reproducibility And Operations Problems

### P2.1 Launchers mutate tracked YAML in place

Evidence:

- `scripts/launch_v4_sf_pilot.sh:43` uses `sed -i` on tracked config.
- `scripts/launch_v5_rollout_heavy.sh:43` uses `sed -i` on tracked config.
- Current git diff shows tracked config changes from `max_steps: 999999` to `max_steps: 136800` in V4/V5 configs.

Impact:

- Reproduction depends on local side effects.
- Git status remains dirty after launch.

Required fix:

- Never mutate tracked configs. Copy to runtime YAML under review/logs or `/data_2` and run that.

### P2.2 Active null-control depends on untracked runtime YAML

Evidence:

- Live command uses `review/0427/logs_train/v5_null_control_gpu3_runtime.yaml`.
- That runtime YAML is untracked.
- Canonical tracked `configs/pet_flow/pet_flow_first_hop_224_v5_null_control.yaml` has `max_steps: 200000`; runtime YAML uses `max_steps: 136800`.

Impact:

- A Gitee checkout cannot reconstruct the exact current NC run.

Required fix:

- Track the runtime YAML for any reported run.
- Include `config.yaml` copied from output directory in the review artifact bundle.

### P2.3 Repo state is dirty and blocks clean remote reproduction

Current dirty/untracked items include:

- Modified tracked configs: V4/V5 rollout configs.
- Modified tracked logs under `review/0426/logs_train`.
- Untracked PID files.
- Untracked P0 eval logs.
- Untracked V5 null-control log/runtime YAML.
- Untracked `review/0429/codex_research_refine_pipeline_review_20260429.md`.

Impact:

- Remote colleagues reviewing Gitee do not see the exact local operational state.

Required fix:

- Decide what is evidence and track it.
- Move stale PID files to `deprecated/` or delete if approved.
- Avoid committing live-growing logs unless snapshots are explicitly copied.

### P2.4 Environment portability is weak

Evidence:

- No tracked `requirements.txt`, `environment.yml`, `pyproject.toml`, `setup.py`, or `Dockerfile` was found.
- Configs hard-code `/home/qujiaxiang/project/RAE` and `/data_2/qujiaxiang`.
- `pet_lr/path_guard.py:5-19` rejects outputs outside `/data_2`.

Impact:

- Remote machines without identical path layout cannot run the pipeline directly.

Required fix:

- Add `environment.yml` or at least an environment note.
- Add a path configuration section documenting required symlinks/data layout.
- Keep `/data_2` guard for local production if desired, but document it.

### P2.5 Determinism is advertised but not fully enforced

Evidence:

- `train_first_hop.py:40-49` enables deterministic algorithms with warn-only.
- Live logs warn that CUDA/cuBLAS and memory-efficient attention are nondeterministic unless additional environment settings are used.
- Launchers do not export `CUBLAS_WORKSPACE_CONFIG`.

Impact:

- Runs are not bitwise reproducible despite `deterministic: true`.

Required fix:

- Export `CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python if strict determinism matters.
- Record CUDA, torch, driver, git SHA, and env name into checkpoint metadata.

### P2.6 `.gitignore` is insufficient for experiment operations

Evidence:

- `.gitignore` only ignores Python bytecode.
- Logs, PID files, and runtime YAMLs drift into dirty/untracked state.

Impact:

- Repository hygiene is poor; important evidence and junk are mixed.

Required fix:

- Define artifact policy:
  - tracked: frozen review logs, final configs, summaries, reports;
  - ignored: live logs, PID files, temp runtime files unless copied to review snapshot.

### P2.7 Default eval/API paths are stale

Evidence:

- `eval_first_hop_224_clip3.py` default config points to `pet_flow_first_hop_224_alignchecked_detfix.yaml`, which is not present locally.
- README still references old example configs.

Impact:

- Running scripts without explicit arguments can fail or evaluate the wrong historical path.

Required fix:

- Make defaults either valid canonical configs or remove defaults and require explicit paths.

## 7. Research Claim Matrix

Allowed now:

- "V3 full-val best reaches NORMAL PSNR 36.865 dB on val using clip_max=3 over 7403 slices."
- "V5 best/step_100000 show NORMAL PSNR gains on full-val, but D20/D10 tradeoffs exist."
- "V5 null-control rolling metrics remain image-loss dominated in weighted scalar loss fractions."
- "V6 Phase I shifts training pressure toward pair loss on average but has not yet tested rollout recovery."
- "E1 supports a decoded-space transport-related discrepancy hypothesis, not a strict MSE attribution."

Not allowed yet:

- "V3 plateau is proven to be caused by image_aux domination."
- "V6 fixes the transport bottleneck."
- "V6 Phase I degradation is harmless."
- "Pair loss is solved in latent space and only rollout is left," unless true latent-chain metrics are added.
- "Transport gradient share is X%," unless actual gradient norms are measured.
- "E1 proves X% of error is transport," unless raw MSE decomposition is implemented.

## 8. Recommended Fix Order

### Fix immediately before next remote review

1. Correct docs that call `val_chain_*_mse` latent-space MSE.
2. Correct or supersede `review/0429/codex_research_refine_pipeline_review_20260429.md` because it contains stale V6 schedule values.
3. Add an explicit note that `best.pt` is rolling-val best, not full-val best.
4. Add a warning that V5/NC resume is EMA warm-start plus raw optimizer state, not exact continuation.
5. Fix Path A launchers by exporting `PYTHONPATH` and rerun Path A.
6. Track the runtime YAML used for V5 null-control or copy the output `config.yaml` into review artifacts.

### Fix before final V5/V6 attribution claims

1. Run NC full-val at matched endpoint: latest numeric step, best, last if available.
2. Run V3/V5/NC Path A using corrected launcher.
3. Add true latent-chain MSE metrics in validation or a diagnostic script.
4. Change E1 output field from `transport_fraction` to diagnostic PSNR-gap wording.
5. Add per-timepoint full-val table to every report, not just NORMAL.

### Fix before running a new V7/V8 method

1. Decide whether pair endpoint loss should remain dt-normalized; if yes, acknowledge duplicate velocity objective.
2. Decide whether pair weights should be normalized to mean 1.
3. Add an open-loop validation-at-train-alpha comparison during ramp.
4. Save raw and EMA model states separately for clean resume/eval semantics.
5. Stop mutating tracked configs in launchers.

## 9. V6-Specific Decision Guidance

Current V6 should not be judged before 50K because `lambda_roll=0` and `alpha=0` by design.

At 50K-60K:

- This is only an early warning phase.
- Check whether `lambda_roll_base` becomes nonzero and `roll_frac` becomes visible.
- Check open-loop validation trend over consecutive rolling windows.
- Also compute a fixed-slice diagnostic if possible.

At 75K:

- If chain metrics are still not trending down, V6 is in trouble.
- But because the official ramp lasts to 150K, failure wording should be "early failure signal," not complete Phase II failure.

At 150K:

- Alpha and lambda should be at end values.
- This is the first point where training rollout distribution should match open-loop eval in the intended V6 schedule.

Do not run `03_v6_fullval.sh` as-is during pilot if the goal is current model assessment. It will evaluate `best.pt` step 6000 and skip latest step unless `last.pt` exists.

## 10. Final Bottom Line

The current project is scientifically salvageable, but several documents are stronger than the code and evidence allow.

The central method question is still open:

> Can a transport-first schedule produce composable open-loop latent trajectories that improve full-val clip3 PSNR across D20/D10/D4/NORMAL without hiding regression in early hops?

Current evidence does not answer this yet. V6 Phase I only shows that pair-heavy local supervision can be optimized while open-loop decoded chain quality remains unstable. The decisive evidence starts after rollout/alpha ramp begins, and final claims require full-val plus corrected diagnostics.
