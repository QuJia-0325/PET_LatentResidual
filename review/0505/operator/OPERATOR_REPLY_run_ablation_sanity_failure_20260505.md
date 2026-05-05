# Operator Reply — Official `run_ablation.sh sanity` Failure Analysis

**Date**: 2026-05-05 CST
**Repo**: `/home/qujiaxiang/project/PET_LatentResidual`
**Branch**: `foc_lite_hop0`
**Remote**: `origin = git@gitee.com:jqu9/PET_LatentResidual.git`
**HEAD at analysis**: `9328222 review(0505): add official run_ablation sanity output`
**Question addressed**: official `run_ablation.sh sanity` output, whether A/B used the same RNG seed, and what the failure means for the sigma-normalize ablation protocol.

---

## 0. Pull / remote state

I fetched `origin/foc_lite_hop0` before this analysis. Remote and local HEAD were identical:

```text
HEAD = origin/foc_lite_hop0 = 9328222a20ad3c2f82b5522706383fb3a43cdac8
```

So there was no newer Gitee commit to merge. The analysis below is based on the current pushed branch plus the official sanity artifacts now copied under `review/0505/logs_sanity/`.

---

## 1. Official answer: formal sanity gate failed

The authoritative output is:

```text
review/0505/logs_sanity/run_ablation_sanity_gpu1_20260503.launch.log
```

The formal `run_ablation.sh sanity` comparator reports:

| Metric | A | B | Relative error | Gate | Result |
|---|---:|---:|---:|---|---|
| `val_pair_total` | `1.537353e-07` | `1.693620e-07` | `10.16%` | Tier 1 `<1e-4` | FAIL |
| `val_rollout_total` | `2.051546e-03` | `2.089454e-03` | `1.85%` | Tier 2 `<1%` | FAIL |
| `val_rollout_step_0_raw` | `1.763927e-03` | `1.768560e-03` | `0.26%` | Tier 3 `<1%` | PASS |
| `val_rollout_step_1_raw` | `1.898111e-03` | `1.902824e-03` | `0.25%` | Tier 3 `<1%` | PASS |
| `val_rollout_step_2_raw` | `2.122597e-03` | `2.162539e-03` | `1.88%` | Tier 3 `<1%` | FAIL |
| `val_rollout_step_3_raw` | `2.395652e-03` | `2.513605e-03` | `4.92%` | Tier 3 `<1%` | FAIL |
| `val_chain_d20_mse` | `7.356653e-04` | `7.351346e-04` | `0.07%` | Tier 4 `<5%` | PASS |
| `val_chain_d10_mse` | `6.598809e-04` | `6.551905e-04` | `0.71%` | Tier 4 `<5%` | PASS |
| `val_chain_d4_mse` | `5.903761e-04` | `5.791343e-04` | `1.90%` | Tier 4 `<5%` | PASS |
| `val_chain_normal_mse` | `5.872322e-04` | `6.619837e-04` | `12.73%` | Tier 4 `<5%` | FAIL |

Final line from the official launcher:

```text
Result: FAIL (one or more tiers exceed threshold or missing)
[run_ablation.sh] Sanity FAILED — sentinel cleared. main/C/D will refuse to launch.
```

Therefore, the official conclusion is **FAIL**, not PASS. Any previous memory-light or manually summarized A/B report must not be used as a formal sanity-pass artifact.

---

## 2. RNG seed answer

A/B used the same configured seed.

Evidence from resolved configs:

| Field | A_sanity | B |
|---|---:|---:|
| `seed` | `42` | `42` |
| `training.deterministic` | `true` | `true` |
| `data.num_workers` | `0` | `0` |
| `training.max_steps` | `20000` | `20000` |
| `training.eval_interval` | `400` | `400` |
| `training.max_val_batches` | `64` | `64` |
| `training.val_window_mode` | `rolling` | `rolling` |

Code evidence:

- `train_first_hop.py:40-49` sets Python, NumPy, Torch CPU and CUDA seeds, and enables deterministic algorithms in warn-only mode when `training.deterministic=true`.
- `train_first_hop.py:1276-1280` reads `seed` from YAML and calls `set_seed()` before model/device construction.
- `review/0502/scripts/run_ablation.sh:50-54` exports `CUBLAS_WORKSPACE_CONFIG=:4096:8` before launching training.

Boundary condition:

The official A/B logs still contain PyTorch nondeterminism warnings:

```text
Memory Efficient attention defaults to a non-deterministic algorithm
adaptive_avg_pool2d_backward_cuda does not have a deterministic implementation, but torch.use_deterministic_algorithms(True, warn_only=True)
```

So the accurate statement is:

> A/B used the same configured RNG seed and deterministic-intended settings, but the training run was not guaranteed bitwise deterministic because deterministic algorithms were enabled with `warn_only=True` and at least two CUDA ops reported nondeterministic behavior.

---

## 3. Code-level interpretation of the failure

### 3.1 The sigma-normalize algebra itself is implemented in the intended location

The intended A/B equivalence is:

```text
A: raw rollout step losses with weights w_v6
B: normalized losses loss_j / n_j with weights w_v6 * n_j
```

The implementation path is:

- `train_first_hop.py:395-464`: computes `step_normalizers`, including `relative_to=preserve_v6_sum`.
- `train_first_hop.py:517-528`: injects `step_normalizers` when `rollout.sigma_normalize.enabled=true`.
- `pet_lr/rollout_first_hop.py:99-104`: computes raw step loss, stores `step_losses_raw`, and divides by `step_normalizers` when present.
- `pet_lr/rollout_first_hop.py:117-119`: computes weighted mean `(stacked * w).sum() / w.sum()`.

This means the code path for B is present and is applied at the rollout loss entry point.

### 3.2 The formal comparator tests end-of-run trained model equality, not only instantaneous algebra

`run_ablation.sh` compares the **last eval row** of the two separately trained runs:

- `review/0502/scripts/run_ablation.sh:149-155`: `last_eval()` selects the last row with validation metrics.
- `review/0502/scripts/run_ablation.sh:163-221`: applies Tier 1-4 thresholds and exits nonzero on failure.
- `review/0502/scripts/run_ablation.sh:392-396`: clears the sentinel and exits `1` when comparator fails.

So the failure means: after two independent 20K training trajectories, the final validation metrics were not close enough under the preregistered thresholds. It does **not** by itself prove that the per-step formula is algebraically wrong.

### 3.3 Why same seed did not guarantee pass

There are two plausible mechanisms visible from code/logs:

1. **Non-bitwise deterministic CUDA kernels**: the logs explicitly report nondeterministic attention and adaptive-pool backward kernels. Because `warn_only=True`, the run continues rather than aborting.
2. **Tiny early numerical differences can be amplified through open-loop rollout**: `pet_lr/rollout_first_hop.py:107-113` feeds mixed/predicted latent state into later hops depending on `alpha`; by step 20K `alpha=1.0` in the 20K sanity schedule, so later-hop metrics are sensitive to earlier trajectory drift.

This is consistent with the metric pattern:

- early raw hops pass or are close (`step_0_raw`, `step_1_raw` pass);
- later raw hops fail (`step_2_raw`, `step_3_raw` fail);
- `val_chain_normal_mse` fails strongly.

That pattern looks more like trajectory divergence / rollout amplification than a simple missing-normalizer bug at hop0.

---

## 4. Important protocol consequence

Because the formal sentinel was cleared, **main/C/D should remain blocked** under the intended protocol.

Relevant launcher behavior:

- `review/0502/scripts/run_ablation.sh:56-58`: defines `SANITY_PASS_SENTINEL`.
- `review/0502/scripts/run_ablation.sh:372-391`: writes the sentinel only on sanity PASS and sufficient steps.
- `review/0502/scripts/run_ablation.sh:392-396`: removes the sentinel on FAIL.
- `review/0502/scripts/run_ablation.sh:399-405`: `main` requires sanity pass before launching A_main/C.
- `review/0502/scripts/run_ablation.sh:407-409`: D also requires sanity pass.

Therefore the correct operational state is:

```text
A_sanity/B official sanity: completed but FAIL
.sanity_pass: absent / cleared
A_main/C/D: should not be launched through gated paths
```

If later runs were launched with `ALLOW_UNGATED=1` or through a manual direct command, they should be labeled as ungated/debug runs, not formal protocol runs.

---

## 5. What should be fixed before rerunning sanity

### Fix / decision A — Do not use `run_sanity_light.sh` as the formal gate

The memory-light sanity is useful for quick plumbing, but it disables full chain metrics and does not exercise Tier 4. The official gate must remain `run_ablation.sh sanity` or a successor that explicitly includes full chain metrics.

### Fix / decision B — Separate algebraic unit test from training-trajectory gate

The current gate conflates two questions:

1. Is the sigma-normalize algebra equivalent on the same model/batch?
2. Do two separately trained 20K trajectories remain close enough under CUDA/kernel noise?

I recommend adding a deterministic unit/golden test that loads a model and one fixed batch, computes A and B losses in the same process, and verifies:

```text
A weighted rollout loss == B weighted rollout loss
A raw per-hop losses == B raw per-hop losses
```

This would directly test the implementation invariant without being polluted by optimizer trajectory drift.

### Fix / decision C — If the formal gate remains trajectory-based, loosen or redefine thresholds

The existing `run_ablation.sh` text says expected Tier 2 error is `1e-6..1e-5`, but actual official Tier 2 error is `1.85%`. That mismatch means one of the following must be true:

- the implementation still has a real equivalence bug;
- the threshold expectation was too optimistic for this model and GPU kernels;
- the gate is testing a stronger condition than needed for the downstream scientific claim.

Before launching A/C/D, we need an explicit decision. My recommendation is not to silently loosen thresholds. First add the same-process algebraic test. If that passes, then redefine the training-trajectory sanity as a stability check with empirically justified thresholds, not as an algebraic exactness proof.

### Fix / decision D — Document same-seed but non-bitwise reproducibility

Future reports should say:

```text
same configured seed = yes
bitwise reproducibility = no, because nondeterministic CUDA ops are warn-only
formal sanity gate = failed on 2026-05-03 run
```

This avoids overclaiming.

---

## 6. Recommended immediate next step

Do **not** treat the current sigma-normalize formal sanity as passed.

Recommended order:

1. Add same-process algebraic A/B equivalence test for `compute_rollout_losses()` / `rollout_multistep_losses_first_hop()`.
2. Re-run `run_ablation.sh sanity` only after the algebraic test passes.
3. Keep the sentinel absent until the official full sanity gate passes.
4. Only then launch formal A_main/C/D under the gated protocol.

---

## 7. Files used for this reply

- `review/0505/logs_sanity/run_ablation_sanity_gpu1_20260503.launch.log`
- `review/0505/logs_sanity/run_ablation_A_sanity_train.log`
- `review/0505/logs_sanity/run_ablation_B_sanity_train.log`
- `review/0505/logs_sanity/run_ablation_A_sanity_config.resolved.yaml`
- `review/0505/logs_sanity/run_ablation_B_sanity_config.resolved.yaml`
- `review/0502/scripts/run_ablation.sh`
- `train_first_hop.py`
- `pet_lr/rollout_first_hop.py`
