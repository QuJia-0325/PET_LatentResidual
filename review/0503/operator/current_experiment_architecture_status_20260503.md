# Current Experiment and Architecture Status - 2026-05-03

**Repo**: `/home/qujiaxiang/project/PET_LatentResidual`  
**Branch**: `foc_lite_hop0`  
**Pulled HEAD**: `23a11d4`  
**Sample time**: `2026-05-03 22:44 CST`  
**Scope**: latest Gitee state, current GPU runs, sigma-normalize ablation architecture, and local pre-C1 questions.

---

## 1. Latest Remote State

The Gitee branch is up to date on the operator host:

```text
23a11d4 review(0503): draft REV1 §10 (round-2 multi-agent absorption) + pre-C1 operator questions
```

The latest commit is documentation/planning only. It changed:

```text
review/0503/local/OPERATOR_QUESTIONS_pre_C1_20260503.md
review/0503/local/REV1_PLAN.md
review/0503/local/REV1_TOOLING_PLAN.md
```

It did **not** patch the actual scripts yet. Therefore every code-level fix in C1-C5d should still be treated as pending.

---

## 2. Current Running Experiments

| Run | Status | GPU | Step state | Metrics path | Interpretation |
|---|---|---:|---|---|---|
| `A_sanity` | finished | was GPU1 | 20000/20000, 50 val rows | `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0502_runs/A_sanity/run/first_hop_224_sigma_norm_A_sanity/metrics.jsonl` | first half of full sanity gate |
| `B_sanity` | running | GPU1 | last train step 18800, last val step 18400, 46 val rows | `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0502_runs/B/run/first_hop_224_sigma_norm_B/metrics.jsonl` | second half of full sanity gate, not complete |
| `V6.1 rollout floor` | running | GPU3 | last train step 86400, last val step 86000 | `/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v6_1_rollout_floor/metrics.jsonl` | legacy/current V6.1 line, not part of sigma-normalize A/C/D lock matrix |
| `A_sanity_light` | running | GPU2 | last train step 28700, last val step 28400 | `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0502_runs_light/20260502_051245/A_sanity_light/run/first_hop_224_sigma_norm_A_sanity_light_20260502_051245/metrics.jsonl` | memory-light A-only sanity, not a full gate |

Current full-sanity sentinel:

```text
review/0502/runs/.sanity_pass: absent
```

So `C`, `D`, and `pair_uniform` remain gated by the current launcher unless `ALLOW_UNGATED=1` is used.

---

## 3. Current Metrics Snapshot

### A_sanity final row

```text
step=20000
val_select_score=0.0020769658
val_chain_normal_mse=0.0005872322
val_rollout_total=0.0020515465
val_pair_total=1.5373535e-07
```

Best single-window `val_select_score` row:

```text
step=400
val_select_score=0.0007849058
val_chain_normal_mse=0.0002290948
val_rollout_total=0.0013998433
val_pair_total=0.0006261284
```

### B_sanity latest row

```text
last_val_step=18400
val_select_score=0.0018183143
val_chain_normal_mse=0.0005981120
val_rollout_total=0.0015367585
val_pair_total=1.5688423e-07
```

Best single-window `val_select_score` row:

```text
step=400
val_select_score=0.0007849058
val_chain_normal_mse=0.0002290948
val_rollout_total=0.0013998345
val_pair_total=0.0006261284
```

### V6.1 latest row

```text
last_val_step=86000
val_select_score=0.0009049836
val_chain_normal_mse=0.0002545511
val_rollout_total=0.0006177456
val_pair_total=1.2144279e-06
```

Best single-window `val_select_score` row:

```text
step=81600
val_select_score=0.0006530711
val_chain_normal_mse=0.0001770941
val_rollout_total=0.0008558025
val_pair_total=1.4759749e-06
```

### A_sanity_light latest row

```text
last_val_step=28400
val_select_score=0.0008783599
val_chain_normal_mse=0.0
val_rollout_total=0.0008783599
val_pair_total=0.0003957904
```

`A_sanity_light` intentionally loads only D50/D20 images and disables full x-rollout image chain metrics. The zero chain metric means it must not be used as a full-val or full-sanity substitute.

---

## 4. What Has Not Started

No formal sigma-normalize main ablation run has started yet:

| Planned run | Current state |
|---|---|
| `A_main` | no metrics path found, not running |
| `A_seed43` | config not yet present, not running |
| `C_uniform` | no metrics path found, not running |
| `D_closed_form` | no metrics path found, not running |
| `A_pair_uniform_spot_aligned` | config not yet present, not running |

This is scientifically useful: the protocol and tooling can still be fixed before the critical A/C comparison starts.

---

## 5. Architecture Summary

The current codebase has two overlapping layers:

1. **Model/trainer layer**: `train_first_hop.py`, `pet_lr/model_first_hop.py`, `pet_lr/rollout_first_hop.py`, and related PET latent data utilities.
2. **Audit/ablation protocol layer**: `review/0502/configs/*.yaml` and `review/0502/scripts/*.sh|*.py`.

The current research pivot is mostly in the second layer. The method is not introducing a new model family at this point; it is trying to make a fair, auditable comparison of rollout step weighting with and without sigma/dt normalization.

Core training architecture:

- Input/output live in latent space produced by the frozen RAE encoder/decoder.
- First-hop transport is trained on the PET timepoint path `D50 -> D20 -> D10 -> D4 -> NORMAL`.
- Pair loss supervises per-hop velocity/endpoint behavior.
- Rollout loss evaluates chained prediction quality under scheduled teacher forcing/self forcing.
- Image auxiliary loss decodes hop0 output and regularizes image-space behavior.
- EMA is enabled and used for eval/checkpoint saving.
- Checkpoint selection uses the computed `val_select_score`, derived from config-side `best_metric: val_multi_objective` and `best_metric_terms`.

Sigma-normalize matrix intent:

- `A_control`: V6-equivalent baseline with original rollout `step_weights=[0.5,2.0,1.5,1.0]` and no sigma-normalize.
- `B_sanity`: sigma-normalize enabled with `preserve_v6_sum`; step weights are transformed so A and B should be mathematically equivalent. This is the implementation sanity gate.
- `C_uniform`: intended main ablation after lock. Sigma-normalize plus uniformized shape, used to test whether the V6 step-weight shape has marginal value beyond scale compensation.
- `D_closed_form`: closed-form/reference check, gated after sanity.
- `A_pair_uniform_spot_aligned`: Risk 4 robustness run to test whether hop0-heavy pair weighting is masking rollout-shape effects.

---

## 6. Main Strengths of the Current Design

1. **A/B sanity gate is conceptually strong.** `B_sanity` is designed to be equivalent to `A_control` after normalization. If it fails, the implementation is wrong before any scientific claim is made.
2. **A/C main comparison has a preregistration direction.** The plan now separates sanity, A_main, threshold locking, C launch, and final decision.
3. **The latest local review correctly caught tool/protocol mismatch.** In particular: `step` vs `global_step`, remote-name portability, checkpoint naming, and A_pair schedule confounding.
4. **Decoder ceiling already exists.** This helps bound claims about latent-decoder reconstruction capacity, as long as it is not used to tune A-vs-C thresholds post hoc.
5. **Current A/C/D main experiments have not started.** That preserves the chance to fix protocol and launch under a clean audit trail.

---

## 7. Main Risks and Blockers

### 7.1 Script schema mismatch is still live

The trainer writes val rows like:

```json
{"event":"val","step":20000,"val_select_score":...}
```

But these scripts still read `global_step`:

```text
review/0502/scripts/lock_effect_size_threshold.py
review/0502/scripts/paired_diff_judge.py
review/0502/scripts/select_best_ckpt_smoothed.py
```

Until C1 lands, lock/judge/smoothing tools can fail or return empty selections on real metrics.

### 7.2 Checkpoint naming mismatch is still live

The trainer writes:

```text
step_020000.pt
best.pt
last.pt
```

`select_best_ckpt_smoothed.py` still searches primarily for:

```text
ckpt_step_*.pt
ckpt_last.pt
```

Until C3 lands, Method-D checkpoint selection is not operational on current training outputs.

### 7.3 Remote-name hardcoding is still live

Operator host:

```text
origin = git@gitee.com:jqu9/PET_LatentResidual.git
```

Therefore `git push gitee ...` and `gitee/foc_lite_hop0` are invalid here. A resolver based on `.review_canonical_remote` must be used before C2/C5b/R1b can be considered portable.

### 7.4 B_sanity is not complete

Current B has 46 val rows and no sentinel. Do not unlock `C`, `D`, or Risk-4 runs based on the current state unless local explicitly accepts the risk.

### 7.5 `A_pair_uniform_spot` must be aligned before it is meaningful

The current spot config is 60K. With ratio-based schedules, that changes phase boundaries relative to A_main. The aligned fix should be 120K plus comment cleanup, not a same-step comparison against a 60K schedule.

### 7.6 `A_sanity_light` is not a substitute for full sanity

It is useful for memory sanity and A-only training behavior, but it loads only D50/D20 image timepoints and has no authoritative full-chain image metrics. Its `val_chain_normal_mse=0.0` is a configuration artifact, not a performance result.

---

## 8. Recommended Execution Order

Recommended order from the operator side:

1. Let B_sanity finish to 20K and write `review/0502/runs/.sanity_pass` if comparator passes.
2. Land C1-C5d tooling fixes before launching any main A/C/D comparison.
3. Land R1a protocol skeleton with physical `A_seed43.yaml` and `A_pair_uniform_spot_aligned.yaml`.
4. Launch `A_main` through `run_ablation.sh A` on the first safe free GPU.
5. Launch `A_seed43` serially or opportunistically, depending on available GPU and host RAM.
6. Lock R1b only after A_main reaches the declared locked window and after the exact Method-D/threshold protocol is frozen.
7. Launch `C_uniform` only after the hard lock gate verifies the lock file exists, is on canonical remote, and has `LOCKED_PROTOCOL_VERSION == "R1b_locked"`.
8. Run `A_pair_uniform_spot_aligned` as robustness, not as a critical-path prerequisite unless the paper claim explicitly depends on Risk 4.

---

## 9. Direct Answers for Local Integration

- Q9: B is running, metrics path provided, 46 val rows copied as a schema fixture.
- Q10: use URL/fragment based canonical remote resolution; do not hardcode `gitee`.
- Q11: choose `a-with-cleanup`; only semantic change is `max_steps=120000`.
- Q12: launch A_main via `PYTHON=/home/qujiaxiang/.conda/envs/rae/bin/python GPU=<N> bash review/0502/scripts/run_ablation.sh A`.
- Q13: physical `A_seed43.yaml`, project-standard data layout under `/data_2/.../review_0502_runs/A_seed43/run`.
- Q14: no free PET-safe GPU at sample time; A_main should wait for B sentinel plus tooling/R1a.
- Q15: no known hard external deadline.

---

## 10. Bottom Line

The current architecture direction is reasonable, but the codebase is not yet in a safe state for the main sigma-normalize A/C experiment. The good news is that A_main/C_uniform have not started, so the remaining issues are fix-before-launch tooling and protocol issues rather than post-hoc contamination.

Do not treat REV1 §10 as implemented until the actual scripts are patched and verified against the real `step`-key metrics and current `step_*.pt` checkpoint names.
