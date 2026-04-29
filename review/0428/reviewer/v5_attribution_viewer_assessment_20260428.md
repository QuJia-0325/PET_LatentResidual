# V5 Attribution Viewer Assessment - 2026-04-28

Update: a 2026-04-29 V5 addendum is available at [v5_attribution_viewer_update_20260429.md](v5_attribution_viewer_update_20260429.md). It extends the null-control evidence from step 100850 to 123550; the core verdict below remains unchanged. V6 is analyzed separately in [v6_phase1_viewer_analysis_20260429.md](v6_phase1_viewer_analysis_20260429.md).

## 0. Reviewer Verdict

当前 V5 attribution 设计方向是正确的，但证据尚未闭环。可以支持的结论是：

> Null-control 快照显示，从 V3 best resume 并保持 V3 recipe 继续训练时，训练仍主要由 image auxiliary weighted loss fraction 主导；rolling-val 没有显示稳定改善。这个结果支持“V3 后段 plateau 可能与 loss recipe / supervision pressure 分配有关”的假设。

但 author 文档中的强结论需要降级：

> “null-control 证明 V3 plateau 是权重配方病，不是训练量不足”目前过强。当前同步日志最多支持这个方向，尚不能作为 claim-level proof。最终判断必须等待 null-control full-val、V5 rollout-heavy full-val 数值同步、以及 Path A diagnostic。

因此当前状态是：**GO for continuing V6 pilot; NO-GO for using V5 attribution as closed causal evidence; NO-GO for approving V6 full 200K solely from current V5 attribution.**

## 1. Evidence Boundary

本 assessment 只基于当前 repo 中可读取的同步文件：

| Evidence | Local path | Status |
|---|---|---|
| Author analysis | `review/0428/local/v5_attribution_analysis_20260428.md` | readable |
| Null-control train snapshot | `review/0428/operator/log_snapshots/v5_null_control_gpu3_train_snapshot_20260428_143153.log` | readable, through step 100850 |
| V3 full-val summary | `review/0427/logs_eval/v3_200k_best_fullval_clip3_summary.log` | readable |
| V5 rollout-heavy counter-review | `review/0427/v5_rollout_heavy_counter_review_20260427.md` | readable |
| Attribution operator script | `review/0428/operator/04_v5_attribution.sh` | readable |
| Remote full-val outputs | `/data_2/qujiaxiang/outputs/PET_LatentResidual/...` | referenced by docs, not locally readable here |

Important limitation: V5 full-val is referenced as existing at remote paths such as `eval_0427_fullval/v5_best` and `eval_0427_fullval/v5_step_100000`, but the actual JSON/CSV/summary files are not synced into the local `review/` tree. Therefore this reviewer cannot verify V5 full-val numbers from the current workspace alone.

## 2. Raw Evidence Tables

### 2.1 V3 Baseline Full-Val

From `review/0427/logs_eval/v3_200k_best_fullval_clip3_summary.log`:

| Metric | Value |
|---|---:|
| checkpoint | `first_hop_224_200k_transport_v3/best.pt` |
| best rolling-val step | 86800 |
| full-val slices | 7403 |
| D50 PSNR clip3 | 42.622378 |
| D20 PSNR clip3 | 35.544402 |
| D10 PSNR clip3 | 35.976761 |
| D4 PSNR clip3 | 36.557899 |
| NORMAL PSNR clip3 | 36.865045 |
| best rolling `val_chain_normal_mse` | 0.0001645001 |
| last rolling `val_chain_normal_mse` | 0.0003011570 |

This gives a reliable full-val anchor. Rolling-val values are included only as training context, not as claim-level metrics.

### 2.2 Null-Control Launch Sanity

From `review/0428/operator/log_snapshots/v5_null_control_gpu3_train_snapshot_20260428_143153.log`:

| Item | Observed value | Reviewer judgment |
|---|---|---|
| run dir | `/data_2/.../first_hop_224_v5_null_control` | expected |
| resume checkpoint | `/data_2/.../first_hop_224_200k_transport_v3/best.pt` | correct control origin |
| restored step | 86800 | matches V3 best rolling step |
| restored best val | 0.000592 | matches V3 best metric context |
| LR schedule | `total_steps_override=200000` | important confound fixed |
| rollout recipe | `lambda_roll=0.2500`, `alpha=1.0000` | V3-like post-ramp recipe |
| image recipe | `lambda_img=0.1200` | V3-like image auxiliary strength |

This means the null-control branch is a valid control for continued training from V3 best without V5 loss changes, at least at launch/config level.

### 2.3 Null-Control Weighted Loss Fractions

The logged `pair_frac`, `roll_frac`, and `img_frac` are weighted loss fractions / supervision pressure proxies. They are not direct gradient share measurements.

| Step | pair_frac | roll_frac | img_frac | Observation |
|---:|---:|---:|---:|---|
| 86850 | 0.009 | 0.050 | 0.940 | image dominated immediately after resume |
| 87200 | 0.027 | 0.072 | 0.901 | image dominated |
| 96000 | 0.018 | 0.055 | 0.927 | image dominated |
| 100000 | 0.032 | 0.097 | 0.871 | transport still minor |
| 100800 | 0.021 | 0.043 | 0.936 | image dominated |
| 100850 | 0.027 | 0.063 | 0.910 | image dominated |

Observation: the author's broad statement that null-control remains image dominated is supported. However, the wording should be “weighted loss fraction” rather than “gradient budget.” Batch-level outliers exist, so the argument should rely on the repeated pattern, not one cherry-picked row.

### 2.4 Null-Control Rolling-Val Trajectory

| Step | window_start | samples | `val_chain_normal_mse` | `val_select_score` | Reviewer note |
|---:|---:|---:|---:|---:|---|
| 87200 | 2752 | 512 | 0.000169 | 0.000631 | close to V3 best rolling context |
| 92800 | 3648 | 428 | 0.000164 | 0.000594 | tail partial window |
| 93200 | 0 | 512 | 0.000159 | 0.000617 | window wrap; not a phase transition |
| 93600 | 64 | 512 | 0.000274 | 0.001047 | worse window |
| 95600 | 384 | 512 | 0.000362 | 0.001400 | worse window |
| 98400 | 832 | 512 | 0.000170 | 0.000620 | good window |
| 98800 | 896 | 512 | 0.000166 | 0.000624 | good window |
| 99600 | 1024 | 512 | 0.000339 | 0.001297 | bad window |
| 100000 | 1088 | 512 | 0.000223 | 0.000825 | mid window |
| 100400 | 1152 | 512 | 0.000209 | 0.000767 | mid window |
| 100800 | 1216 | 512 | 0.000294 | 0.001110 | bad window |

Observation: the null-control rolling curve does not show stable improvement. It alternates between good and bad windows, consistent with rolling validation window composition effects. The 0.000159 at step 93200 is not reliable improvement evidence because it occurs exactly at `window_start=0` after wrap.

### 2.5 V5 Rollout-Heavy Evidence Status

The local 0427 counter-review already corrected the most important interpretive issue:

| Claim from earlier V5 analysis | Reviewer correction |
|---|---|
| step 93200 is a phase transition | unsupported; 92800 to 93200 changes validation window |
| V4/V5 synchronized collapse proves narrow basin | over-inferred; both share resume step and eval schedule |
| `val_pair_total` jump proves irreversible manifold exit | unsupported; later rolling windows show recovery in `val_pair_total` |
| V5/V4 logged `val_select_score` can be compared directly | not rigorous; best metric terms differ |

Current local conclusion for V5 rollout-heavy: rolling-val does not prove stable gain, but also cannot by itself prove final failure. Full-val is the required arbiter.

## 3. Key Findings

### Finding 1: Null-control setup is valid, but not complete

Observation: null-control resumes from V3 best at step 86800, restores EMA, uses `total_steps_override=200000`, and logs V3-like `lambda_roll=0.25`, `lambda_img=0.12`, `alpha=1.0`.

Interpretation: this is the correct control for separating V5 loss changes from continued-training / resume / LR-schedule confounds.

Implication: null-control is scientifically useful, but only after it reaches comparable budget and receives full-val evaluation.

Next step: continue/sync null-control to the agreed endpoint, then run full-val on best/last/latest usable checkpoint.

### Finding 2: Image dominance is real in weighted loss fractions

Observation: visible null-control rows repeatedly show `img_frac` around 0.87-0.94, while `pair_frac + roll_frac` usually stays below 0.13.

Interpretation: under the V3 recipe after resume, image auxiliary supervision remains the dominant weighted objective term.

Implication: it is reasonable to suspect that V3's late-stage training pressure is not transport-first.

Next step: preserve this as a hypothesis and verify with full-val outcomes. Do not state it as complete causal proof yet.

### Finding 3: Rolling-val is not sufficient for claim-level attribution

Observation: `val_chain_normal_mse` fluctuates from 0.000159 to 0.000382+ across rolling windows, with good values reappearing later at different windows.

Interpretation: window composition dominates the observed short-run rolling-val signal.

Implication: neither “null-control improves” nor “null-control fails” should be claimed from rolling-val alone.

Next step: use full-val clip3 PSNR and fixed-window diagnostics for all compared checkpoints.

### Finding 4: V5 rollout-heavy remains unresolved from local evidence

Observation: docs reference remote V5 full-val outputs, but those JSON/CSV summaries are not available in the local workspace. Rolling-val evidence is already known to be confounded.

Interpretation: current local files are insufficient to determine whether V5 rollout-heavy is harmful, neutral, or mildly useful under full-val.

Implication: author's V6 rationale should not rely on an unverified statement that V5 definitively failed under full-val.

Next step: sync V5 full-val summaries into `review/0428/reviewer` or `review/0428/operator/review`, then compare with V3 full-val.

### Finding 5: V6 pilot can continue, but V5 attribution should not greenlight V6 full run

Observation: V6 is from scratch and does not inherit V5 state, so V5 attribution does not block V6 pilot. However, V5 attribution is still incomplete as a causal basis.

Interpretation: V6's early staged gates should be evaluated on their own metrics, while V5 attribution closes in parallel.

Implication: GO for V6 pilot monitoring; NO-GO for direct V6 full 200K approval based only on current V5 attribution.

Next step: require V6 Phase gate plus V5 attribution full-val closure before escalating V6.

## 4. Recommended Edits to Author Framing

The author document should keep the overall direction but soften these statements:

| Current framing | Recommended reviewer-safe framing |
|---|---|
| “image_aux 占据梯度预算” | “image_aux dominates weighted loss fraction / supervision pressure” |
| “null-control 证明 V3 plateau 不是训练量不足” | “null-control rolling snapshot supports the hypothesis that simple continued training is insufficient, pending full-val” |
| “V3 plateau 是权重配方病” | “V3 plateau is consistent with a recipe/supervision-pressure bottleneck, not yet proven as the only cause” |
| “V6 基于正确归因” | “V6 is motivated by the current attribution hypothesis and still requires staged validation” |

Suggested one-sentence replacement:

> Current null-control logs support the diagnosis that V3-like continued training remains image-dominated and does not show stable rolling-val improvement, but full-val and Path A are still required before treating this as a closed causal attribution.

## 5. Required Next Operations

### Priority 1: Sync or generate full-val summaries

Minimum required comparison:

| Branch | Checkpoint | Required metric |
|---|---|---|
| V3 baseline | `v3_best` | full-val `summary_psnr_clip3.NORMAL.mean`, D20/D10/D4 |
| V5 rollout-heavy | `v5_best` | same protocol |
| V5 rollout-heavy | `v5_step_100000` or latest usable step | same protocol |
| V5 null-control | best/last/latest usable checkpoint | same protocol |

The files to sync or summarize should include:

```text
/data_2/qujiaxiang/outputs/PET_LatentResidual/eval_0427_fullval/v5_best/first_hop_224_val_clip3_eval.json
/data_2/qujiaxiang/outputs/PET_LatentResidual/eval_0427_fullval/v5_step_100000/first_hop_224_val_clip3_eval.json
/data_2/qujiaxiang/outputs/PET_LatentResidual/eval_0427_fullval_null_control/null_best/first_hop_224_val_clip3_eval.json
/data_2/qujiaxiang/outputs/PET_LatentResidual/eval_0428_v5_attribution/null_control_best/first_hop_224_val_clip3_eval.json
```

Only one null-control full-val location is needed, but both paths should be checked because 0427 and 0428 scripts use different output bases.

### Priority 2: Complete null-control to comparable budget

The visible snapshot reaches step 100850, about 14050 steps after V3 best. That is useful for early diagnosis but not enough to match the V5 rollout-heavy budget if V5 was evaluated around +50K or at specific saved checkpoints.

After completion, run:

```bash
bash review/0428/operator/04_v5_attribution.sh <GPU_ID>
```

### Priority 3: Run Path A diagnostics

Effect-only full-val answers whether V5 helped. Path A answers whether the rollout exposure gap mechanism improved.

Run or sync:

```bash
bash review/0427/operator/02_pathA_baseline.sh <GPU_ID>
bash review/0427/operator/04_pathA_v5_best.sh <GPU_ID>
```

### Priority 4: Write a final attribution table

After full-val and Path A are available, produce one final table:

| Model | Checkpoint | NORMAL PSNR | D20 PSNR | transport avg | Path A gap | Interpretation |
|---|---|---:|---:|---:|---:|---|
| V3 | best | 36.865 | 35.544 | TBD | TBD | baseline |
| V5 rollout-heavy | best | TBD | TBD | TBD | TBD | pending |
| V5 rollout-heavy | step_100000/latest | TBD | TBD | TBD | TBD | pending |
| V5 null-control | best/latest | TBD | TBD | TBD | TBD | pending |

## 6. Decision Matrix

| Full-val outcome | Interpretation | Action |
|---|---|---|
| V5 > V3, null-control not > V3 | V5 loss recipe likely contributed useful signal | inspect Path A; consider V5-inspired components |
| V5 <= V3, null-control ~= V3 | V5 rollout-heavy recipe likely not useful | continue V6 pilot; do not use V5 as positive evidence |
| V5 <= V3, null-control also degrades | resume/LR/continued-training confound likely important | avoid resume-based conclusions; prioritize from-scratch V6 |
| V5 > V3, null-control also > V3 | continued training / checkpoint selection may explain gain | do not attribute gain to V5 recipe without further controls |
| V5 full-val improves but Path A worsens | effect may come from non-rollout mechanism | revise mechanism claim |
| V5 full-val worsens but Path A improves | mechanism improves but external quality degrades | tune image/pair balance; do not claim end-to-end success |

## 7. Final Reviewer Position

The current best scientific statement is:

> V5 attribution is in progress. The null-control launch is technically correct and its current logs support the diagnosis that V3-like continued training remains image-dominated and does not show stable rolling-val improvement. However, because rolling-val is window-confounded and V5/null-control full-val summaries are not locally available, the causal attribution is not yet closed. V6 pilot can continue as an independent from-scratch experiment, but V5 attribution should not be used as final proof until full-val and Path A are synced and compared.