# V5 Attribution Feedback — 2026-04-28

## Scope

This note reviews the V5 attribution / causal-control part in `review/0428/operator/README.md` and `review/0428/operator/04_v5_attribution.sh`, based on the current code at commit `393e57e`.

V6 has not finished and should not be evaluated yet. V5 attribution is a parallel diagnostic line; it should not block starting V6 because V6 is a from-scratch run and does not resume from V5.

## Current State Observed Locally

- Existing full-val outputs found:
  - `/data_2/qujiaxiang/outputs/PET_LatentResidual/eval_0427_fullval/v3_best`
  - `/data_2/qujiaxiang/outputs/PET_LatentResidual/eval_0427_fullval/v5_best`
  - `/data_2/qujiaxiang/outputs/PET_LatentResidual/eval_0427_fullval/v5_step_100000`
- No local output directory was found for:
  - `/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v5_null_control`
  - `/data_2/qujiaxiang/outputs/PET_LatentResidual/eval_0428_v5_attribution`

Therefore, the V5 attribution loop is not complete locally. We currently have V3/V5 full-val comparison, but not the null-control branch needed to separate V5 loss changes from resume/LR effects.

## Design Assessment

The attribution design is conceptually reasonable.

The intended question is:

> Did V5 change behavior because of the V5 loss recipe, or because we resumed from V3 best and continued training under a particular LR schedule?

The null-control experiment is the correct control for this question: resume from the same V3 checkpoint, keep the V3 loss recipe, continue training for the same additional budget, and then run the same full-val protocol.

Expected interpretation:

| Result | Interpretation |
|---|---|
| V5 improves and null-control does not | V5 loss recipe likely contributed useful signal |
| V5 degrades and null-control does not | V5 loss recipe likely caused degradation |
| Both V5 and null-control degrade | Resume/LR/continued training is likely the main confound |
| Both V5 and null-control improve | Improvement may come from continued training, not necessarily V5 |

This means V5 attribution is useful, but only after null-control training and full-val are actually completed.

## Code-Level Findings

### 1. `04_v5_attribution.sh` is not a true one-click completion script

The README says the script will automatically check 0427 results and only run missing parts. However, if null-control training is missing, the script only prints a command and exits that branch:

- `review/0428/operator/04_v5_attribution.sh:83-85`

```bash
echo "[TODO] Null-control 训练未启动。如需启动:"
echo "  bash review/0427/operator/03_null_control.sh ${GPU_ID}"
```

This is fine as a manual checklist, but it is not a complete automatic attribution pipeline.

Recommendation: either update README wording to say it is a status/check helper, or make the script actually launch `review/0427/operator/03_null_control.sh` when null-control is missing.

### 2. Eval CLI argument is wrong in the 0428 attribution script

`eval_first_hop_224_clip3.py` accepts `--out-dir`, not `--output-dir`.

Current problematic calls:

- `review/0428/operator/04_v5_attribution.sh:36-40`
- `review/0428/operator/04_v5_attribution.sh:50-54`
- `review/0428/operator/04_v5_attribution.sh:76-80`

If the 0427 full-val outputs already exist, the first two paths are skipped and the bug may stay hidden. But any missing eval branch will fail once executed.

Recommendation: replace `--output-dir` with `--out-dir` in `04_v5_attribution.sh`.

### 3. Null-control completion check is too strict

The script treats null-control training as complete only when `last.pt` exists:

- `review/0428/operator/04_v5_attribution.sh:61`

```bash
if [ -d "${NC_DIR}" ] && [ -f "${NC_DIR}/last.pt" ]; then
```

This can skip a valid partially completed null-control run that already has `best.pt` or a usable `step_*.pt` checkpoint.

Recommendation: check for any usable checkpoint, for example `best.pt`, `last.pt`, or latest `step_*.pt`. Report completion status separately from checkpoint availability.

### 4. Null-control full-val output path is inconsistent across scripts

`04_v5_attribution.sh` checks:

- `/data_2/qujiaxiang/outputs/PET_LatentResidual/eval_0427_fullval/null_control_best`

But `review/0427/operator/05_fullval_null_control.sh` writes to:

- `/data_2/qujiaxiang/outputs/PET_LatentResidual/eval_0427_fullval_null_control/null_best`

This mismatch can cause `04_v5_attribution.sh` to think null-control full-val is missing even if `05_fullval_null_control.sh` already finished.

Recommendation: standardize the null-control full-val output path. Prefer reusing `eval_0427_fullval_null_control/null_best` because that is what the existing 0427 script writes.

### 5. `03_null_control.sh` modifies a tracked config file in place

`review/0427/operator/03_null_control.sh` updates `configs/pet_flow/pet_flow_first_hop_224_v5_null_control.yaml` via `sed -i`:

```bash
sed -i "s/^  max_steps: .*/  max_steps: ${MAX_STEPS}/" "${CONFIG}"
```

This makes the git worktree dirty after launching the experiment. It also means repeated launches can silently mutate the tracked config.

Recommendation: generate a runtime config copy under a non-tracked launch directory, or write the resolved config into the experiment output directory before launch.

## Practical Recommendation

For the current phase:

1. Start V6 from scratch and monitor early Go/No-Go metrics. V5 attribution should not block V6.
2. Treat current V5 attribution as incomplete until null-control training plus full-val are done.
3. Before relying on `04_v5_attribution.sh`, fix the `--out-dir` argument and path mismatch.
4. If time is limited, the minimum useful V5 attribution result is: V3 best full-val, V5 best full-val, null-control best full-val under the same clip3 full-val protocol.

## Bottom Line

The V5 attribution idea is sound, but the current operator implementation is only a partial helper and has script-level inconsistencies. The scientific conclusion should not be based on V5 attribution until the null-control branch is completed and evaluated with the same full-val protocol.
