# V5 Operator Fix And GPU3 Launch — 2026-04-28

## Background

This document records the local fixes applied after reviewing the V5 attribution/null-control operator flow.

The target experiment is the V5 null-control branch:

- Resume from V3 best checkpoint.
- Keep the V3/baseline loss recipe.
- Continue training for the same additional budget.
- Use full-val to distinguish V5 loss changes from resume/LR/continued-training effects.

## Fixed Issues

### 1. Avoid tracked config mutation in null-control launch

File: `review/0427/operator/03_null_control.sh`

Previous behavior:

```bash
sed -i "s/^  max_steps: .*/  max_steps: ${MAX_STEPS}/" "${CONFIG}"
```

This modified the tracked config file directly, leaving the git worktree dirty and making repeated launches less auditable.

New behavior:

- Keep `configs/pet_flow/pet_flow_first_hop_224_v5_null_control.yaml` unchanged.
- Copy it to `review/0427/logs_train/v5_null_control_gpu${GPU_ID}_runtime.yaml`.
- Patch only the runtime copy with the resolved `max_steps`.
- Launch training with the runtime config.

This preserves the canonical config and makes the exact launch config available with the training log.

### 2. Add explicit stale output directory guard

File: `review/0427/operator/03_null_control.sh`

Null-control uses `require_fresh_output_dir: true`, so launching into an existing output directory should fail early with a clear message.

New behavior:

- If `/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v5_null_control` already exists, the script exits before launching.
- The operator must then either evaluate the existing run or manually handle the old directory before rerunning.

### 3. Make null-control eval checkpoint fallback robust

File: `review/0428/operator/04_v5_attribution.sh`

Previous behavior:

- The script accepted `best.pt || last.pt` as evidence that null-control existed.
- But the eval branch still only evaluated `best.pt`.
- If only `last.pt` or `step_*.pt` existed, the script could silently skip usable checkpoints.

New behavior:

Checkpoint selection order is now:

1. `best.pt`
2. `last.pt`
3. latest `step_*.pt`

The selected checkpoint is printed before eval.

### 4. Clarify README wording

File: `review/0428/operator/README.md`

The old wording implied that `04_v5_attribution.sh` would one-click complete all V5 attribution work, including null-control training.

The updated wording clarifies:

- `04_v5_attribution.sh` checks/completes full-val pieces.
- If null-control training is missing, run `review/0427/operator/03_null_control.sh` first.

## Validation

Syntax check passed:

```bash
bash -n review/0427/operator/03_null_control.sh
bash -n review/0428/operator/04_v5_attribution.sh
```

## GPU3 Launch Plan

Launch command:

```bash
bash review/0427/operator/03_null_control.sh 3
```

Expected artifacts:

- Training log: `review/0427/logs_train/v5_null_control_gpu3.log`
- PID file: `review/0427/logs_train/v5_null_control_gpu3.pid`
- Runtime config: `review/0427/logs_train/v5_null_control_gpu3_runtime.yaml`
- Output dir: `/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v5_null_control`

After null-control finishes:

```bash
bash review/0428/operator/04_v5_attribution.sh 3
```

This will run or reuse V3/V5/null-control full-val results under the same `clip_max=3` protocol.

