# Round 17 F0/A4 Smoke Test Report

- date: 2026-05-22
- branch: `foc_lite_hop0`
- python: `/home/qujiaxiang/.conda/envs/rae/bin/python`
- GPU used for smoke: GPU0

## Summary

Smoke test passed after fixing one real execution bug in the canonical eval script. The A4 configuration is runnable and the 1-step training smoke confirmed `training.image_aux.lambda_start/max = 0.08` is actually active as `lambda_img=0.0800` in the trainer log.

## Changes Made During Smoke

1. Fixed `review/0505/operator/scripts/eval_first_hop_fullval_psnr_chain_mse.py` repo-root discovery.

The script previously used `Path(__file__).resolve().parents[3]`, which resolves to `.../PET_LatentResidual/review`, not the repository root. Direct execution failed with:

```text
ModuleNotFoundError: No module named 'pet_lr'
```

The script now searches upward for a directory containing `pet_lr/`, making it robust to the nested `review/0505/operator/scripts/` location.

2. Generated A4 configs:

- `review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08.yaml`
- `review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08_smoke.yaml`

The formal A4 yaml differs from V7 only in the intended 4 fields:

- `output_dir`
- `run_name`
- `training.image_aux.lambda_start: 0.04 -> 0.08`
- `training.image_aux.lambda_max: 0.04 -> 0.08`

3. Updated the execution task doc to use the working Python explicitly:

```bash
PYTHON=/home/qujiaxiang/.conda/envs/rae/bin/python
export PYTHONPATH=/home/qujiaxiang/project/PET_LatentResidual:${PYTHONPATH:-}
```

This avoids the default shell `python`, which currently has no `torch` installed.

## Smoke Commands and Results

### F0 canonical eval smoke

Command type: V13 best, `max-slices=2`, `decode-mode=both`, output under `/data_2`.

Result: passed.

Artifacts:

- log: `review/0521/A4_image_aux_lambda_08/smoke_logs/f0_eval_smoke_20260522_011139.log`
- JSON: `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0521_smoke/f0_eval_smoke/v13_best_smoke_max2_fullval_psnr_chain_mse.json`
- CSV: `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0521_smoke/f0_eval_smoke/v13_best_smoke_max2_fullval_psnr_chain_mse_per_slice.csv`

Smoke headline from 2 slices:

```text
normal_psnr: 40.0512 dB
transport_psnr_avg: 39.1374 dB
runtime_sec after loading: 15.96
```

This value is not a scientific result because `max-slices=2`; it only verifies the eval path, model loading, decode path, PSNR_clip3 call, and JSON/CSV writing.

### A4 1-step training smoke

Command type: `train_first_hop.py --config A4_image_aux_lambda_08_smoke.yaml`, `max_steps=1`, no mid-run eval.

Result: passed.

Artifacts:

- log: `review/0521/A4_image_aux_lambda_08/smoke_logs/a4_train_1step_smoke_20260522_011319.log`
- run dir: `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0521_smoke/A4_image_aux_lambda_08/first_hop_224_a4_image_aux_lambda_08_smoke`
- smoke `last.pt`: `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0521_smoke/A4_image_aux_lambda_08/first_hop_224_a4_image_aux_lambda_08_smoke/last.pt`

Key trainer line:

```text
[train] step=00001 ... lambda_img=0.0800 ... grad_total=2.5279e-02 ... Training done.
```

This verifies forward/backward, optimizer step, checkpoint save, and the intended image_aux lambda.

## Observed Costs

Data loading dominates startup:

- `latents_train.pt`: 115.3GB, loaded in 216.1s
- `latents_val.pt`: 32.5GB, loaded in 17.8s during train smoke
- raw image `.pt` files: about 9.2GB each, about 5-6s per load

Current system memory was sufficient for smoke and after completion remained healthy. Root filesystem is tight, so outputs must stay on `/data_2`.

## Potential Problems Before Formal Run

1. **Cold-start IO is nontrivial.**

Every formal training launch loads the 115GB train latent file and val/raw tensors at startup. Expect several minutes before the first train line. This is expected, not a hang.

2. **Default `python` is wrong.**

The default shell `python` does not have `torch`; formal commands must use `/home/qujiaxiang/.conda/envs/rae/bin/python`.

3. **Determinism warnings are present.**

The trainer emits CUDA deterministic warnings from CuBLAS, memory-efficient attention, and adaptive pooling. Current code uses warn-only deterministic behavior, so this does not stop training, but exact bitwise reproducibility is not guaranteed unless `CUBLAS_WORKSPACE_CONFIG` and attention determinism are further controlled.

4. **Smoke checkpoint is large.**

Even 1-step smoke writes a `last.pt` of about 3.6GB under `/data_2`. Do not commit checkpoint files.

## Verdict

A4 is ready to launch as a formal run, provided the launch command uses the `rae` Python and writes logs/artifacts under `review/0521/A4_image_aux_lambda_08` plus `/data_2` output directories.
