# Round 17 Codex Task Execution Fixes

- date: 2026-05-22
- branch: `foc_lite_hop0`
- target file: `review/0521/CODEX_TASK_ROUND17_F0_A4_20260522.md`

## Summary

This patch fixes the remaining execution-level issues in the Round 17 F0/A4 task after the Round17-Prep update. The strategic design is unchanged: run F0 slice-level statistics first, then optionally launch one A4 image_aux-strength probe.

## Fixes Applied

1. **Eval output directory now respects `path_guard`.**

`eval_first_hop_fullval_psnr_chain_mse.py` requires `--out-dir` to be under `/data_2`. The task previously used repo-local paths such as `review/0521/v13_v14_per_slice`, which would fail at runtime. The task now writes eval outputs to `/data_2/...` and copies JSON/CSV/log snapshots back into `review/0521/...` for review and git tracking.

2. **A4 output path corrected.**

`train_first_hop.py` writes to `output_dir / run_name`. With the A4 yaml values, the actual run directory is:

```text
/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0521_runs/A4_image_aux_lambda_08/first_hop_224_a4_image_aux_lambda_08
```

The previous task path incorrectly inserted an extra `/run/`. This is now corrected.

3. **F0 report wording made statistically honest.**

Because patient/volume IDs are not available in the current artifacts, F0 is now explicitly described as **slice-level paired statistics** over 7403 validation slices, not patient-level independent inference.

The F0 script template now includes:

- mean delta
- paired SEM
- paired t / p value
- Cohen's d
- slice win-rate
- slice-level bootstrap 95% CI

4. **V18-cap leakage guard removed from the chain loop.**

The F0 chain `DATA` dictionary excludes V18-cap entirely because V18-cap is direct `decode(z_GT)` substrate and cannot be compared against chain rollout PSNR. The stale conditional guard was removed from the script template to avoid future confusion.

5. **V7 baseline table refreshed.**

The A4 report template now uses the canonical V7 best full-val PSNR values from `review/0511/fullval_psnr_clip3_20260516_173941/artifacts/planf_v7_best_fullval_psnr_chain_mse.json`:

| Timepoint | PSNR_clip3 |
|---|---:|
| D20 | 35.4354 |
| D10 | 35.8194 |
| D4 | 36.3736 |
| NORMAL | 36.7810 |

6. **F0 execution semantics and git remote commands corrected.**

F0 is no longer described as a pure `0 GPU` task. The corrected wording is: F0 does not train, but if V13/V14 per-slice CSV files are missing, canonical eval must first run on one free GPU to generate the CSVs; the paired statistics step itself is CPU-only.

The local gitee remote is named `origin`, so task commands now use:

```bash
git pull --ff-only origin foc_lite_hop0
git push origin foc_lite_hop0
```

## Remaining Caveat

F0 can support slice-level evidence and practical effect-size interpretation. It cannot by itself establish patient-level statistical significance unless patient/volume IDs are later recovered and used for clustered or patient-level resampling.
