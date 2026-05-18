# V18 Metric Correction Report 2026-05-18

## What Was Wrong

In the earlier V18 summary, the `45.6 dB` values were labeled as `PSNR_clip3`. That was incorrect.

Those values were derived from training-time `val_full` fields such as:

- `val_chain_d20_mse`
- `val_chain_d10_mse`
- `val_chain_d4_mse`
- `val_chain_normal_mse`

and then post-hoc converted with:

```text
PSNR = 10 * log10(9 / MSE)
```

This is not the project's canonical full-val evaluation protocol.

## Why It Was Wrong

The project-standard full-val metric is computed by:

- script: `review/0505/operator/scripts/eval_first_hop_fullval_psnr_chain_mse.py`
- function: `src.utils.metrics.calc_psnr_clip3`

That evaluator:

- decodes each sample
- maps image values back to SUV space
- clips to `[0, 3]`
- computes per-slice `PSNR_clip3`
- averages over the full validation set

This is not equivalent to converting an already-averaged training MSE into dB.

## Correct Canonical Results

Full-val split: `n=7403`

### best.pt

- `D20`: `35.4382 dB`
- `D10`: `35.8346 dB`
- `D4`: `36.3944 dB`
- `NORMAL`: `36.8112 dB`

### last.pt

- `D20`: `35.4212 dB`
- `D10`: `35.8439 dB`
- `D4`: `36.4124 dB`
- `NORMAL`: `36.8426 dB`

## Impact On Interpretation

- The corrected V18 results are fully consistent with the historical `~36 dB` range.
- The earlier `45.6 dB` numbers should be ignored for model comparison.
- `last.pt` is only marginally better than `best.pt` on canonical `NORMAL PSNR_clip3` (`+0.0315 dB`).

## Corrected Files

- Corrected summary: [V18_FINAL_RESULTS_20260518.md](/home/qujiaxiang/project/PET_LatentResidual/review/0517/V18_decoder_lora/V18_FINAL_RESULTS_20260518.md)
- Corrected snapshot: [V18_rank32_final_summary_20260518.json](/home/qujiaxiang/project/PET_LatentResidual/review/0517/V18_decoder_lora/run_snapshots/V18_rank32_final_summary_20260518.json)
- Canonical eval report: [V18_FULLVAL_EVAL_20260518.md](/home/qujiaxiang/project/PET_LatentResidual/review/0517/V18_decoder_lora/fullval_eval_20260518/V18_FULLVAL_EVAL_20260518.md)
