# Full-val Eval Comparison (A+C vs C) - 2026-04-19

## Scope
- Evaluation script: `eval_first_hop_224_clip3.py`
- Split: `val`
- Setting: `max-slices=0` (full val)
- Evaluated slices: all runs report `num_eval_slices=7403`

## Mean PSNR (clip3)

| Scheme | Checkpoint | D50 | D20 | D10 | D4 | NORMAL | transport_avg(D20,D10,D4,NORMAL) | all_avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A+C | best.pt | 42.622378 | 35.573978 | 35.941440 | 36.521874 | 36.792607 | 36.207475 | 37.490455 |
| A+C | last.pt | 42.622378 | 35.576589 | 35.953563 | 36.537062 | 36.847919 | 36.228783 | 37.507502 |
| C-only | best.pt | 42.622378 | 35.569034 | 35.942134 | 36.505390 | 36.805522 | 36.205520 | 37.488891 |
| C-only | last.pt | 42.622378 | 35.570788 | 35.948679 | 36.504915 | 36.793369 | 36.204438 | 37.488026 |

## A+C vs C Delta (PSNR, positive means A+C better)

| Pair | D20 | D10 | D4 | NORMAL | transport_avg | all_avg |
|---|---:|---:|---:|---:|---:|---:|
| best.pt | +0.004945 | -0.000694 | +0.016484 | -0.012915 | +0.001955 | +0.001564 |
| last.pt | +0.005801 | +0.004884 | +0.032147 | +0.054550 | +0.024346 | +0.019476 |

## Artifacts
- A+C best JSON: `/data_2/qujiaxiang/outputs/PET_LatentResidual/clip3_eval/ac_irepa_align_50k_best_fullval_20260418/first_hop_224_val_clip3_eval.json`
- A+C last JSON: `/data_2/qujiaxiang/outputs/PET_LatentResidual/clip3_eval/ac_irepa_align_50k_last_fullval_20260418/first_hop_224_val_clip3_eval.json`
- C best JSON: `/data_2/qujiaxiang/outputs/PET_LatentResidual/clip3_eval/schemec_imgaux_boost_50k_best_fullval_20260419/first_hop_224_val_clip3_eval.json`
- C last JSON: `/data_2/qujiaxiang/outputs/PET_LatentResidual/clip3_eval/schemec_imgaux_boost_50k_last_fullval_20260419/first_hop_224_val_clip3_eval.json`

## Logs in this folder
- `ac_fullval_best_existing_20260418.log`
- `ac_fullval_last_existing_20260418.log`
- `schemec_fullval_best_20260419.log`
- `schemec_fullval_last_20260419.log`
