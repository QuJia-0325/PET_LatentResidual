# V13/V14 Training and Full-Val Evaluation Summary

Date: 2026-05-21
Branch: `foc_lite_hop0`

## Scope

This note summarizes the completed V13 and V14 runs and the standard full-val evaluation performed after training completion.

Experiments:

- V13: `true_image_aux_ablation`, image auxiliary loss disabled.
- V14: `true_d_pure`, V7-style seed 1337 setting with image auxiliary supervision enabled.

The purpose is to compare whether removing image auxiliary supervision harms the final rollout/transport quality, and to record fair `best.pt` / `last.pt` full-val results.

## Training Completion

Both runs finished normally at step 160000.

| Experiment | Output run | Final step | Status |
|---|---|---:|---|
| V13 | `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0516_runs/V13_true_image_aux_ablation/run/first_hop_224_v13_true_image_aux_off` | 160000 | completed |
| V14 | `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0516_runs/V14_true_d_pure/run/first_hop_224_v14_v7_seed1337` | 160000 | completed |

Training logs committed with this report:

- `review/0516/V13_true_image_aux_ablation/V13_train_20260518_183547.log`
- `review/0516/V14_true_d_pure/V14_train_20260518_183547.log`

## Trainer-Side Final Full-Val Metrics

These are the final `val_full` records from each run's `metrics.jsonl`, using all 7403 validation slices.

| Experiment | D20 MSE | D10 MSE | D4 MSE | NORMAL MSE | Tail MSE | Select score |
|---|---:|---:|---:|---:|---:|---:|
| V13 image_aux off | 0.000340035 | 0.000309476 | 0.000277580 | 0.000260629 | 0.000282562 | 0.000950047 |
| V14 true_d_pure | 0.000330897 | 0.000296775 | 0.000263167 | 0.000245523 | 0.000268488 | 0.000904133 |

V14 is better than V13 on every chain MSE endpoint. The largest practical difference is at NORMAL, where V14 reduces MSE from `0.000260629` to `0.000245523`.

## Standard Full-Val Evaluation Protocol

Script:

```bash
review/0505/operator/scripts/eval_first_hop_fullval_psnr_chain_mse.py
```

Common settings:

- Split: `val`
- Number of slices: full validation set, 7403 slices
- `--max-slices 0`
- `--batch-size 8`
- `--decode-mode both`
- Metric: repository-standard `PSNR_clip3`
- RAE checkpoint: `/data_2/qujiaxiang/outputs/pet_lora_dinov2_pt_224/best_model.pt`

Full-eval logs committed with this report:

- `review/0516/logs_eval/v13_best_fullval_psnr_clip3_20260521.log`
- `review/0516/logs_eval/v13_last_fullval_psnr_clip3_20260521.log`
- `review/0516/logs_eval/v14_best_fullval_psnr_clip3_20260521.log`
- `review/0516/logs_eval/v14_last_fullval_psnr_clip3_20260521.log`

Full-eval JSON outputs copied into the repo:

- `review/0516/full_eval_json/v13_true_image_aux_off_best_fullval_psnr_chain_mse.json`
- `review/0516/full_eval_json/v13_true_image_aux_off_last_fullval_psnr_chain_mse.json`
- `review/0516/full_eval_json/v14_true_d_pure_best_fullval_psnr_chain_mse.json`
- `review/0516/full_eval_json/v14_true_d_pure_last_fullval_psnr_chain_mse.json`

## Full-Val PSNR_clip3 Results

| Experiment weight | Step | D20 PSNR | D10 PSNR | D4 PSNR | NORMAL PSNR | Transport avg | Tail MSE | NORMAL MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V13 best.pt | 160000 | 35.217159 | 35.597090 | 36.112229 | 36.494330 | 35.855202 | 0.000282562 | 0.000260629 |
| V13 last.pt | 160000 | 35.217159 | 35.597090 | 36.112229 | 36.494330 | 35.855202 | 0.000282562 | 0.000260629 |
| V14 best.pt | 160000 | 35.424869 | 35.810147 | 36.367695 | 36.780632 | 36.095836 | 0.000268488 | 0.000245523 |
| V14 last.pt | 160000 | 35.424869 | 35.810147 | 36.367695 | 36.780632 | 36.095836 | 0.000268488 | 0.000245523 |

## Interpretation

1. V14 is clearly better than V13.

V14 improves NORMAL PSNR over V13 by about `+0.2863 dB` and reduces tail MSE by about `1.407e-5`. This is consistent between trainer-side full-val metrics and the standalone full-eval script.

2. Disabling image_aux is harmful in this setting.

V13 has `lambda_img=0`, and its D20/D10/D4/NORMAL chain metrics are all worse. This supports the view that image auxiliary supervision remains useful for stabilizing or regularizing the transport chain, even though it is not a breakthrough mechanism by itself.

3. best.pt and last.pt are equivalent for these two runs under full-val.

For both V13 and V14, `best.pt` and `last.pt` produce identical full-val metrics. The final step 160000 is also the best checkpoint according to the trainer's full-val selection score.

4. V14 returns to the V7/V18-level regime but does not create a new jump.

V14's NORMAL PSNR is about `36.78 dB`, which is around the previous strong baseline level. Therefore, the result is useful as a controlled confirmation that image_aux should not be removed, but it is not evidence of a new transport breakthrough.

## Recommendation

Do not continue the image_aux-off direction as a primary path. Keep V14/V7-style image auxiliary supervision as the safer default, and focus the next design iteration on mechanisms that directly change the transport mismatch or chain selection objective rather than simply removing image-level auxiliary loss.
