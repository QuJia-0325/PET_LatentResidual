# A4 Image Aux Lambda Bracket: Full-Val Eval Report

Date: 2026-05-25

## Scope

This report summarizes the two completed A4 image-auxiliary-strength experiments:

| Experiment | Config | Main change |
|---|---|---|
| A4-mid | `review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08.yaml` | `image_aux.lambda_start=lambda_max=0.08` |
| A4-low | `review/0521/A4_image_aux_lambda_02/A4_image_aux_lambda_02.yaml` | `image_aux.lambda_start=lambda_max=0.02` |

Both experiments use the same seed (`42`), same transport/rollout settings, same `max_steps=160000`, same `best_metric=val_multi_objective`, and the same RAE decoder checkpoint. The intended comparison is therefore the strength of hop0 image auxiliary supervision.

## Evaluation Protocol

The full-val PSNR evaluation was run with:

- Script: `review/0505/operator/scripts/eval_first_hop_fullval_psnr_chain_mse.py`
- Split: `val`
- `--max-slices 0`, meaning the full validation split
- Number of evaluated slices: `7403`
- Batch size: `8`
- Decode mode: `both`
- PSNR metric: `src.utils.metrics.calc_psnr_clip3`
- Chain MSE definition: `mean((decode_crop(z_chain[t]) - x_rollout[t])^2)`, matching `train_first_hop.py` validation chain metrics

The training script also performed full-val selection at step 160000 (`best_select_full_eval_interval=5000`, `eval_chain_max_unique_slices=0`). The independent PSNR full-val eval reported below was run again for both `best.pt` and `last.pt`.

## Raw Results

Baseline for delta calculation: V7 best from `review/0511/fullval_psnr_clip3_20260516_173941/artifacts/planf_v7_best_fullval_psnr_chain_mse.json`.

V7 best headline:

| Metric | Value |
|---|---:|
| D20 PSNR_clip3 | 35.4354 |
| D10 PSNR_clip3 | 35.8194 |
| D4 PSNR_clip3 | 36.3736 |
| NORMAL PSNR_clip3 | 36.7810 |
| NORMAL chain MSE | 0.000245641 |
| tail chain MSE | 0.000268340 |
| transport PSNR avg | 36.1023 |

A4 full-val results:

| Experiment | lambda_img | Ckpt | Step | Slices | D20 PSNR | D10 PSNR | D4 PSNR | NORMAL PSNR | Delta NORMAL vs V7 | NORMAL MSE | Tail MSE | Transport PSNR Avg |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A4-mid | 0.08 | best | 160000 | 7403 | 35.4918 | 35.8864 | 36.4593 | 36.8939 | +0.1130 | 0.000239971 | 0.000263639 | 36.1828 |
| A4-mid | 0.08 | last | 160000 | 7403 | 35.4918 | 35.8864 | 36.4593 | 36.8939 | +0.1130 | 0.000239971 | 0.000263639 | 36.1828 |
| A4-low | 0.02 | best | 160000 | 7403 | 35.3737 | 35.7536 | 36.2961 | 36.7010 | -0.0800 | 0.000249068 | 0.000271762 | 36.0311 |
| A4-low | 0.02 | last | 160000 | 7403 | 35.3737 | 35.7536 | 36.2961 | 36.7010 | -0.0800 | 0.000249068 | 0.000271762 | 36.0311 |

Machine-readable summaries:

- `review/0521/A4_image_aux_lambda_bracket/A4_BRACKET_FULLVAL_SUMMARY_20260525.json`
- `review/0521/A4_image_aux_lambda_bracket/A4_BRACKET_FULLVAL_SUMMARY_20260525.csv`

## Training Completion Check

Both runs completed normally at step 160000.

A4-mid final training-side full-val line:

- `val_chain_d20_mse=0.000328`
- `val_chain_d10_mse=0.000293`
- `val_chain_d4_mse=0.000258`
- `val_chain_normal_mse=0.000240`
- `val_chain_samples=7403`
- `val_select_score=0.000888`
- New best at step 160000

A4-low final training-side full-val line:

- `val_chain_d20_mse=0.000333`
- `val_chain_d10_mse=0.000300`
- `val_chain_d4_mse=0.000266`
- `val_chain_normal_mse=0.000249`
- `val_chain_samples=7403`
- `val_select_score=0.000915`
- New best at step 160000

For both experiments, `best.pt` and `last.pt` evaluate identically in the independent full-val PSNR run because the final step 160000 checkpoint is also the selected best checkpoint.

## Interpretation

1. A4-mid (`lambda_img=0.08`) improves over V7 best on the full validation set.

   The main headline is `NORMAL PSNR_clip3=36.8939 dB`, which is `+0.1130 dB` over V7 best. D20 also improves by `+0.0564 dB`, and tail chain MSE decreases from `0.000268340` to `0.000263639`.

2. A4-low (`lambda_img=0.02`) is worse than V7 best.

   `NORMAL PSNR_clip3=36.7010 dB`, which is `-0.0800 dB` relative to V7 best. This indicates that too weak image auxiliary supervision does not preserve the benefit.

3. The bracket supports the hypothesis that hop0 image auxiliary supervision has a useful strength threshold.

   Within this controlled two-point bracket, `0.08` is better than both V7 and `0.02`. The result is not monotonic evidence over a dense sweep, but it is enough to reject the idea that merely enabling a tiny image auxiliary term is sufficient.

4. The result should still be treated as single-seed evidence.

   The full-val slice count is large (`7403`), but the training seed is still one seed. For paper-level claims, a paired slice-level bootstrap/win-rate report can strengthen the statement without claiming patient-level statistics.

## Recommendation

Use A4-mid (`lambda_img=0.08`) as the current best A4 bracket result. Do not use A4-low for claims except as a negative/weak-supervision control.

If we continue this line, the next experiment should not simply lower `lambda_img`; the useful direction is either a small local bracket around `0.08` or a mechanism-level analysis of why stronger hop0 image supervision improves D20 and propagates through the chain.

## Attached Logs And Artifacts

A4-mid (`lambda_img=0.08`):

- Config: `review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08.yaml`
- Training log: `review/0521/A4_image_aux_lambda_08/A4_train_20260522_013850_gpu0_tmux.log`
- Training metrics snapshot: `review/0521/A4_image_aux_lambda_08/A4_metrics_20260525_final.jsonl`
- Best eval log: `review/0521/A4_image_aux_lambda_08/fullval_eval/logs/a4_image_aux_lambda_08_best_eval_20260525_012246_gpu0.log`
- Last eval log: `review/0521/A4_image_aux_lambda_08/fullval_eval/logs/a4_image_aux_lambda_08_last_eval_20260525_013441_gpu0.log`
- Best JSON/CSV: `review/0521/A4_image_aux_lambda_08/fullval_eval/artifacts/a4_image_aux_lambda_08_best_fullval_psnr_chain_mse.*`
- Last JSON/CSV: `review/0521/A4_image_aux_lambda_08/fullval_eval/artifacts/a4_image_aux_lambda_08_last_fullval_psnr_chain_mse.*`

A4-low (`lambda_img=0.02`):

- Config: `review/0521/A4_image_aux_lambda_02/A4_image_aux_lambda_02.yaml`
- Training log: `review/0521/A4_image_aux_lambda_02/A4_low_train_20260522_024517_gpu1_tmux.log`
- Training metrics snapshot: `review/0521/A4_image_aux_lambda_02/A4_low_metrics_20260525_final.jsonl`
- Best eval log: `review/0521/A4_image_aux_lambda_02/fullval_eval/logs/a4_image_aux_lambda_02_best_eval_20260525_012246_gpu1.log`
- Last eval log: `review/0521/A4_image_aux_lambda_02/fullval_eval/logs/a4_image_aux_lambda_02_last_eval_20260525_013441_gpu1.log`
- Best JSON/CSV: `review/0521/A4_image_aux_lambda_02/fullval_eval/artifacts/a4_image_aux_lambda_02_best_fullval_psnr_chain_mse.*`
- Last JSON/CSV: `review/0521/A4_image_aux_lambda_02/fullval_eval/artifacts/a4_image_aux_lambda_02_last_fullval_psnr_chain_mse.*`
