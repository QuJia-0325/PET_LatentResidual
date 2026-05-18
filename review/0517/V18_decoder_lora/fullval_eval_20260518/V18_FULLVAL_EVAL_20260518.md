# V18 Full-Val Eval 2026-05-18

## Protocol
- Evaluator: `review/0505/operator/scripts/eval_first_hop_fullval_psnr_chain_mse.py`
- Metric: canonical `calc_psnr_clip3` on full val split (`n=7403` slices)
- Checkpoints:
  - `best.pt` from `first_hop_224_v18_decoder_lora`
  - `last.pt` from `first_hop_224_v18_decoder_lora`
- Training log snapshot included for remote review.

## Headline
- `best.pt` NORMAL `PSNR_clip3`: `36.8112 dB`
- `last.pt` NORMAL `PSNR_clip3`: `36.8426 dB`
- `last - best`: `+0.0315 dB`

## Per-stage PSNR_clip3
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

## Interpretation
- V18 canonical full-val PSNR is in the same range as the historical `~36 dB` results.
- This confirms the earlier `45.64 dB` numbers were not canonical full-val `PSNR_clip3`; they came from an invalid MSE-to-PSNR post-hoc conversion of training `val_full` statistics.
- `last.pt` is slightly better than `best.pt` on `NORMAL` under the canonical evaluator, but the gain is extremely small.

## Files
- Eval logs:
  - `review/0517/V18_decoder_lora/fullval_eval_20260518/logs/v18_best_fullval_20260518.log`
  - `review/0517/V18_decoder_lora/fullval_eval_20260518/logs/v18_last_fullval_20260518.log`
- Eval artifacts snapshot:
  - `review/0517/V18_decoder_lora/fullval_eval_20260518/artifacts/v18_best_fullval_psnr_chain_mse.json`
  - `review/0517/V18_decoder_lora/fullval_eval_20260518/artifacts/v18_best_fullval_psnr_chain_mse_per_slice.csv`
  - `review/0517/V18_decoder_lora/fullval_eval_20260518/artifacts/v18_last_fullval_psnr_chain_mse.json`
  - `review/0517/V18_decoder_lora/fullval_eval_20260518/artifacts/v18_last_fullval_psnr_chain_mse_per_slice.csv`
- Training log snapshot:
  - `review/0517/V18_decoder_lora/fullval_eval_20260518/logs/V18_rank32_train_gpu1_20260517_045325.log`
