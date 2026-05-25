# X3 Image-Aux + Decoder LoRA Full-Val Eval Report

Date: 2026-05-26

## Scope

X3 is the A3-matched short-window additivity test:

- Init: `V7.best` at step 160000.
- Train window: 160000 -> 170000, i.e. 10000 additional steps.
- Changes relative to V7 during this window: decoder LoRA enabled, `image_aux.lambda_start=lambda_max=0.08`, and `decoder_kl_pullback.lambda_kl=0.0`.
- Evaluation: canonical full-val chain eval on `val`, `max_slices=0`, `num_eval_slices=7403`, `PSNR_clip3`, `decode_mode=both`.

This report should not be interpreted as a from-scratch A4-mid + LoRA experiment. It only answers whether a 10k LoRA/high-image-aux adaptation window adds value beyond the existing V7 warmstart.

## Eval Protocol

- Script: `review/0505/operator/scripts/eval_first_hop_fullval_psnr_chain_mse.py`
- Launcher: `review/0525/X3_image_aux_lora/launch_x3_fullval_eval_gpu3.sh`
- Config: `review/0525/X3_image_aux_lora/X3_image_aux_lora.yaml`
- Checkpoints:
  - `best.pt`, checkpoint step 165000
  - `last.pt`, checkpoint step 170000
- Local logs:
  - `review/0525/X3_image_aux_lora/fullval_eval/logs/x3_best_eval_20260526_gpu3.log`
  - `review/0525/X3_image_aux_lora/fullval_eval/logs/x3_last_eval_20260526_gpu3.log`
- Local artifacts:
  - `review/0525/X3_image_aux_lora/fullval_eval/artifacts/x3_image_aux_lora_best_fullval_psnr_chain_mse.json`
  - `review/0525/X3_image_aux_lora/fullval_eval/artifacts/x3_image_aux_lora_last_fullval_psnr_chain_mse.json`
  - matching per-slice CSVs in the same artifact directory

## Raw Results

Baseline values used for deltas:

- V7.best NORMAL PSNR: 36.78095134264647
- A4-mid NORMAL PSNR: 36.89391730892228
- V18.best NORMAL PSNR: 36.811167021208675
- V18.last NORMAL PSNR: 36.84264474364047

| Run | Step | Slices | D20 PSNR | D10 PSNR | D4 PSNR | NORMAL PSNR | Delta vs V7 | Delta vs A4-mid | NORMAL MSE | Tail MSE | Runtime sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| X3.best | 165000 | 7403 | 35.4390269322 | 35.8348284521 | 36.3941115168 | 36.8104495902 | +0.0294982475 | -0.0834677188 | 0.000245506686 | 0.000268399102 | 365.84 |
| X3.last | 170000 | 7403 | 35.4398255339 | 35.8466599199 | 36.4082254154 | 36.8287870105 | +0.0478356678 | -0.0651302984 | 0.000245426530 | 0.000268450270 | 360.88 |

Additional comparisons:

| Comparison | NORMAL PSNR delta |
|---|---:|
| X3.last - X3.best | +0.0183374203 |
| X3.best - V18.best | -0.0007174310 |
| X3.last - V18.last | -0.0138577332 |
| X3.last gain / A4-mid gain over V7 | 0.423452031 |

## Interpretation

1. X3 full-val eval completed cleanly for both `best.pt` and `last.pt`.

   Both logs reached `926/926` batches and wrote JSON plus per-slice CSV artifacts. No X3 eval process remains active.

2. `last.pt` is the better X3 checkpoint under canonical full-val PSNR.

   `X3.last` reaches NORMAL `36.8287870105 dB`, which is `+0.0183374203 dB` above `X3.best`.

3. The short-window additivity result is negative against A4-mid.

   `X3.last` is `+0.0478356678 dB` over V7, but it is still `-0.0651302984 dB` below A4-mid. It recovers only about `42.3%` of A4-mid's gain over V7.

4. X3 stays in the V18 range rather than moving toward A4-mid.

   `X3.best` is essentially tied with `V18.best` (`-0.0007174310 dB`). `X3.last` is below `V18.last` by `-0.0138577332 dB`, though V18.last had a longer 200k endpoint and includes the original V18 setup.

## Conclusion

Within the pre-registered A3-matched window, strong image auxiliary supervision plus decoder LoRA does not show additive value. The clean statement is:

> Under a 10k warmstart adaptation from V7.best, decoder LoRA with `image_aux=0.08` and `lambda_kl=0` remains below A4-mid and does not recover A4-mid's from-scratch gain.

This supports downgrading V18/LoRA as a headline mechanism in the current paper narrative. It does not prove that a from-scratch `image_aux=0.08 + LoRA` run, or a 20k/longer X3-extended run, can never help; those would be separate experiments and should be named separately.

