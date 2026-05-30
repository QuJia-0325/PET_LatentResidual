# A4-mid seed1337 / X1-lite full-val evaluation report (2026-05-30)

## Scope

- Evaluated two completed 160K-step experiments on the full validation split (`n=7403`, `--max-slices 0`):
  - `A4_mid_seed1337`: A4-mid robustness replicate, `training.image_aux.lambda_start=lambda_max=0.08`, full image auxiliary loss.
  - `X1_lite_l1_only`: mechanism falsification probe, decoder-through-image L1 only, no SSIM/seam contribution in the optimized image auxiliary term.
- Evaluation script: `review/0505/operator/scripts/eval_first_hop_fullval_psnr_chain_mse.py`.
- Eval mode: `--split val --batch-size 8 --decode-mode both`.
- Primary metric: full-val `NORMAL` `PSNR_clip3`; secondary metrics are `NORMAL` chain MSE, tail chain MSE, and average transport PSNR.

## Raw results

| Run | Checkpoint | Step | Full-val slices | NORMAL PSNR_clip3 | Δ vs V7.best | Δ vs A4-mid seed42 | NORMAL chain MSE | Tail chain MSE | Avg transport PSNR | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A4_mid_seed1337 | best | 160000 | 7403 | 36.912771 | +0.131819 | +0.018853 | 0.000239688 | 0.000263427 | 36.191337 | 334.8s |
| A4_mid_seed1337 | last | 160000 | 7403 | 36.912771 | +0.131819 | +0.018853 | 0.000239688 | 0.000263427 | 36.191337 | 339.4s |
| X1_lite_l1_only | best | 160000 | 7403 | 36.733205 | -0.047747 | -0.160713 | 0.000247877 | 0.000270557 | 36.064008 | 330.0s |
| X1_lite_l1_only | last | 160000 | 7403 | 36.733205 | -0.047747 | -0.160713 | 0.000247877 | 0.000270557 | 36.064008 | 333.8s |

Reference baselines used for deltas:

- V7.best: `NORMAL PSNR_clip3 = 36.780951`.
- A4-mid seed42: `NORMAL PSNR_clip3 = 36.893917`.
- V14 seed1337 (`lambda=0.04` seed perturbation): `NORMAL PSNR_clip3 = 36.780632`.
- X3 seed1337: `NORMAL PSNR_clip3 = 36.828787`.

## Training/eval evidence

| Run | Training log | Full-val eval logs | Result artifacts |
|---|---|---|---|
| A4_mid_seed1337 | `review/0525/A4_mid_seed1337/A4_mid_seed1337_train_20260526_152836_gpu3.log` | `review/0525/A4_mid_seed1337/fullval_eval/logs/a4_mid_seed1337_best_eval_20260530_gpu0.log`; `review/0525/A4_mid_seed1337/fullval_eval/logs/a4_mid_seed1337_last_eval_20260530_gpu0.log` | `review/0525/A4_mid_seed1337/fullval_eval/artifacts/a4_mid_seed1337_best_fullval_psnr_chain_mse.json`; `review/0525/A4_mid_seed1337/fullval_eval/artifacts/a4_mid_seed1337_last_fullval_psnr_chain_mse.json`; matching per-slice CSVs |
| X1_lite_l1_only | `review/0525/X1_lite_l1_only/X1_lite_train_20260525_222455_gpu1.log` | `review/0525/X1_lite_l1_only/fullval_eval/logs/x1_lite_l1_only_best_eval_20260530_gpu1.log`; `review/0525/X1_lite_l1_only/fullval_eval/logs/x1_lite_l1_only_last_eval_20260530_gpu1.log` | `review/0525/X1_lite_l1_only/fullval_eval/artifacts/x1_lite_l1_only_best_fullval_psnr_chain_mse.json`; `review/0525/X1_lite_l1_only/fullval_eval/artifacts/x1_lite_l1_only_last_fullval_psnr_chain_mse.json`; matching per-slice CSVs |

Both `best.pt` and `last.pt` resolve to the same selected 160K checkpoint for both experiments, so best/last full-val metrics are identical. The training logs also show the final full-val selection scores at 160K:

- A4_mid_seed1337: `val_select_score=0.000887`, `val_chain_normal_mse=0.000240`, `val_chain_tail_mse=0.000263`.
- X1_lite_l1_only: `val_select_score=0.000911`, `val_chain_normal_mse=0.000248`, `val_chain_tail_mse=0.000271`.

## Key findings

1. **A4-mid is robust under the seed1337 replicate.** A4_mid_seed1337 reaches `36.912771`, which is `+0.131819 dB` over V7.best and `+0.018853 dB` over the original A4-mid seed42 result. This passes the pre-declared robustness interpretation: the replicate is within `0.02 dB` of seed42 and is slightly better rather than regressing.
2. **The doubled full image_aux setting remains the paper-headline candidate.** A4_mid_seed1337 also exceeds the V14 seed1337 lambda=0.04 seed perturbation by `+0.132139 dB`, so the gain is not explained by changing the seed alone.
3. **X1-lite falsifies the “L1-only explains A4” hypothesis.** X1_lite_l1_only reaches only `36.733205`, which is `-0.047747 dB` below V7.best, `-0.160713 dB` below A4-mid seed42, and `-0.179566 dB` below A4_mid_seed1337. Decoder-space L1 alone is therefore insufficient; the full image_aux composition and/or its interaction with SSIM/seam terms matters.
4. **A4 improves both PSNR and chain MSE relative to X1-lite.** A4_mid_seed1337 lowers `NORMAL` chain MSE from `0.000247877` to `0.000239688` and tail chain MSE from `0.000270557` to `0.000263427`, matching the PSNR ordering.

## Interpretation

- The current architecture/design story is now stronger than before: the useful lever is not merely “add a pixel-space L1 path through the frozen decoder.” The successful condition is the full A4-mid image auxiliary objective at `lambda=0.08`, while the X1-lite ablation shows that optimizing only the L1 component can underperform the V7 baseline.
- The seed1337 A4 replicate closes the main robustness concern for a paper-scale claim at the current evidence level. This is still one replicate, not a seed sweep, so avoid wording like “statistically seed-invariant”; the defensible wording is that A4-mid reproduces under the approved seed perturbation and improves over the observed V7/V14 seed floor.

## Claim gate

- Local verdict: `partial-to-yes`, scope-limited.
- Supported: A4-mid is reproducible under the approved seed1337 perturbation on this validation protocol, and X1-lite does not recover the A4 gain.
- Not supported: broad cross-dataset/general PET generalization, precise seed variance, or isolated attribution between SSIM and seam individually.
- External Codex result-to-claim review: pending; the MCP call timed out during this archival run, so this report uses the conservative local gate above.

## Suggested next steps

1. Use A4-mid seed42 plus A4_mid_seed1337 as the main result pair in the paper narrative.
2. Keep X1-lite as a mechanism ablation showing L1-only is not sufficient.
3. Do not launch additional A4 lambda or seed sweeps unless a reviewer specifically demands broader uncertainty quantification.
