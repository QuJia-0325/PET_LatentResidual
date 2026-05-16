# Plan-F three-run full-val PSNR_clip3 evaluation

Generated: 2026-05-16 17:39 CST

## Protocol

- Evaluated three completed Plan-F runs: V7, V8_no_image_aux, and V6_NOISE_seed1337.
- Evaluated both `best.pt` and `last.pt` for each run, six checkpoints total.
- Split: full validation set, `max_slices=0`, `num_eval_slices=7403`.
- Metric: `src.utils.metrics.calc_psnr_clip3`, recorded in every JSON under `meta.psnr_metric`.
- Decode mode: `both`; headline table uses default decoded chain outputs without the `_raw` suffix.
- Launch GPU: GPU1, batch size 8, Python `/home/qujiaxiang/.conda/envs/rae/bin/python`.

## Headline results

| Experiment | Ckpt | Step | D20 PSNR | D10 PSNR | D4 PSNR | NORMAL PSNR | Transport avg PSNR | Tail chain MSE | NORMAL chain MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V7 | best | 160000 | 35.4354 | 35.8194 | 36.3736 | 36.7810 | 36.1023 | 0.000268340 | 0.000245641 |
| V7 | last | 160000 | 35.4354 | 35.8194 | 36.3736 | 36.7810 | 36.1023 | 0.000268340 | 0.000245641 |
| V6_NOISE_seed1337 | best | 156400 | 35.4258 | 35.8148 | 36.3575 | 36.7437 | 36.0855 | 0.000269184 | 0.000247104 |
| V6_NOISE_seed1337 | last | 160000 | 35.4217 | 35.8110 | 36.3596 | 36.7550 | 36.0868 | 0.000268958 | 0.000246547 |
| V8_no_image_aux | best | 160000 | 35.2094 | 35.5897 | 36.0963 | 36.4729 | 35.8421 | 0.000283550 | 0.000261979 |
| V8_no_image_aux | last | 160000 | 35.2094 | 35.5897 | 36.0963 | 36.4729 | 35.8421 | 0.000283550 | 0.000261979 |

## Interpretation

- V7 last: NORMAL PSNR delta vs V7 last = +0.0000 dB; tail chain MSE delta = +0.000000000.
- V6_NOISE_seed1337 last: NORMAL PSNR delta vs V7 last = -0.0259 dB; tail chain MSE delta = +0.000000618.
- V8_no_image_aux last: NORMAL PSNR delta vs V7 last = -0.3080 dB; tail chain MSE delta = +0.000015209.
- V7 best and V7 last are numerically identical in this eval; both checkpoints report step 160000, so the final checkpoint is also the selected best checkpoint for this run.
- V8 best and V8 last are also identical at step 160000, but are clearly worse than V7: NORMAL PSNR is lower by 0.3081 dB and tail chain MSE is higher by 0.000015210.
- V6_NOISE best is step 156400 and last is step 160000. Last slightly improves NORMAL PSNR over best (+0.0113 dB) but remains below V7 last by 0.0260 dB on NORMAL PSNR.
- The full-val result supports keeping image_aux/regular V7-style supervision: removing image_aux in V8 degrades all rollout PSNR stages and increases chain MSE.

## Files

- `artifacts/*_fullval_psnr_chain_mse.json`: per-checkpoint aggregate JSON with metric metadata.
- `artifacts/*_fullval_psnr_chain_mse_per_slice.csv`: per-slice PSNR/MSE rows for all 7403 validation slices.
- `status/planf_fullval_psnr_clip3_summary_20260516_173941.csv`: compact comparison table.
- `status/planf_fullval_psnr_clip3_summary_20260516_173941.json`: compact comparison JSON.
- `logs/planf_v7_v8_v6noise_best_last_fullval_psnr_clip3_gpu1_20260516_165948.log`: complete eval log, includes `[all_done] 2026-05-16 17:39:41 CST`.
- `scripts/run_planf_fullval_psnr_clip3_20260516.sh`: exact launch script used for this evaluation.

## Checkpoint paths

- planf_v7_best: `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V7/run/first_hop_224_v7_gronwall_raw/best.pt`
- planf_v7_last: `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V7/run/first_hop_224_v7_gronwall_raw/last.pt`
- planf_v6noise_best: `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V6_NOISE/run/first_hop_224_v6_seed1337/best.pt`
- planf_v6noise_last: `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V6_NOISE/run/first_hop_224_v6_seed1337/last.pt`
- planf_v8_best: `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V8/run/first_hop_224_v8_no_image_aux/best.pt`
- planf_v8_last: `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V8/run/first_hop_224_v8_no_image_aux/last.pt`

