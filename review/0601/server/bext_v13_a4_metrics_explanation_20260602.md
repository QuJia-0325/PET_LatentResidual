# B-ext V13 vs A4 Full-val Metrics Explanation

日期: 2026-06-02  
范围: Task B-ext 已完成的两个 full-val eval: V13 image_aux-off 与 A4-mid composite image_aux。L1-only Task C 仍在训练中，未并入本文件。

## 输入与输出

- V13 原始 JSON: `review/0601/server/bext_v13_first_hop_224_val_clip3_eval.json`
- V13 原始 CSV: `review/0601/server/bext_v13_first_hop_224_val_clip3_eval.csv`
- A4 原始 JSON: `review/0601/server/bext_a4_first_hop_224_val_clip3_eval.json`
- A4 原始 CSV: `review/0601/server/bext_a4_first_hop_224_val_clip3_eval.csv`
- 指标总表: `review/0601/server/metrics_summary.csv`
- seam 分层表: `review/0601/server/seam_strata/seam_strata_summary.csv`
- V13 eval log: `review/0601/server/bext_v13_eval_gpu1_20260601.log`
- A4 eval log: `review/0601/server/bext_a4_eval_gpu2_20260601.log`

两个 eval 均为 `split=val`, `n=7403`, `decode_mode=default`。PSNR 仍使用 `src.utils.metrics.calc_psnr_clip3`。新增指标 SSIM / MS-SSIM / NRMSE / RMSE / MAE / seam / ext-seam 均在 clip3 SUV `[0,3]` 域计算或记录。

## NORMAL headline

| Metric | Role | Direction | V13 | A4 | Delta A4-V13 | 95% block CI | Win-rate |
|---|---|---|---:|---:|---:|---:|---:|
| PSNR | independent | up | 36.494330 | 36.893917 | 0.399587 | [0.386409, 0.413914] | 0.9357 |
| MS-SSIM | independent | up | 0.982886 | 0.984347 | 0.001461 | [0.001382, 0.001540] | 0.9673 |
| NRMSE | independent | down | 0.019025 | 0.018210 | -0.000815 | [-0.000843, -0.000784] | 0.9357 |
| RMSE | independent | down | 0.057074 | 0.054629 | -0.002444 | [-0.002529, -0.002353] | 0.9357 |
| SSIM | training-aligned | up | 0.958004 | 0.962343 | 0.004340 | [0.004190, 0.004492] | 0.9893 |
| MAE | training-aligned | down | 0.014633 | 0.013822 | -0.000811 | [-0.000843, -0.000778] | 0.9846 |
| seam | training-aligned | down | 0.021355 | 0.016552 | -0.004803 | [-0.004999, -0.004612] | 0.9964 |
| ext-seam | supporting | down | 0.010044 | 0.008924 | -0.001120 | [-0.001166, -0.001075] | 0.9954 |

## Seam-risk 分层 NORMAL

分层器固定为 V13 aux-off 的 `seam_NORMAL` 三分位，避免使用 A4 结果做选择器。

| Stratum | n | Threshold | Delta PSNR | PSNR CI | PSNR win | Delta seam % | Delta ext-seam % | Delta NRMSE |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| low | 2468 | <= 0.01163887 | 0.428830 | [0.399491, 0.457924] | 0.9044 | -18.26% | -10.02% | -0.000375 |
| mid | 2467 | (0.01163887, 0.02827142] | 0.356153 | [0.338705, 0.372387] | 0.9311 | -21.13% | -10.23% | -0.000679 |
| high | 2468 | > 0.02827142 | 0.413760 | [0.398032, 0.429802] | 0.9716 | -24.07% | -11.86% | -0.001390 |

## 结论

1. 独立指标支持 A4-mid 优于 V13: NORMAL PSNR 提升 0.3996 dB，NRMSE/RMSE 下降，MS-SSIM 上升；这些不是训练 loss 的直接项，可作为 headline 优越性证据。
2. training-aligned 指标作为一致性佐证: SSIM/MAE/seam 方向均与 independent 指标一致，但论文中不应把它们作为唯一主证据。
3. seam 分层显示 high seam-risk 桶仍有稳定收益；这支持 image_aux 抑制 residual seam artifact，而不是只靠 cherry-picking 个别切片。
4. 这些结果仍只覆盖 V13 vs A4。L1-only Task C 完成后，需要把 L1-only 追加到 `metrics_summary.csv` 和分层表，才能判断 intensity-only 经验 claim 是否成立。

## 边界

- 不包含 lesion ROI / SUV recovery / CNR / reader-study，因此不能声称病灶级或诊断级优越。
- 当前 claim 应限定为 full-reference fidelity 与 residual seam artifact suppression。
