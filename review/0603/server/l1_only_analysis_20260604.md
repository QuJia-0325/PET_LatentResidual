# L1-only 对照三联分析

日期: 2026-06-04

## 结论

- NORMAL PSNR: V13=36.494330, L1-only=36.715786, A4-mid=36.893917。
- L1-only vs V13: ΔPSNR=0.221455 dB, 95% block CI=[0.210945, 0.231976], win-rate=0.8552。
- A4-mid vs V13: ΔPSNR=0.399587 dB, 95% block CI=[0.386968, 0.413370], win-rate=0.9357。
- A4-mid vs L1-only: ΔPSNR=0.178132 dB, 95% block CI=[0.168087, 0.188858], win-rate=0.8179。
- L1-only 复现 A4 PSNR 增益比例: 55.42%。L1-only 只复现了 A4 的一小部分 NORMAL PSNR 增益；复合结构仍有必要，但需要报告拆解数字。

## NORMAL 关键指标

| metric | V13 | L1-only | A4-mid | L1-vs-V13 | A4-vs-V13 | A4-vs-L1 |
|---|---:|---:|---:|---:|---:|---:|
| psnr | 36.494330 | 36.715786 | 36.893917 | 0.221455 | 0.399587 | 0.178132 |
| ms_ssim | 0.982886 | 0.983754 | 0.984347 | 0.000868 | 0.001461 | 0.000593 |
| seam | 0.021355 | 0.019526 | 0.016552 | -0.001829 | -0.004803 | -0.002974 |
| ext_seam | 0.010044 | 0.009512 | 0.008924 | -0.000532 | -0.001120 | -0.000588 |

## 口径

- PSNR 使用 `src.utils.metrics.calc_psnr_clip3`；SSIM/MS-SSIM/NRMSE/RMSE/MAE/seam/ext-seam 使用 clip3 SUV [0,3] 域。
- 置信区间为按 slice 顺序的 64-slice block bootstrap；这是 slice-level 稳健性检查，不等价于 patient-level 显著性。
- D50 是同一 D50 latent passthrough 解码，模型间应当相同；归因主要看 D20/D10/D4/NORMAL。

## 原始文件

- V13_aux_off: `review/0601/server/bext_v13_first_hop_224_val_clip3_eval.csv`
- L1_only: `review/0603/server/c_l1_only_first_hop_224_val_clip3_eval.csv`
- A4_mid: `review/0601/server/bext_a4_first_hop_224_val_clip3_eval.csv`
