# ROI/high-SUV PSNR disambiguation report (2026-05-17)

## Scope

- This is §2 of `review/0516/CODEX_DISAMBIGUATION_RUNBOOK_20260516.md`.
- Protocol: full-val `n=7403`, open-loop chain rollout, `decode-mode=both`, primary rows below use default decode.
- Metrics include canonical `PSNR_clip3`, unclipped SUV-domain PSNR, top-10%/5%/1% GT-SUV masked PSNR, high-gradient masked PSNR, SUVmax error, and top-1% SUVmean error.
- `suvmax_err` and `suvmean_err_*` are signed prediction minus GT; negative means underestimation.

## Main ROI table

| tag | stage | PSNR_clip3 | unclipped PSNR | top10 SUV PSNR | top5 SUV PSNR | top1 SUV PSNR | high-grad PSNR | SUVmax err | top1 SUVmean err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V7_last | D20 | 35.4354 | 32.6990 | 23.2735 | 20.7746 | 16.3675 | 23.2572 | -0.8940 | -0.3038 |
| V7_last | D10 | 35.8194 | 32.5523 | 23.1097 | 20.6909 | 16.4228 | 23.1341 | -0.7543 | -0.2685 |
| V7_last | D4 | 36.3736 | 32.9319 | 23.5128 | 21.0843 | 17.0236 | 23.5299 | -0.5681 | -0.2202 |
| V7_last | NORMAL | 36.7810 | 32.8330 | 23.6121 | 21.0868 | 17.2106 | 23.5516 | -0.4254 | -0.1898 |
| V8_last | D20 | 35.2094 | 32.4882 | 23.1111 | 20.6705 | 16.3849 | 23.0911 | -0.8378 | -0.2872 |
| V8_last | D10 | 35.5897 | 32.3292 | 22.9221 | 20.5384 | 16.3662 | 22.9421 | -0.6903 | -0.2627 |
| V8_last | D4 | 36.0963 | 32.6638 | 23.2743 | 20.8691 | 16.8609 | 23.2844 | -0.5342 | -0.2255 |
| V8_last | NORMAL | 36.4729 | 32.5314 | 23.3421 | 20.8423 | 17.0280 | 23.2765 | -0.3991 | -0.1955 |
| V6_NOISE_last | D20 | 35.4217 | 32.6881 | 23.2602 | 20.7631 | 16.3573 | 23.2442 | -0.8973 | -0.3042 |
| V6_NOISE_last | D10 | 35.8110 | 32.5472 | 23.0998 | 20.6905 | 16.4817 | 23.1319 | -0.7259 | -0.2612 |
| V6_NOISE_last | D4 | 36.3596 | 32.9199 | 23.4971 | 21.0712 | 17.0182 | 23.5175 | -0.5723 | -0.2201 |
| V6_NOISE_last | NORMAL | 36.7550 | 32.7992 | 23.5820 | 21.0511 | 17.1639 | 23.5203 | -0.4373 | -0.1922 |

## V7 - V8 paired deltas

| stage | Δ clip3 | Δ unclipped | Δ top10 | Δ top5 | Δ top1 | Δ high-grad | Δ SUVmax err | Δ top1 SUVmean err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D20 | +0.2260 | +0.2108 | +0.1624 | +0.1041 | -0.0174 | +0.1661 | -0.0562 | -0.0166 |
| D10 | +0.2297 | +0.2230 | +0.1877 | +0.1525 | +0.0566 | +0.1920 | -0.0640 | -0.0058 |
| D4 | +0.2773 | +0.2681 | +0.2385 | +0.2152 | +0.1628 | +0.2455 | -0.0339 | +0.0053 |
| NORMAL | +0.3080 | +0.3016 | +0.2700 | +0.2445 | +0.1827 | +0.2751 | -0.0263 | +0.0057 |

## Paired significance for V7 - V8

| stage | metric | mean delta | paired t | win % |
|---|---|---:|---:|---:|
| D20 | psnr_clip3 | +0.2260 | 43.10 | 85.8% |
| D20 | psnr_top5suv | +0.1041 | 19.57 | 56.1% |
| D20 | psnr_top1suv | -0.0174 | -2.85 | 35.5% |
| D20 | psnr_highgrad | +0.1661 | 35.16 | 74.8% |
| D20 | suvmax_err | -0.0562 | -20.61 | 34.3% |
| D10 | psnr_clip3 | +0.2297 | 72.39 | 90.0% |
| D10 | psnr_top5suv | +0.1525 | 47.57 | 75.9% |
| D10 | psnr_top1suv | +0.0566 | 13.28 | 54.2% |
| D10 | psnr_highgrad | +0.1920 | 60.84 | 85.0% |
| D10 | suvmax_err | -0.0640 | -22.70 | 30.7% |
| D4 | psnr_clip3 | +0.2773 | 79.38 | 91.8% |
| D4 | psnr_top5suv | +0.2152 | 57.76 | 81.9% |
| D4 | psnr_top1suv | +0.1628 | 34.31 | 67.2% |
| D4 | psnr_highgrad | +0.2455 | 67.63 | 87.7% |
| D4 | suvmax_err | -0.0339 | -11.79 | 37.5% |
| NORMAL | psnr_clip3 | +0.3080 | 74.63 | 91.5% |
| NORMAL | psnr_top5suv | +0.2445 | 51.42 | 82.0% |
| NORMAL | psnr_top1suv | +0.1827 | 30.80 | 67.7% |
| NORMAL | psnr_highgrad | +0.2751 | 61.11 | 88.0% |
| NORMAL | suvmax_err | -0.0263 | -9.11 | 42.3% |

## V7 - V6_NOISE reference

| stage | metric | mean delta | paired t | win % |
|---|---|---:|---:|---:|
| D20 | psnr_clip3 | +0.0137 | 6.61 | 54.6% |
| D20 | psnr_top5suv | +0.0115 | 5.44 | 53.7% |
| D20 | psnr_top1suv | +0.0101 | 3.63 | 52.1% |
| D20 | psnr_highgrad | +0.0130 | 5.91 | 54.5% |
| D10 | psnr_clip3 | +0.0083 | 3.59 | 53.5% |
| D10 | psnr_top5suv | +0.0005 | 0.17 | 49.8% |
| D10 | psnr_top1suv | -0.0589 | -17.19 | 39.7% |
| D10 | psnr_highgrad | +0.0022 | 0.88 | 51.0% |
| D4 | psnr_clip3 | +0.0140 | 4.91 | 54.0% |
| D4 | psnr_top5suv | +0.0131 | 4.11 | 51.7% |
| D4 | psnr_top1suv | +0.0054 | 1.30 | 50.1% |
| D4 | psnr_highgrad | +0.0124 | 4.09 | 51.6% |
| NORMAL | psnr_clip3 | +0.0259 | 7.77 | 55.8% |
| NORMAL | psnr_top5suv | +0.0358 | 9.04 | 55.4% |
| NORMAL | psnr_top1suv | +0.0467 | 9.22 | 54.9% |
| NORMAL | psnr_highgrad | +0.0313 | 8.47 | 56.0% |

## Runbook decision matrix

| Observation | Result | Decision impact |
|---|---|---|
| top-1%/5% SUV PSNR much lower than whole-image PSNR | Yes. NORMAL clip3 is 36.78 dB for V7, but NORMAL top5/top1 SUV are 21.09/17.21 dB. The same gap exists for all stages. | Complex/high-uptake regions remain the dominant failure mode; future image losses should consider ROI/high-SUV weighting rather than only global pixel loss. |
| top-1% SUV PSNR(V7) approximately equals PSNR(V8) | Mixed. D20 top1 delta is slightly negative (-0.017 dB), but D10/D4/NORMAL are positive (+0.057/+0.163/+0.183 dB). | Hop0 image_aux does not clearly improve the hardest top1% D20 pixels directly; later top1 gains likely come from chain-level smoothing/coherence. Pure λ-hop0 sweep is not guaranteed to solve ROI artifacts. |
| top-5/top10/high-gradient PSNR(V7) > PSNR(V8) | Yes. NORMAL deltas: top10 +0.270, top5 +0.245, high-gradient +0.275 dB. | V7-style change improves broad ROI/high-gradient quality, but not enough to close the severe absolute ROI gap. |
| SUVmax error in NORMAL worse than D20 | No. V7 SUVmax error improves from D20 -0.894 to NORMAL -0.425; V8 also improves from -0.838 to -0.399. | This does not support multi-hop image_aux from a "NORMAL clinical signal loss is worse" argument. |
| unclipped PSNR trend differs from clip3 PSNR | No major contradiction. V7-V8 deltas are positive in both clip3 and unclipped PSNR, with similar stage ordering. | `clip3` is not reversing the conclusion, but ROI metrics expose much worse absolute behavior in high-SUV regions. |

## Interpretation

- The ROI panel agrees with §1 that the main V7-V8 benefit should not be read as direct multi-hop image_aux evidence. The most difficult D20 top1% SUV region does not improve under V7; if anything, the signed SUV underestimation is slightly worse at D20.
- V7 does improve top5/top10/high-gradient regions and later-stage top1 PSNR, so the effect is real but broad rather than a decisive fix for hottest lesions.
- Absolute ROI PSNR is far below whole-image clip3 PSNR. For NORMAL, V7 has 36.781 dB whole-image clip3 but only 17.211 dB top1 SUV PSNR. This should be treated as an artifact/clinical-risk diagnostic, not as a pass.
- SUVmax errors are negative across all stages and models, meaning systematic underestimation of maxima. The underestimation is largest at D20 and decreases toward NORMAL, so the bottleneck remains early/high-noise input rather than the NORMAL endpoint itself.

## Recommendation

- Do not use ROI results to justify multi-hop/NORMAL-only image_aux as the next primary move.
- If launching a training experiment next, prefer a true single-variable hop0/control experiment or an ROI-weighted loss targeting D20/top-SUV regions. This is more aligned with §1 and ROI evidence than β_NORMAL-only or full multi-hop image_aux.
- Any future claim should report `PSNR_clip3` plus at least top5/top1 SUV PSNR and SUVmax error; whole-image clip3 alone hides the hardest-region failure mode.

## Artifacts

- Tool: `tools/eval_roi_psnr.py`
- Script: `review/0517/disambig/scripts/run_roi_v7_v8_v6noise.sh`
- Log: `review/0517/disambig/logs/roi_v7_v8_v6noise_gpu1_20260517_011502.log`
- Outputs: `review/0517/disambig/roi_psnr/{V7_last,V8_last,V6_NOISE_last}/`
