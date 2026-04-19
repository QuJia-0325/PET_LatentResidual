# A+C Scheme Training Analysis (2026-04-18)

## Run Identity
- Run folder: /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_irepa_align_gpu3_nopbar_20260418_010035
- Metrics file: /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_irepa_align_gpu3_nopbar_20260418_010035/metrics.jsonl
- Raw log file: /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_irepa_align_gpu3_nopbar_20260418_010035.log
- Completion: reached step=50000 and ended with Training done.

## Key Numbers
- Latest train step: 50000
- Latest val step: 50000
- Best val_select_score: 0.000519607906 at step 46400
- Final val_select_score: 0.000671574379 at step 50000
- Improvement from first val to best: 24.26%
- Final relative to best: +29.25%

## Best vs Final (val)

| Item | Best (step 46400) | Final (step 50000) |
|---|---:|---:|
| val_select_score | 0.000519608 | 0.000671574 |
| val_rollout_total | 0.000616317 | 0.000596251 |
| val_hop0_img_total | 0.005290050 | 0.010125812 |
| val_chain_normal_mse | 0.000165654 | 0.000213765 |

## Recent Validation Trend (last 6 evals)

| step | val_select_score | val_rollout_total | val_hop0_img_total | val_chain_normal_mse |
|---:|---:|---:|---:|---:|
| 48000 | 0.000704613 | 0.000883671 | 0.006516326 | 0.000221014 |
| 48400 | 0.000612629 | 0.000850972 | 0.007005272 | 0.000195428 |
| 48800 | 0.001260219 | 0.001326489 | 0.009646507 | 0.000378579 |
| 49200 | 0.001182404 | 0.001153384 | 0.009294428 | 0.000358867 |
| 49600 | 0.000937774 | 0.000787309 | 0.009828796 | 0.000299777 |
| 50000 | 0.000671574 | 0.000596251 | 0.010125812 | 0.000213765 |

## Interpretation
- This run completed full training and produced best.pt, step_050000.pt, and last.pt.
- The best validation objective was reached at step 46400, while the final step is worse than best.
- Recommendation: use best.pt for downstream evaluation/comparison, not last.pt.

## Artifacts
- Best checkpoint: /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_irepa_align_gpu3_nopbar_20260418_010035/best.pt
- Final checkpoint: /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_irepa_align_gpu3_nopbar_20260418_010035/last.pt
- Raw log copied to this folder separately for archival.
