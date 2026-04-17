# 0416 Experiment Conclusion (Full-Set Evaluation)

## Scope
- Repository: `PET_LatentResidual`
- Evaluation mode: full-set (`--max-slices 0`, no subsampling)
- Effective split: `val` (7403 slices)
- Note: `latents_test.pt` is not present in current data directory.

## Run Settings
- Checkpoint: `/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_formal_v3_chainstable/best.pt`
- Config: `configs/pet_flow/pet_flow_first_hop_224_50k_formal_v3_chainstable.yaml`
- E1 device: `cuda:3`
- E2 device: `cuda:1` (another GPU)

## Commands
```bash
# E1 (full-set)
PYTHONPATH=/home/qujiaxiang/project/PET_LatentResidual \
/home/qujiaxiang/.conda/envs/rae/bin/python scripts/diagnose_error_budget.py \
  --config configs/pet_flow/pet_flow_first_hop_224_50k_formal_v3_chainstable.yaml \
  --checkpoint /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_formal_v3_chainstable/best.pt \
  --split val --max-slices 0 --batch-size 8 --device cuda:3

# E2 (full-set, another GPU)
PYTHONPATH=/home/qujiaxiang/project/PET_LatentResidual \
/home/qujiaxiang/.conda/envs/rae/bin/python scripts/diagnose_foc_gap.py \
  --config configs/pet_flow/pet_flow_first_hop_224_50k_formal_v3_chainstable.yaml \
  --checkpoint /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_formal_v3_chainstable/best.pt \
  --split val --max-slices 0 --batch-size 8 --device cuda:1 \
  --out-dir /data_2/qujiaxiang/outputs/PET_LatentResidual/diagnostics/foc_gap_fullval_gpu1
```

## E1 Raw Data (error_budget_val.json)
`n_eval = 7403`

| TP | Ceiling PSNR mean | E2E PSNR mean | Gap_Total (dB) | Gap_Transport (dB) | Gap_Decoder (dB) | transport_fraction |
|---|---:|---:|---:|---:|---:|---:|
| D50 | 42.6224 | 42.6224 | 0.0000 | -inf | inf | 0.0000 |
| D20 | 46.6356 | 35.4881 | 11.1475 | 10.7388 | 0.4087 | 0.9633 |
| D10 | 48.7436 | 35.8620 | 12.8815 | 12.3332 | 0.5483 | 0.9574 |
| D4 | 50.8319 | 36.4207 | 14.4112 | 13.9784 | 0.4328 | 0.9700 |
| NORMAL | 52.6341 | 36.7422 | 15.8918 | 15.6973 | 0.1946 | 0.9878 |

Gate (D20): `transport_fraction = 0.9633` -> **PASS** (>= 0.30)

## E2 Raw Data (foc_gap_val.json)
`n_eval = 7403`

### Summary Scalars
- `half_minus_full_psnr_mean = -0.0445 dB`
- `gap_vs_deficit_correlation = 0.4362`

### Distribution Stats
| Metric | n | mean | std | min | max |
|---|---:|---:|---:|---:|---:|
| gap_abs | 7403 | 0.001239 | 0.000634 | 0.000057 | 0.006039 |
| gap_rel | 7403 | 0.001709 | 0.000868 | 0.000080 | 0.008407 |
| z_full_norm | 7403 | 0.723257 | 0.004808 | 0.714253 | 0.735407 |
| z_half_norm | 7403 | 0.723207 | 0.004792 | 0.713937 | 0.735302 |
| psnr_full_step | 7403 | 35.488083 | 8.699154 | 22.912809 | 74.849111 |
| psnr_half_step | 7403 | 35.443569 | 8.727179 | 22.867774 | 74.821651 |
| psnr_decoder_ceiling_d20 | 7403 | 46.635633 | 6.388339 | 34.371283 | 72.068229 |

Gate (relative gap): `0.1709%` -> **FAIL** (< 5%)

## Final Conclusion
- E1 indicates transport-side dominance in error decomposition.
- E2 full-set gate fails clearly (`0.1709% << 5%`).
- Under current checkpoint/data, the FOC-lite core hypothesis is not sufficiently supported for priority training investment.

## Artifact Paths
- E1 JSON: `/data_2/qujiaxiang/outputs/PET_LatentResidual/diagnostics/error_budget/error_budget_val.json`
- E2 JSON: `/data_2/qujiaxiang/outputs/PET_LatentResidual/diagnostics/foc_gap_fullval_gpu1/foc_gap_val.json`
