# V18 KL Drift Probe Report

## Protocol
- Evaluator: `tools/probe_v18_kl_drift.py`
- Metric: `src.utils.metrics.calc_psnr_clip3`
- Split: `val`, n=7403
- Computation: direct `decode_crop(z_GT)` only; transport rollout is skipped.
- Runtime: 271.6s

## Results

| ckpt | step | D20 | D10 | D4 | NORMAL |
|---|---:|---:|---:|---:|---:|
| V7.best | 160000 | 46.6356 | 48.7436 | 50.8319 | 52.6341 |
| V18.best | 165000 | 46.6850 | 48.7993 | 50.9012 | 52.7314 |
| V18.last | 200000 | 46.7482 | 48.8630 | 50.9635 | 52.7980 |

## KL Drift: V7.best - V18

Positive values mean V18 decoder underperforms V7 decoder on GT latents.

| comparison | D20 | D10 | D4 | NORMAL |
|---|---:|---:|---:|---:|
| V7.best - V18.best | -0.0493 | -0.0557 | -0.0692 | -0.0973 |
| V7.best - V18.last | -0.1126 | -0.1194 | -0.1315 | -0.1639 |

## Verdict

- V18.best NORMAL KL drift: `-0.0973 dB` -> `MODERATE` by abs drift
- V18.last NORMAL KL drift: `-0.1639 dB` -> `MODERATE` by abs drift

## Interpretation

- All signed drifts are negative: `decode_V18(z_GT)` scores higher PSNR than `decode_V7(z_GT)` on every reported timepoint.
- The magnitude is non-negligible, but the direction is an improvement on the GT latent manifold, not a harmful degradation. This weakens the hypothesis that V18 failed because decoder LoRA damaged GT-manifold decoding.
- This report does not launch or recommend a new training run by itself; it is Stage C input only.

## Artifacts

- `KL_DRIFT_SUMMARY.json`
- `KL_DRIFT_PER_SLICE.csv`
- `probe.log`