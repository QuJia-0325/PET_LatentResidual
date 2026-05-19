# V18 KL Drift Probe Report

## Protocol
- Evaluator: `tools/probe_v18_kl_drift.py`
- Metric: `src.utils.metrics.calc_psnr_clip3`
- Split: `val`, n=7403
- Computation: direct `decode_crop(z_GT)` only; transport rollout is skipped.
- Runtime: 1907.6s

## Results

| ckpt | step | D20 | D10 | D4 | NORMAL |
|---|---:|---:|---:|---:|---:|
| V7.best | 160000 | 46.6356 | 48.7436 | 50.8319 | 52.6341 |
| V18.best | 165000 | 46.6850 | 48.7993 | 50.9012 | 52.7314 |
| V18.step170k | 170000 | 46.7212 | 48.8408 | 50.9527 | 52.8027 |
| V18.last | 200000 | 46.7482 | 48.8630 | 50.9635 | 52.7980 |
| V18-cap.last | 170000 | 46.7221 | 48.8420 | 50.9542 | 52.8047 |

## KL Drift: V7.best - Checkpoint

Positive values mean the compared checkpoint underperforms V7 on GT latents.

| comparison | D20 | D10 | D4 | NORMAL |
|---|---:|---:|---:|---:|
| V7.best - V18.best | -0.0493 | -0.0557 | -0.0692 | -0.0973 |
| V7.best - V18.step170k | -0.0855 | -0.0972 | -0.1207 | -0.1686 |
| V7.best - V18.last | -0.1126 | -0.1194 | -0.1315 | -0.1639 |
| V7.best - V18-cap.last | -0.0865 | -0.0984 | -0.1223 | -0.1707 |

## A3 Capacity-Only Matched-Step Delta

Primary delta is `V18-cap.last(170K) - V18.step170k` on direct `decode(z_GT)` PSNR_clip3.

| comparison | D20 | D10 | D4 | NORMAL |
|---|---:|---:|---:|---:|
| V18-cap.last - V18.step170k | +0.0009 | +0.0012 | +0.0015 | +0.0020 |

- A3 primary verdict: capacity-only ~= V18.step170k on NORMAL; LoRA capacity is sufficient for the GT-manifold gain.

## Verdict

- V18.best NORMAL KL drift: `-0.0973 dB` -> `MODERATE` by abs drift
- V18.last NORMAL KL drift: `-0.1639 dB` -> `MODERATE` by abs drift

## Interpretation

- All signed drifts are negative: `decode_V18(z_GT)` scores higher PSNR than `decode_V7(z_GT)` on every reported timepoint. The magnitude is non-negligible, but the direction is an improvement relative to V7 on the GT latent manifold, not a harmful degradation.
- This report does not launch or recommend a new training run by itself; it is Stage C input only.

## Artifacts

- `KL_DRIFT_SUMMARY.json`
- `KL_DRIFT_PER_SLICE.csv`
- `probe.log`
