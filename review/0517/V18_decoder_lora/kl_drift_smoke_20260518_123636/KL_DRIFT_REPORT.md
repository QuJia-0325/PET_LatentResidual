# V18 KL Drift Probe Report

## Protocol
- Evaluator: `tools/probe_v18_kl_drift.py`
- Metric: `src.utils.metrics.calc_psnr_clip3`
- Split: `val`, n=8
- Computation: direct `decode_crop(z_GT)` only; transport rollout is skipped.
- Runtime: 45.0s

## Results

| ckpt | step | D20 | D10 | D4 | NORMAL |
|---|---:|---:|---:|---:|---:|
| V7.best | 160000 | 47.7219 | 50.0119 | 52.1767 | 53.7392 |
| V18.best | 165000 | 47.7412 | 50.0458 | 52.2300 | 53.8118 |
| V18.last | 200000 | 47.7521 | 50.0661 | 52.2554 | 53.8125 |

## KL Drift: V7.best - V18

Positive values mean V18 decoder underperforms V7 decoder on GT latents.

| comparison | D20 | D10 | D4 | NORMAL |
|---|---:|---:|---:|---:|
| V7.best - V18.best | -0.0192 | -0.0339 | -0.0534 | -0.0726 |
| V7.best - V18.last | -0.0301 | -0.0543 | -0.0788 | -0.0733 |

## Verdict

- V18.best NORMAL KL drift: `-0.0726 dB` -> `MODERATE` by abs drift
- V18.last NORMAL KL drift: `-0.0733 dB` -> `MODERATE` by abs drift

## Interpretation

- KL drift is non-negligible for at least one V18 checkpoint. This supports the concern that decoder LoRA moved away from the GT latent manifold and should be considered in Stage C.
- This report does not launch or recommend a new training run by itself; it is Stage C input only.

## Artifacts

- `KL_DRIFT_SUMMARY.json`
- `KL_DRIFT_PER_SLICE.csv`
- `probe.log`
