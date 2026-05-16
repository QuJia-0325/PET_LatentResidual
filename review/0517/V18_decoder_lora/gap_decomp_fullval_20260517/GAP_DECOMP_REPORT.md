# V18 Gap Decomposition Report

- generated_at: 2026-05-17 04:29:29 CST
- config: review/0511/log_snapshots_20260516_163900/configs/V7_config.resolved.yaml
- checkpoint: /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V7/run/first_hop_224_v7_gronwall_raw/best.pt
- n_slices: 7403
- target stage: D20
- elapsed: 223.8s

## Sanity Checks

- Decoder Linear type at last block (must be `Linear`, not `LinearWithLoRA`):
  see `[probe] SANITY:` line in stdout above.
- V7 best.pt decoder LoRA keys (must be empty):
  see `[probe] KEY AUDIT:` line in stdout above.

## Primary Numbers (n=7403)

| metric | mean | median | std | min | max | p05 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `psnr_ceil` PSNR(decode(z_GT_D20), x_D20) | 46.6356 | 45.8531 | 6.3888 | 34.3713 | 72.0681 | 38.8609 | 54.0455 |
| `psnr_transport` PSNR(decode(z_pred^V7), x_D20) | 35.4354 | 34.0377 | 8.6481 | 23.1092 | 74.5604 | 27.0808 | 43.6038 |
| **attackable_gap_dB** | **11.2002** | **12.0247** | 3.0486 | -2.4936 | 15.7607 | 6.0247 | 13.5095 |
| `psnr_transport_vs_ceil` PSNR(decode(z_pred^V7), decode(z_GT_D20)) | 35.9419 | 34.2926 | 9.5943 | 23.1030 | 78.1350 | 27.1853 | 44.1548 |
| `latent_l2_rel` ‖z_pred − z_GT‖₂ / ‖z_GT‖₂ | 0.0250 | 0.0233 | 0.0121 | 0.0018 | 0.1815 | 0.0096 | 0.0464 |

## Pre-Registered Decision Rule (Round 4 agent3 + integration)

| attackable_gap (mean) | action |
|---|---|
| < 2 dB | V18 wrong direction → launch V21 (conv head on frozen V7) |
| 2-5 dB | V18 rank=8 + KL fix + downgrade success threshold to +0.20 dB |
| ≥ 5 dB | V18 rank=32 + KL fix + original success threshold +0.30 dB |

## Verdict

**Decision: `V18_LAUNCH_RANK32_KLFIX`**

attackable_gap mean = 11.200 dB >= 5 dB. V18 has real runway.

Recommended: launch V18 with rank=32 (not rank=8 — decoder is already PET-full-tuned,
need bigger adapter), lambda_kl=0.05 (not 0.5 — see Round 4 agent1 calibration bug),
use_pred_latent=true for KL (focus regularization on predicted-path drift).
Original success threshold +0.30 dB still applies.


## Next Action

See `review/0517/CODEX_RUNBOOK_V18_20260517.md` §4 Day 1 implementation steps.
The pre-registered decision above replaces the launch-vs-no-launch question
that was open in Round 3.
