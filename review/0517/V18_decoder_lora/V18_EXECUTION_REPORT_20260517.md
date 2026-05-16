# V18 Execution Report 2026-05-17

## Summary

- Branch: `foc_lite_hop0`
- Pre-flight full-val probe: completed on GPU1
- Training launched: V18 decoder LoRA rank=32 + KL fix on GPU1
- tmux session: `v18_rank32_gpu1_0517`
- Training starts from V7 `best.pt` checkpoint step `160000` and continues to `max_steps=200000`.

## Phase 0 Full-Val Gap Decomposition

Output directory:

- `review/0517/V18_decoder_lora/gap_decomp_fullval_20260517/`

Key outputs:

- `GAP_DECOMP_REPORT.md`
- `GAP_DECOMP_SUMMARY.json`
- `GAP_DECOMP_PER_SLICE.csv`
- `V7_CKPT_SURVIVAL_CHECK.txt`
- `COMMAND.txt`

Full-val scope:

- split: `val`
- samples: `7403` unique hop0 slices through `Hop0OnlyViewDataset`
- target stage: `D20`
- elapsed: `223.8s`

Primary numbers:

| metric | mean | median | p05 | p95 |
|---|---:|---:|---:|---:|
| `psnr_ceil = PSNR(decode(z_GT_D20), x_D20)` | 46.6356 | 45.8531 | 38.8609 | 54.0455 |
| `psnr_transport = PSNR(decode(z_pred^V7), x_D20)` | 35.4354 | 34.0377 | 27.0808 | 43.6038 |
| `attackable_gap_dB` | 11.2002 | 12.0247 | 6.0247 | 13.5095 |
| `latent_l2_rel` | 0.0250 | 0.0233 | 0.0096 | 0.0464 |

Decision by pre-registered rule:

- `attackable_gap_dB mean = 11.2002 >= 5 dB`
- Decision: `V18_LAUNCH_RANK32_KLFIX`

This is why `review/0517/V18_decoder_lora/V18_decoder_lora.yaml` was updated from `rank: 8` to `rank: 32`.

## V7 Checkpoint Survival

`V7_CKPT_SURVIVAL_CHECK.txt` shows:

- `step_*.pt` count: `16`
- Available checkpoints: `step_010000.pt` through `step_160000.pt`
- `best.pt` and `last.pt` both exist

This preserves the option to run V9a checkpoint re-selection later, but it is not required to launch V18.

## Code Changes Made Before Launch

Files changed:

- `train_first_hop.py`
- `pet_lr/model_first_hop.py`
- `review/0517/V18_decoder_lora/V18_decoder_lora.yaml`

Main implementation points:

- `freeze_rae=false` is allowed only when `training.decoder_lora.enabled=true`; full RAE/decoder finetuning remains blocked.
- Decoder LoRA wraps only the last decoder blocks via RAE's native `LinearWithLoRA`.
- Non-LoRA RAE parameters remain frozen; only `lora_A/lora_B` are trainable.
- Frozen reference RAE is kept out of `state_dict` to avoid saving a second RAE copy in checkpoints.
- Resume remaps V7 decoder base Linear keys, e.g. `...query.weight`, into LoRA-wrapped base keys, e.g. `...query.linear.weight`.
- V7 missing keys are allowed only for fresh LoRA parameters and existing known new modules.
- Step-0 decoder equivalence check is mandatory and passed with `max_abs=0.000e+00`.
- V18 warm-start preserves checkpoint `step=160000` rather than resetting to 0, so LR and rollout schedules remain aligned.
- Optimizer has a separate `decoder_lora` param group: `589,824` trainable LoRA parameters, LR `5e-7` at base LR, weight decay `0`.
- EMA is reset after warm-start loading so EMA shadow matches V7-loaded weights plus zero-init LoRA.
- KL pull-back loss is logged as `decoder_kl`, `lambda_kl`, and `kl_w`.

## Smoke Test

Smoke config:

- `review/0517/V18_decoder_lora/smoke_v18_train/V18_decoder_lora_smoke_1step.yaml`

Smoke logs:

- First attempt stopped at best-metric signature mismatch, after LoRA/remap/equivalence checks had already passed.
- Retry completed one training step successfully.

Retry log:

- `review/0517/V18_decoder_lora/smoke_v18_train/V18_train_smoke_1step_gpu1_retry_20260517_044921.log`

Important smoke lines:

- `remapped 24 decoder base Linear keys to LinearWithLoRA .linear.* keys`
- `24 new-module keys initialized from scratch`
- `step0 decode equivalence OK: max_abs=0.000e+00`
- `architecture changed -> start_step=160000`
- `reset EMA shadow from warm-started model`
- completed `step=160001`

## Formal V18 Training Launch

Command file:

- `review/0517/V18_decoder_lora/V18_TRAIN_COMMAND_20260517.txt`

Training log:

- `review/0517/V18_decoder_lora/logs/V18_rank32_train_gpu1_20260517_045325.log`

Run output directory on data disk:

- `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0517_runs/V18_decoder_lora/run/first_hop_224_v18_decoder_lora`

Repository snapshots of current run outputs:

- `review/0517/V18_decoder_lora/run_snapshots/V18_rank32_run_config_20260517.yaml`
- `review/0517/V18_decoder_lora/run_snapshots/V18_rank32_metrics_snapshot_20260517_0500.jsonl`

Current observed training status at report time:

- process is alive in tmux session `v18_rank32_gpu1_0517`
- GPU1 memory: about `19.3GB`
- first train rows observed at `step=160050` and `step=160100`
- early `decoder_kl` is near zero, expected because LoRA B is zero-initialized and decoder output is initially identical to frozen reference

## Notes / Risks

- The dataloader still loads the full 115.3GB train latent file at startup. This is current project behavior, not introduced by V18.
- Validation full-val best selection runs every `5000` steps, while rolling eval runs every `400` steps. The trainer warns that the grids are not co-located; this is functionally acceptable but should be considered when interpreting early logs.
- Deterministic warn-only mode emits CUDA/cuBLAS warnings during decoder attention and LoRA matmul. This matches existing `warn_only=True` behavior and did not block smoke or launch.
