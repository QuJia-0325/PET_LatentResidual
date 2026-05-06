# Plan F Training Log Snapshot Summary

Snapshot directory: `review/0506/operator/log_snapshots_20260506_161548`

## Location Check

- Gate/supervisor logs were already under `review/0506/operator/logs/`.
- Phase 1 pre-flight train logs were originally under `review/0506/local/logs/` and copied into this snapshot.
- Current main train logs were originally under `review/0505/local/runs/*/train.log` and copied into this snapshot.

## Current Main Training Progress

| arm | latest train step | latest val step | latest val_select_score | log snapshot |
|---|---:|---:|---:|---|
| V7 | 2850 | 02800 | 0.001703 | `main_training/V7_train_snapshot_20260506_161548.log` |
| V8 | 6650 | 06400 | 0.000949 | `main_training/V8_train_snapshot_20260506_161548.log` |
| V6_NOISE | 2150 | 02000 | 0.000852 | `main_training/V6_NOISE_train_snapshot_20260506_161548.log` |

## Phase 1 Pre-flight Logs

- `phase1_preflight/V6_5K_pass1_gpu2.log`
- `phase1_preflight/V6_5K_pass2_gpu2.log`
- `phase1_preflight/V7_5K_pass1_gpu3.log`
- `phase1_preflight/V7_5K_pass2_gpu3.log`

## Notes

- V7/V8/V6_NOISE are all running in tmux sessions at snapshot time.
- V6_NOISE was launched directly on GPU1 per operator request, sharing GPU1 with an existing non-project process.
- System RAM is tight because three project loaders plus existing GPU jobs keep large latent tensors resident; monitor OOM risk.
