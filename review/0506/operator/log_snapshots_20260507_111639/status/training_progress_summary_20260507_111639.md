# Plan F Live Training Log Snapshot Summary

Snapshot time: `20260507_111639`
Snapshot directory: `review/0506/operator/log_snapshots_20260507_111639`

## Current Main Training Progress

| arm | latest train step | latest val step | val_pair_total | val_rollout_total | val_chain_normal_mse | val_select_score | log snapshot |
|---|---:|---:|---:|---:|---:|---:|---|
| V7 | 14150 | 14000 | 0.000003 | 0.002235 | 0.000610 | 0.002046 | `main_training/V7_train_snapshot_20260507_111639.log` |
| V8 | 35900 | 35600 | 0.000002 | 0.001372 | 0.000677 | 0.001791 | `main_training/V8_train_snapshot_20260507_111639.log` |
| V6_NOISE | 16000 | 16000 | 0.000002 | 0.001049 | 0.000453 | 0.001220 | `main_training/V6_NOISE_train_snapshot_20260507_111639.log` |

## Included Files

- `main_training/V7_train_snapshot_20260507_111639.log`
- `main_training/V8_train_snapshot_20260507_111639.log`
- `main_training/V6_NOISE_train_snapshot_20260507_111639.log`
- `status/runtime_status_20260507_111639.txt`
- `status/training_progress_summary_20260507_111639.md`

## Notes

- This is a fixed snapshot of live training logs; the source live logs under `review/0505/local/runs/` continue to grow.
- Phase 1 pre-flight logs were already pushed in the previous snapshot; this snapshot focuses on the latest main-training progress.
