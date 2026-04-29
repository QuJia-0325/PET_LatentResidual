# V5/V6 Training Status Snapshot

Date: 2026-04-30 Asia/Shanghai

## Process Status

| Run | GPU | PID | Status | Last train step | Last val step | Notes |
|---|---:|---:|---|---:|---:|---|
| V5 null-control | 3 | 3845649 | running | 135950 | 135600 | baseline-loss continuation from V3 best; rolling-val currently worse than inherited best |
| V6 transport-first | 1 | 3764014 | running | 51000 | 50800 | just entered early rollout ramp; lambda_roll approx 0.04, alpha approx 0.01 |

## Latest Metrics

| Run | Rolling best step | Rolling best score | Last val score | Last D20 MSE | Last NORMAL MSE |
|---|---:|---:|---:|---:|---:|
| V5 null-control | 92800 | 0.000593790 | 0.001219655 | 0.000485208 | 0.000316948 |
| V6 transport-first | 6000 | 0.000771792 | 0.001063853 | 0.000272985 | 0.000379884 |

## Snapshot Files

- `log_snapshots/v5_null_control_gpu3_train_snapshot_<timestamp>.log`
- `log_snapshots/v6_transport_first_gpu1_train_snapshot_<timestamp>.log`
- `log_snapshots/v5_null_control_metrics_snapshot_<timestamp>.jsonl`
- `log_snapshots/v6_transport_first_metrics_snapshot_<timestamp>.jsonl`

## Interpretation

Training is alive on both runs. The values above are rolling-window validation metrics, not full-val rerank results. Final checkpoint comparison still requires full-val evaluation after each run reaches its intended stopping point.
