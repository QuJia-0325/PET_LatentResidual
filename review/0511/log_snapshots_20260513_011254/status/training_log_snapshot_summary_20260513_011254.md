# Training Log Snapshot 20260513_011254

- copied_at: 2026-05-13 01:12:54 CST
- branch: foc_lite_hop0
- commit: 001c2e3
- source_main_logs: review/0505/local/runs/{V7,V8,V6_NOISE}/train.log
- source_preflight_logs: review/0506/local/logs/V{6,7}_5K_pass{1,2}_gpu{2,3}.log

## Runtime Snapshot
```text
0, NVIDIA RTX A6000, 16831, 49140, 100, 83
1, NVIDIA RTX A6000, 35516, 49140, 100, 82
2, NVIDIA RTX A6000, 18631, 49140, 0, 55
3, NVIDIA RTX A6000, 17983, 49140, 0, 42
```

## Processes
```text
    PID    PPID STAT     ELAPSED %CPU %MEM   RSS CMD
2195795 1117917 Sl+     08:14:12  100  0.8 4579200 python train_ldcon.py
2196656 3997725 Sl+     08:11:01  100  0.9 4813408 python train_ld_deft.py
3970601 3970600 Rl+   6-11:26:35 1775 28.0 148217328 /home/qujiaxiang/.conda/envs/rae/bin/python train_first_hop.py --config /home/qujiaxiang/project/PET_LatentResidual/review/0505/local/runs/V8/config.resolved.yaml
3970603 3970599 Rl+   6-11:26:35 1706 28.0 148323988 /home/qujiaxiang/.conda/envs/rae/bin/python train_first_hop.py --config /home/qujiaxiang/project/PET_LatentResidual/review/0505/local/runs/V7/config.resolved.yaml
4003657 4003644 Rl+   6-11:12:26 1506 28.0 148091396 /home/qujiaxiang/.conda/envs/rae/bin/python train_first_hop.py --config /home/qujiaxiang/project/PET_LatentResidual/review/0505/local/runs/V6_NOISE/config.resolved.yaml
```

## Latest Metrics

| Experiment | log MB | latest train step | latest val step | val_select_score | chain_normal_mse | chain_d20_mse | chain_tail_mse |
|---|---:|---:|---:|---:|---:|---:|---:|
| V7 | 1.96 | 93800 | 93600 | 0.001082 | 0.000292 | 0.000399 | 0.000321 |
| V8 | 2.49 | 122300 | 122000 | 0.000686 | 0.000191 | 0.000238 | 0.000204 |
| V6_NOISE | 2.35 | 113100 | 112800 | 0.001709 | 0.000444 | 0.000681 | 0.000505 |

## Files
- `review/0511/log_snapshots_20260513_011254/main_training/V6_NOISE_train_snapshot_20260513_011254.log` (2462400 bytes)
- `review/0511/log_snapshots_20260513_011254/main_training/V7_train_snapshot_20260513_011254.log` (2059660 bytes)
- `review/0511/log_snapshots_20260513_011254/main_training/V8_train_snapshot_20260513_011254.log` (2610711 bytes)
- `review/0511/log_snapshots_20260513_011254/phase1_preflight/V6_5K_pass1_gpu2.log` (113569 bytes)
- `review/0511/log_snapshots_20260513_011254/phase1_preflight/V6_5K_pass2_gpu2.log` (115400 bytes)
- `review/0511/log_snapshots_20260513_011254/phase1_preflight/V7_5K_pass1_gpu3.log` (114966 bytes)
- `review/0511/log_snapshots_20260513_011254/phase1_preflight/V7_5K_pass2_gpu3.log` (117374 bytes)
- `review/0511/log_snapshots_20260513_011254/status/training_log_snapshot_summary_20260513_011254.md` (1372 bytes)
