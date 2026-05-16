# Plan-F Training Progress and Effect Analysis (20260516_163900)

- generated_at: 2026-05-16 16:39:01 Asia/Shanghai
- branch: foc_lite_hop0
- commit_when_generated: 110f9bd
- snapshot_dir: `review/0511/log_snapshots_20260516_163900`
- metric direction: lower is better for MSE and `val_select_score`.

## Runtime State

```text
0, NVIDIA RTX A6000, 16831, 49140, 100, 81
1, NVIDIA RTX A6000, 18, 49140, 0, 35
2, NVIDIA RTX A6000, 11351, 49140, 100, 73
3, NVIDIA RTX A6000, 17551, 49140, 100, 82
```

Current Plan-F `train_first_hop.py` processes: none. All three Plan-F training jobs have exited.

## Completion Table

| Experiment | Status | latest train step | max_steps | errors | best.pt | last.pt | step_160000.pt |
|---|---|---:|---:|---:|---|---|---|
| V7 | done | 160000 | 160000 | 0 | yes | yes | yes |
| V8 | done | 160000 | 160000 | 0 | yes | yes | yes |
| V6_NOISE | done | 160000 | 160000 | 0 | yes | yes | yes |

## Primary Metric Table

| Experiment | final rolling select | final rolling D20 | final rolling D10 | final rolling D4 | final rolling Normal | best rolling select@step | final full select | final full D20 | final full D10 | final full D4 | final full Normal | best full select@step |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| V7 | 0.000857 | 0.000296 | 0.000276 | 0.000252 | 0.000239 | 0.000610@156400 | 0.000904 | 0.000330 | 0.000296 | 0.000263 | 0.000246 | 0.000904@160000 |
| V8 | 0.000907 | 0.000307 | 0.000290 | 0.000267 | 0.000255 | 0.000650@156400 | 0.000953 | 0.000340 | 0.000310 | 0.000279 | 0.000262 | 0.000953@160000 |
| V6_NOISE | 0.000861 | 0.000297 | 0.000277 | 0.000253 | 0.000240 | 0.000611@156400 trainer-selected (`139200` ties after rounding) | NA | NA | NA | NA | NA | NA |

## Findings

1. All three Plan-F training jobs completed normally at `160000` steps, and no traceback/OOM/killed pattern appears in the logs.
2. On full-val best/final comparison, V7 is lower than V8 for select score: V7 `0.000904` vs V8 `0.000953`, relative reduction `5.14%`.
3. V7 full-val at step 160000: select `0.000904`, Normal `0.000246`, D20 `0.000330`. V8 full-val at step 160000: select `0.000953`, Normal `0.000262`, D20 `0.000340`. This suggests V7's gronwall/raw design is modestly better than removing image aux in V8 under the same full-val protocol.
4. V6_NOISE has no `[val_full]` records because its config does not set `best_select_full_eval_interval`; its `best.pt` is therefore selected by rolling-window val, not full-val. Its trainer-selected best rolling score is printed as `0.000611@156400`; `139200` ties after six-decimal log rounding, so the actual checkpoint should be interpreted from the trainer `new best` line. This rolling-window result is not directly comparable to V7/V8 full-val best scores.
5. Rolling-val can fluctuate substantially: V6_NOISE best rolling `0.000611` at trainer-selected step `156400` is much better than its final rolling `0.000861` at step `160000`. This reinforces that rolling-window best selection is noisy and should be followed by fair full-val eval of `best.pt` and `last.pt`.

## Recommended Next Steps

1. Run fixed-protocol full-val evaluation for `V7`, `V8`, and `V6_NOISE` using both `best.pt` and `last.pt`; V6_NOISE especially needs this because training did not emit full-val records.
2. Use full-val results, not rolling-window val, as the comparison basis for claims. Rolling windows remain useful for training health only.
3. If compute is tight, prioritize V6_NOISE `best.pt`/`last.pt` full-val first, then compare against V7/V8 final full-val already available in logs.

## Log and Config Files in This Snapshot

- `review/0511/log_snapshots_20260516_163900/configs/V6_NOISE_config.resolved.yaml` (12675 bytes)
- `review/0511/log_snapshots_20260516_163900/configs/V7_config.resolved.yaml` (15652 bytes)
- `review/0511/log_snapshots_20260516_163900/configs/V8_config.resolved.yaml` (15687 bytes)
- `review/0511/log_snapshots_20260516_163900/main_training/V6_NOISE_train_20260516_163900.log` (3482197 bytes)
- `review/0511/log_snapshots_20260516_163900/main_training/V7_train_20260516_163900.log` (3513728 bytes)
- `review/0511/log_snapshots_20260516_163900/main_training/V8_train_20260516_163900.log` (3416560 bytes)
- `review/0511/log_snapshots_20260516_163900/status/training_effect_analysis_20260516_163900.md` (4332 bytes)
- `review/0511/log_snapshots_20260516_163900/status/training_metrics_summary_20260516_163900.json` (22200 bytes)
- `review/0511/log_snapshots_20260516_163900/status/training_metrics_table_20260516_163900.csv` (829 bytes)
- `review/0511/log_snapshots_20260516_163900/status/training_status_brief_20260516_163900.md` (4332 bytes)
