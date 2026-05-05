# V6 / V6.1 / Sanity-Light Status and Full-Val Results - 2026-05-05

## Scope

This note records the operator-side status after the three training jobs completed, copies the training logs into `review/0505/logs_train`, and documents the full-val PSNR + decoded chain-MSE evaluation on GPU2/GPU3.

## Training Completion

All three relevant PET jobs have finished; no `train_first_hop.py` process for these runs remains active.

| Run | Final step | Val rows | Best rolling step | Best rolling `val_select_score` | Best rolling `val_chain_normal_mse` | Last rolling `val_select_score` | Last rolling `val_chain_normal_mse` | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| V6 `first_hop_224_v6_transport_first` | 200000 | 500 | 185600 | 0.000607273950 | 0.000170265537 | 0.001349511554 | 0.000354893899 | complete 200k |
| V6.1 `first_hop_224_v6_1_rollout_floor` | 200000 | 500 | 185600 | 0.000606751702 | 0.000170195556 | 0.001357753343 | 0.000358069183 | complete 200k |
| `B_sanity_light` | 50000 | 125 | 44800 | 0.000555750795 | 0.0 | 0.000597511231 | 0.0 | light protocol; chain metrics intentionally unavailable (`samples=0`) |

Interpretation from rolling validation only:

- V6 and V6.1 are nearly indistinguishable in rolling validation.
- V6.1 is marginally better at the best rolling selection row, but the difference is tiny and not meaningful without full-val.
- Both `last.pt` rows look worse than `best.pt` in rolling validation, but full-val below shows this rolling-window signal is not reliable for final selection.
- `B_sanity_light` is complete, but its `val_chain_normal_mse=0.0` is not a valid chain-quality result because the light sanity path does not evaluate full chain image metrics.

## Copied Training Logs

Training logs copied into:

```text
review/0505/logs_train/v6_transport_first_train_200k.log
review/0505/logs_train/v6_1_rollout_floor_train_200k.log
review/0505/logs_train/A_sanity_light_50k_train.log
review/0505/logs_train/B_sanity_light_50k_train.log
```

## Full-Val Evaluation Protocol

Helper script:

```text
review/0505/scripts/eval_first_hop_fullval_psnr_chain_mse.py
```

Metric contract:

- PSNR: `src.utils.metrics.calc_psnr_clip3`.
- Chain MSE: `mean((decode_crop(z_chain[t]) - x_rollout[t])^2)`, matching `train_first_hop.py`'s `val_chain_*_mse` definition.
- `NORMAL` chain MSE in this eval is therefore the full-val version of training's rolling `val_chain_normal_mse`.
- Decode mode: `both`, so JSON contains both default/refined keys and `_raw` keys. In these runs default and raw match, meaning seam refiner is not changing the decoded output in this eval path.

Launched tmux sessions:

| Session | GPU | Work | Final status |
|---|---:|---|---|
| `eval_0505_v6_gpu2` | 2 | V6 `best.pt` then `last.pt` | completed 2026-05-05 16:21:51 CST |
| `eval_0505_v61_gpu3` | 3 | V6.1 `best.pt` then `last.pt` | completed 2026-05-05 16:21:52 CST |

Evaluation logs:

```text
review/0505/logs_eval/v6_best_last_fullval_gpu2.log
review/0505/logs_eval/v6_1_best_last_fullval_gpu3.log
```

Output snapshots copied into:

```text
review/0505/artifacts/
```

Original output directory:

```text
/data_2/qujiaxiang/outputs/PET_LatentResidual/eval_0505_v6_v61_fullval
```

## Full-Val Results

All four evals used the full val split: `num_eval_slices = 7403`.

| Run | Ckpt | Step | NORMAL PSNR clip3 | NORMAL chain MSE | Tail chain MSE | Transport PSNR avg | Runtime sec |
|---|---|---:|---:|---:|---:|---:|---:|
| V6 | best.pt | 185600 | 36.816633633 | 0.000243033704 | 0.000266733347 | 36.109904417 | 423.55 |
| V6 | last.pt | 200000 | 36.821087917 | 0.000242702525 | 0.000266610296 | 36.109416923 | 410.91 |
| V6.1 | best.pt | 185600 | 36.790766564 | 0.000244308949 | 0.000267778256 | 36.099950897 | 425.64 |
| V6.1 | last.pt | 200000 | 36.794051831 | 0.000244004359 | 0.000267685405 | 36.098049875 | 410.33 |

Per-timepoint full-val PSNR and decoded chain MSE:

| Run | Ckpt | D50 PSNR | D20 PSNR | D10 PSNR | D4 PSNR | NORMAL PSNR | D20 MSE | D10 MSE | D4 MSE | NORMAL MSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V6 | best.pt | 42.622377791 | 35.425645358 | 35.812002959 | 36.385335718 | 36.816633633 | 0.000330420400 | 0.000295752110 | 0.000261414228 | 0.000243033704 |
| V6 | last.pt | 42.622377791 | 35.422529175 | 35.808847970 | 36.385202630 | 36.821087917 | 0.000330634724 | 0.000295813899 | 0.000261314464 | 0.000242702525 |
| V6.1 | best.pt | 42.622377791 | 35.423242880 | 35.810486251 | 36.375307890 | 36.790766564 | 0.000330792489 | 0.000296512447 | 0.000262513372 | 0.000244308949 |
| V6.1 | last.pt | 42.622377791 | 35.417367080 | 35.805936273 | 36.374844314 | 36.794051831 | 0.000331033524 | 0.000296626473 | 0.000262425384 | 0.000244004359 |

## Interpretation

- V6 is consistently ahead of V6.1 on full-val NORMAL PSNR and NORMAL chain MSE.
- The margin is small but consistent: V6 best vs V6.1 best is about `+0.0259 dB` NORMAL PSNR and `-1.275e-6` NORMAL MSE; V6 last vs V6.1 last is about `+0.0270 dB` and `-1.302e-6` MSE.
- Within each design, `last.pt` is slightly better than `best.pt` on full-val NORMAL even though rolling validation selected `best.pt` at step 185600. This supports our earlier concern that rolling-val best selection is noisy/local and can disagree with full-val.
- V6.1 rollout_floor does not show a measurable improvement over V6 under this full-val protocol. If anything, it is slightly worse.
- Because default and raw outputs are identical in the JSON headline, seam-refiner/refined-vs-raw is not driving these differences.

## Artifacts

```text
review/0505/artifacts/v6_best_fullval_psnr_chain_mse.json
review/0505/artifacts/v6_best_fullval_psnr_chain_mse_per_slice.csv
review/0505/artifacts/v6_last_fullval_psnr_chain_mse.json
review/0505/artifacts/v6_last_fullval_psnr_chain_mse_per_slice.csv
review/0505/artifacts/v6_1_best_fullval_psnr_chain_mse.json
review/0505/artifacts/v6_1_best_fullval_psnr_chain_mse_per_slice.csv
review/0505/artifacts/v6_1_last_fullval_psnr_chain_mse.json
review/0505/artifacts/v6_1_last_fullval_psnr_chain_mse_per_slice.csv
```
