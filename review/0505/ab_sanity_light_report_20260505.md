# A/B Sanity Light Report (2026-05-05)

## Executive conclusion

The A/B sanity-light experiment completed for both branches. The previously pushed `review/0505/v6_v61_training_and_fullval_status_20260505.md` only mentioned `B_sanity_light` as a status row and did not provide a standalone A/B sanity report; this file fills that gap.

Conclusion: the A/B sanity-light check is consistent with the intended sigma-normalize equivalence. Both runs used the same configured RNG seed (`seed: 42`) and deterministic mode, but the run should be described as **same-seed / deterministic-intended**, not strictly bitwise deterministic, because PyTorch reported warn-only nondeterministic CUDA kernels during training.

## What Was Compared

| Branch | Meaning | Key rollout setting | Output/log snapshot |
|---|---|---|---|
| `A_sanity_light` | V6 baseline-equivalent control | `sigma_normalize` disabled, V6 step weights `[0.5, 2.0, 1.5, 1.0]` | `review/0505/logs_train/A_sanity_light_50k_train.log` |
| `B_sanity_light` | sigma-normalize equivalence sanity | `sigma_normalize.enabled=true`, `relative_to=preserve_v6_sum`, step weights `[1.7908, 1.8606, 0.8691, 0.4795]` | `review/0505/logs_train/B_sanity_light_50k_train.log` |

Both runs used the memory-light protocol: only `D50` and `D20` raw images were loaded, full x-rollout image metrics were disabled, and validation used rolling windows with `max_val_batches=64`. Therefore this is not a full-val PSNR/chain-quality report; it is an implementation sanity check for the sigma-normalize A==B equivalence path.

## RNG / Determinism Evidence

| Item | A | B | Interpretation |
|---|---:|---:|---|
| `seed` | `42` | `42` | same configured RNG seed |
| `training.deterministic` | `true` | `true` | deterministic mode requested |
| `data.num_workers` | `0` | `0` | avoids DataLoader worker RNG divergence |
| `data.batch_size` | `8` | `8` | same main batch size |
| `data.hop0_aux_batch_size` | `2` | `2` | same hop0 aux batch size |
| `val_window_mode` | `rolling` | `rolling` | same rolling-val protocol |
| `max_val_batches` | `64` | `64` | same validation window size |

Code path evidence:

- `train_first_hop.py` calls `random.seed`, `np.random.seed`, `torch.manual_seed`, and `torch.cuda.manual_seed_all` in `set_seed()`.
- `train_first_hop.py` reads `cfg['seed']` and passes `training.deterministic` into `set_seed()` at startup.
- When `deterministic=true`, the trainer sets `torch.backends.cudnn.benchmark=False`, `torch.backends.cudnn.deterministic=True`, and `torch.use_deterministic_algorithms(True, warn_only=True)`.
- `review/0502/scripts/run_sanity_light.sh` exports `CUBLAS_WORKSPACE_CONFIG=:4096:8` and runs `A_sanity_light` then `B_sanity_light` serially.

Important caveat: the logs contain PyTorch warnings for nondeterministic CUDA operations under `warn_only=True`, including memory-efficient attention and `adaptive_avg_pool2d_backward_cuda`. Therefore exact bitwise equality should not be claimed. The correct statement is: **the A/B experiment used the same RNG seed and deterministic-intended settings, but hardware/kernel nondeterminism can still create small drift.**

## Metric Summary

Validation rows common to A/B: 125 (`step=400` through `step=50000`). Selection metric: `val_rollout_total` / `val_select_score`. Chain metrics are intentionally unavailable in light protocol (`val_chain_*_mse=0`, `val_chain_samples=0`).

| Metric | A best | B best | Delta B-A | Relative delta | Best step |
|---|---:|---:|---:|---:|---:|
| `val_rollout_total` best | 0.000561 | 0.000556 | -0.000005 | -0.891% | 44800 for both |
| `val_rollout_total` final | 0.000587 | 0.000598 | +0.000011 | +1.874% | 50000 |
| `val_hop0_img_total` final | 0.010731 | 0.010727 | -0.000004 | -0.037% | 50000 |
| `val_pair_total` final | 0.000299 | 0.000302 | +0.000003 | +1.003% | 50000 |

Across all 125 common validation windows:

| Metric | Mean delta B-A | Max absolute delta | Mean relative delta | Max absolute relative delta |
|---|---:|---:|---:|---:|
| `val_rollout_total` | +0.000017392 | 0.000151 | +1.4133% | 9.1528% |
| `val_hop0_img_total` | -0.000022256 | 0.000114 | -0.3154% | 1.4976% |
| `val_pair_total` | +0.000000248 | 0.000011 | +0.5054% | 25.0000% |

The large relative maxima should not be over-interpreted when denominators are near zero. The absolute differences are small, and the best step matches exactly at 44800.

## Selected Validation Rows

| Step | A `val_rollout_total` | B `val_rollout_total` | Delta B-A | Relative delta | A `val_hop0_img_total` | B `val_hop0_img_total` |
|---:|---:|---:|---:|---:|---:|---:|
| 400 | 0.001400 | 0.001400 | +0.000000 | +0.000% | 0.007738 | 0.007738 |
| 2000 | 0.001160 | 0.001160 | +0.000000 | +0.000% | 0.007805 | 0.007805 |
| 3600 | 0.000907 | 0.000907 | +0.000000 | +0.000% | 0.011182 | 0.011182 |
| 26800 | 0.000842 | 0.000850 | +0.000008 | +0.950% | 0.011026 | 0.010985 |
| 32800 | 0.000760 | 0.000777 | +0.000017 | +2.237% | 0.012136 | 0.012131 |
| 38800 | 0.000676 | 0.000687 | +0.000011 | +1.627% | 0.005680 | 0.005680 |
| 44400 | 0.000618 | 0.000621 | +0.000003 | +0.485% | 0.006596 | 0.006590 |
| 44800 | 0.000561 | 0.000556 | -0.000005 | -0.891% | 0.004914 | 0.004909 |
| 50000 | 0.000587 | 0.000598 | +0.000011 | +1.874% | 0.010731 | 0.010727 |

## Interpretation

1. The A/B sanity-light result supports the implementation-level claim that `preserve_v6_sum` sigma-normalize with rescaled B weights does not materially change the optimization signal relative to A.
2. This is not a method-effect result. If B later differs from A_main/C_uniform/D_closed_form in full-val PSNR, that difference should be interpreted only after running the formal ablation branches and full-val evaluation.
3. The same seed was used, but the warning-only deterministic setting means residual GPU nondeterminism remains. Small deltas at the 1e-5 to 1e-4 MSE level are expected and should not be treated as evidence of a mechanism effect.
4. The light protocol does not validate final NORMAL PSNR or open-loop chain quality. It validates the A/B equivalence plumbing before expensive production ablations.

## Source Artifacts

- `review/0502/runs/sanity_light_20260502_051245/A_sanity_light/config.resolved.yaml`
- `review/0502/runs/sanity_light_20260502_051245/B_sanity_light/config.resolved.yaml`
- `review/0505/logs_train/A_sanity_light_50k_train.log`
- `review/0505/logs_train/B_sanity_light_50k_train.log`
- `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0502_runs_light/20260502_051245/A_sanity_light/run/first_hop_224_sigma_norm_A_sanity_light_20260502_051245/metrics.jsonl`
- `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0502_runs_light/20260502_051245/B_sanity_light/run/first_hop_224_sigma_norm_B_sanity_light_20260502_051245/metrics.jsonl`
