# V6 step160K full-val anchor for Plan F

Date: 2026-05-06
Branch: `foc_lite_hop0`
Commit: `dcdcd2c14761b7edd7b23995b11ba855ecc488cb`

## Purpose

Plan F compares V7 and V8 at `step_160000.pt` against the existing V6 baseline at the same training stage. This anchor is required because:

- V6 was trained for 200K, but `step_150000.pt` does not exist due to `save_interval=20000`.
- `step_160000.pt` exists and is already on the full rollout plateau (`lambda_roll=4.0`).
- V6 `best.pt` and `last.pt` are not the primary anchor: `best.pt` has checkpoint-selection bias, and `last.pt` is at 200K rather than the 160K Plan F endpoint.
- The final V7/V8 primary endpoint should therefore be `V7/V8@step_160000.pt` vs `V6@step_160000.pt` on full-val.

## Command

Physical GPU2 was exposed as `cuda:0` inside the process.

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  /home/qujiaxiang/.conda/envs/rae/bin/python \
  review/0505/operator/scripts/eval_first_hop_fullval_psnr_chain_mse.py \
  --config configs/pet_flow/pet_flow_first_hop_224_v6_transport_first.yaml \
  --checkpoint /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v6_transport_first/step_160000.pt \
  --tag v6_step160k \
  --split val \
  --batch-size 8 \
  --device cuda:0 \
  --decode-mode both \
  --out-dir /data_2/qujiaxiang/outputs/PET_LatentResidual/eval_0505_planf_anchor
```

## Artifacts

Committed copies:

- `review/0505/operator/artifacts/v6_step160k_fullval_psnr_chain_mse.json`
- `review/0505/operator/artifacts/v6_step160k_fullval_psnr_chain_mse_per_slice.csv`
- `review/0505/operator/logs_eval/v6_step160k_fullval_gpu2.log`

Data-disk originals:

- `/data_2/qujiaxiang/outputs/PET_LatentResidual/eval_0505_planf_anchor/v6_step160k_fullval_psnr_chain_mse.json`
- `/data_2/qujiaxiang/outputs/PET_LatentResidual/eval_0505_planf_anchor/v6_step160k_fullval_psnr_chain_mse_per_slice.csv`

## Resource record

Pre-flight state before launch:

| resource | value |
|---|---|
| GPU | physical GPU2, NVIDIA RTX A6000 |
| GPU UUID | `GPU-4915051c-84ae-540c-8e84-5071cff15640` |
| GPU memory before launch | 18 MiB / 49140 MiB |
| GPU util before launch | 0% |
| GPU temperature before launch | 37 C |
| System RAM available before launch | about 476 GiB |
| `/data_2` free space before launch | about 22 TiB |

Observed I/O and runtime:

| item | value |
|---|---:|
| Val latent file | `/data_2/qujiaxiang/lowdose_pet_ct/latents_224/latents_val.pt` |
| Val latent size | 32.5 GB |
| Val latent load time | 19.8 s |
| Raw PET files loaded | D50, D20, D10, D4, plus current dataset's NORMAL source mapping |
| Raw PET file size per load | about 9.2 GB |
| Eval slices | 7403 |
| Batch size | 8 |
| Eval batches | 926 |
| Core eval runtime reported by script | 337.66 s |
| Steady eval speed | about 2.9 batch/s |
| JSON artifact size | 4.8 KB |
| Per-slice CSV artifact size | 3.0 MB |
| Eval log size | 68 KB |

The one-time full-val evaluation is therefore cheap relative to training. The main pressure is host RAM and disk I/O during dataset loading, not GPU memory.

## Full-val result: V6 step160K

| timepoint | PSNR clip3 (dB) | decoded chain MSE |
|---|---:|---:|
| D50 | 42.622378 | 6.524727e-05 |
| D20 | 35.442606 | 3.295092e-04 |
| D10 | 35.829859 | 2.952855e-04 |
| D4 | 36.389247 | 2.617726e-04 |
| NORMAL | 36.800920 | 2.445372e-04 |

Headline:

| metric | value |
|---|---:|
| NORMAL PSNR clip3 | 36.800920 dB |
| NORMAL chain MSE | 2.445372e-04 |
| Tail chain MSE, mean(D10,D4,NORMAL) | 2.671984e-04 |
| Transport PSNR avg, mean(D20,D10,D4,NORMAL) | 36.115658 dB |

## Comparison with existing V6 best/last full-val

| checkpoint | step | NORMAL PSNR | NORMAL MSE | tail MSE | avg transport PSNR |
|---|---:|---:|---:|---:|---:|
| V6 step160K | 160000 | 36.800920 | 2.445372e-04 | 2.671984e-04 | 36.115658 |
| V6 best | 185600 | 36.816634 | 2.430337e-04 | 2.667333e-04 | 36.109904 |
| V6 last200K | 200000 | 36.821088 | 2.427025e-04 | 2.666103e-04 | 36.109417 |

Paired per-slice NORMAL MSE deltas versus step160K:

| comparison | relative NORMAL MSE change | paired Cohen d | t-stat |
|---|---:|---:|---:|
| V6 best - V6 step160K | -0.615% | -0.208 | -17.90 |
| V6 last200K - V6 step160K | -0.750% | -0.208 | -17.86 |

Interpretation:

- V6 continues to improve slightly after 160K, but the practical gain to 200K is below 1% in NORMAL chain MSE.
- This is much smaller than the Plan F primary decision margin of 5%.
- The difference is statistically detectable because full-val has 7403 paired slices, but it is not large enough to justify replacing the Plan F anchor with V6@200K.
- For fair Plan F inference, `V6@step160K` remains the correct primary anchor.

## Rolling-val versus full-val at step160K

The V6 training log has a rolling-val row at step 160000:

| source | NORMAL chain MSE |
|---|---:|
| rolling val, 64 batches / 512 slices | 2.370000e-04 |
| full val, 7403 slices | 2.445372e-04 |
| full vs rolling relative difference | +3.18% |

This validates the previous concern: rolling-val is useful for monitoring but not reliable as a primary decision score. A few-percent discrepancy is possible even on the same checkpoint. Final Plan F decisions should use full-val artifacts, not single rolling-val rows from `metrics.jsonl`.

## Consequence for Plan F

The missing anchor is now available. The formal comparison chain should be:

1. Train V6-seed1337 to 160K and compute `d_pure` from `V6_seed42@step160K` vs `V6_seed1337@step160K` per-slice NORMAL MSE.
2. Set `d_thr = max(0.10, 1.8 * d_pure)`.
3. Train V7 and V8 to 160K with the locked V6-200K schedule.
4. Full-val evaluate V7/V8 `step_160000.pt` and compare against the committed `v6_step160k` anchor.
5. Treat `best.pt` results as secondary only, because full-val best is still a checkpoint-selection procedure.

## Current remaining blockers before final V7/V8 verdict

- `compute_d_pure.py` still needs to be implemented or replaced by an equivalent inline paired-stat script.
- V6-seed1337 must finish before the adaptive threshold is known.
- V7/V8 full-val should use the same evaluator and the same val split/artifact schema used here.
- V8 can only answer latent chain-MSE impact of removing `image_aux`; decode artifact conclusions still need a separate fixed-slice visual study.
