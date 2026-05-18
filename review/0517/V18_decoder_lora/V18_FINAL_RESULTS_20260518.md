# V18 Final Results - 2026-05-18

## Latest Design Context

- V18 = RAE decoder LoRA finetune, rank=32, limited to the last 2 decoder layers.
- Training warm-started from V7 `best.pt` at step `160000` and ran to `max_steps=200000`.
- The latest repo revision fixed the stale decoder-dimension assumption: the actual RAE decoder is `8` layers with `hidden=512` and `intermediate=2048`.
- Round 5 integration in the latest branch also flags two important interpretation risks for the current design: `use_pred_latent=true` can create KL/image_aux gradient conflict (B9), and the earlier `11.2 dB` gap framing overstates what the decoder can actually recover (B10).

## Training Status

Training completed normally.

- `best.pt` step: `165000`
- `last.pt` step: `200000`
- `step_200000.pt` step: `200000`
- full-val checkpoints written at steps `165000, 170000, 175000, 180000, 185000, 190000, 195000, 200000`

The selection metric is the full-val multi-objective score, not rolling-val.

## Final Full-Val Results

Full-val sample count at step `200000`:

- `val_chain_samples = 7403`
- `val_main_batches_evaluated = 3702`

### Best checkpoint by selection metric

`best.pt` (`step=165000`, `best_val=0.0009037493852408773`)

| metric | value |
|---|---:|
| `val_select_score` | `0.0009037493852408773` |
| `val_pair_total` | `0.00011310829821969236` |
| `val_rollout_total` | `0.0007831536440681834` |
| `val_hop0_img_total` | `0.0075349056686846785` |
| `val_chain_d20_mse` | `0.0003304456302737126` |
| `val_chain_d10_mse` | `0.0002966659882599162` |
| `val_chain_d4_mse` | `0.00026309424350887974` |
| `val_chain_normal_mse` | `0.0002454947041527113` |
| `val_chain_tail_mse` | `0.00026841831197383574` |

### Best checkpoint by NORMAL MSE

`step=175000`

| metric | value |
|---|---:|
| `val_select_score` | `0.000904345925061459` |
| `val_chain_normal_mse` | `0.0002453743575473458` |
| `val_chain_d20_mse` | `0.0003310518630994743` |
| `val_chain_d10_mse` | `0.0002971476440382529` |
| `val_chain_d4_mse` | `0.0002633800193038772` |
| `val_pair_total` | `0.00011421653533881156` |
| `val_rollout_total` | `0.0007961236504113152` |
| `val_hop0_img_total` | `0.007545531325407405` |

### Final checkpoint

`last.pt` / `step_200000.pt`

| metric | value |
|---|---:|
| `val_select_score` | `0.00090535337875151` |
| `val_pair_total` | `0.00011698490318593471` |
| `val_rollout_total` | `0.000824408485678813` |
| `val_hop0_img_total` | `0.00756023575548962` |
| `val_chain_d20_mse` | `0.00033179921837840355` |
| `val_chain_d10_mse` | `0.0002976324984604366` |
| `val_chain_d4_mse` | `0.0002638202947701549` |
| `val_chain_normal_mse` | `0.0002453872533079816` |
| `val_chain_tail_mse` | `0.00026894668217952435` |

## PSNR_clip3 Conversion

Using `PSNR = 10 * log10(9 / MSE)`:

| checkpoint | D20 | D10 | D4 | NORMAL |
|---|---:|---:|---:|---:|
| best.pt @ 165000 | 44.3514 dB | 44.8197 dB | 45.3413 dB | 45.6420 dB |
| best-normal @ 175000 | 44.3435 dB | 44.8127 dB | 45.3366 dB | 45.6441 dB |
| last.pt @ 200000 | 44.3337 dB | 44.8056 dB | 45.3293 dB | 45.6439 dB |

## Interpretation

- Training is stable and reaches the configured end without early stop.
- `best.pt` and `last.pt` are effectively tied. The final checkpoint is only marginally different from the best-selected checkpoint.
- The final step does **not** show a meaningful late-stage gain over the best checkpoint. The curve is essentially plateaued by about `165k-175k`.
- If a single checkpoint is needed for downstream comparison, keep `best.pt` as the selection checkpoint and treat `last.pt` as a tie.
- Because the latest review round also flags B9/B10, claims about transport-side causal gain should remain conservative until those issues are rechecked.

## Artifacts

- Training log: [V18_rank32_train_gpu1_20260517_045325.log](/home/qujiaxiang/project/PET_LatentResidual/review/0517/V18_decoder_lora/logs/V18_rank32_train_gpu1_20260517_045325.log)
- Run directory: `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0517_runs/V18_decoder_lora/run/first_hop_224_v18_decoder_lora`
- Launch report: [V18_EXECUTION_REPORT_20260517.md](/home/qujiaxiang/project/PET_LatentResidual/review/0517/V18_decoder_lora/V18_EXECUTION_REPORT_20260517.md)
