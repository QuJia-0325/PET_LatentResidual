# GT Latent Direct Decode Ceiling (clip3) - Full Val

Date: 2026-04-22  
Branch: `foc_lite_hop0`  
Purpose: Verify the upper-bound reconstruction quality of the current stage1 encoder-decoder pair by decoding GT latents directly.

## Evaluation Setup

- Metric: `src.utils.metrics.calc_psnr_clip3` (required `clip_max=3` method)
- Split: `val` (full-val, `n=7403`)
- Timepoints: `D50, D20, D10, D4, NORMAL`
- Latents: `/data_2/qujiaxiang/lowdose_pet_ct/latents_224/latents_val.pt`
- Raw data: `/data_2/qujiaxiang/preprocessed_data_*.pt`
- Stage1 ckpt: `/data_2/qujiaxiang/outputs/pet_lora_dinov2_pt_224/best_model.pt`
- Device: physical `gpu2` (`CUDA_VISIBLE_DEVICES=2`, script uses `--device cuda:0`)
- Command:

```bash
CUDA_VISIBLE_DEVICES=2 TQDM_DISABLE=1 \
/home/qujiaxiang/.conda/envs/rae/bin/python -u scripts/eval_gt_latent_decoder_ceiling_clip3.py \
  --split val --device cuda:0 --batch-size 16 \
  --out-dir /data_2/qujiaxiang/outputs/PET_LatentResidual/gt_latent_decoder_ceiling_20260422
```

## Results (PSNR Ceiling, clip3)

| Timepoint | Mean PSNR (dB) | Std | Min | Max | N |
|---|---:|---:|---:|---:|---:|
| D50 | 42.622386 | 7.256346 | 31.137291 | 72.068481 | 7403 |
| D20 | 46.635633 | 6.388344 | 34.371297 | 72.068178 | 7403 |
| D10 | 48.743580 | 6.023975 | 34.752007 | 72.067465 | 7403 |
| D4 | 50.831939 | 5.790249 | 35.081272 | 72.067398 | 7403 |
| NORMAL | 52.634069 | 5.797238 | 35.230411 | 72.067959 | 7403 |

## Quick Interpretation

- This experiment isolates **encoder+decoder reconstruction upper bound** because transport is bypassed (`GT latent -> decode`).
- The upper bound is high at all timepoints (especially D20/NORMAL and later), so the current encoder-decoder pair is effective in principle.
- Using the latest D1 full-val `best.pt` as reference (from `review/0421/Server/d1_experiment_report_20260421.md`), the gap to ceiling is:
  - D50: `+4.030809 dB`
  - D20: `+10.711663 dB`
  - D10: `+12.649411 dB`
  - D4: `+14.685101 dB`
  - NORMAL: `+16.184410 dB`
- Therefore, the dominant error is not stage1 decode capacity itself; it mainly comes from predicted latent trajectory quality (transport + accumulation).

## Artifacts

- Summary JSON: `review/0422/exp/gt_latent_decoder_ceiling_clip3_val_summary.json`
- Per-slice CSV: `review/0422/exp/gt_latent_decoder_ceiling_clip3_val_per_slice.csv`
- Run log: `review/0422/exp/gt_latent_decoder_ceiling_val_gpu2_20260422_1632.log`
- Eval script: `scripts/eval_gt_latent_decoder_ceiling_clip3.py`
