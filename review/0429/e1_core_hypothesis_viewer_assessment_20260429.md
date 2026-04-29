# E1 Core Hypothesis Viewer Assessment - 2026-04-29

## 0. Verdict

The core hypothesis is **mostly correct, with a necessary wording boundary**.

Supported statement:

> Under the current 224 latent pipeline and V3-family checkpoint evaluated by E1, the dominant quality gap is transport-related decoded-space discrepancy, not decoder ceiling/reconstruction capacity. Therefore, prioritizing transport quality over decoder-only improvement is scientifically reasonable.

Over-strong statement to avoid:

> The gap is 100% pure velocity-field error, or the PSNR diagnostic ratio is a strict MSE-domain linear error contribution.

The revised 0429 explanation is directionally right because it explicitly treats the result as a PSNR-domain diagnostic decomposition and adds MSE-ratio interpretation. The remaining risk is that older code comments and older planning docs used stronger language; the code comments have now been updated to match the revised interpretation.

## 1. Evidence Checked

| Source | What was checked | Viewer result |
|---|---|---|
| `review/0429/e1_error_budget_decomposition_explained.md` | revised explanation and E1/V6 implications | mostly correct |
| `scripts/diagnose_error_budget.py` | actual E1 computation | computation matches document; old comments were too strong |
| `review/0416/conclusion.md` | saved E1 raw table from full val | confirms 7403-slice E1 numbers |
| `review/0422/exp/gt_latent_decoder_ceiling_clip3_val_summary.json` | decoder ceiling values | matches E1 ceiling values |
| `pet_lr/rollout_first_hop.py` | rollout chain output order | matches E1 script usage |
| `pet_lr/model_first_hop.py` | `decode_crop` behavior | supports decoder-based comparisons |
| `configs/pet_flow/pet_flow_first_hop_224_v6_transport_first.yaml` | V6 mechanism mapping | implements transport-first changes described in the doc |
| `train_first_hop.py` | pair/roll/img weighting and logging | confirms `pair_frac`, `roll_frac`, `img_frac` are weighted-loss fractions |

## 2. Raw E1 Table

From the saved full-val E1 table (`n_eval = 7403`):

| TP | Ceiling PSNR | E2E PSNR | Gap total | Gap transport | Gap decoder | transport_fraction |
|---|---:|---:|---:|---:|---:|---:|
| D20 | 46.6356 | 35.4881 | 11.1475 | 10.7388 | 0.4087 | 0.9633 |
| D10 | 48.7436 | 35.8620 | 12.8815 | 12.3332 | 0.5483 | 0.9574 |
| D4 | 50.8319 | 36.4207 | 14.4112 | 13.9784 | 0.4328 | 0.9700 |
| NORMAL | 52.6341 | 36.7422 | 15.8918 | 15.6973 | 0.1946 | 0.9878 |

MSE-ratio check from the same dB gaps:

| TP | MSE_e2e / MSE_ceil | MSE_pred_gt / MSE_ceil | MSE_e2e / MSE_pred_gt |
|---|---:|---:|---:|
| D20 | 13.02x | 11.85x | 1.10x |
| D10 | 19.42x | 17.11x | 1.13x |
| D4 | 27.61x | 24.99x | 1.10x |
| NORMAL | 38.83x | 37.13x | 1.05x |

These numbers support the central claim: decoded-space discrepancy caused by predicted latent trajectory quality is much larger than the decoder ceiling floor.

## 3. Code Verification

The script computes the three quantities described in the 0429 document:

```python
psnr_ceil = calc_psnr_clip3(decode(z_gt), x_raw)
psnr_e2e = calc_psnr_clip3(decode(z_pred), x_raw)
psnr_pred_gt = calc_psnr_clip3(decode(z_pred), decode(z_gt))
```

Then it aggregates mean PSNR and computes:

```python
gap_total = mean(psnr_ceil) - mean(psnr_e2e)
gap_transport = mean(psnr_ceil) - mean(psnr_pred_gt)
gap_decoder = mean(psnr_pred_gt) - mean(psnr_e2e)
```

The rollout chain check is also consistent: `sample_chain_first_hop` returns `[D50, D20, D10, D4, NORMAL]` predictions, so the E1 loop compares each predicted chain point to the matching GT latent/image.

The metric contract is consistent with the project claim metric: `calc_psnr_clip3` converts `[-1, 1]` tensors to SUV, clamps to `[0, 3]`, and uses data range 3.0.

## 4. What The Hypothesis Supports

### Finding 1: Decoder capacity is not the current primary bottleneck

Observation: decoder ceiling PSNR is 46.6-52.6 dB while E2E PSNR is 35.5-36.7 dB.

Interpretation: even with a perfect latent, the decoder can reconstruct much better than the current transported latent trajectory produces.

Implication: decoder-only improvements have limited expected PSNR upside compared with improving the predicted latent trajectory.

Next step: keep decoder monitoring for artifacts, but prioritize transport-side experiments.

### Finding 2: Transport-related decoded discrepancy dominates the observable gap

Observation: `MSE_pred_gt / MSE_ceil` is 11.85x at D20 and 37.13x at NORMAL, while switching the reference from `decode(z_gt)` to `x_raw` only changes MSE by about 1.05-1.13x.

Interpretation: the large observable gap is mostly already present before comparing to raw GT; it appears when `decode(z_pred)` is compared with `decode(z_gt)`.

Implication: the decoded manifestation of latent trajectory error is the main optimization target.

Next step: sync the original E1 JSON/per-slice outputs and compute direct pixel-MSE aggregates to make the MSE-domain statement even cleaner.

### Finding 3: The result is not proof of one specific transport mechanism

Observation: E1 does not distinguish pair velocity bias, rollout exposure bias, accumulated chain drift, decoder Jacobian amplification, or off-manifold sensitivity.

Interpretation: E1 says "transport-side decoded discrepancy dominates"; it does not by itself say "FOC/integration is the cause" or "pair_weight alone fixes it."

Implication: V6's transport-first design is justified as a direction, but the individual V6 modules still need staged attribution.

Next step: combine E1 with Path A / TF-vs-rollout diagnostics and V6 Phase II metrics.

## 5. V6 Implication Check

The V6 mechanism mapping is implemented in config/code:

| E1 implication | V6 mechanism | Code/config check |
|---|---|---|
| improve GT velocity / pair quality | `loss.pair_weight=15.0`, `pair_loss_weights=[2.5,1,1,1]` | present in V6 config; applied by `train_first_hop.py` |
| reduce chain accumulation | rollout lambda 0 -> 4.0 with 50K warmup and 100K ramp | present in V6 config and rollout schedule logic |
| avoid image auxiliary domination | `image_aux.lambda_start=lambda_max=0.04` | present in V6 config; weighted fractions logged |

This means the V6 design is consistent with the E1 hypothesis. It does **not** mean V6 is already validated; current V6 Phase I still needs chain recovery after rollout ramp begins.

## 6. Caveats

| Caveat | Severity | Why it matters |
|---|---|---|
| E1 JSON/per-slice output is not synced locally | medium | local review can verify saved table and code, but cannot recompute all raw aggregates |
| dB-gap ratio is not MSE-linear contribution | high if overstated | the 96-99% number is diagnostic, not energy attribution |
| `psnr_pred_gt` is decoder-mediated | high if overstated | decoder Jacobian/off-manifold sensitivity can amplify or compress latent errors |
| mean-PSNR differences are not the same as mean pixel-MSE decomposition | medium | exact MSE-domain analysis should aggregate MSE directly |
| older docs/code comments used stronger language | medium | can lead to false claims such as "ODE integration error dominates" |

## 7. Recommended Claim Wording

Use this:

> E1 supports that the current E2E quality gap is dominated by transport-related decoded-space discrepancy rather than decoder reconstruction ceiling. In MSE-ratio terms, E2E error is roughly 13-39x the decoder-ceiling error, and `decode(z_pred)` vs `decode(z_gt)` already explains most of that scale. This justifies prioritizing transport-side improvements, while leaving the exact transport sub-mechanism to Path A/V6 staged diagnostics.

Avoid this:

> E1 proves the gap is 96-99% pure velocity-field error, or that decoder effects are mathematically zero.

## 8. Final Position

The core hypothesis is correct as a **directional bottleneck hypothesis**: optimize transport before decoder. It is not correct as a **closed causal proof** of one transport sub-mechanism. The 0429 revised explanation mostly states this boundary correctly; the remaining work is to sync raw E1 outputs and keep V6/Path A responsible for mechanism-level attribution.