# Literature / Architecture / Image-Aux Review

Date: 2026-05-01
Role: viewer / reviewer

## 1. Executive Judgment

The current V6/V6.1 experiment logic is still coherent, but it does not close the architecture question. The strongest current interpretation is:

1. `D50 -> D20` is the dominant single-pair / first-hop difficulty. Project docs support this through both decoded PSNR-gap evidence and pair-level velocity-scale evidence.
2. A pair-first then rollout schedule is a reasonable constraint, but strict pair-only for too long risks GT-manifold overfitting and chain blindness. V6.1's rollout floor is a valid correction that still respects the pair-first principle.
3. The current `image_aux` implementation is weaker and narrower than the phrase "image auxiliary" may imply. It is hop0-only, low-weight, low-level, and by default its seam term is prediction self-continuity rather than GT seam alignment.
4. If V3 has no obvious visual block artifact but complex/high-uptake regions have low PSNR, then the present bottleneck is probably not gross seam artifacts. It is more likely high-SUV local intensity / texture recovery under transport error and decoder off-manifold sensitivity.

## 2. Recent Literature Signals Since 2024

Semantic Scholar was attempted but hit HTTP 429 rate limits without an API key. The table below uses PubMed E-utilities and arXiv pages, with DOI/PMID/arXiv metadata verified from those sources.

| Paper / source | Venue / year | Main idea | Evaluation signals | Relevance to our problem |
|---|---|---|---|---|
| Hashimoto et al., "Deep learning-based PET image denoising and reconstruction: a review" | Radiological Physics and Technology, 2024, PMID 38319563, DOI `10.1007/s12194-024-00780-3` | Reviews PET DL methods as post-processing denoising, direct reconstruction, and iterative reconstruction with neural enhancement | Emphasizes PET-specific reconstruction context rather than generic image metrics only | Supports treating our latent transport as only one part of a reconstruction chain; evaluation should include clinical and PET-quantitative criteria |
| Zhang et al., "Deep Generalized Learning Model for PET Image Reconstruction" | IEEE TMI, 2024, PMID 37428658, DOI `10.1109/TMI.2023.3293836` | Integrates deep learning with ADMM-style iterative optimization; notes pure data-driven DL can blur fine structures | Simulated and real PET, qualitative and quantitative comparisons | Directly matches our concern: pure learned mapping / low-level losses can degrade fine structure, so complex-region PSNR deserves its own metric |
| Xie et al., "Joint diffusion: mutual consistency-driven diffusion model for PET-MRI co-reconstruction" | Physics in Medicine and Biology, 2024, PMID 38981592, DOI `10.1088/1361-6560/ad6117` | Uses PET/MRI data fidelity plus a joint score-based diffusion prior for mutual consistency | Image quality via co-reconstruction consistency and data fidelity | Supports adding consistency constraints rather than relying on image-space visual cleanup alone |
| Dedja et al., "Sequential deep learning image enhancement models improve diagnostic confidence, lesion detectability, and image reconstruction time in PET" | EJNMMI Physics, 2024, PMID 38488923, DOI `10.1186/s40658-024-00632-4` | Sequential DL enhancement for PET images | Reader Likert scores, lesion detectability, diagnostic confidence, noise VOIs | Strong support for a reviewer/reader-style quality axis in addition to PSNR |
| Hopson et al., "Deep Convolutional Backbone Comparison for Automated PET Image Quality Assessment" | IEEE TRPMS, 2024, PMID 39404656, DOI `10.1109/TRPMS.2024.3436697` | Predicts PET image quality from pretrained backbones | Clinical quality labels: global quality rating, pattern recognition, diagnostic confidence | Suggests a future learned PET-IQA proxy may be useful for artifact/quality triage |
| Yang et al., "Investigation of PET image quality with acquisition time/bed and enhancement of lesion quantification accuracy through deep progressive learning" | EJNMMI Physics, 2024, PMID 38195785, DOI `10.1186/s40658-023-00607-x` | Deep progressive learning for PET quality and lesion quantification | Visual score, sharpness, noise, diagnostic confidence, SBR, SNR, CBR, CNR, SUVmax | Gives a concrete evaluation template for our complex/high-uptake region problem |
| Muller et al., "Image Denoising of Low-Dose PET Mouse Scans with Deep Learning" | Molecular Imaging and Biology, 2024, PMID 37875748, DOI `10.1007/s11307-023-01866-x` | Low-count PET denoising validation against filtering baselines | Visual and quantitative comparison over count fractions | Supports dose-fraction stratified evaluation rather than only whole-val averages |
| Ai et al., "RED: Residual Estimation Diffusion for Low-Dose PET Sinogram Reconstruction" | arXiv 2024, `2411.05354` | Uses residual between low-dose and full-dose sinograms instead of Gaussian noise; adds drift correction for reverse-process stability | Low-dose sinogram and reconstruction quality | Highly relevant conceptually: residual path + drift correction targets accumulated prediction errors, similar to our rollout drift issue |
| Hashimoto & Gong, "PET Image Reconstruction Using Deep Diffusion Image Prior" | arXiv 2025 / IEEE TMI 2026 reference on arXiv, `2507.15078` | Alternates diffusion sampling and model fine-tuning guided by PET sinogram; uses HQS to decouple optimization | Simulation + clinical datasets, tracer/scanner generalization | Reinforces data-consistency / prior-guided reconstruction as a stronger architecture direction if pure latent transport saturates |

## 3. What The Literature Suggests For Our Evaluation

The 2024+ PET literature does not support using PSNR alone as the arbiter. The most relevant evaluation additions are:

| Evaluation axis | Concrete metric for this project | Why it matters |
|---|---|---|
| Full-image fidelity | existing full-val clip3 PSNR by D20/D10/D4/NORMAL | Claim-level continuity with prior V3/V6 reports |
| High-uptake / complex ROI fidelity | ROI-PSNR on GT high-SUV masks, e.g. top intensity percentile or SUV-threshold masks after clip3 conversion | Directly tests the user's observation that complex regions have low PSNR despite clean visual appearance |
| Lesion / hotspot quantification | SUVmax/SUVmean error, SBR, CBR, SNR, CNR | Mirrors EJNMMI PET evaluation patterns and is clinically more meaningful than whole-image PSNR |
| Texture / local structure | gradient-mask PSNR or high-gradient MAE; optional SSIM/MS-SSIM in high-uptake mask | Catches over-smoothing and internal texture collapse |
| Seam / periodic artifacts | 14px boundary ratio, extended seam loss, FFT spike around period 14 | Useful as a rule-out metric when visible block artifacts are absent |
| Reader-like quality | small PET-IQA model or human Likert panel: global quality, pattern recognition, diagnostic confidence | Matches recent PET IQA work and avoids over-trusting pixel metrics |

For the current question, the most urgent addition is not another seam-only metric. It is high-uptake / complex-region PSNR and lesion-style SUV/contrast metrics.

## 4. D50 -> D20 Evidence Check

The claim that `D50 -> D20` is the largest single-pair difficulty is supported, with one nuance.

Evidence from project docs:

- `review/plan/transport_breakthrough_research_v6.md` states that pair loss should be hop0-heavy because `D50 -> D20` has `v_std=0.009634` and `CeilGap=11.07 dB`.
- The same V6 plan gives rollout effective signal estimates: hop0 `v_std=0.0096`, hop1 `0.0029`, hop2 `0.00078`, hop3 `0.00014`. Thus the first pair has about `3.3x`, `12.4x`, and `68x` larger velocity scale than hop1/2/3 respectively.
- `docs/experiment_plans/PLAN_V5_PostE1E2_SchemeCA_20260417.md` reports D20 decoded error budget: decoder ceiling `46.64 dB`, E2E `35.49 dB`, total gap `11.15 dB`, transport gap `10.74 dB`, decoder gap `0.41 dB`.
- `review/0429/e1_error_budget_decomposition_explained.md` converts this to MSE ratios: D20 E2E error is `13.03x` the decoder ceiling error; decoded transport discrepancy is `11.86x` the decoder ceiling error.

Nuance:

- Later rows such as NORMAL can have larger cumulative decoded gap, but that is chain accumulation / exposure bias, not evidence that the single pair `D4 -> NORMAL` is harder than `D50 -> D20` in pair velocity scale.

Therefore, the project should continue to treat `D50 -> D20` as the first-hop pair bottleneck, while separately treating NORMAL as the cumulative rollout endpoint.

## 5. Pair-First Then Roll: Agreement And Different View

I agree that a pair-first then rollout schedule is the right high-level constraint for the current codebase. My different view is about the word "first": it should mean pair-dominant first, not necessarily rollout-zero first.

Why strict rollout-zero is risky:

- Pair loss trains GT-input velocity. It does not expose the model to its own predicted intermediate latents.
- V6/V6.1 logs already show the pattern: pair metrics can look good while open-loop chain metrics degrade.
- The RED diffusion paper's drift-correction logic is analogous: long iterative/reverse processes need explicit mechanisms against accumulated drift.

Alternatives that still respect pair-first:

| Method | Does it violate pair-first? | Implementation idea | Expected benefit |
|---|---|---|---|
| Rollout floor | No | V6.1-style `lambda_roll=0.05` from step 1 | Prevents complete chain blindness while keeping pair dominant |
| Shorter pair-only warmup | No | Start rollout ramp at 10K-20K instead of 50K | Reduces time spent overfitting GT-input manifold |
| Pair-stage pixel-aware velocity weighting | No | Reweight pair loss tokens/regions by GT pixel delta or high-SUV mask | Injects image importance into the main pair channel rather than only `image_aux` |
| Pair-stage hard ROI mining | No | Oversample high-SUV/complex-region slices or patches in hop0 pair batches | Targets the exact V3 failure: low complex-region PSNR |
| Jacobian/perceptual pair weighting | No | Approximate decoder-sensitive latent directions or add LPIPS pullback on hop0 | Addresses decoder non-isometry without changing rollout order |
| Roll-stage stop-grad multi-hop pixel consistency | Mostly no, after ramp | Decode intermediate rollout outputs, stop-grad chain input, optimize current hop | Useful only if NORMAL artifacts remain despite good hop0 |

My recommended near-term version is conservative: keep pair-first, but prefer V6.1-style floor or a shorter warmup, and add pair-stage ROI/pixel-aware weighting only after V6/V6.1 full-val plus complex-region metrics decide the failure mode.

## 6. Actual `image_aux` Implementation Review

Current code path:

1. `train_first_hop.py::compute_hop0_image_losses` calls `model.predict_latent_step(...)` on `hop0_batch` only.
2. It decodes `out["z_pred"]` through `model.decode_crop(...)`.
3. It computes `compute_first_hop_image_loss(x_pred, x_gt=hop0_batch["x_dst"])`.
4. V6/V6.1 configs set `lambda_start=lambda_max=0.04`, with `l1_weight=1.0`, `ssim_weight=0.25`, `seam_weight=0.1`, `border_width=14`, `border_weight=2.0`.

Loss components:

- Weighted L1: border-weighted absolute pixel difference to GT.
- SSIM: 5x5-window SSIM loss over single-channel tensors with `data_range=2.0`.
- Default seam: `seam_consistency_loss(x_pred, patch_size=14)`, which penalizes prediction discontinuity and gradient mismatch across 14px grid boundaries inside the prediction.
- Extended seam exists, but V6/V6.1 do not enable `use_extended_seam`. Only extended seam compares predicted seam-zone gradients/curvature with GT.

Implications:

- Current `image_aux` can suppress obvious seam discontinuity, especially around the fixed 14px grid.
- It is not a strong complex-region fidelity loss. It has no LPIPS/perceptual term, no high-SUV ROI weighting, no lesion contrast term, no explicit SUVmax/SUVmean preservation, and no hop1-3 coverage.
- The default seam term can reward smooth self-continuity even when GT has meaningful local intensity variation. This is good for block artifacts, but not sufficient for complex PET intensity structure.

This directly explains the user's observation: if V3 visual inspection has no obvious artifact but complex-region PSNR is low, then the current `image_aux` solved the gross visual seam problem but did not solve fine-grained high-uptake intensity fidelity.

## 7. Recommended Next Experiments

### 7.1 Immediate evaluation before changing training

Run the same evaluation panel on V3 best, V6 best, and later V6.1 best:

1. Full-val clip3 PSNR by D20/D10/D4/NORMAL.
2. High-SUV ROI-PSNR on GT masks, such as top 10%, top 5%, and top 1% SUV pixels after clip3 conversion.
3. High-gradient ROI-PSNR / MAE using GT gradient-magnitude masks.
4. SUVmax/SUVmean error in connected high-uptake components or fixed top-k hotspot regions.
5. Seam rule-out metrics: default seam, extended seam, 14px boundary ratio, optional FFT period-14 spike.

Decision interpretation:

- If V6 beats V3 globally and in high-SUV ROI: current transport-first architecture is provisionally validated.
- If V6 beats global PSNR but loses high-SUV ROI: architecture needs ROI/pixel-aware pair weighting or perceptual/feature auxiliary.
- If V6 improves high-SUV ROI but seam metrics degrade: add artifact-specific seam/extended seam control.
- If all metrics improve but visual readers still reject images: add PET-IQA or reader-style scoring.

### 7.2 Training changes only after the above panel

Recommended order:

1. Keep V6/V6.1 running and run full-val plus ROI panel first.
2. If complex-region PSNR remains the main failure, try pair-stage high-SUV / pixel-delta weighted velocity loss before multi-hop pixel loss.
3. If seam metrics are bad, enable or tune `use_extended_seam`, but do not expect it to fix complex-region PSNR by itself.
4. If ROI fidelity remains bad after pair-stage weighting, test LPIPS-augmented `image_aux` as a low-cost baseline.
5. Use multi-hop pixel consistency only if later-hop decoded outputs are demonstrably pixel-blind after rollout ramp.

## 8. Bottom Line

The project is not blocked by an experiment-logic error. The sharper issue is that the current architecture evaluates and trains too weakly on the exact regions now suspected to fail: complex/high-uptake PET structures.

The immediate next move should be measurement, not another blind architecture jump: add high-SUV ROI PSNR, high-gradient local PSNR, SUV/contrast errors, and seam rule-out metrics to V3/V6/V6.1 evaluation. If those confirm that complex regions are the failure mode, the best architectural next step is pair-stage pixel/ROI-aware transport weighting, not simply increasing `image_aux` and not immediately abandoning pair-first then rollout.