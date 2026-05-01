# V6 / V6.1 Latest Training Log Viewer Update

Date: 2026-05-01
Role: viewer / reviewer

## Evidence Sources

- Gitee/head: `69ffeb3 docs(0430): add latest v6 log snapshots`
- V6 log: `review/0430/operator/log_snapshots/v6_transport_first_gpu1_train_log_snapshot_20260501_132357.log`
- V6.1 log: `review/0430/operator/log_snapshots/v6_1_rollout_floor_gpu3_train_log_snapshot_20260501_132357.log`
- Prior reviewer baseline: `review/0430/reviewer/v6_v6_1_effect_viewer_analysis_20260430.md`
- Literature / architecture follow-up: `review/0430/reviewer/literature_architecture_image_aux_review_20260501.md`

All numbers below are rolling-window validation metrics unless explicitly stated otherwise. They are useful for monitoring and checkpoint selection, but they are not a replacement for full-val clip3 PSNR or direct artifact inspection.

## Revised Evaluation Frame After User Clarification

V3 should not be treated as a clean mechanism baseline. Its strong PSNR is useful as a quality floor / reference, but previous analysis indicates that much of its optimization pressure was carried by `image_aux`, so it does not prove that the transport mechanism itself learned the desired dose-to-dose dynamics.

`image_aux` should also not be treated as a negative component. Its role is clinically important: it was added on the D50-to-D20 stage to suppress decoder artifacts, especially visible block-to-block texture misalignment and local structural discontinuity that can make generated PET images hard for doctors to trust. The failure mode is not "image_aux exists"; the failure mode is "image_aux dominates so much that good PSNR can be obtained without a convincing transport mechanism."

Therefore the correct target for V6/V6.1 is a three-way balance:

1. Preserve or approach V3-level full-val clip3 PSNR.
2. Shift the main learning signal toward transport / chain behavior rather than image-only correction.
3. Keep enough D50-to-D20 `image_aux` pressure to control decoder block artifacts and structural inconsistency.

### Evidence Check From Docs / Review Archive

I re-read the relevant archived docs and review notes after the user clarification. They support the revised frame above:

- `docs/background_first_hop_design.md` explicitly identifies `D50 -> D20` first-hop latent transport quality as the core problem, describes it as the hardest retained 4-hop step, and keeps the rollout state latent-only.
- `docs/main.md` defines `L_img_hop0` as a D50-to-D20-only image auxiliary loss through a frozen decoder, with gradients allowed to flow back to `z_pred` and the transport backbone, while forbidding image/pixel as a second rollout state.
- `docs/experiment_plans/PLAN_V6_D1_SeamRefiner_T1_LambdaSweep_20260419.md` separates two orthogonal goals: artifact removal via seam/refiner-style visual checks, and PSNR improvement via transport/lambda experiments. It also states that decoder gap is small for PSNR but patch artifacts are a clinical blocker.
- `docs/experiment_plans/PLAN_V3_Seam_Stress_and_Hop0_Attribution.md` attributes seam stress to hop0 representation loss from noisy D50 input and proposes D50-to-D20 / D50-to-NORMAL / D20-to-NORMAL stratified checks.
- `review/0429/e1_error_budget_decomposition_explained.md` shows that D50-to-D20 already has a large decoded transport discrepancy (`gap_transport=10.74 dB`, `MSE_pred_gt/MSE_ceil=11.86x`), while later targets show cumulative exposure-bias growth.
- `review/plan/pixel_prior_in_latent_transport.md` says the current `image_aux` direction is correct but too narrow/weak to reshape the velocity field by itself; it also warns that simply increasing `image_aux` can encourage smoothing or conflict with velocity MSE.

This means "D50-to-D20 is the largest transport loss" should be read as: D50-to-D20 is the primary single-hop bottleneck / first transport shock. The later NORMAL row can show a larger cumulative gap because it contains accumulated errors from the whole chain.

### PSNR / Artifact Responsibility Map

The mapping "transport mechanism corresponds to PSNR, image_aux corresponds to artifacts" is accurate as a dominant-responsibility model, but not as a strict causal separation.

| Component | Primary responsibility | Also affects | Failure mode if over/under-used |
|---|---|---|---|
| Transport / rollout / pair velocity | Latent trajectory correctness and most of the PSNR headroom | Artifact severity through off-manifold decoded latents | Weak transport gives low PSNR and cumulative chain drift |
| `image_aux` on D50-to-D20 | Decoder artifact control, local structure, seam/border consistency | PSNR through L1/SSIM/seam terms and decoder-Jacobian pullback | Too weak leaves block texture mismatch; too strong or poorly shaped can smooth PET intensity structure |

So the desired state is not `image_aux` minimization. It is transport-led learning with `image_aux` present enough to suppress block-boundary texture misalignment and prevent clinically unacceptable smoothing/structure loss.

### Experiment Logic vs Architecture Risk

The current experiment design has no obvious logical contradiction: V6 tests whether stronger transport / delayed rollout can recover PSNR headroom, while V6.1 tests whether a small rollout floor is needed to avoid early chain blindness. This is a coherent top-down sequence: first find whether the effect exists, then attribute which component caused it.

However, this does not mean the architecture is problem-free. The main unresolved architecture risks are:

| Risk | Why it matters | Current evidence | What would falsify / confirm it |
|---|---|---|---|
| Latent transport is optimized in a decoder-non-isometric space | Small latent errors can decode into large PET intensity or texture errors | E1 shows decoded transport discrepancy dominates PSNR gap | Full-val PSNR plus decoded D20/NORMAL inspection on V6 best |
| `image_aux` is hop0-only | It can control D50-to-D20 artifacts, but hop1-3 remain pixel-blind | `docs/main.md` keeps `L_img_hop0` only on D50-to-D20 | If NORMAL artifacts remain despite good hop0, multi-hop pixel consistency may be needed |
| Pair loss and rollout loss optimize different execution modes | GT-input pair quality can look good while open-loop chain drifts | V6/V6.1 early phases both show low pair loss but degraded chain metrics | Stable rolling + full-val chain quality after ramp confirms rollout is sufficient |
| Strong pixel auxiliary can smooth structure | It may improve low-level losses while erasing diagnostic PET intensity structure | Pixel-prior review warns against simply increasing `image_aux` | Visual/local-structure checks and D20 PSNR decide whether smoothing is occurring |
| No settled artifact metric exists | A numeric PSNR win can still be clinically unacceptable | Existing plans rely on visual, seam, boundary, and stress-test proxies | A fixed artifact panel or seam/boundary proxy suite is still needed |

So the architecture question is still open. The current architecture is acceptable as the next experimental vehicle, not yet proven as the final architecture. If V6 full-val PSNR approaches or beats V3 while artifact inspection passes, the architecture is provisionally validated. If PSNR improves but block artifacts or smoothing remain, the likely next architectural changes are pixel-aware transport weighting, LPIPS-augmented `image_aux`, or multi-hop pixel consistency rather than only retuning `lambda_img`.

## Executive Verdict

V6 has materially improved since the 04-30 snapshot. It is now at step 129.8K, with the best rolling checkpoint moved from 81.6K to 116K. The best V6 rolling NORMAL MSE is now `1.73e-4`, only about 5.2% worse than the known V3 best rolling NORMAL MSE `1.645e-4`. Because V3 is a PSNR/artifact-quality reference rather than a clean transport-mechanism reference, this comparison should be read as: V6 is approaching the old quality floor while using a more transport-led optimization mix.

V6 is not yet a proven V3 win. Its latest window at 129.6K is `NORMAL=2.05e-4`, worse than its own best, and bad windows still appear after 100K. But the best-window sequence keeps advancing, and the late-stage mean is much better than early Phase I. The correct action is to continue V6 and run full-val on the current best checkpoint around 116K, plus later best checkpoints if new bests appear.

V6.1 remains an early-phase control, not a replacement. It is at step 37.7K with `alpha=0` and `lambda_roll=0.05`, so it has not entered the true rollout ramp. The rollout floor improves the entire 0-40K regime versus V6 at aligned steps, but V6.1's best is still stuck at 12K and the latest window is poor. This says the floor helps but does not by itself solve Phase-I chain drift.

## Raw Latest / Best Table

| Experiment | Latest train step | alpha | lambda_roll | pair_frac | roll_frac | img_frac | Latest val step | Latest select | Latest NORMAL | Best step | Best select | Best NORMAL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V6 | 129,800 | 0.798 | 3.192 | 0.507 | 0.407 | 0.086 | 129,600 | 0.000722 | 0.000205 | 116,000 | 0.000611 | 0.000173 |
| V6.1 | 37,700 | 0.000 | 0.050 | 0.845 | 0.008 | 0.147 | 37,600 | 0.002160 | 0.000763 | 12,000 | 0.000762 | 0.000213 |

Immediate reading:

- V6 has moved into a transport/rollout-led mixed regime: recent train fractions are roughly pair 45-51%, rollout 38-41%, image 9-17%. This is healthier than V3's image-dominated behavior, but the remaining image fraction is not a flaw by itself because it may be needed for artifact control.
- V6.1 is still pair-led and pre-ramp: rollout remains about 1% of weighted pressure on average.
- V6's best is now close enough to V3 that full-val is justified; V6.1 is not yet at a decision point.

## V6: Late Ramp Is Working, But With High Window Volatility

### Rolling validation by stage

| V6 window | Val events | Mean select | Best step | Best select | Mean NORMAL | Best NORMAL |
|---|---:|---:|---:|---:|---:|---:|
| 0-20K | 52 | 0.001211 | 6,000 | 0.000772 | 0.000380 | 0.000224 |
| 20-40K | 50 | 0.001983 | 35,200 | 0.001273 | 0.000842 | 0.000525 |
| 40-50K | 25 | 0.001452 | 46,800 | 0.000953 | 0.000537 | 0.000345 |
| 50-60K | 25 | 0.001163 | 58,400 | 0.000825 | 0.000399 | 0.000276 |
| 60-70K | 28 | 0.001044 | 69,600 | 0.000686 | 0.000319 | 0.000209 |
| 70-80K | 27 | 0.000985 | 75,600 | 0.000678 | 0.000278 | 0.000195 |
| 80-90K | 28 | 0.000955 | 86,800 | 0.000644 | 0.000268 | 0.000190 |
| 90-100K | 26 | 0.000892 | 92,800 | 0.000629 | 0.000249 | 0.000182 |
| 100-110K | 25 | 0.000923 | 104,400 | 0.000634 | 0.000253 | 0.000178 |
| 110-120K | 27 | 0.000934 | 116,000 | 0.000611 | 0.000255 | 0.000173 |
| 120-130K | 25 | 0.000864 | 127,600 | 0.000633 | 0.000238 | 0.000178 |

### Best progression

The log's `new best` records show real progress rather than one isolated lucky window:

| Step | Best select when reached |
|---:|---:|
| 63,600 | 0.000741 |
| 68,000 | 0.000729 |
| 69,600 | 0.000686 |
| 75,200 | 0.000679 |
| 75,600 | 0.000678 |
| 81,200 | 0.000669 |
| 81,600 | 0.000661 |
| 86,800 | 0.000644 |
| 92,800 | 0.000629 |
| 110,000 | 0.000613 |
| 116,000 | 0.000611 |

This pattern supports the idea that V6's rollout ramp is gradually repairing open-loop chain composition.

### Key trajectory points

| V6 step | select | NORMAL | D20 | tail | rollout total |
|---:|---:|---:|---:|---:|---:|
| 50,000 | 0.001157 | 0.000423 | 0.000279 | 0.000329 | 0.000754 |
| 60,000 | 0.000928 | 0.000313 | 0.000251 | 0.000270 | 0.000909 |
| 70,000 | 0.000688 | 0.000196 | 0.000240 | 0.000203 | 0.000879 |
| 80,000 | 0.000874 | 0.000251 | 0.000297 | 0.000258 | 0.000663 |
| 90,000 | 0.001087 | 0.000314 | 0.000362 | 0.000322 | 0.000708 |
| 100,000 | 0.000831 | 0.000230 | 0.000294 | 0.000247 | 0.000840 |
| 110,000 | 0.000613 | 0.000173 | 0.000206 | 0.000183 | 0.000585 |
| 112,800 | 0.001699 | 0.000441 | 0.000679 | 0.000501 | 0.001271 |
| 120,000 | 0.000698 | 0.000196 | 0.000242 | 0.000208 | 0.000535 |
| 124,000 | 0.001112 | 0.000303 | 0.000408 | 0.000330 | 0.001002 |
| 129,600 | 0.000722 | 0.000205 | 0.000243 | 0.000215 | 0.000800 |

Interpretation:

- V6's floor has risen dramatically from the 20-40K collapse, and its best windows now approach the V3 rolling quality reference.
- Volatility remains the main concern. Bad windows still occur at 112.8K and 124.0K, which prevents a clean "solved" conclusion.
- The late 120-130K mean (`NORMAL=2.38e-4`) is not as good as the best checkpoint, but it is substantially better than the pre-ramp mean and suggests no global divergence.
- Recent gradients increase as rollout pressure increases, but the observed values do not indicate an obvious explosion in this snapshot.

## V6.1: Rollout Floor Helps, But Does Not Close Phase I

### V6.1 stage summary

| V6.1 window | Val events | Mean select | Best step | Best select | Mean NORMAL | Best NORMAL |
|---|---:|---:|---:|---:|---:|---:|
| 0-20K | 53 | 0.001126 | 12,000 | 0.000762 | 0.000325 | 0.000213 |
| 20-40K | 45 | 0.001620 | 21,600 | 0.000978 | 0.000601 | 0.000309 |

V6.1's `new best` records stop at 12K:

| Step | Best select when reached |
|---:|---:|
| 400 | 0.000786 |
| 5,600 | 0.000781 |
| 6,000 | 0.000770 |
| 12,000 | 0.000762 |

### Aligned V6 vs V6.1 comparison

| Target step | V6 select | V6 NORMAL | V6.1 select | V6.1 NORMAL | V6.1/V6 select | V6.1/V6 NORMAL |
|---:|---:|---:|---:|---:|---:|---:|
| 12,000 | 0.000816 | 0.000249 | 0.000762 | 0.000213 | 0.934 | 0.855 |
| 16,000 | 0.001141 | 0.000401 | 0.000959 | 0.000283 | 0.840 | 0.706 |
| 20,000 | 0.002751 | 0.000970 | 0.002205 | 0.000621 | 0.802 | 0.640 |
| 24,000 | 0.002378 | 0.001033 | 0.001708 | 0.000580 | 0.718 | 0.561 |
| 28,000 | 0.002104 | 0.000929 | 0.001792 | 0.000729 | 0.852 | 0.785 |
| 32,000 | 0.002686 | 0.001215 | 0.002207 | 0.000911 | 0.822 | 0.750 |
| 36,000 | 0.002126 | 0.000867 | 0.001769 | 0.000642 | 0.832 | 0.740 |
| 37,600 | 0.002563 | 0.001018 | 0.002160 | 0.000763 | 0.843 | 0.750 |

Interpretation:

- V6.1 is consistently better than V6 at the same pre-ramp stage after 12K.
- The improvement is meaningful: 20-40K mean NORMAL improves from V6 `8.42e-4` to V6.1 `6.01e-4`, about 28.6% lower.
- But V6.1 still degrades after 20K and has not made a new best after 12K. The floor mitigates strict-V6's drift; it does not remove the need for the 50K alpha/lambda ramp.

## Hypothesis Assessment

The latest logs strengthen the transport-first hypothesis in a specific, bounded way:

1. Pair-heavy Phase I alone is not enough. Both V6 and V6.1 can have low pair validation loss while chain metrics degrade.
2. Chain supervision is necessary. V6 starts to recover after rollout pressure and alpha ramp turn on; the best sequence after 63.6K is the strongest evidence.
3. A small rollout floor is helpful but insufficient. V6.1 improves same-stage early chain quality, yet still trends worse through 37.6K.
4. Image auxiliary is no longer dominating V6 late training. At 120-130K, image fraction averages about 17.1%, while rollout averages about 37.7% and pair about 45.2%. This is a better mechanism mix than V3, but the remaining image pressure should be preserved unless artifact checks show it is unnecessary.

Therefore, the right conclusion is not "V6.1 beats V6". The right conclusion is: V6 is currently the leading candidate because it has reached the effective ramp region and produced near-V3 rolling quality with a more transport-led supervision mix; V6.1 is a useful design correction that should be allowed to reach 50K-80K before judging whether it should replace V6.

## Artifact-Control Implications

The next evaluation should not rank checkpoints by PSNR or rolling chain MSE alone. A checkpoint that improves transport metrics but reintroduces block texture misalignment would be clinically weaker than the numbers imply. Conversely, a checkpoint with slightly lower transport purity but visibly cleaner D50-to-D20 texture alignment may be preferable. Both artifact modes are unacceptable: block-boundary texture mismatch can directly fail clinical review, while PET intensity flattening / over-smoothing can erase diagnostic structure and usually hurts PSNR as well.

For checkpoint selection, each candidate should be reviewed on:

1. Full-val clip3 PSNR, especially NORMAL and D20.
2. Chain/open-loop metrics, especially whether NORMAL quality holds without GT-input correction.
3. D50-to-D20 artifact behavior, with attention to block-boundary texture discontinuity, local structural misalignment, PET intensity flattening, and over-smoothing.
4. `image_aux` fraction as a balance indicator, not a metric to minimize blindly.

## Recommendations

1. Run full-val clip3 PSNR for V6 best around step 116K immediately if the checkpoint exists. This is now justified because V6 best rolling NORMAL is within about 5.2% of the V3 quality reference.
2. Keep V6 running through at least 150K. It is still in the ramp region, and alpha/lambda have not reached their final values.
3. Track whether V6 can produce a new best after 130K and whether the latest-window mean stabilizes closer to the best. If latest windows remain much worse than best, checkpoint selection becomes important.
4. Keep V6.1 running through 50K and then 80K. The decisive question is whether the floor plus ramp combines to avoid V6's 20-40K collapse and then recovers faster after 50K.
5. Do not claim final success from rolling metrics. The immediate paper/claim-level arbiter should combine full-val clip3 PSNR, chain metrics, and artifact inspection on V6 best, V6 latest, and later V6.1 best once it reaches comparable ramp stages.
