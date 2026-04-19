# Research Review (Round 2): D1 / T1 / Architecture / API

## Scope
- Review date: 2026-04-19
- Reviewed code snapshot: `origin/foc_lite_hop0@a6dafa5`
- Review workspace: `/tmp/petlr_review_latest`
- Methods:
  - Local code+config audit (architecture/design/interface chain)
  - External reviewer (`gpt-5.4`, xhigh) critical pass

## What changed since previous review
- `extended_seam_loss` is now wired into active image loss path via config flags.
- D1 config now enables:
  - `loss.image_aux.use_extended_seam: true`
  - `loss.image_aux.seam_zone_width: 3`
  - `loss.image_aux.seam_weight: 0.20`
- SeamRefiner GroupNorm now uses safer group count logic (`min(8, hidden_channels)`).
- Resume load changed from strict model load to permissive (`strict=False`) with `unexpected_keys` hard-fail.

## Findings (severity ordered)

### High
1. **Baseline framing mismatch risk**  
   V6 D1/T1 configs are C-based variants, not explicit A+C stacked variants.  
   If narrative claims are “on top of A+C”, configs should encode that directly.

2. **Resume compatibility policy is still too permissive**  
   `load_state_dict(..., strict=False)` accepts all missing keys; only `unexpected_keys` fail.  
   This can hide unintended architecture drift during resume and confuse optimizer-state restore.

3. **D1 is fused into decode path, causal attribution must be split**  
   `decode_crop()` always applies refiner when enabled; this affects train-time image aux, val chain decode, and eval decode consistently.  
   Good for deployment-quality reporting, but transport-causal claims need raw/refined split outputs.

### Medium
1. **D1 checkpoint selection objective may not match D1 goal**  
   Current best model selection remains chain-MSE centric; visual seam objective is not a first-class checkpoint selector.

2. **Seam metric naming drift**  
   With extended seam enabled, logged `seam` now reflects extended seam loss; if docs still reference seam_consistency for gate, interpretation can drift.

3. **T1a/T1b design quality is good (clean one-factor sweep)**  
   `lam0_18` vs `lam0_25` differs only by `run_name` and `image_aux.lambda_max`.

4. **A (alignment) gain remains small in current full-val evidence**  
   Treating A as firm mainline still needs stronger multi-seed support.

### Low
1. **GroupNorm “safe” comment is stronger than implementation**  
   `min(8, hidden_channels)` avoids `<8` crash, but does not guarantee divisibility for all channel choices (e.g., 10, 12).

## API/Interface call-chain check
- `config.training.image_aux.lambda_*` -> schedule in `train_first_hop.py`
- `config.loss.image_aux.*` -> `compute_first_hop_image_loss(...)` call args
- `config.first_hop.seam_refiner.*` -> module instantiation and `decode_crop()` behavior
- eval path is consistent: `sample_chain_first_hop(...)` -> `model.decode_crop(...)` -> `calc_psnr_clip3(...)`

## Recommended minimal next-step package (highest value / GPU-week)
1. **No-GPU code hardening first**
   - Add raw-vs-refined decode switch for fair causal reporting.
   - Restrict resume missing-key allowance to known new-module prefixes.
   - Add optimizer/scaler reset path when architecture signature changes.
2. **Run one D1 pilot (10k–15k)**
   - Report raw/refined PSNR and visual seam panel.
3. **Run T1a first (`lambda_max=0.18`)**
   - Gate T1b (`0.25`) on non-regression and full-val gain.
4. **Reconfirm A with at least one extra seed**
   - Decide whether A remains mainline based on stable effect size.

## Verdict
- **Revise** (not reject): implementation is substantially improved and runnable, but claim-to-evidence coupling and resume robustness should be tightened before large-scale expansion.

## Engineer Checklist (Round 2 follow-up)
1. Decode raw/refined separation:
   - Split decode API into raw path + optional refiner path.
   - Eval supports `decode_mode={raw,refined,both}` and outputs both summaries.
2. Resume hardening:
   - Missing-key whitelist only for known new modules.
   - On architecture mismatch: warm-start mode, reset optimizer/scaler/step/best state.
3. D1 selection target alignment:
   - Add seam-oriented validation metrics and checkpoint criterion.
   - Optionally save `best_chain.pt` and seam-oriented `best.pt`.
4. Reporting contract:
   - Transport headline only on `D20/D10/D4/NORMAL`.
   - D50 explicitly labeled as input-decode baseline.
   - D1必须提供 raw/refined 同 checkpoint 对照。
