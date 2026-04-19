# Review: D1 / T1a / T1b and Overall Architecture (2026-04-19)

## Scope
- Reviewed latest remote branch: `origin/foc_lite_hop0`
- Latest commit at review time: `d7a52f8`
- Review target:
  - Overall architecture (A+C as current mainline)
  - D1 (SeamRefiner)
  - T1a/T1b (`image_aux.lambda_max` sweep)

## Key Findings (by severity)

### High
1. **Plan-implementation mismatch for D1 seam objective**
   - `extended_seam_loss` is added, but active image loss path still uses `seam_consistency_loss`.
   - Evidence:
     - `extended_seam_loss` defined in `pet_lr/losses.py`
     - `compute_first_hop_image_loss()` still calls `seam_consistency_loss()` in `pet_lr/losses_first_hop.py`
   - Risk: D1 is not testing the claimed seam objective.

2. **D1 and T1 are not the same causal axis**
   - D1 is decode-side post-refinement; T1 is transport-side supervision strength sweep.
   - Directly comparing “D1 vs T1 gains” as if equivalent can be misleading.
   - Correct comparison should be each vs same A+C baseline.

### Medium
1. **Parallel run is operationally OK but needs evaluation protocol discipline**
   - D1/T1 can run in parallel on separate GPUs.
   - But report should avoid mixing conclusions: visual seam quality claims vs transport PSNR claims.

2. **T1 design is controlled and clean**
   - `lam0_18` vs `lam0_25` differs only in `run_name` and `lambda_max`.
   - Good one-factor ablation design.

3. **Overall architecture direction is reasonable**
   - A+C mainline remains coherent with E1/E2 diagnosis (transport-dominant bottleneck).
   - D1 can be justified as a clinical visual quality branch if evaluated with dedicated seam metrics.

## Verdict
- **Overall**: `Revise then run` (not reject).
- **Why**:
  - Architecture direction is reasonable.
  - But D1 currently has a code-path mismatch against stated seam-loss design.

## Minimum Fix Before Large-Scale Runs
1. Wire D1 to actual seam objective (or explicitly downgrade claim to “SeamRefiner + legacy seam loss”).
2. Keep T1 protocol unchanged (already controlled).
3. Report in two lanes:
   - Transport lane: full-val PSNR on D20/D10/D4/NORMAL.
   - Visual lane: seam-specific metrics + qualitative panels.

## Recommended Minimal Experiment Package (high decision value)
1. A+C baseline (reference run, fixed eval protocol).
2. T1a (`lambda_max=0.18`) only.
3. D1 only (after loss-path fix or explicit claim downgrade).
4. T1b (`lambda_max=0.25`) only if T1a is non-regressive.

## Update (same day, latest commit)
- New remote head moved to `origin/foc_lite_hop0@a6dafa5` (`fix: resolve all 4 audit issues`).
- The previous top issue (extended seam loss not wired) has been fixed.
- A follow-up review note is recorded in:
  - `exp_result/0419/research_review_D1_T1_arch_api_a6dafa5_20260419.md`
