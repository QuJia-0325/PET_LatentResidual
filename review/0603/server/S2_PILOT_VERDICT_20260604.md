# S2 Pilot Verdict

日期: 2026-06-04  
任务: `CODEX_TASK_S2_PILOT_20260604.md`  
执行环境: `/home/qujiaxiang/.conda/envs/rae/bin/python`, GPU0  
口径: matrix-free decoder pullback metric in clip3 SUV [0,3] domain, no training, no checkpoint writes.

## Final Verdict

- **S2.b-pilot: PASS** — `M=JᵀJ` pullback quadratic `q=δᵀMδ=||J_Gδ||²` tracks true nonlinear decode-domain error `e=||G(z_gt+δ)-G(z_gt)||²` substantially better than latent norm `l=||δ||²`.
- **S2.a-pilot: PASS** — decoder pullback metric is strongly anisotropic; top spectrum captures far more energy than a uniform latent-space baseline.
- **S2.c-pilot: FAIL** — top-M image responses do **not** concentrate on the 14px patch-grid more than random latent directions; the patch-grid causal claim is not supported.

## Quantitative Gates

| stage | gate | result | decision |
|---|---|---:|---|
| S2.b | Spearman(q,e) − Spearman(l,e) ≥ 0.15 and log-log Pearson(q,e) > 0.5 | +0.3900; 0.9998 | continue to S2.a |
| S2.a | median top-1/top-8 energy over uniform baseline ≥ 100× | top-1 1463.62×; top-8 833.10× | continue to S2.c |
| S2.c | top-M patch-grid energy / random patch-grid energy ≥ 1.20 | 0.8500 | stop patch-grid causal claim |

## Paper Wording Recommendation

Use:

> The decoder pullback metric provides quantitative support that equal latent-space errors are not equal in image space: `δᵀJ_GᵀJ_Gδ` correlates strongly with actual decoded image error, and the top spectrum of `J_GᵀJ_G` is highly concentrated.

Do **not** use:

> The high-M directions align with the 14px patch lattice, causing seam artifacts.

The S2.c pilot directly fails that claim: top-M responses have lower patch-grid energy fraction than matched random controls. Keep seam as an empirical observation or qualitative hypothesis, not as a demonstrated consequence of patch-grid-aligned top-M directions.

## Artifacts

- S2.b report: `review/0603/server/S2b_pilot_correlation_20260604.md`
- S2.a report: `review/0603/server/S2a_pilot_spectrum_20260604.md`
- S2.c report: `review/0603/server/S2c_pilot_patchgrid_20260604.md`
- Script: `tools/analyze_pullback_metric.py`
- Logs: `review/0603/server/s2b_pilot_gpu0_20260604.log`, `review/0603/server/s2a_pilot_gpu0_20260604.log`, `review/0603/server/s2c_pilot_gpu0_20260604.log`
