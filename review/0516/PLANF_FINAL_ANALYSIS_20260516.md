# Plan-F Final Analysis (20260516)

> **⚠️ 20260516 PEER REVIEW CORRECTION**：本文件中所有“V7 − V8 = image_aux ablation”与“V7 − V6_NOISE = d_pure (seed noise)”的标注都是**错误的**。根据 [review/0511/log_snapshots_20260516_163900/configs/](../0511/log_snapshots_20260516_163900/configs/) 的 resolved config：
>
> | 实验 | step_weights | image_aux | seed |
> |---|---|---|---|
> | **V7** | **[0.6106, 2.0, 2.7041, 2.0423]** (Grönwall raw) | 0.04 | 42 |
> | **V8** | **[0.5, 2.0, 1.5, 1.0]** (V6 经验) | **off** | 42 |
> | **V6_NOISE** | **[0.5, 2.0, 1.5, 1.0]** (V6 经验) | 0.04 | **1337** |
>
> **真实说法**：
> - V7 − V8 = (image_aux on → off) × (Grönwall step_weights → V6 step_weights)
> - V7 − V6_NOISE = (Grönwall → V6 step_weights) × (seed 42 → 1337)
>
> **下文以下所有结论需重新评估**：
> - §2.1 "d_image_aux = 6.65%" —— 实际是 (image_aux+step_weights) 联合效应。
> - §2.1 "d_pure = 0.37%" —— 实际是 (step_weights+seed) 联合效应。
> - §2.2 paired t-stat / win rate 仍然有意义，但**所词描述的效果不是“image_aux 独独带来的 +0.308 dB”**。
> - §4 "cascade 单调放大”可能部分由 V7 step_weights 本身对后期 hop 加权造成，image_aux 贡献未分离。
>
> **补救实验（V13）已设计**：见 [REVIEW_INTEGRATION_20260516.md](./REVIEW_INTEGRATION_20260516.md)。
> V13 = V7 config + image_aux off (保持 Grönwall step_weights, seed=42) → V7 − V13 才是 true image_aux ablation。
> V14 = V7 config + seed=1337 → V7 − V14 才是 true d_pure。

---

- generated_at: 2026-05-16 Asia/Shanghai
- branch: foc_lite_hop0
- commit_when_generated: acc7a9d
- upstream data:
  - training snapshot: [review/0511/log_snapshots_20260516_163900](../0511/log_snapshots_20260516_163900)
  - full-val PSNR_clip3 eval: [review/0511/fullval_psnr_clip3_20260516_173941](../0511/fullval_psnr_clip3_20260516_173941)
  - per-slice CSVs: `review/0511/fullval_psnr_clip3_20260516_173941/artifacts/planf_*_per_slice.csv` (n=7403 each)
- scope: Plan-F three-run comparison (V7 = gronwall+image_aux, V8 = gronwall, no image_aux, V6_NOISE = V6 baseline with seed=1337)
- metric direction: lower MSE / higher PSNR_clip3 = better. PSNR_clip3 uses canonical `src.utils.metrics.calc_psnr_clip3` (window→3 first, then PSNR), identical to RAE evaluation.

---

## 0. TL;DR

1. All three Plan-F runs completed cleanly at step 160000. No traceback / OOM / nan-loss appears in any of the three [main_training](../0511/log_snapshots_20260516_163900/main_training) logs.
2. On full-val PSNR_clip3 with the fixed-endpoint protocol (`step_160000.pt` = `last.pt` for V7/V8/V6_NOISE), V7 strictly dominates V8 across every dose stage:
   - NORMAL PSNR_clip3: V7 36.7810 dB vs V8 36.4729 dB (Δ +0.30803 dB; paired t = 74.6 over 7403 slices; per-slice win rate 91.5%).
   - Transport-avg PSNR_clip3: V7 36.1023 vs V8 35.8421 (Δ +0.2602 dB).
3. The seed-only noise floor (V7 seed42 vs V6_NOISE seed1337, both `last.pt` step 160000, both with image_aux) is small and direction-consistent but tiny:
   - NORMAL ΔPSNR = +0.02594 dB, per-slice win rate 55.8% (≈ coin flip on individual slices).
4. Signal-to-seed-noise ratio (PSNR delta image_aux / PSNR delta pure):
   - D20 16.5×, D10 27.5×, D4 19.8×, NORMAL 11.9×.
5. Plan-F threshold ruling:
   - MSE-based: d_image_aux = 6.65%, d_pure = 0.37%. d_image_aux/d_pure = 18.0× ≫ 1.8× (relative threshold PASS), but d_image_aux < 10% (absolute floor FAIL).
   - PSNR-based: d_image_aux = 0.308 dB, d_pure = 0.026 dB. SNR 12×, well above 1.8× requirement.
   - Net: the gain is statistically overwhelming relative to seed noise but absolutely modest. Decision below in §6.
6. `best.pt` adds no value here over `last.pt`:
   - V7 best.pt = V7 last.pt (same checkpoint, step 160000). Same for V8.
   - V6_NOISE best.pt is step 156400 from rolling-val; on full-val it is slightly *worse* than V6_NOISE last.pt (NORMAL ΔPSNR = -0.0113 dB). This is direct evidence that rolling-window best selection should not be promoted to primary endpoint.
7. Recommendation: freeze V7 `step_160000.pt` as the canonical Plan-F baseline; document image_aux as a *small but reliable* improvement; do not spend further compute on image_aux ablations; reallocate compute to higher-leverage variables.

---

## 1. Raw full-val numbers (canonical calc_psnr_clip3)

From [planf_fullval_psnr_clip3_summary_20260516_173941.csv](../0511/fullval_psnr_clip3_20260516_173941/status/planf_fullval_psnr_clip3_summary_20260516_173941.csv), n_eval_slices = 7403 for every checkpoint.

### 1.1 PSNR_clip3 (dB), all stages

| tag | exp | ckpt | step | D20 | D10 | D4 | NORMAL | transport_avg |
|---|---|---|---:|---:|---:|---:|---:|---:|
| planf_v7_best | V7 | best | 160000 | 35.4354 | 35.8194 | 36.3736 | **36.7810** | 36.1023 |
| planf_v7_last | V7 | last | 160000 | 35.4354 | 35.8194 | 36.3736 | **36.7810** | 36.1023 |
| planf_v6noise_best | V6_NOISE | best | 156400 | 35.4258 | 35.8148 | 36.3575 | 36.7437 | 36.0855 |
| planf_v6noise_last | V6_NOISE | last | 160000 | 35.4217 | 35.8110 | 36.3596 | **36.7550** | 36.0868 |
| planf_v8_best | V8 | best | 160000 | 35.2094 | 35.5897 | 36.0963 | **36.4729** | 35.8421 |
| planf_v8_last | V8 | last | 160000 | 35.2094 | 35.5897 | 36.0963 | **36.4729** | 35.8421 |

### 1.2 NORMAL chain MSE (full-val, n=7403)

| tag | step | NORMAL MSE | tail chain MSE |
|---|---:|---:|---:|
| planf_v7_last | 160000 | 0.000245641 | 0.000268340 |
| planf_v6noise_last | 160000 | 0.000246547 | 0.000268958 |
| planf_v6noise_best | 156400 | 0.000247104 | 0.000269184 |
| planf_v8_last | 160000 | 0.000261979 | 0.000283550 |

### 1.3 Rolling-val "best" (training-time, 64-batch window)

| exp | rolling best select | step | rolling final select | step |
|---|---:|---:|---:|---:|
| V7 | 0.000610 | 156400 | 0.000857 | 160000 |
| V8 | 0.000650 | 156400 | 0.000907 | 160000 |
| V6_NOISE | 0.000611 | 156400 (ties 139200) | 0.000861 | 160000 |

Note: rolling-val numbers above are *not* comparable to full-val 0.000904/0.000953 select numbers in §1.1 — they come from a 64-batch sliding window that is intentionally noisy. They are kept here only to highlight how much rolling-val underestimates the true full-val select score and how much rolling-best fluctuates across steps.

---

## 2. Plan-F decision under the documented threshold

### 2.1 MSE-based threshold (Plan-F primary form)

Formulas (locked during discuss-phase, pre-result):
- d_image_aux = (V8_NORMAL_MSE − V7_NORMAL_MSE) / V7_NORMAL_MSE
- d_pure = |V7_NORMAL_MSE − V6_NOISE_NORMAL_MSE| / V7_NORMAL_MSE
- PASS iff d_image_aux ≥ max(0.10, 1.8 · d_pure)

Plugging in `last.pt` numbers from §1.2:

- d_image_aux = (0.000261979 − 0.000245641) / 0.000245641 = **0.0665** = 6.65%
- d_pure = |0.000245641 − 0.000246547| / 0.000245641 = **0.00369** = 0.369%
- 1.8 · d_pure = 0.664%
- threshold = max(0.10, 0.00664) = **0.10** (= 10.00%)
- d_image_aux ≥ threshold ? **NO (6.65% < 10%)**
- d_image_aux / d_pure = **18.03×** (≫ 1.8× relative requirement)

### 2.2 PSNR-based cross-check (paired t-test over per-slice CSVs)

Computed from per-slice CSVs (`mean ± sd`, paired V7 − other, n = 7403; positive means V7 better):

| stage | Δ(V7 − V8) PSNR | sd | t-stat | win rate | Δ(V7 − V6_NOISE) PSNR | sd | t-stat | win rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D20 | +0.22599 dB | 0.45118 | 43.10 | 85.8% | +0.01368 dB | 0.17794 | 6.61 | 54.6% |
| D10 | +0.22970 dB | 0.27303 | 72.39 | 90.0% | +0.00835 dB | 0.19990 | 3.59 | 53.5% |
| D4 | +0.27728 dB | 0.30056 | 79.38 | 91.8% | +0.01402 dB | 0.24581 | 4.91 | 54.0% |
| NORMAL | +0.30803 dB | 0.35510 | 74.63 | 91.5% | +0.02594 dB | 0.28737 | 7.77 | 55.8% |

PSNR signal-to-seed-noise ratio per stage:

| stage | d_image_aux dB | d_pure dB | ratio |
|---|---:|---:|---:|
| D20 | 0.22599 | 0.01368 | **16.52×** |
| D10 | 0.22970 | 0.00835 | **27.51×** |
| D4 | 0.27728 | 0.01402 | **19.78×** |
| NORMAL | 0.30803 | 0.02594 | **11.87×** |

Interpretation:
- t-stats 43–79 → p-value far below any reasonable threshold (effectively 0 at this sample size).
- d_image_aux/d_pure ≥ 11.9× on every stage, all ≫ 1.8× ratio.
- V7 wins on 86–92% of slices for image_aux comparison vs 53–56% for pure seed comparison — i.e. seed-only difference is a fair coin flip, image_aux difference is a near-uniform win.

### 2.3 Why MSE absolute threshold fails despite huge SNR

The Plan-F 10% absolute floor was set under the prior that image_aux either matters by ≥ 10% NORMAL MSE or is not worth a paper claim. The data say neither: image_aux moves NORMAL MSE by 6.65% (≈ 0.31 dB) with a seed noise of 0.37% (≈ 0.026 dB). The effect is real and clean but small in absolute terms.

We did not commit to a fallback before this run. Decision options below.

---

## 3. Best vs last (selection mechanism check)

Direct evidence on the rolling-best-vs-fixed-endpoint debate:

| exp | best step | best.pt NORMAL PSNR (full-val) | last step | last.pt NORMAL PSNR (full-val) | Δ (best − last) |
|---|---:|---:|---:|---:|---:|
| V7 | 160000 | 36.78095 | 160000 | 36.78095 | 0.00000 (same ckpt) |
| V8 | 160000 | 36.47292 | 160000 | 36.47292 | 0.00000 (same ckpt) |
| V6_NOISE | 156400 | 36.74371 | 160000 | 36.75501 | **−0.01130 (best is worse)** |

Findings:
- V7 and V8 have `best_select_full_eval_interval=5000`. At step 160000 the full-val branch coincidentally produced the best score, so `best.pt` and `last.pt` are literally the same file. Plan-F's secondary endpoint adds nothing here.
- V6_NOISE has no full-val best branch (no `best_select_full_eval_interval`). Its `best.pt` came from rolling-val at step 156400 and is *worse* than `last.pt` (step 160000) under fair full-val evaluation. This is a clean instance of rolling-val best selection picking the wrong checkpoint.

This validates the prior decision (Plan-F audit, round 4) to use `step_160000.pt` as the primary endpoint and treat `best.pt` only as a secondary upper-bound that, in practice, can also be wrong.

---

## 4. Cross-stage pattern (image_aux mechanism evidence)

Δ(V7 − V8) PSNR climbs monotonically along the cascade D20 → D10 → D4 → NORMAL:
- 0.22599 → 0.22970 → 0.27728 → 0.30803 dB.

And the corresponding relative MSE deltas:

| stage | V7 MSE | V8 MSE | Δ MSE | rel Δ |
|---|---:|---:|---:|---:|
| D20 | 0.000330211 | 0.000340270 | 1.0058e-5 | **+3.05%** |
| D10 | 0.000296383 | 0.000310042 | 1.3659e-5 | **+4.61%** |
| D4 | 0.000262997 | 0.000278629 | 1.5632e-5 | **+5.94%** |
| NORMAL | 0.000245641 | 0.000261979 | 1.6338e-5 | **+6.65%** |

Reading: image_aux's benefit accumulates along the rollout. The first hop (D50 → D20) is dominated by easy-to-learn structure, while the deepest target (NORMAL) is where image_aux's pixel-space supervision contributes the most. This is consistent with the original Plan-F motivation: image_aux is a pixel-domain anchor that fights latent-space drift, and drift compounds along the chain.

V7 vs V6_NOISE (pure seed) shows no such monotonic pattern (0.014 → 0.008 → 0.014 → 0.026 dB), which is exactly what we expect from noise — random across stages, slightly larger on NORMAL because NORMAL noise floors are simply higher.

---

## 5. Where the existing comparison tooling needs fixing

The script `review/0505/local/scripts/compare_v6_v7_v8.py` (previously read) loads metrics rows by proximity to a target step, with no filter on `event`. With V7/V8 emitting both rolling `[val]` (event="val") and full `[val_full]` (event="val_full") at potentially adjacent steps, the script can quietly pick a rolling row when a `val_full` row exists.

Action items:
- Filter rows to `event == "val_full"` for V7/V8 when claim-level numbers are needed.
- For V6_NOISE, the script can only return rolling rows; document this and never mix a V6_NOISE rolling row with a V7/V8 full-val row in the same table.
- The clean alternative — already in use here — is to read the full-val PSNR_clip3 artifacts produced by `eval_first_hop_224_clip3.py`, which run a deterministic full-loader pass at the same protocol for all three experiments and write per-slice CSVs.

---

## 6. Decision

The strict Plan-F absolute floor (d_image_aux ≥ 10%) fails. The relative requirement (d_image_aux ≥ 1.8 · d_pure) passes by an order of magnitude. Three options were on the table when Plan-F was authored; pick (B):

- (A) Drop image_aux claim entirely, treat V7 == V8 for paper purposes.
  - Rejected: 12–28× SNR is too clean to discard.
- (B) **Claim image_aux as a small, reliable, monotonically-growing-along-cascade improvement; do not claim a major lever.** ← selected
- (C) Re-tune the 10% floor and pretend Plan-F passed.
  - Rejected: changing thresholds post-hoc is the textbook bad practice. The 10% floor was wrong as a prior; document it as wrong, do not edit it.

Headline wording proposal for the paper / report:
- "image_aux supervision provides a +0.308 dB NORMAL PSNR_clip3 improvement (paired t = 74.6, n = 7403, per-slice win rate 91.5%); the seed-only noise floor measured under identical conditions is 0.026 dB, giving a signal-to-seed-noise ratio of 11.9× on NORMAL and 16–28× on intermediate doses. The gain grows monotonically along the rollout cascade, consistent with image_aux acting as a pixel-domain anchor against latent drift."

Do **not** report it as a >10% MSE win — it is a 6.65% MSE / 0.31 dB PSNR win and that is the honest framing.

---

## 7. Next steps

Priority order; each item is small unless flagged.

1. **Lock canonical baseline.**
   - Pin V7 `step_160000.pt` (= `best.pt`) as the Plan-F primary baseline checkpoint everywhere downstream. Reference path: `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V7/run/first_hop_224_v7_gronwall_raw/step_160000.pt` (and the identical `last.pt`/`best.pt`).
   - Stop iterating on this run.

2. **Fix `compare_v6_v7_v8.py` to filter `event=="val_full"`** before any future use; or deprecate it in favor of the `eval_first_hop_224_clip3.py` pipeline used here (already produces per-slice CSVs and JSON aggregates).

3. **Land a small statistical-test helper** that consumes the per-slice CSVs and emits paired t-stat / win-rate / bootstrap CI. The code in §2.2 of this report is fine as a starting point; promote it into `tools/` or `scripts/` so future ablations get the same treatment automatically.

4. **Plan the next research lever** (this is the genuinely open question; image_aux is closed).
   - Highest expected payoff candidates:
     - Architecture: depth/width sweep on `PETFlowDiTFirstHop`; replace cross-attention with a heavier conditioning path.
     - Loss reformulation: schedule rollout weight or per-stage weighting (current weights 0.50/0.45/0.90/1.50 were set heuristically; SNR analysis in §4 suggests the cascade end is where signal is concentrated, so increasing NORMAL weight may help further).
     - Schedule: cosine restart at 80K; longer training beyond 160K is unlikely to help given the §1.1 plateau, but a warm-restart could.
     - Conditioning: add per-dose embedding refinement, or per-slice position encoding.
   - Lowest payoff (skip unless cheap): more seeds, image_aux variants, sigma schedule micro-tuning.

5. **Decide whether to run one more seed of V7** (seed=2024, say) to give a 3-point seed envelope and tighten the d_pure estimate.
   - Pro: makes the 11.9× SNR statement defensible without a single-seed objection.
   - Con: ~160K-step run takes the same compute V7/V8 took. Worth it if reviewers ask; not worth it preemptively.
   - Recommendation: defer until external review or paper revision asks for it. The current 1-vs-1 seed comparison (V7 seed42 vs V6_NOISE seed1337) is already at 7403-slice paired-t resolution.

6. **Write the next phase plan.** With image_aux settled, frame the next phase around one of the architecture/loss/conditioning levers from item 4 with an explicit pre-registered success threshold (avoiding the "wrong prior" mistake from this round — pick a threshold expressed as a multiple of the now-measured d_pure of 0.026 dB / 0.37% NORMAL MSE rather than an absolute %).

---

## 8. Artifact pointers

- Plan-F training snapshot (final, all three runs done): [review/0511/log_snapshots_20260516_163900](../0511/log_snapshots_20260516_163900)
- Resolved training configs: [V7](../0511/log_snapshots_20260516_163900/configs/V7_config.resolved.yaml), [V8](../0511/log_snapshots_20260516_163900/configs/V8_config.resolved.yaml), [V6_NOISE](../0511/log_snapshots_20260516_163900/configs/V6_NOISE_config.resolved.yaml)
- Full training logs: [V7](../0511/log_snapshots_20260516_163900/main_training/V7_train_20260516_163900.log), [V8](../0511/log_snapshots_20260516_163900/main_training/V8_train_20260516_163900.log), [V6_NOISE](../0511/log_snapshots_20260516_163900/main_training/V6_NOISE_train_20260516_163900.log)
- Full-val PSNR_clip3 report: [PLANF_FULLVAL_PSNR_CLIP3_REPORT_20260516_173941.md](../0511/fullval_psnr_clip3_20260516_173941/PLANF_FULLVAL_PSNR_CLIP3_REPORT_20260516_173941.md)
- Per-slice CSVs used for §2.2 paired stats: `review/0511/fullval_psnr_clip3_20260516_173941/artifacts/planf_{v7,v8,v6noise}_{best,last}_fullval_psnr_chain_mse_per_slice.csv`
- Eval launch script: [run_planf_fullval_psnr_clip3_20260516.sh](../0511/fullval_psnr_clip3_20260516_173941/scripts/run_planf_fullval_psnr_clip3_20260516.sh)
- Eval entry point: [eval_first_hop_224_clip3.py](../../eval_first_hop_224_clip3.py)
