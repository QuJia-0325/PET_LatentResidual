# Research Review (0422) — 0b109f8 Code/Architecture/API Audit

## Meta

- Date: 2026-04-22
- Branch: `foc_lite_hop0`
- Commit reviewed: `0b109f8`
- Scope: code architecture, scheme design, API/invocation contracts, experiment validity
- External reviewer agent:
  - nickname: `Mendel`
  - id: `019db13d-be9c-7f80-a082-45bc0207cfde`
  - model: `gpt-5.4` (`xhigh`)

---

## Round Summary

### Round 1 (external harsh review)

Reviewer identified 7 major risk clusters, with strongest concerns on:
1. N2 standalone not enforcing `--resume` to a fixed transport checkpoint.
2. N1 checkpoint/config semantic mismatch risk (`pixel_forcing_disabled` not reliably encoded in checkpoint compatibility path).
3. Param-group granularity mismatch (`first_hop_weight_decay: 0.0` affects all first-hop params, not only scalar gates).
4. Rolling-window validation used for best-checkpoint selection bias.
5. D1/N2 guard objective coverage mismatch (guarding mostly NORMAL while refiner affects all timepoints).

### Round 2 (convergence)

Reviewer converged to top-5 actionable issues + minimal patch patterns + N1/N2 results-to-claims matrix + 72h execution plan.

---

## Final Findings (Ordered by Severity)

### P0-1: N2 does not enforce resume-to-source transport identity

- Diagnosis: N2 config is described as standalone refiner on frozen transport, but trainer does not require `--resume`.
- Trigger: run N2 config without `--resume` (or wrong checkpoint).
- Impact: refiner may be trained on frozen random first-hop branches, invalidating N2 causal claim.
- Evidence:
  - [/home/qujiaxiang/project/PET_LatentResidual/configs/pet_flow/pet_flow_first_hop_224_20k_seam_refiner_stage2.yaml:1](/home/qujiaxiang/project/PET_LatentResidual/configs/pet_flow/pet_flow_first_hop_224_20k_seam_refiner_stage2.yaml:1)
  - [/home/qujiaxiang/project/PET_LatentResidual/configs/pet_flow/pet_flow_first_hop_224_20k_seam_refiner_stage2.yaml:58](/home/qujiaxiang/project/PET_LatentResidual/configs/pet_flow/pet_flow_first_hop_224_20k_seam_refiner_stage2.yaml:58)
  - [/home/qujiaxiang/project/PET_LatentResidual/train_first_hop.py:1068](/home/qujiaxiang/project/PET_LatentResidual/train_first_hop.py:1068)
  - [/home/qujiaxiang/project/PET_LatentResidual/train_first_hop.py:1140](/home/qujiaxiang/project/PET_LatentResidual/train_first_hop.py:1140)

### P0-2: N1 ablation semantics can be silently mismatched across resume/eval

- Diagnosis: `pixel_forcing_disabled` changes forward semantics but is not robustly enforced as checkpoint semantic compatibility in both train-resume and eval.
- Trigger: evaluate/load checkpoint with mismatched config.
- Impact: N1 conclusion can be silently contaminated.
- Evidence:
  - [/home/qujiaxiang/project/PET_LatentResidual/configs/pet_flow/pet_flow_first_hop_224_50k_pixenc_ablation.yaml:247](/home/qujiaxiang/project/PET_LatentResidual/configs/pet_flow/pet_flow_first_hop_224_50k_pixenc_ablation.yaml:247)
  - [/home/qujiaxiang/project/PET_LatentResidual/train_first_hop.py:1026](/home/qujiaxiang/project/PET_LatentResidual/train_first_hop.py:1026)
  - [/home/qujiaxiang/project/PET_LatentResidual/train_first_hop.py:1510](/home/qujiaxiang/project/PET_LatentResidual/train_first_hop.py:1510)
  - [/home/qujiaxiang/project/PET_LatentResidual/eval_first_hop_224_clip3.py:103](/home/qujiaxiang/project/PET_LatentResidual/eval_first_hop_224_clip3.py:103)

### P1-1: Weight-decay comment and actual optimizer behavior are inconsistent

- Diagnosis: config comment says excluding scalar gates from decay, but current grouping applies `first_hop_weight_decay` to all non-backbone trainable first-hop params.
- Trigger: all active configs with `first_hop_weight_decay`.
- Impact: N1/N2 become mixed-factor changes (especially N2 refiner regularization).
- Evidence:
  - [/home/qujiaxiang/project/PET_LatentResidual/train_first_hop.py:1161](/home/qujiaxiang/project/PET_LatentResidual/train_first_hop.py:1161)
  - [/home/qujiaxiang/project/PET_LatentResidual/train_first_hop.py:1191](/home/qujiaxiang/project/PET_LatentResidual/train_first_hop.py:1191)
  - [/home/qujiaxiang/project/PET_LatentResidual/configs/pet_flow/pet_flow_first_hop_224_50k_imgaux_boost.yaml:149](/home/qujiaxiang/project/PET_LatentResidual/configs/pet_flow/pet_flow_first_hop_224_50k_imgaux_boost.yaml:149)
  - [/home/qujiaxiang/project/PET_LatentResidual/configs/pet_flow/pet_flow_first_hop_224_20k_seam_refiner_stage2.yaml:164](/home/qujiaxiang/project/PET_LatentResidual/configs/pet_flow/pet_flow_first_hop_224_20k_seam_refiner_stage2.yaml:164)

### P1-2: Best-checkpoint selection still biased by rolling validation window

- Diagnosis: step-to-step best comparison is not on fixed subset/full-val.
- Trigger: `val_window_mode: rolling` + truncated validation.
- Impact: best-vs-best comparisons become moving-target and can flip under full-val rerank.
- Evidence:
  - [/home/qujiaxiang/project/PET_LatentResidual/train_first_hop.py:630](/home/qujiaxiang/project/PET_LatentResidual/train_first_hop.py:630)
  - [/home/qujiaxiang/project/PET_LatentResidual/train_first_hop.py:678](/home/qujiaxiang/project/PET_LatentResidual/train_first_hop.py:678)

### P2-1: D1/N2 guard objective may not fully cover all affected timepoints

- Diagnosis: seam refiner is applied across decode path but guard emphasis can miss degradation at intermediate hops.
- Trigger: using `best_d1.pt` for broader chain claims.
- Impact: possible over-claim if D20 seam improves but D10/D4 regress.
- Evidence:
  - [/home/qujiaxiang/project/PET_LatentResidual/configs/pet_flow/pet_flow_first_hop_224_20k_seam_refiner_stage2.yaml:85](/home/qujiaxiang/project/PET_LatentResidual/configs/pet_flow/pet_flow_first_hop_224_20k_seam_refiner_stage2.yaml:85)
  - [/home/qujiaxiang/project/PET_LatentResidual/train_first_hop.py:2132](/home/qujiaxiang/project/PET_LatentResidual/train_first_hop.py:2132)

---

## Not-a-Bug (Confirmed)

1. `decode_crop` refiner apply path change (`hasattr(self, "seam_refiner")`) is acceptable and did not reveal regression in normal config paths.
   - [/home/qujiaxiang/project/PET_LatentResidual/pet_lr/model_first_hop.py:563](/home/qujiaxiang/project/PET_LatentResidual/pet_lr/model_first_hop.py:563)

2. `skip_first_tp` behavior is connected and effective in eval path (D50 default path can skip refiner).
   - [/home/qujiaxiang/project/PET_LatentResidual/eval_first_hop_224_clip3.py:178](/home/qujiaxiang/project/PET_LatentResidual/eval_first_hop_224_clip3.py:178)

3. `.gitignore` + untracking `__pycache__` is a valid and necessary repo hygiene fix.
   - [/home/qujiaxiang/project/PET_LatentResidual/.gitignore:1](/home/qujiaxiang/project/PET_LatentResidual/.gitignore:1)

---

## Minimal Patch Plan (Prioritized)

### Patch A (must do first): enforce N2 resume identity

```python
if train_cfg["freeze_all_except_seam_refiner"] and not resume_enabled:
    raise RuntimeError("N2 standalone requires --resume <audited_transport_ckpt>")
```

Then validate warm-start missing keys only from `seam_refiner.*`.

### Patch B: encode and verify checkpoint semantics for N1/N2

```python
ckpt["first_hop_pixel_enabled"] = not cfg["first_hop"].get("pixel_forcing_disabled", False)
ckpt["config_sha256"] = sha256(canonical_yaml(cfg))
if resume_or_eval and ckpt["first_hop_pixel_enabled"] != current_first_hop_pixel_enabled:
    raise RuntimeError("pixel forcing semantic mismatch")
```

### Patch C: split optimizer param groups by parameter role

- `scalar gates`: `g_pix_raw`, `lambda_hop_raw` => `wd=0`
- `other first-hop weights` => `wd=first_hop_weight_decay`
- `seam_refiner` => explicit `wd_refiner` (do not silently inherit gate policy)
- backbone unchanged

### Patch D: decouple rolling metrics from selection metrics

- Rolling window for monitoring only.
- Fixed full-val (or fixed subset) for checkpoint ranking.
- Offline rerank historical checkpoints before claims.

### Patch E: strengthen D1/N2 guard coverage

- Guard keys include `D20/D10/D4/NORMAL` chain terms, not single-tail emphasis only, for claims beyond hop0 seam.

---

## Results-to-Claims Matrix

## N1 (pixel forcing ON vs N1 ablation OFF)

| Outcome | Allowed Claim | Forbidden Claim |
|---|---|---|
| Significant gain | Whole hop0 pixel-forcing path helps under current recipe. | Encoder representation is universally useful; mechanism is fully proven. |
| Small gain | Pixel forcing is a secondary helper. | Pixel forcing is core novelty with robust dominance. |
| No difference | No reproducible benefit observed under current setup. | Pixel forcing is useless in all settings. |
| Degradation | Current implementation causes negative transfer/instability. | Pixel conditioning is fundamentally invalid for PET. |

## N2 (refined decode vs same frozen transport raw decode)

| Outcome | Allowed Claim | Forbidden Claim |
|---|---|---|
| Significant gain | Post-decoder refiner improves seam/visual metrics on fixed transport. | Transport itself improved; all hops universally improved. |
| Small gain | Refiner provides limited post-processing benefit. | Refiner is primary source of overall chain improvement. |
| No difference | Standalone refiner has no validated net gain. | Decoder artifact is irrelevant in general. |
| Degradation | Current refiner recipe harms chain consistency. | Any seam/post-processing approach is invalid by nature. |

---

## 72-Hour Execution Plan

### Day 1 (code safety, no GPU burn)

1. Implement Patch A/B/C.
2. Add smoke tests:
   - N2 with `freeze_all_except...` and no `--resume` must fail.
   - N1 checkpoint evaluated under non-N1 semantic config must fail.
   - Warm-start missing keys allowed only for `seam_refiner.*`.

Estimated cost: CPU-only engineering.

### Day 2 (no retraining, full-val rerank)

1. Offline full-val rerank for baseline/N1/N2 saved checkpoints (`best.pt`, `last.pt`, periodic checkpoints).
2. Compare training-time best vs full-val best and flag claim flips.

Estimated cost: eval GPU time only (no training).

### Day 3 (minimal recompute only if signal survives rerank)

1. N2 short smoke run with enforced resume identity; verify `raw` path stability against source transport.
2. If stable and signal remains, run only minimal N1/N2 paired reruns needed for publication-grade claims.

Estimated cost: targeted GPU-hours, avoid full broad retraining.

---

## Local Additional Note

- `review/0421/Local/README.md` still contains N2 command text with explicit `--resume` and old path naming in some lines, while config is now `..._standalone`. For consistency, keep one canonical run protocol document to avoid operator confusion.
  - [/home/qujiaxiang/project/PET_LatentResidual/review/0421/Local/README.md:123](/home/qujiaxiang/project/PET_LatentResidual/review/0421/Local/README.md:123)
  - [/home/qujiaxiang/project/PET_LatentResidual/configs/pet_flow/pet_flow_first_hop_224_20k_seam_refiner_stage2.yaml:7](/home/qujiaxiang/project/PET_LatentResidual/configs/pet_flow/pet_flow_first_hop_224_20k_seam_refiner_stage2.yaml:7)

---

## Conclusion

The 0422 patch set fixed several important low-level issues, but publication-risk blockers remain in experiment identity, checkpoint semantics, and fair model selection protocol. The project should prioritize semantic guardrails and full-val rerank before spending more GPU on new claims.

