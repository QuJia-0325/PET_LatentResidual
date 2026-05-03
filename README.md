# PET Latent Residual

Independent workspace for PET-side follow-up experiments around the original RAE project.

This repo currently hosts three lines, all sharing the same `pet_lr/` package and configuration system:

- legacy `v2.1 residual refinement` (entry: `train_v21.py`; configs archived under [configs/pet_flow/archive_20260402/](configs/pet_flow/archive_20260402))
- current `224 first-hop latent transport` line (entry: `train_first_hop.py`; canonical spec in [docs/main.md](docs/main.md))
- σ-normalize / step-weight ablation track (configs `*_50k_foc_lite*` and `*_50k_sigma_*`; pre-registration in [review/0502/POST_V6_NEXT_STEPS.md](review/0502/POST_V6_NEXT_STEPS.md))

The folder is intentionally separate from the original `RAE` project so new ideas do not get mixed into upstream code.

Canonical implementation spec for the active line:

- [docs/main.md](docs/main.md)

Daily progress notes and protocol locks live under [review/](review) (one folder per day, `0416/` … `0503/`); the active milestone is [review/0502/](review/0502).

## Scope

- Keep the latent rollout state as the only transport state.
- Reuse the existing hop-aware 4-hop latent model and frozen RAE decoder from the original RAE project.
- Add a small image-space residual head after decoding (legacy v2.1 line).
- For the first-hop line, add a hop-conditioned latent velocity residual head plus an optional hop-0 image auxiliary loss; the residual output is **not** fed back into the next latent hop.

## Layout

```text
PET_LatentResidual/
  README.md
  CLAUDE.md                    # autonomous-agent project memory
  IDEA_REPORT.md               # cross-experiment idea log
  AUTO_REVIEW.md               # latest auto-review snapshot
  train_v21.py                 # legacy v2.1 entry
  train_first_hop.py           # current first-hop entry
  eval_first_hop_224_clip3.py  # offline eval (chain-MSE / PSNR / LPIPS, clip-3)
  check_alignment_224_clip3.py # latent–image alignment & sanity diagnostics
  configs/pet_flow/
    pet_flow_first_hop_224_*.yaml         # active first-hop / σ-norm configs
    archive_20260402/                     # archived legacy configs (incl. v2.1)
  pet_lr/
    __init__.py  bootstrap.py
    data.py             data_first_hop.py
    model.py            model_first_hop.py
    rollout.py          rollout_first_hop.py
    losses.py           losses_first_hop.py
    ema.py              path_guard.py
  docs/
    main.md
    background_first_hop_design.md
    background_v21_residual_refinement.md
    experiment_plans/
    archive/
      first_hop_implementation_spec_old.md
      v21_224_decoder_adaptor_spec.md
  review/
    0416/ … 0503/        # dated decision / protocol logs
    plan/                # cross-cutting roadmaps
    0502/                # active σ-norm pre-registration milestone
  scripts/
    diagnose_error_budget.py
    diagnose_foc_gap.py
    diagnose_tf_rollout_gap.py
    eval_gt_latent_decoder_ceiling_clip3.py
    launch_v4_sf_pilot.sh
    launch_v5_rollout_heavy.sh
  tools/
    migrate_repo_outputs_to_data_disk.sh   # one-shot output relocation helper
  deprecated/                              # frozen artifacts kept only for traceability
```

## What Lives Here

### Top-level utilities

- `train_v21.py` — minimal training entry for the legacy v2.1 line (kept for reproduction only).
- `train_first_hop.py` — training entry for the 224 first-hop latent transport line (current production).
- `eval_first_hop_224_clip3.py` — offline evaluation harness: walks a checkpoint or ckpt directory, runs latent rollout on the validation slices, and reports `chain_normal_mse / clip-3 PSNR / LPIPS / val_select_score`. Supports `--max-slices 0` for full-validation passes.
- `check_alignment_224_clip3.py` — diagnostic tool: checks latent–image alignment, hop-0 condition wiring, and dataset sanity before launching long runs.

### `pet_lr/` package

- `data.py` — loads aligned 4-hop latent pairs and raw PET slices.
- `data_first_hop.py` — first-hop aligned dataset with dual-loader support (`main` + `hop0 auxiliary`) for hop-0 image supervision.
- `model.py` — wraps the frozen hop-aware latent backbone, frozen RAE decoder, and local residual head.
- `model_first_hop.py` — adds hop-0 pixel forcing and a hop-conditioned latent velocity residual head on top of the hop-aware backbone.
- `rollout.py` — latent-only rollout, preserves the original straight-through mixing idea.
- `rollout_first_hop.py` — first-hop rollout where only step 0 receives the `x_D50` pixel condition.
- `losses.py` — `L1 + SSIM + seam + border-aware weighting + residual penalty` for the v2.1 line.
- `losses_first_hop.py` — hop-0 image auxiliary loss on `Crop(Dec(z_D20_pred))` plus the multi-objective selection score used by Method D.
- `ema.py` — Exponential Moving Average of model weights; toggled per-config (used by every `_50k_*` first-hop run).
- `path_guard.py` — resolves all output paths under `DEFAULT_OUTPUT_ROOT = /data_2/qujiaxiang/outputs/PET_LatentResidual`, refusing accidental writes inside the repo.
- `bootstrap.py` — environment / import shim that exposes the upstream RAE module path to first-hop training without copying code.

### `docs/`

- `main.md` — current canonical implementation spec for the first-hop enhancement line.
- `background_first_hop_design.md` — draft reasoning that led to the first-hop spec.
- `background_v21_residual_refinement.md` — background note for the legacy v2.1 line.
- `experiment_plans/` — per-experiment intent / acceptance docs.
- `archive/v21_224_decoder_adaptor_spec.md`, `archive/first_hop_implementation_spec_old.md` — kept only for traceability.

### `review/`

Daily protocol / decision logs. Each subfolder is a date stamp (`MMDD`). Highlights:

- [review/0502/POST_V6_NEXT_STEPS.md](review/0502/POST_V6_NEXT_STEPS.md) — V6 → ablation transition master plan, with LOCKED §6.6 blinded effect-size pre-registration.
- [review/0502/scripts/](review/0502/scripts) — protocol-grade helpers: `select_best_ckpt_smoothed.py` (Method D ckpt selection), `paired_diff_judge.py` (Risk 4 paired-diff guard), `lock_effect_size_threshold.py` (one-shot pre-registration lock-in).
- `review/plan/` — longer-horizon roadmaps that cross multiple dates.

### `scripts/` and `tools/`

- `scripts/diagnose_*.py` — narrowly-scoped post-mortems for specific gap signatures (error-budget, foc-gap, TF-vs-rollout gap).
- `scripts/eval_gt_latent_decoder_ceiling_clip3.py` — ground-truth-latent decoder ceiling, used as an upper-bound reference in §6.6.
- `scripts/launch_*.sh` — historical pilot-launcher shell scripts; kept for replayability.
- `tools/migrate_repo_outputs_to_data_disk.sh` — one-shot helper used when the repo was moved off the system disk.

## What Still Comes From The Original RAE Project

This workspace reuses stable components from upstream:

- `RAE/code/RAE/src/pet_flow/models/pet_flow_dit_hop.py`
- `RAE/code/RAE/src/pet_flow/inference_pet_flow.py`

That reuse is intentional:

- the original RAE project remains unchanged
- all new experiment-specific code stays here

## First Recommended Experiment

The default first-hop config freezes both:

- the hop-aware latent backbone (only the residual head and hop-0 wiring are trained)
- the RAE decoder

So the first run answers a narrow question:

> Can a small, hop-conditioned velocity residual plus a hop-0 pixel condition reduce structure breaks and seam artifacts without changing latent rollout dynamics?

Active V6 / ablation configs sit under `configs/pet_flow/pet_flow_first_hop_224_50k_*.yaml`; the σ-norm ablation pre-registration explains how to read the resulting numbers without confirmation bias.

## Examples

### Current first-hop line (production)

```bash
cd /home/qujiaxiang/project/PET_LatentResidual

# 50K transport-first V6 backbone:
python train_first_hop.py \
    --config configs/pet_flow/pet_flow_first_hop_224_v6_transport_first.yaml

# Offline evaluation of one or more checkpoints (clip-3 PSNR/LPIPS, chain MSE):
python eval_first_hop_224_clip3.py \
    --config configs/pet_flow/pet_flow_first_hop_224_v6_transport_first.yaml \
    --ckpt   /data_2/qujiaxiang/outputs/PET_LatentResidual/.../best.pt
```

### Legacy v2.1 line (kept for reproduction only)

The v2.1 yaml has been moved to `configs/pet_flow/archive_20260402/`. Only run this if you specifically need to reproduce historical results:

```bash
cd /home/qujiaxiang/project/PET_LatentResidual
python train_v21.py \
    --config configs/pet_flow/archive_20260402/pet_flow_first_hop_224_formal.yaml
```

## Conventions

- All training outputs are written under `/data_2/qujiaxiang/outputs/PET_LatentResidual/<run_name>/`; `pet_lr/path_guard.py` will refuse paths inside the repo.
- Determinism: seeds are config-controlled; rolling-window evaluation uses `max_val_batches=64`, `eval_interval=400`, `val_window_mode=rolling`.
- Protocol changes (window length, threshold values, decision rules) are LOCKED via [review/0502/POST_V6_NEXT_STEPS.md](review/0502/POST_V6_NEXT_STEPS.md) and the lock-in scripts; do **not** modify them in-place. Add a deviation note instead.
