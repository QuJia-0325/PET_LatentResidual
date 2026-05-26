# PET Latent Residual

Independent PET-side workspace for low-dose PET latent transport experiments built around the original RAE project.

This repository is now primarily a **paper-facing experimental record** for the 224 first-hop latent transport line. Older design tracks remain in the tree for provenance, but the current canonical state is the A4 / decoder-aware image auxiliary supervision result described below.

---

## Current State (2026-05-26)

Read this section first. Several older files, especially `docs/main.md`, `review/0502/`, and early `configs/pet_flow/*50k*` plans, are historical and no longer define the active paper narrative.

### Active Claim

The current paper headline candidate is:

> Stronger decoder-aware image auxiliary supervision through the frozen RAE decoder is the dominant lever for PET latent transport quality.

Operationally, this is **A4-mid**:

- base: V7-style first-hop latent transport
- `training.image_aux.lambda_start=lambda_max=0.08`
- full image auxiliary loss (`L1 + 0.25 * SSIM + 0.10 * seam`)
- no decoder LoRA
- no KL pullback
- seed 42

### Canonical Full-Val Anchors

All numbers below are `src.utils.metrics.calc_psnr_clip3`, full validation split, `n=7403` slices, `NORMAL` endpoint.

| run | NORMAL | delta vs V7 | role |
|---|---:|---:|---|
| V13.best (`image_aux=0`) | 36.494330 | -0.286621 | true image_aux-off control |
| V7.best (`image_aux=0.04`) | 36.780951 | 0 | baseline |
| V14.best (`V7 seed=1337`) | 36.780632 | -0.000320 | seed perturbation at lambda=0.04 |
| A4-low.best (`image_aux=0.02`) | 36.700958 | -0.079994 | weak image_aux bracket |
| **A4-mid.best (`image_aux=0.08`)** | **36.893917** | **+0.112966** | **headline result** |
| V18.best (`LoRA + KL`, lambda_img=0.04) | 36.811167 | +0.030216 | secondary decoder-side ablation |
| V18.last (`LoRA + KL`, lambda_img=0.04) | 36.842645 | +0.061693 | secondary decoder-side ablation |
| X3.best (`LoRA + lambda_img=0.08`, KL off) | 36.810450 | +0.029498 | non-additive LoRA ablation |
| X3.last (`LoRA + lambda_img=0.08`, KL off) | 36.828787 | +0.047836 | non-additive LoRA ablation |

Do not reuse stale A4 numbers such as `36.835` or `+0.054`. The canonical A4-mid value is `36.893917`, with `+0.112966 dB` over V7 and `+0.113286 dB` over V14.

### Active Experiments

| run | status | purpose |
|---|---|---|
| X1-lite | running | test whether L1 through the frozen decoder explains most of A4-mid, or whether SSIM/seam components are needed |
| A4-mid-seed1337 | approved / task ready | one seed replicate of A4-mid for paper-headline robustness |
| X3 | done | tested LoRA additivity under `lambda_img=0.08`; result is non-additive |

### Current Stop Rules

- Do not launch V18-clean, V19, decoder-rank sweeps, or X3 extensions unless a new user-signed review explicitly overturns the stop rule.
- Do not launch additional image-aux lambda points (`0.06`, `0.10`, `0.12`, etc.).
- Do not launch a seed sweep. A4-mid-seed1337 is exactly one robustness replicate.
- Do not claim patient-level significance; patient grouping is not recoverable from the current preprocessed artifacts.
- F0 paired-slice/bootstrap is not closed until explicit F0 report artifacts exist.

---

## Start Here For Writing Agents

If you are drafting or reviewing the paper, start from these files in order:

1. `CLAUDE.md` — current canonical state, constraints, and source-of-truth list.
2. `IDEA_REPORT.md` — current active idea stack and superseded 2026-04 ideas.
3. `AUTO_REVIEW.md` — current review gates and historical nightmare-review protocol.
4. `review/0525/REVIEW_INTEGRATION_round18_20260525.md` — strategic integration after A4 bracket.
5. `review/0525/REVIEW_INTEGRATION_round18_prep_20260525.md` — code-level X1/X3 task review.
6. `review/0525/REVIEW_INTEGRATION_round19_next_slot_after_X3_20260526.md` — A4-mid-seed1337 decision.
7. `review/0525/Round19-TODO.md` — current waiting list and next trigger conditions.

For result tables, use raw JSON/CSV artifacts, not prose summaries. Preferred result sources:

- V7: `review/0511/fullval_psnr_clip3_20260516_173941/artifacts/`
- V13/V14: `review/0516/full_eval_json/`
- V18/A3: `review/0517/V18_decoder_lora/`, `review/0517/V18_capacity_only/`
- A4 bracket: `review/0521/A4_image_aux_lambda_bracket/`
- X3: `review/0525/X3_image_aux_lora/fullval_eval/`

---

## Repository Layout

```text
PET_LatentResidual/
  README.md
  CLAUDE.md
  IDEA_REPORT.md
  AUTO_REVIEW.md

  train_first_hop.py              # active training entry for first-hop latent transport
  train_v21.py                    # legacy v2.1 residual-refinement entry, kept for reproduction only
  eval_first_hop_224_clip3.py     # older offline clip3 evaluator
  check_alignment_224_clip3.py    # latent-image alignment audit

  pet_lr/                         # active implementation package
  configs/pet_flow/               # historical and baseline yaml configs
  docs/                           # historical specs and background notes
  review/                         # dated decisions, results, peer reviews, codex tasks
  scripts/                        # historical diagnostic / launch helpers
  tools/                          # probes, migration helpers, small utilities
  deprecated/                     # frozen legacy artifacts
```

---

## Top-Level Files

- `CLAUDE.md` — canonical project memory for agents. It is now the fastest way to learn the current state.
- `IDEA_REPORT.md` — idea ledger. The 2026-04 CCT / delta-B ideas are now backlog; A4 is current.
- `AUTO_REVIEW.md` — review protocol and current review-state snapshot. Use it for adversarial review framing.
- `train_first_hop.py` — active trainer. It implements pair loss, rollout loss, image_aux, optional LoRA, EMA checkpoint saving, and full-val best selection.
- `train_v21.py` — legacy v2.1 entry; do not use for the active paper line.
- `eval_first_hop_224_clip3.py` — older evaluator; current canonical full-val evaluations often use `review/0505/operator/scripts/eval_first_hop_fullval_psnr_chain_mse.py`.
- `check_alignment_224_clip3.py` — alignment check before long runs.

---

## `pet_lr/` Package

| file | role |
|---|---|
| `data_first_hop.py` | active 224 first-hop dataset loader; patient IDs are not preserved in current preprocessed artifacts |
| `model_first_hop.py` | active model wrapper: hop0 pixel forcing, hop residual head, frozen RAE decode/crop path, optional LoRA wrapping |
| `rollout_first_hop.py` | latent rollout training/eval path; only step 0 receives `x_D50` pixel condition |
| `losses_first_hop.py` | hop0 image auxiliary loss wrapper around decoded `z_pred` |
| `losses.py` | weighted L1, SSIM, seam/border losses used by image_aux |
| `decoder_lora.py` | LoRA wrapping for RAE decoder linear layers; V18/A3/X3 use explicit rank 32, not the code default |
| `ema.py` | EMA context manager; saved checkpoints are written inside EMA averaging contexts when EMA is enabled |
| `path_guard.py` | enforces output paths under `/data_2/qujiaxiang/outputs/PET_LatentResidual` |
| `bootstrap.py` | exposes the upstream RAE code path without copying RAE source into this repo |

Important implementation facts:

- `image_aux` flows through the frozen decoder; frozen decoder parameters do not update, but gradients still flow through decoded pixels into `z_pred` and the transport model.
- `loss.image_aux.ssim_weight=0` and `seam_weight=0` remove those terms from the total loss, but raw SSIM/seam losses are still computed and logged.
- `decoder_kl_pullback.use_pred_latent=true` was a historical V18 issue; do not revive KL experiments without a new review.
- `train_first_hop.py` only resumes from the CLI `--resume` path; YAML `training.resume_from` alone does not resume.

---

## `configs/`

`configs/pet_flow/` contains historical and baseline YAML configs. Many older configs correspond to 0420-0506 exploratory tracks and are not the active paper line.

Use these as source templates only with care:

- V7 baseline source: `review/0505/local/configs/V7_gronwall_raw.yaml`
- A4-mid source: `review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08.yaml`
- V18 source: `review/0517/V18_decoder_lora/V18_decoder_lora.yaml`

Do not treat `configs/pet_flow/*50k*` as the current paper headline. That line is historical provenance for V6 / sigma-normalize exploration.

---

## `docs/`

`docs/` contains background and historical implementation specs. As of 2026-05-26, `docs/main.md` is **stale** relative to the current A4 / X1 / X3 paper state.

| path | current status |
|---|---|
| `docs/main.md` | historical first-hop spec; useful provenance, not the active paper spec |
| `docs/background_first_hop_design.md` | background on first-hop design |
| `docs/background_v21_residual_refinement.md` | legacy v2.1 background |
| `docs/experiment_plans/` | early per-experiment design notes |
| `docs/archive/` | frozen older specs |

Before paper drafting deepens, `docs/main.md` should be rewritten or clearly marked as superseded by `CLAUDE.md` + `review/0525` integration files.

---

## `review/` Directory Guide

The `review/` tree is the real project history. Each date folder is an experiment / decision batch.

### Current Paper-Relevant Folders

| folder | role |
|---|---|
| `review/0525/` | current Round 18/19 analysis, X3 results, Round19 TODO, host/supervisor notes, next-slot tasking |
| `review/0521/` | A4 bracket results: A4-low, A4-mid, A4 bracket full-val report; Round17 strategy and slot decisions |
| `review/0517/` | V18 decoder LoRA, A3 capacity-only, KL-drift / direct-decode probes, Round16 reviews |
| `review/0516/` | V13 true image_aux off and V14 true d_pure seed control |
| `review/0511/` | V7 full-val artifacts and log snapshots |
| `review/0505/` | V7 source config and canonical full-val evaluator script under `operator/scripts/` |

### Historical / Provenance Folders

| folder | role |
|---|---|
| `review/0502/` | sigma-normalize pre-registration; historically important but not current headline |
| `review/0503/` | sigma-normalize sanity failure and protocol evidence |
| `review/0416`-`review/0430` | early first-hop / CCT / self-forcing / seam / rollout-heavy exploration |
| `review/plan/` | long-range planning and transport breakthrough notes; many are superseded but useful for future-work framing |

### `review/0525/` Subfolders

| path | role |
|---|---|
| `review/0525/X3_image_aux_lora/` | completed X3 run and full-val eval; shows LoRA is non-additive under strong image_aux |
| `review/0525/server/` | Codex server-side launch/design analysis for X1/X3 |
| `review/0525/HOST/` | host-to-supervisor audit notes |
| `review/0525/Supervisor/` | reserved for supervisor outputs |
| `review/0525/Writer/` | reserved for paper-writing artifacts |

---

## Current Experiment Threads

### X1-lite

Purpose: determine whether L1 through the frozen decoder explains most of A4-mid, or whether SSIM/seam components are needed.

Design: V7-style run with `lambda_img=0.08`, `ssim_weight=0`, `seam_weight=0`, full training to 160K.

Interpretation:

- close to A4-mid: L1-through-decoder is sufficient; M1+M4 dominate.
- interior: CL1 confound (component mechanism vs total gradient budget); write limitation unless user approves X1-v2-balanced.
- near V7: L1-only is insufficient; SSIM/seam or gradient budget likely matters.

### A4-mid-seed1337

Purpose: test whether A4-mid headline is robust to seed 1337.

Design: exact A4-mid clone except `seed=1337`, with isolated `output_dir` and `run_name`.

Stop rule: one seed replicate only; no seed sweep.

### X3

Status: done.

Purpose: test whether decoder LoRA is additive under `lambda_img=0.08` and `lambda_kl=0` in a 10K V7-best warmstart window.

Result: non-additive. X3.last = 36.828787, below A4-mid by -0.065130 dB. V18/X3 remain secondary ablations.

---

## Writing Guidance

### Current Paper Narrative

Recommended main story:

> PET latent transport benefits most from decoder-aware pixel supervision through the frozen RAE decoder. Pure latent objectives are misaligned with decoded-image quality; strengthening the frozen-decoder image auxiliary path yields the dominant improvement.

Do not frame the contribution as:

- a generic lambda sweep,
- a decoder LoRA / KL method,
- a patient-level statistically significant result,
- a new architecture.

### Methods Details To Include

- Image_aux formula: weighted L1 + SSIM + seam through frozen decoder.
- EMA checkpoint semantics: if EMA is enabled, saved best/last checkpoints are EMA-averaged model weights because checkpoint saving happens inside `ema.average_parameters()`.
- Full-val metric: `src.utils.metrics.calc_psnr_clip3`, `n=7403` slices.
- Slice-level limitation: patient IDs are unavailable from current preprocessing.
- Effective hop weights: pair loss + rollout loss should be explained as effective hop weighting, not arbitrary independent knobs.

### Related Work To Add

- REPA / representation alignment / decoder-aware latent learning framing for image_aux.
- Low-dose PET denoising / PET restoration baselines from MICCAI, IEEE TMI, MedIA, TPAMI-style venues.
- Min-SNR / EDM / trajectory consistency / cycle consistency only as future-work framing, unless directly used.

---

## Commands And Path Conventions

All commands are expected to run from the repository root on the server:

```bash
cd /home/qujiaxiang/project/PET_LatentResidual
```

Use the RAE conda env where needed:

```bash
PYTHON=/home/qujiaxiang/.conda/envs/rae/bin/python
export PYTHONPATH=/home/qujiaxiang/project/PET_LatentResidual:${PYTHONPATH:-}
```

Training entry:

```bash
$PYTHON train_first_hop.py --config <yaml> [--resume <ckpt>]
```

Important: `train_first_hop.py` reads resume only from `--resume`; YAML `training.resume_from` alone does not resume.

Canonical full-val evaluator:

```bash
$PYTHON review/0505/operator/scripts/eval_first_hop_fullval_psnr_chain_mse.py \
  --config <run_config.yaml> \
  --checkpoint <best_or_last.pt> \
  --tag <tag> \
  --split val \
  --max-slices 0 \
  --batch-size 8 \
  --decode-mode both \
  --out-dir /data_2/qujiaxiang/outputs/PET_LatentResidual/<eval_dir>
```

All output directories should live under `/data_2/qujiaxiang/outputs/PET_LatentResidual`. `path_guard.py` refuses accidental repo-local outputs.

---

## What Still Comes From The Original RAE Project

This workspace reuses stable upstream RAE components, including:

- `RAE/code/RAE/src/pet_flow/models/pet_flow_dit_hop.py`
- RAE PET decoder checkpoints under `/data_2/qujiaxiang/outputs/pet_lora_dinov2_pt_224/`

The upstream RAE project should remain unchanged unless explicitly approved.

---

## Legacy Lines

- `train_v21.py` and `configs/pet_flow/archive_20260402/` reproduce the legacy v2.1 residual-refinement line only.
- `review/0502` / sigma-normalize remains historical provenance.
- 2026-04 CCT / delta-B-aware / uncertainty-gated ideas in `IDEA_REPORT.md` are backlog, not active.

---

## Quick Checklist For New Agents

Before making claims or edits:

1. Read `CLAUDE.md` current state.
2. Use raw JSON under `review/` for numbers.
3. Confirm whether a result is slice-level only.
4. Check stop rules in `review/0525/Round19-TODO.md`.
5. Do not launch new experiments unless an approved codex task md exists.
6. Do not modify code without explicit user approval.
7. Never cite stale A4 value `36.835 / +0.054`.
