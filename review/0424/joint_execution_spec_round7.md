# Joint Execution Spec — Round 7 Data Collection

**Document type**: **Joint design doc (Reviewer × Author)**  
**Owners**: Author (local), Reviewer (review workflow), **Remote operator** (executing on GPU cluster)  
**Date**: 2026-04-24  
**Predecessor docs**:
- [`review/0424/claude-reviewer/07_pre_registered_verdict_map.md`](../claude-reviewer/07_pre_registered_verdict_map.md) — scoring rules (reviewer-owned)
- [`review/0424/response_to_reviewer/proposer_final_response_r6.md`](./proposer_final_response_r6.md) — Proposer's R6 plan
- [`review/0424/response_to_reviewer/reviewer_signoff_r6.md`](./reviewer_signoff_r6.md) — reviewer's signoff with 3 traps

**Purpose**: One document that a **remote teammate** can read top-to-bottom, copy-paste commands, and produce the 6 artifacts (A1–A6) that close the review loop. **No ambiguity, no cross-reference chase.**

---

## 0. TL;DR for the Remote Operator

If you just want to run things, the minimum sequence is:

1. **Day 0** (preflight, §3). Verify GPU availability, codebase hash, data paths. 30 min.
2. **Day 1–2** (A1 σ_seed-lite, §4). Launch 4 seeds × `imgaux_boost` × 50K in parallel. ~4 GPU-day wall, ~16h wall-clock if 4 GPUs.
3. **Day 2 in parallel** (A2 Path A, §5). Run TF-vs-RO diagnostic on C + N1 best.pt. ~1 GPU-day.
4. **Day 3 gate** (§6). Compute σ_seed → decide 200K go/no-go.
5. **Day 3–5** (A3 Path C, §7). Run decoder-FT diagnostic. ~1 GPU-day.
6. **Day 3–5** (A6 200K conditional, §8). If gate passes, launch with σ-unit stop criterion.
7. **Day 6+** (A4 F2 rewrite, A5 Pix2Pix, §9–10). Lower-priority; don't block on these.

All results land in the specific paths listed in §11. Reviewer Round 7 will read those paths exactly.

---

## 1. Roles & Responsibilities

| Role | Who | What they do | What they don't do |
|---|---|---|---|
| **Author** | Local machine owner (Jiaxiang) | Writes configs, checks ckpt paths, interprets scientific results, writes paper | Does not run large training directly |
| **Remote operator** | GPU cluster teammate | Executes commands, monitors GPU utilization, pushes artifacts back, flags any deviation | Does not make scientific decisions (call Author) |
| **Reviewer** | Nightmare Panel (this review stream) | Scores artifacts against §07 rules, writes Round 7 verdict | Does not tell operator how to run, does not negotiate mid-experiment |

Communication flow: operator follows this doc → produces artifacts in §11 paths → Author interprets → Reviewer scores.

If the operator hits a blocker that's **not** in §12 troubleshooting, escalate to Author (not Reviewer).

---

## 2. Prerequisites

### 2.1 Hardware
- **Minimum**: 1× GPU with ≥24 GB VRAM (e.g. 3090, 4090, A5000).
- **Recommended**: 4× GPUs to run σ_seed-lite in parallel within 16h wall-clock.
- **Not supported**: multi-node training (current code assumes single-node).

### 2.2 Software stack (freeze before kickoff)

Run this once on Day 0 and paste the output into the operator log:

```bash
cd /home/qujiaxiang/project/PET_LatentResidual
git rev-parse HEAD > /tmp/round7_codebase_hash.txt
git status --porcelain > /tmp/round7_git_dirty.txt
/home/qujiaxiang/.conda/envs/rae/bin/python --version > /tmp/round7_python.txt
/home/qujiaxiang/.conda/envs/rae/bin/python -c "import torch; print(torch.__version__, torch.version.cuda)" > /tmp/round7_torch.txt
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > /tmp/round7_gpu.txt

cat /tmp/round7_*.txt
```

**Required condition**: `round7_git_dirty.txt` must be empty (no uncommitted changes). If not empty, commit or stash first.

**Why this matters**: Reviewer §07 §2 row 4 ("σ cannot be computed ... mixed codebase") triggers an R6 revocation penalty (−0.3 Causal). One codebase hash for all 4 σ_seed runs is non-negotiable.

### 2.3 Data paths (must exist and be readable)

```bash
ls /data_2/qujiaxiang/lowdose_pet_ct/latents_224/latents_train.pt
ls /data_2/qujiaxiang/lowdose_pet_ct/latents_224/latents_val.pt
ls /data_2/qujiaxiang/outputs/PET_LatentResidual/alignment_224_audit/alignment_224_clip3_audit.json
```

All three must be non-zero size. If any missing, stop and call Author.

---

## 3. Day 0 Preflight Checklist

Operator runs in order, checks each line:

- [ ] Read this doc top to bottom (takes ~10 min)
- [ ] §2.2 software stack captured to `/tmp/round7_*.txt`
- [ ] §2.3 data files present
- [ ] `nvidia-smi` shows expected GPUs idle
- [ ] `df -h /data_2` shows ≥ 200 GB free (for checkpoints + logs)
- [ ] Previous `imgaux_boost_seed42` checkpoint path known (if reusing for cross-reference only — not as one of the 4 new σ runs; see Trap 1 in signoff)
- [ ] Author confirms no training is currently running on the target GPUs

**Pass condition**: all 7 items checked. Record timestamp in operator log.

---

## 4. Artifact A1 — σ_seed-lite (4 seeds × imgaux_boost × 50K)

### 4.1 Why 4 seeds, not 3

Proposer R6 proposed 3 seeds {42, 123, 456} with seed=42 reused from old run. Reviewer signoff §Trap 1 flagged this as codebase-contaminated. Reviewer signoff §Trap 2 recommends n=4 on a single codebase to get a usable σ CI.

**Agreed plan**: 4 fresh seeds, **all on current codebase hash** (from §2.2).

### 4.2 Configs to create

Author creates 4 config files. They are copies of `pet_flow_first_hop_224_50k_imgaux_boost.yaml` with only `run_name` and `seed` changed. Nothing else touches them.

```bash
cd /home/qujiaxiang/project/PET_LatentResidual/configs/pet_flow
cp pet_flow_first_hop_224_50k_imgaux_boost.yaml pet_flow_first_hop_224_50k_sigma_s42.yaml
cp pet_flow_first_hop_224_50k_imgaux_boost.yaml pet_flow_first_hop_224_50k_sigma_s123.yaml
cp pet_flow_first_hop_224_50k_imgaux_boost.yaml pet_flow_first_hop_224_50k_sigma_s456.yaml
cp pet_flow_first_hop_224_50k_imgaux_boost.yaml pet_flow_first_hop_224_50k_sigma_s789.yaml
```

Author edits each file. Only these two lines change per file:

| Config file | `run_name:` | `seed:` |
|---|---|---|
| `..._sigma_s42.yaml` | `first_hop_224_50k_sigma_s42` | `42` |
| `..._sigma_s123.yaml` | `first_hop_224_50k_sigma_s123` | `123` |
| `..._sigma_s456.yaml` | `first_hop_224_50k_sigma_s456` | `456` |
| `..._sigma_s789.yaml` | `first_hop_224_50k_sigma_s789` | `789` |

**Verification before launch** (operator runs):

```bash
cd /home/qujiaxiang/project/PET_LatentResidual
for s in 42 123 456 789; do
  echo "=== sigma_s$s ==="
  diff configs/pet_flow/pet_flow_first_hop_224_50k_imgaux_boost.yaml \
       configs/pet_flow/pet_flow_first_hop_224_50k_sigma_s$s.yaml
done
```

**Expected output**: each diff shows exactly 2 lines changed (`run_name`, `seed`). If any diff shows extra lines, **stop** — config is wrong.

### 4.3 Launch commands

If 4 GPUs available (preferred, parallel, ~16h):

```bash
cd /home/qujiaxiang/project/PET_LatentResidual
mkdir -p logs/round7

for i in 0 1 2 3; do
  seed=(42 123 456 789)[$i]  # zsh syntax; operator adapts for bash as needed
done

# Pragmatic explicit commands (copy-paste each in its own tmux/screen window):
CUDA_VISIBLE_DEVICES=0 nohup /home/qujiaxiang/.conda/envs/rae/bin/python -u train_first_hop.py \
    --config configs/pet_flow/pet_flow_first_hop_224_50k_sigma_s42.yaml \
    > logs/round7/sigma_s42.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup /home/qujiaxiang/.conda/envs/rae/bin/python -u train_first_hop.py \
    --config configs/pet_flow/pet_flow_first_hop_224_50k_sigma_s123.yaml \
    > logs/round7/sigma_s123.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup /home/qujiaxiang/.conda/envs/rae/bin/python -u train_first_hop.py \
    --config configs/pet_flow/pet_flow_first_hop_224_50k_sigma_s456.yaml \
    > logs/round7/sigma_s456.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup /home/qujiaxiang/.conda/envs/rae/bin/python -u train_first_hop.py \
    --config configs/pet_flow/pet_flow_first_hop_224_50k_sigma_s789.yaml \
    > logs/round7/sigma_s789.log 2>&1 &

jobs
```

If only 2 GPUs, run pairs sequentially (42,123 on GPU0,1 → 456,789). Wall-clock doubles to ~32h.

If only 1 GPU, run all 4 serially. Wall-clock ~64h.

### 4.4 Monitoring

While running, operator checks every 4h:

```bash
tail -n 20 logs/round7/sigma_s42.log
nvidia-smi
```

**Expected**: `step XXXX/50000`, loss decreasing, GPU util > 70%. If any run NaNs or stalls > 30min, kill that run and call Author.

### 4.5 Full-val evaluation after each run

Each run auto-saves `best.pt` and `last.pt`. For each, run the **full-val rerank** (the 0423 protocol that the project uses for leaderboard hygiene):

```bash
for s in 42 123 456 789; do
  CUDA_VISIBLE_DEVICES=0 /home/qujiaxiang/.conda/envs/rae/bin/python -u eval_first_hop_224_clip3.py \
      --config configs/pet_flow/pet_flow_first_hop_224_50k_sigma_s$s.yaml \
      --checkpoint /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_sigma_s$s/best.pt \
      --split val --max-slices 0 --batch-size 8 \
      --out /data_2/qujiaxiang/outputs/PET_LatentResidual/diagnostics/round7_sigma/sigma_s${s}_best.json
done
```

### 4.6 Aggregation (Author or operator runs)

```python
# scripts/round7/compute_sigma_seed.py  -- Author writes this or operator adapts
import json
import numpy as np
from pathlib import Path

runs = {}
for s in [42, 123, 456, 789]:
    path = Path(f"/data_2/qujiaxiang/outputs/PET_LatentResidual/diagnostics/round7_sigma/sigma_s{s}_best.json")
    with open(path) as f:
        d = json.load(f)
    runs[s] = d["transport_avg"]  # key name matches eval output
print("per-seed transport_avg:", runs)
vals = np.array(list(runs.values()))
print(f"mean = {vals.mean():.4f}  std = {vals.std(ddof=1):.4f}")
print(f"σ_upper_80 ≈ 1.5 × σ = {1.5*vals.std(ddof=1):.4f}")
```

### 4.7 Pass/fail gate (per reviewer §07 §2)

| σ_hat | Score impact (reviewer §07 §2) | 200K decision |
|---|---|---|
| ≤ 0.04 | +0.3 Causal | 200K allowed with strict σ-unit gate |
| 0.04–0.08 | +0.2 Causal | 200K allowed with σ_upper_80 gate |
| > 0.08 | +0.1 Causal | **Abandon 200K**, write null paper |
| Cannot compute | −0.3 Causal (R6 revocation) | — |

Artifact A1 delivered: **`/data_2/qujiaxiang/outputs/PET_LatentResidual/diagnostics/round7_sigma/summary.json`** with keys `{per_seed, mean, std, sigma_upper_80, decision}`.

---

## 5. Artifact A2 — Path A (TF vs Rollout Gap)

### 5.1 Required checkpoints

- `imgaux_boost_best` (the C checkpoint; full-val reranked best.pt)
- `pixenc_ablation_best` (the N1 checkpoint; full-val reranked best.pt)

Both checkpoints should already exist from earlier training. Operator verifies:

```bash
ls -lh /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_imgaux_boost/best.pt
ls -lh /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_pixenc_ablation/best.pt
```

If either is missing, **stop** and call Author. Do not substitute with `last.pt` silently — that would breach the reviewer §07 §3 "2 checkpoints required" rule.

### 5.2 Launch

Script already exists: [`scripts/diagnose_tf_rollout_gap.py`](../../scripts/diagnose_tf_rollout_gap.py)

```bash
cd /home/qujiaxiang/project/PET_LatentResidual

# Checkpoint C
CUDA_VISIBLE_DEVICES=0 /home/qujiaxiang/.conda/envs/rae/bin/python -u \
    scripts/diagnose_tf_rollout_gap.py \
    --config     configs/pet_flow/pet_flow_first_hop_224_50k_imgaux_boost.yaml \
    --checkpoint /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_imgaux_boost/best.pt \
    --split val --max-slices 0 --batch-size 8 --device cuda:0 \
    --out-dir   /data_2/qujiaxiang/outputs/PET_LatentResidual/diagnostics/round7_pathA/C

# Checkpoint N1
CUDA_VISIBLE_DEVICES=0 /home/qujiaxiang/.conda/envs/rae/bin/python -u \
    scripts/diagnose_tf_rollout_gap.py \
    --config     configs/pet_flow/pet_flow_first_hop_224_50k_pixenc_ablation.yaml \
    --checkpoint /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_pixenc_ablation/best.pt \
    --split val --max-slices 0 --batch-size 8 --device cuda:0 \
    --out-dir   /data_2/qujiaxiang/outputs/PET_LatentResidual/diagnostics/round7_pathA/N1
```

### 5.3 Expected duration

~4 hours per checkpoint on a 3090/4090 (eval only, no training). Total ~8 GPU-hours = 0.35 GPU-day.

### 5.4 Expected outputs (per reviewer §07 §3 consumption)

- `.../round7_pathA/C/tf_rollout_gap_val.json` with keys `downstream_summary.mean_exposure_gap_dB` and `mean_ceiling_gap_dB`
- `.../round7_pathA/C/tf_rollout_gap_val_per_slice.csv`
- Same two files in `.../round7_pathA/N1/`

Reviewer reads exactly these JSON keys.

### 5.5 Sanity check

`hop0_D50_to_D20` row must show `exposure_gap_dB ≈ 0` (TF == RO by construction). If it doesn't, there's a dataset bug or a non-determinism issue — call Author.

Artifact A2 delivered when all 4 files above are non-zero.

---

## 6. Day 3 Gate — 200K Go/No-Go Meeting

After σ_seed-lite aggregates on Day 3, Author convenes a 10-min decision:

```
σ_hat = <from §4.6>
σ_upper_80 = 1.5 × σ_hat

Decision: 200K ?
  if σ_hat > 0.08:  ABANDON. Skip §8. Proceed to §7 Path C + §9 F2 + §10 Pix2Pix.
  else:             PROCEED. Launch §8 with σ_upper_80 as gate.
```

Decision is recorded in `/data_2/qujiaxiang/outputs/PET_LatentResidual/diagnostics/round7_sigma/200K_decision.md` with:
- σ_hat numeric
- σ_upper_80 numeric
- decision (ABANDON/PROCEED) with reasoning
- timestamp and Author signature

Reviewer §07 §7 Case E explicitly says: **if 200K launches without this file, R6 acceptance is revoked.**

---

## 7. Artifact A3 — Path C (Decoder-FT Diagnostic)

### 7.1 Intent

This is a **diagnostic**, not a new method. It estimates "how much of the 10 dB gap is bound by the frozen RAE decoder". Reviewer §07 §4 grades both extreme outcomes positively.

### 7.2 What to change from imgaux_boost config

Author creates `configs/pet_flow/pet_flow_first_hop_224_pathC_decoder_ft.yaml`:

| Key | Value |
|---|---|
| `training.freeze_rae` | `false` |
| `training.rae_unfreeze_last_n_blocks` | `2` *(requires support in `model_first_hop.build_rae()`; if not supported, Author adds a 5-line helper)* |
| `training.max_steps` | `5000` |
| `training.rae_lr_multiplier` | `0.1` (10× smaller LR for decoder blocks to avoid destroying it) |
| `run_name` | `first_hop_224_pathC_decoder_ft` |
| Resume from | `imgaux_boost_best.pt` (warm-start, not from scratch) |

**Author action**: before operator can launch, Author must verify `model_first_hop.py` supports partial decoder unfreeze. If not, Author adds the helper (30 min of work) or Path C falls back to "full decoder unfreeze with small LR" (riskier but single-line change).

### 7.3 Launch

```bash
CUDA_VISIBLE_DEVICES=1 /home/qujiaxiang/.conda/envs/rae/bin/python -u train_first_hop.py \
    --config configs/pet_flow/pet_flow_first_hop_224_pathC_decoder_ft.yaml \
    --resume-from /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_imgaux_boost/best.pt \
    > logs/round7/pathC.log 2>&1 &
```

### 7.4 Evaluation

After 5K steps:

```bash
CUDA_VISIBLE_DEVICES=1 /home/qujiaxiang/.conda/envs/rae/bin/python -u eval_first_hop_224_clip3.py \
    --config configs/pet_flow/pet_flow_first_hop_224_pathC_decoder_ft.yaml \
    --checkpoint /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_pathC_decoder_ft/last.pt \
    --split val --max-slices 0 --batch-size 8 \
    --out /data_2/qujiaxiang/outputs/PET_LatentResidual/diagnostics/round7_pathC/pathC_val.json
```

### 7.5 Report

Author extracts `ΔPSNR_FT = PSNR_after_NORMAL - PSNR_before_NORMAL` where `before` = imgaux_boost_best at NORMAL.

Artifact A3 delivered: `.../round7_pathC/pathC_val.json` + a 3-line summary in `.../round7_pathC/delta_psnr.md`.

---

## 8. Artifact A6 — 200K Conditional Run

**Only if Day 3 Gate (§6) = PROCEED.**

### 8.1 Config

Use existing v3 200K config if present, else Author creates `pet_flow_first_hop_224_200k_v3.yaml` based on `imgaux_boost.yaml` with `training.max_steps: 200000`.

### 8.2 Pre-registered gate record

Before launch, Author writes `/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_200k_v3/GATE.md`:

```
σ_seed measurement: σ_hat = <value> (from round7_sigma/summary.json)
σ_upper_80 = 1.5 × σ_hat = <value>

SUCCESS condition (at 200K):
  mean(200K) - mean(50K) > 3·σ_upper_80                   AND
  paired bootstrap 95% CI lower bound > σ_upper_80

ABANDON condition (at 150K):
  ΔPSNR(150K - 50K) < σ_upper_80                          OR
  ΔPSNR(150K - 100K) < 0.5·σ_upper_80

Pre-registered on: <timestamp>
Signed: Author (human)
```

This file MUST exist before the run starts. Reviewer §07 §7 Case E flag.

### 8.3 Launch

```bash
CUDA_VISIBLE_DEVICES=3 nohup /home/qujiaxiang/.conda/envs/rae/bin/python -u train_first_hop.py \
    --config configs/pet_flow/pet_flow_first_hop_224_200k_v3.yaml \
    > logs/round7/v3_200k.log 2>&1 &
```

### 8.4 Mid-training check at 150K

At step 150000, operator pauses and reports:
```
PSNR@150K = <val>
PSNR@100K = <val>
PSNR@50K  = <val (from sigma_s42 best.pt)>
```

Author compares against GATE.md conditions and decides continue/abandon. Result recorded in `GATE.md` (append).

---

## 9. Artifact A4 — F2 Rewrite

**Not blocking experiments.** Can happen any day during the 14-day window.

### 9.1 What to rewrite

In `PET_LatentResidual/CLAUDE.md` and `PET_LatentResidual/IDEA_REPORT.md`, find any instance of:

- "decoder amplifies tail error"
- "decoder 对 tail 误差更敏感"
- "每 0.0001 MSE: D20 = 5.4 dB, NORMAL = 9.4 dB" used to support hop0 story
- Anything implying off-manifold amplification is larger at NORMAL

Replace with either:

**Option A** (explanation): "tail-bigger-than-head dB gain reflects that Gap_Transport itself grows monotonically down-chain (10.74 → 15.70 dB per E1); same fractional latent improvement shows as a larger dB number at NORMAL than at D20. This is a dB-unit artifact, not evidence of hop0 mechanism efficacy."

**Option B** (metric change): report all per-hop improvements as `ΔPSNR / Gap_Transport` (unitless) or `1 − (MSE_new / MSE_old)` (relative MSE reduction) throughout the paper.

### 9.2 Verification

After edit, operator (or Author) runs:

```bash
grep -rni "decoder.*amplif\|decoder 对 tail\|0.0001 MSE" \
    PET_LatentResidual/CLAUDE.md PET_LatentResidual/IDEA_REPORT.md
```

**Pass condition**: no output. Artifact A4 delivered when this grep returns empty.

---

## 10. Artifact A5 — Pix2Pix Baseline (lowest priority)

**Can be skipped if budget is tight.** Reviewer §07 §6 says score delta is 0 if not reported (no penalty, just no bonus).

### 10.1 Minimal spec

- Standard U-Net generator + PatchGAN discriminator (pix2pix original)
- Input: D50 raw image (224×224, clamp_max=10.0)
- Output: NORMAL raw image (same space)
- Train on same train split (26,247 slices)
- Evaluate on same val split (7,403 slices) with `calc_psnr_clip3`
- Report NORMAL PSNR only (single number)

### 10.2 Recommended implementation

Use a published pix2pix repo (e.g., https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) with minor adaptation for 1-channel medical images.

Artifact A5 delivered: `.../diagnostics/round7_pix2pix/pix2pix_val_normal.json` with key `psnr_normal`.

---

## 11. Artifact Paths (Reviewer Round 7 reads these exact paths)

```
/data_2/qujiaxiang/outputs/PET_LatentResidual/diagnostics/
├── round7_sigma/
│   ├── sigma_s42_best.json
│   ├── sigma_s123_best.json
│   ├── sigma_s456_best.json
│   ├── sigma_s789_best.json
│   ├── summary.json           # A1 primary artifact
│   └── 200K_decision.md       # Day 3 gate record
├── round7_pathA/
│   ├── C/
│   │   ├── tf_rollout_gap_val.json           # A2 primary
│   │   └── tf_rollout_gap_val_per_slice.csv
│   └── N1/
│       ├── tf_rollout_gap_val.json           # A2 primary
│       └── tf_rollout_gap_val_per_slice.csv
├── round7_pathC/
│   ├── pathC_val.json         # A3 primary
│   └── delta_psnr.md
└── round7_pix2pix/            # A5 (optional)
    └── pix2pix_val_normal.json

/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_200k_v3/
└── GATE.md                    # A6 pre-registered gate + outcome

/home/qujiaxiang/project/PET_LatentResidual/
├── CLAUDE.md                  # A4 (grep passes)
└── IDEA_REPORT.md             # A4 (grep passes)
```

If any path above is missing when operator declares "done", the artifact is not counted.

---

## 12. Troubleshooting (pre-documented blockers)

### 12.1 OOM on σ_seed-lite launch

- Lower `data.batch_size` from 8 to 4 in the 4 sigma configs
- **But do this in ALL 4 configs identically** (reviewer §07 §2 row 4: codebase/config change across seeds = disqualification)

### 12.2 One σ_seed run diverges (NaN)

- Kill that run, restart with same seed
- If NaN reproduces with same seed, call Author (this is a real bug, not a noise event)
- Do NOT "replace that seed" with a different seed silently — document the NaN event in operator log

### 12.3 Path A sanity check fails (hop0 shows nonzero exposure_gap)

- Check that `z_d50` is being used as both TF and RO input at hop 0
- Likely cause: non-deterministic model eval (dropout still active, BN not in eval mode)
- Verify `model.eval()` is called before `sample_one_step_first_hop`
- Call Author if unresolved

### 12.4 Path C unfreeze not supported in model_first_hop.py

- Fallback: full decoder unfreeze with `training.rae_lr: 1e-6` and `training.max_steps: 3000`
- Document the fallback choice in `pathC_val.json` metadata
- This is less clean but still answers the RAE-bound question

### 12.5 Budget emergency: must cut GPU-day

Priority drop order (highest priority kept first):

1. **Keep**: A1 σ_seed-lite (4 GPU-day) — cannot remove
2. **Keep**: A2 Path A (1 GPU-day) — cannot remove
3. **Drop if needed**: A3 Path C (1 GPU-day)
4. **Drop if needed**: A5 Pix2Pix (2 GPU-day)
5. **Drop if σ_hat > 0.08**: A6 200K (2.7 GPU-day)

A4 F2 rewrite has zero GPU cost; never drop.

---

## 13. Operator Completion Report Template

When operator finishes, fills in and commits to `review/0424/response_to_reviewer/operator_completion_report.md`:

```markdown
# Operator Completion Report — Round 7

## Preflight (§3)
- Codebase hash: <git hash>
- Python: <version>
- PyTorch: <version>
- GPUs used: <list>

## σ_seed-lite (§4)
- Seeds run: 42, 123, 456, 789  (all on hash <git hash>)
- Completion timestamps: <list>
- σ_hat: <value>   σ_upper_80: <value>
- 200K decision: ABANDON / PROCEED

## Path A (§5)
- C completion: <timestamp>, exposure_gap=<val>, ceiling_gap=<val>
- N1 completion: <timestamp>, exposure_gap=<val>, ceiling_gap=<val>
- hop0 sanity check (exposure_gap ≈ 0): PASS / FAIL

## Path C (§7)
- Executed: YES / NO
- ΔPSNR_FT at NORMAL: <value>
- Fallback used: NO / YES (describe)

## 200K (§8)
- Executed: YES / NO / ABANDONED_AT_150K
- GATE.md path: <path>
- Result at 200K: <PSNR>
- Gate triggered: PASS / ABANDON at step <N>

## F2 Rewrite (§9)
- grep check: PASS / FAIL
- Files modified: <list>

## Pix2Pix (§10)
- Executed: YES / NO (skipped due to <reason>)
- NORMAL PSNR: <value>

## Blockers encountered
<list, referencing §12 entries when applicable>

## Deviations from this doc
<if none, write "none">
```

---

## 14. Revision Policy

This is a **joint design doc**. Changes require agreement from both Author and Reviewer:

- **Author-only changes allowed**: adding clarifications, fixing typos, adding more troubleshooting entries.
- **Reviewer-only changes allowed**: tightening gate criteria (never loosening).
- **Joint changes required**: changing any artifact delivery path, adding/removing artifacts, changing σ_seed budget, changing 200K stop criterion.

Version: 1.0 (2026-04-24). Bump version + signed commit message on any joint change.

---

## 15. Closing

This document is the operator's complete instruction set. Reviewer's Round 7 scoring rules are in `07_pre_registered_verdict_map.md`. Author's role after kickoff is to interpret data and field operator questions. **Operator does not negotiate with the reviewer directly.**

Go time.

---

*Signed*:
- Author (Proposer): **pending signature**
- Reviewer (Nightmare Panel): **approved, 2026-04-24**
- Remote operator: **pending acknowledgement via §13 report**
