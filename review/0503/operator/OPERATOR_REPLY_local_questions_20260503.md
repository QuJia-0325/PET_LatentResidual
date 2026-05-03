# Operator Reply to Local Questions — 2026-05-03

**Repo**: `/home/qujiaxiang/project/PET_LatentResidual`  
**Branch**: `foc_lite_hop0`  
**HEAD**: `2f9296eb1ac5accb98b8b7ce9f4d2b2a70113314`  
**Remote**: `origin = git@gitee.com:jqu9/PET_LatentResidual.git`  
**Environment**: `/home/qujiaxiang/.conda/envs/rae/bin/python`, PyTorch `2.2.0+cu121`, CUDA `12.1`, 4× RTX A6000 48GB, driver `570.144`.

This file answers `review/0503/local/OPERATOR_QUESTIONS.md` Q1-Q8 from the GPU host side.

---

## Executive Answer

The current remote state is safer than the local side feared: **A_main and C_uniform are not running yet**, and `A_pair_uniform_spot` is also not running. Only the sigma-normalize **full sanity gate** is in progress (`A_sanity` finished 20K; `B` is around 10.4K/20K), plus unrelated/parallel `V6.1` and `sanity_light` jobs.

This means REV1 can still be fixed before A_main reaches 60K, because A_main has not started.

However, there are three immediate blockers in the latest protocol/tooling:

1. **Metrics schema mismatch**: real `metrics.jsonl` uses `step`, not `global_step`. Current `lock_effect_size_threshold.py` and `paired_diff_judge.py` require `global_step`, so they will return zero observations on real training logs unless patched.
2. **Wrong default remote in lock script**: `lock_effect_size_threshold.py` defaults to `--remote gitee`, but this repo only has remote `origin`. Running the script without `--remote origin` will commit locally and fail to push.
3. **Method-D checkpoint filename mismatch**: `select_best_ckpt_smoothed.py` searches `ckpt_step_*.pt` / `ckpt_last.pt`, but our training saves `step_020000.pt` / `last.pt`.

These are implementation blockers independent of the statistical REV1 issues already identified by local/GPT/Claude.

---

## Q1 — Which ablation branches are actually running?

### Current first-hop related GPU processes

| Process | GPU | Status | Current / target step | Notes |
|---|---:|---|---:|---|
| `review/0502/scripts/run_ablation.sh sanity` → `B` | 1 | running | ~10.4K / 20K | full sanity gate second half |
| `A_sanity` | 1 | completed | 20K / 20K | same sanity command first half |
| `first_hop_224_v6_1_rollout_floor` | 3 | running | ~81.6K / 200K | not part of sigma-normalize A/C/D matrix |
| `A_sanity_light` | 2 | running | ~25.2K / 50K | light sanity only; no chain metrics authority |

### Not running / no output directory exists

| Planned run | Current state |
|---|---|
| `A_main` | not running; metrics path missing |
| `C_uniform` | not running; metrics path missing |
| `A_pair_uniform_spot` | not running; metrics path missing |
| `D_closed_form` | not running |

Confirmed missing paths:

```text
/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0502_runs/A_main/run/first_hop_224_sigma_norm_A_main/metrics.jsonl
/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0502_runs/C/run/first_hop_224_sigma_norm_C/metrics.jsonl
/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0502_runs/A_pair_uniform_spot/run/first_hop_224_sigma_norm_A_pair_uniform_spot/metrics.jsonl
```

### Answer to Q1

There are **zero formal A/C/D sigma-normalize main ablation branches currently running**. The only related job is the **sanity gate**.

`A_pair_uniform_spot` is not intentionally assumed to be unnecessary. It simply has not been launched. If §6.4 Risk 4 is kept as a paper robustness claim, this run is still needed after sanity PASS and after the main A/C decision path is clarified.

Recommended priority:

1. Finish full sanity `A_sanity + B`.
2. Apply REV1 protocol/tooling fixes before launching A_main/C_uniform.
3. Launch A_main.
4. Launch C_uniform only under the post-lock / blinded policy that REV1 defines.
5. Launch `A_pair_uniform_spot` as the Risk 4 robustness check if still needed.

---

## Q2 — Can A_main reach step 60000?

### Config answer

`review/0502/configs/A_control.yaml` has:

```text
training.max_steps = 120000
training.eval_interval = 400
training.save_interval = 20000
training.max_val_batches = 64
training.val_window_mode = rolling
```

So the configured A_main endpoint is **120K**, and the locked window `[40000, 60000]` is inside the intended run.

### Current run answer

A_main is **not started**, so it has not approached 60K. This is good: REV1 can still patch the protocol before A_main generates the window used for locking.

### Resource risk

Current disk status:

```text
/data_2: 44T total, 20T used, 22T available, 47% used
/home root: 1.8T total, 1.6T used, 110G available, 94% used
```

Because training outputs are under `/data_2`, data-disk space is not the blocker. Root is tight but not immediately fatal unless scripts/logs accidentally write large products under `/home`.

GPU memory feasibility is likely OK: V6/V6.1 and sanity jobs use ~18.4GB per training process on A6000 48GB. OOM risk is not the primary concern.

### Fallback recommendation

Do **not** pre-authorize shrinking the lock window to `[30000, 50000]` or `[terminal-20000, terminal]` now. Since A_main has not started and resources are sufficient, the correct path is:

```text
wait/fix REV1 -> launch A_main -> require A_main to cross 60000 -> lock using the predeclared window
```

Only if a real failure occurs should a deviation window be considered, and it should be documented as a protocol deviation, not as the default path.

---

## Q3 — Should we add a post-lock C_uniform full-val gate wrapper?

Yes. Strong recommendation: **add the wrapper**.

Reason: the current human-memory-only rule is too weak. The local reviewers are correct that a full-val C result can be produced manually before `EFFECT_SIZE_LOCKED.md` exists, and the current lock script only detects some artifacts after the fact.

Minimum wrapper behavior should be:

1. Refuse C full-val unless `review/0502/EFFECT_SIZE_LOCKED.md` exists.
2. Verify the lock commit is reachable from `origin/foc_lite_hop0`.
3. Refuse if the working tree contains uncommitted protocol/script changes.
4. Run `eval_first_hop_224_clip3.py --max-slices 0` through a single canonical entrypoint.
5. Write a small immutable-ish run manifest recording lock file SHA256, eval command, config path, checkpoint path, git HEAD, and timestamp.

Important implementation detail: use remote `origin`, not `gitee`, unless a `gitee` remote is explicitly added. Current repo remote is only:

```text
origin git@gitee.com:jqu9/PET_LatentResidual.git
```

---

## Q4 — What does the real metrics schema look like?

A_main metrics cannot be provided because A_main has not started.

I created a schema reference from the completed `A_sanity` run:

```text
review/0503/operator/A_sanity_metrics_tail5_schema_reference.jsonl
```

The schema is flat JSONL, but the step field is **`step`**, not `global_step`.

Example keys:

```json
{
  "event": "val",
  "step": 20000,
  "val_select_score": 0.0020769658252561387,
  "val_chain_normal_mse": 0.0005872322064988111,
  "val_rollout_total": 0.0020515464821073692,
  "val_main_window_start_batch": 3136.0
}
```

This is a blocker for current scripts:

- `lock_effect_size_threshold.py` filters rows with `"global_step"`; it will find `N=0` on our real metrics.
- `paired_diff_judge.py` also indexes by `"global_step"`; it will fail to find shared steps.
- `select_best_ckpt_smoothed.py` also uses `global_step` in smoothing, so it needs the same compatibility patch.

Required patch:

```python
def get_step(row):
    if "global_step" in row:
        return int(row["global_step"])
    if "step" in row:
        return int(row["step"])
    return None
```

Then all three scripts should use this helper.

---

## Q5 — Does decoder ceiling already exist?

Yes. It was already run on full val on 2026-04-22.

Artifacts:

```text
/data_2/qujiaxiang/outputs/PET_LatentResidual/gt_latent_decoder_ceiling_20260422/gt_latent_decoder_ceiling_clip3_val_summary.json
/data_2/qujiaxiang/outputs/PET_LatentResidual/gt_latent_decoder_ceiling_20260422/gt_latent_decoder_ceiling_clip3_val_per_slice.csv
/data_2/qujiaxiang/outputs/PET_LatentResidual/gt_latent_decoder_ceiling_20260422/gt_latent_decoder_ceiling_val_gpu2_20260422_1632.log
review/0422/exp/gt_latent_decoder_ceiling_clip3_val_20260422.md
```

Full-val `n=7403`, metric `calc_psnr_clip3`, GT latent direct decode:

| Timepoint | Mean PSNR clip3 | Std | N |
|---|---:|---:|---:|
| D50 | 42.6224 | 7.2563 | 7403 |
| D20 | 46.6356 | 6.3883 | 7403 |
| D10 | 48.7436 | 6.0240 | 7403 |
| D4 | 50.8319 | 5.7902 | 7403 |
| NORMAL | 52.6341 | 5.7972 | 7403 |

So the paper can use this as a separate decoder-ceiling anchor, with the caveat already noted by reviewers: do **not** use it to redefine the A-vs-C threshold post hoc.

---

## Q6 — Should `train_v21.py` move to deprecated?

Operator recommendation: **yes, but not in the same commit as REV1 protocol fixes**.

Rationale:

- `train_v21.py` is explicitly legacy/reproduction-only in README.
- It is unrelated to the sigma-normalize ablation and should not be mixed with preregistration-sensitive changes.
- Moving it now would dirty the protocol diff and make review harder.

Recommended handling:

1. Finish REV1 protocol/tooling fixes first.
2. In a separate cleanup commit, move:

```text
train_v21.py -> deprecated/v21/train_v21.py
```

3. Update README references in the same cleanup commit.

---

## Q7 — Should EFFECT_SIZE_LOCKED.md record environment info?

Yes. Add environment capture.

This is cheap and useful for reviewer defense. At minimum write:

```text
python executable
python version
torch.__version__
torch.version.cuda
CUDA_VISIBLE_DEVICES
nvidia-smi driver_version
GPU model / memory
repo HEAD
remote branch
metrics SHA256 and byte count
```

Current environment from this host:

```text
python: /home/qujiaxiang/.conda/envs/rae/bin/python
python version: 3.10.19
torch: 2.2.0+cu121
cuda: 12.1
GPU: NVIDIA RTX A6000 49140 MiB x4
driver: 570.144
```

Also fix the script remote default to `origin` or require `--remote origin` in the documented command.

---

## Q8 — Are `sigma_seed123` / `sigma_seed456` usable as A_main two seeds?

No. They should **not** be used as A_main second-seed evidence for REV1.

Evidence:

1. There are config files:

```text
configs/pet_flow/pet_flow_first_hop_224_50k_sigma_seed123.yaml
configs/pet_flow/pet_flow_first_hop_224_50k_sigma_seed456.yaml
```

2. There are **no corresponding output directories** under:

```text
/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_sigma_seed123
/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_sigma_seed456
```

3. The configs are not current A_main/V6 transport-first. They are old 50K Scheme C/imgaux-style configs:

| Field | sigma_seed123/456 | current A_main / V6 |
|---|---|---|
| `max_steps` | 50000 | 120000 / 200000 |
| `seed` | 123 / 456 | 42 |
| `rollout.lambda_start/end` | 0.02 -> 0.25 | 0.0 -> 4.0 |
| `rollout.warmup/ramp` | 0.1 / 0.3 | 0.25 / 0.5 |
| `rollout.step_weights` | [1.0, 1.1, 1.2, 1.3] | [0.5, 2.0, 1.5, 1.0] |
| `pair_loss_weights` | [1.0, 1.05, 1.1, 1.2] | [2.5, 1.0, 1.0, 1.0] |
| `best_metric_terms.d20` | 0.15 | 0.50 |

4. Historical docs already flagged them as old naming/configs:

```text
review/0424/consensus_agreement.md: old sigma_seed123/456 need deletion or replacement
review/0424/GPT-operator/remote_audit_assessment_20260424.md: sigma seed consensus not actually landed
review/0424/response_to_reviewer/proposer_final_response_r6.md: these were based on imgaux_boost, old parameters, no 0422 fix
```

### Answer to Q8 category

They are closest to option **(c) abandoned/obsolete old experiment configs**, not A_main seeds.

Do not use them for REV1 paired SD. If REV1 requires a two-seed paired-difference SD for A_main, we need to create a fresh A_main second-seed config from `review/0502/configs/A_control.yaml`, changing only:

```text
seed
run_name
output tag
```

Budget estimate: roughly one A_main-equivalent run to at least the lock window; if only used for noise scale around `[40000,60000]`, it must reach 60K. That is not zero-cost.

---

## Additional Blockers Not Explicitly Asked by Local

### B1 — `lock_effect_size_threshold.py` default remote is wrong

The script default is:

```text
--remote gitee
```

But this repository has only:

```text
origin git@gitee.com:jqu9/PET_LatentResidual.git
```

Patch recommendation: default to `origin`, or detect remotes and prefer `origin` if `gitee` is absent.

### B2 — `select_best_ckpt_smoothed.py` checkpoint filename assumptions are wrong

The script searches:

```text
ckpt_step_*.pt
ckpt_last.pt
```

Our training writes:

```text
step_020000.pt
step_040000.pt
...
best.pt
last.pt
```

Patch recommendation: support both naming schemes.

### B3 — A_main/C_uniform should not be started until REV1 is patched

Because A_main has not started, the cleanest path is available:

```text
patch REV1 -> push -> finish sanity -> launch A_main -> lock X after A_main crosses 60K -> launch/evaluate C under wrapper
```

Starting A_main/C before patching REV1 would create avoidable audit risk.

---

## Recommended Immediate Actions

1. Patch scripts for `step` vs `global_step` compatibility.
2. Patch `lock_effect_size_threshold.py` remote default and hash-race issue.
3. Patch decision rule to handle `rel_diff <= -X` explicitly.
4. Patch Guard 3 / C full-val wrapper before C is launched or evaluated.
5. Patch `select_best_ckpt_smoothed.py` to detect `step_*.pt` and `last.pt`.
6. Do not use `sigma_seed123/456` for REV1; create a fresh A_main second seed if two-seed paired SD is mandatory.
7. Keep decoder-ceiling artifacts as already completed full-val anchor.
