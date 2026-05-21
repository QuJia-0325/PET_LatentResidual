# Peer Review Round 17-Prep — Codex Task F0 + A4

- date: 2026-05-22
- reviewer: GitHub Copilot
-主审对象: `CODEX_TASK_ROUND17_F0_A4_20260522.md`
- scope: execution feasibility only, not Round 17 strategy re-review

## 0. Push Decision

**MODIFY-BEFORE-PUSH.**

F0 + A4 is the right execution direction under Round 17 integration, but the task md is not ready for Codex as-is. It contains execution-breaking path/CLI mismatches, an unsafe cross-substrate paired-t template, and a self-check anti-check pattern that does not actually guard against forbidden variants. These are fixable in the task md; I do not recommend rejecting the plan.

Highest-risk blockers:

1. The canonical eval script uses `--out-dir`, not `--output-dir`; A4 eval and V7 fallback eval commands will fail as written.
2. F0 data paths/schema are wrong: V18 artifacts live under `fullval_eval_20260518/artifacts/`, JSONs are summaries, and per-slice data are CSVs. V13/V14 per-slice CSVs are not in the repo, though eval logs show they were saved under `/data_2/.../eval_0521_v13_v14_fullval_psnr_clip3/`.
3. The F0 template says V18-cap must not mix with chain PSNR, but its loop still computes V18-cap vs V7 for `NORMAL`.
4. The §5 `anti_check ... find ... | grep -q .` pattern is a false guard: the function runs in a pipeline/subshell and does not reliably update `ERR`; `find` itself exits 0 even when no files match.

## 1. Eight Questions

### Q1 — F0 Data Sources

**MODIFY.** V7 per-slice is not actually unknown locally: `review/0511/fullval_psnr_clip3_20260516_173941/artifacts/planf_v7_best_fullval_psnr_chain_mse_per_slice.csv` exists and has the expected `psnr_NORMAL` schema.

V18 per-slice also exists, but at different paths from the task md:

- `review/0517/V18_decoder_lora/fullval_eval_20260518/artifacts/v18_best_fullval_psnr_chain_mse_per_slice.csv`
- `review/0517/V18_decoder_lora/fullval_eval_20260518/artifacts/v18_last_fullval_psnr_chain_mse_per_slice.csv`

The V18 JSON files do **not** contain per-slice arrays; they contain summary stats. V13/V14 local repo files are also summary JSONs only. However, `review/0516/logs_eval/*fullval_psnr_clip3_20260521.log` records saved CSV paths such as `/data_2/qujiaxiang/outputs/PET_LatentResidual/eval_0521_v13_v14_fullval_psnr_clip3/v13_true_image_aux_off_best_fullval_psnr_chain_mse_per_slice.csv`.

Required fix: add F0.0a to locate/copy V13/V14 per-slice CSVs from `/data_2` into `review/0521/F0_inputs/`; only rerun canonical eval if those CSVs are absent. Do not describe F0 as purely existing repo data until those files are copied or verified.

### Q2 — F0 Substrate Consistency

**MODIFY.** The narrative correctly identifies the substrate boundary, but the template violates it. This line:

```python
if 'V18-cap' in name and tp != 'NORMAL':
    continue
```

still computes `V18-cap.last` vs V7 chain PSNR at `NORMAL`, which is exactly the forbidden cross-substrate comparison. That is a bug, not a feature.

Required fix: remove `V18-cap.last` from the main chain `DATA`, or add an explicit `substrate` field and assert all main-F0 inputs have `substrate == "chain_rollout_psnr_clip3"`. If a direct-decode F0b is desired, it must use direct-decode references from `KL_DRIFT_PER_SLICE.csv`, not V7 chain CSV.

### Q3 — A4 YAML Four-Field Diff

**APPROVE with self-check hardening.** The four changed fields are the right functional diff for a from-scratch V7 clone:

- `output_dir`
- `run_name`
- `training.image_aux.lambda_start`
- `training.image_aux.lambda_max`

The V7 YAML has image schedule under `training.image_aux`, while pixel/image loss component weights live under `loss.image_aux`. Trainer code reads these separately, so `loss.image_aux.l1_weight / ssim_weight / seam_weight / border_*` should remain unchanged.

Required self-check additions: explicitly assert `loss.image_aux` subtree equality, `backbone.checkpoint_path` equality, `rae.checkpoint_path` equality, `training.rollout.step_weights` equality, `training.image_aux.enabled == true`, and `seed == 42`.

### Q4 — From-Scratch vs Resume

**APPROVE with wording lock.** A4 should be from-scratch if the goal is a clean single-variable comparison against V7. The launch command has no `--resume`, which is correct. Resume-from-V7 would answer a different patch-adaptation question and would not be comparable to V13/V14/V7 from-scratch controls.

Required small fix: write `from-scratch; do not add --resume` directly in §3.2 next to the command, not only in §0/§3.1 prose.

### Q5 — A4 Stop Rule / Scope Creep

**MODIFY.** The prose stop rule is mostly clear, but the mechanical guard only checks `*A4*v2*.yaml`, so `A4_image_aux_lambda_12.yaml`, `A4_cosine.yaml`, or `A4_ramp.yaml` would bypass it.

Required fix: replace the narrow anti-check with a whitelist check: under `review/0521`, the only allowed A4 YAML path is `review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08.yaml`. Also change the stop sentence to `do not launch any additional image_aux variant, regardless of SUCCESS/MARGINAL/REGRESSION`.

### Q6 — Partial Completion / Failure Recovery

**MODIFY.** Current commit order is clean for success but under-specified for failure. If F0 cannot locate per-slice data, or A4 dies at step 100000, Codex needs a safe stop artifact rather than silent non-push or a misleading `step=160000 done` commit.

Required fix: add §4.4 partial-completion protocol:

- F0 blocked: write and commit `review/0521/F0_STATUS_BLOCKED.md` with missing files, commands attempted, and next required action.
- A4 interrupted: write `review/0521/A4_image_aux_lambda_08/A4_STATUS_INTERRUPTED.md`, copy current `metrics.jsonl` snapshot, and use a truthful commit message such as `step=100000 interrupted`, not `done`.
- Any failure: commit current diagnostic state first, then stop; do not proceed to push narrative claims.

### Q7 — Mechanical Self-Checks

**MODIFY.** Several good checks are already present, but two key classes are missing and one class is broken.

Missing checks to add:

- F0 input CSVs exist and each has exactly 7403 data rows after loading.
- F0 script imports and uses `scipy.stats.ttest_rel` or an equivalent paired test, not an unpaired t-test.
- F0 summary/report explicitly labels substrate as `chain_rollout_psnr_clip3`.
- A4 YAML deep equality for `loss.image_aux`, backbone/RAE checkpoint paths, rollout schedule, and step weights.
- A4 launch log contains `lambda_img=0.0800`; warning is acceptable at +5 min only if a later mandatory check is scheduled.

Broken check to fix:

```bash
anti_check "no A4-v2 yaml" find review/0521 -name '*A4*v2*.yaml' 2>/dev/null | grep -q .
```

This does not work as intended. Use a direct `if find ... -print -quit | grep -q .; then ERR=$((ERR+1)); fi` style block, or write a dedicated helper that receives a glob and performs `find ... -print -quit` inside the helper.

### Q8 — NOT-DO List

**APPROVE with tightening.** Most NOT-DO entries are necessary. Keeping V14b/V14c forbidden is acceptable for this Codex task because Round 17 intentionally scoped Codex to F0 + one A4 probe; extra seeds should require a new user decision.

The paper-claim NOT-DO is not misplaced: Codex may write report text, and the report can easily drift into claim language. Keep it, but sharpen the boundary: Codex may push factual data reports and status reports; any paper narrative or claim ledger update needs user review.

Also keep the V18-cap naming rule. The main risk is not the label itself; the risk is letting `A3` become a new blended substrate name that hides direct-decode vs chain rollout.

## 2. Mandatory Edits Before Push

| § | original / issue | required fix | severity |
|---|---|---|---|
| §2 F0.0 | V18 JSON paths omit `artifacts/` and use `v18_decoder_lora_*` names | Use `review/0517/V18_decoder_lora/fullval_eval_20260518/artifacts/v18_{best,last}_fullval_psnr_chain_mse_per_slice.csv` for per-slice; JSONs are summary only | HIGH |
| §2 F0.0 | V13/V14 JSONs are treated as per-slice sources | Add locate/copy step for `/data_2/.../eval_0521_v13_v14_fullval_psnr_clip3/*_per_slice.csv`; rerun eval only if absent | HIGH |
| §2 F0.1 | V7 per-slice listed as unknown | Prefer existing `review/0511/fullval_psnr_clip3_20260516_173941/artifacts/planf_v7_best_fullval_psnr_chain_mse_per_slice.csv`; fallback rerun only if missing | MED |
| §2 F0.1 / §3 A4.4 | Commands use `--output-dir` | Replace with `--out-dir`; `eval_first_hop_fullval_psnr_chain_mse.py` does not define `--output-dir` | HIGH |
| §2 F0.1 | Mentions possible `--save-per-slice` flag | Remove or correct: current evaluator writes `<tag>_fullval_psnr_chain_mse_per_slice.csv` by default | MED |
| §2 F0.2 | `V18-cap.last` remains in main chain DATA and is computed for NORMAL | Remove from main F0 or enforce substrate field/assertion; optional F0b direct-decode must use direct-decode references only | HIGH |
| §2 F0.2 | Loader left as `NotImplementedError` with no exact schema guidance | Specify CSV schema: chain CSV columns are `slice_idx, psnr_D20, psnr_D10, psnr_D4, psnr_NORMAL`; KL drift CSV columns are `ckpt,step,slice_idx,timepoint,psnr_clip3` | MED |
| §3.2 | From-scratch intent could be missed during launch | Add command-adjacent note: `no --resume; from-scratch single-variable V7 clone` | LOW |
| §4.2 | Commit 4 assumes `step=160000 done` | Add interrupted/partial commit protocol and truthful commit messages | MED |
| §5 | Pipeline anti-checks do not reliably update `ERR` and `find` returns 0 on no match | Replace with direct `find ... -print -quit` checks or a helper that evaluates existence internally | HIGH |
| §5 | Scope-creep anti-check only catches `v2` | Whitelist the one allowed A4 YAML; reject all other `*A4*.yaml` / image_aux variant YAMLs under `review/0521` | MED |
| §5 / F0 report | Slice-level paired-t could be overread as seed-level significance | Add caveat: F0 tests paired slice-level effect on fixed runs; it is not a multi-seed variance estimate | MED |

## 3. New Biases / Failure Modes (B75+)

| ID | bias / failure mode | severity | mitigation |
|---|---|---|---|
| B75 | Artifact optimism: summary JSONs are assumed to contain per-slice data | HIGH | Treat per-slice CSV as first-class input; copy all F0 CSVs into `review/0521/F0_inputs/` before analysis |
| B76 | CLI flag fabrication: `--output-dir` / `--save-per-slice` inferred instead of verified | HIGH | Read `argparse` source; use `--out-dir`; note per-slice CSV is default output |
| B77 | Substrate leakage: V18-cap direct decode can slip into chain paired-t via NORMAL branch | HIGH | Add substrate metadata and hard assert; keep direct-decode F0b separate |
| B78 | Self-check theater: anti-check pipelines look protective but do not update `ERR` correctly | HIGH | Replace with direct existence checks and fail-fast shell blocks |
| B79 | Zero-GPU time optimism: F0 may require locating/copying or regenerating V13/V14 per-slice CSVs | MED | Preflight all F0 inputs before promising runtime; if absent, write blocked status or rerun eval explicitly |
| B80 | Slice-level p-value overclaim: paired-t over slices can become a substitute for seed-level uncertainty | MED | Label F0 as slice-level paired evidence, not multi-seed significance |
| B81 | A4 uniqueness overclaim: `only remaining EV` can invite under-review of variant choice | LOW | Keep Round 17 scope, but word as `only approved new GPU experiment in this task` |

## 4. Final Recommendation

Do not push the current task md to Codex yet. Patch the blockers above, especially the eval CLI, F0 input paths, V18-cap substrate exclusion, and self-check anti-checks. After those edits, the task should be ready without reopening Round 17 strategy.
