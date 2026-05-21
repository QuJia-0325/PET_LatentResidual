# Peer Review Round 17-Prep - Review D Audit

- date: 2026-05-22
- reviewer: **Review D (GitHub Copilot)**
- scope: pre-push execution review for `CODEX_TASK_ROUND17_F0_A4_20260522.md`
- prompt: `PEER_REVIEW_PROMPT_round17_prep_codex_task_20260522.md`
- independence note: I did not read any other Round17-prep reviewer draft.

---

## 0. Push Decision

**Decision: MODIFY-BEFORE-PUSH.**

I agree with the signed Round17 strategy: Hybrid B + F0 + A4-light. The task document is directionally right, but it is **not safe to push as-is**. There are multiple mechanical errors that Codex would likely execute literally:

1. F0 points to summary JSON / wrong V18 paths as if they were per-slice data.
2. Canonical eval commands use `--output-dir`, but the script supports `--out-dir`.
3. A4 edits `transport.image_aux.*`, but V7 and `train_first_hop.py` use `training.image_aux.*`.
4. The self-check anti-check pipeline is broken for `find ... | grep` cases.
5. V18-cap is listed inside the chain paired-t `DATA` path and can still be mixed at `NORMAL`.

These are push blockers because they can cause immediate command failure, a no-op A4 yaml, or a contaminated F0 report.

---

## 1. Evidence Checked

I verified these against the local repo:

- `review/0505/operator/scripts/eval_first_hop_fullval_psnr_chain_mse.py` parser supports `--out-dir`; it does **not** support `--output-dir`.
- The same eval script writes both JSON and per-slice CSV by default: `<tag>_fullval_psnr_chain_mse.json` and `<tag>_fullval_psnr_chain_mse_per_slice.csv`.
- V7 per-slice chain CSV already exists in repo at `review/0511/fullval_psnr_clip3_20260516_173941/artifacts/planf_v7_best_fullval_psnr_chain_mse_per_slice.csv`.
- V18 chain per-slice CSV already exists in repo at `review/0517/V18_decoder_lora/fullval_eval_20260518/artifacts/v18_best_fullval_psnr_chain_mse_per_slice.csv` and `.../v18_last_fullval_psnr_chain_mse_per_slice.csv`.
- V18 task paths in §F0.0 omit `artifacts/` and use `v18_decoder_lora_best...`; those exact JSON paths are not present in the repo.
- V13/V14 committed files under `review/0516/full_eval_json/` are summary JSON only. The logs say per-slice CSVs were saved to `/data_2/qujiaxiang/outputs/PET_LatentResidual/eval_0521_v13_v14_fullval_psnr_clip3/`, but those CSVs are not committed under `review/`.
- V7 yaml has `training.image_aux.lambda_start` / `training.image_aux.lambda_max`, not `transport.image_aux.*`.
- `train_first_hop.py` reads `train_cfg.get("image_aux", {})` and schedules `lambda_img` from that section.
- Image loss shape weights live under `loss.image_aux.*` and should remain unchanged for A4.

---

## 2. Q1 - F0 Data Sources

**Verdict: MODIFY.**

The current F0 data source table is not execution-safe.

Specific issues:

- V13/V14 JSON files do **not** contain per-slice arrays. They contain summary statistics only.
- V18 path names are wrong in the task doc. The committed V18 files are under `fullval_eval_20260518/artifacts/` and use `v18_best...` / `v18_last...` naming.
- V7 per-slice chain CSV is not unknown; it already exists in `review/0511/fullval_psnr_clip3_20260516_173941/artifacts/`.
- V13/V14 per-slice CSVs appear to exist only on `/data_2` according to logs, not in the repo. Codex must either copy them into `review/0521/F0_inputs/` with hashes or regenerate them with the canonical eval script.

Required fix:

Add an explicit **F0.0a input materialization step** before the paired-t script:

1. Use repo CSVs for V7 and V18 where present.
2. Check `/data_2/.../eval_0521_v13_v14_fullval_psnr_clip3/` for V13/V14 per-slice CSVs.
3. If present, copy them into `review/0521/F0_inputs/` and record source paths + SHA256.
4. If absent, rerun V13/V14 canonical evals to produce per-slice CSVs.
5. Fail closed if any comparison has `n != 7403` or missing `slice_idx` / `psnr_<timepoint>` columns.

Also downgrade "F0 is 0 GPU" to: **F0 is 0 training GPU if all per-slice CSVs are already available; otherwise canonical eval regeneration uses eval GPU/time.**

---

## 3. Q2 - F0 Substrate Consistency

**Verdict: MODIFY.**

The prose correctly warns that V18-cap is `direct_probe`, not `canonical_chain`, but the script template violates that warning. This line:

```python
if 'V18-cap' in name and tp != 'NORMAL':
    continue
```

still allows `V18-cap.last` at `NORMAL` to be compared against V7 chain PSNR. That is a substrate-mixing bug, not a feature.

Required fix:

- Remove `V18-cap.last` from the main chain `DATA` dict entirely.
- Add a `substrate` field to every dataset entry.
- Main F0 should assert `substrate == 'canonical_chain'` for all V7/V13/V14/V18 comparisons.
- If V18-cap is analyzed, create a separate F0b direct-probe analysis that uses `KL_DRIFT_PER_SLICE.csv` for both V7 direct and V18-cap direct, with columns `ckpt`, `timepoint`, `slice_idx`, `psnr_clip3`.

The script should refuse cross-substrate comparisons, not rely on reviewer discipline.

---

## 4. Q3 - A4 YAML Four-Field Diff

**Verdict: REJECT as written; MODIFY to correct path.**

The four-field diff idea is correct, but the field path is wrong. V7 yaml uses:

```yaml
training:
  image_aux:
    lambda_start: 0.04
    lambda_max: 0.04
```

The task doc says:

```python
d['transport']['image_aux']['lambda_start'] = 0.08
d['transport']['image_aux']['lambda_max'] = 0.08
```

That will fail with `KeyError` or, if Codex invents the path, silently modify a field the trainer does not read. `train_first_hop.py` reads `train_cfg.get("image_aux", {})`, so the only valid A4 functional diff is:

```python
d['training']['image_aux']['lambda_start'] = 0.08
d['training']['image_aux']['lambda_max'] = 0.08
```

Required fix:

- Replace all `transport.image_aux.lambda_*` references with `training.image_aux.lambda_*`.
- Replace `ALLOWED` with `{'output_dir', 'run_name', 'training.image_aux.lambda_start', 'training.image_aux.lambda_max'}`.
- Add self-checks that `loss.image_aux.l1_weight`, `ssim_weight`, `seam_weight`, `border_width`, `border_weight`, and `seam_patch_size` are identical to V7.
- Add self-check that `backbone.checkpoint_path` is identical to V7.

The current A4 yaml creation section is a push blocker.

---

## 5. Q4 - A4 From-Scratch vs Resume

**Verdict: APPROVE with clarification.**

From-scratch is the right design if the claim is a clean A4 vs V7 single-variable comparison: same seed, same max steps, same schedule, only image_aux lambda changed.

The task already launches without `--resume`, which is correct. But because A3 used warm-start, Codex may pattern-match and add `--resume` unless the task is louder.

Required fix:

- Add a bold line in §A4.2: **Do not pass `--resume`; A4 is from-scratch by design.**
- Add a self-check after launch/report: resolved config or checkpoint metadata must show `step` starts from 0 and no resume checkpoint was loaded.
- Add a NOT-DO entry: "Do not warm-start A4 from V7.best."

---

## 6. Q5 - A4 Stop Rule / Scope Creep

**Verdict: MODIFY.**

The stop rule is conceptually clear, but the anti-check is too narrow. It only searches `*A4*v2*.yaml`, so `A4_image_aux_lambda_12.yaml`, `A4_ramp_lambda_08.yaml`, or `A4_cosine_image_aux.yaml` would bypass it.

Required fix:

- Replace the anti-check with a whitelist: under `review/0521/`, the only allowed A4 yaml path is exactly `review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08.yaml`.
- Add explicit text: no lambda=0.12, no ramp/cosine variant, no A4b/A4c, no follow-up image_aux schedule experiment regardless of A4 outcome.
- In §3.5, add "do not relaunch" to all three branches, not only after the table.

---

## 7. Q6 - Commit Order and Partial Failure

**Verdict: MODIFY.**

The five-commit plan assumes everything succeeds. It needs failure-state commits because this task is being sent to a remote executor that may lose context mid-run.

Required fix:

Add §4.4 partial completion protocol:

- If F0 cannot locate or regenerate per-slice inputs, commit `review/0521/F0_STATUS_BLOCKED.md` with exact missing files, attempted commands, and next command to run.
- If F0 script exists but fails an assertion, commit the script plus `F0_STATUS_FAILED.md`; do not fabricate a report.
- If A4 dies before 5 minutes, commit the yaml + launch log and stop.
- If A4 dies mid-training, copy current `metrics.jsonl`, commit log + metrics snapshot, and write `A4_STATUS_PARTIAL.md` with last step and failure reason.
- Never use commit message `step=160000 done` unless the log contains `Training done.` and checkpoint step is 160000.
- In §0 add: "Any failure state must be committed as STATUS_*.md + current logs before stopping; do not silent-fail."

---

## 8. Q7 - Mechanical Self-Check Coverage

**Verdict: MODIFY.**

Current checks are insufficient, and some are mechanically broken.

High-priority additions:

- Eval script flag check: `--out-dir` exists and `--output-dir` is not used in commands.
- F0 input check: every chain input CSV exists, has `slice_idx`, has `psnr_D20/D10/D4/NORMAL`, and has exactly 7403 rows.
- F0 method check: script imports/uses `scipy.stats.ttest_rel` or implements the exact paired equivalent; no unpaired t-test.
- F0 substrate check: no `V18-cap` appears in the main chain summary JSON.
- A4 yaml check: `training.image_aux.lambda_start == 0.08` and `training.image_aux.lambda_max == 0.08`.
- A4 invariance check: seed, max_steps, rollout step_weights, backbone checkpoint path, `loss.image_aux.*`, `training.rollout.*`, and `lr_schedule.total_steps_override` match V7.
- A4 log check: launch log contains `lambda_img=0.0800` after enough training lines exist.

Mechanical bug in current anti-checks:

```bash
anti_check "no V18-clean yaml" find review/0521 -name '*v18*clean*.yaml' 2>/dev/null | grep -q .
```

This pipes the output of the function, not the `find`, and `find` exits 0 even when it finds nothing. It can print false failures and lose `ERR` updates due to the pipeline subshell.

Required fix:

Use a helper like:

```bash
check_absent_glob() {
  local label="$1" pattern="$2"
  if find review/0521 -name "$pattern" -print -quit | grep -q .; then
    echo "FAIL (should NOT exist): $label"; ERR=$((ERR+1))
  else
    echo "PASS (correctly absent): $label"
  fi
}
```

Or call `anti_check` with `bash -c '! find ... -print -quit | grep -q .'`.

---

## 9. Q8 - NOT-DO List

**Verdict: APPROVE with edits.**

The NOT-DO list is mostly right. I would keep V14b/V14c banned for Codex in this task. Even if a future reviewer wants one more seed, it is outside the current execution doc and would create slot contention / scope creep.

Edits needed:

- Add: do not warm-start A4 from V7.best.
- Add: do not create any A4 variant other than the exact lambda=0.08 yaml.
- Clarify the paper-claim item. Codex is allowed to write data reports (`F0_paired_t_report.md`, `A4_REPORT.md`), but not paper-facing claims or narrative rewrites before user review.
- Keep the V18-cap naming warning, but phrase it as a substrate warning: `V18-cap.last` is an A3 artifact and must not be merged into chain F0.
- Add: do not use summary JSON as per-slice input.

---

## 10. Must-Fix Table

| § | Original / issue | Required fix | Severity |
|---|---|---|---|
| §2.0 | V13/V14 summary JSON listed as per-slice inputs | Add F0.0a materialization: copy `/data_2` per-slice CSVs or regenerate; fail if unavailable | HIGH |
| §2.0 | V18 paths omit `artifacts/` and use nonexistent names | Use `review/0517/V18_decoder_lora/fullval_eval_20260518/artifacts/v18_best...` and `v18_last...` CSV/JSON paths | HIGH |
| §2.1 / §3.4 | Commands use `--output-dir` | Replace with `--out-dir` everywhere | HIGH |
| §2.2 | `V18-cap.last` in chain `DATA`; NORMAL still computed | Remove from main F0; optional F0b direct-probe only with direct V7 reference | HIGH |
| §3.1 | `transport.image_aux.lambda_*` | Replace with `training.image_aux.lambda_*` in prose, Python edit, diff verifier, self-check | HIGH |
| §5 | `anti_check ... find ... | grep` pipeline | Replace with a correct absence helper / whitelist | HIGH |
| §4.2 | Commit 4 assumes complete A4 | Add partial failure commits and status docs | MED |
| §5 | Missing invariance checks | Add checks for `loss.image_aux.*`, backbone path, rollout weights, n=7403, paired-test method | MED |
| §6 | Stop rule only pattern-checks v2/v3 | Whitelist exact A4 yaml; ban lambda=0.12/ramp/cosine variants | MED |
| §0 / §2 | "F0 is 0 GPU" unconditional | Qualify: 0 GPU only if per-slice CSVs already exist; regeneration uses eval GPU/time | MED |
| §3.2 | From-scratch implied by no `--resume` | Make explicit and add NOT-DO for warm-start | LOW |

---

## 11. New Bias / Failure Modes (B75+)

- **B75 - Summary-as-per-slice fabrication.** Treating aggregate JSON as if it contains per-slice arrays lets Codex produce fake paired-t or silently fail late.
- **B76 - CLI flag drift.** The task correctly checks train CLI but uses the wrong eval CLI flag; eval CLI needs the same rigor.
- **B77 - Config-path hallucination.** `transport.image_aux` is a plausible but nonexistent path; the trainer reads `training.image_aux`.
- **B78 - Substrate leakage in code template.** Prose says V18-cap is direct-probe only, but code template leaves a path for NORMAL chain comparison.
- **B79 - Anti-check false confidence.** Broken bash pipelines can print reassuring self-check output while not actually updating failure state.
- **B80 - 0-GPU optimism.** F0 is free only if all per-slice artifacts are accessible; otherwise it becomes canonical eval work.
- **B81 - Atomic-success assumption.** Commit sequencing assumes full completion and lacks recovery artifacts for remote mid-run failure.

---

## 12. Final Review D Verdict

Do **not** push `CODEX_TASK_ROUND17_F0_A4_20260522.md` as-is.

After the HIGH fixes above, the task can become push-ready without strategic re-review. The intended plan is good; the current execution doc is brittle. The minimum safe patch is:

1. Correct F0 input paths and require real per-slice CSV materialization.
2. Correct eval CLI to `--out-dir`.
3. Correct A4 yaml path to `training.image_aux.*`.
4. Remove V18-cap from chain paired-t.
5. Fix the self-check anti-check functions.
6. Add partial failure status protocol.

With those changes, my expected decision would become **READY**.