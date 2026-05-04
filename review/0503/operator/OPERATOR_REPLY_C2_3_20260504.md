# Operator Reply to C2.3 Confirmation Request - 2026-05-04

**Repo**: `/home/qujiaxiang/project/PET_LatentResidual`
**Branch**: `foc_lite_hop0`
**Remote**: `origin = git@gitee.com:jqu9/PET_LatentResidual.git`
**HEAD verified**: `32f2fae04f6a5ecb9a81a89f11d8e8de8dbb235a`
**Requested source**: `review/0503/local/OPERATOR_CONFIRM_REQUEST_C2_3.md`
**Sample time**: 2026-05-04 CST

This file answers the six operator questions in the C2.3 confirmation request and adds an operator-side code review of the current lock / Method-D tooling.

---

## 0. Pull / HEAD State

Gitee branch was already up to date on this host:

```text
From gitee.com:jqu9/PET_LatentResidual
 * branch            foc_lite_hop0 -> FETCH_HEAD
Already up to date.
```

Current HEAD is newer than the request's `de4cc88` because the request file itself was committed later:

```text
32f2fae docs(review/0503): operator confirmation request for C2.2 + C2.3 push
de4cc88 C2.3: Round-5 peer-review absorption — D1 (sanitize git stderr) + E1 (strip URL userinfo)
c01aa27 scripts(review/0502): C2.2 — absorb mid-impl peer review MEDIUM polish (A2/A3/B2a/B3b/B4/B10)
```

Local worktree note: this host still has unrelated historical modified/untracked experiment logs/configs. I did not touch them for this reply.

---

## 1. Test Verification

### Default `python3`

Command:

```bash
python3 -m unittest \
  review.0502.scripts.test_remote_resolver \
  review.0502.scripts.test_metrics_compat
```

Result: **failed 68/69 pass + 1 error**.

Failure reason is environment-only: default `python3` does not have PyYAML, and `test_T16_main_reads_metrics_file_exactly_once` exercises `lock.main()`, which imports `yaml` through `load_yaml()`.

Key stderr:

```text
ModuleNotFoundError: No module named 'yaml'
...
SystemExit: 5
Ran 69 tests in 0.058s
FAILED (errors=1)
```

### RAE environment

Command:

```bash
/home/qujiaxiang/.conda/envs/rae/bin/python -m unittest \
  review.0502.scripts.test_remote_resolver \
  review.0502.scripts.test_metrics_compat
```

Result: **69/69 OK**.

```text
.....................................................................
----------------------------------------------------------------------
Ran 69 tests in 0.059s

OK
```

Operator conclusion: the code passes under the project runtime environment. The confirmation request should not say plain `python3` is sufficient on this host unless PyYAML is installed there; use `/home/qujiaxiang/.conda/envs/rae/bin/python` for reproducible validation.

---

## 2. Q1 - Does Training Pipeline Depend on `review/0502/scripts/`?

Commands and outputs:

```bash
grep -r "from review" pet_lr/ scripts/ tools/ train_*.py 2>/dev/null
# no matches

grep -r "import review" pet_lr/ scripts/ tools/ train_*.py 2>/dev/null
# no matches

grep -r "lock_effect_size_threshold" pet_lr/ scripts/ tools/ train_*.py 2>/dev/null
# no matches

grep -r "_metrics_compat" pet_lr/ scripts/ tools/ train_*.py 2>/dev/null
# no matches
```

Answer: **training does not import `review/0502/scripts/`**. The C2.2/C2.3 lock-tool changes do not affect `pet_lr/`, `train_first_hop.py`, or normal training launchers by Python import dependency.

---

## 3. Q2 - `.review_canonical_remote` Anchor Content

Command:

```bash
cat -A .review_canonical_remote
```

Output:

```text
gitee.com:jqu9/PET_LatentResidual$
```

Answer: single clean line, no BOM, no CRLF, no query/fragment, no credentials. This is valid for this host.

---

## 4. Q3 - Remote URLs on Operator Host

Command:

```bash
git remote -v
```

Output:

```text
origin	git@gitee.com:jqu9/PET_LatentResidual.git (fetch)
origin	git@gitee.com:jqu9/PET_LatentResidual.git (push)
```

Additional check:

```bash
git remote get-url origin
# git@gitee.com:jqu9/PET_LatentResidual.git

git remote get-url --push origin
# git@gitee.com:jqu9/PET_LatentResidual.git
```

Answer: canonical remote name on this host is **`origin`**, not `gitee`. Fetch and push URLs are both canonical Gitee SSH URLs.

---

## 5. Q4 - Gitee SSH Authentication

Command:

```bash
ssh -T git@gitee.com 2>&1
```

Output includes ANSI color codes from Gitee, but semantically says authentication succeeds:

```text
Hi jqu9(@jqu9)! You've successfully authenticated, but GITEE.COM does not provide shell access.
```

Answer: SSH authentication is available. A normal `git push origin foc_lite_hop0` should work from this host, subject to network stability and branch permissions.

---

## 6. Q5 - Where Should Lock Run?

Answer: choose **(a)**.

Training and lock should run on this remote GPU workstation. Reason:

- The metrics files are generated on this machine under `/data_2/qujiaxiang/outputs/PET_LatentResidual/...`.
- Running lock here avoids copying metrics to another machine and avoids a second SHA / transfer-risk surface.
- This host has direct Gitee SSH authentication and canonical remote `origin`.

Operational detail: use the project runtime Python:

```bash
/home/qujiaxiang/.conda/envs/rae/bin/python review/0502/scripts/lock_effect_size_threshold.py ...
```

Do not use default `python3` unless PyYAML is installed there.

---

## 7. Q6 - Current Training State and Distance to Lock

Current PET-related `train_first_hop.py` jobs sampled on 2026-05-04:

| Run | GPU | PID | Current state | Metrics path | Lock relevance |
|---|---:|---:|---|---|---|
| `V6.1 rollout_floor` | 3 | `842073` | latest metrics max step `153600`, latest val step `153600` | `/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v6_1_rollout_floor/metrics.jsonl` | not the sigma-normalize A-main lock run |
| `B_sanity_light` | 2 | `2337360` | latest metrics max step `13150`, latest val step `12800` | `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0502_runs_light/20260502_051245/B_sanity_light/run/first_hop_224_sigma_norm_B_sanity_light_20260502_051245/metrics.jsonl` | light sanity only; not lock data |

Existing completed sanity references:

| Run | State | Val rows | In `[40000,60000]` lock window |
|---|---|---:|---:|
| `A_sanity` | `20000/20000` complete | 50 | 0 |
| `B` full sanity | `20000/20000` complete | 50 | 0 |

Answer: the formal lock stage is **not close yet** because the required formal `A_main` / paired seed data for the preregistered lock has not completed on this host. Current V6.1 and sanity-light metrics should not be used as the effect-size lock input.

---

## 8. Optional C2.4-Tests Timing

Operator recommendation: **Option A**.

C2.4-tests can proceed while training continues because the target files are in `review/0502/scripts/` and are not imported by training. I recommend including at least:

- pushURL verification test, not only fetch URL verification;
- end-to-end checkpoint naming test for `step_*.pt`, `best.pt`, `last.pt`;
- Guard-order regression test for data-side metric-key failure vs N<5;
- subprocess failure mocks for commit/push/error branches.

---

## 9. Additional Operator Code Review Findings

### Finding 1 - Method-D checkpoint selector is still incompatible with real checkpoints

Severity: **HIGH / operational blocker for Method-D selection**.

Real trainer save paths in `train_first_hop.py`:

```text
train_first_hop.py:2420  f"step_{step:06d}.pt"
train_first_hop.py:2521  "best.pt"
train_first_hop.py:2548  "last.pt"
```

Current selector still searches only legacy names in the saved-grid path:

```text
review/0502/scripts/select_best_ckpt_smoothed.py:55   ckpt_step_*.pt / best.pt / ckpt_last.pt
review/0502/scripts/select_best_ckpt_smoothed.py:107  ckpt_dir.glob("ckpt_step_*.pt")
review/0502/scripts/select_best_ckpt_smoothed.py:127  ckpt.get("global_step", None)
review/0502/scripts/select_best_ckpt_smoothed.py:136  _try_load_step("ckpt_last.pt")
review/0502/scripts/select_best_ckpt_smoothed.py:262  ckpt_name = f"ckpt_step_{best['step']}.pt"
```

Empirical probe on real `A_sanity` output:

```bash
A=/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0502_runs/A_sanity/run/first_hop_224_sigma_norm_A_sanity
/home/qujiaxiang/.conda/envs/rae/bin/python review/0502/scripts/select_best_ckpt_smoothed.py \
  --metrics "$A/metrics.jsonl" \
  --ckpt-dir "$A" \
  --neighborhood 2 \
  --include-best-pt \
  --include-last-pt
```

Observed failure:

```text
[info] loaded 450 rows (50 have val metrics)
[info] metric: val_select_score
[info] neighborhood K=2 → averaging 5 eval rows per candidate
No saved ckpts found in ... matching ckpt_step_*.pt; --include-best-pt / --include-last-pt also yielded none
```

But the directory actually contains:

```text
step_020000.pt
best.pt
last.pt
metrics.jsonl
```

Likely root cause: C3 is documented in `REV1_TOOLING_PLAN.md`, but current HEAD has not implemented it. The checkpoint internals also use key `step`, while this script loads only `global_step`.

Recommendation:

- Glob both `step_*.pt` and `ckpt_step_*.pt`.
- Parse zero-padded `step_020000.pt` correctly.
- Probe both `last.pt` and `ckpt_last.pt`.
- Load checkpoint step via `ckpt.get("step", ckpt.get("global_step"))`.
- When printing the final recommended path, prefer an actually existing file rather than constructing `ckpt_step_<step>.pt` blindly.
- Add an end-to-end test with a temporary directory containing `step_020000.pt`, `best.pt`, and `last.pt`.

Until this lands, Method-D selection is not operational on current trainer outputs.

### Finding 2 - Lock warning counts normal train rows as missing `val_select_score`

Severity: **MEDIUM / confusing operator signal**.

`compute_paired_stats_from_rows()` increments `skipped_no_metric` for every in-window row without `val_select_score`. In real metrics files, training rows in the window normally do not have `val_select_score`, so the warning can look like a partial-write/schema problem even when the file is healthy.

Observed during a repo-local smoke lock preview on `A_sanity` tail window:

```text
[warn] compute_paired_cv: skipped 33 in-window row(s) missing `val_select_score` ...
```

But those skipped rows are normal `event=train` rows. Recommendation: only count missing `val_select_score` as suspicious when `event == "val"` or when the row otherwise looks val-like. Do not warn on normal train rows.

### Finding 3 - Canonical remote check verifies fetch URL, not push URL

Severity: **MEDIUM / preregistration push target risk**.

Current host is safe because:

```text
git remote get-url origin        -> git@gitee.com:jqu9/PET_LatentResidual.git
git remote get-url --push origin -> git@gitee.com:jqu9/PET_LatentResidual.git
```

But `lock_effect_size_threshold.py` uses `git remote -v` / `git remote get-url <remote>` for resolver and pre-push TOCTOU. If another machine has `remote.origin.pushURL` configured differently, `git push origin branch` may push to the pushURL rather than the fetch URL that was verified.

Recommendation: immediately before push, verify both:

```bash
git remote get-url <remote>
git remote get-url --push <remote>
```

Both normalized URLs should match `.review_canonical_remote` unless `--force-unsafe-remote` is explicitly set.

### Finding 4 - `python3` command in confirmation request is not portable on this host

Severity: **LOW / documentation-operational**.

The test suite itself is correct under the project runtime, but the requested command uses default `python3`, which lacks PyYAML here. Recommendation: document the expected runtime as:

```bash
/home/qujiaxiang/.conda/envs/rae/bin/python -m unittest \
  review.0502.scripts.test_remote_resolver \
  review.0502.scripts.test_metrics_compat
```

Alternatively, make the specific test skip gracefully when PyYAML is absent, but that would hide a real runtime dependency of `lock_effect_size_threshold.py`.

### Finding 5 - Guard 5 config-side is still permissive

Severity: **LOW-to-MEDIUM / preregistration strictness**.

Current code accepts:

```python
cfg_metric_key in (None, "val_multi_objective", "val_select_score")
```

For formal R1b lock, this should probably be stricter. The intended production config-side objective is `val_multi_objective`, while data-side selection key is `val_select_score`. Allowing `None` or config-side `val_select_score` is convenient for tests/dev, but weakens the preregistered contract.

Recommendation: before formal lock, either:

- require config-side `training.best_metric == "val_multi_objective"`, or
- require an explicit `--allow-dev-best-metric` / `--allow-legacy-best-metric` flag for non-production fixtures.

---

## 10. Operator Recommendation

Do not treat C2.3 as sufficient for formal lock / Method-D launch yet. Environment and C2.3-specific tests pass under the project runtime, but the current stack still needs at least the C3 checkpoint selector fix before Method-D is usable on real trainer outputs.

Recommended next order:

1. Land C3 for `select_best_ckpt_smoothed.py` real checkpoint naming compatibility.
2. Add pushURL verification before preregistration push.
3. Narrow `compute_paired_stats_from_rows()` warnings to val-like rows.
4. Decide whether Guard 5 should become strict before R1b lock.
5. Keep training jobs running; these script changes do not affect `train_first_hop.py` imports.
