# REV1 — Tooling Remediation Plan

**Date**: 2026-05-03 (drafted after operator reply `e95f86a`)
**Scope**: Implementation-layer fixes triggered by operator feedback in
[`OPERATOR_REPLY_local_questions_20260503.md`](../operator/OPERATOR_REPLY_local_questions_20260503.md).
**Companion to**: [`REV1_PLAN.md`](REV1_PLAN.md) (statistical / protocol layer).
**Status**: Plan only. **No code or protocol files have been edited yet.** User approval required before any commit.

---

## 0. Why a separate plan from `REV1_PLAN.md`

[`REV1_PLAN.md`](REV1_PLAN.md) covers statistical findings from external peer reviewers (GPT-5.5 + Claude). Those are **methodology** changes that need user policy decisions (menu A/B/C/D in §8).

This plan covers **mechanical implementation bugs** the operator found by inspecting real artefacts:

- The schema my scripts assume (`global_step`) does not match the schema the trainer writes (`step`).
- The default git remote name is wrong on whichever machine runs the lock script.
- The checkpoint filename pattern my scripts search for is wrong.

These are fail-silent or fail-mismatched bugs that would invalidate the lock-in **regardless** of which menu option is picked in `REV1_PLAN.md`. They should land first, independently, with no policy implications.

This plan is therefore a **prerequisite** to `REV1_PLAN`'s execution, not an alternative to it.

---

## 1. Sequencing context (changed since REV1_PLAN was written)

Per operator reply Q1 + Q2:

| Run | Status |
|---|---|
| `A_main` (= `A_control`) | **not started** — no metrics path on disk |
| `C_uniform` | not started |
| `D_closed_form` | not started |
| `A_pair_uniform_spot` | not started (config exists, my 02:21 add) |
| `A_sanity` | finished 20K (this is what `tail5` schema sample comes from) |
| `B` (sanity second half) | running, ~10.4K / 20K |

**Hard deadline removed.** REV1_PLAN's "before A_main reaches step 60000" constraint is not biting yet. Operator's recommended order:

```
patch REV1 → push → finish sanity → launch A_main → A_main crosses 60K → lock X → C_uniform under wrapper
```

This means we have runway to land tooling fixes (this plan) **and** statistical fixes (REV1_PLAN) cleanly, with verification, before any irreversible launch.

---

## 2. Inventory of changes

### Tier 0 — fail-silent tooling bugs (must land first, no policy involved)

| ID | Bug | File(s) | Symptom on real run | Source |
|---|---|---|---|---|
| **T1** | metrics step key is `step`, scripts read `global_step` | [`lock_effect_size_threshold.py:240,242,244`](../../0502/scripts/lock_effect_size_threshold.py), [`paired_diff_judge.py:138,141,146`](../../0502/scripts/paired_diff_judge.py), [`select_best_ckpt_smoothed.py:122,149,159`](../../0502/scripts/select_best_ckpt_smoothed.py) | All three scripts return `N=0` / no shared steps / empty smoothing window. **Lock script then fails Guard 4 → exit 1**. Fail-loud, but reason is misleading ("insufficient observations" when actually schema mismatch). | Operator reply Q4 + `A_sanity_metrics_tail5_schema_reference.jsonl` |
| **T2** | `lock_effect_size_threshold.py` defaults `--remote=gitee`; on operator's host the canonical gitee URL is registered as `origin`; on my workstation `origin` is the github mirror | [`lock_effect_size_threshold.py:130-131,560`](../../0502/scripts/lock_effect_size_threshold.py) | • On operator host: `git push gitee` fails ("'gitee' is not a git repository"). Commit succeeds → silent half-state.<br>• If we naively change default to `origin`: on my workstation pushes to **github mirror**, not the canonical gitee repo → pre-registration timestamp lands in wrong repo. | Operator reply B1 + my `git remote -v` check |
| **T3** | Ckpt search globs `ckpt_step_*.pt` / `ckpt_last.pt`; trainer writes `step_NNNNNN.pt` / `last.pt` / `best.pt` | [`select_best_ckpt_smoothed.py:50,102,131`](../../0502/scripts/select_best_ckpt_smoothed.py) | Method-D candidate list is empty → script returns no selection. Will be caught at runtime, not silent, but blocks Method-D selection. | Operator reply B2 |

### Tier 1 — schedule-window correction (newly discovered, statistical implication)

| ID | Issue | File | Source |
|---|---|---|---|
| **W1** | LOCKED window `[40000, 60000]` (per `POST_V6_NEXT_STEPS.md §6.6.1` + `lock_effect_size_threshold.py LOCKED_RECOMMENDED_WINDOW`) sits in the **middle of the ramp phase** of A_main's 120K compressed schedule. Per `A_control.yaml` (max_steps=120000, warmup_ratio=0.25, ramp_ratio=0.50) the actual phase boundaries are `warmup [0,30K]` / `ramp [30K,90K]` / `Phase III stable [90K,120K]` — i.e. ramp **lasts 60K steps starting at step 30K**, ending at step 90K, not step 60K. Window [40K,60K] sits at ramp progress 16.7%–50% (α and λ_roll still rising). Any paired_CV / paired_SD computed there is contaminated by deterministic ramp dynamics rather than reflecting steady-state sampling noise. | `POST_V6_NEXT_STEPS.md §2.1` Phase table + §6.6.1 window, [`lock_effect_size_threshold.py:~104`](../../0502/scripts/lock_effect_size_threshold.py) `LOCKED_RECOMMENDED_WINDOW` constant | Local discovery during operator-reply re-read; agent1 review caught a phase-math error in v0.1 of this plan (had written ramp `[30K,60K]` — corrected here to ramp `[30K,90K]`). |

This is technically a **protocol** change, not a pure tooling change — but because A_main has not started, we can fix it **before any window gets locked**.

**Proposed new window**: `[90000, 120000]` (full Phase III stable region, 30K steps).

This is a 50% widening over the original 20K window (`[40K,60K]`), which gives a **~22% reduction in SE_diff** at fixed eval_interval (50 → 75 eval points if interval=400). Free statistical-power gain on top of the noise-contamination fix.

Open question: should the window start at step 90000 (boundary of Phase III) or step 95000 (5K into Phase III, allowing α=1 and λ_roll=4 EMA to fully settle)? See §6 Open question 5.

This shift also affects `REV1_PLAN.md §4` "hard deadline" wording — the deadline becomes "before A_main reaches step **90000** (or 95000)", not 60000. (And since A_main hasn't started, the practical deadline is "before A_main is launched.")

### Tier 2 — additive features per operator request (no behavior changes to existing code)

| ID | Feature | Source |
|---|---|---|
| **F1** | New `review/0502/scripts/run_c_uniform_full_val.sh` wrapper enforcing 5 refusal conditions before allowing any C_uniform full-val to run: (1) `EFFECT_SIZE_LOCKED.md` exists; (2) lock commit reachable from canonical gitee remote; (3) working tree clean; (4) eval invoked through single canonical entrypoint; (5) immutable manifest written (lock SHA256, eval cmd, config path, ckpt path, git HEAD, timestamp, GPU env). | Operator reply Q3 |
| **F2** | Add environment capture block to `EFFECT_SIZE_LOCKED.md` template — minimum 9 fields per operator Q7: python executable + version, torch version, cuda version, `CUDA_VISIBLE_DEVICES`, nvidia-smi driver_version, GPU model/memory, repo HEAD, branch, metrics SHA256 + byte count. Implemented as a helper in `lock_effect_size_threshold.py` that runs `subprocess.run(["nvidia-smi", ...])` etc. and embeds the result. | Operator reply Q7 |

### Tier 3 — deferred (operator explicitly said "not in same commit as REV1")

| ID | Item | Note |
|---|---|---|
| **D1** | Move `train_v21.py` → `deprecated/v21/train_v21.py` and update README references | Operator reply Q6: "yes, but in a separate cleanup commit AFTER REV1 lands". Documenting here so we don't forget. |

---

## 3. Atomic commit plan

Six commits in order. Each is independently testable + revertable. **No squashing** — the chain itself is part of the audit trail (operator reply → tooling fix is a defensible narrative; one giant commit hides the timeline).

```
HEAD (e95f86a, operator reply)
  │
  ├── C1: fix(scripts): step/global_step schema compat                [T1]
  │       • Adds get_row_step(row) helper module-or-inline
  │       • Replaces all 3 scripts' direct r["global_step"] reads
  │       • Adds golden-trace unit test using operator's tail5 jsonl
  │
  ├── C2: fix(lock_effect_size): URL-based remote autodetection       [T2]
  │       • Drops default="gitee" magic
  │       • Adds resolve_canonical_remote() that scans `git remote -v`
  │         for "gitee.com:jqu9/PET_LatentResidual" URL fragment
  │       • --remote stays as explicit override, default=None
  │       • Aborts with actionable error if no remote points at canonical URL
  │       • Inline comment explaining why default="origin" is wrong
  │         (origin = github mirror on my Mac, origin = canonical on op host)
  │
  ├── C3: fix(select_best_ckpt): step_NNNNNN.pt + last.pt naming      [T3]
  │       • glob both `ckpt_step_*.pt` (legacy) and `step_*.pt` (current)
  │       • probe both `ckpt_last.pt` and `last.pt`
  │       • probe `best.pt` (already supported by --include-best-pt)
  │       • Help text updated
  │
  ├── C4: feat(lock_effect_size): environment capture in lock doc     [F2]
  │       • capture_environment() helper: 9 fields per operator Q7
  │       • Embedded into EFFECT_SIZE_LOCKED.md "Environment" section
  │       • Failure to capture nvidia-smi → write "[unavailable: <reason>]"
  │         (don't abort lock; nvidia-smi may not be on lock-time host)
  │
  ├── C5: feat(scripts): post-lock C_uniform full-val wrapper         [F1]
  │       • New file: review/0502/scripts/run_c_uniform_full_val.sh
  │       • Plus tiny Python helper review/0502/scripts/_c_wrapper_guards.py
  │         that does the 5 checks (sh stays thin)
  │       • Refusal conditions exactly per operator Q3
  │       • Manifest written to review/0502/c_uniform_run_manifest_<SHA8>.md
  │
  ├── C6: protocol(0502): window [40000,60000] → [90000,120000]       [W1]
  │       • POST_V6_NEXT_STEPS.md §6.6.1: window numbers + rationale
  │       • lock_effect_size_threshold.py: LOCKED_RECOMMENDED_WINDOW
  │       • Adds §6.6.1.1 "Window choice rationale" subsection citing
  │         §2.1 Phase table (warmup/ramp/Phase III)
  │       • REV1_PLAN.md §4: deadline 60000 → 90000 (or "before A_main
  │         launched" since neither is binding yet)
  │
  └── (later, separate, after REV1_PLAN lands)
      D1: chore(deprecated): move train_v21.py                        [D1]
```

**Total churn estimate**: C1 ~120 LoC, C2 ~50 LoC, C3 ~30 LoC, C4 ~80 LoC, C5 ~150 LoC (mostly new wrapper), C6 ~40 LoC (mostly doc). Combined ~470 LoC across 6 commits.

---

## 4. Verification per commit

Each commit MUST pass its own test before the next is started.

| Commit | Verification | Expected outcome |
|---|---|---|
| **C1** | `python review/0502/scripts/lock_effect_size_threshold.py --metrics-a review/0503/operator/A_sanity_metrics_tail5_schema_reference.jsonl --step-min 18400 --step-max 20000 --no-commit --output /tmp/test_lock.md ...` | Exit code 0 (would have been 1 = Guard 4 fail before fix). Lock doc shows `N = 5`, paired_CV computed from real values. |
| **C1** | Same invocation but with synthetic file having `global_step` key | Exit 0, N=5. **Backward compat preserved.** |
| **C2** | On my Mac: invoke without `--remote`, verify it resolves to `gitee` (not `origin`) | Console prints `[info] resolved canonical remote: gitee → git@gitee.com:jqu9/...` |
| **C2** | Mock test: `git remote -v` returns only github mirror | Exit non-zero with actionable message including `git remote add gitee ...` |
| **C3** | Create test directory with `step_020000.pt` + `step_040000.pt` + `last.pt`, run `select_best_ckpt_smoothed.py` | Both step-files detected, `last.pt` step extracted. |
| **C3** | Test with old-format `ckpt_step_*.pt` + `ckpt_last.pt` | Backward compat preserved. |
| **C4** | Run lock script on test fixture | `EFFECT_SIZE_LOCKED.md` contains all 9 env fields (or `[unavailable]` markers). |
| **C5** | Dry-run wrapper before any lock exists | Exits with refusal: "EFFECT_SIZE_LOCKED.md does not exist". |
| **C5** | Touch a fake `EFFECT_SIZE_LOCKED.md` whose commit is NOT pushed to canonical gitee | Refuses with "lock commit not in canonical remote". |
| **C5** | Working tree dirty | Refuses with "uncommitted changes detected". |
| **C5** | All 3 above conditions satisfied | Wrapper proceeds, writes manifest. |
| **C6** | Lock script default `--step-min` and `--step-max` reflect new window | `python lock_effect_size_threshold.py --help` prints `[90000, 120000]` recommended. |
| **C6** | `POST_V6_NEXT_STEPS.md §6.6.1` cites Phase III explicitly | grep returns `Phase III` in §6.6.1. |

---

## 5. What this plan does NOT do

- **Does NOT touch the statistical formula** (`paired_CV` vs `paired_SD`, floor/slope, decision trichotomy). Those are REV1_PLAN's concern. They will land in a separate REV1 commit after this plan's 6 commits, when user has decided menu A/B/C/D in REV1_PLAN §8.
- **Does NOT lock anything.** No `EFFECT_SIZE_LOCKED.md` is created. Lock-in only happens after A_main crosses the new window's lower bound (90000).
- **Does NOT decide whether `sigma_seed{123,456}` are usable as A_main two seeds.** Operator confirmed they are option (c) abandoned (Q8); REV1_PLAN.md §8 will absorb the corrected cost (a fresh A_control seed=43 run, ~1.5–2 GPU·days for two-seed paired SD).
- **Does NOT run new experiments.** Decoder ceiling already exists per operator Q5; A_pair_uniform_spot launch decision is REV1_PLAN's call.
- **Does NOT move `train_v21.py`.** Operator explicitly requested this be a separate cleanup commit after REV1.
- **Does NOT GPG-sign or OTS-stamp commits.** Tamper-resistance (REV1_PLAN C9) is REV1_PLAN's scope.

---

## 6. Open questions for user before execution

1. **Approve commit C1–C6 sequence as written?** Or split / merge differently?
2. **C2 (URL autodetect)**: I plan to hard-code the canonical URL fragment `gitee.com:jqu9/PET_LatentResidual` as a module constant. Alternative: read it from a `.review_canonical_remote` file at repo root that operator + I both maintain. The constant is simpler; the file is more flexible if we ever rename the gitee repo. Pick one.
3. **C4 environment capture**: do you want it to **abort the lock** if any of the 9 fields cannot be captured (strict — risk of getting stuck if `nvidia-smi` is missing during a CPU-only ad-hoc lock-in), or **write `[unavailable]` markers** (lenient — but a malicious operator could fake fields)? Operator's reply implies lenient. I lean lenient + log a WARNING line.
4. **C5 (C-wrapper)**: should the wrapper be Python-only (matches the Q3-recommended single canonical entrypoint), or Bash-driven calling Python helpers (closer to existing `run_ablation.sh` style)? Operator did not specify.
5. **C6 window shift**: are you OK shifting the LOCKED window from `[40000, 60000]` to `[90000, 120000]` based purely on the local schedule analysis (no third-party reviewer has confirmed this)? Or wait until I send the schedule context back to GPT-5.5 + Claude for independent confirmation? **I lean: ship the shift now (A_main hasn't started, no protocol harm), then send the post-shift protocol to reviewers as confirmation.** Reverse order would block on reviewer turnaround.
6. **Audit trail**: do you want a `REV1_TOOLING_CHANGELOG.md` written as the 7th commit in this series, summarizing what changed and citing operator reply by file path + commit hash? Useful for paper Methods narrative. Costs ~30 min.

---

## 7. Honest self-criticism

Three things this plan does **not** clean up that I should flag:

- **(a)** I authored the original `global_step` assumption in [`lock_effect_size_threshold.py`](../../0502/scripts/lock_effect_size_threshold.py) without reading the trainer's actual metrics writes. I should have asked operator for a tail-of-metrics sample before locking the schema. T1 is fail-silent (returns N=0 quietly); operator only caught it because they ran `wc -l` on a fake invocation. This is the same class of error as REV1_PLAN's C1 — building tooling on assumed behavior without verifying real artefacts.
- **(b)** The `default="gitee"` choice came from my development environment alone. I never asked which remote name operator's machine uses. T2 reveals a missing communication step — for any future tooling that touches git, I should ask operator's `git remote -v` first.
- **(c)** I did not catch W1 (window in ramp) when authoring §6.6.1. The Phase table is in §2.1 of the same document; I should have cross-referenced. I caught it on re-read after operator's reply, but neither external reviewer flagged it because they didn't have the schedule context. This is exactly the kind of error that benefits from "send context to reviewers, ask them to look again" (Open question #5 above).

These are documented here, not glossed over, because the credibility argument for the eventual lock relies on us having a clean audit trail of "found bug → documented → fixed → verified" rather than silent corrections.

---

## 8. Recommended user response

Pick one:

- **(A) Approve C1–C6 as written.** I execute serially, run verification after each, push after C6 succeeds. ETA: ~2–3 hours of agent work. Recommended.
- **(B) Approve a subset.** Tell me which commits to skip / defer. Common partial: skip C5 (wrapper) until C_uniform launch is closer.
- **(C) Approve, but ship C6 (window shift) only after independent reviewer confirmation.** Costs ~30 min reviewer turnaround on top of (A).
- **(D) Reject; reorder before REV1_PLAN.** Not recommended — these are unblockers regardless of REV1_PLAN's policy choices.

I will NOT autonomously execute (A) without explicit approval, because C2 (remote autodetect) and C6 (window shift) both have side effects on cross-machine behavior that you should sign off on.

---

## 9. Multi-agent review response (drafted 2026-05-03 evening)

Four agents (`agent1` / `agent2` / `agent3` / `agent4`) reviewed v0.1 of this plan against the actual codebase, `OPERATOR_REPLY_local_questions_20260503.md`, and `REV1_PLAN.md`. All four independently confirmed:

- T1/T2/T3 are real (verified against `train_first_hop.py` writing `step` not `global_step`, against `git remote -v`, and against the trainer writing `step_NNNNNN.pt` / `last.pt`)
- Plan layering (tooling vs statistical) is correct
- §7 self-criticism is appropriate

This section captures **convergent corrections** that should land before C1 starts. Items prefixed `Fx` are convergent (≥2 agents); items prefixed `Sx` are single-agent suggestions worth considering.

### 9.1 Convergent corrections (lock these into the plan)

| ID | Source | Correction |
|---|---|---|
| **F1** | agent1 + agent4 | **W1 phase math was wrong in v0.1.** v0.1 wrote `ramp [30K,60K] / Phase III [60K,120K]`. Actual: `ramp_ratio=0.50 × 120K = 60K steps starting at step 30K → ramp [30K,90K]`, Phase III `[90K,120K]` (30K wide, not 60K). **Fixed in §2 W1 row.** Window `[40K,60K]` is at ramp progress 16.7%–50%, not "fully in Phase II". The conclusion (window contaminated by ramp dynamics) survives; the math is now correct. |
| **F2** | agent1 + agent2 + agent3 | **C6 (window) and `REV1_PLAN.md` REV1.3 (formula) both edit `POST_V6_NEXT_STEPS.md §6.6.1`.** Splitting them creates two consecutive §6.6.1 revisions in git history → "piecemeal credibility hit" that REV1_PLAN.md §0 explicitly warns against. **Resolution**: promote C6 from "tooling" to "protocol", merge with REV1.3 into a single §6.6.1 patch. Renumbered as **C6 → R1** (protocol-tier, executes after REV1_PLAN approval, not as part of the C1–C5 tooling chain). |
| **F3** | agent2 + agent3 | **`REV1_PLAN.md §1 C12` Guard 5 assertion is wrong.** v0 of REV1_PLAN says tighten Guard 5 to require `best_metric == "val_select_score"` exactly. Verified `A_control.yaml:88` writes `best_metric: val_multi_objective` (config name). The metrics row key `val_select_score` is the *computed* score key the trainer writes. These are different. **Fix**: Guard 5 should assert `best_metric == "val_multi_objective"` AND `val_select_score in metrics_row.keys()` (both must hold). Annotated in `REV1_PLAN.md §9` (operator fact update). |
| **F4** | agent1 + agent2 + agent3 + agent4 | **`REV1_PLAN.md §3 + §8` references stale facts.** Operator Q5 confirmed decoder ceiling already exists (2026-04-22 full-val, NORMAL=52.6341 PSNR, n=7403). Operator Q8 confirmed `sigma_seed{123,456}` are abandoned (option c) — not just "schedule different" but never produced output dirs. **Fix**: `REV1_PLAN.md` §3.4 / §6 / §8 must absorb both facts. Annotated in `REV1_PLAN.md §9` (operator fact update). |
| **F5** | agent1 + agent3 + agent4 | **C5 wrapper's 5 refusal conditions only cover C-fullval-launch-time, not C-training-launch-time.** If operator can train C_uniform before lock and observe `metrics.jsonl` / WandB / stdout, they can selectively decide which ckpt to evaluate post-lock. **Fix**: add **C5.0** (forbid C_uniform training start before `EFFECT_SIZE_LOCKED.md` exists on canonical remote) and **C5.6** (require `A_main step >= max_steps` to prevent ckpt-overwrite during C eval). Updated below in §9.3. |
| **F8** | agent4 + agent1 | **REV1_PLAN.md REV1.3 has a hidden hard dependency on A_seed=43.** `paired_SD_AA = sd(A_seed1 - A_seed2)/mean(A)` requires two A runs at the same schedule. Since `sigma_seed{123,456}` are option (c) abandoned (F4), the only way to obtain `A_seed2` is to run `A_control` with `seed=43` (~1.5–2 GPU·days). **Fix**: REV1.3 cannot land a real formula until A_seed=43 finishes; either (a) land REV1.3 only after A_seed=43 completes, or (b) accept a placeholder with explicit "data pending" annotation (NOT recommended — leaves a credibility-fragile commit in git history). User must choose. |

### 9.2 Single-agent suggestions (proposed for adoption)

| ID | Source | Suggestion | Recommendation |
|---|---|---|---|
| **S1** | agent1 | C2 should read canonical URL from `.review_canonical_remote` file at repo root (git-tracked) instead of hardcoded module constant | **Adopt.** Future-proof against gitee account migration. ~3 LoC. |
| **S2** | agent1 | Add `lock_doc_schema.json` validator + `--validate` mode to lock script | **Defer to post-REV1.** Good idea but adds scope. |
| **S3** | agent1 | Add `training.blinded_metrics: [...]` yaml field; trainer writes but doesn't print blinded keys to stdout | **Defer to REV2.** Real "blinded ablation" infrastructure; out of REV1 scope. |
| **S4** | agent1 | F1 wrapper `out_dir = ${ROOT}/C/full_val_post_lock_<lock_sha8>/` (lock SHA8 in path) | **Adopt as part of C5 manifest.** ~1 LoC. Helps audit trail. |
| **S5** | agent4 | Window `[95000, 120000]` instead of `[90000, 120000]` (skip 5K of Phase III early to let α=1, λ_roll=4 EMA settle) | **Move to user open question 5** (already in §6). I lean [95K,120K]: 25K wide is still 25% wider than original 20K. Final call on user. |
| **S6** | agent4 | Highlight W1's free statistical-power gain: 50 → 75 eval points → ~22% SE reduction at fixed eval_interval=400 | **Adopted** in §2 W1 row revision. |
| **S7** | agent4 | URL match should be `.lower().endswith("/pet_latentresidual.git")` for fork-rename robustness | **Adopt.** ~2 LoC, defensive. |
| **S8** | agent2 | Shared `get_row_step(row)` helper in a small `_metrics_compat.py` module instead of inlined into each of 3 scripts | **Adopt.** Avoids 3 places drifting independently. New file: `review/0502/scripts/_metrics_compat.py` (~30 LoC). |
| **S9** | agent2 | Lock script should hash + parse from same bytes (not parse → hash separately) to avoid race | **Already in REV1_PLAN.md C7.** Cross-reference here. |
| **S10** | agent3 | T3 fix should also verify the printed final command in `select_best_ckpt_smoothed.py` tail uses the new naming convention | **Adopt.** Add to C3 verification matrix. |

### 9.3 Updated commit plan (post-review)

The C-prefix is reserved for **tooling** commits (no protocol effect). The R-prefix is reserved for **protocol** commits (touches `POST_V6_NEXT_STEPS.md §6.x` substantively).

```
HEAD (e95f86a, operator reply)
  │
  ├── C1: fix(scripts): step/global_step schema compat                [T1, F8 helper, S8]
  │       • New file: review/0502/scripts/_metrics_compat.py
  │       • get_row_step(row) helper used by all 3 callers
  │       • Update lock_effect_size_threshold.py / paired_diff_judge.py /
  │         select_best_ckpt_smoothed.py to import + call helper
  │       • Golden-trace test using operator's tail5 jsonl
  │
  ├── C2: fix(lock_effect_size): URL-based remote autodetection       [T2, S1, S7]
  │       • New file: .review_canonical_remote (git-tracked)
  │         contents: "gitee.com:jqu9/PET_LatentResidual"
  │       • resolve_canonical_remote() reads file, scans `git remote -v`,
  │         matches via .lower().endswith() (case-insensitive, fork-resilient)
  │       • --remote stays as explicit override
  │       • Aborts with actionable error if no remote matches
  │
  ├── C3: fix(select_best_ckpt): step_NNNNNN.pt + last.pt naming      [T3, S10]
  │       • glob both `ckpt_step_*.pt` (legacy) and `step_*.pt` (current)
  │       • probe `ckpt_last.pt`, `last.pt`, `best.pt`
  │       • update final printed command tail to use current naming
  │       • help text updated
  │
  ├── C4: feat(lock_effect_size): environment capture in lock doc     [F2 from operator Q7]
  │       • capture_environment() helper: 9 fields per operator Q7
  │       • Embedded into EFFECT_SIZE_LOCKED.md "Environment" section
  │       • Failure to capture nvidia-smi → write "[unavailable: <reason>]"
  │
  ├── C5: feat(scripts): post-lock C_uniform full-val wrapper         [F1 + F5 expansion + S4]
  │       • New file: review/0502/scripts/run_c_uniform_full_val.sh
  │       • Plus tiny Python helper review/0502/scripts/_c_wrapper_guards.py
  │       • Refusal conditions:
  │         C5.0 (NEW): EFFECT_SIZE_LOCKED.md must exist on canonical remote
  │                     BEFORE C_uniform training starts. Wrapper provides
  │                     a sub-command `precheck` that operator runs before
  │                     `bash run_ablation.sh C` to verify lock is in place.
  │         C5.1: EFFECT_SIZE_LOCKED.md exists at full-val time
  │         C5.2: lock commit reachable from canonical gitee remote
  │         C5.3: working tree clean
  │         C5.4: eval invoked through eval_first_hop_224_clip3.py
  │                     (single canonical entrypoint)
  │         C5.5: immutable manifest written to
  │                     review/0502/c_uniform_run_manifest_<lock_sha8>.md
  │                     (lock SHA8 in filename, S4)
  │         C5.6 (NEW): A_main has reached configured max_steps (120K),
  │                     not just "≥ window upper bound". Prevents ckpt
  │                     overwrite during C full-val.
  │
  └── R1 (post-REV1_PLAN approval): protocol(0502): §6.6.1 unified patch
      [merges old C6 + REV1.3 from REV1_PLAN.md per F2]
      • LOCKED window [40K,60K] → [90K,120K] or [95K,120K] (per user)
      • paired_CV_A → paired_SD_AA (requires A_seed=43 data per F8)
      • drop floor (REV1_PLAN.md C10)
      • decision trichotomy adds rel_diff ≤ -X branch (REV1_PLAN.md C2)
      • §6.6.1.1 "Window choice rationale" subsection citing §2.1
      • §6.6.8 changelog citing this REV1 with operator reply +
        4-agent review chain (audit trail)
      • REV1_PLAN.md §4 deadline: 60000 → 90000 (or 95000)
```

**Key change vs v0.1**: C6 is removed from this plan; the window shift becomes part of R1 in `REV1_PLAN.md` to avoid double-revision of §6.6.1. C1–C5 land first as pure tooling; R1 lands after user approves `REV1_PLAN.md §8` and (per F8) after A_seed=43 produces data.

### 9.4 Open question status update

Of the 6 questions in §6:

- **Q1** (commit sequence approval) — re-asked with new C1–C5 (no C6) chain
- **Q2** (URL hardcode vs file) — **resolved by S1 + S7**: file + case-insensitive .endswith
- **Q3** (env capture strict vs lenient) — still open, leaning lenient
- **Q4** (wrapper Python vs bash) — still open, leaning bash thin + Python helpers
- **Q5** (window 90K vs 95K) — **new sub-question per S5**, leaning 95K
- **Q6** (changelog as 7th commit) — still open, leaning yes (becomes part of R1, not a separate commit)

### 9.5 What still needs user input before C1 starts

Two things, neither blocking research integrity but both blocking execution:

1. **Approve C1–C5 (per §8 menu A) plus the §9 corrections.** No statistical or protocol implications until R1.
2. **Decide §6 Q5 (window 90K vs 95K)** so R1 has a concrete number to land. Can be decided later; doesn't block C1–C5.

Everything else (F8 sequencing, R1 vs REV1_PLAN sequencing, A_seed=43 launch) is downstream and can be answered after C1–C5 verify.

> **Note (2026-05-03 late evening)**: §9 was followed by a second round of multi-agent review. See **§10** for the round-2 corrections that supersede parts of §9.3 / §9.4 / §9.5.

---

## 10. Round-2 multi-agent review response (drafted 2026-05-03 late evening)

After §9 v0.2 was posted, the same four agents re-reviewed the v0.2 plans. They independently confirmed §9's corrections were sound, then surfaced **4 new convergent findings** (≥2 agents) and **7 high-value single-agent improvements**. This section captures those; the §9 substance is preserved as an audit trail of v0.2 → v0.3.

### 10.1 Convergent corrections (round 2, lock these in)

| ID | Source | Correction |
|---|---|---|
| **F9** | agent1 + agent4 | **A_pair_uniform_spot has a schedule confound.** `A_pair_uniform_spot.yaml` has `max_steps=60000`, with the same `warmup_ratio=0.25 / ramp_ratio=0.50` as A_main. Because schedule steps are computed as `ratio × max_steps`, the spot run's actual phase boundaries are warmup `[0,15K]` / ramp `[15K,45K]` / Phase III `[45K,60K]` — **not** A_main's `[0,30K] / [30K,90K] / [90K,120K]`. At step=45K, A_main is at ramp progress 25% but A_pair_spot is already in Phase III. **Same step number ≠ same training phase across the two configs.** For A_pair_spot to serve as Risk 4 robustness evidence, schedule must be aligned: either (a) raise `max_steps=120000` and rerun, or (b) set explicit `warmup_steps=30000 / ramp_steps=60000` overrides in spot yaml, or (c) scope-out Risk 4 in paper limitations. Verified: Mac-side python check showed exactly the boundaries above. |
| **F10** | agent1 + agent2 + agent4 | **R1 protocol must be locked BEFORE A_seed=43 launches, not after.** §9.3 has R1 land after A_seed=43 completes, which is the same anti-pattern that produced v0's flaw at a smaller scale (locking the formula after seeing the data the formula will score). Reviewer interpretation: "they peeked at A-arm noise then chose X formula." **Fix per agent2's option Z (resolves §9 F8 X-vs-Y dilemma)**: split R1 into **R1a** (data-independent skeleton: window, formula structure, metric choice, decision rule, A_seed=43 yaml spec, Method-D parameters, ceiling-as-anchor-only clause, §6.6.8 changelog) and **R1b** (numeric instantiation: paired_SD_AA value, X value, EFFECT_SIZE_LOCKED.md). R1a lands BEFORE A_seed=43 launches; R1b lands AFTER A_seed=43 completes by mechanically running the lock script. Standard pre-registration order. |
| **F11** | agent1 + agent3 + agent4 | **C lock-gate must be enforced inside `run_ablation.sh case "C")`, not as an external wrapper precheck.** §9.3 had operator manually run `precheck` before `bash run_ablation.sh C`. Verified: current `run_ablation.sh` has `require_sanity_pass` (sentinel-based hard gate) but no equivalent `require_lock_pass` for the C branch. Operator can forget the precheck → C training launches → trajectory leak. Fix: add `require_lock_pass()` mirroring the existing `require_sanity_pass()` pattern. ~10 lines of bash. New commit C5b in §10.3 below. |
| **F12** | agent2 + agent4 | **`paired_SD_AA` metric is underspecified.** Even after F8's two-seed correction, if paired_SD_AA is computed on `val_select_score` (Method-D multi-objective composite) while headline rel_diff is computed on `val_chain_normal_mse` (single-objective), v0's dimensional inconsistency returns at smaller scale. **Fix in R1a**: paired_SD_AA defined explicitly on `val_chain_normal_mse` paired-seed diff in window. `val_select_score` reserved for Method-D ckpt selection only, not for effect-size scaling. Both metric keys are present in real `metrics.jsonl` (verified against operator's `A_sanity_metrics_tail5_schema_reference.jsonl`). |

### 10.2 Single-agent suggestions (round 2)

| ID | Source | Suggestion | Decision |
|---|---|---|---|
| **S11** | agent3 | Lock-script Guard 6: refuse lock docs containing strings `ceiling` / `headroom` / `near-ceiling` / `52.6341`. Defends Claude Q5 attack (post-hoc ceiling-normalized rel_diff). | **Adopt.** ~5 lines, new commit C5d. |
| **S12** | agent3 | `LOCKED_PROTOCOL_VERSION` constant in `lock_effect_size_threshold.py`. C1 sets `"v0_pending_R1a"`; R1a → `"R1a_pending_data"`; R1b → `"R1b_locked"`. Operator seeing stale value → won't lock. Defends [A]→[D] human-error window. | **Adopt.** Embedded into C1, no new commit. |
| **S13** | agent3 | New script `verify_paired_seed_configs.py`: diffs A_seed=42 vs A_seed=43 yaml, asserts only `seed`/`run_name`/output-dir-related fields differ. SHA256 of diff written into lock doc. | **Adopt.** ~30 lines, new commit C5c. |
| **S14** | agent3 | Move R2 (REV1_PLAN.md REV1.1/REV1.2 hardening) parallel with A_main training instead of sequential before. R2 only matters at lock time so doesn't gate A_main launch. | **Adopt.** Saves ~1 day on critical path. Reflected in §10.3 below. |
| **S15** | agent3 | `REV1_PLAN.md §8` menu text says "REV1.1–REV1.4 as one atomic commit" but §9.3 reorganized the chain. User approving "(A) as written" is now ambiguous. | **Adopt.** Will edit `REV1_PLAN.md §8` in `REV1_PLAN.md §10.4`. |
| **S16** | agent3 vs agent4 | Window decision: 90K vs 95K. agent3 argues 90K (model EMA half-life ≈ 6931 steps so 5K buffer is sub-half-life; 90K gives 75 eval points vs 95K's 62; 95K SE penalty = 10%). agent4 argues 95K (5K Phase III settle buffer). | **Verified math**: 5K buffer = 72% of one EMA half-life → reduces residual EMA non-stationarity by ~39% (not "negligible" as agent3 claims, not "fully settle" as agent4 claims). The 10% SE penalty is real. **Decision deferred to user as still-open Q5**. My current lean: **90K** (statistical power outweighs 39% residual EMA reduction; the rollout schedule λ_roll is stationary in Phase III — ramp `0→4.0` ends at step 90K and λ_roll holds constant at 4.0 for [90K,120K], so schedule contributes zero non-stationarity in this window; the only non-stationarity left is the residual EMA drift, whose 39% reduction does not justify a 10% SE penalty). |
| **S17** | agent2 | §9.4 cost table missing sanity B completion (~0.1–0.2 GPU·day for remaining ~10K steps). Add as line item; small but blocks A_main launch. | **Adopt.** Reflected in `REV1_PLAN.md §10.3` cost table. |

### 10.3 Updated commit chain (round 2) — replaces §9.3

C-prefix = pure tooling, no protocol effect. R-prefix = protocol commits.

```text
HEAD (e95f86a, operator reply)
  │
  ├── C1: fix(scripts): step/global_step compat + LOCKED_PROTOCOL_VERSION  [T1, S8, S12]
  │       New file review/0502/scripts/_metrics_compat.py with get_row_step()
  │       LOCKED_PROTOCOL_VERSION = "v0_pending_R1a" constant in lock script
  │       Golden-trace test using operator's tail5 jsonl
  │
  ├── C2: fix(lock_effect_size): URL-based remote autodetection            [T2, S1, S7]
  │       New file .review_canonical_remote (git-tracked, single line)
  │       resolve_canonical_remote() reads file + scans `git remote -v`
  │       --remote stays as explicit override
  │
  ├── C3: fix(select_best_ckpt): step_NNNNNN.pt + last.pt + best.pt        [T3, S10]
  │       Backward-compat with legacy ckpt_step_*.pt names
  │
  ├── C4: feat(lock_effect_size): environment capture in lock doc          [F2-from-Q7]
  │       9 fields per operator Q7
  │
  ├── C5: feat(scripts): post-lock C_uniform full-val wrapper              [F1+F5+S4]
  │       run_c_uniform_full_val.sh + _c_wrapper_guards.py
  │       5 refusal conditions (C5.1-C5.5) + manifest with lock SHA8
  │
  ├── C5b (NEW): fix(run_ablation): hard lock-gate on case "C")            [F11]
  │       Add require_lock_pass() mirroring require_sanity_pass()
  │       Refuses C training start unless EFFECT_SIZE_LOCKED.md exists
  │       AND its commit is on canonical remote
  │       (mirrors v3.1 sentinel pattern)
  │
  ├── C5c (NEW): feat(scripts): verify_paired_seed_configs.py              [S13]
  │       Diffs A_seed=42 vs A_seed=43 yaml, asserts only allowed deltas
  │       Writes SHA256 of diff for lock doc embedding
  │
  ├── C5d (NEW): fix(lock_effect_size): Guard 6 ceiling-string ban         [S11]
  │     Refuse lock docs containing ceiling / headroom / 52.6341 strings
  │
  └── C5e (NEW — round 2.5 per operator flag 4):                          [Op-flag-4]
        chore(run_ablation): refresh help text for current ckpt naming
        Replace stale `ckpt_*.pt` references with `step_*.pt` / `best.pt` /
        `last.pt`. Documentation-only; no behavior change. Optional but
        cheap (~5 minutes).

  ↓ [B] User approves §10.3 commit chain + decides window per Q5
        (recommended: 90K based on rollout schedule deterministic argument)
        + decides A_pair_uniform_spot per F9: align (a/b) or scope-out (c)

  ├── R1a (NEW protocol skeleton, NO data): atomic protocol commit
  │       • POST_V6_NEXT_STEPS.md §6.6.1 window: [40K,60K] → [90K,120K] (or 95K)
  │       • §6.6.1 formula structure: paired_SD_AA on val_chain_normal_mse  [F12]
  │       • §6.6.1 floor: drop
  │       • §6.6.1 X formula: 3 × paired_SD_AA / sqrt(N_eff)
  │       • §6.6.1.1 Window choice rationale (cite §2.1 Phase table)
  │       • §6.6.3 trichotomy: add rel_diff ≤ -X branch
  │       • §6.6.5 ceiling clause: anchor-only, FORBID rescaling
  │       • A_seed=43 yaml spec (delta from A_control = {seed, run_name, output_dir})
  │       • Method-D parameter spec (LOCKED here, not at R1b)
  │       • §6.6.8 changelog citing PEER_REVIEW_GPT55 + PEER_REVIEW_CLAUDE +
  │         OPERATOR_REPLY + (future) MULTI_AGENT_REVIEW_RECORD
  │       • LOCKED_PROTOCOL_VERSION → "R1a_pending_data"
  │       GPG-sign + OTS-stamp (REV1_PLAN.md C9).

  ↓ [C] Operator launches A_main (= A_seed=42, sole-GPU per Q5 decision)
        sanity B finishes first, then A_main solo until 120K

  ├── R2: REV1.1 + REV1.2 hardening                                        [S14]
  │       Lands parallel with A_main training, anytime before [G].
  │       Critical-path savings: ~2h agent overlap (no GPU contention).

  ↓ [D] A_main reaches step 120000 → ckpt + metrics.jsonl frozen

  ↓ [E] Operator launches A_seed=43 (serial, per Q5 decision; sole-GPU)
        verify_paired_seed_configs.py runs first; SHA256 of yaml-diff stashed.

  ↓ [F] A_seed=43 reaches step 120000

  └── R1b (NEW numeric instantiation): atomic protocol commit
        • EFFECT_SIZE_LOCKED.md generated by lock script
          (--metrics-a1=A_main + --metrics-a2=A_seed=43 + verify SHA)
        • LOCKED_PROTOCOL_VERSION → "R1b_locked"
        Plain commit (no GPG/OTS, per Q4 decision); push to gitee
        immediately as time anchor.

  ↓ [H] C_uniform launched via run_ablation.sh C (now refuses if R1b absent, F11)
  ↓ [I] Decision per R1a §6.6.3 trichotomy
```

**Key changes vs §9.3**: C5b/c/d added (F11/S11/S13); R1 split into R1a + R1b (F10); R2 moved parallel with A_main (S14); A_pair_uniform_spot decision surfaced as required user input per F9.

**Updated 2026-05-03 night per user decisions** (see §10.6): A_seed=43 launches **serial after A_main** (single-GPU constraint per Q5), not parallel. R1a / R1b are **plain commits + gitee push** as time anchor (Q4), not GPG/OTS-signed.

### 10.4 Open questions resolved + still open

Resolved by §10:

- §9 F8 (X vs Y placeholder) — **resolved via R1a/R1b split** per F10 (neither X nor Y)
- §9 Q5 (window 90K vs 95K) — **partial**: math verified above; user still picks but with full numeric tradeoff visible

Still open before C1 starts:

1. **Approve §10.3 commit chain** (replaces v0.2 §9.3)
2. **Window decision** per Q5/S16 — recommend 90K
3. **A_pair_uniform_spot decision** per F9 — recommend (a) align by raising max_steps to 120000, or (c) scope-out if Risk 4 not in main claim
4. **Pre-R1a audit artifact**: should I create `review/0503/local/MULTI_AGENT_REVIEW_RECORD.md` (~150 lines, captures all 4 agents' verbatim findings across both rounds) before R1a commits land? agent3 #8 recommended this so R1a §6.6.8 can cite a primary-source audit file rather than the §9/§10 plan-internal summaries. **Default if no answer**: yes, write it before R1a.

### 10.5 §9 items now superseded by §10

| §9 item | Superseded by | Reason |
|---|---|---|
| §9.3 commit chain (R1 single commit) | §10.3 (R1a + R1b split) | F10 ordering correctness |
| §9.3 R2 sequenced before A_main | §10.3 R2 parallel with A_main | S14 critical-path optimization |
| §9.5 #2 "decide window later" | §10.4 #2 (decide before R1a) | window is now part of R1a skeleton, must be in protocol commit |
| §9 F5 "wrapper precheck command" | §10.1 F11 (hard gate in run_ablation.sh) | precheck command is convention; gate is enforcement |
| §9 (no F12-equivalent) | §10.1 F12 | metric for paired_SD_AA needs explicit choice |
| §9 (no F9-equivalent) | §10.1 F9 | A_pair_uniform_spot schedule confound found |

### 10.6 Confirmed user decisions (2026-05-03 night)

In response to the 6 sub-questions raised after §10.5 was first drafted, the user confirmed the following. These are now **frozen** and govern §10.3 / R1a draft.

| Q | Decision | Where it lands |
|---|---|---|
| **window** | **90K** (i.e., window = `[90000, 120000]`) | R1a §6.6.1 |
| **A_pair_uniform_spot (F9)** | **Option (a): raise `max_steps` to 120000 and rerun** with same `warmup_ratio=0.25 / ramp_ratio=0.50` so phase boundaries align with A_main `[0,30K] / [30K,90K] / [90K,120K]`. Run is parallel-able with A_main / A_seed=43 only if a second GPU is available; otherwise queues after A_seed=43 (operator decides at launch time per slot availability). Risk 4 robustness evidence preserved. | R1a §6.6.5 (cite as Risk 4 robustness; not a primary claim) + new yaml in same R1a commit |
| **MULTI_AGENT_REVIEW_RECORD.md** | **Yes, create before R1a commits land.** ~150–200 lines, captures verbatim findings of agent1/2/3/4 across both rounds + my adopt/defer/reject judgments. Becomes primary source for R1a §6.6.8 changelog. | New file `review/0503/local/MULTI_AGENT_REVIEW_RECORD.md`; created between C5d and R1a |
| **Q1 commit chain** | **Approved as §10.3 (with serial-launch + plain-commit overrides below).** | §10.3 above (annotated) |
| **Q2 A_seed=43 yaml form** | **Option (i): physical yaml lives in R1a commit.** R1a generates `review/0502/configs/A_seed43.yaml` by copying A_control.yaml and overriding only `seed`, `run_name`, `output_dir`. `verify_paired_seed_configs.py` (C5c) is rerun at launch time on the actual on-disk yaml to confirm no drift between R1a-time and launch-time. SHA256 of the yaml-diff is stashed to be embedded in R1b. | R1a §6.6.1 (yaml committed) + C5c re-verifies at [E] |
| **Q3 PEER_REVIEW files** | Already on disk at `review/0503/operator/PEER_REVIEW_GPT55.md` and `PEER_REVIEW_CLAUDE.md`; R1a §6.6.8 cites them by relative path, no new files needed. | R1a §6.6.8 changelog citation |
| **Q4 GPG / OTS** | **Downgrade to plain commit + push to canonical remote as time anchor.** Reasoning: gitee push timestamp is itself a credible weak time anchor (third-party-hosted, not author-controlled). R1a / R1b plain `git commit -s` (sign-off only) + `git push <canonical-remote> foc_lite_hop0` immediately after commit, where `<canonical-remote>` is resolved via `.review_canonical_remote` (on operator host = `origin`, on local Mac = `gitee`). Lock script's environment-capture (C4) records git commit SHA + push timestamp from `git log <canonical>/foc_lite_hop0`. | §10.3 chain annotated above; R1a/R1b commit recipes drop GPG-sign + OTS-stamp lines |
| **Q5 launch order** | **Serial: A_main (= A_seed=42) first → A_seed=43 second.** Single-GPU constraint on operator host. Adds ~1.5–2 days to critical path vs parallel option. R2 (REV1.1/REV1.2 hardening) still lands parallel with A_main since R2 is agent-only work (no GPU contention). | §10.3 chain annotated above |
| **Q6 Method-D parameter inventory** | **Approved my recommendation to inventory now.** Inventory of `select_best_ckpt_smoothed.py` parameters: `--neighborhood K` (default `10`, 21-row total smoothing); `--metric-key` (auto-chain: `val_select_score` first, fallback `val_chain_normal_mse`); `--include-best-pt` (default off); `--include-last-pt` (default off); `--metrics` (path); `--ckpt-dir` (path). **R1a locks**: `K=10`, `metric-key=val_select_score` (composite Method-D key per Phase 6.6 selector design), `include-best-pt=False`, `include-last-pt=False`. Note: Method-D selector key is `val_select_score` (correct here) while paired_SD_AA / headline rel_diff are on `val_chain_normal_mse` per F12 — these are **deliberately different metrics for different purposes** (selection vs effect-size). | R1a §6.6.1 + new sub-bullet `Method-D Parameter Lock` |
| **Q7 run_ablation.sh hard-gate strictness** | **Approved 3-check version: file exists + on canonical remote (`<canonical>/foc_lite_hop0`, resolved per `.review_canonical_remote`) + `LOCKED_PROTOCOL_VERSION == "R1b_locked"`.** Pre-registration's "未公开就不算 locked" principle outweighs operator inconvenience. Operator workflow: commit + immediate push, then launch C. | C5b in §10.3 above; bash implementation in C5b commit |

**Operational consequences for the agent (me) starting C1**:

1. C1 lands `LOCKED_PROTOCOL_VERSION = "v0_pending_R1a"` constant.
2. Between C5d and R1a, write `MULTI_AGENT_REVIEW_RECORD.md` from the verbatim agent1–4 transcripts (round 1 + round 2) plus my adopt/defer/reject column. **This is the only new markdown file the user has explicitly requested in this absorption pass.**
3. R1a draft includes A_seed43.yaml + A_pair_uniform_spot_aligned.yaml (max_steps=120000), Method-D parameter block locked to the values in Q6, window `[90000, 120000]`, paired_SD_AA on `val_chain_normal_mse` (F12), §6.6.5 ceiling-as-anchor-only clause, §6.6.8 changelog citing 4 external sources.
4. R1a + R1b are plain commits, no GPG/OTS. Both pushed to `<canonical-remote>/foc_lite_hop0` immediately on commit (where `<canonical-remote>` is resolved per `.review_canonical_remote` on each host).
5. C5b's `require_lock_pass()` queries `git merge-base --is-ancestor <lock_commit> <canonical>/foc_lite_hop0` for the canonical-remote check, with `<canonical>` resolved at runtime.
6. Cost table in `REV1_PLAN.md §10.3` updated to reflect serial A_main → A_seed=43 (~1.5–2 day critical-path increase).

---

## 11. Operator round-2.5 absorption (drafted 2026-05-03 night)

After §10 was first pushed (commit `23a11d4`), operator answered Q9–Q15 and added 5 unsolicited flags (commit `2d3ae5f` on canonical remote). Files added under `review/0503/operator/`:

- `OPERATOR_REPLY_pre_C1_20260503.md` (148 lines)
- `current_experiment_architecture_status_20260503.md` (275 lines, full GPU/run/architecture audit)
- `B_sanity_metrics_val50_schema_reference.jsonl` (46 val rows — see §11.2 below)

This section freezes operator's answers + flags into the plan. Together with §10.6, all open questions are now resolved before C1 starts.

### 11.1 Operator's 7 answers (frozen)

| Q | Operator answer | Plan effect |
|---|---|---|
| **Q9 sanity B state + path** | (a). B is at step 18800/20000 on GPU1 (PID 803742), `metrics.jsonl` at `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0502_runs/B/run/first_hop_224_sigma_norm_B/metrics.jsonl`. **46 val rows**, last val step=18400 (`val_select_score=0.0018183143`, `val_chain_normal_mse=0.0005981120`). Sentinel `review/0502/runs/.sanity_pass` does NOT exist yet. The 46-row trace is copied to `review/0503/operator/B_sanity_metrics_val50_schema_reference.jsonl` (filename says "val50" but contains 46; will refresh on B completion). | C1 golden-trace test gets a **two-fixture** setup: **5-row** A tail (`A_sanity_metrics_tail5_schema_reference.jsonl`) for minimal smoke + **46-row** B trace (`B_sanity_metrics_val50_schema_reference.jsonl`) for stronger regression coverage. Filename mismatch tolerated; helper text notes "actual row count read at runtime, not asserted to 50". |
| **Q10 canonical-remote interpretation** | (i) URL fragment. `.review_canonical_remote` content **must be** `gitee.com:jqu9/PET_LatentResidual` (not a remote nickname). Resolver scans `git remote -v`, returns the local remote name whose URL contains this fragment. **On operator host the canonical remote is named `origin`, not `gitee`**. Any literal `gitee/foc_lite_hop0` or `git push gitee` in plan/code is operator-host-incompatible. | C2 implementation: `.review_canonical_remote` literally contains the URL fragment `gitee.com:jqu9/PET_LatentResidual`; resolver returns whichever local remote matches. Plan §10.6 Q4/Q7 + operational consequences #4/#5 already reworded above to use `<canonical-remote>`/`<canonical>` placeholders. C5b's `require_lock_pass()` resolves `<canonical>` via the same helper. |
| **Q11 A_pair rerun scope** | (a-with-cleanup). Only semantic change: `training.max_steps: 60000 → 120000`. Keep `pair_loss_weights: [1,1,1,1]` and the spot-specific `pair_dataset_mode` etc. intact. `save_interval: 10000` is acceptable (changes ckpt density only). Clean up stale 60K/200K comments. New file: `review/0502/configs/A_pair_uniform_spot_aligned.yaml`. | R1a generates `A_pair_uniform_spot_aligned.yaml` as 1-field semantic copy + comment cleanup of the source. Diff vs source committed inside R1a so it's reviewable. |
| **Q12 A_main launch command** | **Use `run_ablation.sh A`, not direct `train_first_hop.py`.** `train_first_hop.py` only supports `--config` and `--resume` flags (no `--output-dir`). Canonical recipe: `cd /home/qujiaxiang/project/PET_LatentResidual && PYTHON=/home/qujiaxiang/.conda/envs/rae/bin/python GPU=<free_gpu> bash review/0502/scripts/run_ablation.sh A`. Outputs land at `review/0502/runs/A_main/{config.resolved.yaml, train.log}` and `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0502_runs/A_main/run/first_hop_224_sigma_norm_A_main/{metrics.jsonl, best.pt, last.pt}`. | R1a §6.6.1 reproducibility recipe embeds the operator's exact command verbatim. R1a section explicitly NOTES that `train_first_hop.py` does not accept `--output-dir`; the launcher composes the resolved output dir. |
| **Q13 A_seed=43 output_dir** | (i) physical yaml in R1a, but follow project-standard layout (not date-stamped free-form). Config: `review/0502/configs/A_seed43.yaml`. Allowed deltas vs `A_control.yaml`: `seed: 43`, `run_name: first_hop_224_sigma_norm_A_seed43`, `output_dir: /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0502_runs/A_seed43/run`. Final run dir: `<output_dir>/first_hop_224_sigma_norm_A_seed43/`. | R1a generates `A_seed43.yaml` with exactly these 3 fields different. C5c (`verify_paired_seed_configs.py`) asserts only these allowed deltas. SHA256 of the diff stashed for R1b. |
| **Q14 GPU availability** | All 4 GPUs occupied at sample time. GPU0 unrelated 100%. GPU1 B sanity (until B completes). GPU2 unrelated + A_sanity_light. GPU3 V6.1 47%. **A_main launch precondition: B sentinel + C1–C5e + R1a all landed + at least one A6000 freed up.** No clock-time commitment; operationally "after B sentinel + R1a, on the first free GPU". | `REV1_PLAN.md §10.3` cost table needs annotation: "wall-clock estimate begins from first-free-GPU + R1a, not from now". |
| **Q15 Deadline** | None known to operator host. **No hard deadline.** Treat as such unless PI/user provides a date. | A_pair option (a) stands. No critical-path compression needed. |

### 11.2 Operator's 5 unsolicited flags

| Flag | Operator's wording | My response |
|---|---|---|
| **Op-flag-1** Remote-name portability | Any hard-coded `gitee/foc_lite_hop0` or `git push gitee ...` is incompatible with operator host (canonical = `origin` there). Use the resolver from Q10. | **Already fixed in §10.6 Q4/Q7 + operational consequences #4/#5** (replaced `gitee/foc_lite_hop0` literals with `<canonical>/foc_lite_hop0` portable wording). C2 implementation already targeted URL-fragment resolver; no design change needed. |
| **Op-flag-2** C1/T1 unpatched | `lock_effect_size_threshold.py`, `paired_diff_judge.py`, `select_best_ckpt_smoothed.py` still read `global_step`. Real metrics row uses `step`. | **Confirmed — C1 is exactly this fix.** Implementation pending operator approval to start. |
| **Op-flag-3** C3/T3 unpatched | Trainer writes `step_NNNNNN.pt` (zero-padded 6 digits, e.g., `step_020000.pt`) + `best.pt` + `last.pt`. `select_best_ckpt_smoothed.py` searches `ckpt_step_*.pt` + `ckpt_last.pt`. | **Confirmed — C3 is exactly this fix.** My Mac-side grep verified `train_first_hop.py:2420` writes `f"step_{step:06d}.pt"`. C3 will glob both the legacy `ckpt_step_*.pt` AND the current `step_*.pt`, plus probe `last.pt` / `best.pt` (no `ckpt_` prefix). |
| **Op-flag-4** `run_ablation.sh` help stale | Help text says products are `ckpt_*.pt`; actual products are `step_*.pt` + `best.pt` + `last.pt`. | **New tiny commit C5e added** to §10.3 chain above. ~5 minutes of grep+sed. Not gating; can be batched after C5d. |
| **Op-flag-5** Guard 5 wording | Config-side `best_metric: val_multi_objective`; metrics-side computed key is `val_select_score`. Guard 5 must verify both, not require config `best_metric == val_select_score`. | **Already correct in §9.1 v0.1 fix table** (Tier 3 C12 row). Operator is reaffirming. C1 / lock script's Guard 5 will assert `best_metric == "val_multi_objective"` (config-side) AND `"val_select_score" in metrics_row` (data-side). Both must hold simultaneously. |

### 11.3 Updated commit chain (replaces §10.3 commit-chain box)

The §10.3 chain is unchanged in topology except for **C5e added between C5d and [B] approve gate**, and **launch ordering inside [C]–[E] now matches operator's serial constraint** (already reflected in §10.3). The single source of truth remains §10.3 above with C5e inserted.

For pre-launch readability:

```
HEAD (2d3ae5f, this push will be 2d3ae5f..<new_hash>)
  → C1 (T1+S12, schema compat)
  → C2 (URL-fragment canonical-remote resolver per Q10)
  → C3 (step_NNNNNN.pt + last.pt + best.pt per Q9/Op-flag-3)
  → C4 (env capture)
  → C5 (post-lock C wrapper)
  → C5b (run_ablation.sh require_lock_pass per F11)
  → C5c (verify_paired_seed_configs.py per S13)
  → C5d (Guard 6 ceiling-ban per S11)
  → C5e (run_ablation.sh help refresh per Op-flag-4) — NEW
  → MULTI_AGENT_REVIEW_RECORD.md (between C5e and R1a)
  → R1a (skeleton: window=90K, A_seed43.yaml, A_pair_uniform_spot_aligned.yaml, Method-D K=10 lock, §6.6.8 changelog)
  → operator: B sentinel writes → A_main via run_ablation.sh A → step 120K
  → R2 (REV1.1/REV1.2 hardening, parallel agent-side with A_main training)
  → operator: A_seed=43 via run_ablation.sh A_seed43 (or equivalent) → step 120K
  → R1b (numeric: lock script consumes both metrics.jsonl + verify_paired_seed_configs SHA256)
  → operator: C_uniform via run_ablation.sh C (refused unless require_lock_pass passes)
  → decision per R1a §6.6.3 trichotomy
```

### 11.4 What's resolved + what's now blocking

**Resolved by §11**:

- §10.4 #1 (approve §10.3 chain) — approved by user
- §10.4 #2 (window decision) — locked to 90K
- §10.4 #3 (A_pair F9 decision) — locked to (a) with `max_steps=120000` + comment cleanup
- §10.4 #4 (MULTI_AGENT_REVIEW_RECORD.md decision) — yes; landing in this same checkpoint commit
- §10.6 Q9–Q15 — all 7 answered
- Op-flag-1 — fixed in §10.6 portable wording
- Op-flag-4 — added as C5e

**Still blocking C1 implementation**: nothing. C1 can start as soon as this checkpoint commit lands. C1–C5e are all design-frozen.

**Still blocking R1a**: nothing. R1a content is fully spec'd by §11.1 Q9/Q11/Q12/Q13 + §10.6 Q6 (Method-D lock) + §11.1 Q9 (two-fixture C1 test).

**Still blocking A_main launch (operator-side)**: B sentinel must write before A_main starts. C1–C5e + R1a must merge before A_main starts. At least one A6000 must free up. None of these are blocking the agent's design work.

### 11.5 Self-correction noted on push

Earlier draft of §10.6 said "λ_roll = 0 in Phase III" (claiming schedule contributes zero non-stationarity because the value is zero). This was reworded after I re-read `A_control.yaml:22-26` to **"λ_roll holds constant at 4.0 in Phase III"** — schedule is stationary in [90K,120K] but at the constant `4.0`, not at `0`. The 90K window recommendation is unchanged because schedule stationarity (zero derivative) is what matters, not the absolute value. Already pushed in commit `23a11d4`; this paragraph is the audit-trail note.
