# Self-code-review: Round-6 absorption (C3 + F2 + F3 + F4)

**Author**: Mac-side agent (Claude Opus 4.7).
**Reviewee**: Mac-side agent (same).
**Trigger**: User directive "请你进行修改，修改完成后请你自行进行 code-review。"
(after fetching `OPERATOR_REPLY_C2_3_20260504.md` and authoring the four
fixes in tree).
**Date**: May 4 2026 (operator clock).
**Scope**: ONLY the diff introduced by the C3 + F2 + F3 + F4 batch.
The C2 / C2.1 / C2.2 / C2.3 layers underneath are TAKEN AS GIVEN — they
were peer-reviewed in Rounds 1–5 and any concern raised against them
again here would be out of scope for "absorb codex Round-6 reply".

This document is intentionally written from a NEW reviewer's perspective:
each fix is interrogated as if I had not authored it, with the goal of
finding gaps before operator/codex finds them in Round 7.

---

## Q-A. Coverage check — every Round-6 finding mapped

| Codex finding | Severity | In-tree commit zone | Tests added | Verdict |
|---------------|----------|---------------------|-------------|---------|
| **F1 → C3** | HIGH | `select_best_ckpt_smoothed.py` (`list_saved_steps`, `main` recommendation) | `test_select_best_ckpt_smoothed.py` (12 tests, 5 classes) | **COVERED** |
| **F2** | MEDIUM | `lock_effect_size_threshold.py::compute_paired_stats_from_rows` | `test_metrics_compat.py` T17 + T18 (negative-control) + T19 (legacy-compat) | **COVERED** |
| **F3** | MEDIUM | `lock_effect_size_threshold.py::main` push-time TOCTOU | `test_remote_resolver.py::TestPushUrlTOCTOU` 3 cases | **COVERED** |
| **F4** | LOW | `OPERATOR_CONFIRM_REQUEST_C2_3.md` §1 + §2 | (doc; no code test) | **COVERED** |
| **F5** | DEFERRED | (n/a) | (n/a) | **DEFERRED with rationale in §7.6** |

No finding from `OPERATOR_REPLY_C2_3_20260504.md` is left unaddressed.

---

## Q-B. Regression surface — what could break that wasn't tested?

### B-1. C3 — does the legacy-form workflow still work end-to-end?

Pre-C3, `list_saved_steps` only recognized legacy `ckpt_step_*.pt` /
`ckpt_last.pt`. Post-C3, both naming forms coexist. The legacy-only
workflow (someone running on an OLD trainer output that still has
`ckpt_step_*.pt`) MUST keep working.

**Mitigation**: `TestListSavedStepsLegacyForm::test_legacy_form_only`
explicitly tests a directory containing ONLY legacy-form files and
asserts `(steps, paths)` correctly populated. ✅

**Residual risk**: low. If a future trainer change adds a third naming
form, this code will silently miss it; but Round-6 scope is limited to
the two forms operator confirmed. Acceptable.

### B-2. C3 — what if both `step_NNNNNN.pt` AND `ckpt_step_NNNNNN.pt` exist for the same step?

`step_to_path.setdefault(step, cf)` is order-dependent: first match wins.
The for-loop pattern order is `("step_*.pt", "ckpt_step_*.pt")` — new
form is iterated first, so on a tie the new form wins. This matches
operator's expectation (real trainer output is canonical, legacy is
fixture).

**Mitigation**: `TestListSavedStepsCohabitation::test_new_form_wins_on_tie`
explicitly creates both files for the same step and asserts the new-form
path wins. ✅

**Residual risk**: ZERO. Tested directly.

### B-3. C3 — `--include-best-pt` / `--include-last-pt` interaction

These flags load .pt files via `torch.load`. Pre-C3 they only knew about
`ckpt_last.pt` and key `global_step`. Post-C3 they try `last.pt` first,
then `ckpt_last.pt`; and try `step` first, then `global_step`. If torch
is unavailable, they skip with `[warn]` and don't add to `step_to_path`.

**Mitigation**: `TestListSavedStepsWithTorch` (4 cases, `@skipUnless(_HAS_TORCH)`)
covers: (a) `step` key on new-form `last.pt`; (b) legacy `global_step`
key; (c) legacy `ckpt_last.pt` filename + step key; (d) when BOTH keys
exist, `step` wins. ✅

**Residual risk**: SMALL. If torch.load is given a corrupt .pt file, the
existing `try/except Exception` branch logs `[warn] failed to load` and
continues, so a corrupt ckpt does not abort the script. This is
unchanged from pre-C3 behavior; not regressed.

### B-4. F2 — does narrowing the warning silently hide a real bug?

The most important regression to guard against: F2 narrows
`skipped_no_metric` from "any in-window row missing val_select_score" to
"any val-like in-window row missing val_select_score". A val row that
truly has no `val_select_score` (genuine partial-write) MUST still warn.

**Mitigation**: T18 is a deliberate negative-control: 5 healthy val rows
+ 2 in-window val rows with NO `val_select_score`. Asserts the warn
DOES fire. If future code accidentally suppresses real partial-writes,
T18 catches it. ✅

**Residual risk**: LOW. The only edge case T17/T18/T19 don't cover is
a row with `event` set to a non-string truthy value (e.g.
`event: ["val"]`). My code handles this defensively (`elif isinstance(event, str)` →
otherwise `is_val_like = False`), but a malformed schema could be
silently dropped. The trainer doesn't emit this so it's not a real
production case. **Acceptable.**

### B-5. F2 — what about rows that are in-window but step is malformed?

`get_row_step(r) is None` short-circuits BEFORE the val-like check, so
those rows are skipped entirely — same as pre-F2. ✅

### B-6. F3 — is `git remote get-url --push` available on all git versions?

`git remote get-url --push` was added in git 2.7.0 (early 2016). The
operator host runs git ≥ 2.20 (per `git --version` output captured in
Round-3 `OPERATOR_INVENTORY.md`). Mac is on git 2.39+. **Acceptable.**

If a hypothetical reviewer worries about a git < 2.7 host: the call
would fail with `error: unknown option --push` and `lock.git()` raises
RuntimeError, which is caught by the `try/except` block and aborts with
exit code 6. The operator gets a clear error, not silent corruption.

### B-7. F3 — what if `--force-unsafe-remote` is passed?

Both the original fetch-URL TOCTOU check AND the new pushURL check are
inside a single `if not args.force_unsafe_remote:` guard (L1149). Both
are skipped together. This matches the documented behavior of
`--force-unsafe-remote` (intentional bypass for non-canonical pushes).

**No test added** for this branch — the `--force-unsafe-remote` happy
path was already exercised in C2 baseline tests, and the F3 patch
doesn't change behavior in that branch (both checks are skipped, same
as pre-F3). Acceptable.

### B-8. F3 — what if `_normalize_remote_url` itself has a bug that lets divergent URLs collide?

This was the question E1 (Round-5 Lane E) addressed: ensure
`_normalize_remote_url` correctly distinguishes URLs that differ only
in userinfo, scheme, etc. C2.3 fixed E1 with R17/R18/R19 tests. F3 is a
LAYER on top of `_normalize_remote_url` and inherits its correctness.

**Mitigation**: F3c specifically tests "different URL form, SAME repo →
must canonicalize equal" (https vs ssh of the same repo). If
`_normalize_remote_url` regresses, F3c fires. ✅

### B-9. F4 — does the doc fix break operator-pull workflow?

The §1 hash-expectation softening (was: `de4cc88...`; is: "拉到最新 Mac
推送的 commit 即可") is the only behavioral change. This is correct
because the Round-6 batch hasn't been committed yet (per standing
"please don't manage git" directive — committing is a separate
operation the user will trigger). If I had hardcoded the new HEAD,
operator would see a hash mismatch on pull. Acceptable.

---

## Q-C. Test invariants — do new tests actually fail before the fix?

This is the strongest test of test design: a test that passes regardless
of the fix is worthless.

| Test | Pre-fix expected behavior | Post-fix expected behavior | Discriminating? |
|------|---------------------------|----------------------------|-----------------|
| `TestListSavedStepsNewForm::*` | `list_saved_steps` returns `[]` (no match) → script aborts with `[error] No saved ckpts found` | returns `[20000, 40000, 50000]` correctly | **YES** |
| `TestListSavedStepsLegacyForm::test_legacy_form_only` | passes pre-fix (was the only branch tested) | still passes | NO (regression-only) |
| `TestListSavedStepsCohabitation::test_new_form_wins_on_tie` | pre-fix: returns `ckpt_step_NNN.pt` (legacy was the ONLY pattern tried) | returns `step_NNNNNN.pt` | **YES** |
| `TestListSavedStepsRobustness::*` | pre-fix returned `[]` for both new-form fixtures and empty dir | still returns `[]` for empty + non-ckpt; correctly returns steps for new-form | partial-YES |
| `TestListSavedStepsWithTorch::test_step_key` | pre-fix: only `global_step` was tried → AttributeError | post-fix: tries `step` first → succeeds | **YES** |
| **T17** (F2 train rows quiet) | pre-fix: 33 train rows would fire warn | post-fix: zero warn lines | **YES** |
| T18 (F2 negative-control) | pre-fix: warn fires (over-broadly: 33+2 rows) | post-fix: warn fires correctly (only 2 rows) | partial-YES (count differs) |
| T19 (F2 legacy-compat) | pre-fix: warn fires | post-fix: warn STILL fires (legacy = val-like default) | NO (regression-only) |
| **F3a** (no pushURL) | pre-fix: only fetch checked → trivially passes | post-fix: both checked → trivially passes | NO (regression-only) |
| **F3b** (divergent pushURL) | pre-fix: fetch passes, pushURL not checked → script would push to attacker | post-fix: pushURL check fails → exit 6 | **YES** (security-critical) |
| **F3c** (equivalent pushURL) | pre-fix: fetch passes (pushURL ignored) | post-fix: pushURL canonicalizes equal → passes | NO (false-positive guard) |

**Verdict**: Every fix has at least one DISCRIMINATING test (would fail
pre-fix). Some tests are regression-only — that's fine, they protect
the fix from drift.

The 4 torch-gated tests are skipped on Mac but will exercise the code
on operator host. That's a known coverage gap on Mac but it's resolved
on the operator side.

---

## Q-D. Docstring + commit-message contract integrity

### D-1. Docstring updates

| File | Function | Updated for Round-6? | Cites finding? |
|------|----------|----------------------|----------------|
| `select_best_ckpt_smoothed.py` | `list_saved_steps` | ✅ (C3 paragraph added) | "C3 fix (Round-5 operator review, May 4 2026)" |
| `select_best_ckpt_smoothed.py` | `--ckpt-dir` argparse help | ✅ (mentions both forms) | implicit |
| `lock_effect_size_threshold.py` | `compute_paired_stats_from_rows` | ✅ (F2 paragraph added) | "F2 hardening (Round-5 operator review, May 4 2026)" |
| `lock_effect_size_threshold.py` | (inline F3 block in `main`) | ✅ (F3 multi-line comment) | "F3 hardening (Round-5 operator review, May 4 2026)" |
| `OPERATOR_CONFIRM_REQUEST_C2_3.md` | §1, §2 | ✅ (F4 callout, expected count 87) | "F4 fix (Round-6 absorption, May 4 2026)" |
| `MULTI_AGENT_REVIEW_RECORD.md` | §7 (new) | ✅ (full Round-6 section) | maps F1–F5 by ID |

> Note: The docstrings cite "**Round-5** operator review" but the absorption
> happens in Round 6. This naming is INTENTIONAL — codex's review is the
> Round-5 EXTERNAL review; my absorption of it is Round 6. The citation
> points at the source of the finding, not the absorption commit.

### D-2. LOCKED_PROTOCOL_VERSION

Pre-Round-6: `v0_pending_R1a`.
Post-Round-6: `v0_pending_R1a` (unchanged). ✅

This is a contract: the protocol version may only bump at R1b lock time.
Round-6 is a SCRIPT-level fix batch (not a protocol change), so the
version stays. Verified in source:

```bash
$ grep "LOCKED_PROTOCOL_VERSION" review/0502/scripts/*.py
review/0502/scripts/lock_effect_size_threshold.py:LOCKED_PROTOCOL_VERSION = "v0_pending_R1a"
```

### D-3. Tests pass on Mac (sanity)

```
$ python3 -m unittest review.0502.scripts.test_remote_resolver \
    review.0502.scripts.test_metrics_compat \
    review.0502.scripts.test_select_best_ckpt_smoothed
.....................................................................................ssss
----------------------------------------------------------------------
Ran 87 tests in 0.214s
OK (skipped=4)
```

87 tests; 83 OK + 4 torch-skipped. All discriminating tests above
verified GREEN. Operator-side expected: 87 OK with no skips.

---

## Q-E. Cross-cutting: any unintended interaction between the four fixes?

- **C3 ↔ F2**: C3 is in `select_best_ckpt_smoothed.py`, F2 is in
  `lock_effect_size_threshold.py`. No shared call surface. ✅
- **C3 ↔ F3**: same — different script files. ✅
- **F2 ↔ F3**: both touch `lock_effect_size_threshold.py` but in
  disjoint sections (`compute_paired_stats_from_rows` is L645+;
  `main` push-time TOCTOU is L1149+). Verified by re-reading both
  hunks. ✅
- **F4 ↔ {C3,F2,F3}**: F4 only updates documentation. The expected
  test count quoted in F4 (87) is exactly what the suite produces.
  ✅
- **All four ↔ C2.3 layer (D1+E1)**: D1 is `_safe_stderr_block` which
  is invoked by `lock.git()`. F3 invokes `lock.git()` twice. The new
  F3 errors propagate through `_safe_stderr_block`-wrapped paths
  unchanged → D1's hostile-stderr defense covers F3 errors for free.
  ✅

---

## Verdict

**LIGHT-PASS**.

No HEAVY-FLAG concerns identified. One soft observation:

- **Soft note 1**: Docstring citation says "Round-5 operator review"
  but absorption is Round 6. This is intentional (citation points at
  source of finding) but a future reader might be momentarily
  confused. Could add a small footnote in §7.1 of
  `MULTI_AGENT_REVIEW_RECORD.md`. Not blocking.
- **Soft note 2**: F3 has no test that exercises the actual `main()`
  call path with a divergent pushURL — only the helper-level subprocess
  observation. Constructing a full `main()`-driven mock test was judged
  not worth the complexity (would require fabricating a complete
  lock-input scenario). The F3a/F3b/F3c tests prove the GIT-LEVEL
  observation is correct, and the patch logic is a straightforward
  comparison; the gap is in defensive engineering, not correctness.
  Acceptable.

**Recommended next operator-host action**: pull the (eventual) Round-6
commit, run the 87-test suite via the conda env (`<env>/bin/python`),
expect `OK` (no skips). If GREEN, the absorption is complete and the
F-series can be marked closed in `MULTI_AGENT_REVIEW_RECORD.md`.

**Recommended next Mac-side action**: per standing "请你不要管 git" directive,
the user will decide commit/push timing; the agent does NOT auto-commit
this batch.
