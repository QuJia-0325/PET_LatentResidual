# SELF_REVIEW Round 8 — G1 + G2 + G3 (+ G4 doc + G5 lockstep)

**Date**: May 4 2026
**Branch**: `foc_lite_hop0`
**Round-7 source**: GPT-5.5 Xhigh + Claude 4.7 Extra high cross-AI peer
review of Round-6 batch (commit `27b49bd`). Both verdicts: LIGHT-FLAG.
Three consensus findings (G1 / G2 / G3) + two single-reviewer findings
(G4 / G5). User authorized option (A) — land all five in tree.

**Scope of this self-review**: the in-tree Round-8 patches BEFORE
commit. Same Q-A / Q-B / Q-C / Q-D / Q-E lens as
`SELF_REVIEW_C3_F2_F3_F4.md`.

---

## Q-A. Does each Round-7 finding have a fix that addresses the actual mechanism (not just the symptom)?

| Finding | Symptom (what reviewer reported) | Mechanism (root cause) | Round-8 fix targets | Verdict |
|---------|----------------------------------|------------------------|---------------------|---------|
| **G1** | Multi-pushURL config bypasses F3 canonical check | `git remote get-url --push <remote>` (singular) returns ONLY the first pushURL even when `git push` mirrors to all. F3 enumerated the wrong set. | `--push --all` enumeration + per-URL canonical compare loop + defensive empty-list check + plural pushURL(s) framing in error message | **Mechanism** — fix targets the enumeration scope, not the comparison logic |
| **G2** | C3 silently picks new-form on collision; tie test uses empty bytes | `setdefault` is silent by design. The Round-6 fixture couldn't distinguish "same content, different name" from "different content, same step" | Explicit `if existing.name != cf.name` branch; `[warn]` line names both files + `(kept)` / `(ignored)` labels + `sha256sum` hint | **Mechanism** — fix surfaces the silent-precedence at the moment of conflict |
| **G3** | F5 deferral rationale ("locked-sample contract") was wrong because no locked sample exists yet | Round-6 conflated "must remain stable post-R1a-lock" with "must remain stable pre-R1a-lock". The contract is on the LOCKED artifact, not the lock TOOL. | Strict `cfg_metric_key != "val_multi_objective"` + opt-out flag with mandatory warn + warn cites "R1a lock time" | **Mechanism** — fix tightens the contract at lock time (when the contract starts), not deferred to a future patch |
| **G4** | Doc straggler "这 90 测试" at line 106 | Round-6 F4 was a multi-place rewrite from 69→87; one location missed | Replace 90→97 (current count post-Round-8) AND relabel all call-sites with "Round-6 + Round-8 absorption" prefix | **Mechanism** — both straggler and stale baseline addressed simultaneously |
| **G5** | F3 inline comment said "pushURL" (singular) | Conceptual loop: same single-pushURL framing that produced G1 is still in the code as a comment | Plural `pushURL(s)` framing + dedicated G1-hardening docstring paragraph | **Mechanism** — closes the conceptual loop so future readers don't re-introduce G1 |

**Coverage**: 5/5 findings addressed at the mechanism level. No "I just fixed the symptom" patches.

## Q-B. Does any Round-8 patch regress an existing Round 1-6 test or invariant?

Existing test suite before Round 8: 87 tests. Existing on Mac post-Round-8: still 87 OK (the 10 net-new tests are clean additions; no test was edited or deleted). Verified by full-suite run: `Ran 97 tests in 0.289s OK (skipped=4)`.

Specific invariants checked:

- **G1** does NOT change behavior on the legitimate single-pushURL case (verified: F3a/F3b/F3c continue to pass). It also does NOT regress on the legitimate multi-pushURL-to-same-canonical case (verified: F3e new negative-control test).
- **G2** preserves the Round-6 documented precedence (new-form wins on tie). Verified: `test_new_form_wins_on_tie` still passes; `test_collision_with_different_contents_warns` ALSO asserts the new-form path is the one returned. Round-6 policy intact.
- **G3** does NOT change behavior on the canonical case (`val_multi_objective`). Verified: T16 (whose fixture uses canonical) continues to pass; G3a explicitly asserts canonical → rc=0. The previously-permissive `val_select_score` and "missing key" branches are now strict-rejected, which is the WHOLE POINT — these were never legitimate values, only Round-6 over-permissiveness.
- **G4** is doc-only.
- **G5** is comment-only.

**Verdict**: no regression. 87 → 97, monotone.

## Q-C. Falsifiability — what would each new test fail to catch?

| New test | What it asserts | What it would still miss |
|----------|-----------------|--------------------------|
| **F3d** (G1 discriminating) | `--push --add` × 2 with hostile second URL → `--push` (singular) returns canonical only AND `--push --all` enumerates both AND the loop catches the divergent one | Misses: a git version where `--push --all` itself is unreliable (unlikely; supported since git 2.7). Also misses: a `pushInsteadOf` config that rewrites canonical → hostile at push time (different attack surface from multi-pushURL; not in scope for G1) |
| **F3e** (G1 negative-control) | ssh+https forms of same canonical normalize equal | Misses: a custom canonical-normalize bug that happens to make them equal incorrectly. But `_normalize_remote_url` is already covered by D1/E1 baselines |
| **collision_with_different_contents_warns** (G2 discriminating) | Both files exist with DIFFERENT byte contents → `[warn]` mentions both names + (kept)/(ignored) labels | Misses: a runtime where both `glob`s return SAME path object somehow. But the `existing.name != cf.name` guard handles that case (same name → no warn, just kept) |
| **no_collision_no_warning** (G2 negative-control) | Only new-form files → no warn | Misses: a regression where the warn fires on a different code path. But there's only one place in `list_saved_steps` that builds `step_to_path`, and the test exercises it |
| **G3a–G3f** (6 tests) | All combinations of {canonical, missing, val_select_score, unknown} × {flag, no flag} | Misses: a YAML structure where `training` block exists but `best_metric` is null vs missing (these are equivalent for `dict.get`; G3b covers via "<MISSING>" sentinel). Also misses: a config where `best_metric` is a typo with mixed case (e.g. `Val_Multi_Objective`); strict check is exact-string equal so this fails closed (which is the desired behavior — strict means strict) |

**Coverage gap acceptance**: the misses above are out-of-scope edge cases or cases where the strict check fails in the safe direction (rejection). No false-pass paths identified.

## Q-D. Contract integrity — does Round 8 leak any state into the LOCKED artifact?

`LOCKED_PROTOCOL_VERSION` is unchanged: still `v0_pending_R1a`. The lock-artifact format (`EFFECT_SIZE_LOCKED.md`) is unchanged. The post-lock invariants are unchanged.

What G1/G2/G3 change is exclusively pre-lock validation: tighter pre-conditions on what the operator can run `lock_effect_size_threshold.py` against. Once the lock fires, downstream artifacts and consumer scripts see the same shape they would have seen at Round 6.

The opt-out `--allow-dev-best-metric` is the only addition that could in principle be misused at lock time. Mitigation:
- The argparse help string says "DEVELOPMENT-ONLY ... NEVER pass at R1a lock time".
- The `[warn]` text includes the literal phrase "do not use at R1a lock time".
- The MULTI_AGENT_REVIEW_RECORD §8.4 documents the opt-out as "for development use only — do not pass at R1a lock time".
- Test G3e asserts the warn must appear, which prevents accidental silent bypass.

**Verdict**: contract intact. The opt-out is a development affordance, not a contract loosening.

## Q-E. Cross-cutting integrity

- **Two reviewers, one finding-set**: G1/G2/G3 were raised by BOTH reviewers (consensus); G4/G5 by single reviewers. Round 8 absorbs all five. No reviewer-specific cherry-picking.
- **Empirical verification**: G1's bypass mechanism was independently empirically verified by both reviewers AND by the Mac agent during patch authoring (the `git init + set-url --push --add` ×2 probe was run before writing the F3d test). This was not hypothesis-driven; the test reflects a measured behavior.
- **Test count audit**: 87 (pre) + 2 (G1 F3d/F3e) + 2 (G2 collision/no-collision) + 6 (G3 G3a–G3f) = 97 (post). Verified by `unittest` run; matches §8.9 of MULTI_AGENT_REVIEW_RECORD.
- **Doc count consistency**: OPERATOR_CONFIRM_REQUEST_C2_3.md preface, §2 expected output box, and stderr-stragglers note now ALL say 97 (G4 fix). The historical "69/69" reference at line 60 (Round-5 baseline) is intentionally preserved as a historical fact about the operator's verified Round-5 state.
- **F5 retroactive closure**: §8.7 of MULTI_AGENT_REVIEW_RECORD records that F5 was DEFERRED in Round 6 and is CLOSED by G3 in Round 8. The Round-6 commit message's qualifier ("F5 deferred ... possibly premature — see Round-7 review feedback") anticipated this. No undocumented retraction.
- **Conceptual loop closure (G5)**: the F3 inline comment is now the only place a future reader will land if they're investigating a multi-pushURL question. The G1-hardening paragraph documents the empirical verification + the bypass mechanism + the `--push --all` fix in one place. Future maintainers won't have to dig through MULTI_AGENT_REVIEW_RECORD to learn why the code uses `--all`.

## Verdict

**LIGHT-PASS**.

All five Round-7 findings absorbed at the mechanism level. No regression. No leakage into LOCKED artifact. F5 retroactively closed. Test count auditable. Documentation consistent.

Net new findings during Round-8 self-review: zero.

Recommendation: commit Round 8 atomically with sign-off; LOCKED_PROTOCOL_VERSION remains `v0_pending_R1a`; R1a lock can now proceed without further peer-review absorption.
