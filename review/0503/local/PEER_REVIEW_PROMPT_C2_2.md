# PEER REVIEW PROMPT — C2.2 polish commit (code + architecture audit)

**Reviewer role**: Senior software engineer / staff-level pre-registration tooling auditor. Treat this as a hostile code review of a critical pre-registration script — a silent bug here invalidates the σ-normalize ablation timestamp evidence for a NeurIPS/ICML submission.
**Reviewing model recommendation**: GPT-5.4 xhigh via codex, OR Gemini-2.5-Pro xhigh, OR Claude Opus 4.7 xhigh. **Run at least two reviewers and intersect the findings.**
**Scope**: ONLY the source diff between commits `8a0757a` (C2.1) and `c01aa27` (C2.2 HEAD). NOT the protocol / statistical layer (already reviewed in earlier rounds).
**Time budget**: 30–45 minutes of reviewer reasoning. This is a code+architecture audit, not a statistical audit.

---

## Context (read before the questions)

This is the σ-normalize ablation pre-registration tooling for the PET Latent-Residual paper. The lock script `review/0502/scripts/lock_effect_size_threshold.py` is the one piece of code whose correctness directly underwrites the public timestamp evidence for the blinded analysis. Earlier rounds:

| Round | Output | Reference |
|---|---|---|
| **Round 1** (external paste-back) | 2 desk-reject blockers (statistical) → fixed in REV1_PLAN.md | [`PEER_REVIEW_GPT55.md`](../operator/PEER_REVIEW_GPT55.md) + [`PEER_REVIEW_CLAUDE.md`](../operator/PEER_REVIEW_CLAUDE.md) |
| **Round 2** (4 internal subagents on plan v0.2) | F1–F12 + S1–S17 → adopted into REV1_PLAN.md §10 | [`MULTI_AGENT_REVIEW_RECORD.md`](MULTI_AGENT_REVIEW_RECORD.md) |
| **Round 2.5** (operator) | 7 answers + 5 unsolicited flags → §11 absorption + new C5e commit | [`OPERATOR_REPLY_pre_C1_20260503.md`](../operator/OPERATOR_REPLY_pre_C1_20260503.md) |
| **Round 3** (Phase-1 checkpoint audit) | 5-question light audit → LIGHT-PASS, no blockers | [`PEER_REVIEW_PROMPT_phase1_checkpoint.md`](PEER_REVIEW_PROMPT_phase1_checkpoint.md) |
| **Round 4** (mid-impl 4-lane subagent peer review on C1+C2 stack) | 12 findings (4 BLOCKER+HIGH absorbed in C2.1 = `8a0757a`; 6 MEDIUM absorbed in C2.2 = `c01aa27`) | this prompt |

**You are Round 5: an external audit of the C2.2 polish commit itself.** The hypothesis under test: did the 6 MEDIUM polish fixes (A2 / A3 / B2a / B3b / B4 / B10) actually fix what they claim, AND did the architectural change (A2) introduce any new bug that the test suite does not catch?

**Commit chain**:

```
2288625  C1   step/global_step compat + LOCKED_PROTOCOL_VERSION + SHA race fix + Guard-5 dual-key
6198579  C2   canonical-remote resolver via .review_canonical_remote anchor
8a0757a  C2.1 (HOTFIX) BLOCKER+HIGH from 4-lane review: B5 / B3a / B6 / A5 / A6
c01aa27  C2.2 (POLISH) MEDIUM from 4-lane review: A2 / A3 / B2a / B3b / B4 / B10  ← REVIEW THIS
```

**Status as of this prompt**:
- HEAD is `c01aa27` on branch `foc_lite_hop0`.
- Test suite: `test_metrics_compat.py` 16/16 + `test_remote_resolver.py` 42/42 all green locally on macOS Python 3.11.
- LOCKED_PROTOCOL_VERSION = `v0_pending_R1a` (unchanged across C1→C2.2; protocol not modified).
- No `EFFECT_SIZE_LOCKED.md` artifact has been produced yet by this branch.
- **Repository / branch / remote topology is OUT OF SCOPE.** This audit is purely about source-code correctness and architecture of the four touched files. Do NOT comment on which remote a commit lives on, whether the commit is reachable from any specific remote, when or where it will be merged, or any branch-management decision.

---

## Files to read (in this exact order, all relative to repo root)

### Must read in full

1. **The diff itself**: `git show c01aa27` — focuses your attention on exactly what changed.
2. `review/0502/scripts/lock_effect_size_threshold.py` (1102 lines) — the lock script. **Pay attention to**:
   - `_normalize_remote_url` (B2a fix; fragment/query strip)
   - `_parse_git_remote_v` (B4 rewrite; rsplit-from-right + case-insensitive marker + CRLF tolerance)
   - `_safe_repr_url` (B10 new helper)
   - `_load_anchor` (B3b strict single-line; raises on multi-line)
   - `compute_paired_stats_from_rows` (A2 new pure helper)
   - `compute_paired_cv` (A2 backward-compat wrapper)
   - `main()` from line ~760 down — A2 single-read refactor + Guard 5 reorder
3. `review/0502/scripts/_metrics_compat.py` (85 lines) — A3 docstring caveat on `get_row_step`.
4. `review/0502/scripts/test_metrics_compat.py` (425 lines) — focus on T15 / T16 (new in C2.2) and T13 / T14 (preserved from C2.1).
5. `review/0502/scripts/test_remote_resolver.py` (541 lines) — focus on R5b / R5c / R5d / R11 / R11b / R15 / R16 / TestSafeReprUrl ×4 (new or rewritten in C2.2).

### Skim for invariants only

6. The C2.2 commit message body (`git log -1 --format=%B c01aa27`) — declares 6 absorbed fix IDs and the test invariants the author claims are now enforced. **Use this as the contract you are auditing the code against.**

### Reference (already audited; skim only if your audit forks into them)

7. `review/0503/local/REV1_TOOLING_PLAN.md` — protocol-side context. Don't re-audit, but confirm no protocol assumption is broken by C2.2.
8. `review/0502/POST_V6_NEXT_STEPS.md` §6.6 — pre-registration spec. Confirms what the script must enforce.
9. `review/0503/operator/OPERATOR_REPLY_pre_C1_20260503.md` — operator's Op-flag-5 (data-side `val_select_score` + config-side `val_multi_objective` dual-key Guard 5).

---

## What you are auditing (7 specific questions)

### Q-A. A2 architectural refactor — single-read invariant

Earlier revisions read `metrics_a` THREE times in `main()`: inside `compute_paired_cv`, in the Guard 5 data-side recheck, and while building the markdown observation table. C2.2 collapses all three into ONE `load_metrics_with_hash()` call near the top of `main()` and threads `rows` + `metrics_hash` through. The new pure helper `compute_paired_stats_from_rows(rows, step_min, step_max) -> (cv, mean, std, n, series)` does the math without re-reading the file. T16 patches `load_metrics_with_hash` with a counter and asserts it is called exactly once on the `--no-commit` success path.

**Audit**:

1. Walk the entire `main()` body from arg parse to its final `return`. Is there ANY remaining code path that opens `metrics_a` a second time (directly or via `load_yaml` / `Path.read_text` / `json.loads`)? Specifically: are there any error / fallback branches inside Guard 1 / 2 / 3 / 4 / 5 / config-side / TOCTOU / artifact-write / shell-out blocks that still re-read it? This is a static-analysis question about source-code structure only.
2. Is `compute_paired_cv` (the backward-compat wrapper) still used anywhere besides tests? If yes, that's a hidden second read.
3. T16 only asserts call count on the `--no-commit` happy path. Are there error paths (Guard failures) where the count would still be 1? Or could a Guard 4 / Guard 5 abort path read it twice before exiting?
4. Does T15's "row-form ↔ path-form parity" actually prove the math is identical, or could floating-point summation order differ between the two paths and silently drift?

### Q-B. B3b strict single-line anchor — regression surface

`_load_anchor` previously took the first non-empty line and emitted a stderr warning on multi-line files. C2.2 now RAISES `CanonicalRemoteError` if the file contains anything other than exactly one non-empty content line. Whitespace-only lines (`"   \n"`, `"\t\n"`) do not count.

**Audit**:

1. Is the legitimate-but-edge case "single non-empty line + trailing newline" still accepted? What about CRLF (`"url\r\n"`) — does the strip handle the `\r`?
2. What does the parser do for a file with **only** a UTF-8 BOM and nothing else (`"\ufeff"` then EOF)? Empty error or multi-line error? Which is correct semantically?
3. The `len(nonempty_lines) != 1` check uses a definition of "non-empty" — exactly which whitespace classes does it strip? Is this consistent with the BOM strip in B3a (C2.1)? Concretely: if a malicious actor puts `"\ufeff\n\nurl\n"`, does the BOM line count as non-empty?
4. R11 was rewritten from "first non-empty wins → 0-match downstream raise" to "exactly-one rule → upfront raise". Does this break any legitimate user workflow? Specifically: ops who comment out previous canonical with `# old: ...` and put new on next line — that workflow is now BROKEN. Is that intended? Should the docstring say "no comment syntax supported"?

### Q-C. B4 `_parse_git_remote_v` rewrite — silent regression on real `git remote -v` output

The old parser was `parts = line.split(); name=parts[0], url=parts[1], kind=parts[2]`. The new parser is:

```python
head_split = line.split(None, 1)   # peel off remote name
url, kind = rest.rsplit(None, 1)   # peel off (fetch) marker
if kind.lower() == "(fetch)":      # case-insensitive
    ...
```

**Audit**:

1. Run `git remote -v` on a real repo with at least 3 remotes (or simulate). Does the rsplit-from-right correctly handle the standard `\t` separator git uses between name and URL, plus the space before `(fetch)`?
2. What happens if a remote name itself contains whitespace (impossible per git's own naming rules, but defensive)? Does `split(None, 1)` swallow it or fail?
3. What about lines with `(fetch)` but no remote name (malformed git output)? Old parser returned `parts[2]` index error. New parser?
4. Is "case-insensitive `(fetch)`" actually defensible, or does it open a vulnerability where a hostile remote URL containing the literal substring `(FETCH)` somewhere in its path could fool the parser? Specifically, what about `git@evil.com:fake/(FETCH).git (fetch)` — does this end up keyed as remote `git@evil.com:fake` with URL `(FETCH).git`?
5. CRLF tolerance (R5d): does the line-by-line iteration handle a file with `\r\n` everywhere? `splitlines()` is the standard answer; confirm the implementation uses it, not `split('\n')`.

### Q-D. B10 `_safe_repr_url` — print-site coverage + repr() safety

C2.2 adds `_safe_repr_url(url) -> repr(str(url))` and wraps URLs in 4 print sites: zero-match resolver listing, multi-match resolver listing, `--remote` `check_url` error, TOCTOU pre-push error, plus info prints in `main()`.

**Audit**:

1. **Coverage**: grep the file for ALL `print(` and `sys.stderr.write(` calls. Are there any sites where a URL-bearing string flows in (e.g., from the resolver, from `_load_anchor`, or from in-memory `_parse_git_remote_v` results) and is NOT wrapped in `_safe_repr_url`? This is a static-analysis question about the source code; do not extend it into runtime push behavior.
2. **repr() safety**: Python `repr()` on a string normally uses `'…'` quoting and `\\xNN` escapes for ASCII control chars. But does it escape `\u202E` (right-to-left override) and other Unicode bidi controls? Could a hostile URL string using `\u202E` flip the displayed text without `repr()` neutralizing it? If so, is `_safe_repr_url`'s implementation sufficient or does it need explicit Unicode-class filtering?
3. **Round-trip risk**: if a user copy-pastes a `repr()`-quoted URL from stderr into their shell, would they get back the original URL or a corrupted one? Is this the right tradeoff (safety > usability) for a user-facing diagnostic helper?
4. **Function-purity check**: is `_safe_repr_url` itself free of side effects, deterministic, and total (no exceptions on any `str` input including the empty string and None-coerced inputs)? If it can raise, that's a bug — diagnostics must never crash the diagnostic path.

### Q-E. B2a fragment / query strip — order of operations

`_normalize_remote_url` now strips `#fragment` and `?query` after the scheme/scp parse, before lower-case + `.git` strip.

**Audit**:

1. Walk through normalization for these adversarial inputs and confirm the output:
   - `git@gitee.com:jqu9/repo.git#frag` → ?
   - `git@gitee.com:jqu9/repo.git?token=abc` → ?
   - `https://user:pass@gitee.com/jqu9/repo.git?token=abc` → ? (does the `?` in the query interact with the `@` in the auth?)
   - `git@gitee.com:jqu9/repo.git#frag?query` (fragment THEN query) → ?
   - `git@gitee.com:jqu9/repo.git?query#frag` (query THEN fragment, RFC3986 order) → ?
   - `git@gitee.com:jqu9/repo%23weird.git` (literal `%23` percent-encoded `#`) → does it get treated as a real `#`?
2. Is the strip order `#` first then `?` correct? Or should `?` come first per RFC3986 (where `?query` precedes `#fragment`)?
3. Is "strip fragment+query" the right semantics, or should a fragment-bearing anchor be REJECTED (since it's almost certainly an editor bug)?

### Q-F. Test suite holistic — what invariants are NOT tested

C2.2 adds T15 / T16 / R5b / R5c / R5d / R15 / R16 / TestSafeReprUrl ×4 + rewrites R11 / R11b. Total: 16 + 42 = 58 tests, all passing.

**Audit**:

1. List the invariants the C2.2 commit message CLAIMS are enforced. Cross-check each against the test suite. Is any claim untested?
2. Specifically for A2: T16 asserts call count = 1 only on the `--no-commit` happy path. Are the various Guard-fail / error-exit branches inside `main()` tested for the same single-read invariant? If not, that is a coverage gap — flag it.
3. The C2.2 architectural change moves Guard 5 (data-side) ahead of Guard 4 (N<5). Is there a test that exercises the new ordering — i.e., a fixture with rows present but missing the LOCKED_METRIC_KEY entirely, asserting that the dedicated Guard-5 error message fires rather than Guard 4's generic "N<5"?
4. The `_parse_git_remote_v` and `resolve_canonical_remote` code paths interact with environment-dependent state (whatever the developer's local clone happens to contain). Are there tests that cover these functions hermetically (i.e., without depending on any specific remote being registered locally), so the suite can be re-run on any checkout? Out of scope: which remote names happen to exist on which machine — that is a deployment concern, not a code/architecture concern.

### Q-G. Source-level pre-registration invariants — does the C2.2 polish change anything observable

The lock script's role is to emit a deterministic `EFFECT_SIZE_LOCKED.md` artifact from a fixed input. C2.2 is supposed to be code-internal polish only — no behavioral change from the perspective of an external auditor reading the artifact.

**Audit (source-level only — repository / remote / branch topology is out of scope)**:

1. Has any LOCKED constant changed in C2.2? (Confirm `LOCKED_METRIC_KEY`, `LOCKED_FLOOR`, `LOCKED_SLOPE`, `LOCKED_RECOMMENDED_WINDOW`, `LOCKED_PROTOCOL_VERSION` are byte-identical to C2.1.)
2. Has the format of `EFFECT_SIZE_LOCKED.md` changed in any observable way? (e.g., did the markdown observation table re-order, change column count, or alter SHA-content under the A2 row-reuse refactor?) For an arm with N=45 in-window observations, would C2.1 and C2.2 produce byte-identical lock markdown given the same input file?
3. Does the C2.2 source diff or the new public-facing strings (error messages, help text, info prints) introduce any retroactive claim about the σ-normalize PROTOCOL itself, vs claims about the TOOLING that observes it? Confirm only tooling claims.
4. Does `render_lock_md` consume any state that was changed in C2.2 (e.g., `metrics_hash`, `series`, `n`, `mean`, `std`, `paired_cv`)? If any of these now arrive via a different code path than in C2.1, is the value bit-identical for the same input?

---

## Output format

For each of Q-A through Q-G, please provide:

```
## Q-X verdict

**Severity**: PASS / FLAG / BLOCK

**Finding**: <2-4 sentences. Be specific. Cite file:line.>

**Evidence**: <line numbers in the audited files; quote the offending or load-bearing code if helpful>

**Action**: <if FLAG/BLOCK: minimal patch in pseudo-diff form; if PASS: "no action needed">
```

After all 7 questions, end with:

```
## Overall verdict

<one of:>
- LIGHT-PASS — no source-level blockers; C2.2 code is sound as it stands.
- LIGHT-FLAG — N flags, list them; code is sound modulo these items, address them before next protocol-touching commit.
- HEAVY-FLAG — N flags including ≥1 BLOCK; the C2.2 source as-is has a correctness or architectural defect that must be fixed in a follow-up commit before any further work proceeds on top of it.
```

**Note on scope**: "safe / not safe" here refers strictly to source-code correctness and architectural integrity. Repository topology, branch management, and when/where this commit is merged or pushed are decisions outside the scope of this audit and are explicitly reserved to the operator.

```
## Top-3 latent risks not covered by Q-A..Q-G

If you spot ANY issue not falling cleanly into one of the 7 questions
(e.g., a deeper architectural concern, a test-fragility class, a
threat model the prompt missed), surface it here. Three items max,
ranked by risk.
```

---

## Reviewer ground rules

1. **Be ruthless.** We've already had 4 rounds of review, all of which found something. If you find nothing, double-check — the prior is that we missed something.
2. **Read the diff first** before forming an opinion. The C2.2 commit message body is 130 lines and lists exactly which fix went where.
3. **Run the tests.** If you can execute Python, run `python review/0502/scripts/test_metrics_compat.py` and `python review/0502/scripts/test_remote_resolver.py` and confirm 16+42 pass on your machine. If they don't, that's itself a finding.
4. **Don't re-audit C1 / C2 / C2.1.** Those landed in earlier rounds and are out of scope.
5. **Don't re-audit the statistical / protocol layer.** The σ-normalize protocol, paired_CV math, decision trichotomy, etc., are out of scope. This is a code+architecture review only.
6. **Pseudo-diff format for Action**: use unified-diff syntax (`+` / `-` / context lines) on the file:line where the patch should go. Do NOT include full files.
7. **No emoji. No marketing language. Plain technical English (or 中文 if you prefer; the project is bilingual).**

If anything in this prompt is ambiguous, state your interpretation explicitly in your response and audit under that interpretation rather than asking for clarification.

— end of prompt —
