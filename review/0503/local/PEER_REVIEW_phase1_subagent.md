# PEER REVIEW — Phase-1 checkpoint commit (light plan-diff audit)

**Reviewer**: Subagent (Explore-class) acting as senior ML pre-registration auditor
**Captured**: 2026-05-04 by main agent, in lieu of paste-back from external GPT-5.4 / Codex CLI (terminal dispatch failed due to nested-quote shell state; subagent provides equivalent independent-context review).
**Target**: HEAD `5056c09` plan-diff (REV1_TOOLING_PLAN.md §10/§11 + REV1_PLAN.md §10/§11 + new MULTI_AGENT_REVIEW_RECORD.md + portable-remote rewording)
**Prompt**: `review/0503/local/PEER_REVIEW_PROMPT_phase1_checkpoint.md`
**Verdict**: ⚠️ **LIGHT-FLAG** — 3 substantive flags + 1 minor sub-flag, no BLOCK; proceed to C1 but address before R1a.

---

## Q-A verdict

**Severity**: FLAG

**Finding**: The `<canonical>/foc_lite_hop0` portable wording is correct at the level of the plan, but the resolver semantics specified for `.review_canonical_remote` in §11.1 Q10 — substring-scan a `git remote -v` for the fragment `gitee.com:jqu9/PET_LatentResidual` — has three silent-failure / ambiguity modes that are not addressed: (1) **scheme mismatch**: the fragment contains a colon and therefore only matches the ssh URL form `git@gitee.com:jqu9/...`; an https remote `https://gitee.com/jqu9/PET_LatentResidual.git` (uses `/` after host, not `:`) would *not* match and the resolver would return zero hits even though the remote points at the canonical repo; (2) **prefix-overlap silent match**: the fragment is unanchored, so a sibling/fork remote named `gitee.com:jqu9/PET_LatentResidual_fork.git` *contains* the fragment as a substring and would silently match alongside (or in place of) the canonical; (3) **residual inconsistency with §9.2 S7**, which previously adopted `.lower().endswith("/pet_latentresidual.git")` — this contradicts the §11.1 Q10 substring-on-`gitee.com:jqu9/...` approach and the plan never reconciles them. Combined, these create a real path where C5b's `require_lock_pass()` either passes against the wrong remote (case 2) or refuses on a perfectly valid https setup (case 1).

**Evidence**: `REV1_TOOLING_PLAN.md` §11.1 Q10 (line 530), §10.6 Q4/Q7 + operational consequences #4/#5 (lines 499, 502, 509–510); `REV1_TOOLING_PLAN.md` §9.2 S7 row (in §9.2 single-agent table, "fork-rename robustness"); `OPERATOR_REPLY_pre_C1_20260503.md` Q10 (lines 35–47).

**Action**: Tighten the C2 resolver spec before C2 commits. Pseudo-diff against §11.1 Q10:

```
- `.review_canonical_remote` content: `gitee.com:jqu9/PET_LatentResidual`
- Resolver: return remotes whose URL contains the fragment.
+ `.review_canonical_remote` content: `gitee.com[:/]jqu9/PET_LatentResidual.git$`
+ Resolver:
+   1. parse each `git remote -v` URL; normalize ssh `host:path` ↔ https `host/path`
+      to a canonical `host/path` form, lowercased
+   2. anchor match with trailing `.git` (or `\b`) so `_fork.git` does not match
+   3. if 0 matches → exit non-zero with actionable add-remote hint
+   4. if ≥2 matches → exit non-zero listing all matches and require explicit --remote
```

Also add a one-line note in `MULTI_AGENT_REVIEW_RECORD.md` §3.2 Op-flag-1 row that S7's `.endswith()` approach is *subsumed*, not replaced, by the §11.1 Q10 wording.

---

## Q-B verdict

**Severity**: PASS

**Finding**: Walking the chain `5056c09` → C1 → C2 → C3 → C4 → C5 → C5b → C5c → C5d → C5e → MULTI_AGENT_REVIEW_RECORD.md → R1a → A_main → R2 → A_seed=43 → R1b → C_uniform: dependencies are consistent. C5e and C5b touch the *same file* (`run_ablation.sh`) but disjoint sections — C5b adds `require_lock_pass()` inside `case "C")`, C5e refreshes the help-text block per Op-flag-4. Per §11.1 Op-flag-4 the C5e change is documentation-only with no behavior change, so the C5b hard-gate's semantics are not invalidated. One minor taxonomy bend: §10.3 declares "C-prefix = pure tooling, R-prefix = protocol commits", but R1a and R1b each step `LOCKED_PROTOCOL_VERSION` inside `lock_effect_size_threshold.py`, i.e. R-commits also touch script source. Internally consistent (S12) but should be acknowledged.

**Evidence**: `REV1_TOOLING_PLAN.md` §10.3 commit-chain box (lines ~390–435), §11.3 chain (lines ~556–567), §10.6 row "Q7 run_ablation.sh hard-gate strictness" (line 502), §11.1 Op-flag-4 row (line 544), S12 description.

**Action**: No structural action required. Optional one-line clarification in §10.3 caption: "Note: R1a / R1b are protocol-tier commits but include a single tooling-side line edit (`LOCKED_PROTOCOL_VERSION`); they remain R-prefix because their *substantive* effect is in `POST_V6_NEXT_STEPS.md` and `EFFECT_SIZE_LOCKED.md`."

---

## Q-C verdict

**Severity**: FLAG

**Finding**: I read the first 10 rows of `B_sanity_metrics_val50_schema_reference.jsonl`. **Schema check passes**: every row has `event="val"`, `step` (integer, *not* `global_step`), `val_chain_normal_mse` (numeric), and `val_select_score` (numeric) — exactly the keys the C1 `_metrics_compat.py` golden-trace test needs to assert against. Steps observed: 400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000 — monotone +400. Comparing against `A_sanity_metrics_tail5_schema_reference.jsonl` (the 5-row tail fixture) the keys are identical, so the "two-fixture" claim is structurally valid: A_tail5 covers the late-window range `[18400, 20000]`, B_full46 covers the early/mid range `[400, 18400]`. **Three caveats remain**: (i) **filename-vs-content mismatch creates a refresh race**: when B reaches 20K, operator pledged to refresh the file to ~50 rows; if C1's golden-trace test embeds row-count-dependent assertions (e.g. expected paired_CV value computed over the 46-row contents) the refresh will silently break the test. (ii) **NaN values present**: every val row has `val_chain_normal_raw_mse: NaN` (and several other `_raw_mse` fields). Python `json.loads` accepts this; `orjson` / strict JSON parsers do not. (iii) Cross-row averaging tests will be sensitive to row subset because of training-loss decay (~3 orders of magnitude across the trace).

**Evidence**: `B_sanity_metrics_val50_schema_reference.jsonl` rows 1–10 (read directly); `OPERATOR_REPLY_pre_C1_20260503.md` Q9; `REV1_TOOLING_PLAN.md` §11.1 Q9 row (line 526); `current_experiment_architecture_status_20260503.md` §3 B_sanity row.

**Action**: In C1's design (before the commit lands), pin the B-fixture invocation to a **stable prefix window** so the file refresh is a no-op for the test:

```
- # C1 verification (B fixture)
- python lock_effect_size_threshold.py --metrics-a B_sanity_metrics_val50_schema_reference.jsonl ...
+ # C1 verification (B fixture) — refresh-stable prefix
+ python lock_effect_size_threshold.py --metrics-a B_sanity_metrics_val50_schema_reference.jsonl \
+   --step-min 400 --step-max 18000 --no-commit --output /tmp/test_lock_B.md
+ # Asserts on (val_chain_normal_mse, val_select_score) over rows 1..45 only;
+ # rows 46..50 (added on B completion) are outside the window and do not affect assertion.
```

And add an explicit comment in `_metrics_compat.py` near the JSON load: `# json.loads accepts NaN by default; do not switch to orjson without escaping these values`.

---

## Q-D verdict

**Severity**: FLAG

**Finding**: Structurally `MULTI_AGENT_REVIEW_RECORD.md` is well-organized: §1 (round-1 verbatim cross-table with adopt column + on-disk path index), §2 (round-2 F1–F12 + S1–S17 with adopt/lands-in column), §3 (round-2.5 operator answers + 5 flags), §4 (aggregate metrics). For round-1, the verbatim sources (`PEER_REVIEW_GPT55.md`, `PEER_REVIEW_CLAUDE.md`) live on disk and are correctly cross-referenced. **Two real gaps undermine the "primary-source audit trail" framing R1a §6.6.8 wants to lean on**: (1) **Round-2 internal subagents are opaque**. The four reviewers are referred to as `agent1` / `agent2` / `agent3` / `agent4` "(Explore-class)" with no further provenance — model identity, prompt, whether the four sessions were run independently or shared context. There is no transcript file on disk for these four sessions; §2.1 / §2.2 contain *paraphrased* findings only. A skeptical pre-reg reviewer can credibly claim "your four-agent convergence is just one model's output sampled four times under correlated context." (2) **Plan promised verbatim, file delivers paraphrased**. `REV1_TOOLING_PLAN.md` §10.6 Q3 row literally states "MULTI_AGENT_REVIEW_RECORD.md ... ~150–200 lines, captures **verbatim findings** of agent1/2/3/4" — but §2.1 / §2.2 of the actual file are summary tables, not verbatim. (3) Adopt/defer/reject judgments are traceable to *section-level* refs but not line numbers.

**Evidence**: `MULTI_AGENT_REVIEW_RECORD.md` lines 9–22 (path-index table — round-2 internal subagent transcripts not listed); §2.1 lines 70–85; §2.2 lines 90–110; `REV1_TOOLING_PLAN.md` §10.6 Q3 row (line ~480, "captures verbatim findings of agent1/2/3/4 across both rounds").

**Action**: Either patch the record to honor its own "primary source" claim, or downgrade the claim. Pseudo-diff (downgrade — preferred since round-2 transcripts were never persisted):

```
REV1_TOOLING_PLAN.md §10.6 Q3 row:
- ~150-200 lines, captures verbatim findings of agent1/2/3/4 across both rounds
+ ~150-200 lines, captures *paraphrased* findings of agent1-4 (round-2 transcripts
+  not persisted; round-1 sources are verbatim and on-disk) across both rounds
```

And update both the record's preamble framing and R1a §6.6.8 wording from "primary-source audit trail" to "secondary digest with round-1 verbatim cross-linked, round-2 paraphrased, round-2.5 verbatim cross-linked".

---

## Q-E verdict

**Severity**: FLAG (one minor sub-finding; no new blocker)

**Finding**: Phase-1's three artifacts introduce no *new* desk-reject vulnerabilities. **(E.1) §11.5 self-correction (λ_roll: 0 → 4.0)**: PASS — benign in-plan correction openly acknowledged with the audit-trail note. The corrected claim ("λ_roll holds constant at 4.0 in Phase III") is consistent with `A_control.yaml` lines 22–25 comments, and the locked-window decision (90K) is unchanged because the rationale rests on schedule *stationarity*. **(E.2) C5e on `run_ablation.sh` after C5b lock-gate**: PASS — help-text-only, disjoint from C5b's body, no behavior change. **(E.3) Stale `gitee/foc_lite_hop0` references**: minor FLAG — Phase-1 cleaned up §10.6 of both plans per its scope, but `review/0503/local/OPERATOR_QUESTIONS.md` line 59 still contains the literal: "检查这个 lock 文件的 git commit 是否在远端 `gitee/foc_lite_hop0` 上 → 否则 exit 2". A future operator reaching for that file when implementing R1b could paste a non-portable command.

**Evidence**: `REV1_TOOLING_PLAN.md` §11.5; `A_control.yaml` lines 22–25; `REV1_TOOLING_PLAN.md` §11.3 commit-chain box C5e row; `review/0503/local/OPERATOR_QUESTIONS.md` line 59.

**Action**: One-line edit:

```
review/0503/local/OPERATOR_QUESTIONS.md L59:
- 2. 检查这个 lock 文件的 git commit 是否在远端 `gitee/foc_lite_hop0` 上 → 否则 exit 2
+ 2. 检查这个 lock 文件的 git commit 是否在远端 `<canonical>/foc_lite_hop0` 上 → 否则 exit 2
+    （`<canonical>` 由 `.review_canonical_remote` 解析，详见 `REV1_TOOLING_PLAN.md §11.1 Q10`）
```

---

## Overall verdict

**LIGHT-FLAG — 3 substantive flags + 1 minor sub-flag, no BLOCK; proceed to C1 but address before R1a.**

Flags to resolve before R1a lands:

1. **Q-A**: Tighten the C2 resolver spec to anchor the URL fragment with a `.git` boundary, normalize ssh↔https, and explicitly handle 0-match / ≥2-match cases. Land as a clarification inside §11.1 Q10 *before* C2 is implemented.
2. **Q-C**: Pin C1's B-fixture verification to a stable prefix window (`--step-min 400 --step-max 18000`) so the post-B-completion file refresh is a no-op for the golden-trace test. Land as a one-line edit to `REV1_TOOLING_PLAN.md` §4 verification table for C1.
3. **Q-D**: Reconcile the "verbatim" promise vs the paraphrased delivery in `MULTI_AGENT_REVIEW_RECORD.md` — preferred: downgrade the §10.6 Q3 wording and the R1a §6.6.8 framing from "primary source" to "secondary digest". Land before R1a.
4. **Q-E.3** (minor): scrub the stale `gitee/foc_lite_hop0` in `OPERATOR_QUESTIONS.md` L59.

Q-B is clean. C1 implementation can begin immediately; the four flags above are addressable as small in-plan edits during the C1–C5e window.
