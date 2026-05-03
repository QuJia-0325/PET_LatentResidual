# MULTI_AGENT_REVIEW_RECORD — Sigma-normalize ablation pre-registration

**Drafted**: 2026-05-03 night, between C5e and R1a (per §10.6 decision)
**Purpose**: **Secondary digest** of the multi-round review process that produced `REV1_PLAN.md` v0.3 + `REV1_TOOLING_PLAN.md` v0.3 + the C1–C5e + R1a/R1b commit chain. Round 1 + round 2.5 verbatim sources are on-disk and cross-linked from the path-index table below; round 2 internal subagent findings are paraphrased only (transcripts not persisted). Cited from R1a §6.6.8 changelog as the round-by-round revision history.
**Scope**: Reviews of v0 (`POST_V6_NEXT_STEPS.md` §6.4 + §6.6 + the two scripts) → v0.1 → v0.2 → v0.3 (operator-absorbed) → Phase-1 checkpoint (this record + plan revisions). Round 1 = external GPT-5.5 + Claude paste-back; Round 2 = internal multi-agent re-review of v0.1 / v0.2; Operator round-2.5 = `OPERATOR_REPLY_pre_C1_20260503.md` + `current_experiment_architecture_status_20260503.md`. Phase-1 light audit = `PEER_REVIEW_phase1_subagent.md`.

This file references but does not duplicate the verbatim review content already on disk. For full text:

| Source | On-disk path | Lines |
|---|---|---|
| GPT-5.5 round-1 | `review/0503/operator/PEER_REVIEW_GPT55.md` | full file |
| Claude round-1 | `review/0503/operator/PEER_REVIEW_CLAUDE.md` | full file |
| Operator architecture audit (round-2.5) | `review/0503/operator/overall_design_experiment_matrix_deep_review_20260503.md` | full file |
| Operator local-Q reply (between v0.1 and v0.2) | `review/0503/operator/OPERATOR_REPLY_local_questions_20260503.md` | full file |
| Operator pre-C1 reply (round-2.5) | `review/0503/operator/OPERATOR_REPLY_pre_C1_20260503.md` | full file |
| Operator GPU/run audit (round-2.5) | `review/0503/operator/current_experiment_architecture_status_20260503.md` | full file |
| Round-1 absorption | `REV1_TOOLING_PLAN.md §9` + `REV1_PLAN.md §9` | full sections |
| Round-2 absorption | `REV1_TOOLING_PLAN.md §10` + `REV1_PLAN.md §10` | full sections |
| Round-2.5 absorption | `REV1_TOOLING_PLAN.md §11` + `REV1_PLAN.md §11` | full sections |

---

## 1. Round 1 — External reviewers (paste-back from external sessions)

### 1.1 GPT-5.5 xhigh — verdict: 🛑 DESK REJECT

Headline findings (full text in `PEER_REVIEW_GPT55.md`):

| ID | Finding (paraphrased; verbatim text in source) | Adopt/defer/reject | Lands in |
|---|---|---|---|
| GPT-Q1 | Single-arm `paired_CV_A` is wrong noise estimator for between-arm equivalence claim. 10% floor is unjustified domain margin. | **Adopt** | REV1_PLAN.md §3 REV1.1 (paired_CV_A → paired_SD_AA on val_chain_normal_mse) + REV1.2 (drop 10% floor) |
| GPT-Q2 | Sequence not blind. `metrics.jsonl` whitelist + WandB/TensorBoard streaming + lock-script artifact glob is underinclusive. C trajectory leakage explicitly allowed. | **Adopt** | REV1_TOOLING_PLAN.md §1 Tier 1 C1 (full-tree artifact scan) + Tier 2 C5 (wrapper enforcement) |
| GPT-Q3 | Trichotomy maps `rel_diff < X` (including large negative) to "C ≈ A" — V6-favorable HARK. | **Adopt** | REV1_PLAN.md §3 REV1.4 (decision rule on `\|rel_diff\|` + explicit `rel_diff ≤ −X` branch) |
| GPT-Q4 | AR(1) lag-1 correction may be insufficient when residuals have higher-order autocorrelation. 0.10 threshold is arbitrary. A_pair_uniform_spot may not have been run. | **Adopt** | REV1_TOOLING_PLAN.md §1 Tier 2 C5 (paired_diff_judge AR(p) upgrade — deferred to R1) + REV1_PLAN.md §11.1 Q11 (A_pair rerun = option a) |
| GPT-Q5 | No ceiling number → no near-ceiling/headroom claim. If computed, must be auxiliary anchor only, must not change X. | **Adopt** | REV1_TOOLING_PLAN.md §1 Tier 3 C13 (ceiling-string ban Guard 6) + R1a §6.6.5 ceiling-as-anchor-only clause |
| GPT-Q6 | SHA256 race in `compute_paired_cv` (read at T₀, hash at T₁). Guard 5 only checks `best_metric`, not full run-dir verification. Tamper-resistance requires protected branch + external timestamp. | **Adopt (race + Guard 5); operator-resolved (timestamp)** | C1 implementation (read-once-then-hash); C12 Guard 5 dual-key (config + data per Op-flag-5); §10.6 Q4 plain-commit + canonical-remote push as time anchor |
| GPT-Q7 | 3 attacks (live-peek, side-channel artifact, 200K continuation re-lock). | **Adopt** | All three patches absorbed into C5 wrapper + R1a §6.6.3 single-shot 200K rule |
| GPT-Q8 | Verdict 4/3/5/3, top fix = lock margin pre-observation + wrapper enforcement + paired/block-bootstrap inference. | **Drives v0 → v0.1 rewrite** | Whole REV1_PLAN.md/REV1_TOOLING_PLAN.md effort |

### 1.2 Claude Opus 4.7 xhigh — verdict: ⚠️ MAJOR REVISIONS (desk-reject risk if Q1+Q3 not fixed)

Headline findings (full text in `PEER_REVIEW_CLAUDE.md`):

| ID | Finding (paraphrased) | Adopt/defer/reject | Lands in |
|---|---|---|---|
| Cl-Q1 | Three dimensional flaws: (a) wrong noise estimator (within-arm CV ≠ between-arm SD), (b) 3σ ≠ 3·SE, (c) floor incoherent with slope. | **Adopt all three** | REV1_PLAN.md §3 REV1.1 + §3 REV1.2 + R1a §6.6.1.1 rationale |
| Cl-Q2 | Leak paths: `metrics.jsonl` explicit whitelist (line 219), WandB/TB streaming, window cherry-pick if A already past 40K when written, Method-D criterion drift, force-push amnesia. Concrete missed-files list given. | **Adopt all** | REV1_TOOLING_PLAN.md §1 Tier 1 C1 + Tier 2 C5 + Tier 3 C12 + §10.1 F11 hard gate |
| Cl-Q3 | Negative `rel_diff` falls into V6-favorable bucket. **Two desk-reject blockers** (Q1 + Q3). 200K continuation underspecified. 2X boundary chosen for prose. | **Adopt** | REV1_PLAN.md §3 REV1.4 (trichotomy with `\|rel_diff\|`) |
| Cl-Q4 | AR(1) wrong model for non-stationary mean (training trend) → `N_eff` can be 2.7× overstated → false `no_confound` verdict. 0.10 has two contradictory rationales (defensive in §6.4, floor in §6.6). A_pair_uniform_spot may not exist. | **Adopt** | REV1_PLAN.md §3 REV1.5 (paired_diff_judge AR(p) upgrade) + §11.1 Q11 (A_pair rerun) |
| Cl-Q5 | Without ceiling, no "near-ceiling" / "headroom" / "approaches bound" language. If computed later, separate column only — never retroactively renormalize `rel_diff`. | **Adopt** | REV1_TOOLING_PLAN.md §1 Tier 3 C13 ceiling-string ban + R1a §6.6.5 |
| Cl-Q6 | Concrete bugs: (a) SHA256 race, (b) inclusive window OK, (c) float compare benign, (d) Guard 3 metrics.jsonl whitelist hole, (e) Guard 5 too loose, (f) tamper-resistance conventional only — needs GPG / OTS / immutable mirror. | **Adopt (a)(d)(e); operator-resolved (f)** | C1 (SHA race fix) + Tier 1 C1 full-tree scan + C12 Guard 5 dual-key + §10.6 Q4 plain-commit + canonical-remote push |
| Cl-Q7 | 5 attacks (window cherry-pick, live C peek, Method-D drift, manual lock-doc edit, early-stop A_main at favorable point). | **Adopt all** | C1 lock-script fixes + C5 wrapper + C12 Guard 5 + R1a Method-D parameter lock + N_min check |
| Cl-Q8 | Verdict 4/3/5/3, top fix = `\|rel_diff\|` + explicit `rel_diff < −X` branch + replace single-arm CV with two-seed paired SD. | **Adopt — drives R1a structure** | R1a §6.6.1 + §6.6.3 + new A_seed=43 yaml |

**Round 1 net effect**: v0 protocol has at least 2 desk-reject blockers (paired_CV_A wrong estimator; trichotomy negative-`rel_diff` HARK) + ~6 leak paths + ~5 implementation bugs. Both reviewers converge on these. Drives `REV1_PLAN.md v0` (statistical) + `REV1_TOOLING_PLAN.md v0` (tooling).

---

## 2. Round 2 — Internal multi-agent re-review of v0.1 / v0.2

After v0.1 was drafted absorbing round-1 findings, four subagents (Explore-class) re-reviewed v0.1 and v0.2 in two passes. The structured F-coded + S-coded findings live in `REV1_TOOLING_PLAN.md §10.1` and `§10.2`. Cross-references below; full bullet lists in those sections.

### 2.0 Round-2 subagent provenance + auditability gap

**Provenance**: `agent1`–`agent4` were Explore-class subagents invoked from the main agent's chat session on 2026-05-03; each received the same v0.1 / v0.2 plan files plus repository read-only context; sessions ran independently (no shared context between them). The exact underlying model and the per-session prompts are not preserved on disk.

**Auditability gap (acknowledged per Phase-1 review Q-D, `PEER_REVIEW_phase1_subagent.md`)**: Round-2 subagent transcripts were **not persisted**. §2.1 / §2.2 below are *paraphrased* digests of the F1–F12 + S1–S17 findings as I (the main agent) recorded them when absorbing each finding into §10 of `REV1_TOOLING_PLAN.md`. A skeptical pre-registration reviewer cannot independently verify that four sessions converged on F-prefix items vs one session sampled four times; the F vs S distinction in this record reflects my labeling at absorption time, not transcript-level evidence. R1a §6.6.8 cites this record as a **secondary digest**, not a primary source, for round 2. Round 1 (`PEER_REVIEW_GPT55.md` + `PEER_REVIEW_CLAUDE.md`) and round 2.5 (`OPERATOR_REPLY_pre_C1_20260503.md`) primary sources remain on disk and are cited verbatim in R1a §6.6.8.

### 2.1 Round-2 convergent findings (all 4 agents agreed) — F-prefix

| Code | Headline (full text in `REV1_TOOLING_PLAN.md §10.1`) | Adopt/defer/reject | Lands in |
|---|---|---|---|
| F8 | C1 schema-compat fix needs golden-trace test, not just code change. Operator's tail-5 jsonl is the fixture. | **Adopt** | C1 commit (added test using `A_sanity_metrics_tail5_schema_reference.jsonl`; operator round-2.5 added `B_sanity_metrics_val50_schema_reference.jsonl` for stronger coverage, 46 rows) |
| F9 | A_pair_uniform_spot has `max_steps=60000` — phase boundaries do not align with A_main `[0,30K]/[30K,90K]/[90K,120K]`. Same step ≠ same phase. Cannot do same-step paired. | **Adopt — option (a)** per Q11 | New `A_pair_uniform_spot_aligned.yaml` in R1a, max_steps=120000 |
| F10 | R1 single commit is itself the "lock formula after seeing A_seed=43 data" anti-pattern. Must split into R1a (skeleton, before data) + R1b (numeric, after data). | **Adopt** | §10.3 R1a/R1b split |
| F11 | Operator manual precheck of lock doc is convention, not enforcement. `run_ablation.sh case "C")` needs hard-gate `require_lock_pass()` (file + on canonical remote + protocol version). | **Adopt** | C5b commit |
| F12 | paired_SD_AA metric is implicit. Must be `val_chain_normal_mse` (single-objective) not `val_select_score` (Method-D composite) — otherwise re-introduces v0's dimensional inconsistency at smaller scale. | **Adopt** | R1a §6.6.1 explicit `paired_SD_AA on val_chain_normal_mse` |

### 2.2 Round-2 single-agent or split-vote findings — S-prefix

| Code | Headline (full text in `REV1_TOOLING_PLAN.md §10.2`) | Adopt/defer/reject | Lands in |
|---|---|---|---|
| S1 | C2 needs URL-fragment resolver, not hard-coded remote name. | **Adopt** | C2 + `.review_canonical_remote` |
| S4 | C5 manifest filename should embed `<lock_sha8>` for traceability. | **Adopt** | C5 wrapper |
| S7 | Operator-side remote may not be named `gitee` (later confirmed: `origin`). C2 must scan via URL pattern. | **Adopt** | Same as S1 |
| S8 | C1 helper `get_row_step(row)` should be a separate module (`_metrics_compat.py`), not a one-line lambda. | **Adopt** | C1 new file |
| S10 | C3 must probe `last.pt`/`best.pt` (no `ckpt_` prefix), not just `ckpt_last.pt`. | **Adopt** | C3 commit (operator round-2.5 confirms in Op-flag-3) |
| S11 | Lock script needs Guard 6: refuse lock docs with `ceiling`/`headroom`/`52.6341` strings. Defends Claude Cl-Q5 attack. | **Adopt** | C5d commit |
| S12 | Lock script must declare `LOCKED_PROTOCOL_VERSION` constant for hard-gate version check. | **Adopt** | C1 commit (lands as `"v0_pending_R1a"`; R1b script-edit commit changes to `"R1b_locked"`) |
| S13 | New script `verify_paired_seed_configs.py` to assert A_seed=42 vs A_seed=43 yaml differ only in seed/run_name/output_dir. | **Adopt** | C5c commit |
| S14 | R2 (REV1.1/REV1.2 hardening) is agent-only work; can land parallel with A_main 120K training. Saves ~1 GPU·day on critical path. | **Adopt** | §10.3 chain (R2 parallel) |
| S15 | §8 menu text in REV1_PLAN.md is out of date wrt §10.3 commit chain. | **Adopt** | §10.4 overlay |
| S16 | Window 90K vs 95K decision: math verified. EMA half-life ≈ 6931 steps. λ_roll holds constant 4.0 in Phase III (not 0; corrected from earlier draft) → schedule contributes zero non-stationarity (stationary-at-4.0). 95K imposes 10% SE penalty (62 vs 75 eval points). **Recommend 90K**. | **Adopt** | §10.6 frozen at 90K |
| S17 | §9.4 cost table missing line for sanity B completion (~0.1–0.2 GPU·day). | **Adopt** | §10.3 cost table updated |

### 2.3 Round-2 deferred / rejected items

None of the round-2 findings were rejected. Two items were initially flagged for round-3 but resolved by operator round-2.5:

- **paired_diff_judge AR(p) upgrade** (extension of GPT-Q4 + Cl-Q4): deferred to R2 hardening commit, not blocking C1.
- **Multi-comparison correction across A/B/C/D × multiple metrics**: noted in R2 hardening commit; not in R1a's primary claim path.

---

## 3. Operator round-2.5 — Operator audit + reply + 5 unsolicited flags

After v0.2 (containing §10 round-2 absorption) was pushed to canonical remote at commit `23a11d4`, operator on host responded at commit `2d3ae5f` with three artifacts. Full text on disk; structured absorption in `REV1_TOOLING_PLAN.md §11` + `REV1_PLAN.md §11`.

### 3.1 Operator's 7 answers (Q9–Q15)

Cross-reference: `REV1_TOOLING_PLAN.md §11.1` table is the single source of truth. Adopt/defer/reject summary:

| Q | Adopt/defer/reject | Effect |
|---|---|---|
| Q9 sanity B state + path | **Adopt** | C1 fixture upgraded to two-fixture (A_tail5 + B_full46) |
| Q10 canonical-remote = URL fragment | **Adopt** | `.review_canonical_remote` literally contains `gitee.com:jqu9/PET_LatentResidual` |
| Q11 A_pair scope (option a-with-cleanup) | **Adopt** | New `A_pair_uniform_spot_aligned.yaml` in R1a |
| Q12 A_main launch via `run_ablation.sh A` | **Adopt** | R1a §6.6.1 reproducibility recipe uses launcher |
| Q13 A_seed=43 yaml deltas locked (3 fields) | **Adopt** | C5c verifies exactly these allowed deltas |
| Q14 GPU all occupied | **Adopt — re-anchor cost-table start time** | Wall-clock anchor = "first-free-GPU + R1a", not "now" |
| Q15 No deadline | **Adopt** | A_pair option (a) stands |

### 3.2 Operator's 5 unsolicited flags

| Flag | Adopt/defer/reject | Effect |
|---|---|---|
| Op-flag-1 portable remote | **Adopt** | §10.6 + §11.1 Q4/Q7 + consequences #4/#5 reworded to use `<canonical>/foc_lite_hop0` placeholders |
| Op-flag-2 C1/T1 unpatched | **Adopt — confirms C1 design** | C1 implementation (pending) |
| Op-flag-3 C3/T3 wrong glob + wrong default-key chain | **Adopt — confirms C3 design** | C3 implementation (pending) |
| Op-flag-4 `run_ablation.sh` help text stale | **Adopt — new C5e** | C5e commit added between C5d and [B] approve |
| Op-flag-5 Guard 5 dual-key wording | **Adopt — already split in v0.1** | Reaffirms C12 split: `best_metric == "val_multi_objective"` (config) AND `"val_select_score" in metrics_row` (data) |

---

## 4. Aggregate audit-trail summary

| Metric | Round 1 | Round 2 | Round 2.5 | Round 4 | Round 5 | Total |
|---|---:|---:|---:|---:|---:|---:|
| External / operator reviewers | 2 (GPT-5.5, Claude) | 0 | 1 (operator) | 0 | 2 (agent1 + agent2 paste-back) | 5 |
| Internal subagent reviewers | 0 | 4 | 0 | 4 (mid-impl Lanes A/B/C/D) | 0 | 8 |
| Findings adopted | 16 (8 GPT + 8 Claude) | 17 (5 F + 12 S) | 12 (7 Q + 5 flags) | 10 (4 BLOCKER+HIGH in C2.1; 6 MEDIUM in C2.2) | 2 (D1 + E1, C2.3 in progress) | 57 |
| Findings deferred | 0 | 2 (AR(p), multi-comp) | 0 | 0 | 5 (F1, F2, B1, L1, L2) | 7 |
| Findings rejected | 0 | 0 | 0 | 0 | 2 (BOM-internal edge; PyYAML-env) | 2 |
| Plan-version bumps caused | v0 → v0.1 | v0.1 → v0.2, v0.2 → v0.3 | v0.3 → v0.3 + §11 | none — code-level only | none — code-level only | 4 plan versions |
| New commits added to chain | C1–C5 (5) | C5b, C5c, C5d (3) + R1 split into R1a + R1b | C5e (1) | C2.1 (8a0757a), C2.2 (c01aa27) | C2.3 (in progress) | 14 commits |

R1a §6.6.8 changelog should cite, in this order:

1. `review/0503/operator/PEER_REVIEW_GPT55.md` (round-1 external desk-reject finding)
2. `review/0503/operator/PEER_REVIEW_CLAUDE.md` (round-1 external major-revision finding)
3. `review/0503/local/MULTI_AGENT_REVIEW_RECORD.md` (this file — round-1 + round-2 + round-2.5 + round-4 + round-5 audit-trail compendium)
4. `review/0503/local/REV1_PLAN.md` §11 + `REV1_TOOLING_PLAN.md` §11 (round-2.5 operator absorption)
5. `review/0503/operator/OPERATOR_REPLY_pre_C1_20260503.md` + `current_experiment_architecture_status_20260503.md` + `B_sanity_metrics_val50_schema_reference.jsonl` (round-2.5 operator-side primary sources)
6. `review/0503/local/PEER_REVIEW_PROMPT_C2_2.md` (round-5 prompt under which the C2.2 polish itself was paste-back-audited; primary record of which 7 questions were posed)

This file is intended to remain a frozen snapshot of the multi-round review state at the moment R1a is drafted. Subsequent revisions (e.g., R2 hardening, R1b lock) write their own audit-trail subsections inside R2/R1b commits, not back into this file.

---

## 5. Round 4 — Mid-implementation 4-lane subagent peer review on C1+C2 stack

### 5.1 Provenance

After the C1 (`2288625`) + C2 (`6198579`) commits landed but before any C-stack milestone (i.e., still pre-`run_ablation.sh case "C")` hard-gate), the main agent invoked four internal subagents, each scoped to a different audit lane on the live C1+C2 source tree. Sessions ran independently (no shared context) on 2026-05-04 morning.

| Lane | Scope | Output | Findings prefix |
|---|---|---|---|
| A | Metrics-side: `_metrics_compat.py` + `compute_paired_cv` + Guard 4 / Guard 5 data side | structured severity matrix | A1–A6 |
| B | Resolver-side: `_normalize_remote_url` + `_parse_git_remote_v` + `_load_anchor` + URL print sites | structured severity matrix | B1–B12 |
| C | Test-coverage gap report against the C1+C2 contract | hierarchical gap list (~25KB content excerpt preserved at chat-session-resources path) | C-prefixed |
| D | Workflow / state-machine view of `main()` end-to-end | ordered defect list | D-prefixed |

**Auditability gap (same class as §2.0)**: Round-4 subagent transcripts are not persisted on the canonical-remote tree. The 12-finding severity matrix below is the main agent's labeling at absorption time. The Lane C 25KB content excerpt is preserved at a chat-session-resources path (machine-local, ephemeral); reviewers wishing to verify the pre-absorption findings should request it directly.

### 5.2 Round-4 12-finding severity matrix

Severity classification used by main agent at absorption time:

- **BLOCKER** = pre-registration validity is at risk if shipped uncorrected (e.g., race window re-opens, hash-vs-data drift)
- **HIGH** = silent data-corruption surface; would not be caught by current tests
- **MEDIUM** = correctness-equivalent fix raising the floor (architectural cleanup, adversarial-input hardening)

| Code | Lane | Severity | Headline | Disposition | Lands in |
|---|---|---|---|---|---|
| **B5** | B | BLOCKER | TOCTOU between resolve-time and push-time URL: a `git remote set-url` between the two could redirect the lock commit to a non-canonical remote without the script noticing | C2.1 | `8a0757a` |
| **B3a** | B | BLOCKER | UTF-8 BOM at start of `.review_canonical_remote` produces a 0-match resolver error pointing at the BOM line; legitimate anchor files written from Windows / certain editors silently break | C2.1 | `8a0757a` (load via `utf-8-sig`) |
| **B6** | B | HIGH | Resolver matches anchor as substring of remote URL — `pet_latentresidual` matches `pet_latentresidual_fork`. Boundary-anchored equality required | C2.1 | `8a0757a` (`_normalize_remote_url` + strict `==`) |
| **A5** | A | HIGH | In-window rows with parseable `step` but missing or non-numeric `LOCKED_METRIC_KEY` were silently dropped; trainer partial writes would not surface to the operator | C2.1 | `8a0757a` (stderr `[warn]` line + skipped counts) |
| **A6** | A | MEDIUM (cheap) | `compute_paired_cv` returned series unstably ordered if rows arrived out of order; absorbed in same C2.1 commit because of trivial fix | C2.1 | `8a0757a` |
| **A2** | A | MEDIUM | `main()` re-opens `metrics_a` 3× (`compute_paired_cv` + Guard 5 recheck + observation table) — re-introduces the SHA-vs-data race window that C1 had explicitly closed | C2.2 | `c01aa27` (single-read invariant + new pure helper `compute_paired_stats_from_rows`) |
| **A3** | A | MEDIUM | `get_row_step` truncates non-integral `step` floats (`4.5 → 4`) silently; production rows never have non-integral steps but the docstring did not warn | C2.2 | `c01aa27` (docstring caveat in `_metrics_compat.py`) |
| **B2a** | B | MEDIUM | `_normalize_remote_url` did not strip `#fragment` / `?query` — an editor adding `#L1` to `.review_canonical_remote` silently breaks resolution | C2.2 | `c01aa27` (`#` and `?` strip before scp conversion) |
| **B3b** | B | MEDIUM | `_load_anchor` previously took the first non-empty line on multi-line files with a stderr warning; produced confusing 0-match downstream errors when the first line was a comment | C2.2 | `c01aa27` (strict single-line — raise on multi-line) |
| **B4** | B | MEDIUM | `_parse_git_remote_v` used `line.split()[0:3]` and would mishandle URLs with internal whitespace / CRLF / case-variant `(FETCH)` markers | C2.2 | `c01aa27` (rsplit-from-right + case-insensitive marker + CRLF tolerance) |
| **B10** | B | MEDIUM | Resolver error / info prints emitted untrusted URL bytes directly to stderr — a hostile URL with embedded ANSI / NUL / newline could corrupt the operator's terminal | C2.2 | `c01aa27` (new `_safe_repr_url` helper + 5 wrap sites) |
| **B7** (deferred) | B | LOW | PowerShell / Windows path handling for the resolver — out of scope until a Windows operator is added | DEFERRED post-merge | not in C2.x |

C2.1 (8a0757a) commit message lists B5 / B3a / B6 / A5 / A6.
C2.2 (c01aa27) commit message lists A2 / A3 / B2a / B3b / B4 / B10.

### 5.3 Round-4 deferred / rejected items

- **B7 (Windows path handling)** — deferred post-merge per §11.x policy (no Windows operator on the project today).
- No round-4 finding was rejected.

---

## 6. Round 5 — External 2-reviewer paste-back audit on the C2.2 polish itself

### 6.1 Provenance

After C2.2 (`c01aa27`) landed locally, the main agent drafted [`PEER_REVIEW_PROMPT_C2_2.md`](PEER_REVIEW_PROMPT_C2_2.md) (a 7-question audit prompt — Q-A through Q-G — scoped strictly to source code + architecture of the four touched files; repository / branch / push topology explicitly out-of-scope) and ran two independent reviewers against it on 2026-05-04 afternoon:

| Reviewer | Role | Test verification | Empirical probes run |
|---|---|---|---|
| `agent1` | Adversarial code-review subagent (paste-back) | 16/16 + 42/42 confirmed locally | bidi `\u202E` repr; BOM-only / leading-BOM / internal-BOM anchor; `(FETCH)` substring URL; nameless / 2-field parser input |
| `agent2` | Adversarial code-review subagent (independent — `gsd-code-review` skill loaded) | 16/16 + 42/42 confirmed locally (after installing project's runtime dep `PyYAML` in venv) | fragment / query / fragment+query / userinfo / `%23` URL probes; CRLF / BOM / comment-style anchor probes; hostile `(FETCH)` substring URL |

Both reviewer transcripts were read by the main agent and persisted at `chat-session-resources` paths within the workspace storage (machine-local, ephemeral). The structured 7-question verdicts and the empirical-probe outputs are captured in §6.2 and §6.3 respectively.

### 6.2 Round-5 cross-reviewer intersection matrix (verdict-level)

Findings ranked by **convergence** (both reviewers raised) then severity:

| ID | Description | agent1 | agent2 | Main-agent independent verification | Disposition |
|---|---|---|---|---|---|
| **D1** | Three sites pass `git()` / `git commit` / `git push` raw `proc.stderr` to the operator's terminal (`lock_effect_size_threshold.py:240,989,1087`), bypassing `_safe_repr_url`. This violates the C2.2 commit-message contract that B10 closes the terminal-corruption surface — git's own error messages can echo a hostile remote URL untouched | FLAG | FLAG | ✅ confirmed via grep — 3 raw `{proc.stderr}` / `{res.stderr}` interpolations | **C2.3 (D1)** |
| **E1** | `_normalize_remote_url("https://user:pass@host/path.git")` mis-applies the scp-style `:` heuristic to the userinfo's colon, yielding `user/pass@host/path` instead of `host/path`. B12 docstring had only deprecated `file://`, not userinfo-bearing https | passing-mention (called out as already-known limitation) | FLAG (raised explicitly) | ✅ confirmed via empirical probe: `https://user:pass@gitee.com/jqu9/repo.git?token=abc` → `user/pass@gitee.com/jqu9/repo` | **C2.3 (E1)** |
| F1 | `T16` only asserts the single-read invariant on the `--no-commit` happy path; commit-and-push branch and Guard-fail branches not regression-tested | FLAG | FLAG | ✅ confirmed via static read of `main()`; production push branch contains 0 reads of `metrics_a`, but invariant is held by inspection only | **DEFERRED to test-hardening commit pre-R1a** |
| F2 | Guard-5 ordering shift (data-side before Guard 4) is not exercised end-to-end through `main()` — would let a future regression silently re-order the guards without test breakage | not raised | FLAG | ✅ confirmed | **DEFERRED to test-hardening commit pre-R1a** |
| F3 | No `subprocess.run` mocking anywhere in the test suite; B5 TOCTOU branch + `--remote` URL-mismatch branch + push-failure branch all uncovered | FLAG | FLAG | ✅ confirmed via grep | **DEFERRED to test-hardening commit pre-R1a** |
| B1' | `_load_anchor` docstring does not state explicitly that `#`-style comment syntax is NOT supported | FLAG (doc nit) | PASS | ✅ confirmed; minor user-experience issue | **DEFERRED — fold into C2.3 docstring cleanup or post-merge polish** |
| L1 | Inter-process torn read: trainer is appending to `metrics.jsonl` while `load_metrics_with_hash`'s `read_bytes()` runs; not atomic. A torn last-line is silently swallowed by the JSON-decode-skip loop, but the SHA pinned in `EFFECT_SIZE_LOCKED.md` then references a partial file an external auditor would not be able to reproduce | latent #1 | not raised | true; known but solution-class is protocol-level (snapshot+atomic-rename + manifest extension) | **DEFERRED to R1a §6.6.x** (out-of-scope for C2.x polish) |
| L2 | `find_repo_root` walking up to a `.git` *file* (worktree pointer) vs `.git` *directory* — Pre-reg purity in a worktree-launched invocation needs explicit forbid-or-document | latent #3 | not raised | edge | **DEFERRED to R1a guard if needed; document only** |

### 6.3 Round-5 items raised by a single reviewer that did NOT survive intersection

| ID | Source | Why it did not promote |
|---|---|---|
| Internal BOM (`url1\n\ufeffurl2\n` counts as 2 non-empty lines, "found 2" message slightly less informative than ideal) | agent1 only | extreme edge; behavior is correct (rejection); only the error message is mildly suboptimal |
| `PyYAML` runtime dep tripped up testing in agent2's venv | agent2 only | not a code bug — `lock_effect_size_threshold.py` legitimately imports `yaml`; it is a project runtime dependency, not a test-suite stdlib-only contract |

### 6.4 Round-5 disposition summary

- **C2.3** (in progress, this absorption): D1 + E1.
- **DEFERRED to a dedicated test-hardening commit pre-R1a** (proposed name `C5f` or `C2.4-tests`): F1, F2, F3 + B1' docstring tweak.
- **DEFERRED to R1a / protocol-level**: L1 (inter-process torn read), L2 (worktree).
- **REJECTED**: internal-BOM message phrasing, PyYAML-as-test-bug.

### 6.5 Empirical probe transcripts (machine-verifiable)

Probes run by the main agent independently against `c01aa27` to confirm the cross-reviewer findings before drafting C2.3:

```
$ grep -nE "proc\.stderr|res\.stderr" review/0502/scripts/lock_effect_size_threshold.py
240:            f"git {' '.join(args)} failed (exit {res.returncode}):\n{res.stderr}\n"
989:            sys.stderr.write(f"git commit failed: {proc.stderr}\n")
1087:                f"git push {resolved_remote} {branch} failed:\n{proc.stderr}\n"
```

D1 verified — three unwrapped sites.

```
$ python3 -c "
import sys; sys.path.insert(0, 'review/0502/scripts')
import lock_effect_size_threshold as L
for u in [
    'git@gitee.com:jqu9/repo.git#frag',
    'git@gitee.com:jqu9/repo.git?token=abc',
    'https://user:pass@gitee.com/jqu9/repo.git?token=abc',
    'git@gitee.com:jqu9/repo.git#frag?query',
    'git@gitee.com:jqu9/repo.git?query#frag',
    'git@gitee.com:jqu9/repo%23weird.git',
]:
    print(repr(u), '->', L._normalize_remote_url(u))
"
'git@gitee.com:jqu9/repo.git#frag'                           -> gitee.com/jqu9/repo
'git@gitee.com:jqu9/repo.git?token=abc'                      -> gitee.com/jqu9/repo
'https://user:pass@gitee.com/jqu9/repo.git?token=abc'        -> user/pass@gitee.com/jqu9/repo   ← E1 BUG
'git@gitee.com:jqu9/repo.git#frag?query'                     -> gitee.com/jqu9/repo
'git@gitee.com:jqu9/repo.git?query#frag'                     -> gitee.com/jqu9/repo
'git@gitee.com:jqu9/repo%23weird.git'                        -> gitee.com/jqu9/repo%23weird
```

E1 verified — 1/6 cases produces the wrong normalization (userinfo-bearing https).

### 6.6 C2.3 — proposed source-code modifications

**File**: `review/0502/scripts/lock_effect_size_threshold.py`

1. **D1 fix**: introduce a private `_safe_stderr_block(text)` helper that wraps multi-line subprocess stderr through `text.encode("unicode_escape").decode("ascii")` (so embedded `\x1b` / `\x00` / `\r` / `\n` all render as backslash-escapes on a single logical line each, while still being human-readable). Apply to:
   - `git()` helper L240 (covers all subprocess errors routed through this helper — `git remote get-url`, `git rev-parse HEAD`, `git status --porcelain`, etc.)
   - `git commit` failure path L989
   - `git push` failure path L1087

   This is a **behavior change at the diagnostic layer only** — no successful path emits anything new; only failure paths now show `\\x1b[31m` instead of a literal red ANSI escape.

2. **E1 fix**: in `_normalize_remote_url`, after the scheme strip and BEFORE the scp-`:` heuristic, peel off any leading `userinfo@` if it appears strictly before the first `/`. Concretely: locate the first `/` and the first `@`; if `@` exists and (`/` does not or `@` precedes `/`), advance past `@`. This handles `https://user@host/...`, `https://user:pass@host/...`, `ssh://user@host/...`, and `ssh://user:pass@host/...` symmetrically while leaving scp-style `git@host:path` unchanged (the existing `if s.lower().startswith("git@"):` branch fires first for the scp form). Update the function docstring (B12 paragraph) to reflect that userinfo is now first-class.

**File**: `review/0502/scripts/test_remote_resolver.py`

3. **D1 tests** (new test class `TestSafeStderrBlock`, ~3 cases): plain string passthrough; ANSI escape sequence neutralized; embedded NUL + newline neutralized.

4. **E1 tests** (in `TestNormalizeRemoteUrl`, new R17 / R18 / R19): `https://user@host/...`, `https://user:pass@host/...`, `https://user:pass@host/path?token=abc`. Each must equal the no-userinfo, no-query form.

**Out of scope for C2.3** (per §6.4 disposition): no test changes for F1/F2/F3 — those are reserved for a dedicated test-hardening commit. No docstring rewrite for B1' (will be folded into the next docstring-touching commit).

### 6.7 C2.3 commit-message contract

The C2.3 commit will:
- absorb D1 + E1 only, citing this §6 by file:line
- preserve LOCKED_PROTOCOL_VERSION = `v0_pending_R1a` (no protocol change)
- remain local-only per the standing operator directive on repository topology
- run the full test suite (16/16 + 42/42 + new tests) and require all green before commit
- be signed off (`git commit -s`) following C2.1 / C2.2 precedent
