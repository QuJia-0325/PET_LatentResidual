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

---

## 7. Round 6 — Operator/codex review of C2.3 → C3 + F2 + F3 + F4 absorption

**Source**: `review/0503/operator/OPERATOR_REPLY_C2_3_20260504.md`. Codex
read C2.3 (D1 + E1) on the operator host, ran the test suite (1/69 failed
on default `python3` due to missing PyYAML; 69/69 OK on the conda env), and
raised 5 findings (F1–F5). Round 6 absorbs F1 (renamed C3), F2, F3, F4
in-tree; F5 deferred to R1b lock-policy decision.

### 7.1 Finding inventory

| ID | Severity | Locus | Summary |
|----|----------|-------|---------|
| F1→**C3** | **HIGH (blocker)** | `select_best_ckpt_smoothed.py::list_saved_steps` | only matched legacy `ckpt_step_*.pt` / `ckpt_last.pt` and `global_step` key, but real trainer writes `step_*.pt` / `last.pt` / key=`step` → Method-D non-operational on real outputs |
| F2 | MEDIUM | `lock_effect_size_threshold.py::compute_paired_stats_from_rows` L657-679 | `skipped_no_metric` counter increments on EVERY in-window train row missing `val_select_score`; produces 33-row false-positive `[warn]` on a perfectly healthy `A_sanity` metrics file |
| F3 | MEDIUM | `lock_effect_size_threshold.py::main` pre-push TOCTOU L1109-1130 | only re-checked fetch URL; if `remote.<name>.pushURL` is configured separately, `git push` follows it and bypasses the canonical check |
| F4 | LOW | `OPERATOR_CONFIRM_REQUEST_C2_3.md §2` | recommended bare `python3` which lacks PyYAML on operator host → spurious 1/69 fail |
| F5 | DEFERRED | Guard 5 / R1b lock-policy | Guard 5 currently permissive on absent rolling-window control; codex flagged it as still-open at R1b lock time |

### 7.2 C3 — Method-D selector dual-name compatibility (P0)

**Disposition**: Fix in tree.

**Patch surface**: `review/0502/scripts/select_best_ckpt_smoothed.py`

- `list_saved_steps()` rewritten to (a) glob both `step_*.pt` and `ckpt_step_*.pt`; (b) return `(sorted_steps, step_to_path)`; (c) try both `last.pt` and `ckpt_last.pt` for `--include-last-pt`; (d) fall back ckpt-key from `"step"` → `"global_step"`.
- `--ckpt-dir` help text updated to list both naming forms.
- Final recommendation in `main()` prints the actually-existing `.pt` path discovered in the loop, falling back to a constructed `step_{step:06d}.pt` only when no match exists, with explicit `[warn]` placeholder otherwise.

**Falsifiability**: `review/0502/scripts/test_select_best_ckpt_smoothed.py` (new file, 12 tests, 5 classes):

| Class | Cases | Verifies |
|-------|-------|----------|
| `TestListSavedStepsNewForm` | 2 | new-form zero-padded + unpadded step parsing |
| `TestListSavedStepsLegacyForm` | 1 | legacy-form still works |
| `TestListSavedStepsCohabitation` | 2 | mixed dir: new wins on tie; disjoint steps both kept |
| `TestListSavedStepsRobustness` | 3 | non-ckpt files ignored, invalid suffix skipped, empty dir → `[]` |
| `TestListSavedStepsWithTorch` | 4 (skipUnless torch) | step-key + legacy global_step + ckpt_last.pt + step-preferred |

Mac result: 8 OK + 4 torch-skipped = 12/12 PASS.

### 7.3 F3 — pushURL TOCTOU (P1)

**Disposition**: Fix in tree.

**Patch surface**: `review/0502/scripts/lock_effect_size_threshold.py::main` L1149-1196 (within the existing `if not args.force_unsafe_remote:` guard).

After the existing fetch-URL TOCTOU re-check, add a parallel `git remote get-url --push <remote>` call. If `_normalize_remote_url(push_url) != _normalize_remote_url(resolved_url)`, abort with `ERROR (pushURL mismatch): ... refusing to push`. The new check is a NO-OP on the common case (no separate pushURL configured → `--push` returns the fetch URL and the equality is trivial). The override path is the same `--force-unsafe-remote` flag (already documented as not-recommended for pre-registration).

**Falsifiability**: `review/0502/scripts/test_remote_resolver.py::TestPushUrlTOCTOU` (new class, 3 cases):

| Case | Setup | Expected |
|------|-------|----------|
| F3a | `git init`; `remote add origin <fetch>`; no pushURL | `get-url` and `get-url --push` both return fetch URL; both normalize equal |
| F3b | also `remote set-url --push origin <attacker>` | fetch normalizes canonical; pushURL normalizes DIFFERENT → F3 catches |
| F3c | also `remote set-url --push origin <https form of same repo>` | fetch ssh + pushURL https both normalize to SAME canonical → F3 does NOT false-positive |

These tests use real local `git init` tempdirs (no subprocess mocking), so they exercise the actual git binary's `--push` semantics — the strongest possible signal that F3 holds across git versions on operator host.

### 7.4 F2 — train-row warning narrowing (P2)

**Disposition**: Fix in tree.

**Patch surface**: `review/0502/scripts/lock_effect_size_threshold.py::compute_paired_stats_from_rows` L657-712.

A row is now classified as "val-like" iff `event=="val"` (case-insensitive) OR the `event` field is absent (legacy-fixture compatibility — the operator's `B_sanity_metrics_val50_schema_reference.jsonl` has no event tag but is val-only by construction). Only val-like in-window rows can increment `skipped_no_metric`. Train rows are still IGNORED for series construction (they have no `val_select_score` to plot), but they no longer noisily inflate the warning counter.

The trainer's actual emitted event values are confirmed `"train"` (L2217) and `"val"` (L2503) in `train_first_hop.py`, so the case-insensitive equality is exact.

**Falsifiability**: `test_metrics_compat.py::TestLockComputePairedCV`, three new cases:

| Test | Surface | Asserts |
|------|---------|---------|
| T17 | 5 val + 33 train rows, all in window | N=5 series; **NO** `[warn] compute_paired_cv` line in stderr |
| T18 (negative-control) | 5 val + 2 partial val (event=val, no metric) | warn STILL fires with "2 in-window row(s) missing" |
| T19 (legacy-compat) | rows with no event field, 1 partial | warn fires with "1 in-window row(s) missing" |

T18 is critical: F2 must NOT silence genuine val partial-writes, only the train-row noise. T17 reproduces the operator's empirical "33 spurious skips" condition.

### 7.5 F4 — operator-confirm doc PyYAML guidance (P3)

**Disposition**: Documentation fix only.

**Patch surface**: `review/0503/local/OPERATOR_CONFIRM_REQUEST_C2_3.md §2`.

Recommend `<conda-env>/bin/python -m unittest …` (e.g. `/home/qujiaxiang/.conda/envs/rae/bin/python`) and explicitly state that bare `python3` requires PyYAML. Updated test count: 87 (was 69; +12 from C3, +3 each from F2/F3, +0 from F4). Section 1's HEAD-hash expectation softened to "拉到最新 Mac 推送的 commit 即可" since this Round-6 batch will introduce new commits.

**No test code change** for F4 — the missing-PyYAML failure is a runtime-dependency signal, not a regression. Hiding it (e.g. via `@unittest.skipUnless(_HAS_YAML)`) would mask the fact that the lock script CANNOT run without PyYAML.

### 7.6 F5 — DEFERRED

Guard 5 still aborts only on detected rolling-window control mismatch, not on absence of the control flag. Codex flags this as "permissive at lock time" — i.e. a run with no σ-normalize information at all would currently pass Guard 5. Round 6 explicitly defers this to the R1b protocol bump (where the lock-policy decision can move atomically with the locked sample). Reasoning: tightening Guard 5 mid-`v0_pending_R1a` would break the locked-sample contract; the question is a policy decision, not a bug.

### 7.7 Round-6 absorption summary

| Round | Findings raised | Findings absorbed in tree | Findings deferred |
|-------|-----------------|---------------------------|-------------------|
| 1 (Codex) | 4 | 4 | 0 |
| 2 (Codex follow-up) | 3 | 3 | 0 |
| 2.5 (operator pre-C1) | 1 (schema mismatch) | 1 (C1 patch) | 0 |
| 4 (Codex Lane A) | 4 | 4 (C2.2: A1, A2, A3, A5) | 0 |
| 5 (Lane B + Lane E) | 2 | 2 (C2.3: D1, E1) | 0 |
| **6 (operator/codex)** | **5** | **4 (C3, F2, F3, F4)** | **1 (F5 → R1b)** |
| **Total**             | **19**          | **18**                    | **1**             |

### 7.8 Test-count audit

| Round | Test file | New tests | Cumulative |
|-------|-----------|-----------|------------|
| C2 baseline | test_remote_resolver.py + test_metrics_compat.py | — | 49 |
| C2.2 (A2/A3/A5) | test_metrics_compat.py | +T13/T14/T15/T16 | 53 |
| C2.3 (D1/E1) | test_remote_resolver.py | +D1×3, +E1×3, etc. | 69 |
| **C3 (F1)** | **test_select_best_ckpt_smoothed.py (new)** | **+12 (4 torch-gated)** | **81** |
| **F2** | test_metrics_compat.py | **+T17/T18/T19** | **84** |
| **F3** | test_remote_resolver.py | **+F3a/F3b/F3c** | **87** |

Mac (no torch): 87 tests, 83 OK + 4 skipped → `OK (skipped=4)`.
Operator (with torch): 87 tests, 87 OK → `OK`.

LOCKED_PROTOCOL_VERSION unchanged: still `v0_pending_R1a`.


---

## 8. Round 7 — Cross-AI peer review on Round 6 → Round 8 absorption (G1 + G2 + G3 + G4 + G5)

**Source**: Two parallel external AI reviews requested after the Mac
agent's Round-6 self-review came back LIGHT-PASS. The user (anchored on
the principle that R1a is a one-way door) authorized cross-AI peer
review before committing the locking sequence. Both reviewers received
the same Round-7 prompt (file inventory + per-finding bug+fix+test
summaries + 5 falsifiable questions Q1–Q5).

Both verdicts: **LIGHT-FLAG** (commit Round 6, but address findings
before R1a lock-in pushes). Two reviewers raised three consensus
findings + two single-reviewer findings. Round 8 absorbs all five in
tree.

### 8.1 Finding inventory

| ID | Severity | Origin | Locus | Summary |
|----|----------|--------|-------|---------|
| **G1** | MEDIUM, security-adjacent | Both reviewers (consensus, empirically verified by both) | `lock_effect_size_threshold.py::main` push branch | Round-6 F3 used `git remote get-url --push <remote>` which returns ONLY the first pushURL, but `git push` mirrors to ALL configured pushURLs. A canonical-first-then-hostile multi-pushURL config (via `git remote set-url --push --add`) bypasses the F3 check |
| **G2** | LOW | Both reviewers (consensus) | `select_best_ckpt_smoothed.py::list_saved_steps` | Round-6 C3 silently picks new-form on collision; the `test_new_form_wins_on_tie` test uses empty-byte fixtures so cannot detect content divergence between same-step new+legacy ckpts |
| **G3** | LOW–MEDIUM | Both reviewers (consensus, distinct rationales) | `lock_effect_size_threshold.py::main` Guard 5 config-side | Round-6 deferred F5 to R1b on the rationale that tightening Guard 5 mid-`v0_pending_R1a` would break the locked-sample contract; reviewers rejected this because no locked sample exists yet |
| **G4** | TRIVIAL | Reviewer-1 only | `OPERATOR_CONFIRM_REQUEST_C2_3.md` line 106 | One stragger "这 90 测试" remained after the Round-6 F4 fix (other locations correctly say 87) |
| **G5** | TRIVIAL | Reviewer-2 only | `lock_effect_size_threshold.py` F3 inline comment | F3 comment said `git push <remote> <branch> will go to the pushURL` (singular framing). This single-pushURL framing is the conceptual bug that masked G1 during Round-6 patch authoring |

User decision (May 4 2026): land all five in Round 8, including G3
strict-mode (option A from the Round-7 synthesis).

### 8.2 G1 — F3 multi-pushURL bypass (P0)

**Disposition**: Fix in tree.

**Patch surface**: `review/0502/scripts/lock_effect_size_threshold.py::main` push-branch (within the `if not args.force_unsafe_remote:` guard, after the existing fetch-URL TOCTOU re-check).

Replace the single-pushURL check (`git remote get-url --push <remote>`) with an enumeration via `git remote get-url --push --all <remote>`. Parse the multiline output; require EVERY URL to canonicalize to the resolved canonical via `_normalize_remote_url`. Defensive: if the enumeration returns no URLs, abort (this should be impossible per git docs but a refusal here costs nothing). Error message names the offending pushURL plus the full pushURL list so the operator can see exactly which mirror is divergent.

**Empirical verification**: on git 2.50.1 (Apple Git-155), confirmed that:
- `git remote set-url --push --add origin <canonical>` followed by `git remote set-url --push --add origin <hostile>` produces a remote whose `--push` (singular) returns only `<canonical>` but whose `--push --all` returns both URLs (one per line)
- `git push origin <branch>` mirrors to BOTH bare repos (verified by inspecting `refs/heads/<branch>` in each)

The bypass is therefore real and exploitable on any git ≥ 2.7 host without G1.

**Falsifiability**: `test_remote_resolver.py::TestPushUrlTOCTOU` extended with two new cases (real `git init` tempdirs, no mocking):

| Case | Setup | Asserts |
|------|-------|---------|
| **F3d** (discriminating) | `set-url --push --add` × 2: canonical, then hostile | (a) single `--push` returns only canonical → bypass is real; (b) `--push --all` returns both; (c) per-URL normalize-and-compare flags exactly the hostile one |
| **F3e** (negative-control) | `set-url --push --add` × 2: ssh form, then https form of SAME repo | both pushURLs canonicalize equal — G1 must NOT abort on legitimate mirror configs |

### 8.3 G2 — C3 collision warning (P1)

**Disposition**: Fix in tree.

**Patch surface**: `review/0502/scripts/select_best_ckpt_smoothed.py::list_saved_steps`.

Replace `step_to_path.setdefault(step, cf)` with explicit `get` + branch logic. On cross-name collision (`existing.name != cf.name`), emit `[warn] step <S>: both <kept> (kept) and <ignored> (ignored) exist in <dir>. If their contents differ (sha256sum ...), the recommendation may not match what the eval script loads. Inspect manually.` Precedence is preserved (new-form wins per documented Round-6 policy); we add observability, not policy change. Hard abort was rejected because it would break the legitimate "ran the trainer twice on the same out_dir with different conventions" debug workflow.

**Falsifiability**: `test_select_best_ckpt_smoothed.py::TestListSavedStepsCohabitation` extended with two new cases:

| Test | Setup | Asserts |
|------|-------|---------|
| `test_collision_with_different_contents_warns` (discriminating) | both files exist with DIFFERENT byte contents (`b"new-form-content-AAA"` vs `b"legacy-content-ZZZ"`) | (a) precedence preserved (new-form path returned); (b) `[warn]` line on stdout names BOTH file names AND uses `(kept)` / `(ignored)` labels |
| `test_no_collision_no_warning` (negative-control) | only new-form files, no collision | NO `[warn]` line emitted (G2 must not regress the no-noise property on the common case) |

### 8.4 G3 — Guard 5 strict best_metric (P1)

**Disposition**: Fix in tree.

**Patch surface**: `review/0502/scripts/lock_effect_size_threshold.py::main` Guard 5 config-side block + new `--allow-dev-best-metric` argparse flag.

Tighten the equality from `cfg_metric_key not in (None, "val_multi_objective", "val_select_score")` to `cfg_metric_key != "val_multi_objective"`. Non-canonical values abort with exit 4 and a stderr message naming the actual value AND citing `OPERATOR_REPLY_pre_C1_20260503.md` Op-flag-5 + `MULTI_AGENT_REVIEW_RECORD.md §8 (G3)`. The opt-out path `--allow-dev-best-metric` lets non-canonical values pass but emits a loud `[warn] guard 5 strict bypassed via --allow-dev-best-metric ...` so operators cannot accidentally lock under dev mode.

The opt-out flag's argparse help explicitly says "NEVER pass at R1a lock time — at lock time the strict check is the contract."

**Falsifiability**: new test class `TestGuard5StrictBestMetric` in `test_metrics_compat.py`, 6 cases (drives `main()` end-to-end through the same pattern T16 uses):

| Test | best_metric value | --allow-dev-flag | Expected rc | Asserts |
|------|-------------------|------------------|-------------|---------|
| **G3a** (positive control) | `val_multi_objective` | no | 0 | success (no regression vs T16 baseline) |
| **G3b** (discriminating, was Round-6 false-pass) | (key absent) | no | 4 | stderr names "guard 5 strict" + "None" |
| **G3c** (discriminating, was Round-6 false-pass) | `val_select_score` | no | 4 | stderr names "guard 5 strict" + "val_select_score" |
| **G3d** (regression of Round-6 strictness) | `some_random_metric` | no | 4 | stderr names "guard 5 strict" + the unknown value |
| **G3e** (opt-out positive) | `val_select_score` | yes | 0 | stderr contains `[warn] guard 5 strict bypassed` AND "R1a lock time" gate |
| **G3f** (opt-out harmless on canonical) | `val_multi_objective` | yes | 0 | NO `[warn]` line (prevent dev-flag noise on production-shaped configs) |

T16 was deliberately written with `best_metric: val_multi_objective` (canonical), so it continues to pass under strict G3 without modification.

### 8.5 G4 — doc test-count straggler (P3)

**Disposition**: Fix in tree (doc-only).

**Patch surface**: `review/0503/local/OPERATOR_CONFIRM_REQUEST_C2_3.md` §1 (preface) + §2 (test count + table).

Round-6 F4 updated most call-sites from 69 → 87, but one straggler at line 106 said "这 90 测试". Reviewer-1 caught this. Round-8 updates ALL call-sites to **97** (87 + 2 G1 + 2 G2 + 6 G3) and adds Round-8 callout "(含 Round-8 absorption: G1 + G2 + G3 + G4 + G5)" so operators can tell which absorption batch the doc reflects.

### 8.6 G5 — F3 comment singular→plural pushURL (P0 lockstep with G1)

**Disposition**: Fix in tree (comment-only, in lockstep with G1).

**Patch surface**: `review/0502/scripts/lock_effect_size_threshold.py::main` F3 inline comment block.

Round-6 F3 comment said "`git push <remote> <branch>` will go to the pushURL" (singular). Reviewer-2 noted this single-pushURL framing is exactly what masked the multi-pushURL gap during Round-6 patch authoring. G5 rewrites the comment with explicit pushURL(s) plural framing AND adds a dedicated `G1 hardening (Round-7 cross-AI peer review, May 4 2026)` paragraph documenting the empirical verification, the bypass mechanism, and the `--push --all` enumeration. The conceptual loop is closed.

### 8.7 F5 status update

The Round-7 reviewers rejected the Round-6 deferral rationale ("would break the locked-sample contract") because no locked sample exists yet. G3 absorbs the substance of F5 in tree (strict equality on `val_multi_objective`). Therefore:

- **Round-6 status** of F5: DEFERRED to R1b
- **Round-8 status** of F5: **CLOSED** via G3. The R1b deferral is no longer needed.

The Round-6 commit message's qualifier "F5 deferred (rationale, possibly premature — see Round-7 review feedback)" anticipated this correction.

### 8.8 Round-8 absorption summary

| Round | Findings raised | Findings absorbed in tree | Findings deferred |
|-------|-----------------|---------------------------|-------------------|
| 1 (Codex) | 4 | 4 | 0 |
| 2 (Codex follow-up) | 3 | 3 | 0 |
| 2.5 (operator pre-C1) | 1 | 1 | 0 |
| 4 (Codex Lane A) | 4 | 4 | 0 |
| 5 (Lane B + Lane E) | 2 | 2 | 0 |
| 6 (operator/codex) | 5 | 4 (C3/F2/F3/F4) | 1 (F5 → later closed by G3) |
| **7 (cross-AI peer review)** | **5** | **5 (G1/G2/G3/G4/G5)** | **0** |
| **Total**                      | **24**          | **23 + 1 deferred-then-closed** | **0 (open)** |

### 8.9 Test-count audit

| Round | Test surface | New tests | Cumulative |
|-------|--------------|-----------|------------|
| C2 baseline | `test_remote_resolver.py` + `test_metrics_compat.py` | — | 49 |
| C2.2 (A2/A3/A5) | `test_metrics_compat.py` | +T13/T14/T15/T16 | 53 |
| C2.3 (D1/E1) | `test_remote_resolver.py` | +D1×3, +E1×3, etc. | 69 |
| Round 6 C3 (F1) | `test_select_best_ckpt_smoothed.py` (new) | +12 (4 torch-gated) | 81 |
| Round 6 F2 | `test_metrics_compat.py` | +T17/T18/T19 | 84 |
| Round 6 F3 | `test_remote_resolver.py` | +F3a/F3b/F3c | 87 |
| **Round 8 G1** | `test_remote_resolver.py` | **+F3d/F3e** | **89** |
| **Round 8 G2** | `test_select_best_ckpt_smoothed.py` | **+test_collision_*/test_no_collision_*** | **91** |
| **Round 8 G3** | `test_metrics_compat.py` | **+G3a/G3b/G3c/G3d/G3e/G3f** | **97** |

Mac (no torch): 97 tests, 93 OK + 4 skipped → `OK (skipped=4)`.
Operator (with torch): 97 tests, 97 OK → `OK`.

LOCKED_PROTOCOL_VERSION unchanged: still `v0_pending_R1a`. R1a lock can now proceed without further peer-review absorption.
