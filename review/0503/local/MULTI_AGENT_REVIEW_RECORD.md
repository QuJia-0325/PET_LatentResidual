# MULTI_AGENT_REVIEW_RECORD — Sigma-normalize ablation pre-registration

**Drafted**: 2026-05-03 night, between C5e and R1a (per §10.6 decision)
**Purpose**: Primary-source audit trail for the multi-round review process that produced `REV1_PLAN.md` v0.3 + `REV1_TOOLING_PLAN.md` v0.3 + the C1–C5e + R1a/R1b commit chain. Cited from R1a §6.6.8 changelog so the protocol commit's pre-registration history is independently auditable.
**Scope**: Reviews of v0 (`POST_V6_NEXT_STEPS.md` §6.4 + §6.6 + the two scripts) → v0.1 → v0.2 → v0.3 (operator-absorbed). Round 1 = external GPT-5.5 + Claude paste-back; Round 2 = internal multi-agent re-review of v0.1 / v0.2; Operator round-2.5 = `OPERATOR_REPLY_pre_C1_20260503.md` + `current_experiment_architecture_status_20260503.md`.

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

| Metric | Round 1 | Round 2 | Round 2.5 | Total |
|---|---:|---:|---:|---:|
| External / operator reviewers | 2 (GPT-5.5, Claude) | 0 | 1 (operator) | 3 |
| Internal subagent reviewers | 0 | 4 | 0 | 4 |
| Findings adopted | 16 (8 GPT + 8 Claude) | 17 (5 F + 12 S) | 12 (7 Q + 5 flags) | 45 |
| Findings deferred | 0 | 2 (AR(p), multi-comp) | 0 | 2 |
| Findings rejected | 0 | 0 | 0 | 0 |
| Plan-version bumps caused | v0 → v0.1 | v0.1 → v0.2, v0.2 → v0.3 | v0.3 → v0.3 + §11 | 4 versions |
| New commits added to chain | C1–C5 (5) | C5b, C5c, C5d (3) + R1 split into R1a + R1b (1 → 2) | C5e (1) | C1, C2, C3, C4, C5, C5b, C5c, C5d, C5e, R1a, R2, R1b (12) |

R1a §6.6.8 changelog should cite, in this order:

1. `review/0503/operator/PEER_REVIEW_GPT55.md` (round-1 external desk-reject finding)
2. `review/0503/operator/PEER_REVIEW_CLAUDE.md` (round-1 external major-revision finding)
3. `review/0503/local/MULTI_AGENT_REVIEW_RECORD.md` (this file — round-1 + round-2 + round-2.5 audit-trail compendium)
4. `review/0503/local/REV1_PLAN.md` §11 + `REV1_TOOLING_PLAN.md` §11 (round-2.5 operator absorption)
5. `review/0503/operator/OPERATOR_REPLY_pre_C1_20260503.md` + `current_experiment_architecture_status_20260503.md` + `B_sanity_metrics_val50_schema_reference.jsonl` (round-2.5 operator-side primary sources)

This file is intended to remain a frozen snapshot of the multi-round review state at the moment R1a is drafted. Subsequent revisions (e.g., R2 hardening, R1b lock) write their own audit-trail subsections inside R2/R1b commits, not back into this file.
