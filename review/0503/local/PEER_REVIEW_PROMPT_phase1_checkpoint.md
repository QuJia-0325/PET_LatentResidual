# PEER REVIEW PROMPT — Phase-1 checkpoint commit (light plan-diff audit)

**Reviewer role**: Senior ML reviewer (NeurIPS / ICML pre-registration / pre-reg auditor track).
**Reviewing model recommendation**: GPT-5.4 xhigh via codex, OR GPT-5.5 xhigh, OR Claude Opus 4.7 xhigh.
**Scope**: ONLY plans + multi-agent record. NOT the experiment code (which doesn't exist yet — this checkpoint precedes implementation).
**Time budget**: ~10–20 minutes of reviewer reasoning. This is a light audit, not a full Round-3.

---

## Context

This is the σ-normalize ablation pre-registration project for the PET Latent-Residual paper. Earlier rounds:

- **Round 1** (external paste-back): GPT-5.5 + Claude Opus 4.7 reviewed v0 → produced 2 desk-reject blockers (`paired_CV_A` wrong noise estimator; trichotomy maps `rel_diff < X` including large negative to "C ≈ A"). Verbatim in `review/0503/operator/PEER_REVIEW_GPT55.md` + `PEER_REVIEW_CLAUDE.md`.
- **Round 2** (4 internal subagents on v0.1/v0.2): produced F1–F12 + S1–S17 findings absorbed into `REV1_TOOLING_PLAN.md §10` + `REV1_PLAN.md §10`.
- **Round 2.5** (operator on remote host): answered Q9–Q15 + flagged 5 unsolicited issues. Absorbed into §11 of both plans.

**The Phase-1 checkpoint commit (HEAD `5056c09`) does THREE things**:
1. Replaces `gitee/foc_lite_hop0` literal references in §10.6 of both plans with portable `<canonical>/foc_lite_hop0` placeholder wording (resolved at runtime via `.review_canonical_remote` per Q10).
2. Adds new §11 "Operator round-2.5 absorption" to both plans capturing operator's 7 answers + 5 flags + new tiny commit `C5e` (run_ablation.sh help refresh) into the §10.3 commit chain.
3. Adds new file `review/0503/local/MULTI_AGENT_REVIEW_RECORD.md` (149 lines) — primary-source audit trail of all 3 review rounds for citation in R1a §6.6.8 changelog.

**What hasn't happened yet**: actual code implementation (C1–C5e + R1a/R1b commits). C1 begins immediately after this audit lands.

---

## Files to read (in this exact order, all relative to repo root)

1. `review/0503/local/REV1_TOOLING_PLAN.md` — full file. **Focus on §10 (Round-2 absorption) + §11 (Round-2.5 absorption). §1–§9 are stable, audited in earlier rounds.**
2. `review/0503/local/REV1_PLAN.md` — full file. **Focus on §10 + §11 (mirror of TOOLING side).**
3. `review/0503/local/MULTI_AGENT_REVIEW_RECORD.md` — full file. **This is the new audit trail.**
4. `review/0503/operator/OPERATOR_REPLY_pre_C1_20260503.md` — operator's 7 answers + 5 flags (the input that drove §11).
5. `review/0503/operator/current_experiment_architecture_status_20260503.md` — operator's GPU/run audit.
6. **Skim only** (already-audited round-1 sources): `review/0503/operator/PEER_REVIEW_GPT55.md` + `PEER_REVIEW_CLAUDE.md` — for cross-checking the §10 / §11 absorption claims.

---

## What you are auditing (5 specific questions)

### Q-A. Is the `<canonical>/foc_lite_hop0` placeholder semantics actually portable?

The §10.6 Q4/Q7 wording says:
> R1a / R1b plain `git commit -s` (sign-off only) + `git push <canonical-remote> foc_lite_hop0` immediately after commit, where `<canonical-remote>` is resolved via `.review_canonical_remote` (on operator host = `origin`, on local Mac = `gitee`).

The C5b `require_lock_pass()` semantics says:
> queries `git merge-base --is-ancestor <lock_commit> <canonical>/foc_lite_hop0` for the canonical-remote check, with `<canonical>` resolved at runtime.

**Audit**: Is there a pathological host where this resolution fails? E.g., a host where two remotes both URL-match `gitee.com:jqu9/PET_LatentResidual` (forks)? A host where neither remote URL-matches the canonical? A host where the canonical remote URL is registered under a different protocol (https vs ssh)? Specifically: does the resolver semantics in §11.1 Q10 (URL-fragment scan via `.review_canonical_remote`) handle all these cases gracefully or does it have a silent-failure mode?

### Q-B. Does §11.3's revised commit chain integrate cleanly with §10.3?

§11.3 says the C5e commit slots between C5d and the [B] approve gate. **Audit**: walk the chain HEAD `5056c09` → C1 → C2 → C3 → C4 → C5 → C5b → C5c → C5d → **C5e (NEW)** → MULTI_AGENT_REVIEW_RECORD.md → R1a → A_main → R2 → A_seed=43 → R1b → C_uniform. Are there any missing prerequisites? Any commits whose stated dependencies are violated by the C5e insertion? Does the [B] approve gate's hard-gate check (`require_lock_pass()`) need any update because C5e modified the help text of the same `run_ablation.sh` it gates?

### Q-C. Is `B_sanity_metrics_val50_schema_reference.jsonl` (46 rows) actually a valid C1 fixture?

§11.1 Q9 says:
> The 46-row trace is copied to `review/0503/operator/B_sanity_metrics_val50_schema_reference.jsonl` (filename says "val50" but contains 46; will refresh on B completion). C1 golden-trace test gets a two-fixture setup.

**Audit**: read the first 2-3 rows of this fixture. Does its row schema (`step` key, `event=val` filter, `val_chain_normal_mse` + `val_select_score` keys) actually match what C1's `_metrics_compat.py` should test against? Are there schema differences vs `A_sanity_metrics_tail5_schema_reference.jsonl` (5 rows) that would invalidate the "two-fixture" claim? **Critical**: is the filename mismatch (val50 → 46 rows) a sign of a stale freeze-point that could break the test if B finishes before C1 commits?

### Q-D. Is the MULTI_AGENT_REVIEW_RECORD.md complete enough to cite as primary source in R1a §6.6.8?

Pre-registration credibility depends on the audit trail being independently auditable. **Audit**: does the record's §1 (round-1 verbatim cross-table) + §2 (round-2 F/S codes with adopt-reject column) + §3 (round-2.5 operator absorption) + §4 (aggregate metrics) cover everything an external reviewer would need to verify the protocol's revision history? Specific check: does the record cross-reference the verbatim source file (`PEER_REVIEW_GPT55.md` etc.) so a reviewer can verify the paraphrased findings are not cherry-picked? Are the "adopt/defer/reject" judgments traceable to specific line-numbers in the plan files?

### Q-E. Are there any new pre-registration vulnerabilities introduced by Phase-1 itself?

Phase-1 added §11 + MULTI_AGENT_REVIEW_RECORD.md + portable-remote rewording. **Audit**: did this introduce any new desk-reject vulnerabilities? Specifically:
- Does §11.5 (self-correction note about λ_roll = 4.0 not 0) constitute a retroactive plan modification that breaks pre-reg purity? Or is it a benign clarification?
- Does the C5e commit (cosmetic help-text refresh) create any audit-trail confusion since it touches `run_ablation.sh` after the C5b hard-gate is locked in?
- Does the canonical-remote portable-wording cleanup leave any stale references to `gitee/foc_lite_hop0` that would be ambiguous to a future operator running R1b?

---

## Output format

For each of Q-A through Q-E, please provide:

```
## Q-X verdict

**Severity**: PASS / FLAG / BLOCK

**Finding**: <2-4 sentences>

**Evidence**: <line numbers in the audited files>

**Action**: <if FLAG/BLOCK: minimal patch in pseudo-diff form; if PASS: "no action needed">
```

After all 5 questions, end with:

```
## Overall verdict

<one of:>
- LIGHT-PASS — no blockers, proceed to C1.
- LIGHT-FLAG — N flags, list them; proceed to C1 but address before R1a.
- HEAVY-FLAG — N flags including ≥1 BLOCK; halt C1, fix, then re-audit.
```

Be honest. We've already had 2 desk-reject findings from Round 1 paste-back; if there's a third lurking, surface it now.
