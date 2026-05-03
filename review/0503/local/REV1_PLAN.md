# PEER REVIEW CONVERGENCE + REV1 REMEDIATION PLAN

**Date**: 2026-05-03
**Sources**:

- [`review/0503/operator/PEER_REVIEW_GPT55.md`](../../0503/operator/PEER_REVIEW_GPT55.md) — verdict: 🛑 DESK REJECT
- [`review/0503/operator/PEER_REVIEW_CLAUDE.md`](../../0503/operator/PEER_REVIEW_CLAUDE.md) — verdict: ⚠️ MAJOR REVISIONS (with desk-reject risk on Q1 + Q3)

**Status**: Plan only. **No protocol files have been edited yet** in response to these reviews. User approval required before touching §6.4 / §6.6.

---

## 0. Why this document exists

The protocol revision itself has pre-registration consequences. Every change to §6.4 / §6.6 erodes credibility ("why didn't you lock it right the first time?"); every change made *after* X is locked is fatal. So before touching anything:

1. We snapshot both reviews verbatim. Done.
2. We extract the union and intersection of findings. This document.
3. We propose a single atomic REV1 commit covering the must-fix items, with all rationale traced back to the reviews.
4. **User approves the plan in writing**, then we execute as one commit.

Doing this piecemeal — fixing one issue, pushing, then fixing the next — would itself be a credibility hit (reviewers see a chain of "oh wait, also …" commits). One atomic REV1 commit, signed and externally timestamped, is the only defensible path.

---

## 1. Convergent findings (both reviewers agree → high confidence)

### 🛑 Tier 1 — desk-reject blockers, must fix before A_main hits step 60000

| ID | Finding | GPT-5.5 | Claude | Fix path |
|---|---|---|---|---|
| **C1** | `paired_CV_A` is single-arm rolling-window CV; the headline number `rel_diff` is between-arm full-val. **Dimensional mismatch.** | Q1 (DESK REJECT line) | Q1 #1 (DESK REJECT line) | Replace with two-seed paired-difference SD: `sd((A_seed1 − A_seed2)) / mean(A)`. **Cheap if `sigma_seed123` + `sigma_seed456` are usable** (configs are identical except seed; verified by diff). |
| **C2** | Decision rule has no branch for `rel_diff < 0`. A strongly-negative result (C beats A) maps to "V6 narrative validated" — backwards. | Q3 ("HARKing hole") | Q3 ("HARK-fatal", "two desk-reject-class flaws") | Rewrite trichotomy on `|rel_diff|`; add explicit `rel_diff ≤ −X` branch labeled "V6 narrative falsified — paper reframed as σ-norm-only." |
| **C3** | Guard 3 in `lock_effect_size_threshold.py` whitelists `metrics.jsonl` (line 219) and globs only 4 patterns; misses WandB, tensorboard, stdout logs, sibling dirs. **Live trajectory peek is not detected.** | Q2 (filename list) | Q2 (filename list, line cited) + Attack 2 | Remove whitelist. Snapshot C_uniform's `metrics.jsonl` SHA256 + line count at lock time. Forbid concurrent C_uniform training, OR require operator to set up blinded logging. |

### ⚠️ Tier 2 — must fix before paper submission (not before A_main 60K)

| ID | Finding | GPT-5.5 | Claude | Fix path |
|---|---|---|---|---|
| **C4** | AR(1) lag-1 detrend is wrong for a non-stationary trajectory (training trend dominates). `N_eff` overestimated → false `no_confound`. | Q4 | Q4 #1 (concrete failure mode: ρ₁=0.3 lag-2=0.5 → 2.7× overstated) | Add detrend (linear or moving-average baseline) before computing ρ; OR switch to block-bootstrap N_eff. |
| **C5** | `A_pair_uniform_spot` likely doesn't exist → §6.4 is dead code → Risk 4 unaddressed. | Q4 | Q4 ("§6.4 is dead code") | Either run the spot (~0.5 GPU·day), OR explicitly scope-out in abstract+limitations: "Risk 4 not tested." |
| **C6** | Decoder ceiling not computed. "Near-ceiling" / "remaining headroom" language is unsupported. | Q5 | Q5 ("delete every 'near-ceiling' / 'remaining headroom' phrase") | Run `eval_gt_latent_decoder_ceiling_clip3.py` (~30 min, 1 GPU), OR delete every such phrase before submission. |
| **C7** | SHA256 race condition in `lock_effect_size_threshold.py`: file is parsed at T₀, hashed at T₁; intervening writes break the link. | Q6 | Q6 #1 ("Real bug") | Read file as bytes once → hash bytestring → parse from bytestring. |
| **C8** | Method-D selection criterion is not pinned in §6.6 or the lock script. Subconscious tweak after seeing A_main = leak. | (implicit in Q2) | Q7 Attack 3 | Embed Method-D config SHA256 + window/smoothing params into `EFFECT_SIZE_LOCKED.md`. Or fold Method D into the lock script (run + lock atomically). |
| **C9** | Force-push amnesia: gitee allows force-push by default; the lock-in commit's permanence is conventional, not enforced. | Q6 | Q6 + Q7 Attack 4 | GPG-sign the lock commit + OpenTimestamps stamp on the commit hash. OR push to an immutable mirror (Zenodo DOI, public OSF preregistration). |

### 🟢 Tier 3 — can be left as paper-side scope notes

| ID | Finding | Source | Fix path |
|---|---|---|---|
| **C10** | `0.10` floor + `3·CV` slope are individually defensible but **cannot be defended together** as a single piecewise formula (one branch always dominates). | Claude Q1 #3 | Either drop floor or drop slope. Recommend: keep slope `3·SE_diff` (after C1 fix gives a real SE), drop floor. Document the `< 5%` regime explicitly. |
| **C11** | `2X` boundary is prose-driven, not statistical. | Both Q3 | Replace with `X + 2·SE_diff` (CI-based) after C1 lands. |
| **C12** | Guard 5 (config sanity) is too permissive; allows `best_metric=None`. | Claude Q6 | Tighten to require `best_metric == "val_select_score"`. |
| **C13** | Manual `EFFECT_SIZE_LOCKED.md` edits (§6.6.7 deviation path) are not script-checked. | Claude Q7 Attack 4 | CI hook: re-run lock script on `--check` mode; fail if SHA mismatches. |
| **C14** | Window cherry-pick: window `[40000, 60000]` was authored after some training had occurred. | Claude Q7 Attack 1 | Document the lock-time decision in REV1 changelog (acknowledges, does not fix retrospectively). For future ablations: derive window from config (`[total_steps × 0.66, total_steps]`). |

---

## 2. Divergent findings (one reviewer flagged, the other didn't or disagreed)

| Finding | Reviewer | Other reviewer | Verdict |
|---|---|---|---|
| 200K continuation is "researcher-degrees-of-freedom escape valve" | Both | Both | Agree. Pin: one-shot, same X, same Method-D, mandatory reporting of both 120K and 200K. |
| Force-push tamper resistance | Both | Both | Agree (already C9). |
| `0.10` floor justification (clinical / image-quality) | GPT-5.5 only | n/a | Genuine gap. Domain anchor missing. PET reconstruction has no canonical 10% threshold; this is a vulnerability we can't easily fix retroactively. **Best mitigation**: report on `|rel_diff|` AND on absolute `chain_normal_mse` AND on PSNR-clip3 as a multi-metric panel; let reviewer pick their own threshold. |
| Attack 5 (early-stop A_main at favorable CV) | Claude only | not raised | Real risk. Easy fix: require `N ≥ 20` and `A_main has reached configured total_steps` (not just `≥ step_max`). |

---

## 3. Proposed REV1 plan (4 sub-commits in one atomic push)

The natural temptation is to fix everything in one giant commit. Bad — too risky to review and rollback. Instead, REV1 is a single push of 4 logically separate but commit-grouped changes:

### REV1.1 — `lock_effect_size_threshold.py` hardening

- Remove `metrics.jsonl` whitelist (C3).
- Snapshot C_uniform's `metrics.jsonl` SHA256 + line count + WandB run state at lock time, written into `EFFECT_SIZE_LOCKED.md` (C3).
- Read metrics file as bytes once → hash bytestring → parse from bytes (C7).
- Embed Method-D config SHA256 into the lock document (C8).
- Tighten Guard 5: require `best_metric == "val_select_score"` exactly (C12).
- Tighten Guard 4: require `N ≥ 20` (Claude Attack 5 mitigation).
- Add `--check` mode that re-runs computation on existing lock file and asserts SHA matches (C13).

### REV1.2 — `paired_diff_judge.py` correction

- Detrend `val_select_score` series (linear fit residual) before computing ρ (C4).
- Add block-bootstrap N_eff as alternative output; flag if AR(1) and bootstrap disagree by >2× (C4).
- No threshold change (0.10 stays).

### REV1.3 — `POST_V6_NEXT_STEPS.md` §6.6 REV1

- §6.6.1: replace `paired_CV_A` formula with `paired_SD_AA = sd(A_seed1 − A_seed2)/mean(A)` from `sigma_seed{123,456}`. **Pending operator confirmation that these runs are usable.** (C1)
- §6.6.1: keep slope, drop floor (C10). New formula: `X = 3 · SE_diff` where `SE_diff = paired_SD_AA / √N_eff`.
- §6.6.3: rewrite trichotomy on `|rel_diff|` with explicit `rel_diff ≤ −X` branch (C2).
- §6.6.3: pin 200K continuation rules (which ckpt, no re-locking, mandatory both-numbers reporting).
- §6.6.7: clarify deviation path requires script `--check` re-run.
- Add §6.6.8: "REV1 changelog" listing what changed and why (cite both peer reviews by file path).

### REV1.4 — `POST_V6_NEXT_STEPS.md` §6.4 + §6.6.5 scope adjustments

- §6.4: if Q1 to operator confirms `A_pair_uniform_spot` not running, add explicit "Risk 4 not tested" scope statement (C5).
- §6.6.5: if Q5 to operator confirms decoder ceiling not computed, add "Decoder ceiling deferred to revision" statement; remove all "near-ceiling" / "remaining headroom" prose (C6).

### REV1.5 (separate, ~1 GPU·hour to operator) — supplemental experiments

- Run `eval_gt_latent_decoder_ceiling_clip3.py` (~30 min) to get the ceiling number (C6).
- Run `A_pair_uniform_spot` if it doesn't exist (~0.5 GPU·day) to populate Risk 4 (C5).
- These are operator actions, NOT local edits.

### REV1.6 — Tamper-resistance

- GPG-sign the REV1 commit and the eventual lock-in commit (C9).
- Stamp commit hash with OpenTimestamps (`ots stamp`).
- Mirror the lock commit hash to a Zenodo deposit OR a GitHub Release tag (immutable).

---

## 4. Sequencing constraint

```text
[NOW]                            REV1 plan written + reviews saved (THIS COMMIT)
                                 ↓
[USER APPROVAL]                  User reads this doc + reviews → approves
                                 ↓
[OPERATOR Q1+Q4+Q5+Q8 ANSWER]    Confirm: A_main step? sigma_seed runs usable? ceiling?
                                 ↓
[REV1 EXECUTE]                   Single atomic commit (REV1.1–REV1.4) + GPG/OTS sign + push
                                 ↓
[OPERATOR REV1.5]                Run ceiling + A_pair_uniform_spot in parallel with A_main
                                 ↓
[A_main reaches step 60K]        ← REV1 must already be merged before this point
                                 ↓
[lock_effect_size_threshold.py]  Run on REV1 protocol → EFFECT_SIZE_LOCKED.md
                                 GPG-sign + OTS-stamp → push
                                 ↓
[C_uniform full-val]             Per REV1 §6.6 protocol, post-lock
                                 ↓
[Decision]                       Apply REV1 §6.6.3 trichotomy
```

**Hard deadline for REV1**: **before A_main reaches step 60000** (operator Q2 answer determines the exact wall-clock). Once X is locked under the v0 protocol, REV1 becomes scientifically invalid — reviewers will see "they got the formula wrong, ran the experiment, then revised the formula" and reject regardless of the new formula's quality.

---

## 5. What's NOT in REV1

The reviewers raised some criticisms that are correct but cannot be remediated retrospectively:

| Critique | Why we can't fix it now |
|---|---|
| Window `[40000, 60000]` was authored after training started | True. Acknowledged in REV1 changelog as a methodological vulnerability inherited from v0. Future ablations must derive window from config alone before any training. |
| `0.10` floor lacks domain (clinical / image-quality) anchor | True. The closest defensible anchor is "human-perceptual indistinguishability of decoded images at clip-3 PSNR ≤ 0.5 dB"; we don't have such a study and can't run one in the timeline. **Mitigation**: drop the floor entirely (C10). |
| Concurrent A_main / C_uniform training is itself a leak surface | True. **The right fix would be to kill C_uniform now, lock X first, then start C.** This costs whatever GPU-time is in C right now (sunk cost). Up to user / operator. If we accept the sunk cost, decision is to forbid future ablations from starting C before X is locked, and acknowledge this as a v0 limitation in the paper. |

---

## 6. Open questions for user before execution

1. **`sigma_seed123` / `sigma_seed456`**: which one (if either) is the same arm as A_main? If neither, what's the cheapest way to obtain a second-seed run? (added as Q8 to OPERATOR_QUESTIONS.md)
2. **Concurrent A vs C policy**: do we accept the v0 leak surface (and document it as a limitation), or kill C_uniform now and restart it post-lock? Latter costs ~current C wall-clock; former hurts paper.
3. **GPG / OpenTimestamps**: are you set up to GPG-sign commits? If not, the workaround is push to an extra immutable mirror (e.g., GitHub Release tag, Zenodo DOI). Pick one.
4. **Third opinion**: do you want a third reviewer (e.g., Gemini-2.5-Pro) before REV1 executes, or is the GPT-5.5 + Claude convergence sufficient?

---

## 7. Honest self-criticism (the local agent that wrote v0)

The original protocol had two design intents that I (local agent) did not separate cleanly:

- **Intent A**: lock a number that prevents post-hoc threshold tuning (achieved).
- **Intent B**: have that number be statistically meaningful as an equivalence margin (NOT achieved — the reviewers are right).

I conflated these and wrote a formula that achieves A but not B. The right move at the time would have been:

1. Lock a placeholder formula explicitly labeled "tentative — to be replaced after second seed lands." This preserves Intent A.
2. Wait for the second seed.
3. Replace formula with paired-SD-based version.
4. Document the entire lifecycle in §6.6.8.

Instead, v0 locked a too-strong claim ("statistical justification") into a formula that doesn't carry that load. REV1 fixes this by replacing the formula AND adding §6.6.8 changelog explaining the lineage.

The honest framing for paper Methods: "We pre-registered an effect-size threshold using a two-seed paired-difference SD from `sigma_seed123` and `sigma_seed456` (commit `<hash>`, REV1, timestamped via OpenTimestamps `<receipt>`). An earlier draft of the protocol used a single-arm CV proxy; this was identified as dimensionally inconsistent during external peer review (commit `<hash>`, REV0) and revised before any C_uniform full-val data was observed (commit `<hash>`, REV1, immediately following revision). The chain of custody is reviewable in `review/0502/POST_V6_NEXT_STEPS.md` and `review/0503/operator/PEER_REVIEW_*.md`."

This narrative is **defensible** — it's how good preregistration actually works. v0 locked, peer review caught a flaw, REV1 fixed it before any C data was seen. The fact that we have written evidence of the catch + fix is itself a credibility asset.

---

## 8. Recommended user response

Pick one:

- **(A) Approve the plan as written.** I write REV1.1–REV1.4 as one atomic commit + push, after operator answers Q1/Q4/Q5/Q8. ETA: ~2 hours of agent work after operator answers. **Recommended.**
- **(B) Approve with modifications.** Tell me which Tier 1 / Tier 2 items to skip. I rewrite the plan and re-submit for approval.
- **(C) Get a third reviewer first.** Send `PEER_REVIEW_PROMPT.md` + the two existing reviews to Gemini-2.5-Pro. If Gemini agrees, proceed with (A). If Gemini dissents on a major item, re-evaluate. **Costs ~30 min; ~no downside.**
- **(D) Reject the plan; keep v0 protocol.** Not recommended — both reviewers are right.

I'm not going to autonomously execute (A) without your explicit approval, because the protocol changes are themselves pre-registration-relevant and you should sign off on the framing.
