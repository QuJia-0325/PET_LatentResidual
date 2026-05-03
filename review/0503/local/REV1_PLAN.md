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

---

## 9. Operator fact updates + multi-agent review (drafted 2026-05-03 evening)

Since v0 of this document was drafted, two events updated several of its assumptions:

1. **Operator reply** (`OPERATOR_REPLY_local_questions_20260503.md`, commit `e95f86a`) answered all 8 questions from `OPERATOR_QUESTIONS.md`.
2. **Multi-agent review** (`agent1` / `agent2` / `agent3` / `agent4`) audited this plan + `REV1_TOOLING_PLAN.md` against the codebase and operator reply.

This section captures **fact corrections** and **resolved open questions** without rewriting §1–§8. Reviewers approving menu A/B/C/D in §8 should read §9 first.

### 9.1 Stale assumptions corrected by operator reply

| Section / Item | v0 said | Fact (per operator reply) |
|---|---|---|
| **§1 Tier 2 C5** "A_pair_uniform_spot likely doesn't exist" | unknown if config exists | **Config exists** (`review/0502/configs/A_pair_uniform_spot.yaml`, my 02:21 add) but **not yet launched**. Decision: launch is NOT pre-registration-blocking; can be done before paper submission as Risk 4 robustness check. |
| **§1 Tier 2 C6** "Decoder ceiling not computed" | needs ~30 min run | **ALREADY COMPUTED** (operator Q5): full-val n=7403 on 2026-04-22, NORMAL=52.6341 PSNR (D50/D20/D10/D4 = 42.62 / 46.64 / 48.74 / 50.83). Artefacts at `/data_2/qujiaxiang/outputs/PET_LatentResidual/gt_latent_decoder_ceiling_20260422/`. **Fix**: REV1.4 should change "if not computed → run" to "ceiling exists; cite as anchor; FORBID using ceiling to retroactively redefine X or rescue unfavorable A vs C result". |
| **§3 REV1.3** sigma_seed{123,456} as A_main two seeds | "Pending operator confirmation that these runs are usable" | **Q8 answered: option (c) abandoned.** Not just schedule mismatch; output dirs don't even exist on disk. Configs differ in max_steps (50K vs 120K), seed (123/456 vs 42), λ_roll schedule (`0.02→0.25` vs `0→4`), step_weights, pair_loss_weights, and best_metric_terms. **Implication**: REV1.3 has a hidden hard dependency on a fresh `A_seed=43` run (~1.5–2 GPU·days). REV1.3 cannot land a real `paired_SD_AA = sd(A_seed1 - A_seed2)/mean(A)` until A_seed=43 finishes. |
| **§3 REV1.5** ceiling run | "~30 min, 1 GPU" | Already done. Drop from REV1.5. |
| **§4 sequencing** "before A_main reaches step 60000" | hard deadline | **A_main has not started.** Operator confirmed (Q1 + Q2). Deadline is now "before A_main is launched"; no wall-clock pressure. Combined with the W1 window-shift (REV1_TOOLING §2, ramp ends at 90K not 60K), the deadline becomes step **90000** if A_main does start, but practically it's "REV1 lands → A_main launches → A_main crosses ≥90K → lock". |
| **§3 Tier 3 C12** "Tighten Guard 5 to require `best_metric == val_select_score`" | wrong assumption about config schema | **Config writes `best_metric: val_multi_objective`** (verified at `review/0502/configs/A_control.yaml:88`). The string `val_select_score` is the *metrics-row key* the trainer writes, not the config field. **Fix**: Guard 5 should assert `best_metric == "val_multi_objective"` (config-side) AND `"val_select_score" in metrics_row.keys()` (data-side). Both must hold. |

### 9.2 Convergent agent corrections

Four agents reviewed independently. Convergent recommendations (≥2 agents):

| ID | Source | Correction |
|---|---|---|
| **F2** | agent1 + agent2 + agent3 | **REV1_TOOLING C6 (window) and §3 REV1.3 (formula) both edit `POST_V6_NEXT_STEPS.md §6.6.1`.** Two consecutive §6.6.1 revisions = piecemeal credibility hit (cf. §0 of this doc). **Resolution**: window shift moves OUT of REV1_TOOLING into a new unified protocol commit **R1** that bundles all §6.6.1 changes (window + formula + decision rule + decoder-ceiling clarification). R1 lands as ONE atomic protocol patch. |
| **F8** | agent4 + agent1 | **REV1.3 sequencing depends on A_seed=43.** Two options: **(X)** land R1 only after A_seed=43 completes (clean history, +1.5–2 days but parallel-able with operator launching A_main later); **(Y)** land R1 with placeholder formula now, fill in numbers when A_seed=43 returns (avoids gating REV1, but leaves a "placeholder" commit in pre-registration history — credibility risk). **Recommendation**: choose (X). |
| **F5** | agent1 + agent3 + agent4 | **REV1.1 Guard 3 only blocks C-fullval-launch-time.** Operator can train C_uniform pre-lock and observe `metrics.jsonl` / WandB / stdout, then selectively decide which ckpt to evaluate post-lock. **Resolution**: REV1.1 adds a wrapper precondition (managed by `run_c_uniform_full_val.sh` per REV1_TOOLING C5) that operator MUST run `precheck` before `bash run_ablation.sh C` starts training, and `precheck` refuses if `EFFECT_SIZE_LOCKED.md` is not on canonical remote. (See `REV1_TOOLING_PLAN.md §9.3` C5.0.) |

### 9.3 Updated REV1 commit structure (post §9 corrections)

This supersedes §3 of this doc:

```
[A] REV1_TOOLING C1–C5 (mechanical, no protocol):
    schema compat, URL autodetect, ckpt naming, env capture, C-wrapper.
    Lands first, no menu A/B/C/D dependence. Per REV1_TOOLING §9.3.
    ↓
[B] User approves §8 menu choice (A/B/C/D).
    ↓
[C] Operator launches A_seed=43 (~1.5-2 GPU·days, parallel with sanity).
    ↓ (~1.5-2 GPU·days)
[D] R1 — unified protocol patch:
    - §6.6.1 window:      [40K,60K] → [90K,120K] or [95K,120K]    [from REV1_TOOLING C6]
    - §6.6.1 formula:     paired_CV_A → paired_SD_AA               [REV1.3 + real A_seed=43 data]
    - §6.6.1 floor:       drop                                     [REV1.3, was C10]
    - §6.6.3 trichotomy:  add rel_diff ≤ -X branch                 [REV1.3, was C2]
    - §6.6.5 ceiling:     "exists; do NOT use to redefine X"       [REV1.4, post §9.1]
    - §6.4 Risk 4:        operator decision pending                [REV1.4, post §9.1]
    - §6.6.8 changelog:   v0 → REV1 lineage with operator+agent refs
    Lands as ONE commit, GPG-signed + OTS-stamped.
    ↓
[E] R2 — REV1.1 hardening (paired_diff_judge.py detrend, lock script Guard tightening).
    Per REV1.2 + C7/C12/C13 in original §3.
    ↓
[F] Operator launches A_main (under R1 + R2).
    A_main runs 120K, crosses new lock window upper bound.
    ↓
[G] Lock-in: run lock_effect_size_threshold.py → EFFECT_SIZE_LOCKED.md.
    GPG-signed + OTS-stamped + pushed.
    ↓
[H] Operator runs C_uniform under run_c_uniform_full_val.sh wrapper (R1's C5.0 precheck enforces).
    ↓
[I] Decision per R1 §6.6.3 trichotomy.
```

### 9.4 Updated cost estimate

| Phase | v0 estimate | Revised estimate (per §9) |
|---|---|---|
| REV1.1 + REV1.2 (script changes) | ~2h agent | Same; absorbed into REV1_TOOLING C1–C5 |
| REV1.3 (formula change) | "0 GPU·day if sigma_seed usable" | **~1.5–2 GPU·days** for fresh A_seed=43 (sigma_seed unusable per §9.1); parallel with sanity → ~0 wall-clock added |
| REV1.4 ceiling run | 30 min | **0** (already done) |
| REV1.5 A_pair_uniform_spot | ~0.5 GPU·day | Same; can defer to revision if Risk 4 not in main claim |
| Total wall-clock REV1 → A_main launch | ~1 day | **~2 days** (A_seed=43 dominates; parallelizable) |
| Total GPU·days | ~1.5 | **~3** (A_seed=43 + A_main + C_uniform + optional A_pair_spot) |

### 9.5 What changed in §1–§8 implicitly

These items in §1–§8 are now **partially superseded** by §9. The original numbering is preserved for traceability; the corrections live here.

- **§1 Tier 2 C5** — A_pair_uniform_spot status: config exists, not launched; decision deferred to user (acceptable scope-out)
- **§1 Tier 2 C6** — decoder ceiling: ALREADY DONE; remediation reframed (don't run, just don't abuse)
- **§1 Tier 3 C12** — Guard 5: assert `best_metric == "val_multi_objective"`, NOT `val_select_score`
- **§3 REV1.3** — paired_SD_AA needs A_seed=43 (1.5–2 GPU·days); choose option X (block on data) per F8
- **§3 REV1.5** — ceiling run: REMOVED (done)
- **§4 sequencing** — deadline 60K → 90K (or 95K); A_main has not started → no immediate clock pressure
- **§6 Q1** — answered: sigma_seed{123,456} are option (c) abandoned; need fresh A_seed=43
- **§8** — user choice A/B/C/D applies to **R1** (the unified protocol commit), not the original 4-sub-commit structure

> **Note (2026-05-03 late evening)**: §9 was followed by a second round of multi-agent review. See **§10** for the round-2 corrections that refine §9.

---

## 10. Round-2 multi-agent review absorption (drafted 2026-05-03 late evening)

The same four agents re-reviewed v0.2. They confirmed §9's corrections are sound, then surfaced 4 new convergent findings + 7 single-agent improvements. Full audit-trail-quality details live in `REV1_TOOLING_PLAN.md §10`; this section captures the protocol-layer impact.

### 10.1 Findings affecting REV1_PLAN (cross-reference table)

| Round-2 ID | What it changes in this plan | Cross-ref |
|---|---|---|
| **F9** A_pair_uniform_spot schedule confound | §1 Tier 2 C5, §3 REV1.5, §6 Q9 (new): max_steps=60000 means same warmup_ratio/ramp_ratio yields phase boundaries [0,15K]/[15K,45K]/[45K,60K], not [0,30K]/[30K,90K]/[90K,120K]. **At step=45K, A_main is in ramp 25%, A_pair_spot is in Phase III.** Same step ≠ same phase → cannot do same-step paired comparison without aligning. Decision options: (a) raise spot max_steps to 120000, (b) explicit warmup_steps=30000/ramp_steps=60000 override, (c) scope-out Risk 4 in paper limitations. | `REV1_TOOLING §10.1 F9` |
| **F10** R1 must split into R1a + R1b | §3 REV1.3, §4 sequencing: "lock formula after seeing A_seed=43 data" is the same pre-registration anti-pattern v0 had. R1a (skeleton: window, formula structure, metric, decision rule, A_seed=43 yaml spec, Method-D params, ceiling-anchor-only clause, §6.6.8 changelog) lands BEFORE A_seed=43 launches. R1b (numeric paired_SD_AA + X + EFFECT_SIZE_LOCKED.md) lands AFTER A_seed=43 completes via mechanical lock script. | `REV1_TOOLING §10.1 F10`, `§10.3` |
| **F11** C lock-gate must be hard gate inside `run_ablation.sh` | §1 Tier 1 C1 (lock script alone is insufficient): need bash-level `require_lock_pass()` mirroring existing `require_sanity_pass()`. Operator manual precheck is convention, not enforcement. | `REV1_TOOLING §10.1 F11`, new C5b |
| **F12** paired_SD_AA metric explicit | §3 REV1.3: paired_SD_AA must be on `val_chain_normal_mse` (single-objective, same as headline rel_diff). Using `val_select_score` (Method-D composite) would re-introduce v0's dimensional inconsistency at smaller scale. | `REV1_TOOLING §10.1 F12`, locked in R1a |
| **S11** Guard 6 ceiling-string ban | §1 Tier 3 (new C13): lock script refuses lock docs containing `ceiling`/`headroom`/`52.6341` strings. Defends Claude Q5 attack (post-hoc ceiling normalization). | `REV1_TOOLING §10.2 S11`, C5d |
| **S13** verify_paired_seed_configs.py | §3 REV1.3: diff A_seed=42 vs A_seed=43 yaml, asserts only `seed`/`run_name`/output-dir-related fields differ. SHA256 stashed for R1b lock doc. | `REV1_TOOLING §10.2 S13`, C5c |
| **S14** R2 parallel with A_main | §4 sequencing: REV1.1/REV1.2 hardening doesn't gate A_main launch; can land parallel with training. **Saves ~1 GPU·day on critical path.** | `REV1_TOOLING §10.2 S14` |
| **S15** §8 menu text out of date | §8: A/B/C/D refer to v0 4-sub-commit. Per §9.3 it became R1 (single). Per §10.3 it became R1a + R1b. Need to update §8 verbatim. **Done in §10.4 below.** | `REV1_TOOLING §10.2 S15` |
| **S16** Window 90K vs 95K | §3 REV1.1 / §6 Q5: math verified. EMA decay=0.9999 → half-life ≈ 6931 steps. 5K buffer = 72% of one half-life → ~39% reduction in residual EMA non-stationarity. Window [90K,120K] gives 75 eval points; [95K,120K] gives 62 → 95K imposes **10% SE penalty**. **Recommendation: 90K**, because the actual non-stationarity is in the rollout schedule λ_roll which is deterministic and exactly 0 in Phase III, not the model EMA. User has final say. | `REV1_TOOLING §10.2 S16` |
| **S17** Sanity B completion cost | §9.4 cost table missing line: **+0.1–0.2 GPU·day** for finishing the partial sanity run before A_main launches. Updated table below. | §10.3 cost table |

### 10.2 Updated commit topology (replaces §9.3)

```
HEAD → C1+S12 → C2 → C3 → C4 → C5 → C5b (F11) → C5c (S13) → C5d (S11)
                                                            ↓ [B] approve
                                                            ↓
                                                        R1a (skeleton, no data)
                                                            ↓ [C] launch sanity-B + A_seed=43 + A_main
                                                            ↓
                                                            R2 (parallel with A_main)  ← S14
                                                            ↓ [F] both reach 120K
                                                            ↓
                                                            R1b (numeric lock)
                                                            ↓ [H] C_uniform via run_ablation.sh C
                                                                  (refused unless R1b on canonical)  ← F11
                                                            ↓ [I] decision per R1a §6.6.3 trichotomy
```

`REV1_TOOLING_PLAN §10.3` has the full annotated chain.

### 10.3 Cost table v0.3 (updates §9.4)

> **Updated 2026-05-03 night per Q5 decision**: A_seed=43 launches **serial after A_main** (single-GPU on operator host), not parallel. Wall-clock for end-to-end decision grows by ~1.5–2 days vs the parallel option originally drafted in this section.

| Phase | Wall-clock | GPU·days | Critical path? |
|---|---|---|---|
| C1–C5d (tooling) | ~2h agent | 0 | Yes |
| `MULTI_AGENT_REVIEW_RECORD.md` | ~30 min agent | 0 | Yes (gates R1a) |
| Sanity B completion | ~0.5 day | 0.1–0.2 | Yes (gates A_main) ← S17 |
| R1a (skeleton commit) | ~1h agent | 0 | Yes (gates [C]) |
| A_main 120K (= A_seed=42) | ~1.5–2 days | 1.5–2 | Yes |
| R2 hardening (parallel with A_main) | ~2h agent | 0 | No (parallel, S14) |
| A_seed=43 (serial after A_main, per Q5) | ~1.5–2 days | 1.5–2 | Yes |
| R1b (numeric lock) | ~10 min agent | 0 | Yes |
| C_uniform 120K | ~1.5–2 days | 1.5–2 | Yes |
| A_pair_uniform_spot aligned (per F9 option (a) per Q1) | ~1.5–2 days | 1.5–2 | No (Risk 4 robustness; queues opportunistically when GPU free) |
| **Total wall-clock REV1 → decision (serial path)** | **~7–8 days** | — | — |
| **Total GPU·days (incl. A_pair_spot aligned)** | — | **~5–7** | — |

Compared to the parallel-launch version (~5–6 days, ~3.5 GPU·days excl. A_pair): serial adds ~1.5–2 days wall-clock and pulls A_pair_spot back to a full 120K rerun (option (a) per Q1) for ~1.5–2 GPU·days. The trade-off is justified by single-GPU host availability + Risk 4 robustness preservation.

### 10.4 §8 menu text update (per S15)

The §8 menu's "REV1.1–REV1.4 as one atomic commit" is now misleading. The corrected menu (read §8 with this overlay):

- **(A) Approve the v0.3 plan as written.** I (the agent) execute the C1–C5d commit chain (REV1_TOOLING §10.3), then write R1a, then wait for A_seed=43 + A_main to reach 120K, then run the lock script to produce R1b, then C_uniform launches under hard-gate. **Recommended.** ETA: tooling + R1a in ~3h agent work; full pipeline ~5–6 days wall-clock, ~3.5 GPU·days.
- **(B) Approve with modifications.** Tell me which round-2 items to skip. **Skipping F9/F10/F11/F12 reintroduces specific desk-reject vulnerabilities** documented in REV1_TOOLING §10.1. S11–S17 are individually skippable with measurable risk (S11 = Claude Q5 vulnerability returns; S13 = paired-seed config drift undetectable; etc.).
- **(C) Get a third reviewer first.** §9.3 reviewer round 2 is itself the third reviewer. Going to a fourth is diminishing returns; recommend (A).
- **(D) Reject; keep v0 protocol.** Reintroduces all 14 convergent findings from §1 + 4 round-2 + sigma_seed unusable problem. Not viable.

The §8 menu text body is preserved verbatim above for traceability; **this §10.4 overlay supersedes it**.

### 10.5 Items in §1 / §3 / §6 / §8 superseded by §10

| Section | Item | Superseded by | Reason |
|---|---|---|---|
| §1 Tier 2 C5 | "Decision: launch is NOT pre-registration-blocking" | §10.1 F9 row | Schedule confound makes spot launch non-trivial; needs explicit decision |
| §3 REV1.3 | "paired_SD_AA on val_select_score" implicit assumption | §10.1 F12 row | Must be on val_chain_normal_mse |
| §3 REV1.3 | "R1 lands after A_seed=43" (§9 F8) | §10.1 F10 row | R1a before, R1b after |
| §4 sequencing | R2 sequenced before A_main | §10.1 S14 row | R2 parallel |
| §6 Q5 | window 90K vs 95K (open) | §10.1 S16 row | math verified, recommend 90K, user picks |
| §6 (no Q9 yet) | — | §10.1 F9 (new Q9): A_pair_uniform_spot align-or-scopeout | New required user input |
| §8 | menu text | §10.4 | S15 |
| §9.3 commit chain | R1 single commit | §10.2 (R1a + R1b) | F10 |
| §9.4 cost table | missing sanity B + R2-parallel savings | §10.3 | S14 + S17 |

### 10.6 Confirmed user decisions (2026-05-03 night) — mirrors `REV1_TOOLING_PLAN.md §10.6`

User confirmed all open questions raised after the round-2 absorption. The full decision table lives in `REV1_TOOLING_PLAN.md §10.6` (single source of truth); the protocol-side highlights below summarize what's now frozen for this plan.

| Decision | Effect on this plan |
|---|---|
| **Window = 90K** | §3 REV1.1 freezes to `[90000, 120000]`; cite 75 eval points and λ_roll-stationary-in-Phase-III argument in R1a §6.6.1.1 rationale (λ_roll holds constant at 4.0 throughout [90K,120K] per `A_control.yaml:22-26`, so schedule contributes zero non-stationarity; only residual EMA drift remains, whose 39% reduction at 95K does not offset the 10% SE penalty from losing 13 eval points). |
| **A_pair_uniform_spot → option (a)**: rerun with `max_steps=120000` | §1 Tier 2 C5 + §3 REV1.5: yaml lives in R1a commit; rerun queues opportunistically (no critical-path block); paper Risk 4 robustness preserved. |
| **MULTI_AGENT_REVIEW_RECORD.md = yes, before R1a** | New file written between C5d and R1a; cited in R1a §6.6.8 changelog as primary source for round-1 + round-2 review history. |
| **A_seed=43 yaml → option (i)**: physical yaml in R1a commit | §3 REV1.3: R1a generates `A_seed43.yaml` from `A_control.yaml` overriding only `seed`/`run_name`/`output_dir`; verify_paired_seed_configs.py (C5c) runs at launch [E] to confirm no drift; SHA256 stashed for R1b embedding. |
| **GPG / OTS → plain commit + gitee push as time anchor** | R1a / R1b are plain `git commit -s` followed by immediate `git push gitee foc_lite_hop0`. Lock-script env-capture (C4) records git SHA + gitee push timestamp from `git log gitee/foc_lite_hop0`. §7 lineage narrative still defensible: gitee push is third-party-hosted weak time anchor. |
| **Launch order → serial (A_main → A_seed=43)** | §4 sequencing rewritten: sanity B → A_main 120K → A_seed=43 120K → R1b → C_uniform. R2 still parallel with A_main (agent-only). §10.3 cost table updated above. |
| **Method-D parameter inventory → lock at K=10, metric-key=val_select_score, both --include-* flags off** | R1a §6.6.1 includes a sub-block "Method-D parameter lock". Note the deliberate metric split: Method-D selector uses `val_select_score`, but paired_SD_AA / headline rel_diff use `val_chain_normal_mse` per F12. Both are simultaneously locked in R1a. |
| **C5b 3-check hard gate kept strict** | C lock-gate refuses unless: (1) `EFFECT_SIZE_LOCKED.md` exists, (2) its commit is reachable from `gitee/foc_lite_hop0`, (3) `LOCKED_PROTOCOL_VERSION == "R1b_locked"`. Pre-registration's "un-published ≠ locked" principle preserved. |

**Implication for §8 menu (overlay update)**: option (A) "Approve as written" now means the v0.3 plan with serial launches + plain-commit + option-(a) A_pair + 90K window + Method-D K=10 lock. Wall-clock ~7–8 days, ~5–7 GPU·days total (incl. aligned A_pair_spot rerun).
