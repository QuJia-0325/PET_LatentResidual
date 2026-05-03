# PEER REVIEW — GPT-5.5 xhigh (2026-05-03)

**Reviewer**: GPT-5.5 xhigh (Optimizing tool selection mode; Codex MCP not available, ran as local read-only auditor with `experiment-audit` / `research-review` / `result-to-claim` skills loaded)
**Target**: §6.4 + §6.6 of `review/0502/POST_V6_NEXT_STEPS.md` + `paired_diff_judge.py` + `lock_effect_size_threshold.py`
**Prompt**: `review/0503/local/PEER_REVIEW_PROMPT.md`
**Captured**: 2026-05-03 by user, paste-back from external session
**Verdict**: 🛑 **DESK REJECT**

---

## Q1 — LOCKED formula audit

Not defensible. The desk-reject failure is `review/0502/POST_V6_NEXT_STEPS.md` L656–L658: `X` is computed from `A_main` alone, then used to decide a between-arm claim. A single-arm rolling CV is not an equivalence margin, not an MDE, and not a valid proxy for `sd(C-A)/mean(A)`. The `3σ` language is cosmetic here because the series is autocorrelated, schedule-driven, and checkpoint-selection contaminated. The 10% floor is also unjustified as a domain equivalence margin; if it dominates, the protocol is silently saying "10% chain MSE is negligible" without clinical, image-quality, or headroom evidence. This is a fundamental flaw.

## Q2 — Blinded execution sequence

The sequence is not blind. `review/0502/POST_V6_NEXT_STEPS.md` L676 lets `A_main` and `C_uniform` run to completion before `X` is locked, and L733–L759 explicitly permits A-vs-C training-trajectory monitoring. That is C leakage. Guard 3 is weak: `lock_effect_size_threshold.py` L214–L219 misses `metrics.jsonl`, WandB summaries/history, TensorBoard events, `first_hop_224_val_clip3_eval.csv`, `summary.json`, `results.json`, `fullval_results.json`, `full-validation.json`, `MethodD_selection.md`, and any eval copied outside `--c-uniform-output-dir`. Step 4 proves only that one markdown file was committed before some detectable C full-val artifacts; it does not prove blindness.

## Q3 — Decision-rule trichotomy

The trichotomy has a serious HARKing hole. `review/0502/POST_V6_NEXT_STEPS.md` L721 maps all `rel_diff < X` to "C ≈ A", so a large negative result where `C_uniform` is better than `A_main` is incorrectly swallowed as equivalence. That needs a locked branch, e.g. `rel_diff <= -X`: "uniform beats V6; middle-heavy weighting is harmful under this protocol." The 200K continuation in L722 is only acceptable if it is one-shot, same frozen `X`, same frozen ckpt-selection/full-val rule, and all intermediate results are reported. The `2X` boundary in L723 is prose-driven, not statistical.

## Q4 — paired_diff_judge.py + 0.10 threshold

The paired-diff script is useful but overclaims. The lag-1 correction `paired_diff_judge.py` L183–L196 is only reasonable for roughly AR(1) residuals; rolling validation windows, training drift, schedule phase changes, and higher-order autocorrelation can make `N_eff` too large and falsely certify "no confound." Use block bootstrap/Newey-West or pre-specified non-overlapping windows. The 0.10 Risk-4 threshold rationale `review/0502/POST_V6_NEXT_STEPS.md` L619–L624 is directionally sensible but still arbitrary. If `A_pair_uniform_spot` is not run, the honest paper statement is: "Risk 4 was not tested; conclusions are restricted to the current pair-heavy setup." Stronger attribution requires the supplemental run.

## Q5 — Decoder ceiling auxiliary anchor

No ceiling number, no "near-ceiling" or "X% remaining headroom" claim. The minimum defensible alternative is to remove all ceiling/headroom language and report absolute full-val metrics plus the ablation result only. If computed, decoder ceiling should be a separate descriptive anchor: report `A`, `C`, and ceiling on the same split/metric, maybe include remaining-headroom normalization as an auxiliary table. It must not change `X`, not become a post-hoc denominator, and not rescue an unfavorable `rel_diff`.

## Q6 — Implementation correctness

The inclusive window is fine: the implementation uses `step_min <= step <= step_max`, matching `step ∈ [40000, 60000]`. Float constant assertions are not the main problem. The main problems are enforcement gaps. The script claims Guard 5 verifies the exact run dir `lock_effect_size_threshold.py` L39–L40, but the implementation only checks `best_metric` L429–L433. There is a race: `compute_paired_cv()` reads the live metrics file, then `file_sha256()` hashes it later L440–L458; if training appends during that interval, the hash may not correspond to the data used. Artifact detection is underinclusive. Tamper-resistance is weak unless the lock commit is pushed to a protected, no-force-push branch or externally timestamped; `--no-push` and mutable Gitee history are not reviewer-grade evidence.

## Q7 — Adversarial scenarios

- **Attack 1**: Watch C's `metrics.jsonl`/WandB trajectory before lock, then choose whether to commit, delay, add a deviation note, or abandon the claim | **caught? no** | **patch**: lock `X` before starting C, or run C under blinded logging with no author access until `EFFECT_SIZE_LOCKED.md` is pushed.
- **Attack 2**: Run C full-val before lock but save it as `summary.json`, `clip3_eval.csv`, `fullval_results.json`, or outside the scanned directory | **caught? no** | **patch**: require all C full-val through a wrapper that refuses without a pushed lock file; scan the whole output root and tracker logs, not four filename globs.
- **Attack 3**: Use the grey-zone 200K continuation as an escape valve, then selectively re-lock, rerun, or report only the favorable continuation | **caught? partial** | **patch**: pre-register exactly one continuation, same frozen `X`, same Method-D/full-val rule, no re-locking, and mandatory reporting of 120K and 200K outcomes.

## Q8 — Final verdict

| Axis | Score | Justification |
|---|---|---|
| Pre-registration rigor | 4/10 | It locks a formula, but not blindness; C trajectory leakage is explicitly allowed. |
| Statistical correctness | 3/10 | Single-arm CV is the wrong estimand for a between-arm equivalence claim. |
| Implementation robustness | 5/10 | Useful guards exist, but artifact detection, run-dir verification, live-file hashing, and immutable timestamping are insufficient. |
| Reviewer defensibility | 3/10 | I would not accept "C ≈ A" from this protocol as a credible blinded analysis. |

**VERDICT: 🛑 DESK REJECT**

**TOP FIX**: Lock a domain-justified equivalence margin before any C observation, enforce C full-val through a lock-checking wrapper, and analyze A-vs-C with paired/block-bootstrap or repeated-seed inference rather than `A_main` single-arm CV.
