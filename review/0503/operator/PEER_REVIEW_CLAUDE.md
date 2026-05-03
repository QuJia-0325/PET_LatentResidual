# PEER REVIEW — Claude Opus 4.7 extra-high (2026-05-03)

**Reviewer**: Claude Opus 4.7 extra-high reasoning (local read-only, with `experiment-audit` + `paper-claim-audit` skills loaded)
**Target**: §6.4 + §6.6 of `review/0502/POST_V6_NEXT_STEPS.md` + `paired_diff_judge.py` + `lock_effect_size_threshold.py`
**Prompt**: `review/0503/local/PEER_REVIEW_PROMPT.md`
**Captured**: 2026-05-03 by user, paste-back from external session
**Verdict**: ⚠️ **MAJOR REVISIONS** (with desk-reject risk if Q1 + Q3 not fixed before submission)

---

## Q1 — LOCKED formula audit

The formula is **dimensionally and statistically wrong** in three ways:

1. **Wrong noise estimator (desk-reject blocker).** `paired_CV_A` is `std(val_select_score) / mean(val_select_score)` over A_main alone in `[40000, 60000]`. This is *within-trajectory drift + rolling-window noise of a single arm*, NOT the between-arm paired-difference noise. The headline number `rel_diff = (chain_normal_mse_C − chain_normal_mse_A)/chain_normal_mse_A` is computed from **full-val** (Step 5–6, `--max-slices 0`) on **best-ckpts**, on a **different metric** (`chain_normal_mse`, not `val_select_score`). You are calibrating a between-arm full-val threshold using a within-arm rolling-window CV. The two distributions don't share scale; you can be off by 5–10× in either direction. The principled proxy is `paired_CV_pair = sd((C−A))/mean(A)` from a sister-seed pair (run A twice, compute paired noise from the two seeds).

2. **3σ ≠ 3·CV.** The rationale ("3σ covers 99.7% under Gaussian, physics/RCT standard") confuses *single-observation 3σ* with *3 standard errors of the mean*. The relevant null distribution for `rel_diff` is the *paired-mean's* sampling distribution, whose spread is `SE_diff = SD/√N_eff`, not `SD`. If you genuinely want a 99.7% indistinguishability band, you want `3·SE_diff`, which is several times tighter than `3·CV`.

3. **The 0.10 floor is incoherent with the slope.** When `paired_CV_A < 3.33%` the floor dominates and the formula degenerates to a flat 10%. In the regime the protocol expects (V6 snapshot CV ≈ 5%), `3·CV ≈ 15%`, so the floor *never activates*; conversely when CV ≈ 2% (totally plausible at 200K) the floor *always* dominates. The formula always picks one branch — why not just hardcode that branch? Either drop the floor or drop the slope; you cannot defend both.

## Q2 — Blinded execution sequence

Multiple leak paths survive Guard 3:

- **WandB / tensorboard / metrics.jsonl streaming leak.** `lock_effect_size_threshold.py` Guard 3 globs only `*full_val*`, `*eval*.json`, `*method_d*`, `*selected_ckpt*` and explicitly **whitelists** `metrics.jsonl` (line 219: `suspect = [p for p in suspect if p.name != "metrics.jsonl"]`). C_uniform's training `metrics.jsonl`, WandB summary, tensorboard `events.out.tfevents.*`, stdout logs, `wandb-summary.json`, `summary.csv`, `progress.csv`, anything in tmp, and any `tail -f` on the live job — **all invisible to Guard 3**. Authors can watch C's `val_select_score` converge in real time and never trip a single guard.
- **Window-selection HARK before lock.** §6.6.1 fixes `[40000, 60000]` LOCKED, but if A_main was already past 40K when this section was written, the window choice is itself contaminated by trajectory observation.
- **Method-D top-2 selection leak.** Step 2 happens *before* Step 3. Method D's selection criterion (window, smoothing, tie-breaking) is not pinned in §6.6 or in the lock script. A subconscious tweak to Method D after seeing A_main's trajectory leaks knowledge into the lock-time CV.
- **Subconscious window bias via co-training.** Even without explicit peeking, the pacing of "when do I run the lock script" gets influenced by gut feel about C's progress.
- **Force-push amnesia.** Guard 1 (clean tree) is checked at lock time, but `gitee` allows force-push by default.

**Filenames Guard 3 misses (concrete list):** `metrics.jsonl`, `wandb-summary.json`, `wandb/run-*/files/output.log`, `events.out.tfevents.*`, `progress.csv`, `summary.csv`, `*.tfevents`, `loss.log`, `train.log`, `*.png`, `last.pt`, anything in tmp, anything in a sibling directory not under `--c-uniform-output-dir`.

## Q3 — Decision-rule trichotomy (the second desk-reject blocker)

The rule reads `rel_diff < X → "C ≈ A, σ-norm is the primary lever"`. **Negative `rel_diff` is unhandled and falls into the V6-favorable bucket.** If C_uniform actually *outperforms* A_main (rel_diff = −25%, meaning the σ-normalized uniform-step baseline beats your transport-first middle-heavy V6), the literal LOCKED rule maps that to *"σ-norm is the primary lever — V6 narrative validated."* This is exactly backward. **As written, the protocol cannot distinguish "V6 wins" from "V6 loses badly" — both go in bucket 1. This is the textbook HARK surface and is fatal.** Fix: redefine cells on `|rel_diff|` AND add an explicit `rel_diff < −X` branch labeled "V6 narrative falsified, paper must be rewritten as σ-norm-only."

The "200K continuation" branch is a researcher-degrees-of-freedom escape valve, not a pre-specified analysis. The text says "re-evaluate with the same X" (good), but does not specify (a) which checkpoint to resume from on a tie, (b) whether `paired_CV_A` is recomputed on the longer trajectory, (c) what counts as "still grey" if the new rel_diff lands at exactly 2X.

The 2X boundary is chosen for clean prose; it has no statistical derivation.

## Q4 — paired_diff_judge.py autocorr correction + 0.10 threshold

- **AR(1) lag-1 correction is the wrong model for `val_select_score` time series during training.** The training loss has a strongly *non-stationary mean* (it's monotonically decreasing through any 20K window) and `paired_diff_judge.py` doesn't detrend. Lag-1 ρ on a series with deterministic trend captures the trend, not the residual autocorrelation. `N_eff = N(1−ρ₁)/(1+ρ₁)` will overestimate independence whenever the true correlation extends beyond lag-1. Concrete failure mode: ρ₁ = +0.3 (looks moderate), but lag-2 ρ = +0.5 (because trend dominates) → real N_eff ≈ N/5, script says N_eff ≈ N·0.54 → 2.7× overstated → false `no_confound` verdict.
- **The 0.10 / "defensive false-negative" rationale is post-hoc.** §6.4 says false-negative is desk-reject and false-positive is just disclosure (argues for *low* threshold). But §6.6 also picks 0.10 as the *floor* with the *opposite* rationale. Same number, two different reasoned justifications, two different protocols. Reviewer asks: "Why is 0.10 simultaneously high enough to be defensive in §6.4 and low enough to be a meaningful floor in §6.6?"
- **A_pair_uniform_spot likely doesn't exist.** Operator Q1 admits this is unknown. If never run, **§6.4 is dead code**.

## Q5 — Decoder ceiling

If absent at submission, you cannot use phrasing like "near-ceiling," "X% of remaining headroom," or "approaches the decoder bound." Minimum reviewer-defensible alternative: report `chain_normal_mse` as a raw number, list "decoder ceiling not measured for this submission" as an explicit limitation. If you compute it later, the *only* rigorous integration that doesn't break X is a *separate* sanity-check column; **do not** retroactively renormalize `rel_diff` by the ceiling.

## Q6 — Implementation correctness of lock_effect_size_threshold.py

- **Real bug — SHA256 race condition.** `compute_paired_cv()` calls `load_metrics()` (reads JSONL into memory at time T₀), then later `file_sha256()` re-opens the file at time T₁. If A_main's training process appends rows in `[T₀, T₁]`, the hash covers more bytes than the parsed series. Mitigation: read the file as bytes once, hash that bytestring, then parse from the bytestring.
- **Off-by-one is benign.** Both `step_min` and `step_max` are inclusive.
- **Float comparison is benign.** `0.10` and `3.0` are exact IEEE-754.
- **Guard 3 patterns are insufficient** — see Q2. The deliberate whitelist `if p.name != "metrics.jsonl"` is the single biggest hole.
- **Guard 5 is too loose.** It only checks that A_main config's `best_metric` is in `(None, "val_multi_objective", "val_select_score")` — three permissive options including `None`. Method-D-related config is not checked.
- **Tamper-resistance is conventional, not enforced.** Force-push erases the lock-in commit silently. Required: GPG-signed commit + OpenTimestamps stamp on the commit hash, OR push to an immutable mirror (Zenodo DOI, GitHub Releases asset, public OSF preregistration).

## Q7 — Five adversarial scenarios

- **Attack 1 (window cherry-pick)**: Author observes A_main's WandB curve; sees CV is 1.5% in `[40K, 60K]` but 4% in `[20K, 40K]`. Picks the lower window → smaller X. | **Caught? No.** | **Patch:** Lock window choice in a commit *before* A_main launches OR mandate the window be derived from config alone (e.g., `[total_steps × 0.33, total_steps × 0.5]`).
- **Attack 2 (live C_uniform peek)**: Author tails `C_uniform/.../metrics.jsonl` daily. Guard 3 explicitly whitelists `metrics.jsonl`. | **Caught? No.** | **Patch:** At lock time, snapshot C_uniform's `metrics.jsonl` SHA256 and require `wc -l` to be ≤ a pre-committed bound. Better: forbid *any* concurrent C_uniform training until X is locked.
- **Attack 3 (Method-D criterion drift)**: §6.6 doesn't pin Method D. Author tweaks Method D's selection window/smoothing after observing A_main, justifying it as "smoothing." | **Caught? No.** | **Patch:** Embed Method-D config SHA256 in `EFFECT_SIZE_LOCKED.md`. Or fold Method D into the lock script.
- **Attack 4 (manual edit of EFFECT_SIZE_LOCKED.md)**: §6.6.7 explicitly says "modify by hand in a new commit" is the legal deviation path. Nothing checks the new commit's content against the script's output. | **Caught? No.** | **Patch:** Lock script writes `EFFECT_SIZE_LOCKED.md.sha256` alongside, and a CI hook fails if the file is mutated without a corresponding script re-run + new SHA. Or sign the file with GPG.
- **Attack 5 (early-stop A_main at favorable point)**: Author observes A_main hitting low CV at step 55K, kills the run, locks X with N=15. | **Caught? No** — Guard 4 only requires N ≥ 5. | **Patch:** Require A_main to have actually reached `step_max` (config-declared, not just observed) before lock; abort lock if A_main is not still alive or has not crossed 60K. Better: require N ≥ 20.

## Q8 — Final verdict

| Axis | Score | Justification |
|---|---|---|
| Pre-registration rigor | 4/10 | formula and timestamp are locked, but window-pick, Method-D criterion, and concurrent C_uniform observability are all open leak surfaces. |
| Statistical correctness | 3/10 | paired_CV_A is the wrong noise scale; 3·CV is conflated with 3·SE; trichotomy ignores rel_diff < 0. |
| Implementation robustness | 5/10 | guards are well-structured but metrics.jsonl whitelist + SHA256 race + force-pushable gitee + Method-D-not-pinned leave the protocol porous in practice. |
| Reviewer defensibility | 3/10 | Q3 negative-rel_diff bucket is HARK-fatal; Q1 dimensional mismatch is desk-reject bait. |

**VERDICT: ⚠️ MAJOR REVISIONS** (with desk-reject risk if Q1 + Q3 are not fixed before submission)

**TOP FIX**: Rewrite the decision rule on `|rel_diff|` with an explicit `rel_diff < −X` branch ("σ-norm baseline strictly outperforms V6 → V6 narrative falsified, reframe paper as σ-norm-only"), AND replace single-arm `paired_CV_A` with a two-seed paired-difference SD (run A_main once more at a different seed; that 0.5 GPU·day buys you a defensible noise scale and removes the dimensional mismatch in one shot).

**Secondary musts (any one of these can also sink the paper):**

- Run **or** explicitly scope-out `A_pair_uniform_spot` in the abstract+limitations.
- Compute the decoder ceiling *or* delete every "near-ceiling" / "remaining headroom" phrase from the manuscript.
- Replace AR(1)-lag-1 N_eff with block-bootstrap or detrended-Newey-West.
- Sign the lock-in commit with GPG and stamp it with OpenTimestamps; otherwise reviewers cannot verify the timestamp survives a force-push.
- Pin Method-D config SHA in `EFFECT_SIZE_LOCKED.md`. Pin C_uniform's `metrics.jsonl` byte-count at lock time. Remove the `metrics.jsonl` whitelist from Guard 3.

The protocol's *intent* is honorable and ahead of typical ML papers. Its *execution* has two desk-reject-class flaws (Q1 dimensional, Q3 HARK) and three serious-revision flaws (Q4 stats, Q6 race, Q7 multi-vector leaks). Fix Q1 and Q3 in the next 24 hours; the rest is recoverable in revision.
