# Operator questions — pre-C1 round (2026-05-03 night)

**Status**: Round-2 multi-agent review absorbed (see `REV1_TOOLING_PLAN.md §10` and `REV1_PLAN.md §10`). All major decisions confirmed by user. **7 small operator-side facts are needed before agent starts C1 commits and R1a draft.** Most can be answered in 1–2 lines each.

**Reading order**: skim §1 if you only have 5 minutes; read §2/§3 only if you have time before A_main launch.

This is the **second** operator-questions round. The first (`OPERATOR_QUESTIONS.md` Q1–Q8) was answered in `review/0503/operator/OPERATOR_REPLY_local_questions_20260503.md` commit `e95f86a` and is fully absorbed. Items below are **new** facts surfaced by round-2 multi-agent review + my local code verification. Operator already answered Q1–Q8; please don't revisit those.

---

## 1. Blocking C1 commits (need before agent starts) — 2 questions

### Q9. Sanity B current state + path

**Why I need this**: C1 (T1/T2/T3 fixes) ships a `golden-trace test` against a real `metrics.jsonl`. Schema reference is `review/0503/operator/A_sanity_metrics_tail5_schema_reference.jsonl` (5 rows). For a stronger regression test I'd like a longer trace — either the in-progress sanity B `metrics.jsonl` (any number of rows is fine), or just confirmation that the 5-row reference is the canonical schema and we should rely on it alone.

Please answer **one** of:

- **(a)** Sanity B is at step `<X>`; metrics.jsonl is at `<absolute-path>`; safe to copy first ~50 rows for golden-trace test → I cherry-pick into `review/0503/operator/A_sanity_metrics_long_schema_reference.jsonl`.
- **(b)** Just use the existing 5-row reference; sanity B is still running and will eventually be re-tail'd. → I write the test against the 5-row file alone.
- **(c)** Sanity B has finished (sentinel exists); paths to final metrics.jsonl + ckpt-dir are `<...>`. → I use the finished trace.

If sanity B is in a weird state (paused, restarted, crashed), please describe so I can decide whether C1 should wait for it.

### Q10. Operator-host canonical-remote name + how `.review_canonical_remote` is interpreted

**Why I need this**: C2 introduces `.review_canonical_remote` (git-tracked, single line) so `lock_effect_size_threshold.py --remote auto` resolves the canonical remote without hardcoding a URL. **Problem**: my Mac has `gitee = canonical, origin = github mirror`; operator host has `origin = canonical (gitee URL)`. The same single-line file cannot be both `gitee` and `origin` literally.

Two ways to handle this. Please pick **one**:

- **(i) Single-line URL** in `.review_canonical_remote`, e.g.,
  ```
  git@gitee.com:jqu9/PET_LatentResidual.git
  ```
  Resolver scans `git remote -v`, finds the remote with that URL, returns its name. My Mac → returns `gitee`; operator host → returns `origin`. **Recommended.** Single source of truth, no per-host config.
- **(ii) Comma-separated candidate names**, e.g.,
  ```
  gitee,origin
  ```
  Resolver returns first one that exists on the local repo. Simpler but coarser.

If you prefer (ii), confirm the order: `gitee,origin` (my Mac wins) or `origin,gitee` (operator host wins). It doesn't really matter functionally as long as one matches.

---

## 2. Blocking R1a draft (need before R1a commits) — 3 questions

### Q11. A_pair_uniform_spot rerun scope

You confirmed option (a) — raise `max_steps` to 120000 so the schedule aligns with A_main `[0,30K] / [30K,90K] / [90K,120K]`. Please confirm **what else changes**:

- **(a-only)** Only `max_steps: 60000 → 120000`. Every other field of `A_pair_uniform_spot.yaml` (currently uniform-pair-loss config + spot-specific overrides) stays as-is. → I create `review/0502/configs/A_pair_uniform_spot_aligned.yaml` as a 1-field copy.
- **(a-with-cleanup)** max_steps→120000 **plus** clean up the stale comments at lines 8 / 133–135 that still describe a 200K schedule. No semantic change to other fields. → I do (a-only) plus comment cleanup.
- **(a-from-A_control)** Start from `A_control.yaml`, apply only the **minimum spot-specific deltas** (which fields exactly?). → I'd need you to enumerate the deltas, e.g.:
  ```
  pair_loss_weights: [1.0, 1.0, 1.0, 1.0]   # uniform across hops (vs A_control's [2.5, 1.0, 1.0, 1.0])
  pair_dataset_mode: per_prompt_paired       # whatever the spot-specific switch is
  # ...etc
  ```

**If you don't have time to enumerate**: pick (a-only) and I'll diff A_pair_uniform_spot.yaml vs A_control.yaml myself, then ship that diff in R1a for your review before commit.

### Q12. A_main launch command (for R1a §6.6.1 reproducibility recipe)

R1a §6.6.1 will pin the exact launch command so a third party (reviewer / future-you) can reproduce. Please paste the command you intend to run, e.g.:

```bash
cd /<repo-root>
CUDA_VISIBLE_DEVICES=<N> python train_first_hop.py \
    --config review/0502/configs/A_control.yaml \
    --output-dir /data_2/qujiaxiang/outputs/PET_LatentResidual/A_main_<date> \
    [other flags]
```

Just the actual command line. R1a embeds it verbatim.

### Q13. Output-dir convention for A_seed=43

R1a generates `A_seed43.yaml` from `A_control.yaml` overriding `seed`, `run_name`, `output_dir`. For `output_dir`, two options:

- **(i) R1a hardcodes** something like `/data_2/qujiaxiang/outputs/PET_LatentResidual/A_seed43_120K_<YYYYMMDD>/` (you fill in the date when you launch). → Concrete, reviewable in R1a.
- **(ii) R1a leaves a placeholder** (`output_dir: <set-at-launch>`) and operator fills at launch time. → C5c's `verify_paired_seed_configs.py` allows `output_dir` field to differ between A_seed=42 and A_seed=43 yaml as one of the explicitly-allowed deltas, so this is safe.

Pick one. If (ii), nothing more needed. If (i), tell me the prefix you want.

---

## 3. Soft (helpful but not blocking) — 2 questions

### Q14. GPU availability + estimated A_main launch time

Just for cost-table calibration in `REV1_PLAN.md §10.3`. When (roughly) do you expect to start A_main? Today / tomorrow / next week / "after sanity B finishes which is ETA <X>"? If there's a queue, what's your slot?

**Why I'm asking**: not to apply pressure; just so the plan's "wall-clock ~7–8 days" estimate has an honest start-date anchor.

### Q15. Conference / paper deadline anchor

Is there a deadline driving this work (e.g., conference submission, internal review)? If yes, the date. If no deadline, just say "no hard deadline" and I'll plan accordingly.

**Why I'm asking**: if there's a deadline within 2 weeks, I should probably push back on option (a) for A_pair_uniform_spot (~1.5–2 GPU·days) and propose option (c) scope-out instead. If no deadline, option (a) stands.

---

## 4. Reply format

A short markdown file at `review/0503/operator/OPERATOR_REPLY_pre_C1_20260503.md` with one section per question (Q9–Q15), 1–3 lines each, is plenty. No need for code blocks or long prose.

If a question becomes irrelevant (e.g., sanity B finished while you were typing), just say so and skip.

If anything in `REV1_TOOLING_PLAN.md §10` or `REV1_PLAN.md §10` looks wrong from your operational perspective, **please flag it in the reply** — those §10 sections are still draft until you sign off implicitly via Q9–Q15.

---

## 5. What happens after you reply

1. Agent (me) reads `OPERATOR_REPLY_pre_C1_20260503.md`, updates `REV1_TOOLING_PLAN.md §10.6` and `REV1_PLAN.md §10.6` with your answers (frozen).
2. Agent writes `MULTI_AGENT_REVIEW_RECORD.md` (4 agents × 2 rounds verbatim + adopt/defer/reject column).
3. Agent starts C1 (`fix(scripts): step/global_step compat + LOCKED_PROTOCOL_VERSION`), then C2 → C5d in order.
4. Agent drafts R1a (skeleton commit, no data) and presents for your review.
5. Operator (you) eventually launches sanity-completion → A_main → A_seed=43 (serial per Q5) → R1b numeric lock → C_uniform.

Total agent-side work between now and "ready to launch A_main": ~3–4 hours after your reply lands.
