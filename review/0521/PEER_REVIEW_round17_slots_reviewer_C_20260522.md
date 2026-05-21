# Round 17-Slots Peer Review — Reviewer C (independent)

- **reviewer name**: **C** (consistent with R15/R16/R17/R17-prep)
- date: 2026-05-22
- branch: foc_lite_hop0 (commit 1feaa8b)
- substrate read independently (no other reviewer drafts seen):
  - PEER_REVIEW_PROMPT_round17_slots_stop_rule_revisit_20260522.md (the prompt)
  - CODEX_TASK_ROUND17_F0_A4_20260522.md (stop rule sources §A4.5, §6 NOT-DO #6, §5 anti-check)
  - REVIEW_INTEGRATION_round17_20260522.md (R17 4/4 consensus wording on A4 form)
  - PEER_REVIEW_ROUND17_REVIEWER_A_20260521.md (cross-check reviewer-A's A4 position)
- mandate per prompt §6: give a clear opinion on the governance decision; do not produce 4 options for user decision fatigue.

---

## TL;DR (Reviewer C verdict)

| Item | Verdict |
|---|---|
| Q1 Should stop rule be revisited? | **REJECT revisit on slot-utilization grounds** — content-based revisit would require new substrate (none presented). |
| Q2 Image_aux sweep EV | **LOW marginal EV** — the single A4-mid probe carries most information; A4-low and A4-high are mostly confirmatory. |
| Q3 Main recommendation | **Plan D (守 Round 17 原计划, 1 slot A4-mid only)** PRIMARY; **Plan E (2-slot, A4-low + A4-mid)** as the MINIMUM defensible deviation if user demands a bracket. |
| Q4 3-slot RAM risk | **N/A under D/E** — 1-2 slot needs no staggering / no dataloader-worker cap. |
| Q5 Sweep priority order | **A4-low (λ=0.02) > A4-high (λ=0.12)** by a wide margin if bracket is forced. |
| Q6 Document update scope | **Only if E or higher.** If sticking to D, no doc changes. If E, narrow updates only (stop rule wording, anti-check `wc -l` ≤ 2). |
| Q7 New biases | **4 new (B83 HIGH, B84-B86 MED-LOW).** Headline: B83 — selective quotation of stop rule, ignoring integration md's stricter "one variant only" wording. |

**Main verdict (one line)**: Hold the Round 17 stop rule. Single A4-mid probe is sufficient; idle slots are not a project cost; "slot 利用率" is a sunk-cost-of-idle-resources fallacy. If user has a content-based reason to expand (which they have not articulated in this prompt), Plan E (A4-low + A4-mid, 2 slot, 1 slot buffer) is the maximum acceptable deviation.

---

## 0. Mechanical verification first

### 0.1 What did R17 consensus ACTUALLY say about A4 form?

Per prompt §6 directive, my opinion turns on what consensus was. I grep-verified every A4-related line in `REVIEW_INTEGRATION_round17_20260522.md`:

| line | exact wording |
|---|---|
| 17 | "All 4 reviewers select Hybrid B+A4 ... with **one** cheap image_aux schedule probe in parallel." |
| 123 | "Review D: at-**most-one** pre-registered A4 probe; do not let it scope-creep." |
| 126 | "A4 = **one** image_aux schedule variant, pre-registered with success criterion ≥ +0.05 dB before launch." |
| 156 | "A4 scope limited to **one** pre-registered variant." |
| 173 | "A4 — **one** image_aux schedule probe, pre-registered with stop rule" |
| 196 | "Optionally launch **one** pre-registered A4 image_aux schedule probe if GPU is idle." |

**Six independent statements in the integration md** specify "one variant only". This is the source-of-truth stop rule. The task md §A4.5 "无论 outcome 不 relaunch v2/v3" is a **tactical sub-rule** (the post-result re-run case), not the full stop rule.

### 0.2 The prompt §1 frames the stop rule incorrectly

Prompt §1 reads only §A4.5 and §6 NOT-DO #6 from the codex task md. It then claims:

> "stop rule 没有禁止的: **预先**起 2-3 个 image_aux variant **并行**跑, 形成 sweep curve"

This is **false** when measured against the integration md (the actual source of consensus). All 6 integration-md statements say "one variant", which excludes 2-3 pre-registered variants by definition.

The cleanest read: the prompt's §1 is a textbook selective-quotation bias — quote the weaker wording of the task md (which says only "no v2/v3 after seeing outcome") and ignore the stricter wording of the integration md (which says "one variant only, pre-registered"). This is **B83 (HIGH)**.

### 0.3 What substrate would justify revisiting?

A legitimate revisit would require one of:
1. **New finding** post-R17 that invalidates the "one variant suffices" judgment (e.g., F0 paired-t shows V18 +0.06 dB is real, narrowing the headline-vs-ablation distinction, raising bracket EV).
2. **New constraint** (e.g., paper venue requirement for response curves).
3. **New cost data** (e.g., slot now cheaper / more available than assumed).

None of these are present in the prompt. The stated reason is "**user 观察 '此时只运行了一个训练? 我认为可以同时运行两个, 最多运行三个'**" — pure resource utilization. This is exactly the failure mode the R17 stop rule was set to prevent.

---

## 1. Q1 — Should the stop rule be revisited?

**Verdict: REJECT revisit on slot-utilization grounds. APPROVE revisit only on content grounds (none presented).**

### 1.1 Literal vs spirit reading

Prompt §1 asks three questions about the §A4.5 wording:

| Q | answer |
|---|---|
| Does §A4.5 literally forbid §2.3 Plan A (3-point parallel)? | Literally, §A4.5 alone is ambiguous (it says "no relaunch v2/v3 after outcome"). |
| Does it forbid in spirit? | Yes. The R17 integration md (the actual consensus document) says "one variant" 6 times. Pre-registered 2-3 variants violate the spirit even if they squeak through §A4.5 alone. |
| Should user-revisit be accepted? | **Only if a content-based reason is articulated.** "Slot utilization" is not a content-based reason. |

### 1.2 Why "slot utilization" alone is insufficient

Idle GPU slots are NOT a project cost:
- Slot hours are paid for regardless of utilization.
- The marginal *real* cost is orchestration: writing yamls, monitoring 3 runs, integrating 3 reports, handling potential cross-run failures, maintaining audit hygiene.
- Idle-slot pressure is a **sunk-cost-of-idle-resources** psychological pattern (B84). It feels wasteful but it is not actually wasteful unless the marginal additional run has positive EV ≥ its orchestration cost.

If the project accepts "slot 利用率不够" as sufficient justification to break consensus, then by induction every R17/R18/... stop rule can be broken whenever GPU slots happen to be idle. This is precisely the **stop rule slippage** failure mode (B85).

### 1.3 What WOULD a legitimate revisit look like?

Hypothetical content-based reasons that would change my position:

| reason | would I approve? |
|---|---|
| "F0 paired-t came back and V18 is decisively non-significant — paper now needs additional positive ablation; expand A4 to 2 points for response curve" | YES (content-based). |
| "Reviewer feedback (paper submission) required a response curve — too late to wait for second submission" | YES (external constraint). |
| "I realized after R17 that V13 → V7 alone doesn't bound the local derivative at 0.04, so the single A4-mid is uninterpretable" | MAYBE (content-based, but should have been raised in R17). |
| "Slots are idle" | NO (resource-based, doesn't change EV calculus). |

The prompt only presents the last one. I therefore reject the revisit.

---

## 2. Q2 — Does image_aux sweep have real paper EV?

**Verdict: MODIFY (the prompt's EV framing). The single A4-mid probe is sufficient; bracket adds marginal-only value.**

### 2.1 Existing information

We have two anchored points:
- V13: λ=0, NORMAL = 36.4943
- V7:  λ=0.04, NORMAL = 36.7810

The 2-point slope across [0, 0.04] is `+7.18 dB / unit λ`. This is the **average** slope over [0, 0.04]; tells us nothing about the local slope at 0.04. The function MUST be nonlinear (cannot keep climbing linearly to infinity), but where it bends is unknown.

### 2.2 Information yield of each candidate probe

| probe | most likely outcome | new info | paper EV |
|---|---|---|---|
| A4-mid (λ=0.08) | (a) saturation: 36.78 ± 0.03; (b) climbing: 36.85-36.95; (c) over-weighted: 36.70 or below | Local derivative around 0.04. Discriminates (a) vs (b) vs (c). | **HIGH** — every outcome is paper-actionable. |
| A4-low (λ=0.02) | Most likely 36.65-36.75 (between V13 and V7) | Mid-point of [0, 0.04]. Confirms monotonicity but adds little local-derivative info. | **LOW-MED** — confirmatory rather than discriminating. |
| A4-high (λ=0.12) | Most likely either ties A4-mid (saturation past 0.08) or below V7 (over-weighting). | Bounds the saturation tail. | **LOW-VERY LOW** — confirmatory bound. |

**The bracket-sweep narrative ("3 points form a curve, paper much stronger") is a sweep-curve fetish (B85).** A nonparametric response curve from 4-5 points adds visual rhetoric but rarely changes the headline claim or its statistical defensibility. Reviewers will accept "we identified +0.287 dB at λ=0.04 via V13-V7 comparison; A4 probe at λ=0.08 tests saturation, results consistent with [a/b/c]" with a single probe just as readily.

### 2.3 The "scope creep trigger" question (prompt §2.2 last sub-question)

The prompt's own §2.2 contains the strongest argument for **keeping** the stop rule:

> "如果 sweep 出来 0.08 显著高于 V7 (A4-mid > V7 + 0.05 dB), 是否会触发 user 想再追 λ=0.06 / 0.10 / 0.16 — 形成 stop-rule 的真违规?"

Yes. Exactly. This is the post-result scope creep the R17 stop rule was designed to prevent. A 3-point sweep doesn't eliminate this risk — it amplifies it, because a 3-point dataset suggests an interpolatable curve and invites λ=0.06 follow-up. Better to do the 1-point probe and write the paper.

### 2.4 If a bracket is forced anyway (content-based justification)

The minimum defensible bracket is `{0, 0.04, 0.08} + (optionally) 0.02`:
- 0 and 0.04 are free (V13, V7 already exist).
- 0.08 (A4-mid) is the highest-EV new probe.
- 0.02 (A4-low) is the second-highest-EV new probe, mostly confirmatory.
- 0.12 (A4-high) is the lowest-EV; almost certainly redundant.

Therefore **Plan E (2-slot, λ=0.02 + 0.08) > Plan A (3-slot, λ=0.02 + 0.08 + 0.12)** on EV alone, independent of governance.

---

## 3. Q3 — 5-plan selection

**Verdict: Plan D primary, Plan E secondary.**

### 3.1 Ranking

| Plan | EV | governance cost | overall |
|---|---|---|---|
| **D** (1 slot, A4-mid 0.08 only) | HIGH (single high-information probe) | None — respects R17 consensus | **PRIMARY** |
| **E** (2 slot, A4-low 0.02 + A4-mid 0.08) | HIGH+MED | Stop-rule revision, but to "≤2 variants pre-registered" — narrowest possible deviation | **DEFENSIBLE if content reason given** |
| **A** (3 slot, λ=0.02 + 0.08 + 0.12) | HIGH+MED+LOW = HIGH | Stop-rule fully broken; 3-slot RAM contention; A4-high low EV | **REJECT** — extra slot doesn't justify the governance precedent + RAM risk |
| **B** (A4-bracket + V14b) | HIGH + MED-LOW (V14b is R17-rejected as low priority because F0 substitutes) | Double revisit (stop rule + V14b precedent) | **REJECT** — conflates two governance decisions |
| **C** (A4-mid + V13b) | HIGH + LOW-MED (V13b cleans 0.02 dB train-config confound — marginal paper value) | Adds a non-A4 variant (V13b) under cover of "A4 slot expansion" — opens new variant axis | **REJECT** — V13-V8 +0.02 dB is below noise threshold for paper-relevant claims; not worth 7 days |

### 3.2 Why D wins

1. Respects R17 4/4 consensus ("one variant").
2. Single A4-mid probe captures most of the information yield (§2.2).
3. Leaves 2 slots as genuine buffer for: emergency relaunch if A4 crashes, F0 paired-t V13/V14 per-slice re-eval (~12 min GPU), monitoring overhead, and writing-period responsiveness.
4. No new stop-rule precedent.
5. **Wall-clock identical to E** (both ~7d for the longest run) but with less orchestration cost.

### 3.3 Why E is the maximum acceptable deviation

If user insists on a bracket with a content-based justification:
- E narrows the deviation to 2 variants (A4-low + A4-mid).
- 2-slot has no historical RAM precedent issue (Round 15 already ran 3-slot; 2-slot is conservative).
- Both A4-low and A4-mid have ≥MED EV; A4-high doesn't.
- Stop-rule can be cleanly reworded: "no new image_aux variant launched **after any A4 result is observed**; all variants pre-registered and launched within 24h of first launch" — closes the post-result loophole that the original §A4.5 wording targeted.

User pre-condition for E: a single 2-3 line content-based justification beyond "slot 利用率不够". E.g., "I believe a single probe at 0.08 leaves the [0, 0.04] interval direction-of-bend unresolved, and a response curve is needed for MICCAI submission."

### 3.4 No custom Plan F

I don't see an option I'd prefer over D/E. The prompt's framing of options A-E is comprehensive.

---

## 4. Q4 — 3-slot RAM contention

**Verdict: N/A under primary recommendation; minor under secondary.**

### 4.1 Under D (1 slot)

No contention. Skip the question.

### 4.2 Under E (2 slot)

A4-low and A4-mid have identical model size, dataloader, optimizer footprint (only image_aux λ differs — a scalar). The 2-slot total footprint matches Round 15's confirmed-feasible 3-slot footprint (A3 + V13 + V14), so 2-slot is well below the historical OOM threshold.

Launch protocol for E:
- T=0: launch A4-mid (the higher-EV probe; if anything has to be sacrificed mid-run, sacrifice the lower-EV one)
- T+5min: confirm A4-mid alive (existing §A4.2 health check)
- T+15min: launch A4-low
- T+20min: confirm A4-low alive
- No dataloader worker cap needed (2-slot is below the 3-slot threshold that previously needed worker caps).

### 4.3 Under A (3 slot) — only documented for completeness

If user overrides to A:
- Staggered launch mandatory (T=0, T+30min, T+60min)
- Worker cap mandatory (e.g., 4 from default 8)
- RAM monitor required during first 12h
- These overheads are real and contribute to my A-rejection.

---

## 5. Q5 — Sweep priority

**Verdict: A4-low (λ=0.02) >> A4-high (λ=0.12).**

### 5.1 Why A4-low > A4-high

| dimension | A4-low | A4-high |
|---|---|---|
| Brackets the bend | YES (tests "is 0.04 already past steep zone, or could a lower λ do as well") | NO (tests bound we already know exists) |
| Discriminates 2 hypotheses | YES: (a) function is steep through [0, 0.04] and flat after (likely outcome A4-low << V7); (b) function is shallow through [0, 0.04] (likely outcome A4-low ≈ V7-ε) | NO: only one likely outcome class (saturation OR regression — both confirm bound) |
| Reviewer pressure if absent | Some — "what about lower λ?" | Minimal — "we tested doubling" is plausible |
| Cost if surprising | Surprising A4-low could justify a follow-up at λ=0.01 or 0.03. Manageable. | Surprising A4-high (e.g., +0.05 dB at 0.12) triggers full scope creep (try 0.16, 0.20, ...). High governance risk. |

### 5.2 So if E is chosen

Run A4-low + A4-mid. Skip A4-high entirely.

### 5.3 Prompt §2.2 self-undermines A4-high

> "物理直觉: λ=0.04 已经是项目用了 ~1 年的设置, V7 选 0.04 不是偶然, 大概率是经验调出的 sweet spot"

If this prior is correct, then both A4-low and A4-high probably regress toward V7 (or worse). But A4-low's regression is informative (confirms 0.04 is at/near sweet spot from below); A4-high's regression is just "yes, too much hurts". A4-low is therefore higher EV per the prompt's own argument.

---

## 6. Q6 — Document update scope

**Verdict: only if E. None if D.**

### 6.1 If D (recommended primary)

No document changes. R17 consensus preserved verbatim. Codex task md remains pushable after the R17-prep fixes (B75-B82 from my last review). No new round-of-reviews needed.

### 6.2 If E

Required updates (narrow set):

| § | original | fix |
|---|---|---|
| CODEX_TASK §A4.5 | "无论 outcome, 不 relaunch A4-v2 / v3" | "All image_aux probes pre-registered and launched within 24h of first launch. No new image_aux variant after any A4 result is observed. Maximum 2 pre-registered variants under this revisit. (Round 17-Slots revision.)" |
| CODEX_TASK §5 anti-check #3 | `find … 'A4*v2*.yaml' \| wc -l ≥ 1` | `find review/0521 -type d -name 'A4_*' \| wc -l > 2` (catches >2 A4 directories) |
| CODEX_TASK §6 NOT-DO #6 | "启 A4-v2 / A4-v3 / 其它 image_aux variant" | "启 第 3 个 image_aux variant (e.g., A4-v3, A4-warmup-probe, A4-ramp); maximum 2 (A4-low + A4-mid) under R17-Slots revision" |
| REVIEW_INTEGRATION_round17 lines 126, 156 | "A4 = one image_aux schedule variant" | (add footnote pointing to R17-Slots revision: "Updated 2026-05-22: revised to 'at most two pre-registered image_aux variants, both launched within 24h of first launch'") |
| (new) `A4_image_aux_lambda_02.yaml` | — | Identical 4-field diff (output_dir, run_name, `training.image_aux.lambda_start=0.02`, `training.image_aux.lambda_max=0.02`) |

R17-prep self-check additions (per my last review §7) carry over unchanged for both yamls.

### 6.3 New formal round needed?

For E (2-slot) — NO. The deviation is narrow enough that an inline `R17_SLOTS_DECISION_NOTE.md` documenting the revisit + the 2-3 line content-based justification + the new stop-rule wording suffices. Adding it to the existing R17 thread.

For A/B/C (3-slot or extra dims) — YES, would warrant Round 17-Slots integration md. But I'm rejecting those.

---

## 7. Q7 — New biases (B83-B86)

Next available is B83 (B61-B82 used; B82 was in my R17-prep).

| ID | bias | severity | description |
|---|---|---|---|
| **B83** | Selective quotation of stop rule | **HIGH** | Prompt §1 quotes only CODEX_TASK §A4.5 ("no relaunch v2/v3") and concludes "字面没禁 parallel sweep". But the source of R17 consensus is `REVIEW_INTEGRATION_round17_20260522.md`, which says "**one** variant" 6 times (lines 17, 123, 126, 156, 173, 196). The task md §A4.5 is a tactical sub-rule (post-result re-run only); the integration md is the strong rule (pre-registered count). The prompt skips the integration md entirely in its §1 stop-rule verification. |
| **B84** | Slot-utilization sunk-cost fallacy | MED-HIGH | "此时只运行了一个训练? 我认为可以同时运行两个, 最多运行三个" frames idle slots as cost. Idle slots are not cost — they are paid-for-anyway compute that's available as buffer. The marginal cost is orchestration overhead per additional run, not GPU-time. This is the textbook "we have leftover budget, let's spend it" pattern. |
| **B85** | Sweep-curve fetish | MED | The framing "3 点 sweep > 1 点 single probe" assumes more datapoints → stronger paper. For an ablation where the 1-point signal is already cleanly interpretable (a/b/c outcome classes in §2.2 above), additional points are confirmatory not discriminating. Reviewers accept "we tested λ=0.08 and observed [a/b/c]" without demanding a curve. |
| **B86** | Stop-rule slippage precedent | MED | One revisit with a weak reason ("slots are idle") establishes the pattern that future stop rules will fail under similar pressure. The R17 stop rule was specifically designed to resist this. By accepting revisits on resource-based reasons, the project loses governance teeth for all future stop rules. |

### 7.1 Additional cross-bias note

The prompt §6 directive ("不要因为 '稳妥' 列 4 个方案推给 user 决策疲劳") is correct and I followed it: clear primary D, clear secondary E, clear rejection of A/B/C. But the prompt's own framing of 5 options (rather than asking "should we revisit at all?") implicitly pre-commits to at-least-considering revisit. The most rigorous version of this prompt would have been: "Should we revisit? Y/N. If Y, by how much?" — instead the prompt presents 5 plans of which only one (D) is "no revisit", front-loading the discussion toward "yes". This is a mild **framing bias** but doesn't rise to a B-code on its own.

---

## 8. Output per prompt §5

### 8.1 Per-question

| Q | verdict |
|---|---|
| Q1 Should stop rule be revisited? | **REJECT** revisit on slot-utilization grounds; would approve only with content-based justification. |
| Q2 Image_aux sweep EV | **MODIFY** — single A4-mid is sufficient; bracket adds marginal value. |
| Q3 5-plan selection | **D primary, E secondary**; reject A, B, C. |
| Q4 3-slot RAM | **N/A under D**; minor under E (staggered launch only, no worker cap needed). |
| Q5 Sweep priority | A4-low >> A4-high. |
| Q6 Document update | None under D; narrow updates under E (4 changes listed §6.2). |
| Q7 New biases | 4: B83 HIGH (selective quotation), B84-B86 MED-LOW. |

### 8.2 Main verdict

**Plan D** (1 slot, A4-mid λ=0.08 only) as primary.
**Plan E** (2 slot, A4-low + A4-mid) as secondary contingent on user articulating a content-based reason.

### 8.3 Stop-rule wording update (if E)

"All image_aux probes must be pre-registered and launched within 24h of first launch. No new image_aux variant launched after any A4 result is observed. Maximum 2 pre-registered variants under the R17-Slots revision (2026-05-22)."

If staying on D, no wording update.

### 8.4 Launch protocol

**D**: no protocol changes. Codex task md §A4.2 unchanged.

**E**: T=0 A4-mid → T+5min health → T+15min A4-low → T+20min health → monitor both. No dataloader-worker cap needed (2-slot well below 3-slot RAM threshold). Separate `A4_image_aux_lambda_02.yaml` created with identical 4-field diff (s/0.08/0.02/g; with the **B75 namespace fix from R17-prep**: under `training.image_aux.*`, not `transport.image_aux.*`).

### 8.5 New biases

§7 detailed. B83-B86.

---

## 9. One-paragraph executive summary

The Round 17 stop rule's source of truth is the integration md, which says "**one** image_aux schedule variant" six times in independent statements — not the task md's tactical §A4.5 "no v2/v3 relaunch" wording that the prompt §1 cherry-picks. The user's revisit is motivated solely by slot utilization (a sunk-cost-of-idle-resources fallacy: idle slots are paid-for-anyway compute, not project cost). No new substrate, finding, or external constraint justifies the revisit. The single A4-mid probe at λ=0.08 captures most of the information yield (it discriminates among saturation / still-climbing / over-weighted outcomes, all of which are paper-actionable). Adding A4-low has modest confirmatory EV; adding A4-high has very low EV and high scope-creep risk. **Recommendation: hold Plan D (1 slot, A4-mid only)** as primary; if user articulates a content-based reason, **Plan E (2 slot, A4-low + A4-mid, 1 slot buffer)** is the maximum defensible deviation. Reject Plans A (3-slot full sweep, A4-high low EV), B (conflates V14b governance), and C (V13-V8 +0.02 dB train-config residual is below paper relevance). Four new biases: **B83 (HIGH) selective quotation of stop rule**, B84 slot-utilization fallacy, B85 sweep-curve fetish, B86 stop-rule slippage precedent.

---

## 10. What I did NOT review

- Round 17 strategy (4/4 signed).
- F0 / A4 yaml execution details (R17-prep delivered MODIFY-BEFORE-PUSH with B75-B82; assumed those fixes are being applied in parallel).
- F0 paired-t statistical methodology (Round 17-Stats parallel review per prompt §0).
- V18 / V13 / V14 substrate (R17 signed).
- KL pullback design (R16 signed).
- Whether 3-slot is technically feasible (Round 15 confirmed it is); my rejection of A is on EV+governance grounds, not feasibility.
