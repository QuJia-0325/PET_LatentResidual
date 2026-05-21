# Peer Review Round 17-Slots - Review D Audit

- date: 2026-05-22
- reviewer: **Review D (GitHub Copilot)**
- scope: Stop-rule revisit for A4-light vs A4-bracket slot allocation
- prompt: `PEER_REVIEW_PROMPT_round17_slots_stop_rule_revisit_20260522.md`
- independence note: I did not read any Round17-Slots reviewer drafts.

---

## 0. Executive Verdict

**Main verdict: Option E - A4-bracket 2 points (2 slots: lambda 0.02 and 0.08), leave the third slot as buffer.**

**Backup verdict: Option B - A4-bracket 2 points + V14b, only if a staggered launch health check shows RAM/IO headroom is clearly safe.**

I do not recommend Option A's 3-point A4 bracket with lambda 0.12. The stop rule can be revisited because the user explicitly reopened slot allocation before launch, but the right repair is a **closed, predeclared 2-point bracket**, not a full sweep that quietly turns A4-light into an image_aux tuning campaign.

The response curve value is real, but it has diminishing returns. V13=0 and V7=0.04 already establish the main image_aux effect. Adding 0.02 and 0.08 gives a useful four-point curve around the current setting. Adding 0.12 adds less information, costs the riskiest third slot, and increases the chance that any surprising result triggers another request for 0.06/0.10/0.16.

Note: the prompt says "6 questions" but contains Q1-Q7, so I answer all Q1-Q7 below.

---

## 1. Q1 - Should the Round 17 Stop Rule Be Revisited?

**Verdict: MODIFY.**

The original stop rule does not strictly forbid a predeclared parallel bracket. Its literal target is serial relaunch behavior: A4 finishes, looks disappointing or exciting, and then the project invents A4-v2/v3 to keep searching.

However, the spirit of the rule does constrain this revisit. The rule existed because Hybrid B+A4 was supposed to preserve paper momentum. A bracket is acceptable only if it is closed before launch and explicitly bounded.

My interpretation:

- **Literal reading:** planned parallel variants are not automatically forbidden.
- **Spirit reading:** planned variants are allowed only if the sweep is finite, pre-registered, and cannot expand after results.
- **Governance reading:** user-initiated revisit is valid, but it should produce a new stop rule rather than weaken the old one.

So I accept revisiting the rule, but only for Option E or carefully gated Option B. I reject using the revisit to launch every tempting lambda.

---

## 2. Q2 - Does an Image_aux Sweep Have Real EV?

**Verdict: APPROVE for a small bracket; REJECT for 3-point high-lambda sweep.**

A two-point bracket has real paper EV because image_aux is now the headline mechanism. It can answer whether 0.04 is underpowered, near-optimal, or already over the useful range.

Practical value of the added points:

- **lambda 0.02:** tests whether the V13 to V7 gain is smooth or threshold-like; useful if 0.02 recovers most of the +0.287 dB.
- **lambda 0.08:** tests the original A4 question: whether 0.04 is saturated or underweighted.
- **lambda 0.12:** mostly tests over-regularization. It is useful only after 0.08 looks promising, which is exactly the serial tuning path the stop rule was meant to prevent.

If the bracket shows 0.04 is the sweet spot, that is paper-useful. If 0.08 beats V7 by >=0.05 dB, that is paper-useful too, but it must **not** trigger lambda 0.06/0.10 follow-ups in this phase. A predeclared bracket is evidence; an adaptive sweep is a new project.

---

## 3. Q3 - Strategic Option Choice

**Main: Option E - A4-bracket 2 points (lambda 0.02 and 0.08).**

This is the best EV/risk balance. It uses the user's spare-slot observation constructively, gives the paper a response-curve figure with V13=0, A4-low=0.02, V7=0.04, and A4-mid=0.08, while preserving one slot as a RAM/IO buffer.

**Backup: Option B - A4-low + A4-mid + V14b.**

Use this only if a staggered two-job launch is healthy after 30-60 minutes and the user still wants the third slot used. V14b is a better third job than lambda 0.12 because it strengthens the seed/noise story without expanding the image_aux hyperparameter search.

Rejected options:

- **Option A (0.02/0.08/0.12):** too aggressive. The third lambda has lower marginal paper value and higher scope-creep risk.
- **Option C (A4-mid + V13b):** V13b answers a residual config question, but image_aux headline value is better served by the 0.02/0.08 bracket. V13b is not urgent.
- **Option D (A4-mid only):** defensible under the old stop rule, but now too conservative after the user explicitly reopened slot allocation.

---

## 4. Q4 - RAM / IO Launch Protocol

**Verdict: staggered launch; no direct 3-slot launch.**

Recommended launch protocol for Option E:

1. **T=0:** launch A4-mid (`lambda=0.08`) first.
2. **T=+30min:** check process alive, GPU memory, host RAM, disk IO, and log progress through data loading.
3. **T=+30-60min if healthy:** launch A4-low (`lambda=0.02`).
4. **T=+90min:** repeat health check. Keep slot 3 empty unless moving to backup Option B.

If Option B is chosen after health check:

5. **T=+90min or later:** launch V14b only if the first two jobs are stable and host RAM/IO are not near pressure.

Do not directly launch all three. Round15's staggered pattern exists for exactly this kind of case.

On dataloader workers: V7-class configs already use `num_workers: 0`. Do not invent a worker reduction unless a specific config differs. The right guard is health-check/staggering, not modifying a baseline config field.

Codex should not autonomously decide to fill the third slot. The task should say: launch the third job only if the user selected backup Option B or gives an explicit post-health-check approval.

---

## 5. Q5 - Sweep Priority

**Verdict: A4-mid first, A4-low second, skip A4-high.**

Priority order:

1. **A4-mid, lambda 0.08.** Highest EV because it is the original A4 hypothesis and tests whether 0.04 is underpowered.
2. **A4-low, lambda 0.02.** Second-highest EV because it turns V13/V7 into a real dose-response curve and helps interpret saturation.
3. **V14b, seed 2024.** Best third-slot backup if RAM is safe and a third job is desired.
4. **A4-high, lambda 0.12.** Lowest EV in this phase. More likely to show regression or to tempt adaptive follow-up if it wins.

I would not launch 0.12 before seeing 0.08. And because seeing 0.08 first implies an adaptive follow-up, 0.12 should be excluded from this closed sweep.

---

## 6. Q6 - Integration and Task Update Needed?

**Verdict: APPROVE.**

If at least 2/3 reviewers agree to break A4-light, write a formal Round17-Slots integration document. This needs to be a real governance artifact, not a quiet edit to the task md.

Minimum task updates for Option E:

- Rename A4 section to **A4-bracket-2**.
- Add two yaml files:
  - `A4_image_aux_lambda_02.yaml`
  - `A4_image_aux_lambda_08.yaml`
- Update self-check from `<=1 A4 yaml` to an explicit allowlist of exactly those two yaml files.
- Update NOT-DO #6 to ban `lambda=0.12`, `lambda=0.06`, `lambda=0.10`, ramp/cosine variants, and any A4-v2/v3 after results.
- Add staggered launch protocol and slot-3 buffer language.

New stop rule wording:

> The current paper phase permits one closed image_aux bracket consisting only of lambda 0.02 and lambda 0.08, compared against existing V13 lambda 0 and V7 lambda 0.04. After these runs complete, no additional image_aux lambda, schedule, ramp, cosine, or v2/v3 variants may be launched without a new user-approved review round.

For backup Option B, add V14b as a separate seed-control task, not as an A4 variant, and require post-health-check user approval before launch.

---

## 7. Q7 - New Bias Audit B83+

- **B83 - Slot-utilization anchor.** Empty GPU slots are not automatically wasted; they can be deliberate risk buffer while writing starts.
- **B84 - Sweep-figure bias.** A prettier response curve is not automatically a stronger paper claim if it consumes governance bandwidth and invites hyperparameter search.
- **B85 - Stop-rule slippage.** Reopening a stop rule once can make future stop rules feel provisional. The fix is to replace it with a sharper closed-bracket rule.
- **B86 - High-lambda temptation bias.** Lambda 0.12 feels like a natural bracket endpoint, but its main value appears only after seeing 0.08, which creates adaptive search pressure.
- **B87 - Wall-clock invariance bias.** Three jobs in parallel still consume three GPU-weeks of aggregate compute and increase failure risk; same wall-clock is not same cost.
- **B88 - Memory optimism bias.** Prior 3-slot success does not prove every 3-slot combination is safe. Cold data loading and shared raw/latent files can still produce RAM/IO pressure.
- **B89 - A4-light reinterpretation bias.** Round17 reviewers meant one bounded A4 probe. Recasting that as implicit permission for a sweep is a new decision and must be documented.

---

## 8. Final Review D Decision

Use two slots, not three, for image_aux:

- Launch **A4-mid lambda 0.08** and **A4-low lambda 0.02** under a staggered protocol.
- Leave slot 3 as buffer by default.
- If the user explicitly wants a third job after health checks, use **V14b seed=2024**, not lambda 0.12.
- Write a Round17-Slots integration and update the Codex task before any launch.

This respects the user's slot-utilization concern without letting A4-light become an unbounded tuning campaign.
