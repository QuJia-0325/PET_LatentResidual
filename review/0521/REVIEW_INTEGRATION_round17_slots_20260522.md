# REVIEW INTEGRATION — Round 17-Slots (Stop Rule Revisit)

- date: 2026-05-22
- scope: integrate 4 independent reviews on the stop-rule-revisit / slot-utilization decision
- sources:
  - PEER_REVIEW_ROUND17_SLOTS_REVIEWER_A_20260522.md (Reviewer A)
  - PEER_REVIEW_round17_slots_review_D_20260522.md (Review D)
  - PEER_REVIEW_round17_slots_reviewer_C_20260522.md (Reviewer C)
  - PEER_REVIEW_round17_slots_stop_rule_revisit_copilot_20260522.md (GitHub Copilot)

---

## 1) Strong consensus 4/4

### 1.1 The prompt itself is biased (B83 — selective quotation)

All 4 reviewers independently caught the same flaw in the slot-revisit prompt §1:

> The prompt quoted only `CODEX_TASK §A4.5` ("no v2/v3 relaunch after outcome") and concluded "字面没禁止 parallel sweep". But the actual R17 source-of-truth ([REVIEW_INTEGRATION_round17_20260522.md](./REVIEW_INTEGRATION_round17_20260522.md)) constrains A4 to **one variant** at six independent lines (17, 123, 126, 156, 173, 196).

Severity: HIGH (Reviewer A) / HIGH (Reviewer C) / HIGH (Copilot). Review D didn't tag severity but reached same finding.

→ R17 actual consensus = "one pre-registered A4 probe", not "no-v2/v3-after-outcome". The prompt accidentally weakened the rule.

### 1.2 Plan A (3-slot 0.02/0.08/0.12) and Plan C (V13b) are rejected 4/4

| plan | unanimous rejection reason |
|---|---|
| **A** (3-slot full sweep incl. λ=0.12) | λ=0.12 EV lowest of 3 (confirmatory bound, not informative); challenges user's own 3-slot RAM warning |
| **B** (2-slot bracket + V14b) | V14b directly conflicts R17 integration L179 "V14b/c low priority"; F0 paired-t already covers significance |
| **C** (A4-mid + V13b train-config residual) | V13-V8 +0.02 dB is below paper-reporting threshold; mixes governance scope |

### 1.3 A4-low (λ=0.02) > A4-high (λ=0.12) for slot 2 priority

Even reviewers who hesitated on expansion (Reviewer C) agreed that **if any slot beyond A4-mid is added, A4-low is the right one**:
- λ=0.02 informatively brackets between V13 (λ=0) and V7 (λ=0.04) — tests near-zero linearity
- λ=0.12 only bounds saturation upper end, EV much lower
- V13 (0) and V7 (0.04) endpoints already exist, so 1 extra interior point (0.02) is more informative than 1 extreme bound (0.12)

---

## 2) Verdict divergence (3:1)

| reviewer | main | backup |
|---|---|---|
| Reviewer A | **E** (2-slot 0.02+0.08) | D (守原 1-slot) |
| Review D | **E** (2-slot 0.02+0.08) | B (only after staggered health checks) |
| Copilot | **E** (2-slot 0.02+0.08) | D |
| Reviewer C | **D** (守原 1-slot) | E (only if user gives content-based justification) |

**Tally**: E = 3 primary + 1 secondary; D = 1 primary + 2 secondary; B = 1 secondary; A = 0; C = 0.

Reviewer C's D-primary stance is grounded in B83 (R17 integration md says "one variant"). The other three agree with the bias finding but conclude that **user's content-based revisit signal (PSNR sweep curve) is legitimate revisit trigger**, distinct from raw slot-utilization argument.

---

## 3) Convergent governance recommendations

### 3.1 If expansion happens, it must be closed-form pre-registered

All 3 E-supporters explicitly require:

- **Exactly 2 points, not 3** (no λ=0.12, no λ=0.06, no v2 iterations)
- **Slot 3 stays buffer**, not filled with V14b/V13b/anything
- **New explicit stop rule** replacing old one: "A4-bracket = {0.02, 0.08} pre-registered; on completion, no further image_aux variants regardless of outcome"
- **Staggered launch** (Round 15 B58 precedent): A4-low after A4-mid health check at +30 min

### 3.2 The revisit must be documented as a stop-rule amendment, not a quiet bypass

3/4 require explicit amendment text in CODEX_TASK §A4.5 + integration md, plus standing rule update: "Stop-rule revisits require content-based justification (new substrate or new analytical insight), not resource-availability arguments alone."

This addresses B86 (slippage precedent): if "idle slot" alone is accepted as revisit trigger, future stop rules become non-binding.

### 3.3 Slot-utilization framing is rejected as core argument

All 4 reviewers reject "GPU slot idle = waste" as the primary basis. Reviewer C calls it B84 sunk-cost fallacy; Copilot calls idle slot a "buffer with positive option value"; Reviewers A and D agree slot argument alone is insufficient.

The acceptable revisit basis (per 3 E-supporters) is **content-based**: image_aux is the project's confirmed +0.287 dB main lever (Round 17 X2), so even a 2-point bracket near V7's λ=0.04 produces materially stronger paper figure than 1 point.

---

## 4) Critical sequencing fact (substrate)

**A4-mid (λ=0.08) is already running**. This shifts the question from "should we sweep" to "should we add slot 2 (A4-low λ=0.02) now while A4-mid trains".

Given:
- A4-mid launch is irreversible sunk action
- All 4 reviewers reject A4-high and V14b for slot 2 priority
- 3/4 support adding A4-low as slot 2
- 1/4 (Reviewer C) supports holding all expansion

**Conservative integration**: accept the 3-reviewer majority and add A4-low (λ=0.02) as slot 2, with explicit stop-rule amendment + Reviewer C's caveat documented.

---

## 5) New biases B83-B86 (consolidated)

| ID | bias | severity | source |
|---|---|---|---|
| B83 | Stop-rule selective quotation (cited §A4.5 only, missed integration md "one variant" × 6) | HIGH | A, C, Copilot |
| B84 | Slot-utilization sunk-cost fallacy (idle slot framed as waste) | MED-HIGH | C, Copilot |
| B85 | Sweep-curve fetish (single probe already discriminates 3 outcome classes) | MED | C |
| B86 | Stop-rule slippage precedent (resource-based revisit erodes governance) | MED | C, A |

Root cause pattern: **when user signals slot availability, claude tends to surface plausible expansions and frames them as "low-risk because already running anyway"**. This is structurally similar to Round 15 B55 (user "multi-seed deferred" decision was misquoted to support adding V14). Both cases involve claude expanding scope via selective citation of prior decisions.

**Standing rule recommendation (4/4 implicit)**: any stop-rule revisit must (a) cite **all** prior decision sources, not just convenient ones, and (b) base justification on content/substrate change, not on resource availability.

---

## 6) Final integrated verdict

**Main = Plan E with strict closure** (3 reviewers + sequencing fact):

1. Slot 1: **A4-mid (λ=0.08)** — already running, do not interrupt.
2. Slot 2: **A4-low (λ=0.02)** — staggered launch after A4-mid health check at +30 min.
3. Slot 3: **buffer**, do not fill with V14b/V13b/A4-high.

**New stop rule** (replaces R17 "one variant"):

> "A4-bracket = pre-registered set {λ=0.02, λ=0.08}, exactly 2 points. On completion of both, no further image_aux variants regardless of outcome. Future stop-rule revisits require content-based justification (new substrate / new analytical insight), not resource-availability arguments alone. Documented as Round 17-Slots amendment to Round 17 'A4-light'."

**Backup = Plan D** (Reviewer C primary):

If user does not want to spend the additional 7d × 1 slot or wants to honor Reviewer C's B83 strict reading, hold at A4-mid only. No paper damage — A4-mid alone still tests upward saturation, which combined with V13/V7/V14 existing 3-point evidence supports adequate ablation narrative.

---

## 7) Required actions

### 7.1 Documentation (mandatory before slot 2 launch)

1. Write `A4_image_aux_lambda_02.yaml` (clone V7, only `training.image_aux.lambda_start/max: 0.04 → 0.02`).
2. Update `CODEX_TASK_ROUND17_F0_A4_20260522.md`:
   - §A4.1 → expand to bracket-2pt with both yaml entries
   - §A4.5 → replace stop rule with the new wording above
   - §5 anti-check → broaden to `[[ wc -l > 2 ]]` for `A4_image_aux_lambda_*.yaml`
   - §6 NOT-DO #6 → update to "no λ outside {0.02, 0.08}"
   - Add explicit "Round 17-Slots amendment" reference

### 7.2 Codex execution

After documentation is committed and pushed:
- Codex launches A4-low at +30 min after A4-mid health check (staggered, B58 protocol).
- Same SMOKE_TEST_REPORT pattern: A4-low gets its own smoke before formal launch.
- Both feed into same A4 verdict report when complete.

### 7.3 Round 17 stop-rule amendment note

Add a one-paragraph amendment block at the bottom of [REVIEW_INTEGRATION_round17_20260522.md](./REVIEW_INTEGRATION_round17_20260522.md) noting that R17's "one variant" rule was amended on 2026-05-22 to "exactly 2 points {0.02, 0.08}" per Round 17-Slots 3/4 consensus, with link back to this integration doc.

---

## 8) Round 17-Slots → Round 18 trigger

Round 18 strategic integration triggers only when **at least 2 of 3** are complete:
1. F0 paired-t analysis output (V13/V14/V18 vs V7)
2. A4-mid full-val canonical eval result
3. A4-low full-val canonical eval result

Until then, no further peer review rounds expected. The remaining open thread is Round 17-Stats (slice-level vs patient-level paired-t methodology), which is independent of slot expansion and proceeds on its own track.
