# REVIEW INTEGRATION — Round 17 (Post-V13/V14 Strategic Decision)

- date: 2026-05-22
- scope: integrate 4 independent Round17 reviews for post-V13/V14 strategic decision
- sources:
  - PEER_REVIEW_ROUND17_REVIEWER_A_20260521.md (Reviewer A)
  - PEER_REVIEW_round17_reviewer_C_20260521.md (Reviewer C)
  - PEER_REVIEW_round17_review_D_20260521.md (Review D)
  - PEER_REVIEW_round17_strategy_post_V13V14_copilot_20260521.md (GitHub Copilot)

---

## 1) Strong consensus (4/4)

### 1.1 Strategic main path

**All 4 reviewers select Hybrid B+A4** (paper draft as primary work, with one cheap image_aux schedule probe in parallel).

Reviewer-specific variants:
- Reviewer A: `B + F0 + A4` (adds zero-cost per-slice paired-t as mandatory prerequisite gate)
- Review D: `B + A4-light` (A4 not gating, paper viable without it)
- Reviewer C: standard `Hybrid B+A4` with B' narrative framing
- GitHub Copilot: `Hybrid B+A4` standard

**All 4 reject** options C (architecture pivot), D (data pivot), E (terminate).
**All 4 reject** continuing transport intervention as main path **before** writing paper.
**All 4 list backup = B paper-as-is** if A4 not launched or returns < +0.05 dB.

### 1.2 V18 signal verdict

**All 4 converge: real but ablation-only / decoupling diagnosis. Not headline.**

- V18.last - V7.best = +0.0617 dB is above any reasonable single-seed perturbation, so it is "real".
- But its magnitude is ~5× smaller than image_aux (+0.29 dB), so it cannot carry the paper.
- KL pullback narrative remains dead (Round 16 result holds).
- A3 already explained ~all V18 direct-decode gain as decoder LoRA capacity, not KL.

### 1.3 Paper narrative

**All 4 converge: image_aux is the headline (narrative B').**

Recommended structure (synthesis):
- Main result: image auxiliary supervision is the dominant design choice in PET latent transport (+0.2867 dB single-variable contribution, V14-V13).
- Structural finding: decoder-capacity gains do not propagate through chain (A3); decoder and transport are decoupled.
- Negative controls: KL pullback (A3) and image_aux-off (V13) both fail to improve over V7.
- Ablation: V18 family demonstrates +0.06 dB capacity gain, far below image_aux.

Reject narrative D' (negative-result-only) as anchoring on wrong polarity — the project does have a real positive headline (image_aux).
Reject narrative E' (feasibility-only) as understating the disambiguation contribution.

### 1.4 V13/V14 substrate

All 4 verified the canonical eval numbers from raw JSON:
- V13.best == V13.last NORMAL = 36.4943
- V14.best == V14.last NORMAL = 36.7806
- Three derived facts confirmed: X1 (d_pure |Δ|=0.0004), X2 (image_aux=+0.287), X3 (V13-V8=+0.0214).

---

## 2) Convergent methodological criticism (4/4 raise B61-class)

### 2.1 Single-seed SNR inflation

**All 4 reviewers flag the same methodological issue with prompt §2:**

> Treating V14's single |Δ|=0.0004 dB as if it were a noise standard deviation `σ`, then computing "SNR = 75-150×" for V18 deltas.

Severity assessments differ:
- Reviewer A: HIGH (calls it methodological cherry-pick; points out PLANF L43 already has paired-t = 74.6 for V7 vs V8 on 7403 slices, free to apply to V18 vs V7)
- Reviewer C: LOW (recommends per-slice paired-t fix)
- Review D: MODIFY Q1 verdict
- Copilot: B69 "single-seed SNR inflation"

Integrated resolution:
- The SNR table in prompt §2 must be downgraded from "SNR = 75-150×" to "above single-seed perturbation magnitude".
- **Reviewer A's F0 proposal accepted**: run per-slice paired-t on existing JSON / per-slice CSVs **before** any paper claim about V18 significance. Zero GPU cost.
- If paired-t shows V18 vs V7 is significant, V18 can stay as ablation with a proper p-value.
- If paired-t shows no significance, V18 stays as "small effect at observed magnitude, not statistically separable from seed noise" — still consistent with ablation framing.

### 2.2 Reviewer C found one numerical typo in prompt §2

Sign flip: "V7.best − V6_NOISE.best = −0.0373" should be **+0.0373** (V7 > V6_NOISE). §1.3 has correct sign in the "vs V7.best" column.

This does not change any strategic conclusion but should be noted as `B61-typo` and corrected if prompt is re-circulated.

---

## 3) Meaningful disagreements (medium confidence)

### 3.1 Is F0 (paired-t) a gate or just polish?

- Reviewer A: **mandatory gate** before paper draft commits to "V18 is real signal".
- Reviewer C: nice-to-have; "Hybrid B+A4 is primary, V14b is only outstanding control".
- Review D + Copilot: silent on F0; paper draft can start without it.

Integration resolution:
- Treat F0 as **mandatory before V18 enters the paper as a positive ablation result**.
- Treat F0 as **not blocking** paper outline / image_aux headline drafting.
- F0 is < 1 hour of analysis on existing artifacts; no reason not to do it.

### 3.2 V14b (additional seed) priority

- Reviewer C: V14b is the only high-EV outstanding control.
- Reviewer A: V14b is recommended but not strictly required if F0 paired-t is done.
- Review D + Copilot: V14b is low priority, not gating.

Integration resolution:
- V14b is recommended but not required. F0 (free, paired-t) is the cheaper substitute for the "V18 significance" question.
- If V14b is run, it should be in spare GPU slot only, not displacing A4.

### 3.3 V13 vs V8 = +0.0214 dB and Grönwall step_weights claim

- Reviewer A: **REJECT** prompt §1.4 X3's "Grönwall single-variable effect within noise" claim because V13 and V8 still differ in train config — this is an acknowledgment-then-overreach pattern.
- Reviewers C, D, Copilot: APPROVE with wording softening ("not the dominant contributor, but train-config confound prevents exact-zero claim").

Integration resolution:
- Accept softened wording: "Grönwall step_weights is **not the dominant** contributor to V7 main-line gain; image_aux is. Remaining V13-V8 = +0.02 dB is small and confounded by train config; do not claim Grönwall single effect is exactly zero."

### 3.4 A4 specifics

All 4 reviewers want A4 = image_aux schedule probe. Slight differences on exact form:
- Reviewer C / Copilot: image_aux schedule tuning (ramp / decay / hop0-strength).
- Review D: at-most-one pre-registered A4 probe; do not let it scope-creep.

Integration resolution:
- A4 = one image_aux schedule variant, pre-registered with success criterion ≥ +0.05 dB before launch.
- Launch only if GPU is otherwise idle. If A4 returns < +0.05 dB, do not relaunch a v2.

---

## 4) Integrated bias audit (B61-B74 consolidation)

Cross-reviewer biases (with severity from highest-flagging reviewer):

| ID | bias | severity | source reviewer(s) |
|---|---|---|---|
| B61 | single-seed |Δ| treated as σ in SNR table | HIGH | A |
| B62 | V13-V8 Grönwall overreach (acknowledgment-then-overreach) | HIGH | A |
| B63 | narrative anchor (Q5 4-option menu) | MED | A |
| B64 | strategic option naming conflict (A3 already used as V18-cap label) | MED | A |
| B65 | "事实 X1/X2/X3" framing dressing single estimates as facts | LOW | A |
| B66 | "Hybrid 低风险" conflating cost with risk | LOW | A |
| B61-typo | V7 - V6_NOISE sign flip in prompt §2 | LOW | C |
| B62-seed | unused V14↔V6_NOISE seed=1337 paired comparison (+0.0369 dB) | LOW | C |
| B63-SNR | "75-150× SNR" framing inflates 1-shot estimate | LOW | C |
| B69 | single-seed SNR inflation (= B61) | HIGH | Copilot |
| B70 | image_aux overclaim risk if accepted without train-config caveat | MED | Copilot |
| B71 | Grönwall exact-zero overreach (= B62) | MED | Copilot |
| B72 | V18 sunk-cost headline bias | MED | Copilot |
| B73 | paper-as-is defeatism (B alone underclaims) | LOW | Copilot |
| B74 | Hybrid scope creep (A4 silently becomes A4-v2-v3) | MED | Copilot |

Operational guardrails:
- All paper claims must specify substrate (canonical PSNR_clip3 vs trainer chain MSE).
- V18 significance language requires paired-t result; otherwise downgrade to "above observed single-seed perturbation".
- A4 scope limited to one pre-registered variant.
- Image_aux claim wording must acknowledge V13-V14 share train config so "image_aux = +0.287 dB" is the cleanest single-variable.

---

## 5) Integrated action plan

### 5.1 Do NOW (no GPU, no review wait)

1. **F0 — paired-t on existing data**. Compute per-slice paired-t for V18.best, V18.last, A3.cap.last each vs V7.best on full-val 7403 slices using existing JSON / per-slice CSVs. Report t, p, mean Δ, paired SEM. Owner: claude. Cost: < 1 hour.
2. **Start paper outline with image_aux headline** (narrative B' + C' decoupling supporting). Owner: user + claude.
3. **Update claim ledger**: image_aux SUPPORTED (+0.287 dB), decoupling SUPPORTED, KL REJECTED, V18 PENDING-paired-t.
4. **Fix prompt §2 sign typo** in any document that quotes prompt §2.
5. **Add standing rule**: never use single-seed |Δ| as σ for SNR; require paired-t or multi-seed std.

### 5.2 Launch only after F0 + design memo

1. **A4 — one image_aux schedule probe**, pre-registered with stop rule (`<+0.05 dB → do not re-run`). Launch only if GPU slot is idle. Cost: ~7 days × 1 slot.

### 5.3 Do not launch / hold

1. **V19 / V18-clean / V18 rank-sweep**: low EV after A3+V13+V14.
2. **C (architecture pivot) / D (data pivot)**: out of scope for this paper.
3. **V14b/V14c multi-seed**: low priority; F0 covers significance question more cheaply.
4. **E (terminate)**: rejected; the image_aux + decoupling story is publishable.

### 5.4 Paper-narrative integration

- Headline: image_aux as dominant design choice (+0.287 dB true single-variable).
- Structural: decoder-transport decoupling (A3 + V18 chain ≤ 18% propagation).
- Ablations: V18 capacity gain, KL negative result, image_aux off (V13).
- Significance: paired-t numbers from F0 in every relevant comparison.
- Limitations: single-seed perturbation only; not multi-seed std; train-config confound on Grönwall.

---

## 6) Final integrated verdict

**Round 17 main verdict = Hybrid B + F0 + A4-light**

> Start paper draft now with image_aux as headline. Run F0 (free paired-t analysis) before V18 enters the paper as ablation. Optionally launch one pre-registered A4 image_aux schedule probe if GPU is idle. Do not relaunch on failure. Do not pursue architecture/data pivot or new V18-family runs.

**V18 signal verdict** = real but ablation-only; significance pending F0 paired-t; reframe as decoupling diagnosis component of the structural finding.

**Paper narrative** = B' (image_aux headline) + C' (decoupling structural claim) + KL/V18 as negative-control ablations.

---

## 7) Suggested Round 18 trigger condition

Start Round 18 only when at least one of these is true:
1. F0 paired-t result available and V18 paper-role needs review.
2. A4 schedule probe returns and result requires interpretation.
3. Paper outline / first draft has concrete claims needing peer review.
4. Significant unexpected substrate change (e.g. evaluator bug discovered).

Otherwise, Round 17 is the final strategic decision round before paper-writing phase.
