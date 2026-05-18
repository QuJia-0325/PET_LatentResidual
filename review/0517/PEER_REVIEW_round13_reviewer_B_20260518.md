# Round 13 Peer Review — Reviewer B (independent)

- date: 2026-05-18
- substrate read independently (no draft cross-talk):
  - PEER_REVIEW_PROMPT_round13_project_strategy_20260518.md
  - PLANF_FINAL_ANALYSIS_20260516.md (§1)
  - V18_FINAL_RESULTS_20260518.md
  - KL_DRIFT_REPORT.md + KL_DRIFT_SUMMARY.json
  - pet_lr/model_first_hop.py `decode_crop`
  - pet_lr/decoder_lora.py header (LinearWithLoRA contract)
  - train_first_hop.py line 2230 (KL `z_kl` source)
  - REVIEW_INTEGRATION_round5_20260517.md (B9 history)
- verdict philosophy: §7 constraint #9 says "verdict 必须 grep verify 历史 PSNR". I grep-verified §1.1, §2.1, §4 paths, and the KL implementation **before** answering. Findings drive my recommendation.

---

## TL;DR

| Item | Verdict |
|---|---|
| §2.2 A — "V18 decoder LoRA truly improves GT manifold via KL pullback" | **REJECT (mechanistically impossible)** |
| §2.2 B — "+0.10 dB is generic LoRA capacity, not KL alignment" | **STRONGLY FAVORED** |
| §2.2 C — "evaluator/path artifact" | **WEAK** (decode_crop is structurally identical except for LinearWithLoRA wraps; rule out only if a control says otherwise) |
| Project ceiling = +0.099 dB | **NOT YET ESTABLISHED** — confounded by image_aux (V8 evidence) and by buggy V18 KL config |
| Stage C path | **E + new control** (V13 + V14 + V18-r32-no-KL capacity probe) BEFORE any pivot decision |
| Paper as-is | **REJECT** — narrative ceiling claim is unsupported until V13/V14/no-KL-control land |

---

## 0. New biases found in this prompt (B39+)

Per §6 "优先质疑" mandate and standing rule #9, I grep-verified the prompt's numbers and traced its mechanistic claims. Five new biases:

### B39 — §1.2 silent baseline switch produces two "项目总优化" numbers

- §1.1 table column `Δ vs V6_NOISE` shows V18.last = **+0.099 dB** (against V6_NOISE.best = 36.7437).
- §1.2 row "**V6_NOISE → V18.last (项目总优化) +0.088 dB**" switches baseline to V6_NOISE.last (36.7550) without saying so.
- Same document, same headline metric, two numbers (+0.088 and +0.099) for "项目总优化". The +0.088 number is then used in §2 prose ("0.088 dB 是 ... ceiling 信号?") and Q2/Q5/Q6 reasoning.
- Severity: MODERATE. Not number-fraud, but the smaller number anchors a more pessimistic narrative ("ceiling reached") and isn't flagged as a baseline choice.

### B40 — §1.2 arithmetic gloss is wrong twice in opposite directions

Prompt §1.2 narrative: *"(V7 +0.04 + V18 +0.05 last-vs-V7)"*. Actual paired diffs from §1.1:
| leg | claimed | actual |
|---|---:|---:|
| V6_NOISE.last → V7 | +0.04 | **+0.026** |
| V7 → V18.last | +0.05 | **+0.062** |
| sum | +0.09 | +0.088 |

The endpoints accidentally agree because both legs are mis-rounded in opposite directions. Severity: LOW (numbers don't change the conclusion), but it's exactly the kind of "narrative-anchored arithmetic" §6 said to flag.

### B41 — §2.2 A "transport 吞噬 70%" is best-only, sold as V18-general

`0.0973 - 0.030 = 0.067 absorbed / 0.0973 = 69%` is **V18.best vs V7 NORMAL**.
For V18.last: `0.164 - 0.062 = 0.102 absorbed / 0.164 = 62%`.
Q4 reasons "decoder LoRA 改进 +0.10 dB 但 transport 吞噬 70%" as a uniform V18 fact. It's a V18.best fact. The V18.last picture is friendlier (38% retained, ≈1.6× the best-checkpoint retention). Mild but real anchor toward "transport is hopeless".

### B42 — **§2.2 A is mechanistically impossible** (most important new finding)

The prompt's §2.2 A (claude default) reads:
> "KL pullback `use_pred_latent=true` **不**让 decoder 漂离 GT manifold ... V18 decoder 有 +0.10 dB GT 余量 ... Stage C bottleneck 在 transport 而非 decoder"

But in train_first_hop.py:2230:
```python
z_kl = main_out["z_pred"] if bool(kl_cfg.get("use_pred_latent", True)) else main_batch["z_dst"]
```
V18 yaml has `use_pred_latent=true` (confirmed in Round 5 INTEGRATION §2.2, line 78: *"use_pred_latent=true ... MSE(decode_lora(z_pred), decode_frozen(z_pred))"*). So the KL pullback loss is **`MSE(decode_lora(z_pred), decode_frozen(z_pred))`** — it acts on the **z_pred path**, never on z_GT.

The KL drift probe measures **`decode(z_GT)`** (KL_DRIFT_REPORT.md "Computation: direct decode_crop(z_GT) only").

So **no V18 training loss term ever supervised `decode(z_GT)`**. The +0.10 dB at the z_GT path cannot be credited to KL pullback alignment — KL pullback never touched that path. The +0.10 dB has to come from LoRA generic capacity flowing through the other losses (transport on z_pred, image_aux on z_pred). This is candidate B's story, not A's.

This also explains the "inverse" sign that surprised Round 12: the drift is exactly what you'd expect when you bolt 589K trainable parameters onto a frozen 250M ViT-MAE decoder and let them absorb residual error on a related path — generic capacity bleeds through to neighboring inputs. It's not evidence of decoder ceiling, it's evidence of LoRA being mildly overparameterized.

Severity: **HIGH**. Every Stage C reasoning step downstream of "decoder LoRA gave +0.10 GT margin" (Q4, Q6 D-option, Q2 ceiling claim) needs revisiting.

### B43 — Round 5 already flagged this and Round 13 forgot

Round 5 REVIEW_INTEGRATION §2.2 (line 92) explicitly concluded: *"正确设计应该是 `use_pred_latent=false`"*. V18 still shipped with `=true` (Round 5 line 170: "yaml 的 `use_pred_latent=true` 是 bug"). The V18 KL drift result is therefore from a known-buggy configuration. Round 13 prompt §2.2 reads as though the buggy config were the intended design and treats the inverse drift as a project-strategy signal. This is a regression of standing rule: known-bug experiments should be tagged as such in any "ceiling" reasoning.

---

## 1. Q1 — V18 KL drift interpretation

**Verdict: MODIFY**. Re-rank A/B/C per §2.2:

| candidate | new score | reason |
|---|---|---|
| A — decoder LoRA truly aligned to GT via KL pullback | **REJECT** | KL acts on z_pred path only (line 2230). Cannot have aligned z_GT path. |
| B — generic LoRA capacity (rank=32, +589K) bleeds through | **STRONGLY FAVORED** | Consistent with mechanism. Also explains "inverse" direction without exotic claims. |
| C — evaluator artifact (LinearWithLoRA path difference) | **WEAK** | decode_crop is one function. In eval mode (no dropout), LinearWithLoRA = base Linear forward + (B @ A) @ x. Deterministic. Float-precision delta at 1e-6 level cannot explain 0.10 dB. Rule-out via control. |

**Required control to disambiguate A vs B before any Stage C decision** (NEW Stage C sub-option):

**Control "V18-r32-capacity-only"**:
- Same as V18 yaml, change two things: `lambda_kl=0` and `image_aux` schedule held identical to V18.
- Warmstart from V7.best, 5K-10K steps (cheap probe, ≤1 slot).
- If `decode_V18cap(z_GT) - decode_V7(z_GT) ≈ +0.07 ~ +0.13 dB` (i.e. matches V18 drift within noise) → B confirmed, KL pullback contributed nothing on the probed path. Decoder ceiling claim falls.
- If drift ≈ 0 → A residually defensible (capacity alone does nothing without the *coupled* training signal). Even so, the +0.10 is from "image_aux on z_pred + LoRA capacity", not from KL alignment as advertised.

This is a 5–10K step probe, one slot, ~24h GPU. EV is enormous: it gates whether Stage C transport-intervention reasoning has a foundation.

---

## 2. Q2 — Is +0.088/+0.099 dB the project ceiling?

**Verdict: REJECT** the ceiling framing as currently presented. Three reasons:

1. **Image_aux confound (V8 evidence).** V8 = "Grönwall + image_aux OFF" → −0.27 dB vs V6_NOISE. So image_aux is doing ≥0.3 dB of work somewhere. V13 (Grönwall + image_aux off, isolated from V6/Plan F train-config differences) has not run. Until it does, we don't know whether image_aux ≈ all of the historical gain, in which case the "transport ceiling" framing is wrong: there was never a controlled transport improvement to ceiling against.
2. **Buggy KL config (B42/B43).** V18 ran with `use_pred_latent=true` despite Round 5 declaring this a bug. The "+0.06 dB last-vs-V7" is from a known-misconfigured run. The marginal increment may be a *floor*, not a ceiling.
3. **Decoder/transport split is not yet measured cleanly.** Q4's "decoder 8% of error" comes from Round 7 §2.4 MSE decomposition, but that decomposition assumes V7-style transport + V7-style decoder. Under V18 (LoRA decoder), the split shifts; under V13 (no image_aux), it shifts again. Quoting one number across configs is a category error.

Recommended re-framing: **"+0.099 dB observed under one possibly-confounded trajectory; true ceiling unmeasured until V13/V14 + capacity-only control land."**

If after those three runs the gain is still ≤+0.10 dB and decomposition shows transport plateau is real, then ceiling claim becomes defensible. Right now it's three independent confounds away.

---

## 3. Q3 — Does V8 −0.27 dB prove image_aux is the project pillar?

**Verdict: MODIFY (partial APPROVE)**. V8 vs V6_NOISE differs in more than image_aux:
- image_aux off (the headline change)
- Grönwall step_weights swap
- seed difference (per §1.1: V6_NOISE uses seed=1337, Plan F lineage uses seed=42)
- possibly other train-config drift inherited from Plan F vs V6 ancestry

So V8 −0.27 dB shows: *"the configuration with image_aux OFF, in the Plan F lineage, with Grönwall step_weights and seed=42, underperforms V6_NOISE."* It does not isolate image_aux.

But it does provide a strong **lower bound** on image_aux's load-bearing role: if you switch off image_aux in the otherwise-best lineage, you regress by at least 0.27 dB. That's much larger than the entire historical project Δ. **V13 is the single highest-EV experiment in the project right now**, hands down. Launching it should not be conditional on any Stage C decision — it should be conditional only on slot availability.

**Recommendation: launch V13 immediately, in parallel with the capacity-only control and V14. All three are independent and fit in 3 slots.**

---

## 4. Q4 — Decoder LoRA on GT manifold; transport intervention priority

**Verdict: REJECT the premise**. Q4 builds on §2.2 A (rejected in §1 above). Without A, we cannot conclude "decoder is fine, transport is the bottleneck". The bottleneck attribution is **not yet measured**.

If A *does* survive the capacity-only control (improbable but possible), then transport intervention priorities I'd rank:

| option | priority | rationale |
|---|---|---|
| **D** image_aux schedule tuning | **1st** | Cheap (1 slot, no new code), V8 evidence says image_aux is dominant — high EV per GPU-hour. Test alternative schedules first. |
| **E** multi-step refinement / chain length | 2nd | Modest cost, modifies existing rollout. Could expose whether 92% transport error is rollout-step-count limited. |
| **C** data augmentation / additional split | 3rd | Only after isolating model-side ceiling; otherwise data scaling is wasted. |
| **A** backbone scale-up (DiT base→large) | 4th | Expensive, low diagnostic value, risks 2 weeks of GPU for unclear gain. Defer. |
| **B** backbone architecture change (DiT→flow-matching/consistency) | 5th | Premature without a clean baseline. This is "pivot" disguised as "intervention". |

But again — none of this is actionable until §1 control + V13 + V14 land. **Stop reasoning about transport interventions until the bottleneck attribution is real.**

---

## 5. Q5 — Paper viability

**Verdict: REJECT "paper as-is"; MODIFY narrative options.**

Current data does not support any of the three claude-proposed narratives cleanly:

| narrative | problem |
|---|---|
| "feasibility study" | Without V13 / capacity control, you don't know whether you demonstrated feasibility of *transport* or of *image_aux*. |
| "thorough negative result" | A negative-result paper still needs a clean isolation of the negative. V8 confound + buggy V18 KL = not a clean negative. |
| "decoder LoRA partial improvement" | §1 just rejected this mechanism. Writing the paper around it would be reviewer-bait. |

**My recommended narrative path (post-V13/V14/capacity-control):**
- If V13 shows image_aux ≈ all the gain: paper becomes **"image_aux as a pixel-supervision signal in latent-transport PET"** — that's actually publishable at MICCAI/ISBI workshop as a methods note, possibly main track with strong ablations.
- If V13 shows Grönwall step_weights are doing meaningful work after image_aux is removed: paper becomes a **transport + decoder co-design study** with V18-r32-no-KL as a clean baseline.
- If both V13 and V14 land flat and capacity-only matches V18: paper becomes an **honest negative ("latent transport on PET tops out at ~0.1 dB without backbone change")**, MICCAI workshop level.

Venue: **MICCAI workshop or ISBI** in all three scenarios. TMI / Med Image Anal not realistic at current Δ-magnitude regardless of narrative.

---

## 6. Q6 — Stage C path

**Verdict (strongly): E + new sub-option, NOT D, NOT C.**

Stage C should be **three parallel runs** in 3 slots:
1. **V13** (Grönwall + image_aux off, isolated) — disambiguates the V8 regression.
2. **V14** (d_pure ground truth) — pre-existing plan, still independent value.
3. **V18-r32-capacity-only** (NEW, this review): warmstart V7.best, rank=32 LoRA, `lambda_kl=0`, image_aux schedule identical to V18, ≤10K steps. Probe KL drift identically to KL_DRIFT_REPORT.md protocol.

| original Stage C option | my verdict |
|---|---|
| A: V18-clean (`use_pred_latent=false`) | **DEFER** until capacity-only control runs. The B9 rebuttal in the prompt depends on A being true. |
| B: V18-family sweep (rank/blocks) | **REJECT** — marginal, doesn't move the needle on bottleneck attribution. |
| C: paper draft as-is | **REJECT** — see Q5. |
| D: transport-side intervention | **DEFER** — premise unsupported. |
| **E**: V13+V14 first | **APPROVE, extend to V13+V14+capacity-control** |
| F: pivot architecture | **REJECT, premature** — no evidence framework is exhausted; evidence is that we never measured it cleanly. |
| G: pivot data | **REJECT, premature** — model side not isolated yet. |
| H: terminate / feasibility paper | **REJECT** — three cheap runs gate this decision and they haven't run. |

**Slot allocation (≤3 parallel per standing rule #1):**
- Slot 1: V13 (~7 days)
- Slot 2: V14 (~7 days)
- Slot 3: V18-r32-capacity-only (~24-48h, then free)

Capacity-only finishes first → re-evaluate before V13/V14 land if drift result kills/saves A.

---

## 7. Q7 — Review cadence

**Verdict: MODIFY.** Round 13 itself proves reviewer value at strategic forks (B42 alone justifies this round). But Round 1-12 reviewer-heavy mode for tactical/yaml-level review is high-overhead.

Proposed standing rule **#10**:

> **Round 13+**: reviewer is mandatory when (a) project strategy is in question, (b) a major claim depends on a mechanistic interpretation of a single result, or (c) verdict requires grep-verifying numbers. Reviewer is optional/skip for tactical commits (yaml fields, single-file diffs, audit-prep markdowns) where user+claude already converge.

In concrete terms: **next mandatory reviewer round = after V13+V14+capacity-only land**, not before. Between now and then, user+claude handle launch decisions directly.

---

## 8. Final verdict summary (per §5 format)

### 8.1 Per-question (APPROVE/MODIFY/REJECT)

| Q | verdict | key reason |
|---|---|---|
| Q1 | MODIFY → B favored, A rejected | KL acts on z_pred (line 2230), not z_GT |
| Q2 | REJECT ceiling framing | image_aux + V8 + buggy KL = unresolved confounds |
| Q3 | MODIFY (partial APPROVE) | V8 shows image_aux is load-bearing but doesn't isolate it; V13 highest-EV |
| Q4 | REJECT premise | bottleneck attribution unmeasured; ranking provided conditionally |
| Q5 | REJECT paper-as-is | post-V13/V14/control, MICCAI/ISBI workshop appropriate |
| Q6 | APPROVE E + extend | V13 + V14 + new capacity-only control, 3 slots parallel |
| Q7 | MODIFY | standing rule #10: reviewer mandatory at strategic forks only |

### 8.2 Strategic recommendation (main verdict)

**E (extended) + new sub-option**: launch V13, V14, and V18-r32-capacity-only in parallel. Re-evaluate Stage C *after* the capacity-only control returns (~48h). Do not draft paper, do not pivot, do not run transport intervention until at least the capacity-only control closes the §1 mechanism question.

### 8.3 V18 KL drift interpretation (Q1)

**B** (generic LoRA capacity bleed-through). Mechanism: KL pullback ran on `decode(z_pred)` per train_first_hop.py:2230; probe measured `decode(z_GT)`; no V18 training loss ever supervised the probed path. Inverse drift is most parsimoniously explained by 589K trainable LoRA params absorbing residual error on the z_pred path with side-effect bleed to neighboring z_GT inputs. **New control required** (V18-r32-capacity-only) to falsify candidate A.

### 8.4 Project ceiling judgment (Q2)

**Not yet measurable.** Three confounds in series (V8 image_aux non-isolation, V18 buggy KL config, decoder/transport split varies by config) mean "+0.099 dB" is one trajectory under specific bugs, not a framework ceiling. Three cheap controls (V13, V14, capacity-only) gate the ceiling claim. Pivot or paper-as-is decisions taken before these land are not defensible.

### 8.5 New biases (B39–B43)

- B39: §1.2 baseline switch (+0.088 vs +0.099 same metric, different baseline, unflagged) — MODERATE
- B40: §1.2 arithmetic gloss off in both legs, sum accidentally agrees — LOW
- B41: §2.2 A "70% transport eat" is V18.best-only, sold as V18-general — LOW
- B42: §2.2 A mechanistically impossible (KL on z_pred ≠ probe on z_GT) — **HIGH**
- B43: Round 13 forgot Round 5's `use_pred_latent=true` bug flag — MODERATE

---

## 9. What I did NOT review

- Round 1-12 already-signed content (per §0 boundary)
- v3 audit task push timing (per §7 #8)
- KL drift probe code quality (Round 12 reviewer already verified)
- Anything not falsifiable from the cited substrate

---

## 10. One-paragraph executive summary for user

The prompt's central anchor — that V18's inverse-direction KL drift means "decoder LoRA is improving on the GT manifold, so transport is the bottleneck" — is mechanistically impossible. V18 ran with `use_pred_latent=true` (`train_first_hop.py:2230`), so KL pullback supervised `decode(z_pred)`, never `decode(z_GT)`. The probe measured `decode(z_GT)`. No training loss ever touched the probed path. The +0.10 dB is almost certainly generic LoRA capacity bleed-through, not KL alignment success. Round 5 had already flagged `use_pred_latent=true` as a bug; Round 13 reasoned as though the bug were the design. Before any Stage C pivot/paper/transport-intervention decision, run three parallel cheap controls (V13 for image_aux isolation, V14 for d_pure ground truth, a new V18-r32-capacity-only for the LoRA-vs-KL disambiguation). All other strategy options — pivot architecture, pivot data, paper as-is, transport scale-up — are premature.
