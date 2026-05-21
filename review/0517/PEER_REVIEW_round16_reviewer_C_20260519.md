# Round 16 Peer Review — Reviewer C (independent)

- **reviewer name**: **C** (consistent with R15)
- date: 2026-05-19
- branch: foc_lite_hop0 (commit 962bc91)
- substrate read independently (no other reviewer drafts seen):
  - PEER_REVIEW_PROMPT_round16_A3_results_20260519.md (the prompt)
  - review/0517/V18_capacity_only/V18_CAPACITY_ONLY_A3_REPORT_20260519.md
  - review/0517/V18_capacity_only/kl_drift_with_cap_20260519_180206/KL_DRIFT_REPORT.md
  - review/0517/V18_capacity_only/V18_capacity_only_metrics_20260518.jsonl (full-val rows 165K + 170K)
  - review/0517/V18_decoder_lora/V18_FINAL_RESULTS_20260518.md (V18.best@165K reference)
  - train_first_hop.py:2225-2235 (z_kl source verbatim — B42 mechanism)
- mechanical verifications I ran first: see §0.

---

## TL;DR (Reviewer C verdict)

| Item | Verdict |
|---|---|
| Q1 KL direct-decoder narrative killed? | **APPROVE candidate A**: matched-step Δ=+0.0020 dB is a tie within probe noise; combined with B42, KL pullback contribution to direct-decode is **mechanistically excluded and empirically tied to zero or slightly negative** |
| Q2 chain metric also tied at matched step | **APPROVE — but stronger than prompt frames**: A3@165K is fractionally *better* than V18.best@165K on both chain and select_score |
| Q3 implications for V18/decoder/transport | **MODIFY** — A3 also kills "decoder capacity improves chain" claim; the surviving claim is narrower than prompt suggests |
| Q4 now vs wait | **APPROVE the split, MODIFY the items** — 3 conclusions now, 2 must wait |
| Q5 V18-clean priority | **REJECT V18-clean** — no mechanism survives, ROI dead |
| Q6 paper narrative | **APPROVE narrative B with one qualifier**: decoder capacity improves direct GT-manifold decode by ~+0.17 dB at 10K LoRA training, **but this gain does not propagate through chain rollout** (must say this explicitly) |
| Q7 prompt biases | **3 new (B61-B63)**, all LOW; B61 is "narrative-C anchor" in Q5 |
| Main verdict | **A** (KL direct-decoder narrative basically dead; V18 framework value reduced to a narrow direct-decode demonstration) — but with explicit V13/V14 dependence for chain/significance claims |

---

## 0. Mechanical verifications I ran first

### 0.1 A3 metrics.jsonl matches the report (✓)

I grepped the two full-val rows from `V18_capacity_only_metrics_20260518.jsonl`:

```python
step=165000 -> val_select_score=0.000903735346932283, val_chain_normal_mse=0.00024549206631093893
step=170000 -> val_select_score=0.0009039954006580368,  val_chain_normal_mse=0.00024540630802570037
```

Prompt §1.2 numbers match to all displayed digits. ✓

### 0.2 V18.best@165K reference matches V18_FINAL_RESULTS.md (✓)

`V18_FINAL_RESULTS_20260518.md` lines 30/34/41:
```
best_val=0.0009037493852408773
val_chain_normal_mse=0.0002454947041527113
```

Prompt §2.1 + Q2 numbers match. ✓

### 0.3 Matched-step direct decode delta is real (✓)

KL_DRIFT_REPORT.md confirms `V18-cap.last(170K) NORMAL = 52.8047` and `V18.step170k NORMAL = 52.8027`. My recompute: `52.8047 − 52.8027 = +0.0020 dB`. ✓

### 0.4 train_first_hop.py:2230 unchanged (✓ — B42 still holds)

Re-read lines 2225-2235:
```python
if lambda_kl > 0.0:
    z_kl = main_out["z_pred"] if bool(kl_cfg.get("use_pred_latent", True)) else main_batch["z_dst"]
```
KL is gated off entirely when `lambda_kl == 0.0` (A3 case). When KL is on (V18 case, `use_pred_latent=true`), `z_kl = z_pred` — never `z_dst` (GT). So:
- A3 has zero KL gradient ever.
- V18's KL gradient only ever flows through the z_pred path.
- The direct probe measures `decode(z_GT)` — neither A3 nor V18 has any training loss attached to this path.

B42 mechanism stands. The +0.10 dB ⟨decode_V18(z_GT) − decode_V7(z_GT)⟩ Round 12 anomaly is mechanistically forced to be capacity-driven; A3 empirically confirms.

### 0.5 Independent computation: A3@165K vs V18.best@165K matched-step chain delta

This is the headline I did not see emphasized in either the prompt or the A3 report. Both checkpoints are at step 165000, same V7-warmstart, same LoRA config, same image_aux. Only delta is KL pullback (V18 has `lambda_kl=0.05`, A3 has `lambda_kl=0.0`).

| metric | A3@165K | V18.best@165K | Δ (A3 − V18) | sign |
|---|---:|---:|---:|---|
| `val_select_score` | 0.000903735347 | 0.000903749385 | **−1.41 × 10⁻⁸** | A3 better |
| `val_chain_normal_mse` | 0.000245492066 | 0.000245494704 | **−2.64 × 10⁻⁹** | A3 better |

Both deltas are 7-8 orders of magnitude below the metric itself and far below any plausible float-precision floor (mixed-precision = O(1e-5), float32 = O(1e-7)). **A3 and V18.best at matched step are statistically and numerically indistinguishable; if anything, A3 is fractionally better.** KL pullback at `lambda_kl=0.05` over 5K LoRA training steps contributes ≈ zero (possibly slightly negative) to the chain selection metric.

This is the strongest single piece of A3 evidence the prompt has, and it deserves a row of its own in any downstream summary.

---

## 1. Q1 — Has A3 killed the "KL explains V18 direct GT-manifold gain" narrative?

**Verdict: APPROVE candidate A** — but state it precisely.

The cleanest possible matched-step comparison (V18.step170k vs A3 at step 170K, same V7-warmstart, same LoRA config, same image_aux, same data, KL the *only* difference) gives Δ = **+0.0020 dB on NORMAL direct decode**. Combined with §0.5's near-zero chain delta at step 165K, the evidence is bidirectional: KL contributes ≈ 0 on **both** the direct GT-manifold probe **and** the chain selection metric, **at matched training step**.

Mechanism (B42) had already excluded direct KL-to-z_GT alignment. A3 empirically confirms that even via *indirect* gradient pathways (KL on z_pred → shared decoder weights → bleed to z_GT eval), the contribution is at best statistical noise.

**Most accurate phrasing**:

> "In the V18 buggy-KL configuration (`use_pred_latent=true`, `lambda_kl=0.05`), the KL pullback term contributes ≤ 0.003 dB at matched LoRA-training step on both direct GT-manifold decode and chain selection metric, against a same-step LoRA-capacity baseline. The Round 12 +0.10 dB drift is fully attributable to LoRA decoder capacity (mechanism B42 confirmed by A3 control)."

I reject the "保守说法" framing in the prompt as understating the result. The result is not "KL on direct decode is killed but rollout/path coupling unresolved" — A3 also matches V18 on chain. The only residual unknowns are (a) whether the *corrected* `use_pred_latent=false` variant would have helped (which would require V18-clean to run, ROI now low), and (b) whether KL at much larger `lambda_kl` or longer LoRA training does something different (academic; not worth GPU).

---

## 2. Q2 — Chain metric at matched step

**Verdict: APPROVE the prompt's reading, but it's stronger than presented.**

The prompt asks "does this mean V18's 160K→165K chain improvement is also explained by capacity-only?" Answer: **yes, completely.** §0.5 above gives the numbers. At 5K LoRA training from V7.best, A3 (no KL) and V18 (KL on) produce bit-essentially-identical full-val chain metrics. V18's headline `+0.030 dB` chain gain at 165K vs V7 (from V18_FINAL_RESULTS) is therefore **entirely** the LoRA-capacity component (or, equivalently, the image_aux-gradient-channelled-through-LoRA component); none of it is KL.

The prompt's framing "V18 有效 → 改写为 decoder capacity 微幅有效, 但 KL 设计未显示出额外贡献" is the right rewrite. I'd push it one notch further:

> **V18's training gain on chain is fully captured by an equivalent capacity-only run. KL design contributes ≤ noise at matched step. The "decoder LoRA partial improvement" framing should reduce to "decoder LoRA at rank=32, last 2 blocks adds approximately +0.030 dB chain at 5K LoRA training, attributable to capacity, not to KL pullback design."**

Q2 missing-evidence list: nothing critical for this specific claim. The matched-step comparison at 165K is as clean as control experiments get.

---

## 3. Q3 — V18 / decoder / transport implications

**Verdict: MODIFY the prompt's framing** — A3 evidence is narrower than prompt-Q3 implies for both decoder and transport.

### 3.1 V18 / KL line

- Remaining justification to invest: **near-zero**. KL design contributes ≈ 0 on both probes. Mechanism (B42) is also excluded.
- V18-clean (`use_pred_latent=false`) resurrection value: **none mechanistically supported**. The only argument would be "test whether the *right* KL formulation works", but with A3 showing capacity-only already matches V18, V18-clean would at best tie and at worst lose to A3. No story.
- Recommendation: retire the KL-pullback line in current form. Do not launch V18-clean.

### 3.2 Decoder-capacity line

The prompt frames A3 as proving "decoder capacity 确实能改善 GT-manifold direct decode, 但链路收益极弱". This is correct on the surface but understates a sharper claim:

| metric | direct decode (V18-cap.last − V7.best) | chain (V18-cap@165K − V7.best best-vs-best) |
|---|---:|---:|
| NORMAL | **+0.1707 dB** | **+0.0302 dB**¹ |
| ratio (chain/direct) | — | ~18% |

¹ Inferred: A3@165K chain = V18.best@165K chain (§0.5) and V18.best-vs-V7.best NORMAL chain = +0.0302 dB per V18_FINAL_RESULTS.

So 10K LoRA training delivers ~0.17 dB on GT-manifold direct decode but only ~0.03 dB on chain rollout — about **18%** of the decoder improvement propagates through transport. The other 82% is absorbed (or, more precisely: the chain metric is dominated by transport error such that decoder improvements at this magnitude are largely irrelevant to chain output).

**Sharper Q3.2 conclusion**: decoder is **not** the bottleneck for chain. Further decoder rank/blocks sweeps will produce more direct-decode gain that won't reach chain output. EV of decoder sweeps for the *project goal* (PET chain reconstruction) is very low.

### 3.3 Transport line

The prompt's Q3.3 frames A3 as "strengthening transport-is-real-bottleneck" — **MODIFY**. A3 doesn't prove transport is *the* bottleneck. It proves decoder LoRA gains don't propagate to chain. That's compatible with three different worlds:

| world | description |
|---|---|
| **W1**: transport-error-dominated | Chain error is dominated by transport noise; decoder is already good enough at chain-input distribution; transport intervention has high EV |
| **W2**: jointly saturated | Both decoder and transport contribute, but the framework as a whole has hit a ceiling; pivot needed (architecture / data) |
| **W3**: decoder-mismatched-to-z_pred | LoRA on decoder helped GT-input but didn't help z_pred-input because LoRA was trained mostly with KL on z_pred path which was suboptimal; a decoder LoRA trained explicitly on `MSE(decode(z_pred), x_target)` (= image_aux only, more LoRA training) might show better chain transfer |

A3 alone cannot disambiguate W1/W2/W3. V13 is partially informative (if image_aux ≈ all V7-V8 gain → image_aux already saturates chain transfer of decoder behavior → W3 unlikely). V14 fixes the significance denominator but doesn't disambiguate W1 vs W2.

**To genuinely confirm "transport is the bottleneck", a new experiment is required**: e.g. swap-in a deliberately *better-than-V7* transport (e.g. larger DiT, more sampling steps, multi-step refinement) keeping decoder = V7 frozen, and measure chain Δ. Until that experiment, W1/W2 are equally compatible with A3.

---

## 4. Q4 — Now vs wait

**Verdict: APPROVE the split. Adjust the lists.**

### 4.1 Conclusions that can be drawn NOW (V13/V14-independent)

1. **B42 + A3 jointly bury the KL-pullback success narrative.** Any paper-narrative still pitched on "KL pullback aligned the decoder" must be removed.
2. **V18 chain gain (+0.030 dB best-vs-best, +0.062 dB last-vs-best) is fully attributable to LoRA capacity, not KL.** At matched step, A3 ≈ V18 on chain (§0.5).
3. **Decoder LoRA capacity gains do not propagate to chain ≥ ~18%.** Direct decode improves by +0.17 dB (10K LoRA), chain by +0.03 dB. Decoder is not the chain bottleneck. Further decoder sweeps have low EV.
4. **V18-clean / V19 / V18 rank-sweep should NOT launch.** No mechanism story survives A3.
5. **Standing rule B43 ("V18 is in buggy KL config") is now empirically demonstrated** to be a no-op rather than a confound — the buggy KL contributes ≤ noise, so V18's measured numbers are not corrupted by buggy KL; they are honestly equal to "LoRA capacity + image_aux at rank=32, 2 blocks, 5K–40K LoRA training". The "buggy KL" prefix is no longer required on every V18 number.

### 4.2 Conclusions that MUST WAIT for V13/V14

1. **Whether V7 main-line gain came from image_aux** — V13 disambiguates.
2. **Whether V18 chain Δ (+0.030 / +0.062) exceeds d_pure noise floor** — V14 sets the denominator.
3. **Project ceiling claim** — requires both V13 and V14 (image_aux saturation + significance threshold).
4. **Paper publishability** — depends on whether (V18 chain Δ) > (d_pure noise) AND whether (V13 image_aux Δ) is sufficient to be a paper main result independent of V18.
5. **Whether to pivot to transport-side intervention** — V13 informs whether *anything* improves chain at all by ≥ noise.

### 4.3 Immediate actions claude/user can take NOW

- Update V18_design_rationale.md §5.1/5.2: add Round 16 conclusion that B42 + A3 → KL contribution ≈ 0 at matched step. Remove or qualify any text claiming KL design success.
- Stop drafting any paper text that includes "KL pullback aligned decoder" or "decoder LoRA fixed encoder-decoder asymmetry".
- Do NOT update V18_FINAL_RESULTS.md headline numbers (they're correct), but DO add an interpretation note: "headline +0.030 chain dB is empirically equivalent to a same-step KL-off capacity-only run".

### 4.4 Immediate actions to PAUSE

- All V18-clean / V19 / V18 rank-sweep design work.
- Paper-draft framing of V18 as a "method contribution".
- Any "transport is the bottleneck" claim that goes beyond "A3 shows decoder LoRA gains don't propagate through chain" (W1/W2/W3 unresolved per §3.3).
- Architecture / data pivot deliberation — V13/V14 first.

---

## 5. Q5 — V18-clean / new KL control priority

**Verdict: REJECT V18-clean as high priority. Defer or drop all KL-specific controls.**

| candidate | priority after A3 | reason |
|---|---|---|
| **V18-clean** (`use_pred_latent=false`) | **DROP** | A3 ties V18 on both probes at matched step. V18-clean would, at best, match A3. No paper story. Costs ~7 days GPU for an already-resolved question. |
| **Larger rank / more blocks V18 sweep** | **DROP for project goal** (keep as appendix curiosity at most) | §3.2 shows direct-decode gains don't propagate to chain. Sweeping rank/blocks would push direct decode higher with diminishing chain return. EV near zero for the PET chain task. |
| **Transport-side intervention** | **NOT YET — wait for V13** | §3.3: A3 doesn't distinguish W1/W2/W3. Launching transport intervention now is premature; V13 narrows the search space. |
| (new) **Run V14 noise floor analysis tooling early** | **MEDIUM** | Once V14 lands, the per-slice noise CSV is the denominator for everything. Pre-build the noise-floor analysis script now (no GPU cost) so V14 outputs can be analyzed in <1h after they land. |
| (new) **V18-cap-only longer** (e.g., 30K LoRA training) | **REJECT** | A3 already shows decoder gains don't transmit to chain. Training capacity-only longer at most pushes direct decode higher with no chain transfer. Confirmatory but not informative. |

The only KL-specific question A3 leaves open is "would a *correctly-formulated* KL pullback have helped?" That's hypothetical — there is no formulation of KL that we have empirical reason to expect would beat capacity-only. Reject.

---

## 6. Q6 — Paper narrative

**Verdict: APPROVE narrative B with one critical qualifier. REJECT narrative A. Hold on C until V13/V14.**

| narrative | verdict | reasoning |
|---|---|---|
| **A**: "KL pullback improves decoder manifold, transport absorbs it" | **REJECT — retract immediately** | A3 disproves KL contribution; the manifold-improvement story is mechanism-impossible (B42) and empirically tied to zero. |
| **B**: "Decoder capacity (LoRA) improves GT-manifold direct decode, but improvement does not propagate to chain" | **APPROVE with explicit qualifier** | Honest and exactly what A3 shows. The qualifier "does not propagate" is essential — without it, narrative B can be misread as a method contribution. With it, B is a structural finding about the latent-transport-with-frozen-decoder framework. |
| **C**: "V18 family is marginal; defer paper narrative to V13/V14" | **HOLD** | Correct procedure. V13 tells us if image_aux is the historical signal. V14 tells us noise floor. Without both, no paper-publishability claim is defensible. Once V13/V14 land, the choice between B and a stronger C-variant becomes decidable. |

**Recommended paper-narrative position as of today:**
1. Withdraw all V18-as-method framing from draft.
2. Treat A3 + Round 16 finding as a structural property of the framework: "LoRA decoder capacity at rank=32, last 2 blocks, gives ~+0.17 dB direct GT-manifold decode but ≤+0.05 dB chain rollout PSNR, indicating chain output is decoder-saturation-insensitive in this regime." This is a publishable observation in its own right (negative result + structural insight) but does NOT yet support a "method" claim.
3. Hold final narrative decision until V13+V14 land.

The prompt's Q6 should be amended to make B's "does not propagate" qualifier explicit; without it, narrative B is open to the same mis-anchoring that narrative A had.

---

## 7. Q7 — Round 16 prompt biases (B61+)

**3 new biases identified, all LOW severity.**

### B61 (LOW) — Q5 framing subtly anchors toward "transport main line"

Q5 ends with "C 是否因此被动升为更合理主线". The choice of "被动升" anchors toward "transport rises by default". As §3.3 shows, A3 doesn't distinguish W1/W2/W3, so transport-as-main-line is not automatic. Better framing: "Q5: which (if any) of A/B/C has positive EV given A3? Should default be to pause new launches until V13/V14?"

### B62 (LOW) — Q3.3 frames "transport 反而强化为未解" as if confirmed

Q3.3 says "A3 是否反而强化了 '真正未解的是 z_pred / rollout / chain path' 这个判断". This phrasing presumes transport IS the unsolved part. A3 only shows decoder isn't the chain bottleneck — joint saturation (W2) is equally consistent. Reviewer might overcommit to a "go fix transport" conclusion not supported by A3 alone.

### B63 (LOW) — Q6 narrative-B is presented without "does not propagate" qualifier

Q6 narrative B as written ("decoder capacity 可改善 GT-manifold direct decode, 但 improvement 不自动转化为 transport chain gain") is *almost* right but reads as "transport ate it". The accurate phrasing is stronger: "decoder gain does not translate, full stop — and we don't yet know whether transport ate it, or whether the framework is jointly saturated, or whether the chain metric is insensitive". This matters for paper-narrative honesty. Suggested wording in §6 above.

### What I did NOT find as a bias

- The prompt is **not** mixing direct-decode and chain conclusions (§6 constraint #1 honored).
- The prompt is **not** ignoring B42 (§6 constraint #2 honored).
- The prompt is **not** pre-judging V13/V14 (§6 constraint #3 honored).
- The prompt does NOT fabricate a user quote (Round 15's B55 issue). All numerical claims I sampled match source artifacts.

The prompt is the cleanest substrate I've reviewed in the Round 12-16 series.

---

## 8. Output per §5 format

### 8.1 Per-question

| Q | verdict |
|---|---|
| Q1 KL direct-decoder narrative | **APPROVE candidate A** — KL contribution ≤ noise at matched step, mechanism (B42) excludes direct path |
| Q2 chain at matched step | **APPROVE — stronger than presented** — A3@165K ≈ V18.best@165K to 8-9 sig figs on both metrics, A3 fractionally better |
| Q3 V18/decoder/transport implications | **MODIFY** — A3 also kills "decoder is chain bottleneck" claim; transport-as-main-bottleneck is one of W1/W2/W3, not confirmed |
| Q4 now vs wait | **APPROVE split, adjust items per §4.1-§4.4** |
| Q5 V18-clean / new KL control | **REJECT V18-clean, DROP rank sweep, DELAY transport intervention** |
| Q6 paper narrative | **APPROVE B with "does not propagate" qualifier; REJECT A; HOLD on C until V13/V14** |
| Q7 new biases | **3 new (B61-B63), all LOW** |

### 8.2 Main verdict

**A**: A3 has functionally killed the KL direct-decoder narrative. But with two precisions on top of the prompt's framing:
1. A3 also ties V18 on **chain** at matched step (not just on direct decode), so the kill is broader than "direct-decode only".
2. The remaining story is not "transport is the bottleneck" — it's "decoder LoRA gains don't propagate to chain", which is compatible with multiple framework-saturation pictures (W1/W2/W3 per §3.3).

### 8.3 Immediate executable suggestions

**Do now (V13/V14 still running)**:
- Update V18_design_rationale.md §5: append Round 16 finding (KL ≤ noise at matched step, capacity is the explanation).
- Withdraw "KL pullback aligned decoder" language from any paper/poster/abstract draft.
- Pre-build V14 noise-floor analysis script (no GPU cost, ~30 min, ready when V14 lands).
- Add interpretation footnote to V18_FINAL_RESULTS.md: "headline +0.030 dB chain gain is empirically equivalent to capacity-only at same step".

**Pause now**:
- All V18-clean / V19 / V18 rank-sweep design.
- Transport-intervention launch design (wait V13 to narrow W1/W2/W3).
- Architecture / data pivot debate (wait V13/V14 to set baseline).
- Paper-narrative committal (wait V13/V14 to know if framework has *any* main result).

### 8.4 New biases (B61-B63)

Detailed §7 above. All LOW. No B55-class fabrication; this prompt is cleanly grounded in source artifacts.

---

## 9. One-paragraph executive summary

A3 is the cleanest control experiment in the Round 12-16 series, and the matched-step evidence is stronger than the prompt frames it. At step 170K direct GT-manifold decode, V18-with-KL and V18-without-KL differ by +0.0020 dB (within probe noise). At step 165K chain selection metric, the same comparison differs by −1.4×10⁻⁸ (A3 fractionally better than V18). Combined with the standing B42 mechanism (KL pullback runs on z_pred path, never on z_GT), the KL pullback term in V18's current formulation contributes ≤ noise on **both** direct decode and chain — not just direct decode as the prompt suggests. The V18 framework's surviving claim narrows sharply: LoRA decoder capacity at rank=32, last 2 blocks delivers ~+0.17 dB GT-manifold direct-decode gain at 10K LoRA training, of which only ~+0.03 dB (~18%) reaches the chain output. This is a structural property of the latent-transport-with-frozen-decoder framework, not a method contribution. V18-clean / rank-sweep / new KL controls should be dropped. Transport-side intervention is a candidate but A3 alone cannot confirm transport is THE bottleneck (W1/W2/W3 in §3.3 are equally consistent). The five "wait for V13/V14" items are correct as listed. Paper-narrative committal should be paused until V13/V14 land; the surviving narrative B ("decoder capacity improves direct decode but does not propagate to chain") is publishable as a structural negative result but should not be sold as a method.

---

## 10. What I did NOT review

- V13 / V14 numerical results (still running per prompt; no local artifacts to verify).
- A3 yaml / task md execution quality (covered in R14).
- Round 1-15 already-signed content.
- audit DRAFT release timing.
