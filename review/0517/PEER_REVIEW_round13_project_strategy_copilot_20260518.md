# Peer Review Round 13 — Copilot Independent Project Strategy Review

- date: 2026-05-18
- reviewer: GitHub Copilot
- prompt: `PEER_REVIEW_PROMPT_round13_project_strategy_20260518.md`
- scope: V18 KL drift interpretation, historical PSNR trajectory, Stage C strategy

---

## 0. Executive Verdict

**Main Stage C recommendation: E — launch V13 + V14 disambiguation first, then decide whether to write, pivot, or design a transport-side intervention.**

Do **not** launch V18-clean, V18 rank/block sweeps, or a new transport-side architecture run before V13/V14. The project has two unresolved causal facts that are more decision-relevant than another V18-family variant:

1. the V7-V8 `+0.308 dB` number is still a joint effect, not an image_aux effect;
2. the V7-V6_NOISE / V6_NOISE-V18 deltas are still mixed with seed and step-weight differences, so the `+0.088 dB total project gain` is directionally sobering but not a clean ceiling estimate.

My ceiling judgment: **current V7/V18-style latent-transport + decoder-LoRA family is very likely near its practical ceiling, but the project-wide architecture/data ceiling is not established.** Continue micro-tuning is low EV. The next high-EV action is not a new clever tweak; it is to lock the causal substrate with V13/V14 so the paper/pivot decision is evidence-based.

---

## 0.5 Evidence I Checked

- KL drift artifacts: `V18_decoder_lora/kl_drift_20260518_123834/KL_DRIFT_REPORT.md`, `KL_DRIFT_SUMMARY.json`, `KL_DRIFT_PER_SLICE.csv`, `probe.log`
- Historical PSNR: `../0516/PLANF_FINAL_ANALYSIS_20260516.md`, `V18_decoder_lora/V18_FINAL_RESULTS_20260518.md`, `V18_decoder_lora/V18_METRIC_CORRECTION_REPORT_20260518.md`
- Prior integration: `REVIEW_INTEGRATION_round12_20260518.md`, `NEXT_STAGE_ARCH_CODE_FINAL_20260517.md`
- Code path: `../../pet_lr/model_first_hop.py` `decode_crop`, `../../pet_lr/decoder_lora.py`, `../../train_first_hop.py`, `../../tools/probe_v18_kl_drift.py`, `../../eval_first_hop_224_clip3.py`
- Disambiguation configs: `../0516/V13_true_image_aux_ablation/V13_true_image_aux_off.yaml`, `../0516/V14_true_d_pure/V14_v7_seed1337.yaml`

Mechanical KL drift checks from `KL_DRIFT_PER_SLICE.csv`:

| timepoint | V18.best - V7 mean | paired sd | win rate | V18.last - V7 mean | paired sd | win rate |
|---|---:|---:|---:|---:|---:|---:|
| D20 | +0.049347 | 0.184966 | 98.70% | +0.112561 | 0.418499 | 86.94% |
| D10 | +0.055712 | 0.170656 | 99.03% | +0.119385 | 0.387495 | 86.38% |
| D4 | +0.069228 | 0.153633 | 99.14% | +0.131530 | 0.349236 | 84.67% |
| NORMAL | +0.097290 | 0.177983 | 99.39% | +0.163921 | 0.411655 | 81.17% |

This makes a pure mean-tail artifact unlikely. The probe log also shows `missing=0 unexpected=0` for V7.best, V18.best, and V18.last checkpoint loads.

---

## 1. Seven Questions

### Q1 — V18 KL Drift Inverse Meaning: MODIFY

I choose **A-prime**: the measured fact is real — V18 decodes GT latents better than V7 under the canonical `decode_crop(z_GT)` probe — but the causal interpretation in A is too strong.

- **Reject C as primary explanation.** `decode_crop` uses the same wrapper logic; the V18 path differs because selected decoder linears are wrapped with LoRA, but checkpoint loading is clean and the per-slice win rates are broad. If this were mostly evaluator noise, I would not expect V18.best to beat V7 on 98.7-99.4% of slices across timepoints.
- **Partially accept B.** The result proves trained decoder LoRA improved GT-latent decoding. It does **not** prove the improvement came specifically from the KL pullback design. It could be generic supervised decoder capacity induced by image_aux/rollout gradients plus LoRA trainability.
- **Modify A.** The result refutes “V18 failed because decoder drift damaged the GT manifold.” It does not prove “KL pullback is the mechanism that improved GT manifold.”

New control experiment priority: **not now**. A lambda_kl=0 / use_pred_latent=false control would be scientifically useful, but V18’s end-to-end PSNR gain is too small for another V18-family run to outrank V13/V14.

### Q2 — Project Ceiling: MODIFY

`+0.088 dB` is a credible warning sign but not a clean ceiling estimate.

Reasons:

- The V6/V7/V8 family is confounded by seed and step weights; V6_NOISE-to-V18 is a project-level historical comparison, not a causal ablation.
- The prompt mixes V6_NOISE.best and V6_NOISE.last baselines: V18.last is `+0.099 dB` vs V6_NOISE.best, but `+0.088 dB` vs V6_NOISE.last. Both can be reported, but the label must say which endpoint is used.
- The “11 dB attackable gap” framing is dangerous. It is an RAE round-trip / transport-vs-ceiling gap in dB space, not a linear resource pool where `0.088 / 11` means “0.8% recovered.”

Strategic reading: the **current framework variant** is near ceiling; the **problem** is not proven solved or hopeless. Architecture and data may still have room, but micro-tuning V18/image_aux schedules is low EV.

### Q3 — V8 Regression and image_aux: MODIFY

The V8 regression supports “something about the V7 configuration matters,” but it does **not** isolate image_aux.

V8 differs from V7 in at least image_aux and step_weights. Therefore:

- it is fair to say image_aux remains a prime suspect and should not be casually retired;
- it is not fair to say V8 proves image_aux contributes `0.27-0.31 dB`;
- V13 is mandatory if the paper will claim image_aux as a pillar.

I would launch V13 before any transport redesign. V14 should run too because the project’s meaningful deltas are now in the `0.03-0.10 dB` range, where the true seed envelope matters.

### Q4 — Decoder LoRA Gain vs Transport Swallowing: MODIFY

If A-prime is accepted, the implication is: **decoder LoRA can improve the decoder-side reconstruction surface, but the transport pipeline does not deliver latents that exploit most of that improvement.** That points away from decoder sweeps.

Transport-side priorities after V13/V14:

| priority | option | reviewer view |
|---:|---|---|
| 1 | B: architecture change | Highest EV if continuing research; current single-family latent DiT appears saturated. |
| 2 | C: data increase | High EV if extra PET data is realistically available; otherwise not actionable. |
| 3 | E: multi-step refinement | Worth a cheap probe if it can be eval-time or low-training-cost; do not spend 7 days blind. |
| 4 | A: backbone scale-up | Only after a scaling sanity check; scale-up can preserve the same bottleneck. |
| 5 | D: image_aux schedule tuning | Low EV until V13 proves image_aux has a clean large effect. |

### Q5 — Paper Worthiness: MODIFY

A paper is possible, but the narrative must be narrower and more honest than “we found the winning PET latent transport recipe.”

Most publishable narrative:

1. **PET latent transport feasibility and limitations study** — strongest if V13/V14 quantify which historical gains survive causal cleanup.
2. **Negative result / ceiling analysis for latent transport under PET dose progression** — publishable if framed as rigorous diagnostics, not as failed optimization.
3. **Decoder LoRA partial improvement** — too weak alone; V18 best-vs-best is `+0.030 dB`, last-vs-best `+0.062 dB`, and the endpoint choice is fragile.

Venue fit: ISBI / MICCAI workshop or a careful MICCAI submission if the diagnostic package is strong. TMI / MedIA likely needs either stronger gains, broader datasets, or a more general methodological contribution.

Paper-ready minimum: **V13 + V14 + corrected V18 reporting**. V18-clean is not required unless the paper wants to make a KL pullback mechanism claim.

### Q6 — Stage C Path: APPROVE E

I recommend **E: launch V13 + V14 disambiguation, then decide**.

Reject for now:

- **A V18-clean:** KL drift is opposite of the harmful-drift hypothesis. The run would answer a mechanism question, not the main strategy question.
- **B V18-family sweep:** GT-manifold improvement exists, but end-to-end gains are sub-PARTIAL/KILL depending on endpoint. More decoder capacity is unlikely to rescue the transport bottleneck.
- **C paper as-is:** too many core causal claims remain confounded.
- **D transport-side intervention now:** premature; V13/V14 can materially change whether image_aux/step_weights are pillars or historical noise.
- **F/G/H immediate pivot/terminate:** plausible soon, but V13/V14 are the cheap-in-decision-space evidence needed before making that call.

If slots are constrained: launch **V13 first**, then V14. If both slots are available, run both. While they train, draft the paper skeleton and decision table, but do not commit to claims.

### Q7 — Review Cadence: MODIFY

Reduce review volume after Round 13, but keep review gates for strategic substrate changes.

Recommended standing rule:

- **Strategic claims / pivot decisions / paper claims:** require one focused independent review with mechanical evidence checks.
- **Tactical YAML fields / task docs / code hygiene:** user + Claude/Copilot direct decision is enough unless tests or probes fail.
- **Numerical substrate:** prefer scripts, per-slice CSV checks, and explicit endpoint tables over another long prose review.

After V13/V14 complete, do one compact strategy review. Do not restart a multi-round meta-review loop unless the new data contradicts the current framing.

---

## 2. Main Strategic Recommendation

**Pick E now.** The next decision tree should be:

| V13/V14 outcome | Interpretation | next action |
|---|---|---|
| V13 large, V14 small | image_aux truly matters; old story partially rehabilitated | paper + targeted transport/objective redesign |
| V13 small, V14 small | V7-V8 gain was mostly step_weights/confound | pivot architecture/data or feasibility paper |
| V14 large | all `0.03-0.09 dB` gains are within weak seed envelope | paper must become diagnostic/negative or gather more seeds/data |
| V13 large but still small total V18 gain | image_aux is real but current transport family saturated | architecture/data pivot with image_aux retained as component |

No new V18-family run should start until this table is filled.

---

## 3. V18 KL Drift Interpretation

The KL drift result is best written as:

> V18 decoder LoRA improves direct GT-latent decoding under the canonical probe (`+0.097 dB` best, `+0.164 dB` last on NORMAL), so V18’s weak end-to-end gain is not explained by harmful GT-manifold decoder drift. The result does not by itself identify whether KL pullback, image_aux gradients, or generic LoRA capacity caused the GT-manifold improvement.

This is a useful scientific finding, but it weakens the case for V18-clean as the next run.

---

## 4. Project Ceiling Judgment

Current framework ceiling: **probably close**.

Global project ceiling: **not proven**.

The safest wording is:

> Across the currently tested V6/V7/V18 lineage, full-val NORMAL PSNR improves by only about `0.09-0.10 dB`, while V18 decoder adaptation yields only `+0.030 dB` best-vs-best and `+0.062 dB` last-vs-best over V7. This suggests the current latent transport family has little remaining headroom under the present data/configuration, but causal attribution remains blocked by V13/V14.

---

## 5. New Biases / Drift Risks

### B39 — Ambiguous historical baseline endpoint

The prompt reports both `+0.099 dB` and `+0.088 dB` for V6_NOISE to V18.last depending on whether V6_NOISE.best or V6_NOISE.last is used. This is not fatal, but future tables must label the endpoint explicitly.

### B40 — dB gap linearization

The statement “ate only 0.9% of an 11 dB gap” treats dB as a linear budget. It should be removed or rewritten as a qualitative contrast.

### B41 — KL drift sign to mechanism jump

Negative drift proves V18 improves direct GT-latent decoding; it does not prove KL pullback is the reason.

### B42 — Endpoint-favorable V18 interpretation

V18.last is useful as an endpoint, but best-vs-best remains the stricter trained-selection comparison. Any V18 claim must report both.

---

## 6. Final Answer in Prompt Format

- 5.1 seven questions: Q1 MODIFY, Q2 MODIFY, Q3 MODIFY, Q4 MODIFY, Q5 MODIFY, Q6 APPROVE E, Q7 MODIFY.
- 5.2 strategic recommendation: **E — launch V13+V14 disambiguation first**.
- 5.3 KL drift: **A-prime true as measurement, mechanism unresolved; no immediate V18-clean**.
- 5.4 ceiling: **current framework likely near ceiling; global architecture/data ceiling unresolved**.
- 5.5 new biases: B39-B42 above.
