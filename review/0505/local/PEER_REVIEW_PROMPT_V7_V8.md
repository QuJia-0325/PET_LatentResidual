# 0505 Local → External AI Peer Review Prompt — V7 / V8 Sanity Ablation

**Purpose**: 把这段 prompt 喂给 GPT-5（high reasoning） / Gemini-2.5-Pro / Claude-Opus 等**外部** AI（非本仓库 agent），对 `review/0505/local/` 下的 V7（Grönwall 闭式 raw step_weights）+ V8（拿掉 image_aux）50K sanity 实验设计做 adversarial peer review。
**建议跑两家以上求交**（GPT-5 + Gemini-2.5-Pro 至少其一）。
**用法**: 复制 `--- PROMPT BEGIN ---` 与 `--- PROMPT END ---` 之间的全部内容，**连同附件文件**一并粘贴。**不要改 prompt 措辞**；语气有意 adversarial，目的是在 GPU 投产前抓到方法学漏洞。

---

## 必须随 prompt 一并提供给外部 AI 的附件

| 路径 | 内容 |
|---|---|
| `review/0505/local/ANALYSIS.md` | V7/V8 实验设计 & 决策矩阵全文 |
| `review/0505/local/configs/V7_gronwall_raw.yaml` | V7 完整配置（含 yaml-头注释推导） |
| `review/0505/local/configs/V8_no_image_aux.yaml` | V8 完整配置 |
| `review/0505/local/scripts/run_v7_v8_sanity.sh` | 启动脚本 |
| `review/0505/local/scripts/compare_v6_v7_v8.py` | 后处理对比脚本 |
| `configs/pet_flow/pet_flow_first_hop_224_v6_transport_first.yaml` | V6 baseline 配置（对照参考）|
| `review/plan/ARCHITECTURE_ANALYSIS_20260501.md` 的 §17.3, §18.3.3, §21.1 | Grönwall 闭式解推导出处（如太长可只贴 §21.1）|
| `review/0505/operator/OPERATOR_REPLY_run_ablation_sanity_failure_20260505.md` | A/B sanity FAIL 的 operator 官方回复（理解 σ-normalize 路径污染）|
| `train_first_hop.py` 的 L1600-1640（image_aux 接入）+ L1955-2050（loss 组装与 image_aux gating）+ L2140-2170（watchdog）| V8 配置正确性的代码证据|

---

## --- PROMPT BEGIN ---

You are a top-conference (NeurIPS / ICML / MICCAI) reviewer who specializes in **statistical rigor of deep learning ablations**, **dimensional analysis of theory–experiment matches**, and **HARKing / garden-of-forking-paths detection**. Your reputation depends on catching pre-registration violations, methodological self-deception, and post-hoc rationalization. You are reviewing this submission with the assumption that the authors are **honest but possibly self-deceiving** and possibly under time pressure.

---

### Submission context

- **Domain**: PET image reconstruction via 4-hop latent transport `D50 → D20 → D10 → D4 → NORMAL` (5 timepoints, 4 hops indexed `j = 0..3`).
- **Architecture**: hop-aware DiT backbone + small per-hop residual velocity head; frozen DINOv2 encoder + frozen ViT-MAE decoder (with LoRA on attention); pixel-forcing branch with learnable `gate_pix` and `lambda_hop_0` for hop0 only.
- **V6 baseline** (already trained 200K, the current paper's main result) uses raw-domain `step_weights = [0.5, 2.0, 1.5, 1.0]` and includes an auxiliary "image_aux" loss (L1 + SSIM on `decode(z_pred_0)` vs `x_gt_0`, only for hop0, weighted by `lambda_img` ramped 0 → 0.04).
- **Theoretical claim being tested**: under the clinical-prior weighted objective $J = \sum_{k=1}^{4} \beta_k \|c_k\|^2$ with $\beta = [0.5, 0.45, 0.9, 1.5]$, multi-hop Grönwall error propagation with hop-wise Lipschitz $L_j \approx 1$ predicts the optimal raw-domain step_weights:
  $$w_j^* \propto \frac{\Phi_j}{(\sigma_j \cdot dt_j)^2}, \quad \Phi_j = \sum_{k=j+1}^{4} \beta_k$$
  Plugging in $\sigma_j \cdot dt_j = (0.0289, 0.01473, 0.01163, 0.01058)$ and normalizing to hop1 = 2.0 yields **V7 = `[0.6106, 2.0000, 2.7041, 2.0423]`**. V6 deweights hop2 by 45% and hop3 by 51% relative to this prediction.
- **The two ablations under review**:
  - **V7**: V6 with `step_weights` swapped to the closed-form prediction. Tests whether V6's hop2/hop3 deweighting is empirically optimal or theoretically sub-optimal. Both σ-normalize paths (ON / OFF) yield mathematically equivalent gradients in fp64; V7 takes the **σ-normalize OFF** path because the σ-normalize ON path has a separately-documented A/B sanity FAIL (~10% trajectory drift under fp32 + warn-only deterministic CUDA kernels — see operator reply attachment).
  - **V8**: V6 with `image_aux.enabled: false`. Tests whether image_aux is materially contributing to chain-quality metrics, or whether it was originally added only to suppress a decode-domain artifact that may no longer be needed.
- **Budget**: each arm 50K steps (sanity), then 200K (main) only if sanity supports promotion. V6 50K-step **slice** is taken from the existing V6 200K log — V6 is **not** retrained.

The authors claim the design is fair, the math is sound, and the decision matrix is pre-specified. Your job is to find every place that claim is false, weak, or self-deceiving.

---

### Your task

Read the attached `ANALYSIS.md`, the two yamls, the launch script, the comparator, the V6 baseline yaml, the Grönwall derivation excerpt from `ARCHITECTURE_ANALYSIS_20260501.md`, the operator reply about A/B sanity FAIL, and the relevant code sections of `train_first_hop.py`. Then answer **all eight questions** below in a single structured response (markdown, with headings). Be ruthless. If you'd reject this experiment design as "not informative regardless of outcome," say so and pinpoint which line fails.

---

### Q1 — Math audit: is the V7 closed-form derivation actually correct?

The author asserts $w_j^* \propto \Phi_j / (\sigma_j dt_j)^2$ from a multi-hop Grönwall analysis. Audit the derivation **dimensionally and structurally**:

1. **Lipschitz assumption ($L_j \approx 1$)**: §17.3 of `ARCHITECTURE_ANALYSIS_20260501.md` reports $L_j \in [0.96, 1.13]$ — empirically measured on V6. Is using these *V6-trained* Lipschitz constants to predict the *V7-optimal* step_weights circular? (V7 will train a different model with potentially different $L_j$.) If the closed form is sensitive to $L_j$, by how much could the prediction drift if $L_j$ moves to e.g. $[0.85, 1.30]$ under V7's altered weighting?
2. **Independence assumption**: the formula $w_j^* \propto \Phi_j / \rho_j$ implicitly assumes the per-hop noise contributions $(\sigma_j dt_j)^2$ are independent. The actual model has shared backbone weights + a single LoRA decoder; per-hop residuals through `z_pred_j → z_pred_{j+1}` are correlated. Does ignoring this correlation make the closed-form prediction systematically biased toward over-weighting the late hops (hop2/hop3) — i.e., could V7 be **theoretically wrong** in the direction the authors are testing?
3. **Anchor choice (hop1 = 2.0)**: the normalization is purely a global rescale of `step_weights`. But the authors *also* set `lambda_roll` ramp to `0 → 4.0` (V6) — the **product** `lambda_roll × Σ w_j` is what actually scales the rollout loss vs the pair loss. V7's $\Sigma w = 7.357$ vs V6's $\Sigma w = 5.0$ means V7 effectively pumps **47% more rollout gradient into the optimizer per step** (or 32% less, if lambda_roll is divided by Σ w internally — depends on implementation). The author labels this a "SANITY confound." Is this confound (a) properly accounted for in the decision matrix? (b) Could it dominate any signal the experiment is meant to extract? Quantify.
4. **β prior choice**: $\beta = [0.5, 0.45, 0.9, 1.5]$ is "clinical." Is the closed-form sensitive to β? If a reasonable alternative (e.g., uniform β, or β = standard deviation of clinical evaluation noise per timepoint) would give materially different $w_j^*$, then the V7 weights are partly an artifact of an arbitrary β choice and should be reported as such.

Verdict for Q1: **mathematically sound / mathematically suggestive but not rigorous / mathematically wrong** — pick one and justify.

### Q2 — V6 50K slice as the comparison baseline

V6 is **not retrained**. The authors compare V7 / V8 at step 50000 against V6's metrics.jsonl row at step 50000, taken from the V6 200K main run. This saves ~50% of GPU time but introduces several alignment hazards.

Identify all schedule-mismatch confounds, including but not limited to:

1. **Warmup ramp**: V6 has `warmup_ratio = 0.25`, so `lambda_roll` reaches `lambda_end = 4.0` at step 50000 (= 0.25 × 200K). V7/V8 have `max_steps = 50000` and the same `warmup_ratio = 0.25`, so they reach `lambda_end` at step **12500**. By step 50000, V6 is *just finishing* warmup while V7/V8 have been at `lambda_roll = 4.0` for **37500 steps** of additional Phase III training. **V7/V8 at step 50K are effectively a different training stage than V6 at step 50K.** Does this single confound invalidate the entire comparison? If yes, what is the minimum redesign that fixes it?
2. **LR cosine schedule**: same calculation — V6 at 50K is at ~25% through cosine decay; V7/V8 at 50K are at the *bottom* of cosine decay (lr ≈ `min_lr`). The LRs at "step 50K" are different.
3. **EMA decay 0.9999**: EMA at step 50K averages ~10K effective recent steps. V6 EMA at 50K reflects a *warming* training; V7/V8 EMA at 50K reflects a *converging-on-plateau* training. Comparing val_select_score on EMA weights mixes apples and oranges.
4. **Rolling val noise**: 64 batches × 8 = 512 slices ≠ full 7403-slice eval. The author already flags this; quantify: with rolling-val CV ≈ X% (from V6 200K run), what is the minimum detectable effect (MDE) at α = 0.05? Is the 5% chain_normal_mse promotion threshold (from §6 decision matrix) *resolvable* against this noise floor?
5. **List the minimum redesign**: should the authors instead train V7/V8 with `max_steps = 200000` (full) and only compare at the same global step? Or set V7/V8 `warmup_ratio = 0.0625` so `warmup_steps = 50000 × 0.0625 = 12.5K * (50/200) = 3125`...wait, the author wants V7/V8 at 50K to match V6 at 50K *as a substring of V6's full training*. What's the actual fix?

### Q3 — V8's claim coverage: can it answer the question it asks?

The author states V8 tests:
- (a) does image_aux help chain-MSE? — V8 directly tests this.
- (b) does image_aux suppress decode-domain artifacts? — V8 **does not** test this; needs separate qualitative review.
- (c) is the pair channel alone sufficient? — V8 tests this only if `gate_pix` / `lambda_hop_0` don't drift to the floor (becoming a "dead branch").

Audit:

1. **Original motivation for image_aux**: §21.5 of `ARCHITECTURE_ANALYSIS_20260501.md` (or wherever image_aux was first introduced) presumably gave a reason. Is V8 a fair test of that *original* motivation, or only a fair test of "does V6-with-image_aux outperform V6-without on chain_normal_mse"? If image_aux was added because of decode artifacts, V8 silently re-frames the question to chain MSE — this is a bait-and-switch.
2. **Dead-branch confound**: if V8's `gate_pix` collapses to floor (because pixel forcing no longer receives image-domain gradient), then V8 is *not* "V6 minus image_aux"; it's "V6 minus image_aux **and** minus pixel forcing." The two changes are entangled. Is this acceptable? If yes, the paper should report V8 as such. If no, V8 needs an additional run with `pixel_gate_init` raised to keep the branch alive — and that's a confound on top of a confound.
3. **What V8 cannot prove even on success**: if V8 chain_normal_mse beats V6 by 5%, can the author claim "image_aux was unnecessary"? Or only "image_aux was unnecessary *for chain MSE on the rolling-val window at this single seed*"? Audit the strength of the inference.
4. **Cohen's d**: prior V6 vs V6.1 full-val paired diff had Cohen's d ≈ 0.056 (negligible). If V8 vs V6 lands at the same Cohen's d magnitude, is the 5% chain_normal_mse threshold even meaningful? Or is the design statistically under-powered for the question it asks?

### Q4 — V7 SANITY CONFOUND severity

The yaml header acknowledges V7's $\Sigma w = 7.357$ vs V6's 5.0 means a "32% weaker effective lambda_roll" (or similar — the exact ratio depends on whether the trainer divides by $\Sigma w$). This is a single-line acknowledgment in a comment block.

1. **Is the confound recoverable from the experiment data?** Specifically: can the authors *post hoc* compute "what V7 would have looked like with $\Sigma w$ rescaled to 5.0" from the metrics.jsonl alone? Or does it require V7b (rescaled run)?
2. **V7b's status**: the yaml mentions V7b ([0.4148, 1.3592, 1.8378, 1.3878]) as "optional." But Q1 says V7 vs V6 is *uninterpretable* without V7b. So is V7b actually optional, or is it required for V7 to be informative? If required, the experiment plan should commit to running it; "optional" is a researcher-degrees-of-freedom flag.
3. **The paper's eventual narrative**: regardless of V7 outcome, what is the most charitable narrative the authors *could* run with each of {V7 wins, V7 loses, V7 ties V6, V7b wins, V7b loses}? Is the experiment designed such that **all** outcomes have a publishable story? If so, that's a HARK surface — the ablation cannot falsify any claim.

### Q5 — Decision matrix audit

`ANALYSIS.md §6` gives a 3×2 decision matrix:

V7: { chain_normal_mse < V6 − 5% AND val_multi_objective ≤ V6 → ✅ promote 200K | ≈ → run V7b | > V6 + 1% AND no late-hop gain → ❌ } 
V8: { all chain_*_mse ≤ V6 + 1% AND no decode artifacts → ✅ | ≤ V6 + 1% but artifacts return → 🟡 reduce lambda_max | any chain_*_mse > V6 + 1% → ❌ }

1. **Asymmetry**: V7's promotion threshold is "−5%", V7's rejection threshold is "+1%". Why asymmetric? Is the asymmetry pre-specified for a principled reason (e.g., expected effect direction, prior risk asymmetry)? Or is it a sign that the author has a preferred outcome and is biasing the bar accordingly?
2. **5% vs Cohen's d ≈ 0.056**: as Q3 asked, is 5% even resolvable? At 200K full-val, V6 vs V6.1 had paired Cohen's d ≈ 0.056. Is 5% relative chain_normal_mse change consistent with this Cohen's d, and if so what's the implied SE of the rel_diff estimator on 512-slice rolling val?
3. **`val_multi_objective ≤ V6` AND clause**: this is a multi-objective gate. The weights inside `val_multi_objective` (from `best_metric_terms`) heavily weight chain_normal_mse (1.5) and chain_d4 (0.9) vs chain_d20 (0.5). If V7 trades chain_d20 for chain_normal_mse, val_multi_objective could go either way. Is the author trading away early-hop quality (clinically important for D20 / D10 reconstructions) and hiding it inside an aggregate metric?
4. **"5% rel chain_normal_mse" vs "5% rel val_multi_objective"** — these are different metrics. Which exactly determines promotion? The matrix is ambiguous.
5. **V7b promotion path**: if V7 ties V6 and V7b is run, what is V7b's promotion criterion? It is **not** specified. This is a HARK surface.

### Q6 — Code-level audit

Without running the scripts, audit:

1. **`run_v7_v8_sanity.sh` `resolve_yaml`**: the awk script rewrites `max_steps` and `output_dir`. Does it correctly handle (a) yaml lines with leading whitespace (training.max_steps is nested), (b) the case where `output_dir:` appears in a comment — false positive rewrite? (c) Does it preserve the file's idempotency (running twice with the same args yields identical bytes)?
2. **`compare_v6_v7_v8.py` `closest_row`**: when the V6 metrics.jsonl has rows at steps `[400, 800, ..., 49600, 50000, 50400, ...]`, `closest_row` will pick step 50000 exactly. Fine. But when V7's metrics.jsonl ends abruptly at step 49200 because eval_interval=400 doesn't land on 50000 exactly (50000 / 400 = 125 → does land), the closest row may be at 49600 or 49800 depending on what the trainer wrote. Does this introduce a few-hundred-step bias that, combined with Q2's schedule mismatch, makes the comparison unreliable?
3. **Decision rules in `compare_v6_v7_v8.py`**: the printed rules (§"Decision rules") include text not present in `ANALYSIS.md` decision matrix. Are they consistent? If not, which is the source of truth? (Inconsistency = HARK surface.)
4. **`metrics.jsonl` whitelist**: from prior peer review of 0503 protocol, `metrics.jsonl` was identified as a real-time leak channel. V7/V8 tail-able training logs let the operator peek at val_chain_normal_mse during V7 / V8 training and (subconsciously or otherwise) tune V7b's parameters from V7's partial trajectory. Is there any leak prevention here? If not, does it matter for *this* experiment (50K sanity is not a pre-registered hypothesis test, just an exploratory ablation)?
5. **`require_fresh_output_dir: true`**: V7/V8 yamls have this. The launch script also has a `SANITY_FRESH=1` escape hatch that deletes the previous output. Is this safe (no risk of deleting V6 / V7 / V8 cross-contamination)?

### Q7 — Adversarial scenarios

Construct **three concrete attack scenarios** by which a (subconsciously) biased author could still bias the V7/V8 result through this protocol. For each:

1. The exact mechanism (be specific — name the file, the step, the moment).
2. Whether the protocol or the launch scripts currently catch it.
3. The minimal patch that would close the hole.

Scenarios to consider (you may construct your own):
- (a) Author tails V7's `metrics.jsonl` daily during 50K training, sees chain_normal_mse landing at +3% over V6, decides to "reset and try with smaller pair_loss_weight" → V7 is rerun with a tweaked hyperparam not reflected in the yaml comment changelog.
- (b) Author runs V7 first, sees marginal result, then runs V8 with knowledge of V7's outcome — the V8 decision matrix's "1% / 5%" thresholds could be tweaked subconsciously.
- (c) V6 metrics.jsonl path is unverified — the author later "discovers" the authoritative V6 metrics.jsonl was actually a different run (e.g., V6_200K_replicate3 with a different seed) and silently switches comparison baselines.

### Q8 — Final verdict

Rate the V7/V8 50K sanity ablation design on a 1–10 scale across **five axes**, with 1–2 sentence justifications each:

- **Theoretical grounding** (does Grönwall closed-form actually predict V7 weights, or is the math suggestive at best?)
- **Experimental fairness** (is V6 50K slice a fair comparison baseline given schedule mismatch?)
- **Question coverage** (do V7 and V8 actually answer the questions the authors claim they answer?)
- **Pre-registration / HARK resistance** (could the author retrofit a positive narrative regardless of V7/V8 outcomes?)
- **Implementation hygiene** (do the scripts, yamls, and decision matrix enforce what `ANALYSIS.md` claims?)

End with a **one-paragraph TOP FIX recommendation**: if the authors can only address one issue before launching the GPU job, what should it be?

End with a **GO / NO-GO / GO-AFTER-FIX** verdict.

---

## --- PROMPT END ---

---

## How to use the verdict

After receiving 1–2 external reviews, save them as:

- `review/0505/operator/PEER_REVIEW_GPT5_V7V8.md`
- `review/0505/operator/PEER_REVIEW_GEMINI_V7V8.md`
- (optional) `review/0505/operator/PEER_REVIEW_OPUS_V7V8.md`

Then I (local Claude) will:

1. Cross-reference the reviews — which concerns appear in ≥ 2 reviews ("hard concerns") vs which are single-reviewer ("soft concerns").
2. Triage hard concerns into:
   - **Must-fix-before-launch**: blocking issues that, if left, would make V7/V8 results uninterpretable regardless of outcome.
   - **Document-and-launch**: known caveats already flagged in `ANALYSIS.md §7`; reviewer raised same point.
   - **Defer to full-val**: issues only resolvable after best/last full-val (not 50K sanity).
3. Write `review/0505/local/PEER_REVIEW_TRIAGE.md` with the synthesis + a patched experiment plan if MUST-FIX issues exist.

**Do not launch the GPU job until at least one external review has been processed.** Specifically: Q2 (schedule mismatch) and Q5 (decision matrix asymmetry) are the two most likely "must-fix-before-launch" categories.

---

## Notes on prompt design

- The prompt is **adversarial by construction** (we want the reviewer to find issues, not validate). If the external AI returns "looks great, ship it," treat that as a sign the prompt was too gentle and rerun with stronger framing.
- **Q1 (math)** and **Q2 (schedule mismatch)** are the two most likely places for a serious flaw. Q3–Q5 are HARK / decision-matrix surfaces. Q6–Q7 are hygiene. Q8 is the score.
- Prior 0503 peer review (`review/0503/operator/PEER_REVIEW_CLAUDE.md`) for the σ-norm pre-registration found 4/10 pre-registration rigor and 3/10 statistical correctness. For V7/V8 we expect a *better* score on pre-registration rigor (this is exploratory sanity, not a locked hypothesis test) but possibly a *worse* score on theoretical grounding (Grönwall + Lipschitz assumption is fragile).
- Token budget: this prompt + attachments will likely run 30K–60K tokens. GPT-5 high reasoning and Gemini-2.5-Pro both handle this comfortably; Claude-Opus may need attachments split into 2 messages.
