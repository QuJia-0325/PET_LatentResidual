# 0503 Local → External AI Peer Review Prompt

**Purpose**: 把这段 prompt 喂给 GPT-5 / Gemini-2.5-Pro / Claude-Opus 等外部 AI（非本 agent），做对 §6.4–§6.6 的 adversarial peer review。
**Suggested target reviewers**: GPT-5 (high reasoning) + Gemini-2.5-Pro，至少跑两家求交。
**How to use**: 复制下方 `--- PROMPT BEGIN ---` 与 `--- PROMPT END ---` 之间的全部内容，连同附件文件一并粘贴。**不要改 prompt 措辞**；语气有意 adversarial。

附件（让外部 AI 一并读取）：

1. `review/0502/POST_V6_NEXT_STEPS.md` 全文（重点 §6.4 / §6.6 / §8.6 / §8.7）
2. `review/0502/scripts/paired_diff_judge.py`
3. `review/0502/scripts/lock_effect_size_threshold.py`

---

## --- PROMPT BEGIN ---

You are a top-conference (NeurIPS / ICML / MICCAI) reviewer who specializes in **statistical rigor of deep learning ablations**. Your reputation depends on catching pre-registration violations, p-hacking, garden-of-forking-paths, and HARKing. You are reviewing this submission with the assumption that the authors are **honest but possibly self-deceiving**.

**Submission context**:

- Domain: PET image reconstruction via 4-hop latent transport (D50 → D20 → D10 → D4 → NORMAL).
- Architecture: hop-aware DiT backbone + small hop-residual velocity head; frozen RAE decoder.
- The paper's central empirical claim hinges on a **σ-normalize ablation**: comparing arm `A_main` (V6 transport-first, step-weight middle-heavy) against arm `C_uniform` (B_sanity baseline with σ-normalizer ON, step-weight uniform). The headline number is `rel_diff = (chain_normal_mse_C - chain_normal_mse_A) / chain_normal_mse_A`.
- Authors have committed to a **blinded effect-size pre-registration**: a threshold `X` is locked from `A_main` data alone (before `C_uniform` is fully evaluated), and the decision rule is locked in advance.

**Your task**: Read the attached `POST_V6_NEXT_STEPS.md` (§6.4 paired_diff_judge protocol + §6.6 blinded effect-size pre-registration) plus the two implementation scripts. Then answer the **eight questions below** in a single structured response. Be ruthless. If you'd reject this paper at desk-review, say so and pinpoint exactly which line fails.

---

### Q1 — §6.6.1 / §6.6.3 LOCKED formula audit

The formula is `X = max(0.10, 3 × paired_CV_A)`, where `paired_CV_A` is the rolling-window CV of `val_select_score` on A_main in `step ∈ [40000, 60000]`.

- Is `3σ` (slope=3) statistically defensible as an "indistinguishable" threshold under autocorrelated noise? If you'd argue for a different multiplier, what (e.g., 2σ, Cohen's d, MDE), and why?
- Is the `0.10` floor coherent with the slope? (e.g., when `paired_CV_A < 0.0333`, the floor dominates; can authors justify why 10% rel_diff is "small enough to call indistinguishable" in this domain?)
- Does the **single-arm** CV correctly capture the **between-arm** noise? (i.e., should the proper proxy be `paired_CV_pair = sd((C-A)) / mean(A)`, not `paired_CV_A`?) If yes, this is a fundamental flaw — flag it as a desk-reject blocker.

### Q2 — §6.6.2 blinded execution sequence

The protocol pre-registers steps Step 0 → Step 7. The hard claim: "Step 4 commit hash proves X was locked before C was read."

- Identify any path by which authors could **leak C information into X** without violating the literal text of §6.6.2. (Examples to probe: A_main and C_uniform share a logger? A_main hyperparams were tuned post-hoc with knowledge of preliminary C numbers? `paired_CV_A` is computed on a window the authors picked after seeing C trajectory?)
- Is the `lock_effect_size_threshold.py` Guard 3 (rglob for `*full_val*`, `*eval*.json`, `*method_d*`, `*selected_ckpt*`) sufficient to detect "C has been peeked"? List specific filenames it would miss.
- Does the protocol cover the case where C_uniform's **partial trajectory** (during training, before full-val) is observable via WandB / metrics.jsonl and could subconsciously bias the `paired_CV_A` window choice?

### Q3 — §6.6.3 decision-rule trichotomy

The rule:

| `rel_diff` | Conclusion |
|---|---|
| `< X` | C ≈ A (σ-norm is the lever) |
| `[X, 2X]` | grey zone → 200K continuation |
| `> 2X` | C ≠ A, V6 narrative validated |

- Does the trichotomy cover **`rel_diff < 0` (C is BETTER than A)**? What is the locked interpretation of negative rel_diff? If not specified, this is a HARKing surface.
- The "200K continuation" branch: is this itself a pre-specified analysis or a researcher-degrees-of-freedom escape valve? What stops authors from re-running 200K → re-locking `X` with new `paired_CV_A` → effectively peeking?
- The 2X boundary: is it derived (from any statistical principle) or chosen for clean prose? Audit honestly.

### Q4 — §6.4 paired_diff_judge.py + LOCKED 0.10 threshold

Risk 4 = "is the σ-norm contrast confounded by pair-sampling × step-weight interaction?" The script compares A_main vs `A_pair_uniform_spot`, autocorr-corrects N_eff, decides `no_confound / borderline / confound` at threshold 0.10.

- Is the **autocorr correction** `N_eff = N(1-ρ₁)/(1+ρ₁)` (lag-1 only) appropriate for `val_select_score` time series? When could it overestimate N_eff and falsely confirm "no confound"?
- The threshold 0.10 is "defensive" (false-negative is desk-reject; false-positive is just extra eval). Is this risk asymmetry argued correctly, or is it post-hoc rationalization for a number the authors prefer?
- If `A_pair_uniform_spot` is **not actually run** (a real possibility — see operator question Q1), what is the most honest way to handle Risk 4 in the paper? Specifically: is "we did not run A_pair_uniform_spot because [...]" defensible, or does this require either supplemental experiments or scope reduction?

### Q5 — §6.6.5 auxiliary anchor (decoder ceiling)

The protocol mentions `eval_gt_latent_decoder_ceiling_clip3.py` as an auxiliary anchor.

- If the ceiling number has **not been computed** at submission time, can the paper still credibly claim "near-ceiling performance" or "X% of remaining headroom"? What is the minimum reviewer-defensible alternative?
- If computed, what is the right way to integrate it into the decision rule WITHOUT changing X? (e.g., as a separate sanity check, or as a denominator for normalized rel_diff?)

### Q6 — Implementation correctness of `lock_effect_size_threshold.py`

Without running it, audit the script for:

- Off-by-one in window selection (`step >= step_min` vs `step > step_min`).
- Float comparison hazards in the LOCKED constants.
- Race condition between "compute paired_CV_A" and "git commit": if `metrics.jsonl` is being appended to while the script reads it, can the SHA256 hash fail to match the data the script computed on?
- Whether `--c-uniform-output-dir` rglob actually catches all common artifact name patterns (the script lists `*full_val*`, `*eval*.json`, `*method_d*`, `*selected_ckpt*`).
- Tamper-resistance: can a reviewer trust that the lock-in commit was not retroactively rebased? (Hint: gitee force-push policy.)

### Q7 — Adversarial scenarios / threat model

Construct **three concrete attack scenarios** by which a (subconsciously) biased author could still bias the result through this protocol. For each:

1. The exact mechanism.
2. Whether §6.6 currently catches it.
3. The minimal protocol patch that would close the hole.

### Q8 — Final verdict

Rate the protocol on a 1–10 scale across **four axes**, with 1-sentence justifications each:

- **Pre-registration rigor** (does it actually prevent p-hacking?)
- **Statistical correctness** (is the math right?)
- **Implementation robustness** (do the scripts enforce what the docs claim?)
- **Reviewer defensibility** (would you, as reviewer, accept this as a credible blinded analysis?)

End with **one of**: `ACCEPT (camera-ready)`, `MINOR REVISIONS`, `MAJOR REVISIONS`, `DESK REJECT`. State the single most important fix.

---

**Format your response as**:

```
Q1: <answer>
Q2: <answer>
Q3: <answer>
Q4: <answer>
Q5: <answer>
Q6: <answer>
Q7:
  Attack 1: <mechanism> | caught? <yes/no/partial> | patch: <patch>
  Attack 2: ...
  Attack 3: ...
Q8:
  Pre-registration rigor: <score>/10 — <one-line>
  Statistical correctness: <score>/10 — <one-line>
  Implementation robustness: <score>/10 — <one-line>
  Reviewer defensibility: <score>/10 — <one-line>
  VERDICT: <ACCEPT|MINOR|MAJOR|DESK REJECT>
  TOP FIX: <one sentence>
```

Do not soften your tone for politeness. The authors **want** to be told what's wrong before submission, not after.

## --- PROMPT END ---

---

## 使用说明

### 推荐工作流（30 分钟）

1. **GPT-5 (high reasoning)** via Codex MCP 或网页：粘贴 prompt + 三个附件文件原文。
2. **Gemini-2.5-Pro** via gemini-review MCP：同样粘贴。
3. 把两份输出存到 `review/0503/operator/PEER_REVIEW_GPT5.md` 和 `PEER_REVIEW_GEMINI.md`。
4. 我（本 agent）下次会话时读这两份输出 → 比较两家共识 vs 分歧 → 把"两家都标 MAJOR / DESK REJECT"的项作为必修补丁，"两家分歧"的项作为讨论项。

### 如果时间紧

只跑 GPT-5 一家。重点看 **Q3 (decision-rule trichotomy 是否有 negative rel_diff 漏洞)** 和 **Q7 (adversarial scenarios)**——这两题最容易暴露 desk-reject 风险。

### 如果两家都给 ACCEPT

不太可能。如果真出现，认真复读 Q7 的 attack 场景——"被夸了"通常意味着 prompt 不够 adversarial 或附件不够全。把 `paired_diff_judge.py` 的源码也一并塞进去重跑。

### 不要做的事

- ❌ 不要把外部 AI 的 verdict 直接当结论改 protocol。**先在本地 agent 这边过一遍，挑出真正的 actionable 修补点再动 §6.4–§6.6**。protocol 改一次就要 git commit 一次，每次 commit 都是 pre-registration credibility 的损失（reviewer 会问"为啥不一次锁死"）。
- ❌ 不要在 A_main 跑过 60000 之后再做这件事。X 一旦锁定，protocol 就冻结；外部审计应该**在 X 锁定之前**完成。
