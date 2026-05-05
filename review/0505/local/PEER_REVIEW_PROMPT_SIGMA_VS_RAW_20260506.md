# Round 4 Peer-Review Prompt — σ-normalize Failure vs raw-rollout Path

**Purpose**: 把这段 prompt 喂给 GPT-5.5（high reasoning）+ Claude Opus 4.7（high reasoning）+（可选）Gemini-2.5-Pro。**独立**审计；之后我做交叉核对。

**与前几轮 review 的区别**:

| 轮次 | 审查对象 | 结论 |
|---|---|---|
| Round 1 | V7/V8 实验设计本身 | 4 reviewers 通过 |
| Round 2 | Plan D（V7 resume V6@40K + V8 from-scratch 100K）| EMA-swap 阻断，**Plan D 死亡** |
| Round 3 | Plan F（V7+V8 from-scratch 150K）| GO-WITH-FIXES → 锁定 V3 形态（160K，3-arm） |
| **Round 4（本轮）**| **σ-normalize 失败本身的根本原因 + raw-rollout 路径是否能避开它 + 失败应急预案** | 待审 |

**审查紧迫程度**: 中-高。Plan F 的 V7 已经决定走 raw-rollout（σ-normalize OFF）路径。如果 reviewer 判定 raw-rollout 也会复现同种 trajectory amplification 失败，则 V7 训练完成后 **paired-comparison 的可信度本身存疑**——现在拉响警报代价低，跑完 160K 再发现就晚了。

**用法**: 复制 `--- PROMPT BEGIN ---` 与 `--- PROMPT END ---` 之间的全部内容，连同附件一起粘贴。**不要改 prompt 措辞**；adversarial 语气是有意的。

---

## 必须随 prompt 一并提供给外部 AI 的附件

| 路径 | 内容 | 为什么需要 |
|---|---|---|
| `review/0505/operator/OPERATOR_REPLY_run_ablation_sanity_failure_20260505.md` | **σ-normalize 官方 sanity 失败完整报告**（核心证据） | 必读 |
| `review/0505/operator/logs_sanity/run_ablation_sanity_gpu1_20260503.launch.log` | A/B sanity 比较器原始日志 | 原始数据 |
| `review/0505/operator/logs_sanity/run_ablation_A_sanity_train.log` | A_sanity 训练日志（含 PyTorch 非确定 warning） | 原始数据 |
| `review/0505/operator/logs_sanity/run_ablation_B_sanity_train.log` | B_sanity 训练日志 | 原始数据 |
| `review/0505/operator/logs_sanity/run_ablation_A_sanity_config.resolved.yaml` | A_sanity 决议后 config | 验证种子+确定性配置 |
| `review/0505/operator/logs_sanity/run_ablation_B_sanity_config.resolved.yaml` | B_sanity 决议后 config | 同上 |
| `pet_lr/rollout_first_hop.py` 行段 L40-145 | `rollout_multistep_losses_first_hop` 完整实现（含 step_normalizers / preserve_v6_sum 用法 + return块） | 让 reviewer 验算 σ-normalize 算法等价性 |
| `pet_lr/model_first_hop.py` 行段 L260-340 + L430-440 | **`pair_v_std` 以 `register_buffer(..., persistent=False)` 注册（L336）**，该缓冲区在训练中不变 | 验证σ-normalize的归一化因子是常量 |
| `train_first_hop.py` 行段 L292-330 | **`compute_pair_losses` 完整实现**——pair loss 不调用 rollout_multistep 也不接收 step_normalizers | 验证 pair_loss 结构上独立于σ-normalize |
| `train_first_hop.py` 行段 L395-528 | step_normalizer 计算 + 注入 rollout 路径 | 验证算法注入位置 |
| `review/plan/PLAN_F_V7_V8_FROM_SCRATCH_150K_20260505.md` | Plan F 决策文档（V7 走 raw-rollout 的依据） | 上下文 |
| `review/plan/PLAN_F_PARALLEL_LAUNCH_DESIGN_20260506.md` | 双卡并行启动设计（本轮触发文件） | 上下文 |
| `review/0505/local/configs/V7_gronwall_raw.yaml` 行 240-260 | V7 实际 step_weights + σ-normalize OFF 注释 | V7 当前形态 |

---

## --- PROMPT BEGIN ---

You are a senior reviewer (NeurIPS / ICML / MLSys / SysML level) specializing in **bit-deterministic ML training reproducibility**, **CUDA / cuDNN / cuBLAS kernel non-determinism**, **paired-ablation training-trajectory equivalence**, and **multi-step open-loop rollout error amplification**. Your reputation depends on catching subtle reproducibility traps that turn a 480K-GPU-step paired-ablation experiment into apples-vs-oranges.

You are reviewing **the methodological foundation of Plan F's V7 arm** — specifically, whether the project's decision to switch from a **σ-normalize rollout path** (which formally failed sanity on 2026-05-03) to a **raw-rollout path** (V7 current configuration) is scientifically sound, or whether it merely re-buries the same root cause.

The experiment design (V7 = Grönwall closed-form `step_weights`; 消融 arm = remove `image_aux`; V6_NOISE = seed twin) was peer-reviewed in Rounds 1–3 and is fixed. **Do not re-litigate the design.** Your job is exclusively to assess the σ-normalize failure mechanism and the raw-rollout fallback path.

---

### Background you can take as given (do not audit)

- 4-hop latent-domain cascade `D50 → D20 → D10 → D4 → NORMAL`. Backbone frozen DINOv2 + ViT-MAE decoder + LoRA + per-hop residual head + pixel-forcing branch (only at hop0).
- Rollout total loss = `(Σ_j w_j · ℓ_j) / (Σ_j w_j)` (weighted **mean**, not sum). See `pet_lr/rollout_first_hop.py:117-119`.
- σ-normalize variant: `ℓ_j_normalized = ℓ_j_raw / n_j`, where `n_j` is a per-hop magnitude normalizer (`(σ_j · dt_j)²` family). When `relative_to=preserve_v6_sum`, the normalizer is rescaled so that `Σ w_j · n_j` matches V6's raw `Σ w_j`. The intent is: **at the same model + same batch**, the σ-normalize total loss should be algebraically equal to V6's raw total loss.
- **VERIFIED FACT (do not re-audit)**: `pair_v_std = [0.009634, 0.002946, 0.000775, 0.000141]` is registered with `register_buffer(..., persistent=False)` at `pet_lr/model_first_hop.py:336`. Therefore `n_j` is a **constant across training** (does not depend on optimizer state). In **exact arithmetic**, A's `(w_v6·ℓ).sum()/Σw_v6` = B's `((w_v6·n)·(ℓ/n)).sum()/Σ(w_v6·n)`. **σ-normalize algebraic forward equivalence holds on paper.** Therefore Q1 must distinguish: (α) algebraic equivalence holds in exact arithmetic but FP32 implementation breaks it via reduction-order asymmetry, vs (β) algebraic equivalence is broken even on paper (e.g. wrong yaml weights, wrong `preserve_v6_sum` formula), vs (γ) the failure is purely independent-trajectory drift. These are different invariants with different remediations.
- **`compute_pair_losses` is structurally independent of σ-normalize**: it is defined at `train_first_hop.py:292-330` and does NOT call `rollout_multistep_losses_first_hop`, does NOT pass through `mix_latent`, does NOT receive `step_normalizers`. Pair-loss is a per-batch single-step velocity loss on `hop_idx` randomly drawn from {0,1,2,3}. **A and B should produce identical pair-loss values on identical (model, batch).**
- Implementation: `pet_lr/rollout_first_hop.py:99-104` divides `step_loss` by `step_normalizers[hop_idx]` when `step_normalizers is not None` (B path only). Note: this is `step_loss = step_loss / float(step_normalizers[hop_idx])` — a **Python-scalar division** that introduces a host-side sync only on the B path. `train_first_hop.py:395-528` computes the normalizers and injects them when `rollout.sigma_normalize.enabled=true`.
- Formal A/B sanity comparator (`review/0502/scripts/run_ablation.sh sanity`): trains A_sanity (raw weights, raw rollout) and B_sanity (V6 weights × normalizers, normalized rollout) **independently** for 20K steps each, compares their **last eval-row** validation metrics across 4 tiers. The comparator's threshold designer's note in the launch log expects "FP32 will introduce ~1e-6 noise in tiers 1-3, ~1e-3 in tier 4" — this calibration assumed **bit-deterministic** FP32, NOT warn-only mode. The thresholds may therefore be **structurally over-tight** for the actual training regime.
- 2026-05-03 official run: A/B used `seed=42`, `deterministic=true (warn-only)`, `num_workers=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

---

### The 2026-05-03 σ-normalize failure (you must take these numbers as ground truth)

| Metric | A | B | Relative error | Tier threshold | Result |
|---|---:|---:|---:|---|---|
| `val_pair_total` | 1.537e-7 | 1.694e-7 | **10.16%** | T1 < 1e-4 | **FAIL** |
| `val_rollout_total` | 2.052e-3 | 2.089e-3 | 1.85% | T2 < 1% | **FAIL** |
| `val_rollout_step_0_raw` | 1.764e-3 | 1.769e-3 | 0.26% | T3 < 1% | PASS |
| `val_rollout_step_1_raw` | 1.898e-3 | 1.903e-3 | 0.25% | T3 < 1% | PASS |
| `val_rollout_step_2_raw` | 2.123e-3 | 2.163e-3 | 1.88% | T3 < 1% | **FAIL** |
| `val_rollout_step_3_raw` | 2.396e-3 | 2.514e-3 | 4.92% | T3 < 1% | **FAIL** |
| `val_chain_d20_mse` | 7.357e-4 | 7.351e-4 | 0.07% | T4 < 5% | PASS |
| `val_chain_d10_mse` | 6.599e-4 | 6.552e-4 | 0.71% | T4 < 5% | PASS |
| `val_chain_d4_mse` | 5.904e-4 | 5.791e-4 | 1.90% | T4 < 5% | PASS |
| `val_chain_normal_mse` | 5.872e-4 | 6.620e-4 | **12.73%** | T4 < 5% | **FAIL** |

PyTorch warned during training (both A and B):

```text
Memory Efficient attention defaults to a non-deterministic algorithm
adaptive_avg_pool2d_backward_cuda does not have a deterministic implementation,
  but torch.use_deterministic_algorithms(True, warn_only=True)
```

Sanity sentinel was cleared. Plan F V7 then decided to switch entirely to **raw-rollout** (σ-normalize OFF) — V7 yaml has no `sigma_normalize` block at all, so `step_normalizers` defaults to None and the `step_loss = step_loss / step_normalizers[hop_idx]` branch is never taken.

---

### Failure pattern observation (multiple competing hypotheses; you decide)

Looking at the per-hop column: error **generally increases** with hop index, but hop0 and hop1 are essentially tied at sub-0.3%, then jump at hop2/hop3.

| hop | A vs B raw rel err |
|---|---|
| 0 | 0.26% |
| 1 | 0.25% |
| 2 | 1.88% |
| 3 | **4.92%** |

And `val_chain_normal_mse` (NORMAL endpoint of the open-loop chain through all 4 hops) is the worst single failure at **12.73%**.

This pattern is consistent with **(α) open-loop trajectory amplification of CUDA kernel noise via `mix_latent`** (`pet_lr/rollout_first_hop.py:107-113`). It is **also consistent** with: (β) FP32 reduction-order asymmetry in B's autograd graph (B has `step_loss / float(n_j)` per hop, A doesn't — see `pet_lr/rollout_first_hop.py:102-103`); (γ) optimizer-state divergence after 20K steps under warn-only nondeterministic kernels; (δ) any combination of the above; (ε) something else. **You must decide which dominates in Q1; do not assume.**

### Critical observation: `val_pair_total = 10.16%` falsifies any "rollout-amplification-only" diagnosis

The pair loss is computed per `train_first_hop.py:292-330`. It does NOT invoke `rollout_multistep_losses_first_hop`, does NOT pass through `mix_latent`, does NOT receive `step_normalizers` in any code path. It is a per-batch single-step velocity loss on `hop_idx ∈ {0,1,2,3}` randomly sampled.

Yet `val_pair_total` drifts **10.16%** between A and B after 20K training steps with the same seed (largest single failure other than chain_normal). The threshold-designer expected ~1e-4 (1e-6 of value `~1e-7`), but observed `1.6e-8` absolute drift — this is **≥5 orders of magnitude** above the expectation.

**This single number falsifies the hypothesis "trajectory amplification through `mix_latent` via 4-hop cascade is the sole cause"**, because pair_loss is **structurally outside** the rollout chain. Whatever mechanism produced 10.16% pair_total drift must operate on the *trained model weights themselves* (i.e., 20K steps of SGD under non-bit-deterministic forward+backward yielded measurably different model parameters in A vs B).

**When evaluating Q1(b), you MUST account for the val_pair_total drift as an INDEPENDENT failure that any proposed root-cause hypothesis must reconcile.** A hypothesis is incomplete if it predicts pair_total drift = 0 but the actual drift is 10.16%.

---

### Your three audit questions

**You must answer all three. Each answer must include: (a) one-line verdict, (b) at-least-three-sentence reasoning grounded in the attached code/logs, (c) explicit list of evidence points.**

---

#### Q1 — Why did σ-normalize fail formal sanity?

The author claims (in `OPERATOR_REPLY_run_ablation_sanity_failure_20260505.md` §3.3) that the failure is **CUDA kernel non-determinism amplified through open-loop rollout**, NOT an algebraic equivalence bug in the σ-normalize formula. Specifically:

> 1. PyTorch reports nondeterministic Memory-Efficient-attention and adaptive-pool backward kernels under `warn_only=True`.
> 2. Tiny early numerical differences amplify through 4-hop rollout via `mix_latent`.

**Audit Q1**:

(a) Decompose the author's diagnosis into TWO sub-claims, rate each independently:
  (i) PyTorch warns about nondeterministic Memory-Efficient-attention and `adaptive_avg_pool2d_backward_cuda` kernels under `warn_only=True` — **observable fact**, verifiable in train logs. Rate as VERIFIED / FALSIFIED / INCONCLUSIVE.
  (ii) These kernel warnings, amplified through `mix_latent`, are the **dominant** causal mechanism for the 4-tier failure — **causal claim**. Rate as CORRECT / PARTIAL / WRONG given the val_pair_total constraint above (which is structurally outside `mix_latent`).

(b) Enumerate **ALL** plausible failure mechanisms (not just three) and rank by likelihood given attached evidence + the val_pair_total constraint. At minimum address:
  - (α) Kernel non-determinism + open-loop rollout amplification via `mix_latent`
  - (β) FP32 reduction-order asymmetry from B's `step_loss / float(n_j)` host-scalar sync (only on B path, `rollout_first_hop.py:103`)
  - (γ) Optimizer-state divergence (raw-weight Adam moments accumulated under warn-only forward+backward) producing different weights at step 20K — this alone explains pair_total since pair_loss only depends on weights+batch
  - (δ) DataLoader sample-order divergence between A and B `set_seed()` paths (verifiable by hashing first N batches)
  - (ε) Single-point eval-row noise (V6 Phase III val_select_score CV ≈ 30% empirically) — the comparator uses `last_eval()` which has no aggregation
  - (ζ) Algebraic mis-implementation despite paper-correct algebra (e.g., yaml hard-coded weights `[1.7908, 1.8606, 0.8691, 0.4795]` differ from what `preserve_v6_sum` actually produces at runtime due to dtype/rounding)
  - Any (η), (θ) you find

  Rank by **likelihood** AND **testability** ("can be falsified from existing artifacts alone" vs "requires new test").

(c) Specifically: is there any way to **prove from the existing data** that the σ-normalize **algebraic** path (`step_loss / step_normalizers[hop_idx]`) is correctly implemented? If not, what minimal additional test would prove it?

(d) Would running a **same-process** A/B golden test (load one fixed model + one fixed batch, compute A loss and B loss in the same Python process, compare) actually catch the bug if it's algebraic? Why or why not? Note: `pair_v_std` being a non-persistent buffer means n_j is constant, so `model.eval()` + fixed batch + math-SDPA backend should produce **bit-identical** A and B losses if algebra is correct. State the exact test in the same format as `OPERATOR_REPLY_run_ablation_sanity_failure_20260505.md` §5.2.

---

#### Q2 — Does the raw-rollout path (V7 current) avoid this problem?

V7 abandons σ-normalize entirely. The rollout total loss reverts to:

```text
total = Σ w_j · ℓ_j_raw / Σ w_j
```

where `w_j = [0.6106, 2.0, 2.7041, 2.0423]` (Grönwall closed-form). No `step_normalizers` injection. No A/B comparison gate (V7 stands alone vs V6@step_160000 anchor).

**Audit Q2**:

(a) Does the raw-rollout path **structurally avoid** the failure mode that crashed σ-normalize sanity? Or does it just **hide** the same problem because there's no longer an A/B gate to detect drift? Distinguish: (i) eliminates the σ-normalize-specific FP32 reduction-order asymmetry (`step_loss / float(n_j)` host-sync only on B path); (ii) eliminates the underlying CUDA kernel non-determinism (Memory-Efficient-attention + `adaptive_avg_pool2d_backward_cuda`) — these have **different answers**.

(b) The Plan F decision rule for V7 is `Cohen's d ≥ max(0.10, 1.8 × d_pure)` on paired full-val NORMAL chain MSE between V7@step_160000 and V6@step_160000. **There are TWO different drift bounds at issue, do NOT conflate**:
  - **(b1)** *V7-vs-V6-replica drift bound under same yaml, same seed, different runs of identical code* (this is what V6_NOISE measures) — lower bound on `d_pure` floor.
  - **(b2)** *V7-vs-V6 drift bound under same seed, but different `step_weights` (V7 = [0.6106,2,2.7041,2.0423] vs V6 = [1.7908,1.8606,0.8691,0.4795]) and different rollout `relative_to`* — the actual experimental signal.

  The A/B σ-normalize sanity at 12.73% is **NEITHER (b1) NOR (b2)** — it's V6-vs-(V6×n), a third quantity. It is an **upper bound** on (b1) only if you accept that A and B's algebraic divergence ≤ V6 vs V6-replica's RNG divergence, which is **not obvious**. Conversely it could over- or under-estimate (b1). Discuss whether 12.73% gives any rigorous bound on `d_pure`, and if not, what does.

(c) The V6_NOISE arm (V6-seed1337) is supposed to estimate `d_pure = paired_cohen_d(V6-seed42, V6-seed1337)` as the noise floor. **But V6_NOISE only changes the seed** — it does NOT change `deterministic`/`warn_only`/CUDA kernels. So `d_pure` lower-bounds the noise floor under (RNG seed + warn-only kernel non-determinism + sample-order randomness). Is this an **adequate or inadequate** noise floor for the V7-vs-V6 comparison?

(d) **Threshold-inflation contingency**. Plan F's threshold formula is `d_thr = max(0.10, 1.8 × d_pure)`. The 80%-power MDE at n=7403 is Cohen's d ≈ 0.033. If V6_NOISE measures `d_pure > 0.10`, the effective threshold becomes `1.8 × d_pure`. Quantitatively: based on the 12.73% A/B drift on `chain_normal_mse`, what's your **point estimate and 90% range** for `d_pure` (paired Cohen's d between V6-seed42 and V6-seed1337 on `chain_normal_mse`)? If your point estimate exceeds 0.10, what's the resulting `d_thr`, and does Plan F still have power to detect the realistic V7 effect size (V6→V7 expected effect under Grönwall predicted reduction)?

(e) Specifically: if the σ-normalize sanity failed at 12.73% on `val_chain_normal_mse` between A and B (same seed!), what does this imply for the expected magnitude of `d_pure` between V6-seed42 and V6-seed1337 (different seeds + same kernels)? Will `d_pure` be larger or smaller than 12.73% relative drift? Will Cohen's d (effect size on per-slice paired MSE) be larger or smaller than 0.10?

---

#### Q3 — If raw-rollout V7 also exhibits non-reproducibility, what should the project do?

Suppose V7 finishes 160K steps. The author wants to do a **secondary** sanity check: re-train V7 from scratch with identical settings (same `seed=42`, same yaml, same hardware, same GPU id 2) and verify the two V7 runs converge to within an acceptable threshold of each other. Call this V7-replica.

If V7-replica shows >5% drift on `val_chain_normal_mse` against V7-original, the project has the same problem as σ-normalize, just on the raw-rollout path.

**Audit Q3**:

(a) **Is V7-replica even necessary**, given the project will already have V6_NOISE = V6-seed1337 to estimate the seed-twin noise floor on the V6 codepath? Or does V6_NOISE only validate the **V6** rollout/loss configuration, not the **V7** configuration (different `step_weights` could in principle activate different kernel codepaths)?

(b) If V7-replica is necessary AND fails, list the candidate remediations and rank them by (1) statistical validity of resulting paired comparison, (2) GPU cost, (3) implementation risk:

- **Option A**: Switch entire training to bit-deterministic mode (`torch.use_deterministic_algorithms(True, warn_only=False)`). Likely loses Memory-Efficient-attention speedup (~30-50% slowdown estimated). Requires re-implementing affected ops or accepting failure-to-launch on those ops.
- **Option B**: Disable Memory-Efficient-attention only; keep `warn_only=True` for the rest. Smaller slowdown (~10-20%). May not actually fix the issue if `adaptive_avg_pool2d_backward_cuda` is the dominant non-determinism source.
- **Option C**: Accept non-determinism, redefine the comparison metric to be **distributional** rather than **paired**: report mean ± std over N=3 runs per arm with different seeds, use Welch's t-test instead of paired t-test. Cost: 3× GPU per arm.
- **Option D**: Accept non-determinism, redefine the threshold `d_thr` to absorb it: instead of `d_thr = max(0.10, 1.8 × d_pure)`, use `d_thr = max(0.10, K × d_pure)` with K large enough to ensure power against the empirical worst-case kernel-noise drift. Requires K ≥ 4-5 based on σ-normalize's 12.73%, which would essentially require Cohen's d ≥ 0.4-0.5 to declare a difference — likely too high for the project to detect any realistic Grönwall effect.
- **Option E**: Accept the existing V7 vs V6 comparison as-is, but reframe the conclusion language: replace "V7 outperforms V6" with "V7 differs from V6 by Δ, where Δ is consistent with / exceeds the kernel-noise floor d_pure=K". Make d_pure a **mandatory disclosed metric** in any paper using this experiment.
- **Option F (★ added per multi-agent pre-review)**: **Same-process algebraic golden test BEFORE V7 launch**. Concrete protocol: (1) Load V6@step_160000 in a single Python process; (2) draw one fixed batch and freeze the RNG; (3) call `model.eval()` + use `torch.backends.cuda.enable_math_sdp(True)` (forces math kernel, deterministic); (4) compute A path total loss `(w_v6 · ℓ).sum()/Σw_v6` and B path total loss `((w_v6×n) · (ℓ/n)).sum()/Σ(w_v6×n)`; (5) require bit-identical or ≤ 1e-7 absolute drift. Cost: ~30 GPU-seconds. **Decisive** discriminator between (α)/(β)/(ζ) hypotheses since at `model.eval()` + math-SDPA + fixed batch + non-persistent buffer the only remaining drift source is FP32 reduction-order. If A and B disagree in this regime, algebra is wrong; if they agree, sanity-failure is **definitively** training-time + kernel-noise + 20K-step weight divergence.
- **Option G (★ added per multi-agent pre-review)**: **Paired raw-vs-σnorm evaluator on V6@step_160000 across full validation set**. Cost: ~100 LOC, 0 GPU-day retrain. For each val slice compute both totals; report distribution of `|A_total – B_total| / B_total`. If max drift on full val is sub-1e-4, algebraic equivalence holds in inference; remaining 12.73% must be attributable to training-trajectory divergence — *which then reroutes the entire diagnosis* away from algebra. If max drift is >1%, algebra is broken even at eval and the formula needs revision.
- **Option H (★ added per multi-agent pre-review)**: **Determinism intervention bundle**: simultaneously (i) `torch.backends.cuda.enable_math_sdp(True)` + `enable_flash_sdp(False)` + `enable_mem_efficient_sdp(False)`; (ii) replace `nn.AdaptiveAvgPool2d` (whose backward is the only non-deterministic op flagged in the comparator log) with deterministic `F.avg_pool2d` at fixed kernel size; (iii) `torch.set_float32_matmul_precision("highest")` to force FP32 matmul precision. Estimated overhead: <2% throughput loss. Eliminates **ALL** documented warn-only ops in the train log. Cost: ~50 LOC and one bench run; **not yet attempted**.

(c) Among A–H (in particular F + G + H form a cheap pre-launch diagnostic bundle costing <2 GPU-hours and not blocking V7), which is your **strongest recommendation** for this specific project (small-team, clinical PET imaging, journal/MICCAI-class submission)? Defend your choice in at least 5 sentences referencing the specific Plan F constraints, the per-hop rel-err signature, and the val_pair_total = 10.16% structural-independence constraint.

(d) Is there a **9th option you're aware of** that has not been enumerated? E.g., post-hoc nondeterminism correction via averaging multiple forward passes; switching to TensorFloat-32 OFF; switching from FA-2 to vanilla scaled_dot_product (overlaps with H but separate from full math-SDPA); constraining the affected ops via `torch._dynamo.config.cache_size_limit`; etc. List concretely.

---

### Format your reply as

```text
## Q1 — σ-normalize root cause
- Verdict: [...]
- Reasoning: [...3+ sentences...]
- Evidence: [...bullet list...]
- Recommended additional test: [...]

## Q2 — raw-rollout path adequacy
- Verdict: [...]
- Reasoning: [...]
- Evidence: [...]
- Implication for d_pure floor: [...]

## Q3 — Remediation ranking
- V7-replica necessity: [...]
- Option ranking (A/B/C/D/E/F/G/H + any 9th): [...]
- My strongest recommendation: [...] because [...]
- Caveats / things you would still investigate: [...]
```

---

### Hard constraints on your reply

1. **Distinguish factual vs theoretical claims**:
   - **Factual claims** (e.g., "line X of file Y does Z", "value V is in artifact W") MUST cite an attached file path with line number.
   - **Theoretical / methodological claims** (e.g., "FP32 reduction-order asymmetry produces drift O(ε·N)") MAY cite textbook, established paper, or common knowledge — just be explicit about where the claim's authority comes from.
   - **No vague hand-waving** of either type: every numerical claim must be either reproducible from attached evidence or explicitly flagged as a heuristic/estimate.
2. **No design re-litigation**. If you find issues with V7 step_weights, V6 baseline, or Plan F's d_thr formula, file them as **out-of-scope for this review** in a separate appendix; do not let them dilute Q1/Q2/Q3 verdicts.
3. **Adversarial framing**. Assume the author is wrong by default. Make the author prove correctness, not the other way around. **In particular**: do not be persuaded by the author's own framing in the failure-pattern observation block above — the (α)/(β)/(γ)/(δ)/(ε)/(ζ) enumeration is offered as candidate hypotheses, NOT as a diagnosis to be accepted.
4. **Distinguish algebraic equivalence from trajectory equivalence**. These are different invariants. The σ-normalize failure could be algebraic (formula wrong on paper), reduction-order (formula correct on paper but FP32 implementation diverges from inference value), trajectory (formula correct on paper AND in inference but training trajectory diverges), or any combination. Do not conflate them.
5. **Acknowledge uncertainty explicitly**. If a question requires data the author did not provide (e.g., per-iter raw step losses for a same-process golden test, the resolved per-hop n_j values, raw `step_normalizers` floats from a B-path log), say so and **request the specific artifact**. The author will provide on demand.
6. **The val_pair_total = 10.16% constraint is non-negotiable**. Any Q1 verdict that does not address why pair-loss (which is structurally outside the rollout chain) drifted by 10.16% is **incomplete**. Reject "trajectory amplification only" diagnoses on this basis.

--- PROMPT END ---

---

## After reviewer replies

Each reviewer (GPT-5.5 / Claude Opus 4.7 / optional Gemini-2.5-Pro) returns one report. I will:

1. Cross-check verdicts on Q1/Q2/Q3 — disagreement → re-prompt with disagreement summary.
2. Identify any reviewer's "9th option" (Q3.d) and append to the design-decision register.
3. **If ≥2 reviewers recommend Option F (same-process algebraic golden test) as P0**, run it BEFORE Plan F V7 launch. Per multi-agent internal pre-review (`review/plan/ROUND4_PRE_REVIEW_CONSENSUS_20260506.md`), this is already the strong consensus and should be considered pre-committed.
4. If ≥2 reviewers say **raw-rollout path is also unsafe** (Q2 negative even with Option F passing), suspend Plan F V7 launch and escalate to: implement Option H (math-SDPA + AdaptiveAvgPool replacement) bundle, then re-decide.
5. If ≥2 reviewers recommend a specific remediation (Q3.c), promote it to actionable in [PLAN_F_V7_V8_FROM_SCRATCH_150K_20260505.md](../../plan/PLAN_F_V7_V8_FROM_SCRATCH_150K_20260505.md) §6 (decision thresholds) before launching.
6. If reviewers' point estimate for `d_pure > 0.20` (Q2.d), trigger Plan F's threshold-inflation contingency: switch primary metric to multi-seed Welch (Option C) or fall back to Option E reframing. See `PLAN_F_V7_V8_FROM_SCRATCH_150K_20260505.md` §11 (added 2026-05-06).
7. If reviewers converge on Option E (reframe conclusion language + disclose d_pure), update [ANALYSIS.md](ANALYSIS.md) §6 PRIMARY decision rule and [Plan F §3](../../plan/PLAN_F_V7_V8_FROM_SCRATCH_150K_20260505.md) accordingly.

