# Peer Review Integration — 20260516

- generated_at: 2026-05-16 Asia/Shanghai
- branch: foc_lite_hop0
- commit_when_generated: acc7a9d
- reviewers: agent1 (本地 deep-dive)、agent2 (本地 adversarial)
- inputs: [PEER_REVIEW_PROMPT_image_aux_direction_20260516.md](./PEER_REVIEW_PROMPT_image_aux_direction_20260516.md) + 两份 review 全文
- purpose: 整合两份 review 的发现，列出**已修复的具体问题**与**待补的实验**

---

## 0. 一行结论

两份 review 高度共识：**我之前所有"V7-V8 是 image_aux ablation"和"V7-V6_NOISE 是 d_pure"的标注都是错的**；V7/V8/V6_NOISE 三个 yaml 在 step_weights 上都不同（agent1 用 resolved config 锁死）。在补真正的单变量对照（V13 + V14）之前，**任何基于 V7-V8 paired delta 的结论都不可信**，包括 V9 (β_NORMAL) 与 V11' (强化 hop0) 两个方向。

---

## 1. 两份 review 的整合判定

| Q | agent1 verdict | agent2 verdict | 共识 |
|---|---|---|---|
| Q1 cascade PSNR 单调上升 | partially-accepted（latent v_std vs image PSNR 非等距）| partially-accepted（v_std 衰减 68× vs MSE 仅降 25% 数量级不符；PSNR_clip3 clip 抑制 high-SUV failure；ckpt selector β biased）| **partially-accepted** |
| Q2 哪一跳是瓶颈 | corrected 方向对但 D20 是真瓶颈需更多证据 | **neither**（chain coherence vs shared backbone 未解耦；ckpt selector 循环论证；D20 边际 ΔMSE 实际比 NORMAL 低）| **neither**（更严格）|
| Q3 A/B/C 排序 | A > 其他，但前提是先补对照实验 | B' (廉价 multi-hop) > A > B > C，前面加 §0 量测 gate | **先做 disambiguation 量测；A 和 B' 都可能对** |
| V11 撤回是否过度 | 不算（先做 gate 再说）| **是过度修正**（撤回逻辑只否定了原 V11 论据，没否定结论）| **倾向 agent2**（撤回过度）|

---

## 2. 已立即修复的具体问题

### 2.1 V9 yaml 致命 bug（agent1 发现）

**问题**：之前的 [V9_normal_emphasis.yaml](./V9_normal_emphasis/V9_normal_emphasis.yaml) 顶层用了 `dataset:`，但 [train_first_hop.py:165](../../train_first_hop.py#L165) 全部读 `cfg["data"]`；而且缺少 6 个必需顶层块（`backbone` / `rae` / `first_hop` / `loss` / `lr_schedule` / `transport`），直接跑会 crash。

**修复**：基于 [V7_config.resolved.yaml](../0511/log_snapshots_20260516_163900/configs/V7_config.resolved.yaml) 作为完整 base，只 patch 4 处（output_dir、run_name、`best_metric_terms[normal].weight: 1.50→2.50`、`step_weights: → [0.5870, 2.0, 2.8333, 2.5173]`）。已通过 `yaml.safe_load + key check`，16 个顶层块全部齐全。

### 2.2 PLANF_FINAL_ANALYSIS 与 IMAGE_AUX_ARCHITECTURE_ANALYSIS 的错误标注

**问题**：两份 0516 分析文档把 V7-V8 当作 image_aux 对照、V7-V6_NOISE 当作 d_pure。**这是事实错误**（见下 §3）。

**修复**：在两份文档顶部加 PEER REVIEW CORRECTION 警示块，明确：
- V7-V8 = (image_aux on/off) × (Grönwall vs V6 step_weights)
- V7-V6_NOISE = (Grönwall vs V6 step_weights) × (seed 42 vs 1337)
- 所有基于这两个对照的数字结论需在 V13 + V14 跑完后重新评估。

### 2.3 新增 V13、V14 config（agent1 推荐的纯对照）

- **V13** = V7 config patch image_aux off only（[V13_true_image_aux_off.yaml](./V13_true_image_aux_ablation/V13_true_image_aux_off.yaml)）
- **V14** = V7 config patch seed=1337 only（[V14_v7_seed1337.yaml](./V14_true_d_pure/V14_v7_seed1337.yaml)）

两个 yaml 都基于 V7 resolved config 做单变量 patch，all 16 top-level blocks present，已 `yaml.safe_load` 通过。

---

## 3. agent1 核心发现的直接验证

我跑了 grep 确认 agent1 的因果污染 claim：

```
V7_config.resolved.yaml:251 step_weights:
  - 0.6106  - 2.0000  - 2.7041  - 2.0423    # Grönwall raw
V7_config.resolved.yaml:258 image_aux: enabled=true, lambda_max=0.04
V7_config.resolved.yaml:111 seed: 42

V8_config.resolved.yaml:248 step_weights:
  - 0.5     - 2.0     - 1.5     - 1.0       # V6 经验
V8_config.resolved.yaml:262 image_aux: enabled=false, lambda_max=0.0
V8_config.resolved.yaml:112 seed: 42

V6_NOISE_config.resolved.yaml:216 step_weights:
  - 0.5     - 2.0     - 1.5     - 1.0       # V6 经验
V6_NOISE_config.resolved.yaml:224 image_aux: enabled=true, lambda_max=0.04
V6_NOISE_config.resolved.yaml:91  seed: 1337
```

**事实**：本项目目前**没有任何一组 V vs V 是 image_aux 的单变量对照**，也**没有 V7 配置下的纯 seed 噪声**。V6_NOISE 之所以叫"noise"是因为它本意是 V6 的 seed 控制，但与 V7 比时 step_weights 也不同。

**0516 分析里所有 paired t-test 数字仍然存在**（数据没造假），但它们度量的不是 image_aux 的纯效应，而是**联合效应**。

---

## 4. agent2 推出的 disambiguation 实验（应优先级最高）

agent2 提出两个**不需要新训练**的零成本实验，能在 V13/V14 训出来前就缩小决策空间：

### 4.1 单步 PSNR 面板（区分 chain coherence vs shared backbone）

在 V7 和 V8 现有 checkpoint 上跑：
- 输入 GT z_{i}，模型 forward 一步得到 z_{i+1}_pred
- decode(z_{i+1}_pred) 与 GT x_{i+1} 比 PSNR_clip3
- 对每个 hop 做（不级联）

**判定**：
- 若 V7 vs V8 单步 ΔPSNR ≈ 级联 ΔPSNR → chain coherence 通道主导 → V11' (强化 hop0) 是对的
- 若级联 ΔPSNR >> 单步 ΔPSNR → shared backbone 通道主导 → V11 (multi-hop image_aux) 不该被撤回
- 若两者都有贡献 → 需要两个方向都试

成本：复用 [eval_first_hop_224_clip3.py](../../eval_first_hop_224_clip3.py) 即可，~30 分钟开发 + 30 分钟跑。

### 4.2 ROI-PSNR 面板（[0501 review §3](../0430/reviewer/literature_architecture_image_aux_review_20260501.md)）

在现有 V7/V8/V6_NOISE checkpoint 上加：
- top-10% / 5% / 1% SUV mask 内的 PSNR
- high-gradient mask 内的 PSNR
- SUVmax / SUVmean error
- 未-clip PSNR（验证 clip3 是否抑制了 high-SUV failure）

成本：~1 天开发（per-slice CSV 已存在，只需加 mask 计算）。

### 4.3 Lipschitz 测量（V9_PREREGISTRATION §0 已写好骨架）

[tools/estimate_per_hop_lipschitz.py](../../tools/estimate_per_hop_lipschitz.py) 闭式解部分已 self-verified；只需把模型/数据 wiring 接上。

成本：10 分钟（脚本骨架已写好）。

---

## 5. agent2 独有的 B' 候选（应加入实验清单）

**B' = multi-hop image_aux + v_std² 等比衰减权重**

$$
\lambda_\text{img}^{(j)} = \lambda_\text{img,base} \cdot \frac{(\sigma_j \Delta t_j)^2}{(\sigma_0 \Delta t_0)^2}
$$

代入数：hop0=1.0, hop1=0.26, hop2=0.16, hop3=0.13。

**为什么 B' 比团队撤回的 B 严肃**：
- hop3 image_aux 实际 magnitude 只有 hop0 的 13%，**不会被 latent loss 淹没**（这是我之前反对 multi-hop 的主要理由 —— 它失效了）。
- 可以只跑 hop3-only 子版本，多 ~10% 训练时间。
- 直接验证 chain coherence vs shared backbone 的因果通道（同 §4.1）。

B' 应作为 V15 加入实验队列，与 V13/V14 并列。

---

## 6. agent2 独有的另一个发现：σ-normalize ablation 优先级倒挂

[STEP_WEIGHTS_THEORY_REFERENCE §5.2](./STEP_WEIGHTS_THEORY_REFERENCE.md) 写："σ-normalize rollout loss 是 critical ablation，比 β-axis (V9) 更优先"。

但实际下一步实验队列里 V9 (β-axis) 排在第一位，σ-normalize 没有计划。**这是自相矛盾**。

修复：把 σ-normalize ablation（命名 V16）列入待办，优先级 ≥ V9。

---

## 7. 整合后的最终行动队列（按 prerequisite 排序）

| # | 实验/动作 | 类型 | 成本 | 阻塞下面什么？ |
|---|---|---|---|---|
| 0 | Lipschitz 测量（V7 checkpoint）| 评估，零训练 | 10 min | V9 的 launch 决定 |
| 1 | 单步 PSNR 面板（V7/V8）| 评估，零训练 | 1 h | V11/V11'/B' 之间的选择 |
| 2 | ROI-PSNR 面板（V7/V8/V6_NOISE）| 评估，零训练 | 1 天 | 所有 image_aux 相关 claim |
| 3 | **V13**（true image_aux ablation）| 1 次 160K 训练 | ~7 天 | 后续所有 image_aux 相关 claim |
| 4 | **V14**（true d_pure）| 1 次 160K 训练 | ~7 天 | 所有 paired-t 比较的阈值 |
| 5 | V9（β_NORMAL=2.5）| 1 次 160K 训练 | ~7 天 | 需 §0 gate 通过 |
| 6 | V11' (强化 hop0 image_aux) | 1 次 160K 训练 | ~7 天 | 需 §1 + §2 支持方向 |
| 7 | **V15** (B' multi-hop image_aux, v_std² 衰减) | 1 次 160K 训练 | ~7-9 天 | 需 §1 显示 shared backbone 通道存在 |
| 8 | V16 σ-normalize rollout ablation | 1 次 160K 训练 | ~7 天 | 与 V9/V11'/V15 解耦后单独评估 |

**关键依赖**：
- §0/§1/§2 是零训练 disambiguation，**应立即并行启动**。
- V13/V14 是 ground truth 对照，**无论后续选 V9/V11'/V15 哪个方向都需要**，应该最先排上 GPU。
- V9 在 §0 Lipschitz gate 通过前**不应 launch**（已写入 [V9_PREREGISTRATION.md §0](./V9_normal_emphasis/V9_PREREGISTRATION.md)）。
- V11' / V15 的选择由 §1 单步 PSNR 决定，不要在 §1 完成前先选。

---

## 8. 之前文档的状态

| 文档 | 状态 |
|---|---|
| [PLANF_FINAL_ANALYSIS_20260516.md](./PLANF_FINAL_ANALYSIS_20260516.md) | 顶部加了 CORRECTION，需待 V13/V14 完成后重写 §2-§4 |
| [IMAGE_AUX_ARCHITECTURE_ANALYSIS_20260516.md](./IMAGE_AUX_ARCHITECTURE_ANALYSIS_20260516.md) | 顶部加了 CORRECTION；§1-§2 代码事实仍有效；§10 V11' 推荐**暂停**等 §1+§2 量测结果 |
| [STEP_WEIGHTS_THEORY_REFERENCE.md](./STEP_WEIGHTS_THEORY_REFERENCE.md) | 理论部分仍有效；§7 应升级 σ-normalize 优先级 |
| [V9_PREREGISTRATION.md](./V9_normal_emphasis/V9_PREREGISTRATION.md) | §0 Lipschitz gate 仍有效；阈值表 §4 依赖 V14 的真实 d_pure 重算 |
| [PEER_REVIEW_PROMPT_image_aux_direction_20260516.md](./PEER_REVIEW_PROMPT_image_aux_direction_20260516.md) | "资料 1-2" 部分需在顶部加 confounding 警告再发给后续 reviewer |

---

## 9. 对我自己的元结论

两份 review 都指出我**两次**犯了同型错误：
1. 第一次（推 V11）：用 ablation delta 大小推未挖掘空间。
2. 第二次（撤回 V11 推 V11'）：用 absolute level + selector-biased ckpt 推饱和。

共同 root cause：**没把"训练目标"与"系统性质"分离**。NORMAL 在 chain PSNR 最高、在 ablation delta 也最大，**两件事都部分来自 β_NORMAL=1.5 是 selector 最高权重**这个事实，而我用前者推后者、又反过来用后者推前者。

写入 [/memories/repo/pet_latent_residual_eval.md](/memories/repo/pet_latent_residual_eval.md) 的教训：

> **任何关于 V7/V8/V6_NOISE 之间 PSNR delta 的解读，必须先检查 resolved config 三个轴 (image_aux / step_weights / seed) 哪些被同时改了。当前没有任何一组是单变量对照。**

---

## 10. 我现在做完的事

1. 在 PLANF + IMAGE_AUX 两文档顶部加了 CORRECTION 警示块。
2. 修了 V9 yaml 致命 bug（基于 V7 resolved config patch 重建，16 个顶层块齐全，YAML 已 load 验证）。
3. 新建 V13 (true image_aux ablation) 和 V14 (true d_pure) 的 yaml，同样基于 V7 resolved patch。
4. 写了本文档作为 review 整合的单一入口。

**未做（明确留给你决定的事）**：
- 是否要我现在就实现 §4.1 单步 PSNR 评估代码（基于 eval_first_hop_224_clip3.py 改造）？
- 是否要把 V13/V14 launch 脚本也建好（类似 [run_v9_train.sh](./V9_normal_emphasis/run_v9_train.sh)）？
- V15 (B' multi-hop image_aux with v_std² 衰减) 的 config 要不要也建？

这三件事都是机械动作，我可以继续做，但每件都涉及代码改动（不是 yaml patch），所以等你确认方向。
