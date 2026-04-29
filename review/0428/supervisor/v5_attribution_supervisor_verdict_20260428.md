# Supervisor 裁决：V5 归因分析的 Author/Reviewer 讨论

**日期**：2026-04-28
**角色**：Supervisor
**目的**：仲裁 author 与 reviewer 关于 V5 归因实验（rollout-heavy + null-control）证据强度的分歧
**关联文档**：
- Author：[`review/0428/local/v5_attribution_analysis_20260428.md`](../local/v5_attribution_analysis_20260428.md)
- Reviewer：[`review/0428/reviewer/v5_attribution_viewer_assessment_20260428.md`](../reviewer/v5_attribution_viewer_assessment_20260428.md)
- Operator：[`review/0428/operator/`](../operator/)
- V3 baseline：[`review/0427/logs_eval/v3_200k_best_fullval_clip3_summary.log`](../../0427/logs_eval/v3_200k_best_fullval_clip3_summary.log)

---

## 0. Supervisor 裁决

**Reviewer 立场基本正确，Author 立场方向正确但措辞超前。** 双方分歧不在科学事实，而在**证据强度**与**因果 claim 的提级时机**。

| 议题 | Author 立场 | Reviewer 立场 | Supervisor 裁决 |
|---|---|---|:-:|
| null-control 是否复现 V3 image-dominated 状态 | 证明 | 支持假设 | 偏向 **Reviewer**（rolling-val 不足以"证明"） |
| V3 plateau 是否权重配方病 | 证明 | 与之 consistent，未 closed | 偏向 **Reviewer** |
| V6 是否独立于 V5 归因 | 是（V6 from scratch） | 是（V6 不应以 V5 归因为唯一立项依据，但可独立推进） | 双方一致 |
| rolling-val 0.000159 是否 V5 改善证据 | 已承认是 window 回卷 | 同上 | 双方一致 |
| 当前是否可基于 V5 归因放行 V6 200K full | 隐含"是" | **No-Go**：必须等 full-val + Path A | 偏向 **Reviewer** |
| V6 pilot 是否需要等 V5 归因闭环 | 不需要 | 不需要 | 双方一致 |

**核心裁决**：Author 文档需做 §4 的措辞降级（reviewer 已给出对照表），但 V6 pilot 可继续（双方共识）。**V6 200K 全程放行**需要补齐 V5 full-val + null-control full-val + Path A 三组数据后再做。

---

## 1. 双方共识（不需仲裁）

| 共识 | 内容 |
|---|---|
| C1 | V5 rollout-heavy 的 step 93200 "相变" 是 rolling-val 窗口回卷（window_start=0），不是真实 phase transition |
| C2 | rolling-val 不能作为 claim-level metric，full-val 才能 |
| C3 | V6 是 from scratch 200K，不继承 V5 状态，pilot 可独立推进 |
| C4 | null-control 配置（resume V3 best, total_steps_override=200000, V3 recipe）在 launch 层面是有效控制 |
| C5 | V5 rollout-heavy 与 null-control 的 full-val 数据需要 sync 进 review 树才能形成最终结论 |
| C6 | val_chain_normal_mse 在 rolling-val 中波动 0.000159 ~ 0.000382（2.4×），单 window 数字不可作 claim |

---

## 2. 分歧点逐条仲裁

### 2.1 分歧 A：null-control 当前数据"证明"还是"支持假设"

**Author 表述**："null-control 完全复现了 V3 后段的 image-dominated 状态" / "证明了 V3 的 plateau 是权重配方病，不是训练量不足"

**Reviewer 表述**："null-control rolling snapshot supports the hypothesis... pending full-val"

**事实核查**：

| 指标 | V3 step 130000-172600 | null-control step 86850-100850 | 一致性 |
|---|---|---|---|
| pair_frac | 1-2% | 0.9-3.2% | ✓ |
| roll_frac | 5-7% | 4-9.7% | ✓ |
| img_frac | 92-96% | 87-94% | ✓ |
| val_chain_normal_mse 趋势 | 0.000169-0.000172（不动） | 0.000159-0.000382（波动） | ✓（都没改善）|

数据上 null-control 14K 步行为与 V3 后段一致，author 的"复现"这个事实陈述成立。

**但**："复现 V3 image-dominated"≠"证明 plateau 是权重配方病"。后者是因果 claim，需要：
1. ✓ 证据 A（已有）：保持 V3 recipe 继续训练 → 不改善
2. ✗ 证据 B（缺）：V6 改 recipe 从 scratch 训练 → 改善（V6 仍在 Phase I, 进度 ~7.5%）
3. ✗ 证据 C（缺）：V5 rollout-heavy / V5 null-control full-val 数字

A 单独成立只能证明"在 V3 best 之后保持 V3 recipe 不行"，不能证明"V3 之前的 plateau 也是 recipe 病"——存在替代解释：
- V3 best 已陷入 narrow basin，任何继续训练都退化
- LR cosine 末段太小，已无法逃出局部 optima
- EMA 已收敛到稳态

**裁决**：Author 的事实描述（image_aux 占据 weighted loss fraction 87-94%）✓；因果 claim（"plateau 是权重配方病"）需要降级到"hypothesis supported by null-control short-window data, awaiting full-val + V6 outcome for closure"。

采纳 Reviewer §4 的措辞对照表。

---

### 2.2 分歧 B：weighted loss fraction vs gradient budget

**Author 表述**："image_aux 占据 87-96% 的梯度预算"

**Reviewer 表述**：应改为 "weighted loss fraction / supervision pressure"，不是"gradient share"

**事实核查**：

代码 [`train_first_hop.py` L1936-1940](../../../train_first_hop.py#L1936)：

```python
pair_weighted = pair_loss_weight * pair_losses["total"]
roll_weighted = lambda_roll * rollout_losses["loss_total"]
img_weighted  = lambda_img * loss_img
img_frac = img_weighted / (pair_weighted + roll_weighted + img_weighted + ...)
```

logged 的 `*_frac` 是 **weighted scalar loss 的占比**，不是反向传播 `||grad||` 的占比。两者关系：

$$\|g_{img}\| / \|g_{total}\| \neq L_{img,weighted} / L_{total,weighted}$$

只有当 backbone 是线性映射、各 loss 对参数的 sensitivity 相同时两者才相等，DiT 显然不满足。所以 reviewer 的术语严谨性指控成立。

但实务层面，过去 V3/V4/V5 文档都用 `*_frac` 作为"梯度预算"代理，是一个**项目内已建立的简化口径**。如果改了术语，整个项目历史文档体系都需要重写。

**裁决**：Author 文档**保留 fraction 作为梯度代理的口径**（项目惯例），但**首次出现处加脚注**说明：

> 注：`*_frac` 严格定义是 weighted loss 占比，不是反向梯度 norm 占比；项目内沿用其作为 supervision pressure 的代理。两者数值通常同向，但不等价。

这样既保持术语连贯，又满足 reviewer 的严谨性要求。

---

### 2.3 分歧 C：V6 立项依据是否需要 V5 归因闭环

**Author 暗示**：V6 已经在 V5 归因基础上立项

**Reviewer 立场**：V6 pilot 可继续（独立 experiment），但 V6 200K full 不应基于"V5 归因"作为唯一证据放行

**事实核查**：

V6 plan 的核心论据有三层：
1. V3 200K JSONL 直接观察（V3 后段 image=92%, transport collapse） — **已有**
2. null-control 短窗对照（V3 recipe resume 仍 image-dominated） — **已有但需 full-val 闭环**
3. V5 rollout-heavy 的失败（如果失败）反证"只改 rollout 不够" — **未闭环**

V6 改 pair_weight=15 主要论据来自 (1)，不依赖 (2)(3)。但 V6 plan §0 引用了 "V3 200K 训练数据证明 rollout 梯度占比高（32%）时模型达到 best"，这是基于 V3 jsonl 而非 V5 归因。

所以 V6 立项严格说**不依赖 V5 归因**，依赖 V3 直接观察。V5 归因是**confirmation**，不是 dependency。

**裁决**：

- ✅ V6 pilot 可独立推进（双方共识）
- ⚠️ V6 200K full 放行决策**不应在 +50K Phase II 分歧点之前作出**；届时已有：
  - V6 自身 Phase I/II 的内生 metric
  - V5 attribution full-val 数据（应在此之前 sync 完成）
  - Path A 诊断（应在此之前完成）
- 这与 reviewer "NO-GO for V6 full 200K solely from current V5 attribution" 立场一致

---

### 2.4 分歧 D：V5 rollout-heavy 是否已"判定失败"

**Author 表述**："V5 rollout-heavy 改了权重... 结果也没有明确改善"（隐含失败）

**Reviewer 表述**："current local files are insufficient to determine whether V5 rollout-heavy is harmful, neutral, or mildly useful under full-val"

**事实核查**：

Local repo 没有 V5 full-val JSON/CSV/summary 文件，只引用了远程路径。Author 文档的"失败"判定建立在 rolling-val 数据上，但项目已共识 rolling-val 不是 claim-level metric（C2）。

**裁决**：偏向 Reviewer。Author §3.2 应改为："V5 rollout-heavy rolling-val 未显示明确改善，但 full-val 数据尚未 sync 进 review 树，最终判定 pending"。

---

### 2.5 分歧 E：null-control 的 budget 是否足够

**Reviewer 提出**：null-control 当前 step 100850，仅训了 14K 步（V3 best 之后），如果要与 V5 rollout-heavy（评估在 +50K 或固定 ckpt）对齐，需要继续训练。

**Author 文档未涉及此点**。

**事实核查**：V5 rollout-heavy 的远程文件路径包括 `v5_best` 和 `v5_step_100000`，意味着 V5 rollout-heavy 评估了至少到 step ~150000（V3 86800 + 50K resume 后）。null-control 当前 100850 仅相当于 V5 rollout-heavy 的 **28% 进度**。

**裁决**：Reviewer 的 budget 对齐要求合理。null-control 应继续训练到 V5 rollout-heavy 同等 budget（约 step 130K-150K）才能做对照 full-val。

操作层面：[`review/0428/operator/04_v5_attribution.sh`](../operator/04_v5_attribution.sh) 应在 null-control 完成后执行。

---

## 3. 最终裁决总结

### 3.1 Author 必须修订的内容（采纳 Reviewer §4 措辞表）

| 当前措辞 | 修订为 |
|---|---|
| "image_aux 占据 87-96% 的**梯度预算**" | "image_aux 占据 87-96% 的 **weighted loss fraction**（梯度代理）" |
| "null-control **证明** V3 plateau 不是训练量不足" | "null-control rolling 数据**支持**'继续训练不足以打破 plateau'的假设，待 full-val 闭环" |
| "V3 plateau **是**权重配方病" | "V3 plateau 与 recipe / supervision-pressure 瓶颈**一致**，但未排除 narrow basin / LR floor 等替代解释" |
| "V6 **基于正确归因**" | "V6 受当前归因假设驱动，仍需 staged validation" |
| "V5 rollout-heavy ... 也没有明确改善" | "V5 rollout-heavy rolling-val 未显示明确改善，full-val 待 sync" |

### 3.2 Operator 必须执行的内容（采纳 Reviewer §5）

| 优先级 | 任务 | 阻塞 |
|:-:|---|:-:|
| P1 | sync V5 rollout-heavy + null-control full-val JSON 进 `review/0428/operator/review/` | V6 200K full |
| P2 | null-control 继续训练到 step ~130K-150K | 与 V5 对照 |
| P3 | 执行 Path A diagnostic（`02_pathA_baseline.sh` + `04_pathA_v5_best.sh`） | 因果机制确认 |
| P4 | 完成 reviewer §5 的最终归因表（V3/V5/null-control × full-val + Path A） | V5 归因 closure |

### 3.3 V6 推进决策（双方共识 + supervisor 收敛）

- ✅ V6 pilot 继续（不阻塞）
- ⏸ V6 200K full 放行决策**延至 V6 +50K Phase II 分歧点**，届时同时审查：
  - V6 自身 Phase I/II 内生 metric
  - V5 归因表（P1+P2+P3+P4 全部完成）
  - 决策矩阵（reviewer §6）

### 3.4 V6 pilot 期间的并行任务时序

```
day 0  (now)  : V6 Phase I 启动 (~step 7.5K), null-control step 100850
day 0-2       : null-control 继续到 ~130K + sync V5 full-val
day 2-3       : 执行 Path A diagnostic (V3 baseline + V5 best)
day 3         : 出 V5 attribution 最终归因表
day 4-5       : V6 Phase I 结束 (step 50K), Phase II 分歧点决策
                ├── V6 内生 metric OK + V5 attribution closed → Go V6 200K
                ├── V6 内生 metric OK + V5 attribution unresolved → Pause, debug
                └── V6 内生 metric 退化 → Abort, replan
```

---

## 4. 给 author 与 reviewer 的反馈

### 4.1 给 Author

文档**事实描述准确**，**方向正确**，但有两个习惯性问题需要纠正：

1. **过早把 hypothesis 提级为 proof**：null-control 复现 V3 行为是支持假设，不是 closing causal proof。在等 full-val 之前用"证明"是 over-claim。
2. **混用术语**：weighted loss fraction 与 gradient budget 不等价。建议项目内统一加脚注口径。

V6 立项的真实科学基础（V3 200K JSONL 直接观察 + V3 → null-control 一致）足以支持 V6 pilot 启动；不需要把 null-control 提级为 proof 来"加强"立项依据。

### 4.2 给 Reviewer

Assessment **方法严谨、证据边界清晰、措辞精确**，全文几乎可以直接作为 V5 attribution 的最终模板。两个建议：

1. §4 措辞对照表非常有用，但建议明确告知 author "这是项目惯例下可接受的最强表述"，而不仅是 "recommended"，避免 author 误读为可选建议。
2. §6 决策矩阵列了 6 行，建议明确标注哪一行最可能（基于当前 V3 + null-control + V5 rolling-val 数据，最可能落在 "V5 ≤ V3, null-control ≈ V3" 即"resume 路线整体不有效"），便于 operator 优先准备对应行动。

### 4.3 给 V6 团队

V5 归因不阻塞 V6 pilot，但**V6 200K full 放行需 V5 归因闭环**。请把 P1-P4 排进 V6 Phase I 期间的并行任务，避免 Phase II 分歧点时数据不齐导致决策延误。

---

## 5. 修订后的可信 claim 集（供后续文档引用）

按证据强度从强到弱：

| 强度 | Claim | 支撑证据 |
|:-:|---|---|
| ★★★ 已证 | V3 200K 后段处于 image-dominated 状态（img_frac 87-96%） | V3 JSONL 直接观察 |
| ★★★ 已证 | null-control 在 V3 best 之后保持 V3 recipe 继续训练 14K 步，行为与 V3 后段一致 | null-control snapshot |
| ★★★ 已证 | rolling-val 不能作为 claim-level metric（2.4× 窗口噪声） | C6 |
| ★★★ 已证 | V5 rollout-heavy step 93200 的 rolling-val 改善是 window 回卷伪信号 | counter-review |
| ★★ 强假设 | V3 plateau 与 recipe / supervision-pressure 瓶颈一致 | V3 + null-control 联合证据 |
| ★★ 强假设 | 仅靠 LR continuation 不能打破 plateau | null-control 14K 步 |
| ★ 弱假设 | V3 plateau 的**唯一原因**是 recipe（排除 narrow basin / LR floor） | 待 V6 from scratch 出结果 |
| ★ 弱假设 | V5 rollout-heavy 整体失败 | 待 full-val sync |
| ★ 弱假设 | "加强 rollout"不是有效方向 | 待 V5 full-val + Path A |

V6 plan 当前主要论据应建立在 ★★ 及以上 claim 上；**不应**引用 ★ 级 claim 作为立项依据。
