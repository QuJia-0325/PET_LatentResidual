# 远程审计结果复核说明（GPT-operator）

日期：2026-04-24  
分支：`foc_lite_hop0`  
目的：给远程同事一份可直接阅读的复核说明，判断 `review/0424` 这批远程审计结论哪些合理、哪些过强、哪些尚未落地到当前代码。

---

## 1. 一句话结论

这批远程审计**方向上大体合理**：它正确地把主问题收敛到 transport 误差，而不是 decoder 或 ODE 积分；也正确指出在当前 `36.12-36.21 dB` 的窄差距下，需要先补 variance / causal 诊断，再谈机制性结论。  

但它**还不能被当成“已被代码和实验完全支持的最终结论”**。原因有三类：

1. 文档之间对 `200K` 的执行顺序存在直接冲突。  
2. 部分建议还没有在代码里落地，当前仓库状态下并不能直接执行。  
3. 少数结论表述过强，超出了现有证据能够严格支持的范围。

---

## 2. 我认为合理的部分

### 2.1 把主瓶颈重新定位到 transport，是合理的

`review/0416/conclusion.md` 已经给出 full-val 证据：

- E1: `Gap_Transport` 占总 gap 的 `96-99%`
- E2: half-step vs full-step 差异只有 `0.1709%`
- Oracle: GT latent decode ceiling 在 `46-52 dB`

对应文档：

- `review/0416/conclusion.md`
- `review/0424/consensus_agreement.md:84-89`
- `review/0424/claude-reviewer/breakthrough_analysis_10dB_gap.md`

这条诊断链本身是自洽的：  
`transport-side dominance -> ODE 不是主因 -> decoder 不是主因 -> 继续区分 exposure vs capacity`

### 2.2 把 Path A 作为高信息价值诊断，是合理的

`scripts/diagnose_tf_rollout_gap.py` 的核心设计是：

- 对每一跳同时计算 `teacher-forced` 和 `rollout-fed` 的 1-step 预测；
- 比较 `PSNR_TF - PSNR_RO` 来估 exposure bias；
- 比较 `PSNR_ceiling - PSNR_TF` 来估 intrinsic capacity gap。

这比继续做 hop0 局部权重小修小补更有信息价值，方向上我认同。

对应实现：

- `scripts/diagnose_tf_rollout_gap.py:301-333`
- `review/0424/consensus_agreement.md:40-44`

### 2.3 F2 叙事修正是合理的

把“decoder 非线性放大 tail”改写成“`Gap_Transport` 单调递增，因此同比例改善在 dB 上表现为 tail 更大”，这是更稳妥、更符合现有证据的说法。

对应文档：

- `review/0424/consensus_agreement.md:52-56`

### 2.4 用 `sigma seed` 约束小增益叙事，是合理的方法论提醒

当 effect size 只有 `0.08-0.2 dB` 时，先补同配置多 seed 波动认识，再谈“机制有效 / 训练更久有效”，这个方向是对的。  
尤其是当前已有结果差异本身就很小，这类提醒是必要的。

对应文档：

- `review/0424/consensus_agreement.md:25-38`
- `review/0424/claude-reviewer/06_reasoning_dossier_on_rebuttal.md:126-130`

---

## 3. 我认为不够稳或尚未落地的部分

### 3.1 最大问题：0424 文档内部对 `200K` 是否应立即启动存在冲突

`consensus_agreement.md` 明确写的是：

- `必须在 σ_seed 结果出来后才能启动`  
  见 `review/0424/consensus_agreement.md:32-38`

但 `investigation_and_plan.md` 又明确写：

- `如果 v3-50K 跑完后 GPU 空闲 -> 立即启动 200K`  
  见 `review/0424/investigation_and_plan.md:166-168`

而当前仓库里的 `200K` config 也只是一个“把 50K v3 拉长到 200K”的配置，并没有把文档里要求的 `σ-unit` stop/pass protocol 写进任何配置或训练逻辑：

- `configs/pet_flow/pet_flow_first_hop_224_200k_transport_v3.yaml:4-10`
- `configs/pet_flow/pet_flow_first_hop_224_200k_transport_v3.yaml:45-105`

这说明远程审计目前更像“意见集合”，还不是单一一致的执行规范。

### 3.2 `sigma seed` 共识没有真正落地到当前仓库

共识文档要求：

- 4 个 seed：`42/123/456/789`
- 使用统一命名：`sigma_s42/s123/s456/s789`
- 明确指出旧的 `sigma_seed123/456` 需要删除或替换

对应：

- `review/0424/consensus_agreement.md:95-103`

但当前仓库实际只有：

- `configs/pet_flow/pet_flow_first_hop_224_50k_sigma_seed123.yaml`
- `configs/pet_flow/pet_flow_first_hop_224_50k_sigma_seed456.yaml`

而且没有 `s42`、`s789`，命名也仍是文档点名要替换的旧格式。

因此，`σ_seed 是当前最高优先级` 这个结论在文本上成立，但在仓库里**并没有执行完**。

### 3.3 Path C 目前在代码层面不可执行

共识文档要求给 decoder 增加 partial unfreeze 支持：

- `review/0424/consensus_agreement.md:46-50`
- `review/0424/consensus_agreement.md:105-110`

但当前训练器明确禁止 first-hop trainer 解冻 RAE：

- `train_first_hop.py:1055-1059`

所以 Path C 现在还只是文档建议，不是可以直接启动的实验。

### 3.4 Path A 的 `NEAR_CEILING` 分支推断过强

脚本当前的判决逻辑是：

- 如果 downstream hops 上 `mean_exposure < near_zero`
- 且 `mean_ceiling < 2.0`
- 就给出 `NEAR_CEILING`，并进一步推断“剩余 gap 是 off-manifold decoder non-linearity”

对应代码：

- `scripts/diagnose_tf_rollout_gap.py:362-391`

这个推断跳得太快。  
原因是：单步 teacher-forced 接近 ceiling，只能说明**单步**误差小；但四跳链式 rollout 仍可能因为每一步的小误差累积，在 end-to-end 上形成大的 PSNR deficit。  

因此 Path A 是值得跑的强诊断，但它还不够单独支撑“decoder / off-manifold 是主因”的结论。

### 3.5 “sigma 是所有 PSNR 声明的全局校准”说法偏强

这句话作为方法论提醒是有价值的，但如果按严格统计表述，它说得太满。  
远程文档自己也承认：

1. `σ_seed` 可能随训练长度变化而非稳定常数；  
2. `n=3` 只能给 point estimate，不能给稳健 CI；  
3. slice bootstrap 可能高估独立性。

对应：

- `review/0424/claude-reviewer/06_reasoning_dossier_on_rebuttal.md:126-130`

所以更准确的表述应该是：  
`σ_seed` 是当前项目很合理、很必要的实验治理信号，但不能无条件外推成“所有配置、所有训练长度”的统一 noise floor。

---

## 4. 远程同事应如何解读这批审计

建议按下面的方式阅读，而不要把它当成已经闭环的“最终 verdict”：

1. 把它当成**当前最强的决策框架**，不是最终被代码完全验证的结论。
2. 认可它对主问题的聚焦：现在最值得做的是 `variance calibration + Path A`，而不是继续堆 hop0 小改动。
3. 不要直接把 `200K` 的任何结果写成强机制结论，除非 `σ_seed` 或等价的 variance 校准先补齐。
4. Path C 目前还不能执行，除非先改训练器。
5. Path A 值得优先推进，但它的 verdict 应作为诊断输入，而不是自动当成最终因果定性。

---

## 5. 我的最终判断

**总评：70/100，方向正确，执行未闭环。**

更具体地说：

- **合理**：问题重新定位、Path A 优先级、F2 叙事修正、对小效应的 variance 敏感性。
- **不够稳**：把 `σ_seed` 说成全局统一标尺、把 Path A 某些分支解释得过满。
- **未落地**：4-seed sigma protocol、Path C partial unfreeze、200K 的 sigma-unit gate。

因此，这批远程审计**值得认真采纳为下一阶段实验框架**，但不应被表述为“我们已经通过审计证明了某个机制结论”。

