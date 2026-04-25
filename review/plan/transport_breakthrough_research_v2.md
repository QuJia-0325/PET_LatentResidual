# Transport 突破方向调研与 Idea 设计 — v2

**Date**: 2026-04-24  
**Supersedes**: `transport_breakthrough_research.md` (v1)  
**v2 修订原则**: 方法优先 — 先拿到有效方法再做统计验证；删除 v1 中被证明是 "GPU 浪费" 的前置 σ_seed

---

## 1. v1 → v2 主要变更

| 项目 | v1 | v2 | 理由 |
|---|---|---|---|
| σ_seed 实验 | Day 1-3 前置（4 GPU-day） | **后置到方法验证后**（2 GPU-day，仅 best config） | 目标效应 > 1 dB 时 σ_seed 不影响决策，仅用于论文表格 CI |
| 外部 baseline | 延后到投稿准备 | **与 Phase 2 并行**（1 GPU-day） | 避免"方法成功但无参照"的投稿硬伤 |
| Path A 诊断 | 0.5 GPU-day | 保留 0.5 GPU-day + **梯度范数 logging** | 0 成本验证 Idea 1 的核心前提 |
| Mediation 实验 | 未承诺 | **加入**（复用 Path A 脚本，0.2 GPU-day 增量） | 唯一能救 hop0 因果故事的低成本实验 |
| Idea 1 技术设计 | 从 GT D50 开始 pre-rollout | **增加 alpha ramp + hop0-specific 分支 + velocity target 修正** | 修补 v1 的 4 个技术漏洞 |
| Idea 3 | 列为 VELOCITY_CAPACITY 路径 | **降级为长期研究，不在本投稿周期** | Nightmare KILL §K2（架构迭代是浪费） |
| 成功门槛 | "Δ > 3 dB 就投稿" | **Δ > 5σ_seed AND 有 baseline AND reframe 叙事** | 避免 rebuttal v1 同款陷阱 |

---

## 2. Transport 机制根因分析（核心）

### 2.1 当前 loss 结构的精确描述

从 [train_first_hop.py](../../../train_first_hop.py) L303 和 [rollout_first_hop.py](../../../pet_lr/rollout_first_hop.py) L80-105 可精确读出：

**pair_loss**（权重 **2.0**：velocity_weight=1.0 + endpoint_weight=1.0）：
```
v_target = (z_GT_dst - z_GT_src) / dt / sigma_hop
vel_err  = MSE(v_pred(z_GT_src, t_src, t_dst, hop), v_target)
end_err  = MSE(z_GT_src + v_pred·σ·dt, z_GT_dst)
```

**rollout_loss**（权重 **0.02 → 0.25** ramp，200K v3 的实际 lambda_end=0.25）：
```
for hop in 0..3:
    v_pred = backbone(z_curr, ...)         # z_curr 是 pred 或 mixed
    z_next_pred = z_curr + v_pred·σ·dt
    step_loss = MSE(z_next_pred, z_GT_next)
    z_curr = mix_latent(z_GT_next, z_next_pred, alpha)  # alpha: 0→1 ramp
```

**关键比例**（从 v3 config 计算）：
- pair_loss 有效权重 ≈ 2.0（velocity 1.0 + endpoint 1.0）
- rollout_loss 稳态权重 ≈ 0.25
- **pair_loss 占 backbone 梯度 ≈ 2.0 / 2.25 ≈ 88%**

v1 里的"70-80%"估计偏保守，实际更严重。这**完全确认了 Idea 1 的前提**：backbone 梯度的大头全部在 GT-to-GT 分布上训练。

### 2.2 36.2 dB Plateau 的机制根因（我的判断）

**根因 A — "无纠错 velocity 训练"（最可能）**

pair_loss 的 velocity target：
$$v_{\text{target}}^{\text{pair}} = \frac{z^{\text{GT}}_{\text{dst}} - z^{\text{GT}}_{\text{src}}}{dt \cdot \sigma_{\text{hop}}}$$

注意：**z_src 永远是 GT**。这意味着 backbone 学到的 v(z, t) 只在 GT 轨迹的 ε-邻域定义良好。一旦推理时 z_pred 偏离 GT 轨迹，backbone 没有被训练过"从偏离状态回到 GT 轨迹"的 velocity。

- 推理需要的是 "corrective velocity"：$v^{\text{correct}}(z_{\text{pred}}) = (z^{\text{GT}}_{\text{dst}} - z_{\text{pred}})/(dt \cdot \sigma)$
- pair_loss 训练的 velocity 是 $v^{\text{GT}}(z^{\text{GT}}_{\text{src}})$
- 这两个是**不同函数**，在 OOD 输入上差异任意大

**证据**：pair_loss ≈ 0（单步 GT→GT 精准），但 rollout_loss 不降 → backbone 在 GT 邻域完美，在 pred 分布上漂移。这是 exposure bias 的教科书症状。

**根因 B — rollout loss 信号被稀释（并发因素）**

rollout_loss 权重只有 pair_loss 的 12%。即使 rollout_loss 在 alpha=1 时提供了一些"implicit self-forcing"信号（z_curr 是 pred，target 是 GT，梯度经 z_next_pred 流回），其梯度贡献被 pair_loss 压制。

**证据**：调整 rollout step_weights [1.00→1.30, ...] 和 lambda_end（v3 的修正）未改变 plateau → 信号量不足，不是方向问题。

**根因 C — sigma normalize 的分布错位（二阶因素）**

$\sigma_{\text{hop}}$ = per-hop std of **GT-to-GT velocities**。但推理时 $(z^{\text{GT}}_{\text{dst}} - z_{\text{pred}})$ 的 scale 比 GT-to-GT 大（因为含漂移）。backbone 需要输出的归一化 velocity 其实 > 1，但训练分布集中在 ≈ 1，属于 OOD。

**证据**：velocity_rebalance（v3）启用后无效 → 不是 loss scale 问题，是 backbone 输出分布问题。

### 2.3 为什么 rollout alpha=1 没解决 exposure bias

表面上看，rollout 在 alpha=1 时已经是纯 self-rollout，似乎应该等价于 self-forcing。但有三个原因让它不够：

1. **梯度比例**：88% 梯度来自 pair（GT 分布），12% 来自 rollout（pred 分布）。backbone 的参数优化**由 pair 主导**。
2. **straight-through 的副作用**：当 α < 1 时，forward 用 mixed latent，backward 只流经 z_pred。这让 z_curr 的 forward 值比 pure-pred 更接近 GT，降低了 exposure bias 的曝光强度。
3. **单次前向**：rollout 每步只做一次 backbone forward，但 pair 在同一 step 也有一次 forward（不同 batch）。两个 forward 的梯度同时影响参数，pair 的"回拉"力很强。

**结论**：要真正解决 exposure bias，必须让 **pair_loss 本身**在 pred 分布上训练。这就是 Idea 1 的核心动机。

---

## 3. Idea 1 v2 — Self-Forced Pair Loss（核心方法）

### 3.1 方法设计（修复 v1 的 4 个技术漏洞）

**v1 漏洞 vs v2 修复**：

| v1 漏洞 | v2 修复 |
|---|---|
| hop0 的 z_src 始终是 GT D50，Self-Forcing 对 hop0 无效 | 引入 `hop0_injection_mode` 开关：`gt`（默认）\| `noise`（D50 latent + 可控噪声） |
| velocity target 跳变风险（z_src 变 pred 时 target norm 爆炸） | `alpha_sf`: 0→1 ramp，前 10K 步混合 GT/pred 作为 z_src |
| Phase 2 耗时低估 | 显式承认 +30% 训练时间，预算 6 GPU-day（不是 4） |
| 文献引用未核实 | 基于 [Self-Forcing (Huang et al. 2506.08009)](https://arxiv.org/abs/2506.08009) + [Rolling Forcing (Liu et al. 2509.25161)](https://arxiv.org/abs/2509.25161) 的 pair-level 推广，v2 明确标注这是我们的扩展而非直接套用 |

### 3.2 算法（精确版）

```python
# 在每个 training step，loss 构造前：
global_step = ...
alpha_sf = get_linear_schedule(global_step,
                                warmup=100_000,   # Phase 1 完全不用 SF
                                ramp=10_000,      # Phase 2 前 10K 做 0→1 ramp
                                start=0.0, end=1.0)

if alpha_sf > 0:
    # 1. Detached pre-rollout 从 GT D50 出发生成 z_pred 链
    with torch.no_grad():
        z_chain_pred = [batch["z_rollout"][:, 0]]  # z_D50 (GT 起点)
        for h in range(4):
            out_ro = model.predict_latent_step(
                z_chain_pred[-1], t_src[h], t_dst[h], hop=h,
                x_src_img=batch["x_rollout_first"] if h == 0 else None)
            z_chain_pred.append(out_ro["z_pred"].detach())

    # 2. 对 pair batch 的每个样本，按其 hop_idx 选择对应的 z_pred
    hop = batch["hop_idx"]
    z_src_pred = torch.stack([z_chain_pred[h] for h in hop.tolist()])  # [B, C, H, W]

    # 3. Alpha ramp 混合 GT 和 pred 作为 pair z_src（关键防跳变）
    batch["z_src"] = (1 - alpha_sf) * batch["z_src"] + alpha_sf * z_src_pred

# 4. 正常走 compute_pair_losses —— 
#    v_target = (z_GT_dst - z_src_mixed) / dt / sigma 自动变成 "纠错 velocity"
#    不需要改 compute_pair_losses 本身
```

### 3.3 关键实现细节

**细节 1：velocity target 自动正确**  
因为 `compute_pair_losses` 里 `v_target_raw = (z_dst - z_src) / dt`，我们只需替换 `batch["z_src"]`，target 自动变成 "从 pred 到 GT" 的纠错 velocity。**无需修改 loss 函数**。

**细节 2：sigma 处理**  
sigma 继续用原来的 GT-to-GT 统计量。因为 `(z_GT_dst - z_src_pred)/dt` 的 scale 比 GT-to-GT 大，backbone 需要学会输出 > 1 的归一化 velocity。这是**我们期望发生的事情**——让 backbone 扩展训练分布。  
可选：Phase 2 中重新估计 sigma（online moving average of new v_target norms），但增加复杂度，建议先不做。

**细节 3：梯度范数 logging**（Phase 1 就加）  
在 `compute_pair_losses` 和 `compute_rollout_losses` 返回后，分别计算：
```python
g_pair = torch.autograd.grad(pair_total, model.backbone.parameters(), retain_graph=True)
g_roll = torch.autograd.grad(roll_total, model.backbone.parameters(), retain_graph=True)
log({
    "grad_norm/pair": sum(g.norm()**2 for g in g_pair).sqrt().item(),
    "grad_norm/rollout": sum(g.norm()**2 for g in g_roll).sqrt().item(),
    "grad_ratio/pair_over_total": ...,
})
```
10 行代码，零训练成本。**Phase 1 收集 1K 步数据**即可确认我的 88% 估计。

**细节 4：显存与速度**  
- pre-rollout 在 `torch.no_grad()` 下，显存增加 ≈ 4 × activation 存储 ≈ 15%
- 速度：+4 次 forward / step ≈ +25-30% 训练时间
- 200K 步 Phase 2 ≈ 64h × 1.3 ≈ 83h ≈ **3.5 GPU-day**（100K 步 Phase 2）

### 3.4 Ablation 配对（投稿必需）

| Config | pair z_src | 目的 |
|---|---|---|
| **SF-pair-full**（主） | α_sf=1.0 after 100K | 完整 Self-Forcing |
| **SF-pair-partial**（对照） | α_sf=0.5 after 100K | 验证 ramp 的重要性 |
| **Phase 1 only**（baseline） | α_sf=0 | Plateau 参照 |
| **rollout-up**（对照） | α_sf=0, rollout λ_end=1.0 | 只调 rollout 权重能否达到同等效果？ |

SF-pair-full vs rollout-up 的对比是**方法论贡献的核心 ablation**：证明"self-forcing 必须施加在主梯度路径上（pair），仅靠放大 rollout 权重不够"。

---

## 4. Idea 2 — Stochastic Hop Image Loss（保留，与 Idea 1 叠加）

与 v1 相同，但明确：
- 只在 Phase 2 叠加（Phase 1 先单独跑 SF-pair 验证）
- 需要 `train_include_full_x_rollout: true`（RAM 220→230 GB，仍可承受）
- ~30 行代码

---

## 5. 修正后的执行时间线

```
Day 1-5  (Phase 1 收尾 + 并行诊断)
├─ 200K v3 继续跑 (已在进行, 42K/200K → Day 5 达 100K)
├─ [Day 1] Path A 诊断 + Mediation (0.7 GPU-day, 并行)
├─ [Day 1] 加入 grad_norm logging 到 200K v3（热插入）
└─ [Day 3] 验收 grad_norm 数据：确认 pair 占比 > 70%

Day 5-6  (决策与实施)
├─ 评估 Phase 1 @ 100K: 是否已突破 plateau？
├─ Path A 结果：exposure bias 是否主导？
├─ 实施 SF-pair 代码 (~25 行) + 梯度 logging
└─ 提交 ablation configs

Day 6-10  (Phase 2 主实验)
├─ SF-pair-full (100K 步, 3.5 GPU-day)
├─ SF-pair-partial (100K 步, 3.5 GPU-day) — 可与 full 并行或串行
├─ rollout-up (100K 步, 3.5 GPU-day) — 反驳 "只加 rollout 就够了"
└─ Phase 1 only 继续到 200K 作为 baseline

Day 10-12  (验证与 baseline)
├─ Full-val eval on all 4 configs + paired bootstrap CI
├─ σ_seed on best config (2 seeds 追加 → 3 seeds × best, ~2 GPU-day)
├─ Pix2Pix baseline setup & training (~1 GPU-day)
└─ 汇总：best config + CI + 外部参照

Day 12-14  (论文准备)
├─ 如果 Δ > 5·σ_seed AND baseline 合理：reframe 叙事 + draft workshop paper
└─ 如果 Δ < 1 dB：回看机制，考虑 Idea 2 / 诊断框架路线
```

**总预算**：~14 GPU-day（v1 是 9，多的 5 天分配给：3 个 ablation + σ_seed + baseline）

**关键决策门**：

| Checkpoint | 条件 | 动作 |
|---|---|---|
| Day 3 | grad_ratio pair > 70% | ✅ SF-pair 前提成立，继续 |
|  | grad_ratio pair < 50% | ⚠️ 调整思路：主矛盾在 rollout 权重，不是 pair |
| Day 5 | Path A exposure_gap > 0.3 dB | ✅ Self-Forcing 对路 |
|  | Path A exposure_gap < 0.1 dB | ⚠️ Velocity capacity 问题，Phase 2 可能无效；考虑 Idea 2 单跑 |
| Day 5 | Phase 1 @ 100K 已 > 36.5 dB | 🎉 plateau 已破，Phase 2 重新评估必要性 |
| Day 10 | SF-pair Δ > 1.5 dB vs Phase 1 | ✅ 方法成立，走投稿准备 |
|  | SF-pair Δ < 0.3 dB vs rollout-up | ⚠️ 方法贡献不可辨，需要 reframe |

---

## 6. 成功/失败预案

### 6.1 理想场景（~30% 概率）
- Path A 显示 exposure_gap ≥ 0.5 dB
- SF-pair 在 NORMAL 达 40-42 dB（+3-5 dB）
- σ_seed < 0.1 dB，Pix2Pix baseline 在 37-38 dB
- **故事**："We identify exposure bias in latent multi-hop transport and propose self-forced pair loss. +X dB over plateau, Y% over Pix2Pix."
- **目标**: MIDL / MICCAI 2027 workshop 主提交

### 6.2 部分成功（~40% 概率）
- SF-pair Δ ≈ 1-2 dB，显著但不惊艳
- **故事**：Reframe 为 "systematic diagnosis of cascade latent transport"，SF-pair 是诊断框架下的一个代表性修复
- **目标**: Workshop 负结果/诊断性论文

### 6.3 失败（~30% 概率）
- SF-pair Δ < 0.5 dB，或与 rollout-up 不可区分
- **Pivot 路径**：
  - Idea 2 单独验证（stochastic hop image loss）
  - Decoder FT ceiling 实验（Nightmare §5 Path C）
  - 叙事 reframe 为 "36.2 dB 是 paradigm ceiling" 纯负结果

---

## 7. 相对 v1 的核心哲学修正

v1 的问题是把 Nightmare 的元规则（σ_seed 必跑）僵化套用到"目标效应 >> 可能 σ_seed"的阶段。v2 的排序原则：

> **当目标效应量 Δ_target 远大于假设 σ_seed 时，优先追方法；只在方法验证完成后做 σ_seed 给论文加 CI。当 Δ_target ≈ σ_seed 时，σ_seed 必须前置。**

**决策树**：
```
Δ_target (目标效应) > 5 × σ_seed_prior ?
├─ Yes → 先做方法，后做 σ_seed（本 v2 路径）
└─ No  → 先做 σ_seed，再决定是否还要追方法（Nightmare 原路径）
```

σ_seed_prior 的先验估计：5 configs 在 0.029 dB 跨度内 → σ_seed ≤ 0.1 dB 是合理上界。SF-pair 目标 +2-5 dB = 20-50 × σ_seed_prior → 方法优先合理。

---

## 8. 可证伪预测（按 Nightmare §8.3 元规则预登记）

在 Phase 2 启动前锁定以下预测：

1. **SF-pair-full Δ_transport_avg > 1.5 dB** vs Phase 1 only @ 200K  
   证伪条件：Δ < 0.5 dB → 方法无效，回看机制

2. **SF-pair-full - rollout-up > 0.8 dB**  
   证伪条件：两者在 2σ 内无差 → "pair 主梯度"假设错误，只放大 rollout 即可

3. **grad_ratio (pair/total) at 100K ∈ [70%, 92%]**  
   证伪条件：< 50% → 基础机制分析错误，Idea 1 前提不成立

4. **Path A exposure_gap ∈ [0.2, 2.0] dB**  
   证伪条件：< 0.1 dB → exposure bias 不是主因，capacity 才是

这些预测写入 git commit message 并保留，Phase 2 结果出来后逐条核对。

---

## 9. 一句话判决

> v2 保留 v1 的核心洞察（pair_loss exposure bias 是 plateau 根因），但**彻底修正优先级**（方法优先，σ_seed 后置）、**补齐 4 个技术漏洞**（hop0 分支、velocity target 跳变、耗时估计、文献对齐）、**加入 3 个决策门和 4 个可证伪预测**，预算 14 GPU-day。核心 ablation 是 SF-pair vs rollout-up，证明 self-forcing 必须施加在主梯度路径（pair）上。
