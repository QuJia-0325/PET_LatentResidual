# Research Review: `565c6b0` (Code Architecture / Scheme Design / API Call Chain)

Date: 2026-04-19  
Reviewer mode: `research-review` (local code audit + external xhigh critic)  
Target: `origin/foc_lite_hop0@565c6b0`  
Scope: code architecture, design consistency (D1/T1a/T1b), interface call-chain closure

---

## 0) Executive Summary

Verdict: **Revise**

`565c6b0` 对上轮问题做了“部分落地”：
1. 在 trainer 的 val 里加了 raw/refined 双轨观测（但只覆盖 D20/NORMAL）。
2. 在架构变更 resume 时跳过 optimizer/scaler restore（但 warm-start 语义仍不完整）。

当前结论：
1. `T1a/T1b` 可以继续跑。
2. `D1` 可以探索性跑，但不建议直接作为“正式可汇报结论 run”。
3. 要把 D1 变成可稳健宣称的结果，需要先补 P0 修复包（见第 6 节）。

---

## 1) Evidence Baseline

## 1.1 审计方法

1. 重新拉取远端，确认 `foc_lite_hop0` 最新为 `565c6b0`。
2. 建立隔离 worktree：`/tmp/petlr_foc_lite_audit`。
3. 静态编译检查通过：`train_first_hop.py`, `eval_first_hop_224_clip3.py`, `pet_lr/*_first_hop.py`。
4. 代码差分确认：
   - `e4cd04d..565c6b0` 仅改动 `train_first_hop.py`。
5. 设计一致性检查：
   - D1 配置相对 Scheme C 的增量
   - T1a/T1b 相对 Scheme C 的受控变量
6. 外部 xhigh 审稿器交叉审阅（agent: `019da5db-47f4-73d3-a2cd-24629b331b3e`）。

## 1.2 本次 commit 的实际改动闭包

`565c6b0` 仅改了两件事：
1. `evaluate()` 增加 `val_chain_d20_raw_mse`、`val_chain_normal_raw_mse`（仅在 refiner 开启时计算）。
2. resume 检测到允许的 missing keys 时，跳过 optimizer/scaler 恢复。

未改变：
1. `eval_first_hop_224_clip3.py` 的输出协议（仍是单路 decode 输出）。
2. best checkpoint 的目标函数定义（仍是 chain MSE 体系）。
3. D1 文档 gate 对应字段（仍与“可自动导出字段”存在断层）。

---

## 2) Architecture & Call-Chain Audit

## 2.1 训练路径（主链）

训练步核心调用链：
1. `main_batch -> model.predict_latent_step()`  
2. `compute_pair_losses()`  
3. `compute_rollout_losses()`  
4. `hop0_batch -> compute_hop0_image_losses()`  
5. （可选）`compute_foc_losses()`  
6. （可选）alignment loss（`align_proj` vs `z_dst`）  
7. 总损失反传

关键事实：
1. `compute_hop0_image_losses()` 内部调用 `decode_crop(out["z_pred"])`，使用默认行为。  
2. 当 D1 开启时，默认 decode 路径就是 refined 路径。  
3. 这意味着 D1 并不是“只在推理阶段后处理”，而是直接进入训练损失路径。

## 2.2 验证路径（trainer 内 evaluate）

当前行为：
1. 链路 MSE 默认走 refined decode。  
2. 仅在 `model.seam_refiner_enabled` 时，额外记录：
   - `val_chain_d20_raw_mse`
   - `val_chain_normal_raw_mse`
3. D10/D4 没有 raw 指标。

结论：
1. raw/refined 归因已经“开始接线”，但证据链尚未闭环。

## 2.3 离线 full-val 脚本路径

`eval_first_hop_224_clip3.py` 目前仍是：
1. `x_pred = model.decode_crop(...)`（单路，默认行为）
2. 输出只有单路 PSNR（没有 raw/refined 双路字段）

结论：
1. D1 的“before/after refiner”量化对照，不能通过当前官方评估脚本自动导出。

---

## 3) Scheme Design Consistency Audit (D1 / T1a / T1b)

## 3.1 T1a/T1b 设计一致性

`imgaux_lam0_18.yaml` / `imgaux_lam0_25.yaml` 与 `imgaux_boost.yaml` 相比，仅有两处差异：
1. `run_name`
2. `training.image_aux.lambda_max`

结论：T1a/T1b 是干净的配置消融，设计成立。

## 3.2 D1 设计一致性

`seam_refiner.yaml` 相对 Scheme C 的关键增量：
1. 开启 `first_hop.seam_refiner.enabled=true`
2. 提高 `loss.image_aux.seam_weight`
3. 开启 `use_extended_seam=true` + `seam_zone_width=3`

结论：D1 代码和配置是连通的，模块可训练、损失可生效。

## 3.3 “D1 与 T1 完全独立”表述审计

更准确表述应为：
1. **可并行执行**：是。  
2. **严格正交**：否。  
原因：D1 会改变 `decode_crop` 默认路径并作用于 image_aux 训练对象，而 T1 正在扫 image_aux 权重。

---

## 4) Findings (Severity Ordered)

## High

1. 架构变更 resume 仍非完整 warm-start
   - 证据：
     - 检测 `_arch_changed` 后仅跳过 optimizer/scaler restore。
     - 仍从 checkpoint 读取 `start_step`，并沿用已训练步上下文。
   - 风险：
     - 新模块随机初始化，却被带入后期学习率和损失调度区间。
     - 若 ckpt step 接近 `max_steps`，可能直接退出或有效训练量不足。
   - 影响：高，直接影响实验有效性和可复现解释。

2. D1 的“训练选优 -> full-val 评估 -> 结论”链条未闭环
   - 证据：
     - best 仍按 chain-MSE 选择。
     - full-val 脚本无 raw/refined 双路导出。
     - PLAN 的视觉 gate 字段无法直接由当前官方脚本产出。
   - 风险：
     - `best.pt` 不一定对应“seam 最优点”，结论支持不足。
   - 影响：高，直接影响论文/汇报可信度。

## Medium

3. raw/refined 归因只覆盖 D20/NORMAL，缺 D10/D4  
4. 非 D1 run 的 raw 指标语义不清（`0.0` 易误读为真实值）

## Low

5. PLAN 文案“完全独立”过强，建议改为“可并行但共享 image_aux 路径”

---

## 5) Interface Contract Checklist

当前接口闭包状态：
1. 数据 -> 训练：闭合  
2. 模型 -> 训练损失：闭合  
3. 模型 -> trainer 验证：部分闭合（raw 仅部分 hop）  
4. trainer 验证 -> offline full-val：**未闭合**（字段协议不一致）  
5. offline full-val -> PLAN gate：**未闭合**（D1 gate 所需字段未标准导出）

---

## 6) Minimal Fix Package (Actionable)

## P0 (必须先做，才能把 D1 结论定为正式)

1. 完整 warm-start 语义（for `_arch_changed=True`）
   - 强制：
     - `start_step = 0`
     - `best_val = inf`
     - `best_metric_name_for_ckpt = current config metric`
   - 不恢复 `rng_state`
   - 日志明确标识为 `warm-start`（不是 resume-continuation）

2. D1 评估协议闭环
   - 在 `eval_first_hop_224_clip3.py` 增加双路导出：
     - `psnr_raw_<tp>`
     - `psnr_refined_<tp>`
   - 建议补一个 seam 指标导出（至少 hop0 / chain 关键时间点）
   - 建议增加 D1-aware ckpt：
     - `best_d1.pt` 或显式加入 seam 相关项的 best objective

## P1

1. trainer evaluate 的 raw 指标补齐 D10/D4  
2. 非 refiner run 的 raw 字段输出 `NaN/null` 或不输出该 key

## P2

1. PLAN_V6 文案和 gate 字段与代码导出字段对齐

---

## 7) Run Decision Matrix (Current State)

1. `T1a`（lambda=0.18）：**可以开跑**  
2. `T1b`（lambda=0.25）：**可以开跑**  
3. `D1`（seam refiner）：
   - exploratory: **可以**
   - formal claim run: **不建议**（先做 P0）

建议策略：
1. 先继续 T1a/T1b 收集趋势。
2. 并行实现 P0（warm-start + eval 协议闭环）。
3. 之后重启 D1 正式 run 并以新协议评估。

---

## 8) Final Verdict

`565c6b0` 质量状态：**Revise**  
结论解释：修复方向正确，但关键高风险路径仍未完全闭合，尤其是 D1 的“可宣称证据链”与架构变更 warm-start 语义。
