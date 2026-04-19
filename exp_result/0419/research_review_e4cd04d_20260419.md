# Research Review (Detailed Re-pull Audit): `e4cd04d`

## 1. 审计范围与方法

### 1.1 审计对象
- 分支：`origin/foc_lite_hop0`
- 提交：`e4cd04d` (`fix: address Round 2 Codex review (3 high + 1 low)`)
- 审计日期：2026-04-19

### 1.2 核心关注点
1. 代码架构是否一致、是否引入新耦合。
2. 实验方案（D1 / T1）是否与总体主线（V5 的 C+A 叙事）一致。
3. 接口调用链（config -> train/eval -> model/loss）是否闭环、是否存在兼容性风险。

### 1.3 审计证据来源
- 代码 diff：`a6dafa5..e4cd04d`
- 方案文档：`PLAN_V5`、`PLAN_V6`
- 历史结果：`exp_result/0419/fullval_eval_AC_vs_C_20260419.md`
- 关键代码行号见后文“证据定位”。

---

## 2. 本次提交改动与修复闭环状态

## 2.1 本次提交的实际代码改动
- 改动文件仅 2 个：
  - `pet_lr/model_first_hop.py`
  - `train_first_hop.py`
- 变更规模：`36 insertions(+), 12 deletions(-)`。

## 2.2 与上一轮审查问题的对照

| 上轮问题 | 当前状态 | 结论 |
|---|---|---|
| GN 分组边界不稳 | 改为“<=8 的最大可整除分组” | 已修复 |
| `decode_crop` 无法显式 raw/refined 切换 | 新增 `apply_refiner` 参数 | 半修复（接口有了，调用链未落地） |
| resume missing key 过宽松 | 新增 missing-key 前缀白名单 | 部分修复 |
| 结构变更时 optimizer/scaler 继续恢复 | 行为未改 | 未修复 |

---

## 3. 代码架构审计（Architecture）

## 3.1 已改善项

### 3.1.1 GroupNorm 分组修复正确
- 位置：`pet_lr/model_first_hop.py:140-141`
- 逻辑：
  - `gn_groups = max(g for g in range(1, min(8, hidden_channels)+1) if hidden_channels % g == 0)`
- 影响：
  - 修复了此前 `min(8, hidden_channels)` 在 `hidden_channels=10/12` 等情况下可能触发的不可整除问题。

### 3.1.2 `decode_crop` 提供了可控开关
- 位置：`pet_lr/model_first_hop.py:508-536`
- 逻辑：
  - `apply_refiner=None` 时沿用默认行为（与历史兼容）。
  - `apply_refiner=False` 时可强制获得 raw decode（用于归因对照）。

## 3.2 未闭环项

### 3.2.1 训练与评估调用仍走默认路径，raw/refined 归因未真正落地（High）
- 训练（hop0 image loss）调用：
  - `train_first_hop.py:587` 仍是 `model.decode_crop(...)`（未显式传 `apply_refiner`）
- 验证链路调用：
  - `train_first_hop.py:837-840` 仍是 `model.decode_crop(...)`
- 独立 eval 调用：
  - `eval_first_hop_224_clip3.py:145` 仍是 `model.decode_crop(...)`
- 结果：
  - D1 使 decode 端后处理与 transport 误差在指标上耦合，当前输出仍无法直接回答“transport 本体是否改善”。

---

## 4. 方案设计审计（Design）

## 4.1 D1 方案（SeamRefiner）设计合理性

### 4.1.1 正向点
- D1 配置已启用：
  - refiner：`pet_flow_first_hop_224_50k_seam_refiner.yaml:264-269`
  - extended seam：`...:288-289`
  - seam weight 强化：`...:284`
- 这与“视觉伪影优先”目标一致（V6）。

### 4.1.2 关键风险（Medium）
- D1 的 `best_metric` 仍是 chain-MSE 导向：
  - `pet_flow_first_hop_224_50k_seam_refiner.yaml:61-71`
- 与 V6 中“视觉 seam 优先 gate”有目标错位：
  - V6 gate：`PLAN_V6...:95-100`
- 建议：
  - 至少并行保存 `best_chain.pt` 与 `best_seam.pt`，或引入 seam 目标项到 selection。

## 4.2 T1 方案（lambda sweep）设计合理性

### 4.2.1 一因子设计是干净的（Positive）
- T1a `lambda_max=0.18`：
  - `pet_flow_first_hop_224_50k_imgaux_lam0_18.yaml:143`
- T1b `lambda_max=0.25`：
  - `...lam0_25.yaml:143`
- 相比 Scheme C（0.12）仅更改该主因子（run_name 除外）。

### 4.2.2 执行策略建议
- 先跑 T1a，再按条件触发 T1b（防止高权重直接伤害 roll/pair）。

## 4.3 与总体架构（V5 主线）的一致性

### 4.3.1 当前叙事风险（Medium）
- V5 主线是 Scheme C + Scheme A（A 包含 C）。
- V6 明确写 D1/T1 基于 Scheme C：
  - `PLAN_V6...:70-73`
  - `PLAN_V6...:117`
- 若论文/报告口径写“在 A+C 上继续”，则需提供显式 A+C+D1 / A+C+T1 配置矩阵；否则会造成主线语义歧义。

### 4.3.2 A 的实证支撑仍偏弱
- 已有 full-val 对比显示 A+C 相对 C-only 提升较小：
  - `exp_result/0419/fullval_eval_AC_vs_C_20260419.md:22-23`
- 建议补至少 1 个额外 seed 再决定 A 是否作为后续唯一主线。

---

## 5. 接口调用链审计（API / Call Chain）

## 5.1 Config -> Loss -> Train 的闭环

### 5.1.1 Extended seam 的输入参数链路正确
- config：
  - `loss.image_aux.use_extended_seam`
  - `loss.image_aux.seam_zone_width`
- train 调用：
  - `train_first_hop.py:597-598`
- loss 实现：
  - `pet_lr/losses_first_hop.py:39-44`
- 结论：这条链路是闭合的。

## 5.2 Model decode 接口链路

### 5.2.1 新接口能力具备
- `decode_crop(..., apply_refiner: bool|None=None)`：
  - `pet_lr/model_first_hop.py:508-536`
- 但上层调用尚未显式使用该开关（见 3.2.1）。

## 5.3 Resume 兼容链路

### 5.3.1 模型参数层面
- `strict=False` + unknown missing hard-fail：
  - `train_first_hop.py:1352-1373`
- 相比前版本明显收紧（Positive）。

### 5.3.2 优化器/scaler层面（High）
- 仍是：
  - `optimizer.load_state_dict(...)` at `train_first_hop.py:1380`
  - `scaler.load_state_dict(...)` at `train_first_hop.py:1382`
- 结论：
  - 当“允许 missing keys”代表结构增量 warm-start 时，继续加载历史 optimizer/scaler 仍可能导致训练语义不一致或恢复失败。

---

## 6. 风险分级清单（最终）

## 6.1 High
1. raw/refined 归因在 train/eval 尚未落地（接口有但未用）。
2. 结构增量 resume 场景下 optimizer/scaler 恢复策略仍不安全。

## 6.2 Medium
1. D1 checkpoint 选择目标与“视觉优先 gate”不完全一致。
2. V6 的 C-based 配置与“若宣称 A+C 主线延续”之间存在叙事歧义。
3. A 的优势幅度仍偏小，主线地位应谨慎确认。

## 6.3 Low
1. GroupNorm 分组问题已修复，可从风险列表中降级关闭。

---

## 7. 最小可执行修复清单（按优先级）

## 7.1 P0（建议先改，低成本高价值）
1. 在 `eval_first_hop_224_clip3.py` 增加 `--decode-mode raw|refined|both`，默认 `both`。
2. 在 `train_first_hop.py` 验证阶段同时记录 raw/refined chain 指标（D20/D10/D4/NORMAL）。
3. 当 resume 发生“allowed missing keys”时，切换 warm-start：
   - 跳过 optimizer/scaler 恢复，
   - 重置 step/best-state（或显式策略化）。

## 7.2 P1（建议在下一轮实验前完成）
1. D1 增加 seam-oriented checkpoint 选择（或双 best）。
2. 统一报告口径：
   - transport headline 使用 `D20/D10/D4/NORMAL`，
   - D50 明确标记为 input-decode baseline，
   - D1 报告 raw/refined 同 checkpoint 对照。

## 7.3 P2（实验层面）
1. 如果继续宣称 A 是主线，补 1 seed full-val 复核 A+C vs C-only。
2. T1 执行顺序：先 0.18，再条件触发 0.25。

---

## 8. 结论（Verdict）

- **Verdict: Revise**
- 判定理由：
  - `e4cd04d` 对上轮问题有实质修复（尤其 GN 与 missing-key 白名单），代码可运行性提升明显。
  - 但“评估归因可解释性”和“resume 训练语义安全性”仍未完全闭环。
  - 在这两点补齐前，不建议直接把 D1/T1 大规模结果用于最终论证。

