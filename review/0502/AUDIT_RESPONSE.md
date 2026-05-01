# review/0502 v2 修订记录 — 对外部 agent 审计的逐条回应

> **⚠️ DEPRECATED**：本文件记录 **v2 修订意见**，其中"A 与 B 在 loss_total 上会差 ~25%"的结论已被 v3 推翻。v3 探明原因是 rollout 加权 *平均*分母不同，并引入 `preserve_v6_sum` normalizer 使 A==B 严格数学等价（FP64 残差 ~1e-20）。v4 又修正了 v3 遗留的 path_guard 路由与 sanity comparator 阈值问题。
>
> 最新决策请看：[AUDIT_RESPONSE_v4.md](AUDIT_RESPONSE_v4.md) 【当前活跃】 · [AUDIT_RESPONSE_v3.md](AUDIT_RESPONSE_v3.md) 【deprecated】
>
> 本文件仅作历史记录保留。

---

> 外部 agent 对 v1 提交（README + PLAN + 4 yaml + 2 patches + 3 scripts）做了完整审查，提出 6 个具体问题 + 3 项缺失。本文档记录每条意见与修订。

---

## 🔴 问题 1（关键）：A vs B 的 sanity check 容差判定有 bug

**审计原文**：
> "数学保证 A ≡ B（在 SGD 梯度层面）" 但实际 `loss_total` 会差 ~25%——A 的归一化分母 = Σw_j = 5.0，B 的分母 = Σ(w_j·ρ_j/median(ρ)) = 6.62，比值 0.755 → B 的 loss_total 是 A 的 ~75%。如果 sanity check 真按 5% 阈值判，会全部 fail。

**验证**：

直接看 [`pet_lr/rollout_first_hop.py`](../../pet_lr/rollout_first_hop.py#L107) 末尾：

```python
total = (stacked * w).sum() / w.sum().clamp_min(1e-8)
```

这是 **加权平均**，不是 weighted sum。所以：

* A：分子 = $\sum_j w_j \cdot \rho_j \|\delta v_j\|^2$，分母 = $\sum_j w_j$ = **5.0**
* B：分子 = $\sum_j (w_j n_j) \cdot (\rho_j/n_j) \|\delta v_j\|^2 = \sum_j w_j \rho_j \|\delta v_j\|^2$（与 A 完全相同），分母 = $\sum_j w_j n_j$ = **6.6237**

→ `loss_total_B / loss_total_A = 5.0 / 6.6237 ≈ 0.7549`，差 **24.5%**。审计完全正确。

**这意味着**：

* SGD/AdamW 看到的 effective gradient 也差这个 1.32× 常数因子
* 但 grad_clip=0.5 在大梯度时会把两者都 clip 到同一 norm，部分抵消差异
* 不管怎样，**`loss_total` 不能作为 sanity check 的判定指标**

**修订**：

* 更新 [`SIGMA_NORMALIZE_ABLATION_PLAN.md`](SIGMA_NORMALIZE_ABLATION_PLAN.md) §3.4：明确说明 loss_total 必然差 ~25%（不是 bug），sanity 判据改为**逐 hop val_chain_*_mse 偏差 < 5%**
* 同时强调 `val_pair_total / val_pair_velocity / val_pair_endpoint`（不经 rollout）必须**逐 bit 相同**——这是 RNG 一致性的硬验证
* 更新 [`scripts/run_ablation.sh`](scripts/run_ablation.sh) 的 sanity 命令输出：列出 4 个 chain MSE 比较公式，明文说 "loss_total will differ by ~25% by design — DO NOT use it as sanity gate"
* 更新 [`configs/A_control.yaml`](configs/A_control.yaml) 顶部注释：从 "<5% 偏差" 改为 "per-hop val_chain_*_mse < 5% 偏差，loss_total 会差 ~25% 这是设计内不是 bug"

**结论**：此问题**实质性影响判据正确性**，必须修。已修。

---

## 🟠 问题 2：CUDA stream 同步开销

**审计原文**：
> `step_loss / float(step_normalizers[hop_idx])` 触发 Python float 与 tensor 的乘除——`float(...)` 调用本身会强制 CUDA→CPU 同步……不过当前实现因为 normalizer 是 Python list of float，`float(list[i])` 是 no-op，所以**没有真实开销**。这条只是 future-proofing。

**回应**：审计自己已确认无实际开销。**不修**——保持当前实现简单清晰。

如果未来 normalizer 来自 tensor（比如运行时学习），再升级为：

```python
norm_tensor = torch.tensor(step_normalizers, device=device, dtype=step_loss.dtype)
step_loss = step_loss / norm_tensor[hop_idx]
```

**结论**：审计自我否决，**记录但不修**。

---

## 🟠 问题 3：B sanity 在 AMP 下不严格

**审计原文**：
> A_control.yaml:74 已经设了 `amp: false`，所以这条不影响本次 ablation——但 §3.4 提到 "AMP/bf16 precision 损失" 的措辞容易让人误以为 amp 是开的。建议在 §3.4 显式注明 "本 ablation 关闭 amp（A.yaml L74）以排除精度问题"。

**修订**：[`SIGMA_NORMALIZE_ABLATION_PLAN.md`](SIGMA_NORMALIZE_ABLATION_PLAN.md) §3.4 已加 "**`amp: false`（A_control.yaml L74 已设）→ 排除精度噪声**" 一行。

**结论**：已修。

---

## 🟡 问题 4：D 配置量级危险

**审计原文**：
> D 的 step_weights = [15.89, 3.51, 1.84, 0.95]——hop0 是 15.89。即使在 normalized velocity space 这是设计意图，结合 `pair_weight=15` × `pair_loss_weights[0]=2.5` × `lambda_roll=4.0`，**hop0 的总有效梯度可能接近爆炸边界**。

**修订**：[`configs/D_closed_form.yaml`](configs/D_closed_form.yaml) 顶部注释新增 "⚠️ GRADIENT EXPLOSION RISK ⚠️" 警告块，列出：

* 监控指标：log_grad_norms=true（已开），grad_norm_first_hop > 5.0 警报
* 缓解方案 A：延长 warmup_ratio 到 0.30
* 缓解方案 B：把 step_weights 整体 ×0.5（不影响相对形状）

**结论**：已加警告但**未修改默认值**——让用户先按 plan 跑一遍，看实际 grad_norm 再决定是否调整。

---

## 🟡 问题 5：80K 不够长（关键）

**审计原文**：
> A 在 80K 的 chain_normal_mse 远未到 best（V6 best @ step 156K = 0.000167，但 step 80K ≈ 0.00029）……如果 C @ 80K ≈ A @ 80K（都是 ~0.00029），不能下结论 "C 与 A 等效"，因为可能两者都还在 underfitting……正确判定要等到 Phase II 中段（约 step 100-130K）才有信号。

**修订**：

| 文件 | 修改 |
|------|------|
| [`SIGMA_NORMALIZE_ABLATION_PLAN.md`](SIGMA_NORMALIZE_ABLATION_PLAN.md) §3.3 | 表格更新：阶段 2 = 120K each（不是 80K），并解释为何 80K 不够 |
| [`SIGMA_NORMALIZE_ABLATION_PLAN.md`](SIGMA_NORMALIZE_ABLATION_PLAN.md) §4 | 解读矩阵开头加 "@120K" 限定，加红框警告 |
| [`configs/A_control.yaml`](configs/A_control.yaml) | `max_steps: 80000` → `120000` |
| [`configs/C_uniform.yaml`](configs/C_uniform.yaml) | `max_steps: 80000` → `120000` |
| [`configs/D_closed_form.yaml`](configs/D_closed_form.yaml) | `max_steps: 80000` → `120000` |
| [`scripts/run_ablation.sh`](scripts/run_ablation.sh) | `main` / `closed_form` / `A` / `C` / `D` 的 max_steps override 全部 80000 → 120000 |
| [`README.md`](README.md) | 预算估算更新（7 GPU·days minimum），加阶段 3 200K 选项 |

**结论**：实质性影响 final verdict 可信度，必须修。已修。

---

## 🟡 问题 6：scripts/run_ablation.sh 末尾 sanity 命令被截断

**审计原文**：
> run_ablation.sh:88 的 `diff <(jq -c '.val_chain_normal_mse,.val_rollout_total' ...)/metrics.jsonl)` 命令路径写了 `.../`，是占位符未实际写完。

**修订**：[`scripts/run_ablation.sh`](scripts/run_ablation.sh) 完整重写 sanity 输出：

* 新增 `metrics_path()` helper：从 resolved yaml 自动解析 `output_dir + run_name`
* sanity 命令完成后输出 4 个 per-hop val_chain MSE 比较公式 + 真实 metrics 路径
* 明示 "loss_total will differ by ~25% by design"

**结论**：已修。

---

## 缺失 1：scripts/summarize_run.sh

**审计原文**：
> README 提到"由 `scripts/summarize_run.sh` 自动生成"，但目录里没有这个文件。

**修订**：新增 [`scripts/summarize_run.sh`](scripts/summarize_run.sh)。功能：

* 输入：run_dir（自动从 `config.resolved.yaml` 解析 metrics.jsonl 真实路径）或直接 metrics.jsonl
* 输出：每个 val_chain_d20/d10/d4/normal_mse + val_pair_total + val_rollout_total + val_multi_objective 的 best 值与对应 step
* 已在 fake metrics.jsonl 上 smoke-test 通过

**结论**：已修。

---

## 缺失 2：runtime estimate

**修订**：[`README.md`](README.md) "责任与时间线" section 加：

* A100 80GB bs=8 bf16 off ≈ 65-80 steps/min
* V100 32GB bs=4 ≈ 30-40 steps/min（需 grad accum）
* sanity 1 day, main 6 days, closed-form 3 days, 阶段3 +5 days/condition

**结论**：已修。

---

## 缺失 3：runs/ 应进 .gitignore

**修订**：新增 [`review/0502/.gitignore`](.gitignore) 含 `runs/`。

**结论**：已修。

---

## 总结

| 编号 | 严重度 | 状态 | 涉及文件数 |
|------|-------|------|-----------|
| 1 | 🔴 关键 | 已修 | 3（PLAN, run_ablation.sh, A_control 注释）|
| 2 | 🟠 中 | 不修（无实际影响）| 0 |
| 3 | 🟠 中 | 已修 | 1（PLAN §3.4）|
| 4 | 🟡 低 | 已修（加警告，不改默认值）| 1（D yaml）|
| 5 | 🟡 关键 | 已修 | 7（PLAN §3.3 §4, 4 yaml, run_ablation.sh, README）|
| 6 | 🟡 低 | 已修 | 1（run_ablation.sh）|
| 缺失 1 | 🟡 中 | 已修 | 1（new summarize_run.sh）|
| 缺失 2 | 🟡 低 | 已修 | 1（README）|
| 缺失 3 | 🟡 低 | 已修 | 1（new .gitignore）|

**v2 状态**：审计意见全部回应。问题 1（sanity 判据）和问题 5（步数）是**实质性修正**——v1 跑出来会做出错误判定。问题 2 经审计自我确认无影响，不修。其余皆为 polish。

修订后再次验证（已通过）：

* `verify_normalizers.py` 输出与所有 yaml 数值精确一致
* `apply_patches.sh --check` dry-run 干净通过
* `apply_patches.sh apply` → `python -m py_compile train_first_hop.py pet_lr/rollout_first_hop.py` 通过
* `apply_patches.sh --revert` → `git diff` 为空
* `summarize_run.sh` 在 fake metrics.jsonl 上 smoke-test 通过
* `run_ablation.sh -n` 语法 OK
