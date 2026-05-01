# AUDIT_RESPONSE_v3 — Cross-AI Audit Response (Round 2)

> **⚠️ DEPRECATED**：本文件记录 **v3 修订意见**。v4 在 v3 基础上又修了 2 个一致性 blocker（`output_dir` 与 `pet_lr/path_guard` 冲突、sanity comparator 阈值不一致）+ 其他 doc/cleanup。最新决策请看 [AUDIT_RESPONSE_v4.md](AUDIT_RESPONSE_v4.md)【当前活跃】。
>
> 本文件仅作历史记录保留。

---

> 本文件记录 review/0502 σ-normalize ablation bundle 在 v2 提交后收到的 **4 个独立 AI agent** 的交叉审计反馈，以及 v3 修订的具体改动。v2 修订记录见 [AUDIT_RESPONSE.md](AUDIT_RESPONSE.md)【deprecated】。

## 1. v2 → v3 摘要

| 状态 | v2 | v3 |
|------|----|----|
| normalizer 模式 | `relative_to: median` | `relative_to: preserve_v6_sum` |
| 数学等价性 | A 与 B **梯度方向**等价（loss 差 24.5%）| A 与 B **loss 与梯度都恒等**（残差 < 1e-19，FP32 < 1e-6）|
| Σw_yaml 一致性 | A=5.0, B=6.62, C=4.0, D=22.19（不一致）| A=B=C=D=5.0（v3 锁定）|
| effective lambda_roll | A=0.8, B=0.604, C=1.0, D=0.18（变量混淆）| A=B=C=D=0.8（v3 唯一变量为 step_weights 形状）|
| sanity check 信号 | per-hop val_chain_*_mse | val_pair_total bit-equal + val_rollout_total < 1% + val_chain_* < 5%（v3 多重信号）|
| run_ablation.sh | 不重写 run_name/output_dir → A_sanity/A_main 输出冲突 | 自动重写两者，所有产物落到 `review/0502/runs/<tag>/run/` |
| summarize_run.sh | 依赖 jq | 纯 python，无 jq |
| apply_patches.sh | dirty tree → warn 并继续 | dirty tree → fail-by-default，`--force` 旁路 |
| step_losses_raw 持久化 | 在 train 循环中计算但未写入 metrics.jsonl | `val_rollout_step_*_raw` 写入 metrics.jsonl |

## 2. 4 个 agent 各自的关键发现

### Agent 2：发现 A==B 不数学等价（**最严重 blocker，4/4 agent 一致确认**）

> "A_control 的 `Σw=5.0`，B_sanity 的 `Σw≈6.6237`。`loss_total = (stacked·w).sum() / w.sum()` 在 B 端会被 6.6237 除掉，与 A 端的 5.0 不一致。两者的 SGD 梯度并不真正相等——B 端的 effective lambda_roll 是 A 的 0.755 倍。"

**v3 处理**：完全采纳。引入 `preserve_v6_sum` normalizer 模式（[`scripts/verify_normalizers.py`](scripts/verify_normalizers.py) 已用 FP64 验证恒等性 < 1e-19）：

$$
n_j = \rho_j \;\Big/\; \frac{\sum_k w_{V6,k}\,\rho_k}{\sum_k w_{V6,k}}
\quad\Rightarrow\quad \sum_j w_{V6,j}\,n_j = \sum_j w_{V6,j} = 5.0
$$

数值：$n = [3.5816, 0.9303, 0.5794, 0.4795]$，B step_weights $= [1.7908, 1.8606, 0.8691, 0.4795]$。

### Agent 2：发现 v2 §3.4 关于 "per-hop val_chain MSE 不依赖 normalizer" 的论断错误

> "Plan §3.4 把 chain MSE 当成"实现等价性的最强 sanity"在数学上不准确。chain MSE 测的是训练好的模型，A 和 B effective lambda_roll 不同 → 训练出不同模型 → chain MSE 也会不同。"

**v3 处理**：在 [SIGMA_NORMALIZE_ABLATION_PLAN.md §3.4](SIGMA_NORMALIZE_ABLATION_PLAN.md) 显式承认 v2 的错误论断。v3 用 preserve_v6_sum 后 A==B 真正恒等，所以 4 个信号（`val_pair_total` bit-equal → `val_rollout_total` < 1% → `val_rollout_step_*_raw` < 1% → `val_chain_*` < 5%）逐级判定。

### Agent 3：发现 `run_ablation.sh` 的 `run_name` 冲突

> "`resolve_yaml()` 只覆写 `max_steps`，不动 `run_name` / `output_dir`。`A_control.yaml` 的 `run_name = first_hop_224_sigma_norm_A_control`，所以 `sanity` 把 A 跑成 50K 后，`main` 再用同一份 yaml 跑 120K 时 `require_fresh_output_dir: true` 会拒绝。"

**v3 处理**：扩展 [`scripts/run_ablation.sh`](scripts/run_ablation.sh) 的 `resolve_yaml()`：

```bash
run_name    → first_hop_224_sigma_norm_${tag}      # tag ∈ {A_sanity, A_main, B, C, D}
output_dir  → ${ABLATION_RUNS_DIR}/${tag}/run      # review/0502/runs/<tag>/run
max_steps   → 已有
```

A_sanity / A_main 因此是不同 run_name + 不同 output_dir，不会冲突。所有训练产物都落到 `review/0502/runs/<tag>/run/<run_name>/metrics.jsonl`，`.gitignore` 已排除 `runs/`。

### Agent 3：D 配置的 grad explosion 警告与 normalizer 选择不匹配

> "D 在 v2 用 median normalizer + raw closed-form weight，hop0 step_weight = 15.89 是 V6 baseline 的 31.8×。yaml 警告说"前 5K 步把 grad_clip 临时降到 0.2"。但这里的真正问题是 v2 使用了 median normalizer，让 D 的 Σw 变成 22.19，等于把整体训练强度也放大了。"

**v3 处理**：v3 把 D 切到 preserve_v6_sum + Σw=5.0 rescale。D step_weights = $[3.5795, 0.7910, 0.4149, 0.2146]$，hop0 是 V6 的 7.16× 而非 31.8×（缩了 4.4×）。grad explosion 警告改为 mild 级别，不再要求 grad_clip 临时下调。

### Agent 4：summarize_run.sh 在 GPU host 上 jq 缺失

> "GPU 训练机的 conda env 通常没装 jq。脚本应该用 `python -c` 一行解析 metrics.jsonl，避免 hard fail。"

**v3 处理**：[`scripts/summarize_run.sh`](scripts/summarize_run.sh) 完全去 jq 化，改用 `${PYTHON} - <<PYEOF` heredoc，输出 best per metric + raw 每 hop step_loss + 最后一行的关键字段。

### Agent 4：apply_patches.sh dirty tree 行为太激进

> "`if git status ... | grep -qv '^$'; then` 这个判断永远会触发（grep -qv '^$' 匹配所有非空行，包括 git status 输出的 'M ' 标记）。然后只是 echo WARN 不退出。意外覆盖 in-progress 工作的风险很高。"

**v3 处理**：[`scripts/apply_patches.sh`](scripts/apply_patches.sh) dirty 检查改为 `[[ -n "${dirty}" ]]`，dirty 时**默认 fail with exit 1** 并打印恢复指引，新增 `--force` 旁路（已 manual-验证两种行为）。

### Agent 4：metrics.jsonl 缺少 step_losses_raw

> "README 说"step_losses_raw_*.csv 可选输出"。但实际 patch 里 evaluate_first_hop 只写了 `val_rollout_step_<i>` 进 sums，没有 raw 版本。条件之间无法直接对比 hop-level 行为。"

**v3 处理**：[`patches/train_first_hop.py.patch`](patches/train_first_hop.py.patch) 新增：

* hunk 4：`sums[f"val_rollout_step_{i}_raw"] = 0.0` 初始化
* hunk 6：`step_losses_raw = roll_out.get("step_losses_raw", roll_out["step_losses"])`，循环累加 `sums[f"val_rollout_step_{i}_raw"]`

[`patches/rollout_first_hop.py.patch`](patches/rollout_first_hop.py.patch) 已经在 v2 阶段加了 `step_losses_raw` 返回字段（`step_loss.detach()` 在 `/normalizer` 之前保留），v3 只需在 evaluate 端持久化即可。

### Agent 3 + 4：schedule scaling 警告

> "max_steps 从 200K 改到 50K 时，`warmup_ratio: 0.25` 会被等比缩放到 12.5K（而非保持绝对 50K）。这相当于 A_sanity 用了一个**压缩 schedule** 跑 50K，而不是 V6 的"前 50K 截断"。"

**v3 处理**：[SIGMA_NORMALIZE_ABLATION_PLAN.md §3.3](SIGMA_NORMALIZE_ABLATION_PLAN.md) 显式说明 schedule 等比压缩。结论：

* sanity（A_sanity vs B）@ 50K 是公平的（A 与 B 共享相同压缩 schedule，A==B 等价性不受影响）
* main（A_main vs C）@ 120K 共享相同 schedule，比较 fair——但 Phase III 仅 30K，建议如结果 ambiguous 升级到 200K

### Agent 2：D closed-form 的 L≈1 假设警告

> "Grönwall 闭式解（§18.3.3）假设 hop-wise Lipschitz 系数 L_j ≈ 1。这只在 Phase II 之后（V6 step ≥ 60K）成立。Phase II 期间用 closed-form 权重训练可能不是最优的——这是设计层面的 caveat，不是 bug。"

**v3 处理**：在 [SIGMA_NORMALIZE_ABLATION_PLAN.md §4](SIGMA_NORMALIZE_ABLATION_PLAN.md) D 解读矩阵下加 "L≈1 假设警告" 段，建议主要看 step ≥ 60K 区间的 best ckpt，避免在 Phase II 早期下定论。

## 3. v3 没有采纳的建议

* Agent 4 建议把 `pair_loss_weights` 与 `rollout step_weights` 一起做 2-D sweep：v3 维持 v1 的 single-axis 设计，文中（[SIGMA_NORMALIZE_ABLATION_PLAN.md §3.2](SIGMA_NORMALIZE_ABLATION_PLAN.md)）已声明 "isolate rollout-channel only" 的 scope。
* Agent 3 建议把 V6 anchor 改成可配置的项目级常数：v3 维持 hardcode V6 = [0.5, 2.0, 1.5, 1.0]，但 helper 接受 `anchor_step_weights` override（yaml 已显式列出此字段，便于未来切到不同 anchor）。

## 4. v3 端到端验证

```text
$ cd PET_LatentResidual
$ git status -- pet_lr/rollout_first_hop.py train_first_hop.py
(clean)
$ python review/0502/scripts/verify_normalizers.py
=== Mathematical identity checks ===
weighted_mean(n, w_v6) (must=1.0)            : 1.000000   OK
max |contribA - contribB| per hop (must≈0)   : 2.710505e-20   OK
|sum(w_B) - sum(w_v6)|             (must≈0)  : 0.000000e+00   OK
$ bash review/0502/scripts/apply_patches.sh --check
[check] rollout_first_hop.py.patch         (clean)
[check] train_first_hop.py.patch           (clean)
$ bash review/0502/scripts/apply_patches.sh apply  # full apply
[apply_patches.sh] Verifying divisor math...
[apply_patches.sh] Patches applied.
$ python -m py_compile pet_lr/rollout_first_hop.py train_first_hop.py
(no error)
$ bash review/0502/scripts/apply_patches.sh --revert
[apply_patches.sh] Patches reverted.
$ git diff --stat -- pet_lr/rollout_first_hop.py train_first_hop.py
(empty — round-trip clean)
$ bash -n review/0502/scripts/{apply_patches,run_ablation,summarize_run}.sh
(no errors — syntax OK)
```

dirty-tree refusal 已 manual-验证（人为 `echo "# tmp" >> pet_lr/rollout_first_hop.py` 后 `apply_patches.sh apply` 退出 1 + 打印恢复指引）。

## 5. 仍未解决的设计问题（不阻塞 ablation 启动）

1. **pair_loss_weights[0]=2.5 与 rollout step_weights[0] 的相互作用**：v3 ablation 仅在 rollout 通道扫，pair 通道固定。如 main 结果显示 C ≈ A，可能需要 follow-up 跑 `pair_loss_weights[0]=1.0` 的额外条件。这是 [SIGMA_NORMALIZE_ABLATION_PLAN.md §3.2](SIGMA_NORMALIZE_ABLATION_PLAN.md) 已声明的范围限制，v3 不修。
2. **schedule 是否应改为绝对步数**：v3 维持 ratio-based schedule（`warmup_ratio: 0.25`）以保持代码改动最小。如未来要做更严格的 fair comparison，可以加一个 yaml 字段 `warmup_steps_abs: 50000` 让 ratio 失效。这是非紧急 enhancement，不阻塞 v3 启动。

## 6. 文件改动清单（v2 → v3）

| 文件 | v2 → v3 |
|------|---------|
| [`patches/rollout_first_hop.py.patch`](patches/rollout_first_hop.py.patch) | 不变（已含 step_losses_raw 返回）|
| [`patches/train_first_hop.py.patch`](patches/train_first_hop.py.patch) | 加 preserve_v6_sum 模式 + anchor_step_weights 参数 + val_rollout_step_*_raw 持久化 |
| [`scripts/verify_normalizers.py`](scripts/verify_normalizers.py) | 重写：preserve_v6_sum 计算 + 三个数学恒等性 self-check |
| [`scripts/run_ablation.sh`](scripts/run_ablation.sh) | resolve_yaml() 重写 run_name + output_dir；sanity 段改用 python 而非 jq |
| [`scripts/summarize_run.sh`](scripts/summarize_run.sh) | 完全去 jq 化，纯 python；新增 raw step loss 段 |
| [`scripts/apply_patches.sh`](scripts/apply_patches.sh) | dirty-tree fail-by-default + `--force` 旁路 |
| [`configs/B_sanity.yaml`](configs/B_sanity.yaml) | step_weights `[2.3723, 2.4648, 1.1514, 0.6352]` → `[1.7908, 1.8606, 0.8691, 0.4795]`；relative_to median → preserve_v6_sum；加 anchor_step_weights |
| [`configs/C_uniform.yaml`](configs/C_uniform.yaml) | step_weights `[1, 1, 1, 1]` → `[1.25, 1.25, 1.25, 1.25]`（rescale 到 Σw=5.0）；relative_to median → preserve_v6_sum |
| [`configs/D_closed_form.yaml`](configs/D_closed_form.yaml) | step_weights `[15.8945, 3.5123, 1.8422, 0.9528]` → `[3.5795, 0.7910, 0.4149, 0.2146]`（rescale 到 Σw=5.0）；relative_to median → preserve_v6_sum；grad warning 降级；加 L≈1 caveat |
| [`SIGMA_NORMALIZE_ABLATION_PLAN.md`](SIGMA_NORMALIZE_ABLATION_PLAN.md) | §2.1 重写为 preserve_v6_sum 推导（含分子+分母双等式）；§2.2 三组对照 + 共享 effective lambda_roll；§3.1 normalizer 区间更新；§3.3 加 schedule scaling 段；§3.4 重写 sanity 判定（多信号分级）；§4 D 加 L≈1 警告 |
| [`README.md`](README.md) | 索引更新；v3 修订记录段；输出物路径修正；删除"step_losses_raw_*.csv 可选输出"误导性表述 |
| [`AUDIT_RESPONSE_v3.md`](AUDIT_RESPONSE_v3.md) | **本文件**（新增）|

---

> v3 完成时间：本次 chat session
> 验证状态：本地 macOS apply→compile→revert round-trip 干净，dirty-tree 拒绝行为已 manual 验证，verify_normalizers.py 三个 identity self-check 全部 OK
> 下一步：把 `review/0502/` 整目录推到 GPU host，按 [README.md "快速开始"](README.md) 跑 sanity 50K（约 1 GPU·day）→ 看 val_pair_total bit-equal + val_rollout_total < 1% → 通过则跑 main 120K（约 6 GPU·days）
