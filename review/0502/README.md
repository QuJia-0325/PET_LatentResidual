# review/0502 — σ-normalize Rollout Ablation Bundle

> **目标**：验证 V6 step_weights = [0.5, 2.0, 1.5, 1.0] 是否真的命中 multi-hop Grönwall 闭式解；还是只是在补偿 rollout step_loss 的 σ²·dt² 自然量级。这是论文 §21.9 行动 1 的核心 ablation，决定 §19 contribution 强度。

## 内容索引

| 文件 | 用途 |
|------|------|
| [README.md](README.md) | 本文件——总索引 |
| [SIGMA_NORMALIZE_ABLATION_PLAN.md](SIGMA_NORMALIZE_ABLATION_PLAN.md) | 完整数学推导 + 设计 + 解读矩阵 |
| [POST_V6_NEXT_STEPS.md](POST_V6_NEXT_STEPS.md) | V6 跑完后的 v3.1 执行计划：sanity sentinel gate、20K mini full sanity、预算表与论文口径 |
| [AUDIT_RESPONSE.md](AUDIT_RESPONSE.md) | 第一轮外部 agent 审计意见的逐条回应（v2 修订记录）|
| [AUDIT_RESPONSE_v3.md](AUDIT_RESPONSE_v3.md) | 第二轮 4-agent 交叉审计的回应（v3 修订记录，已 deprecated）|
| [AUDIT_RESPONSE_v4.md](AUDIT_RESPONSE_v4.md) | 第三轮 4-agent 交叉审计的回应（v4 修订记录）|
| [patches/rollout_first_hop.py.patch](patches/rollout_first_hop.py.patch) | unified diff：rollout 函数加 `step_normalizers` 参数 + 返回 `step_losses_raw` |
| [patches/train_first_hop.py.patch](patches/train_first_hop.py.patch) | unified diff：训练循环加 `_compute_sigma_dt_normalizers`（含 `preserve_v6_sum` 模式）+ 注入两个 rollout 入口 + 持久化 `val_rollout_step_*_raw` |
| [configs/A_control.yaml](configs/A_control.yaml) | Control：与 V6 baseline 完全相同（不开 σ-norm，120K steps）|
| [configs/B_sanity.yaml](configs/B_sanity.yaml) | Sanity check：σ-norm + V6-equivalent step_weights，A 与 B 在 val_pair_total 应 bit-equal、val_rollout_total < 1%、val_chain_* < 5%（50K steps）|
| [configs/C_uniform.yaml](configs/C_uniform.yaml) | 主 ablation：σ-norm + uniform [1.25,1.25,1.25,1.25]（120K steps，Σw=5.0=Σw_v6）|
| [configs/D_closed_form.yaml](configs/D_closed_form.yaml) | 闭式解对照：σ-norm + closed-form [3.58, 0.79, 0.41, 0.21]（120K steps，Σw=5.0=Σw_v6，含 L≈1 假设警告）|
| [scripts/apply_patches.sh](scripts/apply_patches.sh) | 在远程仓库 root 应用 patches（含 dry-run + revert + dirty-tree fail-by-default + `--force` 旁路）|
| [scripts/verify_normalizers.py](scripts/verify_normalizers.py) | 离线验证 normalizer 数值与 A==B 数学恒等性（不依赖 GPU）|
| [scripts/run_ablation.sh](scripts/run_ablation.sh) | 远程跑 A/B sanity gate + A/C/D ablation；PASS 后写 `.sanity_pass`，C/D/main 无 sentinel 会拒绝启动 |
| [scripts/summarize_run.sh](scripts/summarize_run.sh) | 从 metrics.jsonl 提取 best ckpt 摘要 + raw 每 hop step_loss（纯 python，无 jq 依赖）|

## 快速开始

```bash
# 1. 在远程仓库 root，确保 git 干净
cd PET_LatentResidual && git status

# 2. 确认 live code 已含 σ-normalize 改动
#    当前仓库已合入 patches；不要重复 apply。只有在全新远程副本缺少
#    _compute_sigma_dt_normalizers / val_rollout_step_*_raw 时才使用 apply_patches.sh。
grep -R "_compute_sigma_dt_normalizers\|val_rollout_step_.*_raw" train_first_hop.py pet_lr/rollout_first_hop.py

# 3. 离线核对 normalizer 数值
python3 review/0502/scripts/verify_normalizers.py

# 4. 跑 mini full sanity gate（A 20K vs B 20K，多层判据：详见 POST_V6_NEXT_STEPS §3.2）
#    Tier 1 val_pair_total            < 1e-4 rel  （bit-equal 目标）
#    Tier 2 val_rollout_total         < 1%   rel  （effective lambda_roll 等价）
#    Tier 3 val_rollout_step_*_raw    < 1%   rel  （per-hop 原始信号诊断）
#    Tier 4 val_chain_*_mse (4 个)    < 5%   rel  （训练噪声容差）
#    脚本结尾会打印每层的 PASS/FAIL，全部 PASS 才算通过 sanity gate，
#    并写入 review/0502/runs/.sanity_pass sentinel。
SANITY_STEPS=20000 bash review/0502/scripts/run_ablation.sh sanity

# 5. A_main 可与 sanity 并行；C 必须等 sanity PASS 写入 .sanity_pass 后再启动。
GPU=0 bash review/0502/scripts/run_ablation.sh A
GPU=0 bash review/0502/scripts/run_ablation.sh C

# 6. 可选：闭式解对照（同样需要 sanity-pass sentinel）
bash review/0502/scripts/run_ablation.sh closed_form   # D, 120K steps

# 7. 查看每条 run 的 best ckpt 摘要（脚本会自动跨索引/数据盘解析）
bash review/0502/scripts/summarize_run.sh review/0502/runs/A_main
bash review/0502/scripts/summarize_run.sh review/0502/runs/C
```

## 阅读顺序建议

1. **先读** [SIGMA_NORMALIZE_ABLATION_PLAN.md](SIGMA_NORMALIZE_ABLATION_PLAN.md) §1-§3（数学等价性 + 4 陷阱，**特别是 §2.1 preserve_v6_sum 推导和 §3.4 sanity 判据**）。
2. 再看 §4 解读矩阵——决定结果出来后怎么写论文（**注意**：80K 不能作为 final verdict，需 120K +）。
3. 执行前看 [POST_V6_NEXT_STEPS.md](POST_V6_NEXT_STEPS.md) §2-§5，按 v3.1 sentinel gate 和双列预算安排 GPU。
4. 最后看 patches 与 configs，确认改动最小、可 revert。

> **审计修订记录**：
> * **v2**：第一轮单 agent 审计后修订 `max_steps 80K→120K`、加 D grad warning、加 `summarize_run.sh`、加 `.gitignore`。详见 [AUDIT_RESPONSE.md](AUDIT_RESPONSE.md)（已 deprecated）。
> * **v3**：第二轮 4-agent 交叉审计后发现 v2 用 `relative_to: median` 时 A 与 B **不数学等价**——`rollout_first_hop.py:107` 是加权"平均"（除以 `Σw`），所以 A==B 同时要求分子和分母相等。v3 引入 `preserve_v6_sum` normalizer 模式 + 把 B/C/D step_weights 都缩放到 `Σw=5.0`，使所有 4 个 condition 共享相同的 effective lambda_roll。同时修了 `run_ablation.sh` run_name/output_dir 冲突、`summarize_run.sh` 的 jq 依赖、`apply_patches.sh` 的 dirty-tree warn-and-continue、加了 `val_rollout_step_*_raw` 持久化。详见 [AUDIT_RESPONSE_v3.md](AUDIT_RESPONSE_v3.md)（已 deprecated）。
> * **v4**：第三轮 4-agent 交叉审计的 2 个一致 blocker：(1) `output_dir` 被 rewrite 到 repo 内会被 `pet_lr/path_guard.resolve_data_disk_dir` 直接拒绝（要求 `/data_2`）；(2) sanity comparator 阈值前后不一致（脚本打印 < 1% 但实际 gate 用 < 5%，且缺 `val_pair_total` bit-equal 与 `val_rollout_step_*_raw` 检查）。v4 把训练产物路由到 `/data_2/.../review_0502_runs/<tag>/`、repo 内只留 `config.resolved.yaml + train.log` 索引；并把 sanity comparator 重写为 4 层 tier（PASS/FAIL 显式逐项报告）。详见 [AUDIT_RESPONSE_v4.md](AUDIT_RESPONSE_v4.md)。
> * **v3.1（当前活跃路线）**：见 [POST_V6_NEXT_STEPS.md](POST_V6_NEXT_STEPS.md)。关键变更是 C/D/main 必须等待 sanity-pass sentinel，mini full sanity 推荐 `SANITY_STEPS=20000`，10K 只作 smoke 且不会解锁 C/D。

## 与 ARCHITECTURE_ANALYSIS_20260501.md 的关系

* §21.9 行动 1 = 本目录的实施
* §21.3 σ²·dt² 替代解释 = 本目录验证的核心假设
* §21.11.4 "emerge 措辞修正" = 本目录把 ablation 从"看是否 emerge"改写为"重新 sweep 验证形状贡献"

修订完后，用本目录结果回填 §18.3.4 / §19.5 / §21.8。

## 输出物

`pet_lr/path_guard.resolve_data_disk_dir` 强制要求 `output_dir` 在 `/data_2` 下，否则 `train_first_hop.py`
启动即抛 RuntimeError。v4 起 `run_ablation.sh` 把训练产物路由到数据盘，并在 repo 内只保留索引文件：

```text
# 索引（in-repo，便于 PR / review）：
review/0502/runs/<tag>/
├── config.resolved.yaml          # 实际使用的 yaml（max_steps + run_name + output_dir 已 rewrite）
└── train.log                     # tee 镜像的完整训练 log

# 训练产物（data disk，受 path_guard 保护）：
/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0502_runs/<tag>/run/first_hop_224_sigma_norm_<tag>/
├── metrics.jsonl                 # 每个 val 周期的所有 metrics（含 v3 新增的 val_rollout_step_*_raw）
├── ckpt_*.pt                     # 各步检查点（保留 best + last）
└── ...                           # 其他训练产物
```

如需把训练产物落到其他盘（仍要在 `/data_2` 下），导出 `ABLATION_OUTPUT_ROOT` 即可：

```bash
export ABLATION_OUTPUT_ROOT=/data_2/<custom>/review_0502_runs
bash review/0502/scripts/run_ablation.sh sanity
```

`metrics.jsonl` 中关键字段（v3 新增以 `_raw` 结尾）：

* `val_rollout_total`：σ-norm 之后的加权总损失（A/B 在 v3 下应 < 1% relative）
* `val_rollout_step_<i>`：σ-norm 之后的每 hop 损失（B 与 A 不直接可比，因 normalizer 不同）
* `val_rollout_step_<i>_raw`：**未除 normalizer** 的原始每 hop 损失（A/B/C/D 之间直接可比，是诊断 hop-level 行为的主要工具）
* `val_chain_d20_mse / d10_mse / d4_mse / normal_mse`：4 个 chain 重构 MSE
* `val_pair_total`：完全不经 rollout 的 pair velocity loss（A/B sanity 时应 bit-equal）
* `val_select_score`：best checkpoint 选择用的目标值；当 yaml 里 `best_metric: val_multi_objective` 时，它就是对应加权和。注意 `val_multi_objective` 不是 `metrics.jsonl` 字段名，只出现在 stdout 的 new-best 文本里。

> **注意**：`review/0502/runs/` 已被 `.gitignore` 排除，避免把 ckpt/log 提交进 repo。

## 责任与时间线

* **预算（v3.1 双列口径；以现场吞吐量为准）**：
  * sanity light (A_50K + B_50K)：@80 steps/min ≈ 1.0 GPU·day；@33 steps/min ≈ 2.4 GPU·days
  * mini full sanity (A_20K + B_20K)：@80 steps/min ≈ 0.42 GPU·day；@33 steps/min ≈ 1.0 GPU·day
  * A_main + C_uniform (120K + 120K)：@80 steps/min ≈ 2.5 GPU·days；@33 steps/min ≈ 6.1 GPU·days
  * closed-form (D_120K，可选)：@80 steps/min ≈ 1.25 GPU·days；@33 steps/min ≈ 3.0 GPU·days
  * **minimum viable = mini full sanity + A_main + C_uniform = ~2.9 GPU·days (@80) / ~7.1 GPU·days (@33)**
  * 阶段 3（200K 完整 Phase III）仅在 main 结果 ambiguous 时启动
* **GPU 速率参考**（仅供估算，实际以您机器为准）：
  * A100 80GB，bs=8，bf16 off ≈ 65-80 steps/min（V6 baseline 实测）
  * V100 32GB，bs=4 ≈ 30-40 steps/min（需 grad accum 或减小 batch）
