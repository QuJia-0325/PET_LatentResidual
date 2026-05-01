# AUDIT_RESPONSE_v4 — Cross-AI Audit Response (Round 3)

> 本文件记录 review/0502 σ-normalize ablation bundle 在 v3 提交后收到的 **4 个独立 AI agent** 的交叉审计反馈，以及 v4 修订的具体改动。v3 修订记录见 [AUDIT_RESPONSE_v3.md](AUDIT_RESPONSE_v3.md)（已 deprecated），v2 见 [AUDIT_RESPONSE.md](AUDIT_RESPONSE.md)（已 deprecated）。

## 1. v3 → v4 摘要

| 状态 | v3 | v4 |
|------|----|----|
| `output_dir` 写入位置 | `review/0502/runs/<tag>/run`（repo 内）| `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0502_runs/<tag>/run`（数据盘）|
| path_guard 兼容 | ❌ `pet_lr/path_guard.resolve_data_disk_dir` 会直接抛 `RuntimeError` 拒绝启动 | ✅ 通过（数据盘路由 + 入口预校验） |
| Sanity comparator 阈值 | 脚本 echo "< 1%" 但 python 实际 gate 是 5%，且只比 chain MSE 一项 | 4 层 tier，逐项 PASS/FAIL；陈述与代码一致 |
| Sanity comparator 覆盖 | 只比 `val_chain_*_mse` | Tier 1 `val_pair_total` < 1e-4 + Tier 2 `val_rollout_total` < 1% + Tier 3 `val_rollout_step_*_raw` < 1% + Tier 4 `val_chain_*_mse` < 5% |
| `*.orig` 残留 | 工作树有 `train_first_hop.py.orig` + `pet_lr/rollout_first_hop.py.orig` | 已 `rm`，工作树干净 |
| `A_control.yaml` 头部注释 | 残留 v2 stale "差 ~25% 不是 bug"、"P1 50K → 150K/200K Pilot 策略" | 改写为 v3/v4 对齐版（A==B 数学严格等价；步数策略由 run_ablation.sh 统一管） |
| `verify_normalizers.py` docstring | "weighted_mean = 2.3311e-04"、"sum=5.003"、"sum=4.999" 等 v2 stale 数字 | 同步为实测值 2.3323e-04、所有 sum=5.0000 |
| `AUDIT_RESPONSE.md`（v2）DEPRECATED 标记 | 缺 | 加 |
| `AUDIT_RESPONSE_v3.md` DEPRECATED 标记 | 缺 | 加 |
| PLAN §3.4 阈值与 README/script 一致 | 文档说 < 5%，但 §3.4 多层判据 < 1%/5%，三处不一致 | 全部统一为 §3.4 多层 tier，README 与 script 引用同一表 |
| PLAN §3.4 FP32/FP64 量级说明 | 只说 "FP32 round-off 残差 < 1e-6" | 显式 caveat：FP64 是 ~1e-20，FP32 是 ~1e-6/1e-3，超过 tier 阈值即 bug |

数学结果**未变**：preserve_v6_sum + step_weights = w_v6 × n 仍然让 A==B 在 SGD 梯度层面恒等（FP64 残差 ~2.7e-20，由 verify_normalizers.py 实测）。v4 全是工程/文档/路径修复。

## 2. 4 个 agent 反馈分组

### Agent 4 完全通过 — 无修订

### Agent 1（5 个 polish）：
1. 🟡 A_control.yaml 头部注释残留 v6 200K + Pilot 策略 → 已重写
2. 🟡 PLAN §2.2 weighted_mean 数值 2.331e-04 vs 实测 2.3323e-04 → 全部对齐
3. 🟡 §3.4 描述与 run_ablation.sh sanity 阈值（1% vs 5%）三处不一致 → 全部用同一套 4-tier 表
4. 🟡 v2 AUDIT_RESPONSE 缺 DEPRECATED → 已加，并把 v3 一并标 deprecated
5. 🟢 §3.4 FP32 vs FP64 残差量级说明 → 已加 caveat

### Agent 2（5 个，含 2 个 blocker）：
1. **🔴 Blocker**：output_dir 被 rewrite 到 repo 内 → path_guard 直接拒绝。**v4 修复**：把训练产物路由到 `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0502_runs/<tag>/run`，repo 内只保留 `config.resolved.yaml + train.log` 索引；run_ablation.sh 启动时还会预校验 `ABLATION_OUTPUT_ROOT` 必须 `/data_2` 前缀。
2. **🔴 by-design**：live 代码没有应用 σ-normalize 实现。这是预期行为（patches 在远程 apply）。v4 在 README 高亮"必须 apply_patches.sh 先于 run_ablation.sh sanity"。
3. **🟠 Medium**：`.orig` 残留（容易误提交）→ 已 `rm`。
4. **🟠 Medium**：sanity comparator 阈值不一致 → 已重写为 4-tier。
5. **🟡 Low/Medium**：A_control.yaml v2 stale 注释 → 已重写。

### Agent 3（4 个）：
1. **High**：同 Agent 2 #1 — output_dir vs path_guard → 已修。
2. **Medium**：同 Agent 2 #4 — comparator 阈值 → 已修。
3. **Low/Medium**：同 Agent 1 #1 / Agent 2 #5 — A_control.yaml 注释 → 已修。
4. **Low**：同 Agent 2 #3 — `.orig` 残留 → 已修。

## 3. 关键修订详情

### 3.1 path_guard 路由（最高优先 blocker）

`pet_lr/path_guard.py`：

```python
DATA_DISK_ROOT = Path("/data_2").resolve()

def resolve_data_disk_dir(path_like, *, arg_name):
    ...
    if DATA_DISK_ROOT != resolved and DATA_DISK_ROOT not in resolved.parents:
        raise RuntimeError(f"{arg_name} must be under {DATA_DISK_ROOT}. ...")
```

`train_first_hop.py:1184` 在启动时调 `resolve_data_disk_dir(output_dir)`，拒绝任何 repo-internal 路径。v3 的 `resolve_yaml()` 把 output_dir rewrite 成 `${ABLATION_RUNS_DIR}/${tag}/run` = `review/0502/runs/<tag>/run`，**直接被拒**。

v4 修复（[scripts/run_ablation.sh](scripts/run_ablation.sh)）：

```bash
ABLATION_INDEX_DIR="${ABLATION_INDEX_DIR:-${REVIEW_DIR}/runs}"                 # in-repo: config + log
ABLATION_OUTPUT_ROOT="${ABLATION_OUTPUT_ROOT:-/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0502_runs}"

# 入口预校验
case "${ABLATION_OUTPUT_ROOT}" in
    /data_2/*|/data_2) : ;;
    *) echo "ERROR: ABLATION_OUTPUT_ROOT must be under /data_2 ..."; exit 1 ;;
esac

# resolve_yaml 中
out_dir="${ABLATION_OUTPUT_ROOT}/${tag}/run"      # 数据盘
indexdir="${ABLATION_INDEX_DIR}/${tag}"           # repo 内（存 config.resolved.yaml + train.log）
```

最终布局：

```text
# 索引（in-repo，便于 PR / review）：
review/0502/runs/<tag>/
├── config.resolved.yaml
└── train.log

# 训练产物（data disk，受 path_guard 保护）：
/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0502_runs/<tag>/run/first_hop_224_sigma_norm_<tag>/
├── metrics.jsonl
├── ckpt_*.pt
└── ...
```

`summarize_run.sh` 已通过解析 `config.resolved.yaml` 的 `output_dir` 自动跳到数据盘读取 metrics.jsonl，调用方式仍是 `bash scripts/summarize_run.sh review/0502/runs/A_main`。

### 3.2 多层 sanity comparator

新版 [`scripts/run_ablation.sh sanity`](scripts/run_ablation.sh) 的 comparator 用 pure-python（无 jq 依赖）实现 4-tier 检查：

| Tier | metric | 阈值 (rel err) | 含义 |
|------|--------|--------------|------|
| 1 | `val_pair_total` | < 1e-4 | bit-equal target，FP32 期望 ~1e-6；不通过 = RNG/数据流 bug |
| 2 | `val_rollout_total` | < 1% | effective lambda_roll 等价；不通过 = sigma_normalize 入口配置错 |
| 3 | `val_rollout_step_*_raw` (per-hop) | < 1% | 未除 normalizer 的原始 step loss；不通过 = step_normalizers 错位 |
| 4 | `val_chain_*_mse` (4 个) | < 5% | 训练好的模型一致性；FP32 训练噪声 + EMA 抖动允许 ~1e-3 量级 |

实现要点：
- 每行打印 `metric, A 值, B 值, rel_err, tier_threshold, PASS/FAIL`
- 末尾打印 overall PASS/FAIL（任一 tier FAIL → overall FAIL）
- 末尾附 FP32 期望 caveat，避免误把训练噪声当 bug

### 3.3 A_control.yaml 头部重写

去掉的 stale v2 内容：
- "raw loss_total 由于 rollout 分母 Σw 不同, A 和 B 会差 ~25%, 这不是 bug" — v3 起 A==B 严格等价（FP64 残差 ~1e-20），这条注释会让读者怀疑实现
- "Pilot 策略：P1 直接用正式权重 50K → 通过后继续训练至 150K/200K，不重启" — 这是 V6 200K 的训练策略，σ-norm ablation 用不到
- "Go/No-Go：+1K pair_loss<5e-4 ..." — 同上，与 σ-norm ablation 无关
- "本 yaml 的 max_steps 已下调到 120K" — 准确说是 run_ablation.sh override 而非 yaml 自带

替换为：v3/v4 对齐的 sanity 50K + main 120K 二阶段说明，明确 max_steps 由 run_ablation.sh override。

### 3.4 .orig 清理

`apply_patches.sh` 在 dry-run 流程中会留下 `*.orig`，v3 之前的本地验证产生了：

```
train_first_hop.py.orig            (119265 bytes, untracked)
pet_lr/rollout_first_hop.py.orig   (5720 bytes, untracked)
```

v4 已 `rm` 这两个文件。**未来策略**：apply_patches.sh 在 apply 成功后应 `rm -f *.orig`（这是它本来就有的 `cleanup` 行为，但 v3 的 dry-run+apply+revert 自动化里有路径漏检），考虑到这是验证步骤产物、不是脚本 bug，暂不改 apply_patches.sh，只清理工作树。

## 4. 验证

| 检查 | 命令 | 结果 |
|------|------|------|
| Math 等价性（v3 仍成立）| `python review/0502/scripts/verify_normalizers.py` | ✅ `weighted_mean(n, w_v6)=1.0`、`max\|contribA-contribB\|=2.71e-20`、`Σw=5.0000` |
| run_ablation.sh syntax | `bash -n review/0502/scripts/run_ablation.sh` | ✅ |
| apply_patches.sh syntax | `bash -n review/0502/scripts/apply_patches.sh` | ✅（未变更）|
| summarize_run.sh syntax | `bash -n review/0502/scripts/summarize_run.sh` | ✅（未变更）|
| .orig 残留 | `git status --porcelain \| grep -E '\.orig$' \| wc -l` | `0` |
| docstring 与实测一致 | 跑 verify_normalizers.py 输出与 docstring "Expected output" 比对 | ✅ 全字段对齐 |

## 5. 已知 by-design 行为（不修）

| 现象 | 解释 |
|------|------|
| live `train_first_hop.py` / `rollout_first_hop.py` 不含 σ-normalize 字段 | patches 必须在远程 GPU 机上 `apply_patches.sh` 后才生效；这是审计与 review 工程的设计——本地 review/0502 目录是 patch 包，不修改主分支代码 |
| sanity comparator 用 rel err 而非绝对差 | rel err 对 lr / batch / 数据规模都鲁棒；绝对阈值需要重新校准 |
| 阶段 3 仍按 200K 设计 | 与 V6 baseline 训练长度一致，给 final verdict 最 fair 的对比；只在 main 阶段 ambiguous (5-10%) 时才启动 |

---

**v4 状态**：4 个 agent 反馈全部应答，2 个 blocker 修复，所有数学/路径/文档/工作树都对齐。可以提交远程。
