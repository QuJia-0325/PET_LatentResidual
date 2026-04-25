# V3.1 重做实验设计 — 2026-04-26

**Supersedes**: V3 实验 B/C（SF-pair / Rollout-Up 50K from C）— schedule bug 污染，结果不可信  
**Inherits**: V3 实验 A（Path A 多 ckpt）✅ 已成立，作为本轮的诊断锚点  
**配套分析**: [./v3_results_analysis_20260426.md](./v3_results_analysis_20260426.md)

---

## 0. 一句话目标

修掉 SF schedule 的 resume 锚点 bug，重新跑 SF-pair vs Rollout-Up 控制变量，**得到能下结论的对比数据**。

---

## 1. 必要前置（按顺序）

### Step P0 — 远端核查 sf_gap_norm（30 min）

确认 SF 替换路径**本身**没坏（与 schedule bug 解耦）。

```bash
# 在远端机执行：
JL=/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_selfforcing_from_C_gpu0_r1/metrics.jsonl
python -c "
import json, statistics as st
gaps, alphas = [], []
with open('$JL') as f:
    for ln in f:
        d = json.loads(ln)
        if d.get('event','train') == 'train' and 'sf_gap_norm' in d:
            gaps.append(d['sf_gap_norm']); alphas.append(d.get('sf_alpha',0))
print(f'n={len(gaps)} | sf_alpha: min={min(alphas):.3f} max={max(alphas):.3f} mean={st.mean(alphas):.3f}')
print(f'sf_gap_norm: min={min(gaps):.6f} max={max(gaps):.6f} mean={st.mean(gaps):.6f}')
"
```

**判定**：
- `sf_alpha mean ≈ 1.0` 全程 → §2 调度 bug 成立，按本计划继续
- `sf_alpha` 有变化（说明 schedule 实际生效，与代码读源不符） → 暂停，重审代码
- `sf_gap_norm ≈ 0` → SF 替换从未生效，§3 代码层修复优先级提升

### Step P1 — 代码层修 SF schedule resume 语义（< 30 min, ~10 行）

在 [`train_first_hop.py`](../../train_first_hop.py) 的 `compute_self_forcing_z_src` 中：

```python
sf_cfg = cfg.get("training", {}).get("self_forcing_pair", {})
if not bool(sf_cfg.get("enabled", False)):
    return None

# v3.1 fix: SF schedule supports resume-relative anchor
schedule_origin = str(sf_cfg.get("schedule_origin", "absolute")).lower()
if schedule_origin == "resume_relative":
    resume_start = int(cfg.get("_runtime", {}).get("resume_start_step", 0))
    effective_step = max(global_step - resume_start, 0)
elif schedule_origin == "absolute":
    effective_step = global_step
else:
    raise ValueError(f"self_forcing_pair.schedule_origin must be 'absolute' or 'resume_relative', got {schedule_origin}")

alpha_sf = get_linear_schedule_value(
    global_step=effective_step,
    warmup_steps=int(sf_cfg.get("warmup_steps", 5000)),
    ramp_steps=int(sf_cfg.get("ramp_steps", 10000)),
    start=float(sf_cfg.get("alpha_sf_start", 0.0)),
    end=float(sf_cfg.get("alpha_sf_end", 1.0)),
)
```

并在 main loop 的 resume 加载完成后注入：

```python
cfg.setdefault("_runtime", {})["resume_start_step"] = int(start_step)
```

### Step P2 — `[train]` 打印行加 SF 诊断字段（5 min）

在 [`train_first_hop.py`](../../train_first_hop.py) 的两处 print（pbar / non-pbar 分支）字符串里追加：

```python
f"sf_alpha={float(sf_info['alpha_sf'].item()) if sf_info else 0.0:.3f} "
f"sf_gap={float(sf_info['gap_norm'].item()) if sf_info else 0.0:.6f} "
```

确保终端能直接看到 SF 是否激活。

### Step P3 — Smoke test（20 min, GPU-0 batch=2）

两组对照 smoke test，**真正能验证 schedule fix**（不是只看 print 字段）：

**Smoke-A（验证 fix 生效）**：从 v3 旧 ckpt（step≈46400）resume，但 config 设：
- `schedule_origin: resume_relative`
- `warmup_steps: 50`, `ramp_steps: 100`（缩小 100×）
- `max_steps: 46600`（即 +200 实际新增步）

期望：`alpha_sf` 在 step 46400→46450 全为 0，46450→46550 单调上升 0→1，46550 之后稳定 1.0。**stdout 直接可见**。

**Smoke-B（reproduce 旧 bug 作为 control）**：同 ckpt resume，但：
- `schedule_origin: absolute`
- 其他同 Smoke-A

期望：`alpha_sf ≡ 1.0` 第 1 步起即满载（reproduce 旧 v3 行为）。

两组对比通过则 P1 代码 fix 已验证；之后才有资格上 B'/C'。

---

## 2. v3.1 实验设计（全部基于修复后的代码）

### 2.1 实验 B' — SF-pair redo（GPU-2，~30h）

**关键变化**：
- `max_steps: <resume_step + 50000>`（**+50K 真实新增训练**，而不是 +3.6K；⚠️ 具体值待 200K v3 完成后确定）
- `schedule_origin: resume_relative`
- `warmup_steps: 5000`（resume-relative 锚点 → 真的有 5K 步纯 GT）
- `ramp_steps: 10000`（5K → 15K 渐进开 SF）
- 15K → 50K 步真正训 SF（35K 步稳态）
- 起点 ckpt：**200K v3 完成后的 best.pt**（而不是过期的 Scheme C best.pt），让起点更强。⚠️ best 仍在更新，需等 200K 完成再锁定

```yaml
# configs/pet_flow/pet_flow_first_hop_224_v3p1_sf_pair.yaml
run_name: first_hop_224_v3p1_sf_pair
training:
  max_steps: 100000             # 占位：实际值 = 200K v3 best 的 step + 50000，待确定后回填
  self_forcing_pair:
    enabled: true
    schedule_origin: resume_relative   # ← v3.1 新增
    alpha_sf_start: 0.0
    alpha_sf_end: 1.0
    warmup_steps: 5000          # 相对 resume 起点
    ramp_steps: 10000
  rollout: { alpha_start: 1.0, alpha_end: 1.0, lambda_start: 0.25, lambda_end: 0.25, ... }
  # ... 其余与 v3 selfforcing_from_C 配置同
optimizer: { lr: 4.0e-5, ... }
lr_schedule: { warmup_ratio: 0.05, ... }   # 5K / 100K = 5% — 与新 max_steps 比例匹配
```

### 2.2 实验 C' — Rollout-Up redo（GPU-3，~30h）

**关键变化**：和 B' 唯一差别是 SF off + λ_roll = 1.0。

```yaml
# configs/pet_flow/pet_flow_first_hop_224_v3p1_rollout_up.yaml
run_name: first_hop_224_v3p1_rollout_up
training:
  max_steps: 100000             # 占位：实际值 = 200K v3 best 的 step + 50000，待确定后回填
  self_forcing_pair: { enabled: false }
  rollout: { lambda_start: 1.00, lambda_end: 1.00, alpha_start: 1.0, alpha_end: 1.0, ... }
optimizer: { lr: 4.0e-5, ... }
lr_schedule: { warmup_ratio: 0.05, ... }
```

### 2.3 控制变量矩阵（v3 → v3.1 只动一处）

| 维度 | v3 B | v3 C | **v3.1 B'** | **v3.1 C'** | 备注 |
|------|------|------|-------------|-------------|------|
| max_steps | 50000 | 50000 | **<resume_step+50K>** | **<resume_step+50K>** | 新增训练 50K 实步；待 200K v3 完成后确定 |
| `schedule_origin` | absolute | n/a | **resume_relative** | n/a | bug 修复 |
| SF warmup/ramp | 5000/10000 (失效) | n/a | **5000/10000 (有效)** | n/a | |
| rollout λ | 0.25 | 1.00 | 0.25 | 1.00 | 唯一 B vs C 差异轴 |
| LR | 4e-5 | 4e-5 | 4e-5 | 4e-5 | |
| LR warmup_ratio | 0.05 | 0.05 | 0.05 | 0.05 | |
| 起点 ckpt | 旧 Scheme C @46400 | 同左 | **200K v3 best.pt（完成后取）** | 同左 | 起点更强；⚠️ best 仍在更新，需等 200K 完成再锁定 |

### 2.4 启动命令

```bash
# ⚠️ 前置条件：200K v3 已完成或 best 不再更新。先核对起点 ckpt 真实步号：
V3_BEST=/data_2/qujiaxiang/outputs/PET_LatentResidual/<200K_v3_run_dir>/best.pt
python -c "import torch; c=torch.load('${V3_BEST}', map_location='cpu'); print('step=', c.get('step'), 'metric=', c.get('best_val'))"
# 记录输出的 step 和 metric，写入 v3.1 config 的注释中
# 然后更新 B'/C' config 中的 max_steps = <best_step> + 50000

# B' (GPU-2):
CUDA_VISIBLE_DEVICES=2 nohup python -u train_first_hop.py \
    --config configs/pet_flow/pet_flow_first_hop_224_v3p1_sf_pair.yaml \
    --resume ${V3_BEST} \
    > review/0426/logs_train/v3p1_sf_pair_gpu2.log 2>&1 &

# C' (GPU-3):
CUDA_VISIBLE_DEVICES=3 nohup python -u train_first_hop.py \
    --config configs/pet_flow/pet_flow_first_hop_224_v3p1_rollout_up.yaml \
    --resume ${V3_BEST} \
    > review/0426/logs_train/v3p1_rollout_up_gpu3.log 2>&1 &
```

---

## 3. 监控（实时 gate，避免 v3 那种事后才发现的悲剧）

**约定**：以下 `BASE_VAL` 指 200K v3 best.pt 的 val_select_score（启动前必须从 ckpt 读出并回填到本节）。

### 3.1 启动 +30 min 检查

- [ ] B' 日志看到 `sf_alpha=0.000 sf_gap=0.000000`（resume 后 +0~+5K 步纯 GT 区间）
- [ ] B' 和 C' 双方 `roll_frac` 占比 ~ 50% 以上（C' λ=1.0 导致 rollout loss 主导）
- [ ] 两边 `loss` 数值合理（不爆炸，不 NaN）
- [ ] 重要：B'/C' 的 `val_select_score` 第一次 eval（通常 +500~+1000 步）不应比 `BASE_VAL` 退化 > 50%

### 3.2 启动 +6h 检查（resume 起点 +6K 步，已进入 SF ramp 区间 5K-15K）

- [ ] B' `sf_alpha` 应当严格非零且单调上升（在 [0.05, 0.15] 区间）
- [ ] B' `sf_gap_norm` 应当 ≈ `sf_z_pred_norm` 的 30–60%（z_src_pred 与 z_gt 有真实差异）
- [ ] B' `val_select_score` 不应比 `BASE_VAL` 退化超过 30%（短期注入代价；超过则触发 §5 fallback 4 的 wait-window）

### 3.3 启动 +24h 检查（resume 起点 +24K 步，已进入 SF 稳态 alpha_sf=1.0）

- [ ] B' / C' `val_select_score` 是否已从注入退化中恢复（应 ≤ `BASE_VAL × 1.10`）
- [ ] B' / C' `val_select_score` 是否开始改善（< `BASE_VAL`）
- [ ] **期望** B' vs C' 的 `grad_total` 量级偏差 > 20%（如果两者完全一致，说明 SF 没产生差异化梯度——回到 v3 的"双双退化"陷阱）

---

## 4. 成功判据（v3.1 收紧版，全部用相对量）

**约定**：`BASE_VAL` = 200K v3 best.pt 的 val_select_score；`BASE_EXPGAP` = 200K v3 best.pt 上的 Path A mean exposure_gap (hops 1–3)；`BASE_CHAINMSE` = 200K v3 best.pt 的 val_chain_normal_mse。三者都要在启动 v3.1 前从 200K v3 ckpt + Path A redo 测出并回填。

| 预测 | 阈值（相对量） | 不满足后果 |
|------|---------------|----------|
| B' final val_select_score < `BASE_VAL` | Δ ≥ 5% （即 ≤ `BASE_VAL × 0.95`）| SF 即使有 warmup 也无效 |
| **B' < C'**（SF 严格优于纯 Rollout-Up）| Δ ≥ 5% (相对 C') | "pair 主梯度"假设错误 |
| B' final val_chain_normal_mse < `BASE_CHAINMSE` | Δ ≥ 8% | SF 没把末端 hop 的 plateau 打破 |
| Path A redo @ B' best.pt — exposure_gap < `BASE_EXPGAP` | Δ ≥ 15% (相对) | exposure 没被 SF 实质压低 |
| B' `sf_alpha` 在 `(resume_start + 5K)` 时**开始**上升 | 单调 0→1 在后续 10K 步内完成 | schedule fix 没生效，stop and debug |

---

## 5. 退路 / Fallback

如果 v3.1 B' 仍然不显著优于 C'（即 SF 真的没用）：

| 失败信号 | 退路 |
|---------|-----|
| B' ≈ C' 且 Δ < 3% | 认 "SF 在该数据规模下不区分于加大 rollout" — 改写论文叙事，把 Path A 多 ckpt 趋势作为主诊断证据，方法贡献改为 "rollout-aware reweighting + 多尺度 chain consistency" |
| B' ≈ baseline 且 sf_alpha 全程 = 1（schedule fix 失败） | 重审 Step P1 代码 + smoke test |
| B' < baseline 但 sf_gap_norm > 0 全程 | SF 是负贡献 → exposure bias 不是主因（与 Path A 矛盾，须重做诊断） |
| B' / C' 都比 200K v3 baseline 退化 | 区分两种情况：(a) 注入早期（< +30K 步）退化是预期现象（SF/rollout 文献中的注入震荡），**继续等 wait-window**；(b) +30K 步仍未恢复 → 改用 step ≈ 30K–50K 的早期 ckpt 作起点（减少 GT 过拟合冲击）|

---

## 6. 时间预算

本计划采取**串行策略**（等 200K v3 完成再启 B'/C'），代价是 GPU-2/GPU-3 闲置 ~1.5 天，收益是起点 ckpt 最优。

备选策略：用 200K v3 当前 best 作 mid-ckpt 立刻起跑（B' vs C' 是相对比较，起点强度不影响相对结论）。**默认走串行；如 200K v3 跑超过 Day 3 仍未完成，切换到并行。**

| Day (从 0426 算) | GPU-0 (空闲) | GPU-1 (200K v3) | GPU-2 (B') | GPU-3 (C') |
|------|---------|----------|------|------|
| 0 PM | P0 远端核查 + P1 代码 fix + P2 print 字段 + P3 smoke A/B | 200K (~95K) | 待启动 | 待启动 |
| 1 | 写 v3p1 B'/C' configs（max_steps 留占位）+ commit | 200K (~110K) | 待启动 | 待启动 |
| 2 | （待分配；可起 Path A redo 准备脚本）| 200K (~125K) | 待启动 | 待启动 |
| 3 | **200K 完成 → Path A redo + 测 BASE_VAL/BASE_EXPGAP/BASE_CHAINMSE → 回填 v3p1 config max_steps** | 200K 完成 | 启动 B' (+0K) | 启动 C' (+0K) |
| 4 | 监控 +24h gate | — | B' (~+15K, ramp 完成) | C' (~+15K) |
| 5 | （待分配） | — | B' (~+30K) | C' (~+30K) |
| 6 | （待分配） | — | B' (~+45K) | C' (~+45K) |
| 7 | 全 val 评估 + 视觉对比 + 写 v3.1 结果总结 | — | B' 完成 | C' 完成 |

---

## 7. Action checklist

- [ ] **(now)** Step P0 远端核查 sf_gap_norm（不需 GPU，~30 min）
- [ ] Step P1 代码层修 schedule_origin（< 30 min）
- [ ] Step P2 print 行加诊断字段（5 min）
- [ ] Step P3 smoke test（10 min）
- [ ] 写 v3.1 B' / C' 两个 config（< 1h，max_steps 留占位符）
- [ ] commit + push 代码 + configs
- [ ] **等 200K v3 完成**（best 不再更新 or step=200K）
- [ ] 核查 200K v3 best.pt 的 step 和 val_score，**回填 config 中的 max_steps**
- [ ] Path A redo on 200K v3 best（多 ckpt：30K/100K/150K/best）
- [ ] 启动 B' / C'（起点 = 200K v3 best.pt）
- [ ] +30 min / +6h / +24h 三档监控
- [ ] B'/C' 完成后全 val 评估 + 视觉
- [ ] v3.1 结果总结文档
