# Transport 突破实验计划 v4 — 2026-04-26

**Supersedes**: v3 + v3.1 redo plan  
**综合输入**: v3 实验结果分析、v3.1 设计、subagent review、Codex GPT-5.4 xhigh review  
**约束**: 仅 1 个实验可同时跑（RAM ~503GB，200K v3 占 ~190GB）

---

## 0. 已确认的事实

| 事实 | 来源 | 状态 |
|------|------|------|
| Exposure bias 是核心瓶颈（gap=4.80 dB，单调上升） | Path A 多 ckpt（v3 实验 A） | ✅ 强确认 |
| v3 B/C 结果不可信（SF schedule bug + 仅 3600 步） | v3 结果分析 | ✅ 确认 |
| `591d34c` 代码修复（schedule_origin）正确 | subagent + Codex 双重确认 | ✅ |
| **SF-pair (scheduled sampling) 方法已证伪** | v4 pilot 实验（alpha≥0.15 后单调恶化） | ❌ **已终止** |
| 200K v3 仍在跑，best 仍在更新 | 服务器状态 | ⏳ 进行中 |
| Rolling val 不可用于 claim 级别对比 | Codex S2 | ⚠️ 需 full-val |
| 只有 1 个 GPU 能用于新实验（RAM 限制） | 硬件约束 | 🔒 |

### SF-pair 证伪详情（2026-04-26 v4 pilot）

- 从 200K v3 best（step=86800）resume，50K 新增步，schedule_origin=resume_relative
- warmup 5K 步 + ramp 10K 步，在 alpha ≈ 0.10 时达到 best（-5.1% vs baseline）
- **alpha ≥ 0.15 后 val 单调恶化**：+74% @ alpha=0.18, +146% @ alpha=0.34
- sf_gap 不收敛（0.002-0.007 震荡），模型未学会消化 SF 信号
- 失败模式与 v3（schedule bug 下 alpha=1.0 的瞬间冲击）一致，证明是方法缺陷而非配置问题
- 详见 [review/0426/v4_sf_pilot_early_analysis_20260426.md](../0426/v4_sf_pilot_early_analysis_20260426.md)

---

## 1. 执行流程（严格串行）

```
Phase 0: 代码加固（本地，不需 GPU）
    ↓
Phase 1: Smoke test（GPU-0，~30 min）
    ↓ 通过
Phase 2: 等 200K v3 完成 → 取 best.pt → Path A redo
    ↓ 确认 exposure bias 仍成立
Phase 3: 10K pilot（GPU-0，~6h）
    ↓ go/no-go 判定
Phase 4: 50K 长跑（GPU-0，~30h）
    ↓
Phase 5: Full-val eval + Path A redo on B' best
```

---

## 2. Phase 0: 代码加固（P0 preflight + P2 per-hop 诊断）

### 2.1 Preflight guard（~20 行）

在 `train_first_hop.py` 的 resume 完成后、训练循环开始前，加硬性检查：

```python
# --- v4 preflight guard ---
sf_cfg = cfg.get("training", {}).get("self_forcing_pair", {})
if sf_cfg.get("enabled", False):
    effective_new_steps = max_steps - start_step
    min_resume_steps = int(sf_cfg.get("min_resume_steps", 10000))
    if effective_new_steps < min_resume_steps:
        raise RuntimeError(
            f"[preflight] SF enabled but effective new steps "
            f"({effective_new_steps} = max_steps {max_steps} - start_step {start_step}) "
            f"< min_resume_steps ({min_resume_steps}). "
            f"Increase max_steps or set min_resume_steps lower."
        )
    origin = str(sf_cfg.get("schedule_origin", "absolute")).lower()
    warmup = int(sf_cfg.get("warmup_steps", 5000))
    ramp = int(sf_cfg.get("ramp_steps", 10000))
    if origin == "absolute" and start_step >= warmup + ramp:
        raise RuntimeError(
            f"[preflight] SF schedule_origin=absolute but resume start_step "
            f"({start_step}) >= warmup+ramp ({warmup}+{ramp}={warmup+ramp}). "
            f"SF warmup/ramp will be skipped entirely. "
            f"Set schedule_origin: resume_relative or adjust warmup/ramp."
        )
    # Audit print
    if origin == "resume_relative":
        eff_0 = 0
        eff_warmup = warmup
        eff_ramp_end = warmup + ramp
    else:
        eff_0 = start_step
        eff_warmup = warmup
        eff_ramp_end = warmup + ramp
    print(
        f"[preflight][SF] origin={origin} | resume_start={start_step} | "
        f"effective_step_at_start={eff_0} | warmup_ends_at_eff={eff_warmup} | "
        f"ramp_ends_at_eff={eff_ramp_end} | new_steps={effective_new_steps}",
        flush=True,
    )
```

### 2.2 Per-hop gap_norm 诊断

`gap_norm` 被 hop_idx=0 样本稀释（hop0 的 z_src_pred ≡ z_src_gt，gap 恒为 0）。
在 `compute_self_forcing_z_src` 返回值中增加 per-hop 统计：

```python
# 在 return 之前：
gap_per_hop = {}
for h in range(num_hops + 1):
    mask = (hop_idx == h)
    if mask.any():
        gap_per_hop[h] = (z_src_pred[mask] - z_src_gt[mask]).abs().mean().item()
```

在 metrics_jsonl 和 stdout 中输出 `sf_gap_h0/h1/h2/h3`。

### 2.3 Schedule audit 打印

训练开始时打印一次完整的 schedule 审计：

```
[schedule-audit] rollout: alpha=1.0(fixed), lambda=0.25(fixed)
[schedule-audit] image_aux: lambda=0.12(fixed)
[schedule-audit] SF: origin=resume_relative, warmup=5000, ramp=10000, effective_step_0=0
[schedule-audit] LR: base=4e-5, warmup_steps=2500, cosine to min=2e-6 over max_steps
```

---

## 3. Phase 1: Smoke test（~30 min）

**目的**：验证 P1 代码修复确实生效，SF alpha 按预期调度。

### Smoke-A（验证 fix）

从 v3 旧 ckpt（step≈46400）resume，缩小调度参数：

```yaml
self_forcing_pair:
  enabled: true
  schedule_origin: resume_relative
  warmup_steps: 50
  ramp_steps: 100
  min_resume_steps: 100
training:
  max_steps: 46600  # +200 steps
  batch_size: 2
```

**期望**：
- step 46400→46450: `sf_alpha=0.000`（warmup 区间）
- step 46450→46550: `sf_alpha` 线性 0→1
- step 46550→46600: `sf_alpha=1.000`

### Smoke-B（复现旧 bug 作为 control）

同 ckpt，但 `schedule_origin: absolute`，其余同 Smoke-A。

**期望**：`sf_alpha=1.000` 从第 1 步起。

**判定**：两组对比通过 → Phase 1 PASS，继续；否则停下 debug。

---

## 4. Phase 2: 200K v3 完成 + Path A redo

### 4.1 等待 200K v3 完成

条件：`step == 200000` 或 `best 连续 20K 步不再更新`。

完成后记录：
```bash
V3_BEST=/data_2/qujiaxiang/outputs/PET_LatentResidual/<200K_v3_run_dir>/best.pt
python -c "import torch; c=torch.load('${V3_BEST}', map_location='cpu'); print('step=', c.get('step'), 'val=', c.get('best_val'))"
```

→ 记录为 `BASE_STEP`、`BASE_VAL`。

### 4.2 Path A redo on 200K v3（多 ckpt）

对 200K v3 的 4 个 checkpoint 跑 Path A 诊断：

| ckpt | 目的 |
|------|------|
| step_30000 | 早期参照 |
| step_100000 | 中期 |
| step_150000 | 后期 |
| best.pt | 最终 |

**关键判定**（Codex claims-matrix 场景 6）：
- 若 200K best 的 exposure_gap **明显下降**（< 3.0 dB） → exposure 假设在长训中不再成立，SF 方向可能不再必要
- 若 exposure_gap **仍 ≥ 4.0 dB 且单调不降** → 确认 SF 方向有价值，继续 Phase 3

记录为 `BASE_EXPGAP`、`BASE_CHAINMSE`。

---

## 5. Phase 3: 10K Pilot（~6h）

### 5.1 配置

只跑 **B'（SF-pair）**，不跑 C'（RAM 限制）。

```yaml
# configs/pet_flow/pet_flow_first_hop_224_v4_sf_pilot.yaml
run_name: first_hop_224_v4_sf_pilot
training:
  max_steps: <BASE_STEP + 10000>      # 待回填
  self_forcing_pair:
    enabled: true
    schedule_origin: resume_relative
    alpha_sf_start: 0.0
    alpha_sf_end: 1.0
    warmup_steps: 2000                 # 10K pilot 中 2K warmup
    ramp_steps: 3000                   # 2K-5K ramp，5K-10K 稳态
    min_resume_steps: 5000
    log_when_zero: true
  rollout:
    lambda_start: 0.25
    lambda_end: 0.25
    alpha_start: 1.0
    alpha_end: 1.0
  # 其余与 200K v3 config 保持一致
optimizer:
  lr: 4.0e-5
  backbone_lr_mult: 0.40
lr_schedule:
  warmup_ratio: 0.05
```

**注意**：pilot 的 SF warmup/ramp 比长跑短（2K/3K vs 5K/10K），因为 10K 预算有限，需要尽快看到 SF 稳态效果。

### 5.2 Go/No-Go 判定（基于 rolling val，仅用于趋势判断）

| 检查点 | 条件 | 动作 |
|--------|------|------|
| +1K（warmup 中） | val_select_score 退化 > 35% vs BASE_VAL | ⛔ 停，检查配置 |
| +5K（SF 稳态开始） | val_select_score 退化 > 15% vs BASE_VAL | ⚠️ 关注但继续 |
| +5K | `sf_alpha` 未达到 1.0 | ⛔ 停，schedule bug |
| +5K | `sf_gap_h1/h2/h3` 全为 0 | ⛔ 停，SF 替换路径 bug |
| +10K | val_select_score 仍 > BASE_VAL × 1.15 | ⛔ 不进入长跑 |
| +10K | val_select_score ≤ BASE_VAL × 1.05 且 sf_gap 下降趋势 | ✅ → Phase 4 |

### 5.3 Pilot 结束后 Full-val eval

不论 go/no-go，对 pilot best.pt 跑一次 full-val（`eval_first_hop_224_clip3.py --decode-mode both`）。
这是 **claim-safe** 的评估，用于和 200K v3 baseline 做精确对比。

---

## 6. Phase 4: 50K 长跑（~30h）

### 6.1 从 pilot best.pt resume

```yaml
# configs/pet_flow/pet_flow_first_hop_224_v4_sf_long.yaml
run_name: first_hop_224_v4_sf_long
training:
  max_steps: <PILOT_BEST_STEP + 50000>  # 待回填
  self_forcing_pair:
    enabled: true
    schedule_origin: resume_relative    # 锚定到 pilot best step
    alpha_sf_start: 1.0                 # pilot 结束时已是 1.0，不需要再 ramp
    alpha_sf_end: 1.0
    warmup_steps: 0
    ramp_steps: 0
    min_resume_steps: 30000
```

**关键**：长跑从 pilot best resume，此时 SF 已是稳态（alpha=1.0），不需要再 warmup。
直接跑 50K 步纯 SF 训练。

### 6.2 监控

| 时间 | 检查 |
|------|------|
| +6h (~10K) | val 趋势、sf_gap 趋势、grad_norm B' vs baseline 偏差 |
| +18h (~30K) | 若 val 仍比 baseline 差 > 10% → **止损** |
| +30h (~50K) | 完成 |

### 6.3 长跑结束后 Full-val eval

对 long-run 的 best.pt 跑 full-val + Path A redo。

---

## 7. Phase 5: 结果判定（Claims Matrix）

基于 Codex review 的 claims matrix，增加 RAM 约束后的简化版：

### 场景 1: B' full-val < BASE_VAL 且 exposure_gap 下降
→ **SF 有效**。下一步：跑 C'（Rollout-Up 对照）归因。

### 场景 2: B' full-val ≈ BASE_VAL（±3%）
→ **SF 无害但无增益**。下一步：检查 sf_gap 趋势——若 gap 在下降，延长到 100K；若 gap 不降，SF 方向终止。

### 场景 3: B' full-val > BASE_VAL × 1.05
→ **SF 有害**。下一步：检查 sf_gap 是否非零——若非零（SF 确实生效但有害），exposure bias 假设在此 regime 不适用；若 gap≈0（SF 从未生效），是实现 bug。

### 场景 4: 200K Path A redo 显示 exposure_gap < 3.0 dB
→ **长训已自然缓解 exposure bias**。SF 不再是必要干预。改写论文叙事为 "rollout-aware reweighting"。

---

## 8. 时间线

| Day | GPU-0 | GPU-1 | 产出 |
|-----|-------|-------|------|
| 0 | **Phase 0**: 代码加固（本地） | 200K v3 继续 | preflight guard + per-hop gap 代码 |
| 0 | **Phase 1**: Smoke A+B (~30 min) | 200K v3 继续 | "调度通过证据" |
| 1-2 | 等待 | 200K v3 继续 | — |
| 3 | **Phase 2**: Path A redo (4 ckpt) | 200K v3 **完成** | BASE_VAL/BASE_EXPGAP/BASE_CHAINMSE |
| 3 PM | **Phase 3**: 10K pilot 启动 | 释放 RAM | — |
| 4 AM | 10K pilot 完成 + full-val | — | go/no-go 判定 |
| 4 PM | **Phase 4**: 50K 长跑启动 | — | — |
| 5-6 | 长跑进行中（+18h 监控） | — | 中间止损判定 |
| 6 PM | 长跑完成 + full-val + Path A redo | — | 最终结果 |
| 7 | **Phase 5**: 结果判定 + 文档 | — | Claims / 论文方向决策 |

---

## 9. 代码改动清单

| 文件 | 改动 | 已完成？ |
|------|------|---------|
| `train_first_hop.py` | P1: `schedule_origin: resume_relative` | ✅ `591d34c` |
| `train_first_hop.py` | P2: stdout 加 `sf_alpha`/`sf_gap` | ✅ `591d34c` |
| `train_first_hop.py` | Phase 0: preflight guard | ❌ 待做 |
| `train_first_hop.py` | Phase 0: per-hop `sf_gap_h{k}` | ❌ 待做 |
| `train_first_hop.py` | Phase 0: schedule audit 打印 | ❌ 待做 |
| config: v4_sf_pilot.yaml | 10K pilot 配置 | ❌ 待做 |
| config: v4_sf_long.yaml | 50K 长跑配置 | ❌ 待做 |

---

## 10. 从 v3 吸取的教训（硬编码进流程）

1. **max_steps 必须是绝对值且 > resume_step + 预期新增步数**。每次 resume 前先 `torch.load(ckpt)['step']` 确认。
2. **任何写死 step 的 schedule，在 resume 路径必须声明锚点**。默认 `absolute`，resume 场景用 `resume_relative`。
3. **关键诊断字段必须打印到 stdout**，不能只写 metrics_jsonl。
4. **rolling val 仅用于趋势监控**。正式结论必须跑 full-val eval。
5. **10K pilot 先行**。只有 pilot 显示正向趋势才进入长跑。
6. **preflight guard 防止静默失败**。启动时检查 schedule 合法性和有效步数。
