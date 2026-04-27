# Reviewer Notes — 0427 Run-Me 实验设计审查

**审查日期**: 2026-04-27
**审查对象**:
- `review/0427/v5_rollout_heavy_analysis_20260427.md`（Claude 主分析）
- `review/0427/v5_rollout_heavy_counter_review_20260427.md`（Codex 反审）
- `review/0427/run_me/`（README + 4 个 shell + null-control config）
**立场**：基本同意 counter-review 的修正；run_me 的设计**方向正确**，但执行前必须修 4 处坑，否则结论仍会带泥。

---

## 1. 对两份分析的整体判断

**接受 counter-review 的修正**：

| Claude 强结论 | Counter-review 反驳 | 我的判断 |
|---|---|---|
| step 93200 是相变点 | rolling-val window 92800 是尾部 partial（54 batches @ start=3648），93200 是头部 full（64 @ start=0），不是同分布 | ✅ 接受。"跳变" 是 window 切换的伪信号 |
| V4/V5 同步崩溃 = 窄盆地 | V4/V5 共享 eval schedule，同步是必然 | ✅ 接受。"窄盆地" 假说证据不足 |
| V4 SF ≡ V5 rollout（同款失败） | V4 是 detached 单步 pred-state；V5 是 alpha=1.0 多步 BPTT，梯度路径完全不同 | ✅ 接受。强制类比掩盖了机制差异 |
| val_pair_total 跳变 = 不可逆推出 manifold | 后续回落到 1e-5 量级，可逆 | ✅ 接受。不能用 rolling-val 单点定义不可逆 |
| V4/V5 logged val_select 可直接比较 | best_metric_terms D20 权重不同（0.15 vs 0.50） | ✅ 接受。必须统一 objective 重算 |

**保留的弱结论**（双方都同意）：
- V5 当前 rolling-val 数据上**没有可靠改善证据**
- V5 的 tail 11.9× effective coefficient（hop3）可能损害 hop0/GT one-step anchor
- 后续判断必须 full-val 或 fixed-window，不能继续用 rolling-val 单点

---

## 2. run_me 设计评价

| 实验 | 评价 |
|---|---|
| 01 full-val eval | ✅ 直接回应 counter-review §7.1 第 1 项 |
| 02 V3 Path A baseline | ✅ 必要；旧 6.32 dB 来自 Scheme C，不可复用 |
| 03 null-control | ✅ 设计精确；分离 "loss 改动" vs "resume + LR" 的关键对照 |
| 04 V5 Path A | ✅ 即便 PSNR 持平，ExpoGap 改善也是机制证据 |

**总评**：4 个实验各自回答一个独立问题，正交性好。判定矩阵（README §"判定矩阵"）逻辑清晰。**前提是配置和评估协议没有混淆**——这正是下面要修的 4 个坑。

---

## 3. 必须修的 4 处问题

### 问题 1（最关键）：Null-control 的 LR cosine 被 `max_steps` 压缩

#### 3.1 现象

[03_null_control.sh L62](../../../scripts/../review/0427/run_me/03_null_control.sh) 使用 `sed` 把 null-control config 的 `max_steps` 改写为 `resume_step + 50000 ≈ 136800`。但 [pet_flow_first_hop_224_v5_null_control.yaml](../../../configs/pet_flow/pet_flow_first_hop_224_v5_null_control.yaml) 继承自 200K v3，原本 `max_steps=200000`、`lr_schedule.warmup_ratio=0.05`。

代码里 LR 由 [train_first_hop.py L1796 `get_warmup_cosine_lr(step, max_steps, ...)`](../../../train_first_hop.py#L1796) 直接用 `max_steps` 计算 cosine progress：

```python
progress = float(step - warmup_steps) / float(max(max_steps - warmup_steps, 1))
cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
return min_lr + (base_lr - min_lr) * cosine
```

这意味着把 `max_steps` 从 200000 改成 136800 后，**step 86800 时的 LR 从 baseline 的 ~6.04e-5 变成 ~3.25e-5（约 54%）**——这正是 V5-main 跑的 LR 曲线。

#### 3.2 后果

null-control 的目的是：在**相同 LR 曲线下**只改 loss 看影响。如果 null-control 用 V5-main 的压缩 LR 曲线，那么：

- null-control 退化 → **无法区分**是 "短训不收敛" 还是 "resume 本身有问题"
- null-control 持平 → 也**无法证明** "V5 的 loss 改动是退化主因"，因为 baseline 在压缩 LR 下原本就持平

**这个坑直接让 03 的判定矩阵作废**。counter-review §7.2 优先级 5 已经点到，但实施层面 run_me 没修。

#### 3.3 修复方案（推荐：方案 A）

代码里**目前没有 `scheduler_total_steps_override`** 这个参数（我之前给的建议是错的）。要让 LR cosine 按 200K 而不是按 max_steps 走，必须二选一：

##### 方案 A（推荐）：3 行代码补丁，加 `lr_schedule.total_steps_override`

`train_first_hop.py` 仅修 2 处：

```python
# 修改 1: L1433 附近
lr_cfg = cfg.get("lr_schedule", {})
lr_sched_enabled = bool(lr_cfg.get("enabled", True))
# === 新增：允许 LR cosine 用与 max_steps 不同的总步数 ===
lr_total_steps = int(lr_cfg.get("total_steps_override", 0)) or max_steps
# === 新增结束 ===
lr_warmup_steps = _resolve_schedule_steps(
    total_steps=lr_total_steps,   # 改这里：原来是 max_steps
    section=lr_cfg,
    steps_key="warmup_steps",
    ratio_key="warmup_ratio",
    default_steps=4000,
)
```

```python
# 修改 2: L1796 附近
if lr_sched_enabled:
    lr_now = get_warmup_cosine_lr(
        step=step,
        max_steps=lr_total_steps,   # 改这里：原来是 max_steps
        base_lr=base_lr,
        warmup_steps=lr_warmup_steps,
        min_lr=lr_min,
    )
```

null-control yaml 加：

```yaml
lr_schedule:
  enabled: true
  warmup_ratio: 0.05
  min_lr: 2.0e-6
  total_steps_override: 200000   # ★ 关键：cosine 仍按 200K 计算
```

**好处**：仅 2 处代码改动，向后兼容（不设 override 时行为不变），物理含义清晰。

**注意**：V5-main 的 launch script 也建议同步加 `total_steps_override: 200000`，否则 V5-main 自己也是在压缩 LR 曲线下评估的——这不是 V5 失败的主因，但**会成为 V5 失败分析的混淆变量**。

##### 方案 B（零代码改动，但有副作用）：把 max_steps 设为 200000，靠 save_interval + 监控手动停

null-control yaml：
```yaml
training:
  max_steps: 200000           # 与 v3 baseline 一致
  save_interval: 5000         # 每 5K 步存 ckpt
```

启动后跑到 step 136800（即新增 50K 步）时**手动 kill -9**。

**坏处**：
- 容易忘记 kill，浪费 GPU
- 没有自动 final ckpt（需要 ckpt 用 step_*.pt 而非 best.pt 作为 final 评估对象）
- 同样的问题在 V5-main 已经发生过，不建议复制这个模式

**结论**：用方案 A。改动成本极低，效益清晰。

---

### 问题 2：01 full-val 用 V3 config eval V3 ckpt、V5 config eval V5 ckpt

[01_fullval_eval.sh L23-25](../../../scripts/../review/0427/run_me/01_fullval_eval.sh) 给 V3 和 V5 的 ckpt **用了不同 config**。但两份 config 的 `best_metric_terms` D20 权重分别是 0.50 和 0.50（V3）vs 0.50（V5——已与 V3 对齐）——**实际上 V5 已经统一过了**，这点比 V4 好。

但 **eval 输出的 select_score 仍然取决于 eval 时的 config**。如果 [eval_first_hop_224_clip3.py](../../../eval_first_hop_224_clip3.py) 内部按 config 的 best_metric_terms 算 select，**V3/V5 比较 select 仍然安全**（D20 权重都是 0.50）。

**但要警惕**：如果未来加 V4 ckpt 进 full-val 对比（V4 D20=0.15），这一坑会复活。

#### 修复建议（轻量）

为防止后续混淆，把 01 改成**所有 ckpt 用同一份"统一评估 config"**：

```bash
# 统一用 V3 config 做 eval（D20=0.50, 与 V5 一致）
EVAL_CONFIG="configs/pet_flow/pet_flow_first_hop_224_200k_transport_v3.yaml"

for ckpt in V3_BEST V5_BEST V5_LATEST; do
    ${PYTHON} eval_first_hop_224_clip3.py \
        --config "${EVAL_CONFIG}" \
        --checkpoint "$ckpt" \
        --split val --max-slices 0 ...
done
```

注意 V5 config 与 V3 在 model 结构上是否一致（backbone / first_hop 部分）。如果一致，统一 eval config 是安全的。

#### 备用建议

不直接看 select_score；只看 per-hop chain MSE / chain PSNR。这些是绝对量，不受 best_metric_terms 影响。判定矩阵以 `chain_normal_psnr` 和 `transport_avg_psnr` 为主，select 只作辅助。

---

### 问题 3：判定矩阵的 0.3 dB 阈值缺乏 noise floor 参考

[README.md "Full-Val（01）结果" 表](../../../scripts/../review/0427/run_me/README.md) 用 ±0.3 dB 作为持平/改善/恶化的边界。这个数字是凭直觉的——可能合理，也可能太严或太松。

#### Counter-review §1.3 给的同 objective 重算数据

| step | window | V4 common | V5 common | delta |
|---:|---:|---:|---:|---:|
| 92000 | 3520 | 0.000832 | 0.000792 | -0.000040 |
| 92400 | 3584 | 0.000722 | 0.000693 | -0.000029 |
| 92800 | 3648 | 0.000614 | 0.000591 | -0.000023 |

平均 delta ≈ -3e-5（select 单位）。如果 select ≈ 1.5 × MSE，则对应 MSE 改善 ≈ -2e-5。
NORMAL 档 baseline MSE ≈ 1.7e-4 → 改善 12% → PSNR 改善 ≈ **0.55 dB**。

但这是 rolling-val 上的；**full-val 的 std 可能更小**（因为样本数 7403 远大于 512）。具体 noise floor 必须实测。

#### 修复建议

01 实验加一步：**对 V3 best.pt 做 5-fold full-val**（把 7403 slices 划成 5 份分别 eval），算 chain_normal_psnr 的 std。

```bash
# 简易实现：用不同 --max-slices --skip-slices 控制起点
for fold in 0 1 2 3 4; do
    ${PYTHON} eval_first_hop_224_clip3.py \
        --config "${V3_CONFIG}" --checkpoint "${V3_BEST}" \
        --split val --max-slices 1480 --skip-slices $((fold * 1480)) \
        --out-dir "${OUT_BASE}/v3_fold${fold}"
done
```

如果 eval 脚本不支持 `--skip-slices`，**可以省略 5-fold**，但**判定矩阵的 0.3 dB 阈值要降级为"经验值，需结合 V3 baseline 自身波动重新校准"**。

**最低要求**：判定矩阵中的 0.3 dB 在结论文档里**必须明确是经验阈值而非统计意义阈值**，避免后续误读。

---

### 问题 4：缺一个固定窗口快速 rerank

counter-review §7.1 第 2 项建议加 fixed-window eval：在**同一个 64-batch 子集**上 eval 多个 ckpt，作为 full-val 的低成本快速验证。

#### 价值

- full-val 单 ckpt 约 20 min；3 ckpt × 4 实验 = ~4h
- fixed-window eval 单 ckpt 约 1 min；3 ckpt × 1 = ~3 min
- 两者**应该方向一致**——如果不一致，说明 full-val 也有未注意到的混淆

#### 修复建议（可选）

`train_first_hop.py` 的 [`_resolve_eval_window`](../../../train_first_hop.py#L737) 现在只支持 `head` / `rolling`。可以在 [eval_first_hop_224_clip3.py](../../../eval_first_hop_224_clip3.py) 里加 `--fixed-window-start <int>`，取固定起点的 64 batches。

**但**：如果时间紧张，**这一项可以不做**——full-val 已经是金标准，fixed-window 只是节省时间。建议作为 P2。

---

## 4. 修复后的执行顺序（替代 README 的）

### Step 0：代码改动（5 分钟）

按问题 1 方案 A 修 `train_first_hop.py` 2 处。

### Step 1：Config 改动

```yaml
# pet_flow_first_hop_224_v5_null_control.yaml
lr_schedule:
  enabled: true
  warmup_ratio: 0.05
  min_lr: 2.0e-6
  total_steps_override: 200000   # ★ 新增
```

```yaml
# 推荐：pet_flow_first_hop_224_v5_rollout_heavy.yaml 同样加（V5-main 重跑也建议加）
lr_schedule:
  ...
  total_steps_override: 200000   # 让 V5 LR 与 baseline 一致，分离 LR 与 loss 影响
```

### Step 2：sanity check

启动 null-control 前，**先 dry-run 5 步**确认 step 86800+1 时 LR 是 ~6e-5 而非 ~3e-5：

```bash
# 短训证 LR 曲线
CUDA_VISIBLE_DEVICES=0 python -u train_first_hop.py \
    --config configs/pet_flow/pet_flow_first_hop_224_v5_null_control.yaml \
    --resume <V3_BEST> 2>&1 | grep -E "lr_schedule|lr=" | head -5
```

### Step 3：执行 P0/P1

```bash
# P0（~1.5h）
bash review/0427/run_me/01_fullval_eval.sh 0    # ⚠️ 修问题 2 后再跑
bash review/0427/run_me/02_pathA_baseline.sh 0

# P1（~30h）
bash review/0427/run_me/03_null_control.sh 2    # ⚠️ 修问题 1 后再跑
bash review/0427/run_me/04_pathA_v5_best.sh 0
```

### Step 4：判定（修订版矩阵）

| full-val V5 vs V3 | null-control 50K | 诊断 |
|---|---|---|
| V5 改善 ≥ 0.3 dB | 任意 | V5 真实有效，rolling-val 掩盖了 |
| V5 持平 | null 持平 | V5 loss 改动无害但也无益；是否继续看 04 ExpoGap |
| V5 持平 | null 改善 | **V5 loss 改动伤害了模型**（被 null-control 揭露）|
| V5 退化 | null 持平 | V5 loss 改动是退化主因 |
| V5 退化 | null 退化 | resume 本身有问题（与 LR 解耦后仍退化）→ 需要排查 dataloader / EMA / 其他 resume bug |

**关键**：null-control 改善的情况是新加的——没有 LR 解耦时不可能出现，解耦后才有意义。

---

## 5. 额外建议：补一个 V5-no-stepweights 消融

如果 01/03 的结论是 "V5 退化、null-control 持平"，那么需要进一步定位**是 lambda_roll 6× 错了，还是 step_weights 反转错了，还是两者叠加**。

最便宜的消融：

```yaml
# pet_flow_first_hop_224_v5_no_stepweights.yaml
# V5 但 step_weights 保持 v3 头重 [1.30, 1.20, 1.10, 1.00]
rollout:
  lambda_start: 1.50            # 与 V5-main 一致
  step_weights: [1.30, 1.20, 1.10, 1.00]   # ★ 改回 v3 头重
image_aux:
  lambda_start: 0.08            # 与 V5-main 一致
```

跑 50K 步。如果**这个版本不退化**——说明退化的元凶是 step_weights 反转（tail 11.9×），不是 lambda 6×。这直接指向 counter-review §7.2 优先级 4 的"detach/teacher variant"方向。

**但**：这一项放在 P2，等 01/03 结论出来再决定是否启动。

---

## 6. 一句话总结

> run_me 设计正确——逻辑上覆盖了 counter-review 的所有反驳。但**问题 1（LR 压缩耦合）不修则 03 的判定矩阵作废**，必须先打 2 行代码补丁加 `lr_schedule.total_steps_override`，再跑 null-control。其余 3 个问题（统一 eval config、noise floor 校准、fixed-window）属于让结论更严谨，可与主流程并行。
