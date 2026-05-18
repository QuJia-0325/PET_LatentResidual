# Codex Task — Stage C v3 V14 launch (slot 3, multi-seed d_pure baseline)

- date: 2026-05-18
- branch: foc_lite_hop0
- status: **READY FOR CODEX EXECUTION** (Round 15 user 决策 1=B 2=B 3=A 4=B)
- 触发: R13 user 决策 A (3 slot 含 V14) + R14 A3 prep 完成 + R15 4/4 reviewer 共识 slot 3 = V14
- 设计来源: [REVIEW_INTEGRATION_round15_20260518.md](./REVIEW_INTEGRATION_round15_20260518.md) §4 (staggered launch) + §5 (V14 task md required)
- 硬约束: ≤ 3 并行训练任务. 本 task 占 slot 3, 与 slot 1 (A3 capacity-only) + slot 2 (V13) 并行
- **supersede 声明**: 本 task supersede [CODEX_TASK_PHASE_A_v3_20260518.md](./CODEX_TASK_PHASE_A_v3_20260518.md) line 9 "Slot 3 永远 0" 约束 (R15 B56 fix: v3 该行属 v3 内部 scope, R13 user 决策 A 含 V14 但 v3 起草时 V14 未 ready, 现在 ready, slot 3 启用)

---

## §0 — 给 codex 的 5 句话总览

1. **0 训练器代码改动, 0 V18/V13 yaml 改动**. V14 yaml 已 ready ([V14_v7_seed1337.yaml](../0516/V14_true_d_pure/V14_v7_seed1337.yaml)), 直接 launch.
2. 目标: 真 d_pure noise floor 测量 (V7 - V14 NORMAL Δ ≈ ±0.01-0.03 dB), paper 所有 ΔPSNR 显著性判定的 noise denominator.
3. **R15 user 决策 staggered launch (Q2=B)**: V14 在 A3 + V13 launch 后 30 min 启动 (IO health check 通过后), 不与 A3/V13 同时秒级 launch.
4. V14 = V7 config + **seed=1337** (从 42 改), 其他全同 V7. from-scratch 160K step, ~7 天.
5. 完成后用 V7 同 evaluator (`eval_first_hop_fullval_psnr_chain_mse.py`) 测 NORMAL Δ, 写 V14_FINAL_RESULTS report. **不直接判断 V18 framework 命运** (那是 R16 战略 review 范围).

---

## §1 — 物理含义 (user 必读)

### 1.1 d_pure noise floor 定义

`d_pure(V7) = |PSNR(V7@seed=42) - PSNR(V7@seed=1337)|` on NORMAL chain.

**含义**: 相同 V7 config 在不同 random seed 下, 训出来的 ckpt 在同 evaluator 测出的 PSNR 差异. 这是**纯 noise**, 不是任何 design 改进.

### 1.2 d_pure 是所有 ΔPSNR 显著性的 denominator

paper 写 "V18 比 V7 提升 +X dB" 时, 是否 statistically significant 取决于 X 是否远大于 d_pure noise:
- V18.best vs V7.best NORMAL Δ = +0.030 dB
- V18.last vs V7.best NORMAL Δ = +0.062 dB
- V13 outcome 待测 (期望 V7-V13 = 真 image_aux 贡献, 量级未知)
- A3 V18-capacity-only 待测 (期望 cap vs V18 ≈ 0 dB if B 候选真)

**没 d_pure 数字, 所有 ΔPSNR claim 都没显著性 baseline**. e.g. +0.030 是否在 noise 内? 假设 d_pure=0.02 → +0.030 是 marginally significant; 假设 d_pure=0.05 → +0.030 不显著.

→ V14 是 paper 必备 noise floor, 不是 "机会主义加跑" (R15 B57 fix: 之前 mislabel).

### 1.3 与 A3 / V13 完全独立

| 维度 | A3 | V13 | V14 |
|---|---|---|---|
| 起点 | V7 best.pt resume | from-scratch | from-scratch |
| seed | 42 (与 V7 同) | 42 (与 V7 同) | **1337** (变量) |
| step_weights | Grönwall (V7 同) | Grönwall (V7 同) | Grönwall (V7 同) |
| image_aux | on (V7 同) | **off** (变量) | on (V7 同) |
| decoder LoRA | rank=32, lambda_kl=0 | 无 | 无 |
| max_steps | 170K | 160K | 160K |

3 个实验**变量隔离**, 互相不干扰 outcome 解读.

---

## §2 — Task A: V14 launch (slot 3, after A3+V13 staggered)

### A0: 前置确认 + IO health check (Round 15 B58 fix)

**触发条件**: A3 + V13 launch 至少 30 min, 两者 process alive + GPU active.

```bash
cd /home/qujiaxiang/project/PET_LatentResidual
git pull --ff-only gitee foc_lite_hop0

# V14 yaml 必须存在
test -f review/0516/V14_true_d_pure/V14_v7_seed1337.yaml && echo "V14 yaml OK" || { echo "FAIL: V14 yaml missing"; exit 1; }

# 验证 V14 yaml 关键字段
python3 << 'PY'
import yaml
cfg = yaml.safe_load(open('review/0516/V14_true_d_pure/V14_v7_seed1337.yaml'))
assert cfg['seed'] == 1337, f"V14 seed should be 1337, got {cfg['seed']}"
assert cfg['training']['max_steps'] == 160000, f"V14 max_steps should be 160000"
assert cfg['training'].get('resume_from', '') == '', "V14 应为 from-scratch (无 resume)"
assert cfg['loss']['image_aux']['enabled'] == True if 'enabled' in cfg['loss'].get('image_aux', {}) else True, "V14 image_aux 应 ON"
print("V14 yaml verified ✓")
PY

# === Round 15 B58 IO health check ===
echo "=== GPU memory before V14 launch ==="
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv

echo "=== disk IO contention check (3 task 共享 latent_dir) ==="
iostat -x 1 5 | tail -20  # 5 秒采样

echo "=== A3 + V13 process alive check ==="
A3_PID=$(cat review/0517/V18_capacity_only/V18_capacity_only_train_*.pid 2>/dev/null | tail -1)
V13_PID=$(cat review/0516/V13_true_image_aux_ablation/V13_train.pid 2>/dev/null | tail -1)
test -n "$A3_PID" && ps -p $A3_PID > /dev/null && echo "A3 alive (PID=$A3_PID) ✓" || echo "WARN: A3 not alive, check before V14 launch"
test -n "$V13_PID" && ps -p $V13_PID > /dev/null && echo "V13 alive (PID=$V13_PID) ✓" || echo "WARN: V13 not alive, check before V14 launch"

# Pass 准则 (全部满足才能 launch V14):
# - GPU memory.free > 12 GB on slot 3 GPU
# - disk %util < 80% sustained
# - A3 + V13 both alive
# 任一 fail → 不 launch V14, 报告 + 等 user
```

### A1: V14 launch (slot 3)

```bash
LAUNCH_TS=$(date +%Y%m%d_%H%M%S)
LAUNCH_LOG=review/0516/V14_true_d_pure/V14_train_${LAUNCH_TS}.log

SLOT3_GPU=<填 nvidia-smi 空闲 GPU id, 不要与 A3 slot 1 或 V13 slot 2 重叠>

export CUDA_VISIBLE_DEVICES=$SLOT3_GPU

# V14 from-scratch, **不需要 --resume**
nohup python train_first_hop.py \
  --config review/0516/V14_true_d_pure/V14_v7_seed1337.yaml \
  > "$LAUNCH_LOG" 2>&1 &
V14_PID=$!

echo "V14 launched: PID=$V14_PID, log=$LAUNCH_LOG"
echo "$V14_PID" > review/0516/V14_true_d_pure/V14_train.pid
```

### A2: launch 后 5 min 验证

```bash
sleep 300
ps -p $V14_PID > /dev/null && V14_ALIVE=1 || V14_ALIVE=0
echo "V14 alive at +5min: $V14_ALIVE"

# 验证 GPU 真在用
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | grep "^$SLOT3_GPU,"

# 验证 V14 effective seed=1337 (R13 reviewer3 evidence-required: grep effective config)
grep -E "seed: ?1337|seed=1337|effective seed.*1337" "$LAUNCH_LOG" | head -3

# 验证 V14 不出现 'Resuming from' (from-scratch)
grep -E "Resuming from|resume from:|resume.*loaded" "$LAUNCH_LOG" | head -3
# 期望: 不出现 (V14 yaml resume_from 为空)
```

### A3: pass 准则 (5 条, Round 14 standing rule B47)

```bash
ERR=0
check() { local desc="$1"; shift; if "$@" >/dev/null 2>&1; then echo "✓ $desc"; else echo "✗ $desc"; ERR=$((ERR+1)); fi; }

check "V14 process alive at +5min" test "$V14_ALIVE" = "1"
check "GPU memory > 1 GB on slot 3" bash -c "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n '$((SLOT3_GPU+1))p' | awk '\$1 > 1000'"
check "V14 effective seed=1337 in log" grep -qE "seed: ?1337|seed=1337" "$LAUNCH_LOG"
check "V14 NOT resuming from any ckpt" bash -c "! grep -qE 'Resuming from|resume from:' '$LAUNCH_LOG'"
check "No CUDA error in log" bash -c "! grep -qE 'CUDA error|OOM|RuntimeError' '$LAUNCH_LOG'"

if [ $ERR -gt 0 ]; then echo "FAIL: $ERR/5 checks failed. KILL V14, report." ; exit 1; fi
echo "PASS: 5/5 alive checks. V14 in slot 3, run for ~7 days."
```

### A4: fail 处理

任一 fail → kill V14 process, 写 `V14_LAUNCH_FAILURE_REPORT_<TS>.md`, **不 launch retry**, 等 user.

---

## §3 — Task B: V14 完成后 eval (~7 天后)

V14 训到 step 160K (`last.pt`), 用同 V7 evaluator:

```bash
TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR=review/0516/V14_true_d_pure/fullval_eval_${TS}
mkdir -p "$OUT_DIR"

python review/0505/operator/scripts/eval_first_hop_fullval_psnr_chain_mse.py \
  --config review/0516/V14_true_d_pure/V14_v7_seed1337.yaml \
  --checkpoint <V14 last.pt 绝对路径> \
  --out-dir "$OUT_DIR" \
  --split val \
  --max-slices 0 \
  --tag v14_seed1337_last \
  2>&1 | tee "$OUT_DIR/eval.log"
```

### B2: 计算 d_pure

写 `V14_FINAL_RESULTS_<TS>.md`:

```markdown
# V14 d_pure Baseline Final Results

## Protocol
- evaluator: review/0505/operator/scripts/eval_first_hop_fullval_psnr_chain_mse.py (与 V7/V18 同)
- metric: calc_psnr_clip3, n=7403, full val
- V14 step: 160000 (from-scratch, seed=1337)

## V14 vs V7 (paired d_pure)

| chain | V7.best (160K, seed=42) | V14.last (160K, seed=1337) | d_pure = |V7 - V14| |
|---|---:|---:|---:|
| D20 | 35.4354 | <X.X> | <X.X> |
| D10 | 35.8194 | <X.X> | <X.X> |
| D4 | 36.3736 | <X.X> | <X.X> |
| **NORMAL** | **36.7810** | **<X.X>** | **<X.X>** |

## d_pure noise floor 应用

- V14 NORMAL d_pure = <X.X> dB → 项目 paired seed noise 上限
- 任何 ΔPSNR < 2× d_pure 在统计上 marginal (低显著性)
- 任何 ΔPSNR ≥ 3× d_pure 在统计上显著 (high confidence)

## 对历史 ΔPSNR 重新校准

| 比较 | ΔPSNR | 相对 d_pure | 显著性 |
|---|---:|---:|---|
| V18.best vs V7.best NORMAL | +0.030 | <X.X>× | <marginal/insignificant/significant> |
| V18.last vs V7.best NORMAL | +0.062 | <X.X>× | <…> |
| V7 vs V8 (Plan F) NORMAL | +0.308 | <X.X>× | <highly significant> (paired t=74.6 已知) |
| (V13 outcome 出来后填) | | | |
| (V18-cap outcome 出来后填) | | | |
```

---

## §4 — Task C: commit + push

```bash
git add review/0516/V14_true_d_pure/V14_train_*.log
git add review/0516/V14_true_d_pure/V14_train.pid  # 暂时, R14 NOT-DO 排除 .pid 但 V14 launch report 用作 reference
git add review/0516/V14_true_d_pure/fullval_eval_*/

# 7 天后 V14 跑完时 commit final results
git status
git diff --cached --stat

git commit -m "Stage C v3 V14 launch (slot 3, multi-seed d_pure baseline)

- yaml: review/0516/V14_true_d_pure/V14_v7_seed1337.yaml (已有, V7 + seed=1337)
- from-scratch 160K step, ~7 days, slot 3
- 与 V7 同 evaluator (eval_first_hop_fullval_psnr_chain_mse.py) 测 NORMAL d_pure
- supersede CODEX_TASK_PHASE_A_v3 line 9 'Slot 3 永远 0' (R15 B56 fix)
- 与 A3 (slot 1 capacity-only) + V13 (slot 2 image_aux ablation) 并行
- Round 15 B58 fix: staggered launch (A3+V13 T=0, V14 T=+30min after IO health check)
- 来源: REVIEW_INTEGRATION_round15 §4-5 + R13 user 决策 A (3 slot 含 V14)"

git push gitee foc_lite_hop0
```

---

## §5 — NOT-DO (此 task)

1. ❌ 不动 V7 best.pt / V18 ckpt / V13 ckpt
2. ❌ 不改 train_first_hop.py / V14 yaml / V13 yaml / V18 yaml 任何代码
3. ❌ 不 launch V14 在 A3 + V13 launch **之前** 或 **同时** (必须 staggered, T=+30min after IO health check, R15 B58)
4. ❌ 不直接判断 V14 outcome 对 V18 framework 命运的含义 (R16 战略 review 范围)
5. ❌ 不顺手 release audit DRAFT
6. ❌ smoke 步骤 fail 时不 `rm -rf` 重跑
7. ❌ 在 commit message 写 "也顺便做了 X"
8. ❌ 不起 Round 16 review (由 user 决定 V14 完成后)
9. ❌ 不修改 V14 yaml (本 task 已 ready, 与 A3 用同 V7 系列字段命名空间)
10. ❌ 不重命名 / 移动 V7 best.pt 绝对路径 (V14 是 from-scratch 不依赖 V7 ckpt, 但 paper d_pure 比较需 V7 ckpt 保留)

---

## §6 — 失败决策树

```
A0 IO health check fail (GPU memory < 12GB / disk %util > 80%)?
  → 不 launch V14. 报告 + 等 user 决策 (是否 reduce A3/V13 batch_size / 暂缓 V14 等 A3 完成)

A1 launch fail (yaml error, OOM, etc.)?
  → kill V14, 写 V14_LAUNCH_FAILURE_REPORT.md, 等 user. 不修 yaml.

A3 pass 准则任一 fail?
  → kill V14, 报告, 等 user.

7 天训练中途 NaN/divergence?
  → 训练循环 watchdog 自然 exit. V14 last.pt 仍可能存在, 报告状态等 user.

B1 eval fail (ckpt loading error)?
  → 检查 ckpt 路径 + config match, 报告. 不修 eval script.

C push 网络/认证 fail?
  → 重试 1 次, 失败保留本地 commit + 报告.

任何步骤遇到 "看起来应该有的 CLI flag / 文件 / 字段 不存在"?
  → 停, 报告. 不杜撰 fallback (B21 守).
```

---

## §7 — 总时长估计

| Task | 时长 |
|---|---|
| A0 IO health check + V14 launch prep | 5 min (在 A3+V13 launch 后 30 min 触发) |
| A1-A3 launch + 5 min verify | 10 min |
| 160K step 训练 (from-scratch) | **~7 天** |
| B1-B2 eval + report | 30 min |
| C commit + push | 5 min |
| **总** | **~7 天** (与 V13 同步完成, 与 A3 48h 之间错位) |

GPU 占用: slot 3 × 7 天. 与 slot 1 (A3 48h then idle) + slot 2 (V13 7 天) 并行.

---

## §8 — R15 standing rules 引用

本 task md 起草遵守:
- **B45** (substrate reviewer 必介入): V14 task md 已经 R15 reviewer 共识 verify
- **B46** (4×2 ΔPSNR 标签): §B2 report 模板 4 chain × seed=42 vs seed=1337 完整列
- **B47** (mechanical script assert): §A3 5 条 check 用 ERR 计数 + exit 1
- **B55** (prompt 引用 user 决策必须 grep verify): R15 §1 引 R13 §6 user 决策 A 含 V14, grep verify ✓ at REVIEW_INTEGRATION_round15 §1.2
- **B56** (supersede 显式声明): §0 header 显式 supersede CODEX_TASK_PHASE_A_v3 line 9 "Slot 3 永远 0"
- **B58** (IO contention 防御): §A0 IO health check before launch
- **B60** (新立 R15, 待写进 design_rationale §5.2): 本 task §1 引用 R13 user 决策 = "Round 13 §6 user 决策 1=A (3 slot 同时 V18-cap + V13 + V14)"
