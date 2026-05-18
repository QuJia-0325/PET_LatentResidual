# Codex Task — Stage C v3 (Round 13 A3): V18-r32-capacity-only

- date: 2026-05-18
- branch: foc_lite_hop0
- status: **READY FOR CODEX EXECUTION** (Round 13 user 决策 = A1/A3/A4)
- 触发: Round 12 KL drift inverse 发现 (V18 在 GT manifold 比 V7 高 +0.10 dB) + Round 13 reviewer3 B42 机制反驳 (use_pred_latent=true → KL pullback 跑在 z_pred 路径, 从未碰 decode(z_GT))
- 设计来源: [REVIEW_INTEGRATION_round13_20260518.md](./REVIEW_INTEGRATION_round13_20260518.md) §3 (Stage C v3 = E 扩展版 V18-capacity-only)
- 硬约束: V18 已完成 slot 1 free, V13 待 launch slot 2 (commit 7486776 待 push), 本 task 占 slot 1, ≤ 3 并行

---

## §0 — 给 codex 的 5 句话总览

1. **0 训练器代码改动, 0 V18 yaml 改动**. 只跑 1 个新 training task: V18-capacity-only (V7 best.pt resume + LoRA rank=32 + **lambda_kl=0** + 10K step).
2. 目标: disambiguate Round 13 §2.2 候选 A (KL 设计正面效果) vs B (LoRA capacity 副产品). B42 机制反驳后, A 候选机制上不可能, B 是默认; capacity-only 实测验证.
3. yaml 已准备: [review/0517/V18_capacity_only/V18_capacity_only.yaml](./V18_capacity_only/V18_capacity_only.yaml), 与 V18 yaml 仅 4 字段 diff (output_dir / run_name / max_steps=170000 / lambda_kl=0.0).
4. **绝对不**改 V18 yaml / train_first_hop.py / V7 best.pt / V18 现有 ckpt. **绝对不**自行 launch V13 (V13 launch 由 commit 7486776 单独 push 后再 codex 跑).
5. 完成后用 KL drift probe **(已就绪 tools/probe_v18_kl_drift.py)** 测 capacity-only ckpt 的 `PSNR(decode_cap(z_GT), x_target)`, 与 V7 / V18 比.

---

## §1 — 物理含义 (user 必读)

### 1.1 B42 机制反驳 (Round 13 reviewer3 grep 实证)

[train_first_hop.py:2230](../../train_first_hop.py):
```python
z_kl = main_out["z_pred"] if bool(kl_cfg.get("use_pred_latent", True)) else main_batch["z_dst"]
```

V18 yaml `use_pred_latent=true` → KL pullback `z_kl = main_out["z_pred"]` → KL pullback **从未碰过 decode(z_GT) 路径**. 

V18 在 GT manifold 上比 V7 高 +0.10 dB 的现象**不是 KL pullback 设计的正面效果**, 机制上必须有别的解释.

### 1.2 候选 B (B42 之后唯一机制上一致的解释 — 实测验证, 不预设结论) [Round 14 B54 fix]

V18 LoRA 589K trainable params 在 z_pred 路径吸 transport residual, 同时**通用增强 decoder capacity** (rank=32, 2.5% of decoder weights). 副作用渗到 z_GT 输入 → +0.10 dB 是 capacity 副产品.

### 1.3 capacity-only 实验设计

| 实验 | KL pullback | LoRA capacity | image_aux | 期望 GT manifold 改进 (按 B 候选) |
|---|---|---|---|---|
| V7 | 无 (frozen decoder) | 无 | 全开 | baseline (52.63 dB on NORMAL) |
| V18 (已跑) | on (lambda_kl=0.05, use_pred_latent=true) | 有 (rank=32) | 全开 | +0.10 dB (实测 52.73) |
| **V18-capacity-only (本 task)** | **off (lambda_kl=0)** | **有 (rank=32, 同 V18)** | **全开 (同 V18)** | **若 B 真: 仍 +0.10 dB / 若 A 真: 显著 < +0.10 dB** |

3 个可能结果:
1. **capacity-only ≈ V18 (+0.10 dB)** → B 候选真, KL pullback 设计无功能, +0.10 dB 是 LoRA capacity 副产品
2. **capacity-only ≈ V7 (0 dB)** → **+0.10 dB 不是 LoRA capacity 副产品** (B 候选 也不成立). 同时 B44 机制上 rule out 直接 KL→z_GT 对齐. 必有**间接机制** (e.g. KL on z_pred 梯度 → 共享 LoRA 权重 → 渗到 z_GT eval 路径). 开 sub-investigation 查具体间接路径, **不复活 A** (A 反驳未变) [Round 14 B53 fix]
3. **capacity-only ∈ (V7, V18)** → 混合 (capacity 部分 + KL 间接 bleed-through)
4. **[NEW] capacity-only < V7** → LoRA capacity 本身有害 (无 KL 约束时 LoRA 被 pair+rollout gradient 撑到损害 GT manifold 的方向), §2.2 B 候选需重读 "capacity 本身在 z_GT 上不是友好"; 不简单 reject A 也不简单 accept B

最可能 1 (机制层支持). 实测决定.

---

## §2 — Task A: Launch V18-capacity-only

### A0: 前置确认

```bash
cd /home/qujiaxiang/project/PET_LatentResidual
git pull --ff-only gitee foc_lite_hop0

# yaml 已就绪
test -f review/0517/V18_capacity_only/V18_capacity_only.yaml && echo "yaml OK" || { echo "FAIL: yaml missing"; exit 1; }

# 验证 yaml 4 字段
python3 << 'PY'
import yaml
cap = yaml.safe_load(open('review/0517/V18_capacity_only/V18_capacity_only.yaml'))
assert cap['output_dir'].endswith('V18_capacity_only/run'), 'output_dir wrong'
assert cap['run_name'] == 'first_hop_224_v18_capacity_only', 'run_name wrong'
assert cap['training']['max_steps'] == 170000, 'max_steps wrong'
assert cap['loss']['decoder_kl_pullback']['lambda_kl'] == 0.0, 'lambda_kl wrong'
assert cap['training']['decoder_lora']['rank'] == 32, 'rank wrong (should be 32, same as V18)'
assert cap['training']['decoder_lora']['last_n_blocks'] == 2, 'last_n_blocks wrong'
print('yaml 4 fields verified ✓')
PY

# V7 best.pt 必须存在 (resume_from 同 V18)
RESUME=/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V7/run/first_hop_224_v7_gronwall_raw/best.pt
test -f "$RESUME" && echo "V7 best.pt OK" || { echo "FAIL: V7 best.pt missing"; exit 1; }

# CLI flag verify (B30 standing rule)
SUPPORTED_FLAGS=$(grep "add_argument" train_first_hop.py | grep -oE "['\"]--[a-z_-]+['\"]" | sort -u | tr -d "'\"" | tr '\n' ' ' | sed 's/ $//')
EXPECTED="--config --resume"
if [ "$SUPPORTED_FLAGS" != "$EXPECTED" ]; then
    echo "FAIL: CLI flag set changed. Detected: '$SUPPORTED_FLAGS'. STOP."
    exit 1
fi
echo "CLI flags verified ✓"

# GPU 状态 (V18 已完成 slot 1 free, V13 未 launch)
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
```

### A1: launch

```bash
TS=$(date +%Y%m%d_%H%M%S)
LAUNCH_LOG=review/0517/V18_capacity_only/V18_capacity_only_train_${TS}.log

# 使用 slot 1 (V18 已完成 free); 若 slot 1 仍被占, 用 nvidia-smi 选 free GPU
SLOT1_GPU=<填 nvidia-smi free GPU id>

export CUDA_VISIBLE_DEVICES=$SLOT1_GPU

START_TS=$(date +%s)
echo "Launch START_TS: $START_TS"

nohup python train_first_hop.py \
  --config review/0517/V18_capacity_only/V18_capacity_only.yaml \
  --resume /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V7/run/first_hop_224_v7_gronwall_raw/best.pt \
  > "$LAUNCH_LOG" 2>&1 &
CAP_PID=$!

echo "V18-capacity-only launched: PID=$CAP_PID, log=$LAUNCH_LOG"
```

### A2: launch 后 5 min 验证

```bash
sleep 300
ps -p $CAP_PID > /dev/null && CAP_ALIVE=1 || CAP_ALIVE=0
echo "alive at +5min: $CAP_ALIVE"

# 验证 GPU 真在用
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | grep "^$SLOT1_GPU,"

# 验证 effective config 中 lambda_kl=0
grep -E "lambda_kl|decoder_kl" "$LAUNCH_LOG" | head -10
```

### A3: pass 准则 (5 条)

- [ ] launch process at +5min alive (CAP_ALIVE=1)
- [ ] GPU memory > 1GB on $SLOT1_GPU
- [ ] log 显示 lambda_kl=0.0 (effective config)
- [ ] log 不出现 RuntimeError / CUDA error
- [ ] log 出现 `[startup] resume from:` 与 `[resume] loaded step=160000` (V7 best.pt resume 起点, Round 14 B51 fix: 原 `Resuming from checkpoint` 是虚构字符串, 实际 trainer print 以上两行)
- [ ] **[NEW] step 5000 后未被 kill_switch 误 terminate** — V18 yaml 继承的 `kill_switch.check_step=5000, min_kl_change_pct=5.0` 在 lambda_kl=0 下可能 spurious-fire. codex 需在起动前 grep watchdog 实现确认 `if lambda_kl == 0: skip kl kill checks` 分支存在; 若不存在, 在 yaml 插入 `loss.decoder_kl_pullback.kill_switch.enabled: false` 或在 launch 前 remove kill_switch 块 (记录 yaml diff +1 字段, 入 commit 说明)

Pass → 等 10K step 完成 (~24-48h, V7 step 160K → step 170K). Fail → kill + 报告.

---

## §3 — Task B: Eval (capacity-only 跑完后)

### B1: KL drift probe 拓展 (~15 行, **必要, 不是条件**)

[tools/probe_v18_kl_drift.py](../../tools/probe_v18_kl_drift.py) 当前 argparse 仅有 `--config --v18-config --v7-ckpt --v18-best-ckpt --v18-last-ckpt --out-dir --batch-size --max-slices --split --device --num-workers` (Round 14 reviewer 2+3 verify). **没有** `--v18-cap-ckpt` 或 `--v18-step170k-ckpt`. **必须先 patch probe.py**, 才能跑 KL drift eval.

#### B0 preflight (probe.py 扩展前, 必验证)

```bash
cd /home/qujiaxiang/project/PET_LatentResidual
python tools/probe_v18_kl_drift.py --help | grep -qE -- '--v18-cap-ckpt|--v18-capacity-ckpt' \
  && echo "✓ probe 已支持 cap ckpt flag" \
  || { echo "✗ probe 不支持, 按下面 patch 护 才能继续"; }
```

#### probe.py patch (需改 ~15 行, **仅改 tools/probe_v18_kl_drift.py, 不动训练器**):

1. argparse: 加 2 行 (`--v18-cap-ckpt`, `--v18-step170k-ckpt`)
2. ckpt 加载逻辑: 加 2 个分支 (如果 arg None 则跳过)
3. 报告表: 加 2 行 PSNR (cap + step170k)
4. 总计 ≤ 15 行, 不改任何训练器/loss/dataloader 代码

patch 后:

```bash
TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR=review/0517/V18_capacity_only/kl_drift_with_cap_${TS}
mkdir -p "$OUT_DIR"

V18_STEP170K=/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0517_runs/V18_decoder_lora/run/first_hop_224_v18_decoder_lora/step_170000.pt
test -f "$V18_STEP170K" && echo "V18@step_170000.pt exists ✓" || { echo "FAIL: V18@step_170000.pt missing on server"; echo "Fallback: use V18.best(165K) + V18.last(200K) only, flag interpretation as ambiguous"; }

CUDA_VISIBLE_DEVICES=$SLOT1_GPU python tools/probe_v18_kl_drift.py \
  --config review/0511/log_snapshots_20260516_163900/configs/V7_config.resolved.yaml \
  --v7-ckpt /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V7/run/first_hop_224_v7_gronwall_raw/best.pt \
  --v18-best-ckpt /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0517_runs/V18_decoder_lora/run/first_hop_224_v18_decoder_lora/best.pt \
  --v18-step170k-ckpt "$V18_STEP170K" \
  --v18-last-ckpt /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0517_runs/V18_decoder_lora/run/first_hop_224_v18_decoder_lora/last.pt \
  --v18-cap-ckpt /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0517_runs/V18_capacity_only/run/first_hop_224_v18_capacity_only/last.pt \
  --out-dir "$OUT_DIR" \
  --batch-size 8 --max-slices 0 \
  2>&1 | tee "$OUT_DIR/probe.log"
```

### B2: 报告

`KL_DRIFT_WITH_CAPACITY_REPORT.md` 包含:

| ckpt | step | NORMAL PSNR(decode(z_GT), x_target) |
|---|---:|---:|
| V7.best | 160000 | 52.6341 (已知) |
| V18.best | 165000 | 52.7314 (已知) |
| **V18@step_170000.pt** (同步矩) | **170000** | **<measure>** |
| V18.last | 200000 | 52.7980 (已知) |
| **V18-cap.last** | **170000** | **<measure>** |

主判决 Δ: **V18-cap.last(170K) − V18@step_170000.pt** 在 NORMAL 上 (同步矩, 未被 cosine LR 阶段差混淆):
- |Δ| < 0.02 dB → **B 候选 confirm** (在同步矩下去掉 KL 几乎不变 GT manifold; 全部是 LoRA capacity)
- Δ < −0.05 dB (capacity-only 明显低) → **A 候选 confirm** (KL 间接贡献 GT manifold +0.05~+0.10)
- 中间 → 混合

辅判决: **V18-cap.last − V7.best** 、 **V18-cap.last − V18.last** (保留 V18.best/V18.last 原有参考点, 但不作为主判决阈值).

---

## §4 — Task C: commit + push

```bash
git add review/0517/V18_capacity_only/V18_capacity_only.yaml
git add review/0517/V18_capacity_only/V18_capacity_only_train_*.log
git add review/0517/V18_capacity_only/kl_drift_with_cap_*/

# 不 add: 任何 ckpt (在 /data_2/), pid
git status
git diff --cached --stat

git commit -m "Stage C v3 A3: V18-r32-capacity-only (KL design vs LoRA capacity disambiguation)

- yaml diff vs V18 (4 字段): output_dir / run_name / max_steps=170K / lambda_kl=0.0
- LoRA rank=32, blocks=last 2, alpha=16 与 V18 全同 (capacity 变量隔离)
- resume_from = V7 best.pt step 160K 与 V18 同起点
- 跑 10K step (V7 之上), KL drift probe 测 capacity-only vs V7/V18 在 GT manifold
- 触发: Round 13 B42 reviewer3 grep train_first_hop.py:2230 机制反驳 §2.2 A
- 不动 V18 yaml / V7 ckpt / train_first_hop.py
- 来源: REVIEW_INTEGRATION_round13 §3.2"

git push gitee foc_lite_hop0
```

---

## §5 — NOT-DO (此 task)

1. ❌ 不改 V18 yaml / V18 现有 ckpt / V7 best.pt
2. ❌ 不改 train_first_hop.py / decoder_lora.py 任何代码
   ✅ **例外 [Round 14 B50 fix]**: `tools/probe_v18_kl_drift.py` 允许 +~15 行扩展 (加 `--v18-cap-ckpt` / `--v18-step170k-ckpt` arg + ckpt 加载分支 + 2 行 PSNR 输出, 见 §B1). 不改训练器 / loss / dataloader.
3. ❌ 不 launch V13 / V14 / V18-clean / V18b 任何变体 (V13 launch 由 commit 7486776 单独 push)
4. ❌ 不直接判 SUCCESS / KILL / paper draft — capacity-only outcome 是 disambig 信息, 阶段 C 最终决策属 user
5. ❌ 不跑任何其他 eval (focus on capacity-only)
6. ❌ 不顺手 release audit DRAFT
7. ❌ smoke 步骤 fail 时不 `rm -rf` 重跑
8. ❌ 在 commit message 写"也顺便做了 X"
9. ❌ 不起 Round 15 review (由 user 决定 capacity-only 完成后是否起)
10. ❌ **[Round 14 B50 add]** 不修改 review/0517/V18_capacity_only/V18_capacity_only.yaml (本 task 已 ready, 不再改)
11. ❌ **[Round 14 B50 add]** 不在 capacity-only 训练期间 push 任何与本 task 无关的 commit
12. ❌ **[Round 14 B50 add]** 不重命名 / 移动 V7 best.pt / V18 ckpt 绝对路径 (resume_from 是绝对)
13. ❌ **[Round 14 B50 add]** 不在 V18_capacity_only/run/ 下 cp 任何 V18 ckpt 文件 (output_dir 隔离已防, cp 会破坏)

---

## §6 — 失败决策树

```
A0 spot-check 失败 (yaml/ckpt/CLI flag)?
  → 停, 报告, 不强行 launch

A1 launch 5 min 后 process 死?
  → kill 残留, 写 LAUNCH_FAILURE_REPORT.md, 等 user

A1 OOM?
  → 减 batch_size 到 4, 再试 1 次. 仍 OOM 则停.
  → 不要 launch 满负载 V13 / V18 等抢同卡

A3 pass 准则任一 fail?
  → kill + 报告, 等 user

10K step 中途 NaN?
  → 训练循环 watchdog 会 raise, 自然 exit. 报告 + 等 user

B1 probe 报错 (e.g. ckpt 加载失败 / probe 无 --v18-cap-ckpt flag)?
  → 改 probe ~10 行加 flag, 不算违反 NOT-DO (probe 是 codex 自管 tool, 不是训练器)
  → 若 probe 改后仍报错, 停 + 报告

C push 网络/认证失败?
  → 重试 1 次, 失败保留本地 commit + 报告
```

---

## §7 — 总时长估计

| Task | 时长 |
|---|---|
| A0 spot-check | 5 min |
| A1-A3 launch + 5 min verify | 10 min |
| 10K step 训练 | ~24-48h (V7 之上 +10K, 与 V18 launch 时同 LR schedule) |
| B1 probe + 报告 | ~10 min (probe 已大部分写好, 加 capacity ckpt arg) |
| C commit + push | 10 min |
| **总** | **~24-48h** (主要是训练时间) |

GPU 占用: slot 1 × 24-48h (V18 已完成 slot 1 free).
