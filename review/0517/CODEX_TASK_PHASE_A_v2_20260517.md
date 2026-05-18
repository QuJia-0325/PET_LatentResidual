# Codex Task — Phase A (v2, Round 8 hard-fixed)

- date: 2026-05-17
- branch: foc_lite_hop0
- version: **v2** (Round 8 整合后重写, supersedes v1 [CODEX_TASK_PHASE_A_20260517.md](./CODEX_TASK_PHASE_A_20260517.md))
- status: **READY TO PUSH after user 过一眼**
- 设计来源: [NEXT_STAGE_ARCH_CODE_FINAL_20260517.md](./NEXT_STAGE_ARCH_CODE_FINAL_20260517.md) §1 + [REVIEW_INTEGRATION_round8_20260517.md](./REVIEW_INTEGRATION_round8_20260517.md) 5 个 hard fix
- user 决策 (Round 7 §6): `全推荐`. user 决策 (Round 8 Q5): `1=折衷` (5 min push + 措辞强制降级)
- 硬约束: 内存 max 3 任务. Slot 1 = V18 在跑 (不可动). 本 task 最多新增 slot 2 = V13. Slot 3 永远 0.

---

## §0 — 给 codex 的 5 句话总览

1. **不停 V18**. 不改 V18 yaml. 不改 train_first_hop.py 任何代码.
2. 5 件事: Task A 改 1 份 doc / Task B V21 retire 4 份 doc / Task C V13 200-step smoke / Task D V13 full launch / Task E 报告 + push.
3. **每个 task 末尾都重复一次**: fail → 停下报告, 不要尝试 fix, 不要 retry, 不要 escalate.
4. V13 是 from-scratch (yaml 无 resume_from), **不需要 `--resume`**.
5. 本版本经过 Round 8 三 reviewer 验证, 已修复 5 个 hard blocker (CLI flag 不存在 / path_guard / pass 准则 grep / 文档结构假设错 / commit 范围). 不要绕过任何修复条目.

---

## §1 — Task A: 修订 V18_design_rationale.md (含 stale facts 修正 + 新增 §2.4 + 新增 §5.2)

### A0: 前置 spot-check (5 min, 必做)

```bash
cd /home/qujiaxiang/project/PET_LatentResidual
FILE=review/0517/V18_decoder_lora/V18_design_rationale.md

# 验证当前结构 (Round 8 reviewer 已 verify, codex 必须再 verify 一次)
grep -n "^### 2\.1\|^### 2\.2\|^### 2\.3\|^### 2\.4\|^## §3\|^## §5\|^### 5\.1\|^### 5\.2" "$FILE"
# 期望输出 (来自当前 head):
#   ### 2.1 设计哲学
#   ### 2.2 V18 与所有撤回候选的 EV 对比
#   ### 2.3 V18 失败的 4 个具体可能 + 应对   ← 注意: 是 failure mode 表, 不是阈值表
#   ## §3 — 为什么不直接全解冻 decoder？
#   ## §5 — 不可逾越的红线
# 期望 *不出现*:
#   ### 2.4 (这就是本 task 要新增的)
#   ### 5.2 (本 task 要新增)
#
# 如果实际输出与上述不符 → 停下报告, 文档已被前 task 改过, 不要硬执行 sed.
```

### A1: 修正 §2.1 stale facts (B25 hard fix)

V18_design_rationale.md §2.1 仍写 "LoRA rank=8, alpha=16" 和 "λ_kl=0.5", 但实际 V18 yaml 是 rank=32 和 λ_kl=0.05. 必须先修正再加 §2.4.

**操作**: 在 §2.1 "设计哲学" 段表格里, 修改两行:

| 原文 | 改为 |
|---|---|
| `LoRA rank=8，alpha=16 (scale=2.0)` | `LoRA rank=32，alpha=16 (scale=0.5; Round 4 audit 后从 rank=8 调整, 真实 trainable=589,824 ≈ 2.5% of decoder)` |
| `KL pull-back loss (λ_kl=0.5)` | `KL pull-back loss (λ_kl=0.05; Round 4 audit 后从 0.5 调小, 防 KL 主导 total loss)` |

**额外修正** §2.2 EV 表里 V18 这一行 ("0.5 - 2.0 dB" → "0.05 - 0.30 dB (区间下界 0.5 dB × 10%, 见 §2.4)"). **不**改 §2.2 表里的 "HIGH (10-40× 其他候选)" — 这是相对评估, 不受 EV 绝对值修正影响.

**实施**: codex 用编辑器手工 patch (sed 跨多行不安全). 改完后:
```bash
# 验证改动
grep -n "rank=32" "$FILE"     # 期望 1+ 行
grep -n "rank=8" "$FILE"      # 期望 0 行 (原文已全替换) 或仅在历史引用上下文
grep -n "λ_kl=0\.05" "$FILE"  # 期望 1+ 行
grep -n "λ_kl=0\.5" "$FILE"   # 期望 0 行 (或仅在历史) — 若 > 0 且不是历史, 停下报告
```

### A2: 新增 §2.4 EV 修正记录

**位置**: §2.3 末尾之后, §3 之前.

**完整文本** (照抄, 不允许改写):

```markdown
### 2.4 Round 5/6/7 EV 修正记录 (不改阈值, 仅更新 interpretation)

#### attackable_gap 不是单一数字

| 量 | 值 | 物理含义 |
|---|---:|---|
| psnr_ceil = PSNR(decode_V7(z_GT), x_target) | 46.64 dB | RAE round-trip 上限 |
| psnr_transport = PSNR(decode_V7(z_pred^V7), x_target) | 35.44 dB | V7 baseline 实测 |
| psnr_transport_vs_ceil = PSNR(decode_V7(z_pred), decode_V7(z_GT)) | 35.94 dB | "若 z_pred=z_GT 仍用 V7 decoder" |

**V18 attackable gap = 区间 [0.05, 3] dB**, 不是单点估计:
- 下界 ~0.5 dB: 假设 V18 LoRA 只能学到 "decode(z_pred) 对齐到 decode(z_GT)" (吃 8% decoder ceiling 部分)
- 上界 ~3 dB: LoRA rank=32 on last 2 blocks (589K trainable / 24M total ≈ 2.5%) 的物理容量约束
- 中位估计 0.1-0.3 dB (落在 PARTIAL 区间)

#### 历史偏差 (作为前车之鉴, 不修改 §2.3 trigger 阈值)

| Bug | 形态 | 识别于 |
|---|---|---|
| B10 | 11.2 dB 当 attackable (B5 同型, 高估) | Round 4 |
| B11 | 0.5 dB 当 attackable (B10 镜像错, 低估) | Round 5 |
| B18 | 在数据出来前 pre-design 每个 outcome 的响应分支 | Round 7 |
| B19 | yaml `decoder_kl_pullback.warmup_steps` 字段名误导, 实际是 ramp_steps (train_first_hop.py:2222-2228 硬编码 warmup_steps=0) | Round 7 |
| B20 | resume 后 `max_steps` 是绝对 step 上限, 不是新训步数. V18-clean from V7 best.pt (step 160000) + max_steps=200000 = 实际仅训 ~40K step | Round 7 |
| B21-B25 | task md 凭记忆杜撰 CLI flag / 文档结构 / 日志格式 (Round 8 在起草 codex task md 时识别) | Round 8 |

#### §2.3 阈值表的解读

§2.3 表的 trigger 列里隐含三档判定:
- **PRIMARY SUCCESS**: 对应 §2.3 行 4 trigger "ΔPSNR < +0.05 + KL drift > 0.05 → 撤回" 的反面, 即 ΔPSNR ≥ +0.30 dB
- **PARTIAL**: ΔPSNR ∈ [+0.05, +0.30] dB → §2.3 行 1/2/3 触发 sweep 行动
- **KILL @ 20K**: ΔPSNR < +0.05 dB

V18 评估时同时报告:
1. 三档分类 (SUCCESS/PARTIAL/KILL) by **原阈值** — 单一判定, 无 dual-track, 无 SECONDARY 列
2. ΔPSNR 在 [0.05, 3] dB 区间的位置 (纯描述性 interpretation)
3. KL drift 数据 (psnr_v18_at_zgt vs 46.64 dB) — 用于阶段 C 决策, 不参与 SUCCESS 定义

本段是 **EV 修正记录, 不是新阈值或新决策路径**. §2.3 原 trigger 阈值是唯一 SUCCESS 判定标准.
```

### A3: 新增 §5.2 Round 5-7 偏差自警

**位置**: §5 红线段, 在 "## §6" 之前. 注意当前 §5 是 4 条红线 (1./2./3./4.), 不是子段 §5.1/§5.2. 本步骤把现有 4 条红线**重组为 §5.1**, 再在其后**新增 §5.2**.

**实施 (照抄结构)**:

```markdown
## §5 — 不可逾越的红线

### 5.1 launch-time 原始红线 (Round 1 写)

1. **V18 init 必须可被 sanity-check** ...(原文不动)
2. **预注册阈值不可事后改** ...(原文不动)
3. **不允许再发 peer review** ...(原文不动)  
4. **Day 9 评估是 binary** ...(原文不动)

### 5.2 Round 5-8 偏差自警 (新增 standing rules)

B12 — 禁止以 "baseline 计算错了" mid-run 改预注册阈值. 修正只能通过 §2.4 文字描述,
      不通过替换 §2.3 trigger 数字, 不通过加 SECONDARY 列, 不通过 dual-track 并列.
      (Round 5 试图把 +0.30 改 +0.15, Round 6/7 reject)

B13 — 禁止 sunk-cost-resume 设计. paired ablation 必须从同一干净起点出发.
      (Round 6 永久撤销 V18b "从 V18 ckpt 续训" 思路, 改 V18-clean from V7)

B14 — yaml 改字段必须 cross-check 联动:
      - LR schedule total_steps_override
      - --resume CLI flag (training.resume_from yaml 字段是死字段, 不被 train_first_hop.py 读取)
      - LoRA→LoRA resume 不被支持 (_assert_v18_step0_equivalence 会 raise non-zero LoRA A/B)
      - require_fresh_output_dir 与 output_dir 改动联动
      - path_guard: output_dir 必须在 /data_2/ 下

B15 — contingent slot 的真实概率必须披露

B16 — "outcome 独立" 框架的 narrow scope: 测量层独立 ≠ 项目优先级层独立

B17 — paired comparison threshold 必须 cross-check 自身 noise floor

B18 — 禁止 pre-design every branch. 数据出来前不为每个 outcome 设计响应分支.

B19 — KL config `warmup_steps` 实际是 ramp_steps; 引用此字段处必须注 [实际=ramp_steps]

B20 — resume 后 max_steps 是绝对 step 上限; resume 实验文档必须显式注 "实际训练 N step"

B21 — 禁止 CLI flag 杜撰. 所有 shell 命令的每个 flag 必须先 grep `add_argument` verify.
      (Round 8 起草中 claude 杜撰了 --max-steps-override / --output-dir-override, 实际不存在)

B22 — 改 yaml 必须 cross-check path_guard / require_fresh_output_dir / 与 train_first_hop.py
      代码字段名一致性. 不能凭记忆.

B23 — push 是不可逆操作. push 节点必须 anchor 到 "可观测训练事件" (e.g. process alive at +Xmin),
      不能 anchor 到 "训练健康/收敛/稳定" 这种长期推断.

B24 — 引用 baseline 必须把具体数字与出处写进文档, 不能"≤ baseline × N"留空让 reader 猜.

B25 — 起草 task md 前必须做"脱稿执行核查": 每个 CLI flag / yaml 字段 / 日志 grep pattern /
      引用文档 section / path 都 spot-check 一次, 不依赖记忆.

      → 综合应用: 起草任何 codex task md 前, 必须先做以下 grep:
        - `grep add_argument <训练器/eval 脚本>`
        - `grep "^### " <要修改的 doc>` 确认 section 结构
        - `grep <字段名> <要写的 yaml>` 确认字段拼写
        - 找一段历史 log 验证 grep pattern 真能匹配
```

### A4: A 段 pass 准则

- [ ] A0 spot-check 输出与期望一致, 否则**停下报告**
- [ ] A1 §2.1 表里 rank=32 和 λ_kl=0.05 出现, rank=8 / λ_kl=0.5 在 §2 范围内只出现在历史引用 (不在配置描述)
- [ ] A2 §2.4 完整出现在 §2.3 之后 §3 之前
- [ ] A3 §5 重组为 §5.1 (原 4 条红线) + §5.2 (新 B12-B25 standing rules)
- [ ] §2.3 的 trigger 数字 (+0.05 / +0.30 / +0.05) **未改动**
- [ ] markdown 渲染验证: `wc -l "$FILE"` 与改动行数大致一致, 没有意外覆盖

**任何一条 fail → 停下报告. 不尝试 fix, 不 retry, 不 escalate**.

---

## §2 — Task B: V21 retire 残留清理 (preflight + 白名单 + 跳过 0 命中)

### B0: preflight grep (10 min, 必做)

```bash
cd /home/qujiaxiang/project/PET_LatentResidual

# 全 0517 范围 grep V21
echo "===== V21 grep results (preflight) ====="
grep -rn "V21" review/0517/ 2>/dev/null | grep -v ".pyc:" | sort
echo "===== END ====="

# 期望命中数 ≥ 5 且 ≤ 50 (sanity)
N=$(grep -rn "V21" review/0517/ 2>/dev/null | grep -v ".pyc:" | wc -l)
echo "Total V21 hits: $N"
if [ "$N" -lt 5 ] || [ "$N" -gt 50 ]; then
    echo "FAIL: V21 hit count $N out of expected [5, 50]. STOP."
    exit 1
fi
```

### B1: 4 文档白名单 + 0 命中跳过

| 文档 | 当前命中 (Round 8 reviewer verify) | 操作 |
|---|---:|---|
| [REVIEW_INTEGRATION_round4_20260517.md](./REVIEW_INTEGRATION_round4_20260517.md) | 19 | 加标记 + 顶部 warning |
| [GAP_DECOMP_REPORT.md](./V18_decoder_lora/gap_decomp_fullval_20260517/GAP_DECOMP_REPORT.md) | 1 | 加标记 + 顶部 warning |
| [CODEX_RUNBOOK_V18_20260517.md](./CODEX_RUNBOOK_V18_20260517.md) | 1 | 加标记 + 顶部 warning |
| [V18_EXECUTION_REPORT_20260517.md](./V18_decoder_lora/V18_EXECUTION_REPORT_20260517.md) | **0** | **跳过, 不动** |

**实施步骤**:

对每份白名单文档, codex 用编辑器 (不是 sed):
1. 文档顶部 (title 之后第一行) 插入 deprecation warning (仅在命中数 > 0 时):
   ```markdown
   > ⚠️ **NOTE (2026-05-17)**: 本文档含已 retired 设计 V21.
   > 见正文中 `[RETIRED 2026-05-17 Round 5: B8]` 标记 + Round 5 B8 偏差识别.
   ```
2. 对每处 V21 设计描述 (form: `V21 = ...` 或 `V21 fallback ...` 或 `V21 conv head`), 在该段落末尾加:
   ```markdown
   > **[RETIRED 2026-05-17 Round 5: B8]** V21 描述与 RAE conv_head.py 实际代码不符.
   > conv_head.py 是 `decoder_pred + unpatchify` 的 token-domain 替换 (输入 [B, N, 512] →
   > Conv + PixelShuffle(14) → 196×196 输出), **不是** post-decode pixel refiner.
   > V21 概念已 retire, 不再列为候选方案.
   > 见 [REVIEW_INTEGRATION_round5_20260517.md](./REVIEW_INTEGRATION_round5_20260517.md) §B8.
   ```
3. **不**对元数据/历史引用 (如 "Round 5 retire V21", "Round 4 提到 V21 fallback") 加标记 — 那是讨论 V21 retire 行为的句子, 不是设计描述

### B2: 出 scope 命中处理

preflight grep 可能在**非白名单**文档命中 V21 (例如 REVIEW_INTEGRATION_round5_20260517.md 自己提到 V21). 这些命中:
- **不动**这些文档
- 在执行报告 §6 自警 section list 出来, 写 "命中但跳过 (out of white-list)"
- 等 user 阶段 B/C 决定是否处理

### B3: pass 准则

- [ ] B0 grep preflight 命中数 ∈ [5, 50], 否则**停下报告**
- [ ] 3 份命中 > 0 的白名单文档都已加 顶部 warning + 段内 retire 标记
- [ ] V18_EXECUTION_REPORT_20260517.md (命中=0) **未被修改**
- [ ] git diff --stat 显示只改动 3 个文件 (不是 4 个), 无其他文件
- [ ] 原文未被删除 (audit trail 保留)
- [ ] 命中但出 scope 的文档列在执行报告 §6, 未被改动

**fail → 停下报告. 不尝试 fix, 不 retry, 不 escalate**.

---

## §3 — Task C: V13 200-step smoke (~20 min)

### C0: 前置确认

```bash
cd /home/qujiaxiang/project/PET_LatentResidual
git pull gitee foc_lite_hop0

# V13 原 yaml 必须存在且未被修改
ORIG=review/0516/V13_true_image_aux_ablation/V13_true_image_aux_off.yaml
test -f "$ORIG" || { echo "FAIL: V13 yaml missing"; exit 1; }

# 记录原 yaml 的 md5, 后续验证不被污染
ORIG_MD5=$(md5sum "$ORIG" | cut -d' ' -f1)
echo "Original V13 yaml md5: $ORIG_MD5"

# 验证 train_first_hop.py CLI 只接受 --config / --resume (B21 防御)
SUPPORTED_FLAGS=$(grep "add_argument" train_first_hop.py | grep -oE "['\"]--[a-z-]+['\"]" | sort -u)
echo "Supported CLI flags: $SUPPORTED_FLAGS"
# 期望: '--config' 和 '--resume', 仅此两个

# GPU 状态: slot 1 = V18 (不动), 至少 1 张空闲
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
echo "Pick an idle GPU id for SMOKE_GPU below"
```

### C1: smoke yaml 副本 (B22 hard fix: output_dir 在 /data_2)

```bash
TS=$(date +%Y%m%d_%H%M%S)
SMOKE_GPU=<填上面 nvidia-smi 空闲 GPU id, 不要与 V18 同卡>

# repo 内只放 smoke yaml 副本和 smoke.log; 训练实际 output 在 /data_2
SMOKE_YAML=review/0516/V13_true_image_aux_ablation/smoke_runs/V13_smoke_${TS}.yaml
SMOKE_OUTPUT_DIR=/data_2/qujiaxiang/outputs/PET_LatentResidual/smoke_runs/v13_smoke_${TS}
SMOKE_LOG=review/0516/V13_true_image_aux_ablation/smoke_runs/V13_smoke_${TS}.log
mkdir -p "$(dirname $SMOKE_YAML)"
mkdir -p "$SMOKE_OUTPUT_DIR"

# 复制原 yaml (不动原文件)
cp "$ORIG" "$SMOKE_YAML"

# 用 python 改 4 字段 (sed 跨 yaml indent 不安全):
python3 - <<EOF
import yaml, pathlib
p = pathlib.Path("$SMOKE_YAML")
cfg = yaml.safe_load(p.read_text())

# 4 字段: max_steps / output_dir / run_name / require_fresh_output_dir
cfg["training"]["max_steps"] = 200
cfg["output_dir"] = "$SMOKE_OUTPUT_DIR"
cfg["run_name"] = "first_hop_224_v13_smoke_${TS}"
cfg["training"]["require_fresh_output_dir"] = False

# 保险: 调小 best_select_full_eval_interval 避免 200 step 内触发 full-val
cfg.setdefault("training", {})["best_select_full_eval_interval"] = 100000

p.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
print(f"smoke yaml written: {p}")
print(f"max_steps = {cfg['training']['max_steps']}")
print(f"output_dir = {cfg['output_dir']}")
print(f"require_fresh = {cfg['training']['require_fresh_output_dir']}")
EOF

# Sanity check: 修改生效
python3 -c "
import yaml
with open('$SMOKE_YAML') as f:
    cfg = yaml.safe_load(f)
assert cfg['training']['max_steps'] == 200, 'max_steps not patched'
assert cfg['output_dir'].startswith('/data_2/'), 'output_dir not in /data_2'
assert cfg['training']['require_fresh_output_dir'] == False, 'require_fresh not False'
print('smoke yaml sanity PASS')
"

# Sanity check: 原 yaml 比特不动
NEW_MD5=$(md5sum "$ORIG" | cut -d' ' -f1)
if [ "$NEW_MD5" != "$ORIG_MD5" ]; then
    echo "FAIL: original V13 yaml was modified ($ORIG_MD5 -> $NEW_MD5). STOP."
    exit 1
fi
echo "Original V13 yaml md5 unchanged: $NEW_MD5"
```

### C2: smoke launch

```bash
export CUDA_VISIBLE_DEVICES=$SMOKE_GPU

# 只用真实存在的 2 个 CLI flag: --config 和 --resume (V13 不需 --resume)
nohup python train_first_hop.py \
  --config "$SMOKE_YAML" \
  > "$SMOKE_LOG" 2>&1 &
SMOKE_PID=$!

echo "Smoke launched: PID=$SMOKE_PID, log=$SMOKE_LOG"
echo "Smoke output dir: $SMOKE_OUTPUT_DIR"
```

### C3: smoke 监控 (阻塞等待, 最多 30 min)

```bash
# 等 smoke 完成 (200 step 应在 5-20 min 内). 上限 30 min, 超过则视为 stuck
TIMEOUT=1800
START=$(date +%s)

while ps -p $SMOKE_PID > /dev/null 2>&1; do
    NOW=$(date +%s)
    if [ $((NOW - START)) -gt $TIMEOUT ]; then
        echo "FAIL: smoke timed out after ${TIMEOUT}s. Killing PID=$SMOKE_PID. STOP."
        kill -9 $SMOKE_PID 2>/dev/null
        exit 1
    fi
    sleep 10
done

wait $SMOKE_PID 2>/dev/null
SMOKE_RC=$?
echo "Smoke exit code: $SMOKE_RC"
```

### C4: pass 准则 (5 条, 全部读 metrics.jsonl, 不 grep stdout)

代码事实 (Round 8 verify):
- `metrics_payload["img"]` 字段名是 `img` (不是 `loss_img`)  
  ([train_first_hop.py:2446](PET_LatentResidual/train_first_hop.py))
- `image_aux.enabled=false` 时 metrics_payload `img=0.0`, 但 `ssim_raw_mean/min/max=nan` ([train_first_hop.py:2166-2178](PET_LatentResidual/train_first_hop.py))
- jsonl 写到 `<output_dir>/metrics.jsonl` ([train_first_hop.py:1817](PET_LatentResidual/train_first_hop.py))

```bash
METRICS=$SMOKE_OUTPUT_DIR/metrics.jsonl

# Pass 准则 1: 程序正常退出
test "$SMOKE_RC" = "0" || { echo "FAIL_1: smoke exit code $SMOKE_RC. STOP."; cat "$SMOKE_LOG" | tail -50; exit 1; }
echo "PASS_1: smoke exit 0"

# Pass 准则 2: metrics.jsonl 存在且非空, train event 行数 >= 50
test -f "$METRICS" || { echo "FAIL_2: no metrics.jsonl at $METRICS. STOP."; exit 1; }
N_TRAIN=$(grep -c '"event": "train"' "$METRICS")
test "$N_TRAIN" -ge 50 || { echo "FAIL_2: only $N_TRAIN train rows in metrics.jsonl. STOP."; exit 1; }
echo "PASS_2: metrics.jsonl has $N_TRAIN train rows"

# Pass 准则 3: image_aux disabled 在事实层 = 所有 train row 的 img 字段为 0.0
python3 - <<EOF
import json
rows = []
with open("$METRICS") as f:
    for l in f:
        try:
            r = json.loads(l)
            if r.get("event") == "train":
                rows.append(r)
        except: pass
img_vals = [r["img"] for r in rows if "img" in r]
assert len(img_vals) >= 50, f"only {len(img_vals)} train rows have 'img' field"
nonzero = [v for v in img_vals if v != 0.0]
assert len(nonzero) == 0, f"FAIL_3: {len(nonzero)} train rows have img != 0: {nonzero[:5]}. STOP."
print(f"PASS_3: all {len(img_vals)} train rows have img=0.0")
EOF

# Pass 准则 4: total loss / pair / roll / grad_norm 关键字段全部 finite (不含 NaN/Inf)
# 注意: ssim_raw_* 在 image_aux=false 时是 nan, 这是预期, 不检查
python3 - <<EOF
import json, math
rows = []
with open("$METRICS") as f:
    for l in f:
        try:
            r = json.loads(l)
            if r.get("event") == "train":
                rows.append(r)
        except: pass
critical = ["loss", "pair", "roll"]   # 关键 finite 字段
nonfinite = []
for r in rows:
    for k in critical:
        if k in r and not math.isfinite(float(r[k])):
            nonfinite.append((r["step"], k, r[k]))
assert len(nonfinite) == 0, f"FAIL_4: nonfinite in critical fields: {nonfinite[:5]}. STOP."
print(f"PASS_4: all critical fields finite in {len(rows)} rows")
EOF

# Pass 准则 5: smoke 完成 wall time ≤ 15 min (绝对上限, 不依赖 V7 baseline)
# 从 metrics.jsonl 第一行和最后一行抓时间戳
python3 - <<EOF
import json, datetime, os
with open("$METRICS") as f:
    lines = [l for l in f if l.strip()]
first_row = json.loads(lines[0])
last_row = json.loads(lines[-1])
# 用文件 mtime 作为粗估 (metrics.jsonl 可能不含 timestamp 字段)
mtime_end = os.path.getmtime("$METRICS")
import pathlib
ctime_start = os.path.getctime("$METRICS")
elapsed_min = (mtime_end - ctime_start) / 60.0
print(f"smoke elapsed: {elapsed_min:.2f} min ({len(lines)} jsonl rows)")
assert elapsed_min <= 15, f"FAIL_5: smoke took {elapsed_min:.2f} min > 15 min limit. STOP."
print(f"PASS_5: elapsed {elapsed_min:.2f} min <= 15 min")
EOF
```

### C5: smoke fail 处理

任何 PASS_X = FAIL → **停下报告**:

```
1. 不要删除 $SMOKE_OUTPUT_DIR (保留 failure 现场)
2. 不要 rm -rf $SMOKE_YAML (保留 yaml 副本)
3. 不要重新 launch smoke (LLM 滑坡防御)
4. 写 SMOKE_FAILURE_REPORT_$(date +%Y%m%d_%H%M%S).md:
   - 哪一条 pass 准则 fail
   - SMOKE_LOG 最后 100 行
   - metrics.jsonl 最后 5 行 (若存在)
   - 系统状态 (nvidia-smi, df -h)
5. 不 launch full V13 (slot 2 维持空)
6. 等 user 决策
```

**fail → 停下报告. 不尝试 fix, 不 retry, 不 escalate**.

---

## §4 — Task D: V13 full launch (slot 2, ~7 天)

**前置**: §3 所有 PASS_X 全部 PASS, smoke 报告无 STOP.

### D1: launch

```bash
cd /home/qujiaxiang/project/PET_LatentResidual

LAUNCH_TS=$(date +%Y%m%d_%H%M%S)
LAUNCH_LOG_PATH=review/0516/V13_true_image_aux_ablation/V13_train_${LAUNCH_TS}.log

# 使用与 smoke 同张 GPU (slot 2)
export CUDA_VISIBLE_DEVICES=$SMOKE_GPU

# V13 是 from-scratch, 不需 --resume
nohup python train_first_hop.py \
  --config review/0516/V13_true_image_aux_ablation/V13_true_image_aux_off.yaml \
  > "$LAUNCH_LOG_PATH" 2>&1 &
V13_PID=$!

# 不 write PID 到 git-tracked 文件 (B14 防御)
echo "V13 launched: PID=$V13_PID, log=$LAUNCH_LOG_PATH"
echo "PID will only be recorded in execution report (not committed)"
```

### D2: launch 后 5 min 验证

```bash
sleep 300
ps -p $V13_PID > /dev/null && V13_ALIVE=1 || V13_ALIVE=0

# 验证 V13 真的在用 GPU
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | grep "^$SMOKE_GPU,"
```

### D3: D pass 准则

- [ ] V13 process at +5 min still alive (`V13_ALIVE=1`)
- [ ] log 包含 `effective image_aux` 或类似 yaml dump 显示 enabled=false (若未输出, 看 metrics.jsonl 是否已开始写入)
- [ ] log 不出现 `Resuming from checkpoint` (V13 是 from-scratch)
- [ ] GPU memory > 1GB (确认 V13 真的在 GPU 上)
- [ ] 无 OOM / CUDA error in stderr

### D4: D fail 处理

任一 fail → kill V13, 写 `LAUNCH_FAILURE_REPORT_$(date +%Y%m%d_%H%M%S).md`, 等 user.

**fail → 停下报告. 不尝试 fix, 不 retry, 不 escalate**.

---

## §5 — Task E: 执行报告 + push gitee (措辞强制降级)

### E1: 写执行报告 (强制降级措辞 + 12 项 NOT-DO 勾选)

文件: `review/0517/PHASE_A_EXECUTION_REPORT_$(date +%Y%m%d_%H%M%S).md`

**模板** (照抄, 不允许重写):

```markdown
# Phase A Execution Report

- date: <TS>
- branch: foc_lite_hop0
- HEAD before: <git rev-parse HEAD>
- HEAD after this commit: <new HEAD>
- task md version: CODEX_TASK_PHASE_A_v2_20260517.md (Round 8 hard-fixed)

## §1 — Task A: V18_design_rationale.md 修订
- A0 spot-check 输出与期望一致: YES/NO
- A1 §2.1 stale facts 修正 (rank=32, λ_kl=0.05): DONE/FAIL
- A2 §2.4 新增: DONE/FAIL
- A3 §5 重组 + §5.2 新增: DONE/FAIL
- §2.3 trigger 数字未改: VERIFIED/VIOLATED

## §2 — Task B: V21 retire
- B0 grep 命中: <N> hits in [5, 50] range: YES/NO
- 命中明细:
  - REVIEW_INTEGRATION_round4: <N1> hits → modified
  - GAP_DECOMP_REPORT: <N2> hits → modified
  - CODEX_RUNBOOK_V18: <N3> hits → modified
  - V18_EXECUTION_REPORT: <N4> hits → SKIPPED (expected 0)
- 出 scope 命中 (其他文档, 未改动):
  - <文件路径>: <hits>
  - (列出所有非白名单命中, 等阶段 B/C 处理)

## §3 — Task C: V13 smoke
- SMOKE_OUTPUT_DIR: <path>
- SMOKE_LOG: <path>
- SMOKE_YAML: <path>
- PASS_1 (exit 0): YES/NO
- PASS_2 (metrics.jsonl ≥ 50 train rows): YES/NO (<N> rows)
- PASS_3 (all img=0.0): YES/NO
- PASS_4 (loss/pair/roll finite): YES/NO
- PASS_5 (≤ 15 min): YES/NO (<X> min)
- 结论: SMOKE PASSED / SMOKE FAILED (→ SMOKE_FAILURE_REPORT 路径)

## §4 — Task D: V13 full launch
- V13_PID: <pid> (仅记录, 不入 git)
- V13_LOG_PATH: <abs path>
- alive at +5 min: YES/NO
- GPU <SMOKE_GPU> memory: <X> MB
- 状态措辞 (B23 防御, 强制原文):
  > "V13 process alive at +5min after launch, GPU utilized.
  >  **未经长期 soak 验证**, 7 天训练健康度由阶段 B/C 跟踪."

## §5 — Slot 占用状态 (push 时)
- Slot 1 (V18 train): step <X>, alive at push time (V18 不在本 task 验证范围)
- Slot 2 (V13 train): step <Y, 大约 50-100>, alive at +5min
- Slot 3: 空

## §6 — NOT-DO 12 项勾选 (codex 必须全部 [x])
- [ ] 1. 未停 V18 / 未改 V18 yaml 任何字段
- [ ] 2. 未 launch V14 / V9 / V21 任何变体
- [ ] 3. 未 launch V18b 任何形态
- [ ] 4. 未准备 V18-clean yaml (推迟到阶段 C)
- [ ] 5. 未做 V18 step 180K early-eval
- [ ] 6. 未起 Round 9 review
- [ ] 7. 未改 train_first_hop.py 任何代码
- [ ] 8. 未加 LoRA→LoRA resume 支持
- [ ] 9. 未优化/重构/重命名/移动任何代码或文件
- [ ] 10. 未跑任何 eval (slot 1 V18 在用 GPU)
- [ ] 11. 未在 commit message 写"也顺便做了 X"
- [ ] 12. 未改 V13 原始 yaml (review/0516/V13_true_image_aux_ablation/V13_true_image_aux_off.yaml)

## §7 — Round 5-8 偏差自查 (6 项 yes/no + 说明)
- B12 (mid-run 改阈值): NO/YES + <说明>; V18_design_rationale §2.3 数字未动
- B13 (sunk-cost-resume): NO/YES + <说明>; 未触及 V18b
- B14 (yaml cross-check): NO/YES + <说明>; smoke yaml 改 4 字段是否都 spot-check 联动
- B18 (pre-design every branch): NO/YES + <说明>; 是否准备了阶段 B/C 内容
- B19 (warmup_steps): NO/YES; A2 §2.4 是否正确写 "[实际=ramp_steps]"
- B20 (resume max_steps): NO/YES; A2 §2.4 是否正确写 "实际训练 N step"
- B21 (CLI flag 杜撰): NO/YES; smoke/launch 命令只用 --config 和 --resume

## §8 — git diff --stat
\`\`\`
<git diff --stat HEAD~1 .. 输出, 显示改动文件 + 行数>
\`\`\`
预期改动文件:
- V18_design_rationale.md (~+60 行)
- REVIEW_INTEGRATION_round4_20260517.md (~+25 行)
- GAP_DECOMP_REPORT.md (~+10 行)
- CODEX_RUNBOOK_V18_20260517.md (~+10 行)
- PHASE_A_EXECUTION_REPORT_<TS>.md (新建)
- smoke_runs/V13_smoke_<TS>.yaml (新建)
- smoke_runs/V13_smoke_<TS>.log (新建)

不在改动列表 (违反则 STOP):
- ❌ V18_decoder_lora.yaml
- ❌ V13_true_image_aux_off.yaml (原 yaml)
- ❌ train_first_hop.py
- ❌ V13_train_<TS>.log (rolling, 不入 git)
- ❌ 任何 *.pid

## §9 — 下一步
- V18 (slot 1) 预计完成时间: <TS>
- V18 step 200K 完成后, user 发起阶段 B task md (V18 200K eval + KL drift)
```

### E2: commit + push gitee

```bash
cd /home/qujiaxiang/project/PET_LatentResidual

# 显式 add (不通配, 避免误 commit)
git add review/0517/V18_decoder_lora/V18_design_rationale.md
git add review/0517/REVIEW_INTEGRATION_round4_20260517.md
git add review/0517/V18_decoder_lora/gap_decomp_fullval_20260517/GAP_DECOMP_REPORT.md
git add review/0517/CODEX_RUNBOOK_V18_20260517.md
git add "review/0517/PHASE_A_EXECUTION_REPORT_${TS}.md"
git add review/0516/V13_true_image_aux_ablation/smoke_runs/V13_smoke_*.yaml
git add review/0516/V13_true_image_aux_ablation/smoke_runs/V13_smoke_*.log

# 显式排除 (B14/B23 防御):
# - 不 add V13_train_*.log (rolling 7-day log)
# - 不 add *.pid
# - 不 add /data_2/ 下任何 smoke training output (不在 repo 内)

# 验证 git status 没有意外文件
echo "===== git status before commit ====="
git status
echo "===== END ====="

# 验证 git diff --stat
echo "===== git diff --stat (after add) ====="
git diff --cached --stat
echo "===== END ====="

# user 应在执行报告 §8 检查这一段输出. 任何意外文件 → STOP.

# commit
git commit -m "Phase A v2: V21 retire + design_rationale §2.4/§5.2 update + V13 smoke pass + launch (slot 2)

- V18_design_rationale: §2.1 stale facts 修正 (rank=8→32, λ_kl=0.5→0.05);
                        §2.4 新增 EV 修正 (区间 [0.05, 3] dB, 不改 §2.3 trigger);
                        §5 重组 + §5.2 新增 B12-B25 standing rules
- V21 retire: 3 文档加 [RETIRED 2026-05-17 Round 5: B8] 标记 + 顶部 warning (不删原文);
              V18_EXECUTION_REPORT 命中=0 跳过
- V13 smoke: 200-step pass (img=0, finite, ≤ 15min), output 在 /data_2/, repo 仅放 yaml+log
- V13 launch: from-scratch on slot 2, alive at +5min, healthy 度由阶段 B/C 跟踪 (B23)
- 不动 V18 (slot 1), slot 3 空
- 来源: NEXT_STAGE_ARCH_CODE_FINAL §1 + Round 8 hard fix"

# push
git push gitee foc_lite_hop0
```

### E3: 完成通知

```
========================================
PHASE A v2 COMPLETE @ <TS>
- V18 (slot 1): step <X>, 继续 (不在本 task 验证范围)
- V13 (slot 2): step <Y>, alive at +5min, soak 由阶段 B/C 跟踪
- slot 3: 空
- next: 等 V18 step 200K (~12-18h), user 发阶段 B task md
- gitee push: SUCCEEDED
========================================
```

---

## §6 — 完整 NOT-DO 列表 (Round 8 扩展, 16 条)

每个 task 内**已经重复**一次 "fail → 停下报告, 不 escalate" 提醒. 本节是集中列表, codex 必须**全部遵守** (E1 §6 12 项勾选式 attestation):

不允许:
1. 停 V18 / 改 V18 yaml 任何字段
2. launch V14 / V9 / V21 任何变体
3. launch V18b 任何形态 (永久撤销)
4. 准备 V18-clean yaml (推迟到阶段 C)
5. V18 step 180K early-eval (Round 7 共识删除)
6. 起 Round 9 review (Round 8 自身就是终点)
7. 改 train_first_hop.py 任何代码 (本 task 不需要)
8. 加 LoRA→LoRA resume 支持 (V18-clean 不走这条路径)
9. 优化 / 重构 / 重命名 / 移动 任何代码或文件
10. 跑任何 eval (slot 1 V18 在用 GPU, 阶段 B 才 eval)
11. 在 commit message 写"也顺便做了 X"
12. 改 V13 原始 yaml (review/0516/V13_true_image_aux_ablation/V13_true_image_aux_off.yaml)
13. commit V13 原始 yaml (它没改动, 不入此 commit)
14. CLI flag 不存在时**改 train_first_hop.py 加 flag** (改用 yaml override; 见 §3 hard fix)
15. smoke fail 时 `rm -rf $SMOKE_OUTPUT_DIR` 重跑 (保留失败现场)
16. push gitee 时执行报告写"V13 健康/稳定/收敛正常"等 7-day 长期推断措辞 (B23)

**任何 NOT-DO 项被违反 → user 会 hard revert. 严格守界限**.

---

## §7 — 失败模式快速决策树

```
Task A0 spot-check 失败 (V18_design_rationale 结构与期望不符)?
  → 文档已被前 task 改过. 停, 写 TASK_A_PRECHECK_FAILURE.md, 等 user
  → 不要尝试 "推测原意" 然后硬执行

Task A1/A2/A3 fail (sed/python 修改失败 / 验证不通过)?
  → 停, 写 TASK_A_FAILURE.md (含错误细节), 等 user
  → 不要 retry sed (LLM 滑坡)

Task B0 grep 命中数超出 [5, 50]?
  → 停, 列命中明细给 user, 等决策
  → 不要 "猜 user 想要哪些"

Task B 改某文档时 markdown 渲染破坏?
  → 停, 报告破坏点, 等 user
  → 不要尝试 "再修"

Task C smoke NaN (PASS_4 fail in critical fields)?
  → 不 launch full V13, 写 SMOKE_FAILURE_REPORT, slot 2 维持空, 等 user
  → 不要 "再 smoke 一次试试"

Task C smoke OOM?
  → 报告 GPU memory peak, slot 2 维持空, 等 user 决定是否减 batch
  → 不要自行减 batch launch (污染配置)

Task C smoke timeout (> 15 min)?
  → 报告, 等 user. 不延长 timeout

Task D launch 5 min 后 dead?
  → kill 残留 process, 写 LAUNCH_FAILURE_REPORT, 等 user
  → 不要 "再 launch 一次"

Task D launch 后 OOM (V13 与 V18 抢同卡)?
  → kill V13, 报告. user 可能需要重新分配 GPU
  → 不要动 V18

Task E git diff --stat 显示意外文件改动?
  → 停, 不 commit, 报告 unexpected files, 等 user
  → 不要 git reset

Task E push 时 git conflict?
  → git fetch + git log 显示远程改动, 报告 conflict 文件, 等 user
  → 不要 git push --force

任何步骤遇到 "看起来应该有的 CLI flag / 文件 / 字段 不存在"?
  → 停, 报告. 不杜撰 fallback, 不 grep 历史 commits 找消失的 flag
```

---

## §8 — 总时长估计 (B22+B25 修复后)

| Task | 时长 |
|---|---|
| A 文档修订 (含 spot-check) | 20 min |
| B V21 retire (含 preflight grep) | 25 min |
| C V13 smoke (含 yaml prep + 5-20 min 训练) | 35 min |
| D V13 full launch + 5 min 验证 | 10 min |
| E 执行报告 + push (含 git diff --stat 验证) | 15 min |
| **总** | **~105 min** |

(v1 估计 65 min 不准. v2 因增加 spot-check / preflight grep / 完整 12 项 attestation 而增长.)

GPU 占用增量: slot 2 从空 → V13 from-scratch (~7 天). Slot 1 (V18) 不动. Slot 3 不动 (空).
