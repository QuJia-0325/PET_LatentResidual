# Codex Task — Phase A (v3, Round 9 hard-fixed)

- date: 2026-05-18
- branch: foc_lite_hop0
- version: **v3** (Round 9 整合后局部修订, supersedes [CODEX_TASK_PHASE_A_v2_20260517.md](./CODEX_TASK_PHASE_A_v2_20260517.md))
- status: **READY TO PUSH after user 过一眼 + 阶段 A 完成后另行起 codex 代码深审 task**
- 设计来源: v2 + [REVIEW_INTEGRATION_round9_20260517.md](./REVIEW_INTEGRATION_round9_20260517.md) 修订选 C (2 hard fix + 5 should-fix)
- 与 v2 的差异: 仅 9 处局部修订 (Round 9 修 7 + Round 10 增 2; ~47 行 markdown), 文档主结构、§4 D / §5 E / §6 NOT-DO / §7 决策树 不变
- 硬约束: 内存 max 3 任务. Slot 1 = V18 在跑 (不可动). 本 task 最多新增 slot 2 = V13. Slot 3 永远 0.

---

## §0 — 给 codex 的 5 句话总览 (不变)

1. **不停 V18**. 不改 V18 yaml. 不改 train_first_hop.py 任何代码.
2. 5 件事: Task A 改 1 份 doc / Task B V21 retire 4 份 doc (白名单 grep) / Task C V13 200-step smoke / Task D V13 full launch / Task E 报告 + push.
3. **每个 task 末尾都重复一次**: fail → 停下报告, 不要尝试 fix, 不要 retry, 不要 escalate.
4. V13 是 from-scratch (yaml 无 resume_from), **不需要 `--resume`**.
5. 本版本经过 Round 8 + Round 9 + Round 10 + Round 11 四轮验证, 9 处 v3 修订 (Round 9 修 7 + Round 10 增 2) 针对 2 hard blocker (B26 / B-V21-scope) + 7 should-fix (B27/B28/B29/B30/B-getctime + R10 git add/diff/item 6/timeout-defensive). 不要绕过任何修复条目.

---

## §0.5 — v2 → v3 修订摘要 (9 处: Round 9 修 7 + Round 10 增 2)

| # | 位置 | 性质 | 内容 |
|---|---|---|---|
| 1 | §C0 末尾 | assert 化 (B30) | SUPPORTED_FLAGS 改 print → if-fail-exit |
| 2 | §C1 python 脚本 | 加字段 (B26) | `cfg["training"]["log_interval"] = 10` (smoke 200 step → ~20 行 jsonl) |
| 3 | §C2 launch 前 | 加 START_TS (B-getctime) | `START_TS=$(date +%s)` 记录 wall clock 起点 |
| 4 | §C4 PASS_2 | 阈值降级 (B26) | `N_TRAIN >= 50` → `N_TRAIN >= 15` (加注释说明依赖 log_interval=10) |
| 5 | §C4 PASS_5 | 重写 (B-getctime) | 删 `os.path.getctime/getmtime`, 改用 `END_TS - START_TS` 算 elapsed |
| 6 | §B0 | 重写 (B-V21-scope hard) | grep 全目录 → 白名单 4 文档 only, sanity 范围 **[10, 35]** (Round 8 实测 ~21 + ±50% margin, 与 §B0 脚本一致) |
| 7 | §1 A1 + §E1 §6 | 文档修正 (B27 + B28 + B29) | §A1 patch 表加 §2.3 row 1 修正; §A1 §2.2 改区间措辞; §E1 §6 attestation 12→16 项 |
| 8 | §E2 + §E1 §8 | git add 显式 + diff 估计 (Round 9 §5.3 should-fix) | `git add` 通配 → `${SMOKE_YAML}` 单文件; §8 diff 估计 "~+60 行" → "~+80-130 行" |
| 9 | §C3 + §E1 §6 item 6 | defensive + 措辞 (Round 10 应修) | timeout 分支补 `END_TS=$(date +%s)` 便于审计; attestation item 6 改 "未自发起任何 review / audit" |

其余 v2 段落 (§0 / §A0 / §A2 / §A3 / §A4 / §B1 / §B2 / §B3 / §C3 / §C4 PASS_1/3/4 / §C5 / §4 / §5 / §6 / §7 / §8) **不变**, 沿用 v2 内容. 本文档仅完整列出**改动段落**, 不重复 unchanged 段落.

阅读 v3 时, 未列出的段落以 v2 为准 (v2 仍在仓库, 不删除).

---

## §1 — Task A 修订: §A1 patch 表扩到 3 行 (B27 + B28)

(其余 A0 / A2 / A3 / A4 不变, 沿用 v2.)

### A1 (v3 重写): 修正 §2.1 stale facts + §2.2 EV + §2.3 row 1

V18_design_rationale.md 现存 3 处 stale facts, **必须**一次性全修, 否则文档自相矛盾 (B27 防御).

**操作**: 在 §2.1 / §2.2 / §2.3 三处, 用编辑器手工 patch:

| # | 位置 | 原文 | 改为 |
|---|---|---|---|
| 1 | §2.1 行 1 | `LoRA rank=8，alpha=16 (scale=2.0)` | `LoRA rank=32，alpha=16 (scale=0.5; V18 launch 前从 rank=8 调整, 真实 trainable=589,824 ≈ 2.5% of decoder, 见 AUDIT_LORA_PARAM_COUNT_20260517.md)` |
| 2 | §2.1 行 5 | `KL pull-back loss (λ_kl=0.5)` | `KL pull-back loss (λ_kl=0.05; V18 launch 前从 0.5 调小, 防 KL 主导 total loss)` |
| 3 | **§2.2 行 V18** (B28 修正, Round 9 共识) | `0.5 - 2.0 dB（11.2 dB gap 的 5-20%）` | `0.05 - 3 dB (区间, 见 §2.4; 中位估计 0.1-0.3 dB)` |
| 4 | **§2.3 row 1 trigger 列 + 应对列** (B27 修正, Round 9 共识) | `LoRA 容量太小 (rank=8 不够)... 跑 V18-r16 / V18-r32 sweep` | `LoRA 容量仍偏小 (rank=32 也可能不够). ΔPSNR ∈ [+0.05, +0.30] | 跑 V18-r64 sweep (推迟到阶段 B 决策)` |

**B28 关键说明** (写进 commit message 注释): §2.2 改区间措辞而**不是**单点 "× 10%", 与 §2.4 EV 区间 [0.05, 3] dB 一致, 不偷偷下调 V18 success ceiling. user Round 7 `全推荐` 决策中 V18 success 概率分布**未改**.

**B27 关键说明**: §2.3 row 1 修正后, 文档 (§2.1 rank=32) 与 (§2.3 row 1 "rank=32 也不够") 自洽; sweep 行动延后到阶段 B, 不在本阶段 A 触发.

**注意**: §2.3 trigger 数字 `+0.05` / `+0.30` / `+0.05` **维持不动** (B12 防御).

**实施验证** (codex 改完后):
```bash
FILE=review/0517/V18_decoder_lora/V18_design_rationale.md

# 验证 3 处 stale facts 全修
grep -n "rank=32" "$FILE"           # 期望 ≥ 2 行 (§2.1 + §2.3)
grep -n "λ_kl=0\.05" "$FILE"        # 期望 ≥ 1 行 (§2.1)
grep -nE "0\.05 - 3 dB|0\.5 - 2\.0 dB" "$FILE"   # 期望: 新文本 1 行, 原 "0.5 - 2.0" 0 行
grep -n "rank=8 不够" "$FILE"       # 期望 0 行 (§2.3 row 1 已改) 或仅在历史引用
grep -n "λ_kl=0\.5" "$FILE"         # 期望 0 行 (§2.1 已改)
grep -nE "trigger.*\+0\.05|trigger.*\+0\.30" "$FILE"   # §2.3 trigger 数字应仍存在
```

任何一条期望未满足 → **停下报告**.

---

## §2 — Task B 修订: §B0 白名单 grep (B-V21-scope hard fix)

(B1 / B2 / B3 不变, 沿用 v2.)

### B0 (v3 重写): preflight grep, 仅白名单文件

```bash
cd /home/qujiaxiang/project/PET_LatentResidual

# 白名单: 4 个文档 (Round 8 已 verify)
WHITELIST=(
  "review/0517/REVIEW_INTEGRATION_round4_20260517.md"
  "review/0517/V18_decoder_lora/gap_decomp_fullval_20260517/GAP_DECOMP_REPORT.md"
  "review/0517/CODEX_RUNBOOK_V18_20260517.md"
  "review/0517/V18_decoder_lora/V18_EXECUTION_REPORT_20260517.md"
)

# 每个白名单文件单独 grep, 输出 per-file 命中数
echo "===== V21 grep results (whitelist preflight) ====="
TOTAL=0
for f in "${WHITELIST[@]}"; do
    if [ ! -f "$f" ]; then
        echo "FAIL: whitelist file missing: $f. STOP."
        exit 1
    fi
    hits=$(grep -c "V21" "$f" 2>/dev/null || echo 0)
    echo "$f: $hits hits"
    TOTAL=$((TOTAL + hits))
done
echo "===== END ====="
echo "Total whitelist V21 hits: $TOTAL"

# Sanity 范围: Round 8 实测 21 (19+1+1+0), 加 ±50% margin → [10, 35]
# 若超出, 文档结构可能已变, 停下报告
# (B31 防御 Round 10: §0.5 行 6 与本范围必须一致, 任一修改另一处同步)
if [ "$TOTAL" -lt 10 ] || [ "$TOTAL" -gt 35 ]; then
    echo "FAIL: whitelist V21 hit count $TOTAL out of expected [10, 35]. STOP."
    exit 1
fi
echo "PASS_B0: whitelist V21 hits in expected range"

# 单独输出 (informational only, 不作为 STOP 条件) 非白名单 V21 命中
echo ""
echo "===== Out-of-scope V21 hits (informational, NOT modified) ====="
grep -rln "V21" review/0517/ 2>/dev/null | grep -v ".pyc:" | sort | while read f; do
    # 跳过白名单
    skip=0
    for w in "${WHITELIST[@]}"; do
        if [ "$f" = "$w" ]; then skip=1; break; fi
    done
    if [ "$skip" = "0" ]; then
        hits=$(grep -c "V21" "$f")
        echo "$f: $hits hits (not modified, will list in execution report §6)"
    fi
done
echo "===== END ====="
```

**B-V21-scope 关键说明** (写进 commit message 注释): v2 用 `grep -rn "V21" review/0517/` 全目录扫描, 实测 ~169 hits 远超 sanity range, 会让 B0 STOP. v3 改为只 grep 4 白名单文件, sanity range 改 [10, 35] (Round 8 实测 21 + ±50% margin). 非白名单 V21 命中**仅 informational 输出**, 不作为 STOP 条件, 不修改, 列在执行报告 §6.

---

## §3 — Task C 修订: §C0 / §C1 / §C2 / §C4 PASS_2 / §C4 PASS_5

(C3 / C5 / 其余 C4 准则 1/3/4 不变, 沿用 v2.)

### C0 (v3 修订): SUPPORTED_FLAGS 改 assert (B30)

```bash
cd /home/qujiaxiang/project/PET_LatentResidual
git pull gitee foc_lite_hop0

# V13 原 yaml 必须存在且未被修改
ORIG=review/0516/V13_true_image_aux_ablation/V13_true_image_aux_off.yaml
test -f "$ORIG" || { echo "FAIL: V13 yaml missing"; exit 1; }
ORIG_MD5=$(md5sum "$ORIG" | cut -d' ' -f1)
echo "Original V13 yaml md5: $ORIG_MD5"

# === v3 修订 (B30): SUPPORTED_FLAGS assert 化 ===
# 验证 train_first_hop.py CLI 真的只接受 --config / --resume (B21 + B30 双防御)
SUPPORTED_FLAGS=$(grep "add_argument" train_first_hop.py | grep -oE "['\"]--[a-z_-]+['\"]" | sort -u | tr -d "'\"" | tr '\n' ' ')
echo "Detected CLI flags: $SUPPORTED_FLAGS"

EXPECTED="--config --resume"
DETECTED_NORMALIZED=$(echo $SUPPORTED_FLAGS | tr -s ' ' | sed 's/ $//')
EXPECTED_NORMALIZED=$(echo $EXPECTED | tr -s ' ' | sed 's/ $//')

if [ "$DETECTED_NORMALIZED" != "$EXPECTED_NORMALIZED" ]; then
    echo "FAIL_C0: train_first_hop.py CLI flag set changed."
    echo "  Expected: '$EXPECTED_NORMALIZED'"
    echo "  Detected: '$DETECTED_NORMALIZED'"
    echo "STOP. 不要绕过, 不要假设新 flag 用法."
    exit 1
fi
echo "PASS_C0: CLI flag set matches expected (--config --resume only)"
# === end v3 修订 ===

# GPU 状态
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
echo "Pick an idle GPU id for SMOKE_GPU below"
```

### C1 (v3 修订): smoke yaml 加 log_interval=10 (B26 hard fix)

```bash
TS=$(date +%Y%m%d_%H%M%S)
SMOKE_GPU=<填上面 nvidia-smi 空闲 GPU id, 不要与 V18 同卡>

SMOKE_YAML=review/0516/V13_true_image_aux_ablation/smoke_runs/V13_smoke_${TS}.yaml
SMOKE_OUTPUT_DIR=/data_2/qujiaxiang/outputs/PET_LatentResidual/smoke_runs/v13_smoke_${TS}
SMOKE_LOG=review/0516/V13_true_image_aux_ablation/smoke_runs/V13_smoke_${TS}.log
mkdir -p "$(dirname $SMOKE_YAML)"
mkdir -p "$SMOKE_OUTPUT_DIR"

cp "$ORIG" "$SMOKE_YAML"

python3 - <<EOF
import yaml, pathlib
p = pathlib.Path("$SMOKE_YAML")
cfg = yaml.safe_load(p.read_text())

# 5 字段 (v2 4 个 + v3 新加 log_interval):
cfg["training"]["max_steps"] = 200
cfg["output_dir"] = "$SMOKE_OUTPUT_DIR"
cfg["run_name"] = "first_hop_224_v13_smoke_${TS}"
cfg["training"]["require_fresh_output_dir"] = False

# === v3 修订 (B26 hard fix): log_interval=10 ===
# 原 V13 yaml log_interval: 50, smoke 200 step 只会写 4 行 train event
# PASS_2 阈值需要 ≥15 行, 必须把 log_interval 降到 10 (200/10=20 行, 留余量)
cfg["training"]["log_interval"] = 10
# === end v3 修订 ===

cfg["training"]["best_select_full_eval_interval"] = 100000   # 避免 200 step 内触发 full-val

p.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
print(f"smoke yaml written: {p}")
print(f"max_steps = {cfg['training']['max_steps']}")
print(f"log_interval = {cfg['training']['log_interval']}")   # v3 verify
print(f"output_dir = {cfg['output_dir']}")
print(f"require_fresh = {cfg['training']['require_fresh_output_dir']}")
EOF

# Sanity check: 5 字段全部生效
python3 -c "
import yaml
with open('$SMOKE_YAML') as f:
    cfg = yaml.safe_load(f)
assert cfg['training']['max_steps'] == 200, 'max_steps not patched'
assert cfg['training']['log_interval'] == 10, 'log_interval not patched (v3 B26 fix)'
assert cfg['output_dir'].startswith('/data_2/'), 'output_dir not in /data_2'
assert cfg['training']['require_fresh_output_dir'] == False, 'require_fresh not False'
print('smoke yaml sanity PASS (5 fields including v3 log_interval)')
"

# Sanity check: 原 yaml md5 不动
NEW_MD5=$(md5sum "$ORIG" | cut -d' ' -f1)
if [ "$NEW_MD5" != "$ORIG_MD5" ]; then
    echo "FAIL: original V13 yaml was modified ($ORIG_MD5 -> $NEW_MD5). STOP."
    exit 1
fi
echo "Original V13 yaml md5 unchanged: $NEW_MD5"
```

### C2 (v3 修订): launch 前记录 START_TS (B-getctime fix)

```bash
export CUDA_VISIBLE_DEVICES=$SMOKE_GPU

# === v3 修订 (B-getctime fix): shell 级 wall clock 起点 ===
START_TS=$(date +%s)
echo "Smoke START_TS: $START_TS ($(date -d @$START_TS '+%Y-%m-%d %H:%M:%S'))"
# === end v3 修订 ===

nohup python train_first_hop.py \
  --config "$SMOKE_YAML" \
  > "$SMOKE_LOG" 2>&1 &
SMOKE_PID=$!

echo "Smoke launched: PID=$SMOKE_PID, log=$SMOKE_LOG"
echo "Smoke output dir: $SMOKE_OUTPUT_DIR"
```

### C3 (v3 部分修订: timeout 分支补 END_TS defensive)

超时等待 30 min 逻辑**主体沒变** (沿用 v2), 仅在 timeout 分支 `exit 1` 之前补一行 `END_TS=$(date +%s)` 以便 codex 多-shell 执行时后续 PASS_5 能读到 wall clock (Round 10 agent2 备选添加).

v2 timeout 分支 原文:
```bash
if [ $((NOW - START)) -gt $TIMEOUT ]; then
    echo "FAIL: smoke timed out after ${TIMEOUT}s. Killing PID=$SMOKE_PID. STOP."
    kill -9 $SMOKE_PID 2>/dev/null
    exit 1
fi
```

v3 修订 (补两行):
```bash
if [ $((NOW - START)) -gt $TIMEOUT ]; then
    echo "FAIL: smoke timed out after ${TIMEOUT}s. Killing PID=$SMOKE_PID. STOP."
    # === v3 修订 (Round 10 Q4 defensive): timeout 分支也记录 END_TS ===
    END_TS=$(date +%s)
    echo "timeout END_TS=$END_TS, elapsed=$((END_TS - START_TS))s"
    # === end v3 修订 ===
    kill -9 $SMOKE_PID 2>/dev/null
    exit 1
fi
```

(其余 C3 逻辑 跳步/超时检查 不变。)

监控段末尾 `wait $SMOKE_PID 2>/dev/null; SMOKE_RC=$?` **之后**, 加 v3 修订:

```bash
# === v3 修订 (B-getctime fix): wall clock 终点 ===
END_TS=$(date +%s)
ELAPSED_SEC=$((END_TS - START_TS))
ELAPSED_MIN=$((ELAPSED_SEC / 60))
echo "Smoke END_TS: $END_TS, elapsed: ${ELAPSED_SEC}s (${ELAPSED_MIN} min)"
# === end v3 修订 ===
```

### C4 (v3 部分修订): PASS_2 阈值降, PASS_5 重写

PASS_1 / PASS_3 / PASS_4 不变 (沿用 v2).

```bash
METRICS=$SMOKE_OUTPUT_DIR/metrics.jsonl

# === PASS_1 (不变, 沿用 v2) ===
test "$SMOKE_RC" = "0" || { echo "FAIL_1: smoke exit code $SMOKE_RC. STOP."; cat "$SMOKE_LOG" | tail -50; exit 1; }
echo "PASS_1: smoke exit 0"

# === v3 修订 PASS_2 (B26 hard fix): 阈值 50 → 15 ===
# 依赖: §C1 已把 log_interval 从 50 改为 10. smoke 200 step → ~20 行 train event.
# 阈值 15 = 20 - 5 安全 margin (覆盖 start_step 边界 / 异常退出后少几行)
test -f "$METRICS" || { echo "FAIL_2: no metrics.jsonl at $METRICS. STOP."; exit 1; }
N_TRAIN=$(grep -c '"event": "train"' "$METRICS")
test "$N_TRAIN" -ge 15 || { echo "FAIL_2: only $N_TRAIN train rows (expected ≥15 with log_interval=10, max_steps=200). STOP."; exit 1; }
echo "PASS_2: metrics.jsonl has $N_TRAIN train rows (expected ~20)"
# === end v3 PASS_2 修订 ===

# === PASS_3 (不变, 沿用 v2) ===
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
assert len(img_vals) >= 15, f"only {len(img_vals)} train rows have 'img' field"   # v3: 15 not 50
nonzero = [v for v in img_vals if v != 0.0]
assert len(nonzero) == 0, f"FAIL_3: {len(nonzero)} train rows have img != 0: {nonzero[:5]}. STOP."
print(f"PASS_3: all {len(img_vals)} train rows have img=0.0")
EOF

# === PASS_4 (不变, 沿用 v2) ===
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
critical = ["loss", "pair", "roll"]
nonfinite = []
for r in rows:
    for k in critical:
        if k in r and not math.isfinite(float(r[k])):
            nonfinite.append((r["step"], k, r[k]))
assert len(nonfinite) == 0, f"FAIL_4: nonfinite in critical fields: {nonfinite[:5]}. STOP."
print(f"PASS_4: all critical fields finite in {len(rows)} rows")
EOF

# === v3 修订 PASS_5 (B-getctime fix): shell elapsed 替代 os.path.getctime ===
# v2 用 os.path.getctime / getmtime 估 elapsed, Linux 上 ctime 是 inode metadata change
# time 不是 creation time, 估出 ELAPSED ≈ 0 trivially pass, 失去检测意义.
# v3 用 §C2 START_TS 与 §C3 END_TS 差, 直接 wall clock.
test $ELAPSED_MIN -le 15 || { echo "FAIL_5: smoke took ${ELAPSED_MIN} min > 15 min. STOP."; exit 1; }
echo "PASS_5: elapsed ${ELAPSED_MIN} min (${ELAPSED_SEC}s) ≤ 15 min"
# === end v3 PASS_5 修订 ===
```

(C5 fail 处理不变, 沿用 v2.)

---

## §4 — Task D 不变 (沿用 v2)

(launch / 5 min 验证 / pass 准则 / fail 处理 完全不变.)

---

## §5 — Task E 修订: §E1 §6 attestation 扩 16 项 (B29) + §E2 git add 显式 + §E1 §8 diff 估计 (Round 9 §5.3 / Round 10)

### E1 模板 §6 (v3 重写, 12 项 → 16 项)

(其余 §E1 §1-§5 / §7-§9, 以及 §E3 不变, 沿用 v2. §E2 见下方 v3 修订.)

```markdown
## §6 — NOT-DO 16 项勾选 (codex 必须全部 [x])

(v3: 与 master §6 一一对应, 从 v2 的 12 项扩到 16 项)

- [ ] 1. 未停 V18 / 未改 V18 yaml 任何字段
- [ ] 2. 未 launch V14 / V9 / V21 任何变体
- [ ] 3. 未 launch V18b 任何形态
- [ ] 4. 未准备 V18-clean yaml (推迟到阶段 C)
- [ ] 5. 未做 V18 step 180K early-eval
- [ ] 6. codex 在 Phase A 执行期间未自发地起任何 review / audit / V18-clean 准备 (user 起 Round X review 不算 codex 违反)
- [ ] 7. 未改 train_first_hop.py 任何代码
- [ ] 8. 未加 LoRA→LoRA resume 支持
- [ ] 9. 未优化/重构/重命名/移动任何代码或文件
- [ ] 10. 未跑任何 eval (slot 1 V18 在用 GPU)
- [ ] 11. 未在 commit message 写"也顺便做了 X"
- [ ] 12. 未改 V13 原始 yaml (review/0516/V13_true_image_aux_ablation/V13_true_image_aux_off.yaml)
- [ ] 13. 未 commit V13 原始 yaml (它没改动, 不入此 commit)                  ← v3 新加
- [ ] 14. CLI flag 不存在时未尝试修改 train_first_hop.py 加 flag             ← v3 新加
- [ ] 15. smoke fail 时未 `rm -rf $SMOKE_OUTPUT_DIR` 重跑 (保留 failure 现场) ← v3 新加
- [ ] 16. 执行报告 + commit message 中未写 "V13 健康/稳定/收敛正常" 等 7-day 长期推断措辞 (B23) ← v3 新加
```

### E1 模板 §8 (v3 修订: diff 估计, Round 9 §5.3 should-fix)

```markdown
## §8 — git diff --stat
\`\`\`
<git diff --stat 输出>
\`\`\`
预期改动文件:
- V18_design_rationale.md (~+80 - 130 行)   ← v3 修正 (原 v2 估 ~+60, 实际 ~110)
- REVIEW_INTEGRATION_round4_20260517.md (~+25 行)
- GAP_DECOMP_REPORT.md (~+10 行)
- CODEX_RUNBOOK_V18_20260517.md (~+10 行)
- PHASE_A_EXECUTION_REPORT_<TS>.md (新建)
- smoke_runs/V13_smoke_<TS>.yaml (新建)
- smoke_runs/V13_smoke_<TS>.log (新建)
```

### E2 commit (v3 修订: git add 显式, Round 9 §5.3 should-fix)

```bash
cd /home/qujiaxiang/project/PET_LatentResidual

# === v3 修订 (Round 9 §5.3 + Round 10): git add 显式单文件, 不用通配符 ===
# (避免旧 smoke 残留被误 add; 与 v2 的 `V13_smoke_*.yaml` `V13_smoke_*.log` 通配不同)
git add review/0517/V18_decoder_lora/V18_design_rationale.md
git add review/0517/REVIEW_INTEGRATION_round4_20260517.md
git add review/0517/V18_decoder_lora/gap_decomp_fullval_20260517/GAP_DECOMP_REPORT.md
git add review/0517/CODEX_RUNBOOK_V18_20260517.md
git add "review/0517/PHASE_A_EXECUTION_REPORT_${TS}.md"
git add "${SMOKE_YAML}"     # §C1 定义的唯一单文件
git add "${SMOKE_LOG}"      # §C2 定义的唯一单文件
# === end v3 修订 ===

# 显式排除 (同 v2):
# - 不 add V13_train_*.log (rolling 7-day log)
# - 不 add *.pid
# - 不 add /data_2/ 下任何 smoke training output

# 验证 git status
echo "===== git status before commit ====="
git status
echo "===== END ====="

# 验证 git diff --stat
echo "===== git diff --cached --stat ====="
git diff --cached --stat
echo "===== END ====="

# 任何意外文件 → STOP, 不 commit, 报告

git commit -m "Phase A v3: V21 retire + design_rationale §2.4/§5.2 + V13 smoke + launch (slot 2)

Round 8 hard fix + Round 9 user 决策 C (2 hard + 5 should-fix) + Round 10 增修订:
- B26 fix: smoke log_interval=10 → metrics.jsonl ~20 行, PASS_2 ≥ 15
- B-V21-scope fix: §B0 白名单 grep (4 文档), sanity [10, 35]
- B27/B28/B29 fix: §A1 §2.3 row 1 + §2.2 区间 + attestation 16 项
- B30/B-getctime fix: SUPPORTED_FLAGS assert + START/END_TS
- Round 10 增: git add 显式 + diff 估 ~+80-130 + timeout END_TS defensive + item 6 措辞
- V13 launch: from-scratch on slot 2, alive at +5min, healthy 评估由阶段 B/C 跟踪 (B23)
- 不动 V18 (slot 1), slot 3 空"

git push gitee foc_lite_hop0
```

(§E3 完成通知 不变, 沿用 v2.)

---

## §6 — Master NOT-DO 列表 (v3 修订 1 处)

(v2 的 16 条全部保留, 仅修 item 6 stale wording → Round 10 应修)

- item 6 (v3): "codex 在 Phase A 执行期间未自发地起任何 review / audit / V18-clean 准备 (user 起 Round X review 不算 codex 违反, 本项约束对象是 codex 自身动作)"

其余 15 条不变.

---

## §7 — 失败决策树 (不变, 沿用 v2)

(完整决策树不变, 含 v2 §7 全部 11 处 fail 路径处理.)

---

## §8 — 总时长估计

(v3 局部修订对 wall clock 时长影响极小, 仍 ~105 min.)

| Task | 时长 |
|---|---|
| A 文档修订 (含 spot-check + §2.3 row 1 修正) | 22 min (+2 vs v2) |
| B V21 retire (含 whitelist preflight grep) | 22 min (-3 vs v2, 因 grep 变快) |
| C V13 smoke (含 yaml prep + log_interval 改) | 35 min (不变) |
| D V13 full launch + 5 min 验证 | 10 min (不变) |
| E 执行报告 + push (含 16 项 attestation + git diff --stat 验证) | 18 min (+3 vs v2) |
| **总** | **~107 min** |

---

## §9 — 阶段 A 完成后下一步 (新增, 非 v2)

阶段 A 完成 + push gitee + V13 在 slot 2 训练后, user 将单独起 **codex 代码深审 task md**, 范围:

| 模块 | 深审目的 |
|---|---|
| train_first_hop.py 完整 (~2800 行) | Round 8/9 已 spot-check 关键段, 但未做整体 review. 阶段 A 完成后 codex 在服务器全文 audit, 找潜在 silent bug |
| pet_lr/decoder_lora.py / model_first_hop.py | V18 LoRA 实现完整性 audit; 为阶段 C V18-clean launch 提供代码层信心 |
| pet_lr/path_guard.py / 训练器 resume 路径 | 完善 B14 standing rule, 把代码层 invariant 写入 design_rationale §5.2 |
| eval_first_hop_224_clip3.py | 阶段 B V18 200K eval 前必须 audit, 确认 PSNR_clip3 计算正确 + V7 baseline 对比逻辑 |
| RAE 子模块 (作为 V18 frozen decoder) | 阶段 C V18-clean from V7 launch 前的最后一道防线 |

**此代码深审是阶段 A 完成的下游 task, 与本 Phase A v3 task 完全独立**. 本 task 范围严格限于阶段 A 5 件事. codex 不可在 Phase A 执行中**顺手**做代码深审 (违反 §6 NOT-DO).

代码深审 task md 起草时机: Phase A v3 push gitee → codex 执行完 → user 收到 PHASE_A_EXECUTION_REPORT → 起 CODEX_CODE_DEEP_REVIEW_REQUEST_<TS>.md.

---

## §10 — v3 修订自查 (Round 10 升级: user 必须独立 grep verify, 不信 claude 自勾)

**Round 10 元教训 (B31)**: claude 自查表 13 项 ✓ 全自勾, 没机制 force 真验证 — 是 B30 同型复发. v3 起 user 验收时**不**信 claude 勾, 必须独立跑下列 grep 命令验证.

### 10.1 自动验证脚本 (user 在 **本地 macOS 仓库** 执行, 不是服务器)

**Round 11 升级 (B33a + B33b + B35)**:
- B33a: row 10 加 `-z` 空值守, 避免 `test "" = ""` vacuous pass
- B33b: 整体加 `ERR` 计数 + `exit 1` if `ERR > 0` (B30 同型第 3 层复发修复)
- B35: 节首显式标注 "本地 macOS 仓库", 区分于服务器执行的 §C0/§C1/§E2 路径

修后 user 可 `bash verify.sh && git push gitee foc_lite_hop0` 链式调用; CI / wrapper 可直接 gate `$?`.

```bash
#!/usr/bin/env bash
# 在 user 本地 macOS 仓库执行 (路径 /Users/jiaxiang/.../PET_LatentResidual)
# 不在服务器 (/home/qujiaxiang/...) 执行 — 服务器执行的是 §C0/§C1/§E2

cd /Users/jiaxiang/Desktop/文件/先进院文件/latent_flow/PET_LatentResidual
F=review/0517/CODEX_TASK_PHASE_A_v3_20260518.md
ERR=0

# helper: check <预期描述> <bash 命令 + args>; pass → ✓, fail → ✗ + ERR++
# 注: 用 "$@" 直接执行 array, 不用 eval (避免二次 parse 弄丢 \[ \$ escape)
check() {
    local desc="$1"
    shift
    if "$@" >/dev/null 2>&1; then
        echo "✓ $desc"
    else
        echo "✗ $desc"
        ERR=$((ERR + 1))
    fi
}

# helper: anti_check (反向断言: pattern 不应命中)
anti_check() {
    local desc="$1"
    shift
    if "$@" >/dev/null 2>&1; then
        echo "✗ $desc"
        ERR=$((ERR + 1))
    else
        echo "✓ $desc"
    fi
}

echo "=== Round 9 + Round 10 + Round 11 修订验证 ==="

# Row 1 (B30): SUPPORTED_FLAGS assert 而非 print
check "row1 (B30 SUPPORTED_FLAGS assert)" \
    grep -q 'FAIL_C0.*train_first_hop.py CLI flag set changed' "$F"

# Row 2 (B26): smoke yaml 设 log_interval=10
check "row2 (B26 log_interval=10)" \
    grep -q 'cfg\["training"\]\["log_interval"\] = 10' "$F"

# Row 3 (B-getctime): START_TS 在 §C2
check "row3 (B-getctime START_TS)" \
    grep -q 'START_TS=\$(date +%s)' "$F"

# Row 4 (B26): PASS_2 阈值 ≥15 active code 命中 + ≥50 active code 0 命中
check "row4 (B26 阈值 ≥15 active)" \
    grep -qE '^test "\$N_TRAIN" -ge 15' "$F"
anti_check "row4 active 残留 ≥50 = 0" \
    grep -qE '^test "\$N_TRAIN" -ge 50' "$F"

# Row 5 (B-getctime): PASS_5 ELAPSED_MIN active + getctime/getmtime active 0 命中
check "row5 (B-getctime PASS_5)" \
    grep -qE '^test \$ELAPSED_MIN -le 15' "$F"
anti_check "row5 active 残留 getctime/getmtime = 0" \
    grep -qE '^[[:space:]]*[^#].*os\.path\.getctime\(|^[[:space:]]*[^#].*os\.path\.getmtime\(' "$F"

# Row 6 (B-V21-scope): §B0 白名单 grep
check "row6 (B-V21-scope 白名单)" \
    grep -q 'WHITELIST=' "$F"

# Row 7a (B27): §2.3 row 1 改 "rank=32 也可能不够"
check "row7a (B27 §2.3 row 1)" \
    grep -q 'rank=32 也可能不够' "$F"

# Row 7b (B28): §2.2 改 "0.05 - 3 dB (区间, 中位 0.1-0.3)"
check "row7b (B28 区间 0.05 - 3 dB)" \
    grep -q '0\.05 - 3 dB' "$F"

# Row 7c (B29): attestation ≥16 项
n_attest=$(grep -c '^- \[ \] [0-9]\+\.' "$F")
if [ "$n_attest" -ge 16 ]; then
    echo "✓ row7c (B29 attestation 实际 $n_attest 项)"
else
    echo "✗ row7c (B29 attestation $n_attest < 16)"
    ERR=$((ERR + 1))
fi

# Row 8a (Round 9 §5.3): git add 用 ${SMOKE_YAML} 显式
check "row8a (git add 显式)" \
    grep -q 'git add "\${SMOKE_YAML}"' "$F"

# Row 8b (Round 9 §5.3): diff 估计 ~+80-130 行
check "row8b (diff 估计 ~+80-130)" \
    grep -q '+80 - 130 行\|+80-130 行' "$F"

# Row 9a (Round 10 Q4 defensive): timeout 分支补 END_TS
check "row9a (timeout END_TS defensive)" \
    grep -q 'Round 10 Q4 defensive' "$F"

# Row 9b (Round 10 item 6): attestation item 6 改 "未自发地"
check "row9b (item 6 措辞)" \
    grep -q 'codex 在 Phase A 执行期间未自发地' "$F"

# Row 10 (B31 mechanical): §0.5 sanity range = §B0 sanity range
# B33a 修: 加 -z 空值守, 防 vacuous pass
range_summary=$(grep '| 6 | §B0 |' "$F" | grep -oE '\[[0-9]+, [0-9]+\]')
range_script=$(grep -oE 'expected \[[0-9]+, [0-9]+\]' "$F" | head -1 | grep -oE '\[[0-9]+, [0-9]+\]')
if [ -z "$range_summary" ] || [ -z "$range_script" ]; then
    echo "✗ row10 grep 返回空 (§0.5 表结构或 §B0 注释可能已变): summary='$range_summary' script='$range_script'"
    ERR=$((ERR + 1))
elif [ "$range_summary" = "$range_script" ]; then
    echo "✓ row10 (B31 §0.5=§B0 range = $range_summary)"
else
    echo "✗ row10 不一致: §0.5=$range_summary vs §B0=$range_script"
    ERR=$((ERR + 1))
fi

# Row 11 (B12 守): §2.3 trigger 数字未改 (V18_design_rationale.md)
DR=review/0517/V18_decoder_lora/V18_design_rationale.md
check "row11 (B12 守 +0.05/+0.30 trigger)" \
    grep -q 'ΔPSNR ∈ \[+0\.05, +0\.30\]' "$DR"

# Row 12 (B34 mechanical): §0.5 摘要数 = §0.5 表实际行数 (B31 同型 calendar check)
# 摘要写 "9 处" → 表必须 9 行
# 注意 pattern 必须精确捕获 "(N 处:" 中的 N, 避免误抓 0.5/v2/v3 等其他数字
summary_count=$(grep -oE '## §0\.5 .*修订摘要 \([0-9]+ 处' "$F" | sed -E 's/.*\(([0-9]+) 处.*/\1/')
table_rows=$(awk '/^\| # \| 位置 \| 性质 \| 内容 \|$/,/^---$/' "$F" | grep -c '^| [0-9]')
if [ -z "$summary_count" ]; then
    echo "✗ row12 (B34) 摘要标题未匹配, 检查 §0.5 标题格式 (期望 '## §0.5 — v2 → v3 修订摘要 (N 处...)')"
    ERR=$((ERR + 1))
elif [ "$summary_count" = "$table_rows" ]; then
    echo "✓ row12 (B34 摘要数=$summary_count = 表行数=$table_rows)"
else
    echo "✗ row12 (B34 摘要数=$summary_count ≠ 表行数=$table_rows)"
    ERR=$((ERR + 1))
fi

echo "=== 验证结束 ==="
if [ "$ERR" -gt 0 ]; then
    echo "FAIL: $ERR 项未通过. 不可 push." >&2
    exit 1
fi
echo "PASS: 全部通过. user 可 push gitee."
exit 0
```

**user 验收 protocol** (Round 11 升级): `bash` 跑上述脚本 (或 chmod +x 后直接执行); exit code 0 = 全 ✓ 可 push, exit code 1 = 任一 ✗ 退回 claude 修. **不依赖视觉数 ✓ 行**, 由 shell exit code 自动 gate.

### 10.2 claude 自勾对照表 (仅供 reference, 不作为验收依据)

| 项 | claude 自勾 | 状态 |
|---|---|---|
| Row 1-9 (Round 8/9 修订) | 详 §0.5 表 | ✓ |
| Row 8a/b (Round 10 should-fix, Round 9 §5.3) | §E2 git add 显式 + §E1 §8 diff 估计 | ✓ |
| Row 9a (Round 10 Q4 defensive) | §C3 timeout 分支补 END_TS | ✓ |
| Row 9b (Round 10 item 6 措辞) | §6 master + §E1 §6 attestation | ✓ |
| Row 10 (B31 mechanical) | §0.5 + §B0 + §1.3 三处 [10, 35] 一致 | ✓ |
| Row 11 (B12 守) | §A4 verify grep 检查 §2.3 trigger 数字 | ✓ |

claude 自勾**不可信**, 必须 §10.1 脚本 verify.
