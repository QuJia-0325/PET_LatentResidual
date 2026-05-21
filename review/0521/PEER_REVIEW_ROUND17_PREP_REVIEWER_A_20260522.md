# Round 17-Prep Peer Review — Reviewer A (independent)

- date: 2026-05-22
- reviewer: **Reviewer A** (GitHub Copilot, Claude Opus 4.7 xhigh, independent draft, did not see Reviewer B/C/D drafts)
- substrate verified via: yaml load + JSON top-level key inspection + find per-slice CSV + eval script CLI flag grep + V7 yaml line-by-line inspection
- 主审对象: `CODEX_TASK_ROUND17_F0_A4_20260522.md` 执行可行性

---

## 0. push 决策 + One-line verdict

**push 决策 = MODIFY-BEFORE-PUSH** (≥ 5 HIGH 必改, 否则 codex 100% 失败)

**One-line**: task md 战略层 OK (Round 17 共识忠实), 但**执行层有 5 个 HIGH fabrication / inversion bugs** (F0 数据源路径错, V7 yaml 字段路径错, eval CLI flag 错, F0 substrate guard 反逻辑, V7 baseline D-stage 数字错), 3 个 MED 缺漏 (partial completion 协议, stop rule anti-check 太窄, scipy 依赖未 verify). 修后可 push.

---

## 1. 5 个 HIGH 必改项 (codex 拿现稿会立即失败)

### H1 [HIGH] F0 数据源路径错 — codex `load_per_slice` 会 KeyError / FileNotFoundError

任务 md §2 `DATA` dict 的 V13/V14/V18 路径**指向 summary-only JSON**, 不含 per-slice 数据.

**实测 V13 JSON 顶层 keys**:
```python
['tag', 'meta', 'summary_psnr_clip3', 'summary_chain_mse', 'headline']
# summary_psnr_clip3 只有 mean/std/min/max, 无 per-slice 数组
```

实际 per-slice CSV 位置 (find 结果):
- ✓ V7.best/V7.last per-slice: `review/0511/fullval_psnr_clip3_20260516_173941/artifacts/planf_v7_best_fullval_psnr_chain_mse_per_slice.csv`
- ✓ V8.best/V8.last per-slice: 同目录
- ✓ V6_NOISE.best/.last per-slice: 同目录
- ✓ V18.last per-slice: `review/0517/V18_decoder_lora/fullval_eval_20260518/artifacts/v18_last_fullval_psnr_chain_mse_per_slice.csv`
- ✗ **V13.best / V14.best / V18.best per-slice CSV NOT in repo** (canonical eval 自动写 per-slice CSV 但 V13/V14/V18.best 这几次 eval 的 CSV 没 commit)

**修法**:
1. §F0.0 (新增子任务): codex 先 `ls` 远端 server 的 eval output 目录:
   ```bash
   ls /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0516_runs/V13_true_image_aux_ablation/run/first_hop_224_v13_true_image_aux_off/
   ls .../V14_true_d_pure/run/first_hop_224_v14_v7_seed1337/
   ls .../V18_decoder_lora/run/first_hop_224_v18_decoder_lora/
   # 找 *_per_slice.csv (canonical eval 自动写, 文件应该在)
   ```
   若 CSV 存在 → `cp` 进 `review/0521/per_slice_cache/`, commit
   若 CSV 不存在 → 重跑 V13/V14/V18.best canonical eval (canonical 脚本line 274 自动写 per-slice CSV, **不需要 patch**, ~30 分钟 × 3 ckpts ≈ 1.5 小时, 不是 0 GPU)

2. §2 `DATA` dict 改为 per-slice CSV 路径, 不是 JSON 路径:
   ```python
   DATA = {
       'V7.best':       'review/0511/fullval_psnr_clip3_20260516_173941/artifacts/planf_v7_best_fullval_psnr_chain_mse_per_slice.csv',
       'V13.best':      'review/0521/per_slice_cache/v13_per_slice.csv',  # 待 codex 准备
       'V14.best':      'review/0521/per_slice_cache/v14_per_slice.csv',  # 待 codex 准备
       'V18.best':      'review/0521/per_slice_cache/v18_best_per_slice.csv',  # 待 codex 准备
       'V18.last':      'review/0517/V18_decoder_lora/fullval_eval_20260518/artifacts/v18_last_fullval_psnr_chain_mse_per_slice.csv',
   }
   ```

3. §F0.4 "F0 是 0 GPU, 30 分钟跑完" 表述 **降级**: "F0 < 1 hour CPU **若 per-slice CSV 全 ready**; 否则 ≤ 2 hour GPU + CPU (3 ckpts re-eval + paired-t)"

### H2 [HIGH] V7 yaml 字段路径错 — A4 yaml clone python 脚本会 KeyError

任务 md §A4.1 写:
```python
d['transport']['image_aux']['lambda_start'] = 0.08
d['transport']['image_aux']['lambda_max']   = 0.08
```

**实测 V7 yaml 实际路径** (line 258-263):
```yaml
training:
  image_aux:               # ← 是 training.image_aux, 不是 transport.image_aux
    lambda_start: 0.04
    lambda_max: 0.04
```

V7 yaml **没有** `transport.image_aux` 路径. python `d['transport']['image_aux']` 会 KeyError. (V7 yaml 有 `loss.image_aux.*` 子项, 但那是 l1_weight / ssim_weight / seam_weight, 不是 lambda)

**修法**: §A4.1 python 改:
```python
d['training']['image_aux']['lambda_start'] = 0.08
d['training']['image_aux']['lambda_max']   = 0.08
```

§A4.1 表格里 4 字段也改:
- `training.image_aux.lambda_start` (不是 `transport.image_aux.lambda_start`)
- `training.image_aux.lambda_max`

§A4.1 verify 脚本里的 `ALLOWED` set 同改:
```python
ALLOWED = {'output_dir', 'run_name', 'training.image_aux.lambda_start', 'training.image_aux.lambda_max'}
```

§5 自检里:
```bash
python3 -c "import yaml; y=yaml.safe_load(open('...')); assert y['transport']['image_aux']['lambda_max']==0.08"
```
也改为 `y['training']['image_aux']['lambda_max']`.

### H3 [HIGH] eval 脚本 CLI flag 错 — codex 会 `unrecognized arguments` 失败

任务 md §F0.1 和 §A4.4 都用 `--output-dir`:
```bash
python review/0505/operator/scripts/eval_first_hop_fullval_psnr_chain_mse.py \
  ...
  --output-dir review/0521/A4_image_aux_lambda_08/fullval_eval \
```

**实测脚本 argparse**:
```python
p.add_argument("--out-dir", ...)  # 是 --out-dir, 不是 --output-dir
```

**修法**: §F0.1 / §A4.4 / 任何 eval 脚本调用 `--output-dir` → `--out-dir`.

### H4 [HIGH] F0 substrate guard 反逻辑 — V18-cap NORMAL 会被错误计入 paired-t

任务 md §2.2 脚本:
```python
if 'V18-cap' in name and tp != 'NORMAL':
    continue  # cap CSV 仅 direct decode; 不与 chain PSNR 同 substrate
```

**逻辑反了**:
- 注释说 V18-cap = direct decode, 不与 chain PSNR 同 substrate → 应**完全跳过 V18-cap**
- 代码却**只跳过 V18-cap 的 D20/D10/D4**, 保留 V18-cap NORMAL → V18-cap NORMAL (direct decode) 会被和 V7 NORMAL (chain rollout) paired-t, **substrate 不一致**

任务 md §6 NOT-DO #9 自己也说:
> F0 把 V18-cap (direct decode) 与 V7 chain PSNR 混算 paired-t — substrate 不一致

但 §2.2 脚本**自己违反**这条 NOT-DO. acknowledgment-then-violate 模式, 同 R16 B61 / R17 B61.

**修法**: §2.2 改为:
```python
if 'V18-cap' in name:
    continue  # excluded: V18-cap CSV is direct decode, V7/V13/V14/V18 chain PSNR are different substrate
```

或直接从 DATA dict 删 `V18-cap.last` 项 (Stage F0b 单独写 task, 不在本 F0 范围).

### H5 [HIGH] V7 baseline D-stage 数字错 — A4 report 模板基线对不上 PLANF

任务 md §A4.5 V7.best 基线行:
```markdown
| V7.best | 160000 | 35.4253 | 35.8105 | 36.3680 | **36.7810** | (ref) |
```

**实测 PLANF L68 V7.best 实际值**:
```
| planf_v7_best | V7 | best | 160000 | 35.4354 | 35.8194 | 36.3736 | 36.7810 | 36.1023 |
```

差异:
| stage | task md | PLANF | Δ |
|---|---:|---:|---:|
| D20 | 35.4253 | **35.4354** | -0.0101 |
| D10 | 35.8105 | **35.8194** | -0.0089 |
| D4 | 36.3680 | **36.3736** | -0.0056 |
| NORMAL | 36.7810 | 36.7810 | 0 (匹配) |

NORMAL 对, 但 D-stages 全部偏低 ~0.01 dB. 这会让 A4 vs V7 对比表的 D-stage 显示假改善 (即使 A4 退到 V7 水平, 也会显示 +0.01 dB).

**修法**: §A4.5 V7 基线行改为 PLANF L68 实际值: D20=35.4354, D10=35.8194, D4=36.3736.

---

## 2. Q1-Q8 逐条

### Q1 — REJECT (现稿 F0 数据源伪造)
- V18.best / V13 / V14 JSON **不含 per-slice 数据**, 只有 summary
- 见 H1 修法

### Q2 — REJECT (现稿 substrate guard 反逻辑)
- 见 H4 修法

### Q3 — REJECT (现稿 V7 yaml 字段路径错)
- `transport.image_aux.*` 不存在, 应是 `training.image_aux.*`
- ALLOWED 字段 4 个是对的, 但路径 prefix 全错
- V7 yaml 还有 `loss.image_aux.{l1_weight,ssim_weight,seam_weight,border_*}` 子项 — task md verify 脚本如果 `ALLOWED = 4`, 这些子项不在 ALLOWED 但只要不改就不会 trigger fail, 所以 OK
- 见 H2 修法

### Q4 — MODIFY (建议显式声明 from-scratch)
- task md launch command 没 `--resume`, 实际是 from-scratch ✓
- 但 task md 文字**没明确写 "from-scratch (not resume)"** → codex 可能 "helpfully" 加 --resume from V7 best.pt
- **修法**: §3.1 launch 上方加一行: "**A4 = from-scratch training, do NOT use --resume**. V7 best.pt 作为对比 baseline 不作为 init."

是否真应 from-scratch?
- from-scratch (~7 天) → A4 vs V7 是干净 single-variable disambig (clone V7 yaml + image_aux lambda 唯一变量, 同 seed/max_steps), 最大 paper-quotable
- resume from V7 best.pt + 10K fine-tune → 类似 A3 patch 实验, 只测 "high lambda 短训" 不测 "high lambda 全程影响", **EV 低**
- Reviewer A 同意 from-scratch, 但 task md 必须**显式**写

### Q5 — MODIFY (stop rule anti-check 太窄, B74 未必挡得住)
- 现 anti-check: `find review/0521 -name '*A4*v2*.yaml'`
- 漏: codex 可创建 `A4_image_aux_lambda_12.yaml` (lambda=0.12), `A4b_image_aux_schedule_ramp.yaml`, `A5_image_aux_*.yaml` 全绕过
- §3.5 verdict 表 3 个分支结尾说 "无论 outcome 不 relaunch A4-v2 / v3" ← 文字 OK 但 anti-check 不严

**修法**: §5 anti-check #3 改为更广 pattern:
```bash
anti_check "no additional image_aux variant yaml" bash -c "ls review/0521/*image_aux*lambda*.yaml 2>/dev/null | grep -v 'A4_image_aux_lambda_08\.yaml' | grep -q ."
anti_check "no A4b/A5/A6 yaml" bash -c "find review/0521 -maxdepth 2 -name '[Aa]4*' -name '*.yaml' 2>/dev/null | grep -v 'A4_image_aux_lambda_08' | grep -q ."
```

### Q6 — MODIFY (partial completion 协议缺失)
- 现 §4 只描述 happy-path commit 顺序
- 若 A4 训到 step 100K 服务器挂, codex 不知该 commit 当前 metrics 还是 silent 等

**修法**: 加 §4.4 "partial completion 协议":
```markdown
### 4.4 Partial completion 协议

A4 训练中断 (e.g. server reboot, OOM):
- 立即 `cp $A4_OUT/metrics.jsonl review/0521/A4_image_aux_lambda_08/A4_metrics_partial_<step>_<TS>.jsonl`
- 立即 commit: `git commit -m "Round 17 A4: partial training snapshot at step=<X>, run interrupted"`
- 写 `review/0521/A4_image_aux_lambda_08/A4_STATUS_INTERRUPTED.md` 说明中断原因 + 最后 step + 是否重启
- **不要 silent 等 user 发现**

F0 数据源缺失 (e.g. V7 per-slice CSV 找不到, eval re-run 也失败):
- 立即 commit `review/0521/F0_STATUS_BLOCKED.md` 描述卡在哪
- 不写 F0_paired_t_report.md (避免 partial report 被误用)
```

### Q7 — MODIFY (机械自检漏 4 个关键 check)

补充 §5 自检:

```bash
# Row 8: A4 yaml loss.image_aux.* 子项未改 (防 image_aux subweight 漂移)
check "A4 yaml loss.image_aux.l1_weight == V7" python3 -c "
import yaml
v7 = yaml.safe_load(open('review/0505/local/configs/V7_gronwall_raw.yaml'))
a4 = yaml.safe_load(open('review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08.yaml'))
for sub in ['l1_weight','ssim_weight','seam_weight','border_width','border_weight','seam_patch_size']:
    assert v7['loss']['image_aux'][sub] == a4['loss']['image_aux'][sub], f'{sub} drifted'
"

# Row 9: A4 训练 log 含 lambda_img=0.0800 (verify yaml 真生效)
if ls review/0521/A4_image_aux_lambda_08/A4_train_*.log 2>/dev/null | grep -q .; then
    check "A4 train log lambda_img=0.0800" grep -q 'lambda_img=0\.0800' review/0521/A4_image_aux_lambda_08/A4_train_*.log
fi

# Row 10: F0 脚本用 scipy.stats.ttest_rel (防 codex 用 unpaired t)
if [ -f tools/paired_t_v18_vs_v7.py ]; then
    check "F0 script uses ttest_rel" grep -q 'from scipy\.stats import ttest_rel' tools/paired_t_v18_vs_v7.py
    anti_check "F0 script NOT use ttest_ind" grep -q 'ttest_ind' tools/paired_t_v18_vs_v7.py
fi

# Row 11: F0 report 每行 n=7403 (防 codex 在缺数据时 silently 用更少 slice)
if [ -f review/0521/F0_paired_t_summary.json ]; then
    check "F0 report n=7403" python3 -c "
import json
d = json.load(open('review/0521/F0_paired_t_summary.json'))
for tp, comps in d.items():
    for name, r in comps.items():
        assert r['n'] == 7403, f'{tp}/{name} n={r[\"n\"]}, expected 7403'
"
fi

# Row 12: scipy 可用 (防 server 上没 scipy)
check "scipy installed" python3 -c "from scipy.stats import ttest_rel"
```

### Q8 — MODIFY 部分条款 (V14b/V14c, paper claim, A3 命名)

| 禁项 | 评估 |
|---|---|
| 改 train_first_hop.py | KEEP — 必要 |
| 改 V7 yaml 本身 | KEEP — 必要 |
| 改 V18/V13/V14 已有 ckpt 或 yaml | KEEP — 必要 |
| V19/V18-clean/V18-rank-sweep | KEEP — Round 17 共识 |
| **V14b/V14c (多 seed)** | **MODIFY** — Round 17 整合 §3.2 明确 "V14b 可选, spare GPU slot only". task md 现稿全禁太严. 改为 "未经 user 显式批准不启 V14b" |
| A4-v2/v3 | KEEP, 但 anti-check 收紧 (Q5) |
| arch/data pivot | KEEP |
| F0 skip sanity | KEEP — 必要 |
| F0 V18-cap 混算 | KEEP — 但 §2.2 代码必须配合修 (H4) |
| **paper 草稿宣称 V18 +0.06 显著** | **MODIFY** — codex 不写 paper, 此条放 task md 走偏. 改为 "F0 report 不要写'V18 +0.06 显著'结论, 只 report mean Δ/t/p; 解读由 user 做" |
| **V18-cap.last 命名为 "A3"** | KEEP, 但**澄清**: codex 是否会在 F0 report 里用 "A3" 指代 V18-cap? 若 F0 report 必引 V18-cap 数据 (e.g. 作为 substrate-mismatch caveat), 应说 "V18-cap" 不说 "A3 result". §6 现有条款 OK |
| **push narrative/claim 类 markdown 前需 user 复审** | **MODIFY** — "数据" vs "解读" 边界 unclear. 改为: codex 可 push (1) yaml, (2) eval 输出 JSON/CSV, (3) script (paired_t_*.py), (4) **状态报告** (e.g. A4_REPORT.md 只填数字, verdict 段保持 placeholder); 不可 push (5) 含 "V18 dead/alive" / "paper 应包含 X" / "narrative 应选 Y" 类**解读语句**的 md |

---

## 3. 必改项汇总表

| § | 原文 | 修改 | severity |
|---|---|---|---|
| §2 DATA dict | `'V13.best': 'review/0516/full_eval_json/v13_*.json'` (V18.best/V14 同) | 改为 per-slice CSV 路径 (或 codex 先准备 CSV) | **HIGH (H1)** |
| §2 脚本前置 | 无 V13/V14/V18.best per-slice 准备步骤 | 加 §F0.0 子任务: 定位/复制/重跑 per-slice CSV | **HIGH (H1)** |
| §A4.1 yaml 改字段 | `transport.image_aux.lambda_*` | `training.image_aux.lambda_*` | **HIGH (H2)** |
| §A4.1 verify ALLOWED | `'transport.image_aux.lambda_start', 'transport.image_aux.lambda_max'` | `'training.image_aux.lambda_start', 'training.image_aux.lambda_max'` | **HIGH (H2)** |
| §5 自检 A4 yaml check | `y['transport']['image_aux']['lambda_max']` | `y['training']['image_aux']['lambda_max']` | **HIGH (H2)** |
| §F0.1, §A4.4 eval flag | `--output-dir` | `--out-dir` | **HIGH (H3)** |
| §2.2 substrate guard | `if 'V18-cap' in name and tp != 'NORMAL': continue` | `if 'V18-cap' in name: continue` | **HIGH (H4)** |
| §A4.5 V7 baseline | D20=35.4253, D10=35.8105, D4=36.3680 | D20=35.4354, D10=35.8194, D4=36.3736 | **HIGH (H5)** |
| §3.1 launch | (无明确说明 from-scratch) | 加 "A4 = from-scratch, do NOT use --resume" | MED |
| §5 anti-check #3 | 仅 `find ... -name '*A4*v2*'` | 扩为 `ls review/0521/*image_aux*lambda*.yaml \| grep -v lambda_08` | MED |
| §4 commit | 无 partial completion 协议 | 加 §4.4 (Q6) | MED |
| §5 自检 | 漏 4 个 check (lambda_img log / ttest_rel / n=7403 / scipy / loss.image_aux subweight) | 加 Row 8-12 (Q7) | MED |
| §F0.4 时间估计 | "F0 是 0 GPU, 30 分钟" | 降级为 "若 per-slice CSV 全 ready < 1h CPU, 否则 ≤ 2h GPU+CPU" | LOW |
| §6 NOT-DO #10 | "paper 草稿宣称 V18 显著" 由 codex 负 | F0 report 不写解读, 由 user 做 | LOW |
| §6 NOT-DO #5 (V14b) | 全禁 | 改 "未经 user 显式批准不启 V14b/c" (与 Round 17 整合 §3.2 一致) | LOW |

---

## 4. 新偏差 B75-B78

### B75 [HIGH] Fabricated CLI flag (`--output-dir`)

任务 md 多处用 `--output-dir`, 实际脚本是 `--out-dir`. 这是**字面 fabrication** — task md 作者没 verify 脚本 argparse.

**模式**: claude 在写 task md 时**凭直觉**生成 CLI flag (`--output-dir` 更常见英文写法), 没 grep 脚本.

**修法**: 加 standing rule B75 "task md 引用任何 CLI flag 必须先 `grep add_argument` verify".

### B76 [HIGH] Fabricated yaml field path (`transport.image_aux`)

任务 md 用 `transport.image_aux.*` 路径, V7 yaml 实际是 `training.image_aux.*`. 与 B75 同型 — 凭直觉编路径 (V7 yaml 顶层有 `transport:` block 所以听起来对), 没 verify.

V7 yaml 顶层 block 包括: `training:`, `transport:`, `loss:`, `optimizer:`, ... `transport:` block 是 trainer 内部 transport 配置 (step_weights / lambda_roll), 与 `training.image_aux` 是不同 block.

**修法**: 加 standing rule B76 "task md 引用任何 yaml 字段路径必须先 `python3 -c \"import yaml; print(yaml.safe_load(open('<path>')))\"` verify".

### B77 [MED] Substrate guard inversion (acknowledgment-then-violate)

§6 NOT-DO #9 明确禁 "F0 把 V18-cap 与 chain PSNR 混算 paired-t", §2.2 注释也说 "substrate 不一致". 但 §2.2 代码**自己违反**, 只跳过 D20/D10/D4, 保留 NORMAL → V18-cap NORMAL (direct decode) 与 V7 NORMAL (chain) 混算.

acknowledgment-then-violate 同 R16 B61 / R17 B61 / R17 B62. **claude 在 markdown 撰写时连续违反自己声明的 standing rule**.

**根本机制**: claude 在写代码段时**复制了一个看起来合理的 guard pattern** (`if X and Y: continue`), 没 trace 它和注释的逻辑等价性. 

**修法**: standing rule B77 "task md 中的 'NOT' 语句必须配对一个机械可测的 anti-check 或 assert; 自然语言禁令不可信".

### B78 [MED] V7 baseline 数字 silent drift (~0.01 dB)

§A4.5 V7 D-stage 数字与 PLANF Table L68 全部低 ~0.01 dB. 来源不明:
- 可能 task md 作者从 stale 来源 copy
- 可能 task md 作者口算 V7 D-stage (基于 V14=36.7806 / V13=36.4943 trying to backfit)
- 可能完全编造

无论哪个, 后果是 A4 vs V7 D-stage 对比表显示**假改善 +0.01 dB**, 误导后续解读.

**修法**: standing rule B78 "task md 中任何 historical baseline 数字必须 verbatim copy from canonical source artifact, 不允许重新打字; 若打字必须 cross-verify".

---

## 5. 元评论 (Reviewer A 立场)

任务 md 战略层 (Round 17 共识忠实) + 文件结构 (9 章节 / 5 atomic commits / 7+4 mechanical check) **质量很高** — 这是项目史上最严密的 codex task md 之一. 显示 R17 整合后 claude 已经 internalize "task md 必须 mechanical-verifiable" 这条 standing rule.

但**执行层 fabrication 5 处** (H1-H5) 暴露**根本机制**: task md 战略骨架是 user + reviewer 共识产出, 但**填充阶段** (CLI flag / yaml field / file path / baseline 数字) **claude 全靠记忆, 没 verify**. 5 个 HIGH 都是同型 — "看起来合理但没核对" 错误. 这与 R17 prompt B61 (V14 单点当 σ, acknowledgment-then-overreach) 同源 — **claude 在 substrate-dense 内容中倾向相信记忆**.

**与 R16 / R15 对比**:
- R16 prompt: 数字层错 (B61 direct/chain 混淆, B64 A3 报告 wording 漂移) — claude 解读层漂移
- R17 prompt: 方法论层错 (B61 SNR 单点当 σ, B62 Grönwall overreach) — claude 推理层漂移
- R17-prep task md: **执行层错** (H1-H5 全是 verify 类) — claude 实施层漂移

漂移层级在向下走. 这意味着 codex 拿现稿执行**100% 失败**:
- H1 (F0 数据源): `load_per_slice` raises FileNotFoundError 或 returns None
- H2 (yaml field): `d['transport']['image_aux']` raises KeyError
- H3 (CLI flag): eval 脚本 raises `unrecognized arguments: --output-dir`
- H4 (substrate guard): F0 report 出错误数字 (V18-cap NORMAL 与 V7 chain NORMAL 混算)
- H5 (baseline 数字): A4 report 表显示假改善

**给 user 的建议**:
- **不要 push 现稿**. 修 H1-H5 后才能 push.
- 修复路径: claude 自己用 verify 脚本逐项过一遍 (yaml 字段 / CLI flag / 文件路径 / 数字 verbatim copy), 再加 §5 自检 Row 8-12.
- 长期 standing rule B75-B78 必须 mark, R18+ task md 必须遵守.
- 短期: 修后让 R17-prep 重 review 一轮 (Reviewer B/C/D), 不要单方面 claude 修完直接 push.
- 战略层不动: F0 + A4-light + paper draft 三轨并行的整体框架是对的.

**Reviewer A 战略立场 (与 R17 一致)**:
| 维度 | 立场 |
|---|---|
| F0 是 paper gate | YES (但前置: per-slice CSV 必须先 ready) |
| A4 from-scratch image_aux=0.08 | YES (但 task md 必须显式说 from-scratch) |
| Push 现稿 | **NO** (5 HIGH 必修) |
| Push 修订稿 (H1-H5 + B75-B78) | YES |

---

## 6. 输出元

- reviewer: **Reviewer A**
- methodology: V7 yaml 行级 inspection + V13 JSON top-key 实测 + find per-slice CSV + eval 脚本 argparse grep + PLANF baseline cross-check + task md 代码段逻辑 trace
- 未与 Reviewer B/C/D 交流
- 关键独立发现:
  1. H1 — F0 DATA dict 路径全错 (JSON 不含 per-slice)
  2. H2 — `transport.image_aux` 路径在 V7 yaml 不存在 (实际 `training.image_aux`)
  3. H3 — `--output-dir` 是 fabrication (实际 `--out-dir`)
  4. H4 — F0 substrate guard 反逻辑 (V18-cap NORMAL 会被错误计入 chain paired-t)
  5. H5 — A4 报告模板 V7 baseline D-stage 全偏低 ~0.01 dB
  6. B75-B78 — 4 个新 standing rule 都是 verify-before-write 同型
