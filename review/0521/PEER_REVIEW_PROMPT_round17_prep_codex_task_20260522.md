# Peer Review Round 17-Prep — CODEX TASK F0 + A4 (Execution Doc)

- date: 2026-05-22
- branch: foc_lite_hop0
- 主审对象: [CODEX_TASK_ROUND17_F0_A4_20260522.md](./CODEX_TASK_ROUND17_F0_A4_20260522.md)
- 上游: [REVIEW_INTEGRATION_round17_20260522.md](./REVIEW_INTEGRATION_round17_20260522.md) 4/4 共识 = Hybrid B + F0 + A4-light
- 触发: Codex task 即将 push 给远端 GPU 服务器执行, push 前最后一道 review
- target reviewers: 2-3 个 AI 独立评审, **不互见草稿**

---

## 0. 本轮范围与边界

Round 17 整合已签 (战略), Round 17 Codex task md 已起草. **本轮只审 Codex task md 的执行可行性**:

**本轮审**:
- F0 paired-t 协议的数据源是否真实存在 (V7 per-slice 数据可能不存在, 需 codex 重跑)
- F0 paired-t 脚本模板的 substrate 一致性约束是否够严
- A4 yaml 4 字段 diff 是否真的覆盖 image_aux schedule 全部需要变的字段
- A4 stop rule "不 relaunch v2/v3" 是否真能防 scope creep
- 机械自检脚本 §5 是否覆盖所有应防的偏差
- NOT-DO 清单 §6 是否漏关键禁项 / 是否过严
- Commit 顺序 §4.2 是否在中途失败时仍 safe (e.g. A4 训到一半挂)
- 整体文件是否有 fabrication (虚构 CLI flag / 不存在的脚本 / 错误的 ckpt 路径)

**本轮不审**:
- Round 17 战略本身 (4/4 已签)
- F0/A4 是否值得做 (Round 17 已签)
- Round 16 / 15 / 14 / 13 / 12 已签内容
- A4 stop rule 是否合理 (Round 17 共识)

---

## 1. 文件结构事实锚点

Codex task md (`CODEX_TASK_ROUND17_F0_A4_20260522.md`) 含 9 个章节:

| § | 内容 | 关键 artifact |
|---|---|---|
| 0 | 5 句话总览 | F0 必做 + A4 GPU 空闲时做 |
| 1 | 物理含义 | Round 17 共识背景 / 派生事实 X1/X2/X3 |
| 2 | **F0 paired-t 协议** | `tools/paired_t_v18_vs_v7.py` (~120 行模板), `review/0521/F0_paired_t_*.{json,md}` |
| 3 | **A4 训练实验** | yaml clone V7, 仅改 4 字段; resume from V7 best.pt? **NO**, A4 是 from-scratch |
| 4 | Commit / push 协议 | 5 atomic commits |
| 5 | 机械自检脚本 | 7 PASS check + 4 anti-check |
| 6 | NOT-DO 清单 | 11 条 |
| 7 | 资料阅读顺序 | 7 个上游 reference |
| 8 | 预期 push 状态 | 5 commit 顺序 |
| 9 | 给 codex 的最后提醒 | 5 条 + 完成后摘要格式 |

---

## 2. 给 reviewer 的 8 个问题

### Q1 — F0 数据源是否真实可达?

§2.0 列了 6 个 per-slice 数据源:

- V7.best (位置**未知**, §2.1 要 codex 用 `find` 定位; 若不存在需重跑 V7 canonical eval)
- V13 / V14 (在 `review/0516/full_eval_json/*.json`, 但**这些 JSON 是否真含 per-slice 数据**, 还是只含 summary mean/std? 需 reviewer verify)
- V18.best / V18.last (在 `review/0517/V18_decoder_lora/fullval_eval_20260518/*.json`, 同上)
- V18-cap.last (在 `kl_drift_with_cap_20260519_180206/KL_DRIFT_PER_SLICE.csv`, **148060 行**, 这个是真 per-slice)

请评估:
- V13/V14/V18 的 JSON 是否真含 per-slice (codex 跑 F0 前必须 verify)?
- 若它们只含 summary, 是否需要补 §F0.0a 子任务: 重跑所有 ckpt 的 canonical eval 出 per-slice CSV?
- §2.1 给 codex 重跑 V7 的方案是否完整 (脚本是否真有 `--save-per-slice` flag, 还是需要 codex 写 patch)?

### Q2 — F0 paired-t 协议 substrate 一致性是否够严?

§2.2 脚本模板的 `DATA` dict 包含 V18-cap (KL_DRIFT_PER_SLICE.csv) 但注释说 "不能直接和 V7 chain PSNR 做 paired-t — substrate 不一致" 然后跳过它. 

请评估:
- 这个 substrate 区分是否清晰: V13/V14/V18.best/V18.last 是 **chain rollout PSNR_clip3**, V18-cap 是 **direct decode(z_GT) PSNR_clip3** 不是同一指标
- 如果 codex 误把 V18-cap 也算进去会出错吗? 脚本里 `if 'V18-cap' in name and tp != 'NORMAL': continue` 这一行其实只跳过 D20/D10/D4, NORMAL 仍会算 — 这是个 bug 还是 feature?
- 是否应在脚本里加 `substrate` 字段, 拒绝跨 substrate paired-t?

### Q3 — A4 yaml 4 字段 diff 是否完整?

§3.1 列的 4 字段:
- `output_dir`
- `run_name`
- `transport.image_aux.lambda_start: 0.04 → 0.08`
- `transport.image_aux.lambda_max: 0.04 → 0.08`

请评估:
- V7 yaml 还有 `loss.image_aux.l1_weight / ssim_weight / seam_weight / border_*` 等子项, A4 是否真的全部不动?
- A4 是不是其实还应该改某个 trainer-side image_aux 路径开关 (e.g. `transport.image_aux.enabled` 还是 `loss.image_aux.enabled`)?
- §3.1 的 yaml verify 脚本用 `ALLOWED = 4` 字段, 是否会漏掉某些必须改的字段 (e.g. seed 应保持 42 这点已 verify, 但 backbone path 是否仍指向 V7 训练用的 backbone)?

### Q4 — A4 是 from-scratch 还是 resume from V7 best.pt?

A4 §3.2 launch command **没有 `--resume`**:

```bash
nohup python train_first_hop.py \
  --config review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08.yaml \
  > "$LAUNCH_LOG" 2>&1 &
```

这意味着 A4 是 from-scratch 训 160K. 请评估:

- A4 from-scratch 是否真合理 (~7 天 GPU)? 还是应该 resume from V7 best.pt @ 160K 然后再训 10K 像 A3 那样?
- 如果 from-scratch, A4 vs V7 比较是公平的 (同 from-scratch, 同 seed=42, 同 max_steps=160K, 只差 image_aux lambda); 如果 resume, A4 是另一个 5-10K patch 实验, EV 不同
- task md 是否应明确写 "from-scratch (not resume)" 避免 codex 误加 --resume?

### Q5 — A4 stop rule 是否真能防 scope creep?

§6 NOT-DO #6: "启 A4-v2 / A4-v3 / 其它 image_aux variant — Round 17 stop rule, 防 scope creep (B74)"
§5 anti-check #3: `anti_check "no A4-v2 yaml" find review/0521 -name '*A4*v2*.yaml' 2>/dev/null | grep -q .`

请评估:
- 这个 anti-check 只检查文件名含 "v2", codex 可能创建 `A4_image_aux_lambda_12.yaml` (lambda=0.12 新探针) 绕过 — 是否应改为更广 pattern?
- A4 完成后, codex 可能"顺手"再跑一个 ramp 或 cosine schedule, 这是否应在 task md 里更显式禁止?
- §3.5 verdict 表的 3 个结果分支 (SUCCESS / MARGINAL / REGRESSION) 是否都明确说 "无论 outcome 不 relaunch"?

### Q6 — Commit 顺序中途失败的恢复路径?

§4.2 列了 5 atomic commits, 但若 A4 训到 100K 服务器挂了, commit 4 (training log + metrics snapshot) 会含 step=100000 而非 160000 — task md 没说这种部分完成的 commit 应该怎么处理.

请评估:
- 是否应加 §4.4 "partial completion 协议": A4 中断时, codex 应 commit 当前 metrics + 一行 status note, 不要 silent
- F0 失败 (e.g. V7 per-slice 找不到, codex 也跑不了 V7 eval) 时, 是否应 commit 一份 `F0_STATUS_BLOCKED.md` 解释卡在哪
- 是否应在 §0 总览加一句: "中途任何 failure 都先 commit current state 再停"

### Q7 — 机械自检 §5 是否漏关键检查?

§5 现有 7 PASS check + 4 anti-check, 含:
- F0 script / report 存在
- A4 yaml 4 字段 verify
- B42 verbatim
- CLI flag 没变
- 既有 ckpt 没被改

请评估漏的检查:
- A4 yaml `loss.image_aux.l1_weight` 等子项是否仍 = V7? (B70 image_aux overclaim risk)
- A4 yaml `seed=42` 已 check, 但 `backbone.checkpoint_path` 是否仍 = V7 backbone? (防 backbone 漂移)
- `tools/paired_t_v18_vs_v7.py` 是否真用 `scipy.stats.ttest_rel` (防 codex 用 unpaired t)?
- F0 报告里的 `n=7403` 是否 verify (防 codex 在缺数据时 silently 用更少 slice)?
- A4 训练日志是否含 `lambda_img=0.0800` (verify yaml 真的生效)?

### Q8 — NOT-DO §6 是否过严或过松?

§6 11 条禁项. 请逐条评估:

| 禁项 | 评估 |
|---|---|
| 改 train_first_hop.py | 必要 |
| 改 V7 yaml 本身 | 必要 |
| 改 V18 / V13 / V14 已有 ckpt 或 yaml | 必要 |
| 启 V19 / V18-clean / V18-rank-sweep | Round 17 共识, 必要 |
| 启 V14b / V14c (多 seed) | 是否过严? 若 codex 有空 GPU + 1 个 seed 是免费 sanity 加强 |
| 启 A4-v2 / A4-v3 / 其它 image_aux variant | Round 17 共识, 必要 |
| 启 architecture / data pivot | 必要 |
| F0 跳过 V14/V13 sanity check | 必要 |
| F0 把 V18-cap 与 chain PSNR 混算 | 必要 |
| 在 paper 草稿里宣称 V18 +0.06 dB 显著 | 这是 user 责任不是 codex, codex 不写 paper draft, 这条放 task md 是否走偏?
| 把 V18-cap.last 命名为 "A3" 在新 doc 里 | 是否真有 codex 误用风险? 还是过于细节?
| push narrative / claim 类 markdown 前需 user 复审 | 是否清晰: codex 可以 push 数据 + 报告但不 push 解读? "数据" 和 "解读" 边界是否够明确?

---

## 3. 资料目录

### 3.1 本轮主审

- [CODEX_TASK_ROUND17_F0_A4_20260522.md](./CODEX_TASK_ROUND17_F0_A4_20260522.md) (本轮唯一主审对象)

### 3.2 上游 context (本轮不重审, 仅 verify reference 正确)

- [REVIEW_INTEGRATION_round17_20260522.md](./REVIEW_INTEGRATION_round17_20260522.md)
- [PEER_REVIEW_PROMPT_round17_strategy_post_V13V14_20260521.md](./PEER_REVIEW_PROMPT_round17_strategy_post_V13V14_20260521.md)
- [V7_gronwall_raw.yaml](../0505/local/configs/V7_gronwall_raw.yaml)
- [eval_first_hop_fullval_psnr_chain_mse.py](../0505/operator/scripts/eval_first_hop_fullval_psnr_chain_mse.py)

### 3.3 验证用脚本

- `train_first_hop.py:2230` (B42 行)
- `tools/probe_v18_kl_drift.py` (probe 同型先例)
- `review/0517/CODEX_TASK_STAGE_C_A3_V18_CAPACITY_ONLY_20260518.md` (A3 task md 同型先例)

---

## 4. 输出格式

每位 reviewer 独立产出 markdown:

### 4.1 8 个问题逐条 (APPROVE / MODIFY / REJECT)

### 4.2 push 决策
请明确选 1:

- **READY (push as-is)**: 任何 MODIFY 都是 nice-to-have, codex 拿现稿能正确执行
- **MODIFY-BEFORE-PUSH (列出必改项)**: 有 ≥ 1 项必改, 否则 codex 会失败 / 数据被污染
- **REJECT (推翻重写)**: task md 战略层或方法论层错, 不能 push

### 4.3 必改项 (若 MODIFY)
按 (location, original wording, fix, severity) 列表:

| § | 原文 | 修改 | severity |
|---|---|---|---|
| ... | ... | ... | HIGH/MED/LOW |

### 4.4 新偏差 (B75+)
若发现 task md 引入新 anchor / overreach / cherry-pick / fabrication.

---

## 5. 约束与提醒

- **本轮不重审** Round 17 战略
- **不假设** 你能跑代码 — 但可读 grep / find 结果, 可读已 push 的 JSON / CSV
- **特别关注**:
  - F0 数据源是否真实 (Q1 决定 F0 能否一击跑通)
  - A4 yaml 4 字段是否真的足够 (Q3 决定 A4 是否被污染)
  - A4 from-scratch vs resume 是否清晰 (Q4 决定 A4 是 7 天还是 1 天)
  - stop rule 是否真能挡 codex 顺手再跑 (Q5 决定 scope 是否守得住)
- **优先质疑**:
  - "F0 30 分钟跑完" 是否乐观 (若需重跑 V7 per-slice eval, 实际 6+ 小时 GPU)
  - "A4 唯一仍有 EV 的实验" 是否过度自信 (是否漏关键 variant)
  - NOT-DO 11 条是否真覆盖 codex 历史上的所有失败模式

---

## 6. 本轮元说明

| 轮次 | 范围 | substrate |
|---|---|---|
| 14 | A3 execution prep | A3 yaml + task md |
| 15 | 3-slot launch decision | slot allocation |
| 16 | A3 结果解释 | A3 报告 |
| 17 | post-V13/V14 strategy | V13/V14 eval + 5 战略选项 |
| **17-prep** | **F0 + A4 execution doc** | **codex task md** |

预期 outcome:

- 最佳: 3/3 reviewer 共识 READY 或 MODIFY-BEFORE-PUSH (修 < 5 项), task md 修后立即 push
- 中等: reviewer 发现 F0 数据源缺失 / A4 字段不足, task md 需补一个章节再 push
- 最差: reviewer 发现方法论或战略层 bug, task md REJECT, 回 Round 17 战略重审

---

## 7. 角色提示

你不是在审战略 (Round 17 已签). 你在审一份**即将 push 给远端 GPU 服务器执行的 markdown**, codex 拿到后会**直接照做**.

如果 task md 里有 fabrication / 含糊 / 漏关键 verify, codex 会**沉默地执行错版本**, user 一周后才发现.

所以本轮 review 的 ROI 计算: 30 分钟 review **必须** 比一周 GPU 浪费 + 重做更便宜. 严起来.

如果你认为 task md 已 ready, 直接说 READY; 不要为了 "稳妥" 多列 nice-to-have 改动让 user 决策疲劳.
