# Round 15 Peer Review — Reviewer A (independent)

- date: 2026-05-18
- reviewer: **Reviewer A** (GitHub Copilot, Claude Opus 4.7 xhigh, independent draft, did not see Reviewer B/C drafts)
- substrate verified via: grep_search / file_search / read_file on 7 artifacts (paths cited inline)
- 主审对象: Round 15 prompt §1.4 (V14 ROI 论据) + §1.2 (3-slot 候选实验) + §2 (6 questions)

---

## 0. One-line verdict

**slot 3 decision = B (V14 launch)**, **launch order = 全部同时 launch**, **V14 task md = 必须 prep (复用 PHASE_A_v3 模板, ~10 min)**, **3 个 HIGH 新偏差 B55/B56/B57 必修**.

但**不是因为** prompt §1.4 给的论据 — 那些论据有 1 处事实错误 (B55). 真正推 B 的理由是 V13/V14 ROI 与 V18-cap outcome **解耦** + paper-necessary noise band, 见 Q1 详解.

---

## 1. 关键事实核对 (independent grep)

| 声明 | 来源 | 验证结果 |
|---|---|---|
| V14 yaml 存在 | `file_search **/V14_v7_seed1337.yaml` | **✓** [V14_v7_seed1337.yaml](review/0516/V14_true_d_pure/V14_v7_seed1337.yaml) (1 result) |
| V13 yaml 存在 | `file_search **/V13_true_image_aux_off.yaml` | **✓** [V13_true_image_aux_off.yaml](review/0516/V13_true_image_aux_ablation/V13_true_image_aux_off.yaml) (1 result) |
| V14 无独立 task md | `file_search **/V14_true_d_pure/*.md` | **✓** 0 results |
| V18 已完成 | Round 14 §0 (3/3 ready) + commit 24de2ab pushed | **✓** |
| Round 13 user 决策 = "不跑 multi-seed" | grep [REVIEW_INTEGRATION_round13 line 177](review/0517/REVIEW_INTEGRATION_round13_20260518.md#L177) | **✗ FALSE** — 决议表 row 1 user 决策 = **A (3 slot 并行 V18-cap + V13 + V14)**, 并在 row 137 明确分配 "slot 3 = V14". prompt §1.4 反对论据 #1 与原决议直接矛盾, 见 B55. |
| PHASE_A_v3 task md "Slot 3 永远 0" | grep [CODEX_TASK_PHASE_A_v3 line 9](review/0517/CODEX_TASK_PHASE_A_v3_20260518.md#L9) | **✓** verbatim. 但 Round 15 prompt §1.1 默默改为 "slot 3 空" — 状态转变 (V18 完成 → free) 未在 prompt 解释, 见 B56. |
| V14 yaml = V7 + seed 1337 ONE 字段 diff | grep V14_v7_seed1337.yaml | **✓** seed: 42→1337, 其他全同 V7 |

---

## 2. 6 个问题逐条回答

### Q1 — V14 加跑 (slot 3) 真的 ROI 高吗?

**verdict: APPROVE V14 加跑, 但 REJECT prompt §1.4 给的反对论据 #1 (含 B55 事实错位)**

#### 1.A prompt §1.4 反对论据 #1 是事实错误

prompt 写 "user Round 13 明确说 '不跑 multi-seed, 先跑出结果, 再考虑 multi-seed'" — **未引证, 未 grep verify, 且与 Round 13 原决议表 row 1 (user 决策 A = 3 slot 并行 含 V14) 直接矛盾**. 我 grep 全部 Round 13 / Round 14 整合文档, 找不到任何 "user 不跑 multi-seed" 的 verbatim 决策原文.

若 user 在 Round 14 之后 / Round 15 prompt 起草之前曾口头说过, prompt 必须**引证会话片段或时间戳**, 否则推翻 R13 user 决策 A = standing rule #5 ("R7/R8/R9/R10/R12/R13/R14 user 决策不可推翻") 的潜在违反.

→ §6 standing constraint #5 在 R15 prompt 自己提了 "R13 '不跑 multi-seed' 是 contextual 还是 absolute, 待 reviewer 评" — 这是个**伪问题**, 因 R13 原决策本身就是 launch V14. prompt 实际在用 reviewer-Q 形式合理化推翻 R13 决策, 应停止.

#### 1.B V14 的真 ROI: paper-necessary noise band, 与 V18 framework 是否 retire 无关

paper 报告任何 ΔPSNR < 0.1 dB 都必须有 d_pure noise band 参考. V18 vs V7 best-vs-best 是 +0.030 dB — 若 V7-V14 noise = ±0.05 dB, 这个 ΔPSNR 在 paper 中被 reviewer 一句话 reject; 若 noise = ±0.01 dB, +0.030 是显著的. **无 V14, V18 paper claim 不可写**.

而且 V14 数据**与 V18 framework 是否 retire 解耦**:
- 若 V18 retire, paper 转 transport-only (V7 / V13 framework), 仍需 V7-V14 noise band 报告 V7 main result
- 若 V18 保留, paper 含 V18, 仍需 V7-V14 noise band 加 V18-V7 比较

所以 V14 在两个 paper narrative 都需要. prompt 说 "V14 数据可能 stale" 是错的 — d_pure noise floor 不会 stale.

#### 1.C "user 不跑 multi-seed" 即使为真, V14 ≠ multi-seed sweep

V14 = V7 + seed=1337 单点测量 (不是 sweep). multi-seed sweep 通常指 ≥3 seed; V14 是 N=2 配对 t-test 的最小要件. 标 "multi-seed" 是 framing 错误 (B55b).

#### 综合
- prompt §1.4 支持论据 (V14 paper-necessary / yaml ready / 解耦) 全成立
- prompt §1.4 反对论据 #1 (user 决策) 事实错误
- prompt §1.4 反对论据 #2 (slot 3 灵活 buffer) 弱: A3 48h 后回, V14 与 V13 都 7 天, V13 已锁 slot 2, 等 A3 outcome 才决 slot 3 = 浪费 48h GPU
- prompt §1.4 反对论据 #3 (V14 数据 stale) 错: 见 1.B

→ **B (slot 3 = V14)** 强推. 

---

### Q2 — 3 slot 真无并发冲突?

**verdict: MODIFY** — 大体可行, 但需补 2 个 spot-check.

#### 2.A GPU 内存
- V18-cap: V7 best.pt + LoRA rank=32, ≈ 10-12 GB (V18 训练实测)
- V13: from-scratch full pipeline, ≈ 18-22 GB
- V14: from-scratch full pipeline, ≈ 18-22 GB (同 V13 backbone)

总 ≈ 46-56 GB. 服务器 GPU 单卡通常 24/40/48/80 GB. **若是单卡, 3 任务并行不可行**. 若多卡 (e.g. A100×4 或 V100×4), 每卡 1 任务 OK.

→ **必须**在 launch 前确认服务器 GPU 拓扑 (`nvidia-smi --query-gpu=index,memory.total --format=csv`). prompt §1.1 表说 "全 GPU × 3 slot", 暗示多卡 — 但未引证. codex 在 A0 spot-check 必须 verify.

#### 2.B Dataloader 冲突
- V13/V14 同读 `/data_2/qujiaxiang/lowdose_pet_ct/latents_224` (同 latent_dir)
- V13/V14 同 dataloader 多进程 (num_workers=0 in V14, 但 V13 yaml 需 verify)
- 3 个 train process 并行读同一磁盘 → I/O 瓶颈风险 (尤其 HDD; NVMe 可忽略)
- 历史: V18 训练时 server 是否曾跑 ≥3 任务? prompt 自己问但没答. **codex 应 query 服务器历史 jobs**.

#### 2.C Watchdog / metrics 互不污染
output_dir 全独立 (V18_capacity_only / V13_true_image_aux_ablation / V14_true_d_pure), metrics jsonl 各写各, OK. 但 launch log 若用 review/0517/*.log 同目录, 三 log 文件名带 TS 应足够隔离.

#### 修复
- A0 spot-check 加: `nvidia-smi` GPU 数 + 单卡内存
- A0 spot-check 加: latents_224 所在盘 `df -hT` (HDD/SSD)
- 历史 query: server `ls /tmp/*.pid 2>/dev/null | wc -l` 或 `pgrep -f train_first_hop` 看当前 process 数

---

### Q3 — Launch order 重要吗?

**verdict: APPROVE 全部同时 launch (B)**, REJECT "A3 先, 等 48h" (因 V13/V14 ROI 与 A3 outcome 解耦, 见 Q1.B).

但加一个小修: **A3 → V13 → V14 间隔 5-10 min launch** (而非同一秒), 这样 launch failure 隔离: 若第 1 个 OOM, 不会带倒后 2 个. 这是低成本好习惯, 不影响并行总时长.

---

### Q4 — V14 缺 task md, 是否阻塞?

**verdict: APPROVE C (复用 PHASE_A_v3 task md 模板)**, ~10 min 工作量.

理由:
- V14 是 V7 + seed=1337 **单字段 diff**, 训练流程与 V13 完全相同 (都是 from-scratch 160K, 同 yaml schedule)
- PHASE_A_v3 Task C+D 模板的 §A0 / §A3 pass 准则 / §commit 流程 / §NOT-DO 可 ~3 行 patch 复用 (改 yaml 路径 + run_name + smoke output_dir)
- 不 prep task md 会触发: B14 (yaml cross-check 缺) + B23 (premature claim) + Round 14 B49 (probe flag 缺) 同型 history. claude 直接 `python --config V14 yaml` 跑 = 缺 pass 准则 / 缺 grep verify / 缺 commit message 规范

修法: claude 在 R15 verdict 后立即起 `CODEX_TASK_V14_LAUNCH_20260518.md` (~150 行, mostly copy from PHASE_A_v3 Task C+D), 与 A3 + V13 同 commit 内 push.

---

### Q5 — 是否漏关键 control 实验?

**verdict: MODIFY** — prompt 列的 4 个候选我评 + 1 个新提议.

| 候选 | Reviewer A 评价 |
|---|---|
| V13-capacity-only (V13 + LoRA r32, no KL) | **不立即跑**. 在 V13 outcome 出来 (~7天) 之前不知道 V13 base 是否本身就 +0.30 dB; 若是, V13-cap-only 也加 LoRA 是 over-engineering 二维 sweep. 等 V13 结果. |
| V8 重训 with V7 train config | **不立即跑**. V8 已撤销 (PLANF_FINAL §0.6 V8 是 regression), 重训为分离 train config 是 nice-to-have, 不影响主 paper. backlog. |
| V18-clean (use_pred_latent=false) 复活 | **REJECT**. Round 6/7 user 永久撤销 = standing rule #5 (R7 不可推翻). KL drift inverse 是 V18-cap (lambda_kl=0) disambig, 不是 V18-clean (use_pred_latent=false) 复活的合法触发. |
| V19 decoder LoRA blocks=[0,1] vs [6,7] | **不立即跑**. 跑 V18-cap outcome 后再考虑. 若 outcome 1 (capacity 真的有用), V19 才有意义; 若 outcome 2 (capacity 无功能), V19 也无意义. |
| **[NEW Reviewer A 提议] paper-blind eval rerun** | 现 PSNR 评估全用 `calc_psnr_clip3` (clip [0,3]). paper 通常用 PSNR (no clip) + SSIM. 在 V13/V14 跑完之前, 复用现有 ckpt 做 SSIM eval (10 min CPU), 加进 paper 工具箱. ROI 高, 0 GPU 成本. |

---

### Q6 — Round 15 新偏差 (B55+)

#### B55 [HIGH] — §1.4 反对 V14 论据 #1 "user 不跑 multi-seed" 事实错位

prompt §1.4 写 "user Round 13 明确说 '不考虑 multi-seed, 先跑出结果, 再考虑 multi-seed' — 严格守这个决策". 

**事实**: 我 grep `multi-seed|不跑|不考虑|seed 1337|V14` 全部 R13 整合文档, **找不到**任何此类 user 决策原文. 相反, [REVIEW_INTEGRATION_round13 line 137](review/0517/REVIEW_INTEGRATION_round13_20260518.md#L137) 显示 slot 3 = V14 (V7 + seed=1337, 160K from-scratch), [line 177](review/0517/REVIEW_INTEGRATION_round13_20260518.md#L177) user 决策 row 1 = A (3 slot 并行 V18-cap + V13 + V14).

**形态**: 凭空构造 user 决策来制造 "反对 V14 论据" 平衡感, 让 prompt 看起来"两边都听" — 但实际是 anchor 到 "或许 slot 3 留空更稳" 的 default. **同型 R12 B23 (last-vs-best 数字配 best-vs-best 标签)**: 也是凭空造一句话来支撑预设结论.

**修法**: 删 §1.4 反对论据 #1 全段, 或改写为 "无引证可考的 user 决策反对 V14; 真正风险是 §6 sunk cost framing 自身". §6 standing constraint #5 删 "R13 '不跑 multi-seed' 是 contextual 还是 absolute, 待 reviewer 评" — 这是 prompt 自己制造的伪问题.

#### B56 [HIGH] — PHASE_A_v3 "Slot 3 永远 0" 约束的状态转变未在 R15 prompt 解释

[CODEX_TASK_PHASE_A_v3 line 9](review/0517/CODEX_TASK_PHASE_A_v3_20260518.md#L9) verbatim: "硬约束: 内存 max 3 任务. **Slot 1 = V18 在跑 (不可动). 本 task 最多新增 slot 2 = V13. Slot 3 永远 0.**"

R15 prompt §1.1 表显示 3 slot 全空 — 物理事实正确 (V18 已完成), 但 prompt **没说明** "Slot 3 永远 0" 这条约束的状态转变: 它是因 V18 完成自然 obsoleted, 还是被 R15 决策推翻? 

**形态**: 静默撤约束. reviewer 看到 R15 prompt 推 "3 slot launch" 时, 必须自己反查 PHASE_A_v3 才能知道这条约束存在过. 同型 R13 B23 (在 prompt 自己声称自纠 B23 的同一段, 又造 B23).

**修法**: R15 prompt §0 加一段 "PHASE_A_v3 §0 '*Slot 3 永远 0*' 约束因 V18 完成而 obsoleted (slot 1 自然 free); 本 task 选项 B 重新激活 slot 3 不构成约束推翻."

#### B57 [MOD] — Q4 选项框定不准确

Q4 三选项 (A: prep task md / B: 不 prep / C: 复用模板) 暗示 "prep ~10 min" 是高成本 — 实际**复用 PHASE_A_v3 Task C+D 模板只需 ~3 行 patch + 10 min 检查** (改 yaml 路径 + run_name + smoke output_dir 三处). prompt 没说复用成本几乎为 0, 让 reviewer 倾向 "不 prep" (选项 B). 

**形态**: 隐藏低成本选项的真实成本, 让"省事"选项看起来更合理. 

**修法**: Q4 选项 C 改 "复用 PHASE_A_v3 Task C+D 模板 (~3 行 patch + 10 min)" 明示成本, 并标为推荐.

#### B58 [LOW] — §1.4 "V14 数据可能 stale (V18 framework retire)" 错

V14 = V7-V14 d_pure noise floor, 与 V18 framework 完全解耦 (V14 不含 LoRA / KL pullback). 无论 V18 保留还是 retire, paper 报告 V7 main result 都需要 noise band. prompt 把 V14 ROI 错与 V18 framework 命运绑定. 见 Q1.B.

**修法**: §1.4 反对论据 #3 删或改 "V14 noise band 与 V18 framework 解耦; 无论 paper 含/不含 V18, 都需要它".

#### B59 [LOW] — §1.2 表 V14 "无 task md" 框为 gap

V14 是 V7 + seed=1337 单字段 diff (V14_v7_seed1337.yaml line 5: `# seed: 42 → 1337`). 这种 trivial diff 复用 PHASE_A_v3 Task C+D 模板成本极低. 标 "task md gap" 让 V14 看起来准备不充分, 实际是 setup 问题不是 ROI 问题. 与 B57 同型.

---

## 3. slot 3 决策 verdict

**B (slot 3 = V14)** — 强推, 但不是因 prompt §1.4 论据.

真正推 B 的理由 (Reviewer A 独立):
1. V14 paper-necessary (Q1.B): 无 d_pure noise band → V18 +0.030/+0.062 dB paper claim 不可写
2. V14 ROI 与 V18 framework 命运解耦 (Q1.B): paper 含/不含 V18, V14 都需要
3. V14 yaml ready + 复用 task md ~10 min: prep 成本几乎为 0
4. V14/V13 与 A3 ROI 解耦 (Q3): A3 outcome 不影响 V13/V14 是否值得跑, 同时 launch 信息密度最大
5. R13 user 决策 A 已 endorse 3 slot 并行 (Q1.A): 不需重新启 standing rule 评估

---

## 4. Launch order verdict

**全部同时 launch (sequential 5-10 min 间隔, 不是同一秒)**

理由: V13/V14 与 A3 解耦 → 等 A3 48h 后启 V13/V14 = 浪费 48h × 2 slot GPU; 5-10 min 间隔启动 = launch failure 隔离, 0 总时长 cost.

---

## 5. V14 task md verdict

**C (复用 PHASE_A_v3 Task C+D 模板, ~3 行 patch + 10 min 检查)** — 推荐, 不是高成本.

claude 在 R15 verdict 后立即起 `CODEX_TASK_V14_LAUNCH_20260518.md` (~150 行 mostly copy), 与 A3 + V13 同 commit push.

---

## 6. 新偏差汇总

| # | severity | 形态 | 修法 |
|---|---|---|---|
| B55 | **HIGH** | §1.4 反对论据 #1 凭空构造 user 决策, 与 R13 决议 A 直接矛盾 | 删全段; §6 standing constraint #5 删 R13 contextual/absolute 伪问题 |
| B56 | **HIGH** | PHASE_A_v3 "Slot 3 永远 0" 约束状态转变未解释 | §0 加一段 obsoleted 说明 |
| B57 | MOD | Q4 选项 C 复用成本被框为 ~10 min 高成本 | 选项 C 改明示 "~3 行 patch + 10 min", 标推荐 |
| B58 | LOW | §1.4 "V14 stale" 与 V18 framework 错绑定 | 改写解耦说明 |
| B59 | LOW | §1.2 "V14 无 task md" 框为 gap | 改 "可复用 PHASE_A_v3 模板" |

**B55/B56 是 ship-blocker** (须修后再 dispatch 给 Reviewer B/C, 否则他们也被同 anchor 误导). B57-B59 可在 R16 整合时修.

---

## 7. 元评论 (Reviewer A 立场)

R15 prompt **整体方向正确**, 但 §1.4 + Q1 的论据合理化框架是 R13 B23 / R14 B53 同型 "anchor + 伪平衡" 偏差. claude 已知答案 (推 B = launch V14), 倒推论据时凭空造了 user 决策来制造 "反对方"; 这种 framing 让 reviewer 落入二选一陷阱 (信 prompt 反对论据 → reject B, 不信 → 在 prompt 既定二元里选 B).

R15 prompt 的真正风险**不是** B 选项错, 是用错论据推 B — 一旦 user 在 R16+ 反问 "你 R15 怎么得出 V14 paper-necessary", 答 "因为 §1.4" 会发现 §1.4 没说 paper-necessary, 只说 "yaml ready / 解耦 / 数据可能 stale". 真正 paper-necessary 论据 (Q1.B noise band) 是 Reviewer A 加的.

**建议**: R15 标准从 "prompt 给的论据" 退一步到 "B 选项是否对", 然后让 reviewer 重新构造 B 论据. 这是 R12/R13 已立 standing rule B45 (substrate review reviewer 必须介入) 的真实意义.

---

## 8. 输出元

- reviewer: **Reviewer A**
- methodology: 全 grep verify (7 个 artifact 5 个 grep 1 个 diff 1 个 file_search)
- 未与 Reviewer B/C 交流
- 撰写时长: 单轮, no iteration
