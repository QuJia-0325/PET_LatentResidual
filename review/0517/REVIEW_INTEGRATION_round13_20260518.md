# Round 13 Review Integration — Project Strategy Re-evaluation

- date: 2026-05-18
- branch: foc_lite_hop0
- 主审对象: [PEER_REVIEW_PROMPT_round13_project_strategy_20260518.md](./PEER_REVIEW_PROMPT_round13_project_strategy_20260518.md)
- 3 reviewer (含独立 grep verify code + per-slice CSV 实算)
- 触发: user "transport 没大涨, decoder 也没大涨" + Round 12 KL drift 反预期

---

## 0. 一行结论

**3/3 reviewer REJECT claude default §2.2 A** (KL pullback 让 V18 改进 GT manifold). Reviewer3 (B42 HIGH) 通过 grep `train_first_hop.py:2230` **机制层反驳**: `z_kl = main_out["z_pred"] if use_pred_latent else main_batch["z_dst"]`, V18 yaml `use_pred_latent=true` → KL pullback **从未碰过 decode(z_GT) 路径**, 不可能让 V18 在 GT manifold "对齐".

**3/3 reviewer 一致推荐 Stage C = E (扩展版)**: 先 launch V13 + V14 disambig + **新增 V18-r32-capacity-only control** (lambda_kl=0) 才能 disambiguate §2.2 A vs B (LoRA generic capacity).

**Round 13 元发现**: claude 作为 prompt 作者犯了 5 个新偏差 (B39-B43), 其中 B42 是 mechanistic impossibility — 表示项目 substrate review 阶段, claude 已经不再可信, 必须 reviewer 介入.

| 维度 | reviewer1 (self-critique) | reviewer2 (per-slice + 推荐 E) | reviewer3 (B42 grep, 反 §2.2 A) | 共识 |
|---|---|---|---|---|
| 数据 verify | ✓ 数字 100% | ✓ NORMAL paired stats + win rate | ✓ + grep train_first_hop.py:2230 | **3/3 数据无误** ✅ |
| §2.2 A "KL 改进 GT manifold" 解读 | B1+B2 警告 anchor | "测量真但非 causal proof" | **B42 机制反驳** | **3/3 reject claude default** 🔴 |
| §2.2 真实候选 | 加 D (z_pred vs z_GT 解耦) | KL 不证明因果, 但 V18 改进 GT manifold 是 real | B (LoRA generic capacity) 最简释 | **A 不成立, B/D 并列** |
| Stage C verdict | (修 prompt 后再选) | **E (V13+V14)** | **E 扩展 (V13+V14+capacity-only)** | **3/3 E (扩展版)** ✅ |
| 项目 ceiling | Q2 需先定标尺 | "framework 头cell, broader ceiling 未解" | "ceiling 判断未到时机" | **3/3 too-early** |
| 新偏差 | B1-B11 (mostly prompt structure) | (隐含) | **B39-B43** (含 HIGH B42) | **B42 hard** 🔴 |

---

## 1. B42 mechanism reversal (reviewer3 独家, project-altering)

### 1.1 代码事实 (claude 二次 verify)

`train_first_hop.py:2230` 实读:
```python
z_kl = main_out["z_pred"] if bool(kl_cfg.get("use_pred_latent", True)) else main_batch["z_dst"]
```

V18 yaml (Round 5 整合 line 78/170 已 flag 但未撤改) `use_pred_latent: true`.

→ KL pullback 计算: `MSE(decode_lora(z_pred), decode_frozen(z_pred))` (z_pred 路径)
→ **从未** 计算: `MSE(decode_lora(z_GT), decode_frozen(z_GT))` (z_GT 路径)

### 1.2 KL drift probe 测量

`tools/probe_v18_kl_drift.py`: `PSNR(decode_V?(z_GT), x_target)` (z_GT 路径).

### 1.3 §2.2 A 不可能成立的逻辑链

claude default §2.2 A 声称: "KL pullback 让 V18 decoder 对齐 GT manifold, B9 假设被反驳".

但 **z_GT 路径在 V18 训练中是 OOD** (从未被任何 loss 监督). KL pullback **机制上** 不可能影响 V18 在 z_GT 上的表现, 无论改进还是漂离.

→ "KL pullback 让 V18 在 GT manifold 改进 +0.10 dB" **机制上不成立**, 是 anchor 同型 (将 KL pullback 这个 anchor 投射到所有 decoder 行为).

### 1.4 真实最简释 (B 候选)

V18 LoRA 在 z_pred 路径吸 transport residual, LoRA 589K 可训参数同时**通用增强 decoder capacity** (rank=32, 2.5% of decoder weights). 副作用渗到 z_GT 输入 → +0.10 dB 改进**是 capacity 副产品, 不是 KL pullback 设计正面效果**.

含义:
- B9 (KL pullback 让 decoder 漂离 GT manifold) **仍未被验证**, 因 KL 在 z_pred 路径上是否漂离也未测
- V18 framework 的 KL pullback 设计**没起作用**, 只是 LoRA capacity 巧合改进 z_GT
- 任何 "decoder LoRA 有余量但被 transport 吞噬" 推理需重审 (V18 改进 z_GT 不等于 "V18 decoder 有 transport 端余量")

---

## 2. 5 个新偏差 (B39-B43, 全 reviewer3 独家)

### 2.1 B42 (HIGH) — §2.2 A mechanism impossibility (见 §1)

### 2.2 B39 (MOD) — baseline silent swap

reviewer3 catch: §1.2 行 5 用 V6_NOISE.best (36.7437) 算 V6→V18 +0.099, 行末用 V6_NOISE.last (36.7550) 算 +0.088. **同一文档同一指标给出 +0.099 + +0.088 两个数**, 后者 anchor 更悲观 narrative (project ceiling).

修法: 统一 baseline (claude 推荐用 V6_NOISE.last = 36.7550 dB 更保守).

### 2.3 B40 (LOW) — 算术四舍五入

§1.2 "+0.04 + +0.05" 求和 ≈ +0.09 是巧合:
- V6→V7 实际 = +0.0373 (rounded to +0.04)
- V7→V18 实际 = +0.0316 / +0.0616 (best/last)
- 真和: 0.0373 + 0.0616 = 0.0989 (NOT 0.09)

修法: 算术显式不四舍五入.

### 2.4 B41 (LOW) — "transport 吞噬 70%" 只对 V18.best

§2.2 A 文本写 "+0.10 GT → +0.030 transport → 吞噬 70%". 但:
- V18.best: 0.0973 → 0.0302 → 吞噬 ~69%
- V18.last: 0.1639 → 0.0617 → 吞噬 ~62%

修法: 区分 best/last, 不当作 V18 通用事实.

### 2.5 B43 (MOD) — 把 Round 5 flag 的 bug 当 intended design

Round 5 §2.2 line 78/170 已 flag `use_pred_latent=true` 是 V18 buggy config (Round 5 三 reviewer 共识), 但 V18 launch 时未撤. Round 13 prompt §2.2 A 把"+0.10 dB GT 改进"解读为 KL pullback 设计的正面效果 — **等于把已确认的 bug 当 intended design 推理**.

修法: §2.2 A 必须 prefix "假设 V18 KL pullback 是 intended design" 才能讨论, 否则机制不通.

---

## 3. Stage C 路径 (3/3 共识 = E 扩展版)

### 3.1 共识细节

| reviewer | Stage C 推荐 |
|---|---|
| reviewer1 | (修 prompt 后再选, 但论据指向 V13+V14 disambig) |
| reviewer2 | **E** (V13+V14 launch, 之后再 V18-clean/sweep/paper) |
| reviewer3 | **E 扩展** (V13+V14 **加** V18-r32-capacity-only control, 3 slot 并行) |

3/3 一致 reject A/B/C/D/F/G/H, 推 E.

### 3.2 V18-r32-capacity-only control (reviewer3 独家提议)

**目的**: disambiguate §2.2 A (KL 设计) vs B (LoRA generic capacity).

**design**:
- V18 yaml diff: `lambda_kl = 0` (其他全同 V18, rank=32, blocks=[6,7], use_pred_latent 字段 moot 因 KL 关闭)
- 从 V7 best.pt resume, ≤10K step (短训, 看 capacity 副作用是否仍达 +0.10 dB GT 改进)
- 1 slot, ~24-48h GPU
- 评估: `PSNR(decode(z_GT), x_target)` (同 KL drift probe), 比对 V18-r32 vs V18-r32-no-KL

**3 个可能结果**:
1. capacity-only ≈ +0.10 dB → A 不真, B 真 (KL 无功能, 是 LoRA capacity)
2. capacity-only ≈ 0 → A 可能真 (KL 间接有功能? 但仍违反 §1.3 mechanism)
3. capacity-only ∈ (0, 0.10) → 混合 (capacity 部分 + KL bleed-through 部分)

最可能 1 (机制层支持). 但实测决定.

### 3.3 3 slot 并行计划 (Stage C v3)

| slot | task | 时长 | 状态 |
|---|---|---|---|
| **slot 1** | **V18-r32-capacity-only** (lambda_kl=0, 10K step from V7) | ~24-48h | **优先 launch, 最快回** |
| **slot 2** | **V13** (image_aux off, 160K from-scratch) | ~7 天 | 已有 yaml |
| **slot 3** | **V14** (V7 + seed=1337, 160K from-scratch) | ~7 天 | 已有 yaml |

**关键**: slot 1 capacity-only 最先回 (~48h), 可能直接 kill §2.2 A → 重排 V13/V14 后续解读 + Stage C 决策.

---

## 4. 项目 ceiling 判断 (Q2)

### 4.1 3/3 共识: too-early to call ceiling

| reviewer | ceiling 判断 |
|---|---|
| reviewer1 | 标尺未定, Q2 需先 "ceiling vs unexplored" 标准 |
| reviewer2 | "framework headroom near-exhausted, broader ceiling 未解 (待 V13/V14 + arch space)" |
| reviewer3 | "+0.099 dB 在三重 confound 下不是 framework ceiling" |

### 4.2 三重 confound (reviewer3 列)

1. **V8 image_aux 未隔离**: V8 改了 step_weights + image_aux + 其他 config, V7-V8 +0.308 不可单 attribute 给 image_aux. V13 必须 launch.
2. **V18 buggy KL** (B43): use_pred_latent=true 是 Round 5 已 flag bug, V18 outcome 不是 "intended V18 design" 的 outcome.
3. **decoder-transport 划分随配置变**: V18 不一定 "decoder side only" — z_pred 路径包含 transport 输出, V18 LoRA 影响整个 decode(z_pred) pipeline.

→ Paper / pivot / 加 transport intervention **都为时过早**. V13 + V14 + capacity-only 三个 control 完成后才能下定论.

---

## 5. 推荐修订 prompt 后再 dispatch? 

reviewer1 推 "修 5 处后 dispatch v2". reviewer2/3 直接独立完成 review 无视 prompt 缺陷.

**claude 推荐**: **不再修 prompt v2** — 3 reviewer 已独立完成 review (含 B42 致命发现), 修 prompt 再发只会重复 review. 直接进 Stage C v3 launch (E 扩展版).

但 **R12 整合 §5 B36 standing rule 升级**: 起草 prompt 含 substrate (PSNR / 代码) 必须先 grep verify, 不允许凭 markdown 转述. R13 prompt 4 处违反 (B39-B43), claude 起草水平在 substrate 上不再可靠.

---

## 6. user 决策点 (3 个)

| # | 决策 | 选项 | claude 推荐 |
|---|---|---|---|
| 1 | Stage C v3 launch 顺序 | A: 3 slot 同时 (V18-cap + V13 + V14) <br>B: slot 1 先 launch V18-cap (48h), 等结果再 V13/V14 <br>C: slot 1 V13 (7 天), slot 1 完成后 V18-cap | **A** (3 slot 并行, 48h 后 V18-cap 回 + V13/V14 在跑, 信息密度最大) |
| 2 | §2.2 A 是否还要 "decoder GT 改进" 当 paper 卖点? | A: 不要 (B42 反驳后, 改进无机制保证) <br>B: 当 "interesting observation, 机制 unknown" 写 paper supplementary <br>C: 等 capacity-only 控制后再决定 | **C** (48h 等 substrate, 不预设 paper narrative) |
| 3 | Round 14+ cadence | A: V13/V14/cap 跑完后起 Round 14 战略 review <br>B: 减少 reviewer 介入, user+claude 直接 <br>C: 把 reviewer-heavy 政策 lock in 直到 paper draft | **A** (战略转折点起 reviewer, 战术细节 user+claude; Round 13 已证 reviewer 必须 catch B42 这种 mechanism bug) |

回复格式: `1=A 2=C 3=A` 或 `按推荐`.

---

## 7. Round 13 元教训

### 7.1 substrate review 阶段 claude 不可信

Round 1-11 markdown 阶段, claude 自纠勉强可用 (虽然 Round 8-11 多次复发). Round 12 V18 substrate 阶段, claude 第一次错 (D20 baseline) + 第二次错 (best vs last label) 都 reviewer 才 catch. Round 13 战略阶段, claude **B42 机制错误** 是新高: 不是数字错, 不是标签错, 是把不可能的机制当 default 解读.

**含义**: substrate review 阶段, claude 不再可独立解读, 必须强制 reviewer 介入. R13 推 R14+ cadence option A.

### 7.2 prompt 自己警告 reviewer 别 anchor, 自己 anchor 5 处

R13 §6 写 "优先质疑 claude default 解读 §2.2 A 是否 anchor 到'decoder LoRA 是有效设计'". 实际 prompt **本身** anchor 5 处 (B39 baseline swap, B40 算术, B41 70% 通用化, B42 mechanism impossibility, B43 已知 bug 当 design). claude 的 anchor 防御机制对自身无效, 只对 reviewer 起作用.

**含义**: 防御 mechanism 必须 outsource 给 reviewer / mechanical script / per-slice CSV 实算, 不能依赖 claude self-discipline.

### 7.3 standing rule #N 升级 (R13 新提案)

**B36** (R10 立, R12 升级): prompt 含 substrate 必须先 grep verify → **R13 升级**: substrate 上 claude 不再独立解读, 必须 reviewer 介入.

**B44 (新)**: V18 family "use_pred_latent=true" 是已知 buggy config (Round 5 flag), 任何基于 V18 实测数据的解读必须 prefix "in V18 buggy config" qualifier, 不当作 intended design outcome.

---

## 8. 立即下一步

待 user 回复决策 1/2/3 → claude:
1. (若 1=A) 起 Stage C v3 launch task md (3 slot 并行 V18-cap + V13 + V14), push gitee
2. (若 2=C) paper narrative 暂缓, 等 48h capacity-only 结果
3. (若 3=A) Round 14 = capacity-only 出来后起战略 review

总 ~30 min 起 task md + push, 然后等 48h-7 天 substrate 回.
