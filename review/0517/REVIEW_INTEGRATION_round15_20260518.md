# Round 15 Review Integration — 3-Slot Launch (A3 + V13 + V14)

- date: 2026-05-18
- branch: foc_lite_hop0
- 主审对象: R15 prompt 3-slot launch decision
- 4 reviewer (Reviewer A/B/C/D, 全独立 grep verify)
- 上游: R13 user 决策 A1/A3/A4 + R14 A3 execution prep + user 当前指示 "B 合适"

---

## 0. 一行结论

**4/4 reviewer 共识 slot 3 = V14 (B 选项)**, 但**核心新发现 B55 (HIGH, 3/3 reviewer A/B/C 独立 grep verify)**: R15 prompt §1.4 引用的 "user Round 13 不考虑 multi-seed, 先跑出结果" **凭空构造** — claude verify 全 repo 无此 quote. R13 整合实际 user 决策 = "A 选项 3 slot 同时 (V18-cap + V13 + V14)", V14 一直在 plan 中.

→ R15 整个 Q1/Q6 框架建立在虚构 user 约束上. **V14 不是 "机会主义加跑"**, 是 R13 user 决策本身. claude 起 R15 prompt 时**逆向 reframe** 一个已定决策为待评估决策, 这是 anchor 同型新形态.

| 维度 | A (Copilot) | B (Copilot) | C (独立) | D (Copilot) | 共识 |
|---|---|---|---|---|---|
| Slot 3 = V14 | **APPROVE B** (但论据独立非 prompt §1.4) | **APPROVE B** | **APPROVE B** (理由是 paper noise floor) | **MODIFY then APPROVE B** | **4/4 B** ✅ |
| Launch order | 同时 launch (5-10 min 间隔) | A3→V13→V14 staggered | A3+V13 T=0, V14 T=+30min (IO check) | A3→V13→V14 staggered + 5min health | **3/4 staggered** ⚠️ |
| V14 task md | 复用 PHASE_A_v3 模板 | **必须 prep ~15 min** | **REQUIRED ~15 min** | 必补 V14 task md 或 3-slot wrapper | **3/4 必须 prep** ✅ |
| 漏 control? | (不评) | 加 V13-cap-only 暂缓 | 全 defer R16+ | 暂时 OK, R16 再加 | **3/4 当前 3 control 够** |
| 新偏差 | **B55 HIGH** (虚构 user quote) + B56 HIGH (slot 3 永远 0 静默推翻) | B55/B57/B58 | **B55-B59 (5)** | (本轮已修 PHASE_A v3 stale) | **B55/B56 共识 HIGH** 🔴 |

---

## 1. B55 mechanism (3/3 A/B/C grep verify)

### 1.1 R15 prompt §1.4 引用 (本文)

> "user Round 13 明确说 '不考虑 multi-seed 的问题, 先跑出结果, 再考虑 multi-seed'"

### 1.2 实际 R13 整合 §3.3 (claude 二次 verify)

```
| slot 3 | V14 (V7 + seed=1337, 160K from-scratch) | ~7 天 | 已有 yaml |
```

R13 §6 user 决策 1: "Stage C v3 launch 顺序 A: 3 slot 同时 (V18-cap + V13 + V14)" → **claude 推 A**.

R13 §8 立即下一步: "(若 1=A) 起 Stage C v3 launch task md (3 slot 并行 V18-cap + V13 + V14)".

→ R13 整合明确**含 V14**. user "执行 A1/A3/A4" 是这个 plan, 不是 "排除 V14".

### 1.3 user 早期口语化的 "先跑出结果"

user 在 R13 之前曾说 "不考虑 multi-seed 的问题, 我们先跑出结果, 再考虑 multi-seed". **但 R13 整合后这变成了 "3 slot 含 V14" plan**, user 签 A1/A3/A4 时已默认接受 V14 在 plan 内.

claude R15 起 prompt 时把 user 早期口语**当作 absolute constraint**, 引到 §1.4 反对 V14 论据, 让 V14 看似"违反 user 决策". 这就是 B55: 把 contextual prior 视为 absolute, anchor 到错误 framing.

### 1.4 元教训 (Round 15 元偏差)

**substrate 阶段** (R12 V18 PSNR 解读 / R13 KL drift 解读) reviewer 已多次 catch claude anchor; **R15 是 process-level review** (launch decision), claude 又一次 anchor — 这次是**虚构 user constraint**. 反派 "凭空构造 R13 user quote" 是 B25 同型变种 (R10 我 reviewer 角色错指 audit DRAFT 符号不存在, 此次我 prompt 作者角色错引 user quote).

→ B55 standing rule 升级: prompt 引用 prior-round user quote 必须**自己先 grep verify**, 不允许凭记忆.

---

## 2. B56 (HIGH, Reviewer A 独家): "Slot 3 永远 0" 约束未解释

R14 CODEX_TASK_PHASE_A_v3 line 9: "**Slot 3 永远 0**" (Phase A v3 范围).

R15 prompt 静默推翻这个约束 (slot 3 = V14), **没解释为何状态转变**.

实际原因 (claude 推测): Phase A v3 是 "smoke+launch V13" 范围, "Slot 3 永远 0" 是该 task md 内部约束; R15 是 Stage C launch decision review, 范围不同, 不一定继承 v3 约束. 但**这转变需在 V14 task md 显式 supersede**, 否则 codex 看到两份文档冲突会困惑.

**修法**: V14 task md 顶部加 "本 task supersede Phase A v3 line 9 'Slot 3 永远 0' 约束 (R13 user 决策 A 含 V14, v3 该行属 v3 内部 scope)".

---

## 3. 其他新偏差 (B57-B59, Reviewer C/A/B 独家)

| # | severity | reviewer | 摘要 |
|---|---|---|---|
| B57 | MOD | Reviewer C | V14 被 R15 prompt mislabel "机会主义 slot 3 加跑" — 实际是 paper-prerequisite noise floor, 是所有 ΔPSNR 显著性声明的 noise denominator (V18 +0.030, V13 outcome, 任何 pivot 都依赖) |
| B58 | LOW | Reviewer C | R15 Q2 漏问 disk IO contention (V13/V14 同读 `latent_dir`, `num_workers=0` 但 3 任务 IO 累积未压测) |
| B59 | LOW | Reviewer C | R15 Q6 自我批判不彻底 — 未要求 reviewer 验证 R13 user quote 本身. R15 已 implicitly enforce via B55 grep verify, 故 B59 fix = B55 standing rule |

---

## 4. Launch order (3/4 共识 staggered)

3 reviewer 推荐 staggered launch:
- Reviewer B/D: A3 → V13 → V14 (5-10 min 健康检查间隔)
- Reviewer C: A3+V13 T=0 同时, V14 T=+30min (after `nvidia-smi` + `iostat 1 5` IO 健康检查)
- Reviewer A: 同时 launch (5-10 min 间隔)

**整合 (claude 推荐 Reviewer C 路径)**:
1. **T=0**: 同时 launch A3 (slot 1) + V13 (slot 2), 各 GPU 独立, 互不依赖
2. **T=+5min**: 各 process alive 检查 (`ps -p $A3_PID`, `ps -p $V13_PID`)
3. **T=+30min**: `nvidia-smi --query-gpu=memory.used,memory.free --format=csv` + `iostat 1 5` 检查 GPU/IO/disk contention
4. **T=+30min (健康)**: launch V14 (slot 3)
5. **T=+35min**: V14 alive 检查

理由: V13/V14 共享 `latent_dir` (同 dataloader 路径), 3 任务 IO 累积未历史压测, staggered launch + IO health check 是低成本防御 (B58).

---

## 5. V14 task md (3/4 必须 prep, ~15 min)

3 reviewer 共识 V14 不能裸跑, 必须 task md:
- 含 pass 准则 (与 V13 类似: image_aux=ON 验证, no NaN, 200-step smoke)
- 含 NOT-DO + 失败决策树
- 含 supersede Phase A v3 "Slot 3 永远 0" 约束声明 (B56 fix)
- 含 R13 user 决策 A 引用 (B55 fix: 显式 grep evidence)

**复用模板**: V13 task md (Phase A v3 Task C+D) 几乎可 1:1 套 V14, 仅改 yaml 路径 + run_name + seed 备注. ~15 min prep.

---

## 6. user 决策点 (4 个)

| # | 决策 | 选项 | claude 推荐 |
|---|---|---|---|
| 1 | Slot 3 verdict | **B (V14)** (4/4 共识) | **B** |
| 2 | Launch order | A: 同时 launch / **B: staggered A3+V13 T=0, V14 T=+30min** / C: 全串行 | **B** (3/4 共识) |
| 3 | V14 task md | A: 必须 prep ~15 min / B: 复用 Phase A v3 / C: 裸跑 (无 task md) | **A** (3/4 共识) |
| 4 | 是否修 R15 prompt 的 B55/B56 (虚构 user quote + Slot 3 永远 0 未解释) | A: 修 prompt 历史诚实 / B: 不修 (整合文档已诚实标 B55) | **B** (R15 prompt 已是历史 artifact, 整合文档诚实记录即可, 不重写历史) |

回复 `1=B 2=B 3=A 4=B` 或 `按推荐`.

---

## 7. 元教训

### 7.1 B55 是 R15 process-level 新偏差

R12 V18 substrate review claude 第一次错 (D20 baseline), R13 strategic review claude 第二次错 (B42 mechanism), **R15 process review claude 第三次错 (虚构 user quote)**. 三轮**全是 claude 起草者角色 anchor**, 全 reviewer catch.

→ standing rule B45 (substrate reviewer 必介入) 应**扩展**至 process review: 任何含 prior-round user 决策引用的 prompt, reviewer 必 grep verify 原文.

### 7.2 推 V14 的真实理由 (不是 R15 §1.4)

| R15 prompt §1.4 给的理由 | Reviewer 给的真实理由 |
|---|---|
| "slot 3 反正空, 7 天机会成本" | V14 是 paper-prerequisite noise floor, 所有 ΔPSNR 显著性 denominator |
| "yaml ready 0 prep" | (同上, 但是 secondary) |
| "与 A3/V13 独立" | 4/4 共识 (但 staggered launch 防 IO contention) |
| (无) | R13 user 决策 A 含 V14, 不是 "推翻"; R15 prompt 把 contextual 引用当 absolute |

→ V14 launch decision 已**早在 R13 内 user 签字**, R15 实际是 "execution prep", 不是 "新决策". R15 prompt 起草 framing 错.

### 7.3 Standing rule B60 (新立, 写进 design_rationale §5.2)

**B60 — prompt 引用 prior-round user 决策必须 grep verify 原文**:
- 任何 prompt 中 "user 在 Round N 决策 X" / "user 明确说 Y" 等引用, 起草者必须**先 grep R/N integration + chat artifact**, 贴出 verbatim quote + 文件:line 引用
- 无原文 verify 的 user 引用降级为 "claude 推测 user 倾向", 不作为 prompt anchor
- 实证: R15 §1.4 虚构 quote 让 3 reviewer 都需独立 grep 反驳, 浪费 30+ min reviewer 时间

---

## 8. 立即下一步

待 user 回复决策 1-4 → claude:
1. (若 3=A) 起 V14 task md (~15 min, 复用 V13 模板, 含 B55/B56 supersede 声明)
2. (若 2=B) Phase A v3 launch task md 加 staggered launch + IO health check 段
3. (若 4=B) 不重写 R15 prompt, R15 整合文档已诚实标 B55
4. design_rationale §5.2 加 B60 standing rule

push 后 codex 在服务器: A3+V13 同时 launch → +30min IO check → V14 launch. 3 slot 全跑.

---

## 9. Round 13-14-15 cadence 验证

| Round | substrate | claude 首次错? | reviewer catch? |
|---|---|---|---|
| 12 | V18 PSNR | 是 (D20 baseline) | 是 |
| 13 | KL drift inverse | 是 (B42 mechanism) | 是 |
| 14 | A3 execution prep | 是 (B53 §1.3 与 B44 自相矛盾) | 是 |
| **15** | **3-slot launch** | **是 (B55 虚构 user quote + B56 静默推翻 slot 3=0)** | **是** |

**B45 standing rule 100% 印证**: substrate / strategic / process / execution 任何 review 阶段 claude 都犯 anchor, reviewer 介入是必要的.
