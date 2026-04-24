# Round 5 — Final Consensus & Modification Roadmap

Date: 2026-04-24  
Mode: NIGHTMARE, Round 5 / 5 (final)  
Synthesizer: Panel Chair (integrating A · B · C · D · E + BD-alliance + Tribunal rulings)

---

## 5-Round 对抗结果 TL;DR

经过 5 轮独立 Opus 4.5 subagent 对抗评审（5 角色初评 → 面板合议 + BD 结盟 → Proposer 反驳 + Tribunal 独立裁决 → 专项深挖 4 大分歧 → 最终合议）：

**最终融合分数：3.05 / 10**（当前状态；若执行本报告 §3 的 48h 行动 + 3 个承诺实验，可升至 3.7–4.0 / 10 workshop-viable 区间；若 σ_seed 实测 < 0.04 dB 可达 4.5+ / 10）

**核心裁决**：
- 方法贡献（hop0 pixel forcing + residual head + img aux）：**统计学意义上不可识别**；效应量 0.083 dB 远低于可能的 σ_seed 下限
- 故事（"first-hop 是瓶颈"）：**与证据相反**——tail 增益 3.7× 于 head 增益
- 实施：**干净但散乱**；核心 ablation 实为单变量 clean（BD 面板原指责"5 个混杂变量"被 Tribunal + Proposer + Chair 独立证伪），但 26 configs + 4 模块中 2 个死代码确属 scope creep
- 因果：**未建立**；6 个实验全落在 36.12–36.21 dB (std ≈ 0.029)，与 null hypothesis 兼容
- 投稿准备度：**离顶会 desk-reject 很近**，工作坊/负结果范式可行

---

## 1. 5 大维度最终评分

| 维度 | 权重 | R1 | R2 融合 | R3 | R4 | R5 终裁 | 说明 |
|---|---:|---:|---:|---:|---:|---:|---|
| **Novelty** | 20% | 3.0 | 3.0 | 3.0 | 3.0 | **3.0** | 组件皆衍生（ControlNet/LoRA/LPIPS）；诊断框架潜在 5/10 但需跨域验证；当前 3/10 |
| **Story** | 25% | 2.5 | 2.5 | 3.5 | 3.5 | **3.5** | Proposer 接受 "first-hop 是瓶颈" 叙事过度；若诚实 reframe 可达 4.5/10 |
| **Implementation** | 15% | 5.3 | 5.3 | 5.3 | **5.8** | **5.8** | clean ablation 验证通过；patient split 验证通过；小瑕疵（D50 列命名、死代码）不致命 |
| **Causal** | 25% | 1.7 | 1.7 | 2.0 | 2.0 | **2.0** | 无 σ_seed → 所有 hop0 主张不可发表；Proposer 已承诺 3-seed 实验 |
| **Submission** | 15% | 2.0 | 2.0 | 2.5 | 2.5 | **2.5** | 顶会 <2% odds；workshop 40–55% odds 取决于 reframe 诚实度 |

**最终融合 = 0.20·3.0 + 0.25·3.5 + 0.15·5.8 + 0.25·2.0 + 0.15·2.5 = 0.60 + 0.875 + 0.87 + 0.50 + 0.375 = 3.225 / 10**

调整后 **3.05 / 10**（因为 Submission 维度 E 在 R1 给了 1/10 baselines，整体拉低）。

### 分数情景表（未来投影）

| 情景 | 条件 | 融合分数 | 含义 |
|---|---|---:|---|
| 当前 | 未做任何新实验 | **3.0** | 不可投稿（任何venue） |
| D4 失败（假设） | 患者泄漏 | 0 | 已排除（R4 验证通过） |
| σ_seed ≥ 0.08 dB | hop0 效应 = 0 | 2.7 | 仅适合纯负结果 workshop |
| σ_seed ≈ 0.04–0.08 dB | hop0 效应被噪声覆盖 | 3.0 | MIDL 负结果 45% |
| σ_seed < 0.04 dB | hop0 效应边缘可辨 | 3.7 | MIDL/MICCAI workshop 55% |
| +1 外部 baseline | Pix2Pix 或 PET DDPM 可比 | +0.5 | 让数字可解读 |
| +Mediation 证据 | tail 增益由 hop0 因果传播 | +0.3 | 故事一致性修复 |
| Reframe 为诊断框架主贡献 | 跨域验证成功 | +0.8 | 新颖性重新定位 |
| +Decoder FT 或 hybrid | 破 36.2 dB plateau 突破 1 dB | +1.5 | 主会 15% odds 打开 |

---

## 2. 5 轮收敛的核心发现（审稿人独立收敛 ≥4/5 票）

### 致命发现（5/5 一致）
1. **Hop0 机制效应 0.083 dB 无统计学意义** — 无 σ_seed 测量，无 CI，无 seed 重复
2. **故事与证据相反** — hop0 意在修第一跳，但 D20 增益（+0.18 dB）远小于 NORMAL 增益（+0.66 dB），差 3.7×
3. **5+ 配置全部撞在 36.12–36.21 dB plateau** — std ≈ 0.029 dB，与 "nothing is helping" 的 null 假设兼容
4. **10 dB oracle 差距无任何机制解释** — 既非架构问题（已排除）亦非训练问题（已排除），但无人给出突破路径
5. **零外部 baseline** — 36.2 dB 无参照系，无法判定是 SOTA 还是 2021 年水平

### 结构发现（4/5 一致）
6. **组件皆为已发表方法的微小特化** — FirstHopPixelEncoder ≈ ControlNet "hop0-only" 变体；HopResidualVelocityHead ≈ per-hop LoRA；L_img_hop0 ≈ perceptual loss
7. **Checkpoint-selection 污染 leaderboard** — val_multi_objective 权重在 C/v2/N1/v3 之间漂移，N1 last > N1 best 是 smoking gun
8. **26 configs + 4 模块中 2 个死代码**（iREPA Spatial Projector + SeamRefiner 在主结果配置中均未启用）

### 被证伪 / 修正的发现
9. BD R2 声称 "N1 配置与 C 差 5+ 个 hyperparameter" → **Proposer + Tribunal + Chair 独立证伪**：imgaux_boost ↔ pixenc_ablation **只差 `pixel_forcing_disabled` 一个变量**（确认 diff 输出：唯一差异是 run_name 和新增一行 `pixel_forcing_disabled: true`）
10. Panel R4 声称 "patient-level split 未验证 → desk-reject 风险" → **C Auditor R4 独立验证**：upstream `RAE/preprocess_lowdose_pet_to_rae.py::split_subjects` 已按 subject 切分（but 仍建议在 PET_LatentResidual 文档中显式声明）

---

## 3. 未来 48 小时必做（最高杠杆）

### Action 1（1 小时）— 文档化数据分割声明
在 `PET_LatentResidual/CLAUDE.md` 的数据部分添加：
> "Train/val split is **patient-level**. Upstream preprocessing (`RAE/preprocess_lowdose_pet_to_rae.py::split_subjects`, line 199–207) partitions subjects by `train_ratio=0.8, seed=42`. All slices from a given subject appear in either train or val, never both. Verified 2026-04-24."

这 1 小时投资防止评审一上来就被 "你的 train/val 怎么切？" 卡住。

### Action 2（立即决策 · 0 小时）— 终止 200K v3 run，重定向 GPU
**理由**：200K stop-criterion（"±0.05 dB of 50K"）的容差窗口已大于 5 配置跨度（0.083 dB），几乎必然触发"abandon"——它测不出任何东西。释放出的 2.7 GPU-day 用于 Exp 1。

如果仍要跑 200K，必须**先**写下可证伪的预测（例："hop0 gain 将从 50K 的 0.08 dB 扩至 200K 的 > 0.20 dB"）并保留在 git。

### Action 3（4 GPU-day）— 启动 seed variance ablation
配置：`imgaux_boost` (C) × 3 seeds {42, 123, 456} + `pixenc_ablation` (N1) × 3 seeds {42, 123, 456} = 6 runs × 50K × 16h。

报告：
- σ_seed for C, σ_seed for N1
- mean ± std for transport_avg
- Paired bootstrap 95% CI on (C − N1), per-sample n=7403
- PASS 门槛：CI 下界 > 0.03 dB

**这一个数字决定整个项目的生死**：
- σ_seed > 0.08 dB → 0.083 dB 效应已死 → 必须 reframe 为纯负结果
- 0.04 < σ_seed < 0.08 → 效应不可辨 → workshop 负结果
- σ_seed < 0.04 dB → 效应边缘成立 → 可争取 workshop 小正结果

---

## 4. 投稿前必做（4–6 周完整路径）

### Week 1: 数据验证 + σ_seed
- Action 1 + 2 + 3 上述

### Week 2: 外部 baseline
- 选 1: Pix2Pix（图像空间，最容易配置）
- 选 2: Palette / PET-DDPM（latent-diffusion，更公平对比）
- 至少 1 个，在同一 val 集上报告 clip3 PSNR + SSIM + LPIPS
- **不需要击败 baseline**——reframe 为负结果时，只要能说 "我们的 latent transport 与 X baseline 在 2σ 内无差" 即可

### Week 3: 诚实重框叙事
标题候选：
> "Pixel Injection Does Not Break the First-Hop Bottleneck: A Systematic Negative Result for Latent Flow Matching in Low-Dose PET"

关键叙事元素：
1. **正面贡献**：首个对 PET 4-hop latent transport 施加完整 A/B/C/D 诊断的系统性研究
2. **负面发现 1**：hop0 机制（pixel forcing + residual head + img aux）在 σ_seed 级别上不可区分
3. **负面发现 2**：latent-only transport 存在 36.2 dB ± 0.03 dB 结构性天花板（5 配置 + 3 seeds 验证）
4. **开放问题**：10 dB oracle 差距在哪一层？encoder/decoder 压缩损失？paradigm 限制？
5. **建议方向**：部分 decoder fine-tuning / 混合 latent-pixel / 减少 hop 数 / 提高 latent 分辨率

### Week 4: 可选——Mediation 实验（0.5 GPU-day）
如果时间允许：
- Arm A：learned hop0 + oracle hops 1–3
- Arm B：oracle hop0 + learned hops 1–3
- 若 (A − baseline)_NORMAL >> (B − baseline)_NORMAL → 确认 tail 增益由 hop0 mediation → 故事稍稍救回来

### Week 5–6: 8 页 workshop 短文撰写 + 内部 mock review

---

## 5. 绝对不能做（KILL LIST）

| # | 行为 | 原因 |
|---|---|---|
| K1 | 启动 200K v3 continuation | 测试错误假设；延迟战术 |
| K2 | 新架构实验（CCT-224 从 test branch 合回、SeamRefiner stage2 等） | Plateau 是结构性，架构迭代是浪费 |
| K3 | 激活 SpatialAlignmentProjector（iREPA）或 SeamRefiner 到主结果管线 | 死代码应保持死亡；启用只会增加 scope 不增加贡献 |
| K4 | 在任何内部文档中继续写 "hop0 mechanism solves first-hop bottleneck" | 证据相反；继续写是科学不诚实 |
| K5 | 瞄准 NeurIPS / ICML / CVPR / MICCAI 主会 | 以当前证据 < 2% odds，desk-reject 高危 |
| K6 | "训练更久就会突破"的叙事 | 5 配置 × 同 plateau 已证伪此 hypothesis |
| K7 | 继续增加 config 数量（当前 26 个） | 配置混乱已是 reviewer 会指责的点 |
| K8 | 在未完成 seed 实验前发起任何投稿 | 无 σ_seed 的 PSNR 表格在 2026 年是自杀 |

---

## 6. 投稿路径选择（按 EV 排序）

### Path 1 · 最高 EV · 负结果 workshop（推荐）
- 目标：MIDL 2027 workshop / MICCAI 2027 workshop / NeurIPS 2026 ML4H workshop
- 时间：4–6 周（从今天起）
- 成本：7 GPU-day（3 实验承诺）+ 2 周写作
- Odds：45–55%（取决于 σ_seed 结果）
- 关键：诚实 reframe，拒绝拔高

### Path 2 · 中等 EV · 诊断框架主贡献
- 目标：ISBI 2027 / MICCAI 2027 短文 / TMI 方法短论
- 时间：6–8 周
- 成本：Path 1 + 跨域验证（应用到 CT denoise cascade 或 video frame interp）
- Odds：25–35%
- 关键：把 A/B/C/D 形式化为可复用工具并开源

### Path 3 · 高成本 · 方法 pivot（不推荐短期内）
- 选项：
  - 3a. 减少 hop 数：2-hop (D50→D10→NORMAL) 或单跳 D50→NORMAL
  - 3b. 混合 latent-pixel transport（放弃 latent-only 主状态）
  - 3c. 部分 decoder fine-tuning（解冻最后 1–2 层）
- 时间：8–12 周
- 成本：新架构 + 充分实验 + 统计
- Odds 在主会：10–15%（但 workshop 兜底 60%+）
- 风险：pivot 失败后 plateau 仍在

### Path 4 · 零 EV · 继续现路线
- 目标：不现实；不再讨论

---

## 7. 5 维具体修改建议

### 7.1 Novelty（3 → 目标 4–5）
- **砍**：所有对 "novel hop-0 only pixel conditioning" 的主张；所有 "novel per-hop residual" 的主张
- **保**：重新包装为 **"系统性 empirical study"**：对 4 种架构干预的完整对比（pixel forcing, residual head, perceptual aux, chainstable scheduling）
- **加**：如果追 diagnostic framework 主贡献路线，必须跨域验证 + 数学化

### 7.2 Story（3.5 → 目标 4.5）
- **核心 pivot**：从 "我们修了第一跳" 改为 "我们系统地测试了 4 种机制，全部失败；发现了 36.2 dB 结构性 ceiling"
- **强化**：诚实讨论 10 dB oracle 差距；明确开放问题
- **删除**：IDEA_REPORT.md 的自评 novelty 分数（8.4/10 自评是 reviewer 的红布）
- **删除**：main.md / CLAUDE.md 中所有 "hop0 是 the bottleneck" 的措辞

### 7.3 Implementation（5.8 → 目标 7）
- **必须**：重命名 `D50_PSNR` 列为 `D50_decode_baseline`（eval_first_hop_224_clip3.py L185-195）
- **必须**：`gate_pix` 在 `pixel_forcing_disabled=True` 时返回 0.0（model_first_hop.py L529）
- **建议**：删除 `SpatialAlignmentProjector` (~60 LoC) 死代码；`SeamRefiner` 移到单独 experimental module 目录
- **建议**：configs 合并归档，只保留 3 个 canonical：baseline, ablation-clean, long-train
- **必须**：在 CLAUDE.md / README 显式声明 patient-level split 并链接到 preprocess 脚本

### 7.4 Causal（2 → 目标 5+）
- **Blocker**：Exp 1（3 seeds × 2 configs）必跑，报告 σ_seed + CI
- **强烈推荐**：Exp 2 mediation test（0.5 GPU-day，极低成本）
- **必须**：每个报表都带 mean ± std、paired bootstrap 95% CI
- **必须**：预登记 hypothesis + 停止准则，再跑新实验

### 7.5 Submission（2.5 → 目标 4.5+）
- 加 1 个外部 baseline（Pix2Pix 或 published PET DDPM）
- 8 页 workshop 短文（不是 main conference）
- 明确 code release plan（patient-split 验证脚本必须随论文开源）
- Dataset & ethics statement

---

## 8. Nightmare 5 Round 审查出的可复用元规则（for future projects）

1. **σ_seed 是 load-bearing number**：任何方法对比不带 seed std + CI 都是死亡主张
2. **Ablation 必须 1 变量**：config diff 要能在 `diff` 里一眼看清
3. **Story-evidence alignment check**：在任何 claim 旁写出 "如果我错了，什么证据会显现" 才算 falsifiable
4. **Plateau across N configs = structural ceiling，不是 "没找到右旋钮"**：N ≥ 3 就该改变范式而不是调参
5. **外部 baseline 永远必要**：自我对比只能证明 "变了"，不能证明 "变好了"
6. **死代码就删**：每一个实现但未启用的模块都是 reviewer 对 "实验卫生" 的负分
7. **诚实负结果 > 夸大正结果**：负结果在 workshop 是 45% 接收，夸大正结果在主会是 2%

---

## 9. 与 `AUTO_REVIEW.md` 既有审查的对比

`AUTO_REVIEW.md` Round 1 已给出 **TotalScore 4.84/10 FAIL** 判定；本次 nightmare 5 轮深挖后给出 **3.05/10** ——分数更低的原因：
- AUTO_REVIEW 当时 Causal 给了 3.3（现已降至 2.0，因 σ_seed 仍缺）
- AUTO_REVIEW 当时未识别 story 反转（tail 增益 > head 增益）
- AUTO_REVIEW 当时未识别 component 衍生性（ControlNet/LoRA/LPIPS 映射）

收敛点：两次审查都 FAIL；都要求 tf_vs_pure_gap、seam_strata、seed-repeat 等证据；都标记 "evidence gate failed"。

---

## 10. 最终一句话判决

> **当前项目是一个工程上干净、但科学上未证实、故事上与数据矛盾的负结果；唯一诚实且高概率可发表的路径是 reframe 为 "系统性 negative-result 分析"，目标 MIDL/MICCAI workshop，并在未来 48 小时内启动 σ_seed 实验以决定最终分数区间。**

---

## Round-by-Round 附录索引

| 文件 | 内容 |
|---|---|
| `00_evidence_pack.md` | 共享事实包（所有 reviewer 起点） |
| `01_round1_initial_reviews.md` | 5 角色独立初评（A · B · C · D · E）|
| `02_round2_panel_chair_convergence.md` | 面板合议 + 5 大争议 |
| `02_round2_merged_BD_story_causal.md` | B+D 联盟深挖故事+因果（含被证伪的 5-confounder claim） |
| `03_round3_proposer_rebuttal_and_rulings.md` | Proposer 反驳 + 3-judge Tribunal 裁决 |
| `03_round3_tribunal_prerule.md` | Tribunal 独立验证（确认 imgaux_boost ↔ pixenc_ablation 为 clean diff） |
| `04_round4_joint_panel_ruling.md` | E+A+B 合议 4 大 disputes + D4 新红旗 |
| `04_round4_impl_audit_final.md`（本文中 C 的实施审计已合入 §3） | Patient split 验证通过；implementation 分数升至 5.8 |
| `05_final_consensus.md`（本文件） | 最终合议 + 修改路线图 |

---

*End of Nightmare Mode 5-Round Multi-Agent Adversarial Review. Final fused score: 3.05 / 10. Next action: Verify within 48h.*
