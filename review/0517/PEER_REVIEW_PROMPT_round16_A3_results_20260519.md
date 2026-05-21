# Peer Review Round 16 — A3 Capacity-Only Result Interpretation

- date: 2026-05-19
- branch: foc_lite_hop0 (commit 962bc91, A3 结果已 merge)
- 主审对象: **A3 capacity-only 结果及其机制/战略含义**
- 上游: Round 13 战略重审 + Round 14 A3 execution prep + Round 15 3-slot launch
- 当前状态: **A3 已完成; V13 / V14 仍在运行**
- target reviewers: 2-3 个 AI 独立评审, **不互见草稿**

---

## 0. 本轮范围与边界

Round 16 不是 launch review, 也不是重新审 A3 task md. 本轮只审:

1. **A3 capacity-only 对 V18 机制解释的冲击有多强**
2. **现在能先下哪些结论, 还必须等待 V13 / V14 哪些结论**
3. **V18 / KL / decoder / transport 这几条研究线各自还剩多少可信空间**

**本轮审**:
- A3 matched-step 结果是否足以判定 "V18 direct decode(z_GT) 提升主要来自 decoder capacity"
- A3 对 "KL pullback 有效" 叙事的打击程度
- A3 对 transport-vs-decoder bottleneck 判断的影响
- 在 V13 / V14 未出结果前, 哪些战略动作可以先做, 哪些还不该做

**本轮不审**:
- V13 / V14 运行是否正确 (它们仍在跑)
- A3 task md / yaml 执行质量 (已由前轮 review + 结果提交覆盖)
- Round 1-15 已签的 launch / standing rules 本身

---

## 1. A3 结果事实锚点 (已 merge)

来源:
- `review/0517/V18_capacity_only/V18_CAPACITY_ONLY_A3_REPORT_20260519.md`
- `review/0517/V18_capacity_only/kl_drift_with_cap_20260519_180206/KL_DRIFT_REPORT.md`
- `review/0517/V18_capacity_only/V18_capacity_only_metrics_20260518.jsonl`

### 1.1 A3 的实验定义

A3 = `V18_capacity_only`:

- warm start: `V7.best.pt @ 160000`
- train to: `170000`
- LoRA rank / blocks: **与 V18 相同**
- `lambda_kl = 0.0`
- `save_interval = 5000`, 因此有 `step_165000.pt` 与 `step_170000.pt`

它的目标不是追求更高最终分数, 而是隔离:

- **保留 decoder capacity**
- **去掉 KL pullback**

### 1.2 A3 raw full-val chain 指标

来自 `metrics.jsonl` 的 full-val 行:

| step | val_select_score | D20 MSE | D10 MSE | D4 MSE | NORMAL MSE | hop0 image total |
|---:|---:|---:|---:|---:|---:|---:|
| 165000 | 0.0009037353 | 0.0003304433 | 0.0002966598 | 0.0002630874 | 0.0002454921 | 0.0075349176 |
| 170000 | 0.0009039954 | 0.0003307622 | 0.0002969189 | 0.0002632126 | 0.0002454063 | 0.0075398990 |
| delta 170K-165K | +0.0000002601 | +0.0000003189 | +0.0000002592 | +0.0000001252 | -0.0000000858 | +0.0000049814 |

最直接读法:

- 170K 相比 165K **没有实质链式提升**
- `val_select_score` 方向上还略差
- NORMAL MSE 略好, 但量级极小

### 1.3 A3 direct `decode(z_GT)` PSNR_clip3 probe

协议:

- split: `val`, `n=7403`
- metric: `src.utils.metrics.calc_psnr_clip3`
- object: `PSNR_clip3(decode_crop(z_GT), x_target)`
- **不经过 transport rollout**

结果:

| checkpoint | step | D20 | D10 | D4 | NORMAL |
|---|---:|---:|---:|---:|---:|
| V7.best | 160000 | 46.6356 | 48.7436 | 50.8319 | 52.6341 |
| V18.best | 165000 | 46.6850 | 48.7993 | 50.9012 | 52.7314 |
| V18.step170k | 170000 | 46.7212 | 48.8408 | 50.9527 | 52.8027 |
| V18.last | 200000 | 46.7482 | 48.8630 | 50.9635 | 52.7980 |
| V18-cap.last | 170000 | 46.7221 | 48.8420 | 50.9542 | 52.8047 |

主判据: matched-step `V18-cap.last(170K) - V18.step170k`

| comparison | D20 | D10 | D4 | NORMAL |
|---|---:|---:|---:|---:|
| V18-cap.last - V18.step170k | +0.0009 | +0.0012 | +0.0015 | +0.0020 |

相对 V7.best:

| comparison | D20 | D10 | D4 | NORMAL |
|---|---:|---:|---:|---:|
| V18.best - V7.best | +0.0493 | +0.0557 | +0.0692 | +0.0973 |
| V18.step170k - V7.best | +0.0855 | +0.0972 | +0.1207 | +0.1686 |
| V18.last - V7.best | +0.1126 | +0.1194 | +0.1315 | +0.1639 |
| V18-cap.last - V7.best | +0.0865 | +0.0984 | +0.1223 | +0.1707 |

### 1.4 当前直观结论候选

**候选 A**: A3 基本坐实 "V18 在 GT latent manifold 上的 direct decode 提升主要来自 LoRA decoder capacity, 而不是 KL pullback"

**候选 B**: A3 只能否定 "KL 是必要条件", 但不能完全否定 KL 在 `170K -> 200K` 或 `z_pred` 路径上有次级作用

**候选 C**: A3 结果虽然强, 但 still insufficient; 必须再跑别的 clean control 才能真正判死 KL 叙事

---

## 2. 已知上游事实 (供 reviewer 约束推理)

### 2.1 V18 canonical full-val 结果

来源: `review/0517/V18_decoder_lora/V18_FINAL_RESULTS_20260518.md`

| checkpoint | D20 | D10 | D4 | NORMAL |
|---|---:|---:|---:|---:|
| V18.best @ 165000 | 35.4382 | 35.8346 | 36.3944 | 36.8112 |
| V18.last @ 200000 | 35.4212 | 35.8439 | 36.4124 | 36.8426 |

对应已知对比:

- V18 best-vs-V7.best = `+0.0302 dB` (NORMAL)
- V18 last-vs-V7.best = `+0.0617 dB` (NORMAL)

### 2.2 B42 仍成立

来自前轮代码审读:

`train_first_hop.py:2230`

```python
z_kl = main_out["z_pred"] if bool(kl_cfg.get("use_pred_latent", True)) else main_batch["z_dst"]
```

因此在 V18 当前配置 `use_pred_latent=true` 下:

- KL pullback **不是直接作用在 `decode(z_GT)` 路径上**
- 任何 "V18 改善 GT-manifold direct decode = KL 直接成功" 的说法, 在机制上都已经极不稳

### 2.3 当前未完成实验

- **V13 仍在运行**: 真 image_aux 单变量分离
- **V14 仍在运行**: d_pure / seed noise floor

所以当前 still missing:

- V7-V8 中 image_aux 单变量贡献到底多大
- 小于 `~0.03-0.06 dB` 的改善里, 有多少只是 d_pure 噪声

---

## 3. 给 reviewer 的 7 个问题

### Q1 — A3 是否已经足以把 "KL 解释 V18 的 GT-manifold direct gain" 基本判死?

请在候选 A/B/C 中选主推, 并解释:

- `V18-cap.last - V18.step170k = +0.0020 dB` 是否已经足够小到可视为 matched-step tie?
- 在 B42 已成立的前提下, 还有没有合理机制能让当前 V18 的 GT-manifold direct gain 主要归因于 KL?
- 你认为现在最准确的表述是:
  - "KL 被基本证伪"
  - "KL 在 direct decode 解释上被证伪, 但在 rollout/path coupling 上仍未判定"
  - 还是更保守的别的说法?

### Q2 — A3 是否也在 chain 指标上几乎复现了 V18@165K?

请只用已知数字判断:

- A3@165K `val_select_score = 0.0009037353`
- V18.best@165K `val_select_score = 0.0009037494`
- A3@165K `NORMAL MSE = 0.0002454921`
- V18.best@165K `NORMAL MSE = 0.0002454947`

问题:

- 这是否意味着 **至少在 160K -> 165K 这段**，V18 已观察到的 chain improvement 也几乎全可由 capacity-only 解释?
- 如果是, 那么 "V18 有效" 这个命题是否应被改写为 "decoder capacity 微幅有效, 但 KL 设计未显示出额外贡献"?
- 如果不是, 你认为这里还缺哪条关键证据?

### Q3 — A3 对 V18 / decoder / transport 三条线分别意味着什么?

请分别评估:

1. **V18/KL 线**:
   - 还剩多少继续投入的理由?
   - `V18-clean(use_pred_latent=false)` 是否还有 resurrection 价值?

2. **decoder capacity 线**:
   - A3 是否证明 "decoder capacity 确实能改善 GT-manifold direct decode, 但链路收益极弱"?
   - 这是否说明 decoder 已不是主瓶颈, 继续 rank/blocks sweep EV 很低?

3. **transport 线**:
   - A3 是否反而强化了 "真正未解的是 z_pred / rollout / chain path" 这个判断?
   - 还是说 A3 只是把问题从 KL 推回到 "整个 V18 story 本来就 marginal"?

### Q4 — 在 V13 / V14 未完成前, 哪些战略结论现在就能下, 哪些必须等?

请分两类列出:

**现在就能下的结论**:
- 例如 "不能再把 `decode(z_GT)` 提升当成 KL 证据"
- 例如 "V18 的主要 direct gain 解释应迁移到 capacity"

**必须等待 V13 / V14 的结论**:
- 例如 "项目 ceiling 是否已逼近"
- 例如 "V18 +0.03/+0.06 dB 是否超过 d_pure 噪声"
- 例如 "V7 主线收益究竟来自 image_aux 还是其他联合变化"

### Q5 — 现在还有没有合理理由继续保留 V18-clean / 新 KL control 作为高优先级?

此前某些路径包括:

- A: `V18-clean (use_pred_latent=false)`
- B: 更大 rank / 更多 decoder blocks 的 V18-family sweep
- C: 聚焦 `z_pred` / rollout 的新 transport-side intervention

请评估:

- A3 之后, A 是否已经跌出高优先级?
- 若还保留 A, 它要回答的唯一问题是什么?
- B 是否已经沦为 "继续优化 direct decode 但对 chain 不敏感" 的低 EV 路线?
- C 是否因此被动升为更合理主线?

### Q6 — 论文叙事现在应如何重写?

请评估哪种叙事最诚实且最稳:

- narrative A: "KL pullback 改善 decoder manifold, 但被 transport 吞噬"  
  当前看是否已经不成立?

- narrative B: "decoder capacity 可改善 GT-manifold direct decode, 但 improvement 不自动转化为 transport chain gain"  
  是否是当前最可信叙事?

- narrative C: "整个 V18 family 是 marginal result; 真正能不能讲 paper, 要等 V13/V14 把 image_aux 和 d_pure 分离后再定"

请明确:

- 现在 paper narrative 是否应该主动撤下 KL 成功故事
- 是否需要把 A3 作为负控制/反证主体写进主文或 appendix

### Q7 — 本 prompt 是否引入了新的偏差 (B61+)?

请特别检查是否存在:

- **过度提前终局偏差**: 因为 A3 很强, 就过早把整个项目战略一次性判死
- **capacity anchor 偏差**: 看到 A3 成功, 就过度把所有收益都归给 decoder capacity
- **direct-decode / chain 混淆偏差**: 把 `decode(z_GT)` 与 transport rollout 的结论混在一起
- **等待依赖偏差**: 明明 V13/V14 还没回来, 却提前把 ceiling / significance 写死

---

## 4. 资料目录

### 4.1 本轮主审对象

- [V18_CAPACITY_ONLY_A3_REPORT_20260519.md](PET_LatentResidual/review/0517/V18_capacity_only/V18_CAPACITY_ONLY_A3_REPORT_20260519.md)
- [KL_DRIFT_REPORT.md](PET_LatentResidual/review/0517/V18_capacity_only/kl_drift_with_cap_20260519_180206/KL_DRIFT_REPORT.md)
- [KL_DRIFT_SUMMARY.json](PET_LatentResidual/review/0517/V18_capacity_only/kl_drift_with_cap_20260519_180206/KL_DRIFT_SUMMARY.json)
- [V18_capacity_only_metrics_20260518.jsonl](PET_LatentResidual/review/0517/V18_capacity_only/V18_capacity_only_metrics_20260518.jsonl)
- [V18_capacity_only_train_gpu1_20260518_173445_tmux.log](PET_LatentResidual/review/0517/V18_capacity_only/V18_capacity_only_train_gpu1_20260518_173445_tmux.log)

### 4.2 上游对照材料

- [V18_FINAL_RESULTS_20260518.md](PET_LatentResidual/review/0517/V18_decoder_lora/V18_FINAL_RESULTS_20260518.md)
- [KL_DRIFT_REPORT.md](PET_LatentResidual/review/0517/V18_decoder_lora/kl_drift_20260518_123834/KL_DRIFT_REPORT.md)
- [REVIEW_INTEGRATION_round13_20260518.md](PET_LatentResidual/review/0517/REVIEW_INTEGRATION_round13_20260518.md)
- [REVIEW_INTEGRATION_round15_20260518.md](PET_LatentResidual/review/0517/REVIEW_INTEGRATION_round15_20260518.md)
- [V18_design_rationale.md](PET_LatentResidual/review/0517/V18_decoder_lora/V18_design_rationale.md)

### 4.3 代码 context

- [train_first_hop.py](PET_LatentResidual/train_first_hop.py) (`z_kl` 选择逻辑)
- [probe_v18_kl_drift.py](PET_LatentResidual/tools/probe_v18_kl_drift.py)
- [model_first_hop.py](PET_LatentResidual/pet_lr/model_first_hop.py) (`decode_crop` path)

---

## 5. 输出格式

每位 reviewer 独立产出 markdown:

### 5.1 7 个问题逐条回答 (APPROVE / MODIFY / REJECT)

### 5.2 主 verdict
请在下列选项中选一个主路径:

- **A**: A3 已足够把 KL direct-decoder 叙事基本判死; 后续主看 transport / z_pred path
- **B**: A3 很强但还不够; 仍应优先补一个新 KL-specific control
- **C**: 现在只允许下局部结论; 战略结论必须等 V13 / V14
- **D**: V18 family 应整体降级, paper narrative 转为 marginal / negative / partial result

### 5.3 立即可执行建议

- 在 V13 / V14 仍运行期间, claude / user 现在就可以做什么
- 哪些动作应明确暂停, 等 V13 / V14 回来再做

### 5.4 新偏差 (B61+)

若发现本 prompt 自身引入新的叙事偷换 / anchor / 过度外推, 请明确点出。

---

## 6. 约束与提醒

- **不要**把 `decode(z_GT)` direct probe 与 transport rollout 结论混为一谈
- **不要**忽略 B42: 当前 V18 的 KL 不是直接作用在 GT path
- **不要**因为 A3 强就自动宣布整个项目结束; V13 / V14 仍是关键未完成依赖
- **特别关注**:
  - A3 是否只证伪 KL 的一种叙事, 还是更大范围地改写了 V18 的主价值
  - A3@165K 与 V18.best@165K 的几乎重合是否意味着 V18 早期 chain gain 也主要是 capacity
  - 现在 paper 应撤下哪些说法, 保留哪些说法

---

## 7. 本轮元说明

| 轮次 | 范围 | substrate |
|---|---|---|
| 12 | V18 final PSNR substrate | canonical eval |
| 13 | project strategy re-evaluation | KL drift + 历史 PSNR |
| 14 | A3 execution prep | yaml + task md |
| 15 | 3-slot launch decision | slot allocation |
| **16** | **A3 结果解释** | **A3 report + raw metrics + matched-step probe** |

预期 Round 16 outcome:

- 最佳: reviewer 对 "A3 改写了 V18 机制解释" 达成强共识, 同时明确哪些战略结论要等 V13/V14
- 中等: reviewer 认为 A3 已很强, 但仍建议补一个更纯的 KL-specific control
- 最差: reviewer 发现当前 prompt 仍在 direct/chain 混淆, 或 A3 证据被过度外推