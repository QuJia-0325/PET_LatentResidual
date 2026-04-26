# Transport 突破实验计划 v5 — 2026-04-26（最终版）

**Supersedes**: v4（SF-pair 方向已暂停）
**核心方向**: Rollout-Heavy 温和版（加大 rollout_loss 权重 + 尾重 step_weights，保护 hop0）
**实验性质**: 组合突破实验（同时改 3 个轴），非单轴机制实验
**资源约束**：GPU 数量充足，**单卡显存有限** → 不增大 batch、不引入大模型

---

## ⚡ TL;DR — 现在就该做的事

```
Step 1（今天）：启动 V5-main（rollout λ=1.5, image_aux=0.08, step_weights=[0.8, 1.0, 1.5, 2.5]）
Step 2（+5K 步）：看 val_pair_total（相变警报 < 0.00005）
Step 3（+10K 步）：看 val_chain_normal_mse 趋势
Step 4（+50K 步）：full-val eval + Path A redo
Step 5（事后）：如果有效 → ablation 归因（§4.1）；如果无效 → §11 备选方向
```

**V5-main 改动（vs 200K v3 baseline，仅 3 个轴）**：
- `rollout.lambda`: 0.25 → **1.5** (6×)
- `image_aux.lambda`: 0.12 → **0.08** (÷1.5)
- `rollout.step_weights`: [1.30, 1.20, 1.10, 1.00]（头重）→ **[0.8, 1.0, 1.5, 2.5]**（尾重）
- **其余参数与 200K v3 严格一致**（lr=8e-5, warmup=0.15, ema=on, pair_weights=头重）

---

## 0. 决策依据摘要

| 实验 | 结论 | 对 V5 的含义 |
|------|------|-------------|
| Path A 多 ckpt (v3) | ExpoGap=4.80 dB，单调上升 | exposure bias 是核心问题 ✅ |
| SF-pair v3 | schedule bug 导致无效 | 修复后重跑 |
| SF-pair v4 pilot | α≤0.10 微弱改善(5%)，α≥0.15 相变崩溃 | SF-pair 方向暂停 |
| rollout alpha 机制 | GT→混合→pred 三阶段 + 每跳 GT loss | 设计正确，权重不足 |
| 梯度占比 | roll_frac ~10%, img_frac ~85% | rollout 信号被淹没 |

**V5 定位**：这是一个**组合突破实验**（同时改 λ_roll + λ_img + step_weights），不是单轴机制实验。成功后需要事后 ablation 归因各因素贡献。

---

## 0.5 风险评估

**最大风险**：rollout 6× + step_weights 方向反转可能改变梯度平衡。

**为什么温和版风险可控**：
1. **rollout 在 V3 200K 已存在 λ=0.25 并稳定**——6× 是放大已内化的信号，不引入新分布
2. **image_aux 仅降 ÷1.5（0.12→0.08）**——像素域锚定保留大部分强度，保护 hop0
3. **每步 GT 监督**——rollout 每跳都有 `||z_pred - z_gt||²` 拉力
4. **val_pair_total 相变监控**：+5K 步即检查，早期发现问题
5. **LR/EMA/pair_weights 与 baseline 一致**——不引入额外混淆变量

> 完整的 SF vs rollout 5 维机制对比表已移至 §10（事后分析）；B4/B5/B1 验证已移至 §11（论文补强证据，**不阻塞主实验**）。

---

## 1. 实验设计

### 1.1 V5-main（组合突破实验，立即启动）

从 200K v3 best.pt（step=86800）resume，50K 新增步。
**仅改 3 个轴**，其余参数与 200K v3 严格一致（已审计确认）。

| 参数 | 200K v3 真实值 | V5-main | 变化 | 理由 |
|------|--------------|---------|------|------|
| `rollout.lambda` | 0.25 | **1.5** | 6× | 将 roll_frac 从 ~10% 提升到 ~30% |
| `image_aux.lambda` | 0.12 | **0.08** | ÷1.5 | 温和降低，保护 hop0 像素质量 |
| `rollout.step_weights` | [1.30, 1.20, 1.10, 1.00]（**头重**） | **[0.8, 1.0, 1.5, 2.5]**（**尾重**） | 方向反转 | 见 §1.2 |
| `self_forcing_pair` | — | **disabled** | — | SF 方向暂停 |
| pair_loss_weights | [1.20, 1.10, 1.05, 1.00] | **不变** | — | 与 v3 一致 |
| best_metric d20 weight | 0.50 | **不变** | — | 与 v3 一致 |
| lr | 8e-5 | **不变** | — | 与 v3 一致 |
| lr warmup_ratio | 0.15 | **不变** | — | 与 v3 一致 |
| ema | enabled, decay=0.9999 | **不变** | — | 与 v3 一致 |

### 1.2 step_weights 设计理由

200K v3 baseline 的 step_weights 是 **[1.30, 1.20, 1.10, 1.00]（头重）**：
优先 hop0（D50→D20），因为它有最大的 latent displacement（pair_v_std=0.009634）。

V5 改为 **[0.8, 1.0, 1.5, 2.5]（尾重）**——这是**方向反转**，不是微调：

```
                ExpoGap   v3 weight   V5 weight   变化
D50→D20         0.00 dB   1.30        0.80        ÷1.6
D20→D10         3.38 dB   1.20        1.00        ÷1.2
D10→D4          4.71 dB   1.10        1.50        ×1.4
D4→NORMAL       6.32 dB   1.00        2.50        ×2.5
```

### 1.3 有效权重对比（使用 200K v3 真实 baseline）

baseline Σw = 1.30+1.20+1.10+1.00 = 4.60
V5 Σw = 0.80+1.00+1.50+2.50 = 5.80

| Hop | v3 有效权重 (λ=0.25) | V5 有效权重 (λ=1.5) | 提升倍数 |
|-----|---------------------|---------------------|---------|
| hop0 | 0.25 × 1.30/4.60 = **0.071** | 1.5 × 0.80/5.80 = **0.207** | 2.9× |
| hop1 | 0.25 × 1.20/4.60 = **0.065** | 1.5 × 1.00/5.80 = **0.259** | 4.0× |
| hop2 | 0.25 × 1.10/4.60 = **0.060** | 1.5 × 1.50/5.80 = **0.388** | 6.5× |
| hop3 | 0.25 × 1.00/4.60 = **0.054** | 1.5 × 2.50/5.80 = **0.647** | **12.0×** |
| image_aux | 0.12 | **0.08** | ÷1.5 |

> **注意 hop3 的 12× 提升**：v3 baseline 的 hop3 权重（1.00）是四跳中最小的（头重设计）；
> V5 的 hop3 权重（2.50）是四跳中最大的（尾重设计）。方向反转 + λ 6× = 12× 提升。

---

## 2. Go/No-Go 监控（V5-main）

### 2.1 相变早期警报（最高优先，0 成本）

| 检查点 | val_pair_total 阈值 | 动作 |
|--------|---------------------|------|
| +5K | > 0.00005 | ⛔ **立即停 → 降 λ_roll 到 1.0 重试** |
| +10K | > 0.00003 | ⚠️ 关注，+15K 再看 |
| +25K | > 0.00002 | ⛔ 止损 |

**理由**：V4 SF pilot 中 val_pair_total 从 0→0.000196 是相变的最早信号（比 val_select 提前 1-2 个 val 点）。V5-main 的 rollout 6× 改变了梯度平衡，需要监控。

### 2.2 效果监控（自顶向下版，看肉眼可见的改善）

**核心问题**：+10K~+25K 步时，val_chain_normal_mse 是否有**肉眼可分辨**的改善？

| 检查点 | val_chain_normal_mse | 解读 |
|--------|---------------------|------|
| +10K | < baseline × 0.90（改善 ≥ 10%）| ✅ 方向对，继续跑 |
| +10K | baseline × 0.95~1.00（改善 ≤ 5%）| 🟡 可能在 noise 内，继续观察到 +25K |
| +10K | > baseline × 1.05 | ⛔ 止损 |
| +25K | < baseline × 0.90 | ✅ 跑完 50K，full-val + Path A |
| +25K | baseline × 0.95~1.05 | ❌ **方向不成立**——梯度强度不是瓶颈，进入 §11 备选 |
| +50K | — | full-val eval + Path A redo on best.pt |

**为什么用 10% 而非 5% 作为成功线**：
- V4 SF best 也有 5%，但那是临界点偶然——5% 在 val 噪声边缘
- 10% 才能脱离 noise，**肉眼可见**的改善
- 如果 V5-main 跑到 +25K 还连 5% 都达不到，就是方向错了，不要再投资

### 2.3 梯度占比监控（次要，仅做诊断）

每 50 步从 stdout `[train]` 行读取：
- `roll_frac` 目标：**40-60%**（从当前 ~10% 推到主导）
- `img_frac` 目标：**20-40%**（从当前 ~85% 下降）
- `pair_frac` 底线：**≥ 2%**（pair 监督不能被压死）

如果 `roll_frac` 没达到 30% → 说明 image_aux ÷3 还不够，梯度被 image 主导——但这是次要问题，不阻塞。

---

## 3. 成功标准（挂钩医生需求）

> **Path A baseline 必须用 step 86800 ckpt 重新跑一次**（旧 6.32 dB 来自 Scheme C，不可直接用）。记为 **ExpoGap_86800**（待测，预计 5.5-6.5 dB）。这是唯一对比基线。

| 优先级 | 指标 | 阈值 | 含义 |
|--------|------|------|------|
| 1 | `val_chain_normal_mse` | **改善 ≥ 10% vs baseline** | NORMAL 档是医生看的输出，10% 是肉眼可见线 |
| 2 | Path A exposure_gap (D4→NORMAL) | ≤ ExpoGap_86800 × 0.85（降 ≥ 15%）| 末端 hop exposure bias 大幅缓解 |
| 3 | `val_select_score` | < baseline × 0.95 | chain 整体改善 ≥ 5% |
| 4 | `val_chain_d20_mse` | 不恶化 > 5% | 保护 D50→D20 质量 |

**最低可接受门槛**（"看得过去"的最低线）：
- val_chain_normal_mse 改善 ≥ 5%
- val_select 不退化
- hop0 PSNR 不退化 > 0.3 dB
- 末端 ExpoGap 至少不变差

如果 V5-main 50K 步后**连最低门槛都达不到**，说明 rollout 加权方向不成立，进入 §11 备选方向。

---

## 4. Claims Matrix（自顶向下版）

只有跑出"看得过去"的结果之后才回头分析。Claims 按结果强度分级：

| V5-main 结果 | 允许的 claim |
|------|-------------|
| ✅ val_chain_normal_mse 改善 ≥ 10% **且** ExpoGap ↓ | "加权 rollout 直接对抗了 cascade exposure bias，方向被证实" |
| ✅ 改善 ≥ 10% 但 ExpoGap 不变 | "权重提升改善了 chain 整体精度，机制留给 §10 事后分析" |
| 🟡 改善 5-10% | "信号微弱，可能在 noise 边缘，full-val 确认是否为真信号" |
| ❌ 改善 < 5% 或持平 | "梯度强度不是当前瓶颈——方向证伪。进入 §11 备选" |
| ❌ < baseline | "rollout 过强压制了 pair/image，降 λ_roll 重试" |
| ⛔ 触发 val_pair_total 相变 | "rollout 在当前 backbone 容量下也会触发盆地切换；降 λ_roll 重试，同时把方向 4（EMA teacher）的优先级提升" |

### 4.1 失败回退路径（简化版）

| V5-main 失败模式 | 下一步 |
|-----------------|--------|
| **+5K 相变** | 立刻停 → 降 λ_roll 到 1.0 重试 |
| **+10K 退化 > 5%** | 停 → 降 λ_roll 到 1.0 重试 |
| **+25K 改善 < 3%** | 跑到 50K 看 → 仍 < 3% → 进入 §11 备选方向 |
| **hop0 PSNR 退化 > 0.5 dB** | 停 → 恢复 image_aux=0.12 重试 |
| **梯度 NaN / loss 爆炸** | 立刻停 → step_weights 改为 [1.0, 1.0, 1.0, 1.0]（均匀）|

---

## 5. 时间线

| GPU | 任务 |
|-----|------|
| GPU-0 | **V5-main**（rollout λ=1.5, image_aux=0.08, 50K steps, from step 86800） |
| GPU-1 | **Path A redo on step 86800 ckpt**（一次性，~2h）→ 完成后待命 |
| GPU-2 | **200K v3 继续**（如未完成）→ 完成后待命 |

### 时间线

| Day | GPU-0 | GPU-1 | 产出 |
|-----|-------|-------|------|
| 0 AM | V5-main config + script 已就绪 | Path A redo on 86800 ckpt | ExpoGap_86800 baseline |
| 0 PM | **V5-main 启动** | 待命 | 实验运行 |
| 1 AM | +5K 相变检查 | — | 第一关 go/no-go |
| 1 PM | +10K 效果检查 | — | 第二关 |
| 2 | +25K 效果检查 | — | 第三关 |
| 3 | +50K 完成 + full-val | Path A redo on V5-main best | 最终结果 |
| 4 | 事后分析（§10）/ 备选方向决策 | — | 决策报告 |

---

## 6. 代码改动

| 文件 | 改动 | 状态 |
|------|------|------|
| config: `pet_flow_first_hop_224_v5_rollout_heavy.yaml` | V5-main config | ✅ 已创建 |
| `scripts/launch_v5_rollout_heavy.sh` | 自动读 ckpt step + 设 max_steps + 启动 | ✅ 已创建 |
| `train_first_hop.py` | **0 行改动** | ✅ 无需修改 |

§11 备选方向涉及的 config / 工具文件等到主实验跑完再看是否需要。

---

## 7. 风险与缓解（极简版）

| 风险 | 缓解 |
|------|------|
| V5-main 触发相变（val_pair_total 跳变） | +5K 步即检查，立即停，降 λ_roll 重试 |
| hop0 像素退化（image_aux ÷1.5） | hop0 PSNR 独立监控，恢复 image_aux=0.12 重试
| pair_loss 被压死（roll_frac > 60%） | pair_frac ≥ 2% 底线，低于即降 λ_roll |
| step_weights 末端过拟合 | val_chain_d20_mse 独立监控；触发回退到 [1.0, 1.0, 1.0, 1.0]（均匀） |
| 86800 不是最强起点 | 直接用 V3 best.pt step 86800，不纠结；如有 final 200K best.pt 优先用 |
| 50K 步全部跑完仍持平 | 进入 §11 备选方向，不再投资 rollout 加权路线 |

---

## 8. 从 V4 吸取的教训（硬编码进 V5 流程）

1. **val_pair_total 是相变早期警报**：独立监控，阈值 0.00005
2. **rolling val 仅用于趋势**：最终结论必须 full-val eval
3. **max_steps 必须是 resume_step + 新增步数**：launch script 自动算
4. **SF 相关功能保持 disabled**
5. **baseline 必须用同 ckpt 重算**：Path A 的 ExpoGap_86800 不能直接套旧 6.32 dB
6. **失败回退路径必须预先想好**：见 §4.1
7. **不做无用功的对照实验阻塞主实验**：B4/B5/B1/方向 5 都是论文补强证据，不阻塞 V5-main 启动

---

## 9. 显存约束下的并行策略

由于**单卡显存有限**，所有实验保持 V3 同样配置：
- 不增大 batch size
- 不引入更大的 backbone（DiT-S 保持，DiT-B/L 暂不考虑）
- 不开 gradient checkpointing 之外的额外内存优化（避免引入未知数）

利用 GPU 数量充足的优势：
- **3 卡并行**（§5）：V5-main + Path A + 待命卡
- 任何回退实验立即在待命卡上启动，不打断主实验
- B5 参数 L2 测量、Path A redo 这类一次性任务在 GPU-1 完成，不占主实验

---

## 10. 事后分析框架（仅在 V5-main 跑出"看得过去"的结果后启动）

> 自顶向下原则：**先有结果，再分析机制**。如果 V5-main 50K 步后没拿到 ≥ 5% 改善，下面所有事后分析都不需要做——因为方向已被证伪。

### 10.1 如果 V5-main 成功（≥ 10% 改善）

需要回答：
1. **改善来自哪里**：roll_frac 占比变化、step_weights 末端加重、image_aux 降低，三者各自贡献多少？
   - 做法：跑 ablation 系列（V5-no-stepweights / V5-image-restore / V5-rollout-1.0）
2. **是否 ExpoGap 直接缓解**：跑 Path A on V5-main best.pt，对比 ExpoGap_86800
3. **机制论证**：补完 SF vs rollout 5 维对比表（见 §10.4）作为论文章节
4. **可推广性**：在更大 scale（如 224→256）或更深 chain（4→6 hops）上是否仍成立

### 10.2 如果 V5-main 性能持平（梯度强度不是瓶颈）

需要回答：
1. **真正瓶颈在哪**：是 backbone 容量？数据多样性？loss 形式？
2. **进入 §11 备选方向决策**：方向 4（EMA teacher）/ 方向 5（结构改动）/ 方向 6（数据增强）
3. 为论文留下 negative result：rollout 加权方向已被实验证伪，节省同行重复

### 10.3 如果 V5-main 触发相变

需要回答：
1. 是否同 V4 SF 量级（B5 参数 L2 距离）
2. 是否 z_curr 没 detach（B4 代码核对）
3. 这些数据**支持 §10.4 的 SF 机制论证**——把这次相变作为 V4 失败的可重复性证据

### 10.4 SF vs rollout 机制对比表（事后写入论文）

V4 Finding D 揭示了"自洽循环 → 参数被推出 GT 盆地"的相变机制。V5 实验将提供 rollout 加权场景下的对比数据点。

| 维度 | SF-pair（V4 失败）| rollout_loss（V5 主推）| 风险等级差异 |
|---|---|---|---|
| **训练历史** | V4 才引入，新分布 | V3 200K 已存在 λ=0.25，模型已适应 | rollout 是放大已被内化的稳定信号 |
| **每步监督密度** | pair_loss 每个 batch 只算 1 跳 endpoint | rollout 在 4 跳 chain 上**每步**计算 `||z_pred − z_gt||²` | rollout 每跳都有 GT 拉力 |
| **输入分布稳定性** | α 从 0 ramp 到 1.0 → 持续漂移 | alpha=1.0 固定 → 分布稳态 | rollout 不引入分布漂移 |
| **自洽循环强度** | z_src_pred 直接进入 pair loss | z_curr 只用作下一步输入，loss 仍以 z_gt 为锚 | rollout 的"自洽"被每步 GT 监督打断 |
| **可证伪信号** | val_pair_total 跳变（V4 实测）| 同样监控 val_pair_total | 复用同一警报 |

**保留风险**：rollout 风险**显著低于 SF**，但**不为零**——梯度方向仍可能慢慢把参数推离 GT 盆地，速度比 SF 慢。

### 10.5 V5 启动后才做的诊断任务（不阻塞主实验）

| 任务 | 描述 | 何时做 |
|------|------|--------|
| B4 | 核对 SF + rollout 的 stop-grad 状态（5 min 代码审查） | V5-main +10K 检查空隙 |
| B5 | 算 V3 正常训练相邻 4K 步参数 L2 距离 → 给"正常移动幅度"baseline | V5-main +25K 检查空隙 |
| B1 | 算 baseline val 自然波动 std → 校准 5%/10% 阈值 | V5-main +5K 后顺手做 |

这些任务**不阻塞主实验**，只是用空闲时间补充诊断证据，方便事后写论文或做决策。

---

## 11. 备选方向（仅在 V5-main 失败时考虑）

### 11.1 方向 4：EMA Teacher Backbone（如 V5-main 触发相变）

**触发条件**：V5-main 触发 val_pair_total 相变 + B5 确认参数移动是同款机制。

#### 设计
维护一个 EMA backbone 作为独立 teacher：
```
θ_teacher_t = β × θ_teacher_{t-1} + (1-β) × θ_student_t
β = 0.999  # 标准 EMA 系数
```
**关键改动**：z_src_pred / z_curr 由 **teacher** 生成（teacher 不接收梯度），student 在 teacher 提供的输入分布上学习。这打破了 V4 SF 的"自洽循环"。

#### 显存影响
- 增量：~200-400 MB（视 dtype）—— 在显存有限约束下**可承受**
- teacher forward 不存 activation（无梯度），开销 < student forward

#### 实施成本
- ~50 行代码（EMA update hook + teacher forward 替换）
- 新 config 项 `ema_teacher: { enabled, beta, start_step }`

### 11.2 方向 5：SF 极保守对照（如 V5-main 持平且想保留 SF 选项）

**触发条件**：V5-main 持平（不是相变，是真没改善）+ 想确认 SF 是否还有未来。

```yaml
# pet_flow_first_hop_224_v5_sf_minimal_control.yaml
self_forcing_pair:
  enabled: true
  alpha_sf_end: 0.05       # 远低于 V4 的 1.0
  warmup_steps: 2000
  ramp_steps: 20000        # 极慢
  schedule_origin: resume_relative
optimizer:
  lr: 1e-5                 # V4 是 4e-5
max_steps: resume_step + 5000   # 只跑 5K 步
rollout: { lambda_start: 0.25 }
image_aux: { lambda_start: 0.12 }
```

**Go/No-Go**：5K 步看 `val_pair_total` 是否跳变。
- 跳变 → SF 方向彻底证伪
- 不跳变 → 方向 4 EMA teacher 优先级提升

### 11.3 方向 6：监督形式改动（如 V5-main 持平且想换思路）

不再调权重，改 loss 形式：
- pair_loss 从 endpoint MSE → multi-step velocity matching
- 引入 perceptual loss on hop3 输出
- 引入 chain consistency loss（要求 hop1+hop2+hop3 chain 输出 = hop3 直接预测）

### 11.4 方向 7：数据 / 监督多样性（最后的兜底）

- 增加更多 dose level（D2 / D1 等）作为 chain 中间监督
- 引入领域先验（PET 物理模型作为 guidance）
- 跨域数据增强

---

## 12. 与 V4 分析报告的引用关系

| 本文档章节 | 引用 v4 分析报告（[v4_sf_pilot_early_analysis_20260426.md](../0426/v4_sf_pilot_early_analysis_20260426.md)）|
|-----------|--------------------------------------------------------------------------------------------|
| §0.5 风险快评 | §10.5 Finding D（自洽循环 + 盆地切换）|
| §2.1 val_pair_total 阈值 0.00005 | §10.5 D.1（V4 跳变到 0.000196 的早期信号）|
| §3 Path A baseline 重算 | §10.1（旧 6.32 dB 来自 Scheme C，非 step 86800）|
| §4 Claims Matrix 强度分级 | §10.5 D.2-D.3（参数空间盆地假设的可证伪点）|
| §10.4 SF vs rollout 对比 | §10.5 D.4 第 4 条（独立 teacher 打破自洽循环）|
| §11.1 方向 4 EMA teacher | §10.5 D.4 第 4 条 |
| §11.2 方向 5 SF 极保守 | §10.5 D.4 和 §6 H6（ramp/LR 是否是 SF 失败原因）|
