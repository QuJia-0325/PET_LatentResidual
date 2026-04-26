# Transport 突破实验计划 v5 — 2026-04-26（最终版）

**Supersedes**: v4（SF-pair 方向已暂停）
**核心方向**: Rollout-Heavy 温和版（加大 rollout_loss 权重，保护 hop0 像素质量）
**资源约束**：GPU 数量充足，**单卡显存有限** → 不增大 batch、不引入大模型

---

## ⚡ TL;DR — 现在就该做的事

```
Step 1（今天）：启动 V5 温和版（rollout λ=1.5, image_aux=0.08, step_weights=[0.8, 1.0, 1.5, 2.5]）
Step 2（+5K~+10K 步）：看 val_pair_total（相变警报）+ val_chain_normal_mse（效果）
Step 3（如果改善）：跑 full-val + Path A，得到一个"看得过去"的结果
Step 4（事后）：回头分析为什么有效 / 为什么这种配置突破了瓶颈
```

**设计哲学**：
- **保护结构层（hop0 D50→D20）**：image_aux 仅降 ÷1.5（0.12→0.08），保留像素域纹理/噪声锚定
- **重点改善细节层（hop3 D4→NORMAL）**：rollout 6× + 末端温和加重
- 医生首先需要正确的解剖结构（hop0），然后才关心纹理细节（hop3）

---

## 0. 决策依据摘要

| 实验 | 结论 | 对 V5 的含义 |
|------|------|-------------|
| Path A 多 ckpt (v3) | ExpoGap=4.80 dB，单调上升 | exposure bias 是核心问题 ✅ |
| SF-pair v3 | schedule bug 导致无效 | 修复后重跑 |
| SF-pair v4 pilot | α≤0.10 微弱改善(5%)，α≥0.15 相变崩溃 | SF-pair 方向暂停 |
| rollout alpha 机制 | GT→混合→pred 三阶段 + 每跳 GT loss | 设计正确，权重不足 |
| 梯度占比 | roll_frac ~10%, img_frac ~85% | rollout 信号被淹没 |

**V5 唯一假设**：加大 rollout_loss 权重，让正确的机制拿到足够的梯度影响力。
**自顶向下版的赌注**：**直接押 1B**——如果方向对，改善应该明显；如果改善不明显，这个方向就不值得继续。

---

## 0.5 风险评估

**最大风险**：rollout 6× 可能改变梯度平衡导致 pair/image 信号被压制。

**为什么温和版风险可控**：
1. **rollout 在 V3 200K 已存在 λ=0.25 并稳定**——6× 是放大已内化的信号，不引入新分布
2. **image_aux 仅降 ÷1.5（0.12→0.08）**——像素域锚定保留大部分强度，保护 hop0 纹理
3. **每步 GT 监督**——rollout 每跳都有 `||z_pred - z_gt||²` 拉力
4. **val_pair_total 相变监控**：+5K 步即检查，早期发现问题

> 完整的 SF vs rollout 5 维机制对比表已移至 §10（事后分析）；B4/B5/B1 验证已移至 §11（论文补强证据，**不阻塞主实验**）。

---

## 1. 实验设计

### 1.1 V5 温和版（主实验，立即启动）

从 200K v3 best.pt（step=86800）resume，50K 新增步。

| 参数 | baseline (200K v3) | V5 | 变化 | 理由 |
|------|-------------------|-----|------|------|
| `rollout.lambda` | 0.25 | **1.5** | 6× | 将 roll_frac 从 ~10% 提升到 ~30% |
| `image_aux.lambda` | 0.12 | **0.08** | ÷1.5 | 温和降低，保护 hop0 像素质量（纹理/噪声锚定） |
| `step_weights` | [1.0, 1.1, 1.2, 1.3] | **[0.8, 1.0, 1.5, 2.5]** | 末端温和加重 | 见 §1.2 |
| `self_forcing_pair` | — | **disabled** | — | SF 方向暂停 |
| 其余参数 | — | **不变** | — | 单轴控制变量 |

### 1.2 step_weights 设计理由

```
                ExpoGap   pair_v_std    val_chain_MSE    step_weight
D50→D20         0.00 dB   0.009634(最大)  0.000243(最大)  0.8（保护结构层）
D20→D10         3.38 dB   0.002946       0.000217         1.0
D10→D4          4.71 dB   0.000775       0.000189         1.5
D4→NORMAL       6.32 dB   0.000141(最小)  0.000175(最小)  2.5（加重细节层）
```

**hop0 保持 0.8 的三个理由**：
1. D50→D20 是结构级恢复（v_std=0.009634，四跳最大），CeilGap=11.07 dB 也最大
2. DINOv2 latent 语义丰富，hop0 恢复的是结构和对比度——如果 hop0 出错，后续不可逆
3. image_aux（像素域唯一锚定）仅降 ÷1.5，配合 step_weight=0.8 共同保护 hop0

**hop3 设为 2.5 的理由**：
- D4→NORMAL ExpoGap=6.32 dB 是四跳最大，是 exposure bias 的重灾区
- 但不用 4.0（激进版），避免末端过拟合风险
- hop3 有效权重 = 1.5 × 2.5 / 5.8 = **0.647**（vs baseline 0.071，提升 9.1×）

### 1.3 有效权重对比表

| Hop | baseline 有效权重 | V5 有效权重 | 提升倍数 |
|-----|------------------|------------|---------|
| hop0 | λ=0.25 × 1.0/4.6 = 0.054 | λ=1.5 × 0.8/5.8 = **0.207** | 3.8× |
| hop1 | λ=0.25 × 1.1/4.6 = 0.060 | λ=1.5 × 1.0/5.8 = **0.259** | 4.3× |
| hop2 | λ=0.25 × 1.2/4.6 = 0.065 | λ=1.5 × 1.5/5.8 = **0.388** | 6.0× |
| hop3 | λ=0.25 × 1.3/4.6 = 0.071 | λ=1.5 × 2.5/5.8 = **0.647** | 9.1× |
| image_aux | 0.12 | **0.08** | ÷1.5 |

---

## 2. Go/No-Go 监控（极简版：1B 直跑）

### 2.1 相变早期警报（最高优先，0 成本）

| 检查点 | val_pair_total 阈值 | 动作 |
|--------|---------------------|------|
| +5K | > 0.00005 | ⛔ **立即停 1B → 启动 1A 兜底** |
| +10K | > 0.00003 | ⚠️ 关注，+15K 再看 |
| +25K | > 0.00002 | ⛔ 止损 |

**理由**：V4 SF pilot 中 val_pair_total 从 0→0.000196 是相变的最早信号（比 val_select 提前 1-2 个 val 点）。1B 因为更激进，相变风险也更高，监控不能省。

### 2.2 效果监控（自顶向下版，看肉眼可见的改善）

**核心问题**：+10K~+25K 步时，val_chain_normal_mse 是否有**肉眼可分辨**的改善？

| 检查点 | val_chain_normal_mse | 解读 |
|--------|---------------------|------|
| +10K | < baseline × 0.90（改善 ≥ 10%）| ✅ 方向对，继续跑 |
| +10K | baseline × 0.95~1.00（改善 ≤ 5%）| 🟡 可能在 noise 内，继续观察到 +25K |
| +10K | > baseline × 1.05 | ⛔ 止损，启动 1A 兜底 |
| +25K | < baseline × 0.90 | ✅ 跑完 50K，full-val + Path A |
| +25K | baseline × 0.95~1.05 | ❌ **方向不成立**——梯度强度不是瓶颈，进入 §11 备选 |
| +50K | — | full-val eval + Path A redo on best.pt |

**为什么用 10% 而非 5% 作为成功线**：
- V4 SF best 也有 5%，但那是临界点偶然——5% 在 val 噪声边缘
- 10% 才能脱离 noise，**肉眼可见**的改善
- 如果 1B 跑到 +25K 还连 5% 都达不到，就是方向错了，不要再投资

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

如果 1B 50K 步后**连最低门槛都达不到**，说明 rollout 加权方向不成立，进入 §11 备选方向。

---

## 4. Claims Matrix（自顶向下版）

只有跑出"看得过去"的结果之后才回头分析。Claims 按结果强度分级：

| 1B 结果 | 允许的 claim |
|------|-------------|
| ✅ val_chain_normal_mse 改善 ≥ 10% **且** ExpoGap ↓ | "加权 rollout 直接对抗了 cascade exposure bias，方向被证实" |
| ✅ 改善 ≥ 10% 但 ExpoGap 不变 | "权重提升改善了 chain 整体精度，机制留给 §10 事后分析" |
| 🟡 改善 5-10% | "信号微弱，可能在 noise 边缘，跑 1A 兜底版做对照确认是真信号" |
| ❌ 改善 < 5% 或持平 | "梯度强度不是当前瓶颈——方向证伪。进入 §11 备选" |
| ❌ < baseline | "rollout 过强压制了 pair/image，启动 1A 兜底" |
| ⛔ 触发 val_pair_total 相变 | "rollout 在当前 backbone 容量下也会触发盆地切换；启动 1A 兜底，同时把方向 4（EMA teacher）的优先级提升" |

### 4.1 失败回退路径（简化版）

| 1B 失败模式 | 下一步 |
|------------|--------|
| **+5K 相变** | 立刻停 1B → 启动 1A 兜底 → 如果 1A 也相变则跳方向 4 |
| **+10K 性能涨 > 5%** | 停 1B → 启动 1A 兜底 |
| **+25K 改善 < 5%** | 不停，跑到 50K 看 → 50K 仍 < 5% → 方向证伪，§11 |
| **hop0 PSNR 退化 > 0.5 dB** | 停 1B → 启动 1A'（image_aux=0.08 而非 0.04）|
| **梯度 NaN / loss 爆炸** | 立刻停 → step_weights 改为 [0.8, 1.0, 1.5, 2.5]（更平）|

---

## 5. 时间线（GPU 数量充足）

**资源说明**：GPU 数量不是约束，**单卡显存才是约束**。所有实验保持 V3 同样的 batch / image size。

### GPU 分配（2 卡主线 + 1 卡备用）

| GPU | 任务 |
|-----|------|
| GPU-0 | **V5-1B 主实验**（rollout λ=2.0, 50K steps, from step 86800） |
| GPU-1 | **Path A redo on step 86800 ckpt**（一次性，~2h，得 ExpoGap_86800）→ 完成后**待命**接管 1A 兜底 |
| GPU-2 | **200K v3 继续**（如未完成）→ 完成后待命 |

### 时间线

| Day | GPU-0 | GPU-1 | 产出 |
|-----|-------|-------|------|
| 0 AM | 准备 1B config + launch script | Path A redo on 86800 ckpt | ExpoGap_86800 baseline |
| 0 PM | **V5-1B 启动** | 待命 | 实验运行 |
| 1 AM | +5K 相变检查 | 视情况接管 1A | 第一关 go/no-go |
| 1 PM | +10K 效果检查 | — | **关键决策点：是否进入备选方向** |
| 2 | +25K 效果检查 | — | 第二关 go/no-go |
| 3 | +50K 完成 + full-val | Path A redo on 1B best | 最终结果 |
| 4 | 事后分析（§10）/ 备选方向决策 | — | 决策报告 |

---

## 6. 代码改动

| 文件 | 改动 | 状态 |
|------|------|------|
| config: `pet_flow_first_hop_224_v5_rollout_heavy.yaml` | 温和版 config | ✅ 已创建 |
| `scripts/launch_v5_rollout_heavy.sh` | 自动读 ckpt step + 设 max_steps + 启动 | ✅ 已创建 |
| `train_first_hop.py` | **0 行改动** | ✅ 无需修改 |

§11 备选方向涉及的 config / 工具文件等到主实验跑完再看是否需要。

---

## 7. 风险与缓解（极简版）

| 风险 | 缓解 |
|------|------|
| 1B 触发相变（val_pair_total 跳变） | +5K 步即检查，立即停跳 1A 兜底 |
| hop0 像素退化（image_aux ÷3 太狠） | hop0 PSNR 独立监控，回退到 1A'（image_aux=0.08）|
| pair_loss 被压死（roll_frac > 60%） | pair_frac ≥ 2% 底线，低于即跑 1A 兜底 |
| step_weights 末端 4× 太陡 → 末端 hop 过拟合 | val_chain_d20_mse 独立监控；触发回退到 [0.8, 1.0, 1.5, 2.5] |
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
7. **不做无用功的对照实验阻塞主实验**：B4/B5/B1/方向 5 都是论文补强证据，不阻塞 1B 启动

---

## 9. 显存约束下的并行策略

由于**单卡显存有限**，所有实验保持 V3 同样配置：
- 不增大 batch size
- 不引入更大的 backbone（DiT-S 保持，DiT-B/L 暂不考虑）
- 不开 gradient checkpointing 之外的额外内存优化（避免引入未知数）

利用 GPU 数量充足的优势：
- **3 卡并行**（§5）：1A 主实验 + 方向 5 对照 + 待命回退卡
- 任何回退实验立即在待命卡上启动，不打断主实验
- B5 参数 L2 测量、Path A redo 这类一次性任务在 GPU-1 完成，不占主实验

---

## 10. 事后分析框架（仅在 1B 跑出"看得过去"的结果后启动）

> 自顶向下原则：**先有结果，再分析机制**。如果 1B 50K 步后没拿到 ≥ 5% 改善，下面所有事后分析都不需要做——因为方向已被证伪。

### 10.1 如果 1B 成功（≥ 10% 改善）

需要回答：
1. **改善来自哪里**：roll_frac 占比变化、step_weights 末端加重、image_aux 降低，三者各自贡献多少？
   - 做法：跑 ablation 系列（1B-no-stepweights / 1B-image_aux=0.08 / 1B-rollout=1.0）
2. **是否 ExpoGap 直接缓解**：跑 Path A on 1B best.pt，对比 ExpoGap_86800
3. **机制论证**：补完 SF vs rollout 5 维对比表（见 §10.4）作为论文章节
4. **可推广性**：在更大 scale（如 224→256）或更深 chain（4→6 hops）上是否仍成立

### 10.2 如果 1B 性能持平（梯度强度不是瓶颈）

需要回答：
1. **真正瓶颈在哪**：是 backbone 容量？数据多样性？loss 形式？
2. **进入 §11 备选方向决策**：方向 4（EMA teacher）/ 方向 5（结构改动）/ 方向 6（数据增强）
3. 为论文留下 negative result：rollout 加权方向已被实验证伪，节省同行重复

### 10.3 如果 1B 触发相变

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
| B4 | 核对 SF + rollout 的 stop-grad 状态（5 min 代码审查） | 1B +10K 检查空隙 |
| B5 | 算 V3 正常训练相邻 4K 步参数 L2 距离 → 给"正常移动幅度"baseline | 1B +25K 检查空隙 |
| B1 | 算 baseline val 自然波动 std → 校准 5%/10% 阈值 | 1B +5K 后顺手做 |

这些任务**不阻塞主实验**，只是用空闲时间补充诊断证据，方便事后写论文或做决策。

---

## 11. 备选方向（仅在 1B 失败时考虑）

### 11.1 方向 4：EMA Teacher Backbone（如 1B 触发相变）

**触发条件**：1B 触发 val_pair_total 相变 + B5 确认参数移动是同款机制。

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

### 11.2 方向 5：SF 极保守对照（如 1B 持平且想保留 SF 选项）

**触发条件**：1B 持平（不是相变，是真没改善）+ 想确认 SF 是否还有未来。

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

### 11.3 方向 6：监督形式改动（如 1B 持平且想换思路）

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
