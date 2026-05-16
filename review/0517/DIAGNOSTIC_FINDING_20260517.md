# Diagnostic finding — image_aux exhausted, decoder is the ceiling (20260517)

- generated_at: 2026-05-17 Asia/Shanghai
- branch: foc_lite_hop0
- commit_when_generated: a7d476c
- triggered by: three independent Round 3 peer reviews (agent1/agent2/agent3) all flagging diagnostic laundering as 4th confirmation bias; agent3 specifically pointing out the freeze_rae hard assertion + the 11.2 dB transport gap untouched.

> **本文件是"M1/M2/M3 + freeze_rae assertion 验证"的直接执行结果。它替代了我们计划中的"5 个 P0 + Round 4 prompt"，因为 agent3 在 Round 3 里指出，"再多设计实验" 本身就是第 4 次 bias。**

---

## §1 — V7 image_aux 路径已经饱和（M1 数据）

从 [review/0511/log_snapshots_20260516_163900/main_training/V7_train_20260516_163900.log](../0511/log_snapshots_20260516_163900/main_training/V7_train_20260516_163900.log) `[val_full]` 行 grep 出 32 个 full-val 评估点，每个含 `val_hop0_img_total`：

| 窗口 | mean `val_hop0_img_total` | n |
|---|---:|---:|
| step 5K (V7 起点) | 0.007965 | 1 |
| step 30K-60K (早期) | 0.007719 | 7 |
| step 100K-130K | 0.007482 | 6 |
| step 130K-160K (末期) | 0.007507 | 7 |

**末期 / 早期 = 0.7507 / 0.7719 = 1.028**。**最后 60% 训练时间里 image_aux full-val loss 只改善 2.7%**。这已经是平台。任何 V11/V11'/V17/V17' 在同一 loss 形状上的改动，**期望增益的上限不会超过这 2.7% 还没榨完的部分**。

---

## §2 — pixel forcing path 是活的，private head 反而被压制（M2/M2b 数据）

同样从 V7 训练 log `[train]` 行：

| 参数 | 初值 (step 0) | step 50K | step 100K | step 160K | 末期/初值 |
|---|---:|---:|---:|---:|---:|
| `gate_pix_value` | 0.020 | 0.029 | 0.036 | **0.0363** | **1.82×（长大 82%）** |
| `lambda_hop[0]` | 0.100 | 0.090 | 0.062 | **0.0565** | **0.57×（收缩 43%）** |

**含义**：

1. `gate_pix` 长大 82% → 模型在**主动使用** pixel forcing 通道（D50 image → latent injection）。所以 image_aux 路径**不是"未激活"**，它是**已激活并已收敛**。
2. `lambda_hop[0]` 收缩 43% → 模型在**主动压制** hop0 private residual head 的输出。这直接**反向证伪** agent3 在 Round 2 提的 V15 假设（"image_aux 信号主要进 private head，所以 multi-hop image_aux 能 unblock 每个 hop 的私有 head"）。事实是模型不要 hop0 private head 长大，只要 backbone + pixel encoder 那部分。

---

## §3 — freeze_rae 硬 assertion 验证（agent3 关键发现）

[train_first_hop.py:1298-1302](../../train_first_hop.py#L1298)：

```python
if not bool(train_cfg_boot.get("freeze_rae", True)):
    raise RuntimeError(
        "docs/main.md requires training.freeze_rae=true for first-hop training. "
        "Decoder unfreezing is not allowed in this trainer."
    )
```

[pet_lr/model_first_hop.py:432-436](../../pet_lr/model_first_hop.py#L432) 还有 `assert_decoder_frozen()` 在训练循环中 enforce。

**结论**：**当前代码物理上禁止 RAE decoder finetune**。撤掉这条 assertion 是任何 decoder-side 实验的前置 1 行代码。

---

## §4 — V7 chain PSNR 距 decoder ceiling 还有 11.2 dB

从 [PLANF_FULLVAL_PSNR_CLIP3_REPORT_20260516_173941.md](../0511/fullval_psnr_clip3_20260516_173941/PLANF_FULLVAL_PSNR_CLIP3_REPORT_20260516_173941.md) 与 [docs/experiment_plans/PLAN_V5_PostE1E2_SchemeCA_20260417.md](../../docs/experiment_plans/PLAN_V5_PostE1E2_SchemeCA_20260417.md)：

| | 值 |
|---|---:|
| V7 chain D20 PSNR_clip3 (full-val n=7403) | 35.435 dB |
| RAE decoder ceiling on D20 | ~46.64 dB |
| **transport gap** | **~11.2 dB** |

V11/V11'/V17/V17' 整族实验在 0.05-0.30 dB 区间打转，**累加起来也吃不到这 11.2 dB 的 10%**。**真正的瓶颈在 decoder 端**，被 §3 的硬 assertion 锁住。

---

## §5 — 6 天 review-meta-review 的客观时间线

| 日期 | 事件 | 新 GPU 训练 launch |
|---|---|---|
| 5/11 | V7/V8/V6_NOISE 完成 160K | — |
| 5/16 | Round 1 review，撤回 V11 → V11' | **0** |
| 5/16 晚 | 推 V13/V14/V15/V16 + §0/§1/§2 disambig | **0** |
| 5/17 上午 | §0/§1/§2 完成，推 V17 | **0** |
| 5/17 下午 | Round 2 整合，撤回 V17 → 5 P0 + Round 3 | **0** |
| 5/17 晚 | Round 3 三位独立 reviewer 都说 "stop reviewing, launch" | **0** |

**6 天 / 3 轮 review / 4 个 candidate / 5 个 P0 / 0 次新 GPU launch**。三位 reviewer 共识：再设计 Round 4 / 再设计第 6 个 P0，**就是第 4 次 confirmation bias 的实现**（用诊断取代决策）。

---

## §6 — 整合后的下一步：撤回 image_aux 围墙花园，攻 decoder 端

### 6.1 撤回的实验

| 实验 | 之前优先级 | 现在 |
|---|---|---|
| V11 / V15 | 已撤回 | 维持撤回（§2 数据反向证伪 private head 假设）|
| V11' / V17 / V17' | 我之前推 | **撤回**。期望增益 < 2.7% × 已饱和路径 残余 = noise floor 量级 |
| V13 (true image_aux ablation) | 待 launch | **撤回**。§1 数据已经显示 image_aux full-val loss 在最后 60% 只改 2.7%；V13 跑出来 ΔPSNR 期望 < 0.05 dB，没诊断价值了 |
| V9 / V9a | 待 launch | **降级**。V9a 仍是廉价 sanity（如果 V7 中间 ckpt 还在服务器）|
| V14 (true d_pure) | 待 launch | **保留**。任何后续实验都需要它做 SNR 标尺 |

### 6.2 新的实验主轴：V18 = RAE decoder LoRA partial unfreeze

agent1 提出 + agent2 + agent3 共同确认。具体配方（三位 reviewer 整合）：

1. **Phase A — RAE inference-time sanity（半天，0 训练）**
   - 取 V7 `best.pt`
   - 把 RAE decoder 设为 train mode（启用 dropout / BN drift 如果有）但**不更新参数**
   - 跑 full-val PSNR_clip3
   - 判定：PSNR 几乎不变 → RAE 是 deterministic operator，安全解冻；PSNR 大幅下降 → RAE 训练时有 dropout/BN，需先稳定
2. **Phase B — V18 LoRA finetune（160K 训练，~7 天）**
   - 撤掉 [train_first_hop.py:1298](../../train_first_hop.py#L1298) 的硬 assert
   - 仅解冻 RAE decoder 最后 1-2 transformer block 的 LoRA-r8
   - decoder LR = 5e-7（比 backbone LR 8e-5 低 100×）
   - 加 KL pull-back 防 decoder 漂走：`(decode(z_GT) - decode_pretrained_frozen(z_GT))^2 × λ_kl`
   - 其余 V7 配置不变
3. **Phase C — 评估**
   - 主指标：V18 full-val NORMAL chain PSNR_clip3 (V18 - V7)
   - 预期：≥ +0.5 dB（保守，1/22 of 11.2 dB gap）；≥ +1.0 dB 才算确认方向

### 6.3 V18 vs 之前所有候选的 EV 对比

| 实验 | 期望 ΔNORMAL PSNR | 成本 | EV/cost |
|---|---:|---:|---:|
| V11 / V11' / V17 / V17' / V15 | 0.02 - 0.10 dB（已饱和路径残余）| 7 天 × 1-3 个 | 极低 |
| V9 (β=2.5) | 0 - 0.05 dB (selector 偏置)| 7 天 | 极低 |
| V13 (true image_aux ablation) | 信息价值，非 PSNR | 7 天 | 中（仅 backward-looking）|
| V14 (true d_pure) | 0（design 是测 noise）| 7 天 | 中（前置条件）|
| **V18 (decoder LoRA)** | **0.5 - 2.0 dB** | **7 天 + 0.5 天 Phase A** | **HIGH** |

---

## §7 — Hard timebox（按三位 reviewer 共识）

| Day | 动作 |
|---|---|
| **Day 0（今天剩余时间）** | (a) 服务器 grep 检查 V7 中间 ckpt 存活 `ls /data_2/.../V7/run/step_*.pt`；(b) Phase A inference-time sanity probe（半天）|
| **Day 1** | 起草 V18 yaml + code patch（撤 assert + LoRA wrapping）+ KL pull-back loss；起草 V14 launch（并行）|
| **Day 2** | V18 训练 launch（GPU 0）+ V14 训练 launch（GPU 1）+ V9a 如果 ckpt 存活则当天结案（CPU 重算）|
| **Day 3+** | 训练跑，不允许再发 Round 4 prompt |

**Day 2 必须有 V18 在 GPU 上跑**。如果 Day 1 起草遇到代码改造障碍超过 1 天，撤回 LoRA-r8 改为更简单的"撤 assert + decoder full unfreeze + LR=5e-8"。**不再设计第 4 轮**。

---

## §8 — 给团队的诚实总结

V11 / V11' / V17 / V17' / V15 这一族我推了 3 轮、撤回 2 轮。三位独立 reviewer 在 Round 3 共识：**这些都是在已饱和路径上微调，不论选哪个都 < d_pure 量级**。**真正的杠杆是 11.2 dB transport gap 那个 RAE decoder ceiling**，而那个 ceiling 被项目自己 6 天前就该看到的一行硬 assertion 锁住。

**我作为 claude 的 meta-错**：3 轮 review 都没问 "RAE decoder 能动吗"。它是默认假设，从未挑战。**这本身就是 confirmation bias 的另一种形式**：把项目历史决定当公理，从来不查它的依据。

**M1/M2/M3 + freeze_rae 验证总共耗时 < 1 小时**，且不需要任何 GPU。它在 5/11（V7 训练完成）那天就能跑。**整整 6 天的 review 链是可以用这 1 小时取代的**。
