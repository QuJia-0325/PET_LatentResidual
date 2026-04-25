# Transport 突破实验计划 v3 — 2026-04-25

**Supersedes**: v2 (transport_breakthrough_research_v2.md)  
**v3 原则**: 并行启动，不等 200K；基于可视化证据直接行动

---

## 1. 当前状态

| 资源 | 状态 |
|------|------|
| GPU-1 | 200K v3 训练中（~58K/200K，best at step=58400） |
| GPU-0/2/3 | **空闲可用** |
| 200K 预计完成 | ~3 天后 |
| 可视化确认 | 级联误差是核心（D50→NORMAL vs D4→NORMAL 差 5-7 dB） |

### 关键洞察（v2 §2 确认）

pair_loss 占 backbone 梯度 **≈ 88%**，但永远在 GT 分布上训练。rollout_loss 只占 ~12%。即使 rollout alpha=1（纯预测输入），backbone 参数优化仍被 pair_loss 的 GT 分布主导。

---

## 2. 不等 200K 的 3 个并行实验

### 实验 A: Path A 诊断（最高优先，0.5 GPU-day）

**目的**：量化 exposure bias vs velocity capacity 的比例。

**可以立即启动**——使用旧 Scheme C 的 best.pt（已存在）。

```bash
# GPU-0:
CUDA_VISIBLE_DEVICES=0 python -u scripts/diagnose_tf_rollout_gap.py \
    --config configs/pet_flow/pet_flow_first_hop_224_50k_imgaux_boost.yaml \
    --checkpoint /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_imgaux_boost_gpu2_nopbar_20260418_010035/best.pt \
    --split val --max-slices 0 --batch-size 8 --device cuda:0 \
    --out-dir /data_2/qujiaxiang/outputs/PET_LatentResidual/diagnostics/pathA_C
```

**4h 内出结果**。verdict 决定后续方向。

### 实验 B: 50K Self-Forcing 快速验证（2 GPU-day）

**不等 200K 完成**——直接从旧 Scheme C best.pt resume，跑 50K 步 Self-Forcing pair_loss。

**理由**：
- 旧 Scheme C best.pt 已经在 GT 分布上训练了 50K 步 = Phase 1 已完成
- 直接做 Phase 2（SF pair_loss）= 验证 Self-Forcing 是否能打破 plateau
- 50K 步 SF 比等 200K→resume 100K SF 快 **3 倍**

**需要先实现代码**（~25 行），然后创建 config：

```yaml
# pet_flow_first_hop_224_50k_selfforcing_from_C.yaml
run_name: first_hop_224_50k_selfforcing_from_C
max_steps: 50000
training:
  self_forcing_pair:
    enabled: true
    alpha_sf_start: 0.0
    alpha_sf_end: 1.0
    warmup_steps: 5000    # 前 5K 步不用 SF（让 optimizer state 适应）
    ramp_steps: 10000     # 5K-15K 渐进从 GT 到 pred z_src
  rollout:
    alpha_start: 1.0      # rollout 固定纯预测（resume 后不需要 ramp）
    alpha_end: 1.0
# --resume 旧 Scheme C best.pt
```

```bash
# GPU-2:
CUDA_VISIBLE_DEVICES=2 python -u train_first_hop.py \
    --config configs/pet_flow/pet_flow_first_hop_224_50k_selfforcing_from_C.yaml \
    --resume /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_imgaux_boost_gpu2_nopbar_20260418_010035/best.pt
```

### 实验 C: Rollout-Up 对照（2 GPU-day）

**消融对照**：只把 rollout loss 权重大幅提升（不做 Self-Forcing），验证是否仅靠加大 rollout 就够。

```yaml
# pet_flow_first_hop_224_50k_rollout_up_from_C.yaml
run_name: first_hop_224_50k_rollout_up_from_C
max_steps: 50000
training:
  rollout:
    alpha_start: 1.0
    alpha_end: 1.0
    lambda_start: 1.0     # rollout 权重从 0.25 提升到 1.0（4×）
    lambda_end: 1.0
  # pair_loss 不变（GT 分布）
# --resume 旧 Scheme C best.pt
```

```bash
# GPU-3:
CUDA_VISIBLE_DEVICES=3 python -u train_first_hop.py \
    --config configs/pet_flow/pet_flow_first_hop_224_50k_rollout_up_from_C.yaml \
    --resume /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_imgaux_boost_gpu2_nopbar_20260418_010035/best.pt
```

### 3 个实验的关系

```
         GPU-0: Path A (4h)
         GPU-1: 200K v3 (继续)
         GPU-2: SF-pair (50K, 16h)
         GPU-3: Rollout-Up (50K, 16h)
              ↓ Day 2
         全部完成 → Full-val eval
              ↓
SF-pair vs Rollout-Up：
  SF >> Rollout-Up → 方法贡献成立
  SF ≈ Rollout-Up → 只需加 rollout 权重，不需要 SF
  SF ≈ baseline   → exposure bias 不是主因
```

---

## 3. 代码实现需求

### Self-Forcing Pair Loss（~25 行）

在 `train_first_hop.py` 的训练循环中，`compute_pair_losses` 调用前插入：

```python
# Self-Forcing pair loss: replace pair z_src with predicted z_src
sf_cfg = cfg["training"].get("self_forcing_pair", {})
sf_enabled = bool(sf_cfg.get("enabled", False))
if sf_enabled:
    alpha_sf = get_linear_schedule_value(
        global_step=step,
        warmup_steps=int(sf_cfg.get("warmup_steps", 5000)),
        ramp_steps=int(sf_cfg.get("ramp_steps", 10000)),
        start=float(sf_cfg.get("alpha_sf_start", 0.0)),
        end=float(sf_cfg.get("alpha_sf_end", 1.0)),
    )
    if alpha_sf > 0 and "z_rollout" in main_batch:
        with torch.no_grad():
            z_chain = [main_batch["z_rollout"][:, 0]]
            for h in range(model.num_hops):
                t_s = torch.full((z_chain[-1].shape[0],), rollout_times[h], device=device)
                t_d = torch.full((z_chain[-1].shape[0],), rollout_times[h+1], device=device)
                hop_t = torch.full((z_chain[-1].shape[0],), h, device=device, dtype=torch.long)
                x_img = main_batch.get("x_rollout_first") if h == 0 else None
                out_sf = model.predict_latent_step(z_chain[-1], t_s, t_d, hop_t, x_img)
                z_chain.append(out_sf["z_pred"].detach())
        hop_idx = main_batch["hop_idx"].long()
        z_src_pred = torch.stack([z_chain[h.item()] for h in hop_idx])
        main_batch["z_src"] = (1 - alpha_sf) * main_batch["z_src"] + alpha_sf * z_src_pred
```

**注意**：这段代码修改的是 `main_batch["z_src"]` 的值，`compute_pair_losses` 的代码**完全不需要修改**——因为它读取 `batch["z_src"]` 来计算 `v_target = (z_dst - z_src) / dt / sigma`。

---

## 4. 时间线

| Day | GPU-0 | GPU-1 | GPU-2 | GPU-3 |
|-----|-------|-------|-------|-------|
| 1 AM | **Path A** (4h) | 200K (~60K) | — | — |
| 1 PM | 空闲 | 200K (~65K) | **实现 SF 代码** | — |
| 2 | Path A eval done | 200K (~75K) | **SF-pair 50K** | **Rollout-Up 50K** |
| 3 | Full-val eval | 200K (~90K) | SF 完成 | RU 完成 |
| 4 | 结果分析 | 200K (~100K) | — | — |
| 5 | 决策 | 200K 继续或 resume | Phase 2 长跑(可选) | — |

**Day 3 就能看到 SF-pair vs Rollout-Up 的初步对比**。不需要等 200K 完成。

---

## 5. 成功判据（可证伪预测）

| 预测 | 阈值 | 如果不满足 |
|------|------|----------|
| SF-pair transport_avg > Scheme C best (36.206) | Δ > 0.3 dB | SF 无效 |
| SF-pair > Rollout-Up | Δ > 0.3 dB | "pair 主梯度"假设错误 |
| Path A exposure_gap (hops 1-3 mean) | > 0.3 dB | 确认 exposure bias 存在 |
| Path A ceiling_gap (hops 1-3 mean) | > 5 dB | velocity capacity 也是问题 |

---

## 6. 风险缓解

| 风险 | 缓解 |
|------|------|
| SF 代码 bug 导致 loss 爆炸 | alpha_sf 前 5K 步 = 0（纯 GT），渐进过渡 |
| resume 不兼容 | 使用 `strict_resume_compat: false` + `resume_allow_config_mismatch: true` |
| SF 50K 步不够 | 如果趋势好但未收敛 → resume 到 100K |
| Rollout-Up λ=1.0 训练不稳定 | 监控 loss，如果爆炸降至 λ=0.5 |
| 显存增加（pre-rollout） | no_grad 下 4 次 forward，预计 +15% 显存 |
