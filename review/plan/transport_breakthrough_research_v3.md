# Transport 突破实验计划 v3 — 2026-04-25

**Supersedes**: v2 (transport_breakthrough_research_v2.md)  
**Revision**: 2026-04-25 PM — applied review feedback (per-sample gather bug,
control-variable tightening, multi-checkpoint Path A, visual eval gate, grad-norm logging).  
**v3 原则**: 并行启动，不等 200K；基于可视化证据直接行动。

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

**多 ckpt 探针（v3 修订）**：单点 ckpt 容易把"训练阶段"特异的曲线
当成结论。改为对 **3 个 ckpt** 跑相同的 Path A 诊断，观察 gap 随训练进度
的演化趋势：

```bash
# GPU-0:
SCHEME_C_DIR=/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_imgaux_boost_gpu2_nopbar_20260418_010035

for CKPT in step_10000.pt step_30000.pt best.pt; do
    CUDA_VISIBLE_DEVICES=0 python -u scripts/diagnose_tf_rollout_gap.py \
        --config configs/pet_flow/pet_flow_first_hop_224_50k_imgaux_boost.yaml \
        --checkpoint "${SCHEME_C_DIR}/${CKPT}" \
        --split val --max-slices 0 --batch-size 8 --device cuda:0 \
        --out-dir "/data_2/qujiaxiang/outputs/PET_LatentResidual/diagnostics/pathA_C_${CKPT%.pt}"
done
```

**4-6h 内出全部结果**。trend 决定后续方向：
- gap 单调上升（exposure 累积）→ SF-pair 必要
- gap 在中段 saturate →  velocity capacity 已经吃满，不仅是 exposure
- gap 不显著（< 0.3 dB）→ 不是 exposure；走 capacity / 数据 / 解码器路线

### 实验 B: 50K Self-Forcing 快速验证（2 GPU-day）

**不等 200K 完成**——直接从旧 Scheme C best.pt resume，跑 50K 步 Self-Forcing pair_loss。

**理由**：
- 旧 Scheme C best.pt 已经在 GT 分布上训练了 50K 步 = Phase 1 已完成
- 直接做 Phase 2（SF pair_loss）= 验证 Self-Forcing 是否能打破 plateau
- 50K 步 SF 比等 200K→resume 100K SF 快 **3 倍**

**v3 修订：控制变量收紧**。和 Rollout-Up 控制实验**只差一个轴**：

| 维度 | SF-pair (B) | Rollout-Up (C) |
|------|-------------|-----------------|
| `self_forcing_pair.enabled` | **true** | false |
| `rollout.lambda_start/end` | 0.25 (与 Scheme C 末态一致) | **1.00 (4×)** |
| `rollout.alpha_start/end` | 1.00 | 1.00 |
| `image_aux.lambda` | 0.12 (与 Scheme C 末态一致) | 0.12 (相同) |
| `optimizer.lr` | 4e-5 (resume LR，相同) | 4e-5 (相同) |
| `lr_schedule.warmup_ratio` | 0.05 (吸收 resume 抖动) | 0.05 (相同) |
| `seed` | 42 | 42 |
| 起点 ckpt | 旧 Scheme C best.pt | 旧 Scheme C best.pt |

**关键**：B 不动 rollout 权重；C 不开 SF。差距完全归因到所测维度。

```bash
# GPU-2 (SF-pair):
CUDA_VISIBLE_DEVICES=2 python -u train_first_hop.py \
    --config configs/pet_flow/pet_flow_first_hop_224_50k_selfforcing_from_C.yaml \
    --resume /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_imgaux_boost_gpu2_nopbar_20260418_010035/best.pt
```

### 实验 C: Rollout-Up 对照（2 GPU-day）

```bash
# GPU-3 (Rollout-Up):
CUDA_VISIBLE_DEVICES=3 python -u train_first_hop.py \
    --config configs/pet_flow/pet_flow_first_hop_224_50k_rollout_up_from_C.yaml \
    --resume /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_imgaux_boost_gpu2_nopbar_20260418_010035/best.pt
```

### 3 个实验的关系

```
         GPU-0: Path A (4-6h, 3 ckpt)
         GPU-1: 200K v3 (继续)
         GPU-2: SF-pair (50K, 16h)
         GPU-3: Rollout-Up (50K, 16h)
              ↓ Day 2-3
         全部完成 → Full-val eval + 视觉对比
              ↓
SF-pair vs Rollout-Up：
  SF >> Rollout-Up → 方法贡献成立（exposure bias 是主因）
  SF ≈ Rollout-Up → 只需加 rollout 权重，不需要 SF
  SF ≈ baseline   → exposure bias 不是主因 → 走 capacity/数据/decoder
```

---

## 3. 代码实现需求（v3 修订版）

### 修复点：v2 草案的 per-sample gather bug

v2 中的草案：

```python
# ❌ BUG：z_chain[h.item()] 是 [B, C, H, W]（所有样本第 h 跳的预测），
#       而不是"第 i 号样本第 hop_idx[i] 跳的预测"。
z_src_pred = torch.stack([z_chain[h.item()] for h in hop_idx])
```

正确做法是先把 chain stack 成 `[B, T, C, H, W]`，再用 `torch.gather` 沿
chain 维做 per-sample 抽取：

```python
z_chain_stack = torch.stack(z_chain, dim=1)             # [B, T, C, H, W]
idx = hop_idx.view(-1, 1, 1, 1, 1).expand(
    -1, 1, *z_chain_stack.shape[2:]
)
z_src_pred = z_chain_stack.gather(1, idx).squeeze(1)    # [B, C, H, W]
```

### 实现位置

新函数 `compute_self_forcing_z_src(model, main_batch, cfg, step, rollout_times)`，
放在 `compute_rollout_losses` 与 `compute_foc_losses` 之间。

主训练循环中（`predict_latent_step` 之前）：

```python
sf_info = compute_self_forcing_z_src(model, main_batch, cfg, step, rollout_times)
if sf_info is not None:
    main_batch["z_src"] = sf_info["z_src_sf"]   # 仅替换值，pair loss 代码完全不变
```

### 设计要点

1. **整段 pre-rollout 在 `@torch.no_grad()` 下执行**——SF 只改输入分布，
   梯度仍来自单次 main `predict_latent_step`。
2. **per-sample gather**（如上）——避免把"all batch at hop k"误当成
   "sample i at hop_idx[i]"。
3. **alpha_sf 调度**：`warmup=5000` 步保持 GT；`5000→15000` 线性 0→1；
   之后保持 1.0（纯 SF）。
4. **诊断字段**：`sf_alpha / sf_gap_norm / sf_z_pred_norm / sf_z_gt_norm`
   写入 metrics_jsonl，用于事后核对 SF 是否真的接管。
5. **`x_src_img` 处理**：pre-rollout 中只在 `h==0` 传 `x_rollout_first`
   （pixel forcing 仅 hop0 生效，与现有 `rollout_multistep_losses_first_hop` 保持一致）。
6. **`log_grad_norms: true`**（两个 config 都启用）——SF vs Rollout-Up 不仅
   比 loss/PSNR，还比 backbone gradient norm，区分"梯度真的变了"还是
   "只是 loss 数值变了"。

---

## 4. 时间线

| Day | GPU-0 | GPU-1 | GPU-2 | GPU-3 |
|-----|-------|-------|-------|-------|
| 1 AM | **Path A × 3 ckpt** | 200K (~60K) | — | — |
| 1 PM | Path A 出图 | 200K (~65K) | **SF-pair 启动** | **Rollout-Up 启动** |
| 2 | 分析 trend | 200K (~75K) | SF-pair (~25K) | Rollout-Up (~25K) |
| 3 | Full-val eval B/C | 200K (~90K) | SF 完成 | RU 完成 |
| 3 PM | **视觉对比 gate** | 200K (~95K) | — | — |
| 4 | 结果分析 | 200K (~100K) | — | — |
| 5 | 决策 | 200K 继续或 resume | Phase 2 长跑(可选) | — |

**Day 3 就能看到 SF-pair vs Rollout-Up 的初步对比**。不需要等 200K 完成。

### Day 3 视觉对比 gate（v3 新增）

仅 PSNR 数字差不能定结论——必须**配合视觉**。Day 3 用同 8 张 val slice
（Day 0 选定的诊断 set），出 `pred_chain × {Scheme C best, B-50K, C-50K}`
的 9-列网格图：

| 列 | 内容 |
|----|------|
| 1-4 | GT D50/D20/D10/D4 |
| 5 | GT NORMAL |
| 6 | Scheme C best chain final |
| 7 | SF-pair (B) chain final |
| 8 | Rollout-Up (C) chain final |
| 9 | per-pixel \|err\| heatmap of B vs C |

**Gate 规则**：如果 PSNR 上 B > C 但视觉看不出差别（artifacts pattern 一致）
→ 不能写为方法贡献，需要重新审视。

---

## 5. 成功判据（可证伪预测，v3 收紧）

| 预测 | 阈值 | 如果不满足 |
|------|------|----------|
| SF-pair transport_avg > Scheme C best (36.206) | Δ > 0.3 dB | SF 无效 |
| **SF-pair > Rollout-Up（chain_normal_psnr）** | **Δ > 0.3 dB** | "pair 主梯度"假设错误 |
| **SF-pair backbone grad_norm 显著偏离 Rollout-Up** | 相对偏差 > 20% | 梯度并未真的换分布 |
| Path A exposure_gap (hops 1-3 mean，best.pt) | > 0.3 dB | 确认 exposure bias 存在 |
| Path A ceiling_gap (hops 1-3 mean，best.pt) | > 5 dB | velocity capacity 也是问题 |
| Path A gap 单调 ↑（10K → 30K → best） | 单调 | 训练中后期才暴露的问题，不是早期 |

---

## 6. 风险缓解

| 风险 | 缓解 |
|------|------|
| SF 代码 bug 导致 loss 爆炸 | alpha_sf 前 5K 步 = 0（纯 GT），渐进过渡；`grad_finite_check: true` 自动捕获 NaN |
| per-sample gather 错位 | 使用 `torch.gather` + 形状 assert（已实现） |
| resume 不兼容 | `strict_resume_compat: false` + `resume_allow_*: true`（B/C 都已配置） |
| SF 50K 步不够 | 如果趋势好但未收敛 → resume 到 100K |
| Rollout-Up λ=1.0 训练不稳定 | 监控 loss，如果爆炸降至 λ=0.5；`loss_balance_dominance_threshold: 0.95` 仅警告不强制 |
| 显存增加（pre-rollout） | no_grad 下 4 次 forward，预计 +15% 显存；batch=8 在 80G 卡仍有余量 |
| PSNR 提升但视觉无改善 | Day 3 视觉 gate（§4） |
| 单 ckpt Path A 偏差 | 多 ckpt 趋势分析（§2 实验 A） |

