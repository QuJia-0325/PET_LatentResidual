# Rebuttal — Codex Audit `f33ddaa` (Reviewer: Mendel / gpt-5.4)

**Date**: 2026-04-22  
**Commit reviewed**: `0b109f8`  
**Response commit**: `a8fd9dd`

---

## P0-1: N2 does not enforce resume-to-source transport identity

**Reviewer finding**: N2 config 声称在 frozen transport 上训练 refiner，但 trainer 不强制 `--resume`。不传 `--resume` 时 refiner 会在随机初始化的 transport 上训练，完全无效。

**Our verdict**: **SUSTAINED — 已修复**

**Fix** (`train_first_hop.py`):  
当 `freeze_all_except_seam_refiner=true` 时，启动阶段检查 `resume_enabled`，不满足则 raise：
```python
if freeze_first_hop and not resume_enabled:
    raise RuntimeError(
        "freeze_all_except_seam_refiner=true requires --resume <transport_checkpoint> ..."
    )
```

---

## P0-2: N1 ablation semantics can be silently mismatched across resume/eval

**Reviewer finding**: `pixel_forcing_disabled` 改变了 forward 语义，但 checkpoint 保存时硬编码 `first_hop_pixel_enabled: True`，resume/eval 时不校验语义一致性。

**Our verdict**: **SUSTAINED — 已修复**

**Fix** (`train_first_hop.py` + `eval_first_hop_224_clip3.py`):

1. **保存**：`first_hop_pixel_enabled` 改为读取 `not model.pixel_forcing_disabled`（原为硬编码 `True`）。
2. **Resume 校验**：加载时比较 `ckpt["first_hop_pixel_enabled"]` 与当前 config 的 `pixel_forcing_disabled`，不一致则 raise。
3. **Eval 校验**：`load_model()` 中同样校验，防止用 N1 checkpoint 配 Scheme C config eval（或反之）。

---

## P1-1: Weight-decay comment and actual optimizer behavior are inconsistent

**Reviewer finding**: 注释说"排除标量 gate 的 decay"，但 `first_hop_weight_decay: 0.0` 实际把所有 first-hop 参数（pixel_encoder conv、hop_residual_head、seam_refiner）的 decay 全部设为 0，不只是标量 gate。

**Our verdict**: **ACKNOWLEDGED — 暂不修复**

**Rationale**:
- 当前 first-hop 参数量较小（pixel_encoder ~52K, hop_residual ~440K），50K 步训练中权重膨胀风险极低
- 拆分 param group（scalar gates vs conv weights vs refiner）增加代码复杂度，收益不明确
- 核心矛盾（weight_decay 压死 gate）已通过 `first_hop_weight_decay: 0.0` 解决
- 如后续训练日志显示 first-hop conv 权重范数异常增长，再精细化 param group 拆分

**Action**: 修正了 config 注释，准确描述当前行为（"exclude first-hop params from L2 decay"而非"exclude scalar gates"）。

---

## P1-2: Best-checkpoint selection still biased by rolling validation window

**Reviewer finding**: `val_window_mode: rolling` 下每次 eval 用不同 val 子集，best checkpoint 比较不公平。

**Our verdict**: **OVERRULED — 误报**

**Evidence**:  
Best checkpoint 选择基于 `val_multi_objective`，该指标由 chain metrics 组成（`val_chain_d20_mse`, `val_chain_d10_mse`, `val_chain_d4_mse`, `val_chain_normal_mse`）。

Chain metrics 的计算路径：
- `eval_chain_on_all_samples: true` — chain 在所有 val 样本上评估
- `eval_chain_unique_slices: true` + `eval_chain_max_unique_slices: 0`（无上限）— 不做截断
- `val_shuffle: false` — 验证集顺序固定

因此 chain PSNR **始终在整个 val 集上计算**，不受 rolling window 影响。Rolling window 只影响 pair loss / hop0 image loss 等辅助监控指标，这些不参与 checkpoint 排名。

**No fix needed.**

---

## P2-1: D1/N2 guard objective may not fully cover all affected timepoints

**Reviewer finding**: SeamRefiner 影响所有 timepoint 的 decode 输出，但 `best_metric_d1_guard_metric` 只 guard `val_chain_normal_mse`。D10/D4 退化不会被检测到。

**Our verdict**: **ACKNOWLEDGED — 暂不修复**

**Rationale**:
- N2 的 `skip_first_tp: true` 已保护 D50 不被 refiner 影响
- NORMAL 是 chain 最末端，对 refiner 最敏感；如果 NORMAL 不退化，中间 hop 退化概率较低
- 当前应**先获取 N2 结果**，如果发现 D10/D4 退化但 NORMAL 未触发 guard，再补充 per-hop guard
- 过早添加多 hop guard 会使 D1 selection lane 过于保守，可能阻止 refiner 学到有用修正

**Planned follow-up**: N2 结果出来后检查 D10/D4 是否退化，若退化则添加：
```yaml
best_metric_d1_guard_metrics:
  - name: val_chain_normal_mse
    rel_tol: 0.03
  - name: val_chain_d10_mse
    rel_tol: 0.05
```

---

## Not-a-Bug (Confirmed)

Reviewer 确认以下修改正确，无回退：

1. `decode_crop` 的 `hasattr(self, "seam_refiner")` 改动 — 正确
2. `skip_first_tp` eval 路径 — 正确
3. `.gitignore` + untrack `__pycache__` — 正确

---

## Summary

| Finding | Severity | Verdict | Action |
|---------|----------|---------|--------|
| P0-1: N2 no `--resume` | CRITICAL | SUSTAINED | ✅ Fixed (`a8fd9dd`) |
| P0-2: pixel semantic mismatch | CRITICAL | SUSTAINED | ✅ Fixed (`a8fd9dd`) |
| P1-1: weight_decay granularity | HIGH | ACKNOWLEDGED | ⏸ Deferred (low risk) |
| P1-2: rolling val bias | HIGH | **OVERRULED** | ❌ False positive |
| P2-1: guard coverage | MEDIUM | ACKNOWLEDGED | ⏸ Post-N2 follow-up |
