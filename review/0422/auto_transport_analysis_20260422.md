# 0422 自动化分析报告（基于代码与产物）

- 生成时间: 2026-04-22 22:58:41
- Git HEAD: `86ccf1f`
- 脚本: `review/0422/generate_transport_analysis_0422.py`

## 1) 关键数值对照（clip3, full-val）

| Timepoint | Ceiling PSNR | D1 best PSNR | Gap (Ceiling - D1) |
|---|---:|---:|---:|
| D50 | 42.622386 | 38.591577 | 4.030809 |
| D20 | 46.635633 | 35.923970 | 10.711663 |
| D10 | 48.743580 | 36.094169 | 12.649411 |
| D4 | 50.831939 | 36.146838 | 14.685101 |
| NORMAL | 52.634069 | 36.449659 | 16.184410 |

- tail 平均 gap（D20/D10/D4/NORMAL）: `13.557646 dB`

## 2) 代码/配置证据检查

| 检查项 | 值/状态 | 证据 |
|---|---|---|
| first_hop_weight_decay | `0.0` | `configs/pet_flow/pet_flow_first_hop_224_50k_schemec_v2.yaml:153`, `train_first_hop.py:1189`, `train_first_hop.py:1219` |
| lambda_hop_init | `0.10` | `configs/pet_flow/pet_flow_first_hop_224_50k_schemec_v2.yaml:261` |
| hop_residual_last_init_std | `0.01` | `configs/pet_flow/pet_flow_first_hop_224_50k_schemec_v2.yaml:264` |
| val_window_mode | `rolling` | `configs/pet_flow/pet_flow_first_hop_224_50k_schemec_v2.yaml:48`, `train_first_hop.py:682` |
| EMA enabled/decay | `enabled=true, decay=0.9999` | `configs/pet_flow/pet_flow_first_hop_224_50k_schemec_v2.yaml:161`, `train_first_hop.py:1167`, `train_first_hop.py:2148`, `pet_lr/ema.py:14` |

## 3) 自动判读（根据当前证据）

- 结论A: **支持 transport 是主瓶颈（高置信）**。
- 结论B: **支持“decoder 非主导瓶颈（限 GT latent/on-manifold 条件）”**。
- 结论C: v2 修复方向总体合理，但当前是 bundle fix（多项同时改动），不能直接归因到单一因素。

## 4) 需要保守处理的表述

- `decoder 完全不是瓶颈` 建议改为：`当前证据支持 decoder 非主导瓶颈（在 GT latent/on-manifold 条件下）`。
  - 文档位置: `review/0422/exp/schemec_v2_experiment_plan.md:17`
- `val_select_score < 0.000500` 仅可作监控阈值，不建议作为唯一 go/no-go 标准。
  - 文档位置: `review/0422/exp/schemec_v2_experiment_plan.md:95`
- `旧实验全部无效` 建议降级为：`旧实验机制性结论需重审`。

## 5) 最小补强动作（1-2天）

1. 做 branch counterfactual eval：`g_pix=0`、`lambda_hop=0`、`both=0`。
2. 做 latent perturbation sweep（`z_gt -> z_pred`），验证 decoder 对 off-manifold 的脆弱性。
3. checkpoint 选择用 fixed full-val/fixed subset rerank，避免仅依赖 rolling-window 分数。

