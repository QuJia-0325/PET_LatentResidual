# 0427 实验设计：V5 失败归因分析

**动机**：Codex counter-review 指出我们之前的 V5 分析有关键缺陷——"step 93200 相变"实际上是 rolling-val window 回卷，不是模型参数变化。需要用严格的评估协议重新判断 V5 是否真的无效。

## 实验列表

| 编号 | 脚本 | 目的 | GPU | 耗时 | 优先级 |
|------|------|------|-----|------|--------|
| **01** | `01_fullval_eval.sh` | Full-val 评估 V3/V5 ckpts，消除 rolling-val 噪声 | 1 卡 | ~1h | **P0** |
| **02** | `02_pathA_baseline.sh` | V3 200K best 的 Path A ExpoGap baseline | 1 卡 | ~30min | **P0** |
| **03** | `03_null_control.sh` | Baseline config resume 继续训 50K（不改 loss） | 1 卡 | ~30h | **P1** |
| **04** | `04_pathA_v5_best.sh` | V5 best 的 Path A ExpoGap（与 02 对比） | 1 卡 | ~30min | **P1** |

## 执行顺序

```
# 第一批（P0，~1.5h，1 卡串行）
bash review/0427/run_me/01_fullval_eval.sh 0
bash review/0427/run_me/02_pathA_baseline.sh 0

# 第二批（P1，并行 2 卡）
bash review/0427/run_me/03_null_control.sh 2    # ~30h
bash review/0427/run_me/04_pathA_v5_best.sh 0   # ~30min
```

## 判定逻辑

### 01 Full-Val 结果解读
| V5 best full-val vs V3 full-val | 结论 |
|--------------------------------|------|
| V5 改善 ≥ 3% | rolling-val 掩盖了真实改善，V5 方向有效 |
| V5 ≈ V3（±3%） | V5 没有改善也没有恶化——需要更多步数或更强干预 |
| V5 恶化 > 3% | V5 方向确实有害，即使消除 rolling-val 噪声 |

### 03 Null-Control 结果解读
| null-control 50K 后 vs V3 baseline | V5 vs null-control | 结论 |
|-----------------------------------|-------------------|------|
| null-control 持平 | V5 恶化 | V5 的 loss 改动是退化主因 |
| null-control 持平 | V5 持平 | 两者都在 plateau，loss 改动无效但无害 |
| null-control 也退化 | V5 也退化 | resume + LR schedule 是主因，不是 V5 loss |
| null-control 改善 | V5 恶化 | V5 loss 改动明确有害 |

## 文件清单

```
review/0427/run_me/
├── README.md                  ← 本文件
├── 01_fullval_eval.sh         ← Full-val 评估
├── 02_pathA_baseline.sh       ← V3 Path A baseline
├── 03_null_control.sh         ← Null-control 训练
└── 04_pathA_v5_best.sh        ← V5 Path A
```

## 依赖

- Null-control config: `configs/pet_flow/pet_flow_first_hop_224_v5_null_control.yaml`
  （已创建，基于 200K v3 config，仅改 run_name + resume 兼容性）
