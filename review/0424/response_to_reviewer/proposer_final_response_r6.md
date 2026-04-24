# Proposer Final Response — Round 6

**Date**: 2026-04-24  
**Responding to**: `reviewer_reply_to_rebuttal.md` + `06_reasoning_dossier_on_rebuttal.md`  
**选择**: **接受**（按 Reviewer §4 修订清单执行）

---

## 1. 逐条回应

### §1 200K 的 σ_seed 前置要求

**接受。** Reviewer 的论证令人信服：

- 0.15 dB 预测在没有 σ_seed 校准下确实无法被外部 reviewer 接受
- "曲线 plateau" 不是操作性定义
- σ-unit 重写的 stop/success criteria 更科学

**承诺**：
1. 先跑 σ_seed-lite（imgaux_boost × 3 seeds {42, 123, 456} × 50K = 3 runs）
2. 用 σ_seed 结果重写 200K 的准则：
   - PASS: `mean(200K) - mean(50K) > 3·σ_seed AND paired bootstrap 95% CI lower > σ_seed`
   - ABANDON at 150K: `Δ(150K-50K) < σ_seed OR Δ(150K-100K) < 0.5·σ_seed`
3. 如果 σ_seed > 0.08 dB，立即砍 200K，写负结果论文

### §2 σ_seed 不能延后

**接受。** 这是我之前思路的错误——把 σ_seed 归类为 "hop0 专用"。

Reviewer 正确指出：**σ_seed 是项目全局的 PSNR 仪表精度**，200K、Path A、baseline 对比都依赖它。不测 σ_seed 等于物理实验没校准天平。

**承诺**：σ_seed-lite 作为最高优先级，与 Path A 并行启动（Day 1-2）。

### §3 F2 概念纠正

**完全接受。** Reviewer 的区分是准确的：

| 概念 | 含义 | E1 实测 |
|------|------|---------|
| Decoder gain（标定效应） | 同样 latent MSE 在 NORMAL 端产生更大 dB 数字 | 确实存在 |
| Off-manifold amplification（机制效应） | Decoder 对 off-manifold 预测更敏感 | NORMAL=0.19 dB ← **最低** |

我之前混淆了两者。tail > head 的原因是 Gap_Transport 单调递增（10.74→15.70 dB），不是 decoder 敏感性。

**承诺**：在所有文档中使用 normalized improvement（ΔPSNR / Gap_Transport）或 latent MSE 相对改善来报告，避免 dB 绝对值的系统性 tail 偏袒。

---

## 2. 修订后的实验执行清单

接受 Reviewer §4 的时间线，具体化为可执行的 config 和命令：

### Phase 1: Day 1-2（并行）

**σ_seed-lite（3 runs, 2 GPU-day）**

需要 3 个 config：imgaux_boost 但 seed 分别为 42, 123, 456。

- seed=42：已有结果（旧 Scheme C best = 36.206）✓
- seed=123：需要新 config + 运行
- seed=456：需要新 config + 运行

实际新增：**2 runs × 50K = 2 × 16h = 32h**（seed=42 已有，复用旧结果）

```bash
# GPU-1: seed=123
CUDA_VISIBLE_DEVICES=1 TQDM_DISABLE=1 python -u train_first_hop.py \
    --config configs/pet_flow/pet_flow_first_hop_224_50k_sigma_seed123.yaml

# GPU-3: seed=456 (v3-50K 完成后空闲)
CUDA_VISIBLE_DEVICES=3 TQDM_DISABLE=1 python -u train_first_hop.py \
    --config configs/pet_flow/pet_flow_first_hop_224_50k_sigma_seed456.yaml
```

**Path A（0.5 GPU-day，与 σ_seed 并行）**

```bash
# 在某个空闲 GPU 或 σ_seed 完成后立即跑
python scripts/diagnose_tf_rollout_gap.py \
    --config configs/pet_flow/pet_flow_first_hop_224_50k_imgaux_boost.yaml \
    --checkpoint <C_best.pt> \
    --split val --max-slices 0 --batch-size 8
```

### Phase 2: Day 3-5（σ_seed 结果出来后）

**读数 σ_seed**：
- 计算 3 个 seed 的 transport_avg
- σ_seed = std([36.206, result_123, result_456])
- 按 verdict 决定 200K 是否启动

**如果 σ_seed < 0.08**：启动 200K（用 σ-unit criteria）  
**如果 σ_seed ≥ 0.08**：砍 200K，把 GPU 给 Path C

**Path C（1 GPU-day，独立于 σ_seed）**：
- 解冻 decoder 最后 2 个 upsample block
- 在 z_pred 上做 5K steps FT
- 测新 PSNR → 估计 RAE bound

### Phase 3: Day 6-14

- 结果汇总 + 叙事定稿
- Pix2Pix baseline（如果 σ_seed 显示比较有意义）
- Workshop 短文撰写

---

## 3. 需要创建的 Config 文件

### σ_seed Config（2 个新文件）

基于 `pet_flow_first_hop_224_50k_imgaux_boost.yaml`，只改 seed 和 run_name：

**Config 1**: `pet_flow_first_hop_224_50k_sigma_seed123.yaml`
- `run_name: first_hop_224_50k_sigma_seed123`
- `seed: 123`

**Config 2**: `pet_flow_first_hop_224_50k_sigma_seed456.yaml`
- `run_name: first_hop_224_50k_sigma_seed456`
- `seed: 456`

其余参数与 imgaux_boost 完全一致（**旧参数，不含 0422 fix**），因为 seed=42 的基准结果就是旧 Scheme C。

---

## 4. 分数路径

| 当前 | 执行 σ_seed + Path A | + Path C + F2 rewrite | + baseline |
|------|---------------------|----------------------|------------|
| **3.25** | **3.8** | **4.4** | **4.9** |

我们的目标是 2 周内达到 4.4（workshop viable）。

---

## 5. 叙事 Reframe 最终确认

从：
> "Pixel Injection Breaks the First-Hop Bottleneck in Latent Flow Matching for Low-Dose PET"

改为：
> "Systematic Diagnosis of Error Propagation in Multi-Hop Latent Flow Matching for Low-Dose PET Reconstruction"

贡献定位：
1. **诊断框架**：A/B/C/D 四维分解（exposure bias / velocity capacity / decoder bound / cascade propagation）
2. **定量证据**：10 dB gap 的 96-99% 归因于 velocity transport（E1），进一步由 Path A 分解
3. **负结论**：hop0 pixel forcing 在当前设置下不可辨，但诊断过程本身有方法论价值
4. **边界估计**：Path C 给出 "frozen RAE 范式内能做到多好" 的上界

---

## 6. 致 Reviewer

感谢 5 轮 + rebuttal 的严格审查。特别感谢：

1. **§4.2 的 E1 反证**（Gap_Decoder 在 NORMAL 最低）—— 纠正了我们对 decoder 放大的错误理解
2. **σ_seed 的全局性论证**（不是 hop0 专用）—— 纠正了我们的方法论盲点
3. **200K 妥协方案**（先 σ 后跑）—— 比我们的原方案既省 GPU 又更科学
4. **Reasoning Dossier** 的透明度 —— 让我们能审计审稿人的推理链

我们将按修订清单执行。下一轮提交：σ_seed + Path A 结果 + 200K 决策。
