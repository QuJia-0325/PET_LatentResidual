# V3 三实验结果分析与诊断 — 2026-04-26

**输入**: commit `3b50782` 的三实验产物（Path A multi-ckpt + SF-pair 50K + Rollout-Up 50K）  
**起点**: V3 plan ([../plan/transport_breakthrough_research_v3.md](../plan/transport_breakthrough_research_v3.md))  
**结论先行**: Path A **强支持** exposure-bias 假设；B/C 不可信，须 redo。

---

## 1. 三实验观测表

### 1.1 Path A 多 ckpt 诊断（实验 A） ✅ 主要科学证据

来源: [../0425/logs_diag/pathA_C_multi_ckpt_gpu0.log](../0425/logs_diag/pathA_C_multi_ckpt_gpu0.log)

| ckpt | mean exposure_gap (hops 1–3) | mean ceiling_gap | LatMSE_RO/TF (D4→NORMAL) | verdict |
|------|-----------------------------:|-----------------:|--------------------------:|---------|
| step_010000 | 4.19 dB | 12.44 dB | 4.32× | EXPOSURE_BIAS_DOMINATES |
| step_030000 | 4.76 dB | 10.09 dB | 5.34× | EXPOSURE_BIAS_DOMINATES |
| best.pt     | **4.80 dB** | **9.52 dB** | **5.64×** | EXPOSURE_BIAS_DOMINATES |

末端 hop 单点（D4→NORMAL @ best）: `PSNR_TF=43.13`, `PSNR_RO=36.81` → exposure 单 hop 6.32 dB。

**趋势分析**：
- exposure_gap 单调上升 4.19 → 4.80 dB（+14.5%）：**训练越久，TF/RO 越分裂**。这是 V3 plan §5 的可证伪预测之一，**满足**。
- ceiling_gap 单调下降 12.4 → 9.5 dB：velocity capacity 在持续吃满 decoder，**capacity 不再是瓶颈**。
- LatMSE_RO/TF 比 5–6×：rollout 路径的 latent 误差已经远超 TF 路径，与 chain visualization 的 5–7 dB cascade 损失定量吻合。

**结论**：exposure bias 是核心，方向对。

### 1.2 SF-pair vs Rollout-Up（实验 B/C） ⚠️ **不可信**

来源:  
- [../0425/logs_train/selfforcing_v3_gpu0_r1.log](../0425/logs_train/selfforcing_v3_gpu0_r1.log)  
- [../0425/logs_train/rolloutup_v3_gpu0_r1.log](../0425/logs_train/rolloutup_v3_gpu0_r1.log)

| 指标 @ step 50000 | B (SF-pair) | C (Rollout-Up) | resume 起点（step 46400） |
|-------------------|------------:|---------------:|--------------------------:|
| `val_select_score` | 0.000672 | 0.000670 | 0.000521 |
| `val_chain_normal_mse` | 0.000213 | n/a (待查) | 0.000161 (45K best) |

**两个红旗**：

#### 🚨 红旗 A — Resume 起点错位

两份 log `[resume] loaded step=46400, best_val=0.000521`，**不是**计划里假设的 step=50000。  
有效新增训练 = **3600 步**（46400 → 50000），而不是计划里的 50000 步。

直接后果：
- SF 调度（warmup=5000, ramp=10000，**写死步数**）按计划应在 5000–15000 区间渐进开 SF，前 5000 步纯 GT。
- 实际 resume 后 `global_step` 立即 ≥ 46400 ≫ 15000 → `alpha_sf` 第 1 步就 = 1.0，**SF 直接满载，无 warmup**。
- LR=4e-5 + 5% warmup（=2500 步）也是按 max_steps=50000 计算，但有效新增 3600 步，warmup 还没过完整就训完。

#### 🚨 红旗 B — B 和 C 几乎完全一致（Δ = 2e-6）

两个完全不同的干预（SF on / Rollout-Up λ×4，且 SF off）得到几乎完全一致的退化结果，**且都比 baseline 倒退 ~29%**（0.000521 → 0.000672）。

可能原因（按概率排序）：
1. SF 实际从未 warmup（红旗 A 直接后果） — 概率最高
2. 3600 步太短，主要是 LR-warmup + 输入分布震荡，未到稳态
3. SF 是否真的替换 z_src **未从 stdout 验证**：`[train]` 行不打 `sf_alpha`/`sf_gap_norm`，只写进 metrics_jsonl
4. log_grad_norms 数据存在（`grad_total ~ 6e-3, grad_backbone ~ 6e-3`），两边量级几乎相同 → 也支持"两边都没真正激活差异"的判断

#### 待确认的诊断（验证 sf 是否真正生效）

```bash
# 远端验证（需在跑实验的机器执行）：
JL=/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_selfforcing_from_C_gpu0_r1/metrics.jsonl
python -c "
import json
for fn in ['$JL']:
    with open(fn) as f:
        for ln in f:
            d = json.loads(ln)
            if d.get('event','train') == 'train':
                print(d.get('step'), d.get('sf_alpha'), d.get('sf_gap_norm'), d.get('sf_z_pred_norm'))
" | head -20
```

**期望输出**：`sf_alpha=1.0`, `sf_gap_norm > 0`（应该和 latent scale ~0.001 同数量级或更大）。
- 若 `sf_gap_norm ≈ 0` → SF 替换路径出 bug（per-sample gather 或 mix 数学错），方法本身从未被测试。
- 若 `sf_gap_norm > 0` 且 `sf_alpha = 1.0` 从第 1 步开始 → 红旗 A 成立，调度 bug 是元凶，不是方法。

---

## 2. SF schedule resume 语义 bug — 根因分析

代码定位: [`train_first_hop.py`](../../train_first_hop.py) line 488-494

```python
alpha_sf = get_linear_schedule_value(
    global_step=global_step,             # ← 用绝对步号
    warmup_steps=int(sf_cfg.get("warmup_steps", 5000)),
    ramp_steps=int(sf_cfg.get("ramp_steps", 10000)),
    start=float(sf_cfg.get("alpha_sf_start", 0.0)),
    end=float(sf_cfg.get("alpha_sf_end", 1.0)),
)
```

`get_linear_schedule_value` 在 `global_step ≥ warmup_steps + ramp_steps` 时直接返回 `end=1.0`。
resume 时 `global_step` 从 46400 起步 → 永远落在尾段 → `alpha_sf ≡ 1.0`。

对比 rollout/image_aux：B/C config 显式写 `warmup_ratio: 0.0, ramp_ratio: 0.0`（即 warmup=ramp=0，
schedule 直接读 `end`，与起点无关，这是可接受的）。SF section 漏了同样的处理。

**两条修法**（任选）：
1. **配置层修法**（最快）：B/C 的 SF config 把 warmup_steps / ramp_steps 改成相对 resume 起点  
   `warmup_steps: 51400` (=46400+5000)，`ramp_steps: 10000` —— 但这违反"config 与 resume 步号无关"的原则。
2. **代码层修法**（更稳）：在 SF helper 内部读 `cfg["training"].get("self_forcing_pair", {}).get("schedule_origin", "absolute")`，支持 `"resume_relative"` 模式：  
   `effective_step = global_step - resume_start_step`。在 main loop 把 resume_start_step 注入 cfg 即可。

**推荐法 2** — 代码改动 < 10 行，behaviour 显式可控。

---

## 3. 已确认成立 / 已证伪的预测

V3 plan §5 的可证伪预测复检：

| 预测 | 阈值 | 实测 | 判定 |
|------|------|------|------|
| Path A exposure_gap @ best | > 0.3 dB | **4.80 dB** | ✅ 强成立 |
| Path A ceiling_gap @ best | > 5 dB | **9.52 dB** | ✅ 成立 (capacity 也仍是问题，但远小于 exposure) |
| Path A gap 单调 ↑ (10K→30K→best) | 单调 | 4.19→4.76→4.80 | ✅ 单调成立 |
| SF-pair > Scheme C best | Δ > 0.3 dB (chain_normal_mse) | 0.000213 vs 0.000161 (退化) | ❌ **不可判定** — schedule bug 污染 |
| SF-pair > Rollout-Up | Δ > 0.3 dB | Δ ≈ 0 | ❌ **不可判定** — 同上 |
| SF-pair grad_norm 偏离 Rollout-Up | 相对偏差 > 20% | grad_total 量级几乎相同 | ❌ **不可判定** — 同上 |

**总结**：诊断（A）成立，干预（B/C）需 redo 才能下结论。

---

## 4. 关于 200K v3 训练（GPU-1，未停）

GPU-1 上的 200K v3 训练仍在跑（resume 前 step ≈ 58K，按 V3 plan 时间线 §4 现在应在 90–95K）。
当前观察：B/C 三天得不出有效结论，但 200K v3 已经 "免费" 跑了 35K+ 步，**它是当前唯一稳定推进的实验**。

**[更新 0426]** 200K v3 **best 仍在更新**——说明模型尚未收敛到 plateau，velocity capacity 还在改善中。
这对 v3.1 redo 有两个影响：
1. **v3.1 的起点 ckpt 不应现在锁定**。应等 200K v3 完成（或 best 连续 20K 步不再更新时）再取 best.pt，否则起点不是最优；
2. **200K v3 的最终 Path A 诊断更有价值**：best 还在更新意味着 exposure_gap 可能还在变化。最终 200K best 的 Path A 多 ckpt 诊断将给出 exposure bias 在长跑下的饱和行为。

**建议**：
- 不停 200K v3，让它继续跑到 200K（约还需 1.5 天）
- **v3.1 B'/C' 的起点 ckpt 在 200K v3 完成后再确定**，用其 best.pt（需在启动前 assert step 和 val_score）
- 在等待 200K 完成期间，先完成 P0（远端核查）、P1（代码修）、P2（stdout 诊断）、P3（smoke test）
- Day 3-4 用 200K final ckpt 重新跑 Path A 多 ckpt（30K / 100K / 150K / best），看 exposure_gap 趋势在 200K 长跑下是否反转 / 饱和

---

## 5. 致命教训（写入 lessons learned）

1. **任何"按 step 写死"的调度，在 `resume` 路径都必须显式声明锚点**。本次 SF 调度沿用 absolute global_step，但 rollout/image_aux 早就因为同样的坑改成 ratio=0 了——SF 只是抄了 rollout 的接口没抄它的 resume 应对。
2. **`[train]` 行 stdout 必须打印关键诊断字段**。仅写 metrics_jsonl 让事后排查只能 ssh 跨机捞文件，急救流程慢一个数量级。
3. **resume 起点必须在 plan 里硬编码并 assert**：本计划假设"50K best.pt"，实际是 46400。Day 0 前应该 `python -c "import torch; print(torch.load('best.pt')['step'])"` 确认。
4. **Δ ≈ 0 + 双双退化** 是"两个干预都没生效"的典型签名，比"两个干预都生效但效果相同"概率高得多。
