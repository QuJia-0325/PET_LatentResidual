# V6 完成后的实验路线图（v3.1 — 修正 v3 事实偏差与脚本 gate 进一步强化）

> 本文档基于 2026-05-02 19:47 的训练快照（[review/0502/log_snapshots/](log_snapshots/)）+ 实测日志数据，规划 V6 完成 200K 后的下一步实验。
> **依赖文档**：[SIGMA_NORMALIZE_ABLATION_PLAN.md](SIGMA_NORMALIZE_ABLATION_PLAN.md) / [README.md](README.md) / [AUDIT_RESPONSE_v4.md](AUDIT_RESPONSE_v4.md)
> **修订记录**：
> - v1 → v2 (2026-05-02, agent1/2/3/4 交叉审计)：修正 eval/check_alignment CLI / V6 best ckpt 口径 / V6.1 lambda 占比与 chain_normal 优势 / V3 baseline 数值
> - v2 → v3 (2026-05-02, agent3/4 二次审计)：修正 V6 vs V3 方向 / eval 输出文件名 / decode-mode both 在 V6 上为 no-op / sanity gate 实际不会自动拦截 / mini full sanity SANITY_STEPS 环境变量
> - v3 → v3.1 (2026-05-02, agent1/2/3/4 三次审计)：【考据】重写 §0 去除造伪的 FP 精度数值（-6e-7 是推测，log 仅 6 位）【一致性】消除 §2.2 与 §7 的并行 gate 矛盾 【调度】修正 §5 V6.1 时间数学 【贯通】明言 metrics.jsonl 只写 `val_select_score`（不是 `val_multi_objective`）【脚本】run_ablation.sh 加 CUBLAS_WORKSPACE_CONFIG / 缺键 fail-strict / sanity sentinel gate。

---

## 0. 当前训练状态（snapshot @ 2026-05-02 19:47）

| Run | 进度 | 当前 best.pt @ 选择口径 | 关键观察 |
|-----|------|----------------------|---------|
| V6 transport-first (200K) | 199 150 / 200 000 (99.6%) | step **185600**, `val_select_score`=0.000607（best_metric=`val_multi_objective`; chain_normal=1.70e-4） | Phase III 中段，等待最后 ~850 步 |
| V6.1 rollout-floor (200K) | 62 950 / 200 000 (31.5%) | step **12000**, `val_select_score`=0.000762（best_metric=`val_multi_objective`; chain_normal=2.13e-4） | Phase II 早期，下文 §4 详细分析 |
| σ-norm sanity light A (50K) | 9 250 / 50 000 (18.5%) | n/a (`require_chain_metrics: false`) | 仍在 warmup，σ-norm 需 step ≥ 12 500 才进入 ramp |

**V6 best ckpt 重要说明**（agent1/3/4 修订 + v3.1 修正）：

| 选择口径 | best step | val_select_score | val_chain_normal_mse | 是 best.pt 吗？ |
|---------|----------|----------------:|---------------------:|----------------|
| `val_multi_objective` (yaml `best_metric`) | **185600**（最后 new best）| 0.000607 (log 6 位截断) | 0.000170 | ✅ **是 best.pt** |
| `val_chain_normal_mse` 单独 | 186000 | 0.000623 (log) | **0.000164** | ❌ 仅在 metrics.jsonl 有数，**无对应可加载 ckpt**（`save_interval=20000` 只保存 step 20K/40K/.../200K）|
| 最近持久化 ckpt | step_180000.pt | 0.000641 (log) | 0.000174 | ✅ 可加载，但这不是 best.pt；chain_normal 比 best.pt 还差 +2.4% |

> **错误被证伪 + 造假证据被除去**（agent3/4 v3.1 修正）：v2/v3 都在驳 agent1 “best.pt = step 162400” 时使用了“按 FP 精度 0.0006075 严格大于 0.0006069，diff = -6e-7”这种具体数字 — **这些数字是推测**，log 只保存 6 位有效数字（两者均显示 0.000607）。真正的**硬证据**只有三条：
> 1. V6 训练日志中明示存在 `[val] new best val_multi_objective(...) = ... at step=185600`
> 2. 代码路径是 `if key < best_val: ... save_checkpoint(best.pt)`（[train_first_hop.py L2440–L2510](../../train_first_hop.py)）
> 3. 初始仅当严格小于才会记录为 new best 并覆写 best.pt → 所以 step 185600 处 val_select_score **严格小于** step 162400 处的值
>
> 具体该差多少 (1e-6个位 / 1e-7 / 或者更小) 不能从日志本身验证。若 paper 需要精确数字，**必须从 `metrics.jsonl` (FP32 ASCII 序列化到 ~1e-7) 读 162400 与 185600 两行的 `val_select_score`** 。

**重要键名说明**（agent1 v3.1 修正）：
- yaml 中 `training.best_metric: val_multi_objective` **仅是配置名**。选 best.pt 使用的实际加权和以 `val_select_score` 这个键写入到 metrics.jsonl（[train_first_hop.py L2446](../../train_first_hop.py): `metrics["val_select_score"] = float(key)`）。
- `val_multi_objective` **不是** metrics.jsonl 里的键，仅出现于 stdout 的 “new best” 字样。任何 jq / python 汇总脚本 **都该读 `val_select_score`**，否则拿到空值。
- log [val] 行中同一个数字打印为 `val_select_score=0.000607`；metrics.jsonl 里是 FP32 ASCII，精度足够区分 162400 与 185600。

**系统性 separation**（agent1 代表性问题）：multi_obj best step (185600) 与 chain_normal best step (186000) 仅差 400 步；更明显的 separation 在 multi_obj 连续 two new-best 之间（step 162400 → 185600 跨 23 200 步，进展仅 6 位以后的粒度），这意味着 V6 在 Phase III 中后期 **multi_obj 平原期**与 **chain_normal 仍在缓慢下降**是同时发生的 trajectory-level 现象，paper 必须同时报两套数字避免 cherry-pick 嫌疑。

**V6 schedule 三阶段**：

```text
Phase I  (0     – 50000)  : 纯 GT，alpha=0, λ_roll=0
Phase II (50000 – 150000) : alpha & λ_roll 同步线性 ramp 到 4.0
Phase III(150000 – 200000): alpha=1, λ_roll=4.0 全负荷 → final
```

**sanity light yaml 要点**（agent4 提示）：[`A_sanity_light.yaml`](configs/A_sanity_light.yaml) 中 `max_steps: 120000` 仅作 fallback，[`run_sanity_light.sh`](scripts/run_sanity_light.sh) 通过 `MAX_STEPS=50000` 强制 override 到 50K。直接读 yaml 会被误导。

---

## 1. V6 完成时（剩余 ~850 步, < 1 GPU·hour）

### 1.1 立即 eval V6 best ckpt（PSNR/seam 复核）

```bash
# 在任意空闲 GPU 上：
python eval_first_hop_224_clip3.py \
    --config configs/pet_flow/pet_flow_first_hop_224_v6_transport_first.yaml \
    --checkpoint /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v6_transport_first/best.pt \
    --split val \
    --max-slices 0 \
    --out-dir /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v6_transport_first_eval_best \
    --decode-mode default \
    --device cuda:<free>
```

**关于 EMA**：训练保存 `best.pt` 时是用 `with ema.average_parameters(): save_checkpoint(...)`（[`train_first_hop.py` L2510-2530](../../train_first_hop.py)）写入，所以 `state_dict["model"]` **就是** EMA 参数。eval 直接加载 best.pt 即拿到 EMA 版本，**无需也没有 `--use_ema` 开关**。

**关于 `--decode-mode default` vs `both`**（agent2 修订）：V6 训练时 `seam_refiner_enabled = False`（实测 V6 log 中 `val_chain_*_raw_mse=nan`，证明 refiner 未启用）。`--decode-mode both` 的 raw 分支与 default 完全一样，CSV 会多出一块完全重复的列。**使用 default 避免误导“已验证 raw vs refined”**。

**这一步 eval 输出什么**（agent3 修订）：eval_first_hop_224_clip3.py 会在 `--out-dir` 下写：
- `first_hop_224_val_clip3_eval.json`：含 `summary_psnr_clip3` / `summary_seam_consistency` / `summary_extended_seam` / `per_slice` 四个结构化块
- `first_hop_224_val_clip3_eval.csv`：per-slice PSNR clip3 表
- **不输出** chain MSE，**不输出** FID（脚本未实现 FID）。chain MSE / multi_objective 数字直接从 V6 训练日志的 `metrics.jsonl` 中固化即可（best.pt 对应 step 185600）

### 1.2 ~~alignment check~~ → 改为：从 V6 metrics.jsonl 固化数字

[`check_alignment_224_clip3.py`](../../check_alignment_224_clip3.py) 检查的是 **`latents_224.pt` ↔ `preprocessed_data_*.pt` ↔ RAE LoRA decoder** 三者的对齐，是数据集级 stage-1 审计工具，**不接收任何 first-hop ckpt 参数**（参数只有 `--latents-dir / --raw-data-dir / --rae-ckpt / --out-dir`）。它无法用来"验证 V6 ckpt 在所有 hop 上的 alignment 指标"——v1 文档此处概念错位，v2 删除该步骤。

**替代方案**：从 V6 `metrics.jsonl` 提取 best.pt 对应行的所有指标作为 paper 锚点：

```bash
PYTHON=/home/qujiaxiang/.conda/envs/rae/bin/python
${PYTHON} - <<'PYEOF'
import json
path = "/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v6_transport_first/metrics.jsonl"
target_step = 185600
target = None
for line in open(path):
    obj = json.loads(line)
    if obj.get("event") == "val" and obj.get("step") == target_step:
        target = obj
        break
print(json.dumps(target, indent=2, ensure_ascii=False))
PYEOF
```

输出即作为 `V6 best.pt` 的 paper 数字（`val_select_score ≈ 0.000607`，`val_chain_normal_mse ≈ 1.70e-4`，等等）。注意：从 metrics.jsonl 读的是 `val_select_score`、`val_chain_normal_mse` 这些键；**不是** `val_multi_objective`（后者仅是 yaml 配置名，不会被写入 metrics.jsonl）。

> **⚠️ 不要用 `summarize_run.sh` 的输出作为 paper 表数字**（agent6 v3.1 follow-up）。
> `summarize_run.sh` 报告的是 *per-metric independent minima*：对每个指标独立扫描 metrics.jsonl 取最小值的那一行——不同指标可能来自 **不同的 step**。它适合做"快速看 chain 是否塌"或"per-hop raw 诊断"。
> **paper 里 A/B/C/D 的对比必须用 best.pt 那一行作为整体**（即按 `val_select_score` 最小选出的 step，所有指标都报这一 step 的值）——上面 §1.2 的 python snippet 才是正确路径。两者口径不一样，混用会让 A/C/D 的差距变大或变小，直接影响论断。

### 1.3 EMA CV 快算（agent3 v3.1 推荐 — 不占 GPU，马上能拿到 V6 vs V3 第一个差异化证据）

§6 提出 V6 vs V3 在 chain MSE 上 essentially tied 后，需从 **训练动态稳定性** 维度拿出差异化证据。最便宜的证据是 “V6 last-50K 窗口内 chain_normal 的 coefficient of variation (std/mean) < V3 的 CV” — 如果这项成立，paper 可写 “V6 transport-first 产生 cleaner Phase III dynamics 与 V3 相同级别 chain MSE 共存”。

```bash
PYTHON=/home/qujiaxiang/.conda/envs/rae/bin/python
${PYTHON} - <<'PYEOF'
import json, statistics

def load_window(path, last_steps=50000):
    rows = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("event") == "val" and "val_chain_normal_mse" in obj:
                rows.append((obj["step"], obj["val_chain_normal_mse"]))
    if not rows:
        return None
    last_step = max(r[0] for r in rows)
    cutoff = last_step - last_steps
    win = [v for s, v in rows if s >= cutoff]
    return win

for name, path in [
    ("V6", "/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v6_transport_first/metrics.jsonl"),
    ("V3", "/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v3_baseline/metrics.jsonl"),  # adjust path
]:
    win = load_window(path, last_steps=50000)
    if win is None:
        print(f"{name}: no data at {path}")
        continue
    mean = statistics.mean(win)
    std  = statistics.stdev(win) if len(win) > 1 else 0.0
    cv = std / mean if mean > 0 else float("nan")
    print(f"{name}: n={len(win)}  mean={mean:.6e}  std={std:.6e}  CV={cv:.4f}")
PYEOF
```

期望输出形式：`V6: n=125 mean=2.1e-4 std=4.2e-5 CV=0.20` vs `V3: n=125 mean=2.1e-4 std=6.3e-5 CV=0.30`（示例，具体数字待实跑出）。这个表结果可直接进 paper 的 “Training stability comparison” 表。

---

## 2. σ-norm 主 ablation（GPU 0 立即启动）

V6 跑完后 GPU 0 释放。**σ-norm ablation 是当前最高优先级**——它是 review/0502/ 整个 bundle 的目的。

### 2.1 为什么 A_main 必须独立重跑 120K

虽然 V6 200K log 包含 step=120K 的 metrics，但 **不能** 复用为 A_main——schedule 阶段不同：

| | warmup 长度 | ramp 长度 | step 120K 处所处阶段 |
|---|------------|-----------|---------------------|
| V6 (200K, ratio=0.25/0.50) | 50 000 | 100 000 | Phase II 70%（α=0.7, λ_roll=2.8）|
| A_main (120K, ratio=0.25/0.50) | 30 000 | 60 000 | Phase III 早期（α=1.0, λ_roll=4.0, 已稳定 30K 步）|

→ A_main 与 C_uniform 都在 120K 压缩 schedule 下训练，schedule 一致才能做单变量对比；**V6 step=120K 的指标只能用作"schedule 在 120K 处大致是什么样"的参考，不能作为 ablation 的 control**。

> **隐含风险**（agent4 提示）：A_main 120K 压缩 schedule 与 V6 200K 完整 schedule 在最终性能上**不一定等价**。论文 narrative 应以"在 120K 压缩 schedule 下 A vs C 的相对差异"为主 claim，避免直接报"σ-norm 在 200K 下的最终 chain MSE"——后者需要单独跑 200K A/C 来验证（PLAN §3.3 阶段 3，~10 GPU·days）。

### 2.2 启动顺序（v3.1 修正 — 取消与 §7 冲突的并行命令）

```bash
# GPU 0 → A_main (120K, ~1.0 GPU·day @ 80 steps/min on A100)
# A_main 使用 V6 reference step_weights，不依赖 sigma_dt code path → 可与 sanity 并行。
# 这个并行路径走 `A` 快捷跱（无 gate）而不是 `main`（`main` 是双门 gate 的串行快捷跱）
GPU=0 bash review/0502/scripts/run_ablation.sh A

# C/D 依赖 sigma_dt normalizer code path，必须等 sanity Tier 1-4 全 PASS。
# v3.1 脚本会拒绝启动除非存在 sanity-pass sentinel（由 run_ablation.sh sanity 在 PASS 后写入）:
GPU=0 bash review/0502/scripts/run_ablation.sh C   # 120K, ~1.0 GPU·day
GPU=0 bash review/0502/scripts/run_ablation.sh D   # 120K, ~1.0 GPU·day，可选

# 资源充足且确信 sanity 已 PASS 时，可用串行快捷跱（A→C，两者双门 gate）:
GPU=0 bash review/0502/scripts/run_ablation.sh main
```

**A_main best.pt 路径**（agent4 v3.1 修正，为后续 eval/summarize 给出明确路径）：
- 路径模板：`${ABLATION_OUTPUT_ROOT}/<tag>/run/first_hop_224_sigma_norm_<tag>/best.pt`
- 默认 ABLATION_OUTPUT_ROOT = `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0502_runs`
- 常用路径：
  - A_main: `/data_2/.../review_0502_runs/A_main/run/first_hop_224_sigma_norm_A_main/best.pt`
  - C    : `/data_2/.../review_0502_runs/C/run/first_hop_224_sigma_norm_C/best.pt`
  - D    : `/data_2/.../review_0502_runs/D/run/first_hop_224_sigma_norm_D/best.pt`
- metrics.jsonl 与 best.pt 同目录。**paper 表用 §1.2 的 python snippet**（按 best.pt 步号取整行）；[`scripts/summarize_run.sh`](scripts/summarize_run.sh) 仅做 per-metric 独立最小值的快速诊断，不要把它的输出直接抄进 paper 表（agent6 v3.1 follow-up，详见 §1.2 警告框）。

> **不再有的 v3 文本**：v3 §2.2 仍写 “sanity light Tier 1-3 已 PASS 即可并行 GPU 0=A & GPU 1=C”。**这与 §7 的 “C 必须等 Tier 1-4 全 PASS” 直接冲突**（agent1/2/4 同时指出）。v3.1 以 §7 为准删除该并行命令。A_main 仍可与 sanity 并行（不依赖 sigma_dt path）；C/D 独立启动需 sentinel。

### 2.3 主 ablation 通过/失败判据

**判据口径**（与 yaml `best_metric` 对齐）：以 `best.pt`（即按 `val_multi_objective` 选出）的两个独立维度为准：

- **主指标**：`val_select_score`（yaml `best_metric: val_multi_objective` 对应的加权和：0.5·d20 + 0.45·d10 + 0.9·d4 + 1.5·normal）
- **次指标**：`val_chain_normal_mse`（end-of-chain 单点 MSE，更敏感于末段 hop 难度）

**rel diff 定义**（agent2 修订）：`rel = (X_best.pt[metric] - A_best.pt[metric]) / A_best.pt[metric]`，次指标必须 **同向过阈**才认为结果一致（即主指标说 C≈A，次指标也必须说 C≈A）。如两者结论相反，记为 `unstable result`，必须扩到 200K 才能下定论。

| 结果（按 best.pt 处的 multi_obj rel）| 解读 | 论文 narrative |
|-------------------------------------|------|--------------|
| **C ≈ A**（rel diff < 5%）| step_weights shape 不重要 | 仅在**当前 pair-weighted transport-first 设置**下成立：rollout step-weight shape 的边际贡献不明显、可被 σ²·dt² normalizer 吸收。**不能外推到** “PET hop-level 难度由 σ·dt 几何主导”之类的强因果 claim |
| **C 显著差 vs A**（rel diff > 10%）| shape 至关重要 | 需 D 对照才能确定是 V6 [0.5,2,1.5,1] 偶然好还是 closed-form 才是最优 |
| **A 显著差 vs C/D**（V6 weights 不优）| 需重新 sweep step_weights | 推迟 paper，启动 V7 weight sweep |
| **A vs C 在 5%-10% 之间** | 不能下定论 | 启动 PLAN §3.3 阶段 3：A/C 双双延到 200K（+10 GPU·days）|

> **额外 caveat**（agent4 修订）：若 C 显著差 vs A，**不能立即下“shape 重要”结论**。A_main 120K 与 C_uniform 120K 都是压缩 schedule，但 C 的 step_weights 是 uniform [1.25,1.25,1.25,1.25]，这可能与压缩 schedule 产生**交互偏差** (uniform shape 下 hop3 梯度噪声 × 压缩 schedule)。需补跡 C @ 200K full schedule 才能区分“shape 不重要” vs “shape × schedule 交互”。

> **主指标与次指标冲突的可能性**（agent1 修订）：V6 自己在 multi_obj best step (185600) 与 chain_normal best step (186000) 间仅差 400 步，但 162400 → 185600 为 multi_obj 连续 two new-best 间隔 23 200 步，表明 Phase III 中后期 multi_obj 与 chain_normal 存在 trajectory-level systematic separation。A vs C 的主指标与次指标给出相反结论的概率**不低**。**Paper 必须同时报两套数字**，不能只以其中一项写 hero metric。

---

## 3. σ-norm sanity gate（GPU 2，独立线，gate 模式）

`run_sanity_light.sh` 现在仍在 GPU 2 跑 A_sanity_light（50K），跑完会自动接 B_sanity_light。

### 3.1 sanity light comparator 仅覆盖 Tier 1-3（agent2 提示）

[`scripts/run_sanity_light.sh`](scripts/run_sanity_light.sh) 末尾的 comparator 与 [`scripts/run_ablation.sh sanity`](scripts/run_ablation.sh) 不同：

| | run_sanity_light.sh | run_ablation.sh sanity |
|---|--------------------|------------------------|
| max_steps | 50 000（runner override）| 50 000 |
| chain eval | **关闭** (`require_chain_metrics: false`) | 开启 |
| Tier 1 `val_pair_total` (rel < 1e-4) | ✅ | ✅ |
| Tier 2 `val_rollout_total` (rel < 1%) | ✅ | ✅ |
| Tier 3 `val_rollout_step_*_raw` (rel < 1%) | ✅ | ✅ |
| Tier 4 `val_chain_*_mse` (rel < 5%) | ❌ 不验证 | ✅ |

→ 当前 GPU 2 上跑的是 light 版本，**只能验证 σ-normalize 数学入口正确性，不能保证 A/B 在 chain MSE 上端到端等价**。

### 3.2 推荐增补：mini-sanity full（20K + chain eval ON）

为了在启动 C/D 之前覆盖 Tier 4，建议在 light A/B 全部 PASS 后**插入一次 full-eval mini sanity**：

```bash
# v3.1 (agent3 修订)：默认 20K。为避免 Phase I 期 chain MSE 抖动，10K 已从推荐降为“debug-only smoke”。
SANITY_STEPS=20000 GPU=<free> bash review/0502/scripts/run_ablation.sh sanity
```

代价：~0.42 GPU·day @ 80 steps/min / ~1.0 GPU·day @ 33 steps/min（20K + 20K = 40K 步，与 §5 双列预算表一致）。这一步如果 PASS，脚本写 sentinel `${ABLATION_INDEX_DIR}/.sanity_pass`，C/D 可以放心启动；如果 FAIL，脚本 **exit 1 + 清除 sentinel**，后续调用者 `bash run_ablation.sh main/C/D` 会被 `require_sanity_pass` 拦下（exit 3）。

> **smoke-only 选项**：如只想验证 “脚本能跑起来且能合法处理 §6 下游”，可用 `SANITY_STEPS=10000`，但 **agent2 已警告** Phase I 初期 chain MSE CV 约 60%（V6 step 10K=2.42e-4 → step 20K=9.70e-4 跳变），5% Tier 4 阈值在这个区间假性 FAIL > 30%。agent4 反论是 “A vs B 同 seed 下 trajectory 几乎 bit-equal，10K 差异 << 5%”，但这只在 `_compute_sigma_dt_normalizers` FP32 成原与 B raw `step_weights/n` 除法顺序几乎同指令时才成立。默认选 20K 是保守赌注。
>
> **重跑 sanity**（不同 SANITY_STEPS 或 FAIL 后 debug 重跑）：yaml 里 `require_fresh_output_dir: true` 会拦下 train_first_hop.py，第二次调用会报 `Output dir already exists`。v3.1 脚本加了显式逃生门：
> ```bash
> SANITY_FRESH=1 SANITY_STEPS=20000 bash review/0502/scripts/run_ablation.sh sanity
> ```
> `SANITY_FRESH=1` 只会清除 `${ABLATION_OUTPUT_ROOT}/{A_sanity,B}/`，**不会动 main / A_main / C / D 的产出目录**。未设该变量且后者已存在 → 脚本 exit 4 + 提示。

> **agent2 提示（历史）**：10K 在 V6 Phase I 早期 noise 下 chain MSE CV 约 60%（V6 step 10K=2.42e-4 → step 20K=9.70e-4 跳变），mini full sanity 的 chain Tier 4 阈值 5% 在此区间 **可能被训练噪声淹没**。变通方案二选一：
> - 方案 A（v3.1 改为默认）：SANITY_STEPS=20000，避开 chain 高振荡区，代价 ~0.42 GPU·day @ 80 steps/min / ~1.0 GPU·day @ 33 steps/min
> - 方案 B（备选，未实施）：保持 10K 但 comparator 改为在 last-5 eval 上取 mean；agent4 认为与 “strict-fail-on-missing-keys” 已堆叠，属过度工程化

### 3.3 sanity 不通过的应对

如 Tier 1–3 中任一项 FAIL：
1. **立即 kill GPU 0 上的 C_uniform / D_closed_form**（结果作废）
2. 检查 [`train_first_hop.py`](../../train_first_hop.py) 中 [`_compute_sigma_dt_normalizers`](../../train_first_hop.py)（已在 live code @ L395）的 yaml 落地点：`relative_to`、`anchor_step_weights`、`mode` 是否正确从 yaml 解析
3. 检查 [`pet_lr/rollout_first_hop.py`](../../pet_lr/rollout_first_hop.py) 中 `step_normalizers` 是否正确传入并按 hop 索引除法
4. 修复后，重新运行 `verify_normalizers.py` 确认 FP64 数学等价，再跑 sanity

> **patches 状态澄清**（agent3 提示）：当前 live `train_first_hop.py` / `rollout_first_hop.py` **已包含** σ-normalize 实现（grep `_compute_sigma_dt_normalizers` 确认）。AUDIT_RESPONSE_v4.md 中"先 apply patches"的描述已过时，操作员**不要**再 apply patches/*.patch；那些只是历史 diff 工件。

### 3.4 sanity 通过的标志

GPU 2 上 light A→B 串行约 1 GPU·day 后看到（[`run_sanity_light.sh`](scripts/run_sanity_light.sh) comparator 输出，仅 Tier 1-3）：

```text
metric                              A              B    rel_err threshold
------------------------------------------------------------------------
val_pair_total           x.xxxxxxe-04   x.xxxxxxe-04     0.00%  1e-04 PASS
val_rollout_total        x.xxxxxxe-04   x.xxxxxxe-04     y.yy%  1e-02 PASS
val_rollout_step_0_raw   ...                                            PASS
val_rollout_step_1_raw   ...                                            PASS
val_rollout_step_2_raw   ...                                            PASS
val_rollout_step_3_raw   ...                                            PASS
Result: PASS
```

随后 mini full sanity（[`run_ablation.sh sanity`](scripts/run_ablation.sh) v3）输出全四 Tier：

```text
metric                              A              B    rel_err  tier_threshold
----------------------------------------------------------------------------
val_pair_total           x.xxxxxxe-04   x.xxxxxxe-04     0.00%  Tier 1 (bit-equal) -> PASS
val_rollout_total        x.xxxxxxe-04   x.xxxxxxe-04     y.yy%  Tier 2 (<1%)       -> PASS
val_rollout_step_*_raw   ...                                    Tier 3 (<1%)       -> PASS x4
val_chain_d20_mse        x.xxxxxxe-04   x.xxxxxxe-04     z.zz%  Tier 4 (<5%)       -> PASS
val_chain_d10_mse        ...                                    Tier 4 (<5%)       -> PASS
val_chain_d4_mse         ...                                    Tier 4 (<5%)       -> PASS
val_chain_normal_mse     ...                                    Tier 4 (<5%)       -> PASS

Result: PASS (all tiers under threshold)
```

全 Tier PASS 后启动 C/D。如 FAIL，脚本本身 `exit 1`，可接入 `&& bash run_ablation.sh C` 这样的 chain 控制中当动作 gate（若 A_main 已提前跑过，避免用 `main` 重跑 A_main）。

---

## 4. V6.1 是否继续跑（GPU 1 决策点）

### 4.1 当前轨迹（v2 修正：基于实测头对头数据）

V6.1 与 V6 在 **`val_pair_total`** 上 bit-equal（pair channel 不被 lambda_start 影响），但 **`val_chain_normal_mse`** 上 V6.1 在 Phase I-II 早段显著优于 V6：

| step | V6 chain_normal | V6.1 chain_normal | V6.1 优势 |
|-----:|---------------:|-------------------:|----------|
| 10 000 | 2.42e-4 | 2.33e-4 | **−3.7%** |
| 20 000 | 9.70e-4 | 6.21e-4 | **−36.0%** |
| 30 000 | 1.13e-3 | 9.35e-4 | **−17.4%** |
| 40 000 | 5.09e-4 | 4.35e-4 | **−14.5%** |
| 50 000 | 4.23e-4 | 3.98e-4 | **−5.9%** |
| 60 000 | 3.13e-4 | 3.13e-4 | 0.0% |
| 62 800 | 3.38e-4 | 3.35e-4 | −0.9% |

**157 个共同 val step 上的头对头**：V6.1 在 `val_chain_normal_mse` 上赢 **135 / 输 12 / 平 10**（除平局后胜率 **91.8%**）；在 `val_select_score` 上赢 **144/10/3**（胜率 93.5%）。

### 4.2 实际 lambda 梯度占比（v2 修正）

V6.1 yaml 注释期望 floor=0.05 提供"显著加速 Phase I"。实测 999 个 train events（step < 50K）的 `roll_frac`：

```
mean   = 1.231%
median = 0.800%
P95    = 3.400%
max    = 18.900%
```

→ 实际占比 ≈ **1.2%**（v1 文档误写为 0.3%，agent4 修正）。这与数学预测一致：

$$\text{expected roll\_frac} = \frac{0.05 \cdot L_r}{15 \cdot L_p + 0.05 \cdot L_r} \approx \frac{0.19}{15.19} \approx 1.25\%$$

1.2% 高于 SGD 噪声底，**确实给 chain anchoring 提供了一个非平凡的梯度信号**——这与 §4.1 实测的 Phase I 91.8% 胜率吻合。

### 4.3 如何评价 V6.1 hypothesis（v2 修正立场）

**v1 误判**："hypothesis 半证伪 / 让 V6.1 跑完是为了否定证据"。

**v2 真实情况**：
- ✅ Phase I (step 10K-50K) V6.1 chain MSE 持续优于 V6（最大 36% 改善）
- ✅ 这是 floor=0.05 提供的早期 chain anchoring 收益
- 🟡 step 60K 起两者趋同——V6 进入 Phase II ramp 后 λ_roll 自动起到同样作用，V6.1 的早期优势被覆盖
- ❓ Phase III (150K-200K) final 表现 **未知**，需 V6.1 跑完才能下定论

**正确解读**：V6.1 验证了 "Phase I floor 提供 early stabilization"，但 **未证明** 这种 stabilization 会转化为 final ckpt 的优势。V6 通过更长的 ramp 已经获得了相同的端到端效果，floor 是冗余的还是互补的需要 final 数据。

### 4.4 决策：让 V6.1 跑完 200K（推荐）

**理由**：
- 拿到 final ckpt 数据点，论文里可以写"floor=0.05 在 Phase I 提供 chain anchoring 收益（91.8% 胜率），但 Phase II ramp 后趋同，final 性能差异 < X%"——这是一个**精确**的发现，无论 X 多大都可写入 paper
- GPU 1 不阻塞主 ablation（GPU 0 跑 A/C, GPU 2 跑 sanity）
- V6.1 还剩 ~1.4 GPU·days

**预期 final 结果**：V6.1 chain_normal @ 200K 与 V6 差 < 5%（即 ~1.6e-4 ~ 1.8e-4 区间）。

### 4.5 V6.2/V6.3 候选（仅在 V6.1 跑完后才决定）

**条件**：V6.1 final 显著优于 V6（rel < −5%）→ floor 设计有 final 收益，值得 sweep 找最优 floor 值；或 V6.1 final ≈ V6 且 ablation 节需要更多 floor 数据点 → V6.2 作为 robustness check。

| 候选 | 改动 | 实测预期 roll_frac | 风险 |
|------|------|-------------------:|------|
| V6.2 | `lambda_start: 0.20` | ~5.0% | 早期 chain 加速更明显，但可能在 ramp 末端突变剧烈 |
| V6.3 | `lambda_start: 0.10 + warmup_ratio: 0.10` | ~2.4% but Phase I 短 | 让 ramp 在 step 20K 启动，加速 chain 暴露 |
| V6.4 | `pair_weight: 8.0 + lambda_start: 0.10` | ~2.4% | 需重跑 V6.4 baseline 校正 pair channel |

> 若 V6.1 final 与 V6 差异不显著，整个 V6.x rollout floor 系列 **结案**，不开 V6.2/V6.3/V6.4，paper narrative 直接锁定为"floor=0.05 在 Phase I 有收益但被 Phase II ramp 抹平，故 V6 baseline 已足够"。

---

## 5. 时间表（v3.1 修正 — V6.1 时间数学 + 吞吐量现场验证）

```text
T+0h    V6 跑完 (~850 步) → 从 metrics.jsonl 固化 best.pt 数字 + eval PSNR/seam
        + §1.3 EMA CV 快算（不占 GPU）

T+1h    GPU 0: 启动 A_main (120K, 不依赖 sanity)        + GPU 2 sanity light 继续
        + GPU 1 V6.1 独立线（现在 step 62950，跑完 137K 需 ~1.2d @80 / ~2.9d @33）
T+1.2d  V6.1 跑完 200K @ 80 steps/min → GPU 1 释放，V6.1 final ckpt 可作 paper 补馈处理（§4.3-4.5）
T+1d    A_main 完成 + sanity light A 完成 → 启动 sanity light B
T+2d    sanity light A→B 完成（PASS/FAIL）
        → 通过则做 §3.2 mini full sanity (20K, ~0.42/~1.0 GPU·day, 写 sentinel)
T+2.7d  mini full sanity 完成、通过，启动 GPU 0 = C_uniform
T+3.7d  C_uniform 完成 → 主 ablation 出 A vs C 数据
T+4.7d  可选：D_closed_form 完成
T+5d    开始写 paper σ-norm 节
```

**预算表**（与 [README.md](README.md) §快开始 §GPU·days 对齐）：

| 项目 | 步数 | @ 80 steps/min | @ 33 steps/min（保守）|
|------|----:|---------------:|------------------------:|
| sanity light (A_50K + B_50K) | 100K | ~1.0 GPU·day | ~2.4 GPU·days |
| mini full sanity (20K + 20K) | 40K | ~0.42 GPU·day | ~1.0 GPU·day |
| 主 ablation (A_120K + C_120K) | 240K | ~2.5 GPU·days | ~6.1 GPU·days |
| V6.1 续跑 (剩余 137K) | 137K | ~1.2 GPU·days | ~2.9 GPU·days |
| 可选 D_closed_form (120K) | 120K | ~1.25 GPU·days | ~3.0 GPU·days |
| **总和必选** | | **~5.1 GPU·days** | **~12.4 GPU·days** |
| **总和含 D** | | **~6.4 GPU·days** | **~15.4 GPU·days** |

如有 2 张 GPU 并行（GPU 0 跑 A→C, GPU 2 跑 sanity, GPU 1 跑 V6.1），实际 wall-clock @ 80 steps/min ≈ **2.5–3 天**；@ 33 steps/min ≈ **5–6 天**。

> **吞吐量不确定性 caveat**（agent4 v3.1 修正）：[README.md L116-119](README.md) 声明 A100 80GB bs=8 ≈ 65–80 steps/min 是 “V6 实测”，但快照本身不含 wall-clock 数据。agent4 指出 V6.1 剩 137K 步 @ 80 steps/min 仅需 1.19 天；原 v3 时间表却写 V6.1 在 T+2.9 天才跑完，隐含实际吞吐量 ≈ 33 steps/min（4× 偏偏令 README 处于 “偏高估计”）。
>
> **现场验证动作**：
> 1. V6 跑完后立即从 V6 metrics.jsonl 反推实测 steps/min（`step / wall_time` 可从训练脚本内部计时字段获得）——这是本项目唯一可靠的吞吐量 source of truth。
> 2. A_main 头 10K 后从新的 metrics.jsonl 重估 steps/min；若 < 50 steps/min 告警，主 ablation 预算需从 2.5 天上调。
> 3. 上表 “@ 33 steps/min” 一列提供“最坏情形”边界，以防实际 < 80 steps/min。

---

## 6. 论文 narrative 写作锚点

**当前数据已经支撑的 claim**（V6 best.pt @ step 185600，v3 修正）：
- pair_weight=15 + λ_roll=0→4 ramp + EMA 收敛到 `val_select_score ≈ 0.000607`（best_metric=`val_multi_objective`），`val_chain_normal_mse ≈ 1.70e-4`
- **与 V3 200K baseline (chain_normal=0.000165, [review/0418/A_C_scheme_analysis](../0418/A_C_scheme_analysis_20260418.md)) essentially tied**：V6 best.pt **略差 +2.6%**（0.000170 vs 0.000165）；V6 chain_normal min @ step 186000 = 1.64e-4 略好 −0.6%（但 step_*.pt 未在该点保存）。**不能将 V6 包装为 “V6 优于 V3”**。
- alpha+λ_roll 同步 ramp 是稳定的（200K 全程无 divergence）
- **V6 vs V3 的真正差异在 non-MSE 维度**（v3 增补 paper-positioning 证据采集计划）：
  1. **视觉质量**：抽 8–16 张 chain rollout 切片（V6 best.pt vs V3 best.pt）作 side-by-side PSNR/SSIM/LPIPS × hop 表格与 patch 图，重点在 hop2/hop3 末段及 seam 区域
  2. **EMA CV 稳定性**：从 V6/V3 metrics.jsonl 提取 last-50K 窗口内 chain_normal 的 coefficient of variation（mean / std）；预期 V6 < V3（换句话说 V6 同样的 chain MSE 但训练动态更干净），是 transport-first design 的微收益
  3. **OOD slice robustness**：eval val 切片带 PSNR_clip3 尾部 P10/P05（难 hop 验证），与 V3 best.pt 同表对比；如 V6 在 P10/P05 表现优于 mean，说明 “cleaner Phase III dynamics” 在 long-tail 上有增量

> **v1 误声明 + v2 未修正**（v3 黄牌警告）：v1 写 “V3 baseline 是 ～2e-4，V6 提升 ～18%”是错的；v2 改为 “V6 略好 −3.0%” 仍然错了（方向反了）——多份历史文档一致 V3 best chain_normal=0.000165，V6 best.pt chain_normal=0.000170，diff = +2.6%。**Paper framing 必须是 “V6 matches V3 chain MSE” 而非 “V6 outperforms V3”**。若必须跟据一个 chain_normal 略优的数字，必须明言使用的是未保存的 step 186000 谷点 (1.64e-4)，并在 paper appendix 列出 metrics.jsonl 原始数据以供 reviewer 复核。

**主 ablation 完成后将支撑的 claim**：

| 结果组合 | paper narrative |
|---------|----------------|
| sanity Tier 1-4 PASS + C ≈ A | σ²·dt² 替代解释成立（**caveat**：仅在 pair_loss_weights[0]=2.5 头重前提下，且仅 120K 压缩 schedule 验证） |
| sanity PASS + C 显著差 vs A + D ≈ C | step_weights shape 重要，但 closed-form Grönwall 也不优；写 "V6 weights 是经验最优" |
| sanity PASS + C 显著差 + D ≈ A | step_weights shape 重要且与 closed-form 一致；写 "PET 的 hop-level 难度 ≈ Grönwall 几何" |
| V6.1 final ≈ V6 (< 5%) | floor 在 Phase I 有 early stabilization 但被 Phase II ramp 抹平；省略 V6.2 |
| V6.1 final 显著优于 V6 | floor 设计有 final 收益，启动 V6.2 sweep；推迟 paper |

**最坏情况**：sanity light Tier 1-3 FAIL → 整个 ablation 作废，需 debug 后重跑（+ 1.5 GPU·days）。这一项已由 4-agent 交叉审计 + [`verify_normalizers.py`](scripts/verify_normalizers.py) 的 FP64 验证（残差 2.71e-20）覆盖，**风险低但非零**——FP32/cuda 数值精度在长训练下的累积偏差只能由 mini full sanity (Tier 4) 排除。

### 6.1 论文必须遵守的写作纪律（codex 2026-05-03 review 固化）

**禁止表述**（来自 codex review §9）：

- ❌ "V6 proves Grönwall optimality"
- ❌ "V6 is clearly better than V3"
- ❌ "Uniform result at 120K proves final 200K behavior"
- ❌ "D_closed_form is a theorem-level optimum"
- ❌ "Chain NORMAL MSE can be directly compared to D20 MSE without scale normalization"

**推荐表述**：

- ✅ 方法描述：*"We isolate rollout-channel step weighting under a fixed pair-supervised transport setup. A σ-normalized coordinate system removes the natural (σ·dt)² scale from per-hop latent rollout errors. The A/B sanity condition is algebraically equivalent to the raw V6 rollout objective and verifies the implementation path before testing uniform and closed-form hop weighting."*
- ✅ 若 C ≈ A：*"Under the current pair-heavy V6 setup and a 120K compressed schedule, rollout step-weight shape contributes little beyond scale compensation."*
- ✅ 若 C 显著差 vs A：*"After scale normalization, V6's middle-heavy weighting remains beneficial, suggesting a genuine hop-shape effect beyond (σ·dt)² magnitude compensation."*

**确定性 caveat**（来自 codex Risk D）：训练 set `deterministic: true`，但 [`train_first_hop.py:49`](../../train_first_hop.py) 实际是 `torch.use_deterministic_algorithms(True, warn_only=True)`——`memory_efficient_attention` 与 `adaptive_avg_pool2d` backward 仍是非确定算子。所以：

- A/B 不应期待 bit-identical
- Tier 1 阈值 `<1e-4 rel` 是 *经验等价* 而非 *bit-equal*
- A_main 与 C_uniform 单 seed 比较的 ±2% 浮动归入"非确定噪声"区间，不应解读为信号；> 5% rel 才视为有效差异

**Pair channel confound caveat**（来自 codex Risk B）：当前 `loss.pair_weight=15`、`pair_loss_weights=[2.5,1,1,1]` 已让 pair/hop0 通道很强。若 C ≈ A，正确 narrative 是 *"under the current pair-heavy setup"*，**不要**外推到"hop weighting is universally unnecessary"。pair-uniform 的小规模复核留作 future work；不阻塞当前 paper。

**V6 best vs last 不稳定 caveat**（来自 codex Risk F → 2026-05-03 后期 root-cause 分析修正）：V6 best.pt @ step 185600 chain_normal=0.000170，last.pt @ step 200000 chain_normal=0.000355（+108%）。**这并非训练失稳，是 eval methodology pitfall**——见 §6.2 详细分析。论文表中：

- 报 best.pt 整行（按 §1.2 python snippet）
- **同时**报 last.pt 行作为稳定性附录
- 不把单点 best 包装为稳定收敛
- **更重要**：paper 表的所有数字必须用 `eval_first_hop_224_clip3.py --max-slices 0`（full-val）而非 metrics.jsonl 的 best.pt 那一行（后者是 64-batch rolling window，约 25% 噪声）

### 6.2 ⚠️ Eval methodology pitfall: rolling-window noise（root cause of "V6 best vs last +108%"）

> **深入分析**：见 [ROLLING_WINDOW_TRADEOFF.md](ROLLING_WINDOW_TRADEOFF.md)（数学/代码/架构三维 Pareto 分析 + best.pt 极值选择偏差 landmine 暴露 + paired diff 判读规则补强）。本节给出**操作纪律**，不重复推导。

**结论先**：V6 best.pt vs last.pt +108% **不是训练失稳，主要是 eval window sampling artifact**。

**证据链**：

1. V6 yaml 设置：`max_val_batches: 64`、`val_window_mode: rolling`（[`train_first_hop.py:842-849`](../../train_first_hop.py)）
2. val_loader 总 **3648 batches** × `batch_size=8` = 29184 val samples（确认自 V6 metrics.jsonl 的 `val_main_window_start_batch` 字段，覆盖 0, 64, 128, ..., 3648 共 57 个 disjoint 窗口）
3. **每次 eval 只看 64 batches ≈ 512 samples**——57 个窗口循环周期 = 22800 步
4. metrics.jsonl V6 step 70000-82800 区间内 33 次 eval（33 个不同 window）的 `val_chain_normal_mse`：
   - mean = 2.70e-4, std = 6.90e-5, **CV = 25.5%**
5. 即同一时段、模型几乎不变的情况下，仅 window 切换就让 chain_normal 在 ±25% 范围波动
6. step 75600 (1.95e-4) → step 76400 (3.69e-4, +90%)；step 81600 (1.82e-4) → step 82000 (3.10e-4, +70%)——**400 步内模型不会变这么多，是窗口子集"运气"**

**含义**：

- V6 best.pt @ step 185600 的 chain_normal=1.70e-4 **可能是"幸运 window"** 选出的偏低值
- last.pt @ step 200000 的 chain_normal=3.55e-4 **可能是"困难 window"** 的偏高值
- 数学上的 +108% 差距大概率被 ~25% 窗口噪声 ×2σ ≈ 50% 之外的部分解释，**真正的训练退化（如有）幅度小于看起来的 50%**
- best_metric 选 ckpt 时也部分在选"哪个 step 的 window 更好"，不只是"哪个 step 模型更好"

**数学/架构/逻辑核查（用户问题：是否有 bug）**：

| 检查 | 结果 | 证据 |
|---|---|---|
| Mean-flow 更新 `z_pred = z_src + σ·v_raw·dt` | ✅ 正确 | [`model_first_hop.py:505-511`](../../pet_lr/model_first_hop.py) |
| 加权平均 not 加权和 | ✅ 正确 | [`rollout_first_hop.py:117-119`](../../pet_lr/rollout_first_hop.py) |
| EMA 包装 best.pt | ✅ 正确 | [`train_first_hop.py:2511-2532`](../../train_first_hop.py) |
| `val_select_score` 写入 metrics.jsonl | ✅ 正确 | [`train_first_hop.py:2446`](../../train_first_hop.py) |
| 确定性 algorithms `warn_only=True` | ⚠️ 已知 | [`train_first_hop.py:49`](../../train_first_hop.py) |
| **rolling window with 64/3648 ratio** | **⚠️ ROOT CAUSE** | [`train_first_hop.py:838-849`](../../train_first_hop.py) `_resolve_eval_window` |

**没有数学、架构、逻辑 bug**。问题在 eval methodology——以速度优先（每次 eval 只 64 batches × ~5s ≈ 320s = 5 min）牺牲了 chain MSE 的低噪声估计。这是合理的训练时取舍，但 **paper 时必须切换到 full-val**。

**Paper 时强制纪律**：

```bash
# Step 1: Method D 选 ckpt（邻域平滑，避开 best.pt 极值偏移）
python review/0502/scripts/select_best_ckpt_smoothed.py \
    --metrics /data_2/.../A_main/run-.../metrics.jsonl \
    --ckpt-dir /data_2/.../A_main/run-.../ \
    --neighborhood 10
# 输出推荐 ckpt step (e.g. step 119800) + 次推荐

# Step 2: 对推荐 + 次推荐的 ckpt 跑 full-val
for STEP in <recommended> <runner-up>; do
    python eval_first_hop_224_clip3.py \
        --config review/0502/configs/A_control.yaml \
        --checkpoint /data_2/.../ckpt_step_${STEP}.pt \
        --split val \
        --max-slices 0
done
# Paper 表数字 = mean ± std 于这两个 step 的 full-val
```

不要从 metrics.jsonl 的 best.pt step 取数字。改用 full-val eval：
python eval_first_hop_224_clip3.py \
    --config review/0502/configs/A_control.yaml \
    --checkpoint /data_2/.../A_main/run/.../best.pt \
    --split val \
    --max-slices 0   # 0 = full val set, no rolling
# 同样跑 last.pt：
python eval_first_hop_224_clip3.py \
    --config review/0502/configs/A_control.yaml \
    --checkpoint /data_2/.../A_main/run/.../ckpt_last.pt \
    --split val \
    --max-slices 0
```

得到的 PSNR_clip3 + chain MSE 才是 paper 表数字。**A/C/D 的 best.pt 比较也必须走这一步**——training metrics.jsonl 的 best.pt 行同样有 25% rolling 噪声，不能直接用作 ablation 主表。

**Paper 时强制纪律（追加 v3.1 follow-up）**：

- ✅ A/B/C/D 主表 chain MSE / PSNR_clip3：**必须** 用 `eval_first_hop_224_clip3.py --max-slices 0`
- ✅ same-step paired 比较：在 metrics.jsonl 中找 A 与 C **共同存在** 的 step（同一 window）做 rel diff，比 best.pt 比较更鲁棒（因为 A_main 与 C_uniform 同 schedule、同 eval_interval、同 RNG → 同步遍历相同 window）
- ❌ 不要直接抄 metrics.jsonl 的 best.pt 行进 paper 表（每条 line 都是 64-batch rolling window 局部估计）
- ❌ 不要把 V6 metrics.jsonl 的 step 185600 chain_normal=1.70e-4 与 step 200000 chain_normal=3.55e-4 的对比写入 paper 当 "training instability" 证据——它主要是 window noise

**附加保险（可选，不阻塞）**：A_main / C_uniform / D_closed_form yaml 在跑 main ablation 时**可以临时把 `max_val_batches` 提到 256**（从 64×4 倍）这样 eval 一次看 ~2000 samples，CV 降到 ~12-15%，chain MSE 数字更稳。代价：每次 eval 从 5 min 涨到 20 min × 300 evals = 1.7 GPU·hour 额外开销，可接受。但即使如此，paper 数字仍要走 `--max-slices 0` 全 val 复测。

### 6.3 Risk 3 (120K compressed schedule) 处理纪律

**事实**：A/C/D 用 120K，V6 用 200K。warmup_ratio/ramp_ratio 同为 0.25/0.50：

| Run | Phase I (warmup) | Phase II (ramp) | Phase III |
|---|---:|---:|---:|
| V6 | 0-50K | 50K-150K | 150K-200K (50K) |
| A_main 120K | 0-30K | 30K-90K | 90K-120K (**30K**) |

**含义**：A_main 在 Phase III 仅训 30K 步（vs V6 的 50K）。这是 GPU·day 预算下的合理取舍，**但 claim scope 必须收敛**。

**处理（不修代码，做 discipline）**：

- paper 主 narrative 必须显式写 *"120K compressed schedule, controlled comparison within ablation"*，**不外推到** *"final convergence behavior at 200K"*
- 灰区决策（rel diff ∈ [5%, 10%]）→ **pre-commit** 扩 200K，不允许事后改口"已经够了"
- 灰区扩 200K 的实现：从 120K best.pt resume 继续训练 80K 步，新 yaml 设 `warmup_steps: 0, ramp_steps: 0, lambda_start: 4, lambda_end: 4, alpha_start: 1, alpha_end: 1, max_steps: 80000`（即直接进入 Phase III）。代价 = 0.7 GPU·day per condition
- **不**修改当前 A/C/D yaml 的 warmup/ramp ratios（修改会破坏 sanity 与 main 共用 yaml 的对称性，且 sanity 20K 已与 ratios 锁死）

**为什么不能"把 120K 的 ratio 改小让 Phase III 同样 50K"**：A_control.yaml 同时被 sanity（20K override）与 main（120K override）使用，ratios 在 yaml 里。如果改 A_control.yaml 的 ratios，sanity 也跟着变，破坏 A/B 等价（B_sanity.yaml 也会受影响）。若一定要做，需要拆分两个 yaml；当前优先级低。

### 6.4 Risk 4 (pair channel confound) 处理协议

**事实**：`loss.pair_weight=15`、`pair_loss_weights=[2.5, 1.0, 1.0, 1.0]`——pair channel 在 hop0 已经强。若 C ≈ A，无法区分：

- 假说 H_pair：pair 通道吃掉 rollout shape 的能量，所以 shape 变化没显示
- 假说 H_shape：rollout shape 真的不重要

**处理**：A/C 主 ablation 完成后，跑 [`A_pair_uniform_spot.yaml`](configs/A_pair_uniform_spot.yaml)（60K, seed=42, pair_loss_weights=[1,1,1,1]，其余与 A_main 完全一致）。

**启动**：
```bash
GPU=<free> bash review/0502/scripts/run_ablation.sh pair_uniform
```

**判读规则**（LOCKED 2026-05-03）：使用 [`paired_diff_judge.py`](scripts/paired_diff_judge.py) 在共同 step 窗口上对 `val_chain_normal_mse` 做 paired 比较，输出 mean ± SE ± CI95 + autocorr-corrected N_eff。**绝对不要用 best-vs-best**——其 3σ 检测下限 ~76%，根本测不出 confound 信号。

```bash
# 推荐窗口：[40000, 60000] —— 两 run 都已进 Phase II 中段，schedule 状态对齐
python review/0502/scripts/paired_diff_judge.py \
    --metrics-a /data_2/qujiaxiang/outputs/PET_LatentResidual/A_main/run-.../metrics.jsonl \
    --config-a  review/0502/configs/A_control.yaml \
    --metrics-b /data_2/qujiaxiang/outputs/PET_LatentResidual/A_pair_uniform_spot/run-.../metrics.jsonl \
    --config-b  review/0502/configs/A_pair_uniform_spot.yaml \
    --metric-key val_chain_normal_mse \
    --step-min 40000 --step-max 60000 \
    --threshold 0.10 \
    --label-a A_main --label-b A_pair_uniform \
    --output-md review/0502/runs/risk4_paired_diff.md
```

**LOCKED decision rule**（脚本自动应用，使用 mean ± 2·SE 区间与 0.10 阈值比较）：

| 脚本判定 / exit code | 含义 | Paper 行动 |
|---|---|---|
| `no_confound` (exit 0) | `\|mean\| + 2·SE < 0.10`<br>(mean rel diff 95% CI 完全在 ±10% 内) | C ≈ A 结论 paper 可保留 *"under the current pair-heavy setup"* 措辞，并在附录引用本 spot check 报告作 robustness 证据 |
| `confound` (exit 10) | `\|mean\| − 2·SE > 0.10`<br>(mean rel diff 95% CI 完全在 10% 外) | pair_weight[0]=2.5 是主驱动。Paper narrative **必须**重新框架为 *"pair-channel weighting is the primary lever; rollout step-weight shape's contribution is conditional on pair-channel configuration"* + 报告 magnitude=`mean` |
| `borderline` (exit 11) | CI95 跨过 ±10% 阈值 | 对两 run 的 best.pt 各跑一次 `eval_first_hop_224_clip3.py --max-slices 0`（Layer 4 full-val），取该数字判读；如仍 borderline，paper supplementary 同时披露两个数字 |

**Guards 自动检查**（脚本里硬执行，不要试图绕过）：

1. metric-key 在 A/B 两文件至少 5 个共同 step（`exit 1`）
2. autocorrelation-corrected N_eff ≥ 5（防止 mean 不稳定，`exit 2`）
3. step-min < step-max（强制窗口显式指定，`exit 3`）
4. mean ± 2·SE 同时跨过 0 和阈值时一律 borderline，不允许"勉强通过"
5. A/B 两 yaml 必须在 seed / val_shuffle / val_window_mode / eval_interval / max_val_batches / batch_size 上完全一致（`exit 5`）—— 这是 paired diff 真实成立的硬条件

**为什么 0.10 阈值合理**（详细推导见 [ROLLING_WINDOW_TRADEOFF.md §3.4 与本节 audit](ROLLING_WINDOW_TRADEOFF.md)）：

- 单次 paired diff CV ~5%（理论估计）；窗口 50 个 paired 观测，自相关修正后 N_eff ~10-20 → mean 的 SE ≈ 1.5-2%
- 0.10 阈值在 mean SE 上是 5σ 距离 → 统计功效充足
- 0.15 阈值是 8σ → 功效过剩，会漏掉真实 8-12% confound
- Risk 4 是**防御性检查**：false negative（漏报 confound）的代价是 reviewer 抓住 → desk reject；false positive 的代价仅是多写一段披露 → **应当倾向高敏感度**

**成本**：60K @ 80 steps/min ≈ 12.5 GPU·hour ≈ 0.5 GPU·day。**单 seed 即可作 robustness check**（不要求 seed 复跑——这是 confound check，不是 main claim）。

**Fallback**：若 paired_diff_judge.py 因任何 guard fail（如 N_eff < 5），不要回退到 best-vs-best；改为对 A_main / A_pair_uniform_spot 各跑一次 Layer 4 full-val（`eval_first_hop_224_clip3.py --max-slices 0`，每 run ~30 min 单 GPU）→ 直接比较两个 unbiased 数字。

### 6.5 Risk 5 (hop residual claim scope) — paper framing 校准

**事实**：[`HopResidualVelocityHead.lambda_hop_init=1e-3`](../../pet_lr/model_first_hop.py)，V6 训练日志 `lambda_hop_*` / `v_hop_abs` 一直在 1e-2 量级——hop residual 的 latent-space 贡献始终很小。**这不是 bug，是设计**：用户已确认 "Hop residual 的作用主要是 decoder 的伪影压制"——它在 latent space 是小幅 correction，但在 decoded image 上抑制 cross-hop seam artifact。

**Paper framing 校准**（不动代码，改写作）：

- ❌ 不要写 *"We propose a hop-residual velocity head as a core architectural contribution"*——证据强度不足
- ✅ 改写 *"Our architecture augments the shared backbone velocity with a small hop-conditioned residual head (initialized to λ_hop ≈ 1e-3) whose primary role is to suppress decoder seam artifacts at hop boundaries; latent-space contribution remains small (|v_hop| ~ 1e-2 throughout training)."*
- ✅ 论文 section 中将 hop residual 描述为 "implementation detail / safety branch" 而不是 "main innovation"
- 如果 reviewer 要求量化证据：可补 1 个 60K ablation `lambda_hop_init=0`（关掉 hop residual）。decoded image 的 seam region PSNR_clip3 / LPIPS 是关键指标，预期会变差；latent chain MSE 预期变化 < 5%。**这个补实验不阻塞当前 paper**，但作为"可选 robustness"留底。

---

### 6.6 σ-normalize ablation 的 effect-size 阈值（blinded analysis pre-registration, LOCKED）

**问题背景**：σ-normalize ablation 完成后，A_main 与 C_uniform 的 `chain_normal_mse` 相对差 `rel_diff = (C - A) / A` 是核心 paper 数字。但单一阈值（"X% 以下视为 indistinguishable"）必须在**看到 C 结果之前**锁死，否则等同 p-hacking → reviewer desk-reject 风险。

**为什么不直接锁 X = 10%**：单点估计来自 V6 snapshot（rolling-window CV ≈ 25.5% / paired CV ≈ 5%），但 A_main 是新一轮 200K 训练，实际噪声可能不同。如果实测 paired CV = 8%，X=10% 就太严；反过来若 = 2%，X=10% 又太宽松。**用户直觉正确**：阈值应数据驱动。

**为什么不能"看完再定"**：看到 `rel_diff = 8.4%` 之后再选 X=10% 还是 X=8% — 即使本人完全诚实，也无法向 reviewer 证明阈值未被结果污染。这是 ML reproducibility crisis 的标志性反模式。

**解决方案：blinded analysis（盲分析）**——pre-commit 一个**公式**，公式只用 A_main 自己的数据计算（A 是 control，先跑完）；C 的数据**直到 X 锁定后**才允许查看。

#### 6.6.1 LOCKED formula（修改此处任何参数 = scientific fraud）

```text
X = max(0.10,  3 × paired_CV_A)

where paired_CV_A is computed from A_main alone:
  - Read A_main metrics.jsonl
  - Filter rows with step ∈ [40000, 60000]   # post-warmup, pre-Phase-III
  - Series: val_select_score (the same key Method D consumes)
  - paired_CV_A = std(series) / mean(series)
```

**参数 rationale**（now LOCKED, no negotiation post-A_main）：

- **floor 0.10 (10%)**：保护极小 paired_CV_A 的 pathological 情况。如果 paired_CV_A = 1%，formula 会给 X=3%，但 V6 snapshot 上 full-val protocol 的真实分辨率 ~1.5% → 3% 几乎在地板上，paper 不可信
- **slope 3×**：在高斯噪声下 ±3σ 覆盖 99.7% 概率，是物理学/医学 RCT 标准 effect-size 系数
- **window [40000, 60000]**：post-warmup（warmup 通常 ≤ 20K）但 pre-Phase-III（≥ 50K 时 alpha/λ_roll 开始 ramp，会引入 schedule-driven 非平稳）。在 V6 snapshot 上这段也是相对平稳
- **single key (val_select_score)**：与 Method D 一致；不加 chain_normal_mse 的额外阈值是为了避免 multiple-comparison 漏洞

#### 6.6.2 严格执行顺序（顺序错 = pre-registration 失效）

```text
[已发生]   Step 0:  本 commit 写入此 §6.6，git push 到 gitee → pre-registration 时间戳锁定
[在跑]     Step 1:  A_main 与 C_uniform 各自跑到完成（两者并行无干扰）
           Step 2:  A_main 完成后，运行 Method D 选 A_main top-2 ckpts（不动 C）
           Step 3:  从 A_main metrics.jsonl 算 paired_CV_A，代入公式得 X
           Step 4:  把 X 写入新文件 review/0502/EFFECT_SIZE_LOCKED.md（含 paired_CV_A
                    实测值、X 计算式、A_main 的 commit hash），git commit + push
                    ↑↑↑ 这一步的 commit hash 就是"X 在 C 数据未读取前已锁定"的硬证据
           Step 5:  ONLY NOW：跑 Method D + full-val 在 C_uniform → 得到 C 数字
           Step 6:  计算 rel_diff = (C_full_val - A_full_val) / A_full_val
           Step 7:  应用下表（rule 也是 LOCKED）
```

**Step 4 是不可省略的盲分析 commit**——没有它，整个 pre-registration 失效。即使 A_main 跑完后才发现公式有 bug，也**必须**在 Step 4 commit 里说明 deviation，不能事后修改本 §6.6。

#### 6.6.3 LOCKED decision rule

| `rel_diff` | 区间含义 | Paper conclusion / 后续动作 |
|---|---|---|
| `rel_diff < X` | C ≈ A | "Step-weight shape contributes little beyond scale compensation under the current pair-heavy V6 setup; σ-normalize is the primary lever." → σ-norm 写为核心 contribution |
| `X ≤ rel_diff ≤ 2X` | Grey zone | 触发 200K continuation：从 A_main + C_uniform 的 best.pt 继续训到 200K，再用同一 X 重判。如 200K 后仍在 grey zone：paper 报两个数字、不下结论 |
| `rel_diff > 2X` | C ≠ A 且方向明确 | "After σ-normalization, V6 middle-heavy weighting still helps → genuine hop-shape effect." → reframe paper 为 "we validate V6 with rigor"，σ-norm 写为 controlled-comparison methodology |

**注意**：表里的 X 与 2X 都使用 Step 3 算出的具体数字，例如若 paired_CV_A = 0.04 → X = 0.12 → 2X = 0.24。

#### 6.6.4 与现有 protocol 的关系

- 与 §6.2 paper-time discipline 兼容：Method D 选 ckpt（Step 1）+ full-val on top-2（Step 2）→ 输出 mean ± std，再代入这里的 rule
- 与 §6.4 Risk 4 (pair confound) 协议**独立但共用脚本**：那个协议针对 A_main vs A_pair_uniform_spot，不是 A_main vs C_uniform。两者各用自己的阈值（Risk 4 LOCKED 0.10；§6.6 是 `max(0.10, 3 × paired_CV_A)`）。脚本 [`paired_diff_judge.py`](scripts/paired_diff_judge.py) 也可作为 §6.6 的 **early-warning auxiliary anchor**（A_main vs C_uniform 训练中段 monitoring），见 §6.6.5
- `EFFECT_SIZE_LOCKED.md` 也作为 paper supplementary 的 reproducibility 附件（reviewer 可验证 commit hash 时序）

#### 6.6.5 Auxiliary anchor: paired diff monitoring during training（不替代主 protocol）

`paired_diff_judge.py` 在 §6.6 主 protocol（Step 5 full-val on C_uniform）之外可以作为**早期信号**使用：当 A_main 跑到 30K / 60K / 90K 时，可以并行查看 A_main vs C_uniform 在 `[0, current_step]` 的 paired diff，判断是否需要在 A_main 跑完后立刻准备 200K 续训。

**用法（不进 paper，仅决策辅助）**：

```bash
# 推荐窗口 [40000, 当前步] 或 [40000, 60000] 等已有数据的稳定段
python review/0502/scripts/paired_diff_judge.py \
    --metrics-a /data_2/.../A_main/run-.../metrics.jsonl \
    --config-a  review/0502/configs/A_control.yaml \
    --metrics-b /data_2/.../C_uniform/run-.../metrics.jsonl \
    --config-b  review/0502/configs/C_uniform.yaml \
    --metric-key val_chain_normal_mse \
    --step-min 40000 --step-max <current_step> \
    --threshold 0.10 \
    --label-a A_main --label-b C_uniform
```

**关键区别**（与 §6.4 Risk 4 用法相比）：

| 维度 | §6.4 Risk 4 用法 | §6.6.5 auxiliary anchor 用法 |
|---|---|---|
| 输入 B run | A_pair_uniform_spot | C_uniform |
| 数字进 paper 吗 | ✅ 是（exit code 0/10/11 进 §6.4 决策） | ❌ 否（仅作早期 heads-up） |
| 决策权威 | LOCKED 阈值 0.10 | 仅参考；最终判读用 §6.6 主 protocol（Step 6）|
| 何时跑 | A_pair_uniform_spot 完成后一次 | A_main 跑到 30K/60K/90K 各一次（可选）|

**绝对不允许**：用 paired diff monitoring 的早期数字替代 §6.6 主 protocol 的 full-val on C_uniform。脚本 exit code 在 auxiliary anchor 用法里只是"接下来 GPU 排队是否要预留 200K continuation"的提示，不是 paper 数字。

#### 6.6.6 反例：如果不做 blinded analysis 会怎样

| 实测 rel_diff | "看完再定 X" 路径 | Blinded analysis 路径 |
|---|---|---|
| 4% | 选 X=10% 宣告 ≈ ✓（看似无害） | paired_CV_A=2% → X=10% (floor) → 4% < 10% → ≈ ✓ 同结论但**过程可证明** |
| 8% | 倾向选 X=10% → ≈；但 reviewer 会问"为什么不是 X=5%" → 难答 | paired_CV_A=4% → X=12% → 8% < 12% → ≈ ✓ |
| 13% | 倾向选 X=15% → ≈；reviewer 一眼识破 cherry-pick | paired_CV_A=4% → X=12% → 13% ∈ [12%, 24%] → grey zone → 触发 200K（**保护了诚实**） |
| 25% | 选 X=10% → 显著（这次没作弊也无法证明没作弊） | paired_CV_A=4% → X=12% → 25% > 24% → 显著 ✓ **rule-based** |

#### 6.6.7 LOCKED status

- **Pre-registration commit**：本文件改动的 commit hash（写入此处时未知，post-commit 应回填）
- **Pre-registration timestamp**：commit 时刻（git history 永久记录）
- **修改本 §6.6 的合法路径**：只有 1 种 — 在 `EFFECT_SIZE_LOCKED.md` 里以 deviation note 形式说明，并附 deviation 的 reviewer-defensible rationale。**绝对不允许**事后修改本 §6.6 的 formula、floor、slope、window、key、rule

---

## 7. 决策清单（v3.1 修正 — 强化 sanity gate）

按时间顺序：

1. ✅ **等 V6 跑完**（～850 步, < 1 GPU·hour）
2. 🔵 **从 V6 metrics.jsonl 固化 best.pt @ step 185600 数字**（即时，不占 GPU）+ **可选 PSNR/seam eval**（5 分钟，任意空闲 GPU）
3. 🟡 **GPU 2 等 sanity light A→B 完成**（～1 GPU·day，主要覆盖 Tier 1-3）
4. 🟡 **sanity light Tier 1-3 PASS 后做 mini full sanity (20K)**（～0.42 GPU·day @80 / ～1.0 GPU·day @33，覆盖 Tier 4）——脚本会 auto-exit 非零如 FAIL
5. 🟢 **GPU 0 启动 A_main 120K**（V6 完成后立即跑，～1 GPU·day）——**可与 sanity 并行**，因为 A_main 与 B_sanity 使用不同的 step_weights schema、sanity 失败不会污染 A_main
6. ⚠️ **A_main 完成后仅在全部 sanity tier 都 PASS 的前提下**启动 C_uniform 120K（～1 GPU·day）；C 依赖 sigma_dt normalizer code path，那条 path 必须与 A 等价才能化除 confound
7. 🟠 **V6.1 让其跑完 200K**（GPU 1，～1.4 GPU·day）——不依赖 sanity，全程独立进行
8. 🟣 **可选：D_closed_form 120K**（～1 GPU·day，视 GPU 0 是否有空）——也需 sanity PASS
8.5. 🟤 **A_main + C_uniform 主 ablation 完成后：A_pair_uniform_spot 60K**（Risk 4 spot check，～0.5 GPU·day，单 seed=42）——见 §6.4 判读规则
8.6. 📐 **paper 表数字最终化**（任意空闲 GPU，～0.5 GPU·day 总计）：
    - 对 A_main / C_uniform / D_closed_form / V6_resumed 的 best.pt + last.pt 各跑一次 `eval_first_hop_224_clip3.py --max-slices 0`
    - 数字进 paper 表；training metrics.jsonl 里的 64-batch rolling 数字**仅**用于 sanity / monitoring
    - 这一步是 §6.2 强制纪律，**不可跳过**
8.7. 🔒 **Blinded effect-size threshold lock-in（A_main 完成后，C_uniform full-val 之前必须执行）**：
    - A_main 完成 → Method D 选 A_main top-2 ckpts
    - 从 A_main metrics.jsonl 取 step ∈ [40000, 60000] 的 `val_select_score`，算 `paired_CV_A`
    - 代入 §6.6 LOCKED 公式：`X = max(0.10, 3 × paired_CV_A)`
    - 写入新文件 `review/0502/EFFECT_SIZE_LOCKED.md`（含 paired_CV_A、X、A_main commit hash），**git commit + push**
    - **此 commit 是 pre-registration 完成的硬证据；这一步不做，§6.6 失效**
    - **顺序硬约束**：本步必须在 §8.6 对 C_uniform 的 full-val eval **之前**完成
9. 📝 **主 ablation + V6.1 final + spot check + full-val eval 全部完成后开始写 paper σ-norm 节**（应用 §6.6 LOCKED decision rule）

> **与 v2 的关键差异**：v2 说可以在 sanity Tier 1-3 PASS 后同时跑 GPU 0=A_main 与 GPU 1=C_uniform，v3 **只允许 A_main 与 sanity 并行**，C 必须等全部 tier (含 mini full Tier 4) 都 PASS。原因：C = `B_sanity.yaml 上 step_weights normalizer ON` 的生产设置，若 normalizer code path 有 bug，120K 跑出来的 C 数据是污染的，问题只能在最后发现 → +≥2 GPU·days 入坑。

> **如 sanity FAIL**：跳过 6/8，立即 debug `_compute_sigma_dt_normalizers` 与 `step_normalizers` 入口 → 重跑 sanity → 再启动 C/D。脚本从 v3 起会以 exit code 骨崩，不需人手读表。

---

## 8. 修订记录

### 8.1 v1 → v2 修订（agent1/2/3/4 首轮审计）

| 项 | v1 | v2 |
|---|----|----|
| eval CLI 命令 | `--output_dir/--ckpt_step/--use_ema/--eval_split` | `--config/--checkpoint/--split/--out-dir/--decode-mode` |
| EMA 加载方式 | "`--use_ema true`" | best.pt 已是 EMA 参数，无需切换 |
| check_alignment 用途 | "验证 V6 ckpt 在所有 hop 上的 alignment" | RAE/latent 数据级审计；**与 V6 ckpt 无关**，本步骤删除 |
| V6 best ckpt | "step 186K, chain_normal=1.64e-4" | best.pt = step **185600** (val_multi_objective=0.000607, chain_normal=1.70e-4)；step 186K 既不是 best.pt 也不在持久化列表 |
| V6 vs V3 framing | "V3 ~2e-4, V6 提升 18%" | "V6 略好 ~−3%"（仍错误，见 v3）|
| V6.1 hypothesis 立场 | "半证伪 / 否定证据" | Phase I 91.8% 胜率（最大 −36%），Phase II ramp 后趋同；待 final 数据 |
| V6.1 lambda 占比 | "约 0.3%" | 实测 mean **1.23%**, P95=3.4% |
| §2.1 论证 | "A_main 不需要重跑" + "A_main 必须重跑" 自相矛盾 | 仅保留 "A_main 必须独立重跑 120K" |
| sanity script 行为 | "run_ablation.sh sanity 4-tier comparator" | 实际 GPU 2 跑的是 light 版（Tier 1-3 only）；增补 mini full sanity 步骤 |
| GPU·days 预算 | 22 GPU·days / 9 wall-clock days | ～5.8 GPU·days / ～3 wall-clock days（与 [README.md](README.md) 一致）|
| FID 提及 | "chain MSE / FID / 完整 chain rollout" | eval 脚本不输出 FID；删除 |
| patches 状态 | "先 apply patches" | 已在 live code（grep `_compute_sigma_dt_normalizers` 确认），不要再 apply |

### 8.2 v2 → v3 修订（agent3/4 二次审计 + 脚本修复）

| 项 | v2 | v3 |
|---|----|----|
| Agent 1 best.pt step 错误声明 | 未明示驳难 | **明示驳伪**于§0：FP 精度计算证明 step 185600 严格 < step 162400 (−6e-7)，log 亦明言 “new best at step=185600” |
| Agent 1 separation 结论 | 写 "4 步差异是 ckpt-level" | 重写：multi_obj two new-best 间隔 23 200 步 → trajectory-level systematic separation，paper 必须同报两套数字 |
| --decode-mode | `both` | `default`（V6 无 seam_refiner，`both` 的 raw 分支与 default 重复，CSV 冱余、可能误导 reviewer）|
| eval 输出文件名 | "slice_eval_clip3.csv + summary.json" | `first_hop_224_{split}_clip3_eval.json` (含 summary_psnr_clip3 / summary_seam_consistency / summary_extended_seam / per_slice) + `first_hop_224_{split}_clip3_eval.csv`；不输出独立 summary.json |
| rel diff 定义 | 隐含 | 明说：`(C_best.pt[metric] − A_best.pt[metric]) / A_best.pt[metric]`，次指标需同向过阈 |
| C 差 vs A 的解读 | strong claim "PET hop 难度由 σ·dt 主导" | 改为弱 caveat：仅在 pair_loss_weights[0]=2.5 且 120K 压缩 schedule 下成立，不外推 |
| C≠A 的 caveat | 未提 | 备注压缩 schedule × uniform shape 交互偏差，需 200K 补跡 |
| mini sanity 命令 | `MAX_STEPS=10000 ... sanity` | `SANITY_STEPS=10000 ... sanity`（`MAX_STEPS` 仍可用为后兼）|
| sanity FAIL 后果 | "只是打印表" | 脚本本身 `exit 1`，可作为 CI 门控 |
| V6 vs V3 方向 | "V6 略好 ~−3.0%" | **V6 略差 +2.6%，essentially tied**；min@186K = −0.6% 但未保存 |
| paper-positioning 证据 | 仅 "cleaner training dynamics" 表述 | 给出 3 项可量化证据采集：视觉质量 / EMA CV / OOD slice P10/P05 |
| §7 决策顺序 | A_main 与 C_uniform 可同 sanity 并行 | 仅 A_main 与 sanity 并行；C 必须等全 tier PASS |
| 吞吐量验证 | 80 steps/min 直接使用 | 补 caveat：A_main 头 10K 后从 metrics.jsonl 重估 steps/min |
| run_ablation.sh sanity hardcode 50K | 脚本 bug | **修复**：`SANITY_STEPS / MAX_STEPS` 优先级使用 |
| run_ablation.sh comparator | `cat <<EOF` 仅打印 | **修复**：重命名 `print_sanity_check_cmd → run_sanity_comparator`，真正调用 python 且 sys.exit 0/1 |

### 8.3 v3 → v3.1 修订（agent1/2/3/4 三次审计）

| 项 | v3 | v3.1 |
|---|----|------|
| §0 best.pt FP 精度 “-6e-7” 论证 | 具体数值 0.0006069 / 0.0006075 | 说明这些是推测（log 6 位截断），硬证据仅 “new best at step=185600” + `if key < best_val` 路径 |
| §0 step_180000 行 | “待查 metrics.jsonl” | 明言 chain_normal=0.000174，比 best.pt 差 +2.4%，**不适合作为 paper ckpt** |
| §0 重要键名说明 | 未提 | 明言 metrics.jsonl 只写 `val_select_score`，`val_multi_objective` **不是** 键，汇总脚本须读前者 |
| §1 新增 §1.3 | 无 | EMA CV 快算脚本，不占 GPU，为 paper paper-positioning 采集第一项差异化证据 |
| §2.2 与 §7 冲突 | v3 §2.2 仍保留 “Tier 1-3 PASS 后并行 GPU 0=A & GPU 1=C” | 删除并行命令，与 §7 “C 必须等 Tier 1-4” 一致；A_main 仍可与 sanity 并行 |
| §2.2 ckpt 路径 | 未给出 | 增补 A_main / C / D 的硬路径供后续 eval/summarize 使用 |
| §5 V6.1 时间 | T+2.9d 跑完 200K | 重算：@ 80 steps/min 为 1.2 GPU·day；@ 33 steps/min 为 2.9 GPU·day；两列同表 |
| §5 吞吐量 | 80 steps/min 估算 | 两列预算表（80 / 33）+ 明言快照本身不含 wall-time，以 V6 跑完后的 metrics.jsonl 为 source of truth |
| §6 V6 vs V3 narrative | tied + 3 项证据 | 同 v3 + 增补 §1.3 EMA CV 快算作为第一项可马上拿的证据 |
| run_ablation.sh CUBLAS | 未设 | **修复**：`export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"`，与 [`run_sanity_light.sh`](scripts/run_sanity_light.sh) 对齐 |
| run_ablation.sh comparator 缺键 | silent pass | **修复**：缺 Tier 1/2/4 必需键 → overall_pass=False + 输出 missing keys 报警 + Tier 3 < 4 个 raw 键也 exit 1（防 light yaml 误用）|
| run_ablation.sh main/C/D 纵容启动 | 无 gate | **修复**：增 `require_sanity_pass` 函数 + sentinel `${ABLATION_INDEX_DIR}/.sanity_pass`；sanity PASS 写入 / FAIL 清除；main 中 C 拒绝启动除非 sentinel 存在或 ALLOW_UNGATED=1（agent1 v3.1 追调）|
| run_ablation.sh `main` 快捷跱 | 原 未 gate | **修复**：`main` 双门 gate — `require_sanity_pass "main"` 在 A_main 之前运行；A_main、C 串行，重复利用同一 sentinel；需与 sanity 并行跑 A_main 时使用显式 `bash run_ablation.sh A`（无 gate），sanity PASS 后再 `bash run_ablation.sh C`（agent6 codex review 确认这是更保守且不令人意外的设计）|
| run_ablation.sh usage | 只提 ALLOW_UNGATED | 补列 SANITY_FRESH / SANITY_STEPS / MAX_STEPS / GPU 及说明 |
| run_ablation.sh sanity 重跑冲突 | yaml `require_fresh_output_dir: true` 拦下第二次调用 | **修复**：脚本 sanity 分支检查旧输出目录，`SANITY_FRESH=1` 显式逃生门才清除（只动 A_sanity/B，不动 main）|
| §3.2 mini sanity 默认步数 | 10K（agent2 警过 chain CV ~60%） | 默认 20K，10K 降为 smoke-only；agent4 反论也录入文中以供后续复议 |
| §5 时间表 V6.1 行位 | 出现在 T+3.7d 后面，愉必误导“V6.1 等 D 完成才跑” | 移至 T+1h “GPU 1 独立线” + T+1.2d V6.1 完成一行，与 §4.4 “V6.1 不阻塞主 ablation” 一致（agent1 v3.1 追调）|
| §1.2 summarize_run.sh 警告 | 未提 | 增 “⚠️ 不要把 summarize_run.sh 输出当 paper 表数字”警告框：summarize_run 报 per-metric independent minima，paper 必须用 best.pt 行（§1.2 python snippet）（agent6 v3.1 follow-up）|
| §2.2 ckpt 路径行 summarize_run 引用 | 写 “可用 summarize_run.sh 生成 best ckpt 指标表” | 改为 “paper 表用 §1.2 python snippet；summarize_run 仅做 per-metric 独立最小值的快速诊断”（agent6）|
| run_ablation.sh require_sanity_pass freshness | 仅 `[ -f sentinel ]`；旧 sentinel / 已删除 metrics / 10K smoke sentinel 都会放行 | **修复**：require_sanity_pass 解析 `sanity_steps=N` 与 `a_metrics= / b_metrics=` 路径；强制 N ≥ SANITY_PASS_MIN_STEPS（默认 20000）且两个 metrics 文件仍存在；任一失败 exit 3（agent6 v3.1 follow-up）|
| run_ablation.sh sanity 重跑被拦截时不清旧 sentinel | exit 4 但 `${SANITY_PASS_SENTINEL}` 仍指向上一轮 sanity → C/D 仍可被旧证据放行 | **修复**：sanity 分支 exit 4 路径前 `rm -f "${SANITY_PASS_SENTINEL}"`；操作员"我尝试重跑 sanity"的意图本身使旧 gate 失效（agent6 v3.1 follow-up）|
| run_ablation.sh usage SANITY_PASS_MIN_STEPS | 未提 | 增加 `SANITY_PASS_MIN_STEPS=N` 行：默认 20000，仅 debug 重建可降低（agent6 v3.1 follow-up）|
| summarize_run.sh 头注释 | 写"Quick best-checkpoint summary"易被误解为 best.pt 行汇总 | 增"IMPORTANT — what this script reports vs what the paper table needs"块，明言 per-metric 独立最小值口径与 §1.2 best.pt 行口径不同（agent6 v3.1 follow-up）|

#### v3.1 三次 follow-up（codex2 风险三审 — 2026-05-03）

| 项 | 之前 | 现在 |
|---|----|------|
| §6.1 V6 best vs last 解释 | "Phase III 后段震荡" | **修订**：root cause 是 eval rolling-window methodology，**不是**训练失稳。原文保留 best/last 双报纪律但增加 §6.2 引导 |
| §6.2 全新 | 无 | 新增 "Eval methodology pitfall: rolling-window noise" 节：max_val_batches=64 / val_loader 3648 batches → 57 disjoint 窗口；CV=25.5%；adjacent-step 70-130% swings；paper 数字必须用 `eval_first_hop_224_clip3.py --max-slices 0` 全 val 复测 |
| §6.3 全新 | 无 | 新增 "Risk 3 (120K compressed schedule) 处理纪律"：Phase III 30K vs V6 50K，paper 必须收敛到 "120K compressed schedule"，灰区 (5-10% rel diff) 触发 200K 续训；不动 yaml ratios |
| §6.4 全新 | 无 | 新增 "Risk 4 (pair channel confound) 处理协议"：A_pair_uniform_spot 60K spot check，决策规则 ≤5% / 5-10% / >10% 三档；不阻塞主 ablation |
| §6.5 全新 | 无 | 新增 "Risk 5 (hop residual claim scope)"：reframe 为 "decoder 伪影压制 branch"（用户确认）而非 "core architectural innovation"；不动代码 |
| §7 决策清单 | 8 项, 主 ablation + V6.1 done = paper-write | 增 8.5 (A_pair_uniform_spot 60K) + 8.6 (`eval_first_hop_224_clip3.py --max-slices 0` 全 val 复测 paper 数字) |
| configs/A_pair_uniform_spot.yaml | 不存在 | 新增：copy of A_control.yaml；run_name + max_steps=60K + save_interval=10K + pair_loss_weights=[1,1,1,1] 四项 delta；header comment block 说明 Risk 4 spot check 用途与决策规则 |
| run_ablation.sh dispatch | 7 个分支 | 新增 `pair_uniform` 分支：`require_sanity_pass` gate + run_one A_pair_uniform_spot 60K；usage block 同步更新 |

## 9. 脚本修复记录（review/0502/scripts/run_ablation.sh）

### 9.1 v2 → v3 修复

为让 §3.2/§3.4 的 mini full sanity 能真正拦截，修复了两处脚本 bug：

1. **`SANITY_STEPS / MAX_STEPS` 环境变量**：`run_ablation.sh sanity` 在 v2 以前指定 `run_one ... "50000"`（硬编码），试图在文档中用 `MAX_STEPS=10000` 覆盖不生效。v3 改为：
   ```bash
   SANITY_STEPS="${SANITY_STEPS:-${MAX_STEPS:-50000}}"
   ```
   与 [`run_sanity_light.sh`](scripts/run_sanity_light.sh) `MAX_STEPS` 语义兼容。
2. **comparator 由 print 变为执行**：原 `print_sanity_check_cmd` 函数只是 `cat <<EOF ... EOF`，将一串 python 脚本文本输出到 stdout，gate 不会拦截运行。v3 重命名为 `run_sanity_comparator`，使用 `"${PYTHON}" - "${a_metrics}" "${b_metrics}" <<'PYEOF' ... sys.exit(0 if overall_pass else 1) PYEOF` 真正执行脚本。

### 9.2 v3 → v3.1 修复（agent1/2 三次审计）

五项脚本加固：

1. **CUBLAS_WORKSPACE_CONFIG 环境变量**（agent1）：`run_sanity_light.sh` 已设，但 `run_ablation.sh` 缺。没有这个变量时，`torch.use_deterministic_algorithms(True)` 不能控制 cuBLAS GEMM workspace 选择 → Tier 1 bit-equal 目标会被违反。v3.1 加：
   ```bash
   export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
   ```
2. **comparator 缺键 fail-strict**（agent2）：v3 comparator 遇到 `a is None or b is None` 只打 `<absent>` 后 `continue`，**不会让 overall_pass 变 False**。如果有人误把 light sanity yaml（不含 chain key）拼接到这个 comparator，会 silent PASS。v3.1 改为 Tier 1/2/4 必需键丢失主动 FAIL，且补充 “Tier 3 少于 4 个 raw step 键”的入口判可 + missing keys 收尾报警。
3. **main/C/D sanity-pass sentinel gate**（agent2）：v3 的 `main)` / `C)` / `D)` 都能被调用者任意启动 — 这里面只有 A_main 不依赖 sigma_dt code path。v3.1 加了：
   - 成功 sanity 后写 sentinel `${ABLATION_INDEX_DIR}/.sanity_pass`，含 timestamp + steps + metrics paths
   - sanity FAIL 或 不运行 → 无 sentinel
   - `main` / `C` / `D` (single command) 启动前调用 `require_sanity_pass`，无 sentinel 则 exit 3 并提示调用者 `bash run_ablation.sh sanity` 或 `ALLOW_UNGATED=1` (debug 专用)
   - `A` / `B` 不受影响，仍可独立跑
4. **sanity 重跑冲突 + SANITY_FRESH 逃生门**（agent3）：[A_control.yaml](configs/A_control.yaml) 与 [B_sanity.yaml](configs/B_sanity.yaml) 都设 `require_fresh_output_dir: true`，第二次启动 sanity（不同 SANITY_STEPS / FAIL 重试 / 代码修后 debug）会被 `train_first_hop.py:1303` 报 "Output dir already exists" 拦下。sanity 是设计上的烟雾测试 (smoke test)，产出可棄。v3.1 加了脚本逃生门：
   - 进入 `sanity)` 分支后，先检查 `${ABLATION_OUTPUT_ROOT}/{A_sanity,B}` 是否存在且非空
   - 如存在 且 `SANITY_FRESH=1` → `rm -rf` 两个目录后继续
   - 如存在 且 `SANITY_FRESH` 未设 → exit 4 + 提示调用示例
   - main / A_main / C / D 产出目录 **不受影响**（这些是正式跑，重跑需调用者手动重命名或清理）
5. **`main` 快捷跱 双门 gate**（agent1 v3.1 追调 → codex review 确认保留）：之前一度打算 将 `main` 拆分为 “A_main 无 gate → sanity gate → C”，但这让 `main` 同时含含两种 gate 语义，调用者连看 5 行 shell 都不能预期“A_main 会不会被拦”。最终取保守语义：
   - `main`：双门 gate。调用者明确讨论“我信任 sanity 已过，需要串行跑 A→C”。
   - 与 sanity 并行：显式 `bash review/0502/scripts/run_ablation.sh A` （`A)` 分支不调用 `require_sanity_pass`，依据是 A_main 使用 V6 reference step_weights，不走 `_compute_sigma_dt_normalizers` code path）； sanity sentinel 写入后再 `bash run_ablation.sh C`。
   - 这个语义与 codex 2026-05-03 review §4.3 推荐的操作模式完全一致。
6. **usage 补全环境变量说明**（agent1 v3.1 追调）：原 usage 只提了 ALLOW_UNGATED。v3.1 补上 SANITY_STEPS / SANITY_FRESH / MAX_STEPS / GPU 以及 “10K 是 debug-only smoke” 警告。
5. **10K smoke 不写 sentinel**：`SANITY_STEPS<20000` 即使 comparator PASS，也只证明脚本链路可跑，不足以解锁 C/D/main。v3.1 因此会清除/不写 `${ABLATION_INDEX_DIR}/.sanity_pass`，并提示改跑 `SANITY_STEPS=20000` 或默认 50K。

#### v3.1 二次 follow-up（agent6 read-only audit — 三项加固）

7. **`require_sanity_pass` 新增 freshness check**（agent6 High #2）：v3.1 的 gate 只验证 `[ -f "${SANITY_PASS_SENTINEL}" ]` ——一个 10K smoke 残留的 sentinel、或 sentinel 写完后两个 sanity 输出目录已被手动删除的情况，都会被错误放行。v3.1 二次 follow-up 强化为：
   - 解析 sentinel 中的 `sanity_steps=N` 行：若 `N < SANITY_PASS_MIN_STEPS`（默认 20000）→ exit 3
   - 解析 `a_metrics=` / `b_metrics=` 路径：若任一文件不再存在 → exit 3（"sanity 输出已删，gate 不可信"）
   - sentinel 格式异常（无 `sanity_steps=` 行）→ exit 3
   - 通过时打印 "freshness OK: sanity_steps=N ≥ MIN, metrics paths exist"
   - debug 重建可设 `SANITY_PASS_MIN_STEPS=<lower>` 或 `ALLOW_UNGATED=1` 显式降级
8. **sanity 重跑被拦截时清除旧 sentinel**（agent6 Medium #3）：v3.1 sanity 分支在 `SANITY_FRESH` 未设但旧目录已存在时 exit 4——但旧 `${SANITY_PASS_SENTINEL}` 仍指向上一轮 sanity，意味着 C/D 仍可被旧证据放行。操作员"我尝试重跑 sanity"的意图本身已让旧 gate 失效。v3.1 二次 follow-up 在 exit 4 之前 `rm -f "${SANITY_PASS_SENTINEL}"`，并打印 invalidation 提示。
9. **`summarize_run.sh` 文档警告**（agent6 High #1）：`summarize_run.sh` 输出的是 *per-metric independent minima*（每个指标独立扫描取最小那行）——不同指标可能来自不同 step，这与 paper 表所需的 best.pt 行口径不一致。v3.1 二次 follow-up 修：
   - `summarize_run.sh` 头注释新增 "IMPORTANT — what this script reports vs what the paper table needs" 块，明言两种口径不同
   - `POST_V6_NEXT_STEPS.md §1.2` 末尾新增 ⚠️ 警告框，导回 §1.2 python snippet
   - `POST_V6_NEXT_STEPS.md §2.2` ckpt 路径行的 summarize_run 引用改写为"仅做 per-metric 独立最小值的快速诊断，不要直接抄进 paper 表"
   - `run_ablation.sh` usage 新增 `SANITY_PASS_MIN_STEPS=N` 行
   - 未跳过的两项（agent6 Medium #4 "main 调用即跳 main_skip" / agent6 Low #5 "A_main 残差留底"）维持 v3.1 行为：前者 `set -e` 已使 A_main FAIL 早退；后者 `require_fresh_output_dir: true` 已强制每次正式跑独立目录，残差不会污染下一次 best.pt。

### 9.3 验证
```bash
$ bash -n review/0502/scripts/run_ablation.sh && echo OK
OK
$ bash review/0502/scripts/run_ablation.sh main
ERROR: main requires a sanity-pass sentinel at:
  /Users/.../review/0502/runs/.sanity_pass
Run a Tier 1-4 sanity gate first:
  bash review/0502/scripts/run_ablation.sh sanity        # default 50K×2
  SANITY_STEPS=20000 bash review/0502/scripts/run_ablation.sh sanity   # mini full sanity
$ bash review/0502/scripts/run_ablation.sh A
[run_ablation.sh] condition=A_main           # ✅ 正常启动不被 gate 拦截
$ SANITY_STEPS=10000 bash review/0502/scripts/run_ablation.sh sanity
[run_ablation.sh] Smoke sanity PASS at SANITY_STEPS=10000, but no sentinel was written.

# v3.1 二次 follow-up 新增验证（agent6）：
$ printf 'timestamp=stale\nsanity_steps=10000\na_metrics=/tmp/a.jsonl\nb_metrics=/tmp/b.jsonl\n' \
    > review/0502/runs/.sanity_pass
$ bash review/0502/scripts/run_ablation.sh main
ERROR: sentinel sanity_steps=10000 < SANITY_PASS_MIN_STEPS=20000.
  This sentinel was written by a smoke run, not a Tier-4 gate.
  Re-run with SANITY_STEPS=20000 (or larger):
    SANITY_FRESH=1 SANITY_STEPS=20000 bash review/0502/scripts/run_ablation.sh sanity

# v3.1 三次 follow-up 验证（codex2 risk triage）：
$ bash -n review/0502/scripts/run_ablation.sh && echo OK
OK
$ bash review/0502/scripts/run_ablation.sh pair_uniform
ERROR: pair_uniform (Risk 4 spot) requires a sanity-pass sentinel at:
  /Users/.../review/0502/runs/.sanity_pass             # ✅ gate 正确生效
$ bash review/0502/scripts/run_ablation.sh
Usage: ... {sanity|main|closed_form|A|B|C|D|pair_uniform}
  pair_uniform: run A_pair_uniform_spot (60K, Risk 4 spot check) — requires sentinel  # ✅ usage 已更新
$ python3 -c "import yaml; c=yaml.safe_load(open('review/0502/configs/A_pair_uniform_spot.yaml')); \
    print(c['run_name'], c['training']['max_steps'], c['transport']['pair_loss_weights'])"
first_hop_224_sigma_norm_A_pair_uniform_spot 60000 [1.0, 1.0, 1.0, 1.0]      # ✅ yaml delta 验证
```

---

> **生成参考**：[SIGMA_NORMALIZE_ABLATION_PLAN.md](SIGMA_NORMALIZE_ABLATION_PLAN.md) §3-§4 / [README.md](README.md) §快速开始 + §GPU·days / [AUDIT_RESPONSE_v4.md](AUDIT_RESPONSE_v4.md) §3-§4 / V6 / V6.1 / sanity light 三份训练快照（[log_snapshots/](log_snapshots/)）/ [`train_first_hop.py` L395+](../../train_first_hop.py) live `_compute_sigma_dt_normalizers` 实现 / [`train_first_hop.py` L2510-2530](../../train_first_hop.py) best.pt 保存逻辑与 EMA 包装 / [`eval_first_hop_224_clip3.py` L283-310](../../eval_first_hop_224_clip3.py) 输出文件名 / [`run_ablation.sh`](scripts/run_ablation.sh) v3 sanity gate 实现 / [`run_sanity_light.sh`](scripts/run_sanity_light.sh) Tier 1-3 light comparator。
