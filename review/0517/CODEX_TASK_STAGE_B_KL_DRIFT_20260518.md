# Codex Task — Stage B Step 1: KL Drift Measurement (V18 vs V7)

- date: 2026-05-18
- branch: foc_lite_hop0
- status: **READY FOR CODEX EXECUTION** (Round 12 user decision 2=A: 先测 KL drift)
- 触发: V18 训练已完成 (200K), Round 12 reviewer 共识 V18 outcome = sub-PARTIAL (best-vs-best +0.030 dB), 需 KL drift 数据按 V18_design_rationale §2.3 row 4 双触发判 confirm KILL / 灰区
- 硬约束: V18 已 free (slot 1 已空), V13 不在 slot 2 (v3 未 push), 本 task 占 slot 1 唯一 GPU 约 30-60 min
- 设计来源: [REVIEW_INTEGRATION_round12_20260518.md](./REVIEW_INTEGRATION_round12_20260518.md) §4 (阶段 C decision matrix 输入)

---

## §0 — 给 codex 的 5 句话总览

1. **0 训练**, 0 yaml 改动. 只跑 1 个 eval probe (~30-60 min, 1 GPU).
2. 目标: 实测 KL drift = `PSNR(decode_V18(z_GT), x_target)` vs `PSNR(decode_V7(z_GT), x_target) = 46.64 dB` 的差异 (V18 decoder 是否漂离 GT manifold).
3. **不动 V18 yaml / 不改 train_first_hop.py / 不 launch 任何训练**.
4. 复用现有 [probe_v18_gap_decomposition.py](../../tools/probe_v18_gap_decomposition.py) 框架, **新建** sister script `probe_v18_kl_drift.py` (~80 行, 与 gap_decomp 同 structure 但 swap V7 best → V18 best).
5. push gitee 后等 user 阶段 C 决策, 不自发 launch V18-clean / paper draft / 任何其他 task.

---

## §1 — 物理含义 (Round 12 user 必读)

`psnr_v18_at_zgt = PSNR(decode_V18(z_GT), x_target)` 度量:
- **如果 V18 decoder 已漂离 GT manifold** (B9 同型: KL pullback `use_pred_latent=true` 让 decoder 朝错方向学), `psnr_v18_at_zgt` 显著 < V7 `psnr_ceil = 46.64 dB`
- **如果 V18 decoder 仍在 GT manifold 附近**, `psnr_v18_at_zgt ≈ 46.64 dB` (KL pullback 工作正常)

KL drift = |psnr_v7_at_zgt - psnr_v18_at_zgt| (on NORMAL chain)

### 1.1 决策矩阵 (Round 12 共识)

按 V18_design_rationale §2.3 row 4 + NEXT_STAGE_ARCH_CODE_FINAL §3.1 3×3 矩阵:

| KL drift | 含义 | V18 outcome confirm | Stage C |
|---|---|---|---|
| < 0.05 dB (NEGLIGIBLE) | KL pullback 工作良好, V18 仍在 GT manifold | sub-PARTIAL gray (不 confirm KILL) | 灰区, **user 决策** best-vs-best=KILL vs last-vs-best=PARTIAL |
| 0.05 - 1.0 dB (MODERATE) | KL 部分漂离 | confirm KILL (双触发) | backbone/data 方向, paper "V18 limitation" 章节 |
| > 1.0 dB (SIGNIFICANT) | KL 严重漂离, V18 走错方向 | confirm KILL + B9 confirmed | backbone/data 方向, paper 加 B9 attribution |

---

## §2 — Task A: Build `probe_v18_kl_drift.py` (~80 行, 复用 gap_decomp framework)

### A1: 文件位置

新建: `tools/probe_v18_kl_drift.py`

### A2: 基于 [tools/probe_v18_gap_decomposition.py](../../tools/probe_v18_gap_decomposition.py) 的 diff

复用其全部 dataloader / model load / decode crop pipeline. 只改:

| 段 | gap_decomp | kl_drift |
|---|---|---|
| 输入 ckpt | V7 best.pt (single) | **V7 best.pt + V18 best.pt + V18 last.pt (3 ckpt)** |
| 度量 | psnr_ceil / psnr_transport / latent_l2_rel | **PSNR(decode_V?(z_GT), x_target) on D20/D10/D4/NORMAL × 3 ckpts** |
| 报告 | GAP_DECOMP_REPORT.md | **KL_DRIFT_REPORT.md** |
| Per-slice CSV | GAP_DECOMP_PER_SLICE.csv | **KL_DRIFT_PER_SLICE.csv** (n=7403 × 4 chains × 3 ckpts) |
| Summary JSON | GAP_DECOMP_SUMMARY.json | **KL_DRIFT_SUMMARY.json** |

### A3: 核心 forward pseudocode

```python
# for each (ckpt_name, ckpt_path) in [(V7_best, ...), (V18_best, ...), (V18_last, ...)]:
#     load model with ckpt
#     for batch in val_loader (n=7403):
#         z_GT = batch["z_dst"]  # ground truth latent (D20/D10/D4/NORMAL)
#         x_target = batch["x_dst"]  # ground truth pixel
#         with torch.no_grad():
#             x_hat = model.rae.decode(z_GT)  # decode GT latent directly (no transport!)
#         psnr = calc_psnr_clip3(x_hat, x_target)  # per-slice PSNR
#         记录 per-slice 到 csv, 累积 mean 到 json
```

**关键**: `model.rae.decode(z_GT)` 直接 decode ground-truth latent, **跳过整个 transport pipeline**. 这隔离了 decoder 的 GT-manifold 性能, 不混入 transport error.

### A4: Output 模板

`KL_DRIFT_REPORT.md`:

```markdown
# V18 KL Drift Probe Report

## Protocol
- Evaluator: `tools/probe_v18_kl_drift.py`
- Metric: `src.utils.metrics.calc_psnr_clip3` (same as V18 final eval)
- Split: full val (n=7403)
- Compare: PSNR(decode(z_GT), x_target) across V7.best / V18.best / V18.last

## Results

| ckpt | step | D20 | D10 | D4 | NORMAL |
|---|---:|---:|---:|---:|---:|
| V7.best | 160000 | <X.X> | <X.X> | <X.X> | <X.X> |
| V18.best | 165000 | <X.X> | <X.X> | <X.X> | <X.X> |
| V18.last | 200000 | <X.X> | <X.X> | <X.X> | <X.X> |

## KL Drift (V7.best - V18.X)

| comparison | D20 | D10 | D4 | NORMAL |
|---|---:|---:|---:|---:|
| V7.best - V18.best | <Δ> | <Δ> | <Δ> | <Δ> |
| V7.best - V18.last | <Δ> | <Δ> | <Δ> | <Δ> |

## Verdict (Stage C input, Round 12 决策矩阵)

- V18.best NORMAL KL drift = <X> dB
- V18.last NORMAL KL drift = <X> dB
- Verdict (per Round 12 §1.1):
  - NEGLIGIBLE (<0.05) / MODERATE (0.05-1.0) / SIGNIFICANT (>1.0)

## Interpretation

- 若 NEGLIGIBLE: KL pullback 工作良好, V18 在 GT manifold 附近. V18 sub-PARTIAL outcome 不是 decoder 漂离原因, 是 decoder LoRA 容量不足 / transport error 不可被 decoder 端补偿. → 灰区, user 决策.
- 若 MODERATE/SIGNIFICANT: KL 漂离 confirm KILL (双触发), V18 family retire, backbone/data 方向.
```

---

## §3 — Task B: 执行

### B1: 前置确认

```bash
cd /home/qujiaxiang/project/PET_LatentResidual
git pull --ff-only gitee foc_lite_hop0

# 验证 ckpt 存在
test -f /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V7/run/first_hop_224_v7_gronwall_raw/best.pt && echo "V7 best OK" || echo "V7 MISSING"
test -f /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0517_runs/V18_decoder_lora/run/first_hop_224_v18_decoder_lora/best.pt && echo "V18 best OK" || echo "V18 best MISSING"
test -f /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0517_runs/V18_decoder_lora/run/first_hop_224_v18_decoder_lora/last.pt && echo "V18 last OK" || echo "V18 last MISSING"

# 验证 probe_v18_gap_decomposition.py 可参考
test -f tools/probe_v18_gap_decomposition.py || { echo "FAIL: gap_decomp probe missing"; exit 1; }

# GPU 状态 (V18 已完成, V13 未 launch, slot 1 应 free)
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
```

### B2: 实施新 probe (codex 自己写)

按 §A3 pseudocode + §A2 diff 表, 基于 `probe_v18_gap_decomposition.py` 改 ~80 行写 `probe_v18_kl_drift.py`. **必须复用** `Hop0OnlyViewDataset` / `PETFirstHopAligned4HopDataset` / `PETFlowDiTFirstHop` / `calc_psnr_clip3` 现有 import.

### B3: 跑

```bash
TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR=review/0517/V18_decoder_lora/kl_drift_${TS}
mkdir -p "$OUT_DIR"

CUDA_VISIBLE_DEVICES=0 python tools/probe_v18_kl_drift.py \
  --config review/0511/log_snapshots_20260516_163900/configs/V7_config.resolved.yaml \
  --v7-ckpt /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V7/run/first_hop_224_v7_gronwall_raw/best.pt \
  --v18-best-ckpt /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0517_runs/V18_decoder_lora/run/first_hop_224_v18_decoder_lora/best.pt \
  --v18-last-ckpt /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0517_runs/V18_decoder_lora/run/first_hop_224_v18_decoder_lora/last.pt \
  --out-dir "$OUT_DIR" \
  --batch-size 8 --max-slices 0 \
  2>&1 | tee "$OUT_DIR/probe.log"
```

### B4: pass 准则 (5 条)

- [ ] 程序正常退出 (exit 0)
- [ ] `KL_DRIFT_REPORT.md` 生成且含 V7/V18.best/V18.last 三行 PSNR
- [ ] `KL_DRIFT_PER_SLICE.csv` 行数 = n_eval × 3 ckpts (e.g. 7403×3=22209 行 + 1 header)
- [ ] `KL_DRIFT_SUMMARY.json` 含 `meta.psnr_metric = src.utils.metrics.calc_psnr_clip3` (与 V18 final eval 同 evaluator)
- [ ] Verdict 显式标 NEGLIGIBLE / MODERATE / SIGNIFICANT (per §1.1)

任何 fail → 写 KL_DRIFT_FAILURE_REPORT.md, 停, 等 user.

---

## §4 — Task C: commit + push

```bash
cd /home/qujiaxiang/project/PET_LatentResidual

git add tools/probe_v18_kl_drift.py
git add review/0517/V18_decoder_lora/kl_drift_${TS}/

# 验证 git status
git status
git diff --cached --stat

git commit -m "Stage B Step 1: V18 KL drift probe (Round 12 decision 2=A)

- 新 probe tools/probe_v18_kl_drift.py (基于 gap_decomp 改 80 行)
- 度量 PSNR(decode_V?(z_GT), x_target) on V7.best / V18.best / V18.last
- Verdict 输入阶段 C 决策矩阵 (V18 outcome = sub-PARTIAL, 待 KL drift confirm)
- 不动 V18 yaml, 不改 train_first_hop.py, 0 训练
- 来源: REVIEW_INTEGRATION_round12 §4"

git push gitee foc_lite_hop0
```

---

## §5 — NOT-DO (此 task)

1. ❌ 不动 V18 yaml / V18 ckpt / train_first_hop.py
2. ❌ 不 launch V13 / V14 / V9 / V18b / V18-clean 任何训练
3. ❌ 不修复发现的 bug (只跑 probe, 修复决策属 user)
4. ❌ 不起 Round 13 / 不二次 audit 自己
5. ❌ 不顺手做未列范围的事 (audit DRAFT 不 release, paper draft 不写)
6. ❌ KL drift > SIGNIFICANT 时**不**直接 launch backbone 实验 (Stage C 决策属 user)
7. ❌ KL drift < NEGLIGIBLE 时**不**直接判 SUCCESS / paper draft (sub-PARTIAL 灰区属 user 决策)
8. ❌ 在 commit message 写"也顺便做了 X"
9. ❌ 报告里写 "V18 confirmed KILL/SUCCESS/PARTIAL" — Verdict 仅 NEGLIGIBLE/MODERATE/SIGNIFICANT, Stage C 决策属 user

---

## §6 — 失败决策树

```
B2 probe code 编写失败 (e.g. dataloader API 不熟)?
  → 停, 写报告, 不强行 launch. 不要"再试一次".

B3 probe 运行 OOM?
  → 减小 batch_size 到 4 或 2, 再跑 1 次. 仍 OOM 则停.

B3 probe 跑出 PSNR < 30 dB 或 > 50 dB (异常 sanity)?
  → 停, 检查 ckpt 加载 / dataloader z_GT 是否正确, 写报告, 不上 commit

B4 任一 pass fail?
  → 停, 写 KL_DRIFT_FAILURE_REPORT.md, 等 user

C push 时 git conflict?
  → git fetch + 报告, 不 git push --force
```

---

## §7 — 总时长估计

| Task | 时长 |
|---|---|
| A 写 probe_v18_kl_drift.py (~80 行) | 20 min |
| B 跑 probe (n=7403 × 3 ckpts) | 30-60 min |
| C 报告整合 + commit + push | 10 min |
| **总** | **~60-90 min** |

GPU 占用: slot 1 × 30-60 min. V18 已完成所以 slot 1 free, 不冲突.
