# Codex Server Runbook — Disambiguation Experiments §0/§1/§2

- date: 2026-05-16
- author: claude (claude-opus-4.5) for codex on the GPU server
- context: 两份 peer review ([REVIEW_INTEGRATION_20260516.md](./REVIEW_INTEGRATION_20260516.md)) 指出 V7/V8/V6_NOISE 不是单变量 ablation，所有 image_aux 相关 claim 在补真实对照前不可信。本 runbook 列出 **三个零训练 / 零 GPU-day 的前置实验**，必须在任何新训练 (V9/V11'/V13/V14/V15/V16) 启动前先做完。
- estimated cost: §0 = 10 min, §1 = 2-4 h dev + 1 h run, §2 = 1 day dev + 2-3 h run
- branch: foc_lite_hop0
- input checkpoints (server-local):
  - V7 last: `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V7/run/first_hop_224_v7_gronwall_raw/last.pt`
  - V7 best: `…/V7/run/first_hop_224_v7_gronwall_raw/best.pt`
  - V8 last: `…/V8/run/first_hop_224_v8_no_image_aux/last.pt`
  - V8 best: `…/V8/run/first_hop_224_v8_no_image_aux/best.pt`
  - V6_NOISE last: `…/V6_NOISE/run/first_hop_224_v6_seed1337/last.pt`
  - V6_NOISE best: `…/V6_NOISE/run/first_hop_224_v6_seed1337/best.pt`
- input configs: `review/0511/log_snapshots_20260516_163900/configs/V{7,8,V6_NOISE}_config.resolved.yaml`
- output root: `review/0517/disambig/` （需要新建）

---

## §0 — Lipschitz 测量（最便宜，必须最先跑）

### 目标

验证 [STEP_WEIGHTS_THEORY_REFERENCE.md §5.1](./STEP_WEIGHTS_THEORY_REFERENCE.md) 的"`L_i ≈ 1`"假设。它是 V7 step_weights 闭式解和后续 V9 (β_NORMAL) 推导的共同前提。L_i 偏离 1 越大，整套闭式解越不靠谱。

### 已有工件

- 脚本骨架：[tools/estimate_per_hop_lipschitz.py](../../tools/estimate_per_hop_lipschitz.py)
- 闭式解数学已 self-verified；只剩 `# TODO:` 标记的 model / data wiring。

### 服务器端任务

请 codex 完成 3 个具体 TODO，**只改 tools/estimate_per_hop_lipschitz.py，不动其他代码**：

1. **TODO #1 — 构建 model + load checkpoint**：参考 [eval_first_hop_224_clip3.py:80-130](../../eval_first_hop_224_clip3.py) 的 build_model 路径。复用该文件里的 `build_backbone` / `PETFlowDiTFirstHop` 构造 + `model.load_state_dict(sd["model"], strict=True)`。
2. **TODO #2 — 采样 z_at_hop[k]**：参考 [pet_lr/data_first_hop.py](../../pet_lr/data_first_hop.py) `PETFirstHopAligned4HopDataset`。每个 hop k 取 64 个 slice 的 `z_src`（hop=k 的 batch 里的 z_src）；可以遍历 dataloader 直到每个 hop 凑够 64 个。
3. **TODO #3 — 调用 model**：用 `model.predict_latent_step(z_src=z, t_src=..., t_dst=..., hop_idx=torch.full((B,), k, dtype=torch.long), x_src_img=x_src if k==0 else None)`；从 V7 config `data.t_map` 读 t_src/t_dst。

### 运行命令

```bash
cd /home/qujiaxiang/project/PET_LatentResidual
git pull --ff-only gitee foc_lite_hop0

OUT=review/0517/disambig/lipschitz
mkdir -p $OUT

for EXP in V7 V8 V6_NOISE; do
  CFG=review/0511/log_snapshots_20260516_163900/configs/${EXP}_config.resolved.yaml
  case $EXP in
    V7)       CKPT=/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V7/run/first_hop_224_v7_gronwall_raw/last.pt ;;
    V8)       CKPT=/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V8/run/first_hop_224_v8_no_image_aux/last.pt ;;
    V6_NOISE) CKPT=/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V6_NOISE/run/first_hop_224_v6_seed1337/last.pt ;;
  esac
  CUDA_VISIBLE_DEVICES=0 /home/qujiaxiang/.conda/envs/rae/bin/python \
    tools/estimate_per_hop_lipschitz.py \
    --config $CFG --checkpoint $CKPT \
    --out $OUT/${EXP}_lipschitz.json \
    --n-samples 64 --eps 1e-3
done
```

### 期望输出

每个 JSON 含：

```json
{
  "L_per_hop": [L0, L1, L2, L3],
  "closed_form_w_with_measured_L": [w0, w1, w2, w3],
  "current_step_weights": [s0, s1, s2, s3],
  "alignment_distance_pct": 0.0
}
```

### 判定

来自 [V9_PREREGISTRATION §0](./V9_normal_emphasis/V9_PREREGISTRATION.md)：

| alignment_distance_pct (V7) | max(L_j) | 判定 |
|---|---|---|
| < 10% | < 1.10 | **L≈1 confirmed**：V9 可 launch；V7 step_weights 是真闭式解 |
| 10-25% | < 1.10 | **partial**：V9 可 launch 但需记录 measured L_j |
| ≥ 25% OR max(L_j) ≥ 1.15 | — | **L≈1 FAILS**：V9 不可 launch；整个 step_weights 闭式解推导需用 measured L 重派 |

### 交付

把 `review/0517/disambig/lipschitz/*.json` + 一份 50 行的 `LIPSCHITZ_REPORT.md`（含 alignment 判定）commit 到 gitee。

---

## §1 — 单步 PSNR 面板（区分 chain coherence vs shared backbone）

### 目标

agent2 peer review 指出 V7 vs V8 的 NORMAL +0.308 dB 至少有两个来源：

- **通道 A (chain coherence)**：hop0 学得好 → 后续起点更准 → 累积到 NORMAL
- **通道 B (shared backbone)**：image_aux 修改 backbone → 所有 hop 直接受益

两个通道都能产生"delta 沿链放大"的现象，但临床决策完全不同。**用单步 PSNR 可以解耦**。

### 方法

对每个 hop k 独立做：
1. 输入 GT z_{k} 与 GT x_{k}（对 k=0 提供，其他 None）
2. 模型 forward 一步：`model.predict_latent_step(z_src=z_k, t_src=t_k, t_dst=t_{k+1}, hop_idx=k, x_src_img=...)`
3. decode(z_pred) 与 GT x_{k+1} 比 PSNR_clip3
4. 注意：**不级联**，每个 hop 都用 GT z_{k} 作为输入

这与 [eval_first_hop_224_clip3.py](../../eval_first_hop_224_clip3.py) 现有的链式评估（用 z_pred 级联）**不同**，需要新写。

### 实现建议

复制 `eval_first_hop_224_clip3.py` 为 `tools/eval_per_hop_singlestep_clip3.py`，把 `sample_chain_first_hop` 调用替换为：

```python
def per_hop_singlestep(model, batch, rollout_times, image_size):
    """For each hop k, predict z_{k+1} from GT z_{k}, decode, return PSNR per hop."""
    results = {}  # {hop_k: {"psnr": tensor[B], "mse": tensor[B]}}
    for k in range(len(rollout_times) - 1):
        z_src = batch["z_rollout"][:, k]      # GT z_{k}, [B, C, H, W]
        x_gt  = batch["x_rollout"][:, k+1]    # GT image at target [B, 1, H, W]
        x_src = batch["x_rollout"][:, 0] if k == 0 else None  # only hop0 gets D50 image
        t_src = torch.full((z_src.shape[0],), float(rollout_times[k]),   device=z_src.device)
        t_dst = torch.full((z_src.shape[0],), float(rollout_times[k+1]), device=z_src.device)
        hop   = torch.full((z_src.shape[0],), k, device=z_src.device, dtype=torch.long)
        out   = model.predict_latent_step(z_src=z_src, t_src=t_src, t_dst=t_dst, hop_idx=hop, x_src_img=x_src)
        x_pred = model.decode_crop(out["z_pred"], crop_size=image_size)
        # PSNR_clip3 per-slice (reuse calc_psnr_clip3 from eval_first_hop_224_clip3.py)
        results[f"hop{k}"] = {
            "psnr": calc_psnr_clip3(x_pred, x_gt, reduction="none"),  # [B]
            "mse":  ((x_pred - x_gt) ** 2).mean(dim=(1,2,3)),         # [B]
        }
    return results
```

dataset 配 `val_include_full_x_rollout: true`（V7/V8 yaml 已开），所以 batch 里有 `x_rollout` 全部 5 个 timepoint。

### 运行命令

```bash
cd /home/qujiaxiang/project/PET_LatentResidual
git pull --ff-only gitee foc_lite_hop0

OUT=review/0517/disambig/per_hop_singlestep
mkdir -p $OUT

for EXP_CKPT in V7:last V8:last V7:best V8:best V6_NOISE:last; do
  EXP=${EXP_CKPT%:*}
  KIND=${EXP_CKPT#*:}
  CFG=review/0511/log_snapshots_20260516_163900/configs/${EXP}_config.resolved.yaml
  case $EXP in
    V7)       BASE=/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V7/run/first_hop_224_v7_gronwall_raw ;;
    V8)       BASE=/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V8/run/first_hop_224_v8_no_image_aux ;;
    V6_NOISE) BASE=/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V6_NOISE/run/first_hop_224_v6_seed1337 ;;
  esac
  CKPT=$BASE/${KIND}.pt
  CUDA_VISIBLE_DEVICES=0 /home/qujiaxiang/.conda/envs/rae/bin/python \
    tools/eval_per_hop_singlestep_clip3.py \
    --config $CFG --checkpoint $CKPT \
    --max-slices 0 --decode-mode both \
    --out-dir $OUT/${EXP}_${KIND}
done
```

### 期望输出

每个目录含：

- `per_hop_singlestep_psnr.json`：每个 hop 的均值 PSNR / MSE
- `per_hop_singlestep_per_slice.csv`：每 slice × 每 hop 的 PSNR / MSE（用于后续 paired-t 分析）

### 判定（zero-training disambiguation）

定义：

- `Δ_cascade(k) = PSNR_clip3(V7 chain → x_k) - PSNR_clip3(V8 chain → x_k)`  （来自 [PLANF_FULLVAL_PSNR_CLIP3_REPORT_20260516_173941.md](../0511/fullval_psnr_clip3_20260516_173941/PLANF_FULLVAL_PSNR_CLIP3_REPORT_20260516_173941.md)）
- `Δ_singlestep(k) = PSNR_clip3(V7 single-step z_k→x_{k+1}) - PSNR_clip3(V8 single-step z_k→x_{k+1})`（本 §1 输出）

| 模式 | 判定 | 决策含义 |
|---|---|---|
| `Δ_singlestep(0) ≈ Δ_cascade(NORMAL)` 且 `Δ_singlestep(k≥1) ≈ 0` | **通道 A 主导**（chain coherence） | V11' (强化 hop0) 是对的；不需要 multi-hop image_aux |
| `Δ_singlestep(k) ≈ Δ_cascade(k)` 对所有 k 成立 | **通道 B 主导**（shared backbone） | V11 / V15 (multi-hop image_aux) 不该被撤回 |
| 两者都有显著贡献 | **mixed** | V11' 和 V15 都该试，且应该解释它们的边际贡献 |

### 交付

`review/0517/disambig/per_hop_singlestep/*` + 一份 60 行 `SINGLESTEP_REPORT.md` 含判定 + paired t-test (Δ_singlestep vs 0)。

---

## §2 — ROI-PSNR / 高 SUV 区域面板

### 目标

[0501 review §3](../0430/reviewer/literature_architecture_image_aux_review_20260501.md) 已经明确指出：

- PSNR_clip3 把 SUV > 3 截掉了；**临床真正关心的失败模式（肿瘤热点）全在 SUV > 3**
- 当前 image_aux 也是 hop0-only + low-weight + 没有 ROI 加权
- 是否要强化 hop0 image_aux (V11') 或扩展 multi-hop (V11/V15)，**取决于复杂区域是否真有 PSNR 失败**

### 评估指标（来自 0501 review §3 表）

每个 checkpoint × 每个 hop 输出：

1. **未-clip PSNR**（不做 SUV<3 clip 的 PSNR）
2. **top-10% SUV mask 内的 PSNR** （GT mask 内 only）
3. **top-5% SUV mask 内的 PSNR**
4. **top-1% SUV mask 内的 PSNR**
5. **high-gradient mask 内的 PSNR**（GT gradient magnitude > p90 阈值）
6. **SUVmax 误差**：max(x_pred) - max(x_gt)
7. **SUVmean 误差**（高 SUV 区域均值差）

### 实现建议

写 `tools/eval_roi_psnr.py`：
- 接受 chain 解码后的 5 张图像（D50/D20/D10/D4/NORMAL）
- 对每张 compute 上述 7 个指标
- 输出 per-slice CSV + 汇总 JSON

数据来源：复用 [eval_first_hop_224_clip3.py](../../eval_first_hop_224_clip3.py) 的链式采样 (`sample_chain_first_hop`)，但 decode 后把图像保留，喂给 `compute_roi_metrics(x_pred, x_gt)`。

伪代码：

```python
def compute_roi_metrics(x_pred, x_gt):
    """x_pred, x_gt: [B, 1, H, W] in SUV units."""
    out = {}

    # Unclipped PSNR
    mse = ((x_pred - x_gt) ** 2).flatten(1).mean(dim=1)
    # peak based on x_gt max per slice (clinical convention)
    peak = x_gt.flatten(1).max(dim=1).values
    out["psnr_unclipped"] = 10.0 * torch.log10(peak ** 2 / mse.clamp_min(1e-12))

    # ROI masks from x_gt
    for top_pct, name in [(0.10, "top10"), (0.05, "top5"), (0.01, "top1")]:
        thresh = torch.quantile(x_gt.flatten(1), 1 - top_pct, dim=1, keepdim=True).view(-1, 1, 1, 1)
        mask = (x_gt >= thresh).float()
        out[f"psnr_{name}suv"] = _masked_psnr(x_pred, x_gt, mask)

    # High-gradient mask
    grad = _gradient_magnitude(x_gt)
    g_thresh = torch.quantile(grad.flatten(1), 0.9, dim=1).view(-1, 1, 1, 1)
    g_mask = (grad >= g_thresh).float()
    out["psnr_highgrad"] = _masked_psnr(x_pred, x_gt, g_mask)

    # SUV stats
    out["suvmax_err"] = x_pred.flatten(1).max(dim=1).values - x_gt.flatten(1).max(dim=1).values
    out["suvmean_err"] = (x_pred * (x_gt > 0.5).float()).flatten(1).sum(dim=1) / ((x_gt > 0.5).float().flatten(1).sum(dim=1).clamp_min(1)) \
                       - (x_gt   * (x_gt > 0.5).float()).flatten(1).sum(dim=1) / ((x_gt > 0.5).float().flatten(1).sum(dim=1).clamp_min(1))

    return out
```

注：实际 PET SUV 阈值（0.5、3.0 等）需要与项目数据归一化方式校准；可以先以 GT 自身分位数为基准，避免引入 hard threshold。

### 运行命令

```bash
cd /home/qujiaxiang/project/PET_LatentResidual
git pull --ff-only gitee foc_lite_hop0

OUT=review/0517/disambig/roi_psnr
mkdir -p $OUT

for EXP_CKPT in V7:last V8:last V6_NOISE:last; do
  EXP=${EXP_CKPT%:*}
  KIND=${EXP_CKPT#*:}
  CFG=review/0511/log_snapshots_20260516_163900/configs/${EXP}_config.resolved.yaml
  case $EXP in
    V7)       BASE=/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V7/run/first_hop_224_v7_gronwall_raw ;;
    V8)       BASE=/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V8/run/first_hop_224_v8_no_image_aux ;;
    V6_NOISE) BASE=/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V6_NOISE/run/first_hop_224_v6_seed1337 ;;
  esac
  CKPT=$BASE/${KIND}.pt
  CUDA_VISIBLE_DEVICES=0 /home/qujiaxiang/.conda/envs/rae/bin/python \
    tools/eval_roi_psnr.py \
    --config $CFG --checkpoint $CKPT \
    --max-slices 0 \
    --out-dir $OUT/${EXP}_${KIND}
done
```

### 判定

| 现象 | 决策含义 |
|---|---|
| top-1%/5% SUV PSNR 显著低于全图 PSNR | **复杂区域是失败模式** → image_aux 应加 ROI 加权 (V11c) 优先于 λ sweep |
| top-1% SUV PSNR(V7) ≈ PSNR(V8) | hop0 image_aux 没改善复杂区域 → V11' (强化 hop0 λ) 不会有效 |
| top-1% SUV PSNR(V7) > PSNR(V8) | hop0 image_aux 部分改善复杂区域 → V11' 方向对 |
| SUVmax 误差 在 NORMAL 比 D20 大 | NORMAL 端临床信号丢失多 → 支持 multi-hop image_aux (V11/V15) |
| unclipped PSNR 与 clip3 PSNR 趋势不同 | clip3 在掩盖 high-SUV failure → 后续所有 claim 必须同时报告两个 |

### 交付

`review/0517/disambig/roi_psnr/*` + 一份 `ROI_REPORT.md` 含判定矩阵 + V7/V8/V6_NOISE 三套数据的对照表。

---

## §3 — 阻塞关系

§0/§1/§2 三个实验**互相独立**，可以并行：
- §0 在 GPU 0 跑（10 min）
- §1 在 GPU 1 跑（每 ckpt ~5 min eval × 5 ckpts）
- §2 在 GPU 2 跑（每 ckpt ~10 min eval × 3 ckpts）

但任何一个完成后都应该立即把结果 commit + push gitee，以便后续 V13/V14/V15 设计能用上。

---

## §4 — 后续决策树（基于 §0/§1/§2 结果）

```
Lipschitz §0:
  - L≈1 confirmed   → V7/V9/V11' step_weights 推导有效
  - L偏离>25%       → 整套 step_weights 都要重派；V9 暂停

Singlestep §1:
  - channel A 主导  → 推 V11' (强化 hop0 image_aux)
  - channel B 主导  → 推 V15 (multi-hop image_aux + v_std² 衰减)
  - mixed          → V11' 和 V15 都做，互为对照

ROI §2:
  - 复杂区域失败    → image_aux 改 ROI 加权 (V11c) 比改 λ 优先级高
  - 复杂区域 OK     → 当前 image_aux 形式合理，方向选 λ sweep / multi-hop
```

V13 (true image_aux ablation) 与 V14 (true d_pure) 的 yaml 已在 [review/0516/V13_*](./V13_true_image_aux_ablation/) 和 [review/0516/V14_*](./V14_true_d_pure/) draft，**但它们的优先级与 launch 时机要在 §0/§1/§2 完成后才能确定**。

---

## §5 — 给 codex 的明确边界

**可以做**：
- 改 / 新建 `tools/*.py`
- 改 / 新建 `review/0517/disambig/*`
- 改 / 新建本文档（`review/0516/CODEX_DISAMBIGUATION_RUNBOOK_20260516.md`）的执行记录段
- commit + push gitee 自己的工件

**不要做**：
- 改 `train_first_hop.py` / `pet_lr/**` / `configs/**`（这些是训练核心，需经过 review）
- launch 任何新训练（V9/V11'/V13/V14/V15/V16）
- 在 §0/§1/§2 完成前先动 V13/V14 yaml（它们是 draft）
- 修改 `review/0516/REVIEW_INTEGRATION_20260516.md` 或其他 review 文档（这些是讨论结果，需 claude 同步）

完成 §0/§1/§2 后，请 codex 在本文档末尾追加一段简短执行记录（运行时间、判定结论一句话），然后 commit。我会基于结果决定 V13/V14/V15 的下一步。

---

## §6 — Codex execution record (2026-05-17)

- §0 Lipschitz gate completed on GPU1. Outputs: `review/0517/disambig/lipschitz/`, report: `review/0517/disambig/lipschitz/LIPSCHITZ_REPORT.md`, commit `49312b4`. Decision: measured per-hop `L_i` are all close to 1 (`max L <= 1.0091`), so the Gronwall/closed-form step-weight assumption is acceptable for V7/V9-style reasoning.
- §1 per-hop single-step full-val completed on GPU1. Outputs: `review/0517/disambig/per_hop_singlestep/`, report: `review/0517/disambig/per_hop_singlestep/SINGLESTEP_REPORT.md`, commit `5190659`. Decision: V7-V8 gain is concentrated in hop0 (`D50->D20 +0.2260 dB`); hop1-hop3 single-step deltas are near zero or negative. Channel A / chain coherence is the dominant interpretation, not a uniform shared-backbone direct gain.
- §2 ROI/high-SUV full-val completed on GPU1. Outputs: `review/0517/disambig/roi_psnr/`, report: `review/0517/disambig/roi_psnr/ROI_REPORT.md`. Decision: high-SUV/top1 regions are the clear failure mode, but D20 top1 SUV does not improve under V7 vs V8; SUVmax underestimation is largest at D20 and decreases toward NORMAL. This does not support NORMAL-only or multi-hop image_aux as the immediate primary move.
- Practical next step recommendation: do not launch V9/V11/V15 solely from the old V7-V8 NORMAL delta. Prefer either true single-variable controls (V13/V14-style) or a D20/top-SUV ROI-weighted hop0 experiment, with `PSNR_clip3 + top5/top1 SUV PSNR + SUVmax error` as required reporting metrics.
