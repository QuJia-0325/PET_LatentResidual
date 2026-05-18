# Next-Stage Architecture & Code Design — Final (post Round 7)

- date: 2026-05-17 深夜
- branch: foc_lite_hop0（草稿, push 前需 user 过一眼）
- 决策来源: [REVIEW_INTEGRATION_round7_20260517.md §6](./REVIEW_INTEGRATION_round7_20260517.md), user 选择 = **全推荐** (1接/2A/3加/4C/5四档/6做)
- 替换/Supersede: [NEXT_STAGE_EXPERIMENT_DESIGN_20260517.md](./NEXT_STAGE_EXPERIMENT_DESIGN_20260517.md)（保留作 audit trail, 但本文档优先）
- 硬约束: 服务器内存最多 3 个并行训练任务（Slot 1 = V18 ~step 180K/200K, 剩 12-18h）

---

## §0 — TL;DR

**今天 (阶段 A, 0 GPU)**: 文档清理 (§2.4 EV 区间, B12-B20 自警) + V21 retire 标记 + V13 smoke + V13 launch slot 2. **不**做 V18 评估前的任何 V18-clean 准备工作.

**Day +1~+2 (阶段 B)**: V18 step 200K eval + KL drift 测量 (重用 GAP_DECOMP probe).

**Day +2~+3 (阶段 C)**: 用 V18 真实数据 + KL drift 数据, **当场 git commit** V18-clean pre-register (含 4 档 threshold), 或转 backbone 方向.

整份设计的元教训 (B18): **不为还没出的数据预先设计响应分支**.

---

## §1 — 阶段 A: 0 GPU 文档与启动 (今天)

### 1.1 V18_design_rationale.md §2.4 新增 — EV 修正记录

**阈值表 (§2.3) 完全不动**. 原 PRIMARY +0.30 / PARTIAL [+0.05, +0.30] / KILL +0.05 维持.

新增 §2.4 内容:

```markdown
### 2.4 Round 5/6/7 EV 修正记录 (不改阈值, 只更新 interpretation)

#### attackable_gap 不是单一数字

| 量 | 值 | 物理含义 |
|---|---:|---|
| psnr_ceil = PSNR(decode_V7(z_GT), x_target) | 46.64 dB | RAE round-trip 上限 |
| psnr_transport = PSNR(decode_V7(z_pred^V7), x_target) | 35.44 dB | V7 baseline |
| psnr_transport_vs_ceil = PSNR(decode_V7(z_pred), decode_V7(z_GT)) | 35.94 dB | "若 z_pred=z_GT 仍用 V7 decoder" |

**V18 attackable gap = 区间 [0.05, 3] dB**, 不是单点:
- 下界 ~0.5 dB: 假设 V18 LoRA 只能学到 "decode(z_pred) 对齐到 decode(z_GT)" (吃 8% decoder ceiling 部分)
- 上界 ~3 dB: LoRA rank=32 on last 2 blocks (589K trainable / 24M total ≈ 2.5%) 的物理容量约束
- 中位估计 0.1-0.3 dB (落在 PARTIAL 区间)

#### 历史偏差 (作为前车之鉴, 不修改阈值)

| Bug | 形态 | Round |
|---|---|---|
| B10 | 11.2 dB 当 attackable (B5 同型, 高估) | 4 |
| B11 | 0.5 dB 当 attackable (B10 镜像错, 低估) | 5 |
| B18 | 在数据出来前 pre-design 每个 outcome 的响应分支 | 7 |
| B19 | yaml `decoder_kl_pullback.warmup_steps` 字段名误导, 实际是 `ramp_steps` (代码硬编码 `warmup_steps=0`) | 7 |
| B20 | resume 后 `max_steps` 是绝对 step 上限, 不是新训步数. V18-clean from V7 best.pt (step 160000) + max_steps=200000 = 仅训 40K 不是 200K | 7 |

#### 评估方式

V18 200K eval 时同时报告:
1. SUCCESS / PARTIAL / KILL by **原阈值** (单一判定, 无 dual-track)
2. ΔPSNR 在 [0.05, 3] dB 区间的位置 (纯描述性 interpretation, 不参与 SUCCESS 定义)
3. KL drift 数据 (见阶段 B), 用于 PARTIAL outcome 时判断是否触发 V18-clean
```

### 1.2 V18_design_rationale.md §5.2 自警条目扩展

```markdown
### 5.2 Standing rules (Round 1-7 累积)

B12 — 禁止以 "baseline 错了" mid-run 改预注册阈值. 修正只能通过 §2.4 文字描述,
      不通过替换 PRIMARY 数字, 不通过加 SECONDARY 列, 不通过 dual-track 并列.

B13 — 禁止 sunk-cost-resume 设计. paired ablation 必须从同一干净起点出发.
      (具体: 永久撤销 V18b "从 V18 ckpt 续训" 思路, 用 V18-clean from V7 替代)

B14 — yaml 改字段必须 cross-check 联动:
      - LR schedule total_steps_override
      - --resume CLI flag (training.resume_from yaml 字段是死字段)
      - LoRA→LoRA resume 不被支持 (_assert_v18_step0_equivalence 会 raise)

B15 — contingent slot 的真实概率必须披露 (e.g. V18-clean trigger ~50-70% 概率,
      不是 "0% 默认")

B16 — "outcome 独立" 框架的 narrow scope: 测量层独立 ≠ 项目优先级层独立.
      但 launch 决策只用测量层独立性.

B17 — paired comparison threshold 必须 cross-check 自身 noise floor (e.g. V18 vs
      V18-clean 的 paired noise ≠ V7 vs V7-seed1337 的 paired noise)

B18 — 禁止 pre-design every branch. 数据出来前不为每个 outcome 设计响应.

B19 — KL config `warmup_steps` 实际是 ramp_steps; 任何提到此字段处必须注 [实际=ramp_steps]

B20 — resume 后 max_steps 是绝对 step 上限; 任何 resume 实验文档必须注 "实际训练 N step"
```

### 1.3 V21 retire 残留清理 (本次必做)

在以下文档加 `[RETIRED 2026-05-17 Round 5: B8 — V21 描述与 conv_head.py 不符, 它是 token-domain 替换 (decoder_pred + unpatchify), 不是 post-decode pixel refiner]` 标记 (不删原文, 保留 audit trail):

- [REVIEW_INTEGRATION_round4_20260517.md](./REVIEW_INTEGRATION_round4_20260517.md) §6.2, §7 (V21 fallback yaml + decision tree)
- [V18_EXECUTION_REPORT_20260517.md](./V18_decoder_lora/V18_EXECUTION_REPORT_20260517.md) §Phase 0 (V21 launch 触发条件)
- [GAP_DECOMP_REPORT.md](./V18_decoder_lora/gap_decomp_fullval_20260517/GAP_DECOMP_REPORT.md) "Pre-Registered Decision Rule" 表
- [CODEX_RUNBOOK_V18_20260517.md](./CODEX_RUNBOOK_V18_20260517.md) 任何 V21 引用

每个被污染文档**顶部** ToC 加 1 行显眼 deprecation warning:
> **NOTE (2026-05-17)**: 本文档含已 retired 设计 V21. 见标记 `[RETIRED ...]` + Round 5 B8.

### 1.4 V13 launch (slot 2, 含 200-step smoke)

#### 1.4.1 V13 yaml 验证 (代码事实)

[V13_true_image_aux_off.yaml](../0516/V13_true_image_aux_ablation/V13_true_image_aux_off.yaml):
- `loss.image_aux.enabled: false`
- `loss.image_aux.lambda_max: 0` (双保险)
- `training.max_steps: 160000` (from-scratch, 与 V7 schedule lock)
- **无 resume_from** — V13 是 from-scratch, **不需 --resume**

[train_first_hop.py:1770, 2160-2161, 2368](PET_LatentResidual/train_first_hop.py) 验证 `image_aux.enabled=false` 分支:
```python
img_enabled = bool(image_aux_cfg.get("enabled", True))   # = False for V13
...
if img_enabled:                                          # 跳过 compute_hop0_image_losses
    img_losses = compute_hop0_image_losses(model, hop0_batch, cfg)
...
if img_enabled:                                          # loss 合并时 image_aux 项跳过
    total_loss += ...
```

**V13 路径在代码层面是 well-defined**, 但**从未在历史训练中跑过** (V7/V8/V18 都 image_aux=true). 必须 smoke 验证.

#### 1.4.2 V13 smoke (200 step, ~20 min)

```bash
# 在服务器, slot 2 空闲
cd /home/qujiaxiang/project/PET_LatentResidual
mkdir -p review/0516/V13_true_image_aux_ablation/smoke_runs
nohup python train_first_hop.py \
  --config review/0516/V13_true_image_aux_ablation/V13_true_image_aux_off.yaml \
  --max-steps-override 200 \
  --output-dir-override review/0516/V13_true_image_aux_ablation/smoke_runs/smoke_$(date +%Y%m%d_%H%M%S) \
  > review/0516/V13_true_image_aux_ablation/smoke_runs/smoke.log 2>&1 &
```

(注: 若 `train_first_hop.py` 无 `--max-steps-override` flag, codex 改用 yaml override 文件; 否则可考虑临时改 yaml 跑 smoke 再恢复)

**Smoke pass 标准** (codex 必须全部满足才能进 1.4.3):
- [ ] log 显示 `image_aux.enabled = false` (effective config print)
- [ ] 训练 metrics 里 `loss_img = 0.0` (200 step 内每一步都 0)
- [ ] 200 step 内**无任何 NaN** (loss_total, loss_pair, loss_roll, grad_norm)
- [ ] coverage / watchdog 指标正常 (与 V7 smoke 同等水平)
- [ ] GPU memory < 同硬件 V7 训练值的 1.1× (排除内存泄漏)
- [ ] step throughput ≥ V7 baseline 的 0.95× (image_aux 跳过应该更快, 不能更慢)

#### 1.4.3 V13 full launch

Smoke pass 后:
```bash
cd /home/qujiaxiang/project/PET_LatentResidual
nohup python train_first_hop.py \
  --config review/0516/V13_true_image_aux_ablation/V13_true_image_aux_off.yaml \
  > review/0516/V13_true_image_aux_ablation/V13_train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > review/0516/V13_true_image_aux_ablation/V13_train.pid
```

**预计时长**: ~7 天 (with V18 同 slot 1 + slot 3 空, 1 张卡, 160K step)
**Success 标准 (V13)**: 不预注册 SUCCESS/KILL; V13 是诊断不是优化. 任何 ΔPSNR(V13 vs V7) 都是有效信息.
**评估指标**: ΔPSNR(V13 vs V7) on NORMAL/D20/D10/D4, 4-tp 一致性 sanity, 用于 quantify image_aux 在 V7-V8 +0.308 dB 中真实贡献.

### 1.5 阶段 A 显式 NOT-DO

绝对不做:
- ❌ dual-track 阈值表 (Round 7 3/3 reject)
- ❌ SECONDARY +0.05 dB 列 (在 PARTIAL 隐式 trigger, 加列冗余)
- ❌ step 180K early-eval 三档判定 (B4 同型, agent2 还指出依赖字段 metrics.jsonl 不存在)
- ❌ V18-clean 预先 pre-register (B18 反模式, 推迟到阶段 C)
- ❌ V18-clean yaml draft (推迟)
- ❌ 起 Round 8 review (违反 "不再 review" 教训, 7 轮已穷尽)
- ❌ 改 V18 yaml 任何字段 (V18 在跑, 触发 config mismatch)
- ❌ V18b 任何形态 (永久撤销, Round 6 共识)

### 1.6 阶段 A push gitee

完成 1.1-1.4 后, git commit:
- V18_design_rationale.md 更新 (§2.4 + §5.2)
- 4 个 V21 残留文档加标记
- V13 smoke + launch logs (不 commit smoke run 数据本身, 只 commit log 摘要)
- 本文档 (NEXT_STAGE_ARCH_CODE_FINAL_20260517.md)

push 到 gitee. 阶段 A 结束.

---

## §2 — 阶段 B: V18 200K eval + KL drift 测量 (Day +1~+2)

### 2.1 触发条件

V18 训练自然到 max_steps=200K 完成 (~12-18h 后, 取决于剩余时间). slot 1 释放.

### 2.2 V18 200K full-val eval

#### 2.2.1 执行命令

```bash
cd /home/qujiaxiang/project/PET_LatentResidual
python eval_first_hop_224_clip3.py \
  --config review/0517/V18_decoder_lora/V18_decoder_lora.yaml \
  --checkpoint <V18 run dir>/checkpoints/step_200000.pt \
  --output-dir review/0517/V18_decoder_lora/eval_200K_$(date +%Y%m%d_%H%M%S) \
  --eval-mode chain_normal \
  --baseline-checkpoint <V7 best.pt 绝对路径>
```

(具体 CLI flag 可能与代码不完全一致, codex 按实际 [eval_first_hop_224_clip3.py](PET_LatentResidual/eval_first_hop_224_clip3.py) argparse 调整)

#### 2.2.2 输出 V18_FINAL_EVAL_<TS>.md

报告内容:

```markdown
# V18 200K Final Eval

## §1 主要数字
- ΔPSNR(V18 vs V7) on NORMAL_chain (PSNR_clip3): <值> dB
- ΔPSNR on D20/D10/D4 (4-tp 一致性 sanity):
  - D20: <值> dB
  - D10: <值> dB
  - D4: <值> dB
  - 一致性: ALL POSITIVE / MIXED / ALL NEGATIVE

## §2 判定 (by 原阈值)
- PRIMARY threshold: +0.30 dB
- 判定: SUCCESS / PARTIAL / KILL
- 在 EV 区间 [0.05, 3] dB 位置: HIGH (≥+0.5) / MID (+0.1~+0.5) / LOW (<+0.1)

## §3 KL drift 测量 (见 §2.3)
- psnr_v18_at_zgt: <值> dB
- vs psnr_v7_at_zgt = 46.64 dB
- KL drift: |psnr_v18_at_zgt − 46.64| = <值> dB
- 判定: NEGLIGIBLE (<0.5) / MODERATE (0.5-1.0) / SIGNIFICANT (>1.0)

## §4 V18 loss/grad 健康指标 (最后 5K step 摘要)
- loss_total trend: stable / drifting
- KL loss component (λ_kl × loss_kl) 占总 loss 比例: <%>
- grad_norm: stable / blow-up

## §5 阶段 C 决策 input (不在本文档做决策, 见 §3)
- V18 outcome × KL drift = (SUCCESS/PARTIAL/KILL × NEGLIGIBLE/MODERATE/SIGNIFICANT)
- 用此 cell 查 §3 决策表
```

### 2.3 KL drift 测量 (Round 7 agent3 独家高 EV 提议)

#### 2.3.1 目的

直接量化 B9 (KL pullback `use_pred_latent=true` 与 image_aux 梯度对抗) 是否真的把 V18 decoder 拉离了 GT manifold.

#### 2.3.2 方法 (重用 GAP_DECOMP probe)

修改 [tools/probe_v18_gap_decomposition.py](PET_LatentResidual/tools/probe_v18_gap_decomposition.py) 或新建 `tools/probe_v18_kl_drift.py`, ~50 行:

```python
# 伪代码
from pet_lr.model_first_hop import build_model_from_cfg
import yaml, torch

cfg = yaml.safe_load(open("review/0517/V18_decoder_lora/V18_decoder_lora.yaml"))
model = build_model_from_cfg(cfg)
ckpt = torch.load("<V18 step_200000.pt>", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval().cuda()

# 用 val split 算 PSNR(decode_V18(z_GT), x_target)
psnr_v18_at_zgt = []
for batch in val_loader:
    z_gt = batch["z_dst"].cuda()       # ground truth latent
    x_target = batch["x_dst"].cuda()   # ground truth pixel
    with torch.no_grad():
        x_v18 = model.rae.decode(z_gt)   # V18 LoRA decoder on z_GT (no transport involved)
    psnr_v18_at_zgt.append(calc_psnr_clip3(x_v18, x_target))

print(f"psnr_v18_at_zgt = {np.mean(psnr_v18_at_zgt):.4f} dB")
print(f"psnr_v7_at_zgt  = 46.64 dB (from GAP_DECOMP)")
print(f"KL drift = {abs(np.mean(psnr_v18_at_zgt) - 46.64):.4f} dB")
```

**资源**: ~30 min single GPU, 与 V18 200K eval 共用一张卡 (slot 1, V18 已释放). V13 在 slot 2 不受影响.

#### 2.3.3 解读

| KL drift | 含义 | 触发 |
|---|---|---|
| < 0.5 dB | NEGLIGIBLE — V18 decoder 仍贴近 GT manifold, KL pullback 在工作 → **B9 实际影响小** | V18-clean 不需要做 (KL 没问题) |
| 0.5 - 1.0 dB | MODERATE — 部分 drift, B9 可能 marginal helpful | 需结合 V18 outcome 综合判断 |
| > 1.0 dB | SIGNIFICANT — KL pullback 没拉住, V18 decoder 漂走 → **B9 是真 ceiling** | V18-clean 高 EV, 必跑 |

---

## §3 — 阶段 C: 数据驱动决策 (Day +2~+3)

### 3.1 3×3 决策矩阵

将 V18 outcome (SUCCESS/PARTIAL/KILL) × KL drift (NEGLIGIBLE/MODERATE/SIGNIFICANT) 组合, 决定 slot 3 行动:

| V18 outcome \ KL drift | NEGLIGIBLE (<0.5) | MODERATE (0.5-1.0) | SIGNIFICANT (>1.0) |
|---|---|---|---|
| **SUCCESS** (ΔPSNR ≥ +0.30) | slot 3 = null. Paper draft. | slot 3 = null. Paper draft + KL drift note. | slot 3 = null. Paper draft + B9 limitation note. (V18 SUCCESS 即使 KL drift 大, 也不需 V18-clean — 上限已达成) |
| **PARTIAL** ([+0.05, +0.30]) | slot 3 = **backbone/data 方向 phase plan**. B9 不是 bottleneck, V18 已达 LoRA 容量上限. | slot 3 = **暂缓决策, 与 user 沟通**. B9 unclear, V18-clean ROI 不明. | slot 3 = **V18-clean from V7 launch** (按 §3.2 pre-register). 高 EV. |
| **KILL** (<+0.05) | slot 3 = backbone/data/architecture 方向. V18 family retire. | slot 3 = backbone 方向. V18-clean 也救不了 KILL. | slot 3 = backbone 方向. KL drift 显著说明 decoder LoRA 整体方向错, B9 fix 不够. |

**9 个 cell 中 V18-clean 实际触发只在 1 个 cell** (PARTIAL × SIGNIFICANT). 这把 V15 "contingent slot inflation" 担忧降到最低.

### 3.2 V18-clean pre-register (仅在 §3.1 触发时执行)

#### 3.2.1 触发动作: 当场 git commit

```bash
# 在 V18_FINAL_EVAL.md 写完后, 立即:
cd /home/qujiaxiang/project/PET_LatentResidual
mkdir -p review/0517/V18_clean_kl_gt_anchor

# 创建 pre-register 文件
cat > review/0517/V18_clean_kl_gt_anchor/V18_CLEAN_PREREGISTER_$(date +%Y%m%d_%H%M%S).md <<'EOF'
# V18-clean Pre-Register (signed at <timestamp>)

Triggered by: V18 PARTIAL (ΔPSNR=<v> dB) × KL drift SIGNIFICANT (<d> dB).

Pre-registered success threshold (4 档, B17 noise floor 校正):
| ΔPSNR(V18-clean vs V18) | 判定 | Action |
|---|---|---|
| < +0.02 dB | B9 影响可忽略 | paper 一句话 limitation, 不改 V18 默认 |
| [+0.02, +0.08) | inconclusive (信噪比不足, 落在 d_pure noise band) | 不 escalate, paper 报告区间 |
| [+0.08, +0.15) | B9 fix materially helpful | paper 报告两个数字, 推荐未来 V18 默认 use_pred_latent=false |
| ≥ +0.15 dB | B9 fix strongly effective | 改 V18 默认配置, 重新 evaluate 是否 supersede V18 |

4-tp 一致性 sanity (必须报告, 不参与 threshold 判定):
- NORMAL/D20/D10/D4 同符号 → high confidence
- 反向 → 标记 INCONCLUSIVE, 降级到 lower tier
EOF

git add review/0517/V18_clean_kl_gt_anchor/V18_CLEAN_PREREGISTER_*.md
git commit -m "Pre-register V18-clean (PARTIAL + SIGNIFICANT KL drift trigger)"
git push gitee
```

**关键**: pre-register **必须先 commit + push 再 launch 实验**. 这是真 commitment, 防止数据出来后挑数 (B12).

#### 3.2.2 V18-clean yaml 设计

`review/0517/V18_clean_kl_gt_anchor/V18_clean_kl_gt_anchor.yaml`, 基于 V18 yaml, **只改 3 字段**:

| 字段 | V18 | V18-clean | 理由 |
|---|---|---|---|
| `output_dir` | `.../V18_decoder_lora/run` | `.../V18_clean_kl_gt_anchor/run` | 隔离 ckpt |
| `run_name` | `first_hop_224_v18_decoder_lora` | `first_hop_224_v18_clean_kl_gt_anchor` | wandb 区分 |
| `loss.decoder_kl_pullback.use_pred_latent` | `true` | **`false`** | B9 ablation 的 1 bit |

**不改** (全部保持与 V18 一致, 确保 paired comparison cleanness):
- rank=32, blocks=[6,7]
- λ_kl=0.05, λ_img=0.04
- lr_schedule.total_steps_override=200000
- warmup_steps=2000 (B19 注: 实际是 ramp_steps; V7 best.pt 起点 step≈160000 已 saturate, 实际 KL 从第 1 步起就是满 0.05)
- max_steps=200000 (B20 注: 这是绝对 step 上限; 从 step 160000 起跑实际只训 ~40K step, 非 200K)
- seed (与 V18 同 launch seed)
- 其他全部继承

#### 3.2.3 V18-clean launch (must include `--resume`)

```bash
cd /home/qujiaxiang/project/PET_LatentResidual

# slot 3 launch (假设 slot 1 已空闲于 eval 之后, slot 2 = V13)
nohup python train_first_hop.py \
  --config review/0517/V18_clean_kl_gt_anchor/V18_clean_kl_gt_anchor.yaml \
  --resume /data_2/qujiaxiang/.../V7/run/.../best.pt \
  > review/0517/V18_clean_kl_gt_anchor/V18_clean_train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > review/0517/V18_clean_kl_gt_anchor/V18_clean_train.pid

# 注意: 
# - --resume 必须显式 (training.resume_from yaml 字段是死字段, 不被 train_first_hop.py 读取)
# - V18 当时也是这样 launch, 见 V18_TRAIN_COMMAND_20260517.txt
```

#### 3.2.4 V18-clean 真实成本

| 量 | 值 | 来源 |
|---|---:|---|
| 起点 step | 160000 | V7 best.pt |
| 终点 step | 200000 | max_steps |
| 实际训练步数 | **~40000** | B20 修正 |
| 实际训练时长 | ~1.4 天 | V18 同 40K 用时 |
| GPU 占用 | slot 3 × 1.4 天 | 而非 7 天 (V15 担忧弱化) |

#### 3.2.5 V18-clean 评估 (~Day +4)

```bash
python eval_first_hop_224_clip3.py \
  --config review/0517/V18_clean_kl_gt_anchor/V18_clean_kl_gt_anchor.yaml \
  --checkpoint <V18-clean run dir>/checkpoints/step_200000.pt \
  --output-dir review/0517/V18_clean_kl_gt_anchor/eval_200K_$(date +%Y%m%d_%H%M%S) \
  --baseline-checkpoint <V18 step_200000.pt>
```

**Primary comparison**: V18-clean step_200000.pt vs V18 step_200000.pt (matched endpoint).
**Secondary**: V18-clean best.pt vs V18 best.pt (best-selection criterion 必须相同, 见 §4.2).

按 §3.2.1 pre-registered 4 档判定, 写 `V18_CLEAN_FINAL_EVAL_<TS>.md`. 不允许临时改 threshold.

### 3.3 阶段 C 的 git discipline

| 步骤 | git 操作 |
|---|---|
| V18_FINAL_EVAL.md 写完 | commit + push (任何人看到的 ΔPSNR 都被 git history 锁定) |
| §3.1 决策矩阵查表后 (V18-clean trigger 或 backbone 转向) | commit 决策 doc + push |
| V18-clean pre-register (仅 trigger 时) | commit + push **before launch** |
| V18-clean launch log | commit log 摘要 |
| V18-clean eval 结果 | commit + push, 按 pre-registered 4 档严格判定 |

---

## §4 — 代码层面安全网

### 4.1 V13 路径完整性

代码事实 (Round 7 agent2 + 本次 verify):

| 路径 | 是否 `image_aux.enabled=false` 时跳过 | 验证 |
|---|---|---|
| `compute_hop0_image_losses` | ✅ 跳过 | [train_first_hop.py:2160-2161](PET_LatentResidual/train_first_hop.py) `if img_enabled:` |
| image loss 在 `total_loss` 中合并 | ✅ 跳过 | [train_first_hop.py:2368](PET_LatentResidual/train_first_hop.py) `if img_enabled:` |
| `img_enabled` 解析 | ✅ 正确 | [train_first_hop.py:1770](PET_LatentResidual/train_first_hop.py) `bool(image_aux_cfg.get("enabled", True))` |

⚠️ Smoke 必须验证: log 显示 `effective image_aux.enabled = False`, 200 step 内 `loss_img = 0`.

### 4.2 V18-clean 不触发 LoRA→LoRA resume blocker

代码事实 (Round 6/7 agent2 + 本次 verify):

- [train_first_hop.py:98 `_remap_state_dict_for_decoder_lora`](PET_LatentResidual/train_first_hop.py): 把 V7 (pre-LoRA) Linear key 映射到 V18 LinearWithLoRA 的 base key. **V7→V18 必经路径**.
- [train_first_hop.py:130 `_assert_v18_step0_equivalence`](PET_LatentResidual/train_first_hop.py): V18 step 0 必须 output-equivalent to frozen decoder, 否则 raise. **要求 LoRA A/B 是 fresh (零初始化)**.

V18-clean from V7 best.pt:
- ✅ 触发 `_remap_state_dict_for_decoder_lora` (V7 是 pre-LoRA)
- ✅ LoRA A/B 是 fresh (V18-clean 自己也是新 LoRA wrap)
- ✅ `_assert_v18_step0_equivalence` 通过 (output equal frozen decoder, err < 1e-5)

**V18-clean 走的就是 V18 已验证的 resume 路径, 不需要任何代码补丁**.

(对比: V18b "从 V18 ckpt 续" 会 fail step0_equivalence, 因为 V18 LoRA A/B 已训过非零. 这是 Round 6 agent2 hard blocker 的 root cause, 也是 V18b 被永久撤销的代码理由.)

### 4.3 KL schedule 行为 (B19 + V18-clean 推论)

代码事实:
```python
# train_first_hop.py:2222-2228
lambda_kl = get_linear_schedule_value(
    global_step=step,            # absolute, resume 后从 ckpt step 起
    warmup_steps=0,              # 硬编码
    ramp_steps=int(kl_cfg.get("warmup_steps", 0)),   # yaml warmup_steps 实际是 ramp_steps
    start=0.0, end=lambda_kl_max,
)
```

V18-clean from V7 best.pt (step ≈ 160000):
- step=160001: `(160001-0)/2000` clipped → λ_kl = 0.05 (满载)
- **V18-clean 第 1 步起就是满 KL**, 不会重新 ramp

**含义**: V18-clean yaml `warmup_steps: 2000` 是装饰性 (保持与 V18 yaml bit-level 对称), 实际不起作用. 这必须写在 V18-clean pre-register doc 里, 防止未来 reviewer 误以为 V18-clean 有 KL warmup 期.

### 4.4 best.pt 选择 criterion 一致性

代码事实需 codex 二次确认 (CODEX_CODE_AUDIT_REQUEST §6.3):

- V18 best.pt 选择: `val_chain_normal_mse` 最低 (推测) — 训练器 metrics.jsonl 只有 MSE
- V18-clean best.pt 必须用**同 criterion** (否则 V18-clean best vs V18 best 是 confounded)

⚠️ 若 V18-clean yaml 与 V18 yaml 在 `training.best_metric` / `best_select_*` 字段有任何差异, 必须显式 align.

### 4.5 评估 PSNR_clip3 路径 (Round 7 B 阶段依赖)

代码事实 (Round 7 agent2 揭示):
- 训练器 metrics.jsonl 只记 MSE (`val_chain_normal_mse`)
- PSNR_clip3 只在离线 `eval_first_hop_224_clip3.py` 算

阶段 B 必须用 `eval_first_hop_224_clip3.py`, 不能从 metrics.jsonl 抓 PSNR.

---

## §5 — Resource accounting (max=3 硬约束)

时间线下 slot 占用:

```
                    Slot 1            Slot 2          Slot 3
今天 (阶段 A)       V18 (180→200K)    V13 smoke→launch  空
Day +1              V18 (190→200K)    V13 (~5%)        空
Day +2 (阶段 B)     V18 200K eval +    V13 (~15%)       空
                   KL drift probe                       
Day +2 (阶段 C)     [decision]        V13 (~20%)       [contingent launch]
Day +4              [free]            V13 (~30%)       V18-clean (~0% or null)
Day +4 (V18-clean eval, if triggered)  V13 (~30%)       V18-clean step_200000 eval
Day +8              [free]            V13 (~60%)       [free]
Day +10             [free]            V13 finish       [free]
```

**任何时刻 ≤ 3 任务** ✅.

特别注意:
- 阶段 B eval (V18 200K + KL drift) 在 V18 释放后的 slot 1 上, 与 V13 (slot 2) 不冲突. eval 总时长 ~1h, 不长占.
- 阶段 C V18-clean 若 launched, 占 slot 3 × 1.4 天 (而非 7 天, B20 修正).
- V18-clean eval ~30 min, 在 slot 3 V18-clean 训完后立即跑, 不与 V13 抢卡 (因为不同 GPU).

**Eval 与 train 同卡冲突防护**: 任何 full-val PSNR eval 占同张卡时, 应在 train pause window (next checkpoint 间隔) 内插入, 或单独排队. V18 200K eval 是 V18 已结束, 不冲突.

---

## §6 — 失败模式与回滚

### 6.1 阶段 A 失败模式

| 故障 | 现象 | 回滚 |
|---|---|---|
| V13 smoke NaN | smoke.log 出现 nan | 不 launch full V13, 进 debug. slot 2 转 V14 (V7 + seed 1337, low-EV 但安全 fallback) |
| V13 smoke loss_img ≠ 0 | log 显示 image loss 非零 | 不 launch, 调查 `image_aux.enabled=false` 是否被某代码路径忽略 |
| V13 smoke GPU OOM | 内存不够 | 不 launch, 检查 batch_size 配置 |
| V13 smoke throughput < V7 0.95× | image_aux 跳过反而更慢 | 怀疑有未跳过的 forward 路径, debug |

任何一条不通过 → V13 不 launch, slot 2 维持 null. 不影响 V18 (slot 1) 和 阶段 B.

### 6.2 阶段 B 失败模式

| 故障 | 现象 | 回滚 |
|---|---|---|
| V18 train 异常崩溃在 step 200K 之前 | crash log | 用最后 ckpt eval (≤ step 200K), 标 "incomplete run" |
| eval_first_hop_224_clip3.py 报错 | eval crash | 检查 ckpt + config 是否匹配, 必要时用 step_195000.pt 备份 |
| KL drift probe 报错 | probe crash | 必须 fix 才进阶段 C (KL drift 是决策关键 input) |

### 6.3 阶段 C 失败模式

| 故障 | 现象 | 回滚 |
|---|---|---|
| V18-clean launch raise (step0_equivalence fail) | RuntimeError | 不应该发生 (V7→V18 已验证). 若发生, 检查 V7 best.pt 是否被改动 |
| V18-clean training NaN/divergence | crash | 不预设 fix, 报告并 retire V18-clean branch (B9 fix 不 robust) |
| V18-clean eval 出 ΔPSNR 落在 4 档边界 | 例如 +0.020001 dB | 严格按 pre-registered 4 档, 不允许 round. 边界值归低 tier |

---

## §7 — claude 自警 (Round 7 carry forward + 本次)

继承 Round 1-7 偏差识别 (B1-B20, 见 §1.2). 本次起草中:

- **B18 复发尝试 #1**: 起草 §3.1 决策矩阵时, 我一度想为 "SUCCESS × SIGNIFICANT KL drift" 这个 cell 写 "考虑后续 ablation". 但 SUCCESS 已意味 V18 达成上限, 任何 ablation 都不能 supersede. 撤回, 改为 "Paper draft + B9 limitation note", 不增加 GPU commitment.
- **B14 复发尝试 #1**: 起草 §3.2.2 V18-clean yaml diff 时, 列了 3 字段但漏了 `output_dir` 和 `run_name`. 已补全. 这两个非内容字段, 但漏掉会让 V18-clean ckpt 写到 V18 目录, 覆盖 V18 数据.
- **B20 自警生效**: 起草 §3.2.4 真实成本表时, 主动标 "40000 step (B20 修正)", 没默认写 "200K step / 7 天".

---

## §8 — 下一步动作

**今天 user 阅读本文档** → 过 ok → claude 起草 `CODEX_TASK_PHASE_A_20260517.md` (阶段 A 给 codex 的可执行 task md) → user 再过一眼 → push gitee.

codex 在服务器执行阶段 A → 反馈结果 → 阶段 B 启动 (V18 200K 完成 trigger).

---

## §9 — 文档关系图

```
NEXT_STAGE_ARCH_CODE_FINAL_20260517.md (本文档, 最终架构 & 代码设计)
  ├── supersedes: NEXT_STAGE_EXPERIMENT_DESIGN_20260517.md (保留 audit trail)
  ├── 决策来源: REVIEW_INTEGRATION_round7_20260517.md §6 (user = 全推荐)
  ├── 代码事实来源:
  │     ├── train_first_hop.py:98-160, 1770, 2160-2161, 2222-2228, 2368
  │     ├── pet_lr/decoder_lora.py
  │     ├── V18_TRAIN_COMMAND_20260517.txt (resume CLI 实证)
  │     └── GAP_DECOMP_REPORT.md (EV 区间数据)
  ├── 阶段 A 待 codex task md: CODEX_TASK_PHASE_A_20260517.md (尚未起草)
  ├── 阶段 B 待 codex task md: 阶段 A 完成后起草
  └── 阶段 C 待 codex task md: 阶段 B 完成 + V18 数据已知后起草
```
