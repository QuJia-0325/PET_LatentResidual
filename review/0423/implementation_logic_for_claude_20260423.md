# PET_LatentResidual 实现逻辑说明（给 Claude）

日期：2026-04-23  
分支：`foc_lite_hop0`  
目的：把当前实现链路讲清楚，尤其说明 backbone 来源、PET_LatentResidual 在其上的增量机制，以及为什么当前判断瓶颈在 transport。

---

## 1. 一句话全景

当前系统不是“从零训练一个 transport 模型”，而是：

1. 先在 RAE 侧训练一个 224 的 4-hop hop-aware mean-flow backbone（这一步是 smoke 规模，2000 steps）；
2. PET_LatentResidual 读取这个 backbone checkpoint 作为初始化；
3. 在其上加入 first-hop 机制（pixel forcing + hop residual），并做 50k 微调训练与 clip3 评估。

---

## 2. Backbone 来源调查结果（已核实）

### 2.1 真实 checkpoint 与元数据

目标 checkpoint（PET_LatentResidual 配置里引用）：
- `/data_2/qujiaxiang/outputs/pet_flow_224_small_rollout_hopaware_smoke/mean_flow_224_small_rollout4hop_hopaware_smoke/best.pt`

元数据（直接加载 checkpoint 得到）：
- `epoch = 1`
- `global_step = 2000`
- `best_metric_name = rollout_total`
- `best_metric = 0.01848421052227097`
- `target_normalize = False`
- `rollout_path = [D50, D20, D10, D4, NORMAL]`
- `model_ema` 存在，`model` 也存在
- `pos_embed` 形状：`(1, 256, 384)`（16x16 token，对应 224/14）

结论：这是一个 **2000-step smoke backbone**，不是长训练 backbone。

### 2.2 配置与日志位置

同源输出目录（RAE 侧）实际包含：
- `config.yaml`
- `train.log`
- `checkpoints/ckpt_step_{500,1000,1500,2000}.pt`
- `best.pt`

关键日志信息（`train.log`）：
- 启动：`2026-03-31 23:51:58`
- `Starting ... max_steps=2000`
- 结束：`Training complete! Steps: 2000, Best rollout_total: 0.0185`

### 2.3 /data_2 与 /home/.../RAE/outputs 是否一致

两处 `best.pt` 校验一致：
- 文件大小一致（`3281512190`）
- `sha256` 完全一致：
  - `4f49766f1b87239ac82e455633345cc3fd5c0d87570fef0bf03342069cd12cf5`

说明：PET_LatentResidual 引用的 checkpoint 与 RAE 输出目录中的 checkpoint 是同版本。

---

## 3. RAE 侧（backbone 预训练）到底做了什么

训练入口代码：
- `/home/qujiaxiang/project/RAE/code/RAE/src/pet_flow/train_pet_flow_rollout_4hop_hopaware.py`

对应配置模板：
- `/home/qujiaxiang/project/RAE/code/RAE/224_bundle/pet_flow_mf_224_small_rollout4hop_hopaware_smoke.yaml`

关键训练设定（来自 `config.yaml`）：
- `max_steps: 2000`
- `batch_size: 4`, `grad_accum_steps: 8`
- `ema.enabled: true`, `ema.decay: 0.9999`
- `transport.target_normalize: false`
- rollout：
  - `path = [D50, D20, D10, D4, NORMAL]`
  - `alpha_start=0.0 -> alpha_end=1.0`
  - `step_weights = [1.50, 1.25, 1.0, 1.0]`

因此，这个 backbone 的性质是：
- 结构对齐主任务（4-hop hop-aware）
- 但训练预算偏小（2000 steps），更像 pipeline/smoke 验证权重

---

## 4. PET_LatentResidual 如何接入这个 backbone

核心代码：
- `pet_lr/model_first_hop.py:569-595`

逻辑是：
1. 构建 `PETFlowDiTDHHopAware`（架构参数从当前 PET config 读）；
2. 从 `backbone.checkpoint_path` 加载 checkpoint；
3. 优先读取 `model_ema`（否则 `model`）；
4. 严格 `load_state_dict(..., strict=True)`；
5. 返回 `target_normalize`（来自 checkpoint 元数据）。

这说明：
- PET_LatentResidual 并没有重写 backbone 架构；
- 它直接继承了 RAE 侧训练出来的权重语义（含 `target_normalize`）。

---

## 5. PET_LatentResidual 在 backbone 上新增了什么

在 `PETFlowDiTFirstHop` 中新增/控制：

1. **hop0 pixel forcing**（仅第一跳使用像素条件）
2. **hop residual velocity head**（hop-specific 速度残差）
3. rollout + hop0 image aux 联合训练策略
4. checkpoint/语义 guard（如 resume 兼容、N2 约束）

训练入口：
- `train_first_hop.py`

关键点（v2）：
- `freeze_backbone: false`（允许微调 backbone）
- 优化器分组为 `backbone` 和 `first_hop`
- `first_hop_weight_decay: 0.0`（当前实现是 first_hop 整组 0wd，不只标量 gate）
- 启用 EMA（与 RAE 习惯对齐）

---

## 6. 目前瓶颈判断为何指向 transport

已知对照（full-val, clip3）：
- decoder ceiling（GT latent 直解码）远高于当前 transport 结果
- D20/D10/D4/NORMAL gap 约 `10.7 ~ 16.2 dB`

因此当前合理表述应是：
- **“在 GT latent / on-manifold 条件下，decoder 不是主导瓶颈；当前主误差更可能来自 transport 预测链路”**

注意不要写成绝对句：
- 不建议写“decoder 完全不是瓶颈”
- 更不建议写“旧实验全部无效”

---

## 7. Claude 最容易混淆的点（直接澄清）

1. **混淆点 A：backbone 是不是 PET_LatentResidual 里训练出来的？**
- 不是。backbone 初始权重来自 RAE 侧输出（smoke 2000-step）。

2. **混淆点 B：现在训练是不是只在训 first-hop 分支？**
- 不是。当前 v2 `freeze_backbone=false`，backbone 与 first-hop 分支都在训（除 frozen RAE 解码器）。

3. **混淆点 C：checkpoint 用的是 raw model 还是 EMA model？**
- 加载时优先 `model_ema`（`build_backbone` 明确如此）。

4. **混淆点 D：target_normalize 是从配置写死吗？**
- 不是。它从 backbone checkpoint 元数据读取并返回给 PET 模型。

---

## 8. 对后续实验决策的直接含义

1. 现有 backbone 仅 2000-step，预训练充分性存在疑问；
2. v2 的 first-hop 修复（wd/init/ema）可以继续验证，但不能替代 backbone 质量问题本身；
3. 若 v2 改善有限，需优先评估：
- 是微调策略问题，还是 backbone 预训练预算不足；
- 是否需要先在 RAE 侧补一个更长训练（例如 >=20k / 50k）backbone 再迁移。

---

## 9. 关键证据文件清单

- Backbone 调查需求：
  - `review/0423/backbone_checkpoint_investigation.md`
- RAE 输出（同源）：
  - `/home/qujiaxiang/project/RAE/outputs/pet_flow_224_small_rollout_hopaware_smoke/mean_flow_224_small_rollout4hop_hopaware_smoke/config.yaml`
  - `/home/qujiaxiang/project/RAE/outputs/pet_flow_224_small_rollout_hopaware_smoke/mean_flow_224_small_rollout4hop_hopaware_smoke/train.log`
- PET 侧加载逻辑：
  - `pet_lr/model_first_hop.py:569-595`
- PET 侧训练逻辑：
  - `train_first_hop.py:1139-1230`
- 当前 v2 配置：
  - `configs/pet_flow/pet_flow_first_hop_224_50k_schemec_v2.yaml`


---

## 10. `resume` 使用情况（基于训练日志的直接证据）

结论先行：

- **主线一阶段训练（如 Scheme C v2）**：未使用 `--resume`（fresh start，backbone 从 smoke checkpoint 初始化）。
- **N1（pixenc ablation）**：使用了 `--resume`（从 step_010000 继续）。
- **N2（seam_refiner standalone）**：使用了 `--resume`（并且代码层面要求必须 resume）。
- **RAE smoke backbone 训练本身**：从头跑到 2000 step（日志无 resume 迹象）。

### 10.1 证据表

| 运行 | 是否使用 resume | 关键日志证据 | 对“起点权重”的含义 |
|---|---|---|---|
| RAE backbone smoke (`mean_flow_224_small_rollout4hop_hopaware_smoke`) | 否 | `Starting ... max_steps=2000`，`Training complete! Steps: 2000` | 该权重本身是从头训练得到 |
| PET Scheme C v2 (`first_hop_224_50k_schemec_v2`) | 否 | startup 段仅有 `loading config/run dir/...`，**无** `resume from` | 有效起点 = smoke backbone + 新初始化 first-hop 模块 |
| PET N1 (`first_hop_224_50k_pixenc_ablation`) | 是 | `[startup] resume from: ...step_010000.pt`，`[resume] loaded step=10000` | 有效起点 = N1 已训练 checkpoint（覆盖构模时的smoke加载） |
| PET N2 (`first_hop_224_20k_seam_refiner_standalone`) | 是 | `[startup] resume from: ...imgaux_boost.../best.pt`，`[resume] loaded step=0 ... warm-start` | 有效起点 = stage1 transport ckpt（再新增 seam_refiner 模块） |

### 10.2 日志摘录（关键行）

- RAE smoke：
  - `2026-03-31 23:53:47 - INFO - Starting 4-hop hop-aware rollout training: max_steps=2000`
  - `2026-04-01 01:16:04 - INFO - Training complete! Steps: 2000, Best rollout_total: 0.0185`

- Scheme C v2：
  - `1:[startup] loading config: configs/pet_flow/pet_flow_first_hop_224_50k_schemec_v2.yaml`
  - `2:[startup] run dir: /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_schemec_v2`
  - （startup 段未出现 `resume from`）

- N1：
  - `3:[startup] resume from: /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_pixenc_ablation/step_010000.pt`
  - `61:[resume] loaded step=10000, ...`

- N2：
  - `3:[startup] resume from: /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_imgaux_boost_gpu2_nopbar_20260418_010035/best.pt`
  - `102:[resume] loaded step=0, best_val=inf, best_metric_name=val_multi_objective`

---

## 11. 附件：已归档到本仓库的对应训练日志

为避免口头争议，以下日志已复制到本目录：

- `review/0423/logs_resume_evidence/backbone_smoke_train.log`
- `review/0423/logs_resume_evidence/schemec_v2_train_gpu3_20260422_1118.log`
- `review/0423/logs_resume_evidence/n1_pixenc_ablation_train_gpu1_resume_20260422_1109.log`
- `review/0423/logs_resume_evidence/n2_seam_refiner_standalone_train_gpu3_20260422_0249.log`

如果 Claude 只看一个判断句，请用：

> “我们当前主线 Scheme C v2 训练没有用 `--resume`；但 N1/N2 这些特定实验是用了 `--resume` 的。所有结论请按 run 区分，不要混为一条。”

