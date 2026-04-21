# 0421 代码问题与修复方案（N1/N2 最新提交审查）

## 范围

- 分支：`foc_lite_hop0`
- 审查对象：N1/N2 实验矩阵对应最新代码与配置
- 相关文件：
  - `train_first_hop.py`
  - `pet_lr/model_first_hop.py`
  - `eval_first_hop_224_clip3.py`
  - `configs/pet_flow/pet_flow_first_hop_224_20k_seam_refiner_stage2.yaml`
  - `configs/pet_flow/pet_flow_first_hop_224_50k_pixenc_ablation.yaml`
  - `exp_result/0421/Local/README.md`

---

## 发现的问题（按严重度）

### P0（阻断运行）`train_cfg` 在赋值前被使用

- 位置：
  - `train_first_hop.py:1140` 使用 `train_cfg.get("freeze_first_hop_modules", False)`
  - `train_first_hop.py:1203` 才定义 `train_cfg = cfg["training"]`
- 影响：
  - 训练启动时会触发 `UnboundLocalError`，N2 无法开跑。
- 修复方案：
  1. 在 `main()` 中将 `train_cfg = cfg["training"]` 提前到第一次使用前（建议放在 `train_cfg_boot` 附近）。
  2. 删除后续重复赋值，避免二义性。
- 验证：
  - 运行 `python -m py_compile train_first_hop.py`
  - 本地 smoke：`--config ... --max_steps` 小步测试，确认能进入训练循环。

---

### P1（高风险配置偏差）N2 YAML 存在重复且冲突的 resume 兼容键

- 位置（同一文件重复定义）：
  - `pet_flow_first_hop_224_20k_seam_refiner_stage2.yaml:57-60`
  - `pet_flow_first_hop_224_20k_seam_refiner_stage2.yaml:109-112`
- 现象：
  - 前半段是“放宽兼容”；
  - 后半段被覆盖为“严格兼容”。
  - YAML 解析后后者生效，导致注释目标与实际行为相反。
- 影响：
  - N2 `--resume Scheme C best.pt` 可能因签名/配置差异而报错，或者行为不符合实验设计预期。
- 修复方案：
  1. 删除重复键，仅保留一组兼容策略。
  2. 若目标是 stage-2 warm-start，建议保留：
     - `strict_resume_compat: false`
     - `resume_allow_metric_mismatch: true`
     - `resume_allow_config_mismatch: true`
     - `resume_allow_missing_compat_metadata: true`
  3. 在 YAML 顶部注释写明“该配置用于跨方案 warm-start，不做 strict compatibility”。
- 验证：
  - 启动时确认不再出现 resume 兼容性拒绝报错。
  - 日志中应看到模型加载成功且进入 step 训练。

---

### P2（语义风险）`freeze_first_hop_modules` 实现范围过宽

- 位置：
  - `train_first_hop.py:1142-1144`
- 现象：
  - 当前逻辑是冻结“除了 `seam_refiner` 前缀外的所有可训练参数”，不只是 first-hop 模块。
- 影响：
  - 该开关名字与实际行为不一致，后续复用时容易误用。
  - 若未来引入新可训练模块，也会被意外冻结。
- 修复方案（两种择一）：
  1. **改名法（低改动）**：将配置项改为 `freeze_all_except_seam_refiner`，并同步文档；
  2. **限域法（推荐）**：仅冻结 first-hop 相关模块（`pixel_encoder`/`hop_residual_head`/`g_pix_raw` 等），保留其他模块按各自开关控制。
- 验证：
  - 启动日志打印 trainable 参数统计，确认与预期一致。
  - 检查 `optimizer.param_groups` 中参数数量变化合理。

---

### P3（可观测性偏差）N1 日志描述与实际行为不一致

- 位置：
  - `model_first_hop.py:327` 日志写“gate forced to 0”
  - 实际 `gate_pix_value()` 仍按 `softplus(g_pix_raw)+floor` 计算（`model_first_hop.py:436-438`）
  - 真正禁用靠 `_apply_hop0_pixel_forcing` 早返回（`model_first_hop.py:449-450`）
- 影响：
  - 监控和排障时可能误判 gate 数值语义。
- 修复方案：
  1. 调整日志为“pixel forcing path bypassed; gate metric kept for monitoring”。
  2. 在训练日志里增加 `pixel_forcing_disabled` 标记，避免后续分析歧义。
- 验证：
  - N1 训练日志语义清晰，`pix_delta_abs_hop0` 接近 0 的行为可解释。

---

## 对实验矩阵更新本身的结论

- `exp_result/0421/Local/README.md` 的 N1/N2 目标、gate、命令链条总体清晰。
- 主要阻断不是设计层面，而是实现细节（P0/P1）会影响可运行性与可复现性。

---

## 建议的修复执行顺序

1. 先修 `P0`（保证训练脚本可启动）。
2. 立刻修 `P1`（确保 N2 resume 策略与实验意图一致）。
3. 再修 `P2`（语义一致性，防止后续误用）。
4. 最后修 `P3`（提升日志可解释性）。

---

## 最小验收清单（修复后）

1. `python -m py_compile train_first_hop.py eval_first_hop_224_clip3.py pet_lr/model_first_hop.py`
2. N2 用目标配置 + `--resume` 跑到至少 `step=50`，无 resume 兼容报错。
3. N1 跑到至少 `step=50`，确认 `pixel_forcing_disabled` 行为和日志一致。
4. 各 run 目录写入 `config.yaml` 与 `metrics.jsonl` 正常。

