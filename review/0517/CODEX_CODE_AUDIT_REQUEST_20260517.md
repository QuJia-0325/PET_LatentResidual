# 给 Codex 的代码审核请求清单

- date: 2026-05-17
- 触发: Round 6 agent2 read 训练器 ~1500 行后识别多处与 V18-clean / V18b 直接相关的代码 hard blocker
- 目的: 在草稿 [NEXT_STAGE_EXPERIMENT_DESIGN_20260517.md](./NEXT_STAGE_EXPERIMENT_DESIGN_20260517.md) push 之前, 通过 codex 提供完整代码片段验证以下问题
- 风险等级: HIGH (任何一条没核对清楚, V18-clean launch 时会出 silent bug 或直接 raise)

---

## 1. resume CLI vs yaml 字段

### 1.1 现状 (我们目前的理解)

`train_first_hop.py:1381` 显示:
```python
resume_path = str(args.resume).strip()
```

`grep` 结果显示**只有 1 处** `resume_from`, 而该 1 处在 `review/0517/V18_decoder_lora/V18_decoder_lora.yaml` 里, **不在 train_first_hop.py**。

### 1.2 请 codex 确认

- **Q1.1**: train_first_hop.py 是否真的**完全不读** `cfg["training"]["resume_from"]` 字段? 请贴出 args 解析的完整片段 (`argparse` setup) + 所有 `args.resume` 出现位置 (含 `if args.resume` 的判定逻辑)
- **Q1.2**: V18 当时是怎么 launch 的? launch command 里是否有 `--resume <V7 best.pt>`? 如果是, V18_decoder_lora.yaml 里的 `training.resume_from` 字段是**死字段** (无效配置), 还是被某个 wrapper script 读取后翻译成 `--resume`?
- **Q1.3**: 是否有 launch wrapper (例如 `scripts/launch_first_hop.sh` 或类似) 读 yaml 的 resume_from 并构造 `--resume` 参数? 若有, 请贴出该 script

### 1.3 为什么重要

V18-clean (从 V7 出发) launch command 必须显式 `--resume <V7 best.pt>`。如果 wrapper 自动从 yaml 读 → V18-clean yaml 写 V7 路径即可; 如果没有 wrapper → 启动命令必须人工指定。这影响 §4.3 的 codex 执行指令。

---

## 2. LoRA 检查点 resume 路径

### 2.1 现状

`train_first_hop.py:1843, 1862, 1895` 显示存在:
- `_remap_state_dict_for_decoder_lora(...)` — 把 base Linear weights remap 成 LoRA wrapper 形式
- `_is_lora_key(k)` — 判断 key 是否是 LoRA-specific
- `_assert_v18_step0_equivalence(model, device, cfg)` — V18 step 0 必须 output-equivalent to frozen decoder

这套显然是为 **V7 (非-LoRA) → V18 (LoRA)** warm-start 设计的。

### 2.2 请 codex 确认

- **Q2.1**: 请贴出 `_remap_state_dict_for_decoder_lora` 完整实现 + 调用点的 if-condition (`train_first_hop.py:1843` 上下 30 行)
- **Q2.2**: 请贴出 `_assert_v18_step0_equivalence` 完整实现 (`train_first_hop.py:130-` 起始约 80 行)。它检查什么? V18 step 0 必须等价于 frozen decoder 才能通过吗?
- **Q2.3**: 若我们从 V18 自己的某个 step (例如 V18 step_120000.pt) resume, `_remap_state_dict_for_decoder_lora` 会判定 "已经是 LoRA, 不 remap" 然后跳过吗? 还是会 raise? 请贴出 if-branch 逻辑
- **Q2.4**: `_assert_v18_step0_equivalence` 在 LoRA→LoRA resume 时是否仍被调用? 如果是, 它会因为 LoRA weights 非零 (V18 已训过) 而 raise 吗?
- **Q2.5**: 是否有任何 "LoRA→LoRA resume" 的 test case / smoke run 历史? grep 一下 `decoder_lora_remapped` 在 log 里的状态

### 2.3 为什么重要

agent2 推断 "LoRA→LoRA resume 会被 step0_equivalence assertion raise"。如果这个推断正确, V18-clean 设计 (从 V7 出发) 是**唯一可行路径**; 任何 "从 V18 ckpt 续" 的实验都需要先打代码补丁。需要 codex 确认推断或贴出反例。

---

## 3. resume_allow_config_mismatch 字段冲突

### 3.1 现状

`train_first_hop.py:1399`:
```python
allow_cfg_mismatch = bool(train_cfg_boot.get("resume_allow_config_mismatch", False))
```

agent1 在 V18_decoder_lora.yaml 里发现这个字段被设置**两次** (last-wins = `false`):
- 第 162 行附近: `resume_allow_config_mismatch: true`
- 第 189 行附近: `resume_allow_config_mismatch: false`

### 3.2 请 codex 确认

- **Q3.1**: 请贴出 V18 launch 时实际生效的 `resume_allow_config_mismatch` 值 (从 wandb 启动日志或 stdout 抓 effective config)
- **Q3.2**: 触发 config mismatch 的字段集合 = ? 请贴出 `train_first_hop.py:1395-1420` 附近完整逻辑, 看 "什么算 mismatch"。例如改 use_pred_latent 是否算? 改 lambda_kl 是否算?

### 3.3 为什么重要

V18-clean 改 `use_pred_latent` 字段 + `warmup_steps` (相对 V18 baseline)。若 V18-clean 从 V7 出发 (不是从 V18 ckpt), 这个 mismatch 检查可能根本不触发 (因为对照的是 V7 train config 而不是 V18 train config)。需要确认。

---

## 4. KL pullback loss 实现细节

### 4.1 现状

`train_first_hop.py:2220-2237` (agent2 已贴):
```python
z_kl = main_out["z_pred"] if use_pred_latent else main_batch["z_dst"]
loss_kl = compute_kl_pullback_loss(rae_lora, rae_frozen, z_gt=z_kl, ...)
```

`pet_lr/decoder_lora.py:225-260` 显示 `compute_kl_pullback_loss = MSE(decode_lora(z_kl), decode_frozen(z_kl))`.

### 4.2 请 codex 确认

- **Q4.1**: 请贴出 `compute_kl_pullback_loss` 完整实现 (`pet_lr/decoder_lora.py:225-260`)。是否真的就是 MSE? 是否有 detach / stop_gradient 操作?
- **Q4.2**: `decode_frozen(z_kl)` 是否做了 `torch.no_grad()` 包装? 如果没有, frozen path 也会触发梯度 (尽管 frozen params requires_grad=False, 但 activations 仍可能 retain graph)
- **Q4.3**: 请贴出 V18 训练过程中 `loss_kl` 与 `loss_image_aux` 在 backward 之前是怎么合并的 (`train_first_hop.py:2240-2260`)。是 `total = loss_main + lambda_kl * loss_kl + lambda_img * loss_img_aux + ...` 这种形式吗?

### 4.3 为什么重要

V18-clean 假设 `use_pred_latent=false` 切到 `z_kl = main_batch["z_dst"]`。若 `z_dst` 路径的 KL 实现有任何 bug (例如 grad detach 缺失, 导致 KL 反向触发 latent transformer 参数), V18-clean 就不是 clean B9 ablation。

---

## 5. image_aux loss path (Round 6 agent2 提到的 hop0 vs main batch 区分)

### 5.1 现状

agent2 报告: image_aux 用的是 hop0 auxiliary batch 经 `predict_latent_step → decode_crop(out["z_pred"]) → x_dst`, **不是** main batch。

### 5.2 请 codex 确认

- **Q5.1**: 请贴出 `compute_hop0_image_losses` (`train_first_hop.py:760-890` 范围内) 完整实现
- **Q5.2**: 请贴出 V18 训练 step 里 image_aux 是怎么调用的 (`train_first_hop.py:880-1160` 范围内 `loss_img` 相关行)
- **Q5.3**: image_aux 的 batch 与 main batch 是同时 forward 的吗? 还是两次独立 forward? 这影响计算图共享情况
- **Q5.4**: image_aux 的 `decode_crop` 走的是 `rae_lora` 还是 `rae_frozen`? (这决定 image_aux 与 KL 是否真的作用在同一套 LoRA 参数上)

### 5.3 为什么重要

B9 (KL vs image_aux 梯度对抗) 的精确机制需要看 image_aux 真实路径。若 image_aux 走 `rae_lora.decode(z_pred_hop0)` 而 KL 走 `rae_lora.decode(z_pred_main)`, 两者参数共享 (rae_lora) 但 input 不同。梯度对抗的程度取决于:
- z_pred_main vs z_pred_hop0 的相关性
- 两个 batch 的样本重叠度

这个机制细节决定 V18-clean 预期 ΔPSNR 的量级 (§4.3 的 +0.05 dB threshold)。

---

## 6. LR schedule 与 checkpoint 间隔

### 6.1 请 codex 确认

- **Q6.1**: 请贴出 V18 实际的 `lr_schedule.total_steps_override` 生效值 + cosine schedule 在 step 200K 时的 lr 值 (从 wandb 抓或 inline print)。验证 agent1 推算 "decoder lr ≈ 1.25e-8 at step 200K"
- **Q6.2**: V18 checkpoint 保存间隔 = ? grep `save_interval|checkpoint_interval` 的实际 yaml 值; 列出 V18 当前已有 ckpt files (例如 `ls -la run/checkpoints/`)
- **Q6.3**: best.pt 选择规则 = ? 是 val_chain_normal_psnr_clip3 最高吗? 还是 val_loss 最低?

### 6.2 为什么重要

V18-clean 从 V7 best.pt 出发, 需要确认 "V7 best.pt" 选择 criterion 与 V18-clean 评估 criterion 一致 (否则比较不公平)。

---

## 7. 训练器是否有 LoRA→LoRA resume 测试

### 7.1 请 codex 确认

- **Q7.1**: `tests/` 目录下是否有 LoRA→LoRA resume 的 unit/integration test? `grep -rn "decoder_lora.*resume\|lora.*resume" tests/`
- **Q7.2**: 是否有人 (历史 commit) 跑过 "从 V18 ckpt 续训" 类型的实验? 看 git log 找 V18b / V19 / resume_from_v18 之类的 yaml

### 7.2 为什么重要

如果有历史 LoRA→LoRA resume 经验, 我们就不必猜测 step0_equivalence 是否 raise; 可以直接看历史 log。

---

## 8. 输出格式 (codex 回复时请按以下结构)

```markdown
# Response to Code Audit Request 20260517

## §1 resume CLI vs yaml
- Q1.1: <代码片段 + 结论>
- Q1.2: <V18 launch command>
- Q1.3: <wrapper script 是/否>

## §2 LoRA resume path
- Q2.1: <_remap_state_dict_for_decoder_lora 完整代码>
- Q2.2: <_assert_v18_step0_equivalence 完整代码>
- Q2.3: <LoRA→LoRA 分支判定>
- Q2.4: <step0_equivalence 在 LoRA→LoRA 时行为>
- Q2.5: <历史 log 状态>

## §3 config mismatch
- Q3.1: <V18 effective 值>
- Q3.2: <mismatch 字段集合>

## §4 KL pullback
- Q4.1: <compute_kl_pullback_loss 完整代码>
- Q4.2: <decode_frozen no_grad 状态>
- Q4.3: <loss 合并代码>

## §5 image_aux path
- Q5.1: <compute_hop0_image_losses 完整代码>
- Q5.2: <V18 训练 step 里 image_aux 调用>
- Q5.3: <batch 关系>
- Q5.4: <rae_lora vs rae_frozen>

## §6 LR + ckpt
- Q6.1: <step 200K 实际 lr>
- Q6.2: <ckpt 列表 + 间隔>
- Q6.3: <best.pt criterion>

## §7 历史
- Q7.1: <test 是否存在>
- Q7.2: <历史 LoRA→LoRA resume 实验>

## §8 codex 自己发现的 caveat
<任何代码层 caveat 我们没问到的>
```

---

## 9. 优先级 & 阻塞关系

| 问题 | 优先级 | 阻塞什么 |
|---|---|---|
| Q1 (resume CLI) | HIGH | V18-clean launch command 写法 |
| Q2 (LoRA resume) | HIGH | 确认 V18-clean from V7 是唯一可行路径 |
| Q4 (KL pullback) | HIGH | V18-clean 是不是 clean B9 ablation |
| Q5 (image_aux) | MEDIUM | B9 精确机制 + V18-clean 预期 ΔPSNR |
| Q3 (config mismatch) | MEDIUM | V18-clean launch 不被 raise |
| Q6 (LR/ckpt) | LOW | V7 best.pt 选择 criterion |
| Q7 (历史) | LOW | 加速确认 |

**Q1+Q2+Q4 任何一条卡住, NEXT_STAGE_EXPERIMENT_DESIGN §4.3 (V18-clean) 都需要重写或推迟**。

---

## 10. claude 的不确定性自陈

我目前对 train_first_hop.py 的认识基于:
- agent2 Round 6 read (我相信 agent2 但没独立 verify 每一行)
- 我自己刚才 spot-check 的 grep 结果 (验证了 `args.resume` 唯一性 + `_remap_state_dict_for_decoder_lora` / `_assert_v18_step0_equivalence` 存在)
- V18_decoder_lora.yaml 字段 read

我**没有**独立 verify:
- agent2 报告的 "image_aux 用 hop0 batch" (Q5)
- agent1 报告的 "V18 yaml 两次 resume_allow_config_mismatch" (Q3)
- "step 200K 时 lr = 1.25e-8" 推算 (Q6.1)

所以这份审核请求**主要目的是 second-source verify**, 不是 codex 重新发现已知问题。
