# Plan F — V7 + 消融 arm 双卡并行启动设计

**Date**: 2026-05-06 CST
**Status**: DESIGN-LOCKED, awaits user GPU-launch authorization
**Companion docs**:
- 上游决策: [PLAN_F_V7_V8_FROM_SCRATCH_150K_20260505.md](PLAN_F_V7_V8_FROM_SCRATCH_150K_20260505.md)
- 实验设计: [../0505/local/ANALYSIS.md](../0505/local/ANALYSIS.md)
- launcher: [../0505/local/scripts/run_v7_v8_sanity.sh](../0505/local/scripts/run_v7_v8_sanity.sh)
- σ-normalize 失败证据: [../0505/operator/OPERATOR_REPLY_run_ablation_sanity_failure_20260505.md](../0505/operator/OPERATOR_REPLY_run_ablation_sanity_failure_20260505.md)

---

## 1. 背景与触发原因

Plan F 当前包含 3 个独立 arm（[Plan F §3](PLAN_F_V7_V8_FROM_SCRATCH_150K_20260505.md)）:

| Tag | 别名 | 设计 | 用途 |
|---|---|---|---|
| **V7** | Grönwall 闭式解 arm | step_weights=[0.6106, 2.0, 2.7041, 2.0423], image_aux on (V6 同) | 测 raw-rollout step_weights SHAPE 假说 |
| **V8** | **消融 arm**（更准确叫法）| V6 step_weights, image_aux **disabled** | 测 image_aux 对 hop0 / 整链贡献的下界 |
| **V6_NOISE** | seed=1337 二号 V6 | V6-seed42 的种子孪生（仅 RNG 不同）| 计算 d_pure noise floor |

每个 arm 独占一张 GPU 跑 160K steps from-scratch。原 launcher 只支持 sequential（`both` / `all` 案例分支顺序运行）。本文档归档 **`parallel` 模式** 的设计——允许 V7 和消融 arm **同时**在两张卡上独立运行，缩短 wall-clock 时间约 50%。

> **命名约定**：用户偏好 "消融 arm" / "no-image-aux 消融" 而非 "V8"。文件路径与 yaml run_name 暂保留 `V8` 字符串以避免重跑 integrity check + 路径变量冲突；对话与文档显示层使用 "消融 arm"。完整改名（yaml + run_name + output_dir + launcher case）仍待批准。

---

## 2. GPU 拓扑与默认分配

| GPU | Arm | yaml | 输出根 |
|---|---|---|---|
| `cuda:2` | V7 (Grönwall 闭式) | `review/0505/local/configs/V7_gronwall_raw.yaml` | `${ABLATION_OUTPUT_ROOT}/V7/run/` |
| `cuda:3` | 消融 (no image_aux) | `review/0505/local/configs/V8_no_image_aux.yaml` | `${ABLATION_OUTPUT_ROOT}/V8/run/` |

环境变量覆盖：
- `GPU_V7=2`（默认）
- `GPU_V8=3`（默认；命名仍沿用 V8 与代码 tag 一致）

V6_NOISE arm **不**进入 `parallel` 模式——它必须在与 V6-seed42 严格同卡同环境下跑（控制 GPU 漂移），单独走 `V6_NOISE` 子命令。

---

## 3. 并发安全性论证

### 3.1 输出路径不冲突

V7 和消融 arm 写入完全不同的目录树：

```text
/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/
├── V7/run/first_hop_224_v7_gronwall_raw/   ← V7 写入
└── V8/run/first_hop_224_v8_no_image_aux/   ← 消融 arm 写入
```

`resolve_yaml()` 在 launcher 里把每个 arm 的 `output_dir` 重写到自己的 `${ABLATION_OUTPUT_ROOT}/${tag}/run`。两个 tag 不同，不可能 collide。

### 3.2 In-repo index 也不冲突

```text
review/0505/local/runs/
├── V7/{config.resolved.yaml, train.log}
└── V8/{config.resolved.yaml, train.log}
```

每个 tag 独立子目录。

### 3.3 freshness gate 串行化

`require_fresh_output_dir: true` 在两个 yaml 都开着。launcher 在 launch 任何 python 之前**串行**预检：

```text
prep_one V7  →  resolve yaml + 检查 V7 输出是否存在
prep_one V8  →  resolve yaml + 检查 V8 输出是否存在
（任一失败则 exit 4，两个 python 都不启动）
train_one V7 &        ← 然后才并行启动
train_one V8 &
wait
```

→ 不会出现 "V7 启动了但 V8 因为 freshness 失败导致 V7 半训练状态" 的问题。

### 3.4 RNG / 数据加载并发安全性

两个 python 进程是完全独立的 PID，各自调用 `set_seed(42)`、各自构造 DataLoader、各自占用一张物理 GPU（通过 `CUDA_VISIBLE_DEVICES`）。共享的只读资源：
- 训练集 (`/data_2/.../train_*.pt`)：read-only mmap，并发读没有竞争
- DINOv2 backbone 权重：read-only checkpoint，并发读无竞争

写入资源各自独占。

### 3.5 CUBLAS / cuDNN 状态

`CUBLAS_WORKSPACE_CONFIG=:4096:8` 在 launcher 顶部 export，被两个子进程继承。每个进程拥有自己 GPU 的独立 cuBLAS handle，不共享 workspace。

---

## 4. launcher 实现

### 4.1 关键函数拆分

旧版 `run_one()` 单体函数被拆成两个原子操作：

```text
prep_one(tag, src, steps)             ← 同步：resolve yaml + freshness check
train_one(tag, src, steps, gpu)       ← 阻塞：CUDA_VISIBLE_DEVICES=gpu python train ...
run_one(tag, src, steps, gpu?)        ← 兼容包装（prep + train），保留旧行为
```

### 4.2 `parallel` case 分支

```bash
parallel)
    prep_one  V7   .../V7_gronwall_raw.yaml   ${SANITY_STEPS}
    prep_one  V8   .../V8_no_image_aux.yaml   ${SANITY_STEPS}
    set +e
    train_one V7   .../V7_gronwall_raw.yaml   ${SANITY_STEPS}  ${GPU_V7} &
    pid_v7=$!
    train_one V8   .../V8_no_image_aux.yaml   ${SANITY_STEPS}  ${GPU_V8} &
    pid_v8=$!
    wait ${pid_v7}; rc_v7=$?
    wait ${pid_v8}; rc_v8=$?
    set -e
    (( rc_v7 != 0 || rc_v8 != 0 )) && exit 1
    ;;
```

**为什么 `set +e`/`set -e` 包夹**：
- `set -euo pipefail` 默认对失败 PID 立即 abort
- 我们要的是"两个都失败也要分别报告 rc"——必须临时关 errexit
- `wait $pid` 在 errexit 关闭时返回该 PID 的 rc
- 完成后 restore errexit 并显式 exit on failure

### 4.3 用户启动命令

```bash
# 默认 GPU 2/3
SANITY_FRESH=1 SANITY_STEPS=160000 \
  bash review/0505/local/scripts/run_v7_v8_sanity.sh parallel

# 自定义 GPU (e.g. 6/7)
GPU_V7=6 GPU_V8=7 SANITY_FRESH=1 SANITY_STEPS=160000 \
  bash review/0505/local/scripts/run_v7_v8_sanity.sh parallel
```

### 4.4 监控

启动后 launcher 打印：

```text
[parallel] launched V7 pid=12345 on GPU2, V8 pid=12346 on GPU3
[parallel] follow logs:
  tail -f review/0505/local/runs/V7/train.log
  tail -f review/0505/local/runs/V8/train.log
```

两个 train.log 是**完整 tee**（不是 nohup 截断）。每个进程都会持续 flush wandb / metrics.jsonl 到自己的 `output_dir`。

---

## 5. wall-clock 预估

| 模式 | V7 | 消融 | wall-clock |
|---|---|---|---|
| `both` (sequential) | 160K steps × ~0.65 s/step | 160K steps × ~0.65 s/step | ~2 × 28.9h ≈ 58h |
| `parallel` (双卡) | 同上，并发 | 同上，并发 | ~28.9h |

→ 节省约 29 h（剩余 V6_NOISE arm 28.9h 单独跑，或者用第三张卡再并发）。

> 单卡 step time 估计来源：V6 200K 训练耗时约 36h（用户提供），即约 0.65 s/step。参数与显存对 V7/消融 arm 完全相同，step time 应一致。

---

## 6. 与现有协议的兼容性

| 协议项 | 是否影响 |
|---|---|
| Plan F §3 决策矩阵 | 不变 |
| Plan F §6 d_thr 阈值 | 不变 |
| ANALYSIS §5 Stage 0 V6_NOISE 启动 | 不变（V6_NOISE 仍走 `V6_NOISE` 子命令） |
| Round 1/2/3 peer-review 设计锁定 | 不变（仅改启动方式，不改任何 hyperparam） |
| 48/42 integrity check | 不影响（只增加 launcher case 分支） |

---

## 7. Open question 记录到外部 peer review（Round 4）

并发执行本身没有未决问题，但**触发本轮工作的根本原因是 σ-normalize ablation 失败**（见 [OPERATOR_REPLY_run_ablation_sanity_failure_20260505.md](../0505/operator/OPERATOR_REPLY_run_ablation_sanity_failure_20260505.md)）：

> 2026-05-03 official `run_ablation.sh sanity` failed with `val_pair_total` 10.16% drift, `val_chain_normal_mse` 12.73% drift. 同种子 / 同 deterministic 设置（warn-only）下两条 20K 训练轨迹无法收敛到同一权重。Memory Efficient attention + `adaptive_avg_pool2d_backward_cuda` 的非确定性 kernel 在 PyTorch warn-only mode 下不被强制阻断。Sentinel 已被清除，A_main / C / D 拒绝启动。

V7 选择走 **raw-rollout 路径**（σ-normalize OFF）正是为了绕开这条断头路。但这引出三个未决问题需要外部 reviewer 审计：

**Q1**：σ-normalize 路径为什么会失败？是数学等价性 bug，还是 CUDA kernel 非确定性导致的 trajectory amplification？

**Q2**：raw-rollout 路径（V7 当前走的方向）是否能避开 Q1 的根本问题，还是只是把同一问题埋得更深？

**Q3**：如果 raw-rollout 路径在 V7 训练完成后仍出现类似的"同种子不复现"现象，应该如何处理？

→ 这三个问题的 peer-review prompt 在 [PEER_REVIEW_PROMPT_SIGMA_VS_RAW_20260506.md](../0505/local/PEER_REVIEW_PROMPT_SIGMA_VS_RAW_20260506.md)（Round 4）。

---

## 8. Audit trail

| 日期 | 事件 |
|---|---|
| 2026-05-05 | Plan F V3 锁定（V6-seed1337 noise-baseline arm 加入，3 arm 总 480K GPU） |
| 2026-05-05 | 48/42 integrity check PASS（V6_seed1337.yaml + adaptive d_thr） |
| 2026-05-06 | 用户选 A（保留 V7 hop0=0.6106，事后用消融 arm 反推效应）|
| 2026-05-06 | 用户偏好 "消融 arm" 替代 "V8"（对话层调整，文件名待定） |
| 2026-05-06 | **本文档**：launcher `parallel` 模式归档（GPU 2/3 双卡并发） |
| 2026-05-06 | Round 4 peer-review prompt 起草（σ-normalize vs raw-rollout） |

---

## 9. 文件清单

| 文件 | 状态 | 修改 |
|---|---|---|
| `review/0505/local/scripts/run_v7_v8_sanity.sh` | UPDATED | 新增 `prep_one` / `train_one` / `parallel` case；run_one 加 4-th gpu 参数 |
| `review/plan/PLAN_F_PARALLEL_LAUNCH_DESIGN_20260506.md` | NEW | 本文档 |
| `review/0505/local/PEER_REVIEW_PROMPT_SIGMA_VS_RAW_20260506.md` | NEW | Round 4 peer-review prompt |
| `review/plan/PLAN_F_V7_V8_FROM_SCRATCH_150K_20260505.md` | UNCHANGED | 上游决策文档不变 |
| `review/0505/local/ANALYSIS.md` | UNCHANGED | 实验决策矩阵不变 |
| `review/0505/local/configs/V*.yaml` | UNCHANGED | 任何 hyperparam 都没动 |

