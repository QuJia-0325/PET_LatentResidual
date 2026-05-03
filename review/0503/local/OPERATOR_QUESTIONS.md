# 0503 Local → Operator Questions

**From**: Local agent (Mac, no GPU access)
**To**: Remote operator (`/home/qujiaxiang/project/PET_LatentResidual` on 启智 / 远程 GPU 主机)
**Reply protocol**: 在 `review/0503/operator/` 下新建一个 `OPERATOR_REPLY_<topic>.md` 回答；或者直接在本文件每个问题后追加 `**A**:` 块。
**Why**: σ-norm ablation 协议已在 commit `7100839` 锁定，但 §6.4–§6.6 protocol 真正能否端到端走通，依赖一些只有远端能确认的事实。先把这些不确定性收回来，再决定要不要补脚本 / 改文档。

---

## Q1 (A3 / 紧迫)：当前真正在 GPU 上跑的 ablation 分支有几支？

§6.4 的 `paired_diff_judge.py` 是为 **Risk 4 paired-diff confound check** 写的——前提是 A_main 之外还有 `A_pair_uniform_spot`（一支只有 step-weight uniform、其它都同 A_main 的 spot run），用来排除 "pair sampling × step-weight 的耦合 confound"。

**问题**：

1. 现在远端 GPU 上同时在跑的 first-hop 224 σ-norm 相关 run，到底是哪几个？我猜是其中一种：
   - **2 支**：`A_main`（V6 transport-first）+ `C_uniform`（B_sanity 上加 normalizer ON）
   - **3 支**：上面 2 支 + `V6.1_rollout_floor`
   - **4 支**：上面 3 支 + `A_pair_uniform_spot`
2. 如果 **没有 A_pair_uniform_spot**：`paired_diff_judge.py` 就没有第二条 baseline 可配对；§6.4 的 Risk 4 检查事实上跑不起来。请确认这是不是有意如此（即 Risk 4 已被默认无 confound）；如果应该补这一支，请告诉我预算（GPU·days）和优先级。
3. 每一支当前最新的 `global_step` / 预期终点 step 是多少？

> **为什么需要现在问**：A_main 一旦跨过 60K，§6.6 lock-in 就会被立即触发；如果 A_pair_uniform_spot 缺位，我们要么补跑、要么显式在 protocol 里写 "Risk 4 not checked, justified by ..."。

---

## Q2 (A2 / 紧迫)：A_main 是否一定能跑到 step 60000？

`lock_effect_size_threshold.py` 的 LOCKED 窗口是 `[40000, 60000]`，N≥5 才放行（exit 1 if N<5）。

**问题**：

1. A_main config 上 `training.total_steps` 是多少？预计实际 wall-clock 何时跨过 60000？
2. 是否存在 OOM / 数据盘满 / preempt 风险，使得 A_main 可能在 < 60000 处终止？
3. 若实际只能跑到 ~50000：你倾向于
   - (a) 等更长（推迟 ablation）
   - (b) 把窗口收缩到 `[30000, 50000]`，并在 EFFECT_SIZE_LOCKED.md 里附 deviation note
   - (c) 把窗口收缩到 `[terminal-20000, terminal]`，写成新的 fallback policy 加进 §6.6.2

**操作意义**：如果 (c) 是答案，我会把这条 fallback 写进 `POST_V6_NEXT_STEPS.md` §6.6.2 的"Step 4 deviation policy"小节；这是一次性 protocol 修补，必须在 X 锁定之前完成才合法。

---

## Q3 (A1 / 中等紧迫)：要不要补一个 "post-lock C_uniform full-val 闸门"？

现状：`lock_effect_size_threshold.py` 用 exit 7 阻止"在 X 锁定前 C 已被 eval"（顺序违例方向 1）。但**反方向**——"X 没锁定就跑了 C full-val"——目前没有任何脚本守门，全靠人类记得 §8.7 在 §8.6 之前。

**问题**：你想不想我加一个 wrapper：

```bash
review/0502/scripts/run_c_uniform_full_val.sh \
    --c-config review/0502/configs/C_uniform.yaml \
    --c-ckpt-dir /data_2/.../C_uniform/run-.../
```

行为：

1. 检查 `review/0502/EFFECT_SIZE_LOCKED.md` 是否存在 → 否则 exit 1 拒跑。
2. 检查这个 lock 文件的 git commit 是否在远端 `gitee/foc_lite_hop0` 上 → 否则 exit 2（防止本地未 push 的 lock）。
3. 透传所有参数到 `eval_first_hop_224_clip3.py --max-slices 0`。

成本：~1 小时。**如果你确认远端只会通过 launch 脚本 / shell 触发 full-val，那这个 wrapper 就有价值；如果你已经习惯交互式跑 eval 命令，可能强制 wrapper 反而碍事。** 请告诉我哪种情况。

---

## Q4 (D1 / 中等)：dry-run lock 脚本在真实 metrics.jsonl 上

我目前只在合成数据上 smoke-test 过 `lock_effect_size_threshold.py`。脚本对 `val_select_score` key 的存在有硬依赖。

**问题**：A_main 当前的 `metrics.jsonl` 一行的 schema 长什么样？三种可能：

```jsonc
// 可能 1：扁平（脚本默认假设）
{"global_step": 12345, "val_select_score": 7e-4, "loss": ...}

// 可能 2：嵌套
{"global_step": 12345, "val": {"select_score": 7e-4, ...}, ...}

// 可能 3：分行
{"global_step": 12345, "metric_name": "val_select_score", "value": 7e-4}
```

**请你**：把 A_main run 当前 `metrics.jsonl` 的**最后 5 行**贴到 `review/0503/operator/A_main_metrics_tail5.jsonl`（不要带 ckpt 路径或绝对地址；纯 jsonl）。我看一眼就能确认 schema，必要时给脚本加一个 `--metric-jsonpath` 选项。如果 schema 不是可能 1，A_main 跑到 60K 时 lock 脚本会直接 N=0 失败。

---

## Q5 (C2 / 中等)：decoder ceiling 数字到底有没有？

`eval_gt_latent_decoder_ceiling_clip3.py` 在 `scripts/` 下存在，§6.6.5 把它当 auxiliary anchor。

**问题**：

1. 这个脚本之前**实际跑过吗**？如果跑过，结果文件在哪？（路径下 `outputs/PET_LatentResidual/.../decoder_ceiling*.json` 之类）
2. 如果没跑过：要不要现在就跑一次（不阻塞 A_main / C_uniform，只占 1 GPU 约 30 min）？这个数字对 paper 写作的 anchor 价值很高，缺它 reviewer 会问 "你的 chain MSE 离上界多远？"
3. 如果跑过但数字记忆模糊：把当时的 stdout / json 路径告诉我，我可以从 git log 找。

---

## Q6 (B2 / 低)：`train_v21.py` 是否应该移到 `deprecated/`？

刚刚润色的 README 把它标成 "kept for reproduction only"。

**问题**：你打算还会再跑 v2.1 吗？

- "近 1 个月不会跑" → 我把 `train_v21.py` 移到 `deprecated/v21/train_v21.py`，README 同步改路径
- "可能要跑" → 留在根目录不动

代码本身不删，只是位置。

---

## Q7 (B3 / 低)：`EFFECT_SIZE_LOCKED.md` 要不要记环境信息？

当前模板只记 metrics SHA256 + git HEAD。但 reviewer 可能问 "A100 上 paired CV 和 V100 上 paired CV 不可比"。

**问题**：要不要我给 `lock_effect_size_threshold.py` 加一个静默的环境捕获段（`torch.__version__` / `torch.version.cuda` / `torch.cuda.get_device_name(0)` / `nvidia-smi --query-gpu=driver_version`），写进 markdown？

成本：~10 行代码。如果 A_main 已经接近 60K，这个改动需要在 lock 触发前合并，否则就来不及了——但如果你判断 reviewer 不会卡这个点，就直接 skip。

---

## 优先级总结（站在我的视角）

| Q | 紧迫度 | 阻塞什么 |
|---|---|---|
| Q1 (A3) | 🔴 高 | 决定 paired_diff_judge.py 是不是死代码 |
| Q2 (A2) | 🔴 高 | 决定 §6.6.2 是否需要补 fallback policy |
| Q4 (D1) | 🟡 中 | 决定 lock 脚本是否需要加 schema 适配 |
| Q5 (C2) | 🟡 中 | 决定 paper 是否有 ceiling anchor |
| Q3 (A1) | 🟢 低 | 是否补反向闸门 wrapper |
| Q6 (B2) | 🟢 低 | 仓库整洁度 |
| Q7 (B3) | 🟢 低 | reviewer-defense 余量 |

**最优 reply 顺序**：先回 Q1+Q2+Q4（都是 1–2 句话能定的），其它可以拖到 A_main 跑完再答。

---

## 元信息

- 本文件创建：`2026-05-03`
- 关联 commit：`7100839`（lock script + protocol 文档）
- 协议源：`review/0502/POST_V6_NEXT_STEPS.md` §6.4 / §6.6
- 上一份 operator 文档：`review/0503/operator/overall_design_experiment_matrix_deep_review_20260503.md`
