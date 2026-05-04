# OPERATOR CONFIRMATION REQUEST — C2.2 + C2.3 已 push 到 `gitee/foc_lite_hop0`

**Author**: Mac-side agent (Claude Opus)
**Audience**: 远程 operator (via codex CLI)
**Date**: 2026-05-04
**Branch**: `foc_lite_hop0`
**Remote HEAD**: `de4cc88` (was `8a0757a` before this push)
**LOCKED_PROTOCOL_VERSION**: `v0_pending_R1a` (跨 push 不变)
**Action expected from operator**: pull → 跑测试 → 在 `review/0503/operator/OPERATOR_REPLY_C2_3_<date>.md` 中逐条回答 §3 中的 6 个问题

---

## 0. TL;DR

`gitee/foc_lite_hop0` 现在比上次（C2.1 / `8a0757a`）多了 2 个 commit：

```
de4cc88  C2.3  Round-5 peer-review absorption — D1 (sanitize git stderr) + E1 (strip URL userinfo)
c01aa27  C2.2  absorb mid-impl peer review MEDIUM polish (A2/A3/B2a/B3b/B4/B10)
8a0757a  C2.1  ← gitee 上次的 HEAD（你之前看到的）
```

**改动只动 `review/0502/scripts/` 下的 lock 工具 + `review/0503/local/` 下的审计记录文档。**
**训练 pipeline 不受影响**（lock 脚本不在训练侧的 import 链上）。

需要 operator 做的事情：
1. **拉取** — §1
2. **跑测试** — §2（期望 97 tests OK，含 Round-8 absorption: G1 + G2 + G3 + G4 + G5）
3. **回答 6 个问题** — §3（用于让 Mac 侧确认远端环境与代码兼容）
4. **决定 C2.4-tests 时机** — §4（可选，不阻塞）

---

## 1. 拉取指令

```bash
cd <your repo path>
git fetch gitee
git checkout foc_lite_hop0
git pull --ff-only gitee foc_lite_hop0
git rev-parse HEAD                                    # 拉到最新 Mac 推送的 commit 即可
git log --oneline 8a0757a..HEAD                       # 含 C2.3 + C3/F2/F3/F4 absorption 的若干 commit
```

> 注：本文档原版（C2.3 时期）写了 `de4cc881` 作为期望 HEAD，那是 C2.3
> 那一次推送的 commit。Round-6 absorption（C3 + F2 + F3 + F4）之后会
> 有新的 commit，HEAD 会前进。只要 `--ff-only` 成功并且 §2 的测试
> 跑通就 OK。

如果 `git pull --ff-only` 报错（即出现非 fast-forward），**不要强制合并**；直接在回复中说明本机 HEAD 与冲突信息，Mac 侧再分析。

---

## 2. 测试验证（必做）

> **F4 fix (Round-6 absorption, May 4 2026)**：必须使用带 PyYAML 的解释器。
> Round-5 反馈中 operator 报告 T16 在 host `python3` 上 1/69 失败，原因是
> 默认 `python3` 没有 PyYAML，而 `lock_effect_size_threshold.py` 在
> module-load 时就 `import yaml`。如果改用 `<conda-env>/bin/python`
> 就 69/69 OK。**切勿** 在 default `python3` 下运行此命令然后报告
> "测试失败"，那是环境缺包，不是代码 regression。

请明确使用训练用的 conda env，例如 operator 的 `rae` env：

```bash
# ✅ 正确：直接调用 conda env 的 python（不需要先 activate）
/home/<your-user>/.conda/envs/rae/bin/python -m unittest \
    review.0502.scripts.test_remote_resolver \
    review.0502.scripts.test_metrics_compat \
    review.0502.scripts.test_select_best_ckpt_smoothed
```

或先 activate 再跑：

```bash
conda activate rae           # 或你训练用的任何 env
python -m unittest \
    review.0502.scripts.test_remote_resolver \
    review.0502.scripts.test_metrics_compat \
    review.0502.scripts.test_select_best_ckpt_smoothed
```

如果你的 env 名字不是 `rae`，把路径替换为你训练用的 conda env。
快速判断：能跑 `train_first_hop.py` 的 python 就一定能跑测试
（训练入口同样依赖 PyYAML）。

**期望输出**（Round-6 + Round-8 = C3/F2/F3/F4 + G1/G2/G3/G4/G5 absorption 之后）：

```
.................................................................................................
----------------------------------------------------------------------
Ran 97 tests in <X>s

OK
```

> 数字说明：69 (C2.3 baseline) + 12 (C3 select_best e2e) + 3 (F2 train-row warning narrowing) + 3 (F3 pushURL TOCTOU) + 2 (G1 multi-pushURL) + 2 (G2 collision warn) + 6 (G3 strict best_metric) = **97**。
> F4 / G4 / G5 是文档/comment 修复，不引入测试。
> 4 个 torch-gated test (`TestListSavedStepsWithTorch::*`) 在
> 有 torch 的 env 上会跑过 (97 OK)，在无 torch 的 env 上会以 `skipped` 计入仍是 PASS
> (`Ran 97 tests ... OK (skipped=4)`)。
> **OK** 是唯一可接受的整体结论；不接受 `FAILED (errors=*)`。

如果不是 OK：
- 把完整 stderr 粘到回复
- 不要修改任何代码尝试 fix（先回 Mac 这边分析）
- 仍然继续训练（这 97 测试是 lock 工具内部的，不阻塞训练）

---

## 3. 需要 operator 回答的 6 个问题

### Q1. 训练 pipeline 是否依赖 `review/0502/scripts/` 下的代码？

C2.2 / C2.3 改动了：

- `review/0502/scripts/lock_effect_size_threshold.py`
- `review/0502/scripts/_metrics_compat.py`（C2.2 只改 docstring）
- `review/0502/scripts/test_remote_resolver.py`
- `review/0502/scripts/test_metrics_compat.py`

预期：训练侧 (`pet_lr/`、`train_first_hop.py`、`train_v21.py`、`scripts/run_ablation.sh` 等) **不**从 `review/0502/scripts/` import 任何东西。请 operator 在远端 grep 验证：

```bash
grep -r "from review" pet_lr/ scripts/ tools/ train_*.py 2>/dev/null
grep -r "import review" pet_lr/ scripts/ tools/ train_*.py 2>/dev/null
grep -r "lock_effect_size_threshold" pet_lr/ scripts/ tools/ train_*.py 2>/dev/null
grep -r "_metrics_compat" pet_lr/ scripts/ tools/ train_*.py 2>/dev/null
```

**回答格式**：
- 4 条命令各自的输出（粘贴 stdout，无输出就写 "no matches"）
- 如果有任何 match，请贴出完整行 + 文件路径

---

### Q2. `.review_canonical_remote` anchor 文件的内容

锁脚本 `lock_effect_size_threshold.py` 在 lock 阶段会读 `.review_canonical_remote` 来确认推送目标。C2.2 加严了多行检测（B3b），C2.3 修复了 userinfo URL 误判（E1）。

请 operator 在远端跑：

```bash
cat -A .review_canonical_remote
```

（`-A` 会显示所有不可见字符，包括 `\r`、BOM、行尾 `\n` 等。）

**期望形式之一**（按 operator 实际配置）：

- `git@gitee.com:jqu9/PET_LatentResidual.git$`（scp 形式 + 单行）
- `https://gitee.com/jqu9/PET_LatentResidual.git$`（https 干净形式 + 单行）
- `gitee.com/jqu9/PET_LatentResidual$`（裸 host/path 形式 + 单行）

**警告形式**（如果命中其中之一，请告知 Mac 侧）：

- 含 `M-bM-+M-?` 字样 → BOM，C2.1 起已自动剥离，但请确认是预期还是误存
- 多于 1 行非空内容 → C2.2 起会 raise `CanonicalRemoteError`，必须删除
- 含 `https://user:pass@` 或 `https://<token>@` → C2.3 已修复但建议改成 SSH 形式
- 含 `#frag` 或 `?token=...` → C2.2 起自动剥离，但请确认非误存

**回答格式**：粘贴 `cat -A` 的输出。

---

### Q3. `gitee` remote 在 operator 机器上的 fetch URL

锁脚本通过 `git remote -v` 找匹配 `.review_canonical_remote` 锚点的 remote 名。请 operator 跑：

```bash
git remote -v
```

**回答格式**：粘贴完整 stdout。

如果 operator 机器上 remote 名 ≠ `gitee`（比如叫 `origin` 或 `upstream`），需要 Mac 侧侧重新评估锚点匹配是否还成立。

---

### Q4. operator 机器上是否已配置可推送的 SSH key 到 gitee？

锁脚本在生成 `EFFECT_SIZE_LOCKED.md` 后会**自动 `git push`**到 canonical remote（除非加 `--no-push`）。请 operator 跑（**不要真推**）：

```bash
ssh -T git@gitee.com 2>&1
```

**期望**：返回 `Hi <username>! You've successfully authenticated...` 或类似消息。

**回答格式**：粘贴完整 stdout/stderr。

如果 SSH 不通：要么 operator 在 lock 时手动 push（`lock_effect_size_threshold.py --no-push` 然后 `git push gitee foc_lite_hop0`），要么先把 key 配上。

---

### Q5. operator 计划在哪台机器上跑 lock？

锁的产物 `EFFECT_SIZE_LOCKED.md` 是 σ-normalize ablation 的预注册公示证据，commit 时间戳就是预注册时间。**lock 必须发生在所有 Method-A in-window data 完成之后**。

请 operator 确认：

- (a) 训练继续在远端 GPU 工作站跑，lock 也在同一台跑（最稳）
- (b) 训练在远端跑，metrics 拉到本地 Mac 后由 Mac 侧跑 lock（需要把 metrics_a 文件传过来；不推荐，多一次网络传输 + SHA 风险）
- (c) 其他方案

**回答格式**：选择 a / b / c，如果是 c 请简述。

---

### Q6. 当前训练状态 + 距离 lock 还有多远

只为了让 Mac 侧理解时序节奏，不需要精确数字：

- 训练当前在哪个 epoch / step？
- step ∈ [step_min, step_max] 的 in-window 数据已收集了多少行（粗略估计即可）？
- 预计还要多久能进入 lock 阶段？

**回答格式**：自由文字，不超过 5 行。

---

## 4. 可选 — C2.4-tests 时机决定

Round 5 peer review 还有 4 项被延后到独立 commit（**纯测试 + docstring 加固，不改产品代码逻辑**）：

| ID | 内容 | 触发风险 |
|---|---|---|
| F1 | T16 单读不变量扩展到 Guard-fail / push-fail / commit-fail 等错误分支 | 低（一旦未来 main() 改成误读两次，T16 happy-path 测不出来） |
| F2 | Guard 5 (data-side) 优先于 Guard 4 (N<5) 的 reorder 用 end-to-end 测试穿透 | 低（一旦未来 reorder 被回退，单元测覆盖不到） |
| F3 | 引入 `unittest.mock` patch `subprocess.run`，覆盖 B5 TOCTOU + push-failure + URL-mismatch 分支 | 低（这些分支目前只在 live-repo 中触发） |
| B1' | `_load_anchor` docstring 显式说明不支持 `#` 注释语法 | 极低（用户体验改进） |

请 operator 选一个：

- **(选项 A — 推荐)** Mac 侧立即开始 C2.4-tests 编写，operator 这边继续跑训练，互不干扰；做完再 push 一个 commit 让 operator 拉
- **(选项 B)** 等 operator 验证完 C2.3（即 §2 测试通过 + §3 回复完毕）再启动 C2.4-tests
- **(选项 C)** C2.4-tests 不必单独 commit，等 R1a 一起出
- **(选项 D)** 其他

**回答格式**：A / B / C / D，如果是 D 请简述。

---

## 5. 不需要 operator 回答的事项（说明只是为了避免重复审计）

- 远端 `EFFECT_SIZE_LOCKED.md` 何时产出 / 哪台机器跑 lock — Q5 已涵盖；不需另答
- `master` / `idea2_alpha_cct_gpu3` 分支的合并时机 — 不在本次审查范围
- GitHub `origin` 与 Gitee `gitee` 的镜像策略 — 不在本次审查范围
- 任何统计 / 协议层修改（`paired_CV` 数学、决策三分法、σ-normalize 流程）— C2.2 / C2.3 都未触及，不需 operator 重审

---

## 6. 审计 trail 自助查阅

如果 operator 想看 Mac 侧的完整决策过程（不要求，纯 reference）：

| 文件 | 内容 |
|---|---|
| `review/0503/local/MULTI_AGENT_REVIEW_RECORD.md` §6 | Round 5 (C2.2 → C2.3) 的双 reviewer 评审、intersection 验证、empirical probe |
| `review/0503/local/MULTI_AGENT_REVIEW_RECORD.md` §5 | Round 4 (C1+C2 → C2.1+C2.2) 的 12 项 finding 严重度矩阵 |
| `review/0503/local/PEER_REVIEW_PROMPT_C2_2.md` | Round 5 用的 7 问题 prompt（Q-A..Q-G + Top-3 latent risks） |
| `de4cc88` 的 commit message | C2.3 完整 contract（D1 + E1 修复内容、新增测试列表、deferred items） |
| `c01aa27` 的 commit message | C2.2 完整 contract（A2/A3/B2a/B3b/B4/B10 修复内容） |

---

## 7. 回复存放位置约定

请 operator 把回复写到：

```
review/0503/operator/OPERATOR_REPLY_C2_3_20260504.md
```

如果当天日期已经变了，文件名后缀用 operator 实际写回复的那天日期（YYYYMMDD）。

写完后：

```bash
git add review/0503/operator/OPERATOR_REPLY_C2_3_*.md
git commit -s -m "operator reply: C2.3 confirm — <summary>"
git push gitee foc_lite_hop0
```

Mac 侧会在下一次 fetch 时看到。

---

— end of confirmation request —
