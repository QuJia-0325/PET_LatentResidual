# Review Summary (Post-Idea2)

## Round 1: Idea2 Outcome Check
- Evidence:
  - `idea2_best_full` 相比 `chainstable_best_full` 在 D10/D4/NORMAL 略退化。
  - `idea2_last_full` 可优于 `chainstable_best_full`。
  - 新补充：`chainstable_last_full` 也优于 `chainstable_best_full`，说明 best 选择存在窗口偏差风险。
- Ruling:
  Idea2 方向非零收益，但当前 recipe 不是稳定主线。

## Round 2: Mechanism Audit
- Main issues:
  1) matching/conflict 机制在当前实现与配置下不形成稳健可解释收益。
  2) rolling-window best 与 full-val ranking 不一致。
  3) 后期出现“局部损失变好但链路指标变差”的目标拉扯。
- Ruling:
  先收敛到更小、更硬的机制，减少不可解释自由度。

## Round 3: Remaining Ideas Re-Ranking

| Idea | 目标 | 复杂度 | 预期收益 | 决策 |
|---|---|---:|---:|---|
| FOC-lite (Idea3) | 直接压首跳数值漂移 | 低 | 高 | PASS (Primary) |
| PLPP-lite (Idea1) | 修分块与缝隙纹理 | 中低 | 中 | PASS (Supporting) |
| MAP-hop-lite (Idea5) | 更稳轨迹约束 | 中 | 中 | HOLD |
| DSGC (Idea4) | 自适应加权/引导 | 高 | 不确定 | HOLD |
| Alpha-CCT扩展 (Idea2扩展) | 分布对齐 | 中高 | 现阶段不稳 | DROP as primary |

## Consensus
1. 主线改为 `chainstable(last)` + `FOC-lite`。
2. 视觉修复只作为第二阶段可选支线（PLPP-lite），并受 chain guardrail 限制。
3. 评估协议必须固定：对称 full-val，不再把 rolling-window best 当最终证据。

## Remaining Reviewer Concerns to Validate
- C1 是否在 3 seeds 下稳定成立。
- PLPP 是否只是“变好看”而非“链路不退化”。
- 结果是否在高 seam-risk 桶同样成立。
