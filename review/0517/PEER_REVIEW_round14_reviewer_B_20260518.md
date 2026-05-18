# Round 14 Peer Review — Reviewer B (independent)

- date: 2026-05-18
- substrate read independently (no draft cross-talk):
  - PEER_REVIEW_PROMPT_round14_A3_capacity_only_20260518.md (the prompt)
  - review/0517/V18_capacity_only/V18_capacity_only.yaml (read in full)
  - review/0517/V18_decoder_lora/V18_decoder_lora.yaml (read via mechanical diff)
  - CODEX_TASK_STAGE_C_A3_V18_CAPACITY_ONLY_20260518.md (full §0-§7)
  - V18_decoder_lora/V18_design_rationale.md §5.2 (B43-B47)
  - train_first_hop.py:1356-1357 (CLI flags), :1399-1403, :1902-1935 (resume-compat path), :2230 (z_kl source)
  - tools/probe_v18_kl_drift.py:45-55 (argparse surface)
- mechanical yaml diff: ran `flatten(v18) vs flatten(cap)` over 195 fields → exactly **4 differences** (output_dir, run_name, training.max_steps, loss.decoder_kl_pullback.lambda_kl). Numbers match task md.
- mandate per prompt §4: focus on executability + standing-rule landing + bias detection.

---

## TL;DR

| Item | Verdict |
|---|---|
| yaml | **READY TO USE** (mechanical diff = 4 fields exactly) |
| task md | **MODIFY before push** (3 small wording fixes + 1 mechanical gap; not blocking) |
| §5.2 standing rules B43-B47 | **MODIFY** — rules are sound but all 5 are *cultural*, not *mechanical*; B47 is closest. Worth flagging this honestly. |
| Push order | **B** (push commit 7486776 first, then A3 as isolated commit) |
| New biases | **B48-B54** (7 new), all LOW/MOD — none block push |

---

## 0. Mechanical verifications I ran first

1. **yaml field diff**: 195 leaf fields in each; symmetric difference = 4 fields, all match the table in prompt §1 Q1. ✓
2. **train_first_hop.py CLI flag set** = `--config --resume` (lines 1356-1357). Task md A0 EXPECTED string matches. ✓
3. **train_first_hop.py:2230** z_kl source verbatim:
   ```python
   z_kl = main_out["z_pred"] if bool(kl_cfg.get("use_pred_latent", True)) else main_batch["z_dst"]
   ```
   Task md §1.1 quote matches. ✓ (B42 mechanism reversal stands.)
4. **probe_v18_kl_drift.py** current argparse surface (lines 45-55):
   - `--config --v18-config --v7-ckpt --v18-best-ckpt --v18-last-ckpt --out-dir --batch-size --max-slices --split --device --num-workers`
   - **does NOT have `--v18-cap-ckpt`**. Task md §3 B1 invokes this flag as if it existed.
5. **resume-compat dead key**: yaml lines 182 and 244 both define `resume_allow_config_mismatch` (first `true`, then `false`). YAML last-wins → effective `false`. Inherited from V18 yaml (same duplicate). Not a blocker; flagged as B51.

---

## 1. Q1 — yaml 4-field diff precision

**Verdict: APPROVE.**

Mechanical diff over 195 leaf fields returned exactly the 4 expected differences. Specifically verified equal across:
- `training.decoder_lora.rank=32`, `last_n_blocks=2`, `alpha=16`, `dropout=0.0`, `init_scale_zero=true`, all 6 `target_keywords`
- `training.resume_from` = V7 best.pt (identical absolute path)
- `lr_schedule.total_steps_override=200000` (cosine reference frame identical → at any common absolute step, V18 and capacity-only see identical lr)
- `optimizer.decoder_lr_mult=0.00625`, `decoder_weight_decay=0.0`, all other optimizer fields
- `loss.image_aux.{l1_weight,ssim_weight,seam_weight,...}` (image_aux path identical)
- `loss.decoder_kl_pullback.use_pred_latent=true` (preserved, moot under lambda_kl=0)
- `seed=42`, `freeze_rae=false`, all rollout / pair_loss_weights / first_hop / rae / backbone / data fields

No additional field should change for the disambiguation to be clean. The yaml is mechanically correct.

---

## 2. Q2 — Is capacity truly "isolated"?

**Verdict: MODIFY (one significant comparability gap, two minor concerns).**

### 2.1 max_steps=170000 creates a training-duration mismatch (NEW, B48)

This is the biggest substantive issue I found. The pre-registered comparison in task md §3 B2 reads:

| ckpt | step | LoRA training steps from V7 |
|---|---:|---:|
| V7.best | 160000 | 0 |
| V18.best | 165000 | **5K** |
| V18.last | 200000 | **40K** |
| V18-cap.last | 170000 | **10K** |

`capacity-only.last` is at **10K LoRA training**, but the headline pre-registered comparison ("≥ +0.07 dB" threshold) is calibrated against **V18.best @ 5K LoRA training**. Mismatched LoRA-training durations. Per B46 (4×2 comparison-label rule), the cleanest reading is **capacity-only @ 5K LoRA vs V18.best @ 5K LoRA**, then independently **capacity-only @ 10K LoRA vs V18.last/2 @ ~10K LoRA**.

The yaml has `save_interval=10000` and resume_from=V7@160K → ckpts emit at 170K only. So `step_165000.pt` (the matched-LoRA-training comparand to V18.best) will **not exist** unless save_interval is reduced.

**Recommended yaml fix (minor):**
```diff
-  save_interval: 10000
+  save_interval: 5000   # capacity-only: emit step_165000.pt for matched LoRA-training comparison with V18.best
```
Cost: ~one extra ckpt write at step 165K, negligible.

Without this fix, the disambiguation is still meaningful but not as clean as it could be. The `best.pt` mechanism (best_select_full_eval_interval=5000) might happen to land at 165K, but that's selection-by-chain-MSE, not by training-step.

### 2.2 use_pred_latent=true preserved (yaml + header comment)

The yaml retains `use_pred_latent: true` because lambda_kl=0 makes the lookup moot. Task md §0.5 acknowledges this. **I agree with the choice**: changing two fields at once (lambda_kl=0 AND use_pred_latent=false) would re-confound the disambiguation. Keep `use_pred_latent=true`, but **add a single-line comment in yaml** at that field:
```yaml
    # CAPACITY-ONLY: kept identical to V18 yaml so the only computational delta
    # vs V18 is lambda_kl=0. Field is moot here (KL loss multiplied by 0).
    use_pred_latent: true
```
This pre-empts a reviewer-misreading or future codex-confused edit ("oh, capacity-only inherited the buggy flag, let me fix it"). LOW priority.

### 2.3 lr stage matching

At step 170K both V18 and capacity-only see lr ≈ 8.1e-6 (cosine, total_steps_override=200000, warmup ends at 30K). lr stage match is exact at any common step. ✓ — but this is an *abstract* match; the practical question is "does the LoRA gradient signal at 5K LoRA training look the same in both runs?". Since both runs share the same V7@160K starting point, same image_aux schedule (locked-in V6 absolute steps), same optimizer, only the KL term differs. Under lambda_kl=0, capacity-only at step k has the *same effective gradient* as V18 at step k *would have had* with KL disabled. ✓ control is mechanistically clean.

---

## 3. Q3 — task md §1 anchoring & outcome coverage

**Verdict: MODIFY (3 wording issues, none blocking).**

### 3.1 §1.2 "默认最简释" anchors codex toward B (B54)

The prompt itself flagged this risk under Q3. Confirming: §1.2 header reads *"候选 B (默认最简释)"*. After B42, B is the only mechanistically consistent candidate — but calling it "default simplest explanation" up front nudges codex to interpret borderline outcomes as B. Suggested rewording:

```diff
-### 1.2 候选 B (默认最简释)
+### 1.2 候选 B (B42 之后唯一机制上一致的解释 — 实测验证)
```

This shifts framing from "we already know it's B, go confirm" to "B is what the mechanism predicts; this experiment is the falsification test".

### 3.2 §1.3 outcome 2 contradicts B44 standing rule (B53)

Task md §1.3 outcome 2 reads: *"capacity-only ≈ V7 (0 dB) → A 候选可能真"*. But §5.2 B44 (added in this same push) states: *"任何 'KL pullback 让 V18 decoder 在 GT manifold 对齐' 的解读机制上不可能成立"*. Outcome 2's "A 候选可能真" allows the very interpretation B44 rules out.

The honest framing for outcome 2 is: *"capacity-only ≈ V7 → +0.10 dB in V18 is **not** explained by LoRA capacity alone. B44 rules out direct KL-to-GT alignment as the mechanism. Therefore some indirect mechanism (e.g. KL on z_pred indirectly shapes shared decoder weights → bleed to z_GT eval path) is at work. Open a sub-investigation; do **not** revive A."*

Suggested patch:
```diff
-2. **capacity-only ≈ V7 (0 dB)** → A 候选可能真 (但仍违反 §1.1 机制, 需二次调查; 可能 KL 间接经 z_pred 训练影响 z_GT 路径)
+2. **capacity-only ≈ V7 (0 dB)** → +0.10 dB **不是** LoRA capacity 副产品. B44 已 rule out 直接 KL→z_GT 对齐. 必有间接机制 (e.g. KL on z_pred → 共享权重梯度 → 渗到 z_GT eval 路径). 开 sub-investigation, **不**复活 A.
```

### 3.3 §B1 invokes non-existent --v18-cap-ckpt flag (B49)

probe_v18_kl_drift.py line 45-55 has 11 args; `--v18-cap-ckpt` is not among them. Task md §3 B1 invokes:
```bash
  --v18-cap-ckpt /data_2/.../V18_capacity_only/.../last.pt \
```
And §6 failure tree at "B1" hedges: *"若 probe 不支持新 flag, codex 改 ~10 行 (加 ckpt arg + 第 4 行 PSNR 输出)"*. The "若" (if) is misleading — the flag definitely does not exist, so the ~10-line probe.py edit is **required**, not contingent. Suggested patch:

```diff
-### B1: KL drift probe 复用
-
-V18-capacity-only step 170K (`last.pt`) 跑完后, 复用现有 [tools/probe_v18_kl_drift.py](../../tools/probe_v18_kl_drift.py), 加 capacity-only ckpt 作为第 4 个比较点.
+### B1: KL drift probe 扩展 (~10 行)
+
+probe_v18_kl_drift.py 当前 argparse (lines 45-55) 没有 `--v18-cap-ckpt`, **必须**加. 改动范围:
+- 加 `p.add_argument("--v18-cap-ckpt", default=None)` (1 行)
+- ckpt 加载逻辑加 capacity-only 分支 (~3 行)
+- 报告输出加第 4 行 PSNR (~3 行)
+- 总计 ≤ 10 行, **仅改 tools/probe_v18_kl_drift.py**, 不动训练器.
```

This makes §5 NOT-DO #2 ("不改 train_first_hop.py / decoder_lora.py 任何代码") **non-contradictory** with §B1 — probe.py is explicitly *not* in NOT-DO list. But §5 should add an explicit allow-line for clarity (B50):

```diff
 5. ❌ 不改 train_first_hop.py / decoder_lora.py 任何代码
+   ✅ 例外: tools/probe_v18_kl_drift.py 允许 +~10 行加 capacity-only ckpt 参数 (§B1).
```

### 3.4 Coverage of 3 outcomes — 4th outcome ("capacity-only **worse** than V7")

Task md §1.3 has 3 outcomes (cap ≈ V18 / cap ≈ V7 / cap mid). The prompt §1 Q3 asked about a 4th: "capacity-only 显著差于 V7, 即 LoRA capacity 反而**有害**". I think this 4th outcome is **mechanistically unlikely but not impossible**:
- LoRA init_scale_zero=true → step 0 output exactly = V7. So the *trajectory* over 10K steps could in principle degrade decode(z_GT) if image_aux gradient on z_pred pushes LoRA params in a direction harmful for z_GT decoding.
- Probability: low (image_aux is on the same decode head as eval; harm would also harm image_aux loss, which the optimizer minimizes).
- But worth listing for completeness.

Suggested addendum:
```diff
 3. **capacity-only ∈ (V7, V18)** → 混合 (capacity 部分 + KL 间接 bleed-through)
+4. **capacity-only < V7 (负 drift)** → LoRA capacity 有害 (image_aux 梯度推 LoRA 到对 z_GT decoding 不利方向). 机制上不太可能但需 cover. → 决策: V18 +0.10 dB 来自 KL 设计的不对称效应; 重新审 B42 mechanism.
```

---

## 4. Q4 — Are B43-B47 actionable defenses or cultural notes?

**Verdict: MODIFY (rules are well-stated but the cultural-vs-mechanical claim deserves an honest table).**

| rule | substance | enforcement | classification |
|---|---|---|---|
| B43 — V18 "in buggy KL config" prefix | Tag every V18 number with "in V18 buggy KL config" prefix | **No mechanical enforcement.** No template, no script, no CI. Pure convention. | **CULTURAL** |
| B44 — KL pullback decoupled from decode(z_GT) | Rules out "KL aligned GT manifold" as an interpretation | **No mechanical enforcement.** Cited in design_rationale; only enforced when claude or reviewer reads it. | **CULTURAL** |
| B45 — Substrate-stage reviewer mandatory | Strategic forks must trigger reviewer | **No mechanical enforcement.** Relies on user invoking review skill. Round 13 already proved value but no gate. | **CULTURAL** |
| B46 — 4×2 comparison labels | Each ΔPSNR must show (best-vs-best, last-vs-best) × 4 chain points | **Partially mechanical**: prompt §6 Q4 says "需要 reviewer verify in next round". No script that fails on missing labels. | **CULTURAL-with-protocol** |
| B47 — Mechanical scripts assert-on-fail | grep verify scripts must exit 1 on any ✗ | **Closest to mechanical**: explicit protocol (exit 1, -z guard, ERR counter). Still relies on user *running* the script, not on CI. | **PROTOCOL** |

**My honest take**: all 5 are cultural, B47 is "protocol that humans must execute". None is a CI hook or a template generator. The §5.2 wording "permanent standing rules" overstates enforcement strength — it makes them sound mechanical when they're norms.

Suggested addition to §5.2 (honest framing):

```diff
+#### Enforcement honesty: cultural vs mechanical
+
+B43-B47 are *cultural* (B43-B45) or *protocol* (B46-B47) rules. None has CI / template
+/ pre-commit enforcement. They depend on claude / reviewer / user honoring them at each
+invocation. Mechanical enforcement (e.g. lint script that scans markdown for `V18` not
+prefixed by "in V18 buggy KL config") is a future enhancement, not in this commit.
```

This costs nothing and preempts a reviewer flagging "your rules don't enforce themselves".

---

## 5. Q5 — NOT-DO completeness

**Verdict: MODIFY (4 missing items, all low-risk).**

Suggested additions to §5:

```diff
+10. ❌ 不修改 review/0517/V18_capacity_only/V18_capacity_only.yaml (本 task 已 ready)
+11. ❌ 不在 capacity-only 训练期间 push 任何 unrelated commit
+12. ❌ 不重命名 / 移动 V7 best.pt / V18 ckpt 文件路径 (resume_from 是绝对路径)
+13. ❌ 不在 V18_capacity_only/run/ 下 cp 任何 V18 ckpt 文件 (output_dir 隔离已防, 但 cp 操作会破坏)
+   ✅ 例外 (来自 §3.3): tools/probe_v18_kl_drift.py 允许 +~10 行扩展 (见 §B1)
```

---

## 6. Q6 — New biases (B48-B54)

### B48 (MODERATE) — capacity-only step mismatch vs V18.best

`max_steps=170000` + `save_interval=10000` + `resume_from=V7@160K` → capacity-only emits ckpt at step_170000 only. V18.best is at 165K (5K LoRA training). The cleanest disambiguation needs a capacity-only ckpt at 165K (matched LoRA training). **Fix: yaml change `save_interval: 10000 → 5000`** (1-line; negligible cost).

### B49 (LOW) — `--v18-cap-ckpt` flag does not exist; §B1 treats it as supported

probe.py argparse has no such flag. §6 failure tree hedges with "若 probe 不支持". The "若" is misleading — the ~10-line probe edit is **required**, not conditional. **Fix: rewrite §B1 to state the probe extension up front** (see §3.3 above).

### B50 (LOW) — §5 NOT-DO #2 vs §B1 probe edit (contradiction)

§5 NOT-DO #2 says "不改 train_first_hop.py / decoder_lora.py 任何代码"; §B1 (correctly) edits probe.py. Probe.py is *not* in NOT-DO #2, so it's technically allowed, but a careful codex reading this will hesitate. **Fix: explicit `✅ 例外` line in §5** (see §5 above).

### B51 (LOW) — yaml duplicate keys (inherited from V18)

Lines 182 and 244 both declare `resume_allow_config_mismatch` (true, then false). YAML last-wins → effective `false`. Lines 183 and 245 both declare `resume_allow_missing_compat_metadata` (false, false — harmless). The line-182/183 block under `training:` is dead code inherited from V18 yaml. Not a blocker. **Fix: defer to next yaml hygiene pass; not worth holding the push.**

### B52 (LOW) — kill_switch.check_step=5000 dead under resume

`loss.decoder_kl_pullback.kill_switch.check_step: 5000` is absolute, but training resumes at step 160000. Step 5000 is never reached. Dead config, inherited from V18 (where it was already dead). Doubly moot under lambda_kl=0. Not a blocker.

### B53 (MODERATE) — §1.3 outcome 2 contradicts §5.2 B44

See §3.2 above. Same commit introduces B44 ("KL→z_GT alignment 机制不可能") and §1.3 outcome 2 ("A 候选可能真"). Two parts of one push contradict each other. **Fix: rewrite outcome 2** (patch in §3.2).

### B54 (LOW) — §1.2 anchoring "默认最简释"

The prompt §1 Q3 itself flagged this risk; confirming after review. **Fix: §3.1 patch above.**

### Round-up

7 new biases total. None block push. Highest priority is **B48 (save_interval fix)** because it affects experiment cleanness, then **B53 + B49 + B50** (task md wording).

---

## 7. Output per §3 format

### 7.1 Per-question

| Q | verdict | reason |
|---|---|---|
| Q1 yaml 4-field diff | APPROVE | mechanical diff = exactly 4 fields ✓ |
| Q2 capacity isolation | MODIFY | B48 step mismatch; recommend save_interval=10000→5000 |
| Q3 task md §1 anchoring | MODIFY | B54 wording, B53 outcome 2 contradicts B44, B49 probe flag |
| Q4 standing rules mechanical | MODIFY | rules are cultural/protocol; add enforcement-honesty note |
| Q5 NOT-DO completeness | MODIFY | 4 small additions (#10-#13 + probe exemption) |
| Q6 new biases | 7 new (B48-B54) | LOW/MODERATE; none block push |

### 7.2 Overall verdict (3 substrates)

- **yaml**: **READY TO USE** — but recommend pre-push 1-line edit `save_interval: 10000 → 5000` (B48). Without this, experiment still works; with it, comparability is cleaner.
- **task md**: **MODIFY then push** — 4 small wording patches (§1.2, §1.3, §B1, §5). All <30 lines total. No blockers.
- **§5.2 standing rules**: **MODIFY** — add enforcement-honesty paragraph; don't claim mechanical enforcement where there is none. Substance of B43-B47 is sound.

### 7.3 Push order

**Option B**: push commit 7486776 first (V13 launch unblocked, content already reviewed in prior rounds), then A3 as isolated commit after wording patches. Rationale:
- 7486776 is already in commit form, lower risk of further edits
- A3 has 4 wording patches outstanding (B48/B49/B50/B53/B54 fixes)
- After A3 patches land, A3 commit is clean and reviewable in its own diff
- Codex picks up V13 (slot 2) and A3 (slot 1) independently — order doesn't affect runtime parallelism since slot 1 is free now

Reject option A (one bundled commit): bundling re-opens 7486776 for re-edit and loses its atomicity.
Reject option C (A3 first, V13 deferred 1 week): no upside; V13 has its own scientific value independent of A3 outcome.

### 7.4 New biases

B48-B54 detailed in §6 above. Top three to address pre-push:
1. **B48** (yaml): `save_interval: 10000 → 5000` — 1 line, enables matched-training-step comparison
2. **B53** (task md §1.3 outcome 2): rewrite to be consistent with §5.2 B44
3. **B49 + B50** (task md §B1 + §5 NOT-DO #2): probe.py extension is required, not optional; explicit exemption in NOT-DO

---

## 8. One-paragraph executive summary

The yaml is mechanically clean: 195-field diff produces exactly the 4 advertised fields. The task md is broadly correct but has one comparability gap (B48: `save_interval=10000` precludes a matched-LoRA-training-step ckpt at 165K against V18.best) and three small wording issues (B49 probe flag wrongly framed as conditional, B53 outcome 2 contradicts the new B44 standing rule, B54 §1.2 anchor wording flagged by the prompt itself). The five new standing rules B43-B47 are substantively sound but are all *cultural/protocol*, not *mechanical* — claiming "permanent standing rule" overstates enforcement strength; recommend adding an honest enforcement-classification paragraph. None of these issues block the push. Recommended order: push 7486776 first to unblock V13, then patch A3's four wording/yaml items and push as an isolated commit.

---

## 9. What I did NOT review (per prompt §0 boundary)

- Round 1-13 already-signed content
- V13 design / commit 7486776 contents
- audit DRAFT release
- Round 12 KL drift probe code quality (Round 12 already verified)
