# Codex 任务包 — seam 伪影 provenance(A)+ seam_strata_summary 分层证据(B)

日期: 2026-06-01
分支: `foc_lite_hop0`
执行顺序: **A 先 B 后**(A 的 decoder/variant 对照结论决定 B 的分层口径与论文 novelty 措辞,B 在 A 锁定后再跑)

---

## 为什么是这个顺序

- **A 必须先做**: 当前论文 novelty 拟定位为"首个 latent 生成式(RAE+Meanflow)PET 重建 + 压制
  patch 硬拼接伪影"。但"before(有缝)→ after(无缝)"对照在仓库里**没有任何成文定义**
  (`seam_strata_summary.csv` 从 0424 一直挂到 `AUTO_REVIEW.md` 仍缺失)。若不先查清当前冻结
  decoder 到底是 `conv_head`(单源 PixelShuffle,硬缝)还是 `conv_overlap_head`(高斯重叠,抑缝),
  就无法界定"伪影抑制"归 RAE 架构还是归 PET 侧 `image_aux`。评审一查 RAE decoder 即可识破误 claim。
- **B 依赖 A**: B 的分层证据(高 seam 切片改善最大)只有在 A 确认了"before 对应 image_aux-off(V13),
  而非 RAE decoder 本身"之后,才能正确表述为"`image_aux` 进一步压制残余 seam",而不是越界 claim
  "我们修复了硬拼接"。

---

## 全局硬约束(A、B 通用)

- **不训练、不改主干网络**;只允许 B 中对 `evaluate()/write_csv` 做最小落盘改动。
- PSNR 一律用 `src.utils.metrics.calc_psnr_clip3`(先调窗到 3 再算),禁止其它口径。
- 输出写 `/data_2/qujiaxiang/...` 与 `review/`,**禁止写仓库内 outputs**
  (`eval_first_hop_224_clip3.py` 有 `ensure_repo_local_outputs_absent` 保护)。
- 统计纪律: **不写 patient-level p 值**(病人分组不可恢复);用 effect size +
  64-slice block-bootstrap 95% CI + win-rate;`scipy` 不可用,t 统计自算 mean/SEM。
- 不臆测、不编数;所有结论附文件+行号或原始 CSV 路径,可复算。

### 关键事实(供 Codex 对齐)

- eval 入口: `eval_first_hop_224_clip3.py`,full-val `n=7403`。
- 现状: eval **已计算** `seam_consistency_loss(x_pred, patch=14)`(预测自身 patch 网格不连续度,
  无需 GT,即"硬拼接伪影"量化值)与 `extended_seam_loss(x_pred, x_gt, patch=14, zone=3)`,
  **但只聚合**,per-slice CSV **仅含 `psnr*` 列**,无 per-slice seam 列。
- 冻结 decoder 来自 RAE: `RAE/src/stage1/decoders/{conv_head,conv_overlap_head,unetr_head}.py`。
  `conv_head` = single-source PixelShuffle(硬拼接,有缝);`conv_overlap_head` = 高斯加权重叠(抑缝)。
- 关键 variant: V13(`image_aux` λ=0,aux-off 基线)、V7(λ=0.04)、A4-mid(λ=0.08,headline)。
- 锚定数(NORMAL,canonical PSNR_clip3): V13=36.494330,V7=36.780951,A4-mid=36.893917。

---

## 任务 A — 确认 before/after decoder 对照,坐实 novelty claim(先做)

```
仓库: PET_LatentResidual (branch: foc_lite_hop0)
服务器: /home/qujiaxiang/project/PET_LatentResidual
冻结 decoder 来自 RAE: RAE/src/stage1/decoders/{conv_head,conv_overlap_head,unetr_head}.py

背景: 论文 novelty 拟定位为"首个 latent 生成式(RAE+Meanflow)PET 重建 + 压制
patch 硬拼接伪影"。需要坐实"before(有缝)→ after(无缝)"对照到底对应什么,
否则 novelty claim 站不住。conv_head=single-source PixelShuffle(硬拼接,有缝);
conv_overlap_head=高斯加权重叠(抑制缝)。

请只做只读代码考古,不训练、不改主干、不臆测,逐条给出文件+行号证据:

1. 确定 V7 / A4-mid / V13 三个 run 在【训练】和【评估】时实际加载并冻结的是哪个
   decoder head 类。给出:配置键、加载代码路径行号、checkpoint 里 decoder
   权重对应的 head 类型。三者是否完全一致?

2. 判定项目历史上是否存在过一个用 conv_head(single-source,硬缝)做 decoder 的
   阶段,其输出就是"有明显伪影"的 before 状态;若存在,给出它被 conv_overlap_head
   替换的 commit / 时间点 / 证据。

3. 明确区分两种"伪影抑制"分别归谁:
   (a) RAE 的 decoder head 架构(conv_head -> conv_overlap_head)—— 属 RAE 贡献;
   (b) PET 侧 image_aux 的 seam loss 在 transport 训练时把 z_pred 拉回可解码流形
       —— 属本工作贡献。
   给出代码证据,说明在当前冻结 decoder(已是 overlap?)下,A4-mid 相对 V13(aux off)
   的伪影改善具体由哪条路径产生。

4. 盘点服务器端(含 /data_2/qujiaxiang/...)是否已有任何 seam/伪影 before-after
   可视化图或生成脚本;若有,逐张标注其 provenance(哪个 checkpoint / head / variant /
   decode_mode 产生)。

产出: 一份 markdown(写入 review/0601/ 或指定目录),含上述 4 点的证据表,
并明确给出一句可写进论文的、有代码背书的 novelty 边界陈述
(哪些能 claim 是本工作的、哪些必须归 RAE)。
```

**A 的验收门槛**: 必须明确回答"`paper/main.pdf` 里若有 before/after 伪影图,其 before 面板对应
哪个 head/variant/checkpoint"。若结论是"当前 decoder 早已是 conv_overlap_head",则论文措辞
必须改为"`image_aux` 压制残余 seam",**不得** claim "修复硬拼接"。A 未结清前 **不要动 B**。

---

## 任务 B — 生成 seam_strata_summary.csv + 分层/局部指标(A 锁定后再做)

```
仓库: PET_LatentResidual (branch: foc_lite_hop0)
eval 入口: eval_first_hop_224_clip3.py
指标硬规范: PSNR 必须用 src.utils.metrics.calc_psnr_clip3(先调窗到3再算)。
现状: eval 已计算 seam_consistency_loss(x_pred, patch=14)(预测自身patch网格不连续度,
无需GT,即"硬拼接伪影"量化值)与 extended_seam_loss(x_pred,x_gt,patch=14,zone=3),
但只聚合,per-slice CSV 仅含 psnr* 列,无 per-slice seam 列。

目标: 生成 review 里反复缺失的 seam_strata_summary.csv,用分层证据证明
image_aux 的伪影抑制【真实、且集中在高 seam 切片】,而非切片选择偏差(回应历史 D2 裁决)。

步骤:
1. 最小改动 evaluate()/write_csv,使 per-slice 行额外落盘:
   seam_pred_{tp}(=seam_consistency_loss 每切片值)、ext_seam_{tp}。不改 PSNR 口径。
2. 用现有 best.pt 重跑 full-val eval(val, n=7403)两个 run:
   - V13(image_aux λ=0, aux-off 基线)
   - A4-mid(image_aux λ=0.08, headline)
   输出写 /data_2/.../outputs 下(仓库内禁写,脚本有 ensure_repo_local_outputs_absent)。
   若服务器已存在带 seam 的 per-slice 产物则直接复用,勿重复跑。
3. 分层(关键,避免循环/选择偏差):
   - 用【V13(aux-off)】每切片 seam_pred_NORMAL 作为与处理无关的 seam-risk 分层器,
     按分位数切 low/mid/high 三桶(报告各桶切片数与阈值)。
   - 在每桶内做【配对】比较 A4-mid 减 V13:报告
     ΔPSNR_NORMAL(均值、64-slice block-bootstrap 95% CI、win-rate)、
     Δseam_pred(伪影下降量、相对%)、Δext_seam。
   - 三桶全部报告,不许只挑 high 桶。预期:high-seam 桶 ΔPSNR 与 seam 下降最大。
4. 同样对 D20/D10/D4/NORMAL 全 timepoint 各出一版(至少 NORMAL 与 D50->D20)。

统计纪律: 不写 patient-level p 值(病人分组不可恢复);用 effect size + slice-level
block-bootstrap + win-rate;scipy 不可用,t 用 mean/SEM 自算。

产出:
- review/0601/seam_strata/seam_strata_summary.csv(列: timepoint, stratum,
  n, seam_threshold, psnr_V13, psnr_A4mid, dPSNR_mean, dPSNR_blkCI_lo/hi, win_rate,
  seam_V13, seam_A4mid, dseam_abs, dseam_pct)。
- 一份 markdown 解读: 是否支持"image_aux 在高 seam 切片上抑制伪影最强"的结论,
  以及全图 PSNR(+0.11dB)为何低估感知改善(seam 是局部高频,PSNR 全图平均不敏感)。
- 所有数字附原始 per-slice CSV 路径,可复算。
```

**B 的验收门槛**: 分层器必须用**与处理无关**的基线(V13 aux-off)seam,做**配对**(per-slice)比较,
三桶全报告。若 high-seam 桶 ΔPSNR 与 Δseam 显著大于 low-seam 桶 → 支持"伪影抑制真实且定位准确",
补齐 `seam_strata_summary.csv` 这一从 4 月挂到现在的缺失硬伤。

---

## 战略提醒(留给作者)

1. **任务 A 可能是双刃剑**: 若当前冻结 decoder 早已是 `conv_overlap_head`(缝已被 RAE 架构修掉),
   则 before 图只能对应 image_aux-off(V13),论文只能 claim "`image_aux` 进一步压制残余 seam"。
   先把这个搞清楚再定 novelty 措辞,A 必须先于论文定稿。
2. **任务 B 是把 "+0.11 dB" 翻盘成强故事的关键**: 分层若显示高 seam 桶改善最大,即得"全图 PSNR
   低估、但伪影抑制真实且定位准确"的硬证据,远强于单一 0.11 dB 数字。
3. **两项外部风险仍未解**(本任务包不覆盖): (a) effect-size 框架 —— 经 seam 重定位已缓解,
   **前提是 B 量化成功**;(b) 缺同 split 上对外部已发表 PET 方法的对照 —— 在做任何 "beats prior
   methods" claim 前必须补齐。
