# 架构图 Prompt 合集 v3 — GPT Image 2 / ACL RR / ICLR / ICML

本文档把 PET_LatentResidual 的总体结构、V6 transport-first 训练策略、E1 误差预算分析整理成可直接复制到 GPT Image 2 的高质量图像生成 prompt。

目标不是“好看的科研插图”，而是能进入 ACL RR / ICLR / ICML / NeurIPS 论文或 rebuttal PDF 的方法图：结构正确、文字可读、证据边界严谨、审稿人 5 秒内能看懂主张。

---

## 0. 全局设计锁定（复制任意单张 prompt 时也建议带上）

### GPT Image 2 通用生成契约

```text
Create a direct-use, publication-quality academic figure for a top-tier machine learning / medical imaging conference paper.

GLOBAL STYLE CONTRACT:
- White background only. No decorative background, no texture, no gradient canvas.
- Vector-like flat design with crisp anti-aliased edges, aligned geometry, and balanced whitespace.
- Use a restrained 5-color academic palette:
  - frozen / encoder-decoder modules: deep teal #2D6A6A
  - trainable transport modules: warm amber #D4893F
  - auxiliary pixel / stabilizing paths: muted sage #7B9E6B
  - supervision / error / caveat signals: soft rose #C17B8E
  - text / axes / arrows: cool slate #4B5563
- Use tinted fills with 0.6-1.0 pt desaturated borders. Avoid heavy black outlines.
- Use rounded rectangles with 8-12 px radius. Keep corners consistent.
- Use subtle same-color gradients only inside major modules, barely visible. No glossy, 3D, glowing, neon, rainbow, or heavy shadow effects.
- Typography: clean sans-serif, similar to Helvetica / Inter / Arial. Use medium weight for module names, regular weight for annotations. Text must remain legible when the figure is reduced to double-column paper width.
- Render all labels and formulas exactly as written. Do not invent extra labels. Do not create pseudo-text, misspellings, watermark-like random strings, or decorative microtext.
- Keep label text short. If a label is long, use a two-line label inside the same module rather than shrinking it below readability.
- All arrows must have clear source and target. Use dark slate arrows, 1.5-2.5 pt for main flow and 0.75-1.0 pt for secondary paths. Use filled or open arrowheads consistently.
- Avoid arrow crossings. Use curved connectors only when they improve clarity; otherwise use straight orthogonal or horizontal connectors.
- Do not use emoji. Use simple vector pictograms only when they clarify a module: lock for frozen, small grid for latent, small PET slice thumbnail for image domain.
- Do not use clip-art, cartoon icons, 3D perspective, photorealistic equipment, or decorative medical stock imagery.
- PET thumbnails should be stylized synthetic slices: circular brain cross-section, PET-like warm colormap, no patient identifiers, no anatomy labels.
- The final image should look like a carefully designed figure from an ICLR / ICML / ACL paper, not a slide deck or beginner flowchart.
```

### 统一视觉语义

| 视觉编码 | 含义 |
|---|---|
| Teal | frozen RAE encoder / frozen ViT-MAE decoder / GT-associated references |
| Amber | trainable latent transport / DiT backbone / rollout optimization |
| Sage | first-hop pixel prior / stabilizing auxiliary branch / sensitivity product |
| Rose | supervision losses / error halos / caveats |
| Slate | text, axes, neutral arrows, equations |

### 生成后必须检查

1. 所有时间态顺序必须是 `D50 → D20 → D10 → D4 → NORMAL`。
2. 主 rollout state 只能是 latent；pixel branch 只能在 hop0 作为条件输入，不能画成第二条 state。
3. 图 4 的 E1 只能表述为 `decoded-space transport discrepancy` 或 `transport-related decoded discrepancy`，不能写成 `pure velocity-field error`。
4. 图 3 的 V6 Phase I chain 退化只能画成“risk / gate / to be verified in Phase II”，不能画成 V6 已经成功或失败。
5. 如果任何文字拼错、公式缺符号、箭头方向错误、数字被改写，直接重生成，不要手工补救用于最终稿。

---

## 图 1：Overall Pipeline（PET latent transport 全局流程）

**图的任务**：纯架构图——展示数据流和模块连接关系。**不包含训练 loss**（loss 移到图 3 训练策略图中）。

**推荐尺寸**：`2600×900 px`，宽高比约 `2.9:1`（去掉底部 supervision 后更扁）。

### Prompt v3.2 — GPT-5.3 多轮润色后的最终版

```text
Create a minimal, publication-quality architecture figure for a top-tier ML conference (ICML / ICLR / NeurIPS). The figure must be clean, uncluttered, and visually balanced, focusing on a single core idea: a latent-only 4-hop transport pipeline with a first-hop-only pixel prior. This is a PURE ARCHITECTURE DIAGRAM — no training losses, no supervision signals.

CANVAS:
- Wide panoramic layout, aspect ratio ~2.9:1 (2600×900 px)
- Pure white background, no panels, no background boxes
- Generous whitespace and margins

COLOR PALETTE (soft, desaturated):
- Teal (#2D6A6A): frozen modules (encoder, decoder)
- Amber (#D4893F): transport hops
- Sage (#7B9E6B): hop0 pixel prior
- Rose (#C17B8E): supervision signals
- Slate (#4B5563): text and arrows

TYPOGRAPHY:
- Clean sans-serif font
- No bold heavy titles
- Mathematical symbols italic

--------------------------------------------------
LEFT: INPUT AND ENCODER (small, de-emphasized)
--------------------------------------------------
- A circular PET brain scan (noisy, low-dose), warm yellow-green tone
- Label below: "Low-dose PET (D50)"

- Arrow to a small teal module:
  "Frozen RAE Encoder"
  sub-label: "DINOv2-LoRA"
- Include a small lock icon inside encoder

- Output: a small latent grid icon (16×16 style)
- Label: "z_D50"

--------------------------------------------------
CENTER: LATENT TRANSPORT (MAIN FOCUS)
--------------------------------------------------
- Four amber rounded rectangles, evenly spaced, slightly larger than encoder/decoder

Labels on each:
- "Hop 0: D50→D20"
- "Hop 1: D20→D10"
- "Hop 2: D10→D4"
- "Hop 3: D4→NORMAL"

Inside each hop:
- ONLY a single centered formula:
  "Δz = f_θ(z, t)"
- No internal block diagrams, no horizontal lines

Extra:
- On Hop 0, add small grey annotation "(bottleneck)"

Between hops:
- small latent grid icons labeled:
  z_D20, z_D10, z_D4, z_NORMAL

Main arrows:
z_D50 → Hop0 → z_D20 → Hop1 → z_D10 → Hop2 → z_D4 → Hop3 → z_NORMAL

--------------------------------------------------
HOP0 PIXEL PRIOR (SUBTLE SECONDARY PATH)
--------------------------------------------------
- Thin sage-colored arrow from input image → small module above Hop 0

Module label:
"Pixel prior (hop0 only)"

- Below it: a small circle labeled "g_pix"

- Connect into Hop 0 only

- Small annotation:
"not part of rollout"

--------------------------------------------------
SCHEDULED MIXING (VERY SUBTLE)
--------------------------------------------------
- Small diamond nodes between hops labeled "α"

- Two thin inputs:
  solid teal: "GT"
  dashed amber: "pred"

- Keep very small and low contrast

--------------------------------------------------
RIGHT: DECODER AND OUTPUT (small)
--------------------------------------------------
- z_NORMAL → teal module:
  "Frozen Decoder"
  sub-label: "ViT-MAE"

- Include lock icon

- Output: clean PET image (same colormap but smoother)

- Label:
"Reconstructed normal-dose PET"

--------------------------------------------------
STRICT RULES:
--------------------------------------------------
- No loss signals (L_pair, L_roll, L_img) — those belong in Fig 3
- No supervision arrows or rose-colored elements
- No mini-decoder / D20 pred inset
- No explanation panels, no legend box
- No horizontal bars inside hops
- No decorative elements
- Keep everything minimal and balanced
- This is ARCHITECTURE ONLY — data flow and module connections

FINAL STYLE:
- Clean, academic, minimal
- Similar to Nature / NeurIPS best paper figures
- High clarity, low visual noise
```

### Prompt v3.1 — 针对当前 Image 1 的局部修正

如果 GPT Image 2 已经生成了整体 pipeline 还不错、但底部 `L_pair / L_roll / L_img` 很丑的版本，直接把当前图作为 reference image，并使用下面的 refinement prompt：

```text
Refine the attached Figure 1 while preserving the main PET latent transport pipeline.

KEEP UNCHANGED:
- Keep the left-to-right pipeline, PET thumbnails, frozen encoder, four amber hop cards, latent grids, scheduled α-mixing nodes, hop0 pixel prior branch, frozen decoder, and legend style.
- Keep the same academic color palette and white background.
- Keep all core labels and arrow directions in the main pipeline.

REDESIGN ONLY THE BOTTOM SUPERVISION AREA:
- The current bottom supervision layout looks too much like three detached UI buttons with messy long red curves. Replace it with a clean integrated academic annotation rail.
- Remove the large left-side "Supervision Signals" label and remove the vertical stack of three large rose badges.
- Under the four transport cards, create a slim aligned supervision band occupying no more than 15-18% of the figure height.

NEW SUPERVISION LAYOUT:
1. L_pair row:
  - Place a small rose text chip "L_pair" directly under the left edge of the transport chain, not on the far left margin.
  - Put the annotation "GT-input velocity + endpoint" beside it in small slate/rose text.
  - Draw four short vertical dashed rose taps upward, one tap to each hop card. These taps should be short and nearly vertical, not diagonal curves.

2. L_roll row:
  - Place one thin rose bracket below the L_pair row, spanning exactly from Hop 0 to Hop 3.
  - Center the label on the bracket: "L_roll: open-loop chain consistency".
  - Use clean bracket end caps. No arrowheads. No sweeping curves.

3. L_img local inset:
  - Place this only under Hop 0 / Hop 1, not across the full figure.
  - Draw a small local path: "L_img: hop0 image auxiliary" → tiny teal mini-decoder "Dec(z_D20_pred)" → small D20 PET thumbnail "D20 pred".
  - Connect Hop 0 to the mini-decoder with one short downward dashed rose arrow.
  - Do not connect L_img to the final NORMAL output.

NEGATIVE REQUIREMENTS:
- No long sweeping rose curves.
- No fan-shaped supervision lines.
- No large pill-shaped loss buttons on the left margin.
- No red lines crossing the central latent chain.
- No clutter in the bottom band.

The revised bottom supervision area should look like fine, precise annotations in an ICLR / ICML architecture figure, not like a UI control panel.
```

---

## 图 2：PETFlowDiTFirstHop 内部结构

**图的任务**：解释单步 transport 模型内部如何工作：`z_src` 与 hop/time conditioning 进入共享 DiT；hop0 可加 pixel prior；输出经 hop residual 与 `Δt` 得到 `z_pred`。

**推荐尺寸**：`2200×1100 px`，宽高比约 `2:1`。

### Prompt v3.2 — GPT Image 2

```text
Create a minimal, publication-quality model architecture diagram for a top-tier ML conference (ICML / ICLR / NeurIPS). Clean, uncluttered, visually balanced. Focus on showing the single-step latent velocity prediction path shared across all hops.

CANVAS:
- 2200×1100 px, pure white background
- Left-to-right data flow
- Generous whitespace, no background panels

COLOR PALETTE (soft, desaturated):
- Teal (#2D6A6A): frozen / reference
- Amber (#D4893F): trainable backbone and heads
- Sage (#7B9E6B): hop0 pixel prior
- Lavender (#8B7FB5): conditioning signals
- Slate (#4B5563): text, arrows, equations

TYPOGRAPHY:
- Clean sans-serif, no bold heavy titles
- Mathematical symbols italic
- Module names inside modules, annotations outside

--------------------------------------------------
LEFT: INPUTS
--------------------------------------------------

Main input (bottom-left):
- Small latent grid icon labeled "z_src"
- Below: "16²×768" in small slate text
- Thick slate arrow → merge node "⊕"

Conditioning (top-left):
- Three small lavender circles: "t_src", "t_dst", "hop"
- Arrow → small lavender module "Cond. MLP"
- Output: thin lavender bar "c(t, hop)"
- Thin lavender lines from c inject into backbone (very subtle)

Pixel prior (far top-left, subtle):
- Tiny PET thumbnail "x_D50"
- → small sage module:
  "Pixel Encoder"
  sub: "Conv→Pool→Proj"
- → gate circle "g_pix"
- → merge node "⊕" (same as main input merge)
- Small annotation: "hop₀ only"

--------------------------------------------------
CENTER: SHARED BACKBONE
--------------------------------------------------

Single amber rounded rectangle, the largest element:
- Label inside:
  "Shared DiT-S"
  sub: "12×(d=384) + 2×(d=2048)"
- Below module: "198.7M params, shared across all hops"
- Show conditioning c entering from top as thin lavender arrow
- No internal block diagrams — keep the interior clean

--------------------------------------------------
RIGHT: OUTPUT PATH
--------------------------------------------------

Backbone output "h" splits into two paths:

Upper path:
- → small amber module "Velocity Head"
- → "v_backbone"

Lower path:
- → small amber module:
  "Hop Residual"
  sub: "1×1 Conv"
- → gate circle "λ_hop"
- → "λ_hop · r_hop"

Both merge at "⊕" node → "v_total"

Final equation in a clean slate badge:
"z_pred = z_src + Δt · v_total"

Output: small latent grid icon "z_pred"

--------------------------------------------------
BOTTOM: SHARED PATH ANNOTATION
--------------------------------------------------

Thin dashed outline around entire figure labeled:
"predict_latent_step(...)"

Below, three small slate text labels in a row:
"pair training" · "rollout step" · "inference"
(showing all three use the same path)

--------------------------------------------------
STRICT RULES:
--------------------------------------------------
- No internal transformer layer bars inside backbone
- No pixel features entering decoder
- No separate trunk per hop
- Residual head must be visually smaller than backbone
- No decorative elements, no legend box
- Keep everything minimal and balanced
```

---

## 图 3：V6 Transport-First 三阶段训练策略

**图的任务**：表达 V6 的训练压力重分配策略 + **三类 loss 的作用位置**（从 Fig 1 迁移到此）。Phase I 强化 GT-input transport；Phase II 逐步引入 rollout / α-mixing；Phase III open-loop refinement。保留 50K-75K chain-quality gate。

**推荐尺寸**：`2200×1500 px`，宽高比约 `1.47:1`（增加一个 panel 放 supervision）。

### Prompt v3.3 — GPT Image 2

```text
Create a minimal, publication-quality multi-panel training figure for a top-tier ML conference (ICML / ICLR / NeurIPS). Clean data visualization, no chartjunk, Tufte-inspired. This figure covers BOTH the training schedule AND supervision design (losses moved here from the architecture figure).

CANVAS:
- 2200×1500 px, pure white background
- Four vertically stacked panels, shared x-axis (0 to 200K steps)
- L-shaped axes only, no boxed frames
- Generous vertical spacing between panels

COLOR PALETTE (same series):
- Teal (#2D6A6A): pair / GT-input
- Amber (#D4893F): rollout / chain
- Sage (#7B9E6B): image auxiliary
- Rose (#C17B8E): risk / gate
- Slate (#4B5563): axes and text

TYPOGRAPHY:
- Clean sans-serif, math symbols italic
- Inline labels at curve ends, no separate legend boxes

--------------------------------------------------
PHASE BANDS (very subtle, spanning all panels)
--------------------------------------------------
Three barely-visible vertical color washes:
- 0–50K: pale teal, label above: "Phase I: Velocity Precision"
- 50K–150K: pale amber, label: "Phase II: Chain Transition"
- 150K–200K: pale gray-rose, label: "Phase III: Refinement"

Labels appear once above panel (a), small caps, slate text.

--------------------------------------------------
PANEL (a): HYPERPARAMETER SCHEDULE
--------------------------------------------------
Y-axis left: "Value" (0 to 4.5)
Y-axis right: "Pair weight" (0 to 16)

Four curves:
1. Amber solid: "λ_roll" — 0 in Phase I, ramp 0→4.0 in Phase II, flat 4.0 in Phase III
2. Slate dashed: "α" — same timing, ramp 0→1
3. Sage thin line: "λ_img = 0.04" — constant, near bottom
4. Teal thin line (right axis): "pair_weight = 15" — constant, near top

Inline labels at right end of each curve. No legend box.

--------------------------------------------------
PANEL (b): LOSS FRACTION EVOLUTION
--------------------------------------------------
Y-axis: "Weighted loss fraction" (0% to 100%)

Smooth stacked area chart (river diagram style):
- Teal area (bottom): "pair" — starts ~85%, decreases to ~27%
- Amber area (middle): "rollout" — starts 0%, grows to ~70%
- Sage area (top): "image" — starts ~15%, shrinks to ~3%

Boundaries are smooth bezier curves, 40% opacity fills.

Rose callout at 50K–75K region:
"chain MSE must recover here"
(arrow pointing to early Phase II, not to final result)

--------------------------------------------------
PANEL (c): HOP × CHANNEL BUBBLE MATRIX
--------------------------------------------------

Rows: "hop₀", "hop₁", "hop₂", "hop₃"
Columns: "pair weight", "rollout weight", "image"

Circles sized by value:
- pair: 2.5, 1.0, 1.0, 1.0 (hop₀ largest, teal)
- rollout: 0.5, 2.0, 1.5, 1.0 (hop₁ largest, amber)
- image: 0.04, —, —, — (only hop₀, tiny sage)

Values as small text next to each bubble.
Sage annotation arrow → hop₁ rollout bubble: "sweet spot"

--------------------------------------------------
BOTTOM: PHASE GATE TIMELINE
--------------------------------------------------
Thin horizontal timeline with ticks: 50K, 65K, 75K, 150K, 200K
Rose label at 65K: "go/no-go gate"
Small slate text: "Phase I chain lag is expected, not a verdict"

--------------------------------------------------
PANEL (d): SUPERVISION DESIGN (from architecture)
--------------------------------------------------
A compact schematic showing how three losses attach to the 4-hop chain.
Draw a simplified horizontal strip of four small amber boxes "H0 H1 H2 H3" at the top of this panel.

Below the strip, show three loss annotations:

1. L_pair (rose, dashed vertical taps):
   - Four short dashed rose lines, one going UP to each hop box
   - Label: "L_pair: velocity + endpoint (GT input, λ_p=15)"
   - Annotation: "per-hop, pair_loss_weights = [2.5, 1.0, 1.0, 1.0]"

2. L_roll (rose, horizontal bracket):
   - One thin rose bracket spanning all 4 hop boxes
   - Label: "L_roll: chain consistency (pred input, λ_r: 0→4)"
   - Annotation: "step_weights = [0.5, 2.0, 1.5, 1.0]"

3. L_img (sage, local to H0):
   - One short dashed sage line from H0 downward
   - Label: "L_img: pixel recon (hop0 only, λ_i=0.04)"
   - Small text: "decode(z_D20_pred) vs x_GT_D20"

Keep this panel compact — it's a supervision schematic, not a full architecture repeat.

--------------------------------------------------
STRICT RULES:
--------------------------------------------------
- Do not portray V6 as succeeded or failed
- Exact values: pair_weight=15, λ_img=0.04, λ_roll 0→4, α 0→1
- No chartjunk, no box frames, no heavy grid
- No legend boxes
- No decorative elements
- Keep everything minimal
```

---

## 图 4：E1 误差预算分解（Transport vs Decoder）

**图的任务**：E2E gap 主要是 transport-related decoded-space discrepancy，decoder floor 很小。区分 PSNR-domain diagnostic 与 MSE-ratio，避免 overclaim。

**推荐尺寸**：`2400×1100 px`，宽高比约 `2.2:1`。

### Prompt v3.2 — GPT Image 2

```text
Create a minimal, publication-quality error budget figure for a top-tier ML conference (ICML / ICLR / NeurIPS). Two clean panels side by side. The visual must make one thing instantly clear: transport-related error is an order of magnitude larger than decoder floor.

CANVAS:
- 2400×1100 px, pure white background
- Two panels side by side (55% / 45% split)
- No background boxes, generous whitespace

COLOR PALETTE (same series):
- Teal (#2D6A6A): decoder ceiling
- Amber (#D4893F): transport-related gap
- Rose (#C17B8E): decoder floor
- Slate (#4B5563): E2E, axes, text

TYPOGRAPHY:
- Clean sans-serif, math italic
- No bold titles, small panel labels "(a)" "(b)"

--------------------------------------------------
PANEL (a): PSNR WATERFALL (D50→D20)
--------------------------------------------------

Vertical waterfall chart, three levels:

Top: teal bar at 46.64 dB
Label: "Decoder ceiling: 46.64 dB"

Large amber drop: 10.74 dB
Label: "Transport discrepancy"
Small annotation: "MSE ≈ 11.9×"

Tiny rose drop: 0.41 dB
Label: "Decoder floor"
Small annotation: "MSE ≈ 1.1×"

Bottom: slate bar at 35.49 dB
Label: "E2E: 35.49 dB"

The amber segment must be ~26× taller than rose — the size contrast IS the message.

Right side of panel (a): compact 4-group bar comparison
Groups: D20, D10, D4, NORMAL
Each group: tall teal bar (ceiling) + short slate bar (E2E)
Gap grows left-to-right. Small amber annotation: "gap grows along cascade"

--------------------------------------------------
PANEL (b): MSE RATIOS (log-scale)
--------------------------------------------------

Horizontal bar chart, log x-axis (1× to 50×)

Rows: "D50→D20", "→D10", "→D4", "→NORMAL"

Each row:
- Long amber bar: E2E/ceiling = 13.0×, 19.4×, 27.6×, 38.8×
- Tiny rose dot near 1×: E2E/pred-GT = 1.10×, 1.13×, 1.10×, 1.05×

Dashed vertical line at 1× (parity)
Amber bars grow dramatically top to bottom.
Rose dots barely visible — that IS the point.

--------------------------------------------------
BOTTOM FOOTNOTE
--------------------------------------------------
Small slate text spanning both panels:
"PSNR gaps are diagnostic proxies. MSE ratios are the physically meaningful quantification."

--------------------------------------------------
STRICT RULES:
--------------------------------------------------
- Do not write "96% of error energy"
- Do not write "pure velocity error"
- Use "transport discrepancy" and "decoder floor"
- Keep exact numbers as specified
- No legend box, no decorative elements
- Amber-vs-rose scale contrast must be visually immediate
```

---

## 图 5：4-Hop Exposure Bias 与 Scheduled α-Mixing

**图的任务**：训练时 α-mixing vs 推理时 open-loop 的分布偏移，hop1 是 rollout 敏感性 sweet spot。

**推荐尺寸**：`2600×1100 px`，宽高比约 `2.4:1`。

### Prompt v3.2 — GPT Image 2

```text
Create a minimal, publication-quality conceptual diagram for a top-tier ML conference (ICML / ICLR / NeurIPS). Two rows showing training vs inference in a 4-hop latent cascade, plus a small sensitivity inset.

CANVAS:
- 2600×1100 px, pure white background
- Left 70%: two horizontal rows (training / inference)
- Right 30%: compact sensitivity plot
- Generous spacing, no background panels

COLOR PALETTE (same series):
- Teal (#2D6A6A): GT-associated
- Amber (#D4893F): prediction-associated
- Rose (#C17B8E): error halos
- Sage (#7B9E6B): sensitivity sweet spot
- Slate (#4B5563): text, arrows

TYPOGRAPHY:
- Clean sans-serif, math italic
- Row titles small and subtle

--------------------------------------------------
TOP ROW: TRAINING (scheduled mixing)
--------------------------------------------------
Title: "Training: scheduled α-mixing"

Five state nodes left-to-right:
- z_D50: fully teal circle
- z_D20: bicolor circle (mostly teal, small amber slice)
- z_D10: bicolor (roughly half-half)
- z_D4: bicolor (mostly amber, small teal slice)
- z_NORMAL: fully amber circle

Between nodes: small amber card "f_θ"
Before each mixed node: tiny diamond "α"
- solid teal input "GT"
- dashed amber input "pred"

The bicolor circles are the KEY visual — they must be clean pie-chart style divisions, not messy gradients.

--------------------------------------------------
BOTTOM ROW: INFERENCE (open-loop)
--------------------------------------------------
Title: "Inference: open-loop rollout"

Same five positions:
- z_D50: teal (known input)
- z_D20, z_D10, z_D4, z_NORMAL: all fully amber

Error halos (soft rose concentric rings):
- z_D20: barely visible (2px ring)
- z_D10: small (4px)
- z_D4: medium (7px)
- z_NORMAL: large (11px)

Same f_θ cards between nodes.
NO α diamonds in inference row.

Vertical dashed rose line between training z_D20 and inference z_D20:
"distribution shift begins"

--------------------------------------------------
RIGHT INSET: HOP SENSITIVITY
--------------------------------------------------
Small clean plot:
X-axis: "hop" (0, 1, 2, 3)

Three curves:
1. Amber dashed "ExpoGap" — increasing
2. Teal solid "v_std" — decreasing
3. Sage filled area "ExpoGap × v_std" — peaks at hop 1

Sage arrow → hop 1 peak: "sweet spot"
Below plot: "step_weights: [0.5, 2.0, 1.5, 1.0]"

--------------------------------------------------
STRICT RULES:
--------------------------------------------------
- No GT input after z_D50 in inference row
- No pixel images — this is about latent states only
- Bicolor circles must be clean, not blurred
- Error halos subtle but visible
- No legend box, no decorative elements
- Keep everything minimal
```

---

## 6. 推荐生成流程

1. 先生成图 1，用它锁定配色、线宽、字体密度和 PET thumbnail 风格。
2. 图 2 紧跟图 1 生成，检查模块名字和实现路径是否完全正确。
3. 图 4 优先于图 3 / 图 5，因为它是 reviewer 最容易质疑的证据边界图。
4. 图 3 生成后重点检查：`λ_roll: 0 → 4.0`、`pair_weight = 15`、`λ_img = 0.04`、`50K-75K gate` 是否准确。
5. 图 5 生成后重点检查：training row 有 α-mixing，inference row 没有 GT mixing。

## 7. 质量审查清单（每张图都过一遍）

| 检查项 | 通过标准 |
|---|---|
| 文本 | 无拼写错误、无伪文字、公式符号完整 |
| 箭头 | 每条箭头 source/target 清楚，无反向箭头 |
| 科学边界 | 没有把诊断性结论画成严格因果证明 |
| 视觉 | 白底、协调色、无 3D/发光/彩虹/重阴影 |
| 缩放 | 缩到论文双栏宽度仍能读主要标签 |
| 一致性 | 五张图共享同一色板和模块语义 |

## 8. 如果 GPT Image 2 文字仍不稳定

优先重新生成，而不是接受错误文字。若三次仍不稳定，使用同一 prompt 生成“结构干净、文字较少”的版本，然后在 Illustrator / Inkscape / Figma 中叠加 LaTeX 或矢量文字。最终投稿图必须以文字正确为第一优先级；漂亮但拼错的图不能进入 ACL RR / ICLR / ICML 材料。
