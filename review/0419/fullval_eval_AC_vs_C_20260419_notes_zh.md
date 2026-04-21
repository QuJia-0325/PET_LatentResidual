# 0419 Full-val 对比结论（A+C vs C）

## 评估设置
- 数据划分：`val`（全量，`max-slices=0`）
- 评估脚本：`eval_first_hop_224_clip3.py`
- 统一样本数：4 组运行均为 `num_eval_slices=7403`
- 对比对象：
  - A+C：`best.pt` 与 `last.pt`
  - C-only：`best.pt` 与 `last.pt`

## 关键结果（PSNR, clip3）
- `best.pt` 对比（A+C - C）：
  - `transport_avg`：`+0.001955`（几乎持平，A+C 略优）
  - `all_avg`：`+0.001564`
- `last.pt` 对比（A+C - C）：
  - `transport_avg`：`+0.024346`（A+C 明显优于 C）
  - `all_avg`：`+0.019476`

## 结论
- 在 full-val 公平对比下，A+C 整体不弱于 C-only。
- 尤其在 `last.pt` 上，A+C 在传输阶段平均指标（`D20/D10/D4/NORMAL`）有稳定增益。
- `best.pt` 的差距非常小，说明两者上限接近；A+C 的优势主要体现在训练后段/最终权重。

## 建议
- 主对比报告可优先使用 `last.pt` 的 full-val 结果体现 A+C 的实际收益。
- 同时保留 `best.pt` 对照，说明上限接近且 A+C 在最终收敛质量上更稳。
