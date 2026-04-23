# Backbone-LR1 Full-Val Eval Report (Best vs Last)

Date: 2026-04-23
Branch: `foc_lite_hop0`

## 1. Runs executed (as requested)

- GPU1: `backbone_lr1 best.pt`
- GPU3: `backbone_lr1 last.pt`

Commands were launched in parallel via tmux.

## 2. Evaluation protocol

- Dataset: `val` full set (`n=7403`)
- Metric: clip3 PSNR (formula-equivalent fast implementation)
- Script: `review/0423/eval_single_first_hop_clip3_fast.py`
- Batch size: `10`

Note: we first attempted canonical `eval_first_hop_224_clip3.py`, but it was extremely slow for full-val in this environment; for completion and reproducibility, results below use the fast equivalent clip3 evaluator.

## 3. Results

| ckpt | D50 | D20 | D10 | D4 | NORMAL | transport_avg | all_avg | runtime_sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| backbone_lr1_best | 42.622379 | 35.558931 | 35.908639 | 36.471447 | 36.712880 | 36.162974 | 37.454855 | 572.0 |
| backbone_lr1_last | 42.622379 | 35.565742 | 35.926397 | 36.493110 | 36.767850 | 36.188275 | 37.475096 | 566.7 |

Delta (`last - best`):
- `transport_avg`: **+0.025300**
- `all_avg`: **+0.020240**
- D20: `+0.006811`
- D10: `+0.017758`
- D4: `+0.021662`
- NORMAL: `+0.054970`

## 4. Conclusion

- Under full-val clip3, **`last.pt` is better than `best.pt`** for `backbone_lr1`.
- This again indicates rolling-window in-training best selection can diverge from full-val ranking.
- For fair cross-scheme comparison, use full-val rerank outputs as primary evidence.

## 5. Artifacts and logs

Fast eval artifacts (tracked in repo):
- `review/0423/artifacts/backbone_lr1_best_fullval_clip3_fast.json`
- `review/0423/artifacts/backbone_lr1_best_fullval_clip3_fast.csv`
- `review/0423/artifacts/backbone_lr1_last_fullval_clip3_fast.json`
- `review/0423/artifacts/backbone_lr1_last_fullval_clip3_fast.csv`

Execution logs:
- `review/0423/logs_eval/backbone_lr1_best_fullval_fast_20260423.log`
- `review/0423/logs_eval/backbone_lr1_last_fullval_fast_20260423.log`

(Initial slow full-val attempts kept as run evidence)
- `review/0423/logs_eval/backbone_lr1_best_fullval_20260423.log`
- `review/0423/logs_eval/backbone_lr1_last_fullval_20260423.log`
