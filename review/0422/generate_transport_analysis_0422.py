#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


TIMEPOINTS = ["D50", "D20", "D10", "D4", "NORMAL"]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _find_line(path: Path, needle: str) -> int:
    lines = _read_text(path).splitlines()
    for i, line in enumerate(lines, start=1):
        if needle in line:
            return i
    return -1


def _parse_d1_best_psnr(path: Path) -> Dict[str, float]:
    text = _read_text(path)
    out: Dict[str, float] = {}
    pattern = re.compile(r"\| PSNR (D50|D20|D10|D4|NORMAL) \|\s*([0-9.]+)\s*\|")
    for m in pattern.finditer(text):
        tp = m.group(1)
        out[tp] = float(m.group(2))
    missing = [tp for tp in TIMEPOINTS if tp not in out]
    if missing:
        raise RuntimeError(f"Failed to parse D1 best PSNR for: {missing}")
    return out


def _parse_ceiling(path: Path) -> Dict[str, float]:
    payload = json.loads(_read_text(path))
    summary = payload.get("summary_psnr_clip3", {})
    out: Dict[str, float] = {}
    for tp in TIMEPOINTS:
        if tp not in summary:
            raise RuntimeError(f"Missing {tp} in ceiling summary json")
        out[tp] = float(summary[tp]["mean"])
    return out


def _extract_config_scalar(path: Path, key: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*:\s*(.+?)\s*$")
    for line in _read_text(path).splitlines():
        m = pattern.match(line)
        if m:
            return m.group(1).strip()
    return "NOT_FOUND"


def _git_head(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
        return out
    except Exception:
        return "UNKNOWN"


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def _ref(path: Path, line: int, root: Path) -> str:
    if line > 0:
        return f"{_rel(path, root)}:{line}"
    return _rel(path, root)


def _render_table(rows: List[Tuple[str, float, float, float]]) -> str:
    lines = [
        "| Timepoint | Ceiling PSNR | D1 best PSNR | Gap (Ceiling - D1) |",
        "|---|---:|---:|---:|",
    ]
    for tp, c, d1, g in rows:
        lines.append(f"| {tp} | {c:.6f} | {d1:.6f} | {g:.6f} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate automatic 0422 transport-bottleneck analysis from code + artifacts."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output markdown path. Default: review/0422/auto_transport_analysis_<date>.md",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    today = dt.date.today().strftime("%Y%m%d")
    out_path = (
        args.out.resolve()
        if args.out is not None
        else repo_root / "review" / "0422" / f"auto_transport_analysis_{today}.md"
    )

    ceiling_json = repo_root / "review" / "0422" / "exp" / "gt_latent_decoder_ceiling_clip3_val_summary.json"
    d1_report = repo_root / "review" / "0421" / "Server" / "d1_experiment_report_20260421.md"
    v2_plan = repo_root / "review" / "0422" / "exp" / "schemec_v2_experiment_plan.md"
    v2_cfg = repo_root / "configs" / "pet_flow" / "pet_flow_first_hop_224_50k_schemec_v2.yaml"
    train_py = repo_root / "train_first_hop.py"
    ema_py = repo_root / "pet_lr" / "ema.py"

    ceiling = _parse_ceiling(ceiling_json)
    d1 = _parse_d1_best_psnr(d1_report)
    rows: List[Tuple[str, float, float, float]] = []
    for tp in TIMEPOINTS:
        c = ceiling[tp]
        b = d1[tp]
        rows.append((tp, c, b, c - b))

    tail_gaps = [g for tp, _, _, g in rows if tp != "D50"]
    tail_gap_mean = sum(tail_gaps) / len(tail_gaps)

    wd = _extract_config_scalar(v2_cfg, "first_hop_weight_decay")
    lam_init = _extract_config_scalar(v2_cfg, "lambda_hop_init")
    head_init = _extract_config_scalar(v2_cfg, "hop_residual_last_init_std")
    val_mode = _extract_config_scalar(v2_cfg, "val_window_mode")
    ema_enabled = _extract_config_scalar(v2_cfg, "enabled")
    ema_decay = _extract_config_scalar(v2_cfg, "decay")

    ln_train_wd = _find_line(train_py, "first_hop_weight_decay =")
    ln_train_group = _find_line(train_py, '"weight_decay": first_hop_weight_decay')
    ln_train_val_mode = _find_line(train_py, 'val_window_mode", "rolling"')
    ln_train_ema = _find_line(train_py, "ema_cfg = cfg.get(\"ema\", {})")
    ln_train_ema_ctx = _find_line(train_py, "ema.average_parameters()")
    ln_cfg_wd = _find_line(v2_cfg, "first_hop_weight_decay:")
    ln_cfg_lam = _find_line(v2_cfg, "lambda_hop_init:")
    ln_cfg_head = _find_line(v2_cfg, "hop_residual_last_init_std:")
    ln_cfg_val = _find_line(v2_cfg, "val_window_mode:")
    ln_cfg_ema = _find_line(v2_cfg, "ema:")
    ln_ema_class = _find_line(ema_py, "class EMA")
    ln_plan_assert = _find_line(v2_plan, "frozen decoder 完全不是瓶颈")
    ln_plan_hard_gate = _find_line(v2_plan, "val_select_score **< 0.000500**")

    strongest_claim = "支持 transport 是主瓶颈（高置信）" if tail_gap_mean > 8.0 else "主瓶颈结论证据不足"
    decoder_claim = (
        "支持“decoder 非主导瓶颈（限 GT latent/on-manifold 条件）”"
        if rows[0][3] > 2.0 and tail_gap_mean > 8.0
        else "decoder 结论需保守表述"
    )

    lines: List[str] = []
    lines.append("# 0422 自动化分析报告（基于代码与产物）")
    lines.append("")
    lines.append(f"- 生成时间: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Git HEAD: `{_git_head(repo_root)}`")
    lines.append(f"- 脚本: `{_rel(Path(__file__).resolve(), repo_root)}`")
    lines.append("")
    lines.append("## 1) 关键数值对照（clip3, full-val）")
    lines.append("")
    lines.append(_render_table(rows))
    lines.append("")
    lines.append(f"- tail 平均 gap（D20/D10/D4/NORMAL）: `{tail_gap_mean:.6f} dB`")
    lines.append("")
    lines.append("## 2) 代码/配置证据检查")
    lines.append("")
    lines.append("| 检查项 | 值/状态 | 证据 |")
    lines.append("|---|---|---|")
    lines.append(f"| first_hop_weight_decay | `{wd}` | `{_ref(v2_cfg, ln_cfg_wd, repo_root)}`, `{_ref(train_py, ln_train_wd, repo_root)}`, `{_ref(train_py, ln_train_group, repo_root)}` |")
    lines.append(f"| lambda_hop_init | `{lam_init}` | `{_ref(v2_cfg, ln_cfg_lam, repo_root)}` |")
    lines.append(f"| hop_residual_last_init_std | `{head_init}` | `{_ref(v2_cfg, ln_cfg_head, repo_root)}` |")
    lines.append(f"| val_window_mode | `{val_mode}` | `{_ref(v2_cfg, ln_cfg_val, repo_root)}`, `{_ref(train_py, ln_train_val_mode, repo_root)}` |")
    lines.append(f"| EMA enabled/decay | `enabled={ema_enabled}, decay={ema_decay}` | `{_ref(v2_cfg, ln_cfg_ema, repo_root)}`, `{_ref(train_py, ln_train_ema, repo_root)}`, `{_ref(train_py, ln_train_ema_ctx, repo_root)}`, `{_ref(ema_py, ln_ema_class, repo_root)}` |")
    lines.append("")
    lines.append("## 3) 自动判读（根据当前证据）")
    lines.append("")
    lines.append(f"- 结论A: **{strongest_claim}**。")
    lines.append(f"- 结论B: **{decoder_claim}**。")
    lines.append("- 结论C: v2 修复方向总体合理，但当前是 bundle fix（多项同时改动），不能直接归因到单一因素。")
    lines.append("")
    lines.append("## 4) 需要保守处理的表述")
    lines.append("")
    lines.append("- `decoder 完全不是瓶颈` 建议改为：`当前证据支持 decoder 非主导瓶颈（在 GT latent/on-manifold 条件下）`。")
    lines.append(f"  - 文档位置: `{_ref(v2_plan, ln_plan_assert, repo_root)}`")
    lines.append("- `val_select_score < 0.000500` 仅可作监控阈值，不建议作为唯一 go/no-go 标准。")
    lines.append(f"  - 文档位置: `{_ref(v2_plan, ln_plan_hard_gate, repo_root)}`")
    lines.append("- `旧实验全部无效` 建议降级为：`旧实验机制性结论需重审`。")
    lines.append("")
    lines.append("## 5) 最小补强动作（1-2天）")
    lines.append("")
    lines.append("1. 做 branch counterfactual eval：`g_pix=0`、`lambda_hop=0`、`both=0`。")
    lines.append("2. 做 latent perturbation sweep（`z_gt -> z_pred`），验证 decoder 对 off-manifold 的脆弱性。")
    lines.append("3. checkpoint 选择用 fixed full-val/fixed subset rerank，避免仅依赖 rolling-window 分数。")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote report: {out_path}")


if __name__ == "__main__":
    main()
