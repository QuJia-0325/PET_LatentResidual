"""Schema-compatibility helpers for review/0502 metrics.jsonl rows.

Background: the trainer (`train_first_hop.py`) writes one JSON-lines row per
val-eval into `metrics.jsonl`. The historical tooling under `review/0502/scripts/`
was written assuming the row carries a `global_step` key. Operator inspection
of an actually-running v6.0 / sigma-normalize run (round-2.5 reply, file
`OPERATOR_REPLY_pre_C1_20260503.md`) confirmed the trainer-emitted rows use:

    {"event": "val", "step": <int>, "val_chain_normal_mse": ..., "val_select_score": ..., ...}

i.e. the integer step is keyed `step`, not `global_step`. This module provides
a single source of truth for reading the step out of a metrics row, tolerant
of either naming.

Locked semantics (R1a §6.6.1 will reference this module by SHA):
  - Prefer `step` (current trainer output).
  - Fall back to `global_step` (legacy / stale fixtures).
  - Return None if neither key is present or the value cannot be cast to int
    (callers must skip the row, not treat it as step=0).

This module deliberately does NO float-coercion of metric values, only step
parsing — metric-value semantics are the caller's responsibility.

NaN tolerance note: the trainer emits `NaN` literals for `*_raw_mse` fields
when the raw-mse computation is disabled. Python's stdlib `json.loads` accepts
`NaN` by default. Do NOT switch any caller of this helper to `orjson` (or any
strict-JSON parser) without first sanitizing the upstream rows — `orjson`
rejects `NaN` and would break loading both fixtures
(`A_sanity_metrics_tail5_schema_reference.jsonl`,
`B_sanity_metrics_val50_schema_reference.jsonl`) and any current
production `metrics.jsonl`.

Used by:
  - lock_effect_size_threshold.py   (Step 3 paired_CV_A computation)
  - paired_diff_judge.py            (paired diff over shared steps)
  - select_best_ckpt_smoothed.py    (K-side smoothing window selection)
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

# Order matters: the current trainer key `step` is checked first; `global_step`
# is the legacy fallback. If both are present (defensive — should not happen
# in production rows) the current key wins.
_STEP_KEYS_PREFERRED_FIRST: tuple[str, ...] = ("step", "global_step")


def get_row_step(row: Mapping[str, Any]) -> Optional[int]:
    """Return the integer step for a metrics.jsonl row, or None if absent.

    Accepts either `step` (current trainer) or `global_step` (legacy / older
    fixtures). Returns None if neither key is present, or if the value cannot
    be cast to int (e.g. None, missing, malformed). Callers MUST treat None
    as "skip this row" — never as step=0.

    Args:
        row: a single decoded JSON object from a metrics.jsonl line.

    Returns:
        int step value, or None if the row has no usable step key.

    A3 caveat (Lane A peer review, May 4 2026): non-integral float values
    are silently truncated by ``int(v)`` (e.g. ``step=4.5`` → ``4``,
    ``step=-1.9`` → ``-1`` per Python's truncate-toward-zero rule). The
    trainer never emits non-integral steps in production; if a row has
    one this should be treated as a SCHEMA error by the caller. Do not
    depend on the truncation behavior for any control flow.
    """
    for key in _STEP_KEYS_PREFERRED_FIRST:
        if key not in row:
            continue
        v = row[key]
        if v is None:
            continue
        try:
            return int(v)
        except (TypeError, ValueError):
            continue
    return None


def has_step(row: Mapping[str, Any]) -> bool:
    """Convenience: True iff `get_row_step(row)` would return non-None."""
    return get_row_step(row) is not None
