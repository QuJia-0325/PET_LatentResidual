#!/usr/bin/env python3
"""Golden-trace tests for `_metrics_compat.get_row_step` + the three callers
that consume metrics.jsonl rows.

Run from repo root:
    python review/0502/scripts/test_metrics_compat.py

Or:
    cd review/0502/scripts && python test_metrics_compat.py

Two-fixture coverage (per F8 + Phase-1 review Q-C):

  fixture A: review/0503/operator/A_sanity_metrics_tail5_schema_reference.jsonl
             5 rows, late-window steps [18400, 20000], minimal smoke test.

  fixture B: review/0503/operator/B_sanity_metrics_val50_schema_reference.jsonl
             46 rows, early/mid window. Phase-1 review Q-C requires that
             we use a refresh-stable prefix (steps in [400, 18000], i.e.
             rows 1..45) so when the operator refreshes the file from 46
             rows to ~50 rows on B's completion, this test continues to pass
             (the added rows at steps > 18000 are out of window).

This test is self-contained and uses only stdlib. It exercises:

  T1: get_row_step on `step`-schema row → returns int
  T2: get_row_step on `global_step`-schema row → returns int (legacy)
  T3: get_row_step on row with neither key → returns None
  T4: get_row_step on row with non-castable value → returns None
  T5: get_row_step prefers `step` when both present
  T6: A-fixture (5 rows) loads, all have val_select_score + val_chain_normal_mse
  T7: A-fixture step values match operator-published values
  T8: B-fixture (46 rows) loads under refresh-stable [400, 18000] prefix
  T9: B-fixture row 45 (step=18000) is the inclusive upper bound
  T10: NaN values in `*_raw_mse` fields are loaded without error
  T11: lock_effect_size_threshold.compute_paired_cv on B-fixture prefix
       window [400, 18000] yields N=45, mean>0, std>0, cv finite
  T12: paired_diff_judge.index_by_step on B-fixture yields exactly 46 entries
       (the function uses no window filter; fixture has 46 rows total)

Exit 0 = all pass. Non-zero = first failed test name + reason on stderr.
"""

from __future__ import annotations

import json
import math
import os
import sys
import unittest
from pathlib import Path

# Make `_metrics_compat` and the three callers importable when run from
# either repo root or scripts dir.
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

REPO_ROOT = SCRIPTS_DIR.parent.parent.parent  # …/PET_LatentResidual
A_FIXTURE = REPO_ROOT / "review/0503/operator/A_sanity_metrics_tail5_schema_reference.jsonl"
B_FIXTURE = REPO_ROOT / "review/0503/operator/B_sanity_metrics_val50_schema_reference.jsonl"

# Refresh-stable prefix per Phase-1 review Q-C. The operator pledged to refresh
# B-fixture to ~50 rows when sanity-B reaches 20K; rows added at step > 18000
# must not affect this test.
B_PREFIX_STEP_MIN = 400
B_PREFIX_STEP_MAX = 18000


# ---------- helpers ---------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    raw = path.read_bytes()
    for line in raw.splitlines():
        s = line.decode("utf-8", errors="replace").strip()
        if not s:
            continue
        rows.append(json.loads(s))  # NaN literals OK — see _metrics_compat docstring
    return rows


# ---------- tests -----------------------------------------------------------

class TestMetricsCompat(unittest.TestCase):
    """T1–T5: unit tests on the `get_row_step` helper itself."""

    def setUp(self):
        from _metrics_compat import get_row_step
        self.get_row_step = get_row_step

    def test_T1_step_schema(self):
        self.assertEqual(self.get_row_step({"step": 400, "x": 1.0}), 400)

    def test_T2_global_step_legacy(self):
        self.assertEqual(self.get_row_step({"global_step": 60000, "x": 1.0}), 60000)

    def test_T3_no_step_key(self):
        self.assertIsNone(self.get_row_step({"x": 1.0, "y": 2.0}))

    def test_T4_non_castable(self):
        self.assertIsNone(self.get_row_step({"step": None}))
        self.assertIsNone(self.get_row_step({"step": "not_an_int"}))
        self.assertIsNone(self.get_row_step({"global_step": [1, 2, 3]}))

    def test_T5_step_wins_over_global_step(self):
        # When both keys are present (defensive — should not happen in
        # production), `step` wins per `_metrics_compat._STEP_KEYS_PREFERRED_FIRST`.
        self.assertEqual(self.get_row_step({"step": 100, "global_step": 999}), 100)


class TestAFixture(unittest.TestCase):
    """T6–T7: A_sanity_metrics_tail5 (5 rows, late-window)."""

    @classmethod
    def setUpClass(cls):
        if not A_FIXTURE.exists():
            raise unittest.SkipTest(f"A_fixture missing: {A_FIXTURE}")
        cls.rows = _load_jsonl(A_FIXTURE)

    def test_T6_five_rows_with_required_keys(self):
        from _metrics_compat import get_row_step
        self.assertEqual(len(self.rows), 5)
        for i, r in enumerate(self.rows):
            self.assertIsNotNone(get_row_step(r), f"row {i} has no usable step key")
            self.assertIn("val_select_score", r, f"row {i} missing val_select_score")
            self.assertIn("val_chain_normal_mse", r, f"row {i} missing val_chain_normal_mse")
            self.assertEqual(r.get("event"), "val", f"row {i} not a val event")

    def test_T7_step_values(self):
        from _metrics_compat import get_row_step
        steps = [get_row_step(r) for r in self.rows]
        # Operator published: tail-5 covers [18400, 18800, 19200, 19600, 20000]
        self.assertEqual(steps, [18400, 18800, 19200, 19600, 20000])


class TestBFixturePrefixStable(unittest.TestCase):
    """T8–T10: B_sanity_metrics_val50 (46 rows now; ~50 after refresh).

    All assertions use the refresh-stable prefix [400, 18000] so they remain
    valid when B-fixture grows."""

    @classmethod
    def setUpClass(cls):
        if not B_FIXTURE.exists():
            raise unittest.SkipTest(f"B_fixture missing: {B_FIXTURE}")
        cls.rows = _load_jsonl(B_FIXTURE)

    def test_T8_prefix_loads(self):
        from _metrics_compat import get_row_step
        prefix_rows = [
            r for r in self.rows
            if (s := get_row_step(r)) is not None
            and B_PREFIX_STEP_MIN <= s <= B_PREFIX_STEP_MAX
        ]
        # 400, 800, ..., 18000 = 45 rows under inclusive bounds.
        self.assertEqual(len(prefix_rows), 45,
                         f"expected 45 rows in prefix [{B_PREFIX_STEP_MIN}, {B_PREFIX_STEP_MAX}], "
                         f"got {len(prefix_rows)}")
        for r in prefix_rows:
            self.assertIn("val_select_score", r)
            self.assertIn("val_chain_normal_mse", r)

    def test_T9_inclusive_upper_bound(self):
        from _metrics_compat import get_row_step
        steps_in_prefix = sorted(
            s for r in self.rows
            if (s := get_row_step(r)) is not None
            and B_PREFIX_STEP_MIN <= s <= B_PREFIX_STEP_MAX
        )
        # First step is 400; last step under inclusive [400, 18000] is exactly 18000.
        self.assertEqual(steps_in_prefix[0], 400)
        self.assertEqual(steps_in_prefix[-1], 18000)

    def test_T10_nan_tolerated_in_raw_mse(self):
        # The trainer emits `val_chain_normal_raw_mse: NaN` (and similar) when
        # raw-mse is disabled. Python json.loads accepts NaN; orjson does not.
        # This test asserts that loading + iterating these rows does NOT raise.
        nan_field_present = False
        for r in self.rows:
            v = r.get("val_chain_normal_raw_mse")
            if isinstance(v, float) and math.isnan(v):
                nan_field_present = True
                break
        self.assertTrue(nan_field_present,
                        "fixture lacks expected NaN field; if upstream changed, "
                        "update the test rather than switching the JSON parser")


class TestLockComputePairedCV(unittest.TestCase):
    """T11: lock_effect_size_threshold.compute_paired_cv on B-fixture prefix."""

    @classmethod
    def setUpClass(cls):
        if not B_FIXTURE.exists():
            raise unittest.SkipTest(f"B_fixture missing: {B_FIXTURE}")

    def test_T11_compute_paired_cv_on_b_prefix(self):
        # Import the lock script as a module. It also imports `_metrics_compat`
        # via the same SCRIPTS_DIR path we inserted at module-load time.
        import lock_effect_size_threshold as lock
        cv, mean, std, n, series, sha = lock.compute_paired_cv(
            B_FIXTURE, B_PREFIX_STEP_MIN, B_PREFIX_STEP_MAX
        )
        # 45 rows in [400, 18000] (steps 400, 800, …, 18000)
        self.assertEqual(n, 45,
                         f"compute_paired_cv saw N={n}, expected 45 over "
                         f"refresh-stable prefix [{B_PREFIX_STEP_MIN}, "
                         f"{B_PREFIX_STEP_MAX}]")
        self.assertGreater(mean, 0.0)
        self.assertGreater(std, 0.0)
        self.assertTrue(math.isfinite(cv))
        self.assertEqual(len(series), 45)
        # SHA must be a 64-char hex digest
        self.assertEqual(len(sha), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in sha))


class TestPairedDiffIndex(unittest.TestCase):
    """T12: paired_diff_judge.index_by_step on B-fixture prefix.

    Use the refresh-stable subset (steps ≤ B_PREFIX_STEP_MAX) instead of the
    full file so this assertion remains valid after the B-fixture refresh."""

    @classmethod
    def setUpClass(cls):
        if not B_FIXTURE.exists():
            raise unittest.SkipTest(f"B_fixture missing: {B_FIXTURE}")

    def test_T12_index_by_step(self):
        import paired_diff_judge as pdj
        rows = _load_jsonl(B_FIXTURE)
        # Restrict to the refresh-stable prefix manually before indexing.
        from _metrics_compat import get_row_step
        prefix_rows = [
            r for r in rows
            if (s := get_row_step(r)) is not None
            and B_PREFIX_STEP_MIN <= s <= B_PREFIX_STEP_MAX
        ]
        idx = pdj.index_by_step(prefix_rows, "val_chain_normal_mse")
        self.assertEqual(len(idx), 45,
                         f"index_by_step yielded {len(idx)} keyed rows, expected 45")
        self.assertIn(400, idx)
        self.assertIn(18000, idx)
        for step, val in idx.items():
            self.assertGreater(val, 0.0,
                               f"step {step} has non-positive metric: {val}")


# ---------- main ------------------------------------------------------------

if __name__ == "__main__":
    # Use TextTestRunner with verbosity=2 so each Tname is printed.
    unittest.main(verbosity=2)
