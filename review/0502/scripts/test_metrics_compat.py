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
  T13: compute_paired_cv emits a stderr warning when in-window rows are
       skipped due to missing/invalid metric value (A5 hardening, Lane A HIGH).
  T14: negative-control — compute_paired_cv stays silent on clean B-fixture.

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

    def test_T13_compute_paired_cv_warns_on_skipped_rows(self):
        """A5 hardening (Lane A HIGH): rows with parseable `step` in window
        but missing `val_select_score` (or with a non-numeric value) must
        emit a stderr warning so trainer partial-writes don't vanish
        silently from the locked sample.
        """
        import io
        import tempfile
        import contextlib
        import lock_effect_size_threshold as lock

        # Build a synthetic metrics file: 5 valid rows + 2 in-window rows
        # missing the metric key + 1 in-window row with non-numeric value +
        # 1 out-of-window row missing the key (must NOT count toward warn).
        good_rows = [
            {"step": s, "event": "val", "val_select_score": 0.50}
            for s in (400, 800, 1200, 1600, 2000)
        ]
        bad_no_key = [
            {"step": 2400, "event": "val"},   # in-window, no val_select_score
            {"step": 2800, "event": "val"},   # in-window, no val_select_score
        ]
        bad_value = [
            {"step": 3200, "event": "val", "val_select_score": "not_a_number"},
        ]
        out_of_window = [
            {"step": 99999, "event": "val"},  # outside [400, 18000]
        ]
        all_rows = good_rows + bad_no_key + bad_value + out_of_window

        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False
        ) as f:
            for r in all_rows:
                f.write(json.dumps(r) + "\n")
            tmp_path = Path(f.name)
        try:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                cv, mean, std, n, series, sha = lock.compute_paired_cv(
                    tmp_path, 400, 18000
                )
            warning = stderr.getvalue()
            # The 5 good rows survive, the 2-no-key + 1-bad-value get warned.
            self.assertEqual(n, 5,
                             f"expected 5 surviving rows, got {n}")
            self.assertIn("compute_paired_cv", warning)
            self.assertIn("skipped", warning)
            # The two no-key in-window rows are reported.
            self.assertIn("2 in-window row(s) missing", warning)
            # The single bad-value row is reported separately.
            self.assertIn("1 row(s) with non-numeric", warning)
            # The out-of-window row must NOT be counted.
            self.assertNotIn("3 in-window", warning)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_T14_compute_paired_cv_no_warning_on_clean_data(self):
        """Negative-control for T13: when all in-window rows are clean,
        compute_paired_cv emits NO warning to stderr."""
        import io
        import contextlib
        import lock_effect_size_threshold as lock
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            cv, mean, std, n, series, sha = lock.compute_paired_cv(
                B_FIXTURE, B_PREFIX_STEP_MIN, B_PREFIX_STEP_MAX
            )
        self.assertEqual(n, 45)
        # No "[warn]" line should be emitted on clean B-fixture.
        self.assertNotIn("[warn] compute_paired_cv",
                         stderr.getvalue(),
                         "B-fixture is clean; warning is a false positive")

    def test_T15_compute_paired_stats_from_rows_matches_path_form(self):
        """A2 architectural fix (Lane A peer review): the path-form
        wrapper and the row-form helper must produce the same statistics
        when fed equivalent inputs. This guards the refactor: if a future
        change drifts the two paths apart, lock-script integration vs
        unit-test calls would silently disagree.
        """
        import lock_effect_size_threshold as lock

        rows, sha = lock.load_metrics_with_hash(B_FIXTURE)
        cv_p, mean_p, std_p, n_p, series_p, sha_p = lock.compute_paired_cv(
            B_FIXTURE, B_PREFIX_STEP_MIN, B_PREFIX_STEP_MAX
        )
        cv_r, mean_r, std_r, n_r, series_r = (
            lock.compute_paired_stats_from_rows(
                rows, B_PREFIX_STEP_MIN, B_PREFIX_STEP_MAX
            )
        )
        self.assertEqual(n_p, n_r, "row-form N differs from path-form N")
        self.assertEqual(series_p, series_r,
                         "row-form series differs from path-form series")
        self.assertAlmostEqual(cv_p, cv_r, places=12)
        self.assertAlmostEqual(mean_p, mean_r, places=12)
        self.assertAlmostEqual(std_p, std_r, places=12)
        # Path form additionally returns the SHA of the same byte read.
        self.assertEqual(sha_p, sha)

    def test_T16_main_reads_metrics_file_exactly_once(self):
        """A2 architectural fix: the lock-script main() previously read
        metrics_a THREE times (compute_paired_cv + Guard 5 recheck +
        observation table). The refactor collapses these into one read
        tied to the SHA in EFFECT_SIZE_LOCKED.md. We verify by patching
        ``load_metrics_with_hash`` with a counter and running main() to
        the no-commit success path on a synthetic fixture.
        """
        import io
        import contextlib
        import tempfile
        import unittest.mock
        import lock_effect_size_threshold as lock

        # Synthesise a minimal valid lock-input set.
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            metrics = tdp / "metrics.jsonl"
            metrics.write_text("\n".join(
                json.dumps({
                    "step": s, "event": "val",
                    "val_select_score": 0.50,
                    "val_chain_normal_mse": 0.001,
                })
                # Step window must match LOCKED_RECOMMENDED_WINDOW = (40000, 60000)
                # so we don't trip the deviation-note guard. 51 rows → ≥5.
                for s in range(40000, 60001, 400)
            ) + "\n")
            cfg = tdp / "config.yaml"
            cfg.write_text("training:\n  best_metric: val_multi_objective\n")
            output = tdp / "EFFECT_SIZE_LOCKED.md"

            counter = {"calls": 0}
            real_load = lock.load_metrics_with_hash

            def counting_load(p):
                counter["calls"] += 1
                return real_load(p)

            with unittest.mock.patch.object(
                lock, "load_metrics_with_hash", side_effect=counting_load
            ):
                # Build a fake argv and invoke main() under --no-commit so
                # we don't touch git.
                argv = [
                    "lock_effect_size_threshold.py",
                    "--metrics-a", str(metrics),
                    "--config-a", str(cfg),
                    "--output", str(output),
                    "--no-commit",
                    "--allow-dirty",
                ]
                with unittest.mock.patch.object(sys, "argv", argv):
                    # find_repo_root walks up looking for `.git`; create a
                    # fake one in our tmpdir so it stops there.
                    (tdp / ".git").mkdir()
                    # Suppress all stdout/stderr noise from main().
                    with contextlib.redirect_stdout(io.StringIO()), \
                         contextlib.redirect_stderr(io.StringIO()):
                        rc = lock.main()

            self.assertEqual(rc, 0, "main() did not reach --no-commit success")
            self.assertEqual(
                counter["calls"], 1,
                f"metrics_a was loaded {counter['calls']} times; expected 1 "
                f"(A2 architectural single-read invariant broken)"
            )

    def test_T17_train_rows_in_window_do_not_warn(self):
        """F2 hardening (Round-5 operator review, May 4 2026): in-window
        rows with ``event="train"`` (and other non-val schemas) NEVER
        carry ``val_select_score``. Counting them as ``skipped_no_metric``
        produces a false-positive ``[warn] compute_paired_cv: skipped N
        in-window row(s) missing 'val_select_score'`` line on perfectly
        healthy metrics files (operator observed 33 spurious skips on
        ``A_sanity`` tail-window). The warning must only fire on
        val-like rows (event=="val" or event absent for legacy
        fixtures).
        """
        import io
        import tempfile
        import contextlib
        import lock_effect_size_threshold as lock

        # 5 healthy val rows (paired-CV minimum N).
        good_val = [
            {"step": s, "event": "val", "val_select_score": 0.50}
            for s in (400, 800, 1200, 1600, 2000)
        ]
        # 33 in-window train rows that legitimately lack val_select_score.
        # These must NOT be counted in the skipped warning.
        train_rows = [
            {"step": s, "event": "train", "loss": 0.1}
            for s in range(500, 4500, 121)
        ]
        all_rows = good_val + train_rows

        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False
        ) as f:
            for r in all_rows:
                f.write(json.dumps(r) + "\n")
            tmp_path = Path(f.name)
        try:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                cv, mean, std, n, series, sha = lock.compute_paired_cv(
                    tmp_path, 400, 18000
                )
            self.assertEqual(n, 5, "the 5 val rows must form the series")
            warning = stderr.getvalue()
            self.assertNotIn(
                "[warn] compute_paired_cv", warning,
                "F2: train rows in window must not trigger the "
                "val_select_score-missing warning",
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_T18_val_rows_missing_metric_still_warn(self):
        """F2 negative-control: val-event rows that genuinely lack
        ``val_select_score`` (true trainer partial-write) MUST still
        trigger the warning. F2 only suppresses train-row noise.
        """
        import io
        import tempfile
        import contextlib
        import lock_effect_size_threshold as lock

        good = [
            {"step": s, "event": "val", "val_select_score": 0.50}
            for s in (400, 800, 1200, 1600, 2000)
        ]
        # Real partial-writes: explicitly val-event but no metric key.
        partial_val = [
            {"step": 2400, "event": "val"},
            {"step": 2800, "event": "val"},
        ]
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False
        ) as f:
            for r in good + partial_val:
                f.write(json.dumps(r) + "\n")
            tmp_path = Path(f.name)
        try:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                cv, mean, std, n, series, sha = lock.compute_paired_cv(
                    tmp_path, 400, 18000
                )
            self.assertEqual(n, 5)
            warning = stderr.getvalue()
            self.assertIn("[warn] compute_paired_cv", warning)
            self.assertIn("2 in-window row(s) missing", warning)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_T19_event_absent_treated_as_val_for_legacy_fixtures(self):
        """F2 backward-compat: rows with no ``event`` field default to
        val-like, because legacy fixtures (e.g. the operator's
        ``B_sanity_metrics_val50_schema_reference.jsonl``) have no
        explicit event tag but every row IS a val measurement.
        Otherwise F2 would silently drop legacy fixtures from the
        warning surface, hiding partial-write defects.
        """
        import io
        import tempfile
        import contextlib
        import lock_effect_size_threshold as lock

        good_legacy = [
            {"step": s, "val_select_score": 0.50}   # no "event" key
            for s in (400, 800, 1200, 1600, 2000)
        ]
        partial_legacy = [
            {"step": 2400},  # no event, no metric — must warn
        ]
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False
        ) as f:
            for r in good_legacy + partial_legacy:
                f.write(json.dumps(r) + "\n")
            tmp_path = Path(f.name)
        try:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                cv, mean, std, n, series, sha = lock.compute_paired_cv(
                    tmp_path, 400, 18000
                )
            self.assertEqual(n, 5)
            warning = stderr.getvalue()
            self.assertIn("1 in-window row(s) missing", warning)
        finally:
            tmp_path.unlink(missing_ok=True)


class TestGuard5StrictBestMetric(unittest.TestCase):
    """G3 hardening (Round-7 cross-AI peer review, May 4 2026): the
    Round-6 absorption deferred F5 to R1b on the rationale that
    tightening Guard 5 mid-`v0_pending_R1a` would break the
    locked-sample contract. Round-7 cross-AI review rejected that
    rationale because no locked sample exists yet, so tightening now
    cannot pollute any artifact.

    G3 contract: at lock time, ``training.best_metric`` MUST be
    ``val_multi_objective`` exactly. Any other value (None/missing,
    ``val_select_score``, an unknown key, etc.) aborts with exit 4
    unless ``--allow-dev-best-metric`` is passed.

    These tests drive ``main()`` end-to-end through the same minimal
    fixture pattern T16 uses (so the entire pre-Guard-5 pipeline
    runs); only the config YAML's ``best_metric`` value varies.
    """

    def _run_main(self, best_metric_value, *, allow_dev_flag=False):
        """Drive main() with a synthetic fixture whose config has the
        given ``best_metric`` (or no best_metric line at all if value
        is the sentinel string ``"<MISSING>"``). Returns
        (rc, stderr_text).
        """
        import io
        import contextlib
        import tempfile
        import unittest.mock
        import lock_effect_size_threshold as lock

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            metrics = tdp / "metrics.jsonl"
            metrics.write_text("\n".join(
                json.dumps({
                    "step": s, "event": "val",
                    "val_select_score": 0.50,
                    "val_chain_normal_mse": 0.001,
                })
                for s in range(40000, 60001, 400)
            ) + "\n")
            cfg = tdp / "config.yaml"
            if best_metric_value == "<MISSING>":
                cfg.write_text("training:\n  seed: 0\n")  # no best_metric key
            else:
                cfg.write_text(
                    f"training:\n  best_metric: {best_metric_value}\n"
                )
            output = tdp / "EFFECT_SIZE_LOCKED.md"
            (tdp / ".git").mkdir()

            argv = [
                "lock_effect_size_threshold.py",
                "--metrics-a", str(metrics),
                "--config-a", str(cfg),
                "--output", str(output),
                "--no-commit",
                "--allow-dirty",
            ]
            if allow_dev_flag:
                argv.append("--allow-dev-best-metric")

            stderr = io.StringIO()
            with unittest.mock.patch.object(sys, "argv", argv), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(stderr):
                rc = lock.main()
            return rc, stderr.getvalue()

    def test_G3a_canonical_value_passes(self):
        """Positive control: the canonical value
        ``val_multi_objective`` continues to pass under strict G3
        (and matches T16's pre-G3 baseline)."""
        rc, stderr = self._run_main("val_multi_objective")
        self.assertEqual(rc, 0,
                         f"canonical best_metric must pass; stderr={stderr!r}")

    def test_G3b_missing_best_metric_strict_aborts(self):
        """Strict abort: a config with no ``best_metric`` key at all
        used to silently pass (Round-6 allowed None). G3 rejects with
        exit 4 and a stderr message naming the actual value (None).
        """
        rc, stderr = self._run_main("<MISSING>")
        self.assertEqual(rc, 4)
        self.assertIn("guard 5 strict", stderr)
        self.assertIn("None", stderr)

    def test_G3c_val_select_score_strict_aborts(self):
        """Strict abort: ``val_select_score`` (config-side) was
        permitted by Round-6 but is NOT canonical. Locking against a
        run that selected on `val_select_score` directly (rather than
        on `val_multi_objective`) would record a different selection
        rule than reviewers expect."""
        rc, stderr = self._run_main("val_select_score")
        self.assertEqual(rc, 4)
        self.assertIn("guard 5 strict", stderr)
        self.assertIn("val_select_score", stderr)

    def test_G3d_unknown_key_strict_aborts(self):
        """Strict abort: any unknown best_metric value also aborts
        (this branch was already in Round-6, kept under strict)."""
        rc, stderr = self._run_main("some_random_metric")
        self.assertEqual(rc, 4)
        self.assertIn("guard 5 strict", stderr)
        self.assertIn("some_random_metric", stderr)

    def test_G3e_dev_flag_bypasses_with_warning(self):
        """Opt-out: passing ``--allow-dev-best-metric`` lets
        non-canonical values pass, BUT must emit a [warn] line
        surfacing the bypass and the actual value, so operators
        cannot accidentally lock under dev mode without seeing it.
        """
        rc, stderr = self._run_main(
            "val_select_score", allow_dev_flag=True,
        )
        self.assertEqual(rc, 0,
                         f"opt-out must succeed; stderr={stderr!r}")
        self.assertIn("[warn] guard 5 strict bypassed", stderr)
        self.assertIn("val_select_score", stderr)
        self.assertIn("R1a lock time", stderr,
                      "the warn must explicitly say 'do not use at "
                      "R1a lock time'")

    def test_G3f_dev_flag_with_canonical_does_not_warn(self):
        """Opt-out is harmless on canonical: even with
        ``--allow-dev-best-metric``, the canonical value still hits
        the equality branch FIRST, so no warn fires. (Otherwise
        anyone running with the dev flag would see noise on every
        production-shaped invocation.)"""
        rc, stderr = self._run_main(
            "val_multi_objective", allow_dev_flag=True,
        )
        self.assertEqual(rc, 0)
        self.assertNotIn("[warn] guard 5 strict bypassed", stderr,
                         "dev flag with canonical value must not warn")


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
