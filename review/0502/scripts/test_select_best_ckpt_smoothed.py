#!/usr/bin/env python3
"""Tests for review/0502/scripts/select_best_ckpt_smoothed.py — C3 fix.

C3 (Round-5 operator review, May 4 2026): the trainer
(`train_first_hop.py`) saves files as `step_{step:06d}.pt` and
`best.pt` / `last.pt`, but the selector previously globbed only
`ckpt_step_*.pt` and `ckpt_last.pt`, making Method-D non-operational
on real trainer outputs. These tests guard the new dual-naming
discovery and the schema-tolerant ckpt-internals key.

The tests are torch-free (no real .pt loaded) — empty bytes suffice
because `list_saved_steps` only inspects filenames for the saved-grid
path. Tests for `--include-best-pt` / `--include-last-pt` would need
torch and are skipped unless torch is importable.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Allow `from select_best_ckpt_smoothed import ...` when running from repo
# root via `python -m unittest review.0502.scripts.test_select_best_ckpt`.
sys.path.insert(0, str(Path(__file__).parent))

from select_best_ckpt_smoothed import list_saved_steps


class TestListSavedStepsNewForm(unittest.TestCase):
    """C3: trainer-native naming (`step_NNNNNN.pt`)."""

    def test_new_form_zero_padded_six_digits(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "step_020000.pt").write_bytes(b"")
            (d / "step_040000.pt").write_bytes(b"")
            (d / "step_153600.pt").write_bytes(b"")
            steps, m = list_saved_steps(d, include_best=False, include_last=False)
            self.assertEqual(steps, [20000, 40000, 153600])
            self.assertEqual(m[20000].name, "step_020000.pt")
            self.assertEqual(m[153600].name, "step_153600.pt")

    def test_new_form_unpadded_step(self):
        # Defensive: not the trainer's actual format, but should still work
        # because we parse the integer suffix of the stem regardless of
        # zero-padding.
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "step_42.pt").write_bytes(b"")
            steps, m = list_saved_steps(d, include_best=False, include_last=False)
            self.assertEqual(steps, [42])
            self.assertEqual(m[42].name, "step_42.pt")


class TestListSavedStepsLegacyForm(unittest.TestCase):
    """C3 backward-compat: `ckpt_step_*.pt` still discovered."""

    def test_legacy_form_only(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "ckpt_step_10000.pt").write_bytes(b"")
            (d / "ckpt_step_20000.pt").write_bytes(b"")
            steps, m = list_saved_steps(d, include_best=False, include_last=False)
            self.assertEqual(steps, [10000, 20000])
            self.assertEqual(m[10000].name, "ckpt_step_10000.pt")


class TestListSavedStepsCohabitation(unittest.TestCase):
    """C3: a directory containing both naming forms is deduped by step."""

    def test_new_form_wins_on_tie(self):
        # If the same step exists under both names, prefer the new
        # (trainer-native) form because that is what training is
        # currently writing on the operator host.
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "step_020000.pt").write_bytes(b"")
            (d / "ckpt_step_20000.pt").write_bytes(b"")
            steps, m = list_saved_steps(d, include_best=False, include_last=False)
            self.assertEqual(steps, [20000])
            self.assertEqual(m[20000].name, "step_020000.pt")

    def test_disjoint_steps_under_both_forms(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "step_020000.pt").write_bytes(b"")
            (d / "ckpt_step_50000.pt").write_bytes(b"")
            steps, m = list_saved_steps(d, include_best=False, include_last=False)
            self.assertEqual(steps, [20000, 50000])
            self.assertEqual(m[20000].name, "step_020000.pt")
            self.assertEqual(m[50000].name, "ckpt_step_50000.pt")

    def test_collision_with_different_contents_warns(self):
        """G2 hardening (Round-7 cross-AI peer review, May 4 2026): a
        directory containing BOTH naming forms at the same step with
        DIFFERENT byte contents must (a) still resolve to the new-form
        path (precedence preserved), and (b) emit a `[warn]` line on
        stdout naming both files so the operator can `sha256sum` them.

        The Round-6 ``test_new_form_wins_on_tie`` only verified path
        precedence with empty-byte fixtures; it could not catch a real
        crash-recovery scenario where the two files disagree on
        ckpt contents. G2 closes that observability gap.
        """
        import io
        import contextlib
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "step_020000.pt").write_bytes(b"new-form-content-AAA")
            (d / "ckpt_step_20000.pt").write_bytes(b"legacy-content-ZZZ")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                steps, m = list_saved_steps(
                    d, include_best=False, include_last=False,
                )
            # (a) precedence preserved: new-form wins.
            self.assertEqual(steps, [20000])
            self.assertEqual(m[20000].name, "step_020000.pt")
            # (b) collision warning: must name BOTH files so operator
            # can manually verify (sha256sum etc).
            warning = stdout.getvalue()
            self.assertIn("[warn] step 20000", warning)
            self.assertIn("step_020000.pt", warning)
            self.assertIn("ckpt_step_20000.pt", warning)
            # The kept file should be flagged "(kept)" and the ignored
            # one "(ignored)" so the operator can identify which is
            # which without re-deriving precedence.
            self.assertIn("(kept)", warning)
            self.assertIn("(ignored)", warning)

    def test_no_collision_no_warning(self):
        """G2 negative-control: a directory containing only ONE naming
        form at each step (the common case on the operator host: only
        ``step_*.pt`` exists) must NOT emit any collision warning. F4
        operators have repeatedly complained about noisy false-positive
        warns; G2 must not regress on that surface.
        """
        import io
        import contextlib
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "step_020000.pt").write_bytes(b"")
            (d / "step_040000.pt").write_bytes(b"")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                steps, m = list_saved_steps(
                    d, include_best=False, include_last=False,
                )
            self.assertEqual(steps, [20000, 40000])
            self.assertNotIn("[warn]", stdout.getvalue())


class TestListSavedStepsRobustness(unittest.TestCase):
    """C3: unrelated files in the directory must not blow up the parser."""

    def test_non_ckpt_files_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "step_010000.pt").write_bytes(b"")
            (d / "metrics.jsonl").write_text("{}\n")
            (d / "config.yaml").write_text("seed: 0\n")
            (d / "wandb").mkdir()
            steps, m = list_saved_steps(d, include_best=False, include_last=False)
            self.assertEqual(steps, [10000])

    def test_invalid_step_suffix_skipped(self):
        # A file whose suffix is not an integer must be skipped silently.
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "step_invalid.pt").write_bytes(b"")
            (d / "step_010000.pt").write_bytes(b"")
            steps, m = list_saved_steps(d, include_best=False, include_last=False)
            self.assertEqual(steps, [10000])

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            steps, m = list_saved_steps(d, include_best=False, include_last=False)
            self.assertEqual(steps, [])
            self.assertEqual(m, {})


# ---------- best.pt / last.pt loading (requires torch) ----------------------

try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@unittest.skipUnless(_HAS_TORCH, "torch not available; skipping ckpt-load tests")
class TestListSavedStepsWithTorch(unittest.TestCase):
    """C3: best.pt / last.pt step extraction with new key (`step`) and
    legacy key (`global_step`) fallback."""

    def _save_ckpt(self, path: Path, payload: dict) -> None:
        import torch
        torch.save(payload, path)

    def test_best_pt_with_step_key(self):
        # Trainer-native: ckpt internal key is "step".
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self._save_ckpt(d / "best.pt", {"step": 7777, "model": {}})
            steps, m = list_saved_steps(d, include_best=True, include_last=False)
            self.assertIn(7777, steps)
            self.assertEqual(m[7777].name, "best.pt")

    def test_last_pt_with_global_step_legacy(self):
        # Legacy fixtures: ckpt internal key is "global_step".
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self._save_ckpt(d / "last.pt", {"global_step": 8888, "model": {}})
            steps, m = list_saved_steps(d, include_best=False, include_last=True)
            self.assertIn(8888, steps)
            self.assertEqual(m[8888].name, "last.pt")

    def test_last_pt_legacy_filename(self):
        # Legacy-named ckpt_last.pt (older fixtures) still works.
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self._save_ckpt(d / "ckpt_last.pt", {"step": 9999, "model": {}})
            steps, m = list_saved_steps(d, include_best=False, include_last=True)
            self.assertIn(9999, steps)
            self.assertEqual(m[9999].name, "ckpt_last.pt")

    def test_step_key_preferred_over_global_step(self):
        # If both keys present, "step" wins (matches trainer's own
        # resume code at train_first_hop.py:1785).
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self._save_ckpt(d / "best.pt", {"step": 100, "global_step": 200, "model": {}})
            steps, m = list_saved_steps(d, include_best=True, include_last=False)
            self.assertIn(100, steps)
            self.assertNotIn(200, steps)


if __name__ == "__main__":
    unittest.main()
