#!/usr/bin/env python3
"""Golden tests for the canonical-remote resolver in
`lock_effect_size_threshold.py` (C2 per Phase-1 review Q-A / §11.1 Q10).

Run from repo root::

    python review/0502/scripts/test_remote_resolver.py

Or from this dir::

    cd review/0502/scripts && python test_remote_resolver.py

Coverage map (Phase-1 review Q-A + §11.1 Q10 acceptance):

  R1  ssh ↔ https equivalence: scp-style ssh and https URLs that point at
      the same path normalize to the same string.
  R2  trailing `.git` boundary: `repo.git` and `repo` normalize equal.
  R3  case-insensitivity: `PET_LatentResidual` ≡ `pet_latentresidual`.
  R4  fork boundary: `repo_fork.git` does NOT match `repo`. (Q-A (b))
  R5  `git remote -v` parser splits on whitespace and keeps only (fetch).
  R6  resolver returns (name, url) on unique match.
  R7  resolver raises with a helpful message on 0-match. (Q-A (c))
  R8  resolver raises with a helpful message on ≥2-match. (Q-A (d))
  R9  resolver raises if `.review_canonical_remote` is missing.
  R10 resolver raises if `.review_canonical_remote` is empty.
  R11 multi-line anchor → first non-empty line wins (with stderr warn).
  R12 happy-path on the actual repo: anchor file matches at least one
      configured remote (skipped if not in a git repo).
  R13 anchor file with leading UTF-8 BOM (B3a Lane B BLOCKER hardening):
      `_load_anchor` strips the BOM via encoding='utf-8-sig'.
  R14 end-to-end resolve succeeds even with a BOM-prefixed anchor file.

The resolver accepts an injectable `remotes_provider` callable so we can
unit-test the matching logic without spawning git subprocesses or mutating
real remote configuration.
"""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory

# Make the lock script importable.
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import lock_effect_size_threshold as L  # noqa: E402

REPO_ROOT = SCRIPTS_DIR.parent.parent.parent  # …/PET_LatentResidual
ANCHOR_PATH = REPO_ROOT / L.CANONICAL_REMOTE_FILENAME
ANCHOR_VALUE = "gitee.com:jqu9/PET_LatentResidual"


# ---------- pure helpers ----------------------------------------------------

class TestNormalizeRemoteUrl(unittest.TestCase):
    """R1–R4 + edge cases on `_normalize_remote_url()`."""

    def test_R1a_scp_ssh_normalizes(self):
        self.assertEqual(
            L._normalize_remote_url("git@gitee.com:jqu9/PET_LatentResidual.git"),
            "gitee.com/jqu9/pet_latentresidual",
        )

    def test_R1b_https_normalizes(self):
        self.assertEqual(
            L._normalize_remote_url("https://gitee.com/jqu9/PET_LatentResidual.git"),
            "gitee.com/jqu9/pet_latentresidual",
        )

    def test_R1c_ssh_url_with_user(self):
        self.assertEqual(
            L._normalize_remote_url("ssh://git@gitee.com/jqu9/PET_LatentResidual.git"),
            "gitee.com/jqu9/pet_latentresidual",
        )

    def test_R1d_ssh_url_no_user(self):
        self.assertEqual(
            L._normalize_remote_url("ssh://gitee.com/jqu9/PET_LatentResidual.git"),
            "gitee.com/jqu9/pet_latentresidual",
        )

    def test_R1e_http_normalizes(self):
        self.assertEqual(
            L._normalize_remote_url("http://gitee.com/jqu9/PET_LatentResidual.git"),
            "gitee.com/jqu9/pet_latentresidual",
        )

    def test_R1f_anchor_form_normalizes_to_same(self):
        # The anchor file's form (no scheme, no `.git`) must normalize to the
        # same string as the ssh and https forms above.
        self.assertEqual(
            L._normalize_remote_url("gitee.com:jqu9/PET_LatentResidual"),
            "gitee.com/jqu9/pet_latentresidual",
        )

    def test_R2_dot_git_stripped(self):
        with_git = L._normalize_remote_url("git@gitee.com:jqu9/repo.git")
        without_git = L._normalize_remote_url("git@gitee.com:jqu9/repo")
        self.assertEqual(with_git, without_git)

    def test_R3_case_insensitive(self):
        self.assertEqual(
            L._normalize_remote_url("git@GITEE.COM:JQU9/PET_LatentResidual.git"),
            "gitee.com/jqu9/pet_latentresidual",
        )

    def test_R4_fork_boundary_distinct(self):
        # Per Q-A (b): `_fork.git` must NOT compare equal to anchor.
        anchor = L._normalize_remote_url("gitee.com:jqu9/PET_LatentResidual")
        forked = L._normalize_remote_url(
            "https://gitee.com/jqu9/PET_LatentResidual_fork.git")
        self.assertNotEqual(anchor, forked)
        self.assertEqual(anchor, "gitee.com/jqu9/pet_latentresidual")
        self.assertEqual(forked, "gitee.com/jqu9/pet_latentresidual_fork")

    def test_R4b_substring_does_not_silently_match(self):
        # Adversarial case: an attacker registers a remote with a URL that
        # is a SUPERSTRING of the anchor. After normalization+stripping
        # `.git` they must compare unequal.
        anchor = L._normalize_remote_url("gitee.com:jqu9/PET_LatentResidual")
        attacker = L._normalize_remote_url(
            "https://gitee.com/jqu9/PET_LatentResidual2.git")
        self.assertNotEqual(anchor, attacker)

    def test_R4c_path_prefix_does_not_silently_match(self):
        # Another adversarial case: same repo name under a different user.
        anchor = L._normalize_remote_url("gitee.com:jqu9/PET_LatentResidual")
        other_user = L._normalize_remote_url(
            "https://gitee.com/attacker/PET_LatentResidual.git")
        self.assertNotEqual(anchor, other_user)

    def test_trailing_slash_tolerated(self):
        self.assertEqual(
            L._normalize_remote_url("https://gitee.com/jqu9/PET_LatentResidual.git/"),
            "gitee.com/jqu9/pet_latentresidual",
        )

    def test_whitespace_tolerated(self):
        self.assertEqual(
            L._normalize_remote_url("  git@gitee.com:jqu9/PET_LatentResidual.git  "),
            "gitee.com/jqu9/pet_latentresidual",
        )


# ---------- parser ----------------------------------------------------------

class TestParseGitRemoteV(unittest.TestCase):
    """R5: `git remote -v` parser keeps only (fetch) URLs."""

    def test_R5_basic(self):
        stdout = (
            "origin\thttps://github.com/jqu9/PET_LatentResidual.git (fetch)\n"
            "origin\thttps://github.com/jqu9/PET_LatentResidual.git (push)\n"
            "gitee\tgit@gitee.com:jqu9/PET_LatentResidual.git (fetch)\n"
            "gitee\tgit@gitee.com:jqu9/PET_LatentResidual.git (push)\n"
        )
        out = L._parse_git_remote_v(stdout)
        self.assertEqual(set(out.keys()), {"origin", "gitee"})
        self.assertEqual(out["origin"], "https://github.com/jqu9/PET_LatentResidual.git")
        self.assertEqual(out["gitee"], "git@gitee.com:jqu9/PET_LatentResidual.git")

    def test_R5_skips_malformed_lines(self):
        stdout = "garbage_line\nname only\nname\turl (fetch)\n"
        out = L._parse_git_remote_v(stdout)
        self.assertEqual(out, {"name": "url"})

    def test_R5_empty_stdout(self):
        self.assertEqual(L._parse_git_remote_v(""), {})

    def test_R5_distinct_fetch_push_urls(self):
        # Some workflows use distinct fetch/push URLs (e.g. mirror push).
        # Resolver only checks fetch.
        stdout = (
            "gitee\tgit@gitee.com:jqu9/PET_LatentResidual.git (fetch)\n"
            "gitee\tgit@private.example.com:jqu9/different.git (push)\n"
        )
        out = L._parse_git_remote_v(stdout)
        self.assertEqual(out, {"gitee": "git@gitee.com:jqu9/PET_LatentResidual.git"})


# ---------- resolver: pure-fn paths -----------------------------------------

class TestMatchRemoteByAnchor(unittest.TestCase):
    """`_match_remote_by_anchor` returns a list of matching pairs.

    These bypass file-system + subprocess by calling the matcher directly.
    """

    def test_unique_match_ssh(self):
        remotes = {
            "origin": "https://github.com/jqu9/PET_LatentResidual.git",
            "gitee":  "git@gitee.com:jqu9/PET_LatentResidual.git",
        }
        matches = L._match_remote_by_anchor(ANCHOR_VALUE, remotes)
        self.assertEqual(matches, [("gitee", "git@gitee.com:jqu9/PET_LatentResidual.git")])

    def test_unique_match_https_form(self):
        # Operator host: canonical is https-form named `origin`.
        remotes = {"origin": "https://gitee.com/jqu9/PET_LatentResidual.git"}
        matches = L._match_remote_by_anchor(ANCHOR_VALUE, remotes)
        self.assertEqual(matches, [("origin", "https://gitee.com/jqu9/PET_LatentResidual.git")])

    def test_zero_match(self):
        remotes = {
            "origin": "https://github.com/jqu9/PET_LatentResidual.git",
            "fork":   "git@gitee.com:other/PET_LatentResidual.git",
        }
        matches = L._match_remote_by_anchor(ANCHOR_VALUE, remotes)
        self.assertEqual(matches, [])

    def test_two_match(self):
        # Pathological but possible: two named remotes pointing at the same URL.
        remotes = {
            "gitee":  "git@gitee.com:jqu9/PET_LatentResidual.git",
            "gitee2": "https://gitee.com/jqu9/PET_LatentResidual.git",
        }
        matches = L._match_remote_by_anchor(ANCHOR_VALUE, remotes)
        self.assertEqual(len(matches), 2)
        self.assertEqual({m[0] for m in matches}, {"gitee", "gitee2"})

    def test_fork_does_not_match(self):
        # R4 redux at the matcher level.
        remotes = {"fork": "https://gitee.com/jqu9/PET_LatentResidual_fork.git"}
        self.assertEqual(L._match_remote_by_anchor(ANCHOR_VALUE, remotes), [])


# ---------- resolver: full integration --------------------------------------

class TestResolveCanonicalRemote(unittest.TestCase):
    """R6–R11: end-to-end calls to `resolve_canonical_remote()` using
    a tempdir as repo_root and an injected `remotes_provider`."""

    def _make_repo_with_anchor(self, content: str) -> "TemporaryDirectory[str]":
        td = TemporaryDirectory()
        anchor_file = Path(td.name) / L.CANONICAL_REMOTE_FILENAME
        anchor_file.write_text(content)
        return td

    def test_R6_unique_match_returns_pair(self):
        td = self._make_repo_with_anchor(ANCHOR_VALUE + "\n")
        try:
            remotes = {
                "origin": "https://github.com/jqu9/PET_LatentResidual.git",
                "gitee":  "git@gitee.com:jqu9/PET_LatentResidual.git",
            }
            name, url = L.resolve_canonical_remote(
                Path(td.name),
                remotes_provider=lambda: remotes,
            )
            self.assertEqual(name, "gitee")
            self.assertEqual(url, "git@gitee.com:jqu9/PET_LatentResidual.git")
        finally:
            td.cleanup()

    def test_R7_zero_match_raises(self):
        td = self._make_repo_with_anchor(ANCHOR_VALUE + "\n")
        try:
            remotes = {"origin": "https://github.com/jqu9/PET_LatentResidual.git"}
            with self.assertRaises(L.CanonicalRemoteError) as ctx:
                L.resolve_canonical_remote(
                    Path(td.name),
                    remotes_provider=lambda: remotes,
                )
            msg = str(ctx.exception)
            self.assertIn("no remote matches", msg)
            self.assertIn(ANCHOR_VALUE, msg)
            # Must list the configured remotes for actionable feedback.
            self.assertIn("origin", msg)
            self.assertIn("--remote", msg)
        finally:
            td.cleanup()

    def test_R8_two_match_raises(self):
        td = self._make_repo_with_anchor(ANCHOR_VALUE + "\n")
        try:
            remotes = {
                "gitee":  "git@gitee.com:jqu9/PET_LatentResidual.git",
                "gitee2": "https://gitee.com/jqu9/PET_LatentResidual.git",
            }
            with self.assertRaises(L.CanonicalRemoteError) as ctx:
                L.resolve_canonical_remote(
                    Path(td.name),
                    remotes_provider=lambda: remotes,
                )
            msg = str(ctx.exception)
            self.assertIn("2 remotes match", msg)
            self.assertIn("gitee", msg)
            self.assertIn("gitee2", msg)
            self.assertIn("--remote", msg)
        finally:
            td.cleanup()

    def test_R9_missing_anchor_raises(self):
        td = TemporaryDirectory()
        try:
            with self.assertRaises(L.CanonicalRemoteError) as ctx:
                L.resolve_canonical_remote(
                    Path(td.name),
                    remotes_provider=lambda: {"any": "url"},
                )
            self.assertIn("missing anchor file", str(ctx.exception))
            self.assertIn(L.CANONICAL_REMOTE_FILENAME, str(ctx.exception))
        finally:
            td.cleanup()

    def test_R10_empty_anchor_raises(self):
        td = self._make_repo_with_anchor("   \n")  # whitespace only
        try:
            with self.assertRaises(L.CanonicalRemoteError) as ctx:
                L.resolve_canonical_remote(
                    Path(td.name),
                    remotes_provider=lambda: {"any": "url"},
                )
            self.assertIn("empty", str(ctx.exception))
        finally:
            td.cleanup()

    def test_R11_multiline_first_nonempty_wins(self):
        td = self._make_repo_with_anchor(
            "# this is a header comment\n"
            "\n"
            "gitee.com:jqu9/PET_LatentResidual\n"
            "trailing-junk\n"
        )
        try:
            remotes = {"gitee": "git@gitee.com:jqu9/PET_LatentResidual.git"}
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                # First non-empty line is "# this is a header comment" — that
                # MUST be treated as the anchor (we don't have comment
                # syntax). So we expect a 0-match raise here, since the
                # comment line doesn't match any remote.
                with self.assertRaises(L.CanonicalRemoteError):
                    L.resolve_canonical_remote(
                        Path(td.name),
                        remotes_provider=lambda: remotes,
                    )
            # The warning is emitted unconditionally for multi-line anchor.
            self.assertIn("multiple lines", stderr.getvalue())
        finally:
            td.cleanup()

    def test_R11b_no_match_when_first_line_is_anchor(self):
        # Same test but the first line IS the canonical anchor. Verifies the
        # fallback "first non-empty line wins" picks the right one when
        # extra blank lines surround it.
        td = self._make_repo_with_anchor(
            "\n"
            "gitee.com:jqu9/PET_LatentResidual\n"
            "\n"
        )
        try:
            remotes = {"gitee": "git@gitee.com:jqu9/PET_LatentResidual.git"}
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                name, url = L.resolve_canonical_remote(
                    Path(td.name),
                    remotes_provider=lambda: remotes,
                )
            self.assertEqual(name, "gitee")
        finally:
            td.cleanup()

    def test_no_remotes_at_all_raises(self):
        td = self._make_repo_with_anchor(ANCHOR_VALUE + "\n")
        try:
            with self.assertRaises(L.CanonicalRemoteError) as ctx:
                L.resolve_canonical_remote(
                    Path(td.name),
                    remotes_provider=lambda: {},
                )
            self.assertIn("no fetch remotes", str(ctx.exception))
        finally:
            td.cleanup()

    def test_R13_load_anchor_strips_utf8_bom(self):
        """B3a (Lane B BLOCKER): anchor file with leading UTF-8 BOM
        (0xEF 0xBB 0xBF, e.g. saved by a Windows editor) must NOT silently
        break canonical-remote matching. `_load_anchor` opens with
        encoding='utf-8-sig' so the BOM is consumed."""
        td = TemporaryDirectory()
        try:
            anchor_file = Path(td.name) / L.CANONICAL_REMOTE_FILENAME
            anchor_file.write_bytes(b"\xef\xbb\xbf" + ANCHOR_VALUE.encode("utf-8") + b"\n")
            text = L._load_anchor(Path(td.name))
            self.assertEqual(text, ANCHOR_VALUE,
                             "BOM should be stripped; got {!r}".format(text))
            # Sanity: the BOM bytes are NOT in the returned string.
            self.assertNotIn("\ufeff", text)
        finally:
            td.cleanup()

    def test_R14_resolve_works_with_bom_anchor(self):
        """End-to-end: resolver succeeds even when anchor file has BOM."""
        td = TemporaryDirectory()
        try:
            anchor_file = Path(td.name) / L.CANONICAL_REMOTE_FILENAME
            anchor_file.write_bytes(b"\xef\xbb\xbf" + ANCHOR_VALUE.encode("utf-8") + b"\n")
            remotes = {"gitee": "git@gitee.com:jqu9/PET_LatentResidual.git"}
            name, url = L.resolve_canonical_remote(
                Path(td.name),
                remotes_provider=lambda: remotes,
            )
            self.assertEqual(name, "gitee")
        finally:
            td.cleanup()


# ---------- happy-path on the actual repo ----------------------------------

class TestResolveOnLiveRepo(unittest.TestCase):
    """R12: anchor + actual `git remote -v` resolve to ≥1 match.

    Skipped if `.git` not present (e.g. running outside a checkout)."""

    def test_R12_anchor_resolves_to_known_remote(self):
        if not (REPO_ROOT / ".git").exists():
            self.skipTest("not in a git repo")
        if not ANCHOR_PATH.exists():
            self.skipTest(".review_canonical_remote not present")
        # Use the real provider (no injection).
        try:
            name, url = L.resolve_canonical_remote(REPO_ROOT)
        except L.CanonicalRemoteError as exc:
            self.fail(f"live-repo resolve raised: {exc}")
        # On the author's Mac the canonical remote name is `gitee`. On the
        # operator's host it is `origin`. We don't assert which — just that
        # the URL normalizes to the anchor.
        self.assertEqual(
            L._normalize_remote_url(url),
            L._normalize_remote_url(ANCHOR_PATH.read_text().strip()),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
