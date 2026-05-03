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
  R11 multi-line anchor → HARD ERROR (B3b hardening; was "first non-empty
      wins" fallback before). Comments and stray content lines are rejected
      explicitly so downstream errors are not misleading.
  R11b blank-padded single-line anchor still resolves under the strict
      single-line rule (whitespace-only lines do not count).
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

    def test_R15_url_fragment_stripped(self):
        """B2a hardening (Lane B peer review): URL fragments (``#...``)
        are stripped before comparison. This keeps a stray
        ``host/path.git#frag`` from comparing unequal to ``host/path``
        purely because of an editor-added anchor."""
        self.assertEqual(
            L._normalize_remote_url("https://gitee.com/jqu9/PET_LatentResidual.git#frag"),
            "gitee.com/jqu9/pet_latentresidual",
        )
        self.assertEqual(
            L._normalize_remote_url("git@gitee.com:jqu9/PET_LatentResidual.git#section/sub"),
            "gitee.com/jqu9/pet_latentresidual",
        )

    def test_R16_url_query_stripped(self):
        """B2a hardening: query strings (``?...``) are likewise stripped
        before path comparison."""
        self.assertEqual(
            L._normalize_remote_url("https://gitee.com/jqu9/PET_LatentResidual.git?token=abcd"),
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

    def test_R5b_case_insensitive_fetch_marker(self):
        """B4 hardening (Lane B peer review): the parser must accept the
        fetch marker case-insensitively. Git always emits lowercase
        ``(fetch)``; this is defensive against future output changes."""
        stdout = (
            "gitee\tgit@gitee.com:jqu9/repo.git (FETCH)\n"
            "gitee\tgit@gitee.com:jqu9/repo.git (Push)\n"
        )
        out = L._parse_git_remote_v(stdout)
        self.assertEqual(out, {"gitee": "git@gitee.com:jqu9/repo.git"})

    def test_R5c_url_with_internal_whitespace_kept_intact(self):
        """B4 hardening: rsplit-from-the-right keeps URLs with spaces in
        them as a single token. Such URLs should not occur in practice,
        but the parser must not silently truncate them."""
        # ``url with space.git`` is the URL field; everything between
        # the first whitespace (after ``name``) and the last whitespace
        # (before ``(fetch)``) is the URL.
        stdout = "weird\tfile:///path with space/repo.git (fetch)\n"
        out = L._parse_git_remote_v(stdout)
        self.assertEqual(out, {"weird": "file:///path with space/repo.git"})

    def test_R5d_crlf_lines_tolerated(self):
        """B4 hardening: CRLF line endings (Windows / mixed checkouts) do
        not leak into the parsed URL or marker."""
        stdout = (
            "gitee\tgit@gitee.com:jqu9/repo.git (fetch)\r\n"
            "gitee\tgit@gitee.com:jqu9/repo.git (push)\r\n"
        )
        out = L._parse_git_remote_v(stdout)
        self.assertEqual(out, {"gitee": "git@gitee.com:jqu9/repo.git"})


# ---------- safe URL repr (B10) --------------------------------------------

class TestSafeReprUrl(unittest.TestCase):
    """B10 hardening (Lane B peer review): printed URLs are wrapped through
    `_safe_repr_url` so embedded ANSI escape sequences and control characters
    do not corrupt the user's terminal when they are surfaced via stderr.
    """

    def test_plain_url_quoted(self):
        # Result is a single-quoted Python repr; the URL is preserved.
        out = L._safe_repr_url("git@gitee.com:jqu9/repo.git")
        self.assertIn("git@gitee.com:jqu9/repo.git", out)
        # Must be a quoted form, not the raw string.
        self.assertTrue(out.startswith("'") or out.startswith("\""))

    def test_ansi_escape_neutralised(self):
        hostile = "https://gitee.com/\x1b[1;31mDELETE\x1b[0m/repo.git"
        out = L._safe_repr_url(hostile)
        # repr() escapes \x1b as \\x1b — the literal ESC byte must NOT be
        # present in the output.
        self.assertNotIn("\x1b", out)
        self.assertIn("\\x1b", out)

    def test_nul_byte_neutralised(self):
        hostile = "https://gitee.com/\x00/repo.git"
        out = L._safe_repr_url(hostile)
        self.assertNotIn("\x00", out)
        self.assertIn("\\x00", out)

    def test_newline_neutralised(self):
        # A newline in a URL field would otherwise let an attacker forge
        # extra log lines via the resolver's error message.
        hostile = "https://gitee.com/jqu9/repo.git\n[info] hijacked"
        out = L._safe_repr_url(hostile)
        # The repr escapes the newline as \n, so the physical newline is
        # gone and the attacker's "[info] hijacked" line cannot start at
        # column 0 of a fresh stderr line.
        self.assertNotIn("\n[info] hijacked", out)
        self.assertIn("\\n", out)


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

    def test_R11_multiline_anchor_raises(self):
        """B3b hardening (Lane B peer review): a multi-line anchor file is
        a hard error. The previous "first non-empty line wins" fallback
        produced confusing downstream errors (e.g. a comment line yielding
        a 0-match resolver error); the new behavior surfaces the offending
        file with its line count immediately."""
        td = self._make_repo_with_anchor(
            "# this is a header comment\n"
            "\n"
            "gitee.com:jqu9/PET_LatentResidual\n"
            "trailing-junk\n"
        )
        try:
            remotes = {"gitee": "git@gitee.com:jqu9/PET_LatentResidual.git"}
            with self.assertRaises(L.CanonicalRemoteError) as ctx:
                L.resolve_canonical_remote(
                    Path(td.name),
                    remotes_provider=lambda: remotes,
                )
            msg = str(ctx.exception)
            self.assertIn("exactly one", msg)
            # Found 3 non-empty lines in the fixture (header, anchor, junk).
            self.assertIn("found 3", msg)
        finally:
            td.cleanup()

    def test_R11b_blank_padded_single_line_anchor_resolves(self):
        """Counterpart to R11: a file with a single content line surrounded
        by blank/whitespace-only lines IS still a valid single-line anchor
        under the strict B3b rule. Whitespace-only lines do not count
        toward the non-empty-line tally."""
        td = self._make_repo_with_anchor(
            "\n"
            "gitee.com:jqu9/PET_LatentResidual\n"
            "\n"
        )
        try:
            remotes = {"gitee": "git@gitee.com:jqu9/PET_LatentResidual.git"}
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
