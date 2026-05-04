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

    def test_R17_https_userinfo_stripped(self):
        """E1 (Round-5 external peer review of C2.2): URL userinfo of
        the form ``user@`` or ``user:token@`` placed strictly before the
        first ``/`` must be stripped before the scp ``:`` heuristic
        runs. Without the strip, the ``:`` inside ``user:token`` was
        misread as the scp ``host:path`` separator and the function
        returned a wrong ``user/token@host/path`` form."""
        self.assertEqual(
            L._normalize_remote_url("https://user@gitee.com/jqu9/PET_LatentResidual.git"),
            "gitee.com/jqu9/pet_latentresidual",
        )
        self.assertEqual(
            L._normalize_remote_url(
                "https://user:pass@gitee.com/jqu9/PET_LatentResidual.git"
            ),
            "gitee.com/jqu9/pet_latentresidual",
        )

    def test_R18_ssh_url_userinfo_stripped(self):
        """E1: ``ssh://`` URL form with explicit user (or user:token)
        must collapse to the same canonical form as the no-user
        ``ssh://`` URL. R1c already covers ``ssh://git@host/path``;
        this case extends to arbitrary userinfo."""
        self.assertEqual(
            L._normalize_remote_url(
                "ssh://user:tok@gitee.com/jqu9/PET_LatentResidual.git"
            ),
            "gitee.com/jqu9/pet_latentresidual",
        )
        self.assertEqual(
            L._normalize_remote_url(
                "ssh://gituser@gitee.com/jqu9/PET_LatentResidual.git"
            ),
            "gitee.com/jqu9/pet_latentresidual",
        )

    def test_R19_userinfo_with_query_and_fragment(self):
        """E1 + B2a interaction: a URL carrying userinfo, a query, and
        a fragment all at once must still normalize to the canonical
        ``host/path`` form. Composition test for the three independent
        strips (userinfo, query, fragment)."""
        self.assertEqual(
            L._normalize_remote_url(
                "https://user:tok@gitee.com/jqu9/PET_LatentResidual.git?token=abc#frag"
            ),
            "gitee.com/jqu9/pet_latentresidual",
        )

    def test_R20_scp_form_unaffected_by_userinfo_strip(self):
        """E1 regression guard: the existing scp-style ``git@host:path``
        form must still normalize correctly. The new userinfo strip
        runs unconditionally, which means it ALSO peels off the leading
        ``git@`` for scp-style inputs (since the ``@`` precedes the
        first ``/``). The scp ``:`` heuristic that follows must then
        still convert ``host:path`` to ``host/path``."""
        self.assertEqual(
            L._normalize_remote_url("git@gitee.com:jqu9/PET_LatentResidual.git"),
            "gitee.com/jqu9/pet_latentresidual",
        )
        # Pathological scp form with userinfo before git@ is not in our
        # supported input grammar; we only assert it does not crash.
        result = L._normalize_remote_url(
            "user:tok@gitee.com:jqu9/PET_LatentResidual.git"
        )
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


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


# ---------- safe stderr block (D1) -----------------------------------------

class TestSafeStderrBlock(unittest.TestCase):
    """D1 hardening (Round-5 external peer review of C2.2): subprocess
    stderr surfaced to the operator goes through `_safe_stderr_block` so
    git's own diagnostics — which can echo a hostile remote URL or
    embed ANSI / NUL / newline bytes — cannot corrupt the operator's
    terminal. This complements `_safe_repr_url`, which is for single
    URL strings; `_safe_stderr_block` is for multi-line subprocess
    stderr blobs.
    """

    def test_plain_text_passthrough(self):
        out = L._safe_stderr_block("fatal: unable to access 'host'")
        # No control bytes in the input, so the output is unchanged.
        self.assertEqual(out, "fatal: unable to access 'host'")

    def test_ansi_escape_neutralised(self):
        hostile = "\x1b[31mfatal\x1b[0m: hostile error"
        out = L._safe_stderr_block(hostile)
        self.assertNotIn("\x1b", out)
        self.assertIn("\\x1b", out)

    def test_nul_byte_neutralised(self):
        hostile = "fatal:\x00 NUL injected"
        out = L._safe_stderr_block(hostile)
        self.assertNotIn("\x00", out)
        self.assertIn("\\x00", out)

    def test_newline_collapsed(self):
        # Multi-line stderr (the common shape from `subprocess.run`) is
        # collapsed to a single logical line via `\n` escapes. This
        # prevents an attacker from forging additional stderr lines
        # starting at column 0.
        hostile = "fatal: unable to access 'host'\n[info] forged log line"
        out = L._safe_stderr_block(hostile)
        self.assertNotIn("\n[info] forged log line", out)
        self.assertIn("\\n[info] forged log line", out)

    def test_carriage_return_neutralised(self):
        # \r alone could overwrite the previous logical line on a
        # terminal; must be escaped.
        hostile = "fatal: error\rOK"
        out = L._safe_stderr_block(hostile)
        self.assertNotIn("\r", out)
        self.assertIn("\\r", out)

    def test_none_returns_sentinel(self):
        # Defensive: None inputs (e.g. a subprocess that produced no
        # stderr) must not raise and must not echo "None" via str().
        out = L._safe_stderr_block(None)
        self.assertEqual(out, "<None>")

    def test_url_in_stderr_blob_neutralised(self):
        # The motivating threat: git's own message echoes the remote
        # URL, which itself can be hostile. The stderr-block sanitizer
        # is what closes that surface for the 3 git() / commit / push
        # call sites that do not route through _safe_repr_url.
        url = "https://gitee.com/\x1b[1;31mhostile\x1b[0m/repo.git"
        stderr_blob = f"fatal: unable to access '{url}': could not connect\n"
        out = L._safe_stderr_block(stderr_blob)
        self.assertNotIn("\x1b", out)
        self.assertNotIn("\n", out)
        self.assertIn("\\x1b", out)


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


class TestPushUrlTOCTOU(unittest.TestCase):
    """F3 hardening (Round-5 operator review, May 4 2026): the pre-push
    re-verification must check BOTH the fetch URL and the dedicated
    pushURL, because `git push <remote>` follows ``remote.<name>.pushURL``
    when configured. A canonical fetch URL with a divergent pushURL would
    silently push to a non-canonical target and bypass the pre-registration
    timestamp guarantee.

    These tests use real ``git`` subprocesses on a tempdir to verify the
    F3 helper sequence (``git remote get-url`` vs
    ``git remote get-url --push``) returns what we expect, and that
    ``_normalize_remote_url`` correctly distinguishes them.
    """

    def _git_init_with_remote(self, tmpdir, fetch_url, push_url=None):
        """Create a minimal repo with origin's fetch URL = ``fetch_url`` and
        (optionally) a different pushURL. Returns the tmpdir Path."""
        import subprocess
        repo = Path(tmpdir)
        # `git init` is enough; we don't need a working tree to manipulate
        # remote.* config keys.
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "remote", "add", "origin", fetch_url],
                       cwd=repo, check=True)
        if push_url is not None:
            subprocess.run(
                ["git", "remote", "set-url", "--push", "origin", push_url],
                cwd=repo, check=True,
            )
        return repo

    def test_F3a_no_push_url_configured_returns_fetch_url(self):
        """Baseline: when no separate pushURL is configured,
        ``git remote get-url --push origin`` returns the fetch URL.
        F3's pushURL check is therefore trivially satisfied on the
        common case (no false positive)."""
        import subprocess
        import tempfile
        fetch = "git@gitee.com:jqu9/PET_LatentResidual.git"
        with tempfile.TemporaryDirectory() as td:
            repo = self._git_init_with_remote(td, fetch_url=fetch,
                                              push_url=None)
            fetch_observed = L.git(["remote", "get-url", "origin"],
                                   cwd=repo)
            push_observed = L.git(["remote", "get-url", "--push", "origin"],
                                  cwd=repo)
            self.assertEqual(L._normalize_remote_url(fetch_observed),
                             L._normalize_remote_url(fetch))
            self.assertEqual(L._normalize_remote_url(push_observed),
                             L._normalize_remote_url(fetch),
                             "no pushURL configured: --push must equal fetch")

    def test_F3b_divergent_push_url_is_observable(self):
        """The bug F3 fixes: ``git push origin`` would silently follow a
        configured pushURL even if the fetch URL is canonical. This test
        proves the pushURL is observable AND that it normalizes
        differently from the canonical, i.e. F3 has falsifiable signal.
        """
        import tempfile
        fetch = "git@gitee.com:jqu9/PET_LatentResidual.git"   # canonical
        push = "git@gitee.com:attacker/PET_LatentResidual.git"  # divergent
        with tempfile.TemporaryDirectory() as td:
            repo = self._git_init_with_remote(td, fetch_url=fetch,
                                              push_url=push)
            fetch_observed = L.git(["remote", "get-url", "origin"],
                                   cwd=repo)
            push_observed = L.git(["remote", "get-url", "--push", "origin"],
                                  cwd=repo)
            # Fetch URL is unchanged (canonical) — fetch-side check passes.
            self.assertEqual(L._normalize_remote_url(fetch_observed),
                             L._normalize_remote_url(fetch))
            # PushURL diverges — F3 catches this; pre-F3 code would not.
            self.assertNotEqual(
                L._normalize_remote_url(push_observed),
                L._normalize_remote_url(fetch),
                "divergent pushURL must normalize differently from canonical",
            )

    def test_F3c_divergent_but_equivalent_push_url_passes(self):
        """Negative-control for F3: a configured pushURL that is in a
        DIFFERENT URL form (e.g. https vs ssh) but points at the SAME
        canonical repo must NOT trigger the F3 abort. ``_normalize_remote_url``
        is form-agnostic, so https://gitee.com/jqu9/PET_LatentResidual.git
        and git@gitee.com:jqu9/PET_LatentResidual.git canonicalize to the
        same string and F3 must accept the configuration.
        """
        import tempfile
        fetch = "git@gitee.com:jqu9/PET_LatentResidual.git"
        push = "https://gitee.com/jqu9/PET_LatentResidual.git"  # same repo
        with tempfile.TemporaryDirectory() as td:
            repo = self._git_init_with_remote(td, fetch_url=fetch,
                                              push_url=push)
            push_observed = L.git(["remote", "get-url", "--push", "origin"],
                                  cwd=repo)
            self.assertEqual(
                L._normalize_remote_url(push_observed),
                L._normalize_remote_url(fetch),
                "https and ssh forms of same repo must canonicalize equal",
            )

    def test_F3d_multiple_push_urls_first_canonical_second_divergent(self):
        """G1 hardening (Round-7 cross-AI peer review, May 4 2026): a
        remote may have MULTIPLE pushURLs configured via
        ``git remote set-url --push --add``. ``git push`` mirrors to all
        of them, but the Round-6 F3 patch only checked
        ``git remote get-url --push <remote>`` which returns just the
        FIRST. A canonical-first-then-hostile multi-pushURL config
        therefore bypassed the Round-6 F3 check.

        This test simulates that bypass directly:
          1. fetch URL: canonical
          2. pushURL #1: canonical (added via --push --add)
          3. pushURL #2: hostile (added via --push --add)

        It then asserts that:
          (a) ``--push`` (single, Round-6 form) returns ONLY the first
              pushURL → the bypass is real;
          (b) ``--push --all`` (G1 form) returns BOTH pushURLs;
          (c) at least one of the enumerated pushURLs normalizes
              differently from the canonical → G1's iteration-then-check
              loop catches the divergent mirror.

        If git ever changes ``--push --all`` semantics, this test will
        fail loudly rather than silently regressing the canonical check.
        """
        import subprocess
        import tempfile
        canonical = "git@gitee.com:jqu9/PET_LatentResidual.git"
        hostile = "git@evil.com:attacker/PET_LatentResidual.git"
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(
                ["git", "remote", "add", "origin", canonical],
                cwd=repo, check=True,
            )
            # Multi-pushURL: append two pushURLs (first canonical, second
            # hostile). `--add` flag is what creates the multi-URL state
            # — without `--add` the second `set-url --push` would
            # *replace* the first.
            subprocess.run(
                ["git", "remote", "set-url", "--push", "--add",
                 "origin", canonical],
                cwd=repo, check=True,
            )
            subprocess.run(
                ["git", "remote", "set-url", "--push", "--add",
                 "origin", hostile],
                cwd=repo, check=True,
            )

            # (a) Round-6 form sees only the first pushURL (canonical) →
            # would falsely pass the Round-6 F3 check.
            single = L.git(["remote", "get-url", "--push", "origin"],
                           cwd=repo)
            self.assertEqual(
                L._normalize_remote_url(single),
                L._normalize_remote_url(canonical),
                "single --push must return canonical (the bypass)",
            )

            # (b) G1 form enumerates BOTH.
            block = L.git(
                ["remote", "get-url", "--push", "--all", "origin"],
                cwd=repo,
            )
            urls = [u for u in block.splitlines() if u.strip()]
            self.assertEqual(
                len(urls), 2,
                f"--push --all must enumerate both pushURLs; got {urls!r}",
            )

            # (c) At least one enumerated URL diverges from canonical →
            # G1's per-URL normalize-and-compare catches the bypass.
            canonical_norm = L._normalize_remote_url(canonical)
            divergent = [u for u in urls
                         if L._normalize_remote_url(u) != canonical_norm]
            self.assertEqual(
                len(divergent), 1,
                f"exactly one URL ({hostile!r}) must diverge from "
                f"canonical; got divergent={divergent!r}",
            )
            self.assertEqual(
                L._normalize_remote_url(divergent[0]),
                L._normalize_remote_url(hostile),
                "the divergent URL must be the hostile one",
            )

    def test_F3e_multiple_push_urls_all_canonical_pass(self):
        """G1 negative-control: multiple pushURLs that ALL canonicalize
        to the same repo (e.g. ssh + https forms of the same gitee path)
        must NOT trigger the G1 abort. This is a legitimate "mirror to
        the same target via two protocols" config.
        """
        import subprocess
        import tempfile
        canonical_ssh = "git@gitee.com:jqu9/PET_LatentResidual.git"
        canonical_https = "https://gitee.com/jqu9/PET_LatentResidual.git"
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(
                ["git", "remote", "add", "origin", canonical_ssh],
                cwd=repo, check=True,
            )
            subprocess.run(
                ["git", "remote", "set-url", "--push", "--add",
                 "origin", canonical_ssh],
                cwd=repo, check=True,
            )
            subprocess.run(
                ["git", "remote", "set-url", "--push", "--add",
                 "origin", canonical_https],
                cwd=repo, check=True,
            )
            block = L.git(
                ["remote", "get-url", "--push", "--all", "origin"],
                cwd=repo,
            )
            urls = [u for u in block.splitlines() if u.strip()]
            self.assertEqual(len(urls), 2)
            canonical_norm = L._normalize_remote_url(canonical_ssh)
            for u in urls:
                self.assertEqual(
                    L._normalize_remote_url(u), canonical_norm,
                    f"all pushURLs must canonicalize equal; got {u!r}",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
