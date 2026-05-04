#!/usr/bin/env python3
"""Lock the σ-normalize ablation effect-size threshold X (per §6.6 of POST_V6_NEXT_STEPS.md).

This script automates §6.6.2 Step 3-4 of the blinded analysis pre-registration:

    Step 3: Read A_main metrics.jsonl, compute paired_CV_A from val_select_score
            in [step_min, step_max] window.
    Step 4: Apply LOCKED formula  X = max(0.10, 3 × paired_CV_A).
            Write EFFECT_SIZE_LOCKED.md (containing paired_CV_A, X, metrics file
            content hash, training run's git commit hash, formula citation).
            Optionally git commit + push.

LOCKED parameters (these MUST NOT be configurable post-hoc — modifying them is
scientific fraud per §6.6.7):

    floor       0.10
    slope       3.0
    metric_key  val_select_score
    window      step ∈ [step_min, step_max]   (caller specifies, but cannot
                                                widen post-hoc to game results)

The script enforces the LOCKED parameters: caller cannot pass --floor, --slope,
or --metric-key. Window is required and recommended to be [40000, 60000] per
§6.6.1, but caller can specify a different window only if they document why
in EFFECT_SIZE_LOCKED.md (deviation note).

Order-violation guards (these prevent the most common pre-registration breakage):

    Guard 1: working tree must be clean (no uncommitted changes that could
             retroactively pollute the lock-in commit).
    Guard 2: EFFECT_SIZE_LOCKED.md must NOT already exist (else lock has already
             happened — running again = retroactive modification = fraud).
    Guard 3: NO C_uniform full-val artifact may exist when this script runs.
             We detect this by scanning for files matching --c-uniform-output-dir
             that suggest a Layer 4 full-val has already been computed. If found,
             exit 7 (sequence violation: C results were peeked at before X locked).
    Guard 4: A_main metrics.jsonl must contain ≥ 5 rows in the [step_min, step_max]
             window with val_select_score present.
    Guard 5: A_main config yaml must declare the exact training_run_dir we read
             metrics from (sanity check that we're locking against the right run).

Exit codes:
    0   Success: EFFECT_SIZE_LOCKED.md written and (optionally) committed+pushed.
    1   Guard 4 failed (insufficient observations).
    2   Guard 1 failed (dirty working tree).
    3   Guard 2 failed (lock file already exists).
    4   Guard 5 failed (config / run-dir mismatch — config-side OR data-side).
    5   PyYAML missing.
    6   Canonical-remote resolution failed (missing `.review_canonical_remote`,
        no remote matches the URL fragment, OR ≥2 remotes match — see C2 in
        REV1_TOOLING_PLAN.md §11.1 Q10 / Phase-1 review Q-A).
    7   Guard 3 failed (C_uniform results detected — sequence violation).
    8   Git operation failed.

Usage:
    # 1. Sanity preview (no commit)
    python review/0502/scripts/lock_effect_size_threshold.py \
        --metrics-a /data_2/.../A_main/run-.../metrics.jsonl \
        --config-a  review/0502/configs/A_control.yaml \
        --output    review/0502/EFFECT_SIZE_LOCKED.md \
        --step-min  40000 \
        --step-max  60000 \
        --no-commit

    # 2. Real lock-in (commit + push)
    python review/0502/scripts/lock_effect_size_threshold.py \
        --metrics-a /data_2/.../A_main/run-.../metrics.jsonl \
        --config-a  review/0502/configs/A_control.yaml \
        --output    review/0502/EFFECT_SIZE_LOCKED.md \
        --step-min  40000 \
        --step-max  60000 \
        --c-uniform-output-dir /data_2/.../C_uniform/

Status: Implements §6.6.2 Step 3-4 (LOCKED 2026-05-03 in commit 867b5c0).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Local helper: schema-tolerant step extraction. See `_metrics_compat.py` for
# the rationale (trainer writes `step`; legacy fixtures may use `global_step`).
from _metrics_compat import get_row_step

# ---- LOCKED parameters per §6.6.1 — DO NOT EXPOSE AS CLI ARGS ----
LOCKED_FLOOR = 0.10
LOCKED_SLOPE = 3.0
LOCKED_METRIC_KEY = "val_select_score"
LOCKED_RECOMMENDED_WINDOW = (40000, 60000)

# ---- Pre-registration protocol version (S12 from REV1_TOOLING_PLAN §10.2) ----
# This constant is checked by `run_ablation.sh` `require_lock_pass()` (C5b)
# before C_uniform launches; only `"R1b_locked"` permits the launch. The
# value transitions across the commit chain:
#     C1   sets   "v0_pending_R1a"      (this file: pre-R1a tooling baseline)
#     R1a  sets   "R1a_pending_data"    (skeleton committed; A_main / A_seed=43 not yet at 120K)
#     R1b  sets   "R1b_locked"          (numeric paired_SD_AA + X embedded; C may launch)
# Any other value at C-launch time → hard-gate refuses.
LOCKED_PROTOCOL_VERSION = "v0_pending_R1a"
LOCKED_FORMULA_CITATION = (
    "X = max(0.10, 3 × paired_CV_A)  per POST_V6_NEXT_STEPS.md §6.6.1"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--metrics-a", required=True, type=Path,
                   help="Path to A_main metrics.jsonl")
    p.add_argument("--config-a", required=True, type=Path,
                   help="Path to A_main yaml (used by guard 5 sanity check)")
    p.add_argument("--output", required=True, type=Path,
                   help="Path to write EFFECT_SIZE_LOCKED.md (typically "
                        "review/0502/EFFECT_SIZE_LOCKED.md)")
    p.add_argument("--step-min", type=int, default=LOCKED_RECOMMENDED_WINDOW[0],
                   help=f"Inclusive lower step bound. LOCKED default = "
                        f"{LOCKED_RECOMMENDED_WINDOW[0]}. If you change this "
                        f"you MUST document why in the deviation note.")
    p.add_argument("--step-max", type=int, default=LOCKED_RECOMMENDED_WINDOW[1],
                   help=f"Inclusive upper step bound. LOCKED default = "
                        f"{LOCKED_RECOMMENDED_WINDOW[1]}. Same deviation rule.")
    p.add_argument("--c-uniform-output-dir", type=Path, default=None,
                   help="Path to C_uniform run output dir (or its eval output "
                        "dir). If provided, guard 3 scans for 'full_val' or "
                        "Method-D-selected ckpt artifacts that would indicate "
                        "C results were peeked at before X was locked.")
    p.add_argument("--deviation-note", type=str, default=None,
                   help="Required if --step-min or --step-max differ from the "
                        "LOCKED default window. Free-text rationale that will "
                        "be embedded in EFFECT_SIZE_LOCKED.md.")
    p.add_argument("--no-commit", action="store_true",
                   help="Don't auto-commit. Useful for previewing the X value "
                        "before locking in. Files are still written to --output.")
    p.add_argument("--no-push", action="store_true",
                   help="Commit but don't push. Default is to push to the "
                        "canonical remote (resolved per `.review_canonical_remote` "
                        "— see C2 in REV1_TOOLING_PLAN.md) so the pre-registration "
                        "timestamp is publicly verifiable.")
    p.add_argument("--remote", default=None,
                   help="Git remote to push to. If omitted, resolved at runtime "
                        "by reading `.review_canonical_remote` at repo root and "
                        "matching it against `git remote -v`. On the operator's "
                        "host the canonical remote is named `origin`; on the "
                        "author's Mac it is named `gitee`. Pass an explicit name "
                        "to bypass the resolver (e.g. when ambiguous). The "
                        "supplied name is still verified to point at the "
                        "canonical anchor URL unless --force-unsafe-remote is "
                        "also set.")
    p.add_argument("--force-unsafe-remote", action="store_true",
                   help="DANGEROUS: skip the canonical-anchor verification of "
                        "the URL behind --remote, AND skip the pre-push TOCTOU "
                        "re-verification. Default behavior (B5/B6 hardening) "
                        "is to compare `git remote get-url <name>` against the "
                        "anchor both right after argparse and immediately "
                        "before push. Only set this flag when intentionally "
                        "testing against a fork.")
    p.add_argument("--allow-dirty", action="store_true",
                   help="DANGEROUS: bypass guard 1 (clean working tree). Only "
                        "for development testing; never use in production.")
    return p.parse_args()


def load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError:
        sys.stderr.write(
            "ERROR: PyYAML not installed. `pip install pyyaml` and retry.\n"
        )
        sys.exit(5)
    with path.open() as f:
        return yaml.safe_load(f) or {}


def load_metrics(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_metrics_with_hash(path: Path) -> tuple[list[dict], str]:
    """Read file as bytes ONCE, hash those bytes, then parse from them.

    Fixes the Cl-Q6 SHA256-race bug (Round-1 review): previously
    `compute_paired_cv()` called `load_metrics()` (file open at T0) and
    `file_sha256()` (separate file open at T1); if the trainer appended
    rows in [T0, T1] the hash would cover bytes never seen by the parser.
    This helper guarantees the returned rows and hash refer to the exact
    same byte stream.
    """
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    rows: list[dict] = []
    for line in raw.splitlines():
        s = line.decode("utf-8", errors="replace").strip()
        if not s:
            continue
        try:
            rows.append(json.loads(s))
        except json.JSONDecodeError:
            continue
    return rows, sha


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git(args: list[str], cwd: Path | None = None) -> str:
    res = subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        # D1 (Round-5 peer review): wrap raw subprocess stderr through
        # _safe_stderr_block so embedded ANSI/NUL/CR bytes from a hostile
        # remote URL echoed back by git itself cannot corrupt the
        # operator's terminal.
        sys.stderr.write(
            f"git {' '.join(args)} failed (exit {res.returncode}):\n"
            f"{_safe_stderr_block(res.stderr)}\n"
        )
        raise RuntimeError(f"git command failed: git {' '.join(args)}")
    return res.stdout.strip()


# ---------------------------------------------------------------------------
# Canonical-remote resolver (C2 per Phase-1 review Q-A / §11.1 Q10)
# ---------------------------------------------------------------------------
#
# `.review_canonical_remote` at repo root contains a single line URL fragment
# (e.g. `gitee.com:jqu9/PET_LatentResidual`). The resolver scans `git remote -v`
# and returns the local remote name whose URL points at that same repository,
# regardless of the local nickname (`gitee` on the author's Mac, `origin` on
# the operator's host) and regardless of ssh-vs-https URL form.
#
# Q-A (Phase-1 review) tightened the matching semantics to:
#   (a) Normalize each candidate URL to a single canonical lowercased
#       `host/path` form by collapsing ssh `host:path` ↔ https `host/path`.
#   (b) Strip a trailing `.git` so the comparison is anchored at the repo
#       boundary (otherwise `gitee.com/jqu9/PET_LatentResidual_fork.git`
#       could spuriously match an `endswith()`-style check).
#   (c) On 0 matches → exit 6 with a hint pointing at the missing remote.
#   (d) On ≥2 matches → exit 6 listing all matches and require explicit
#       `--remote` override.
#   (e) On exactly 1 match → return (remote_name, original_url).

CANONICAL_REMOTE_FILENAME = ".review_canonical_remote"


class CanonicalRemoteError(Exception):
    """Raised when the canonical remote cannot be uniquely resolved.

    Carries human-readable text suitable for printing to stderr; the caller
    converts this into exit code 6.
    """


def _safe_repr_url(url: str) -> str:
    """Return a printable repr of a URL with control characters and ANSI
    escape sequences neutralised.

    B10 hardening (Lane B peer review): a hostile or accidentally-corrupt
    git remote URL could contain ``\\x1b[...`` escape sequences or NUL /
    other C0 / DEL bytes that, when written to a terminal via
    ``sys.stderr``, alter rendering or the active tty mode. Wrapping
    every printed URL through this helper produces a single-line,
    single-quoted form (Python ``repr``) where any non-printable byte is
    rendered as ``\\xHH``. The helper is purely cosmetic / defensive; it
    is NOT part of the matching contract — matching always goes through
    ``_normalize_remote_url``.
    """
    if url is None:
        return "<None>"
    return repr(str(url))


def _safe_stderr_block(text: str) -> str:
    """Return a printable form of an arbitrary subprocess stderr blob with
    control characters and ANSI escape sequences neutralised.

    D1 hardening (Round-5 external peer review of C2.2): ``_safe_repr_url``
    handles single URL strings but is unsuitable for multi-line stderr
    captured from ``subprocess.run``. A hostile remote URL embedded in
    git's own error message (e.g. ``fatal: unable to access '<URL>'``)
    can re-introduce the terminal-corruption surface that B10 closed, if
    the script writes ``proc.stderr`` to ``sys.stderr`` verbatim.

    We use ``unicode_escape`` so that:
      - ``\\x1b`` ANSI sequences render as the literal four bytes
        ``\\x1b`` (no terminal mode change).
      - NUL / C0 / DEL bytes render as ``\\xHH``.
      - Real newlines render as the literal two bytes ``\\n``, collapsing
        the entire stderr blob onto a single logical line. This keeps
        diagnostic output legible in a log file while removing the
        terminal-corruption surface.

    Like ``_safe_repr_url``, this helper is purely cosmetic / defensive
    and is NOT part of any matching or signing contract.
    """
    if text is None:
        return "<None>"
    s = str(text)
    try:
        return s.encode("unicode_escape").decode("ascii")
    except (UnicodeEncodeError, UnicodeDecodeError):  # pragma: no cover
        # Fallback: raw repr() always succeeds, never echoes bytes.
        return repr(s)


def _normalize_remote_url(url: str) -> str:
    """Normalize a git remote URL to a lowercased ``host/path`` form with any
    trailing ``.git`` stripped. The boundary anchor matters: per Q-A (b),
    `pet_latentresidual_fork` and `pet_latentresidual` must compare unequal.

    Supported input forms:
      - ``git@host:path``                       (scp-style ssh)
      - ``ssh://[user[:tok]@]host/path``        (URL-style ssh, optional userinfo)
      - ``https://[user[:tok]@]host/path``      (https, optional userinfo)
      - ``http://[user[:tok]@]host/path``       (plain http — accepted for completeness)
      - bare ``host/path``                      (left as-is then lowercased)

    Notes:
      - Whitespace at the boundaries is ignored.
      - URL fragments (``#...``) and query strings (``?...``) are stripped
        before comparison so that ``host/path.git#frag`` does not silently
        compare unequal to ``host/path``. (B2a, Lane B peer review.)
      - URL userinfo (``user@`` or ``user:token@``) is stripped after the
        scheme is removed, so a CI-injected credential prefix does not
        cause a real-mismatch nor get confused with the scp-style ``:``
        in ``host:path``. (E1, Round-5 external peer review of C2.2.)
      - The function is a pure helper so it can be unit-tested without a
        real repo. Anything beyond the listed forms (e.g. file paths,
        weird custom protocols) returns the raw lowercased input minus a
        trailing ``.git`` — callers must not expect that to be a security
        boundary; the actual matching is a strict equality check after
        normalization.

      ``file://`` and unusual scheme/port forms (e.g. ``ssh://host:22/``)
      are NOT first-class supported; pre-registration only ever runs over
      the canonical https / ssh forms above. (B12 documentation, Lane B.)
    """
    s = url.strip()
    # ssh:// or ssh://git@
    sl = s.lower()
    if sl.startswith("ssh://"):
        s = s[len("ssh://"):]
    # https:// or http://
    elif sl.startswith("https://"):
        s = s[len("https://"):]
    elif sl.startswith("http://"):
        s = s[len("http://"):]

    # E1 (Round-5 peer review): strip URL userinfo (``user@`` or
    # ``user:token@``) if it appears strictly before the first ``/``.
    # Must run BEFORE the scp ``:`` heuristic — otherwise the ``:`` in
    # ``user:token`` triggers a spurious scp-style host/path conversion
    # for ``https://user:token@host/path``. The scp-style ``git@host:path``
    # form is handled by the dedicated branch immediately below; that
    # branch fires when the scheme strip above did NOT match (i.e. the
    # URL had no ``ssh://`` / ``https://`` / ``http://`` prefix), so the
    # leading ``git@`` is still intact at that point.
    slash = s.find("/")
    at = s.find("@")
    if at != -1 and (slash == -1 or at < slash):
        s = s[at + 1:]

    # scp-style ssh prefix: strip leading "git@" if present (without scheme).
    if s.lower().startswith("git@"):
        s = s[len("git@"):]

    # B2a (Lane B peer review): strip URL fragment and query before scp
    # conversion so a stray `#frag` or `?q=v` cannot mask a real mismatch.
    # We strip fragment first (`#`), then query (`?`).
    if "#" in s:
        s = s.split("#", 1)[0]
    if "?" in s:
        s = s.split("?", 1)[0]

    # scp-style host:path → host/path. We do this AFTER scheme/user
    # stripping so we do not mistake the `:` in `https://` for an scp
    # separator. Only convert if the segment BEFORE the first `:` has no
    # `/`, i.e. it looks like a bare hostname (the `host` half of
    # `host:path`). This also handles the anchor-file form
    # `gitee.com:jqu9/PET_LatentResidual` which is identical in shape to
    # the scp-ssh path component minus the `git@` and the `.git`.
    if ":" in s and "/" not in s.split(":", 1)[0]:
        s = s.replace(":", "/", 1)

    # Strip trailing slash first so `host/path.git/` and `host/path.git`
    # both fall through to the `.git`-stripping branch below.
    s = s.rstrip("/")
    # Strip trailing .git boundary
    if s.lower().endswith(".git"):
        s = s[: -len(".git")]
    return s.lower()


def _parse_git_remote_v(stdout: str) -> dict[str, str]:
    """Parse `git remote -v` stdout into {remote_name: fetch_url}.

    Each line of `git remote -v` looks like::

        gitee   git@gitee.com:jqu9/PET_LatentResidual.git (fetch)
        gitee   git@gitee.com:jqu9/PET_LatentResidual.git (push)

    We retain only the (fetch) URL per remote (per-remote fetch and push
    URLs may differ; canonical match is on fetch).

    B4 hardening (Lane B peer review): the URL field is split via
    ``rsplit(None, 2)`` from the right so URLs containing literal
    whitespace (rare but possible via misconfigured ``git remote add``)
    are preserved as a single token. The ``(fetch)``/``(push)`` marker is
    matched case-insensitively to be defensive against future git output
    quirks.
    """
    out: dict[str, str] = {}
    for line in stdout.splitlines():
        line = line.rstrip("\r\n")
        if not line.strip():
            continue
        # rsplit gives us [name+url, url, kind]. Wait — that's wrong:
        # we need [name, url, kind]. Format is name<tab>url<space>(fetch),
        # so a left-anchored split on the first whitespace gives
        # ``name`` then the rest as ``url (fetch)``. We then rsplit
        # the rest from the right with maxsplit=1 to peel off ``(fetch)``.
        head_split = line.split(None, 1)
        if len(head_split) < 2:
            continue
        name, rest = head_split[0], head_split[1]
        url_kind = rest.rsplit(None, 1)
        if len(url_kind) < 2:
            continue
        url, kind = url_kind[0], url_kind[1]
        if kind.lower() == "(fetch)":
            out[name] = url
    return out


def _match_remote_by_anchor(
    anchor: str,
    remotes: dict[str, str],
) -> list[tuple[str, str]]:
    """Return the list of (remote_name, original_url) pairs whose normalized
    URL exactly matches the normalized anchor. Length-0 = no match,
    length-1 = unique resolution, length-≥2 = ambiguous.
    """
    target = _normalize_remote_url(anchor)
    matches: list[tuple[str, str]] = []
    for name, url in remotes.items():
        if _normalize_remote_url(url) == target:
            matches.append((name, url))
    return matches


def _load_anchor(repo_root: Path) -> str:
    """Read and validate `.review_canonical_remote`. Shared by
    `resolve_canonical_remote()` and the `--remote` validation path.

    B3a hardening: opens with ``encoding='utf-8-sig'`` so a stray UTF-8 BOM
    (e.g. from a Windows editor) does not silently break the match.

    B3b hardening (Lane B peer review): a multi-line anchor file is now a
    HARD ERROR rather than a "first non-empty line wins" fallback. Comments
    or stray blank lines that previously slipped through silently — and
    raised a confusing "no remote matches" error downstream — now surface
    immediately with a clear message identifying the offending file.

    Returns the anchor text (the unique non-empty line, stripped). Raises
    `CanonicalRemoteError` on missing / empty / multi-line.
    """
    anchor_path = repo_root / CANONICAL_REMOTE_FILENAME
    if not anchor_path.exists():
        raise CanonicalRemoteError(
            f"missing anchor file `{CANONICAL_REMOTE_FILENAME}` at {repo_root}. "
            f"Create it with the canonical URL fragment, e.g.\n"
            f"    echo 'gitee.com:jqu9/PET_LatentResidual' > "
            f"{anchor_path}\n"
            f"Or pass `--remote <name>` to bypass the resolver."
        )
    # B3a (Lane B BLOCKER): use utf-8-sig so a leading BOM is stripped.
    raw = anchor_path.read_text(encoding="utf-8-sig")
    # Normalise CR/LF line endings before splitting.
    text = raw.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise CanonicalRemoteError(
            f"anchor file {anchor_path} is empty. Write a single line "
            f"containing the canonical URL fragment, "
            f"e.g. `gitee.com:jqu9/PET_LatentResidual`."
        )
    nonempty_lines = [ln for ln in text.split("\n") if ln.strip()]
    # B3b (Lane B peer review): no silent multi-line fallback.
    if len(nonempty_lines) != 1:
        preview = "\n".join(f"    {ln!r}" for ln in nonempty_lines[:3])
        more = (f"\n    ... and {len(nonempty_lines) - 3} more line(s)"
                if len(nonempty_lines) > 3 else "")
        raise CanonicalRemoteError(
            f"anchor file {anchor_path} must contain exactly one "
            f"non-empty line; found {len(nonempty_lines)}:\n"
            f"{preview}{more}\n"
            f"Remove comments / extra lines and retry."
        )
    return nonempty_lines[0].strip()


def resolve_canonical_remote(
    repo_root: Path,
    remotes_provider: "callable[[], dict[str, str]] | None" = None,
) -> tuple[str, str]:
    """Resolve the canonical remote name + URL.

    Args:
      repo_root: the git repo's working tree root.
      remotes_provider: optional injected callable returning ``{name: url}``;
        used by tests to bypass the real ``git remote -v`` invocation. If
        omitted, the function shells out to ``git remote -v`` in
        ``repo_root``.

    Raises:
      CanonicalRemoteError: anchor file missing OR empty OR no remotes
        match OR ≥2 remotes match. The exception message is suitable for
        printing directly; the caller should map it to exit code 6.

    Returns:
      (remote_name, original_fetch_url) on unique match.
    """
    anchor = _load_anchor(repo_root)

    if remotes_provider is not None:
        remotes = remotes_provider()
    else:
        try:
            stdout = git(["remote", "-v"], cwd=repo_root)
        except RuntimeError as exc:
            raise CanonicalRemoteError(
                f"`git remote -v` failed in {repo_root}: {exc}"
            ) from exc
        remotes = _parse_git_remote_v(stdout)

    if not remotes:
        raise CanonicalRemoteError(
            f"no fetch remotes configured in {repo_root}. "
            f"Add one with `git remote add <name> <url>` and retry."
        )

    matches = _match_remote_by_anchor(anchor, remotes)
    if len(matches) == 0:
        listing = "\n".join(
            f"    {n}\t{_safe_repr_url(u)}"
            for n, u in sorted(remotes.items())
        )
        raise CanonicalRemoteError(
            f"no remote matches canonical anchor `{anchor}`. Configured "
            f"fetch remotes:\n{listing}\n"
            f"Add a remote pointing at `{anchor}.git` and retry, or pass "
            f"`--remote <name>` explicitly."
        )
    if len(matches) >= 2:
        listing = "\n".join(
            f"    {n}\t{_safe_repr_url(u)}" for n, u in matches
        )
        raise CanonicalRemoteError(
            f"{len(matches)} remotes match canonical anchor `{anchor}`:\n"
            f"{listing}\n"
            f"Refusing to guess. Pass `--remote <name>` to disambiguate."
        )
    return matches[0]


def find_repo_root(path: Path) -> Path:
    """Find git repo root by walking up from path."""
    cur = path.resolve()
    while cur != cur.parent:
        if (cur / ".git").exists():
            return cur
        cur = cur.parent
    raise RuntimeError(f"No .git found above {path}")


def check_clean_tree(repo_root: Path) -> tuple[bool, str]:
    out = git(["status", "--porcelain"], cwd=repo_root)
    return (out == ""), out


def detect_c_uniform_artifacts(c_dir: Path | None) -> list[Path]:
    """Guard 3: detect any artifact that suggests C_uniform was already eval'd.

    Looks for:
      - eval_*.json or *full_val* files in the C_uniform output dir
      - Method-D-selected ckpt eval outputs
    """
    if c_dir is None:
        return []
    if not c_dir.exists():
        return []
    suspect: list[Path] = []
    patterns = ("*full_val*", "*eval*.json", "*method_d*", "*selected_ckpt*")
    for pat in patterns:
        suspect.extend(c_dir.rglob(pat))
    # Filter out training metrics.jsonl itself (that's allowed to exist;
    # it's only forbidden to have run a SEPARATE full-val on top of it)
    suspect = [p for p in suspect if p.name != "metrics.jsonl"]
    return suspect


def compute_paired_stats_from_rows(
    rows: list[dict],
    step_min: int,
    step_max: int,
) -> tuple[float, float, float, int, list[float]]:
    """Compute paired CV statistics directly from already-loaded metric rows.

    Returns ``(paired_CV, mean, std, N, raw_values)``. The hash is NOT
    returned because the caller already has it from the same single read
    (see `compute_paired_cv` below for the convenience wrapper).

    A2 architectural fix (Lane A peer review, May 4 2026): the lock
    workflow used to re-open `metrics.jsonl` THREE times — once inside
    `compute_paired_cv`, once in the Guard 5 data-side recheck, once
    while building the markdown observation table. The middle read in
    particular re-introduced a small race window that the Cl-Q6 SHA fix
    had explicitly closed. By passing pre-loaded `rows` through this
    helper (and reusing them in `main()`), all three reads collapse to
    a single ``load_metrics_with_hash`` invocation tied to the SHA that
    is embedded in EFFECT_SIZE_LOCKED.md.

    A5 hardening (Lane A peer review): in-window rows whose ``step`` is
    parseable but whose `LOCKED_METRIC_KEY` is missing or non-numeric
    are tallied and surfaced via a `[warn]` line on stderr — never
    silently discarded. Out-of-window rows are correctly ignored.

    F2 hardening (Round-5 operator review, May 4 2026): the warning
    only fires on **val-like** in-window rows. The trainer emits two
    kinds of in-window rows: (1) `event="val"` rows that DO carry
    ``val_select_score`` and are the true measurement, and (2)
    `event="train"` rows (and other non-val schema rows) that NEVER
    carry val metrics. Counting (2) as "skipped" produces a confusing
    33-row warning on perfectly healthy metrics files (operator
    observed this on `A_sanity` tail-window). A row is val-like if
    its ``event`` field equals "val" (case-insensitive). Rows with
    no ``event`` field default to val-like for legacy fixtures
    (e.g. `B_sanity_metrics_val50_schema_reference.jsonl` has no
    explicit event marker but every row IS a val measurement).
    """
    series: list[float] = []
    skipped_no_metric = 0
    skipped_bad_value = 0
    for r in rows:
        step = get_row_step(r)
        if step is None:
            continue
        in_window = step_min <= step <= step_max
        # F2: a row is val-like iff event == "val" or event is absent
        # (legacy fixtures are val-only, so absence defaults to val-like).
        # event == "train" or any other explicit non-val tag means the
        # row is NOT a val measurement and the absence of val_select_score
        # is expected, not suspicious.
        event = r.get("event")
        if event is None:
            is_val_like = True
        elif isinstance(event, str):
            is_val_like = event.lower() == "val"
        else:
            # Defensive: non-string event field is treated as non-val.
            is_val_like = False
        if LOCKED_METRIC_KEY not in r:
            if in_window and is_val_like:
                skipped_no_metric += 1
            continue
        try:
            val = float(r[LOCKED_METRIC_KEY])
        except (TypeError, ValueError):
            if in_window:
                skipped_bad_value += 1
            continue
        if in_window and val > 0.0:
            series.append(val)

    if skipped_no_metric or skipped_bad_value:
        sys.stderr.write(
            f"[warn] compute_paired_cv: skipped {skipped_no_metric} in-window "
            f"row(s) missing `{LOCKED_METRIC_KEY}` and {skipped_bad_value} "
            f"row(s) with non-numeric value (potential trainer partial-write).\n"
        )

    n = len(series)
    if n < 5:
        return 0.0, 0.0, 0.0, n, series
    mean = sum(series) / n
    var = sum((v - mean) ** 2 for v in series) / (n - 1)
    std = math.sqrt(var)
    cv = std / mean if mean > 0 else 0.0
    return cv, mean, std, n, series


def compute_paired_cv(
    metrics_path: Path,
    step_min: int,
    step_max: int,
) -> tuple[float, float, float, int, list[float], str]:
    """Returns (paired_CV, mean, std, N, raw_values, metrics_sha256).

    Backward-compatible convenience wrapper around
    ``compute_paired_stats_from_rows``. Used by tests and any caller that
    only needs the one-shot ``path -> stats`` flow. The lock script's
    ``main()`` itself now uses the row-based path so the metrics file is
    read exactly once per lock.

    The metrics SHA256 is returned from the SAME byte read used to populate
    `raw_values`, eliminating the Cl-Q6 race window between parse-time and
    hash-time.
    """
    rows, sha = load_metrics_with_hash(metrics_path)
    cv, mean, std, n, series = compute_paired_stats_from_rows(
        rows, step_min, step_max
    )
    return cv, mean, std, n, series, sha


def render_lock_md(
    args: argparse.Namespace,
    paired_cv: float,
    mean: float,
    std: float,
    n: int,
    x_value: float,
    metrics_hash: str,
    train_git_commit: str,
    deviation_note: str | None,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    using_default_window = (
        args.step_min == LOCKED_RECOMMENDED_WINDOW[0]
        and args.step_max == LOCKED_RECOMMENDED_WINDOW[1]
    )

    lines: list[str] = []
    lines.append("# EFFECT_SIZE_LOCKED.md — σ-normalize ablation X threshold")
    lines.append("")
    lines.append(
        "**STATUS: LOCKED.** This file is the pre-registration artifact "
        "for [POST_V6_NEXT_STEPS.md §6.6](review/0502/POST_V6_NEXT_STEPS.md). "
        "Modifying any number below = scientific fraud."
    )
    lines.append("")
    lines.append("## Lock-in metadata")
    lines.append("")
    lines.append(f"- **Lock timestamp**: {now}")
    lines.append(f"- **Generated by**: `review/0502/scripts/lock_effect_size_threshold.py`")
    lines.append(f"- **A_main metrics path**: `{args.metrics_a}`")
    lines.append(f"- **A_main metrics SHA256**: `{metrics_hash}`")
    lines.append(f"- **A_main config path**: `{args.config_a}`")
    lines.append(f"- **Repo HEAD at lock time**: `{train_git_commit}`")
    lines.append(f"- **LOCKED_PROTOCOL_VERSION**: `{LOCKED_PROTOCOL_VERSION}`")
    lines.append("")
    lines.append("## Computation")
    lines.append("")
    lines.append(f"- **Formula (LOCKED at commit 867b5c0)**: {LOCKED_FORMULA_CITATION}")
    lines.append(f"- **Metric key**: `{LOCKED_METRIC_KEY}`")
    lines.append(
        f"- **Window**: `step ∈ [{args.step_min}, {args.step_max}]`"
        + ("" if using_default_window
           else f"  ⚠️ **DEVIATION** from LOCKED default "
                f"`[{LOCKED_RECOMMENDED_WINDOW[0]}, "
                f"{LOCKED_RECOMMENDED_WINDOW[1]}]`")
    )
    lines.append(f"- **N (observations in window)**: {n}")
    lines.append(f"- **mean(val_select_score)**: {mean:.6e}")
    lines.append(f"- **std(val_select_score)**: {std:.6e}")
    lines.append(f"- **paired_CV_A** = std / mean = `{paired_cv:.4f}` ({paired_cv * 100:.2f}%)")
    lines.append(
        f"- **3 × paired_CV_A** = `{3.0 * paired_cv:.4f}` ({3.0 * paired_cv * 100:.2f}%)"
    )
    lines.append(f"- **floor (LOCKED)** = `{LOCKED_FLOOR:.4f}` ({LOCKED_FLOOR * 100:.0f}%)")
    lines.append("")
    lines.append("## Result")
    lines.append("")
    lines.append(f"## **X = max(floor, 3 × paired_CV_A) = `{x_value:.4f}` ({x_value * 100:.2f}%)**")
    lines.append("")
    lines.append("## Decision rule (LOCKED, per §6.6.3)")
    lines.append("")
    lines.append(
        f"After Method D + full-val on **C_uniform** top-2 ckpts, compute "
        f"`rel_diff = (mean(C_top2) - mean(A_top2)) / mean(A_top2)`."
    )
    lines.append("")
    lines.append("| `rel_diff`                           | Conclusion / Action                                                                                                                                                                                                                  |")
    lines.append("|---|---|")
    lines.append(
        f"| `rel_diff < {x_value:.4f}` ({x_value * 100:.2f}%)             | **C ≈ A.** σ-norm is the primary lever. Paper claim: \"step-weight shape contributes little beyond scale compensation under the current pair-heavy V6 setup.\"                                          |"
    )
    lines.append(
        f"| `{x_value:.4f} ≤ rel_diff ≤ {2 * x_value:.4f}` ({x_value * 100:.2f}–{2 * x_value * 100:.2f}%)        | **Grey zone.** Run 200K continuation on A_main best.pt + C_uniform best.pt; re-evaluate with the same X. If still grey after 200K: report both numbers, no claim.                                                                       |"
    )
    lines.append(
        f"| `rel_diff > {2 * x_value:.4f}` ({2 * x_value * 100:.2f}%)            | **C ≠ A.** \"After σ-normalization, V6 middle-heavy weighting still helps → genuine hop-shape effect.\" Reframe paper as 'we validate V6 with rigor', σ-norm becomes controlled-comparison methodology.                                  |"
    )
    lines.append("")
    if deviation_note:
        lines.append("## Deviation note")
        lines.append("")
        lines.append(deviation_note)
        lines.append("")
    lines.append("## Raw observations")
    lines.append("")
    lines.append(
        "First / last 5 of the in-window val_select_score series (full series "
        "reproducible from `metrics.jsonl @ {sha256}` and the LOCKED window).".format(
            sha256=metrics_hash[:16]
        )
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()

    # ---- Refuse CLI overrides of LOCKED parameters via undocumented args ----
    # (argparse doesn't expose these; this is just a sanity assertion.)
    assert LOCKED_METRIC_KEY == "val_select_score"
    assert LOCKED_FLOOR == 0.10
    assert LOCKED_SLOPE == 3.0

    # ---- Window deviation: require a deviation note ----
    using_default_window = (
        args.step_min == LOCKED_RECOMMENDED_WINDOW[0]
        and args.step_max == LOCKED_RECOMMENDED_WINDOW[1]
    )
    if not using_default_window and not args.deviation_note:
        sys.stderr.write(
            f"ERROR: --step-min/--step-max differ from LOCKED default "
            f"[{LOCKED_RECOMMENDED_WINDOW[0]}, {LOCKED_RECOMMENDED_WINDOW[1]}]; "
            f"you must pass --deviation-note explaining why. This goes into "
            f"EFFECT_SIZE_LOCKED.md and becomes part of the public pre-reg.\n"
        )
        return 4

    # ---- Resolve paths ----
    output = args.output.resolve()
    metrics_a = args.metrics_a.resolve()
    config_a = args.config_a.resolve()
    repo_root = find_repo_root(output)

    # ---- Guard 2: lock file must not already exist ----
    if output.exists():
        sys.stderr.write(
            f"ERROR (guard 2): {output} already exists. Lock has already been "
            f"performed. Re-running this script = retroactive modification = "
            f"scientific fraud per §6.6.7. If you really need to deviate, "
            f"add a deviation note to the existing file MANUALLY in a new "
            f"git commit; do NOT use this script.\n"
        )
        return 3

    # ---- Guard 1: clean working tree ----
    if not args.allow_dirty:
        clean, dirty_files = check_clean_tree(repo_root)
        if not clean:
            sys.stderr.write(
                f"ERROR (guard 1): working tree is dirty. Lock-in commit must "
                f"be atomic — uncommitted changes in the tree could pollute the "
                f"pre-registration timestamp. Commit or stash first.\n\n"
                f"Dirty files:\n{dirty_files}\n\n"
                f"To bypass (DEV ONLY, never in production), pass --allow-dirty.\n"
            )
            return 2

    # ---- Guard 3: no C_uniform artifacts ----
    suspect = detect_c_uniform_artifacts(args.c_uniform_output_dir)
    if suspect:
        sys.stderr.write(
            f"ERROR (guard 3): SEQUENCE VIOLATION. C_uniform artifacts already "
            f"exist in {args.c_uniform_output_dir} — implies C results were "
            f"already evaluated before X was locked. This destroys the blinded "
            f"analysis pre-registration.\n\n"
            f"Suspect files:\n"
        )
        for p in suspect[:20]:
            sys.stderr.write(f"  {p}\n")
        if len(suspect) > 20:
            sys.stderr.write(f"  ... and {len(suspect) - 20} more\n")
        sys.stderr.write(
            "\nRecovery options:\n"
            "  1. If those artifacts pre-date the A_main run (they're left over "
            "     from an earlier experiment), move/delete them and retry.\n"
            "  2. If they were really computed on the current C_uniform run "
            "     before reading this script, the §6.6 pre-reg is broken; you "
            "     must add a deviation note to POST_V6_NEXT_STEPS.md §6.6 and "
            "     accept reduced reviewer-defensibility.\n"
        )
        return 7

    # ---- Guard 5 (config side per Op-flag-5): config-side best_metric ----
    # Operator round-2.5 reply (`OPERATOR_REPLY_pre_C1_20260503.md` Op-flag-5)
    # clarified that production configs declare `best_metric: val_multi_objective`
    # (config-side) while the metrics row's selection-aware key is
    # `val_select_score` (data-side computed). Both must be checked. The data-
    # side check is performed inside `compute_paired_cv()` via the LOCKED_METRIC_KEY
    # filter (rows missing `val_select_score` are skipped → if 0 rows survive,
    # Guard 4 catches it as N<5).
    cfg_a = load_yaml(config_a)
    cfg_metric_key = cfg_a.get("training", {}).get("best_metric")
    if cfg_metric_key not in (None, "val_multi_objective", "val_select_score"):
        sys.stderr.write(
            f"ERROR (guard 5 config-side): A_main config best_metric="
            f"'{cfg_metric_key}' is unexpected. §6.6 was designed assuming "
            f"val_multi_objective → val_select_score key (see "
            f"POST_V6_NEXT_STEPS.md §0 + OPERATOR_REPLY_pre_C1_20260503.md "
            f"Op-flag-5). Aborting to avoid locking against the wrong run.\n"
        )
        return 4

    # ---- Single read of metrics_a (A2 architectural fix, Lane A peer
    # review). Earlier revisions read this file three times — inside
    # `compute_paired_cv`, in the Guard 5 data-side recheck, and in the
    # markdown observation table — re-introducing a small race window
    # that the SHA-from-same-bytes fix had explicitly closed. We now
    # read once and pass `rows` (and the same `metrics_hash`) through
    # all three downstream consumers.
    rows, metrics_hash = load_metrics_with_hash(metrics_a)

    # ---- Guard 5 (data-side per Op-flag-5): val_select_score must appear
    # in actual metrics rows. Runs BEFORE compute so a schema mismatch
    # surfaces with its own dedicated error rather than being swallowed
    # by Guard 4's generic "N<5" message.
    has_select = any(LOCKED_METRIC_KEY in r for r in rows)
    if not has_select:
        sys.stderr.write(
            f"ERROR (guard 5 data-side): metrics file {metrics_a} contains no "
            f"row with key '{LOCKED_METRIC_KEY}'. Per Op-flag-5 the data-side "
            f"selection key is computed by the trainer (Method-D selector); "
            f"its absence means either the run uses a different selector or "
            f"the metrics file is from a stale schema. Aborting.\n"
        )
        return 4

    # ---- Compute paired_CV_A from already-loaded rows. The SHA captured
    # alongside the byte read above flows directly into the lock MD —
    # parse and hash come from the same in-memory bytes.
    paired_cv, mean, std, n, series = compute_paired_stats_from_rows(
        rows, args.step_min, args.step_max
    )

    # ---- Guard 4: minimum N ----
    if n < 5:
        sys.stderr.write(
            f"ERROR (guard 4): only {n} val_select_score observations in "
            f"window [{args.step_min}, {args.step_max}] of {metrics_a}. "
            f"Need ≥ 5. Either A_main hasn't reached step_max yet, or the "
            f"metric key is missing. Wait for training to progress further.\n"
        )
        return 1

    # ---- Apply LOCKED formula ----
    x_value = max(LOCKED_FLOOR, LOCKED_SLOPE * paired_cv)

    # ---- Compute artifacts ----
    # metrics_hash already obtained from the single read above (race-free).
    try:
        repo_head = git(["rev-parse", "HEAD"], cwd=repo_root)
    except RuntimeError:
        repo_head = "<unknown — git rev-parse failed>"
    md = render_lock_md(
        args=args,
        paired_cv=paired_cv,
        mean=mean,
        std=std,
        n=n,
        x_value=x_value,
        metrics_hash=metrics_hash,
        train_git_commit=repo_head,
        deviation_note=args.deviation_note,
    )

    # Append raw observations (first 5 + last 5). Reuses the same `rows`
    # loaded above — no third file read.
    md_lines = md.rstrip("\n").split("\n")
    md_lines.append("| Index | step | val_select_score |")
    md_lines.append("|---|---|---|")
    in_window: list[tuple[int, float]] = []
    for r in rows:
        step = get_row_step(r)
        if step is None or LOCKED_METRIC_KEY not in r:
            continue
        try:
            val = float(r[LOCKED_METRIC_KEY])
        except (TypeError, ValueError):
            continue
        if args.step_min <= step <= args.step_max and val > 0.0:
            in_window.append((step, val))
    show_rows = in_window[:5] + in_window[-5:] if len(in_window) > 10 else in_window
    for i, (step, val) in enumerate(show_rows):
        md_lines.append(f"| {i} | {step} | {val:.6e} |")
    if len(in_window) > 10:
        md_lines.insert(len(md_lines) - 5, f"| ... | ... | (showing 5+5 of {len(in_window)} obs) |")
    md = "\n".join(md_lines) + "\n"

    # ---- Write file ----
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(md)
    print(f"Written: {output}")
    print()
    print(f"  paired_CV_A = {paired_cv:.4f} ({paired_cv * 100:.2f}%)")
    print(f"  3 × paired_CV_A = {3.0 * paired_cv:.4f}")
    print(f"  floor = {LOCKED_FLOOR}")
    print(f"  X = max(floor, 3·CV) = {x_value:.4f} ({x_value * 100:.2f}%)")
    print(f"  metrics SHA256 = {metrics_hash}")
    print()

    # ---- Commit + push (unless --no-commit) ----
    if args.no_commit:
        print("--no-commit set: file written but NOT committed.")
        print("To finalize the lock-in, commit this file and push.")
        return 0

    rel_output = output.relative_to(repo_root)
    commit_msg = (
        f"review/0502: EFFECT_SIZE_LOCKED.md — X={x_value:.4f} "
        f"(paired_CV_A={paired_cv:.4f}, n={n})\n\n"
        f"Per POST_V6_NEXT_STEPS.md §6.6.2 Step 4. Locks the σ-normalize ablation\n"
        f"effect-size threshold X before any C_uniform full-val eval. This commit\n"
        f"is the pre-registration timestamp evidence.\n\n"
        f"  formula:        max(0.10, 3 × paired_CV_A)\n"
        f"  paired_CV_A:    {paired_cv:.4f} ({paired_cv * 100:.2f}%)\n"
        f"  floor:          {LOCKED_FLOOR:.2f}\n"
        f"  X:              {x_value:.4f} ({x_value * 100:.2f}%)\n"
        f"  window:         step ∈ [{args.step_min}, {args.step_max}]\n"
        f"  metric:         {LOCKED_METRIC_KEY}\n"
        f"  N:              {n}\n"
        f"  metrics SHA256: {metrics_hash}\n"
        f"  LOCKED_PROTOCOL_VERSION: {LOCKED_PROTOCOL_VERSION}\n"
    )
    try:
        git(["add", str(rel_output)], cwd=repo_root)
        # Use stdin to safely handle commit message
        proc = subprocess.run(
            ["git", "commit", "-F", "-"],
            cwd=repo_root,
            input=commit_msg,
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            # D1: sanitize stderr (see _safe_stderr_block).
            sys.stderr.write(
                f"git commit failed: {_safe_stderr_block(proc.stderr)}\n"
            )
            return 8
        commit_hash = git(["rev-parse", "HEAD"], cwd=repo_root)
        print(f"Committed: {commit_hash}")
    except RuntimeError:
        return 8

    if args.no_push:
        print("--no-push set: committed locally but not pushed.")
        print("To finalize the public pre-registration, push to remote.")
        return 0

    # ---- Resolve canonical remote (C2 per Phase-1 review Q-A / Q10) ----
    # If user passed --remote explicitly, that overrides the resolver. This
    # is the documented escape hatch for the ≥2-match ambiguity case.
    if args.remote is not None:
        resolved_remote = args.remote
        # B6 hardening (Lane B HEAVY-FLAG): still verify the supplied name's
        # URL matches the canonical anchor unless --force-unsafe-remote.
        # This catches typos like `--remote orign` that would otherwise push
        # to a non-canonical (or non-existent) remote silently.
        if args.force_unsafe_remote:
            resolved_url = "(unverified, --force-unsafe-remote)"
            print(f"[warn] --force-unsafe-remote: skipping canonical-anchor "
                  f"verification for --remote {resolved_remote!r}")
        else:
            try:
                check_url = git(["remote", "get-url", resolved_remote],
                                cwd=repo_root)
            except RuntimeError as exc:
                sys.stderr.write(
                    f"ERROR: cannot read URL for --remote "
                    f"{resolved_remote!r}: {exc}\n"
                )
                return 6
            try:
                anchor_text = _load_anchor(repo_root)
            except CanonicalRemoteError as exc:
                sys.stderr.write(
                    f"ERROR (--remote anchor verification): {exc}\n"
                )
                return 6
            if _normalize_remote_url(check_url) != _normalize_remote_url(anchor_text):
                sys.stderr.write(
                    f"ERROR: --remote {resolved_remote!r} URL "
                    f"({_safe_repr_url(check_url)}) does not match canonical "
                    f"anchor ({anchor_text!r}). Pass --force-unsafe-remote "
                    f"to override (NOT recommended for pre-registration).\n"
                )
                return 6
            resolved_url = check_url
        print(f"[info] using explicit --remote {resolved_remote!r} \u2192 "
              f"{_safe_repr_url(resolved_url)}")
    else:
        try:
            resolved_remote, resolved_url = resolve_canonical_remote(repo_root)
        except CanonicalRemoteError as exc:
            sys.stderr.write(f"ERROR (canonical-remote resolver): {exc}\n")
            return 6
        print(f"[info] resolved canonical remote: {resolved_remote} \u2192 "
              f"{_safe_repr_url(resolved_url)}")

    # B5 hardening (Lane B BLOCKER): immediately before push, re-read the
    # remote URL and compare to what we resolved. Catches a TOCTOU race
    # where another process did `git remote set-url` between resolution
    # (or argparse) and push, redirecting the pre-registration timestamp
    # to a non-canonical repo. Skip if --force-unsafe-remote.
    #
    # F3 hardening (Round-5 operator review, May 4 2026): also verify
    # the PUSH URL, not only the fetch URL. `git remote -v` and
    # `git remote get-url <remote>` both return the FETCH URL by
    # default. If `remote.<name>.pushURL` is configured separately
    # (e.g. operator points fetch at a read-only mirror but pushes
    # to a write target), `git push <remote> <branch>` will go to
    # the pushURL — bypassing our fetch-URL canonical check entirely.
    # Verify both shapes and require both to canonicalize to the
    # resolved URL.
    if not args.force_unsafe_remote:
        try:
            push_time_url = git(["remote", "get-url", resolved_remote],
                                cwd=repo_root)
        except RuntimeError as exc:
            sys.stderr.write(
                f"ERROR: cannot re-verify remote {resolved_remote!r} "
                f"pre-push: {exc}\n"
            )
            return 6
        if _normalize_remote_url(push_time_url) != _normalize_remote_url(resolved_url):
            sys.stderr.write(
                f"ERROR (TOCTOU): remote {resolved_remote!r} URL changed "
                f"between resolution ({_safe_repr_url(resolved_url)}) and "
                f"push ({_safe_repr_url(push_time_url)}). Refusing to push "
                f"to non-canonical target. You committed locally; "
                f"investigate and retry.\n"
            )
            return 6

        # F3: verify the dedicated pushURL configuration. `--push` makes
        # git report what `git push` would actually use; if no separate
        # pushURL is configured, this returns the fetch URL (so the check
        # is trivially satisfied on the common case). If a pushURL IS
        # configured and disagrees with the resolved canonical, refuse.
        try:
            push_url_explicit = git(
                ["remote", "get-url", "--push", resolved_remote],
                cwd=repo_root,
            )
        except RuntimeError as exc:
            sys.stderr.write(
                f"ERROR: cannot re-verify pushURL of remote "
                f"{resolved_remote!r} pre-push: {exc}\n"
            )
            return 6
        if _normalize_remote_url(push_url_explicit) != _normalize_remote_url(resolved_url):
            sys.stderr.write(
                f"ERROR (pushURL mismatch): remote {resolved_remote!r} "
                f"has a dedicated pushURL "
                f"({_safe_repr_url(push_url_explicit)}) that does not "
                f"canonicalize to the resolved URL "
                f"({_safe_repr_url(resolved_url)}). `git push` would "
                f"silently target the pushURL, bypassing the canonical "
                f"check. Refusing to push. To override (NOT recommended "
                f"for pre-registration), pass --force-unsafe-remote.\n"
            )
            return 6

    try:
        # Detect current branch
        branch = git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
        proc = subprocess.run(
            ["git", "push", resolved_remote, branch],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            # D1: sanitize stderr (see _safe_stderr_block). git's own
            # "fatal: unable to access '<URL>'" echoes the remote URL
            # verbatim, which is the surface B10 was meant to close.
            sys.stderr.write(
                f"git push {_safe_repr_url(resolved_remote)} {branch} failed:\n"
                f"{_safe_stderr_block(proc.stderr)}\n"
                f"You committed locally; push manually to finalize.\n"
            )
            return 8
        print(f"Pushed: {resolved_remote}/{branch}")
    except RuntimeError:
        return 8

    print()
    print("EFFECT_SIZE_LOCKED.md is now public. X is LOCKED.")
    print("You may now proceed to §6.6.2 Step 5 (Method D + full-val on C_uniform).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
