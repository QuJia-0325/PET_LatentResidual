#!/usr/bin/env bash
# review/0502/scripts/apply_patches.sh
#
# Apply sigma-normalize ablation patches to repo working tree.
# Idempotent: re-running after success is a no-op (patch will refuse).
#
# Usage:
#   cd <repo root>            # i.e. PET_LatentResidual/
#   bash review/0502/scripts/apply_patches.sh             # apply (fails on dirty target files)
#   bash review/0502/scripts/apply_patches.sh --force     # apply even if target files are dirty
#   bash review/0502/scripts/apply_patches.sh --revert    # revert
#   bash review/0502/scripts/apply_patches.sh --check     # dry-run only
#
# Required tools: git, patch, python3 (for verify step).

set -euo pipefail

# Resolve repo root (script lives in review/0502/scripts/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="$(cd "${SCRIPT_DIR}/../patches" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"

# Sanity: confirm we are inside the right repo.
for f in pet_lr/rollout_first_hop.py train_first_hop.py; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: cannot find ${f}; please run from PET_LatentResidual/ root." >&2
        exit 1
    fi
done

PATCHES=(
    "${PATCH_DIR}/rollout_first_hop.py.patch"
    "${PATCH_DIR}/train_first_hop.py.patch"
)

MODE="${1:-apply}"
FORCE=0
if [[ "${MODE}" == "--force" || "${MODE}" == "force" ]]; then
    FORCE=1
    MODE="apply"
fi

case "${MODE}" in
    apply)
        # Fail if working tree is dirty in the target files (do NOT clobber uncommitted work).
        # `git status --porcelain` emits one line per changed file (' M ...' for unstaged,
        # 'M  ...' for staged, '?? ...' for untracked). Any non-empty line means dirty.
        dirty=$(git status --porcelain pet_lr/rollout_first_hop.py train_first_hop.py 2>/dev/null || true)
        if [[ -n "${dirty}" ]]; then
            if [[ "${FORCE}" -eq 1 ]]; then
                echo "WARN: target files have uncommitted changes; --force given, applying with patch -p1 --merge." >&2
                echo "${dirty}" >&2
            else
                echo "ERROR: target files have uncommitted changes:" >&2
                echo "${dirty}" >&2
                echo "" >&2
                echo "Refusing to apply patches over a dirty tree. Either:" >&2
                echo "  (a) commit/stash the existing changes first, then re-run, or" >&2
                echo "  (b) run with --force to merge anyway." >&2
                exit 1
            fi
        fi
        echo "[apply_patches.sh] Verifying divisor math..."
        python3 review/0502/scripts/verify_normalizers.py >/dev/null
        echo "[apply_patches.sh] Applying patches with --fuzz=10 (line drift tolerance)..."
        for p in "${PATCHES[@]}"; do
            echo "  -> $(basename "$p")"
            patch -p1 --fuzz=10 < "$p"
        done
        echo "[apply_patches.sh] Patches applied. Run 'git diff' to review changes."
        ;;
    --check|check)
        for p in "${PATCHES[@]}"; do
            echo "  [check] $(basename "$p")"
            patch -p1 --fuzz=10 --dry-run < "$p"
        done
        ;;
    --revert|revert)
        # Reverse-apply patches in opposite order.
        for ((i=${#PATCHES[@]}-1; i>=0; i--)); do
            p="${PATCHES[$i]}"
            echo "  [revert] $(basename "$p")"
            patch -p1 --fuzz=10 -R < "$p"
        done
        echo "[apply_patches.sh] Patches reverted."
        ;;
    *)
        echo "Usage: $0 [apply|--force|--check|--revert]" >&2
        exit 2
        ;;
esac
