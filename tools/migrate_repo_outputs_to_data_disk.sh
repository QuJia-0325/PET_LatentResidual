#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src="${repo_root}/outputs"
dst="/data_2/qujiaxiang/outputs/PET_LatentResidual"

if [ ! -e "$src" ] && [ ! -L "$src" ]; then
  echo "No repository-local outputs directory found: $src"
  exit 0
fi

if [ -L "$src" ]; then
  echo "Repository-local outputs symlink is not allowed: $src" >&2
  echo "Remove the symlink after confirming artifacts already live under $dst" >&2
  exit 1
fi

mkdir -p "$dst"
shopt -s dotglob nullglob
for item in "$src"/*; do
  base="$(basename "$item")"
  if [ -e "$dst/$base" ] || [ -L "$dst/$base" ]; then
    echo "Refusing to overwrite existing destination: $dst/$base" >&2
    exit 1
  fi
  mv "$item" "$dst/$base"
done
rmdir "$src"
echo "Moved repository-local outputs into: $dst"
