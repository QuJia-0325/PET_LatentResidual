from __future__ import annotations

import sys
from pathlib import Path


def resolve_rae_code_path(rae_root: str | Path) -> Path:
    root = Path(rae_root).expanduser().resolve()
    code_path = root / "code" / "RAE"
    if not code_path.exists():
        raise FileNotFoundError(f"RAE code path not found: {code_path}")
    return code_path


def ensure_rae_importable(rae_root: str | Path) -> Path:
    code_path = resolve_rae_code_path(rae_root)
    code_str = str(code_path)
    if code_str not in sys.path:
        sys.path.insert(0, code_str)
    return code_path
