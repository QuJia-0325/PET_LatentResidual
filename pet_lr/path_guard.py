from __future__ import annotations

from pathlib import Path

DATA_DISK_ROOT = Path("/data_2").resolve()
DEFAULT_OUTPUT_ROOT = DATA_DISK_ROOT / "qujiaxiang" / "outputs" / "PET_LatentResidual"


def resolve_data_disk_dir(path_like: str | Path, *, arg_name: str) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (Path.cwd() / path).resolve()
    if DATA_DISK_ROOT != resolved and DATA_DISK_ROOT not in resolved.parents:
        raise RuntimeError(
            f"{arg_name} must be under {DATA_DISK_ROOT}. got: {resolved}. "
            "Do not write outputs into the repository/system disk."
        )
    return resolved


def ensure_repo_local_outputs_absent(repo_root: str | Path) -> None:
    outputs_dir = Path(repo_root).expanduser().resolve() / "outputs"
    if outputs_dir.exists() or outputs_dir.is_symlink():
        raise RuntimeError(
            "Repository-local outputs directory/symlink is not allowed: "
            f"{outputs_dir}. Move artifacts to {DEFAULT_OUTPUT_ROOT} and remove it first."
        )
