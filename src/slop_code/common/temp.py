"""Temporary-directory helpers for host/container interoperability."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

TEMP_ROOT_ENV = "SLOP_CODE_TMPDIR"


def temporary_directory() -> tempfile.TemporaryDirectory[str]:
    """Create a temporary directory under an optional shared host root.

    Colima can bind-mount paths under the macOS user directory, but not the
    default ``/private/var`` temporary tree. Setting ``SLOP_CODE_TMPDIR`` keeps
    workspaces and agent CLI homes in a path visible to the Docker VM.
    """
    configured_root = os.environ.get(TEMP_ROOT_ENV)
    if configured_root is None:
        return tempfile.TemporaryDirectory()

    if not configured_root.strip():
        raise ValueError(
            f"{TEMP_ROOT_ENV} must be a non-empty absolute path"
        )

    configured_path = Path(configured_root).expanduser()
    if not configured_path.is_absolute():
        raise ValueError(f"{TEMP_ROOT_ENV} must be an absolute path")

    root = configured_path.resolve()
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(
            f"{TEMP_ROOT_ENV} must point to a directory: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=root)
