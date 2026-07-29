"""Temporary-directory helpers for host/container interoperability."""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path

TEMP_ROOT_ENV = "SLOP_CODE_TMPDIR"
NAMED_PROFILE_TEMP_DIR = Path("tmp") / "scbench-v2"


def configure_named_profile_temp_root(repository_root: Path) -> Path:
    """Configure a Docker-visible temporary root for a named profile.

    Named benchmark profiles are expected to run in Docker on macOS as well as
    Linux. macOS Docker VMs commonly cannot bind-mount the default
    ``/private/var`` temporary tree, so use a repository-local path when the
    caller has not supplied an explicit override. Explicit overrides are never
    replaced and are validated by :func:`temporary_root_path`.
    """
    if TEMP_ROOT_ENV in os.environ:
        # Explicit absolute overrides may intentionally point through a host
        # symlink to a Docker-shared volume. Preserve that existing contract.
        return temporary_root_path()

    # Deliberately preserve lexical components so a symlink cannot disappear
    # before the component-by-component lstat validation below.
    repository = Path(os.path.abspath(repository_root))  # noqa: PTH100
    automatic_root = repository / NAMED_PROFILE_TEMP_DIR
    current = Path(automatic_root.anchor)
    for part in automatic_root.parts[1:]:
        current /= part
        try:
            current_stat = current.lstat()
        except FileNotFoundError:
            # Another run may create the same component after our lstat.
            # ``exist_ok`` closes that benign first-use race; the lstat below
            # still rejects a concurrently inserted symlink or non-directory.
            current.mkdir(exist_ok=True)
            current_stat = current.lstat()
        if stat.S_ISLNK(current_stat.st_mode):
            raise ValueError(
                "Automatic named-profile temporary root cannot traverse "
                f"a symlink: {current}"
            )
        if not stat.S_ISDIR(current_stat.st_mode):
            raise NotADirectoryError(
                "Automatic named-profile temporary root component is not "
                f"a directory: {current}"
            )

    resolved_repository = repository.resolve(strict=True)
    resolved_root = automatic_root.resolve(strict=True)
    if not resolved_root.is_relative_to(resolved_repository):
        raise ValueError(
            "Automatic named-profile temporary root escapes the repository: "
            f"{resolved_root}"
        )
    os.environ[TEMP_ROOT_ENV] = str(resolved_root)
    return resolved_root


def temporary_root_path() -> Path:
    """Return the validated root for benchmark-owned temporary files."""
    configured_root = os.environ.get(TEMP_ROOT_ENV)
    if configured_root is None:
        return Path(tempfile.gettempdir()).resolve()

    if not configured_root.strip():
        raise ValueError(f"{TEMP_ROOT_ENV} must be a non-empty absolute path")

    configured_path = Path(configured_root).expanduser()
    if not configured_path.is_absolute():
        raise ValueError(f"{TEMP_ROOT_ENV} must be an absolute path")

    root = configured_path.resolve()
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(
            f"{TEMP_ROOT_ENV} must point to a directory: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    return root


def temporary_directory() -> tempfile.TemporaryDirectory[str]:
    """Create a temporary directory under an optional shared host root.

    Colima can bind-mount paths under the macOS user directory, but not the
    default ``/private/var`` temporary tree. Setting ``SLOP_CODE_TMPDIR`` keeps
    workspaces, snapshot archives, and agent CLI homes in a path visible to the
    Docker VM.
    """
    return tempfile.TemporaryDirectory(dir=temporary_root_path())
