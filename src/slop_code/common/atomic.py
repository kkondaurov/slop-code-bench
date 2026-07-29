"""Fail-closed atomic persistence for benchmark evidence files."""

from __future__ import annotations

import errno
import os
import secrets
import stat
from contextlib import suppress
from pathlib import Path


class UnsafeAtomicWriteError(OSError):
    """Raised when an evidence path could escape through a symlink."""


def _unsafe(message: str, path: Path) -> UnsafeAtomicWriteError:
    return UnsafeAtomicWriteError(errno.ELOOP, message, str(path))


def _open_directory_without_symlinks(path: Path) -> int:
    """Open a directory one component at a time without following links."""
    absolute = path.absolute()
    flags = os.O_RDONLY
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute.anchor, flags)
    try:
        for component in absolute.parts[1:]:
            try:
                next_descriptor = os.open(
                    component,
                    flags,
                    dir_fd=descriptor,
                )
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise _unsafe(
                        "refusing to write through an unsafe parent path",
                        path,
                    ) from exc
                raise
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def atomic_write_text(
    target: Path,
    content: str,
    *,
    encoding: str = "utf-8",
) -> None:
    """Atomically replace ``target`` without following target or parent links.

    The caller must create the parent directory first. A directory descriptor
    pins that parent for the complete operation, while ``O_EXCL`` and
    ``O_NOFOLLOW`` make the randomly named temporary file safe even if an
    untrusted pre-existing run directory contains a symlink trap.
    """
    parent = target.parent
    if parent.is_symlink():
        raise _unsafe("refusing to write through a symlink parent", parent)
    if target.is_symlink():
        raise _unsafe("refusing to replace a symlink target", target)

    try:
        parent_fd = _open_directory_without_symlinks(parent)
    except OSError as exc:
        if parent.is_symlink():
            raise _unsafe(
                "refusing to open a symlink parent",
                parent,
            ) from exc
        raise

    temporary_name = f".{target.name}.{secrets.token_hex(12)}.tmp"
    temporary_fd: int | None = None
    try:
        try:
            target_stat = os.stat(
                target.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            target_stat = None
        if target_stat is not None and stat.S_ISLNK(target_stat.st_mode):
            raise _unsafe("refusing to replace a symlink target", target)

        open_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        open_flags |= getattr(os, "O_NOFOLLOW", 0)
        temporary_fd = os.open(
            temporary_name,
            open_flags,
            0o666,
            dir_fd=parent_fd,
        )
        encoded = content.encode(encoding)
        with os.fdopen(temporary_fd, "wb", closefd=True) as handle:
            temporary_fd = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

        # Re-check the target under the pinned directory immediately before
        # replace. ``os.replace`` replaces a link itself; it never follows it.
        try:
            target_stat = os.stat(
                target.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            target_stat = None
        if target_stat is not None and stat.S_ISLNK(target_stat.st_mode):
            raise _unsafe("refusing to replace a symlink target", target)

        os.replace(
            temporary_name,
            target.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        os.fsync(parent_fd)
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        with suppress(FileNotFoundError):
            os.unlink(temporary_name, dir_fd=parent_fd)
        os.close(parent_fd)
