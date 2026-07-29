"""Snapshot and diff functionality for execution environments.

This module provides filesystem state capture and comparison capabilities:

- **Snapshot**: Create compressed archives of directory state with file filtering
- **SnapshotDiff**: Compare snapshots to produce line-by-line file diffs (text files only)
- **File type detection**: Automatically handles text vs binary files
- **Glob filtering**: Include/exclude files using glob patterns

Example:
    >>> from pathlib import Path
    >>> from slop_code.execution.snapshot import Snapshot
    >>>
    >>> before = Snapshot.from_directory(cwd=Path("workspace"), env={})
    >>> # ... make changes ...
    >>> after = Snapshot.from_directory(cwd=Path("workspace"), env={})
    >>>
    >>> diff = before.diff(after)
    >>> for path, file_diff in diff.file_diffs.items():
    ...     print(f"{path}: {file_diff.change_type}")

See Also:
    - docs/execution/snapshots.md for detailed usage guide
"""

from __future__ import annotations

import contextlib
import difflib
import fnmatch
import hashlib
import hmac
import os
import shutil
import stat
import tarfile
from collections.abc import Generator
from collections.abc import Iterable
from datetime import datetime
from enum import Enum
from pathlib import Path
from pathlib import PurePosixPath
from pathlib import PureWindowsPath
from typing import BinaryIO

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from slop_code.execution.assets import ResolvedStaticAsset
from slop_code.execution.models import EnvironmentSpec
from slop_code.execution.models import ExecutionError
from slop_code.logging import get_logger

logger = get_logger(__name__)

DEFAULT_ARCHIVE_NAME = "slop_code_snapshot"
SYMLINK_TARGET_TYPE_PAX = "SLOPCODE.symlink_target_type"
ATIME_NS_PAX = "SLOPCODE.atime_ns"
MTIME_NS_PAX = "SLOPCODE.mtime_ns"
ROOT_MEMBER_NAME = "."

IS_WINDOWS = os.name == "nt"


def _normalize_compression(compression: str) -> tuple[str, str]:
    """Convert compression name to tarfile mode and file extension.

    Args:
        compression: Compression algorithm name ("gz", "bz2", "xz", "none", or "").

    Returns:
        Tuple of (tar_mode, file_extension) for use with tarfile module.

    Raises:
        ValueError: If compression algorithm is not supported.

    Example:
        >>> _normalize_compression("gz")
        ('w:gz', '.tar.gz')
        >>> _normalize_compression("xz")
        ('w:xz', '.tar.xz')
    """
    compression = (compression or "").lower()
    mode_map = {
        "gz": "w:gz",
        "bz2": "w:bz2",
        "xz": "w:xz",
        "": "w",
        "none": "w",
    }
    if compression not in mode_map:
        raise ValueError(
            f"Unsupported compression '{compression}'. "
            f"Choose one of {list(mode_map.keys())}."
        )
    tar_mode = mode_map[compression]
    ext_map = {
        "w:gz": ".tar.gz",
        "w:bz2": ".tar.bz2",
        "w:xz": ".tar.xz",
        "w": ".tar",
    }
    return tar_mode, ext_map[tar_mode]


def _tar_read_mode(compression: str) -> str:
    """Convert compression name to tarfile read mode.

    Args:
        compression: Compression algorithm name ("gz", "bz2", "xz", "none", or "").

    Returns:
        Tarfile read mode string (e.g., "r:gz", "r:xz").

    Raises:
        ValueError: If compression algorithm is not supported.
    """

    compression = (compression or "").lower()
    mode_map = {
        "gz": "r:gz",
        "bz2": "r:bz2",
        "xz": "r:xz",
        "": "r:",
        "none": "r:",
    }
    if compression not in mode_map:
        raise ValueError(
            f"Unsupported compression '{compression}'. "
            f"Choose one of {list(mode_map.keys())}."
        )
    return mode_map[compression]


def _resolve_archive_path(
    cwd: Path, save_dir: Path | None, tar_ext: str
) -> Path:
    """Determine the full path for saving a snapshot archive.

    Creates a unique filename using timestamp and UUID to prevent collisions.
    If save_dir is None, saves in cwd. Otherwise saves in save_dir with
    the workspace name.

    Args:
        cwd: The workspace directory being snapshotted.
        save_dir: Optional directory to save archive. If None, saves in cwd.
        tar_ext: File extension including compression (e.g., ".tar.gz").

    Returns:
        Resolved absolute path for the archive file.

    Raises:
        FileExistsError: If save_dir exists but is not a directory.

    Example:
        >>> cwd = Path("/workspace")
        >>> save_dir = Path("/snapshots")
        >>> path = _resolve_archive_path(cwd, save_dir, ".tar.gz")
        >>> # Returns: /snapshots/20251010T140732.abc123de.workspace.tar.gz
    """
    import uuid

    now = datetime.now().strftime("%Y%m%dT%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    if save_dir is None:
        return (
            cwd / f"{now}.{unique_id}.{DEFAULT_ARCHIVE_NAME}{tar_ext}"
        ).resolve()

    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    elif not save_dir.is_dir():
        raise FileExistsError(
            f"Save directory already exists and is not a directory: {save_dir}"
        )
    return (save_dir / f"{now}.{unique_id}.{cwd.name}{tar_ext}").resolve()


def _matches_any(patterns: Iterable[str], rel_posix: str) -> bool:
    """Return True when any glob pattern matches the relative path.

    Some user-provided glob patterns (for example ``"**/*"``) expect paths to
    contain at least one separator. When evaluating files located at the
    workspace root (e.g. ``"main.py"``), those patterns would otherwise fail to
    match because the relative path lacks a separator. To make the matching
    behaviour align with common shell-style expectations, we evaluate both the
    raw relative path and a ``"./"``-prefixed variant against each pattern.
    """

    candidates = (rel_posix, f"./{rel_posix}")
    for pattern in patterns:
        for candidate in candidates:
            if fnmatch.fnmatch(candidate, pattern):
                return True
    return False


def _walk_candidates(
    cwd: Path,
    ignore_globs: set[str],
    keep_globs: set[str],
) -> tuple[set[Path], set[Path], set[Path], set[Path]]:
    other_paths: set[Path] = set()
    matched_paths: set[Path] = set()
    symlink_directory_paths: set[Path] = set()
    candidate_directory_paths: set[Path] = set()

    for root, dirs, files in os.walk(cwd, topdown=True, followlinks=False):
        root_path = Path(root)

        # Prune directories early (using trailing slash to match dir globs)
        kept_dirs = []
        for d in list(dirs):
            abs_dir = root_path / d
            rel_path = abs_dir.relative_to(cwd)
            rel_dir = rel_path.as_posix() + "/"
            if _matches_any(ignore_globs, rel_dir):
                continue
            if abs_dir.is_symlink():
                if not keep_globs or _matches_any(keep_globs, rel_dir):
                    matched_paths.add(rel_path)
                    symlink_directory_paths.add(rel_path)
                else:
                    other_paths.add(rel_path)
                continue
            candidate_directory_paths.add(rel_path)
            kept_dirs.append(d)
        dirs[:] = kept_dirs

        # Collect files
        for f in files:
            abs_path = root_path / f
            rel_path = abs_path.relative_to(cwd)
            rel_posix = rel_path.as_posix()

            if _matches_any(ignore_globs, rel_posix):
                other_paths.add(rel_path)
                continue

            if not keep_globs or _matches_any(keep_globs, rel_posix):
                matched_paths.add(rel_path)

    if keep_globs:
        directory_paths = {
            path
            for path in candidate_directory_paths
            if _matches_any(keep_globs, f"{path.as_posix()}/")
        }
        for matched_path in matched_paths:
            for parent in matched_path.parents:
                if parent == Path():
                    break
                if parent in candidate_directory_paths:
                    directory_paths.add(parent)
        other_paths.update(candidate_directory_paths - directory_paths)
    else:
        directory_paths = candidate_directory_paths

    return (
        matched_paths,
        other_paths,
        symlink_directory_paths,
        directory_paths,
    )


def _safe_archive_member_path(name: str) -> Path:
    """Validate and convert an archive member path without host traversal."""
    pure_path = PurePosixPath(name)
    windows_path = PureWindowsPath(name)
    if (
        pure_path.is_absolute()
        or not pure_path.parts
        or "\\" in name
        or windows_path.is_absolute()
        or bool(windows_path.drive)
    ):
        raise ExecutionError(f"Unsafe snapshot archive member: {name!r}")
    if any(part in {"", ".", ".."} for part in pure_path.parts):
        raise ExecutionError(f"Unsafe snapshot archive member: {name!r}")
    return Path(*pure_path.parts)


def _safe_relative_symlink_target(link_path: Path, target: str) -> str:
    """Reject absolute or workspace-escaping symlink targets."""
    pure_target = PurePosixPath(target)
    windows_target = PureWindowsPath(target)
    if (
        pure_target.is_absolute()
        or not pure_target.parts
        or "\\" in target
        or windows_target.is_absolute()
        or bool(windows_target.drive)
    ):
        raise ExecutionError(
            f"Unsafe snapshot symlink target for {link_path}: {target!r}"
        )

    resolved_parts = list(PurePosixPath(link_path.as_posix()).parent.parts)
    for part in pure_target.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not resolved_parts:
                raise ExecutionError(
                    "Snapshot symlink escapes the workspace: "
                    f"{link_path} -> {target}"
                )
            resolved_parts.pop()
            continue
        resolved_parts.append(part)
    return target


def _ensure_safe_output_parent(target_dir: Path, out_path: Path) -> None:
    """Create archive parents while refusing traversal through symlinks."""
    relative = out_path.relative_to(target_dir)
    current = target_dir
    for part in relative.parts[:-1]:
        current /= part
        if current.is_symlink():
            raise ExecutionError(
                f"Snapshot extraction parent is a symlink: {current}"
            )
        current.mkdir(exist_ok=True)


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    """Return metadata that changes when a captured path is replaced/mutated."""
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


@contextlib.contextmanager
def _open_parent_directory_fd(
    cwd: Path,
    rel_path: Path,
) -> Generator[int | None, None, None]:
    """Open every parent with no-follow semantics when the platform supports it."""
    supports_dir_fd = os.open in os.supports_dir_fd
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if not supports_dir_fd or not no_follow or not directory:
        current = cwd
        if current.is_symlink():
            raise ExecutionError(f"Snapshot root is a symlink: {cwd}")
        for part in rel_path.parts[:-1]:
            current /= part
            if current.is_symlink():
                raise ExecutionError(
                    f"Snapshot path parent is a symlink: {current}"
                )
        yield None
        return

    opened: list[int] = []
    flags = os.O_RDONLY | no_follow | directory | getattr(os, "O_CLOEXEC", 0)
    try:
        current_fd = os.open(cwd, flags)
        opened.append(current_fd)
        for part in rel_path.parts[:-1]:
            current_fd = os.open(part, flags, dir_fd=current_fd)
            opened.append(current_fd)
        yield current_fd
    except OSError as exc:
        raise ExecutionError(
            f"Snapshot path changed or traverses a symlink: {rel_path}"
        ) from exc
    finally:
        for descriptor in reversed(opened):
            os.close(descriptor)


@contextlib.contextmanager
def _open_regular_file_no_follow(
    cwd: Path,
    rel_path: Path,
) -> Generator[tuple[BinaryIO, os.stat_result], None, None]:
    """Pin a regular file descriptor without following the final component."""
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    if no_follow:
        flags |= no_follow

    with _open_parent_directory_fd(cwd, rel_path) as parent_fd:
        try:
            if parent_fd is None:
                descriptor = os.open(cwd / rel_path, flags)
            else:
                descriptor = os.open(
                    rel_path.name,
                    flags,
                    dir_fd=parent_fd,
                )
        except OSError as exc:
            raise ExecutionError(
                f"Snapshot file changed or became a symlink: {rel_path}"
            ) from exc

        try:
            source_stat = os.fstat(descriptor)
            if not stat.S_ISREG(source_stat.st_mode):
                raise ExecutionError(
                    f"Unsupported snapshot file type: {rel_path}"
                )
            with os.fdopen(descriptor, "rb", closefd=False) as source:
                yield source, source_stat
        finally:
            os.close(descriptor)


@contextlib.contextmanager
def _open_directory_no_follow(
    cwd: Path,
    rel_path: Path,
) -> Generator[tuple[int | None, os.stat_result], None, None]:
    """Pin a real directory without following its final component."""
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if os.open not in os.supports_dir_fd or not no_follow or not directory:
        path = cwd / rel_path
        try:
            before = path.lstat()
        except OSError as exc:
            raise ExecutionError(
                f"Snapshot directory changed or became a symlink: {rel_path}"
            ) from exc
        if not stat.S_ISDIR(before.st_mode):
            raise ExecutionError(
                f"Unsupported snapshot directory type: {rel_path}"
            )
        yield None, before
        try:
            after = path.lstat()
        except OSError as exc:
            raise ExecutionError(
                f"Snapshot directory changed during capture: {rel_path}"
            ) from exc
        if _stat_identity(before) != _stat_identity(after):
            raise ExecutionError(
                f"Snapshot directory changed during capture: {rel_path}"
            )
        return

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | directory
        | no_follow
    )
    with _open_parent_directory_fd(cwd, rel_path) as parent_fd:
        try:
            if parent_fd is None:
                descriptor = os.open(cwd / rel_path, flags)
            else:
                descriptor = os.open(
                    rel_path.name,
                    flags,
                    dir_fd=parent_fd,
                )
        except OSError as exc:
            raise ExecutionError(
                f"Snapshot directory changed or became a symlink: {rel_path}"
            ) from exc

        try:
            source_stat = os.fstat(descriptor)
            if not stat.S_ISDIR(source_stat.st_mode):
                raise ExecutionError(
                    f"Unsupported snapshot directory type: {rel_path}"
                )
            yield descriptor, source_stat
            after = os.fstat(descriptor)
            if _stat_identity(source_stat) != _stat_identity(after):
                raise ExecutionError(
                    f"Snapshot directory changed during capture: {rel_path}"
                )
        finally:
            os.close(descriptor)


def _read_symlink_no_follow(
    cwd: Path,
    rel_path: Path,
) -> tuple[str, os.stat_result]:
    """Read a symlink through a pinned parent and reject concurrent replacement."""
    with _open_parent_directory_fd(cwd, rel_path) as parent_fd:
        try:
            if parent_fd is None:
                before = (cwd / rel_path).lstat()
                target = str((cwd / rel_path).readlink())
                after = (cwd / rel_path).lstat()
            else:
                before = os.stat(
                    rel_path.name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                target = os.readlink(rel_path.name, dir_fd=parent_fd)
                after = os.stat(
                    rel_path.name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
        except OSError as exc:
            raise ExecutionError(
                f"Snapshot symlink changed during capture: {rel_path}"
            ) from exc
    if not stat.S_ISLNK(before.st_mode) or (
        _stat_identity(before) != _stat_identity(after)
    ):
        raise ExecutionError(
            f"Snapshot symlink changed during capture: {rel_path}"
        )
    return target, before


def _tar_info_from_stat(
    rel_path: Path,
    source_stat: os.stat_result,
    *,
    symlink_target: str | None = None,
    symlink_target_is_directory: bool = False,
) -> tarfile.TarInfo:
    """Build archive metadata from descriptor-pinned filesystem metadata."""
    info = tarfile.TarInfo(rel_path.as_posix())
    info.mode = stat.S_IMODE(source_stat.st_mode)
    info.uid = source_stat.st_uid
    info.gid = source_stat.st_gid
    info.mtime = source_stat.st_mtime
    info.pax_headers[ATIME_NS_PAX] = str(source_stat.st_atime_ns)
    info.pax_headers[MTIME_NS_PAX] = str(source_stat.st_mtime_ns)
    if stat.S_ISDIR(source_stat.st_mode):
        info.type = tarfile.DIRTYPE
        info.size = 0
    elif symlink_target is None:
        info.type = tarfile.REGTYPE
        info.size = source_stat.st_size
    else:
        info.type = tarfile.SYMTYPE
        info.size = 0
        info.linkname = symlink_target
        info.pax_headers[SYMLINK_TARGET_TYPE_PAX] = (
            "directory" if symlink_target_is_directory else "file"
        )
    return info


def _prepare_safe_extraction_root(target_dir: Path) -> Path:
    """Create an extraction root without accepting symlinked components."""
    absolute = target_dir.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            current_stat = current.lstat()
        except FileNotFoundError:
            current.mkdir()
            current_stat = current.lstat()
        if stat.S_ISLNK(current_stat.st_mode):
            raise ExecutionError(
                f"Snapshot extraction root traverses a symlink: {current}"
            )
        if not stat.S_ISDIR(current_stat.st_mode):
            raise ExecutionError(
                f"Snapshot extraction root is not a directory: {current}"
            )
    return absolute


def _supports_descriptor_safe_extraction() -> bool:
    """Return whether the host exposes the no-follow dir-fd primitives used."""
    return bool(
        getattr(os, "O_NOFOLLOW", 0)
        and getattr(os, "O_DIRECTORY", 0)
        and os.open in os.supports_dir_fd
        and os.mkdir in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
        and os.unlink in os.supports_dir_fd
        and os.symlink in os.supports_dir_fd
        and os.utime in os.supports_dir_fd
        and os.utime in os.supports_follow_symlinks
    )


@contextlib.contextmanager
def _open_safe_extraction_root(
    target_dir: Path,
) -> Generator[tuple[Path, int | None], None, None]:
    """Create and pin an extraction root without traversing symlinks."""
    absolute = Path(os.path.abspath(target_dir))  # noqa: PTH100
    if not _supports_descriptor_safe_extraction():
        yield _prepare_safe_extraction_root(absolute), None
        return

    flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
    )
    current_fd: int | None = None
    try:
        current_fd = os.open(absolute.anchor, flags)
        for part in absolute.parts[1:]:
            if part in {"", ".", ".."}:
                raise ExecutionError(
                    f"Unsafe snapshot extraction root component: {part!r}"
                )
            with contextlib.suppress(FileExistsError):
                os.mkdir(part, dir_fd=current_fd)
            try:
                next_fd = os.open(part, flags, dir_fd=current_fd)
            except OSError as exc:
                raise ExecutionError(
                    "Snapshot extraction root changed or traverses a symlink: "
                    f"{absolute}"
                ) from exc
            os.close(current_fd)
            current_fd = next_fd
        yield absolute, current_fd
    finally:
        if current_fd is not None:
            os.close(current_fd)


@contextlib.contextmanager
def _open_safe_output_parent(
    root_fd: int,
    relative_path: Path,
) -> Generator[int, None, None]:
    """Open/create an archive member's parent beneath a pinned root."""
    flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
    )
    current_fd = os.dup(root_fd)
    try:
        for part in relative_path.parts[:-1]:
            with contextlib.suppress(FileExistsError):
                os.mkdir(part, dir_fd=current_fd)
            try:
                next_fd = os.open(part, flags, dir_fd=current_fd)
            except OSError as exc:
                raise ExecutionError(
                    "Snapshot extraction parent changed or is a symlink: "
                    f"{relative_path.parent}"
                ) from exc
            os.close(current_fd)
            current_fd = next_fd
        yield current_fd
    finally:
        os.close(current_fd)


def _remove_existing_output(parent_fd: int, name: str) -> None:
    """Unlink a non-directory final component relative to a pinned parent."""
    try:
        existing = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(existing.st_mode):
        raise ExecutionError(
            f"Snapshot archive member conflicts with directory: {name}"
        )
    # A concurrent removal is benign. Creation below still uses O_EXCL.
    with contextlib.suppress(FileNotFoundError):
        os.unlink(name, dir_fd=parent_fd)


def _parse_exact_integer(
    encoded: str,
    *,
    field: str,
    path: str,
) -> int:
    """Parse canonical integer metadata without accepting ambiguous text."""
    try:
        value = int(encoded)
    except (TypeError, ValueError) as exc:
        raise ExecutionError(
            f"Invalid snapshot {field} metadata for {path}: {encoded!r}"
        ) from exc
    if str(value) != encoded:
        raise ExecutionError(
            f"Invalid snapshot {field} metadata for {path}: {encoded!r}"
        )
    return value


def _member_mtime_ns(member: tarfile.TarInfo) -> int:
    """Read the exact captured mtime, including nanoseconds."""
    encoded = member.pax_headers.get(MTIME_NS_PAX)
    if encoded is None:
        return int(float(member.mtime) * 1_000_000_000)
    return _parse_exact_integer(
        encoded,
        field="mtime",
        path=member.name,
    )


def _member_atime_ns(member: tarfile.TarInfo) -> int:
    """Read exact captured atime, falling back for legacy snapshots."""
    encoded = member.pax_headers.get(ATIME_NS_PAX)
    if encoded is None:
        return _member_mtime_ns(member)
    return _parse_exact_integer(
        encoded,
        field="atime",
        path=member.name,
    )


@contextlib.contextmanager
def _open_verified_archive(
    archive_path: Path,
    expected_checksum: str,
) -> Generator[BinaryIO, None, None]:
    """Verify an archive and parse it through the same open descriptor."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(archive_path, flags)
    except OSError as exc:
        raise ExecutionError(
            f"Could not safely open snapshot archive: {archive_path}"
        ) from exc

    try:
        source_stat = os.fstat(descriptor)
        if not stat.S_ISREG(source_stat.st_mode):
            raise ExecutionError(
                f"Snapshot archive is not a regular file: {archive_path}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            digest = hashlib.md5(usedforsecurity=False)
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(descriptor)
            if _stat_identity(source_stat) != _stat_identity(after):
                raise ExecutionError(
                    f"Snapshot archive changed while verifying: {archive_path}"
                )
            actual_checksum = digest.hexdigest()
            if not hmac.compare_digest(actual_checksum, expected_checksum):
                raise ExecutionError(
                    "Snapshot archive checksum mismatch: "
                    f"expected {expected_checksum}, got {actual_checksum}"
                )
            source.seek(0)
            yield source
    finally:
        os.close(descriptor)


def _apply_descriptor_metadata(
    descriptor: int,
    member: tarfile.TarInfo,
    *,
    uid: int | None,
    gid: int | None,
) -> None:
    """Restore owner, mode, and exact mtime through a pinned descriptor."""
    if uid is not None and gid is not None:
        os.fchown(descriptor, uid, gid)
    os.fchmod(descriptor, member.mode & 0o777)
    atime_ns = _member_atime_ns(member)
    mtime_ns = _member_mtime_ns(member)
    os.utime(descriptor, ns=(atime_ns, mtime_ns))


def _ensure_directory_member_at(
    root_fd: int,
    relative_path: Path,
) -> None:
    """Create one real directory beneath a pinned extraction root."""
    with _open_safe_output_parent(root_fd, relative_path) as parent_fd:
        name = relative_path.name
        with contextlib.suppress(FileExistsError):
            os.mkdir(name, 0o700, dir_fd=parent_fd)
        try:
            current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as exc:
            raise ExecutionError(
                f"Snapshot directory changed while creating: {relative_path}"
            ) from exc
        if not stat.S_ISDIR(current.st_mode):
            raise ExecutionError(
                f"Snapshot directory conflicts with non-directory: {relative_path}"
            )


@contextlib.contextmanager
def _open_safe_directory_at(
    root_fd: int,
    relative_path: Path,
) -> Generator[int, None, None]:
    """Open an extracted directory without following any path component."""
    flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
    )
    with _open_safe_output_parent(root_fd, relative_path) as parent_fd:
        try:
            descriptor = os.open(
                relative_path.name,
                flags,
                dir_fd=parent_fd,
            )
        except OSError as exc:
            raise ExecutionError(
                f"Snapshot directory changed before metadata restore: "
                f"{relative_path}"
            ) from exc
        try:
            yield descriptor
        finally:
            os.close(descriptor)


def _write_regular_member_at(
    parent_fd: int,
    member: tarfile.TarInfo,
    source: BinaryIO,
    *,
    uid: int | None,
    gid: int | None,
) -> None:
    """Write a regular member without reopening any pathname ancestor."""
    name = Path(member.name).name
    _remove_existing_output(parent_fd, name)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=parent_fd)
    except OSError as exc:
        raise ExecutionError(
            f"Snapshot output changed while creating: {member.name}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        _apply_descriptor_metadata(
            descriptor,
            member,
            uid=uid,
            gid=gid,
        )
        if uid is not None and gid is not None:
            os.fchown(parent_fd, uid, gid)
    finally:
        os.close(descriptor)


def _write_symlink_member_at(
    parent_fd: int,
    member: tarfile.TarInfo,
    target: str,
    *,
    target_is_directory: bool,
    uid: int | None,
    gid: int | None,
) -> None:
    """Write a symlink relative to a pinned parent directory."""
    name = Path(member.name).name
    _remove_existing_output(parent_fd, name)
    try:
        os.symlink(
            target,
            name,
            target_is_directory=target_is_directory,
            dir_fd=parent_fd,
        )
    except OSError as exc:
        raise ExecutionError(
            f"Snapshot output changed while linking: {member.name}"
        ) from exc
    if uid is not None and gid is not None:
        os.chown(
            name,
            uid,
            gid,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        os.fchown(parent_fd, uid, gid)
    atime_ns = _member_atime_ns(member)
    mtime_ns = _member_mtime_ns(member)
    os.utime(
        name,
        ns=(atime_ns, mtime_ns),
        dir_fd=parent_fd,
        follow_symlinks=False,
    )


def _symlink_target_is_directory(member: tarfile.TarInfo) -> bool:
    """Read portable symlink target-type metadata from a snapshot member."""
    target_type = member.pax_headers.get(SYMLINK_TARGET_TYPE_PAX)
    if target_type in {None, "file"}:
        return False
    if target_type == "directory":
        return True
    raise ExecutionError(
        "Unsupported snapshot symlink target type: "
        f"{member.name} -> {target_type!r}"
    )


class Snapshot(BaseModel):
    """Snapshot that stores file contents in a compressed archive.

    Only the contents of the matched paths are included in the archive snapshot.

    Args:
        archive: The path to the compressed archive that contains the snapshot.
        checksum: The checksum of the snapshot.
        compression: The compression algorithm used for the archive.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    archive: Path
    checksum: str
    compression: str = "gz"
    timestamp: datetime = Field(default_factory=datetime.now)
    env: dict[str, str] = Field(default_factory=dict)
    matched_paths: set[Path] = Field(default_factory=set)
    other_paths: set[Path] = Field(default_factory=set)
    owns_archive_parent: bool = False

    def __repr__(self) -> str:
        return (
            f"Snapshot(path={self.path}, timestamp={self.timestamp}, "
            f"env={list(self.env.keys())}, "  # type: ignore
            f"archive={self.archive.stat().st_size / (1024**2):.2f} MB, "
            f"checksum={self.checksum[:8]}, "
            f"matched_paths={len(self.matched_paths):,}, "
            f"other_paths={len(self.other_paths):,})"
        )

    @classmethod
    def from_directory(
        cls,
        cwd: Path,
        env: dict[str, str],
        compression: str = "gz",
        save_path: Path | None = None,
        keep_globs: set[str] | None = None,
        ignore_globs: set[str] | None = None,
    ) -> Snapshot:
        """Snapshot the directory by creating a compressed archive with the
        patterns specified by the execution environment.

        Args:
            cwd: The directory to snapshot.
            compression: The compression algorithm to use.
            save_path: Optional directory where the archive should be stored.

        Returns:
            Snapshot containing archive metadata.
        """
        if not cwd.exists() or not cwd.is_dir():
            raise ExecutionError(
                f"`cwd` must be an existing directory, got: {cwd!s}"
            )

        tar_mode, tar_ext = _normalize_compression(compression)
        archive_path = _resolve_archive_path(cwd, save_path, tar_ext)

        logger.debug(
            "Snapshotting directory",
            cwd=cwd,
            compression=compression,
            save_path=save_path,
            keep_globs=keep_globs,
            ignore_globs=ignore_globs,
        )
        ignore_globs = ignore_globs or {
            "*.pyc",
            "venv/*",
            ".venv/*",
            "**/.DS_Store",
        }
        root_stat = cwd.stat(follow_symlinks=False)
        if not stat.S_ISDIR(root_stat.st_mode):
            raise ExecutionError(f"Snapshot root is not a directory: {cwd}")

        (
            matched_paths,
            other_paths,
            symlink_directory_paths,
            directory_paths,
        ) = _walk_candidates(cwd, ignore_globs, keep_globs or set())
        logger.debug(
            "Creating archive",
            num_matched_paths=len(matched_paths),
            num_other_paths=len(other_paths),
            archive_path=str(archive_path),
        )

        archive_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(
                str(archive_path),
                mode=tar_mode,  # type: ignore[arg-type]
                format=tarfile.PAX_FORMAT,
            ) as tf:
                tf.addfile(
                    _tar_info_from_stat(Path(ROOT_MEMBER_NAME), root_stat)
                )
                for rel_path in sorted(
                    matched_paths | directory_paths,
                    key=lambda path: path.as_posix(),
                ):
                    abs_path = cwd / rel_path
                    try:
                        candidate_stat = abs_path.lstat()
                    except OSError as exc:
                        raise ExecutionError(
                            f"Snapshot path disappeared during capture: {rel_path}"
                        ) from exc

                    if rel_path in directory_paths:
                        if not stat.S_ISDIR(candidate_stat.st_mode):
                            raise ExecutionError(
                                "Snapshot directory changed during capture: "
                                f"{rel_path}"
                            )
                        with _open_directory_no_follow(
                            cwd,
                            rel_path,
                        ) as (_, directory_stat):
                            tf.addfile(
                                _tar_info_from_stat(rel_path, directory_stat)
                            )
                        continue
                    if stat.S_ISLNK(candidate_stat.st_mode):
                        target, symlink_stat = _read_symlink_no_follow(
                            cwd,
                            rel_path,
                        )
                        _safe_relative_symlink_target(rel_path, target)
                        tf.addfile(
                            _tar_info_from_stat(
                                rel_path,
                                symlink_stat,
                                symlink_target=target,
                                symlink_target_is_directory=(
                                    rel_path in symlink_directory_paths
                                ),
                            )
                        )
                        continue
                    if not stat.S_ISREG(candidate_stat.st_mode):
                        raise ExecutionError(
                            f"Unsupported snapshot file type: {rel_path}"
                        )

                    with _open_regular_file_no_follow(
                        cwd,
                        rel_path,
                    ) as (source, source_stat):
                        tf.addfile(
                            _tar_info_from_stat(rel_path, source_stat),
                            source,
                        )
                        after = os.fstat(source.fileno())
                        if _stat_identity(source_stat) != _stat_identity(after):
                            raise ExecutionError(
                                "Snapshot file changed during capture: "
                                f"{rel_path}"
                            )
        except BaseException:
            archive_path.unlink(missing_ok=True)
            raise

        logger.debug(
            "Calculating checksum",
            archive_path=str(archive_path),
            size=f"{archive_path.stat().st_size / (1024**2):.4f} MB",
        )

        hash_md5 = hashlib.md5(usedforsecurity=False)
        with archive_path.open("rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        checksum = hash_md5.hexdigest()

        return cls(
            path=cwd,
            env=env,
            archive=archive_path,
            checksum=checksum,
            compression=compression,
            matched_paths=matched_paths,
            other_paths=other_paths,
        )

    def diff(self, other: Snapshot) -> SnapshotDiff:
        """Create a diff between this snapshot and another.

        Args:
            other: Snapshot to compare against

        Returns:
            SnapshotDiff showing the differences
        """
        logger.debug(
            "Creating diff between snapshots",
            from_checksum=self.checksum[:8],
            to_checksum=other.checksum[:8],
            verbose=True,
        )
        return SnapshotDiff.from_snapshots(self, other)

    def _extract_contents(self) -> Generator[tuple[Path, bytes], None, None]:
        """Extract all file contents from the snapshot archive.

        Yields:
            Tuples of (file_path, file_contents) for each file in the archive
        """
        read_mode = _tar_read_mode(self.compression)
        file_count = 0

        logger.debug(
            "Extracting contents from snapshot archive",
            archive=self.archive,
            compression=self.compression,
            verbose=True,
        )

        with (
            _open_verified_archive(self.archive, self.checksum) as source,
            tarfile.open(  # type: ignore[call-overload]
                fileobj=source,
                mode=read_mode,
            ) as tf,
        ):
            for member in tf.getmembers():
                if member.isfile():
                    path = _safe_archive_member_path(member.name)
                    extracted = tf.extractfile(member)
                    if extracted is not None:
                        file_count += 1
                        yield path, extracted.read()

        logger.debug(
            "Extracted file contents",
            file_count=file_count,
            verbose=True,
        )

    def extract_contents(self) -> dict[Path, bytes]:
        """Materialize all file contents from the snapshot archive.

        Returns:
            Dictionary mapping file paths to their contents as bytes
        """
        return dict(self._extract_contents())

    def extract_to_path(self, target_dir: Path) -> None:
        """Extract the snapshot contents to a directory.

        Args:
            target_dir: Directory to extract contents to
        """
        logger.debug(
            "Extracting snapshot to directory",
            target_dir=target_dir,
            verbose=True,
        )

        # Get the target uid/gid (e.g. from env vars set by Docker).
        uid: int | None = None
        gid: int | None = None
        if not IS_WINDOWS:
            uid = int(os.environ.get("HUID", os.getuid()))
            gid = int(os.environ.get("HGID", os.getgid()))

        file_count = 0
        directory_members: list[tuple[Path, tarfile.TarInfo]] = []
        read_mode = _tar_read_mode(self.compression)
        with (
            _open_verified_archive(self.archive, self.checksum) as source,
            _open_safe_extraction_root(target_dir) as (
                target_dir,
                root_fd,
            ),
            tarfile.open(  # type: ignore[call-overload]
                fileobj=source,
                mode=read_mode,
            ) as tf,
        ):
            root_member: tarfile.TarInfo | None = None
            for member in tf.getmembers():
                if member.name in {ROOT_MEMBER_NAME, f"{ROOT_MEMBER_NAME}/"}:
                    if root_member is not None or not member.isdir():
                        raise ExecutionError(
                            "Invalid snapshot workspace-root metadata member"
                        )
                    root_member = member
                    continue
                path = _safe_archive_member_path(member.name)

                if member.isdir():
                    if root_fd is not None:
                        _ensure_directory_member_at(root_fd, path)
                    else:
                        out_path = target_dir / path
                        _ensure_safe_output_parent(target_dir, out_path)
                        try:
                            current = out_path.lstat()
                        except FileNotFoundError:
                            out_path.mkdir(mode=0o700)
                            current = out_path.lstat()
                        if not stat.S_ISDIR(current.st_mode):
                            raise ExecutionError(
                                "Snapshot directory conflicts with "
                                f"non-directory: {out_path}"
                            )
                    directory_members.append((path, member))
                elif member.isfile():
                    extracted = tf.extractfile(member)
                    if extracted is None:
                        raise ExecutionError(
                            f"Could not read snapshot member: {member.name}"
                        )
                    if root_fd is not None:
                        with _open_safe_output_parent(root_fd, path) as parent_fd:
                            _write_regular_member_at(
                                parent_fd,
                                member,
                                extracted,
                                uid=uid,
                                gid=gid,
                            )
                    else:
                        out_path = target_dir / path
                        _ensure_safe_output_parent(target_dir, out_path)
                        if out_path.is_symlink() or out_path.exists():
                            if out_path.is_dir() and not out_path.is_symlink():
                                raise ExecutionError(
                                    "Snapshot file conflicts with directory: "
                                    f"{out_path}"
                                )
                            out_path.unlink()
                        try:
                            with out_path.open("xb") as output:
                                shutil.copyfileobj(
                                    extracted,
                                    output,
                                    length=1024 * 1024,
                                )
                        except OSError as exc:
                            raise ExecutionError(
                                "Snapshot output changed while creating: "
                                f"{member.name}"
                            ) from exc
                        if uid is not None and gid is not None:
                            os.chown(
                                out_path,
                                uid,
                                gid,
                                follow_symlinks=False,
                            )
                            os.chown(out_path.parent, uid, gid)
                        out_path.chmod(member.mode & 0o777)
                        atime_ns = _member_atime_ns(member)
                        mtime_ns = _member_mtime_ns(member)
                        os.utime(
                            out_path,
                            ns=(atime_ns, mtime_ns),
                            follow_symlinks=False,
                        )
                elif member.issym():
                    target = _safe_relative_symlink_target(
                        path,
                        member.linkname,
                    )
                    target_is_directory = _symlink_target_is_directory(member)
                    if root_fd is not None:
                        with _open_safe_output_parent(root_fd, path) as parent_fd:
                            _write_symlink_member_at(
                                parent_fd,
                                member,
                                target,
                                target_is_directory=target_is_directory,
                                uid=uid,
                                gid=gid,
                            )
                    else:
                        out_path = target_dir / path
                        _ensure_safe_output_parent(target_dir, out_path)
                        if out_path.is_symlink() or out_path.exists():
                            if out_path.is_dir() and not out_path.is_symlink():
                                raise ExecutionError(
                                    "Snapshot symlink conflicts with directory: "
                                    f"{out_path}"
                                )
                            out_path.unlink()
                        out_path.symlink_to(
                            target,
                            target_is_directory=target_is_directory,
                        )
                        if uid is not None and gid is not None:
                            os.chown(
                                out_path,
                                uid,
                                gid,
                                follow_symlinks=False,
                            )
                            os.chown(out_path.parent, uid, gid)
                        atime_ns = _member_atime_ns(member)
                        mtime_ns = _member_mtime_ns(member)
                        os.utime(
                            out_path,
                            ns=(atime_ns, mtime_ns),
                            follow_symlinks=False,
                        )
                else:
                    raise ExecutionError(
                        "Unsupported snapshot archive entry type: "
                        f"{member.name}"
                    )

                file_count += 1

            for path, member in sorted(
                directory_members,
                key=lambda item: len(item[0].parts),
                reverse=True,
            ):
                if root_fd is not None:
                    with _open_safe_directory_at(root_fd, path) as descriptor:
                        _apply_descriptor_metadata(
                            descriptor,
                            member,
                            uid=uid,
                            gid=gid,
                        )
                else:
                    out_path = target_dir / path
                    _ensure_safe_output_parent(target_dir, out_path)
                    try:
                        current = out_path.lstat()
                    except OSError as exc:
                        raise ExecutionError(
                            "Snapshot directory changed before metadata "
                            f"restore: {out_path}"
                        ) from exc
                    if not stat.S_ISDIR(current.st_mode):
                        raise ExecutionError(
                            "Snapshot directory changed before metadata "
                            f"restore: {out_path}"
                        )
                    if uid is not None and gid is not None:
                        os.chown(out_path, uid, gid, follow_symlinks=False)
                    out_path.chmod(member.mode & 0o777)
                    atime_ns = _member_atime_ns(member)
                    mtime_ns = _member_mtime_ns(member)
                    os.utime(
                        out_path,
                        ns=(atime_ns, mtime_ns),
                        follow_symlinks=False,
                    )

            if root_member is not None:
                if root_fd is not None:
                    _apply_descriptor_metadata(
                        root_fd,
                        root_member,
                        uid=uid,
                        gid=gid,
                    )
                else:
                    if uid is not None and gid is not None:
                        os.chown(target_dir, uid, gid)
                    target_dir.chmod(root_member.mode & 0o777)
                    os.utime(
                        target_dir,
                        ns=(
                            _member_atime_ns(root_member),
                            _member_mtime_ns(root_member),
                        ),
                        follow_symlinks=False,
                    )
            elif uid is not None and gid is not None:
                if root_fd is not None:
                    os.fchown(root_fd, uid, gid)
                else:
                    os.chown(target_dir, uid, gid)

        logger.debug(
            "Extracted snapshot to directory",
            target_dir=target_dir,
            file_count=file_count,
            verbose=True,
        )

    def extract_text_contents(self) -> dict[Path, str]:
        """Extract text file contents from the snapshot archive.

        Returns:
            Dictionary mapping file paths to their text contents
        """
        text_contents: dict[Path, str] = {}

        for path, data in self._extract_contents():
            if not _is_binary(data):
                text = _decode_text(data)
                if text is not None:
                    text_contents[path] = text

        logger.debug(
            "Extracted text contents",
            total_files=len(text_contents),
            verbose=True,
        )
        return text_contents

    @classmethod
    def from_environment_spec(
        cls,
        cwd: Path,
        env_spec: EnvironmentSpec,
        static_assets: dict[str, ResolvedStaticAsset] | None = None,
        env: dict[str, str] | None = None,
    ) -> Snapshot:
        """Create a snapshot from an environment specification.

        Args:
            cwd: Directory to snapshot
            env_spec: Environment specification
            static_assets: Optional static assets
            env: Additional environment variables

        Returns:
            Snapshot created according to the environment specification
        """
        logger.debug(
            "Creating snapshot from environment spec",
            cwd=cwd,
            compression=env_spec.snapshot.compression,
            has_static_assets=bool(static_assets),
            verbose=True,
        )
        owns_archive_parent = env_spec.snapshot.archive_save_dir is None
        snapshot = cls.from_directory(
            cwd=cwd,
            env=env_spec.get_full_env(env or {}),
            compression=env_spec.snapshot.compression,
            save_path=env_spec.get_archive_save_dir(),
            ignore_globs=env_spec.get_ignore_globs(static_assets),
            keep_globs=env_spec.snapshot.keep_globs,
        )
        return snapshot.model_copy(
            update={"owns_archive_parent": owns_archive_parent}
        )

    def cleanup(self) -> None:
        """Clean up the snapshot archive file."""
        logger.debug(
            "Cleaning up snapshot",
            verbose=True,
            snapshot=self.checksum[:8],
            archive=self.archive,
        )
        self.archive.unlink(missing_ok=True)
        if self.owns_archive_parent:
            with contextlib.suppress(OSError):
                self.archive.parent.rmdir()


class FileChangeType(str, Enum):
    """Enum representing the type of change to a file."""

    CREATED = "created"
    DELETED = "deleted"
    MODIFIED = "modified"


class FileDiff(BaseModel):
    """Represents changes to a single file between two snapshots."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    change_type: FileChangeType
    is_binary: bool = False

    # For text files
    diff_text: str | None = None
    lines_added: int = 0
    lines_removed: int = 0

    # For binary files
    old_size: int | None = None
    new_size: int | None = None

    def __repr__(self) -> str:
        if self.change_type == FileChangeType.CREATED:
            size_info = (
                f", size={self.new_size}"
                if self.is_binary
                else f", +{self.lines_added} lines"
            )
            return f"FileDiff({self.path}, CREATED{size_info})"
        if self.change_type == FileChangeType.DELETED:
            size_info = (
                f", size={self.old_size}"
                if self.is_binary
                else f", -{self.lines_removed} lines"
            )
            return f"FileDiff({self.path}, DELETED{size_info})"
        if self.is_binary:
            return f"FileDiff({self.path}, MODIFIED, binary: {self.old_size} -> {self.new_size})"
        return f"FileDiff({self.path}, MODIFIED, +{self.lines_added}/-{self.lines_removed})"

    def get_stats(self) -> dict[str, int]:
        return {
            "added": self.lines_added,
            "removed": self.lines_removed,
        }


class SnapshotDiff(BaseModel):
    """Represents the differences between two snapshots."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    from_checksum: str
    to_checksum: str
    from_timestamp: datetime
    to_timestamp: datetime
    file_diffs: dict[Path, FileDiff]

    def _file_stats(self) -> tuple[int, int, int]:
        created = sum(
            1
            for d in self.file_diffs.values()
            if d.change_type == FileChangeType.CREATED
        )
        deleted = sum(
            1
            for d in self.file_diffs.values()
            if d.change_type == FileChangeType.DELETED
        )
        modified = sum(
            1
            for d in self.file_diffs.values()
            if d.change_type == FileChangeType.MODIFIED
        )
        return created, deleted, modified

    def __repr__(self) -> str:
        created, deleted, modified = self._file_stats()
        return (
            f"SnapshotDiff({self.from_checksum[:8]} -> {self.to_checksum[:8]}, "
            f"+{created} -{deleted} ~{modified} files)"
        )

    @classmethod
    def from_snapshots(
        cls,
        from_snapshot: Snapshot,
        to_snapshot: Snapshot,
    ) -> SnapshotDiff:
        """Create a diff between two snapshots.

        Only compares text files that can be decoded as strings.

        Args:
            from_snapshot: The original snapshot (the "before" state).
            to_snapshot: The new snapshot (the "after" state).

        Returns:
            A SnapshotDiff object containing the differences.
        """
        # Generate checksums for snapshots
        from_checksum = _generate_snapshot_checksum(from_snapshot)
        to_checksum = _generate_snapshot_checksum(to_snapshot)

        logger.debug(
            "Creating diff between snapshots",
            from_checksum=from_checksum[:8],
            to_checksum=to_checksum[:8],
        )

        # Extract text contents from both snapshots
        from_contents = from_snapshot.extract_text_contents()
        to_contents = to_snapshot.extract_text_contents()

        # Get all file paths from both snapshots
        all_paths = set(from_contents.keys()) | set(to_contents.keys())

        file_diffs: dict[Path, FileDiff] = {}

        for path in all_paths:
            from_text = from_contents.get(path)
            to_text = to_contents.get(path)

            if from_text is None and to_text is not None:
                # File was created
                file_diffs[path] = _create_text_file_diff(
                    path, None, to_text, FileChangeType.CREATED
                )
            elif from_text is not None and to_text is None:
                # File was deleted
                file_diffs[path] = _create_text_file_diff(
                    path, from_text, None, FileChangeType.DELETED
                )
            elif from_text != to_text:
                # File was modified
                file_diffs[path] = _create_text_file_diff(
                    path, from_text, to_text, FileChangeType.MODIFIED
                )

        logger.debug(
            "Diff created",
            num_changed_files=len(file_diffs),
        )

        return cls(
            from_checksum=from_checksum,
            to_checksum=to_checksum,
            from_timestamp=from_snapshot.timestamp,
            to_timestamp=to_snapshot.timestamp,
            file_diffs=file_diffs,
        )

    def get_stats(self) -> dict[str, int]:
        created, deleted, modified = self._file_stats()
        lines_added = lines_removed = 0
        for file in self.file_diffs.values():
            if file.is_binary:
                continue
            lines_added += file.lines_added
            lines_removed += file.lines_removed
        return {
            "created": created,
            "deleted": deleted,
            "modified": modified,
            "lines_added": lines_added,
            "lines_removed": lines_removed,
        }


def _generate_snapshot_checksum(snapshot: Snapshot) -> str:
    """Generate a checksum for a snapshot."""
    return snapshot.checksum


def _is_binary(data: bytes) -> bool:
    """Detect if file data is binary using heuristics.

    Uses multiple indicators to determine if data is binary:
    1. Presence of null bytes (strong indicator)
    2. Ratio of non-printable characters (>30% threshold)

    Tabs, line feeds, and carriage returns are considered printable.
    Only the first 8KB is sampled for performance.

    Args:
        data: Raw file contents to analyze.

    Returns:
        True if data appears to be binary, False if likely text.

    Example:
        >>> _is_binary(b"Hello, world!")
        False
        >>> _is_binary(b"\\x89PNG\\r\\n\\x1a\\n")
        True
        >>> _is_binary(b"Text with\\x00null byte")
        True
    """
    # Check for null bytes (strong indicator of binary)
    if b"\x00" in data:
        return True

    # Sample the first 8KB for performance
    sample = data[:8192]
    if not sample:
        return False

    # Count non-text bytes
    non_text = sum(
        1
        for byte in sample
        if byte < 0x20 and byte not in (0x09, 0x0A, 0x0D)  # tab, LF, CR
    )

    # If more than 30% non-text, consider binary
    return (non_text / len(sample)) > 0.3


def _decode_text(data: bytes) -> str | None:
    """Attempt to decode bytes as text using multiple encodings.

    Tries common encodings in order: UTF-8, Latin-1, CP1252.
    Returns None if all decoding attempts fail.

    Args:
        data: Raw bytes to decode.

    Returns:
        Decoded string if successful, None if all encodings fail.

    Note:
        Latin-1 can decode any byte sequence, so it serves as a fallback
        that will almost never return None. CP1252 handles Windows-specific
        characters.

    Example:
        >>> _decode_text(b"Hello")
        'Hello'
        >>> _decode_text("Café".encode('utf-8'))
        'Café'
    """
    encodings = ["utf-8", "latin-1", "cp1252"]
    for encoding in encodings:
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, AttributeError):
            continue
    return None


def _create_text_file_diff(
    path: Path,
    from_text: str | None,
    to_text: str | None,
    change_type: FileChangeType,
) -> FileDiff:
    """Create a FileDiff for a single text file.

    Args:
        path: The relative path to the file.
        from_text: The original file contents (None if created).
        to_text: The new file contents (None if deleted).
        change_type: The type of change.

    Returns:
        A FileDiff object representing the changes.
    """
    from_text = from_text or ""
    to_text = to_text or ""

    # Generate unified diff
    from_lines = from_text.splitlines(keepends=True)
    to_lines = to_text.splitlines(keepends=True)

    # Use difflib to generate a unified diff
    diff_lines = list(
        difflib.unified_diff(
            from_lines,
            to_lines,
            fromfile=str(path),
            tofile=str(path),
            lineterm="",
            n=3,  # context lines
        )
    )

    # Join diff lines into a single string
    diff_text = "\n".join(diff_lines) if diff_lines else ""

    # Count added and removed lines
    lines_added = 0
    lines_removed = 0

    for line in diff_lines:
        if line.startswith("+") and not line.startswith("+++"):
            lines_added += 1
        elif line.startswith("-") and not line.startswith("---"):
            lines_removed += 1

    return FileDiff(
        path=path,
        change_type=change_type,
        is_binary=False,
        diff_text=diff_text,
        lines_added=lines_added,
        lines_removed=lines_removed,
    )


def _extract_text_contents_from_directory(
    directory: Path,
    ignore_globs: set[str] | None = None,
) -> dict[Path, str]:
    """Extract text file contents from a directory.

    Args:
        directory: Path to the directory to scan.
        ignore_globs: Optional set of glob patterns to ignore.

    Returns:
        Dictionary mapping relative file paths to their text contents.
    """
    if not directory.exists() or not directory.is_dir():
        return {}

    ignore_globs = ignore_globs or {
        "*.pyc",
        "venv/*",
        ".venv/*",
        "**/.DS_Store",
    }
    text_contents: dict[Path, str] = {}

    for root, dirs, files in os.walk(
        directory, topdown=True, followlinks=False
    ):
        root_path = Path(root)

        # Prune directories early
        kept_dirs = []
        for d in list(dirs):
            rel_dir = (root_path / d).relative_to(directory).as_posix() + "/"
            if _matches_any(ignore_globs, rel_dir):
                continue
            kept_dirs.append(d)
        dirs[:] = kept_dirs

        # Process files
        for f in files:
            abs_path = root_path / f
            rel_path = abs_path.relative_to(directory)
            rel_posix = rel_path.as_posix()

            if _matches_any(ignore_globs, rel_posix):
                continue

            try:
                data = abs_path.read_bytes()
                if not _is_binary(data):
                    text = _decode_text(data)
                    if text is not None:
                        text_contents[rel_path] = text
            except OSError as e:
                logger.warning(
                    "Failed to read file",
                    path=str(abs_path),
                    error=str(e),
                )
                continue

    return text_contents


def _compute_directory_checksum(directory: Path) -> str:
    """Compute a checksum for a directory based on its file contents.

    Args:
        directory: Path to the directory.

    Returns:
        MD5 hex digest of the directory contents.
    """
    if not directory.exists():
        return "empty"

    hash_md5 = hashlib.md5(usedforsecurity=False)

    # Get all files sorted by path for deterministic ordering
    all_files: list[Path] = []
    for root, _, files in os.walk(directory, followlinks=False):
        for f in files:
            all_files.append(Path(root) / f)

    for file_path in sorted(all_files):
        try:
            rel_path = file_path.relative_to(directory)
            hash_md5.update(rel_path.as_posix().encode())
            hash_md5.update(file_path.read_bytes())
        except OSError:
            continue

    return hash_md5.hexdigest()


def create_diff_from_directories(
    from_dir: Path | None,
    to_dir: Path,
    from_checksum: str | None = None,
    to_checksum: str | None = None,
) -> SnapshotDiff:
    """Create a SnapshotDiff between two directories.

    This function is useful for regenerating diffs from extracted snapshot
    directories without needing the original tar archives.

    Args:
        from_dir: The "before" directory. If None, treated as empty (all files
            in to_dir will be marked as created).
        to_dir: The "after" directory.
        from_checksum: Optional checksum for from_dir. If not provided, computed
            from directory contents.
        to_checksum: Optional checksum for to_dir. If not provided, computed
            from directory contents.

    Returns:
        A SnapshotDiff representing the changes between directories.

    Example:
        >>> diff = create_diff_from_directories(
        ...     from_dir=Path("checkpoint_1/snapshot"),
        ...     to_dir=Path("checkpoint_2/snapshot"),
        ... )
        >>> print(diff)
    """
    # Extract text contents from both directories
    from_contents: dict[Path, str] = {}
    if from_dir is not None:
        from_contents = _extract_text_contents_from_directory(from_dir)

    to_contents = _extract_text_contents_from_directory(to_dir)

    # Compute checksums if not provided
    if from_checksum is None:
        from_checksum = (
            _compute_directory_checksum(from_dir) if from_dir else "empty"
        )
    if to_checksum is None:
        to_checksum = _compute_directory_checksum(to_dir)

    # Get all file paths from both directories
    all_paths = set(from_contents.keys()) | set(to_contents.keys())

    file_diffs: dict[Path, FileDiff] = {}

    for path in all_paths:
        from_text = from_contents.get(path)
        to_text = to_contents.get(path)

        if from_text is None and to_text is not None:
            # File was created
            file_diffs[path] = _create_text_file_diff(
                path, None, to_text, FileChangeType.CREATED
            )
        elif from_text is not None and to_text is None:
            # File was deleted
            file_diffs[path] = _create_text_file_diff(
                path, from_text, None, FileChangeType.DELETED
            )
        elif from_text != to_text:
            # File was modified
            file_diffs[path] = _create_text_file_diff(
                path, from_text, to_text, FileChangeType.MODIFIED
            )

    # Use directory mtime or current time for timestamps
    now = datetime.now()
    from_timestamp = now
    to_timestamp = now

    if from_dir is not None and from_dir.exists():
        with contextlib.suppress(OSError):
            from_timestamp = datetime.fromtimestamp(from_dir.stat().st_mtime)

    if to_dir.exists():
        with contextlib.suppress(OSError):
            to_timestamp = datetime.fromtimestamp(to_dir.stat().st_mtime)

    logger.debug(
        "Created diff from directories",
        from_dir=str(from_dir) if from_dir else "empty",
        to_dir=str(to_dir),
        num_changed_files=len(file_diffs),
    )

    return SnapshotDiff(
        from_checksum=from_checksum,
        to_checksum=to_checksum,
        from_timestamp=from_timestamp,
        to_timestamp=to_timestamp,
        file_diffs=file_diffs,
    )
