"""Main entry point for checkpoint metric extraction.

This module provides the orchestrating function that combines all metric
extractors to produce a complete checkpoint metrics dictionary.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import subprocess
import tempfile
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from slop_code.common import QUALITY_DIR
from slop_code.common import SNAPSHOT_DIR_NAME
from slop_code.common.atomic import _open_directory_without_symlinks
from slop_code.common.atomic import atomic_write_text
from slop_code.logging import get_logger
from slop_code.metrics.checkpoint.delta import compute_checkpoint_delta
from slop_code.metrics.checkpoint.extractors import get_evaluation_metrics
from slop_code.metrics.checkpoint.extractors import get_inference_metrics
from slop_code.metrics.checkpoint.extractors import get_quality_metrics
from slop_code.metrics.checkpoint.extractors import get_rubric_metrics

logger = get_logger(__name__)

SCB_CHECK_NAME = "scb-check"
SCB_CHECK_VERSION = "0.1.3"
SCB_CHECK_REQUIREMENT = f"{SCB_CHECK_NAME}=={SCB_CHECK_VERSION}"
SCB_CHECK_RECORD_FILENAME = f"{SCB_CHECK_NAME}-{SCB_CHECK_VERSION}.json"
SCB_CHECK_TIMEOUT_SECONDS = 600
SCB_CHECK_PROJECT_RELATIVE = "configs/scbench-v2/evaluator"
SCB_CHECK_PROJECT_DIR = (
    Path(__file__).resolve().parents[4] / SCB_CHECK_PROJECT_RELATIVE
)
SCB_CHECK_PROJECT_PATH = SCB_CHECK_PROJECT_DIR / "pyproject.toml"
SCB_CHECK_LOCK_PATH = SCB_CHECK_PROJECT_DIR / "uv.lock"
SCB_CHECK_PROJECT_ENV = "SLOP_CODE_SCB_CHECK_PROJECT_DIR"
SCB_CHECK_VENV_ENV = "SLOP_CODE_SCB_CHECK_VENV_DIR"
SCB_CHECK_ERROR_STREAM_LIMIT = 8192
SCB_CHECK_REPORT_STREAM_LIMIT = 64 * 1024 * 1024
SCB_CHECK_SNAPSHOT_MAX_ENTRIES = 100_000
SCB_CHECK_SNAPSHOT_MAX_BYTES = 1024 * 1024 * 1024
SCB_CHECK_SNAPSHOT_HASH_ALGORITHM = "sha256-type-path-mode-size-v3"
SCB_CHECK_SNAPSHOT_HASH_DOMAIN = b"slop-code.snapshot-tree.v3\0"
_SENSITIVE_OUTPUT = re.compile(
    r"(?i)(?:authorization\s*[:=]\s*(?:bearer|basic)?\s*\S+|"
    r"(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)"
    r"\s*[:=]\s*\S+|(?:sk-|sess-|gh[opusr]_|xox[baprs]-)\S+)"
)
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def _evaluator_project_dir() -> Path:
    configured = os.environ.get(SCB_CHECK_PROJECT_ENV)
    return (
        Path(configured).absolute()
        if configured
        else SCB_CHECK_PROJECT_DIR.absolute()
    )


def _evaluator_project_path() -> Path:
    return _evaluator_project_dir() / "pyproject.toml"


def _evaluator_lock_path() -> Path:
    return _evaluator_project_dir() / "uv.lock"


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        try:
            number = float(value)
        except OverflowError:
            return None
        return number if math.isfinite(number) else None
    return None


def _error_record(
    exc: BaseException,
    *,
    phase: str,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Build JSON-safe evaluator failure evidence."""

    def stream_text(value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, bytes):
            text = value.decode(errors="replace")
        else:
            text = str(value)
        text = _SENSITIVE_OUTPUT.sub("<redacted>", text)
        if len(text) <= SCB_CHECK_ERROR_STREAM_LIMIT:
            return text
        omitted = len(text) - SCB_CHECK_ERROR_STREAM_LIMIT
        return (
            text[:SCB_CHECK_ERROR_STREAM_LIMIT]
            + f"\n<truncated {omitted} characters>"
        )

    error: dict[str, Any] = {
        "phase": phase,
        "type": type(exc).__name__,
        "message": stream_text(exc) or type(exc).__name__,
    }
    if command is not None:
        error["command"] = command
    if isinstance(exc, subprocess.CalledProcessError):
        error["returncode"] = exc.returncode
        error["stdout"] = stream_text(exc.stdout)
        error["stderr"] = stream_text(exc.stderr)
    elif isinstance(exc, subprocess.TimeoutExpired):
        error["timeout_seconds"] = exc.timeout
        error["stdout"] = stream_text(exc.stdout)
        error["stderr"] = stream_text(exc.stderr)
    return error


def _evaluator_command(executable: str, *arguments: str) -> list[str]:
    return [
        "uv",
        "run",
        "--frozen",
        "--project",
        str(_evaluator_project_dir()),
        executable,
        *arguments,
    ]


def _scb_check_command(*arguments: str) -> list[str]:
    return _evaluator_command(SCB_CHECK_NAME, *arguments)


def _scb_check_environment() -> dict[str, str]:
    configured_cache = os.environ.get("UV_CACHE_DIR")
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("UV_")
    }
    environment["UV_NO_CONFIG"] = "1"
    # The cache location cannot change dependency resolution under --frozen,
    # but preserving it lets callers select a writable cache on restricted
    # hosts. Other ambient uv overrides remain scrubbed.
    if configured_cache:
        environment["UV_CACHE_DIR"] = str(Path(configured_cache).absolute())
    configured_venv = os.environ.get(SCB_CHECK_VENV_ENV)
    if configured_venv:
        environment["UV_PROJECT_ENVIRONMENT"] = str(
            Path(configured_venv).absolute()
        )
    return environment


def _bounded_stream_text(stream: Any, limit: int) -> str:
    stream.flush()
    stream.seek(0)
    content = stream.read(limit + 1)
    if isinstance(content, bytes):
        text = content.decode(errors="replace")
    else:
        text = str(content)
    if len(content) <= limit:
        return text
    return text[:limit] + "\n<truncated at collection limit>"


def _run_captured_command(
    command: list[str],
    *,
    timeout: int,
    stdout_limit: int,
) -> subprocess.CompletedProcess[str]:
    """Run with disk-backed streams and bounded in-memory collection."""
    with (
        tempfile.TemporaryFile(mode="w+b") as stdout_file,
        tempfile.TemporaryFile(mode="w+b") as stderr_file,
    ):
        try:
            completed = subprocess.run(  # noqa: S603
                command,  # noqa: S607 - fixed commands use the active uv.
                check=False,
                stdout=stdout_file,
                stderr=stderr_file,
                timeout=timeout,
                env=_scb_check_environment(),
            )
        except subprocess.TimeoutExpired as exc:
            raise subprocess.TimeoutExpired(
                exc.cmd,
                exc.timeout,
                output=_bounded_stream_text(
                    stdout_file,
                    SCB_CHECK_ERROR_STREAM_LIMIT,
                ),
                stderr=_bounded_stream_text(
                    stderr_file,
                    SCB_CHECK_ERROR_STREAM_LIMIT,
                ),
            ) from exc

        # Test doubles may return text directly instead of writing the supplied
        # file descriptors; real subprocesses always use the disk-backed path.
        returncode = getattr(completed, "returncode", 0)
        collection_limit = (
            SCB_CHECK_ERROR_STREAM_LIMIT if returncode != 0 else stdout_limit
        )
        returned_stdout = getattr(completed, "stdout", None)
        if returned_stdout is None:
            stdout = _bounded_stream_text(stdout_file, collection_limit)
            stdout_file.seek(0, os.SEEK_END)
            stdout_size = stdout_file.tell()
        else:
            stdout = str(returned_stdout)
            stdout_size = len(stdout.encode())
        returned_stderr = getattr(completed, "stderr", None)
        stderr = (
            _bounded_stream_text(stderr_file, SCB_CHECK_ERROR_STREAM_LIMIT)
            if returned_stderr is None
            else str(returned_stderr)[:SCB_CHECK_ERROR_STREAM_LIMIT]
        )
        if returncode != 0:
            raise subprocess.CalledProcessError(
                returncode,
                command,
                output=stdout[:SCB_CHECK_ERROR_STREAM_LIMIT],
                stderr=stderr,
            )
        if stdout_size > stdout_limit:
            raise ValueError(
                "evaluator stdout exceeded collection limit "
                f"({stdout_limit} bytes)"
            )
        return subprocess.CompletedProcess(
            command,
            returncode,
            stdout=stdout,
            stderr=stderr,
        )


def scb_check_lock_sha256() -> str:
    """Return the digest of the complete frozen evaluator dependency graph."""
    return _sha256_regular_file(_evaluator_lock_path())


def _sha256_regular_file(path: Path) -> str:
    """Hash a pinned regular file and reject path or byte drift."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        parent_fd = _open_directory_without_symlinks(path.parent)
    except OSError as exc:
        raise ValueError(f"cannot safely open evaluator input {path}: {exc}") from exc
    try:
        try:
            descriptor = os.open(path.name, flags, dir_fd=parent_fd)
        except OSError as exc:
            raise ValueError(
                f"cannot safely open evaluator input {path}: {exc}"
            ) from exc
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(
                    f"evaluator input is not a regular file: {path}"
                )
            digest = hashlib.sha256()
            with os.fdopen(os.dup(descriptor), "rb", closefd=True) as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            after = os.fstat(descriptor)
            path_after = os.stat(
                path.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            if (
                _filesystem_identity(before) != _filesystem_identity(after)
                or _filesystem_identity(after)
                != _filesystem_identity(path_after)
            ):
                raise ValueError(
                    f"evaluator input changed while hashing: {path}"
                )
            return digest.hexdigest()
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_fd)


def _evaluator_identity() -> dict[str, str]:
    project_dir = _evaluator_project_dir()
    return {
        "project_dir": str(project_dir),
        "project_sha256": _sha256_regular_file(
            project_dir / "pyproject.toml"
        ),
        "lock_sha256": _sha256_regular_file(project_dir / "uv.lock"),
    }


def _require_evaluator_identity(
    expected: dict[str, str],
    *,
    phase: str,
) -> None:
    actual = _evaluator_identity()
    if actual != expected:
        raise ValueError(
            f"evaluator project changed during {phase}: "
            f"expected {expected!r}, got {actual!r}"
        )


def _scb_check_lock_evidence() -> dict[str, Any]:
    lock_path = _evaluator_lock_path()
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    packages = lock.get("package")
    if not isinstance(packages, list):
        raise ValueError("SCBench v2 evaluator lock has no package list")
    package_evidence: list[dict[str, Any]] = []
    for package in packages:
        if not isinstance(package, dict):
            raise ValueError("SCBench v2 evaluator lock has an invalid package")
        wheels = package.get("wheels")
        wheel_values = wheels if isinstance(wheels, list) else []
        wheel_hashes = sorted(
            wheel.get("hash")
            for wheel in wheel_values
            if isinstance(wheel, dict) and isinstance(wheel.get("hash"), str)
        )
        sdist = package.get("sdist")
        sdist_hash = sdist.get("hash") if isinstance(sdist, dict) else None
        package_evidence.append(
            {
                "name": package.get("name"),
                "version": package.get("version"),
                "sdist_hash": sdist_hash,
                "wheel_hashes": wheel_hashes,
            }
        )
    package_evidence.sort(key=lambda value: str(value["name"]))
    return {
        "lock_path": str(lock_path),
        "lock_sha256": scb_check_lock_sha256(),
        "requires_python": lock.get("requires-python"),
        "packages": package_evidence,
    }


def scb_check_preflight() -> dict[str, Any]:
    """Resolve the frozen evaluator before any named-profile model work."""
    evaluator_identity = _evaluator_identity()
    resolved_version = _resolved_scb_check_version()
    uv_result = _run_captured_command(
        ["uv", "--version"],
        timeout=30,
        stdout_limit=SCB_CHECK_ERROR_STREAM_LIMIT,
    )
    python_result = _run_captured_command(
        _evaluator_command(
            "python",
            "-c",
            "import platform; print(platform.python_version())",
        ),
        timeout=30,
        stdout_limit=SCB_CHECK_ERROR_STREAM_LIMIT,
    )
    lock_evidence = _scb_check_lock_evidence()
    _require_evaluator_identity(evaluator_identity, phase="preflight")
    return {
        "status": "verified",
        "evaluator": SCB_CHECK_NAME,
        "requirement": SCB_CHECK_REQUIREMENT,
        "resolved_version": resolved_version,
        "uv_version": uv_result.stdout.strip(),
        "python_version": python_result.stdout.strip(),
        "project_path": str(_evaluator_project_path()),
        "project_sha256": evaluator_identity["project_sha256"],
        **lock_evidence,
    }


def is_valid_scb_check_metadata(value: object) -> bool:
    """Whether a checkpoint owns a complete, reproducible measurement."""
    if not isinstance(value, dict):
        return False
    snapshot_hash = value.get("snapshot_tree_sha256")
    project_hash = value.get("environment_project_sha256")
    lock_hash = value.get("environment_lock_sha256")
    try:
        evaluator_identity = _evaluator_identity()
    except (OSError, ValueError):
        return False
    return (
        value.get("evaluator") == SCB_CHECK_NAME
        and value.get("requested_version") == SCB_CHECK_VERSION
        and value.get("resolved_version") == SCB_CHECK_VERSION
        and value.get("status") == "measured"
        and value.get("record_persisted") is True
        and value.get("snapshot_preserved") is True
        and isinstance(snapshot_hash, str)
        and _SHA256_HEX.fullmatch(snapshot_hash) is not None
        and value.get("snapshot_hash_algorithm")
        == SCB_CHECK_SNAPSHOT_HASH_ALGORITHM
        and isinstance(project_hash, str)
        and _SHA256_HEX.fullmatch(project_hash) is not None
        and isinstance(lock_hash, str)
        and _SHA256_HEX.fullmatch(lock_hash) is not None
        and project_hash == evaluator_identity["project_sha256"]
        and lock_hash == evaluator_identity["lock_sha256"]
    )


def _scb_check_metrics_from_report(report: dict[str, Any]) -> dict[str, Any]:
    """Extract checkpoint metrics owned by scb-check's report."""
    def score(name: str) -> float:
        value = _number(report.get(name))
        if value is None or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"scb-check report {name!r} must be a finite number "
                "between 0 and 1"
            )
        return value

    def optional_count(name: str) -> int | None:
        if name not in report:
            return None
        value = _number(report[name])
        if value is None or value < 0 or not value.is_integer():
            raise ValueError(
                f"scb-check report {name!r} must be a non-negative integer"
            )
        return int(value)

    verbosity = score("verbosity")
    erosion = score("erosion")
    clone_loc = optional_count("clone_loc")
    verbosity_flagged_loc = optional_count("verbosity_flagged_loc")
    total_loc = optional_count("total_loc")

    if total_loc is not None:
        for name, value in (
            ("clone_loc", clone_loc),
            ("verbosity_flagged_loc", verbosity_flagged_loc),
        ):
            if value is not None and value > total_loc:
                raise ValueError(
                    f"scb-check report {name!r} cannot exceed 'total_loc'"
                )

    metrics: dict[str, Any] = {
        "verbosity": verbosity,
        "erosion": erosion,
    }
    if clone_loc is not None:
        metrics["cloned_sloc_lines"] = clone_loc
    if verbosity_flagged_loc is not None:
        metrics["verbosity_flagged_sloc_lines"] = verbosity_flagged_loc

    if total_loc is None or total_loc == 0:
        return metrics

    if clone_loc is not None:
        metrics["cloned_pct"] = clone_loc / total_loc
    if verbosity_flagged_loc is not None:
        metrics["verbosity_flagged_pct"] = verbosity_flagged_loc / total_loc

    return metrics


def _update_length_prefixed_hash(
    digest: Any,
    value: bytes,
) -> None:
    """Add an unambiguous byte field to a snapshot digest."""
    digest.update(len(value).to_bytes(8, byteorder="big"))
    digest.update(value)


def _snapshot_tree_sha256(snapshot_dir: Path) -> str:
    """Hash the preserved source tree used by the quality evaluator."""
    if snapshot_dir.is_symlink():
        raise ValueError(f"snapshot root is an unsupported symlink: {snapshot_dir}")
    if not snapshot_dir.is_dir():
        raise ValueError(f"snapshot is not a directory: {snapshot_dir}")

    digest = hashlib.sha256()
    digest.update(SCB_CHECK_SNAPSHOT_HASH_DOMAIN)
    pending_directories = [snapshot_dir]
    entries_seen = 0
    bytes_seen = 0

    while pending_directories:
        directory = pending_directories.pop()
        entries: list[os.DirEntry[str]] = []
        with os.scandir(directory) as scanner:
            for entry in scanner:
                entries_seen += 1
                if entries_seen > SCB_CHECK_SNAPSHOT_MAX_ENTRIES:
                    raise ValueError(
                        "snapshot exceeds entry limit "
                        f"({SCB_CHECK_SNAPSHOT_MAX_ENTRIES})"
                    )
                entries.append(entry)
        entries.sort(key=lambda entry: os.fsencode(entry.name))

        child_directories: list[Path] = []
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(snapshot_dir).as_posix()
            relative_bytes = os.fsencode(relative)
            if entry.is_symlink():
                raise ValueError(
                    "snapshot contains unsupported symlink: "
                    f"{relative} -> {path.readlink()}"
                )

            if entry.is_dir(follow_symlinks=False):
                _update_length_prefixed_hash(digest, b"directory")
                _update_length_prefixed_hash(digest, relative_bytes)
                mode = stat.S_IMODE(
                    entry.stat(follow_symlinks=False).st_mode
                )
                _update_length_prefixed_hash(
                    digest,
                    mode.to_bytes(4, byteorder="big"),
                )
                _update_length_prefixed_hash(digest, b"")
                child_directories.append(path)
                continue

            if not entry.is_file(follow_symlinks=False):
                raise ValueError(
                    f"snapshot contains unsupported file type: {relative}"
                )

            file_stat = entry.stat(follow_symlinks=False)
            size = file_stat.st_size
            if size < 0 or bytes_seen + size > SCB_CHECK_SNAPSHOT_MAX_BYTES:
                raise ValueError(
                    "snapshot exceeds byte limit "
                    f"({SCB_CHECK_SNAPSHOT_MAX_BYTES})"
                )
            _update_length_prefixed_hash(digest, b"file")
            _update_length_prefixed_hash(digest, relative_bytes)
            mode = stat.S_IMODE(file_stat.st_mode)
            _update_length_prefixed_hash(
                digest,
                mode.to_bytes(4, byteorder="big"),
            )
            digest.update(size.to_bytes(8, byteorder="big"))

            file_bytes = 0
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    file_bytes += len(chunk)
                    if file_bytes > size:
                        raise ValueError(
                            f"snapshot file changed while hashing: {relative}"
                        )
                    digest.update(chunk)
            if file_bytes != size:
                raise ValueError(
                    f"snapshot file changed while hashing: {relative}"
                )
            bytes_seen += file_bytes

        pending_directories.extend(reversed(child_directories))
    return digest.hexdigest()


def _filesystem_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _copy_snapshot_directory(
    source_fd: int,
    destination: Path,
    *,
    prefix: Path,
    counters: dict[str, int],
) -> None:
    """Copy one source directory through pinned no-follow descriptors."""
    directory_before = os.fstat(source_fd)
    entries = list(os.scandir(source_fd))
    entries.sort(key=lambda entry: os.fsencode(entry.name))
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    destination_flags |= getattr(os, "O_NOFOLLOW", 0)

    for entry in entries:
        relative = prefix / entry.name
        counters["entries"] += 1
        if counters["entries"] > SCB_CHECK_SNAPSHOT_MAX_ENTRIES:
            raise ValueError(
                "snapshot exceeds entry limit "
                f"({SCB_CHECK_SNAPSHOT_MAX_ENTRIES})"
            )
        before = os.stat(
            entry.name,
            dir_fd=source_fd,
            follow_symlinks=False,
        )
        if stat.S_ISLNK(before.st_mode):
            raise ValueError(
                "snapshot contains unsupported symlink: "
                f"{relative.as_posix()}"
            )
        target = destination / entry.name
        if stat.S_ISDIR(before.st_mode):
            child_fd = os.open(
                entry.name,
                directory_flags,
                dir_fd=source_fd,
            )
            try:
                child_before = os.fstat(child_fd)
                if _filesystem_identity(child_before) != _filesystem_identity(
                    before
                ):
                    raise ValueError(
                        "snapshot directory changed while capturing: "
                        f"{relative.as_posix()}"
                    )
                target.mkdir(mode=0o700)
                _copy_snapshot_directory(
                    child_fd,
                    target,
                    prefix=relative,
                    counters=counters,
                )
                child_after = os.fstat(child_fd)
                path_after = os.stat(
                    entry.name,
                    dir_fd=source_fd,
                    follow_symlinks=False,
                )
                if (
                    _filesystem_identity(child_after)
                    != _filesystem_identity(child_before)
                    or _filesystem_identity(path_after)
                    != _filesystem_identity(child_after)
                ):
                    raise ValueError(
                        "snapshot directory changed while capturing: "
                        f"{relative.as_posix()}"
                    )
                target.chmod(stat.S_IMODE(before.st_mode))
            finally:
                os.close(child_fd)
            continue
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(
                "snapshot contains unsupported file type: "
                f"{relative.as_posix()}"
            )
        if (
            before.st_size < 0
            or counters["bytes"] + before.st_size
            > SCB_CHECK_SNAPSHOT_MAX_BYTES
        ):
            raise ValueError(
                "snapshot exceeds byte limit "
                f"({SCB_CHECK_SNAPSHOT_MAX_BYTES})"
            )
        input_fd = os.open(
            entry.name,
            source_flags,
            dir_fd=source_fd,
        )
        output_fd: int | None = None
        try:
            opened = os.fstat(input_fd)
            if _filesystem_identity(opened) != _filesystem_identity(before):
                raise ValueError(
                    "snapshot file changed while capturing: "
                    f"{relative.as_posix()}"
                )
            output_fd = os.open(target, destination_flags, 0o600)
            copied = 0
            with (
                os.fdopen(os.dup(input_fd), "rb", closefd=True) as source,
                os.fdopen(output_fd, "wb", closefd=True) as output,
            ):
                output_fd = None
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    copied += len(chunk)
                    if copied > before.st_size:
                        raise ValueError(
                            "snapshot file changed while capturing: "
                            f"{relative.as_posix()}"
                        )
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
                os.fchmod(output.fileno(), stat.S_IMODE(before.st_mode))
            after = os.fstat(input_fd)
            path_after = os.stat(
                entry.name,
                dir_fd=source_fd,
                follow_symlinks=False,
            )
            if (
                copied != before.st_size
                or _filesystem_identity(after) != _filesystem_identity(opened)
                or _filesystem_identity(path_after)
                != _filesystem_identity(after)
            ):
                raise ValueError(
                    "snapshot file changed while capturing: "
                    f"{relative.as_posix()}"
                )
            counters["bytes"] += copied
        finally:
            if output_fd is not None:
                os.close(output_fd)
            os.close(input_fd)

    directory_after = os.fstat(source_fd)
    if _filesystem_identity(directory_after) != _filesystem_identity(
        directory_before
    ):
        raise ValueError(
            "snapshot directory changed while capturing: "
            f"{prefix.as_posix() or '.'}"
        )


@contextmanager
def _captured_snapshot(
    snapshot_dir: Path,
) -> Iterator[tuple[Path, str]]:
    """Yield a private byte-for-byte snapshot and verify it after use."""
    absolute = snapshot_dir.absolute()
    try:
        source_fd = _open_directory_without_symlinks(absolute)
    except OSError as exc:
        raise ValueError(f"cannot safely open snapshot {absolute}: {exc}") from exc
    try:
        source_before = os.fstat(source_fd)
        with tempfile.TemporaryDirectory(
            prefix="slop-code-scb-check-snapshot-"
        ) as temporary:
            captured = Path(temporary) / SNAPSHOT_DIR_NAME
            captured.mkdir()
            _copy_snapshot_directory(
                source_fd,
                captured,
                prefix=Path(),
                counters={"entries": 0, "bytes": 0},
            )
            source_after = absolute.stat(follow_symlinks=False)
            if _filesystem_identity(source_after) != _filesystem_identity(
                source_before
            ):
                raise ValueError(
                    f"snapshot root changed while capturing: {absolute}"
                )
            captured_sha256 = _snapshot_tree_sha256(captured)
            yield captured, captured_sha256
            final_sha256 = _snapshot_tree_sha256(captured)
            if final_sha256 != captured_sha256:
                raise ValueError(
                    "captured snapshot changed during evaluator use: "
                    f"{captured}"
                )
    finally:
        os.close(source_fd)


def _write_scb_check_record(
    checkpoint_dir: Path,
    metadata: dict[str, Any],
    report: dict[str, Any] | None = None,
) -> None:
    """Persist versioned evaluator evidence without modifying the snapshot."""
    quality_dir = checkpoint_dir / QUALITY_DIR
    if quality_dir.is_symlink():
        raise OSError(f"scb-check quality directory is a symlink: {quality_dir}")
    quality_dir.mkdir(parents=True, exist_ok=True)
    record = {"schema_version": 1, **metadata, "report": report}
    atomic_write_text(
        quality_dir / SCB_CHECK_RECORD_FILENAME,
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _persist_scb_check_record(
    checkpoint_dir: Path,
    metadata: dict[str, Any],
    report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist evidence and return metadata that states the real outcome."""
    persisted = {
        **metadata,
        "record_persisted": True,
        "record_error": None,
    }
    try:
        _write_scb_check_record(checkpoint_dir, persisted, report)
    except OSError as exc:
        record_error = _error_record(exc, phase="persist_record")
        logger.error(
            "Failed to persist scb-check measurement record",
            checkpoint_dir=str(checkpoint_dir),
            error=record_error,
        )
        return {
            **metadata,
            "record_persisted": False,
            "record_error": record_error,
        }
    return persisted


def _scb_check_metadata(
    *,
    status: str,
    resolved_version: str | None,
    snapshot_sha256: str | None,
    evaluator_identity: dict[str, str] | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if evaluator_identity is None:
        evaluator_identity = _evaluator_identity()
    return {
        "evaluator": SCB_CHECK_NAME,
        "requested_version": SCB_CHECK_VERSION,
        "resolved_version": resolved_version,
        "status": status,
        "snapshot_path": SNAPSHOT_DIR_NAME,
        "snapshot_preserved": snapshot_sha256 is not None,
        "snapshot_tree_sha256": snapshot_sha256,
        "snapshot_hash_algorithm": SCB_CHECK_SNAPSHOT_HASH_ALGORITHM,
        "record_path": f"{QUALITY_DIR}/{SCB_CHECK_RECORD_FILENAME}",
        "environment_project_path": str(_evaluator_project_path()),
        "environment_project_sha256": evaluator_identity["project_sha256"],
        "environment_lock_path": str(_evaluator_lock_path()),
        "environment_lock_sha256": evaluator_identity["lock_sha256"],
        "error": error,
    }


def _resolved_scb_check_version() -> str:
    """Resolve the pinned evaluator from the current verified project."""
    command = _scb_check_command("--version")
    completed = _run_captured_command(
        command,
        timeout=SCB_CHECK_TIMEOUT_SECONDS,
        stdout_limit=SCB_CHECK_ERROR_STREAM_LIMIT,
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line]
    if not lines:
        raise ValueError("scb-check --version returned no version")
    resolved = lines[-1].removeprefix(f"{SCB_CHECK_NAME} ").strip()
    if resolved != SCB_CHECK_VERSION:
        raise ValueError(
            f"resolved scb-check {resolved!r}, expected {SCB_CHECK_VERSION!r}"
        )
    return resolved


def _get_scb_check_metrics(checkpoint_dir: Path) -> dict[str, Any]:
    """Run scb-check for composite quality metrics for a checkpoint."""
    snapshot_dir = checkpoint_dir / SNAPSHOT_DIR_NAME
    try:
        evaluator_identity = _evaluator_identity()
    except (OSError, ValueError) as exc:
        metadata = _scb_check_metadata(
            status="failed",
            resolved_version=None,
            snapshot_sha256=None,
            evaluator_identity={"project_sha256": "", "lock_sha256": ""},
            error=_error_record(exc, phase="verify_evaluator_inputs"),
        )
        metadata = _persist_scb_check_record(checkpoint_dir, metadata)
        return {"scb_check": metadata}
    if not snapshot_dir.exists():
        metadata = _scb_check_metadata(
            status="missing_snapshot",
            resolved_version=None,
            snapshot_sha256=None,
            evaluator_identity=evaluator_identity,
            error={
                "phase": "locate_snapshot",
                "type": "FileNotFoundError",
                "message": f"snapshot directory not found: {snapshot_dir}",
            },
        )
        metadata = _persist_scb_check_record(checkpoint_dir, metadata)
        return {"scb_check": metadata}

    snapshot_sha256: str | None = None
    resolved_version: str | None = None
    report: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    phase = "capture_snapshot"
    command: list[str] | None = None
    try:
        with _captured_snapshot(snapshot_dir) as (
            captured_snapshot,
            captured_sha256,
        ):
            snapshot_sha256 = captured_sha256
            phase = "resolve_version"
            command = _scb_check_command("--version")
            resolved_version = _resolved_scb_check_version()
            phase = "verify_evaluator_inputs"
            _require_evaluator_identity(
                evaluator_identity,
                phase="version resolution",
            )
            phase = "evaluate_snapshot"
            command = _scb_check_command(
                "check",
                "--report",
                "--include-all",
                str(captured_snapshot),
            )
            completed = _run_captured_command(
                command,
                timeout=SCB_CHECK_TIMEOUT_SECONDS,
                stdout_limit=SCB_CHECK_REPORT_STREAM_LIMIT,
            )
            report_value = json.loads(completed.stdout)
            if not isinstance(report_value, dict):
                raise ValueError("scb-check report is not a JSON object")
            json.dumps(report_value, allow_nan=False)
            metrics = _scb_check_metrics_from_report(report_value)
            report = report_value
            phase = "verify_evaluator_inputs"
            _require_evaluator_identity(
                evaluator_identity,
                phase="snapshot evaluation",
            )
            phase = "verify_captured_snapshot"
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as exc:
        error = _error_record(
            exc,
            phase=phase,
            command=command,
        )
        metadata = _scb_check_metadata(
            status="failed",
            resolved_version=resolved_version,
            snapshot_sha256=snapshot_sha256,
            evaluator_identity=evaluator_identity,
            error=error,
        )
        metadata = _persist_scb_check_record(checkpoint_dir, metadata)
        logger.error(
            "scb-check failed",
            checkpoint_dir=str(checkpoint_dir),
            snapshot_dir=str(snapshot_dir),
            error=error,
        )
        return {"scb_check": metadata}

    if report is None or metrics is None:
        raise AssertionError("successful scb-check evaluation produced no report")
    metadata = _scb_check_metadata(
        status="measured",
        resolved_version=resolved_version,
        snapshot_sha256=snapshot_sha256,
        evaluator_identity=evaluator_identity,
    )
    metadata = _persist_scb_check_record(checkpoint_dir, metadata, report)
    if not metadata["record_persisted"]:
        metadata = {
            **metadata,
            "status": "failed",
            "error": metadata["record_error"],
        }
        return {"scb_check": metadata}
    return {
        **metrics,
        "scb_check": metadata,
    }


def get_checkpoint_metrics(
    checkpoint_dir: Path,
    prior_metrics: dict | None = None,
    prior_checkpoint_dir: Path | None = None,
    is_first: bool = False,  # noqa: FBT001,FBT002
    is_last: bool = False,  # noqa: FBT001,FBT002
) -> dict:
    """Extract all metrics for a checkpoint directory.

    Combines evaluation, inference, quality, and rubric metrics into a single dict.

    Args:
        checkpoint_dir: Path to the checkpoint directory.
        prior_metrics: Metrics from previous checkpoint (for percentage deltas).
        prior_checkpoint_dir: Path to previous checkpoint directory (for mass deltas).
        is_first: Whether this is the first checkpoint.
        is_last: Whether this is the last checkpoint.
    Returns:
        Dictionary with all metrics combined. Keys use dot-notation for namespacing.

    Raises:
        MetricsError: If any metric extraction fails.
    """
    metrics = {
        **get_evaluation_metrics(checkpoint_dir),
        **get_inference_metrics(checkpoint_dir),
        **get_quality_metrics(checkpoint_dir),
        **get_rubric_metrics(checkpoint_dir),
    }

    # Add rubric density metric if applicable
    if "rubric_total_flags" in metrics and metrics.get("loc", 0) > 0:
        metrics["rubric_per_loc"] = (
            metrics["rubric_total_flags"] / metrics["loc"]
        )

    metrics.update(_get_scb_check_metrics(checkpoint_dir))

    # Compute deltas from prior checkpoint
    delta = compute_checkpoint_delta(prior_metrics, metrics)

    return {
        "is_first": is_first,
        "is_last": is_last,
        **metrics,
        **delta,
    }
